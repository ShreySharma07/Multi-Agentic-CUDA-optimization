# scripts/migrate_kb.py
"""
Migrate KnowledgeBase entries written before the memory fixes.

Three things were wrong with the old records:
  * no `techniques` field — `strategy_used` was free prose, so the planner (which
    reasons over playbook technique IDs) could never match an entry;
  * no `baseline` field — the legacy standalone path measured against a naive
    kernel (2-4x routine) and the torch path measures against PyTorch/cuBLAS
    (~1.0x is excellent). Both wrote to the same store, so a bare speedup is
    ambiguous;
  * insights written under the wrong baseline semantics — the reflector was never
    told what it was comparing against, so it labelled "99% of cuBLAS" as
    "complexity that did not translate into measurable speedup".

The era is recoverable: the legacy path stored kernel_name with a ".cu" suffix
(it used Path.name), the torch path stores the bare task stem. That is a reliable
discriminator, so entries can be migrated rather than blanket-purged.

Failure lessons are baseline-INDEPENDENT ("cp.async needs commit/wait groups")
and are always kept. Success entries whose insight demonstrably misreads a
near-parity result are dropped, because their prose is wrong and cannot be
regenerated without the original code (which was never stored).

    python scripts/migrate_kb.py            # dry run, prints the plan
    python scripts/migrate_kb.py --apply    # back up, then rewrite
"""
from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Agents.playbook import technique_names          # noqa: E402
from knowledge.store import KnowledgeBase            # noqa: E402

DB_DIR = Path("knowledge/db")

TORCH_BASELINE = "PyTorch eager (cuBLAS/cuDNN)"
LEGACY_BASELINE = "the original unoptimized CUDA kernel (NOT PyTorch)"

# Prose the old reflector used -> playbook technique IDs.
PHRASE_TO_TECHNIQUE = {
    "shared memory tiling": "shared_memory_tiling",
    "shared-memory tiling": "shared_memory_tiling",
    "shared memory tiled": "shared_memory_tiling",
    "shared-memory tiled": "shared_memory_tiling",
    "tiled matrix": "shared_memory_tiling",
    "tiling": "shared_memory_tiling",
    "register blocking": "register_blocking",
    "register-blocked": "register_blocking",
    "register blocked": "register_blocking",
    "micro-tile": "register_blocking",
    "warp tiling": "warp_tiling",
    "warp-level tiling": "warp_tiling",
    "float4": "vectorized_loads",
    "vectorized load": "vectorized_loads",
    "vectorised load": "vectorized_loads",
    "vectorized memory": "vectorized_loads",
    "bank conflict": "bank_conflict_avoidance",
    "padded shared": "bank_conflict_avoidance",
    "double buffer": "double_buffering",
    "prefetch": "double_buffering",
    "cp.async": "async_copy",
    "async copy": "async_copy",
    "loop unroll": "loop_unrolling",
    "unrolling": "loop_unrolling",
    "warp shuffle": "warp_shuffle_reduction",
    "__shfl": "warp_shuffle_reduction",
    "tree reduction": "shared_memory_tree_reduction",
    "grid-stride": "grid_stride_loop",
    "grid stride": "grid_stride_loop",
    "fast math": "fast_math_intrinsics",
    "__expf": "fast_math_intrinsics",
    "intrinsic": "fast_math_intrinsics",
    "occupancy": "occupancy_tuning",
    "coalesc": "coalesced_global_access",
    "__restrict__": "read_only_cache",
    "__ldg": "read_only_cache",
    "read-only cache": "read_only_cache",
    "tensor core": "tensor_core_mma",
    "wmma": "tensor_core_mma",
    "online softmax": "online_softmax",
    "constant memory": "constant_memory_filters",
    "thread coarsen": "multi_element_per_thread",
    "elements per thread": "multi_element_per_thread",
}

# An insight that dismisses a near-parity result is simply wrong under the torch
# baseline: 0.85-1.0x means 85-100% of a hand-tuned vendor library.
DISMISSIVE = (
    "did not translate", "no measurable", "not translate into",
    "no improvement", "not worth", "failed to improve", "no speedup",
    "did not yield", "no gain", "negligible", "no benefit",
)

KNOWN = set(technique_names())


def recover_techniques(meta: dict) -> list[str]:
    """Pull playbook technique IDs out of the old prose fields."""
    blob = " ".join(
        str(meta.get(k, "")) for k in ("strategy_used", "insight", "applicable_when")
    ).lower()
    found: list[str] = []
    for phrase, tech in PHRASE_TO_TECHNIQUE.items():
        if phrase in blob and tech in KNOWN and tech not in found:
            found.append(tech)
    return found


def era_baseline(meta: dict) -> str:
    """Legacy standalone runs stored kernel_name with a .cu suffix; torch runs
    store the bare task stem."""
    return LEGACY_BASELINE if str(meta.get("kernel_name", "")).endswith(".cu") else TORCH_BASELINE


def classify(meta: dict) -> tuple[str, str]:
    """(action, reason) — 'drop' or 'keep'."""
    name = str(meta.get("kernel_name", ""))
    result = str(meta.get("result", ""))
    insight = str(meta.get("insight", "")).lower()
    try:
        sp = float(meta.get("speedup", 0) or 0)
    except (TypeError, ValueError):
        sp = 0.0

    if name == "test_kernel":
        return "drop", "smoke-test fixture from store.py __main__"

    # Failure lessons hold regardless of what the speedup was measured against.
    if result != "success":
        return "keep", "failure lesson (baseline-independent)"

    baseline = era_baseline(meta)
    if baseline == TORCH_BASELINE and sp >= 0.85 and any(d in insight for d in DISMISSIVE):
        return "drop", f"insight dismisses {sp:.2f}x vs cuBLAS as a non-result — wrong"

    return "keep", "ok"


def main() -> None:
    apply = "--apply" in sys.argv

    kb = KnowledgeBase()
    raw = kb.collection.get(limit=10_000, include=["metadatas"])
    metas = raw["metadatas"]
    ids = raw["ids"]
    print(f"KnowledgeBase: {len(metas)} entries\n")

    keep, drop = [], []
    for m in metas:
        action, reason = classify(m)
        (drop if action == "drop" else keep).append((m, reason))

    print(f"DROP ({len(drop)}):")
    for m, why in drop:
        print(f"  - {str(m.get('kernel_name'))[:34]:36} {float(m.get('speedup',0) or 0):5.2f}x  {why}")

    print(f"\nKEEP + MIGRATE ({len(keep)}):")
    recovered = 0
    for m, _ in keep:
        techs = recover_techniques(m)
        if techs:
            recovered += 1
        m["_techniques"] = ",".join(techs)
        m["_baseline"] = era_baseline(m)

    for m, _ in keep[:8]:
        era = "legacy" if m["_baseline"] == LEGACY_BASELINE else "torch "
        print(f"  {era} {float(m.get('speedup',0) or 0):5.2f}x  "
              f"{str(m.get('kernel_name'))[:30]:32} -> {m['_techniques'] or '(none recovered)'}")
    if len(keep) > 8:
        print(f"  ... and {len(keep) - 8} more")

    print(f"\ntechnique IDs recovered for {recovered}/{len(keep)} kept entries")

    if not apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to migrate.")
        return

    backup = DB_DIR.parent / f"db_backup_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copytree(DB_DIR, backup)
    print(f"\nbacked up {DB_DIR} -> {backup}")

    kb.reset()
    for m, _ in keep:
        rec = {k: v for k, v in m.items() if not k.startswith("_")}
        rec["techniques"] = m["_techniques"]
        rec["baseline"] = m["_baseline"]
        kb.store(rec)

    print(f"migrated. KnowledgeBase now holds {kb.count()} entries "
          f"(dropped {len(drop)}, deduped {len(keep) - kb.count()}).")


if __name__ == "__main__":
    main()

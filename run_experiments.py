# run_experiments.py
"""
Headless experiment runner for KARMA.
Runs optimization loop on KernelBench / SGLang / own kernels.
Saves results to CSV after every kernel (crash-safe).

Each agent (coder / classifier / planner / reflector) is an independent class
with its own prompt, its own LLM provider and its own session id, all configured
in karma.yaml — so they can run on different models/vendors and never share
conversation context.

optimize_one() flow (per kernel):
  pre_flight (ncu metrics; Redis cache is best-effort and never load-bearing)
    → baseline benchmark (mean ± std over TIMED_RUNS)
    → ClassifierAgent.classify()   [once, if USE_CLASSIFIER]  — soft prior:
        kernel_type / bottleneck / preferred_strategies / confidence
        (also logged to results/classifier_log.csv for offline accuracy scoring)
    → KB.retrieve()                keyed by kernel_type to sharpen retrieval
    → round loop (1..ROUNDS):
        PlanningAgent.plan()       [each round, if USE_PLANNER] — turns the
            prior + metrics + KB + history into ONE concrete strategy+changes;
            free to choose "other" (escape hatch — prior is never a hard filter)
          → CoderAgent.optimize() emits the .cu (implements the plan exactly,
            or freeform if planner disabled)
          → GATES, cheapest first, each short-circuiting the round:
              compile → dims_match (regex, µs) → validate (runs CPU ref)
              → check_determinism (N full re-runs; catches races)
              → benchmark (baseline re-measured back-to-back; mean ± std)
          → reflect → KB.store; failed strategy names feed the next plan's avoid
    → best kernel + metrics written to results/experiments.csv

Ablation flags: --no-kb  --no-reflect  --no-classifier  --no-planner
Disabling the classifier and planner reproduces the original pipeline.
Config:         --config <file.yaml>   (default: karma.yaml)
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import csv
import json
import re
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# pipeline
from pipeline.compiler import compile_cuda
from pipeline.pre_flight import pre_flight
from pipeline.validator import run_validation, check_determinism
from pipeline.benchmarker import benchmark
from pipeline.dims import dims_match

# config + providers (LLM-agnostic: each agent picks its own provider/model)
from config import load_config, ConfigError
from Agents.providers import build_provider

# agents — each owns its class, prompt, provider and session id
from Agents.coder import CoderAgent
from Agents.classifier import ClassifierAgent, append_classifier_log
from Agents.planner import PlanningAgent

# knowledge (novel contribution)
from knowledge.store import KnowledgeBase
from knowledge.reflector import ReflectorAgent

# ── Config ─────────────────────────────────────────────────────────────
RESULTS_CSV  = "results/experiments.csv"
ROUNDS       = 3        # 3 for lab sessions, 5 for paper runs
WARMUP_RUNS  = 3
TIMED_RUNS   = 10
TIMEOUT_SEC  = 10
DETERMINISM_RUNS = 3    # full CPU-reference re-runs used to catch race conditions

# Ask the driver rather than hardcoding: prompts previously said "sm_86 / RTX
# A4000 Ampere" while compile_cuda auto-detected the real arch (sm_89 / Ada on
# this box), so the coder was tuning for hardware that wasn't there.
try:
    from pipeline.compiler import get_gpu_arch
    GPU_ARCH = get_gpu_arch()
except Exception:
    GPU_ARCH = "sm_86"
CONFIG_PATH  = "karma.yaml"
USE_KB         = True    # set False for ablation: no knowledge base
USE_REFLECT    = True    # set False for ablation: no reflection agent
USE_CLASSIFIER = True    # set False for ablation: no per-kernel classification
USE_PLANNER    = True    # set False for ablation: no per-round planning

# ── Initialize KnowledgeBase + agents ──────────────────────────────────
# Each agent gets its OWN provider (from karma.yaml) and its OWN session id, so
# they can run on different models and never share conversation context.
kb = KnowledgeBase() if USE_KB else None
coder = classifier = planner = reflector = None


def build_agents(config_path: str = CONFIG_PATH) -> None:
    """Construct every agent from karma.yaml. Fails fast on misconfiguration."""
    global coder, classifier, planner, reflector

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        print(f"\n  CONFIG ERROR: {e}\n")
        raise SystemExit(1)

    print("  LLM config:")
    for agent_name in ("coder", "classifier", "planner", "reflector"):
        print(f"    {agent_name:11} {cfg.for_agent(agent_name).redacted()}")

    coder = CoderAgent(build_provider(cfg.for_agent("coder")), session_id="coder")
    if USE_CLASSIFIER:
        classifier = ClassifierAgent(build_provider(cfg.for_agent("classifier")), session_id="classifier")
    if USE_PLANNER:
        planner = PlanningAgent(build_provider(cfg.for_agent("planner")), session_id="planner")
    if USE_REFLECT:
        reflector = ReflectorAgent(build_provider(cfg.for_agent("reflector")), session_id="reflector")


# ── Pydantic result schema ─────────────────────────────────────────────
class ExperimentResult(BaseModel):
    timestamp:            str
    kernel:               str
    source:               str
    bottleneck:           str
    occupancy:            float = 0.0
    compute_throughput:   float = 0.0
    dram_throughput:      float = 0.0
    baseline_ms:          float = 0.0
    baseline_ms_initial:  float = 0.0  # first measurement, before any round ran
    baseline_ms_final:    float = 0.0  # freshest re-measurement, taken back-to-back with best_ms
    baseline_drift_pct:   float = 0.0  # |final - initial| / initial -- thermal/clock drift signal
    best_ms:              float = 0.0
    best_ms_std:          float = 0.0  # stddev of the winning round's timed runs
    best_speedup:         float = 0.0
    best_round:           int   = 0
    rounds_total:         int   = 0
    compile_failures:     int   = 0
    validation_failures:  int   = 0
    dims_mismatches:      int   = 0  # rounds rejected for changing problem size
    unstable_rounds:      int   = 0  # rounds that failed their own check on a timed run
    kb_entries_used:      int   = 0
    kernel_type:              str = ""   # ClassifierAgent label
    classifier_confidence:    str = ""   # high | medium | low
    strategies_tried:         str = ""   # semicolon-joined per-round strategy names
    planner_used_other_count: int = 0    # rounds where the planner went off-taxonomy
    pytorch_ref:          Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────
def append_result(row: dict):
    """
    Append one result row, keyed by column name rather than position.

    ExperimentResult's schema has changed before without the CSV header being
    migrated (a previous run appended a `pytorch_ref` column with no header
    update, so that row silently has one more value than the header has
    columns). DictWriter with a bare `fieldnames=row.keys()` only writes a
    header on first-ever creation, so any later schema change just appends
    misaligned columns forever after. Detect a header mismatch here and
    migrate the whole file onto the union of columns instead.
    """
    Path("results").mkdir(exist_ok=True)
    csv_path = Path(RESULTS_CSV)
    fieldnames = list(row.keys())

    if csv_path.exists():
        with open(csv_path, newline="") as f:
            existing_rows = list(csv.DictReader(f))
        existing_header = list(existing_rows[0].keys()) if existing_rows else []

        if existing_header and existing_header != fieldnames:
            union_fields = fieldnames + [c for c in existing_header if c not in fieldnames]
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=union_fields, restval="")
                writer.writeheader()
                for old_row in existing_rows:
                    old_row.pop(None, None)  # overflow values DictReader stashed under None
                    writer.writerow(old_row)
                writer.writerow(row)
            print(f"  → schema changed — migrated {RESULTS_CSV} to the new column set")
            return

        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)
    else:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
    print(f"  → saved to {RESULTS_CSV}")


def already_done(kernel_name: str) -> bool:
    if not Path(RESULTS_CSV).exists():
        return False
    with open(RESULTS_CSV) as f:
        return any(kernel_name in line for line in f)


def safe_float(val, default=0.0) -> float:
    try:
        return float(str(val).replace("%", "").strip())
    except Exception:
        return default


async def _reflect_and_store(*, kernel_name, bottleneck, round_num, code, result,
                             speedup=0.0, error=""):
    """Reflect on one round's outcome and persist the insight. No-op if the
    reflector or the KB is disabled (ablation)."""
    if not (USE_REFLECT and reflector and kb):
        return
    insight = await reflector.reflect(
        kernel_name=kernel_name, bottleneck=bottleneck, round_num=round_num,
        code=code, result=result, speedup=speedup, error=error,
    )
    kb.store(insight)
    print(f"    KB stored: {insight['strategy_used'][:50]}")


# ── Core optimization loop ─────────────────────────────────────────────
async def optimize_one(kernel_path: Path, source: str = None) -> dict:
    name   = kernel_path.name
    source = source or kernel_path.read_text()

    print(f"\n{'='*60}")
    print(f"KERNEL: {name}")
    print(f"{'='*60}")

    # ── pre-flight ─────────────────────────────────────────────────
    print("  running pre-flight...")
    occ = comp = dram = 0.0
    hint = "unknown"
    metrics_ctx = ""

    try:
        pf = pre_flight(source)
    except Exception as e:
        pf = {"status": "error"}
        print(f"  pre-flight exception: {e}")

    if pf.get("status") == "success":
        m    = pf["metrics"]
        occ  = safe_float(m.get("occupancy", 0))
        comp = safe_float(m.get("compute_throughput", 0))
        dram = safe_float(m.get("dram_throughput", 0))
        hint = "memory-bound" if dram > comp else "compute-bound"
        metrics_ctx = (
            f"Hardware profile (Nsight Compute):\n"
            f"  Occupancy: {occ}%\n"
            f"  Compute throughput: {comp}%\n"
            f"  DRAM throughput: {dram}%\n"
            f"  Bottleneck: {hint}\n"
        )
        print(f"  pre-flight ok · occ={occ}% dram={dram}% → {hint}")
    else:
        print(f"  pre-flight unavailable — no metrics")

    metrics = {"occupancy": occ, "compute_throughput": comp, "dram_throughput": dram}

    # ── baseline timing (before loop; re-measured each round, see below) ──
    print("  benchmarking baseline...")
    baseline_ms = 0.0
    baseline_ms_initial = 0.0
    baseline_std = 0.0
    base_ok, base_bin = compile_cuda(str(kernel_path))
    if base_ok:
        base_stats = benchmark(base_bin, warmup=WARMUP_RUNS, runs=TIMED_RUNS)
        baseline_ms = base_stats["mean_ms"]
        baseline_ms_initial = baseline_ms
        baseline_std = base_stats["std_ms"]
        cov = (baseline_std / baseline_ms * 100) if baseline_ms else 0.0
        print(f"  baseline: {baseline_ms:.3f}ms (±{baseline_std:.3f}ms, {cov:.1f}% CoV over {base_stats['n']} runs)")
    else:
        print("  baseline compile failed — speedup will be 0")

    # ── classification (once per kernel, soft prior for the planner) ──
    classification = None
    kernel_type_out = ""
    classifier_conf_out = ""
    if USE_CLASSIFIER and classifier:
        classification = await classifier.classify(source, metrics)
        kernel_type_out = classification["kernel_type"]
        classifier_conf_out = classification["confidence"]
        print(f"  classifier: type={kernel_type_out} bottleneck={classification['bottleneck']} "
              f"conf={classifier_conf_out} · prefers={classification['preferred_strategies']}")
        print(f"    rationale: {classification['rationale']}")
        append_classifier_log(name, classification)

    # ── KnowledgeBase retrieval ────────────────────────────────────
    # Keyed by the classifier's kernel_type when available so retrieval matches
    # on the compute pattern rather than the (often opaque) file name.
    retrieval_key = kernel_type_out if (USE_CLASSIFIER and classification) else name
    kb_ctx = ""
    kb_used = 0
    kb_results = []
    if USE_KB and kb:
        kb_results = kb.retrieve(bottleneck=hint, kernel_name=retrieval_key)
        kb_used = len(kb_results)
        if kb_results:
            kb_ctx = "Relevant past optimizations from KnowledgeBase:\n"
            for past in kb_results:
                kb_ctx += (
                    f"  - {past.get('bottleneck','?')} kernel: "
                    f"{past.get('strategy_used','?')} → "
                    f"{past.get('speedup',0)}x. "
                    f"Insight: {past.get('insight','?')}. "
                    f"Avoid if: {past.get('avoid_if','?')}\n"
                )
            kb_ctx += "\nUse these insights to inform your optimization.\n\n"
            print(f"  KB: {kb_used} relevant past entries found")
        else:
            print(f"  KB: no relevant history (total entries: {kb.count()})")

    # ── optimization rounds ────────────────────────────────────────
    best_speedup = None
    best_round   = None
    best_code    = source
    best_std     = 0.0
    baseline_drift_pct = 0.0
    planner_other_count = 0
    history      = []

    strategy_hint = {
        "memory-bound": "Focus: coalesced access, float4 vectorized loads, shared memory tiling.",
        "compute-bound": "Focus: fast math (__expf, __fdividef), loop unrolling, occupancy.",
        "unknown": "Apply the most impactful optimization you can identify.",
    }.get(hint, "")

    for r in range(1, ROUNDS + 1):
        print(f"  round {r}/{ROUNDS}: asking agent...")

        # build history context
        history_ctx = ""
        if history:
            history_ctx = "Previous attempts this session:\n"
            for h in history:
                if h["result"] == "success":
                    history_ctx += f"  Round {h['round']}: {h['speedup']:.2f}x speedup\n"
                elif h["result"] == "compile_failed":
                    history_ctx += f"  Round {h['round']}: COMPILE FAILED — {h.get('error','')[:100]}\n"
                elif h["result"] == "validation_failed":
                    history_ctx += f"  Round {h['round']}: VALIDATION FAILED — math incorrect\n"
                elif h["result"] == "dims_mismatch":
                    history_ctx += (
                        f"  Round {h['round']}: REJECTED — changed problem size "
                        f"({h.get('error','')[:100]}). Keep dimensions identical.\n"
                    )
                elif h["result"] == "unstable":
                    history_ctx += (
                        f"  Round {h['round']}: REJECTED — nondeterministic, failed its own "
                        f"check on a later run. Likely a race condition.\n"
                    )
                elif h["result"] == "empty_response":
                    history_ctx += f"  Round {h['round']}: agent returned no valid code\n"
            history_ctx += "\n"

        best_ctx = ""
        if best_speedup:
            best_ctx = f"Current best: {best_speedup:.2f}x — beat this.\n\n"

        # ── planning (once per round; turns prior+metrics+KB+history into a plan) ──
        current_strategy = "agent_choice"
        if USE_PLANNER and planner:
            current_plan = await planner.plan(
                classification, metrics, kb_results, history, best_speedup
            )
            current_strategy = current_plan.get("strategy", "agent_choice")
            if current_plan.get("strategy_is_other"):
                planner_other_count += 1
            print(f"  round {r}: plan → {current_strategy}"
                  + (" (other)" if current_plan.get("strategy_is_other") else ""))
            guidance = (
                "Implement EXACTLY this plan. Do not choose a different strategy:\n"
                + json.dumps(current_plan, indent=2)
                + "\n\n"
            )
        else:
            guidance = f"{strategy_hint}\n\n"

        # The prompt (hard constraints, harness contract, output rules) lives in
        # CoderAgent — this loop only supplies context.
        try:
            optimized = await coder.optimize(
                best_code,
                round_num=r, rounds=ROUNDS, gpu_arch=GPU_ARCH,
                guidance=guidance, metrics_ctx=metrics_ctx,
                kb_ctx=kb_ctx, history_ctx=history_ctx, best_ctx=best_ctx,
            )
        except Exception as e:
            print(f"  round {r}: coder call failed — {e}")
            optimized = ""

        if not optimized or len(optimized) < 30:
            print(f"  round {r}: invalid/empty output — skipping")
            history.append({"round": r, "result": "empty_response", "strategy": current_strategy})
            continue

        tmp = Path(f"kernels/tmp_exp_r{r}.cu")
        tmp.write_text(optimized)

        # ── compile ────────────────────────────────────────────────
        print(f"  round {r}: compiling...")
        ok, result = compile_cuda(str(tmp))

        if not ok:
            err = str(result)[:400]
            print(f"  round {r}: COMPILE FAILED — {err[:80]}")
            history.append({"round": r, "result": "compile_failed", "error": err, "strategy": current_strategy})
            # feed error into next round
            best_code = (
                "// COMPILE ERROR — fix these exact errors first:\n"
                + "\n".join(f"// {line}" for line in err.splitlines()[:10])
                + f"\n\n{optimized}"
            )
            # reflect on failure
            await _reflect_and_store(
                kernel_name=name, bottleneck=hint, round_num=r,
                code=optimized, result="compile_failed", error=err,
            )
            continue

        # ── problem-size equivalence gate ───────────────────────────
        # A kernel can pass validation yet still be "faster" only because it
        # quietly shrank the problem (e.g. N_TEST). This is a regex over the
        # source (microseconds) whereas run_validation() executes the binary and
        # a single-threaded CPU reference (seconds-to-minutes), so it runs first.
        dims_ok, dims_msg = dims_match(source, optimized)
        if not dims_ok:
            print(f"  round {r}: PROBLEM-SIZE MISMATCH — {dims_msg}")
            history.append({"round": r, "result": "dims_mismatch", "error": dims_msg, "strategy": current_strategy})
            best_code = (
                f"// REJECTED — you changed the problem dimensions: {dims_msg}\n"
                f"// Dimensions must stay IDENTICAL to the baseline. A smaller\n"
                f"// problem is not a valid optimization and is checked automatically.\n"
                f"// (Tuning knobs like TILE_SIZE/BLOCK_DIM ARE free to change.)\n\n"
                + optimized
            )
            await _reflect_and_store(
                kernel_name=name, bottleneck=hint, round_num=r,
                code=optimized, result="dims_mismatch", error=dims_msg,
            )
            continue

        # ── validate ───────────────────────────────────────────────
        print(f"  round {r}: validating...")
        val_ok, val_msg = run_validation(result)

        if not val_ok:
            msg = str(val_msg)[:200]
            print(f"  round {r}: VALIDATION FAILED — {msg[:60]}")
            history.append({"round": r, "result": "validation_failed", "strategy": current_strategy})
            best_code = (
                f"// VALIDATION FAILED — output wrong.\n"
                f"// Message: {msg}\n"
                f"// Fix the math. Do not restructure the kernel.\n\n"
                + optimized
            )
            await _reflect_and_store(
                kernel_name=name, bottleneck=hint, round_num=r,
                code=optimized, result="validation_failed", error=msg,
            )
            continue

        # ── determinism gate ───────────────────────────────────────
        # Passing validation once proves nothing about a racy kernel. Re-run the
        # full CPU-reference check a few times; a kernel that only sometimes
        # agrees is not a result, however fast it measures.
        print(f"  round {r}: checking determinism...")
        det_ok, det_msg = check_determinism(result, runs=DETERMINISM_RUNS)
        if not det_ok:
            print(f"  round {r}: UNSTABLE — {str(det_msg)[:70]}")
            history.append({"round": r, "result": "unstable", "strategy": current_strategy})
            best_code = (
                "// REJECTED — this kernel passed validation once but failed a\n"
                "// repeat run, so it is not deterministically correct (race\n"
                "// condition). Every thread in a block must reach every\n"
                "// __syncthreads(); never return early before a barrier.\n"
                f"// Detail: {str(det_msg)[:200]}\n\n"
                + optimized
            )
            await _reflect_and_store(
                kernel_name=name, bottleneck=hint, round_num=r,
                code=optimized, result="unstable", error=str(det_msg)[:200],
            )
            continue

        # ── benchmark ──────────────────────────────────────────────
        # Re-measure the baseline back-to-back with the candidate so both
        # numbers are taken under the same thermal/clock conditions, instead
        # of trusting a baseline measured minutes (and rounds) earlier.
        fresh_base = benchmark(base_bin, warmup=WARMUP_RUNS, runs=TIMED_RUNS)
        if fresh_base["n"] > 0:
            baseline_ms = fresh_base["mean_ms"]
        baseline_drift_pct = (
            100.0 * abs(baseline_ms - baseline_ms_initial) / baseline_ms_initial
            if baseline_ms_initial > 0 else 0.0
        )

        opt_stats = benchmark(result, warmup=WARMUP_RUNS, runs=TIMED_RUNS)
        opt_ms = opt_stats["mean_ms"]

        if not opt_stats["stable"]:
            # Belt-and-braces only: fires for legacy kernels that ignore
            # BENCH_ONLY and still print their SUCCESS/FAILURE token on timed
            # runs. Real race detection already happened in check_determinism().
            print(f"  round {r}: WARNING — a timed run printed a failure token")

        if opt_ms <= 0 or baseline_ms <= 0:
            print(f"  round {r}: benchmark returned 0 — binary may not print 'GPU Time'")
            speedup = 0.0
        else:
            speedup = round(baseline_ms / opt_ms, 4)

        cov = (opt_stats["std_ms"] / opt_ms * 100) if opt_ms else 0.0
        print(f"  round {r}: ✓ {opt_ms:.3f}ms (±{opt_stats['std_ms']:.3f}ms, {cov:.1f}% CoV) "
              f"→ {speedup:.2f}x speedup [baseline {baseline_ms:.3f}ms, drift {baseline_drift_pct:+.1f}%]")
        history.append({"round": r, "speedup": speedup, "result": "success", "strategy": current_strategy})

        # track best
        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round   = r
            best_code    = optimized
            best_std     = opt_stats["std_ms"]
            Path("kernels/results").mkdir(exist_ok=True)
            Path(f"kernels/results/{name}").write_text(optimized)
            print(f"  new best — saved kernels/results/{name}")
        else:
            print(f"  round {r}: {speedup:.2f}x < best {best_speedup:.2f}x — discarded")
            # reset to best code for next round (don't build on worse code)
            best_code = source

        # reflect on success
        await _reflect_and_store(
            kernel_name=name, bottleneck=hint, round_num=r,
            code=optimized, result="success", speedup=speedup,
        )

        # convergence check
        successes = [h for h in history if h.get("result") == "success"]
        if len(successes) >= 2:
            prev, curr = successes[-2]["speedup"], successes[-1]["speedup"]
            if abs(curr - prev) < 0.01:
                print("  converged — stopping early")
                break

    # ── cleanup temp files ─────────────────────────────────────────
    for r_clean in range(1, ROUNDS + 1):
        tmp = Path(f"kernels/tmp_exp_r{r_clean}.cu")
        if tmp.exists():
            tmp.unlink()

    # ── build result ───────────────────────────────────────────────
    n_compile = sum(1 for h in history if h["result"] == "compile_failed")
    n_val     = sum(1 for h in history if h["result"] == "validation_failed")
    n_dims    = sum(1 for h in history if h["result"] == "dims_mismatch")
    n_unstable = sum(1 for h in history if h["result"] == "unstable")
    strategies_tried = ";".join(h.get("strategy", "agent_choice") for h in history)

    print(f"\n  RESULT: best={'%.4f' % best_speedup if best_speedup else '0'}x  "
          f"round={best_round}  compile_fails={n_compile}  val_fails={n_val}  "
          f"dims_mismatches={n_dims}  unstable={n_unstable}  "
          f"baseline_drift={baseline_drift_pct:.1f}%  kb_entries={kb.count() if kb else 0}")

    result = ExperimentResult(
        timestamp=datetime.now().isoformat(),
        kernel=name,
        source=str(kernel_path),
        bottleneck=hint,
        occupancy=occ,
        compute_throughput=comp,
        dram_throughput=dram,
        baseline_ms=baseline_ms,
        baseline_ms_initial=baseline_ms_initial,
        baseline_ms_final=baseline_ms,
        baseline_drift_pct=baseline_drift_pct,
        best_ms=round(baseline_ms / best_speedup, 4) if best_speedup and best_speedup > 0 else 0.0,
        best_ms_std=best_std,
        best_speedup=best_speedup or 0.0,
        best_round=best_round or 0,
        rounds_total=len(history),
        compile_failures=n_compile,
        validation_failures=n_val,
        dims_mismatches=n_dims,
        unstable_rounds=n_unstable,
        kb_entries_used=kb_used,
        kernel_type=kernel_type_out,
        classifier_confidence=classifier_conf_out,
        strategies_tried=strategies_tried,
        planner_used_other_count=planner_other_count,
    )
    return result.model_dump()


# ── Run modes ──────────────────────────────────────────────────────────
async def run_own_kernels():
    skip = {"tmp_", "best_", "baseline_", "temp_"}
    kernels = sorted(
        f for f in Path("kernels").glob("*.cu")
        if not any(f.name.startswith(s) for s in skip)
    )
    print(f"Found {len(kernels)} own kernels")
    Path("kernels/results").mkdir(exist_ok=True)
    for kp in kernels:
        if already_done(kp.name):
            print(f"  already in CSV — skipping {kp.name}")
            continue
        row = await optimize_one(kp)
        append_result(row)


def find_kernelbench_path() -> Path | None:
    for p in Path("KernelBench").rglob("level1"):
        return p.parent
    return None


async def run_kernelbench(level: int = 1, max_kernels: int = 10):
    kb_base = find_kernelbench_path()
    if not kb_base:
        print("KernelBench not found. Run: git clone https://github.com/ScalingIntelligence/KernelBench")
        return

    kb_path = kb_base / f"level{level}"
    if not kb_path.exists():
        print(f"Level {level} not found. Available: {list(kb_base.iterdir())}")
        return

    files = sorted(kb_path.glob("*.py"))[:max_kernels]
    print(f"Found {len(files)} kernels at {kb_path}")
    Path("kernels/kernelbench").mkdir(parents=True, exist_ok=True)
    Path("kernels/results").mkdir(exist_ok=True)

    for py_file in files:
        cu_name = py_file.stem + ".cu"

        if already_done(cu_name):
            print(f"\n  already in CSV — skipping {py_file.name}")
            continue

        print(f"\n{'='*60}")
        print(f"KernelBench: {py_file.name}")

        pytorch_code = py_file.read_text()

        print(f"  generating baseline CUDA from PyTorch...")
        try:
            baseline_cuda = await coder.generate_baseline(pytorch_code, gpu_arch=GPU_ARCH)
        except Exception as e:
            print(f"  baseline generation failed — skipping: {e}")
            continue

        if not baseline_cuda or not baseline_cuda.startswith("#include"):
            print(f"  failed to generate valid baseline — skipping")
            continue

        cu_path = Path(f"kernels/kernelbench/{cu_name}")
        cu_path.write_text(baseline_cuda)

        ok, err = compile_cuda(str(cu_path))
        if not ok:
            print(f"  baseline compile failed — skipping: {str(err)[:80]}")
            continue

        print(f"  baseline compiled — optimizing...")
        row = await optimize_one(cu_path, source=baseline_cuda)
        row["source"]      = f"kernelbench_l{level}"
        row["pytorch_ref"] = py_file.name
        append_result(row)


def find_sglang_kernels() -> list[Path]:
    all_cu = list(Path("sglang").rglob("*.cu"))
    skip_patterns = {"test", "benchmark", "example", "cmake"}
    return sorted(
        f for f in all_cu
        if not any(s in str(f).lower() for s in skip_patterns)
    )


async def run_sglang(max_kernels: int = 5):
    kernels = find_sglang_kernels()
    if not kernels:
        print("No SGLang .cu files found — run: find sglang/ -name '*.cu' | head -10")
        return

    print(f"Found {len(kernels)} SGLang .cu files")
    Path("kernels/sglang").mkdir(parents=True, exist_ok=True)
    Path("kernels/results").mkdir(exist_ok=True)

    for kp in kernels[:max_kernels]:
        dest = Path("kernels/sglang") / kp.name
        dest.write_text(kp.read_text())

    for kp in sorted(Path("kernels/sglang").glob("*.cu"))[:max_kernels]:
        if already_done(kp.name):
            print(f"  already in CSV — skipping {kp.name}")
            continue
        row = await optimize_one(kp)
        row["source"] = "sglang"
        append_result(row)


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "own"

    # ablation flags
    if "--no-kb" in sys.argv:
        USE_KB = False
        kb = None
        print("⚠  ABLATION MODE: KnowledgeBase disabled")
    if "--no-reflect" in sys.argv:
        USE_REFLECT = False
        print("⚠  ABLATION MODE: ReflectionAgent disabled")
    if "--no-classifier" in sys.argv:
        USE_CLASSIFIER = False
        print("⚠  ABLATION MODE: ClassifierAgent disabled")
    if "--no-planner" in sys.argv:
        USE_PLANNER = False
        print("⚠  ABLATION MODE: PlanningAgent disabled")

    # optional: --config path/to/other.yaml
    if "--config" in sys.argv:
        CONFIG_PATH = sys.argv[sys.argv.index("--config") + 1]

    print(f"\nGPU arch: {GPU_ARCH}")
    build_agents(CONFIG_PATH)   # honours the ablation flags set above
    print()

    if mode == "own":
        asyncio.run(run_own_kernels())
    elif mode == "kernelbench":
        level = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 1
        count = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 10
        asyncio.run(run_kernelbench(level=level, max_kernels=count))
    elif mode == "sglang":
        count = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 5
        asyncio.run(run_sglang(max_kernels=count))
    else:
        print("Usage: python run_experiments.py [own|kernelbench|sglang] [level] [max_kernels]")
        print("Flags: --no-kb  --no-reflect  --no-classifier  --no-planner  --config <file.yaml>")
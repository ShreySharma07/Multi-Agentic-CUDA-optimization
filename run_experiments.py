# run_experiments.py
"""
Headless experiment runner for KARMA.
Runs optimization loop on KernelBench / SGLang / own kernels.
Saves results to CSV after every kernel (crash-safe).

Each agent (coder / classifier / planner / reflector) is an independent class
with its own prompt, its own LLM provider and its own session id, all configured
in karma.yaml — so they can run on different models/vendors and never share
conversation context.

TWO EVALUATION PATHS
--------------------
kernelbench  → torch-extension evaluation (CUDA Agent methodology). Python owns
               the inputs, the PyTorch correctness oracle and the clock; the LLM
               writes ONLY {cuda_source, cpp_source}. The model cannot shrink the
               problem, weaken a tolerance, or grade itself — so dims_match,
               BENCH_ONLY and check_determinism are not needed here and are not
               used. See pipeline/torch_eval.py.
own, sglang  → legacy standalone-.cu path, unchanged (self-contained program with
               main(), its own CPU reference and timing).

optimize_one_torch() flow (per kernelbench task):
  load_task()            import the KernelBench module (Model/get_inputs)
    → probe_task()       does it fit in VRAM? if not, SKIP before spending any
                         tokens (Windows silently pages past VRAM into system RAM,
                         which produces plausible timings that only measure PCIe)
    → measure_baselines()  PyTorch eager, and torch.compile when available
                           (unavailable on Windows: triton has no wheel)
    → ROUND 0: coder.translate() → compile → validate, up to translate_retries
    → ext pre-flight     ncu on the round-0 kernel → occupancy/dram/compute
                         (degrades to bottleneck="unknown" on ERR_NVGPUCTRPERM)
    → classify (soft prior) → KB.retrieve keyed by kernel_type
    → rounds 1..ROUNDS:
        plan → coder.optimize() → compile_extension
          → validate   (Gate A: correct vs PyTorch on 3 seeds;
                        Gate B: deterministic — catches races)
          → benchmark_interleaved  (eager vs karma, interleaved to cancel drift)
          → reflect → KB.store; failed strategy feeds the next plan's avoid list
        early stop when speedup_vs_compile >= target (target_met)
    → best kernel → kernels/results/{task}.cu + {task}_binding.cpp
    → row → results/experiments.csv

Ablation flags: --no-kb  --no-reflect  --no-classifier  --no-planner
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

import torch
from pydantic import BaseModel

# pipeline
from pipeline.compiler import compile_cuda
from pipeline.pre_flight import pre_flight
from pipeline.validator import run_validation, check_determinism
from pipeline.benchmarker import benchmark
from pipeline.dims import dims_match
from pipeline.torch_eval import (
    load_task, probe_task, measure_baselines, compile_extension,
    validate as validate_ext, benchmark_interleaved, run_off,
)
from pipeline.ext_profiler import profile_extension

# config + providers (LLM-agnostic: each agent picks its own provider/model)
from config import load_config, ConfigError
from Agents.providers import build_provider

# agents — each owns its class, prompt, provider and session id
from Agents.coder import CoderAgent, build_signature_ctx
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
# The legacy own/sglang path measures against the ORIGINAL naive kernel, not
# PyTorch. Both paths write to the SAME KnowledgeBase, so an entry must record
# what it was measured against -- otherwise a 2.5x legacy row outranks a 0.98x
# torch row that is in fact the far better kernel.
LEGACY_BASELINE = "the original unoptimized CUDA kernel (NOT PyTorch)"
USE_KB         = True    # set False for ablation: no knowledge base
USE_REFLECT    = True    # set False for ablation: no reflection agent
USE_CLASSIFIER = True    # set False for ablation: no per-kernel classification
USE_PLANNER    = True    # set False for ablation: no per-round planning

# ── Initialize KnowledgeBase + agents ──────────────────────────────────
# Each agent gets its OWN provider (from karma.yaml) and its OWN session id, so
# they can run on different models and never share conversation context.
kb = KnowledgeBase() if USE_KB else None
coder = classifier = planner = reflector = None
EVAL = None   # EvalSettings from karma.yaml; populated by build_agents()


def build_agents(config_path: str = CONFIG_PATH) -> None:
    """Construct every agent from karma.yaml. Fails fast on misconfiguration."""
    global coder, classifier, planner, reflector, EVAL

    try:
        cfg = load_config(config_path)
    except ConfigError as e:
        print(f"\n  CONFIG ERROR: {e}\n")
        raise SystemExit(1)

    EVAL = cfg.eval
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
    dims_mismatches:      int   = 0  # legacy path only (torch path can't be cheated this way)
    unstable_rounds:      int   = 0  # rounds rejected as nondeterministic
    kb_entries_used:      int   = 0
    kernel_type:              str = ""   # ClassifierAgent label
    classifier_confidence:    str = ""   # high | medium | low
    strategies_tried:         str = ""   # semicolon-joined per-round strategy names
    applied_techniques:       str = ""   # cumulative stack in the BEST kernel
    planner_used_other_count: int = 0    # rounds where the planner went off-taxonomy
    pytorch_ref:          Optional[str] = None

    # ── torch-extension path ───────────────────────────────────────────
    status:               str   = "ok"   # ok | translate_failed | skipped_oom | error
    eager_ms:             float = 0.0    # PyTorch eager baseline
    eager_std:            float = 0.0
    compile_ms:           Optional[float] = None   # torch.compile (None on Windows: no triton)
    karma_ms:             float = 0.0    # best kernel
    karma_std:            float = 0.0
    speedup_vs_eager:     float = 0.0
    speedup_vs_compile:   float = 0.0
    target_met:           bool  = False  # beat torch.compile by the configured margin
    translate_retries:    int   = 0      # round-0 attempts needed for a correct port
    peak_gb:              float = 0.0    # peak GPU memory of the reference op


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
                             speedup=0.0, error="", applied=None,
                             baseline="PyTorch eager (cuBLAS/cuDNN)"):
    """Reflect on one round's outcome and persist the insight. No-op if the
    reflector or the KB is disabled (ablation).

    `applied` and `baseline` matter: without them the reflector cannot tell what a
    speedup of 0.99x means, and it once concluded that reaching 99% of cuBLAS was
    "complexity that did not translate into measurable speedup".
    """
    if not (USE_REFLECT and reflector and kb):
        return
    insight = await reflector.reflect(
        kernel_name=kernel_name, bottleneck=bottleneck, round_num=round_num,
        code=code, result=result, speedup=speedup, error=error,
        applied=applied, baseline=baseline,
    )
    kb.store(insight)
    techs = insight.get("techniques") or "(none)"
    print(f"    KB stored [{result}, {speedup:.2f}x]: {techs}")


# ── Torch-extension loop (kernelbench) ─────────────────────────────────
async def optimize_one_torch(py_file: Path) -> dict:
    """
    CUDA Agent methodology: Python owns inputs, correctness and timing; the LLM
    writes only the kernel + binding. See the module docstring for the flow.
    """
    name = py_file.stem
    print(f"\n{'='*60}")
    print(f"TASK: {name}")
    print(f"{'='*60}")

    row = dict(
        timestamp=datetime.now().isoformat(),
        kernel=f"{name}.cu",
        source="kernelbench",
        pytorch_ref=py_file.name,
        bottleneck="unknown",
    )

    # ── load + VRAM feasibility (before spending a single token) ───────
    try:
        task = load_task(py_file)
    except Exception as e:
        print(f"  load failed: {e}")
        return ExperimentResult(**row, status="error").model_dump()

    fits, why, peak_gb = probe_task(task, headroom=EVAL.vram_headroom)
    row["peak_gb"] = peak_gb
    if not fits:
        print(f"  SKIPPED — {why}")
        return ExperimentResult(**row, status="skipped_oom").model_dump()

    # ── baselines: PyTorch is the reference, not the LLM ───────────────
    print("  measuring PyTorch baselines...")
    base = measure_baselines(task, warmup=EVAL.warmup, runs=EVAL.runs)
    eager_ms, compile_ms = base["eager_ms"], base["compile_ms"]
    if not eager_ms or eager_ms <= 0:
        return ExperimentResult(**row, status="error").model_dump()

    row.update(eager_ms=eager_ms, eager_std=base["eager_std"], compile_ms=compile_ms)
    cmp_txt = f"{compile_ms:.3f}ms" if compile_ms else "n/a"
    print(f"  eager: {eager_ms:.3f}ms (±{base['eager_std']:.3f})  "
          f"compile: {cmp_txt}  peak: {peak_gb:.2f}GB")

    # ── ROUND 0: faithful translation ──────────────────────────────────
    # forward() must accept the activation inputs AND the module's parameters
    # (Linear/Conv weights, norm affine + running stats). The kernel is a free
    # function, so Python passes them explicitly; the coder is told the exact
    # ordered signature. Without this, any task with learned weights fails with a
    # TypeError at call time.
    input_specs = [(f"x{i}", tuple(t.shape), str(t.dtype).replace("torch.", ""))
                   for i, t in enumerate(task.inputs())]
    param_specs = task.param_specs()
    signature_ctx = build_signature_ctx(input_specs, param_specs)
    if param_specs:
        print(f"  signature: {len(input_specs)} input(s) + {len(param_specs)} parameter(s)")

    cuda_src = cpp_src = None
    err_fb = ""
    translate_retries = 0

    for attempt in range(1, EVAL.translate_retries + 1):
        print(f"  round 0 (translate) attempt {attempt}/{EVAL.translate_retries}...")
        out = await coder.translate(task.source, gpu_arch=GPU_ARCH,
                                    signature_ctx=signature_ctx, error_feedback=err_fb)
        if not out:
            err_fb = "Your reply could not be parsed as the required JSON object."
            translate_retries = attempt
            continue

        mod, cerr = compile_extension(
            out["cuda_source"], out["cpp_source"], timeout_s=EVAL.compile_timeout_s
        )
        if mod is None:
            print(f"    compile failed: {cerr.splitlines()[0][:70] if cerr else '?'}")
            err_fb = f"COMPILER ERROR:\n{cerr[:2000]}"
            translate_retries = attempt
            continue

        vok, vmsg = validate_ext(mod, task)
        if not vok:
            print(f"    incorrect: {vmsg[:70]}")
            err_fb = f"VALIDATION FAILED:\n{vmsg}"
            translate_retries = attempt
            continue

        cuda_src, cpp_src = out["cuda_source"], out["cpp_source"]
        translate_retries = attempt
        print(f"    translation correct ({vmsg})")
        break

    row["translate_retries"] = translate_retries
    if cuda_src is None:
        print(f"  TRANSLATE FAILED after {EVAL.translate_retries} attempts")
        return ExperimentResult(**row, status="translate_failed").model_dump()

    # baseline kernel timing (round 0 is the thing rounds 1..N must beat)
    r0 = benchmark_interleaved(mod, task, warmup=EVAL.warmup, runs=EVAL.runs)
    best_cuda, best_cpp = cuda_src, cpp_src
    best_speedup = r0["speedup_vs_eager"]
    best_ms, best_std, best_round = r0["karma_ms"], r0["karma_std"], 0
    print(f"  round 0: {r0['karma_ms']:.3f}ms → {best_speedup:.2f}x vs eager")

    # ── pre-flight: profile the round-0 kernel ─────────────────────────
    print("  profiling (ncu)...")
    occ = comp = dram = 0.0
    hint = "unknown"
    metrics_ctx = ""
    pf = profile_extension(cuda_src, cpp_src, py_file, Path.cwd(), timeout_s=EVAL.ncu_timeout_s)
    if pf.get("status") == "success":
        m = pf["metrics"]
        occ, comp, dram = (safe_float(m.get(k, 0)) for k in
                           ("occupancy", "compute_throughput", "dram_throughput"))
        hint = "memory-bound" if dram > comp else "compute-bound"
        metrics_ctx = (
            f"Hardware profile (Nsight Compute):\n"
            f"  Occupancy: {occ}%\n  Compute throughput: {comp}%\n"
            f"  DRAM throughput: {dram}%\n  Bottleneck: {hint}\n\n"
        )
        print(f"    occ={occ}% dram={dram}% → {hint}")
    else:
        print(f"    unavailable — {str(pf.get('error_message'))[:70]}")

    row.update(bottleneck=hint, occupancy=occ, compute_throughput=comp, dram_throughput=dram)
    metrics = {"occupancy": occ, "compute_throughput": comp, "dram_throughput": dram}

    # ── classify (soft prior) ──────────────────────────────────────────
    classification = None
    if USE_CLASSIFIER and classifier:
        classification = await classifier.classify(task.source, metrics)
        print(f"  classifier: {classification['kernel_type']} / "
              f"{classification['bottleneck']} / {classification['confidence']}")
        append_classifier_log(f"{name}.cu", classification)
        row["kernel_type"] = classification["kernel_type"]
        row["classifier_confidence"] = classification["confidence"]

    # ── KB retrieval ───────────────────────────────────────────────────
    kb_ctx, kb_results = "", []
    if USE_KB and kb:
        key = classification["kernel_type"] if classification else name
        kb_results = kb.retrieve(
            bottleneck=hint, kernel_name=key,
            techniques=(classification or {}).get("preferred_strategies"),
        )
        if kb_results:
            # Label the outcome AND the baseline. These used to be listed under
            # "Relevant past optimizations" with no sign that some had failed
            # outright, and with no way to tell a 3x-vs-naive row (legacy path)
            # from a 0.98x-vs-cuBLAS row, which is the far better kernel.
            kb_ctx = "Past rounds on similar kernels:\n"
            for p in kb_results:
                try:
                    sp = float(p.get("speedup", 0) or 0)
                except (TypeError, ValueError):
                    sp = 0.0
                techs = str(p.get("techniques", "")).strip() or "(unrecorded)"
                # Test on "cuBLAS": the legacy baseline string contains the words
                # "NOT PyTorch", so a `"PyTorch" in ..." check matches it too.
                # NB: named entry_baseline, not `base` — `base` is the
                # measure_baselines() dict in this scope and shadowing it broke
                # the run-off.
                entry_baseline = str(p.get("baseline", ""))
                vs = ("vs PyTorch/cuBLAS" if "cuBLAS" in entry_baseline
                      else "vs a naive kernel (NOT comparable to PyTorch numbers)")
                kb_ctx += (f"  - [{p.get('result','?')}, {sp:.2f}x {vs}] {techs}\n"
                           f"      {p.get('insight','?')}\n")
            kb_ctx += "\n"
        print(f"  KB: {len(kb_results)} relevant entries")
    row["kb_entries_used"] = len(kb_results)

    # ── optimization rounds ────────────────────────────────────────────
    # `applied` is the cumulative stack of techniques currently in the BEST kernel.
    # It is what makes rounds additive: the planner is told what must be preserved,
    # the coder is told to keep it and layer on top. Without this the agent swaps
    # techniques and each round undoes the last.
    # target_met is NOT decided in the loop — it is settled after the run-off, on a
    # torch.compile timing taken in the same sitting as the winning kernel.
    history, planner_other, target_met = [], 0, False
    applied: list[str] = []
    err_fb = ""
    cc = torch.cuda.get_device_capability()
    cc_int = cc[0] * 10 + cc[1]

    # Every kernel that passed validation, kept for the end-of-run run-off. Each
    # round's speedup is measured at a different moment and this GPU throttles, so
    # per-round ratios are NOT comparable to each other — the finalists get re-raced
    # back-to-back before we declare a winner. Round 0 (the naive port) is a
    # candidate too: it is the thing everything else must actually beat.
    candidates: list[dict] = [
        {"round": 0, "mod": mod, "cuda": cuda_src, "cpp": cpp_src,
         "applied": [], "speedup": best_speedup}
    ]

    for r in range(1, ROUNDS + 1):
        history_ctx = ""
        if history:
            history_ctx = "Previous attempts this session:\n"
            for h in history:
                if h["result"] == "success":
                    history_ctx += f"  Round {h['round']}: {h['strategy']} → {h['speedup']:.2f}x\n"
                else:
                    history_ctx += f"  Round {h['round']}: {h['strategy']} → FAILED ({h['result']})\n"
            history_ctx += "\n"

        best_ctx = f"Current best: {best_speedup:.2f}x vs PyTorch eager — beat it.\n\n"

        strategy = "agent_choice"
        add_techs: list[str] = []
        guidance = ""
        if USE_PLANNER and planner:
            plan = await planner.plan(
                classification, metrics, kb_results, history, best_speedup,
                applied=applied, cc=cc_int,
            )
            strategy = plan.get("strategy", "agent_choice")
            add_techs = plan.get("add_techniques", []) or []
            if plan.get("strategy_is_other"):
                planner_other += 1
            guidance = (
                "PLAN FOR THIS ROUND — implement it, and reason beyond it if the code or the\n"
                "metrics tell you something it missed:\n"
                + json.dumps(plan, indent=2) + "\n\n"
            )
            keep = plan.get("keep_techniques", [])
            print(f"  round {r}: plan → add {add_techs or [strategy]}"
                  + (f" | keep {keep}" if keep else ""))
        else:
            print(f"  round {r}: asking coder...")

        # Techniques this round is trying to ADD. On failure these are what the KB
        # records as having failed; on success they become part of the stack.
        attempted = add_techs or ([strategy] if strategy != "agent_choice" else [])

        out = await coder.optimize(
            best_cuda, best_cpp,
            round_num=r, rounds=ROUNDS, gpu_arch=GPU_ARCH,
            pytorch_source=task.source, guidance=guidance, metrics_ctx=metrics_ctx,
            kb_ctx=kb_ctx, history_ctx=history_ctx, best_ctx=best_ctx,
            error_feedback=err_fb, signature_ctx=signature_ctx, applied=applied,
        )
        if not out:
            print(f"  round {r}: unparseable reply")
            history.append({"round": r, "result": "empty_response", "strategy": strategy})
            err_fb = "Your reply could not be parsed as the required JSON object."
            continue

        # compile
        mod, cerr = compile_extension(
            out["cuda_source"], out["cpp_source"], timeout_s=EVAL.compile_timeout_s
        )
        if mod is None:
            print(f"  round {r}: COMPILE FAILED — {cerr.splitlines()[0][:60] if cerr else '?'}")
            history.append({"round": r, "result": "compile_failed", "strategy": strategy})
            err_fb = f"COMPILER ERROR:\n{cerr[:2000]}"
            # On a failure, record the techniques that were ATTEMPTED — that is what
            # a future planner needs in order to avoid them.
            await _reflect_and_store(kernel_name=name, bottleneck=hint, round_num=r,
                                     code=out["cuda_source"], result="compile_failed",
                                     error=cerr[:200], applied=attempted)
            continue

        # correctness + determinism (PyTorch is the oracle)
        vok, vmsg = validate_ext(mod, task)
        if not vok:
            kind = "unstable" if "ondeterministic" in vmsg else "validation_failed"
            print(f"  round {r}: {kind.upper()} — {vmsg[:60]}")
            history.append({"round": r, "result": kind, "strategy": strategy})
            err_fb = f"VALIDATION FAILED:\n{vmsg}"
            await _reflect_and_store(kernel_name=name, bottleneck=hint, round_num=r,
                                     code=out["cuda_source"], result=kind, error=vmsg[:200],
                                     applied=attempted)
            continue

        # benchmark
        b = benchmark_interleaved(mod, task, warmup=EVAL.warmup, runs=EVAL.runs)
        speedup = b["speedup_vs_eager"]
        vs_compile = (compile_ms / b["karma_ms"]) if (compile_ms and b["karma_ms"] > 0) else 0.0
        err_fb = ""

        cov = (b["karma_std"] / b["karma_ms"] * 100) if b["karma_ms"] else 0.0
        print(f"  round {r}: ✓ {b['karma_ms']:.3f}ms (±{b['karma_std']:.3f}, {cov:.1f}% CoV) "
              f"→ {speedup:.2f}x vs eager"
              + (f", {vs_compile:.2f}x vs compile" if vs_compile else ""))
        history.append({"round": r, "result": "success", "strategy": strategy, "speedup": speedup})

        # The stack THIS kernel carries: whatever the running best had, plus what
        # this round added. Computed unconditionally, because the per-round ratio
        # is drift-contaminated and the run-off below may well overturn it. Deriving
        # it only inside `if speedup > best_speedup` lost the technique record for
        # any round that lost the drifty comparison but won the run-off — which is
        # exactly the case the run-off exists to catch.
        round_stack = list(applied)
        for t in attempted:
            if t not in round_stack:
                round_stack.append(t)

        if speedup > best_speedup:
            best_cuda, best_cpp = out["cuda_source"], out["cpp_source"]
            best_speedup, best_ms, best_std, best_round = speedup, b["karma_ms"], b["karma_std"], r
            applied = list(round_stack)   # later rounds must preserve these
            print(f"    new best — stack now: {applied or ['(unnamed)']}")

        # Keep it as a finalist regardless: this round's ratio and an earlier
        # round's are not directly comparable (see the run-off below).
        candidates.append({
            "round": r, "mod": mod,
            "cuda": out["cuda_source"], "cpp": out["cpp_source"],
            "applied": round_stack, "speedup": speedup,
        })

        await _reflect_and_store(kernel_name=name, bottleneck=hint, round_num=r,
                                 code=out["cuda_source"], result="success", speedup=speedup,
                                 applied=attempted or round_stack)

        # Early stop on beating torch.compile — but NOT on `vs_compile`, which is
        # drift-contaminated: compile_ms was measured once at the start and this
        # kernel is timed minutes later. Deciding on it once inflated 1.07x to
        # 1.38x. The honest number only exists after the run-off, so the target is
        # evaluated there.
        if compile_ms and vs_compile >= EVAL.target_speedup_vs_compile:
            print(f"  provisionally beating torch.compile ({vs_compile:.2f}x, drift-prone) "
                  f"— stopping; the run-off will confirm")
            break

    # ── run-off: re-race the finalists back-to-back ────────────────────
    # Per-round ratios were measured minutes apart on a GPU whose clocks drift, so
    # the round with the best *recorded* ratio is not necessarily the best kernel.
    # Re-measure them in one sitting, round-robin, and let that decide.
    # Only serious contenders: re-racing the naive round-0 kernel 30x costs minutes
    # (it can be an order of magnitude slower) and it cannot win. Cap by recorded
    # speedup — that ranking is drift-contaminated, but it is more than good enough
    # to separate contenders from no-hopers.
    finalists = sorted(candidates, key=lambda c: -c.get("speedup", 0.0))[:EVAL.runoff_finalists]

    # torch.compile races AS AN ARM, not as a number remembered from the start of
    # the run. compile_ms was captured in measure_baselines() minutes ago; dividing
    # it by a freshly-timed kernel once turned a true 1.07x into a reported 1.38x.
    refs = {"compile": base["compiled"]} if base.get("compiled") is not None else {}
    speedup_vs_compile = 0.0

    if finalists:
        print(f"  run-off: re-racing {len(finalists)} finalist(s)"
              + (" + torch.compile" if refs else "") + " back-to-back...")
        win = run_off(finalists, task, warmup=EVAL.warmup, runs=EVAL.runs, refs=refs)

        for c in win.get("runoff", []):
            mark = "  <-- winner" if c["round"] == win["round"] else ""
            print(f"    round {c['round']:>2}: {c['karma_ms']:8.3f}ms  "
                  f"{c['speedup_vs_eager']:.3f}x vs eager{mark}")
        for k, ms in win.get("ref_ms", {}).items():
            print(f"    {k:>8}: {ms:8.3f}ms")

        if win["round"] != best_round:
            print(f"    NOTE: run-off overturned round {best_round} in favour of "
                  f"round {win['round']} — the earlier ranking was thermal drift")

        best_cuda, best_cpp = win["cuda"], win["cpp"]
        best_round = win["round"]
        best_ms, best_std = win["karma_ms"], win["karma_std"]
        best_speedup = win["speedup_vs_eager"]
        applied = win.get("applied", applied)
        eager_ms = win["eager_ms"]           # freshest baseline, same sitting as the winner
        row["eager_ms"] = eager_ms

        # Both measured in the same sitting -> directly comparable.
        speedup_vs_compile = win.get("speedup_vs_ref", {}).get("compile", 0.0)
        if refs:
            row["compile_ms"] = win["ref_ms"]["compile"]
            compile_ms = win["ref_ms"]["compile"]

    # target_met is decided HERE, on the drift-free number, not on the provisional
    # per-round one that stopped the loop.
    target_met = bool(compile_ms and speedup_vs_compile >= EVAL.target_speedup_vs_compile)
    if compile_ms:
        verdict = "MET" if target_met else "not met"
        print(f"  vs torch.compile: {speedup_vs_compile:.3f}x "
              f"(target {EVAL.target_speedup_vs_compile}x — {verdict})")

    Path("kernels/results").mkdir(parents=True, exist_ok=True)
    Path(f"kernels/results/{name}.cu").write_text(best_cuda)
    Path(f"kernels/results/{name}_binding.cpp").write_text(best_cpp)

    # ── result ─────────────────────────────────────────────────────────
    row.update(
        best_ms=best_ms, best_ms_std=best_std, best_round=best_round,
        karma_ms=best_ms, karma_std=best_std,
        speedup_vs_eager=best_speedup, speedup_vs_compile=speedup_vs_compile,
        best_speedup=best_speedup, target_met=target_met,
        baseline_ms=eager_ms,
        rounds_total=len(history),
        compile_failures=sum(1 for h in history if h["result"] == "compile_failed"),
        validation_failures=sum(1 for h in history if h["result"] == "validation_failed"),
        unstable_rounds=sum(1 for h in history if h["result"] == "unstable"),
        strategies_tried=";".join(h["strategy"] for h in history),
        applied_techniques=";".join(applied),
        planner_used_other_count=planner_other,
        status="ok",
    )

    print(f"\n  RESULT: {best_speedup:.2f}x vs eager"
          + (f" | {speedup_vs_compile:.2f}x vs compile" if speedup_vs_compile else "")
          + f" | round={best_round} | stack={applied or ['(none)']}")
    return ExperimentResult(**row).model_dump()


# ── Core optimization loop (LEGACY: own/ and sglang/ standalone .cu) ────
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
            optimized = await coder.optimize_standalone(
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
                baseline=LEGACY_BASELINE,
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
                baseline=LEGACY_BASELINE,
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
                baseline=LEGACY_BASELINE,
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
                baseline=LEGACY_BASELINE,
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
    print(f"Found {len(files)} tasks at {kb_path}")
    Path("kernels/results").mkdir(parents=True, exist_ok=True)

    for py_file in files:
        if already_done(py_file.stem + ".cu"):
            print(f"\n  already in CSV — skipping {py_file.name}")
            continue

        try:
            row = await optimize_one_torch(py_file)
        except Exception as e:
            print(f"  ERROR on {py_file.name}: {e.__class__.__name__}: {e}")
            continue

        row["source"] = f"kernelbench_l{level}"
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
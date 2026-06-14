# run_experiments.py
"""
Headless experiment runner for KARMA.
Runs optimization loop on KernelBench / SGLang / own kernels.
Saves results to CSV after every kernel (crash-safe).
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import csv
import re
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

# pipeline
from pipeline.compiler import compile_cuda
from pipeline.pre_flight import pre_flight
from pipeline.validator import run_validation
from pipeline.benchmarker import benchmark

# agent
from Agents.coder import safe_chat, runner, USER_ID, SESSION_ID

# knowledge (novel contribution)
from knowledge.store import KnowledgeBase
from knowledge.reflector import reflect

# ── Config ─────────────────────────────────────────────────────────────
RESULTS_CSV  = "results/experiments.csv"
ROUNDS       = 3        # 3 for lab sessions, 5 for paper runs
WARMUP_RUNS  = 3
TIMED_RUNS   = 10
TIMEOUT_SEC  = 10
GPU_ARCH     = "sm_86"
USE_KB       = True      # set False for ablation: no knowledge base
USE_REFLECT  = True      # set False for ablation: no reflection agent

# ── Initialize KnowledgeBase ───────────────────────────────────────────
kb = KnowledgeBase() if USE_KB else None


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
    best_ms:              float = 0.0
    best_speedup:         float = 0.0
    best_round:           int   = 0
    rounds_total:         int   = 0
    compile_failures:     int   = 0
    validation_failures:  int   = 0
    kb_entries_used:      int   = 0
    pytorch_ref:          Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────
def extract_cuda_code(text: str) -> str:
    # strip markdown fences and "cppcopy" artifacts
    for fence in ["```cpp", "```cuda", "```c", "cppcopy", "```"]:
        if fence in text:
            after = text.split(fence, 1)[1]
            end = after.find("```")
            text = after[:end] if end != -1 else after
            break
    # find actual CUDA start
    if "#include" in text:
        return text[text.find("#include"):].strip()
    if "__global__" in text:
        return text[text.find("__global__"):].strip()
    return text.strip()


def append_result(row: dict):
    Path("results").mkdir(exist_ok=True)
    exists = Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
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

    # ── baseline timing (once, before loop) ────────────────────────
    print("  benchmarking baseline...")
    baseline_ms = 0.0
    base_ok, base_bin = compile_cuda(str(kernel_path))
    if base_ok:
        baseline_ms = benchmark(base_bin)
        print(f"  baseline: {baseline_ms:.3f}ms")
    else:
        print("  baseline compile failed — speedup will be 0")

    # ── KnowledgeBase retrieval ────────────────────────────────────
    kb_ctx = ""
    kb_used = 0
    if USE_KB and kb:
        kb_results = kb.retrieve(bottleneck=hint, kernel_name=name)
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
                elif h["result"] == "empty_response":
                    history_ctx += f"  Round {h['round']}: agent returned no valid code\n"
            history_ctx += "\n"

        best_ctx = ""
        if best_speedup:
            best_ctx = f"Current best: {best_speedup:.2f}x — beat this.\n\n"

        prompt = (
            f"You are a CUDA expert optimizing for RTX A4000 ({GPU_ARCH} Ampere).\n"
            f"Round {r} of {ROUNDS}. {strategy_hint}\n\n"
            f"{metrics_ctx}"
            f"{kb_ctx}"
            f"{history_ctx}"
            f"{best_ctx}"
            f"HARD CONSTRAINTS:\n"
            f"- float32 only. No half / half2 / __half / fp16.\n"
            f"- No cuda/cmath, no std::complex headers.\n"
            f"- Apply ONE focused optimization per round.\n"
            f"- If previous round had a compile error, fix THAT error first.\n\n"
            f"OUTPUT RULES — no exceptions:\n"
            f"- Return ONLY the raw .cu file.\n"
            f"- First character must be '#' (from #include).\n"
            f"- No markdown, no backticks, no explanation.\n"
            f"- Must compile: nvcc -O2 -arch={GPU_ARCH} -lcublas -lcurand\n\n"
            f"KERNEL TO OPTIMIZE:\n{best_code}"
        )

        raw = await safe_chat(prompt, runner, USER_ID, SESSION_ID)
        optimized = extract_cuda_code(raw)

        if not optimized or len(optimized) < 30:
            print(f"  round {r}: invalid/empty output — skipping")
            history.append({"round": r, "result": "empty_response"})
            continue

        tmp = Path(f"kernels/tmp_exp_r{r}.cu")
        tmp.write_text(optimized)

        # ── compile ────────────────────────────────────────────────
        print(f"  round {r}: compiling...")
        ok, result = compile_cuda(str(tmp))

        if not ok:
            err = str(result)[:400]
            print(f"  round {r}: COMPILE FAILED — {err[:80]}")
            history.append({"round": r, "result": "compile_failed", "error": err})
            # feed error into next round
            best_code = (
                "// COMPILE ERROR — fix these exact errors first:\n"
                + "\n".join(f"// {line}" for line in err.splitlines()[:10])
                + f"\n\n{optimized}"
            )
            # reflect on failure
            if USE_REFLECT and kb:
                insight = await reflect(
                    safe_chat, runner, USER_ID, SESSION_ID,
                    kernel_name=name, bottleneck=hint, round_num=r,
                    code=optimized, result="compile_failed", error=err
                )
                kb.store(insight)
                print(f"    KB stored: {insight['strategy_used'][:50]}")
            continue

        # ── validate ───────────────────────────────────────────────
        print(f"  round {r}: validating...")
        val_ok, val_msg = run_validation(result)

        if not val_ok:
            msg = str(val_msg)[:200]
            print(f"  round {r}: VALIDATION FAILED — {msg[:60]}")
            history.append({"round": r, "result": "validation_failed"})
            best_code = (
                f"// VALIDATION FAILED — output wrong.\n"
                f"// Message: {msg}\n"
                f"// Fix the math. Do not restructure the kernel.\n\n"
                + optimized
            )
            if USE_REFLECT and kb:
                insight = await reflect(
                    safe_chat, runner, USER_ID, SESSION_ID,
                    kernel_name=name, bottleneck=hint, round_num=r,
                    code=optimized, result="validation_failed", error=msg
                )
                kb.store(insight)
                print(f"    KB stored: {insight['strategy_used'][:50]}")
            continue

        # ── benchmark ──────────────────────────────────────────────
        opt_ms = benchmark(result)
        if opt_ms <= 0 or baseline_ms <= 0:
            print(f"  round {r}: benchmark returned 0 — binary may not print 'GPU Time'")
            speedup = 0.0
        else:
            speedup = round(baseline_ms / opt_ms, 4)

        print(f"  round {r}: ✓ {opt_ms:.3f}ms → {speedup:.2f}x speedup")
        history.append({"round": r, "speedup": speedup, "result": "success"})

        # track best
        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round   = r
            best_code    = optimized
            Path("kernels/results").mkdir(exist_ok=True)
            Path(f"kernels/results/{name}").write_text(optimized)
            print(f"  new best — saved kernels/results/{name}")
        else:
            print(f"  round {r}: {speedup:.2f}x < best {best_speedup:.2f}x — discarded")
            # reset to best code for next round (don't build on worse code)
            best_code = source

        # reflect on success
        if USE_REFLECT and kb:
            insight = await reflect(
                safe_chat, runner, USER_ID, SESSION_ID,
                kernel_name=name, bottleneck=hint, round_num=r,
                code=optimized, result="success", speedup=speedup
            )
            kb.store(insight)
            print(f"    KB stored: {insight['strategy_used'][:50]}")

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

    print(f"\n  RESULT: best={'%.4f' % best_speedup if best_speedup else '0'}x  "
          f"round={best_round}  compile_fails={n_compile}  val_fails={n_val}  "
          f"kb_entries={kb.count() if kb else 0}")

    result = ExperimentResult(
        timestamp=datetime.now().isoformat(),
        kernel=name,
        source=str(kernel_path),
        bottleneck=hint,
        occupancy=occ,
        compute_throughput=comp,
        dram_throughput=dram,
        baseline_ms=baseline_ms,
        best_ms=round(baseline_ms / best_speedup, 4) if best_speedup and best_speedup > 0 else 0.0,
        best_speedup=best_speedup or 0.0,
        best_round=best_round or 0,
        rounds_total=len(history),
        compile_failures=n_compile,
        validation_failures=n_val,
        kb_entries_used=kb_used,
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

        # enforce GPU Time output format in baseline generation
        cuda_prompt = (
            f"Convert this PyTorch operation to a standalone CUDA kernel for RTX A4000 (sm_86).\n\n"
            f"PYTORCH REFERENCE:\n{pytorch_code}\n\n"
            f"MANDATORY REQUIREMENTS:\n"
            f"- Complete standalone .cu file with main()\n"
            f"- Use cudaEvent_t for timing. Print EXACTLY: printf(\"GPU Time: %f ms\\n\", elapsed_ms);\n"
            f"- Compare GPU output to CPU reference. Print SUCCESS if match (tolerance 1e-4), else FAILURE\n"
            f"- float32 only. No half types.\n"
            f"- Start with #include, no markdown, no backticks\n"
            f"- Must compile: nvcc -O2 -arch=sm_86 -lcublas -lcurand\n"
        )

        print(f"  generating baseline CUDA from PyTorch...")
        raw = await safe_chat(cuda_prompt, runner, USER_ID, SESSION_ID)
        baseline_cuda = extract_cuda_code(raw)

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
        print("Flags: --no-kb  --no-reflect")
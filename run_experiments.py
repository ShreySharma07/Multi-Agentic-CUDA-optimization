# run_experiments.py
import asyncio
import csv
import re
import subprocess
import importlib.util
from pathlib import Path
from datetime import datetime

from pipeline.compiler import compile_cuda
from pipeline.pre_flight import pre_flight
from pipeline.validator import run_validation
from Agents.coder import safe_chat, runner, USER_ID, SESSION_ID


from pydantic import BaseModel
from typing import Optional

class ExperimentResult(BaseModel):
    timestamp: str
    kernel: str
    source: str
    bottleneck: str
    occupancy: float        # was str
    compute_throughput: float  # was str
    dram_throughput: float     # was str
    baseline_ms: float
    best_speedup: float = 0.0      # default 0 instead of None
    best_round:   int   = 0
    best_ms:      float = 0.0
    rounds_total: int
    compile_failures: int
    validation_failures: int
    pytorch_ref: Optional[str] = None


RESULTS_CSV = "results/experiments.csv"
GPU_ARCH    = "sm_86"
WARMUP_RUNS = 3    # was 10
TIMED_RUNS  = 10   # was 30
TIMEOUT_SEC = 8    # was 20
ROUNDS      = 3    # was 5 — for quick runs, use 3

# ── Code extraction ────────────────────────────────────────────────────
def extract_cuda_code(text: str) -> str:
    text = text.replace("cppcopy", "").replace("```", "")
    if "#include" in text:
        return text[text.find("#include"):].strip()
    return text.strip()

# ── Real benchmarker ───────────────────────────────────────────────────
def benchmark(binary_path: str) -> float:
    """
    Run binary, parse 'GPU Time: X ms' from stdout.
    Returns mean ms over TIMED_RUNS, or 0.0 on failure/timeout.
    """
    def run_once():
        try:
            r = subprocess.run(
                [binary_path],
                capture_output=True, text=True,
                timeout=TIMEOUT_SEC
            )
            m = re.search(r'GPU\s*Time[:\s]+([\d.]+)', r.stdout)
            if m:
                return float(m.group(1))
        except subprocess.TimeoutExpired:
            pass
        except Exception:
            pass
        return None

    # warmup
    for _ in range(WARMUP_RUNS):
        run_once()

    times = []
    for _ in range(TIMED_RUNS):
        t = run_once()
        if t is not None:
            times.append(t)

    if not times:
        print(f"    [bench] WARNING: no timing output from {Path(binary_path).name}")
        return 0.0

    mean = sum(times) / len(times)
    return round(mean, 4)

# ── CSV writer ─────────────────────────────────────────────────────────
def append_result(row: dict):
    Path("results").mkdir(exist_ok=True)
    file_exists = Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  → saved to {RESULTS_CSV}")

# ── Core optimization loop (headless) ──────────────────────────────────
async def optimize_one(kernel_path: Path, source: str = None) -> dict:
    name   = kernel_path.name
    source = source or kernel_path.read_text()

    print(f"\n{'='*60}")
    print(f"KERNEL: {name}")
    print(f"{'='*60}")

    # ── pre-flight ─────────────────────────────────────────────────────
    print("  running pre-flight...")
    try:
        pf = pre_flight(source)
    except Exception as e:
        pf = {"status": "error"}
        print(f"  pre-flight exception: {e}")

    metrics_ctx = ""
    hint = "unknown"
    occ = comp = dram = "0"

    if pf.get("status") == "success":
        m    = pf["metrics"]
        try:
            occ  = float(str(m.get("occupancy",         "0")).replace("%",""))
            comp = float(str(m.get("compute_throughput","0")).replace("%",""))
            dram = float(str(m.get("dram_throughput",   "0")).replace("%",""))
        except Exception:
            occ = comp = dram = 0.0
        try:
            hint = "memory-bound" if float(str(dram)) > float(str(comp)) else "compute-bound"
        except Exception:
            pass
        metrics_ctx = (
            f"Hardware profile (Nsight Compute):\n"
            f"  Occupancy: {occ}%\n"
            f"  Compute throughput: {comp}%\n"
            f"  DRAM throughput: {dram}%\n"
            f"  Bottleneck: {hint}\n"
        )
        print(f"  pre-flight ok · occ={occ}% dram={dram}% → {hint}")
    else:
        print(f"  pre-flight unavailable ({pf.get('stage','?')}) — no metrics context")

    # ── baseline timing (compile original once, time it) ───────────────
    print("  benchmarking baseline...")
    baseline_ms = 0.0
    _base_ok, _base_bin = compile_cuda(str(kernel_path))
    if _base_ok:
        baseline_ms = benchmark(_base_bin)
        print(f"  baseline: {baseline_ms:.3f}ms")
    else:
        print("  baseline compile failed — speedup will be 0")

    # ── optimization loop ──────────────────────────────────────────────
    best_speedup = None
    best_round   = None
    best_code    = source
    history      = []

    for r in range(1, ROUNDS + 1):
        print(f"  round {r}/{ROUNDS}: asking agent...")

        # build history string for prompt
        history_ctx = ""
        if history:
            history_ctx = "Previous attempts this session:\n"
            for h in history:
                if h["result"] == "success":
                    history_ctx += f"  Round {h['round']}: {h['speedup']:.2f}x speedup\n"
                elif h["result"] == "compile_failed":
                    history_ctx += f"  Round {h['round']}: COMPILE FAILED — {h.get('error','')[:120]}\n"
                elif h["result"] == "validation_failed":
                    history_ctx += f"  Round {h['round']}: VALIDATION FAILED — math wrong\n"
                elif h["result"] == "empty_response":
                    history_ctx += f"  Round {h['round']}: agent returned no code\n"
            history_ctx += "\n"

        best_ctx = ""
        if best_speedup:
            best_ctx = f"Current best: {best_speedup:.2f}x speedup — improve on this.\n\n"

        strategy_hint = {
            "memory-bound": (
                "Focus on: coalesced global memory access, "
                "vectorized loads (float4), shared memory tiling."
            ),
            "compute-bound": (
                "Focus on: fast math (__expf, __fdividef), "
                "loop unrolling, reducing register pressure."
            ),
            "unknown": "Apply the most impactful optimization you can identify.",
        }.get(hint, "")

        prompt = (
            f"You are a CUDA expert optimizing for RTX A4000 ({GPU_ARCH} Ampere).\n"
            f"Round {r} of {ROUNDS}.\n\n"
            f"{metrics_ctx}"
            f"Optimization hint: {strategy_hint}\n\n"
            f"{history_ctx}"
            f"{best_ctx}"
            f"HARD CONSTRAINTS:\n"
            f"- float32 only. No half / half2 / __half / fp16 types.\n"
            f"- No cuda/cmath, no std::complex headers.\n"
            f"- Apply ONE focused optimization — do not rewrite everything.\n"
            f"- All pointer arguments remain float*.\n\n"
            f"OUTPUT RULES — no exceptions:\n"
            f"- Return ONLY the raw .cu file.\n"
            f"- First character must be '#' (from #include).\n"
            f"- No markdown, no backticks, no explanation.\n"
            f"- Must compile: nvcc -O2 -arch={GPU_ARCH}\n"
            f"- If previous round had a compile error, fix THOSE exact errors first.\n\n"
            f"KERNEL TO OPTIMIZE:\n{best_code}"
        )

        raw       = await safe_chat(prompt, runner, USER_ID, SESSION_ID)
        optimized = extract_cuda_code(raw)

        # guard: reject empty or non-CUDA output
        if not optimized or not optimized.startswith("#include"):
            print(f"  round {r}: invalid output — skipping")
            history.append({"round": r, "result": "empty_response"})
            continue

        tmp = Path(f"kernels/tmp_exp_r{r}.cu")
        tmp.write_text(optimized)

        # compile
        print(f"  round {r}: compiling...")
        ok, result = compile_cuda(str(tmp))

        if not ok:
            err = str(result)[:400]
            print(f"  round {r}: COMPILE FAILED — {err[:80]}")
            history.append({"round": r, "result": "compile_failed", "error": err})
            # feed error into next round's code context
            best_code = (
                "// PREVIOUS COMPILE ERROR — fix these before any other change:\n"
                + "\n".join(f"// {line}" for line in err.splitlines())
                + f"\n\n{optimized}"
            )
            continue

        # validate correctness
        print(f"  round {r}: validating...")
        val_ok, val_msg = run_validation(result)

        if not val_ok:
            print(f"  round {r}: VALIDATION FAILED — {str(val_msg)[:60]}")
            history.append({"round": r, "result": "validation_failed"})
            best_code = (
                f"// VALIDATION FAILED — output does not match CPU baseline.\n"
                f"// Validation message: {str(val_msg)[:200]}\n"
                f"// Fix the math. Do not change the kernel structure.\n\n"
                + optimized
            )
            continue

        # benchmark
        opt_ms = benchmark(result)
        if opt_ms <= 0 or baseline_ms <= 0:
            print(f"  round {r}: benchmark returned 0 — binary may not print 'GPU Time'")
            speedup = 0.0
        else:
            speedup = round(baseline_ms / opt_ms, 4)

        print(f"  round {r}: ✓ {opt_ms:.3f}ms → {speedup:.2f}x speedup")
        history.append({"round": r, "speedup": speedup, "result": "success"})

        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round   = r
            best_code    = optimized
            Path("kernels/results").mkdir(exist_ok=True)
            Path(f"kernels/results/{name}").write_text(optimized)
            print(f"  new best — saved kernels/results/{name}")
        else:
            print(f"  round {r}: {speedup:.2f}x < best {best_speedup:.2f}x — discarded")

        # convergence: stop if last 2 successes improved by < 1%
        successes = [h for h in history if h.get("result") == "success"]
        if len(successes) >= 2:
            prev, curr = successes[-2]["speedup"], successes[-1]["speedup"]
            if abs(curr - prev) < 0.01:
                print("  converged — stopping early")
                break

    n_compile = sum(1 for h in history if h["result"] == "compile_failed")
    n_val     = sum(1 for h in history if h["result"] == "validation_failed")
    # replace the print at the end
    print(f"\n  RESULT: best={'%.4f' % best_speedup if best_speedup else 'None'}x  "
      f"round={best_round}  compile_fails={n_compile}  val_fails={n_val}")

    result = ExperimentResult(
    timestamp=datetime.now().isoformat(),
    kernel=name,
    source=str(kernel_path),
    bottleneck=hint,
    occupancy=occ,
    compute_throughput=comp,
    dram_throughput=dram,
    baseline_ms=baseline_ms,
    best_ms=round(baseline_ms / best_speedup, 4) if best_speedup else 0.0,
    best_speedup=best_speedup or 0.0,
    best_round=best_round or 0,
    rounds_total=len(history),
    compile_failures=n_compile,
    validation_failures=n_val,
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
        row = await optimize_one(kp)
        append_result(row)


def find_kernelbench_path() -> Path | None:
    candidates = [
        Path("KernelBench/kernelbench/dataset"),
        Path("KernelBench/KernelBench/dataset"),
        Path("KernelBench/KernelBench"),
        Path("KernelBench/data"),
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    for p in Path("KernelBench").rglob("level1"):
        return p.parent
    return None

# in run_kernelbench(), skip kernels already in results CSV
def already_done(kernel_name: str) -> bool:
    if not Path(RESULTS_CSV).exists():
        return False
    with open(RESULTS_CSV) as f:
        return any(kernel_name in line for line in f)


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

        print(f"\n{'='*60}\nKernelBench: {py_file.name}")
        pytorch_code = py_file.read_text()

        cuda_prompt = (
            f"Convert this PyTorch operation to a standalone CUDA kernel for RTX A4000 (sm_86).\n\n"
            f"PYTORCH REFERENCE:\n{pytorch_code}\n\n"
            f"REQUIREMENTS:\n"
            f"- Complete standalone .cu file with main()\n"
            f"- main() allocates data, runs kernel, compares to CPU, prints 'GPU Time: X ms'\n"
            f"- Prints 'SUCCESS' if GPU and CPU outputs match (tolerance 1e-4), else 'FAILURE'\n"
            f"- float32 only\n"
            f"- Start with #include, no markdown, no backticks\n"
            f"- Must compile: nvcc -O2 -arch=sm_86\n"
        )

        print("  generating baseline CUDA from PyTorch...")
        raw           = await safe_chat(cuda_prompt, runner, USER_ID, SESSION_ID)
        baseline_cuda = extract_cuda_code(raw)

        if not baseline_cuda.startswith("#include"):
            print("  failed to generate baseline — skipping")
            continue

        cu_name = py_file.stem + ".cu"
        if already_done(cu_name):
            print(f"  already in CSV — skipping")
            continue

        cu_path = Path(f"kernels/kernelbench/{cu_name}")
        cu_path.write_text(baseline_cuda)

        ok, err = compile_cuda(str(cu_path))
        if not ok:
            print(f"  baseline compile failed — skipping: {str(err)[:80]}")
            continue

        print("  baseline compiled — optimizing...")
        row = await optimize_one(cu_path, source=baseline_cuda)
        row["source"]       = f"kernelbench_l{level}"
        row["pytorch_ref"]  = py_file.name
        append_result(row)


def find_sglang_kernels() -> list[Path]:
    all_cu       = list(Path("sglang").rglob("*.cu"))
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
        print(f"  copied: {kp.name}")

    for kp in sorted(Path("kernels/sglang").glob("*.cu"))[:max_kernels]:
        row = await optimize_one(kp)
        row["source"] = "sglang"
        append_result(row)


# ── Entry point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "own"

    if mode == "own":
        asyncio.run(run_own_kernels())
    elif mode == "kernelbench":
        level = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        asyncio.run(run_kernelbench(level=level, max_kernels=count))
    elif mode == "sglang":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        asyncio.run(run_sglang(max_kernels=count))
    else:
        print("Usage: python run_experiments.py [own|kernelbench|sglang] [level] [max_kernels]")
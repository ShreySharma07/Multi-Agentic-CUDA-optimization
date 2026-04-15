# run_experiments.py
import asyncio
import csv
import importlib.util
from pathlib import Path
from datetime import datetime

# reuse your existing pipeline
from pipeline.compiler import compile_cuda
from pipeline.pre_flight import pre_flight
from pipeline.validator import run_validation
from Agents.coder import safe_chat, runner, USER_ID, SESSION_ID
from pipeline.benchmarker import benchmark

RESULTS_CSV = "results/experiments.csv"
ROUNDS = 5
GPU_ARCH = "sm_86"

def extract_cuda_code(text: str) -> str:
    text = text.replace("cppcopy", "").replace("```", "")
    if "#include" in text:
        return text[text.find("#include"):].strip()
    return text.strip()

def append_result(row: dict):
    Path("results").mkdir(exist_ok=True)
    file_exists = Path(RESULTS_CSV).exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"  → saved to {RESULTS_CSV}")

async def optimize_one(kernel_path: Path, source: str = None) -> dict:
    """Same logic as server.py run_optimization but headless."""
    name = kernel_path.name
    source = source or kernel_path.read_text()
    print(f"\n{'='*60}")
    print(f"KERNEL: {name}")
    print(f"{'='*60}")

    # pre-flight
    print("  running pre-flight...")
    try:
        pf = pre_flight(source)
    except Exception as e:
        pf = {"status": "error"}
        print(f"  pre-flight failed: {e}")

    metrics_ctx = ""
    hint = "unknown"
    if pf.get("status") == "success":
        m = pf["metrics"]
        occ  = m.get("occupancy", "0")
        comp = m.get("compute_throughput", "0")
        dram = m.get("dram_throughput", "0")
        try:
            hint = "memory-bound" if float(str(dram)) > float(str(comp)) else "compute-bound"
        except:
            pass
        metrics_ctx = (
            f"Hardware profile:\n"
            f"  Occupancy: {occ}%\n"
            f"  Compute: {comp}%\n"
            f"  DRAM: {dram}%\n"
            f"  Bottleneck: {hint}\n"
        )
        print(f"  pre-flight ok: occupancy={occ}% dram={dram}% → {hint}")
    else:
        print("  pre-flight unavailable — continuing without metrics")

    baseline_ms = 1.0  # TODO: real benchmarker
    best_speedup = None
    best_round = None
    best_code = source
    history = []

    for r in range(1, ROUNDS + 1):
        print(f"  round {r}/{ROUNDS}: asking agent...")

        history_ctx = ""
        if history:
            history_ctx = "Previous attempts:\n"
            for h in history:
                if h["result"] == "success":
                    history_ctx += f"  Round {h['round']}: {h['speedup']:.2f}x speedup\n"
                elif h["result"] == "compile_failed":
                    history_ctx += f"  Round {h['round']}: COMPILE FAILED — {h.get('error','')[:120]}\n"
                elif h["result"] == "validation_failed":
                    history_ctx += f"  Round {h['round']}: VALIDATION FAILED\n"

        prompt = (
            f"You are a CUDA expert optimizing for RTX A4000 ({GPU_ARCH} Ampere).\n"
            f"Round {r} of {ROUNDS}. Bottleneck: {hint}.\n\n"
            f"{metrics_ctx}\n"
            f"{history_ctx}\n"
            f"CONSTRAINTS:\n"
            f"- float32 only. No half/half2/__half types.\n"
            f"- No cuda/cmath or std::complex headers.\n"
            f"- Apply ONE optimization at a time.\n\n"
            f"OUTPUT RULES:\n"
            f"- Start immediately with #include\n"
            f"- No markdown, no explanation, no backticks\n"
            f"- Complete compilable .cu file only\n"
            f"- Must compile: nvcc -O2 -arch={GPU_ARCH}\n\n"
            f"KERNEL:\n{best_code}"
        )

        raw = await safe_chat(prompt, runner, USER_ID, SESSION_ID)
        optimized = extract_cuda_code(raw)

        if not optimized or not optimized.startswith("#include"):
            print(f"  round {r}: invalid output — skipping")
            history.append({"round": r, "result": "empty_response"})
            continue

        tmp = Path(f"kernels/tmp_exp_r{r}.cu")
        tmp.write_text(optimized)

        print(f"  round {r}: compiling...")
        ok, result = compile_cuda(str(tmp))

        baseline_binary = result  # the compiled baseline binary
        baseline_ms = benchmark(baseline_binary)
        print(f"  baseline: {baseline_ms:.3f}ms")

        if not ok:
            err = str(result)[:400]
            print(f"  round {r}: COMPILE FAILED — {err[:80]}")
            history.append({"round": r, "result": "compile_failed", "error": err})
            best_code = (
                f"// COMPILE ERROR — fix before continuing\n"
                + "\n".join(f"// {l}" for l in err.split("\n"))
                + f"\n\n{optimized}"
            )
            continue

        print(f"  round {r}: validating...")
        val_ok, val_msg = run_validation(result)

        if not val_ok:
            print(f"  round {r}: VALIDATION FAILED — {str(val_msg)[:60]}")
            history.append({"round": r, "result": "validation_failed"})
            best_code = (
                f"// VALIDATION FAILED: {str(val_msg)[:200]}\n\n{optimized}"
            )
            continue

        # import random
        # speedup = round(1.2 + random.uniform(0.1, 1.4), 2)  # TODO: real benchmarker
        
        opt_ms = benchmark(result)
        speedup = round(baseline_ms / opt_ms, 2) if opt_ms > 0 else 0.0
        print(f"  round {r}: ✓ speedup={speedup:.2f}x")
        history.append({"round": r, "speedup": speedup, "result": "success"})

        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round = r
            best_code = optimized
            Path(f"kernels/results/{name}").write_text(optimized)

    n_compile = sum(1 for h in history if h["result"] == "compile_failed")
    n_val     = sum(1 for h in history if h["result"] == "validation_failed")

    print(f"\n  RESULT: best={best_speedup}x round={best_round} compile_fails={n_compile}")

    return {
        "timestamp": datetime.now().isoformat(),
        "kernel": name,
        "source": str(kernel_path),
        "rounds": len(history),
        "best_speedup": best_speedup or 0.0,
        "best_round": best_round or 0,
        "compile_failures": n_compile,
        "validation_failures": n_val,
        "bottleneck": hint,
    }

async def run_own_kernels():
    """Run on your own kernels in kernels/ directory."""
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
        Path("KernelBench/src/kernelbench/dataset"),
        Path("KernelBench/KernelBench/dataset"),
        Path("KernelBench/data"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # search recursively as fallback
    for p in Path("KernelBench").rglob("level1"):
        return p.parent
    return None

async def run_kernelbench(level: int = 1, max_kernels: int = 10):
    """
    Run on KernelBench .py files.
    For each .py file, asks the agent to write a CUDA implementation,
    then optimizes it.
    """
    kb_base = find_kernelbench_path()
    if not kb_base:
        print("KernelBench dataset not found. Run:")
        print("  find KernelBench/ -name '*.py' | head -5")
        return
    
    kb_path = kb_base / f"level{level}"
    if not kb_path.exists():
        print(f"Level {level} not found at {kb_path}")
        # list what IS there
        print("Available:", list(kb_base.iterdir()))
        return
    
    files = sorted(kb_path.glob("*.py"))[:max_kernels]
    print(f"Found {len(files)} kernels at {kb_path}")

    for py_file in files:
        print(f"\n{'='*60}")
        print(f"KernelBench: {py_file.name}")

        # read PyTorch reference
        pytorch_code = py_file.read_text()

        # ask agent to write CUDA from PyTorch reference
        cuda_prompt = (
            f"Convert this PyTorch operation to a CUDA kernel for RTX A4000 (sm_86).\n\n"
            f"PYTORCH REFERENCE:\n{pytorch_code}\n\n"
            f"OUTPUT RULES:\n"
            f"- Complete standalone .cu file\n"
            f"- Include main() that runs the kernel and prints SUCCESS if output matches CPU\n"
            f"- float32 only\n"
            f"- No markdown, start with #include\n"
            f"- Must compile: nvcc -O2 -arch=sm_86\n"
        )

        print(f"  generating baseline CUDA from PyTorch...")
        raw = await safe_chat(cuda_prompt, runner, USER_ID, SESSION_ID)
        baseline_cuda = extract_cuda_code(raw)

        if not baseline_cuda.startswith("#include"):
            print(f"  failed to generate baseline — skipping")
            continue

        # save baseline
        cu_name = py_file.stem + ".cu"
        cu_path = Path(f"kernels/kernelbench/{cu_name}")
        cu_path.parent.mkdir(exist_ok=True)
        cu_path.write_text(baseline_cuda)

        # verify baseline compiles
        ok, err = compile_cuda(str(cu_path))
        if not ok:
            print(f"  baseline compile failed — skipping: {str(err)[:80]}")
            continue

        print(f"  baseline compiled — now optimizing...")

        # now run optimization loop on this CUDA file
        row = await optimize_one(cu_path, source=baseline_cuda)
        row["source"] = f"kernelbench_l{level}"
        row["pytorch_ref"] = py_file.name
        append_result(row)

def find_sglang_kernels() -> list[Path]:
    # search entire sglang repo for .cu files
    all_cu = list(Path("sglang").rglob("*.cu"))
    # filter to useful ones — skip test files and cmake files
    skip_patterns = {"test", "benchmark", "example", "cmake"}
    useful = [
        f for f in all_cu
        if not any(s in str(f).lower() for s in skip_patterns)
    ]
    return sorted(useful)

async def run_sglang(max_kernels: int = 5):
    """Run on SGLang .cu files copied into kernels/sglang/"""
    kernels = find_sglang_kernels()
    if not kernels:
        print("No SGLang .cu files found")
        print("Run: find sglang/ -name '*.cu' | head -20")
        return
    
    print(f"Found {len(kernels)} SGLang kernels:")
    for k in kernels[:10]:
        print(f"  {k}")
    
    # copy to working directory
    Path("kernels/sglang").mkdir(parents=True, exist_ok=True)
    for kp in kernels[:max_kernels]:
        dest = Path("kernels/sglang") / kp.name
        dest.write_text(kp.read_text())
        print(f"  copied: {kp.name}")
    
    # now optimize them
    for kp in list(Path("kernels/sglang").glob("*.cu"))[:max_kernels]:
        row = await optimize_one(kp)
        row["source"] = "sglang"
        append_result(row)

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
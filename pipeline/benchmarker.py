# pipeline/benchmarker.py
import subprocess
import re
import os
import statistics

GPU_TIME_RE = re.compile(r"GPU Time:\s*([\d.]+)")
FAIL_RE = re.compile(r"\b(FAILURE|MISMATCH|INCORRECT)\b")


def benchmark(binary_path: str, warmup: int = 5, runs: int = 20, timeout: int = 10) -> dict:
    """
    Runs the compiled binary `warmup` times (discarded) then `runs` times,
    parsing the printed 'GPU Time: <ms>' line from each timed run.

    Returns:
      {
        "mean_ms": float,  # 0.0 if no run produced a usable time
        "std_ms":  float,  # population stddev across timed runs -- how noisy
                            # this measurement was, so a caller can tell a
                            # real speedup from run-to-run jitter
        "n":       int,    # number of timed runs that produced a value
        "stable":  bool,   # False if ANY timed run printed a failure token
      }

    BENCH_ONLY=1 is set in the child environment. Generated kernels are
    instructed (see run_experiments.py prompts) to skip their CPU-reference
    recompute when it's set, so the 20 timed runs only pay for the GPU
    kernel, not for re-validating on the CPU every pass -- that recompute is
    what run_validation() is for, on its own single run. Kernels that don't
    check the env var still work exactly as before (they just re-validate
    every timed run too); either way every run's output is scanned for a
    failure token, so a kernel that is only *sometimes* correct can't slip a
    fast-but-wrong run into the average undetected.
    """
    env = dict(os.environ, BENCH_ONLY="1")

    for _ in range(warmup):
        try:
            subprocess.run([binary_path], capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"mean_ms": 0.0, "std_ms": 0.0, "n": 0, "stable": False}

    times = []
    stable = True
    for _ in range(runs):
        try:
            r = subprocess.run(
                [binary_path], capture_output=True, text=True, timeout=timeout, env=env
            )
        except subprocess.TimeoutExpired:
            break

        out = (r.stdout or "") + (r.stderr or "")
        if FAIL_RE.search(out.upper()):
            stable = False

        match = GPU_TIME_RE.search(out)
        if match:
            times.append(float(match.group(1)))

    if not times:
        return {"mean_ms": 0.0, "std_ms": 0.0, "n": 0, "stable": stable}

    mean = sum(times) / len(times)
    std = statistics.pstdev(times) if len(times) > 1 else 0.0
    return {"mean_ms": mean, "std_ms": std, "n": len(times), "stable": stable}

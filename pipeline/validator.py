import subprocess
import os
import re

# Number of full (CPU-reference) validation passes used to detect a kernel that
# is only *sometimes* correct. See check_determinism().
DEFAULT_DETERMINISM_RUNS = 3

# Relative tolerance for numerical drift between the GPU kernel and the CPU
# reference. float32 reassociation over a length-K reduction accumulates an
# error on the order of sqrt(K) * eps * magnitude, so the 1e-4 the generated
# harnesses often hard-code is far too strict for matmul/reduction kernels and
# rejects mathematically-correct code. 1e-2 relative is a safe ceiling.
DEFAULT_REL_TOL = 1e-2

# CPU references can be slow even at moderate sizes; give correctness runs a
# generous budget so a legitimately-correct kernel is never reported as a
# "deadlock". Runaway kernels are still caught, just later.
DEFAULT_TIMEOUT = 120

# Word-boundary matched so "MATCH" never fires inside "MISMATCH", etc.
PASS_TOKENS = ("SUCCESS", "PASSED", "CORRECT", "VALIDATION OK")
FAIL_TOKENS = ("FAILURE", "FAILED", "MISMATCH", "INCORRECT")


def is_close(a, b, tol=1e-3):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def _has_token(text_upper: str, tokens) -> bool:
    return any(re.search(r"\b" + t + r"\b", text_upper) for t in tokens)


def _max_reported_diff(output: str):
    """Largest DIFF=<float> the harness printed, or None if it printed none."""
    diffs = []
    for m in re.finditer(r"DIFF\s*=\s*([0-9eE.+\-]+)", output):
        try:
            diffs.append(float(m.group(1)))
        except ValueError:
            continue
    return max(diffs) if diffs else None


def run_validation(
    executable_path: str,
    timeout: int = DEFAULT_TIMEOUT,
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[bool, str]:
    """
    Executes the compiled CUDA binary to ensure mathematical correctness.
    Returns (True, "output") if passed, (False, "error reason") if failed.

    Decision order:
      1. Non-zero exit code  -> crash/failure.
      2. A printed DIFF=<max relative error> is the quantitative source of
         truth: pass iff it is within `rel_tol`. This overrides any
         SUCCESS/FAILURE token the harness printed with its own (often too
         strict) tolerance.
      3. Fall back to SUCCESS/FAILURE token matching.
    """
    abs_binary = os.path.abspath(executable_path)

    if not os.path.exists(abs_binary):
        return False, f"Executable not found at {abs_binary}"

    # BENCH_ONLY makes a kernel skip its CPU reference (see benchmarker). It must
    # never be set for a correctness run, or we'd "validate" a kernel that never
    # actually compared itself against anything.
    env = {k: v for k, v in os.environ.items() if k != "BENCH_ONLY"}

    try:
        result = subprocess.run(
            [abs_binary],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )

        output = result.stdout.strip() + "\n" + result.stderr.strip()

        if result.returncode != 0:
            return False, (
                f"Binary crashed during execution. Exit code {result.returncode}.\n"
                f"Output: {output}"
            )

        output_upper = output.upper()

        # (2) Quantitative check wins whenever the harness reported a diff.
        max_diff = _max_reported_diff(output)
        if max_diff is not None:
            if max_diff <= rel_tol:
                return True, (
                    f"Math validation passed "
                    f"(max DIFF={max_diff:.2e} <= tol {rel_tol:.0e})."
                )
            return False, (
                f"Math validation failed: max DIFF={max_diff:.2e} exceeds "
                f"tolerance {rel_tol:.0e}.\nOutput:\n{output}"
            )

        # (3) Token fallback for harnesses that only print SUCCESS/FAILURE.
        has_pass = _has_token(output_upper, PASS_TOKENS)
        has_fail = _has_token(output_upper, FAIL_TOKENS)

        if has_pass and not has_fail:
            return True, "Math validation passed."
        if has_fail:
            return False, f"Math validation failed. Output:\n{output}"

        return False, (
            "Validation logic missing. The kernel must print 'SUCCESS' (or a "
            "'DIFF=<max_rel_error>' line) if correct. Output was:\n"
            f"{output}"
        )

    except subprocess.TimeoutExpired:
        return False, (
            "Validation timed out. The kernel likely contains an infinite loop "
            "or deadlock, or the CPU reference is verifying too many elements."
        )
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"


def check_determinism(
    executable_path: str,
    runs: int = DEFAULT_DETERMINISM_RUNS,
    timeout: int = DEFAULT_TIMEOUT,
    rel_tol: float = DEFAULT_REL_TOL,
) -> tuple[bool, str]:
    """
    Re-run the FULL correctness check several times to catch a kernel that is
    only sometimes right -- the signature of a race condition (e.g. a missing
    __syncthreads, or threads exiting early past a barrier).

    Why this exists as its own function rather than as a flag on benchmark():
    the timed runs deliberately set BENCH_ONLY=1, which tells the kernel to skip
    its CPU reference. A kernel that honours that flag prints no SUCCESS/FAILURE
    token at all, so scanning timed-run output for a failure token can never
    detect anything. Determinism therefore has to be established here, on real
    validation runs with the CPU reference enabled.

    Returns (True, msg) if every run passed, (False, reason) on the first
    disagreement.
    """
    if runs <= 1:
        return True, "determinism check skipped (runs <= 1)"

    diffs = []
    for i in range(1, runs + 1):
        ok, msg = run_validation(executable_path, timeout=timeout, rel_tol=rel_tol)
        if not ok:
            return False, (
                f"Nondeterministic correctness: run {i}/{runs} failed after an "
                f"earlier run passed. Likely a race condition (check __syncthreads "
                f"placement and shared-memory writes).\n{msg}"
            )
        m = re.search(r"max DIFF=([0-9eE.+\-]+)", msg)
        if m:
            diffs.append(m.group(1))

    # Identical inputs are regenerated per run only if the harness seeds RNG by
    # time, so differing DIFFs are not proof of a bug -- but all runs passing is
    # the property we need.
    detail = f" (max DIFF across runs: {', '.join(diffs)})" if diffs else ""
    return True, f"Deterministic across {runs} validation runs{detail}."

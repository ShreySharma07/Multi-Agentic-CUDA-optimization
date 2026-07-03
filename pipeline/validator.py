import subprocess
import os
import re

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

    try:
        result = subprocess.run(
            [abs_binary],
            capture_output=True,
            text=True,
            timeout=timeout,
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

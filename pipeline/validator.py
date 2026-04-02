import subprocess
import os

def run_validation(executable_path: str) -> tuple[bool, str]:
    """
    Executes the compiled CUDA binary to ensure mathematical correctness.
    Returns (True, "output") if passed, (False, "error reason") if failed.
    """
    abs_binary = os.path.abspath(executable_path)
    
    if not os.path.exists(abs_binary):
         return False, f"Executable not found at {abs_binary}"

    try:
        # Run binary with a timeout (in case the LLM wrote an infinite loop)
        result = subprocess.run(
            [abs_binary],
            capture_output=True,
            text=True,
            timeout=30  # 30 seconds should be plenty for a validation run
        )

        output = result.stdout.strip() + "\n" + result.stderr.strip()
        
        if result.returncode != 0:
            return False, f"Binary crashed during execution. Exit code {result.returncode}.\nOutput: {output}"
        
        output_upper = output.upper()
        
        if "SUCCESS" in output_upper:
            return True, "Math validation passed."
        elif "FAILURE" in output_upper or "ERROR" in output_upper:
            return False, f"Math validation failed. Output:\n{output}"
        else:
            return False, f"Validation logic missing. The kernel must print 'SUCCESS' if correct. Output was:\n{output}"
        
    except subprocess.TimeoutExpired:
        return False, "Validation timed out. The kernel likely contains an infinite loop or deadlock."
    except Exception as e:
        return False, f"Unexpected validation error: {str(e)}"
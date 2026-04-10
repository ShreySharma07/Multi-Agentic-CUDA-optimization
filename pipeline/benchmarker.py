# pipeline/benchmarker.py
import subprocess
import re
from pathlib import Path

def benchmark(binary_path: str, warmup: int = 20, runs: int = 100) -> float:
    """
    Returns mean execution time in milliseconds.
    Requires the kernel binary to print timing itself,
    OR we time it externally with cudaEvents.
    """
    times = []
    
    # warmup
    for _ in range(warmup):
        subprocess.run([binary_path], capture_output=True, timeout=30)
    
    # timed runs
    for _ in range(runs):
        result = subprocess.run(
            [binary_path], capture_output=True, text=True, timeout=30
        )
        # parse "GPU Time: X ms" from stdout
        match = re.search(r'GPU Time:\s*([\d.]+)\s*ms', result.stdout)
        if match:
            times.append(float(match.group(1)))
    
    if not times:
        return 0.0
    
    return sum(times) / len(times)
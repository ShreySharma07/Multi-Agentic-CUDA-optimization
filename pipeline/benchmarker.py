# pipeline/benchmarker.py
import subprocess
import re
from pathlib import Path

def benchmark(binary_path: str, warmup=5, runs=20, timeout=10) -> float:
    times = []
    for _ in range(warmup):
        try:
            subprocess.run([binary_path], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0
    for _ in range(runs):
        try:
            r = subprocess.run([binary_path], capture_output=True, text=True, timeout=timeout)
            match = re.search(r'GPU Time:\s*([\d.]+)', r.stdout)
            if match:
                times.append(float(match.group(1)))
        except subprocess.TimeoutExpired:
            break
    return sum(times)/len(times) if times else 0.0
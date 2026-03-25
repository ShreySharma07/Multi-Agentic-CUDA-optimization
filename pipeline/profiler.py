import subprocess
from pathlib import Path


def run_ncu_profile(binary: str):
    result = subprocess.run(
        ["ncu", "--set", "full", "--csv", f"./{binary}"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return {"error": result.stderr}

    return {"profile": result.stdout}

import csv
import io

def parse_ncu_profile(profile_csv: str):
    """Extract key metrics from Nsight Compute output."""
    
    metrics = {}

    reader = csv.reader(io.StringIO(profile_csv))

    for row in reader:
        if len(row) < 2:
            continue

        metric_name = row[0]
        metric_value = row[1]

        if "sm__warps_active" in metric_name:
            metrics["occupancy"] = metric_value

        if "dram__throughput" in metric_name:
            metrics["dram_throughput"] = metric_value

        if "gld_efficiency" in metric_name:
            metrics["memory_coalescing"] = metric_value

    return {"metrics": metrics}
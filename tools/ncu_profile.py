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
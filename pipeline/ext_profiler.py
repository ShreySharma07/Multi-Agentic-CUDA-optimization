# pipeline/ext_profiler.py
"""
Nsight Compute profiling for a torch CUDA extension.

The old pre_flight() profiled a standalone binary. An extension has no binary --
it's a .pyd loaded into a Python process -- so we generate a tiny driver script
that rebuilds the (already cached) extension and calls forward() once, then run
`ncu` on that.

Reuses pipeline.profiler.parse_ncu_profile so the metric extraction stays in one
place. Redis is deliberately not used here: torch's own on-disk extension cache
(keyed by our content hash) already dedupes rebuilds.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from . import profiler, utils

DRIVER = '''\
import json, sys, torch
from pathlib import Path
sys.path.insert(0, r"{repo}")
from pipeline.torch_eval import load_task, compile_extension

spec = json.loads(Path(r"{spec}").read_text())
task = load_task(Path(spec["task"]))
mod, err = compile_extension(spec["cuda"], spec["cpp"])
if mod is None:
    print("EXT_BUILD_FAILED", err[:200]); sys.exit(1)

inputs = task.inputs(42)
with torch.no_grad():
    mod.forward(*inputs)          # the one launch ncu profiles
torch.cuda.synchronize()
'''


def profile_extension(
    cuda_source: str,
    cpp_source: str,
    task_path: Path,
    repo_root: Path,
    timeout_s: int = 120,
) -> dict:
    """
    Returns {"status": "success", "metrics": {...}} or {"status": "error", ...}.

    Degrades exactly like pre_flight(): any failure (no counter permission, no
    ncu, timeout) yields an error dict and the caller falls back to
    bottleneck="unknown" rather than aborting the run.
    """
    tmp = Path(tempfile.mkdtemp(prefix="karma_ncu_"))
    spec_path = tmp / "spec.json"
    driver_path = tmp / "bench_once.py"

    spec_path.write_text(json.dumps({
        "task": str(task_path),
        "cuda": cuda_source,
        "cpp": cpp_source,
    }))
    driver_path.write_text(DRIVER.format(repo=str(repo_root), spec=str(spec_path)))

    try:
        ncu = utils.find_cuda_tool("ncu")
    except FileNotFoundError as e:
        return {"status": "error", "stage": "profiler", "error_message": str(e)}

    cmd = [ncu, "--set", "full", "--csv", "--target-processes", "all",
           sys.executable, str(driver_path)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "error", "stage": "profiler",
                "error_message": f"ncu timed out after {timeout_s}s"}
    except Exception as e:
        return {"status": "error", "stage": "profiler", "error_message": str(e)}

    out = (result.stdout or "") + (result.stderr or "")

    if "ERR_NVGPUCTRPERM" in out:
        return {"status": "error", "stage": "profiler", "error_message": (
            "ERR_NVGPUCTRPERM: no permission to read GPU performance counters. "
            "Set HKLM\\SYSTEM\\CurrentControlSet\\Services\\nvlddmkm\\Global\\NVTweak"
            "\\RmProfilingAdminOnly = 0 (DWORD) and REBOOT, or run as Administrator."
        )}

    if result.returncode != 0 or "EXT_BUILD_FAILED" in out:
        return {"status": "error", "stage": "profiler",
                "error_message": f"ncu exit {result.returncode}: {out.strip()[:300]}"}

    metrics = profiler.parse_ncu_profile(result.stdout)["metrics"]
    if not any(metrics.values()):
        return {"status": "error", "stage": "profiler",
                "error_message": "ncu produced no usable metrics"}

    return {"status": "success", "metrics": metrics}

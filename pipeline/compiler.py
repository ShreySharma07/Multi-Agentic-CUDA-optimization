import subprocess
from pathlib import Path
import torch

def get_gpu_info():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"

def compile_cuda(file_path):

    executable = file_path.replace(".cu", "")

    arch_type = get_gpu_info()

    cmd = ['nvcc', str(file_path), '-o', executable, '-O3', f'-arch={arch_type}', '-lineinfo']

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode == 0:
        print("Compilation successful")
        return (True, file_path)
    else:
        print("Compilation Failed!")
        return (False, process.stderr)
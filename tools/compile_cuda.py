import subprocess
from pathlib import Path

def compile_cuda(file_path):

    executable = file_path.replace(".cu", "")

    cmd = ['nvcc', str(file_path), '-o', executable]

    process = subprocess.run(cmd, capture_output=True, text=True)

    if process.returncode == 0:
        print("Compilation successful")
        return executable
    else:
        print("Compilation Failed!")
        print(process.stderr)
        return None
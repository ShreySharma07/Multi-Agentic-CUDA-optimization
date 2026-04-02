import subprocess
import torch
import os
import utils

def get_gpu_arch():
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"

def compile_cuda(file_path: str):
    # absolute paths so it works from anywhere
    abs_file_path = os.path.abspath(file_path)
    executable = abs_file_path.replace(".cu", "")
    arch = get_gpu_arch()
    
    nvcc_path = utils.find_cuda_tool('nvcc')
    
    cmd = [nvcc_path, abs_file_path, '-o', executable, '-O3', f'-arch={arch}', '-lineinfo']

    try:
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode == 0:
            print(f"Compilation successful for {arch}")
            return (True, executable) 
        else:
            # stdout just in case nvcc hides the error there
            error_msg = process.stderr.strip() or process.stdout.strip()
            return (False, error_msg)
            
    except Exception as e:
        return (False, f"Critical Compiler Error: {str(e)}")
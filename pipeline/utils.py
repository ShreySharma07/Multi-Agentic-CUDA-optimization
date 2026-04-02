import shutil
import glob

def find_cuda_tool(tool_name: str) -> str:
    """
    Dynamically hunts for CUDA tools (nvcc, ncu) even if sudo strips the PATH.
    """
    # 1. normal system PATH first
    path = shutil.which(tool_name)
    if path: 
        return path

    # 2. Fallback: Search common Linux installation directories
    patterns = [
        f"/usr/local/cuda/bin/{tool_name}",         # Default CUDA
        f"/usr/local/cuda-*/bin/{tool_name}",       # Versioned CUDA (like your 13.1)
        f"/opt/nvidia/nsight-compute/*/{tool_name}",# Custom NCU installs
        f"/usr/local/NVIDIA-Nsight-Compute*/{tool_name}" 
    ]

    for p in patterns:
        matches = glob.glob(p)
        if matches:
            # Sort reverse so if there's multiple versions, it grabs the newest
            return sorted(matches, reverse=True)[0]

    raise FileNotFoundError(f"Critical: Could not locate {tool_name} on this system.")
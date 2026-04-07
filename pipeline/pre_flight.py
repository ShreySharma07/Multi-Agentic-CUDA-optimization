import hashlib
import pathlib
from . import compiler, profiler, cache
import json
import os

def pre_flight(source_code : str)->dict:
    hashed_code = hashlib.sha256(source_code.encode('utf-8')).hexdigest()

    cached_data = cache.get_cached_metrics(hashed_code)

    if cached_data:
        return {
            "status":"success",
            "hash":hashed_code,
            "metrics":cached_data,
            "source":"redis cache"
        }

    temp_dir = pathlib.Path("kernels")
    temp_dir.mkdir(exist_ok=True)

    temp_dir_path = temp_dir / f"baseline_{hashed_code[:8]}.cu"
    
    try:
        with open(temp_dir_path, 'w') as f:
            f.write(source_code)
        
        success, compiled_path = compiler.compile_cuda(str(temp_dir_path))

        if not success:
            return{
            "status":"error",
            "stage":"compilation",
            "hash":hashed_code,
            "result":compiled_path
        }

        executable = compiled_path

        profiler_raw = profiler.run_ncu_profile(executable)

        if isinstance(profiler_raw, dict) and "error" in profiler_raw:
            return {
                "status":"error",
                "stage":"profiler",
                "hash": hashed_code,
                "error_message": profiler_raw["error"]
            }
        
        metrics = profiler.parse_ncu_profile(profiler_raw['profile'])

        cache.save_to_cache(hashed_code, metrics["metrics"])

        return {
            "status":"success",
            "hash":hashed_code,
            "metrics":metrics['metrics']
        }
    
    finally:
        if temp_dir_path.exists():
            temp_dir_path.unlink()
        
        executable_file = pathlib.Path(str(temp_dir_path).replace(".cu", ""))
        if executable_file.exists():
            executable_file.unlink()

if __name__ == '__main__':
    cu_file_path = '/home/lab/project_26/Multi_Agent_CUDA_optimization/kernels/sigmoid_kernel.cu'

    with open(cu_file_path, 'r') as f:
        source_code = f.read()
    
    result = pre_flight(source_code=source_code)
    print(json.dumps(result, indent=2))
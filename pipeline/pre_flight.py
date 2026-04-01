import hashlib
import pathlib
from pipeline import compiler, profiler
import json


def pre_flight(source_code : str)->dict:
    hashed_code = hashlib.sha256(source_code)
    
    with open('kernels/baseline_temp.cu', 'w') as f:
        f.write(source_code)
    
    source = pathlib.Path('kernels/baseline_temp.cu')
    try:
        _, compiled_path = compiler.compile_cuda(source)
    except Exception as e:
        print(f"compilation failed {e}")
    
    profiler_path = profiler.run_ncu_profile(compiled_path)
    metrics = profiler.parse_ncu_profile(profiler_path)

    return {
        "hashed_code":hashed_code,
        "metrics":metrics
    }
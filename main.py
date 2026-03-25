from pipeline import compiler, profiler
from Agents import coder
import os
import time

import os
import asyncio
from pathlib import Path

# Import your pipeline tools
from pipeline.compiler import compile_cuda
from pipeline.profiler import run_ncu_profile, parse_ncu_profile

# Import your agent setup (assuming you exposed an 'ask_coder' function or similar)
from Agents.coder import chat, runner, USER_ID, SESSION_ID

# A helper function to strip the markdown formatting from the LLM response
def extract_cuda_code(llm_response: str) -> str:
    """Extracts raw C++ code from markdown block with safety checks."""
    # 1. Safety check: Did the LLM actually return a string?
    if not llm_response or not isinstance(llm_response, str):
        return ""

    # 2. Extract the code
    if "```cpp" in llm_response:
        code = llm_response.split("```cpp")[1].split("```")[0]
        return code.strip()
    elif "```cuda" in llm_response:
        code = llm_response.split("```cuda")[1].split("```")[0]
        return code.strip()
    
    return llm_response.strip()

async def main():
    print("=== KARMA MVP Optimization Loop ===")
    
    # 1. Deterministic File Selection (No AI here)
    target_file = Path("kernels/sigmoid_kernel.cu")
    if not target_file.exists():
        print(f"Error: Could not find {target_file}")
        return

    with open(target_file, "r") as f:
        original_code = f.read()

    # 2. Setup the Optimization Loop
    max_retries = 3
    current_prompt = f"Optimize this CUDA kernel for an Ampere A4000 GPU:\n\n```cpp\n{original_code}\n```"
    
    optimized_code = ""
    executable_path = ""
    compile_success = False

    for attempt in range(1, max_retries + 1):
        print(f"\n[Round {attempt}/{max_retries}] Asking CoderAgent for optimizations...")
        
        # Call your agent (Make sure your chat() function returns the string!)
        response_text = await chat(current_prompt, runner, USER_ID, SESSION_ID)
        
        # Extract the raw code and save it to a temporary file
        optimized_code = extract_cuda_code(response_text)
        temp_file = Path("kernels/temp_optimized.cu")
        
        with open(temp_file, "w") as f:
            f.write(optimized_code)
            
        print(f"[Round {attempt}] Compiling temp_optimized.cu...")
        
        # 3. Deterministic Compilation
        success, result = compile_cuda(str(temp_file))
        
        if success:
            print(f"[Round {attempt}] Compilation Successful! ✅")
            executable_path = result
            compile_success = True
            break # Exit the loop, we have a working binary!
        else:
            print(f"[Round {attempt}] Compilation Failed. ❌ Feeding error back to Agent...")
            # 4. The Feedback Loop: Tell the LLM exactly why it failed
            current_prompt = (
                f"Your previous code failed to compile with the following error:\n"
                f"```\n{result}\n```\n"
                f"Please fix the compilation errors and output the full corrected CUDA code."
            )

    # 5. Profiling (If we successfully compiled)
    if compile_success:
        print("\n=== Running Nsight Compute Profiler ===")
        profile_raw = run_ncu_profile(executable_path)
        
        if "error" in profile_raw:
            print("Profiler encountered an error:")
            print(profile_raw["error"])
        else:
            # Parse the CSV output into a clean dictionary
            metrics = parse_ncu_profile(profile_raw["profile"])
            print("\n📊 Optimization Results:")
            for key, value in metrics["metrics"].items():
                print(f"  - {key}: {value}")
    else:
        print("\n❌ Optimization failed: Could not produce a compiling kernel after maximum retries.")

if __name__ == "__main__":
    asyncio.run(main())
import subprocess
import csv
import io
import os
import utils

def run_ncu_profile(binary: str):
    # 1. Convert to an absolute path to prevent NCU pathing bugs
    abs_binary = os.path.abspath(binary)
    
    # 2. Safety check: Did the compiler actually create the file?
    if not os.path.exists(abs_binary):
         return {"error": f"Executable not found at {abs_binary}. Compiler may have failed silently."}

    ncu_path = utils.find_cuda_tool('ncu')
    try:
        result = subprocess.run(
            # 3. Pass the absolute path
            [ncu_path, "--set", "full", "--csv", abs_binary],
            capture_output=True,
            text=True,
            timeout=300 
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if not error_msg:
                error_msg = result.stdout.strip()
                
            return {"error": f"NCU Exit Code {result.returncode}:\n{error_msg}"}

        return {"profile": result.stdout}
        
    except FileNotFoundError:
        return {"error": "ncu command not found. Is Nsight Compute installed and in your PATH?"}
    except subprocess.TimeoutExpired:
        return {"error": "Profiler timed out. The kernel likely contains an infinite loop."}
    except Exception as e:
        return {"error": f"Unexpected Python error: {str(e)}"}

def parse_ncu_profile(profile_csv: str):
    metrics = {
        "occupancy": 0.0,
        "dram_throughput": 0.0,
        "compute_throughput": 0.0
    }
    
    # Use csv.reader to safely handle commas inside kernel names
    reader = csv.reader(io.StringIO(profile_csv))
    
    header = None
    name_idx = -1
    val_idx = -1
    
    for row in reader:
        # Skip empty or malformed rows
        if not row or len(row) < 3:
            continue
        
        # 1. Dynamically find the header columns
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            name_idx = row.index("Metric Name")
            val_idx = row.index("Metric Value")
            continue
            
        # 2. Extract the data if we found the headers
        if header and len(row) > max(name_idx, val_idx):
            m_name = row[name_idx]
            
            try:
                # Clean up the value (remove commas like '223,763')
                raw_val = row[val_idx].replace(',', '')
                m_val = float(raw_val)
            except ValueError:
                continue # Skip rows that don't have numeric values
            
            # 3. Match against the metrics we care about for KARMA
            if "Achieved Occupancy" in m_name or "sm__warps_active" in m_name:
                metrics["occupancy"] = m_val
            elif "DRAM Throughput" in m_name or "dram__throughput" in m_name:
                metrics["dram_throughput"] = m_val
            elif "Compute (SM) Throughput" in m_name or "sm__throughput" in m_name:
                metrics["compute_throughput"] = m_val

    return {"metrics": metrics}
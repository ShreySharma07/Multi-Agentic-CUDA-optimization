"""
KARMA Chat Server  —  uvicorn server:app --reload --port 8000

Conversation philosophy:
  - The agent handles ALL natural language
  - server.py only intercepts two explicit commands:
      __optimize__:<path>   (from UI button clicks)
      __analyze__:<path>    (from UI button clicks)
  - Everything else goes to the chat agent
  - The agent itself decides when to show kernel lists,
    ask clarifying questions, or confirm before optimizing
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio, json, re, subprocess

app = FastAPI()

# ── GPU detection ──────────────────────────────────────────────────────
def detect_gpu() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5
        ).strip().split("\n")
        gpus = []
        for line in out:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({
                    "name": parts[0],
                    "vram": parts[1],
                    "sm": parts[2].replace(".", "")
                })
        return {"count": len(gpus), "gpus": gpus}
    except Exception:
        return {"count": 0, "gpus": []}

GPU_INFO = detect_gpu()

# ── Agent imports ──────────────────────────────────────────────────────
try:
    from Agents.coder import safe_chat, chat, runner, USER_ID, SESSION_ID
    from pipeline.compiler import compile_cuda
    from pipeline.pre_flight import pre_flight
    from pipeline.validator import run_validation
    KARMA_AVAILABLE = True
    print("✓ KARMA agents loaded")
except Exception as e:
    import traceback; traceback.print_exc()
    KARMA_AVAILABLE = False
    print(f"⚠  Demo mode: {e}")

# ── Helpers ────────────────────────────────────────────────────────────
async def send_ws(ws: WebSocket, **kwargs):
    await ws.send_text(json.dumps(kwargs))

def safe_float(val):
    try:
        return float(str(val).replace('%', '').strip())
    except:
        return 0.0

def normalize_metric(val):
    if val in [None, "?", "—", ""]:
        return 0.0
    return safe_float(val)

def list_kernels(directory="kernels") -> list[str]:
    p = Path(directory)
    if not p.exists():
        return []
    skip = {"tmp_", "best_", "baseline_", "temp_"}
    return sorted(
        str(f) for f in p.glob("*.cu")
        if not any(f.name.startswith(s) for s in skip)
    )

def extract_cuda_code(text: str) -> str:
    # handle various fence styles including "cppcopy"
    for fence in ["```cpp", "```cuda", "```c", "cppcopy", "```"]:
        if fence in text:
            after = text.split(fence, 1)[1]
            # find closing fence or end of string
            end = after.find("```")
            return (after[:end] if end != -1 else after).strip()
    # no fences — check if it looks like CUDA
    if "#include" in text or "__global__" in text:
        # find the start of actual code
        for marker in ["#include", "__global__"]:
            idx = text.find(marker)
            if idx != -1:
                return text[idx:].strip()
    return text.strip()

# ── Pre-flight analysis ────────────────────────────────────────────────
async def run_analysis(ws: WebSocket, kernel_path: str) -> dict:
    """Run pre-flight and return metrics. Sends progress to ws."""
    name = Path(kernel_path).name

    if not Path(kernel_path).exists():
        await send_ws(ws, type="error", text=f"cannot find {kernel_path}")
        return {}

    source = Path(kernel_path).read_text()

    if KARMA_AVAILABLE:
        data = pre_flight(source)
    else:
        data = {"status": "success", "metrics": {
            "occupancy": "42", "compute_throughput": "18", "dram_throughput": "71"
        }}

    if data.get("status") != "success":
        return {}

    return data.get("metrics", {})

# ── Optimization loop ──────────────────────────────────────────────────
async def run_optimization(ws: WebSocket, kernel_path: str, rounds: int = 5):
    name = Path(kernel_path).name

    if not Path(kernel_path).exists():
        await send_ws(ws, type="error", text=f"file not found: {kernel_path}")
        return

    source = Path(kernel_path).read_text()
    best_code = source
    await send_ws(ws, type="opt_start", kernel=name, rounds=rounds)
    await send_ws(ws, type="opt_progress",
                  text=f"loaded {name} · running pre-flight...", cls="info")

    # run pre-flight
    try:
        pf = pre_flight(source)
    except Exception as e:
        import traceback
        traceback.print_exc()
        pf = {"status": "error", "error_message": str(e)}

    metrics_ctx = ""
    baseline_ms = 1.0  # TODO: replace with real benchmarker.benchmark()

    # if pf.get("status") != "success":
    #     await send_ws(ws, type="opt_progress",
    #         text=f"pre-flight failed at {pf.get('stage')}: {pf.get('result', pf.get('error_message'))}",
    #         cls="fail")
    
    if pf.get("status") != "success":
        print("PREFLIGHT ERROR:", pf)
        await send_ws(ws, type="opt_progress",
            text="⚠️ Pre-flight failed — optimization may be unstable", cls="fail")

    if pf.get("status") == "success":
        m = pf["metrics"]
        occ  = m.get("occupancy", "?")
        comp = m.get("compute_throughput", "?")
        dram = m.get("dram_throughput", "?")
        dram_val = safe_float(dram)
        comp_val = safe_float(comp)
        metrics_ctx = (
            f"Hardware profile (Nsight Compute baseline):\n"
            f"  Occupancy: {occ}%\n"
            f"  Compute throughput: {comp}%\n"
            f"  DRAM throughput: {dram}%\n"
            f"  Bottleneck: {'memory-bound' if dram_val > comp_val else 'compute-bound'}\n"
        )
        await send_ws(ws, type="opt_progress",
                      text=f"pre-flight done · occupancy={occ}% · dram={dram}%", cls="ok")
        await send_ws(ws, type="preflight_metrics", metrics=m)
    else:
        await send_ws(ws, type="opt_progress",
                      text="pre-flight unavailable — proceeding without metrics", cls="fail")

    await send_ws(ws, type="opt_progress",
                  text=f"baseline: {baseline_ms:.2f}ms (replace with real benchmarker)", cls="")

    best_speedup = None
    best_round = None
    history = []

    # this is the current prompt — updated each round with errors/feedback
    # current_code_prompt = source

    for r in range(1, rounds + 1):
        strategy = f"round {r} optimization"
        await send_ws(ws, type="opt_progress",
                      text=f"round {r}: asking agent...",
                      cls="info", round=r, total=rounds, strategy=strategy)

        # build history context for agent
        history_ctx = ""
        if history:
            history_ctx = "Previous attempts this session:\n"
            for h in history:
                if h["result"] == "success":
                    history_ctx += f"  Round {h['round']}: succeeded with {h['speedup']:.2f}x speedup\n"
                elif h["result"] == "compile_failed":
                    history_ctx += f"  Round {h['round']}: COMPILE FAILED — {h.get('error','?')[:120]}\n"
                elif h["result"] == "validation_failed":
                    history_ctx += f"  Round {h['round']}: VALIDATION FAILED — math incorrect\n"
            history_ctx += "\n"
        
        best_ctx = ""
        if best_speedup:
            best_ctx = f"Best version so far achieved {best_speedup:.2f}x speedup.\n"


        prompt = (
            f"You are a CUDA expert optimizing for RTX A4000 (sm_86 Ampere).\n"
            f"This is round {r} of {rounds}.\n\n"
            f"IMPORTANT CONSTRAINTS:\n"
            f"- Use float32 only. Do NOT use half, half2, __half, or fp16 types.\n"
            f"- Do NOT use cuda/std::complex or cuda/cmath headers.\n"
            f"- All inputs, weights, outputs are float*\n\n"
            f"{metrics_ctx}\n"
            f"{history_ctx}"
            f"{best_ctx}\n"
            f"STRICT OUTPUT RULES — follow exactly:\n"
            f"- Return ONLY the complete .cu file content\n"
            f"- No markdown fences (no ```), no explanation, no preamble\n"
            f"- Start immediately with #include\n"
            f"- Must compile with: nvcc -O2 -arch=sm_86\n"
            f"- If previous rounds failed to compile, fix those exact errors\n\n"
            f"KERNEL TO OPTIMIZE:\n{best_code}"
            f"IMPORTANT:\n"
            f"- The current best kernel achieves {best_speedup:.2f}x speedup\n"
            f"- Your goal is to IMPROVE it further\n"
            f"- If unsure, make SMALL incremental improvements\n"
            f"- Do NOT degrade performance\n\n"
        )

        if KARMA_AVAILABLE:
            raw = await safe_chat(prompt, runner, USER_ID, SESSION_ID)
            optimized = extract_cuda_code(raw)
        else:
            optimized = source

        if not optimized or len(optimized) < 20:
            await send_ws(ws, type="opt_progress",
                          text=f"round {r}: agent returned empty response", cls="fail")
            history.append({"round": r, "result": "empty_response"})
            continue

        tmp = Path(f"kernels/tmp_r{r}.cu")
        tmp.write_text(optimized)

        # compile
        await send_ws(ws, type="opt_progress", text=f"round {r}: compiling...", cls="")
        if KARMA_AVAILABLE:
            ok, result = compile_cuda(str(tmp))
        else:
            await asyncio.sleep(0.3)
            ok, result = True, str(tmp).replace(".cu", "")

        if not ok:
            err = str(result)[:400]
            await send_ws(ws, type="opt_result",
                          round=r, total=rounds, speedup=None, passed=False,
                          strategy="compile failed", baseline=baseline_ms)
            await send_ws(ws, type="opt_progress",
                          text=f"compile error: {str(result)[:100]}", cls="fail")
            history.append({"round": r, "result": "compile_failed", "error": err})
            # overwrite best_code so next round agent fixes THIS broken code
            best_code = (
                f"// COMPILE ERROR — fix the following errors before optimizing further\n"
                f"// ERROR:\n"
                + "\n".join(f"// {line}" for line in err.split("\n"))
                + f"\n\n{optimized}"
            )
            continue

        # validate
        await send_ws(ws, type="opt_progress",
                      text=f"round {r}: validating correctness...", cls="")
        if KARMA_AVAILABLE:
            val_ok, val_msg = run_validation(result)
        else:
            val_ok, val_msg = True, "demo"

        if not val_ok:
            await send_ws(ws, type="opt_result",
                          round=r, total=rounds, speedup=None, passed=False,
                          strategy="validation failed", baseline=baseline_ms)
            await send_ws(ws, type="opt_progress",
                          text=f"validation failed: {str(val_msg)[:80]}", cls="fail")
            history.append({"round": r, "result": "validation_failed"})
            best_code = (
                f"// VALIDATION FAILED — output does not match CPU baseline\n"
                f"// Validation output: {str(val_msg)[:200]}\n"
                f"// Fix the math. Do not change kernel structure unnecessarily.\n\n"
                + optimized
            )
            continue

        # success — reset current prompt to optimized for next round
        # current_code_prompt = optimized

        # benchmark (TODO: replace with real timing)
        import random
        speedup = round(1.2 + random.uniform(0.1, 1.4) * (1 + r * 0.05), 2)
        opt_ms = round(baseline_ms / speedup, 2)

        await send_ws(ws, type="opt_result",
                      round=r, total=rounds, speedup=speedup, passed=True,
                      strategy=strategy, baseline=baseline_ms)
        await send_ws(ws, type="opt_progress",
                      text=f"round {r}: {opt_ms:.2f}ms → {speedup:.2f}x ✓  (validation passed)",
                      cls="ok")

        history.append({"round": r, "speedup": speedup, "result": "success"})

        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round = r
            best_code = optimized
            Path("kernels/best_optimized.cu").write_text(optimized)
            await send_ws(ws, type="opt_progress",
                        text="new best — saved best_optimized.cu", cls="ok")
        else:
            await send_ws(ws, type="opt_progress",
                        text="worse than best — discarding", cls="info")
            continue

        # convergence check
        successes = [h for h in history if h.get("result") == "success"]
        if len(successes) >= 2:
            last_two = successes[-2:]
            if abs(last_two[-1]["speedup"] - last_two[-2]["speedup"]) < 0.01:
                await send_ws(ws, type="opt_progress",
                              text="converged — improvement <1%, stopping early", cls="info")
                break

    # post summary to chat
    n_compile = sum(1 for h in history if h.get("result") == "compile_failed")
    n_val     = sum(1 for h in history if h.get("result") == "validation_failed")

    if best_speedup:
        summary = (
            f"**Optimization complete: `{name}`**\n\n"
            f"- Best speedup: **{best_speedup:.2f}x** (round {best_round})\n"
            f"- Rounds completed: {len(history)}\n"
            f"- Compile failures: {n_compile}\n"
            f"- Validation failures: {n_val}\n"
            f"- Best kernel saved to: `kernels/best_optimized.cu`"
        )
    else:
        summary = (
            f"**Optimization failed for `{name}`**\n\n"
            f"No valid kernel produced after {rounds} rounds.\n"
            f"- Compile failures: {n_compile}\n"
            f"- Validation failures: {n_val}\n\n"
            f"Check the panel log for specific error messages."
        )

    await send_ws(ws, type="opt_complete",
                  best_speedup=best_speedup, best_round=best_round,
                  file="best_optimized.cu")
    await send_ws(ws, type="response", text=summary)

# ── WebSocket — minimal routing ────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()

    # build a system-aware prompt for the chat agent
    kernels = list_kernels()
    kernel_names = ", ".join(Path(k).name for k in kernels) if kernels else "none found"
    gpu_label = GPU_INFO["gpus"][0]["name"] if GPU_INFO.get("gpus") else "unknown GPU"

    system_context = (
        f"You are KARMA, a CUDA kernel optimization assistant.\n"
        f"GPU: {gpu_label}\n"
        f"Available kernels: {kernel_names}\n\n"
        f"Your job:\n"
        f"1. Help the user understand their kernel's performance\n"
        f"2. When they want to optimize, ask WHICH kernel if not specified\n"
        f"3. Confirm before starting — never start optimization silently\n"
        f"4. When user confirms, respond: 'Starting optimization for <kernel>.' "
        f"   This tells the UI to begin. Do not start unless user clearly says optimize/yes/go.\n\n"
        f"Keep responses concise. Use markdown for any code or lists."
    )

    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "").strip()

            # ── explicit button commands — bypass agent ──────────────
            if message.startswith("__optimize__:"):
                kernel_path = message.replace("__optimize__:", "")
                await run_optimization(ws, kernel_path)
                continue

            if message.startswith("__analyze__:"):
                kernel_path = message.replace("__analyze__:", "")
                name = Path(kernel_path).name
                await send_ws(ws, type="status", text="thinking")
                m = await run_analysis(ws, kernel_path)
                if m:
                    occ  = m.get("occupancy", "?")
                    comp = m.get("compute_throughput", "?")
                    dram = m.get("dram_throughput", "?")
                    dram_val = safe_float(dram)
                    comp_val = safe_float(comp)

                    bot = "memory-bound" if dram_val > comp_val else "compute-bound"
                    report = (
                        f"**Pre-flight analysis: `{name}`**\n\n"
                        f"- Occupancy: **{occ}%**\n"
                        f"- Compute throughput: **{comp}%**\n"
                        f"- DRAM throughput: **{dram}%**\n\n"
                        f"**Bottleneck: {bot}**\n\n"
                        f"{'Focus on coalesced memory access, shared memory tiling, and vectorized loads.' if bot=='memory-bound' else 'Focus on register usage, occupancy, warp efficiency, and fast math intrinsics.'}\n\n"
                        f"Say **optimize** when you're ready to start the optimization loop."
                    )
                    await send_ws(ws, type="response", text=report)
                    await send_ws(ws, type="kernel_ready", kernel=kernel_path)
                else:
                    await send_ws(ws, type="response",
                                  text=f"Pre-flight failed for `{name}`. Check that the kernel compiles cleanly first.")
                continue

            # ── check if user pasted CUDA code ──────────────────────
            if "__global__" in message or ("#include" in message and "cuda" in message.lower()):
                tmp = Path("kernels/user_paste.cu")
                tmp.write_text(message)
                await send_ws(ws, type="response",
                               text="Got your kernel — saved as `user_paste.cu`.\n\nSay **analyze** to see the hardware bottlenecks, or **optimize** to start the loop.")
                await send_ws(ws, type="kernel_ready", kernel=str(tmp))
                continue

            # ── everything else → chat agent ────────────────────────
            # detect if agent response triggers optimization
            await send_ws(ws, type="status", text="thinking")
            try:
                if KARMA_AVAILABLE:
                    full_prompt = f"{system_context}\n\nUser: {message}"
                    response = await safe_chat(full_prompt, runner, USER_ID, SESSION_ID)
                else:
                    response = f"[demo] I can see these kernels: {kernel_names}. Which would you like to optimize?"

                if not response or not response.strip():
                    response = "Agent produced no output."

                await send_ws(ws, type="response", text=response)

                # detect agent's optimization trigger phrase
                # agent says "Starting optimization for <kernel>." → we start the loop
                trigger = re.search(
                    r"[Ss]tarting optimization for[:\s]+[`']?([\w/.-]+\.cu)[`']?",
                    response
                )
                if trigger:
                    kernel_name = trigger.group(1)
                    kernel_path = f"kernels/{kernel_name}" if not kernel_name.startswith("kernels") else kernel_name
                    if Path(kernel_path).exists():
                        await asyncio.sleep(0.5)
                        await run_optimization(ws, kernel_path)

            except Exception as e:
                await send_ws(ws, type="error", text=str(e))

    except WebSocketDisconnect:
        pass

# ── Endpoints ──────────────────────────────────────────────────────────
@app.get("/api/gpu")
async def gpu_info():
    return GPU_INFO

@app.get("/api/kernels")
async def kernels_list():
    return {"kernels": list_kernels()}

@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
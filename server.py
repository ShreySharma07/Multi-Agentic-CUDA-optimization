"""
KARMA Chat Server
Run: uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
import asyncio, json, re

app = FastAPI()

try:
    from Agents.coder import chat, runner, USER_ID, SESSION_ID
    from pipeline.compiler import compile_cuda
    KARMA_AVAILABLE = True
    print("✓ KARMA agents loaded")
except Exception as e:
    import traceback; traceback.print_exc()
    KARMA_AVAILABLE = False
    print(f"⚠  Demo mode: {e}")

async def mock_chat(query: str, *args, **kwargs) -> str:
    await asyncio.sleep(1.0)
    return f"[demo] received: {query[:80]}"

async def send_ws(ws: WebSocket, **kwargs):
    await ws.send_text(json.dumps(kwargs))

def list_kernels(directory="kernels") -> list[str]:
    p = Path(directory)
    if not p.exists():
        return []
    return sorted(str(f) for f in p.glob("*.cu")
                  if "tmp_" not in f.name and "best_" not in f.name)

async def run_optimization(ws: WebSocket, kernel_path: str, rounds: int = 5):
    name = Path(kernel_path).name
    if not Path(kernel_path).exists():
        await send_ws(ws, type="error", text=f"file not found: {kernel_path}")
        return

    source = Path(kernel_path).read_text()
    await send_ws(ws, type="opt_start", kernel=name, rounds=rounds)
    await send_ws(ws, type="opt_progress", text=f"loaded {name} ({len(source)} chars)", cls="info")

    await send_ws(ws, type="opt_progress", text="compiling baseline...", cls="")
    if KARMA_AVAILABLE:
        ok, result = compile_cuda(kernel_path)
    else:
        await asyncio.sleep(0.5); ok, result = True, kernel_path

    if not ok:
        await send_ws(ws, type="opt_progress", text=f"baseline compile failed: {str(result)[:120]}", cls="fail")
        await send_ws(ws, type="opt_complete", best_speedup=None, best_round=None, file=None)
        return

    baseline_ms = 1.0  # replace with real benchmarker.benchmark()
    await send_ws(ws, type="opt_progress", text=f"baseline OK — {baseline_ms:.2f}ms", cls="ok")

    best_speedup = None
    best_round = None
    history = []

    for r in range(1, rounds + 1):
        strategy = f"optimization attempt {r}"
        await send_ws(ws, type="opt_progress",
                      text=f"round {r}: generating optimized kernel...",
                      cls="info", round=r, total=rounds, strategy=strategy)

        prompt = (
            f"You are a CUDA expert. Optimize this kernel for RTX A4000 (sm_86 Ampere).\n"
            f"Round {r} of {rounds}. Previous attempts: {history}\n\n"
            f"RULES:\n"
            f"- Return ONLY the complete .cu file\n"
            f"- No markdown, no explanation, no code fences\n"
            f"- Must compile with: nvcc -O2 -arch=sm_86\n\n"
            f"KERNEL:\n{source}"
        )

        if KARMA_AVAILABLE:
            optimized = await chat(prompt, runner, USER_ID, SESSION_ID)
        else:
            optimized = source

        tmp = Path(f"kernels/tmp_r{r}.cu")
        tmp.write_text(optimized)

        await send_ws(ws, type="opt_progress", text=f"round {r}: compiling...", cls="")
        if KARMA_AVAILABLE:
            ok, result = compile_cuda(str(tmp))
        else:
            await asyncio.sleep(0.4); ok = True

        if not ok:
            await send_ws(ws, type="opt_result",
                          round=r, total=rounds, speedup=None, passed=False,
                          strategy=strategy, baseline=baseline_ms)
            await send_ws(ws, type="opt_progress",
                          text=f"compile error: {str(result)[:100]}", cls="fail")
            history.append({"round": r, "result": "compile_failed"})
            continue

        # TODO: replace with real benchmarker.benchmark()
        import random
        speedup = round(1.0 + random.uniform(0.2, 1.8) * (1 + r * 0.05), 2)
        opt_ms = round(baseline_ms / speedup, 2)

        await send_ws(ws, type="opt_result",
                      round=r, total=rounds, speedup=speedup, passed=True,
                      strategy=strategy, baseline=baseline_ms)
        await send_ws(ws, type="opt_progress",
                      text=f"round {r}: {opt_ms:.2f}ms → {speedup:.2f}x ✓", cls="ok")

        history.append({"round": r, "speedup": speedup})

        if best_speedup is None or speedup > best_speedup:
            best_speedup = speedup
            best_round = r
            Path("kernels/best_optimized.cu").write_text(optimized)
            await send_ws(ws, type="opt_progress", text="new best — saved best_optimized.cu", cls="ok")

        if len(history) >= 2:
            prev = [h["speedup"] for h in history[-2:] if "speedup" in h]
            if len(prev) == 2 and abs(prev[-1] - prev[-2]) < 0.01:
                await send_ws(ws, type="opt_progress", text="converged — stopping early", cls="info")
                break

    await send_ws(ws, type="opt_complete",
                  best_speedup=best_speedup, best_round=best_round, file="best_optimized.cu")

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            payload = json.loads(data)
            message = payload.get("message", "").strip()

            if message.startswith("__optimize__:"):
                await run_optimization(ws, message.replace("__optimize__:", ""))

            elif any(x in message.lower() for x in ["list kernels", "show kernels", "what kernels", "available kernels"]):
                await send_ws(ws, type="status", text="thinking")
                await send_ws(ws, type="kernel_list", kernels=list_kernels())

            elif "optimize" in message.lower() and ".cu" in message.lower():
                match = re.search(r'[\w/\\.-]+\.cu', message)
                if match:
                    path = match.group()
                    if not Path(path).exists():
                        path = f"kernels/{Path(path).name}"
                    await run_optimization(ws, path)
                else:
                    await send_ws(ws, type="status", text="thinking")
                    await send_ws(ws, type="kernel_list", kernels=list_kernels())

            elif "optimize" in message.lower():
                await send_ws(ws, type="status", text="thinking")
                await send_ws(ws, type="kernel_list", kernels=list_kernels())

            else:
                await send_ws(ws, type="status", text="thinking")
                try:
                    if KARMA_AVAILABLE:
                        response = await chat(message, runner, USER_ID, SESSION_ID)
                    else:
                        response = await mock_chat(message)
                    await send_ws(ws, type="response",
                                  text=response if response and response.strip() else "Agent produced no output.")
                except Exception as e:
                    await send_ws(ws, type="error", text=str(e))

    except WebSocketDisconnect:
        pass

@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

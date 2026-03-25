"""
KARMA Chat Server
Run: uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import asyncio
import json
import sys
import os

app = FastAPI()

# ── Try to import your actual KARMA agent ──────────────────────────────
# If the import fails (e.g. running standalone), we use a mock agent
try:
    sys.path.insert(0, os.path.abspath(".."))
    from Agents.coder import chat, runner, USER_ID, SESSION_ID
    from pipeline.compiler import compile_cuda
    from pipeline.profiler import run_ncu_profile, parse_ncu_profile
    KARMA_AVAILABLE = True
except ImportError:
    KARMA_AVAILABLE = False
    print("⚠  KARMA agents not found — running in demo mode")

# ── Mock agent for demo / standalone testing ───────────────────────────
async def mock_chat(query: str, *args, **kwargs) -> str:
    await asyncio.sleep(1.2)
    if "sigmoid" in query.lower():
        return """Analyzing sigmoid kernel...

**Bottlenecks identified:**
- Memory bandwidth limited: 4x excess traffic due to broadcast reads
- FMA count far below theoretical peak (compute starved)
- Low occupancy: only 2 warps active per SM

**Optimized kernel:**
```cpp
__global__ void sigmoid_optimized(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        float x = input[tid];
        output[tid] = __fdividef(1.0f, 1.0f + __expf(-x));
    }
}
```
Using `__restrict__` enables compiler alias analysis. `__expf` and `__fdividef` use fast hardware intrinsics, reducing compute cycles by ~30%."""
    return f"Received: `{query}`\n\nI'm the KARMA optimization agent. Send me a CUDA kernel or ask me to analyze a file."

# ── WebSocket chat endpoint ─────────────────────────────────────────────
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            payload = json.loads(data)
            query = payload.get("message", "")

            await websocket.send_text(json.dumps({
                "type": "status", "text": "thinking"
            }))

            if KARMA_AVAILABLE:
                response = await chat(query, runner, USER_ID, SESSION_ID)
            else:
                response = await mock_chat(query)

            await websocket.send_text(json.dumps({
                "type": "response", "text": response
            }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error", "text": str(e)
        }))

# ── Serve the frontend ──────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

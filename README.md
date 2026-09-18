# KARMA — Multi-agent CUDA kernel optimization

A closed-loop system that profiles a CUDA kernel with Nsight Compute, has a
coder agent rewrite it one optimization at a time, then compiles, validates
against a CPU reference and benchmarks each candidate on the GPU. A reflector
agent stores what worked in a ChromaDB knowledge base for the next kernel.

- `run_experiments.py` — headless loop over own kernels, KernelBench L1 or SGLang
- `server.py` + `index.html` — FastAPI WebSocket chat UI with a live optimization panel
- `pipeline/` — pre-flight, compiler, profiler, validator, benchmarker, Redis cache
- `Agents/coder.py` — Gemini coder agent on Google ADK
- `knowledgeBase/` — reflector + ChromaDB store
- `results/experiments.csv` — measured runs on an RTX A4000 (sm_86)
- `website/` — public site, OAuth login and desktop early-access waitlist (see its README)

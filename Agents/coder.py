# Agents/coder.py
"""
CoderAgent — writes the CUDA.

Owns both of its prompts (baseline generation and per-round optimization) and
talks to whatever LLMProvider its config selects. No google-adk, no Runner, no
shared session: the previous implementation routed all four agents through a
single ADK session, so every agent's replies accumulated in every other agent's
context and grew without bound over a run.
"""
from __future__ import annotations

from Agents.json_utils import extract_cuda_code
from Agents.providers import LLMProvider

# Constraints that apply to every kernel this agent emits. Kept as one block so
# baseline generation and optimization can't drift apart.
HARNESS_CONTRACT = """CORRECTNESS HARNESS (follow exactly):
- Keep problem dimensions MODERATE so the single-threaded CPU reference finishes in
  a few seconds and host allocations stay under ~1GB. For a reduction/matmul, keep
  the reduction dimension <= 512 and total output <= ~4M elements.
- Do the naive CPU reference ONCE, then compare GPU vs CPU by the RELATIVE error:
  rel = fabs(gpu-cpu) / fmaxf(1e-6f, fabsf(cpu)). Track max_rel.
- Because GPU and CPU accumulate reductions in different orders, use a RELATIVE
  tolerance that scales with reduction depth K:
      float tol = 1e-4f * sqrtf((float)K);   // K = length of the reduction, min 1
- Print EXACTLY one line:  printf("DIFF=%e\\n", max_rel);
- Then print SUCCESS if max_rel <= tol, else FAILURE.
- Print EXACTLY: printf("GPU Time: %f\\n", milliseconds);   (number only, no units)
- At the very start of main(), check: if (getenv("BENCH_ONLY") != NULL).
  When set, SKIP the CPU reference and the DIFF/SUCCESS/FAILURE prints entirely --
  still allocate, run the GPU kernel, and print the 'GPU Time' line as always.
"""

CORRECTNESS_RULES = """CUDA CORRECTNESS RULES — violating these produces wrong answers:
- Every thread in a block must reach EVERY __syncthreads(). Never `return` early
  (bounds check, triangular mask, etc.) before a barrier the rest of the block
  will hit. Instead, let all threads run the loop and mask only the final global
  write, where divergence is safe.
- Never read shared memory a thread never wrote for this tile.
"""

OUTPUT_RULES = """OUTPUT RULES — no exceptions:
- Return ONLY the raw .cu file.
- First character must be '#' (from #include).
- No markdown, no backticks, no explanation.
"""


class CoderAgent:
    """Generates baseline kernels and per-round optimizations."""

    def __init__(self, provider: LLMProvider, session_id: str = "coder"):
        self.provider = provider
        self.session_id = session_id

    async def _ask(self, prompt: str) -> str:
        raw = await self.provider.complete(prompt, session_id=self.session_id)
        return extract_cuda_code(raw)

    # ── baseline: PyTorch reference -> standalone .cu ──────────────────
    async def generate_baseline(self, pytorch_code: str, *, gpu_arch: str) -> str:
        prompt = (
            f"Convert this PyTorch operation to a standalone CUDA kernel for {gpu_arch}.\n\n"
            f"PYTORCH REFERENCE:\n{pytorch_code}\n\n"
            f"MANDATORY REQUIREMENTS:\n"
            f"- Complete standalone .cu file with main()\n"
            f"- Use cudaEvent_t for timing.\n"
            f"- float32 only. No half types.\n"
            f"- Must compile: nvcc -O2 -arch={gpu_arch} -lcublas -lcurand\n\n"
            f"{CORRECTNESS_RULES}\n"
            f"{HARNESS_CONTRACT}\n"
            f"{OUTPUT_RULES}"
        )
        return await self._ask(prompt)

    # ── one optimization round ────────────────────────────────────────
    async def optimize(
        self,
        kernel_source: str,
        *,
        round_num: int,
        rounds: int,
        gpu_arch: str,
        guidance: str = "",
        metrics_ctx: str = "",
        kb_ctx: str = "",
        history_ctx: str = "",
        best_ctx: str = "",
    ) -> str:
        prompt = (
            f"You are a CUDA expert optimizing for {gpu_arch}.\n"
            f"Round {round_num} of {rounds}.\n\n"
            f"{guidance}"
            f"{metrics_ctx}"
            f"{kb_ctx}"
            f"{history_ctx}"
            f"{best_ctx}"
            f"HARD CONSTRAINTS:\n"
            f"- float32 only. No half / half2 / __half / fp16.\n"
            f"- No cuda/cmath, no std::complex headers.\n"
            f"- Apply ONE focused optimization per round.\n"
            f"- If previous round had a compile error, fix THAT error first.\n"
            f"- PRESERVE THE HARNESS: keep main(), the CPU reference, the 'GPU Time'\n"
            f"  print, the 'DIFF=%e' print and the SUCCESS/FAILURE logic EXACTLY as\n"
            f"  given. Optimize only the device kernel(s) and their launch config —\n"
            f"  never delete or weaken the verification, and never change the\n"
            f"  tolerance to force a pass.\n"
            f"- Do NOT change any PROBLEM-SIZE constant (N, N_TEST, matrix dims, batch,\n"
            f"  channels). This is checked automatically and such a round is rejected.\n"
            f"  Tuning knobs (TILE_SIZE, BLOCK_DIM, THREADS_PER_BLOCK, UNROLL_FACTOR)\n"
            f"  ARE free to change — that is the point.\n"
            f"- Keep the BENCH_ONLY environment-variable check intact if present.\n\n"
            f"{CORRECTNESS_RULES}\n"
            f"{OUTPUT_RULES}"
            f"- Must compile: nvcc -O2 -arch={gpu_arch} -lcublas -lcurand\n\n"
            f"KERNEL TO OPTIMIZE:\n{kernel_source}"
        )
        return await self._ask(prompt)


# ── Backwards-compatibility shim ───────────────────────────────────────
# server.py and main.py still call the old ADK-era entry point:
#     safe_chat(prompt, runner, USER_ID, SESSION_ID) -> raw text
# The ADK arguments are now meaningless (there is no Runner and no shared
# session), but keeping the signature lets those callers work unchanged.
# Prefer CoderAgent directly in new code.
USER_ID = "user_1"
SESSION_ID = "session_001"
runner = None  # ADK Runner is gone; kept so `from Agents.coder import runner` works

_default_coder: CoderAgent | None = None


def _get_default_coder() -> CoderAgent:
    """Build the coder from karma.yaml on first use (not at import time)."""
    global _default_coder
    if _default_coder is None:
        from config import load_config
        from Agents.providers import build_provider

        cfg = load_config()
        _default_coder = CoderAgent(build_provider(cfg.for_agent("coder")))
    return _default_coder


async def safe_chat(prompt: str, runner=None, user_id: str = "", session_id: str = "") -> str:
    """Deprecated. Returns the RAW model text (callers do their own extraction)."""
    agent = _get_default_coder()
    return await agent.provider.complete(prompt, session_id=session_id or "legacy")


chat = safe_chat

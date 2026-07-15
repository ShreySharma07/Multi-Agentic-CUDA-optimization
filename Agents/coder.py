# Agents/coder.py
"""
CoderAgent — writes the CUDA kernel and its pybind binding, and nothing else.

Under the torch-extension methodology the model no longer authors a whole
program: Python owns the inputs, the PyTorch reference and the clock. So the
contract here is narrow on purpose — emit {"cuda_source", "cpp_source"} and
that is all. The old prompt's instructions about main(), printing "GPU Time",
BENCH_ONLY and preserving problem dimensions are gone, because those were only
ever there to police cheats that are now structurally impossible.

Two modes:
  translate() — round 0. Faithful, naive, correctness-only port of the PyTorch op.
  optimize()  — rounds 1..N. Plan-driven improvement of the current best kernel.
"""
from __future__ import annotations

from Agents.json_utils import extract_json
from Agents.providers import LLMProvider

# Shared contract. Both modes emit the same JSON shape, so the two prompts can't
# drift apart on the thing the pipeline actually parses.
OUTPUT_CONTRACT = """OUTPUT FORMAT — return ONLY a raw JSON object, no markdown, no prose:
{
  "cuda_source": "<complete .cu translation unit>",
  "cpp_source":  "<forward declaration + PYBIND11_MODULE block>"
}

cuda_source MUST:
- #include <torch/extension.h>
- define your __global__ kernel(s)
- define a C++ entry point:  torch::Tensor forward(<tensor args>)
  (use std::vector<torch::Tensor> only if the op returns several tensors)
- forward() must accept EXACTLY the argument list given under "forward() SIGNATURE"
  below, in that order, and return the same shape/dtype PyTorch returns
- launch the kernel; call .contiguous() on tensors you index directly

cpp_source MUST be:
  torch::Tensor forward(<same signature>);
  PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("forward", &forward); }

JSON rules: the sources are JSON strings — escape newlines as \\n and quotes as \\".
Do not wrap the JSON in code fences.
"""


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


def build_signature_ctx(input_specs, param_specs) -> str:
    """
    Describe forward()'s FULL argument list. The module's learned parameters
    (Linear/Conv weights, norm affine + running stats) are NOT part of the
    activation inputs — the kernel is a free function, so Python passes them
    explicitly, after the inputs, in this exact order. Getting this wrong is a
    hard TypeError at call time, so it is stated unambiguously.

    input_specs / param_specs: lists of (name, shape_tuple, dtype_str).
    """
    lines = ["forward() SIGNATURE — your C++ forward() MUST take these tensors in EXACTLY",
             "this order (Python calls forward(inputs..., params...)):", ""]
    n = 1
    lines.append("  activation inputs (what get_inputs() returns):")
    for i, (_, shape, dt) in enumerate(input_specs):
        lines.append(f"    arg {n}: x{i}  shape {tuple(shape)}  {dt}")
        n += 1
    if param_specs:
        lines.append("  model parameters (the nn.Module's weights/buffers, already on GPU,")
        lines.append("  holding the SAME values the reference uses — read the PyTorch source")
        lines.append("  to see how each is applied, e.g. nn.Linear does x @ weight.T + bias):")
        for name, shape, dt in param_specs:
            lines.append(f"    arg {n}: {_sanitize(name):24} shape {tuple(shape)}  {dt}")
            n += 1
    else:
        lines.append("  (this module has NO parameters — forward() takes only the inputs above)")
    lines.append("")
    lines.append("Do NOT invent, re-initialise, or hardcode any parameter — use the tensors")
    lines.append("passed in. Do NOT add or drop arguments; the call site is fixed.")
    return "\n".join(lines) + "\n"

CORRECTNESS_RULES = """CUDA CORRECTNESS RULES — violating these silently produces wrong answers:
- Every thread in a block must reach EVERY __syncthreads(). Never `return` early
  (bounds check, triangular mask, etc.) before a barrier the rest of the block
  will hit. Let all threads run the loop and mask only the final global write,
  where divergence is safe.
- Never read shared memory a thread never wrote for the current tile.
- Do not assume the input is contiguous — call .contiguous() or handle strides.
- Match PyTorch's dtype. Do not hardcode float if the task may be another dtype.
"""


class CoderAgent:
    """Emits {cuda_source, cpp_source}. Returns None when the model's reply
    cannot be parsed, so the caller records the round as an empty response."""

    def __init__(self, provider: LLMProvider, session_id: str = "coder"):
        self.provider = provider
        self.session_id = session_id

    async def _ask_json(self, prompt: str) -> dict | None:
        raw = await self.provider.complete(prompt, session_id=self.session_id)
        data = extract_json(raw)
        if not data:
            return None

        cuda = data.get("cuda_source")
        cpp = data.get("cpp_source")
        if not isinstance(cuda, str) or not cuda.strip():
            return None
        if not isinstance(cpp, str) or not cpp.strip():
            # A missing binding is recoverable — compile_extension appends a
            # standard PYBIND11_MODULE when one is absent.
            cpp = ""
        return {"cuda_source": cuda, "cpp_source": cpp}

    # ── round 0: faithful translation ─────────────────────────────────
    async def translate(self, pytorch_source: str, *, gpu_arch: str,
                        signature_ctx: str = "", error_feedback: str = "") -> dict | None:
        feedback = ""
        if error_feedback:
            feedback = (
                "YOUR PREVIOUS ATTEMPT FAILED. Fix this exact problem first:\n"
                f"{error_feedback}\n\n"
            )

        prompt = (
            f"Translate this PyTorch module into a CUDA extension for {gpu_arch}.\n\n"
            f"{feedback}"
            f"GOAL: a FAITHFUL, NAIVE implementation. Correctness only.\n"
            f"Do NOT optimize. Do not tile, vectorize, or fuse. A simple, obviously\n"
            f"correct kernel is exactly what is wanted here — it is the baseline that\n"
            f"later rounds will improve on.\n\n"
            f"PYTORCH MODULE:\n{pytorch_source}\n\n"
            f"{signature_ctx}\n"
            f"{CORRECTNESS_RULES}\n"
            f"{OUTPUT_CONTRACT}"
        )
        return await self._ask_json(prompt)

    # ── rounds 1..N: plan-driven optimization ─────────────────────────
    async def optimize(
        self,
        cuda_source: str,
        cpp_source: str,
        *,
        round_num: int,
        rounds: int,
        gpu_arch: str,
        pytorch_source: str = "",
        guidance: str = "",
        metrics_ctx: str = "",
        kb_ctx: str = "",
        history_ctx: str = "",
        best_ctx: str = "",
        error_feedback: str = "",
        signature_ctx: str = "",
        applied: list[str] | None = None,
    ) -> dict | None:
        feedback = ""
        if error_feedback:
            feedback = (
                "THE PREVIOUS ROUND FAILED. Fix this before anything else:\n"
                f"{error_feedback}\n\n"
            )

        ref = f"REFERENCE PyTorch semantics (must match exactly):\n{pytorch_source}\n\n" if pytorch_source else ""

        applied_ctx = ""
        if applied:
            applied_ctx = (
                "OPTIMIZATIONS ALREADY IN THIS KERNEL — every one of these must SURVIVE "
                "your edit:\n  " + ", ".join(applied) + "\n\n"
            )

        prompt = (
            f"You are a CUDA expert optimizing a kernel for {gpu_arch}.\n"
            f"Round {round_num} of {rounds}.\n\n"
            f"{feedback}"
            f"{applied_ctx}"
            f"{guidance}"
            f"{metrics_ctx}"
            f"{kb_ctx}"
            f"{history_ctx}"
            f"{best_ctx}"
            f"{ref}"
            f"{signature_ctx}\n"
            f"THIS CHANGE IS CUMULATIVE — read this twice:\n"
            f"- A fast kernel is a STACK of techniques applied TOGETHER, each covering a\n"
            f"  different bottleneck. Shared-memory tiling AND register blocking AND wide\n"
            f"  loads AND a conflict-free layout — all at once, not one instead of another.\n"
            f"- KEEP every optimization already present and ADD the new one(s) on top.\n"
            f"- Do NOT rewrite the kernel from scratch. Do NOT drop an existing technique to\n"
            f"  make room for a new one. Replacing a working technique with a different one\n"
            f"  typically loses everything the first one bought.\n"
            f"- If a new technique genuinely subsumes an existing one, say so in a comment\n"
            f"  and make sure the result is strictly better, not merely different.\n"
            f"- You may apply MULTIPLE techniques in one round when they belong together.\n\n"
            f"HARD CONSTRAINTS:\n"
            f"- The kernel must remain numerically correct: it is checked against PyTorch\n"
            f"  on three random seeds AND for run-to-run determinism (a race is caught here).\n"
            f"- Keep forward()'s signature and return type unchanged.\n"
            f"- You may call at:: ops or cuBLAS via at::cuda::getCurrentCUDABlasHandle() for\n"
            f"  sub-steps, but the core improvement must be your own kernel unless the plan\n"
            f"  says otherwise.\n"
            f"- Tile sizes, unroll factors, block dims and vector widths are yours to choose\n"
            f"  and tune for {gpu_arch} — they are not fixed by anyone.\n\n"
            f"REASON FIRST, THEN WRITE:\n"
            f"The plan and the playbook are a prior, not a cage. If the metrics or the code\n"
            f"itself tell you something the plan missed — a different technique, an extra one,\n"
            f"or a restructuring of the algorithm — do that instead, and leave a brief comment\n"
            f"in the kernel saying why. Measured speedup against PyTorch is what counts.\n\n"
            f"{CORRECTNESS_RULES}\n"
            f"{OUTPUT_CONTRACT}\n"
            f"CURRENT KERNEL (cuda_source):\n{cuda_source}\n\n"
            f"CURRENT BINDING (cpp_source):\n{cpp_source}\n"
        )
        return await self._ask_json(prompt)

    # ── legacy: standalone .cu (own/ and sglang/ modes) ───────────────
    # These modes still evaluate a self-contained program with its own main(),
    # CPU reference and timing. They are untouched by the torch-extension
    # migration, which applies to the kernelbench path only.
    async def optimize_standalone(
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
        from Agents.json_utils import extract_cuda_code

        prompt = (
            f"You are a CUDA expert optimizing for {gpu_arch}.\n"
            f"Round {round_num} of {rounds}.\n\n"
            f"{guidance}{metrics_ctx}{kb_ctx}{history_ctx}{best_ctx}"
            f"HARD CONSTRAINTS:\n"
            f"- float32 only. No half / half2 / __half / fp16.\n"
            f"- Apply ONE focused optimization per round.\n"
            f"- PRESERVE THE HARNESS: keep main(), the CPU reference, the 'GPU Time'\n"
            f"  print, the 'DIFF=%e' print and the SUCCESS/FAILURE logic EXACTLY as\n"
            f"  given. Never weaken the verification or change the tolerance.\n"
            f"- Do NOT change any PROBLEM-SIZE constant (N, N_TEST, batch, channels);\n"
            f"  this is checked automatically. Tuning knobs (TILE_SIZE, BLOCK_DIM,\n"
            f"  UNROLL_FACTOR) ARE free to change.\n"
            f"- Keep the BENCH_ONLY environment-variable check intact if present.\n\n"
            f"{CORRECTNESS_RULES}\n"
            f"OUTPUT RULES:\n"
            f"- Return ONLY the raw .cu file. First character must be '#'.\n"
            f"- No markdown, no backticks, no explanation.\n"
            f"- Must compile: nvcc -O2 -arch={gpu_arch} -lcublas -lcurand\n\n"
            f"KERNEL TO OPTIMIZE:\n{kernel_source}"
        )
        raw = await self.provider.complete(prompt, session_id=self.session_id)
        return extract_cuda_code(raw)


# ── Backwards-compatibility shim ───────────────────────────────────────
# server.py and main.py still call the ADK-era entry point:
#     safe_chat(prompt, runner, USER_ID, SESSION_ID) -> raw text
# The ADK arguments are inert (no Runner, no shared session). Prefer CoderAgent
# directly in new code.
USER_ID = "user_1"
SESSION_ID = "session_001"
runner = None

_default_coder: CoderAgent | None = None


def _get_default_coder() -> CoderAgent:
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

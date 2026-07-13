# knowledge/reflector.py
"""
ReflectorAgent — turns one optimization round's outcome into a structured
insight for the KnowledgeBase.

TWO THINGS THIS PROMPT MUST GET RIGHT, BOTH LEARNED THE HARD WAY:

1. THE REFLECTOR MUST KNOW WHAT THE SPEEDUP IS MEASURED AGAINST.
   Under the old standalone harness the baseline was the LLM's own naive kernel,
   so 0.99x genuinely meant "this round achieved nothing". Under the torch
   -extension methodology the baseline is PyTorch eager — i.e. cuBLAS/cuDNN,
   hand-tuned vendor code — so 0.99x means "we reached 99% of a vendor library",
   which is close to the best outcome available. Handed a bare "Speedup: 0.99x",
   the reflector concluded "the added complexity did not translate into
   measurable speedup" and wrote that into the KB as a lesson to avoid the very
   stack that had just achieved it. Interpretation scale is therefore stated
   explicitly below.

2. IT MUST SPEAK THE SAME VOCABULARY AS THE PLANNER.
   `strategy_used` used to come back as free prose ("Applied shared-memory tiling
   with register blocking, float4 vectorized loads..."). The planner reasons over
   technique IDs from Agents/playbook.py, so a prose entry can never match a
   technique name and the KB could never tell the planner "double_buffering
   failed on this kernel type before". Techniques are now emitted as IDs.
"""
from __future__ import annotations

from Agents.playbook import technique_names
from Agents.providers import LLMProvider

REFLECTION_PROMPT = """You are analyzing one round of CUDA kernel optimization. Extract a structured,
reusable insight for a knowledge base that a planner will consult on FUTURE kernels.

HOW TO READ THE SPEEDUP — this is measured against PyTorch eager, which for common
ops means cuBLAS / cuDNN: hand-tuned vendor libraries, not a naive baseline.
  >= 1.0x  : beat the vendor library. Exceptional.
  0.9-1.0x : reached ~90-100% of vendor performance. A STRONG result, not a failure.
  0.5-0.9x : respectable for generated code; real headroom remains.
  < 0.5x   : substantially slower than the vendor library.
Do NOT describe a 0.9-1.0x result as "no improvement" or "not worth the complexity".
Judge the round by whether it moved the kernel CLOSER to (or past) the vendor
library, not by whether the number exceeds 1.0.

Respond in EXACTLY this format — no other text, no explanation:
TECHNIQUES: <comma-separated technique IDs from the list below; "unknown" if none apply>
STRATEGY: <one-line description of what was tried>
INSIGHT: <one sentence on why it worked or failed>
APPLICABLE_WHEN: <one sentence on when to use this again>
AVOID_IF: <one sentence on when NOT to use it>

TECHNIQUE IDS (use these exact strings in TECHNIQUES):
{techniques}

Attempt details:
"""


def _parse_field(text: str, field: str) -> str:
    """Parse 'FIELD: value' from the response."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.upper().startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return "unknown"


def _parse_techniques(text: str) -> list[str]:
    """Technique IDs, grounded against the playbook so a hallucinated name can't
    enter the KB and silently fail to match anything later."""
    raw = _parse_field(text, "TECHNIQUES")
    if raw == "unknown":
        return []
    known = set(technique_names())
    return [t for t in (p.strip() for p in raw.split(",")) if t in known]


class ReflectorAgent:
    def __init__(self, provider: LLMProvider, session_id: str = "reflector"):
        self.provider = provider
        self.session_id = session_id

    async def reflect(
        self,
        *,
        kernel_name: str,
        bottleneck: str,
        round_num: int,
        code: str,
        result: str,          # success | compile_failed | validation_failed | dims_mismatch | unstable
        speedup: float = 0.0,
        error: str = "",
        applied: list[str] | None = None,
        baseline: str = "PyTorch eager (cuBLAS/cuDNN)",
    ) -> dict:
        prompt = (
            REFLECTION_PROMPT.replace("{techniques}", ", ".join(technique_names()))
            + f"Kernel: {kernel_name}\n"
            + f"Kernel bottleneck: {bottleneck}\n"
            + f"Round: {round_num}\n"
            + f"Result: {result}\n"
            + f"Baseline being compared against: {baseline}\n"
            + f"Speedup vs that baseline: {speedup:.3f}x\n"
            + (f"Techniques already in the kernel: {', '.join(applied)}\n" if applied else "")
            + (f"Error: {error[:300]}\n" if error else "")
            + f"Code (first 600 chars):\n{code[:600]}\n"
        )

        try:
            raw = await self.provider.complete(prompt, session_id=self.session_id)
        except Exception as e:
            print(f"    [reflect] LLM call failed: {e}")
            raw = ""

        techniques = _parse_techniques(raw)

        return {
            "kernel_name":     kernel_name,
            "bottleneck":      bottleneck,
            "round":           round_num,
            "result":          result,
            "speedup":         speedup,
            # What the speedup was measured against. The torch path compares to
            # PyTorch/cuBLAS (so ~1.0x is excellent) while the legacy path compares
            # to a naive kernel (where 2.5x is routine). Both write to the same KB,
            # so a bare number is ambiguous without this.
            "baseline":        baseline,
            # Technique IDs the planner can actually match on. Falls back to the
            # stack we know was applied, so an entry is never vocabulary-less.
            "techniques":      ",".join(techniques or (applied or [])),
            "strategy_used":   _parse_field(raw, "STRATEGY"),
            "insight":         _parse_field(raw, "INSIGHT"),
            "applicable_when": _parse_field(raw, "APPLICABLE_WHEN"),
            "avoid_if":        _parse_field(raw, "AVOID_IF"),
            "error":           error[:200] if error else "",
        }

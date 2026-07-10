# knowledge/reflector.py
"""
ReflectorAgent — turns one optimization round's outcome into a structured
insight for the KnowledgeBase.

Own class, own prompt, own LLMProvider (see karma.yaml `agents.reflector`).
Like the other agents it never raises into the pipeline: a failed reflection
yields an "unknown" insight rather than aborting the round.
"""
from __future__ import annotations

from Agents.providers import LLMProvider

REFLECTION_PROMPT = """You are analyzing a CUDA kernel optimization attempt. Extract a structured insight.

Respond in EXACTLY this format — no other text, no explanation:
STRATEGY: <one-line description of what optimization was tried>
INSIGHT: <one sentence on why it worked or failed>
APPLICABLE_WHEN: <one sentence on when to use this strategy again>
AVOID_IF: <one sentence on when NOT to use this strategy>

Attempt details:
"""


def _parse_field(text: str, field: str) -> str:
    """Parse 'FIELD: value' from the response."""
    for line in (text or "").splitlines():
        line = line.strip()
        if line.upper().startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return "unknown"


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
    ) -> dict:
        prompt = (
            REFLECTION_PROMPT
            + f"Kernel: {kernel_name}\n"
            + f"Bottleneck: {bottleneck}\n"
            + f"Round: {round_num}\n"
            + f"Result: {result}\n"
            + f"Speedup: {speedup:.2f}x\n"
            + (f"Error: {error[:200]}\n" if error else "")
            + f"Code (first 400 chars):\n{code[:400]}\n"
        )

        try:
            raw = await self.provider.complete(prompt, session_id=self.session_id)
        except Exception as e:
            print(f"    [reflect] LLM call failed: {e}")
            raw = ""

        return {
            "kernel_name":     kernel_name,
            "bottleneck":      bottleneck,
            "round":           round_num,
            "result":          result,
            "speedup":         speedup,
            "strategy_used":   _parse_field(raw, "STRATEGY"),
            "insight":         _parse_field(raw, "INSIGHT"),
            "applicable_when": _parse_field(raw, "APPLICABLE_WHEN"),
            "avoid_if":        _parse_field(raw, "AVOID_IF"),
            "error":           error[:200] if error else "",
        }

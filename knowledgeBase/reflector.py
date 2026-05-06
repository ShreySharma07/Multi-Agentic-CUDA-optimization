# knowledge/reflector.py
from Agents.coder import safe_chat, runner, USER_ID, SESSION_ID
import asyncio

REFLECTION_PROMPT = """
You are analyzing a CUDA kernel optimization attempt.

Given this attempt, extract a structured insight.

Respond in this EXACT format (no other text):
STRATEGY: <one-line description of what was tried>
INSIGHT: <one sentence on why it worked or failed>
APPLICABLE_WHEN: <one sentence on when to use this again>
AVOID_IF: <one sentence on when NOT to use this>

Attempt details:
"""

async def reflect(
    kernel_name: str,
    bottleneck: str,
    round_num: int,
    code: str,
    result: str,       # "success" | "compile_failed" | "validation_failed"
    speedup: float,
    error: str = "",
) -> dict:
    """Extract structured insight from one optimization round."""

    prompt = (
        REFLECTION_PROMPT
        + f"Kernel: {kernel_name}\n"
        + f"Bottleneck: {bottleneck}\n"
        + f"Round: {round_num}\n"
        + f"Result: {result}\n"
        + f"Speedup: {speedup:.2f}x\n"
        + (f"Error: {error[:200]}\n" if error else "")
        + f"Code snippet (first 500 chars):\n{code[:500]}\n"
    )

    raw = await safe_chat(prompt, runner, USER_ID, SESSION_ID)

    # parse the structured response
    insight = {
        "kernel_name":       kernel_name,
        "bottleneck":        bottleneck,
        "round":             round_num,
        "result":            result,
        "speedup":           speedup,
        "strategy_used":     _parse_field(raw, "STRATEGY"),
        "insight":           _parse_field(raw, "INSIGHT"),
        "applicable_when":   _parse_field(raw, "APPLICABLE_WHEN"),
        "avoid_if":          _parse_field(raw, "AVOID_IF"),
        "error":             error[:200],
    }

    return insight

def _parse_field(text: str, field: str) -> str:
    for line in text.splitlines():
        if line.startswith(f"{field}:"):
            return line[len(field)+1:].strip()
    return "unknown"
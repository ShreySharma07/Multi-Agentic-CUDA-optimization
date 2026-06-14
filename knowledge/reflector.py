# knowledge/reflector.py
import asyncio


REFLECTION_PROMPT = """You are analyzing a CUDA kernel optimization attempt. Extract a structured insight.

Respond in EXACTLY this format — no other text, no explanation:
STRATEGY: <one-line description of what optimization was tried>
INSIGHT: <one sentence on why it worked or failed>
APPLICABLE_WHEN: <one sentence on when to use this strategy again>
AVOID_IF: <one sentence on when NOT to use this strategy>

Attempt details:
"""


async def reflect(
    safe_chat_fn,
    runner,
    user_id: str,
    session_id: str,
    kernel_name: str,
    bottleneck: str,
    round_num: int,
    code: str,
    result: str,       # "success" | "compile_failed" | "validation_failed"
    speedup: float = 0.0,
    error: str = "",
) -> dict:
    """
    Call the LLM to extract a structured insight from one optimization round.
    Returns a dict ready to be stored in KnowledgeBase.
    """

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
        raw = await safe_chat_fn(prompt, runner, user_id, session_id)
    except Exception as e:
        print(f"    [reflect] LLM call failed: {e}")
        raw = ""

    insight = {
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

    return insight


def _parse_field(text: str, field: str) -> str:
    """Parse 'FIELD: value' from LLM response."""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith(f"{field}:"):
            return line[len(field) + 1:].strip()
    return "unknown"
# Agents/json_utils.py
"""Tolerant JSON extraction shared by the agents that ask for structured output.

Deliberately a free function rather than a base class: the agents are otherwise
independent (own prompt, own provider, own fallback), and inheritance would only
couple them.
"""
from __future__ import annotations

import json


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object out of an LLM response, or None.

    Handles the usual sins: markdown fences, prose before/after, trailing commentary.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        obj = json.loads(text[start:end + 1])
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def extract_cuda_code(text: str) -> str:
    """Strip markdown fences / 'cppcopy' artifacts and return raw .cu source."""
    for fence in ["```cpp", "```cuda", "```c", "cppcopy", "```"]:
        if fence in text:
            after = text.split(fence, 1)[1]
            end = after.find("```")
            text = after[:end] if end != -1 else after
            break
    if "#include" in text:
        return text[text.find("#include"):].strip()
    if "__global__" in text:
        return text[text.find("__global__"):].strip()
    return text.strip()

# Agents/providers/__init__.py
"""
Provider registry.

Vendor SDKs are imported lazily, inside each provider's __init__, so you only
need the SDK for the provider(s) you actually configure -- `pip install openai`
is not a prerequisite for running on Gemini.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .base import LLMError, LLMProvider

if TYPE_CHECKING:  # avoid importing config at runtime (keeps this module standalone)
    from config import AgentSettings

_REGISTRY = {}


def _load(name: str) -> type[LLMProvider]:
    if name not in _REGISTRY:
        if name == "gemini":
            from .gemini import GeminiProvider as cls
        elif name == "anthropic":
            from .anthropic import AnthropicProvider as cls
        elif name == "openai":
            from .openai import OpenAIProvider as cls
        elif name == "ollama":
            from .ollama import OllamaProvider as cls
        else:
            raise ValueError(
                f"Unknown provider {name!r}. Supported: gemini, anthropic, openai, ollama"
            )
        _REGISTRY[name] = cls
    return _REGISTRY[name]


def build_provider(settings: "AgentSettings") -> LLMProvider:
    """Construct the provider an agent's config asks for."""
    cls = _load(settings.provider)
    return cls(
        model=settings.model,
        api_key=settings.api_key,
        temperature=settings.temperature,
        max_retries=settings.max_retries,
        timeout=settings.timeout,
        base_url=settings.base_url,
    )


__all__ = ["LLMProvider", "LLMError", "build_provider"]

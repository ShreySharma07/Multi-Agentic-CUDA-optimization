# Agents/providers/anthropic.py
"""Anthropic Claude via the official anthropic SDK."""
from __future__ import annotations

from .base import LLMProvider

# Claude requires an explicit output cap. Kernels are long; leave plenty of room.
DEFAULT_MAX_TOKENS = 16384


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, *args, max_tokens: int = DEFAULT_MAX_TOKENS, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' provider needs the Anthropic SDK: pip install anthropic"
            ) from e

        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=self.api_key, timeout=self.timeout)

    async def _generate(self, prompt: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        # content is a list of blocks; concatenate the text ones
        return "".join(block.text for block in resp.content if block.type == "text")

    def _is_retryable(self, exc: Exception) -> bool:
        import anthropic

        return isinstance(
            exc,
            (
                anthropic.RateLimitError,
                anthropic.APITimeoutError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ),
        )

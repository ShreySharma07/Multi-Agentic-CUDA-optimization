# Agents/providers/openai.py
"""OpenAI (and any OpenAI-compatible endpoint) via the official openai SDK."""
from __future__ import annotations

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' provider needs the OpenAI SDK: pip install openai"
            ) from e

        # base_url lets this class also serve vLLM / LM Studio / Together / etc.
        self._client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    async def _generate(self, prompt: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""

    def _is_retryable(self, exc: Exception) -> bool:
        import openai

        return isinstance(
            exc,
            (
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
                openai.InternalServerError,
            ),
        )

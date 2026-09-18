# Agents/providers/gemini.py
"""Google Gemini via the google-genai SDK (no ADK, no Runner, no session)."""
from __future__ import annotations

from .base import LLMProvider


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'gemini' provider needs the google-genai SDK: pip install google-genai"
            ) from e

        self._genai = genai
        self._client = genai.Client(api_key=self.api_key)

    async def _generate(self, prompt: str) -> str:
        from google.genai import types

        resp = await self._client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        return resp.text or ""

    def _is_retryable(self, exc: Exception) -> bool:
        from google.genai import errors

        if isinstance(exc, errors.APIError):
            # 429 = rate limit / RESOURCE_EXHAUSTED, 5xx = transient server error
            code = getattr(exc, "code", None)
            return code == 429 or (isinstance(code, int) and 500 <= code < 600)

        # httpx connect/read timeouts surface as plain OSError subclasses
        return isinstance(exc, (TimeoutError, ConnectionError))

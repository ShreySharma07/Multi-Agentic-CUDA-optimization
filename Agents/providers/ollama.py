# Agents/providers/ollama.py
"""Local models via an Ollama server. No SDK, no API key -- just HTTP."""
from __future__ import annotations

from .base import LLMProvider

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'ollama' provider needs httpx: pip install httpx"
            ) from e

        self._httpx = httpx
        self._url = (self.base_url or DEFAULT_BASE_URL).rstrip("/")

    async def _generate(self, prompt: str) -> str:
        async with self._httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self._url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    def _is_retryable(self, exc: Exception) -> bool:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError):
            return exc.response.status_code >= 500
        return isinstance(exc, (httpx.TimeoutException, httpx.ConnectError))

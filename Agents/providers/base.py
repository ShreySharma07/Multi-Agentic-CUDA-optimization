# Agents/providers/base.py
"""
Provider-agnostic LLM interface.

Every agent talks to exactly one LLMProvider and never imports a vendor SDK.
A provider is stateless: `complete(prompt) -> str`. There is no conversation
history, which is deliberate -- all four KARMA agents are single-turn (send one
prompt, parse one reply), and the previous ADK implementation shared a single
session across all of them, so each agent's output silently accumulated in every
other agent's context.

`session_id` is therefore a *correlation id* for tracing, not conversation state.

Retry/backoff lives here so it is identical across providers. Providers only
declare which of their own exceptions are worth retrying, via `_is_retryable`.
"""
from __future__ import annotations

import abc
import asyncio
import random
import time


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion after all retries."""


class LLMProvider(abc.ABC):
    #: short provider name, e.g. "gemini" -- used in trace output
    name: str = "base"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.2,
        max_retries: int = 5,
        timeout: int = 120,
        base_url: str | None = None,
    ):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.base_url = base_url

    # ── subclass contract ──────────────────────────────────────────────
    @abc.abstractmethod
    async def _generate(self, prompt: str) -> str:
        """One raw call to the vendor SDK. May raise anything."""

    @abc.abstractmethod
    def _is_retryable(self, exc: Exception) -> bool:
        """True for rate limits / transient network / 5xx."""

    # ── public API ─────────────────────────────────────────────────────
    async def complete(self, prompt: str, *, session_id: str = "") -> str:
        """
        Retry with exponential backoff + jitter. Returns "" only if the provider
        legitimately produced empty text; genuine failures raise LLMError so a
        misconfigured key surfaces immediately rather than being mistaken for a
        model that "returned no code".
        """
        last_exc: Exception | None = None
        attempt = 0

        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                text = await self._generate(prompt)
                self._trace(session_id, attempt, time.perf_counter() - started, ok=True)
                return text or ""
            except Exception as e:  # noqa: BLE001 - normalized below
                last_exc = e
                self._trace(session_id, attempt, time.perf_counter() - started, ok=False, exc=e)

                if not self._is_retryable(e) or attempt == self.max_retries:
                    break

                # 2s, 4s, 8s ... capped, with jitter to avoid thundering herd
                backoff = min(2 ** attempt, 60) + random.uniform(0, 1)
                print(f"    [{self.name}] {e.__class__.__name__} — retrying in "
                      f"{backoff:.1f}s ({attempt}/{self.max_retries})")
                await asyncio.sleep(backoff)

        raise LLMError(
            f"{self.name}:{self.model} failed after {attempt} attempt(s): "
            f"{last_exc.__class__.__name__}: {last_exc}"
        ) from last_exc

    def _trace(self, session_id, attempt, elapsed, *, ok, exc=None):
        """Single choke point for observability. Kept intentionally minimal for
        now; a JSONL sink slots in here without touching any agent."""
        if not ok and exc is not None:
            return  # retry loop already reports; avoid double-printing

    def describe(self) -> str:
        return f"{self.name}:{self.model}"

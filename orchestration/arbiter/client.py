"""LLM client interface and an Anthropic adapter. The Arbiter only needs one method, so any provider can be added."""
from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str, *, timeout_s: float) -> str: ...


class AnthropicClient:
    """Calls the Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment (never from code).

    The model is configurable: pass `model=` or set SENTINEL_ARBITER_MODEL. A small fast model suits a 3 s budget.
    No tools are passed, temperature is 0, and the reply is capped at 200 tokens.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 200, send_temperature: bool = True):
        self.model = model or os.environ.get("SENTINEL_ARBITER_MODEL", "claude-haiku-5-5")
        self.max_tokens, self.send_temperature = max_tokens, send_temperature
        self._client = None

    def _get(self):
        if self._client is None:
            import anthropic                                   # lazy: tests and other providers do not need it
            self._client = anthropic.Anthropic(max_retries=0)  # no hidden retries inside a 3 s budget
        return self._client

    def complete(self, system: str, user: str, *, timeout_s: float) -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}], timeout=timeout_s)
        if self.send_temperature:
            kwargs["temperature"] = 0
        resp = self._get().messages.create(**kwargs)
        return "".join(getattr(b, "text", "") for b in resp.content)

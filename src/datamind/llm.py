"""LLM client abstraction and the Claude implementation.

Every backend (Claude, or the local fine-tuned model served by vLLM) implements
:class:`LLMClient` and returns an :class:`LLMResponse` carrying token usage,
cost and latency, so the agent can fill ``QueryMetrics`` uniformly.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict

import anthropic
import httpx

from datamind.pricing import GpuPricing, ModelPricing, resolve_pricing
from datamind.schema import Route

__all__ = ["ClaudeClient", "LLMClient", "LLMError", "LLMResponse", "Message", "VLLMClient"]

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model call failed or returned unusable output."""


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: float
    model: str
    route: Route
    avg_logprob: float | None = None  # mean token log-probability, if the backend reports it


class LLMClient(Protocol):
    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        """Return the model's reply. Raises :class:`LLMError` on failure."""
        ...


class ClaudeClient:
    """Async Anthropic Messages API client with cost accounting.

    Transient failures (connection errors, 429, 5xx) are retried by the SDK
    (``max_retries``); anything that still fails surfaces as :class:`LLMError`.

    ``temperature`` is omitted from requests unless you set it, because newer
    Claude models may reject sampling parameters.
    """

    route = Route.CLAUDE

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        *,
        api_key: str | None = None,
        pricing: ModelPricing | None = None,
        max_tokens: int = 1024,
        temperature: float | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if max_tokens < 1 or timeout_s <= 0 or max_retries < 0:
            raise ValueError("max_tokens, timeout_s and max_retries are out of range")
        self._model = model
        self._pricing = pricing or resolve_pricing(model)
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_s, max_retries=max_retries
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": [dict(m) for m in messages],
        }
        if self._temperature is not None:
            request["temperature"] = self._temperature

        started = time.perf_counter()
        try:
            response = await self._client.messages.create(**request)
        except anthropic.APIError as exc:
            raise LLMError(f"Claude API error: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        text = "".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not text:
            raise LLMError("Claude returned no text content")

        prompt_tokens = int(response.usage.input_tokens)
        completion_tokens = int(response.usage.output_tokens)
        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._pricing.cost(prompt_tokens, completion_tokens),
            latency_ms=latency_ms,
            model=self._model,
            route=self.route,
        )

    async def aclose(self) -> None:
        await self._client.close()


class VLLMClient:
    """Async client for a vLLM OpenAI-compatible server (``vllm serve``).

    Cost is attributed by GPU time via :class:`GpuPricing` (see its docstring).
    Requests ask for token log-probabilities so the router can use the mean
    log-probability of the generated SQL as a confidence signal.

    Run the benchmark on the same machine as the server (e.g. inside the Colab
    notebook against ``localhost``): latency measured through a public tunnel
    includes network time that is not model time.
    """

    route = Route.LOCAL

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000/v1",
        api_key: str | None = None,
        pricing: GpuPricing,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_tokens < 1 or timeout_s <= 0 or temperature < 0:
            raise ValueError("max_tokens, timeout_s and temperature are out of range")
        self._model = model
        self._pricing = pricing
        self._max_tokens = max_tokens
        self._temperature = temperature
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/", headers=headers, timeout=timeout_s
        )

    @property
    def model(self) -> str:
        return self._model

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *(dict(m) for m in messages)],
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "logprobs": True,
        }
        started = time.perf_counter()
        try:
            response = await self._client.post("chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise LLMError(f"vLLM server returned {exc.response.status_code}: {detail}") from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise LLMError(f"vLLM request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            choice = body["choices"][0]
            text = str(choice["message"]["content"] or "").strip()
            usage = body["usage"]
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError(f"Unexpected vLLM response shape: {exc!r}") from exc
        if not text:
            raise LLMError("vLLM returned no text content")

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=self._pricing.cost(latency_ms),
            latency_ms=latency_ms,
            model=self._model,
            route=self.route,
            avg_logprob=_mean_logprob(choice),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


def _mean_logprob(choice: dict[str, Any]) -> float | None:
    """Mean token log-probability of a chat-completion choice, or None if absent."""
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not content:
        return None
    values = [t["logprob"] for t in content if isinstance(t, dict) and isinstance(t.get("logprob"), (int, float))]
    return sum(values) / len(values) if values else None

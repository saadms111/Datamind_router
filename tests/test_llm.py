"""Tests for ClaudeClient using a fake Anthropic client (no network)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest

from datamind.llm import ClaudeClient, LLMError
from datamind.pricing import ModelPricing
from datamind.schema import Route


class FakeAnthropic:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._response = response
        self._error = error
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return self._response


def _response(text: str = "```sql\nSELECT 1\n```", inp: int = 1000, out: int = 200) -> Any:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=inp, output_tokens=out),
    )


def _client(fake: FakeAnthropic, **kwargs: Any) -> ClaudeClient:
    return ClaudeClient(
        "claude-sonnet-5-5", pricing=ModelPricing(2.0, 10.0), client=fake, **kwargs
    )


def test_generate_returns_usage_and_cost() -> None:
    fake = FakeAnthropic(_response(inp=1_000_000, out=100_000))
    reply = asyncio.run(_client(fake).generate(system="s", messages=[{"role": "user", "content": "q"}]))
    assert reply.text.startswith("```sql")
    assert (reply.prompt_tokens, reply.completion_tokens) == (1_000_000, 100_000)
    assert reply.cost_usd == pytest.approx(3.0)
    assert reply.route is Route.CLAUDE
    assert reply.model == "claude-sonnet-5-5"
    assert reply.latency_ms >= 0


def test_temperature_only_sent_when_set() -> None:
    fake = FakeAnthropic(_response())
    asyncio.run(_client(fake).generate(system="s", messages=[{"role": "user", "content": "q"}]))
    assert "temperature" not in fake.calls[0]

    fake = FakeAnthropic(_response())
    asyncio.run(
        _client(fake, temperature=0.0).generate(system="s", messages=[{"role": "user", "content": "q"}])
    )
    assert fake.calls[0]["temperature"] == 0.0


def test_api_error_becomes_llm_error() -> None:
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    client = _client(FakeAnthropic(error=err))
    with pytest.raises(LLMError):
        asyncio.run(client.generate(system="s", messages=[{"role": "user", "content": "q"}]))


def test_empty_reply_is_llm_error() -> None:
    client = _client(FakeAnthropic(_response(text="   ")))
    with pytest.raises(LLMError):
        asyncio.run(client.generate(system="s", messages=[{"role": "user", "content": "q"}]))

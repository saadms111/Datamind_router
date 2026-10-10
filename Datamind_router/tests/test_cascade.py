"""Tests for the cascade router, GPU pricing and VLLMClient (all offline)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import DatabaseCatalog
from datamind.llm import LLMError, LLMResponse, Message, VLLMClient
from datamind.pricing import GpuPricing
from datamind.schema import QueryMetrics, QueryRequest, QueryStatus, Route, RoutePreference
from router.cascade import CascadeAgent, RoutingPolicy


# ------------------------------------------------------------------ GPU pricing
def test_gpu_pricing_arithmetic() -> None:
    assert GpuPricing(3.6).cost(1000.0) == pytest.approx(0.001)  # 1s of a $3.60/h GPU
    assert GpuPricing(3.6, concurrent_requests=4).cost(1000.0) == pytest.approx(0.00025)
    assert GpuPricing(0.0).cost(5000.0) == 0.0


def test_gpu_pricing_validation() -> None:
    with pytest.raises(ValueError):
        GpuPricing(-1.0)
    with pytest.raises(ValueError):
        GpuPricing(1.0, concurrent_requests=0)
    with pytest.raises(ValueError):
        GpuPricing(1.0).cost(-1.0)


# ------------------------------------------------------------------- VLLMClient
def _chat_body(
    text: str = "```sql\nSELECT 1\n```", logprobs: list[float] | None = (-0.1, -0.3)  # type: ignore[assignment]
) -> dict[str, Any]:
    choice: dict[str, Any] = {"message": {"role": "assistant", "content": text}}
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": "t", "logprob": lp} for lp in logprobs]}
    return {"choices": [choice], "usage": {"prompt_tokens": 120, "completion_tokens": 8}}


def _client(handler: Any, pricing: GpuPricing | None = None) -> VLLMClient:
    http = httpx.AsyncClient(
        base_url="http://vllm.test/v1/", transport=httpx.MockTransport(handler), timeout=5.0
    )
    return VLLMClient("datamind-qwen", pricing=pricing or GpuPricing(3.6), client=http)


def _generate(client: VLLMClient) -> LLMResponse:
    return asyncio.run(client.generate(system="sys", messages=[{"role": "user", "content": "q"}]))


def test_vllm_success_and_request_shape() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_body())

    reply = _generate(_client(handler))
    assert seen["url"] == "http://vllm.test/v1/chat/completions"
    assert seen["json"]["model"] == "datamind-qwen"
    assert seen["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert seen["json"]["messages"][1] == {"role": "user", "content": "q"}
    assert seen["json"]["logprobs"] is True and seen["json"]["temperature"] == 0.0
    assert reply.text.startswith("```sql")
    assert (reply.prompt_tokens, reply.completion_tokens) == (120, 8)
    assert reply.avg_logprob == pytest.approx(-0.2)
    assert reply.route is Route.LOCAL and reply.model == "datamind-qwen"
    assert reply.cost_usd == pytest.approx(reply.latency_ms / 1000 / 3600 * 3.6)


def test_vllm_missing_logprobs_gives_none() -> None:
    reply = _generate(_client(lambda r: httpx.Response(200, json=_chat_body(logprobs=None))))
    assert reply.avg_logprob is None


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json=_chat_body(text="   ")),
        httpx.Response(200, text="not json"),
    ],
)
def test_vllm_failures_become_llm_error(response: httpx.Response) -> None:
    with pytest.raises(LLMError):
        _generate(_client(lambda r: response))


def test_vllm_connection_error_becomes_llm_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(LLMError):
        _generate(_client(handler))


# ---------------------------------------------------------------------- metrics
def test_metrics_merge_keeps_latest_confidence() -> None:
    a = QueryMetrics(confidence=-0.5, llm_calls=1)
    b = QueryMetrics(confidence=None, llm_calls=1)
    assert a.merge(b).confidence == -0.5
    assert a.merge(QueryMetrics(confidence=-0.1)).confidence == -0.1


# --------------------------------------------------------------------- cascade
class ScriptedLLM:
    def __init__(
        self, replies: Sequence[str], route: Route, *, logprob: float | None, cost: float, latency_ms: float
    ) -> None:
        self._replies = list(replies)
        self._route, self._logprob, self._cost, self._latency = route, logprob, cost, latency_ms
        self.calls = 0

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            text=f"```sql\n{self._replies.pop(0)}\n```",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=self._cost,
            latency_ms=self._latency,
            model=f"{self._route.value}-model",
            route=self._route,
            avg_logprob=self._logprob,
        )


@pytest.fixture
def catalog(tmp_path: Path) -> DatabaseCatalog:
    db_dir = tmp_path / "dbs" / "music"
    db_dir.mkdir(parents=True)
    conn = sqlite3.connect(db_dir / "music.sqlite")
    conn.execute("CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany("INSERT INTO singer (name, age) VALUES (?, ?)", [("a", 30), ("b", 20), ("c", 40)])
    conn.commit()
    conn.close()
    return DatabaseCatalog.from_directory(tmp_path / "dbs")


def _cascade(
    catalog: DatabaseCatalog,
    local_replies: Sequence[str],
    remote_replies: Sequence[str],
    *,
    local_logprob: float | None = -0.05,
    policy: RoutingPolicy | None = None,
) -> tuple[CascadeAgent, ScriptedLLM, ScriptedLLM]:
    local_llm = ScriptedLLM(local_replies, Route.LOCAL, logprob=local_logprob, cost=0.0001, latency_ms=50.0)
    remote_llm = ScriptedLLM(remote_replies, Route.CLAUDE, logprob=None, cost=0.01, latency_ms=800.0)
    cfg = AgentConfig(max_attempts=1)
    agent = CascadeAgent(
        DataMindAgent(local_llm, catalog, cfg), DataMindAgent(remote_llm, catalog, cfg), policy
    )
    return agent, local_llm, remote_llm


def _ask(agent: CascadeAgent, preference: RoutePreference = RoutePreference.AUTO):
    request = QueryRequest(question="names?", db_id="music", route_preference=preference)
    return asyncio.run(agent.answer(request))


def test_local_success_is_accepted_and_claude_not_called(catalog: DatabaseCatalog) -> None:
    agent, local_llm, remote_llm = _cascade(catalog, ["SELECT name FROM singer"], [])
    response = _ask(agent)
    assert response.status is QueryStatus.SUCCESS
    assert response.metrics.route is Route.LOCAL
    assert (local_llm.calls, remote_llm.calls) == (1, 0)
    assert response.metrics.cost_usd == pytest.approx(0.0001)


def test_guard_rejection_escalates_and_costs_are_summed(catalog: DatabaseCatalog) -> None:
    agent, local_llm, remote_llm = _cascade(catalog, ["DROP TABLE singer"], ["SELECT name FROM singer"])
    response = _ask(agent)
    assert response.status is QueryStatus.SUCCESS and response.rows == [["a"], ["b"], ["c"]]
    m = response.metrics
    assert m.route is Route.CLAUDE and m.model == "claude-model"
    assert m.llm_calls == 2 and m.total_tokens == 220
    assert m.cost_usd == pytest.approx(0.0101)
    assert m.latency_ms >= 0  # merged latency includes the wasted local attempt
    assert (local_llm.calls, remote_llm.calls) == (1, 1)


def test_execution_error_escalates(catalog: DatabaseCatalog) -> None:
    bad = "SELECT name FROM singer UNION SELECT name, age FROM singer"
    agent, _, remote_llm = _cascade(catalog, [bad], ["SELECT name FROM singer"])
    response = _ask(agent)
    assert response.metrics.route is Route.CLAUDE and remote_llm.calls == 1


def test_low_confidence_escalates_even_if_sql_ran(catalog: DatabaseCatalog) -> None:
    agent, _, remote_llm = _cascade(
        catalog,
        ["SELECT name FROM singer"],
        ["SELECT name FROM singer ORDER BY age"],
        local_logprob=-0.9,
        policy=RoutingPolicy(min_confidence=-0.2),
    )
    response = _ask(agent)
    assert response.metrics.route is Route.CLAUDE and remote_llm.calls == 1
    assert response.rows == [["b"], ["a"], ["c"]]


def test_high_confidence_stays_local(catalog: DatabaseCatalog) -> None:
    agent, _, remote_llm = _cascade(
        catalog, ["SELECT name FROM singer"], [], local_logprob=-0.05, policy=RoutingPolicy(-0.2)
    )
    assert _ask(agent).metrics.route is Route.LOCAL and remote_llm.calls == 0


def test_missing_confidence_is_accepted(catalog: DatabaseCatalog) -> None:
    agent, _, remote_llm = _cascade(
        catalog, ["SELECT name FROM singer"], [], local_logprob=None, policy=RoutingPolicy(-0.2)
    )
    assert _ask(agent).metrics.route is Route.LOCAL and remote_llm.calls == 0


@pytest.mark.parametrize(
    ("preference", "expected_route", "local_calls", "remote_calls"),
    [(RoutePreference.LOCAL, Route.LOCAL, 1, 0), (RoutePreference.CLAUDE, Route.CLAUDE, 0, 1)],
)
def test_forced_routes(
    catalog: DatabaseCatalog,
    preference: RoutePreference,
    expected_route: Route,
    local_calls: int,
    remote_calls: int,
) -> None:
    agent, local_llm, remote_llm = _cascade(
        catalog, ["SELECT name FROM singer"], ["SELECT name FROM singer"]
    )
    response = _ask(agent, preference)
    assert response.metrics.route is expected_route
    assert (local_llm.calls, remote_llm.calls) == (local_calls, remote_calls)


def test_forced_local_failure_is_not_escalated(catalog: DatabaseCatalog) -> None:
    agent, _, remote_llm = _cascade(catalog, ["DROP TABLE singer"], ["SELECT name FROM singer"])
    response = _ask(agent, RoutePreference.LOCAL)
    assert response.status is QueryStatus.REJECTED and remote_llm.calls == 0


def test_both_backends_failing_returns_claude_failure(catalog: DatabaseCatalog) -> None:
    agent, _, _ = _cascade(catalog, ["DROP TABLE singer"], ["DELETE FROM singer"])
    response = _ask(agent)
    assert response.status is QueryStatus.REJECTED
    assert response.metrics.llm_calls == 2 and response.metrics.route is Route.CLAUDE


def test_policy_rejects_positive_threshold() -> None:
    with pytest.raises(ValueError):
        RoutingPolicy(min_confidence=0.5)

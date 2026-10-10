"""Tests for the FastMCP server and the backend factory.

The server is exercised through FastMCP's in-memory client (no sockets, no
network, no API key); the agent behind it uses a scripted LLM.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import DatabaseCatalog
from datamind.llm import LLMResponse, Message
from datamind.schema import QueryRequest, QueryResponse, QueryStatus, Route
from mcp_server.server import ServerConfig, create_server
from router.cascade import CascadeAgent
from router.factory import BackendSettings, ConfigError, build_answerer


class ScriptedLLM:
    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        return LLMResponse(
            text=f"```sql\n{self._replies.pop(0)}\n```",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=0.002,
            latency_ms=5.0,
            model="fake-model",
            route=Route.CLAUDE,
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


def _server(
    catalog: DatabaseCatalog, replies: Sequence[str] = (), config: ServerConfig | None = None
) -> FastMCP:
    agent = DataMindAgent(ScriptedLLM(replies), catalog, AgentConfig(max_attempts=1))
    return create_server(agent, catalog, config)


def run_client(server: FastMCP, scenario: Callable[[Client], Awaitable[Any]]) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await scenario(client)

    return asyncio.run(go())


def call(server: FastMCP, name: str, args: dict[str, Any]) -> Any:
    return run_client(server, lambda c: c.call_tool(name, args))


def call_error(server: FastMCP, name: str, args: dict[str, Any]) -> str | None:
    """Message of the error a tool call produces, or None if it succeeded."""

    async def scenario(client: Client) -> str | None:
        try:
            result = await client.call_tool(name, args)
        except ToolError as exc:
            return str(exc)
        if getattr(result, "is_error", False):
            return " ".join(getattr(block, "text", "") for block in result.content)
        return None

    return run_client(server, scenario)


# ----------------------------------------------------------------------- discovery
def test_lists_expected_tools_all_read_only(catalog: DatabaseCatalog) -> None:
    tools = run_client(_server(catalog), lambda c: c.list_tools())
    assert {t.name for t in tools} == {
        "query_database",
        "list_databases",
        "describe_database",
        "validate_sql",
    }
    assert all(t.annotations is not None and t.annotations.readOnlyHint is True for t in tools)


def test_list_and_describe_database(catalog: DatabaseCatalog) -> None:
    server = _server(catalog)
    assert call(server, "list_databases", {}).structured_content == {"databases": ["music"]}
    described = call(server, "describe_database", {"db_id": "music"}).structured_content
    assert described["tables"] == {"singer": ["singer_id", "name", "age"]}
    assert "CREATE TABLE singer" in described["schema_ddl"]
    assert "Unknown database" in (call_error(server, "describe_database", {"db_id": "nope"}) or "")


# ------------------------------------------------------------------ query_database
def test_query_success_returns_rows_and_metrics(catalog: DatabaseCatalog) -> None:
    server = _server(catalog, ["SELECT name FROM singer"])
    payload = call(server, "query_database", {"question": "names?", "db_id": "music"}).structured_content
    assert payload["status"] == "success"
    assert payload["columns"] == ["name"]
    assert payload["rows"] == [["a"], ["b"], ["c"]]
    assert payload["row_count"] == 3
    assert payload["metrics"]["llm_calls"] == 1
    assert payload["metrics"]["cost_usd"] == pytest.approx(0.002)
    assert payload["metrics"]["route"] == "claude"
    assert payload["validation"]["is_valid"] is True


def test_rejected_sql_is_data_not_a_tool_error(catalog: DatabaseCatalog) -> None:
    server = _server(catalog, ["DROP TABLE singer"])
    result = call(server, "query_database", {"question": "wipe it", "db_id": "music"})
    payload = result.structured_content
    assert payload["status"] == "rejected"
    assert "non_select_statement" in payload["error"]
    assert payload["rows"] == []


def test_max_rows_is_applied(catalog: DatabaseCatalog) -> None:
    server = _server(catalog, ["SELECT name FROM singer"])
    payload = call(
        server, "query_database", {"question": "all", "db_id": "music", "max_rows": 2}
    ).structured_content
    assert payload["row_count"] == 2 and payload["truncated"] is True


def test_unknown_database_is_an_error(catalog: DatabaseCatalog) -> None:
    message = call_error(_server(catalog), "query_database", {"question": "q", "db_id": "nope"})
    assert message is not None and "Unknown database" in message


@pytest.mark.parametrize(
    "args",
    [
        {"question": "q", "db_id": "music", "max_rows": 0},
        {"question": "q", "db_id": "music", "max_rows": 5000},
        {"question": "   ", "db_id": "music"},
        {"question": "q", "db_id": "../etc"},
        {"question": "q", "db_id": "music", "route": "gpt"},
    ],
)
def test_invalid_arguments_are_errors(catalog: DatabaseCatalog, args: dict[str, Any]) -> None:
    assert call_error(_server(catalog), "query_database", args) is not None


class RecordingAnswerer:
    """Returns a canned success and records the requests it receives."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.requests: list[QueryRequest] = []
        self.delay_s = delay_s
        self.active = 0
        self.max_active = 0

    async def answer(self, request: QueryRequest) -> QueryResponse:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_s)
        finally:
            self.active -= 1
        return QueryResponse(
            request_id=request.request_id,
            question=request.question,
            db_id=request.db_id,
            status=QueryStatus.SUCCESS,
            sql="SELECT 1",
            columns=["x"],
            rows=[[1]],
        )


def test_route_is_forwarded(catalog: DatabaseCatalog) -> None:
    answerer = RecordingAnswerer()
    server = create_server(answerer, catalog)
    call(server, "query_database", {"question": "q", "db_id": "music", "route": "local"})
    assert answerer.requests[0].route_preference.value == "local"
    assert answerer.requests[0].max_rows == 100


def test_request_timeout_is_an_error(catalog: DatabaseCatalog) -> None:
    server = create_server(
        RecordingAnswerer(delay_s=5.0), catalog, ServerConfig(request_timeout_s=0.2)
    )
    message = call_error(server, "query_database", {"question": "q", "db_id": "music"})
    assert message is not None and "request limit" in message


def test_concurrency_is_bounded(catalog: DatabaseCatalog) -> None:
    answerer = RecordingAnswerer(delay_s=0.05)
    server = create_server(answerer, catalog, ServerConfig(max_concurrent=2))

    async def scenario(client: Client) -> list[Any]:
        return await asyncio.gather(
            *(client.call_tool("query_database", {"question": f"q{i}", "db_id": "music"}) for i in range(5))
        )

    results = run_client(server, scenario)
    assert len(results) == 5 and len(answerer.requests) == 5
    assert 1 <= answerer.max_active <= 2


def test_internal_errors_are_masked(catalog: DatabaseCatalog) -> None:
    class Exploding:
        async def answer(self, request: QueryRequest) -> QueryResponse:
            raise RuntimeError("secret internal detail")

    message = call_error(
        create_server(Exploding(), catalog), "query_database", {"question": "q", "db_id": "music"}
    )
    assert message is not None and "secret internal detail" not in message


# --------------------------------------------------------------------- validate_sql
def test_validate_sql(catalog: DatabaseCatalog) -> None:
    server = _server(catalog)
    ok = call(server, "validate_sql", {"sql": "SELECT name FROM singer"}).structured_content
    assert ok["is_valid"] is True and ok["tables"] == ["singer"] and ok["normalized_sql"]

    bad = call(server, "validate_sql", {"sql": "DROP TABLE singer"}).structured_content
    assert bad["is_valid"] is False and bad["violations"][0]["code"] == "non_select_statement"

    column = call(
        server, "validate_sql", {"sql": "SELECT salary FROM singer", "db_id": "music"}
    ).structured_content
    assert {v["code"] for v in column["violations"]} == {"unknown_column"}
    assert call(
        server, "validate_sql", {"sql": "SELECT salary FROM singer"}
    ).structured_content["is_valid"] is True  # no db_id: structural checks only

    assert call_error(server, "validate_sql", {"sql": "SELECT 1", "db_id": "nope"}) is not None


# ---------------------------------------------------------------------- resources
def test_resources(catalog: DatabaseCatalog) -> None:
    server = _server(catalog)
    schema = run_client(server, lambda c: c.read_resource("datamind://schema/music"))
    assert "CREATE TABLE singer" in schema[0].text
    databases = run_client(server, lambda c: c.read_resource("datamind://databases"))
    assert json.loads(databases[0].text) == ["music"]


# ------------------------------------------------------------------- server config
def test_server_config_validation() -> None:
    with pytest.raises(ValueError):
        ServerConfig(max_concurrent=0)
    with pytest.raises(ValueError):
        ServerConfig(request_timeout_s=0)


# ------------------------------------------------------------------------- factory
def _close(closers: Sequence[Any]) -> None:
    async def go() -> None:
        for closer in closers:
            await closer.aclose()

    asyncio.run(go())


def test_factory_requires_api_key_for_claude(
    monkeypatch: pytest.MonkeyPatch, catalog: DatabaseCatalog
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        build_answerer(BackendSettings(backend="claude"), catalog)


def test_factory_requires_explicit_gpu_rate(
    monkeypatch: pytest.MonkeyPatch, catalog: DatabaseCatalog
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    for backend in ("local", "routed"):
        with pytest.raises(ConfigError, match="gpu_hourly_usd"):
            build_answerer(BackendSettings(backend=backend), catalog)


@pytest.mark.parametrize(
    "settings",
    [
        BackendSettings(backend="nope"),
        BackendSettings(backend="claude", claude_input_price=1.0),  # prices come in pairs
        BackendSettings(backend="claude", claude_model="no-such-model"),  # no pricing known
        BackendSettings(backend="routed", gpu_hourly_usd=0.0, min_confidence=0.5),  # logprob must be <= 0
    ],
)
def test_factory_rejects_bad_settings(
    monkeypatch: pytest.MonkeyPatch, catalog: DatabaseCatalog, settings: BackendSettings
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with pytest.raises(ConfigError):
        build_answerer(settings, catalog)


def test_factory_builds_each_backend(monkeypatch: pytest.MonkeyPatch, catalog: DatabaseCatalog) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    claude, closers = build_answerer(BackendSettings(backend="claude"), catalog)
    assert isinstance(claude, DataMindAgent) and len(closers) == 1
    _close(closers)

    local, closers = build_answerer(BackendSettings(backend="local", gpu_hourly_usd=0.0), catalog)
    assert isinstance(local, DataMindAgent) and len(closers) == 1
    _close(closers)

    routed, closers = build_answerer(
        BackendSettings(backend="routed", gpu_hourly_usd=0.5, min_confidence=-0.15), catalog
    )
    assert isinstance(routed, CascadeAgent) and len(closers) == 2
    _close(closers)

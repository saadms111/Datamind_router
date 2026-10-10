"""Agent tests with a scripted LLM (no network)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import DatabaseCatalog, UnknownDatabaseError
from datamind.llm import LLMError, LLMResponse, Message
from datamind.schema import QueryRequest, QueryStatus, Route


class ScriptedLLM:
    def __init__(self, replies: Sequence[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        self.calls.append(list(messages))
        item = self._replies.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMResponse(
            text=item,
            prompt_tokens=100,
            completion_tokens=20,
            cost_usd=0.001,
            latency_ms=5.0,
            model="fake-model",
            route=Route.LOCAL,
        )


def sql_block(sql: str) -> str:
    return f"```sql\n{sql}\n```"


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


def ask(agent: DataMindAgent, question: str = "Who are the youngest singers?"):
    return asyncio.run(agent.answer(QueryRequest(question=question, db_id="music")))


def test_success_first_try(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("SELECT name FROM singer ORDER BY age LIMIT 2")])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.SUCCESS
    assert response.columns == ["name"]
    assert response.rows == [["b"], ["a"]]
    assert response.row_count == 2
    assert response.validation is not None and response.validation.is_valid
    assert response.sql == response.validation.normalized_sql
    m = response.metrics
    assert (m.prompt_tokens, m.completion_tokens, m.total_tokens) == (100, 20, 120)
    assert m.llm_calls == 1 and m.route is Route.LOCAL and m.model == "fake-model"
    assert m.cost_usd == pytest.approx(0.001) and m.latency_ms > 0
    response.model_dump_json()  # must be serialisable


def test_prompt_contains_schema_and_question(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("SELECT 1 FROM singer")])
    ask(DataMindAgent(llm, catalog), "How many singers?")
    prompt = llm.calls[0][0]["content"]
    assert "CREATE TABLE singer" in prompt and "How many singers?" in prompt


def test_retry_after_guard_rejection(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("DROP TABLE singer"), sql_block("SELECT count(*) FROM singer")])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.SUCCESS
    assert response.rows == [[3]]
    assert response.metrics.llm_calls == 2
    assert response.metrics.total_tokens == 240
    assert response.metrics.cost_usd == pytest.approx(0.002)
    second_call = llm.calls[1]
    assert [m["role"] for m in second_call] == ["user", "assistant", "user"]
    assert "rejected" in second_call[2]["content"]


def test_retry_after_execution_error(catalog: DatabaseCatalog) -> None:
    bad = "SELECT name FROM singer UNION SELECT name, age FROM singer"  # column-count mismatch
    llm = ScriptedLLM([sql_block(bad), sql_block("SELECT name FROM singer")])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.SUCCESS
    assert response.metrics.llm_calls == 2
    assert "SQLite returned an error" in llm.calls[1][2]["content"]


def test_exhausted_attempts_report_last_failure(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("DROP TABLE singer"), sql_block("DELETE FROM singer")])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.REJECTED
    assert response.error and "non_select_statement" in response.error
    assert response.sql == "DELETE FROM singer"
    assert response.rows == [] and response.metrics.llm_calls == 2


def test_unknown_column_is_rejected_by_schema_check(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("SELECT salary FROM singer")])
    response = ask(DataMindAgent(llm, catalog, AgentConfig(max_attempts=1)))
    assert response.status is QueryStatus.REJECTED
    assert response.error and "unknown_column" in response.error
    assert response.metrics.llm_calls == 1


def test_no_sql_in_reply(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM(["I cannot help with that.", "Still no."])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.GENERATION_ERROR
    assert response.sql is None and response.validation is None


def test_llm_error_is_not_retried(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([LLMError("boom"), sql_block("SELECT 1 FROM singer")])
    response = ask(DataMindAgent(llm, catalog))
    assert response.status is QueryStatus.GENERATION_ERROR
    assert response.error == "boom"
    assert response.metrics.llm_calls == 0 and len(llm.calls) == 1


def test_max_attempts_one_does_not_retry(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("DROP TABLE singer"), sql_block("SELECT 1 FROM singer")])
    response = ask(DataMindAgent(llm, catalog, AgentConfig(max_attempts=1)))
    assert response.status is QueryStatus.REJECTED
    assert len(llm.calls) == 1


def test_row_limit_is_applied(catalog: DatabaseCatalog) -> None:
    llm = ScriptedLLM([sql_block("SELECT name FROM singer")])
    request = QueryRequest(question="all", db_id="music", max_rows=2)
    response = asyncio.run(DataMindAgent(llm, catalog).answer(request))
    assert response.status is QueryStatus.SUCCESS
    assert response.row_count == 2 and response.truncated


def test_unknown_database_raises(catalog: DatabaseCatalog) -> None:
    agent = DataMindAgent(ScriptedLLM([]), catalog)
    with pytest.raises(UnknownDatabaseError):
        asyncio.run(agent.answer(QueryRequest(question="q", db_id="nope")))

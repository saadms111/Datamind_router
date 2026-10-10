"""Tests for the evaluation runner and report (scripted LLM, no network)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path

import pytest

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import DatabaseCatalog
from datamind.llm import LLMResponse, Message
from datamind.schema import Route
from eval.loader import DEFAULT_DATA_DIR, SpiderDataError, SpiderExample, load_spider_sample
from eval.report import percentile, render_markdown, summarize, write_report
from eval.runner import OracleLLM, run_eval
from eval.scoring import FailureCategory


class ScriptedLLM:
    """Replies in order; each reply costs $0.01 and 100ms."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        return LLMResponse(
            text=f"```sql\n{self._replies.pop(0)}\n```",
            prompt_tokens=100,
            completion_tokens=10,
            cost_usd=0.01,
            latency_ms=100.0,
            model="fake",
            route=Route.CLAUDE,
        )


DDL = "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"


@pytest.fixture
def env(tmp_path: Path) -> tuple[DatabaseCatalog, Path]:
    db_dir = tmp_path / "databases" / "music"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "music.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute(DDL.rstrip(";"))
    conn.executemany("INSERT INTO singer (name, age) VALUES (?, ?)", [("a", 30), ("b", 20), ("c", 40)])
    conn.commit()
    conn.close()
    return DatabaseCatalog.from_directory(tmp_path / "databases"), db_path


def make_example(i: int, db_path: Path, gold: str, question: str = "q") -> SpiderExample:
    return SpiderExample(
        example_id=f"ex-{i}",
        db_id="music",
        question=f"{question} {i}",
        gold_sql=gold,
        schema_ddl=DDL,
        db_path=db_path,
    )


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)
    assert percentile([7], 95) == 7
    assert percentile([], 95) == 0.0
    with pytest.raises(ValueError):
        percentile([1], 101)


def test_run_scores_and_categorises(env: tuple[DatabaseCatalog, Path], tmp_path: Path) -> None:
    catalog, db_path = env
    examples = [
        make_example(0, db_path, "SELECT name FROM singer WHERE age > 25"),  # correct
        make_example(1, db_path, "SELECT name FROM singer ORDER BY age"),  # wrong ordering
        make_example(2, db_path, "SELECT name FROM singer"),  # wrong row count
        make_example(3, db_path, "SELECT name FROM singer"),  # guard rejected (x2 attempts)
        make_example(4, db_path, "SELECT name, age FROM singer"),  # wrong column count
    ]
    replies = [
        "SELECT name FROM singer WHERE age > 25",
        "SELECT name FROM singer ORDER BY age DESC",
        "SELECT name FROM singer WHERE age > 25",
        "DROP TABLE singer",
        "DELETE FROM singer",
        "SELECT name FROM singer",
    ]
    agent = DataMindAgent(ScriptedLLM(replies), catalog, AgentConfig(max_attempts=2))
    seen: list[int] = []
    results = asyncio.run(run_eval(agent, examples, progress=lambda d, t, r: seen.append(d)))

    assert seen == [1, 2, 3, 4, 5]
    assert [r.correct for r in results] == [True, False, False, False, False]
    assert [r.failure_category for r in results] == [
        None,
        FailureCategory.WRONG_ORDERING,
        FailureCategory.WRONG_ROW_COUNT,
        FailureCategory.GUARD_REJECTED,
        FailureCategory.WRONG_COLUMN_COUNT,
    ]
    assert results[3].llm_calls == 2 and results[3].error

    summary = summarize(results, run_name="t", created_at="2026-01-01T00:00:00+00:00")
    assert summary.n_examples == 5 and summary.n_correct == 1
    assert summary.execution_accuracy == pytest.approx(0.2)
    assert summary.total_prompt_tokens == 600 and summary.total_completion_tokens == 60
    assert summary.total_cost_usd == pytest.approx(0.06)
    assert summary.cost_per_1k_queries_usd == pytest.approx(12.0)
    assert summary.avg_llm_calls == pytest.approx(6 / 5)
    assert summary.failure_counts[FailureCategory.GUARD_REJECTED.value] == 1
    assert summary.accuracy_by_route["claude"].n == 5
    assert summary.latency_ms.p50 > 0

    out = tmp_path / "out"
    write_report(out, summary, results)
    lines = (out / "results.jsonl").read_text().splitlines()
    assert len(lines) == 5 and json.loads(lines[0])["example_id"] == "ex-0"
    assert json.loads((out / "summary.json").read_text())["n_correct"] == 1
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "Execution accuracy" in report and "guard_rejected" in report
    assert render_markdown(summary, results) == report


def test_unknown_database_becomes_harness_error(env: tuple[DatabaseCatalog, Path]) -> None:
    catalog, db_path = env
    ex = make_example(0, db_path, "SELECT name FROM singer")
    bad = ex.model_copy(update={"db_id": "missing"})
    agent = DataMindAgent(ScriptedLLM([]), catalog)
    (result,) = asyncio.run(run_eval(agent, [bad]))
    assert not result.correct
    assert result.failure_category is FailureCategory.HARNESS_ERROR
    assert result.status is None


def test_concurrency_preserves_order(env: tuple[DatabaseCatalog, Path]) -> None:
    catalog, db_path = env
    examples = [make_example(i, db_path, "SELECT name FROM singer") for i in range(6)]
    agent = DataMindAgent(OracleLLM(examples), catalog)
    results = asyncio.run(run_eval(agent, examples, concurrency=3))
    assert [r.example_id for r in results] == [e.example_id for e in examples]
    assert all(r.correct for r in results)
    with pytest.raises(ValueError):
        asyncio.run(run_eval(agent, examples, concurrency=0))


def test_oracle_scores_100_percent_on_real_sample() -> None:
    try:
        examples = load_spider_sample(DEFAULT_DATA_DIR, ensure=False)
    except SpiderDataError as exc:
        pytest.skip(f"Spider sample not built: {exc}")
    catalog = DatabaseCatalog.from_directory(DEFAULT_DATA_DIR / "databases")
    agent = DataMindAgent(OracleLLM(examples), catalog, AgentConfig(max_attempts=1))
    results = asyncio.run(run_eval(agent, examples, concurrency=4))
    failures = [(r.example_id, r.failure_category, r.error) for r in results if not r.correct]
    assert not failures, failures

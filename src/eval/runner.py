"""Run the DataMind agent over Spider examples and score each result."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from datamind.agent import DataMindAgent
from datamind.catalog import CatalogError, UnknownDatabaseError
from datamind.executor import ExecutionError, SafeExecutor
from datamind.llm import LLMResponse, Message
from datamind.prompts import build_user_prompt
from datamind.schema import QueryRequest, QueryStatus, Route
from eval.loader import SpiderExample
from eval.scoring import (
    FailureCategory,
    classify_status,
    compare_results,
    complexity_bucket,
    feature_mismatches,
    gold_has_order_by,
    mismatch_category,
)

__all__ = ["ExampleResult", "OracleLLM", "evaluate_example", "run_eval"]

logger = logging.getLogger(__name__)

_RESULT_ROW_CAP = 1000
_GOLD_TIMEOUT_S = 10.0


class ExampleResult(BaseModel):
    """Flat, JSON-friendly record of one scored example."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str | None = None
    status: QueryStatus | None = None  # None only for harness errors
    correct: bool = False
    failure_category: FailureCategory | None = None
    feature_mismatches: list[str] = Field(default_factory=list)
    complexity: str = "unknown"
    error: str | None = None
    result_truncated: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    llm_calls: int = 0
    model: str | None = None
    route: str | None = None


class OracleLLM:
    """Test double that answers every prompt with the gold SQL.

    Running the full harness with it must score 100%; any shortfall is a bug in
    the harness, guard or executor rather than in a model.
    """

    route = Route.LOCAL

    def __init__(self, examples: Sequence[SpiderExample]) -> None:
        self._by_prompt = {build_user_prompt(e.question, e.schema_ddl): e.gold_sql for e in examples}

    async def generate(self, *, system: str, messages: Sequence[Message]) -> LLMResponse:
        sql = self._by_prompt.get(messages[0]["content"])
        if sql is None:
            raise KeyError("OracleLLM received a prompt it was not built for")
        return LLMResponse(
            text=f"```sql\n{sql}\n```",
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
            latency_ms=0.0,
            model="oracle",
            route=self.route,
        )


def _harness_error(example: SpiderExample, message: str) -> ExampleResult:
    return ExampleResult(
        example_id=example.example_id,
        db_id=example.db_id,
        question=example.question,
        gold_sql=example.gold_sql,
        failure_category=FailureCategory.HARNESS_ERROR,
        complexity=complexity_bucket(example.gold_sql),
        error=message,
    )


async def evaluate_example(agent: DataMindAgent, example: SpiderExample) -> ExampleResult:
    """Run one example through the agent and score it against the gold query."""
    gold_executor = SafeExecutor(example.db_path, max_rows=_RESULT_ROW_CAP, timeout_s=_GOLD_TIMEOUT_S)
    try:
        gold = await asyncio.to_thread(gold_executor.execute, example.gold_sql)
    except ExecutionError as exc:
        return _harness_error(example, f"gold SQL failed: {exc}")

    request = QueryRequest(
        request_id=example.example_id,
        question=example.question,
        db_id=example.db_id,
        max_rows=_RESULT_ROW_CAP,
    )
    try:
        response = await agent.answer(request)
    except (UnknownDatabaseError, CatalogError) as exc:
        return _harness_error(example, str(exc))

    correct = False
    category = classify_status(response.status)
    mismatches: list[str] = []
    truncated = False
    if response.status is QueryStatus.SUCCESS:
        truncated = response.truncated or gold.truncated
        match = compare_results(
            response.rows, gold.rows, ordered=gold_has_order_by(example.gold_sql)
        )
        correct = match.matched
        if not correct and match.kind is not None:
            category = mismatch_category(match.kind)
    if not correct and response.sql:
        mismatches = feature_mismatches(response.sql, example.gold_sql)

    m = response.metrics
    return ExampleResult(
        example_id=example.example_id,
        db_id=example.db_id,
        question=example.question,
        gold_sql=example.gold_sql,
        predicted_sql=response.sql,
        status=response.status,
        correct=correct,
        failure_category=None if correct else category,
        feature_mismatches=mismatches,
        complexity=complexity_bucket(example.gold_sql),
        error=response.error,
        result_truncated=truncated,
        prompt_tokens=m.prompt_tokens,
        completion_tokens=m.completion_tokens,
        cost_usd=m.cost_usd,
        latency_ms=m.latency_ms,
        llm_calls=m.llm_calls,
        model=m.model,
        route=m.route.value if m.route else None,
    )


async def run_eval(
    agent: DataMindAgent,
    examples: Sequence[SpiderExample],
    *,
    concurrency: int = 1,
    progress: Callable[[int, int, ExampleResult], None] | None = None,
) -> list[ExampleResult]:
    """Evaluate all examples, returning results in input order.

    ``concurrency=1`` (default) gives clean per-query latency numbers; higher
    values finish sooner but latencies then include queueing and rate-limit effects.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    semaphore = asyncio.Semaphore(concurrency)
    done = 0
    total = len(examples)

    async def one(example: SpiderExample) -> ExampleResult:
        nonlocal done
        async with semaphore:
            try:
                result = await evaluate_example(agent, example)
            except Exception as exc:  # noqa: BLE001 - one bad example must not abort a long run
                logger.exception("Unexpected failure on %s", example.example_id)
                result = _harness_error(example, f"{type(exc).__name__}: {exc}")
        done += 1
        if progress is not None:
            progress(done, total, result)
        return result

    return list(await asyncio.gather(*(one(e) for e in examples)))

"""The DataMind Text-to-SQL agent.

Pipeline per request::

    prompt -> LLM -> extract SQL -> AST guard -> safe execution -> response
                ^                        |               |
                +---- feedback + retry --+---------------+

* Guard rejections and SQLite errors are fed back to the model for another
  attempt (up to ``AgentConfig.max_attempts`` total LLM calls).
* Timeouts and LLM transport failures are *not* retried.
* The SQL that actually executes is the guard's regenerated ``normalized_sql``.
* Token usage, cost and latency are accumulated across all attempts into
  ``QueryMetrics``; ``latency_ms`` is end-to-end wall time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from datamind.catalog import DatabaseCatalog, DatabaseInfo
from datamind.executor import ExecutionError, ExecutionResult, ExecutionTimeout, SafeExecutor
from datamind.guardrails import GuardrailConfig, SQLGuard
from datamind.llm import LLMClient, LLMError, LLMResponse, Message
from datamind.prompts import SYSTEM_PROMPT, build_feedback_prompt, build_user_prompt, extract_sql
from datamind.schema import (
    ASTValidationResult,
    QueryMetrics,
    QueryRequest,
    QueryResponse,
    QueryStatus,
)

__all__ = ["AgentConfig", "DataMindAgent"]

logger = logging.getLogger(__name__)

_MAX_ERROR_CHARS = 500
_MAX_VIOLATIONS_IN_ERROR = 5


@dataclass(frozen=True, slots=True)
class AgentConfig:
    max_attempts: int = 2
    guardrails: GuardrailConfig = field(default_factory=GuardrailConfig)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")


def _describe_violations(validation: ASTValidationResult) -> str:
    parts = [f"{v.code.value}: {v.message}" for v in validation.violations[:_MAX_VIOLATIONS_IN_ERROR]]
    return "; ".join(parts)[:_MAX_ERROR_CHARS]


def _call_metrics(reply: LLMResponse) -> QueryMetrics:
    return QueryMetrics(
        prompt_tokens=reply.prompt_tokens,
        completion_tokens=reply.completion_tokens,
        cost_usd=reply.cost_usd,
        latency_ms=reply.latency_ms,
        llm_calls=1,
        model=reply.model,
        route=reply.route,
        confidence=reply.avg_logprob,
    )


class DataMindAgent:
    def __init__(
        self,
        llm: LLMClient,
        catalog: DatabaseCatalog,
        config: AgentConfig | None = None,
    ) -> None:
        self._llm = llm
        self._catalog = catalog
        self._config = config or AgentConfig()
        self._guards: dict[str, SQLGuard] = {}

    def _guard_for(self, info: DatabaseInfo) -> SQLGuard:
        guard = self._guards.get(info.db_id)
        if guard is None:
            guard = SQLGuard(info.schema, self._config.guardrails)
            self._guards[info.db_id] = guard
        return guard

    async def answer(self, request: QueryRequest) -> QueryResponse:
        """Answer one question.

        Query-level failures (rejected SQL, SQLite errors, timeouts, LLM errors)
        are reported in the returned ``QueryResponse``; they do not raise.

        Raises:
            UnknownDatabaseError: ``request.db_id`` is not in the catalog.
            CatalogError: the database could not be read.
        """
        info = self._catalog.get(request.db_id)
        guard = self._guard_for(info)
        executor = SafeExecutor(info.path, max_rows=request.max_rows, timeout_s=request.timeout_s)

        started = time.perf_counter()
        messages: list[Message] = [
            {"role": "user", "content": build_user_prompt(request.question, info.schema_ddl)}
        ]
        metrics = QueryMetrics()
        status = QueryStatus.GENERATION_ERROR
        error: str | None = "No generation attempt was made"
        sql: str | None = None
        validation: ASTValidationResult | None = None
        result: ExecutionResult | None = None

        for attempt in range(1, self._config.max_attempts + 1):
            try:
                reply = await self._llm.generate(system=SYSTEM_PROMPT, messages=messages)
            except LLMError as exc:
                status, error = QueryStatus.GENERATION_ERROR, str(exc) or "LLM call failed"
                break
            metrics = metrics.merge(_call_metrics(reply))

            candidate = extract_sql(reply.text)
            if candidate is None:
                status, error = QueryStatus.GENERATION_ERROR, "Model reply contained no SQL query"
                sql, validation = None, None
                feedback = build_feedback_prompt("your reply did not contain a SQL query")
            else:
                sql = candidate
                validation = guard.validate(candidate)
                if validation.is_valid:
                    try:
                        result = await asyncio.to_thread(
                            executor.execute, validation.normalized_sql or ""
                        )
                    except ExecutionTimeout as exc:
                        status, error = QueryStatus.TIMEOUT, str(exc)
                        break
                    except ExecutionError as exc:
                        status = QueryStatus.EXECUTION_ERROR
                        error = (str(exc) or "SQLite execution failed")[:_MAX_ERROR_CHARS]
                        feedback = build_feedback_prompt(f"SQLite returned an error: {error}")
                    else:
                        status, error = QueryStatus.SUCCESS, None
                        break
                else:
                    status = QueryStatus.REJECTED
                    error = _describe_violations(validation)
                    feedback = build_feedback_prompt(f"it was rejected by the validator ({error})")

            if attempt < self._config.max_attempts:
                messages.append({"role": "assistant", "content": reply.text})
                messages.append({"role": "user", "content": feedback})

        wall_ms = (time.perf_counter() - started) * 1000.0
        final_metrics = metrics.model_copy(update={"latency_ms": wall_ms})

        if status is QueryStatus.SUCCESS and result is not None and validation is not None:
            response = QueryResponse(
                request_id=request.request_id,
                question=request.question,
                db_id=request.db_id,
                status=status,
                sql=validation.normalized_sql,
                columns=result.columns,
                rows=result.rows,
                truncated=result.truncated,
                validation=validation,
                metrics=final_metrics,
            )
        else:
            response = QueryResponse(
                request_id=request.request_id,
                question=request.question,
                db_id=request.db_id,
                status=status,
                sql=sql,
                validation=validation,
                metrics=final_metrics,
                error=error or "Unknown failure",
            )

        logger.info(
            "query request_id=%s db=%s status=%s llm_calls=%d tokens=%d cost_usd=%.6f latency_ms=%.0f",
            response.request_id,
            response.db_id,
            response.status.value,
            final_metrics.llm_calls,
            final_metrics.total_tokens,
            final_metrics.cost_usd,
            wall_ms,
        )
        return response

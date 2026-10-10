"""Pydantic models shared by the DataMind agent, router, benchmarks and MCP server.

All models are immutable (``frozen=True``) and reject unknown fields so that
malformed payloads fail loudly at the API boundary instead of propagating.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    model_validator,
)

__all__ = [
    "ASTValidationResult",
    "ASTViolation",
    "DbId",
    "QueryMetrics",
    "QueryRequest",
    "QueryResponse",
    "QueryStatus",
    "Route",
    "RoutePreference",
    "SQLValue",
    "ViolationCode",
]

# Spider-style database identifiers: safe to embed in file paths.
DbId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,128}$")]

# JSON-safe scalar types a result cell may contain. Producers must convert
# ``bytes`` (e.g. SQLite BLOBs) to ``str`` before building a ``QueryResponse``.
SQLValue = str | int | float | bool | None


class _StrictModel(BaseModel):
    """Base class: immutable, no unknown fields, enum values serialised as values."""

    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=False)


class Route(StrEnum):
    """Model that actually served a generation."""

    LOCAL = "local"
    CLAUDE = "claude"


class RoutePreference(StrEnum):
    """Caller-controlled routing hint."""

    AUTO = "auto"
    LOCAL = "local"
    CLAUDE = "claude"


class QueryStatus(StrEnum):
    """Terminal state of a query."""

    SUCCESS = "success"
    REJECTED = "rejected"  # blocked by AST guardrails
    GENERATION_ERROR = "generation_error"  # LLM call failed or returned no SQL
    EXECUTION_ERROR = "execution_error"  # SQL was valid but failed at runtime
    TIMEOUT = "timeout"


class ViolationCode(StrEnum):
    """Machine-readable reason an AST guardrail fired."""

    PARSE_ERROR = "parse_error"
    EMPTY_STATEMENT = "empty_statement"
    MULTIPLE_STATEMENTS = "multiple_statements"
    NON_SELECT_STATEMENT = "non_select_statement"
    FORBIDDEN_NODE = "forbidden_node"
    FORBIDDEN_FUNCTION = "forbidden_function"
    UNKNOWN_TABLE = "unknown_table"
    UNKNOWN_COLUMN = "unknown_column"
    TOO_COMPLEX = "too_complex"


class QueryRequest(_StrictModel):
    """A natural-language question to answer against a specific database."""

    request_id: str = Field(
        default_factory=lambda: uuid4().hex,
        min_length=1,
        max_length=64,
        description="Caller-supplied or auto-generated correlation id.",
    )
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ] = Field(description="Natural-language question.")
    db_id: DbId = Field(description="Identifier of the target database.")
    route_preference: RoutePreference = Field(
        default=RoutePreference.AUTO,
        description="Force a backend, or let the router decide.",
    )
    max_rows: int = Field(
        default=100, ge=1, le=1000, description="Maximum rows returned to the caller."
    )
    timeout_s: float = Field(
        default=10.0, gt=0.0, le=60.0, description="SQL execution timeout in seconds."
    )


class QueryMetrics(_StrictModel):
    """Token, cost and latency accounting for one query (possibly several LLM calls)."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0, description="Total LLM cost in USD.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="End-to-end wall time.")
    llm_calls: int = Field(default=0, ge=0)
    model: str | None = Field(default=None, description="Model that produced the final SQL.")
    route: Route | None = None
    confidence: float | None = Field(
        default=None,
        description="Mean token log-probability of the final generation, when the backend "
        "reports it (closer to 0 is more confident). None for backends without logprobs.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: QueryMetrics) -> QueryMetrics:
        """Combine metrics from sequential steps (e.g. a retry after a failed attempt).

        Counters and cost are summed; ``other`` wins for ``model``/``route`` since it
        represents the later (final) step.
        """
        return QueryMetrics(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            latency_ms=self.latency_ms + other.latency_ms,
            llm_calls=self.llm_calls + other.llm_calls,
            model=other.model or self.model,
            route=other.route or self.route,
            confidence=other.confidence if other.confidence is not None else self.confidence,
        )


class ASTViolation(_StrictModel):
    """A single guardrail violation found while walking the SQL AST."""

    code: ViolationCode
    message: str = Field(min_length=1)
    node_type: str | None = Field(default=None, description="sqlglot expression class name.")
    fragment: str | None = Field(
        default=None, max_length=500, description="Offending SQL snippet, truncated."
    )


class ASTValidationResult(_StrictModel):
    """Outcome of validating generated SQL against the guardrail policy."""

    is_valid: bool
    dialect: str = "sqlite"
    statement_type: str | None = Field(
        default=None, description="Root sqlglot expression class, e.g. 'Select'."
    )
    tables: list[str] = Field(default_factory=list, description="Referenced tables, lower-cased.")
    violations: list[ASTViolation] = Field(default_factory=list)
    normalized_sql: str | None = Field(
        default=None, description="Canonical SQL to execute; set only when valid."
    )

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.is_valid:
            if self.violations:
                raise ValueError("is_valid=True cannot have violations")
            if not self.normalized_sql:
                raise ValueError("is_valid=True requires normalized_sql")
        else:
            if not self.violations:
                raise ValueError("is_valid=False requires at least one violation")
            if self.normalized_sql is not None:
                raise ValueError("normalized_sql must be None when is_valid=False")
        return self

    @classmethod
    def ok(
        cls,
        normalized_sql: str,
        *,
        statement_type: str = "Select",
        tables: list[str] | None = None,
        dialect: str = "sqlite",
    ) -> Self:
        return cls(
            is_valid=True,
            dialect=dialect,
            statement_type=statement_type,
            tables=tables or [],
            normalized_sql=normalized_sql,
        )

    @classmethod
    def rejected(
        cls,
        violations: list[ASTViolation],
        *,
        statement_type: str | None = None,
        tables: list[str] | None = None,
        dialect: str = "sqlite",
    ) -> Self:
        return cls(
            is_valid=False,
            dialect=dialect,
            statement_type=statement_type,
            tables=tables or [],
            violations=violations,
        )


class QueryResponse(_StrictModel):
    """Full result of answering a :class:`QueryRequest`."""

    request_id: str
    question: str
    db_id: DbId
    status: QueryStatus
    sql: str | None = Field(default=None, description="SQL that was (or would have been) run.")
    columns: list[str] = Field(default_factory=list)
    rows: list[list[SQLValue]] = Field(default_factory=list)
    truncated: bool = Field(
        default=False, description="True if more rows existed than max_rows allowed."
    )
    validation: ASTValidationResult | None = None
    metrics: QueryMetrics = Field(default_factory=QueryMetrics)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def row_count(self) -> int:
        return len(self.rows)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.status is QueryStatus.SUCCESS:
            if not self.sql:
                raise ValueError("status=success requires sql")
            if self.error is not None:
                raise ValueError("status=success must not carry an error")
        else:
            if not self.error:
                raise ValueError(f"status={self.status.value} requires an error message")
            if self.rows:
                raise ValueError("rows must be empty for non-success responses")
        if self.rows and self.columns:
            width = len(self.columns)
            if any(len(row) != width for row in self.rows):
                raise ValueError("every row must match the number of columns")
        return self

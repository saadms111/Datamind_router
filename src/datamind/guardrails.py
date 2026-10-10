"""AST-level SQL guardrails built on sqlglot.

``SQLGuard.validate`` parses model-generated SQL and only accepts a single,
read-only SELECT-style statement. It never executes anything. Checks:

* exactly one statement, whose root is SELECT / UNION / INTERSECT / EXCEPT
* no DDL, DML, PRAGMA, transaction or other command nodes anywhere in the tree
* no dangerous SQLite functions (``load_extension``, ``readfile`` ...) or
  ``pragma_*`` table-valued functions
* no schema-qualified tables (``main.t``, ``temp.t``, attached databases) and no
  ``sqlite_*`` system tables
* no ``WITH RECURSIVE`` unless explicitly enabled
* complexity limits (SQL length, AST node count, join count)
* optionally, when a schema is supplied: every table and column must exist

On success ``normalized_sql`` is regenerated *from the validated AST*. Callers
should execute that string, not the original text, so that what was validated is
exactly what runs (no parser-differential gap).

The column check is deliberately lenient (query-wide scope rather than strict
per-SELECT scope) to avoid rejecting valid queries. It follows SQLite's own
fallback: an unresolved double-quoted identifier is treated as a string literal.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from datamind.schema import ASTValidationResult, ASTViolation, ViolationCode

__all__ = ["DEFAULT_FORBIDDEN_FUNCTIONS", "GuardrailConfig", "SQLGuard"]

logger = logging.getLogger(__name__)

DEFAULT_FORBIDDEN_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {
        "load_extension",
        "readfile",
        "writefile",
        "edit",
        "fts3_tokenizer",
        "zeroblob",  # memory-exhaustion vectors
        "randomblob",
    }
)

_FORBIDDEN_NODE_NAMES: Final[tuple[str, ...]] = (
    "Insert",
    "Update",
    "Delete",
    "Drop",
    "Create",
    "Alter",
    "AlterTable",
    "Command",
    "Pragma",
    "Merge",
    "Set",
    "Use",
    "Transaction",
    "Commit",
    "Rollback",
    "TruncateTable",
    "Into",
    "Attach",
    "Detach",
    "Copy",
    "LoadData",
)
# Resolved by name so the module tolerates sqlglot versions that rename/remove nodes.
_FORBIDDEN_NODES: Final[tuple[type[exp.Expression], ...]] = tuple(
    cls for name in _FORBIDDEN_NODE_NAMES if isinstance(cls := getattr(exp, name, None), type)
)
_SELECT_ROOTS: Final = (exp.Select, exp.Union, exp.Intersect, exp.Except)

_MAX_VIOLATIONS: Final = 20
_FRAGMENT_CHARS: Final = 200
_ANSI_RE: Final = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True, slots=True)
class GuardrailConfig:
    """Tunable policy limits for :class:`SQLGuard`."""

    dialect: str = "sqlite"
    max_sql_chars: int = 10_000
    max_nodes: int = 2_000
    max_joins: int = 10
    allow_recursive_cte: bool = False
    forbidden_functions: frozenset[str] = DEFAULT_FORBIDDEN_FUNCTIONS

    def __post_init__(self) -> None:
        if self.max_sql_chars < 1 or self.max_nodes < 1 or self.max_joins < 0:
            raise ValueError("GuardrailConfig limits must be positive")


def _clean_error(exc: Exception) -> str:
    """First line of a sqlglot error, without ANSI highlighting, bounded in length."""
    text = _ANSI_RE.sub("", str(exc)).strip()
    first = text.splitlines()[0] if text else type(exc).__name__
    return first[:300]


def _fragment(node: exp.Expression, dialect: str) -> str:
    try:
        return node.sql(dialect=dialect)[:_FRAGMENT_CHARS]
    except SqlglotError:
        return type(node).__name__


def _add(
    violations: list[ASTViolation],
    code: ViolationCode,
    message: str,
    *,
    node: exp.Expression | None = None,
    dialect: str = "sqlite",
) -> None:
    if len(violations) >= _MAX_VIOLATIONS:
        return
    violations.append(
        ASTViolation(
            code=code,
            message=message,
            node_type=type(node).__name__ if node is not None else None,
            fragment=_fragment(node, dialect) if node is not None else None,
        )
    )


def _single(code: ViolationCode, message: str) -> list[ASTViolation]:
    return [ASTViolation(code=code, message=message)]


class SQLGuard:
    """Validates SQL against the read-only policy; see module docstring."""

    def __init__(
        self,
        schema: Mapping[str, Collection[str]] | None = None,
        config: GuardrailConfig | None = None,
    ) -> None:
        """
        Args:
            schema: optional ``{table: [columns]}`` mapping (case-insensitive). When
                given, unknown tables and columns are rejected.
            config: policy limits; defaults to :class:`GuardrailConfig`.
        """
        self._config = config or GuardrailConfig()
        self._schema: dict[str, frozenset[str]] | None = (
            None
            if schema is None
            else {t.lower(): frozenset(c.lower() for c in cols) for t, cols in schema.items()}
        )

    def validate(self, sql: str) -> ASTValidationResult:
        cfg = self._config
        text = sql.strip()
        if not text:
            return ASTValidationResult.rejected(
                _single(ViolationCode.EMPTY_STATEMENT, "SQL is empty"), dialect=cfg.dialect
            )
        if len(text) > cfg.max_sql_chars:
            return ASTValidationResult.rejected(
                _single(
                    ViolationCode.TOO_COMPLEX,
                    f"SQL is {len(text)} characters; limit is {cfg.max_sql_chars}",
                ),
                dialect=cfg.dialect,
            )

        try:
            statements = [s for s in sqlglot.parse(text, read=cfg.dialect) if s is not None]
        except SqlglotError as exc:
            return ASTValidationResult.rejected(
                _single(ViolationCode.PARSE_ERROR, _clean_error(exc)), dialect=cfg.dialect
            )

        if not statements:
            return ASTValidationResult.rejected(
                _single(ViolationCode.EMPTY_STATEMENT, "No SQL statement found"),
                dialect=cfg.dialect,
            )
        if len(statements) > 1:
            return ASTValidationResult.rejected(
                _single(
                    ViolationCode.MULTIPLE_STATEMENTS,
                    f"Expected one statement, found {len(statements)}",
                ),
                statement_type=type(statements[0]).__name__,
                dialect=cfg.dialect,
            )

        tree = statements[0]
        root = tree
        while isinstance(root, exp.Subquery):
            root = root.this
        statement_type = type(root).__name__
        if not isinstance(root, _SELECT_ROOTS):
            return ASTValidationResult.rejected(
                _single(
                    ViolationCode.NON_SELECT_STATEMENT,
                    f"Only SELECT queries are allowed; got {statement_type}",
                ),
                statement_type=statement_type,
                dialect=cfg.dialect,
            )

        violations: list[ASTViolation] = []
        tables = self._walk(tree, violations)

        if self._schema is not None:
            unknown = sorted(t for t in tables if t not in self._schema)
            for table in unknown:
                _add(violations, ViolationCode.UNKNOWN_TABLE, f"Unknown table '{table}'",
                     dialect=cfg.dialect)
            if not unknown:
                self._check_columns(tree, tables, violations)

        table_list = sorted(tables)
        if violations:
            return ASTValidationResult.rejected(
                violations, statement_type=statement_type, tables=table_list, dialect=cfg.dialect
            )

        try:
            normalized = tree.sql(dialect=cfg.dialect)
        except SqlglotError as exc:
            return ASTValidationResult.rejected(
                _single(ViolationCode.PARSE_ERROR, f"Could not regenerate SQL: {_clean_error(exc)}"),
                statement_type=statement_type,
                tables=table_list,
                dialect=cfg.dialect,
            )
        return ASTValidationResult.ok(
            normalized, statement_type=statement_type, tables=table_list, dialect=cfg.dialect
        )

    # ------------------------------------------------------------------ internals
    def _walk(self, tree: exp.Expression, violations: list[ASTViolation]) -> set[str]:
        """Single pass over the AST; returns the set of real (non-CTE) table names."""
        cfg = self._config
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        tables: set[str] = set()
        node_count = 0

        for node in tree.find_all(exp.Expression):
            node_count += 1
            if node_count > cfg.max_nodes:
                _add(violations, ViolationCode.TOO_COMPLEX,
                     f"AST exceeds {cfg.max_nodes} nodes", dialect=cfg.dialect)
                break

            if isinstance(node, _FORBIDDEN_NODES):
                _add(violations, ViolationCode.FORBIDDEN_NODE,
                     f"{type(node).__name__} is not allowed", node=node, dialect=cfg.dialect)
                continue

            if isinstance(node, exp.Func):
                name = (
                    node.name.lower() if isinstance(node, exp.Anonymous) else node.sql_name().lower()
                )
                if name in cfg.forbidden_functions or name.startswith("pragma_"):
                    _add(violations, ViolationCode.FORBIDDEN_FUNCTION,
                         f"Function '{name}' is not allowed", node=node, dialect=cfg.dialect)

            elif isinstance(node, exp.With):
                if node.args.get("recursive") and not cfg.allow_recursive_cte:
                    _add(violations, ViolationCode.FORBIDDEN_NODE,
                         "WITH RECURSIVE is not allowed", node=node, dialect=cfg.dialect)

            elif isinstance(node, exp.Table):
                name = node.name.lower()
                if node.args.get("db") or node.args.get("catalog"):
                    _add(violations, ViolationCode.FORBIDDEN_NODE,
                         "Schema-qualified table references are not allowed",
                         node=node, dialect=cfg.dialect)
                if name.startswith(("sqlite_", "pragma_")):
                    _add(violations, ViolationCode.FORBIDDEN_NODE,
                         f"System table '{name}' is not allowed", node=node, dialect=cfg.dialect)
                if name and name not in cte_names:
                    tables.add(name)

        joins = sum(1 for _ in tree.find_all(exp.Join))
        if joins > cfg.max_joins:
            _add(violations, ViolationCode.TOO_COMPLEX,
                 f"Query has {joins} joins; limit is {cfg.max_joins}", dialect=cfg.dialect)
        return tables

    def _check_columns(
        self, tree: exp.Expression, tables: set[str], violations: list[ASTViolation]
    ) -> None:
        """Best-effort column existence check (query-wide scope)."""
        assert self._schema is not None  # noqa: S101 - guarded by caller
        dialect = self._config.dialect

        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        subquery_aliases = {s.alias.lower() for s in tree.find_all(exp.Subquery) if s.alias}
        local_scopes = cte_names | subquery_aliases

        alias_to_tables: dict[str, set[str]] = defaultdict(set)
        for table in tree.find_all(exp.Table):
            name = table.name.lower()
            if not name or name in cte_names:
                continue
            alias_to_tables[name].add(name)
            alias_to_tables[(table.alias_or_name or name).lower()].add(name)

        known_columns: set[str] = set()
        for table_name in tables:
            known_columns |= self._schema.get(table_name, frozenset())
        output_names = {
            expr.alias.lower()
            for select in tree.find_all(exp.Select)
            for expr in select.expressions
            if expr.alias
        }

        for column in tree.find_all(exp.Column):
            if isinstance(column.this, exp.Star):
                continue
            name = column.name.lower()
            qualifier = column.table.lower()

            if qualifier:
                candidates = alias_to_tables.get(qualifier)
                if candidates is None:
                    if qualifier not in local_scopes:
                        _add(violations, ViolationCode.UNKNOWN_TABLE,
                             f"Unknown table or alias '{qualifier}'", node=column, dialect=dialect)
                    continue
                if not any(name in self._schema.get(t, frozenset()) for t in candidates):
                    _add(violations, ViolationCode.UNKNOWN_COLUMN,
                         f"Column '{name}' not found in {', '.join(sorted(candidates))}",
                         node=column, dialect=dialect)
            elif name not in known_columns and name not in output_names:
                # SQLite treats an unresolvable "double-quoted" identifier as a string literal.
                is_quoted = isinstance(column.this, exp.Identifier) and bool(
                    column.this.args.get("quoted")
                )
                if not is_quoted:
                    _add(violations, ViolationCode.UNKNOWN_COLUMN,
                         f"Unknown column '{name}'", node=column, dialect=dialect)

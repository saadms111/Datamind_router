"""Execution-accuracy scoring and failure analysis.

Scoring follows Spider's execution-accuracy idea: a prediction is correct when it
returns the same result set as the gold query. Row order only matters when the
gold query has an ORDER BY; column order is tolerated (permutations are tried
for up to ``_MAX_PERMUTED_WIDTH`` columns); floats are compared to 6 decimals.

Complexity buckets are a *heuristic* derived from the gold SQL's AST. They are not
Spider's official hardness labels, so do not compare them with published numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import permutations
from typing import Final

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from datamind.schema import QueryStatus, SQLValue

__all__ = [
    "FailureCategory",
    "MatchResult",
    "MismatchKind",
    "classify_status",
    "compare_results",
    "complexity_bucket",
    "feature_mismatches",
    "gold_has_order_by",
    "sql_features",
]

_MAX_PERMUTED_WIDTH: Final = 6
_FLOAT_DIGITS: Final = 6

Row = Sequence[SQLValue]


class MismatchKind(StrEnum):
    COLUMN_COUNT = "column_count"
    ROW_COUNT = "row_count"
    ORDERING = "ordering"
    VALUES = "values"


class FailureCategory(StrEnum):
    GENERATION_ERROR = "generation_error"
    GUARD_REJECTED = "guard_rejected"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"
    WRONG_COLUMN_COUNT = "wrong_column_count"
    WRONG_ROW_COUNT = "wrong_row_count"
    WRONG_ORDERING = "wrong_ordering"
    WRONG_VALUES = "wrong_values"
    HARNESS_ERROR = "harness_error"


_STATUS_CATEGORY: Final = {
    QueryStatus.GENERATION_ERROR: FailureCategory.GENERATION_ERROR,
    QueryStatus.REJECTED: FailureCategory.GUARD_REJECTED,
    QueryStatus.EXECUTION_ERROR: FailureCategory.EXECUTION_ERROR,
    QueryStatus.TIMEOUT: FailureCategory.TIMEOUT,
}
_MISMATCH_CATEGORY: Final = {
    MismatchKind.COLUMN_COUNT: FailureCategory.WRONG_COLUMN_COUNT,
    MismatchKind.ROW_COUNT: FailureCategory.WRONG_ROW_COUNT,
    MismatchKind.ORDERING: FailureCategory.WRONG_ORDERING,
    MismatchKind.VALUES: FailureCategory.WRONG_VALUES,
}


def classify_status(status: QueryStatus) -> FailureCategory | None:
    """Failure category for a non-success status; None for SUCCESS."""
    return _STATUS_CATEGORY.get(status)


def mismatch_category(kind: MismatchKind) -> FailureCategory:
    return _MISMATCH_CATEGORY[kind]


@dataclass(frozen=True, slots=True)
class MatchResult:
    matched: bool
    kind: MismatchKind | None = None


def _norm_cell(value: SQLValue) -> SQLValue:
    return round(value, _FLOAT_DIGITS) if isinstance(value, float) else value


def _norm_rows(rows: Sequence[Row]) -> list[tuple[SQLValue, ...]]:
    return [tuple(_norm_cell(c) for c in row) for row in rows]


def _same(a: list[tuple[SQLValue, ...]], b: list[tuple[SQLValue, ...]], ordered: bool) -> bool:
    return a == b if ordered else Counter(a) == Counter(b)


def compare_results(pred_rows: Sequence[Row], gold_rows: Sequence[Row], *, ordered: bool) -> MatchResult:
    """Compare predicted and gold result sets (see module docstring for the rules)."""
    pred, gold = _norm_rows(pred_rows), _norm_rows(gold_rows)
    if pred and gold and len(pred[0]) != len(gold[0]):
        return MatchResult(False, MismatchKind.COLUMN_COUNT)
    if len(pred) != len(gold):
        return MatchResult(False, MismatchKind.ROW_COUNT)
    if _same(pred, gold, ordered):
        return MatchResult(True)

    width = len(gold[0]) if gold else 0
    if 1 < width <= _MAX_PERMUTED_WIDTH:
        identity = tuple(range(width))
        for perm in permutations(range(width)):
            if perm == identity:
                continue
            if _same([tuple(row[i] for i in perm) for row in pred], gold, ordered):
                return MatchResult(True)

    if ordered and Counter(pred) == Counter(gold):
        return MatchResult(False, MismatchKind.ORDERING)
    return MatchResult(False, MismatchKind.VALUES)


def _parse(sql: str) -> exp.Expression | None:
    try:
        return sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError:
        return None


def gold_has_order_by(sql: str) -> bool:
    tree = _parse(sql)
    return tree is not None and tree.args.get("order") is not None


def _count(tree: exp.Expression, *types: type[exp.Expression]) -> int:
    return sum(1 for _ in tree.find_all(*types))


def sql_features(sql: str) -> dict[str, int]:
    """Structural feature counts of a query; empty if it cannot be parsed."""
    tree = _parse(sql)
    if tree is None:
        return {}
    return {
        "joins": _count(tree, exp.Join),
        "group_by": _count(tree, exp.Group),
        "having": _count(tree, exp.Having),
        "order_by": _count(tree, exp.Order),
        "limit": _count(tree, exp.Limit),
        "distinct": _count(tree, exp.Distinct),
        "nested_selects": max(_count(tree, exp.Select) - 1, 0),
        "set_ops": _count(tree, exp.Union, exp.Intersect, exp.Except),
        "aggregates": _count(tree, exp.AggFunc),
        "like": _count(tree, exp.Like),
    }


def feature_mismatches(predicted_sql: str, gold_sql: str) -> list[str]:
    """Names of structural features whose counts differ between prediction and gold."""
    pred, gold = sql_features(predicted_sql), sql_features(gold_sql)
    if not pred or not gold:
        return []
    return sorted(name for name in gold if pred.get(name, 0) != gold[name])


def complexity_bucket(gold_sql: str) -> str:
    """Heuristic difficulty bucket: easy / medium / hard / extra (``unknown`` if unparsable)."""
    f = sql_features(gold_sql)
    if not f:
        return "unknown"
    score = (
        f["joins"]
        + f["group_by"]
        + f["having"]
        + f["nested_selects"]
        + f["set_ops"]
        + min(f["distinct"], 1)
        + min(f["limit"], 1)
        + min(f["order_by"], 1)
    )
    if score <= 1:
        return "easy"
    if score <= 3:
        return "medium"
    if score <= 5:
        return "hard"
    return "extra"

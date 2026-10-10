"""Tests for eval.scoring."""

from __future__ import annotations

import pytest

from datamind.schema import QueryStatus
from eval.scoring import (
    FailureCategory,
    MismatchKind,
    classify_status,
    compare_results,
    complexity_bucket,
    feature_mismatches,
    gold_has_order_by,
    sql_features,
)


def test_identical_results_match() -> None:
    assert compare_results([[1, "a"], [2, "b"]], [[1, "a"], [2, "b"]], ordered=True).matched


def test_unordered_ignores_row_order() -> None:
    assert compare_results([[2], [1]], [[1], [2]], ordered=False).matched


def test_ordered_detects_wrong_order() -> None:
    result = compare_results([[2], [1]], [[1], [2]], ordered=True)
    assert not result.matched and result.kind is MismatchKind.ORDERING


def test_column_permutation_is_tolerated() -> None:
    assert compare_results([["a", 1], ["b", 2]], [[1, "a"], [2, "b"]], ordered=False).matched


def test_duplicates_matter() -> None:
    result = compare_results([[1], [1]], [[1], [2]], ordered=False)
    assert not result.matched and result.kind is MismatchKind.VALUES


def test_column_count_mismatch() -> None:
    assert compare_results([[1, 2]], [[1]], ordered=False).kind is MismatchKind.COLUMN_COUNT


def test_row_count_mismatch() -> None:
    assert compare_results([[1]], [[1], [2]], ordered=False).kind is MismatchKind.ROW_COUNT


def test_float_tolerance_and_int_float_equality() -> None:
    assert compare_results([[1.00000001, 2.0]], [[1.0, 2]], ordered=False).matched


def test_both_empty_match_and_empty_vs_nonempty() -> None:
    assert compare_results([], [], ordered=False).matched
    assert compare_results([], [[1]], ordered=False).kind is MismatchKind.ROW_COUNT


def test_none_values() -> None:
    assert compare_results([[None]], [[None]], ordered=False).matched
    assert not compare_results([[None]], [[0]], ordered=False).matched


def test_gold_has_order_by() -> None:
    assert gold_has_order_by("SELECT a FROM t ORDER BY a")
    assert not gold_has_order_by("SELECT a FROM t")
    assert not gold_has_order_by("SELECT (")  # unparsable -> False


def test_sql_features() -> None:
    f = sql_features(
        "SELECT DISTINCT a, count(*) FROM t JOIN u ON t.id = u.id "
        "WHERE a IN (SELECT a FROM v) GROUP BY a HAVING count(*) > 1 ORDER BY a LIMIT 3"
    )
    assert f["joins"] == 1 and f["group_by"] == 1 and f["having"] == 1
    assert f["order_by"] == 1 and f["limit"] == 1 and f["distinct"] == 1
    assert f["nested_selects"] == 1 and f["aggregates"] >= 1
    assert sql_features("SELECT (") == {}


def test_feature_mismatches() -> None:
    gold = "SELECT a FROM t GROUP BY a"
    assert feature_mismatches("SELECT a FROM t", gold) == ["group_by"]
    assert feature_mismatches(gold, gold) == []
    assert feature_mismatches("SELECT (", gold) == []


def test_complexity_buckets_are_monotonic() -> None:
    order = ["easy", "medium", "hard", "extra"]
    queries = [
        "SELECT a FROM t",
        "SELECT a FROM t ORDER BY a LIMIT 1",
        "SELECT a FROM t JOIN u ON 1=1 GROUP BY a HAVING count(*) > 1",
        "SELECT DISTINCT a FROM t JOIN u ON 1=1 WHERE a IN (SELECT a FROM v) "
        "GROUP BY a HAVING count(*) > 1 ORDER BY a LIMIT 1",
    ]
    buckets = [complexity_bucket(q) for q in queries]
    assert buckets == sorted(buckets, key=order.index)
    assert buckets[0] == "easy" and buckets[-1] in {"hard", "extra"}
    assert complexity_bucket("SELECT (") == "unknown"


def test_classify_status() -> None:
    assert classify_status(QueryStatus.SUCCESS) is None
    assert classify_status(QueryStatus.REJECTED) is FailureCategory.GUARD_REJECTED
    assert classify_status(QueryStatus.TIMEOUT) is FailureCategory.TIMEOUT
    assert classify_status(QueryStatus.EXECUTION_ERROR) is FailureCategory.EXECUTION_ERROR
    assert classify_status(QueryStatus.GENERATION_ERROR) is FailureCategory.GENERATION_ERROR

"""Integration check: guard + executor against the real 50-example Spider sample.

Gold queries are known-good, so every one must (a) pass the guard with a
schema-aware column check, (b) execute, and (c) return the same rows when run
from the guard's regenerated SQL as from the original text.
"""

from __future__ import annotations

import pytest

from datamind.executor import ExecutionError, SafeExecutor, read_schema
from datamind.guardrails import SQLGuard
from eval.loader import DEFAULT_DATA_DIR, SpiderDataError, load_spider_sample


@pytest.fixture(scope="module")
def examples():
    try:
        return load_spider_sample(DEFAULT_DATA_DIR, ensure=False)
    except SpiderDataError as exc:
        pytest.skip(f"Spider sample not built: {exc}")


def _canonical(rows: list[list]) -> list[str]:
    return sorted(repr(r) for r in rows)


def test_gold_sql_passes_guard_and_roundtrips(examples) -> None:
    failures: list[str] = []
    for ex in examples:
        guard = SQLGuard(read_schema(ex.db_path))
        verdict = guard.validate(ex.gold_sql)
        if not verdict.is_valid:
            reasons = "; ".join(f"{v.code.value}: {v.message}" for v in verdict.violations)
            failures.append(f"[{ex.example_id}] guard rejected gold SQL ({reasons})\n    {ex.gold_sql}")
            continue

        executor = SafeExecutor(ex.db_path, max_rows=1000, timeout_s=10.0)
        try:
            original = executor.execute(ex.gold_sql)
            regenerated = executor.execute(verdict.normalized_sql or "")
        except ExecutionError as exc:
            failures.append(f"[{ex.example_id}] execution failed: {exc}\n    {ex.gold_sql}")
            continue

        if not original.truncated and _canonical(original.rows) != _canonical(regenerated.rows):
            failures.append(
                f"[{ex.example_id}] normalized SQL changed results\n"
                f"    original:   {ex.gold_sql}\n    normalized: {verdict.normalized_sql}"
            )

    assert not failures, f"{len(failures)}/{len(examples)} gold queries failed:\n" + "\n".join(failures)

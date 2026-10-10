"""Unit tests for datamind.guardrails.SQLGuard."""

from __future__ import annotations

import pytest

from datamind.guardrails import GuardrailConfig, SQLGuard
from datamind.schema import ViolationCode

SCHEMA = {
    "singer": ["singer_id", "name", "age", "country"],
    "concert": ["concert_id", "name", "singer_id", "year"],
}


def codes(result) -> set[ViolationCode]:
    return {v.code for v in result.violations}


@pytest.fixture
def guard() -> SQLGuard:
    return SQLGuard(SCHEMA)


@pytest.fixture
def schemaless() -> SQLGuard:
    return SQLGuard()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM singer",
        "SELECT name, age FROM singer WHERE age > 30 ORDER BY age DESC LIMIT 5;",
        "SELECT NAME FROM SINGER",
        'SELECT name FROM singer WHERE country = "France"',  # SQLite double-quote literal fallback
        "SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id",
        "SELECT name FROM singer UNION SELECT name FROM concert",
        "SELECT name AS n FROM singer ORDER BY n",
        "SELECT s.n FROM (SELECT name AS n FROM singer) AS s",
        "WITH old AS (SELECT * FROM singer WHERE age > 50) SELECT name FROM old",
        "SELECT t.* FROM singer t",
        "SELECT country, count(*) FROM singer GROUP BY country HAVING count(*) > 1",
    ],
)
def test_valid_queries_pass(guard: SQLGuard, sql: str) -> None:
    result = guard.validate(sql)
    assert result.is_valid, [v.message for v in result.violations]
    assert result.normalized_sql
    assert result.normalized_sql.upper().startswith(("SELECT", "WITH"))


def test_tables_are_reported_lowercase(guard: SQLGuard) -> None:
    result = guard.validate("SELECT 1 FROM Singer JOIN CONCERT ON 1 = 1")
    assert result.tables == ["concert", "singer"]


@pytest.mark.parametrize("sql", ["", "   ", ";"])
def test_empty_rejected(schemaless: SQLGuard, sql: str) -> None:
    result = schemaless.validate(sql)
    assert not result.is_valid
    assert codes(result) == {ViolationCode.EMPTY_STATEMENT}


@pytest.mark.parametrize("sql", ["SELECT (1 + FROM t", "SELECT 'unterminated"])
def test_parse_errors_rejected(schemaless: SQLGuard, sql: str) -> None:
    result = schemaless.validate(sql)
    assert not result.is_valid
    assert codes(result) == {ViolationCode.PARSE_ERROR}


def test_multiple_statements_rejected(schemaless: SQLGuard) -> None:
    result = schemaless.validate("SELECT 1; DROP TABLE singer")
    assert codes(result) == {ViolationCode.MULTIPLE_STATEMENTS}


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE singer",
        "DELETE FROM singer",
        "UPDATE singer SET age = 1",
        "INSERT INTO singer (name) VALUES ('x')",
        "CREATE TABLE x (a INT)",
        "ALTER TABLE singer ADD COLUMN z INT",
        "PRAGMA table_info(singer)",
        "ATTACH DATABASE 'other.db' AS o",
    ],
)
def test_non_select_rejected(schemaless: SQLGuard, sql: str) -> None:
    result = schemaless.validate(sql)
    assert not result.is_valid
    assert result.normalized_sql is None


def test_forbidden_function(schemaless: SQLGuard) -> None:
    result = schemaless.validate("SELECT load_extension('evil.so')")
    assert ViolationCode.FORBIDDEN_FUNCTION in codes(result)


def test_pragma_table_function(schemaless: SQLGuard) -> None:
    result = schemaless.validate("SELECT * FROM pragma_table_info('singer')")
    assert not result.is_valid


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM sqlite_master", "SELECT * FROM main.singer", "SELECT * FROM temp.singer"]
)
def test_system_and_qualified_tables_rejected(schemaless: SQLGuard, sql: str) -> None:
    result = schemaless.validate(sql)
    assert ViolationCode.FORBIDDEN_NODE in codes(result)


def test_recursive_cte_rejected_by_default(schemaless: SQLGuard) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT * FROM c"
    assert ViolationCode.FORBIDDEN_NODE in codes(schemaless.validate(sql))
    permissive = SQLGuard(config=GuardrailConfig(allow_recursive_cte=True))
    assert permissive.validate(sql).is_valid


def test_unknown_table(guard: SQLGuard) -> None:
    result = guard.validate("SELECT * FROM orders")
    assert ViolationCode.UNKNOWN_TABLE in codes(result)


def test_unknown_column(guard: SQLGuard) -> None:
    result = guard.validate("SELECT salary FROM singer")
    assert ViolationCode.UNKNOWN_COLUMN in codes(result)


def test_unknown_qualified_column(guard: SQLGuard) -> None:
    result = guard.validate("SELECT T1.salary FROM singer AS T1")
    assert ViolationCode.UNKNOWN_COLUMN in codes(result)


def test_unknown_alias(guard: SQLGuard) -> None:
    result = guard.validate("SELECT x.name FROM singer AS T1")
    assert ViolationCode.UNKNOWN_TABLE in codes(result)


def test_complexity_limits() -> None:
    tiny = SQLGuard(config=GuardrailConfig(max_joins=1))
    sql = "SELECT 1 FROM a JOIN b ON 1=1 JOIN c ON 1=1"
    assert ViolationCode.TOO_COMPLEX in codes(tiny.validate(sql))
    short = SQLGuard(config=GuardrailConfig(max_sql_chars=10))
    assert ViolationCode.TOO_COMPLEX in codes(short.validate("SELECT 1 FROM singer"))


def test_normalized_sql_is_regenerated_from_ast(schemaless: SQLGuard) -> None:
    result = schemaless.validate("select   name   from   singer   -- trailing comment")
    assert result.is_valid
    assert "--" not in (result.normalized_sql or "") or result.normalized_sql.strip().endswith("*/")

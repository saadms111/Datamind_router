"""Unit tests for datamind.executor."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from datamind.executor import ExecutionError, ExecutionTimeout, SafeExecutor, read_schema


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
    conn.executemany(
        "INSERT INTO singer (name, age) VALUES (?, ?)", [(f"s{i}", 20 + i) for i in range(10)]
    )
    conn.commit()
    conn.close()
    return path


def test_basic_select(db_path: Path) -> None:
    result = SafeExecutor(db_path).execute("SELECT name, age FROM singer ORDER BY age LIMIT 2")
    assert result.columns == ["name", "age"]
    assert result.rows == [["s0", 20], ["s1", 21]]
    assert not result.truncated


def test_row_cap_sets_truncated(db_path: Path) -> None:
    result = SafeExecutor(db_path).execute("SELECT * FROM singer", max_rows=3)
    assert len(result.rows) == 3
    assert result.truncated


def test_blob_and_infinity_are_json_safe(db_path: Path) -> None:
    result = SafeExecutor(db_path).execute("SELECT x'00ff', 1e999")
    assert result.rows == [["<BLOB 2 bytes>", "inf"]]


def test_long_cells_are_truncated(db_path: Path) -> None:
    executor = SafeExecutor(db_path, max_cell_chars=5)
    assert executor.execute("SELECT 'abcdefghij'").rows == [["abcde…"]]


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM singer",
        "DROP TABLE singer",
        "CREATE TABLE x (a INT)",
        "INSERT INTO singer (name) VALUES ('x')",
        "ATTACH DATABASE ':memory:' AS m",
        "PRAGMA table_info(singer)",
        "SELECT * FROM sqlite_master",
    ],
)
def test_executor_blocks_writes_and_introspection(db_path: Path, sql: str) -> None:
    with pytest.raises(ExecutionError):
        SafeExecutor(db_path).execute(sql)
    # Data untouched.
    assert SafeExecutor(db_path).execute("SELECT count(*) FROM singer").rows == [[10]]


def test_forbidden_function_denied(db_path: Path) -> None:
    with pytest.raises(ExecutionError):
        SafeExecutor(db_path).execute("SELECT zeroblob(10)")


def test_syntax_error(db_path: Path) -> None:
    with pytest.raises(ExecutionError):
        SafeExecutor(db_path).execute("SELEC 1")


def test_timeout(db_path: Path) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    with pytest.raises(ExecutionTimeout):
        SafeExecutor(db_path, timeout_s=0.3).execute(sql)


def test_missing_database(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SafeExecutor(tmp_path / "nope.sqlite")


def test_read_schema(db_path: Path) -> None:
    assert read_schema(db_path) == {"singer": ["singer_id", "name", "age"]}

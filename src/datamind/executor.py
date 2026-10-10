"""Defence-in-depth SQLite execution.

The AST guard (``guardrails.py``) is the first line of defence; this module is
the second and does not trust it. Every query runs on a fresh connection that is:

* opened read-only (``mode=ro``) with ``PRAGMA query_only = ON``
* restricted by a SQLite *authorizer* that allows only SELECT/READ/RECURSIVE and
  non-blacklisted functions (so DDL, DML, ATTACH, PRAGMA and ``sqlite_*`` reads
  are denied by the engine itself)
* bounded in wall-clock time via a progress handler that aborts the statement
* bounded in result size (rows fetched ``max_rows + 1``; long cells truncated)

Results are converted to JSON-safe values matching ``datamind.schema.SQLValue``.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import time
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from datamind.guardrails import DEFAULT_FORBIDDEN_FUNCTIONS
from datamind.schema import SQLValue

__all__ = [
    "ExecutionError",
    "ExecutionResult",
    "ExecutionTimeout",
    "SafeExecutor",
    "read_schema",
]

logger = logging.getLogger(__name__)

# Authorizer action codes (fallbacks are the stable values from sqlite3.h).
_SQLITE_READ: Final[int] = getattr(sqlite3, "SQLITE_READ", 20)
_SQLITE_SELECT: Final[int] = getattr(sqlite3, "SQLITE_SELECT", 21)
_SQLITE_FUNCTION: Final[int] = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_SQLITE_RECURSIVE: Final[int] = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_PASSIVE_ACTIONS: Final = frozenset({_SQLITE_SELECT, _SQLITE_RECURSIVE})

_PROGRESS_INTERVAL: Final = 1_000  # SQLite VM instructions between timeout checks
_DEFAULT_MAX_CELL_CHARS: Final = 2_000


class ExecutionError(Exception):
    """The query failed to run (syntax error, denied operation, missing DB, ...)."""


class ExecutionTimeout(ExecutionError):
    """The query exceeded its wall-clock budget and was aborted."""


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    columns: list[str]
    rows: list[list[SQLValue]]
    truncated: bool
    elapsed_ms: float


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def read_schema(db_path: Path) -> dict[str, list[str]]:
    """Return ``{table: [column, ...]}`` for a SQLite database (user tables only).

    Raises:
        FileNotFoundError: if ``db_path`` does not exist.
        ExecutionError: if the database cannot be read.
    """
    path = Path(db_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ExecutionError(f"Cannot open database {path}: {exc}") from exc
    try:
        conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
        names = [
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' ORDER BY name"
            )
        ]
        return {
            name: [str(col[1]) for col in conn.execute(f"PRAGMA table_info({_quote_identifier(name)})")]
            for name in names
        }
    except sqlite3.Error as exc:
        raise ExecutionError(f"Cannot read schema of {path}: {exc}") from exc
    finally:
        conn.close()


def _to_json_safe(value: object, max_cell_chars: int) -> SQLValue:
    if value is None or isinstance(value, (int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        if len(value) > max_cell_chars:
            return value[:max_cell_chars] + "…"
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<BLOB {len(bytes(value))} bytes>"
    return str(value)


class SafeExecutor:
    """Runs read-only queries against one SQLite database with hard limits."""

    def __init__(
        self,
        db_path: Path,
        *,
        max_rows: int = 100,
        timeout_s: float = 10.0,
        max_cell_chars: int = _DEFAULT_MAX_CELL_CHARS,
        forbidden_functions: Collection[str] = DEFAULT_FORBIDDEN_FUNCTIONS,
    ) -> None:
        if max_rows < 1 or timeout_s <= 0 or max_cell_chars < 1:
            raise ValueError("max_rows, timeout_s and max_cell_chars must be positive")
        path = Path(db_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Database not found: {path}")
        self._db_path = path
        self._max_rows = max_rows
        self._timeout_s = timeout_s
        self._max_cell_chars = max_cell_chars
        self._forbidden = frozenset(f.lower() for f in forbidden_functions)

    def _authorizer(
        self,
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        source: str | None,
    ) -> int:
        if action == _SQLITE_FUNCTION:
            name = (arg2 or "").lower()
            denied = name in self._forbidden or name.startswith("pragma_")
            return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
        if action == _SQLITE_READ:
            denied = (arg1 or "").lower().startswith("sqlite_")
            return sqlite3.SQLITE_DENY if denied else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_OK if action in _PASSIVE_ACTIONS else sqlite3.SQLITE_DENY

    def execute(
        self,
        sql: str,
        *,
        max_rows: int | None = None,
        timeout_s: float | None = None,
    ) -> ExecutionResult:
        """Execute one query.

        Raises:
            ExecutionTimeout: the time budget was exceeded.
            ExecutionError: any other failure, including operations denied by the
                authorizer.
            ValueError: invalid ``max_rows`` / ``timeout_s``.
        """
        limit = self._max_rows if max_rows is None else max_rows
        budget = self._timeout_s if timeout_s is None else timeout_s
        if limit < 1 or budget <= 0:
            raise ValueError("max_rows must be >= 1 and timeout_s must be > 0")

        started = time.perf_counter()
        deadline = time.monotonic() + budget
        timed_out = False

        def progress() -> int:
            nonlocal timed_out
            if time.monotonic() > deadline:
                timed_out = True
                return 1  # non-zero aborts the running statement
            return 0

        try:
            conn = sqlite3.connect(f"{self._db_path.as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise ExecutionError(f"Cannot open database {self._db_path}: {exc}") from exc

        try:
            conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
            conn.execute("PRAGMA query_only = ON")  # must precede the authorizer
            conn.set_authorizer(self._authorizer)
            conn.set_progress_handler(progress, _PROGRESS_INTERVAL)
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description or ()]
            fetched = cursor.fetchmany(limit + 1)
        except sqlite3.Error as exc:
            if timed_out:
                raise ExecutionTimeout(f"Query exceeded {budget:.1f}s time limit") from exc
            raise ExecutionError(str(exc)) from exc
        finally:
            conn.close()

        truncated = len(fetched) > limit
        rows = [
            [_to_json_safe(cell, self._max_cell_chars) for cell in row] for row in fetched[:limit]
        ]
        return ExecutionResult(
            columns=columns,
            rows=rows,
            truncated=truncated,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

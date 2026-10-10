"""Registry of SQLite databases the agent may query."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from datamind.executor import ExecutionError, read_schema

__all__ = ["CatalogError", "DatabaseCatalog", "DatabaseInfo", "UnknownDatabaseError"]

logger = logging.getLogger(__name__)

_DB_ID_RE: Final = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


class CatalogError(RuntimeError):
    """A registered database could not be read."""


class UnknownDatabaseError(LookupError):
    """The requested ``db_id`` is not registered."""


@dataclass(frozen=True, slots=True)
class DatabaseInfo:
    db_id: str
    path: Path
    schema_ddl: str
    schema: dict[str, list[str]]


def _read_ddl(path: Path) -> str:
    try:
        conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise CatalogError(f"Cannot open database {path}: {exc}") from exc
    try:
        conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    except sqlite3.Error as exc:
        raise CatalogError(f"Cannot read schema of {path}: {exc}") from exc
    finally:
        conn.close()
    ddl = "\n\n".join(f"{str(row[0]).strip().rstrip(';')};" for row in rows)
    if not ddl:
        raise CatalogError(f"Database {path} contains no tables")
    return ddl


class DatabaseCatalog:
    """Maps ``db_id`` to a database file and lazily caches its schema."""

    def __init__(self, paths: Mapping[str, Path]) -> None:
        for db_id, path in paths.items():
            if not _DB_ID_RE.match(db_id):
                raise ValueError(f"Invalid db_id: {db_id!r}")
            if not Path(path).is_file():
                raise FileNotFoundError(f"Database file not found for '{db_id}': {path}")
        self._paths = {db_id: Path(p).resolve() for db_id, p in paths.items()}
        self._cache: dict[str, DatabaseInfo] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_directory(cls, root: Path) -> DatabaseCatalog:
        """Register every ``<root>/<db_id>/<db_id>.sqlite`` (Spider layout)."""
        root = Path(root)
        if not root.is_dir():
            raise FileNotFoundError(f"Database directory not found: {root}")
        paths = {
            child.name: child / f"{child.name}.sqlite"
            for child in sorted(root.iterdir())
            if child.is_dir()
            and _DB_ID_RE.match(child.name)
            and (child / f"{child.name}.sqlite").is_file()
        }
        if not paths:
            raise CatalogError(f"No <db_id>/<db_id>.sqlite databases found in {root}")
        return cls(paths)

    @property
    def db_ids(self) -> list[str]:
        return sorted(self._paths)

    def get(self, db_id: str) -> DatabaseInfo:
        """Return schema info for ``db_id``.

        Raises:
            UnknownDatabaseError: ``db_id`` is not registered.
            CatalogError: the database cannot be read.
        """
        path = self._paths.get(db_id)
        if path is None:
            raise UnknownDatabaseError(f"Unknown database '{db_id}'")
        with self._lock:
            cached = self._cache.get(db_id)
            if cached is not None:
                return cached
            try:
                schema = read_schema(path)
            except ExecutionError as exc:
                raise CatalogError(str(exc)) from exc
            info = DatabaseInfo(db_id=db_id, path=path, schema_ddl=_read_ddl(path), schema=schema)
            self._cache[db_id] = info
            return info

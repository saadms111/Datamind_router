"""Prompt construction and SQL extraction for the Text-to-SQL agent."""

from __future__ import annotations

import re
from typing import Final

__all__ = ["SYSTEM_PROMPT", "build_feedback_prompt", "build_user_prompt", "extract_sql"]

SYSTEM_PROMPT: Final = """You are an expert SQLite analyst. Translate the user's question into a single SQLite query.

Rules:
- Reply with exactly one SQL statement inside a ```sql fenced block and nothing else.
- The query must be read-only: a SELECT (optionally with CTEs or set operations). Never modify data or schema.
- Use only the tables and columns that appear in the schema.
- Do not use schema-qualified names, PRAGMA statements, or sqlite_* tables.
- Use single quotes for string literals.
- Return only the columns the question asks for."""

_FENCE_RE: Final = re.compile(r"```[ \t]*(?:sqlite|sql)?[ \t]*\r?\n?(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)
_BARE_SQL_RE: Final = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def build_user_prompt(question: str, schema_ddl: str) -> str:
    return f"Database schema:\n{schema_ddl}\n\nQuestion: {question}"


def build_feedback_prompt(problem: str) -> str:
    return (
        f"That query could not be used: {problem}\n"
        "Return a corrected single read-only SELECT query in a ```sql fenced block."
    )


def extract_sql(text: str) -> str | None:
    """Pull the SQL out of a model reply, or return None if there is none.

    Prefers the first fenced block (tolerating an unterminated fence from a
    truncated reply); otherwise accepts a reply that is itself a bare query.
    """
    match = _FENCE_RE.search(text)
    candidate = match.group(1).strip() if match else text.strip()
    if not candidate:
        return None
    if match is None and not _BARE_SQL_RE.match(candidate):
        return None
    return candidate

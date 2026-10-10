"""Download and materialise a deterministic 50-example subset of Spider.

Sources
-------
* Question / gold-SQL pairs: ``xlangai/spider`` (``validation`` split; the Spider
  test set is withheld by the authors).
* SQLite databases: the ``HAL-9001/spider-databases`` re-host of the official Yale
  archive, pinned to a commit and verified by SHA-256.

Output layout (under ``data/spider_sample/`` by default)::

    examples.jsonl                     one SpiderExample per line
    manifest.json                      provenance + parameters
    tables.json                        Spider tables.json filtered to used databases
    databases/<db_id>/<db_id>.sqlite   only the databases the sample needs

Selection is seeded and round-robin across databases, and only keeps examples
whose gold SQL is a SELECT that executes successfully against its database.

CLI::

    PYTHONPATH=src python -m eval.loader --size 50 --seed 13
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time
import zipfile
from collections import defaultdict, deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Final

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import HfHubHTTPError
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

__all__ = [
    "ChecksumMismatchError",
    "DEFAULT_DATA_DIR",
    "SampleManifest",
    "SampleNotFoundError",
    "SpiderDataError",
    "SpiderExample",
    "build_spider_sample",
    "download_databases_archive",
    "extract_archive_member",
    "index_databases_archive",
    "load_spider_sample",
    "read_schema_ddl",
    "main",
]

logger = logging.getLogger(__name__)

# --- Source configuration -----------------------------------------------------
QUESTIONS_REPO: Final = "xlangai/spider"
QUESTIONS_SPLIT: Final = "validation"
DATABASES_REPO: Final = "HAL-9001/spider-databases"
DATABASES_ARCHIVE: Final = "spider_data.zip"
DATABASES_REVISION: Final = "4a01bbac6520cd35b216db9e1724e5e1ada60aa4"
DATABASES_SHA256: Final = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"

# --- Defaults -------------------------------------------------------------------
DEFAULT_DATA_DIR: Final = Path(__file__).resolve().parents[2] / "data" / "spider_sample"
DEFAULT_SAMPLE_SIZE: Final = 50
DEFAULT_SEED: Final = 13
DEFAULT_GOLD_TIMEOUT_S: Final = 10.0

_EXAMPLES_FILE: Final = "examples.jsonl"
_MANIFEST_FILE: Final = "manifest.json"
_TABLES_FILE: Final = "tables.json"
_DATABASES_DIR: Final = "databases"
_GOLD_FETCH_LIMIT: Final = 1000
_HASH_CHUNK: Final = 1 << 20
_SELECT_RE: Final = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_SAFE_DB_ID_RE: Final = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

DbId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]{1,128}$")]


# --- Errors ---------------------------------------------------------------------
class SpiderDataError(RuntimeError):
    """Base error for anything that goes wrong preparing or reading the sample."""


class SampleNotFoundError(SpiderDataError):
    """The sample has not been built in the requested directory."""


class ChecksumMismatchError(SpiderDataError):
    """A downloaded artifact did not match its pinned SHA-256."""


# --- Data models ----------------------------------------------------------------
class SpiderExample(BaseModel):
    """One evaluation example, bound to a local SQLite database."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    example_id: str
    db_id: DbId
    question: str
    gold_sql: str
    schema_ddl: str
    db_path: Path  # absolute at load time; not persisted in examples.jsonl


class SampleManifest(BaseModel):
    """Provenance and parameters of a built sample."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    sample_size: int
    seed: int
    example_count: int
    questions_dataset: str
    questions_split: str
    databases_repo: str
    databases_revision: str
    archive_sha256: str
    db_ids: list[str]


@dataclass(frozen=True, slots=True)
class _Candidate:
    db_id: str
    question: str
    gold_sql: str


# --- Low-level helpers ------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database read-only, tolerating Spider's non-UTF-8 text."""
    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    return conn


def _gold_sql_executes(db_path: Path, sql: str, timeout_s: float) -> bool:
    """Return True if ``sql`` runs to completion (or first 1000 rows) within the timeout."""
    deadline = time.monotonic() + timeout_s
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error as exc:
        logger.warning("Cannot open %s: %s", db_path, exc)
        return False
    try:
        # Returning non-zero aborts the running statement.
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        conn.execute(sql).fetchmany(_GOLD_FETCH_LIMIT)
        return True
    except sqlite3.Error as exc:
        logger.debug("Gold SQL failed on %s: %s (%s)", db_path.name, exc, sql)
        return False
    finally:
        conn.close()


def _schema_ddl(db_path: Path) -> str:
    """Return the CREATE TABLE statements of a database, one per table."""
    try:
        conn = _connect_readonly(db_path)
    except sqlite3.Error as exc:
        raise SpiderDataError(f"Cannot open database {db_path}: {exc}") from exc
    try:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
            "AND sql IS NOT NULL ORDER BY name"
        ).fetchall()
    except sqlite3.Error as exc:
        raise SpiderDataError(f"Cannot read schema of {db_path}: {exc}") from exc
    finally:
        conn.close()
    ddl = "\n\n".join(f"{str(row[0]).strip().rstrip(';')};" for row in rows)
    if not ddl:
        raise SpiderDataError(f"Database {db_path} contains no tables")
    return ddl


# --- Source retrieval ---------------------------------------------------------------
def _fetch_candidates() -> list[_Candidate]:
    """Load question/SQL pairs, dropping malformed rows and non-SELECT gold queries."""
    try:
        dataset = load_dataset(QUESTIONS_REPO, split=QUESTIONS_SPLIT)
    except Exception as exc:  # noqa: BLE001 - `datasets` raises many unrelated error types
        raise SpiderDataError(
            f"Failed to load {QUESTIONS_REPO}[{QUESTIONS_SPLIT}] from the Hugging Face Hub: {exc}"
        ) from exc

    candidates: list[_Candidate] = []
    for row in dataset:
        db_id = str(row.get("db_id") or "").strip()
        question = str(row.get("question") or "").strip()
        gold_sql = str(row.get("query") or "").strip()
        if not (db_id and question and gold_sql):
            continue
        if not _SAFE_DB_ID_RE.match(db_id) or not _SELECT_RE.match(gold_sql):
            continue
        candidates.append(_Candidate(db_id=db_id, question=question, gold_sql=gold_sql))

    if not candidates:
        raise SpiderDataError(f"{QUESTIONS_REPO}[{QUESTIONS_SPLIT}] yielded no usable examples")
    return candidates


def _download_archive(cache_dir: Path | None) -> Path:
    """Fetch the pinned database archive and verify its checksum."""
    try:
        archive = Path(
            hf_hub_download(
                repo_id=DATABASES_REPO,
                filename=DATABASES_ARCHIVE,
                repo_type="dataset",
                revision=DATABASES_REVISION,
                cache_dir=cache_dir,
            )
        )
    except (HfHubHTTPError, OSError, ValueError) as exc:
        raise SpiderDataError(
            f"Failed to download {DATABASES_ARCHIVE} from {DATABASES_REPO}: {exc}"
        ) from exc

    logger.info("Verifying %s ...", archive.name)
    actual = _sha256_file(archive)
    if actual != DATABASES_SHA256:
        raise ChecksumMismatchError(
            f"SHA-256 mismatch for {archive}: expected {DATABASES_SHA256}, got {actual}. "
            "Delete the cached file and retry."
        )
    return archive


def _index_archive(zf: zipfile.ZipFile) -> tuple[dict[str, str], str | None]:
    """Map ``db_id`` -> archive member for each SQLite file, and locate tables.json."""
    db_index: dict[str, str] = {}
    tables_member: str | None = None
    for name in zf.namelist():
        parts = PurePosixPath(name).parts
        if len(parts) >= 3 and parts[-3] == "database" and parts[-1] == f"{parts[-2]}.sqlite":
            db_index[parts[-2]] = name
        elif parts and parts[-1] == "tables.json":
            if tables_member is None or len(parts) < len(PurePosixPath(tables_member).parts):
                tables_member = name
    if not db_index:
        raise SpiderDataError("Archive contains no database/<db_id>/<db_id>.sqlite files")
    return db_index, tables_member


def _extract_member(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, dest.open("wb") as out:
        shutil.copyfileobj(src, out, _HASH_CHUNK)


def _read_tables_subset(
    zf: zipfile.ZipFile, member: str | None, db_ids: set[str]
) -> list[dict[str, Any]] | None:
    if member is None:
        logger.warning("tables.json not found in archive; skipping")
        return None
    try:
        data = json.loads(zf.read(member))
    except (json.JSONDecodeError, KeyError, zipfile.BadZipFile) as exc:
        logger.warning("Could not read %s from archive: %s", member, exc)
        return None
    if not isinstance(data, list):
        logger.warning("Unexpected tables.json structure; skipping")
        return None
    return [t for t in data if isinstance(t, dict) and t.get("db_id") in db_ids]


# --- Sampling ---------------------------------------------------------------------------
def _select_candidates(
    candidates: Sequence[_Candidate],
    staged_dbs: Path,
    size: int,
    seed: int,
    gold_timeout_s: float,
) -> list[_Candidate]:
    """Seeded round-robin selection across databases, keeping only executable gold SQL."""
    by_db: dict[str, list[_Candidate]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for cand in candidates:
        key = (cand.db_id, cand.question)
        if key not in seen:
            seen.add(key)
            by_db[cand.db_id].append(cand)

    rng = random.Random(seed)
    db_order = sorted(by_db)
    rng.shuffle(db_order)
    queues: dict[str, deque[_Candidate]] = {}
    for db_id in db_order:
        items = by_db[db_id]
        rng.shuffle(items)
        queues[db_id] = deque(items)

    chosen: list[_Candidate] = []
    while len(chosen) < size and queues:
        for db_id in list(queues):
            queue = queues[db_id]
            db_path = staged_dbs / db_id / f"{db_id}.sqlite"
            while queue:
                cand = queue.popleft()
                if _gold_sql_executes(db_path, cand.gold_sql, gold_timeout_s):
                    chosen.append(cand)
                    break
            if not queue:
                del queues[db_id]
            if len(chosen) >= size:
                break

    if len(chosen) < size:
        raise SpiderDataError(
            f"Only {len(chosen)} valid examples available; requested {size}"
        )
    return chosen


# --- Persistence --------------------------------------------------------------------------
def _install(stage_out: Path, data_dir: Path) -> None:
    """Replace managed artifacts in ``data_dir`` with freshly built ones.

    The manifest is removed first and installed last, so an interrupted install
    is detected as "no sample" rather than a half-written one. Files in
    ``data_dir`` that this module does not manage are left untouched.
    """
    install_order = (_DATABASES_DIR, _TABLES_FILE, _EXAMPLES_FILE, _MANIFEST_FILE)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in reversed(install_order):
        target = data_dir / name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    for name in install_order:
        src = stage_out / name
        if src.exists():
            shutil.move(str(src), str(data_dir / name))


def _read_sample(data_dir: Path) -> tuple[SampleManifest, list[SpiderExample]]:
    manifest_path = data_dir / _MANIFEST_FILE
    examples_path = data_dir / _EXAMPLES_FILE
    if not manifest_path.is_file() or not examples_path.is_file():
        raise SampleNotFoundError(f"No Spider sample found in {data_dir}")

    try:
        manifest = SampleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        lines = examples_path.read_text(encoding="utf-8").splitlines()
    except (ValidationError, OSError, UnicodeDecodeError) as exc:
        raise SpiderDataError(f"Corrupt Spider sample in {data_dir}: {exc}") from exc

    examples: list[SpiderExample] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("record is not a JSON object")
            db_id = str(record.get("db_id", ""))
            example = SpiderExample.model_validate(
                {**record, "db_path": data_dir / _DATABASES_DIR / db_id / f"{db_id}.sqlite"}
            )
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise SpiderDataError(f"{examples_path}:{lineno}: invalid example: {exc}") from exc
        if not example.db_path.is_file():
            raise SpiderDataError(f"{examples_path}:{lineno}: missing database {example.db_path}")
        examples.append(example)

    if len(examples) != manifest.example_count:
        raise SpiderDataError(
            f"Manifest lists {manifest.example_count} examples but {examples_path} has {len(examples)}"
        )
    return manifest, examples


# --- Public API -----------------------------------------------------------------------------
def build_spider_sample(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
    gold_timeout_s: float = DEFAULT_GOLD_TIMEOUT_S,
) -> list[SpiderExample]:
    """Download Spider sources and (re)build the sample in ``data_dir``.

    Raises:
        ValueError: for invalid arguments.
        SpiderDataError: for download, integrity, or data problems.
    """
    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    if gold_timeout_s <= 0:
        raise ValueError("gold_timeout_s must be > 0")

    data_dir = data_dir.resolve()
    data_dir.parent.mkdir(parents=True, exist_ok=True)

    candidates = _fetch_candidates()
    archive = _download_archive(cache_dir)

    with tempfile.TemporaryDirectory(prefix=".spider_stage_", dir=data_dir.parent) as tmp:
        stage = Path(tmp)
        staged_dbs = stage / "raw"
        stage_out = stage / "out"

        try:
            zf = zipfile.ZipFile(archive)
        except (zipfile.BadZipFile, OSError) as exc:
            raise SpiderDataError(f"Cannot open archive {archive}: {exc}") from exc

        with zf:
            db_index, tables_member = _index_archive(zf)
            missing = {c.db_id for c in candidates} - set(db_index)
            if missing:
                logger.warning("Databases missing from archive: %s", ", ".join(sorted(missing)))
            available = [c for c in candidates if c.db_id in db_index]
            if not available:
                raise SpiderDataError("No candidate examples have a matching database")

            for db_id in sorted({c.db_id for c in available}):
                _extract_member(zf, db_index[db_id], staged_dbs / db_id / f"{db_id}.sqlite")

            chosen = _select_candidates(available, staged_dbs, sample_size, seed, gold_timeout_s)
            db_ids = sorted({c.db_id for c in chosen})
            tables = _read_tables_subset(zf, tables_member, set(db_ids))

        ddl_by_db = {
            db_id: _schema_ddl(staged_dbs / db_id / f"{db_id}.sqlite") for db_id in db_ids
        }
        examples = [
            SpiderExample(
                example_id=f"spider-dev-{i:03d}",
                db_id=cand.db_id,
                question=cand.question,
                gold_sql=cand.gold_sql,
                schema_ddl=ddl_by_db[cand.db_id],
                db_path=data_dir / _DATABASES_DIR / cand.db_id / f"{cand.db_id}.sqlite",
            )
            for i, cand in enumerate(chosen)
        ]
        manifest = SampleManifest(
            sample_size=sample_size,
            seed=seed,
            example_count=len(examples),
            questions_dataset=QUESTIONS_REPO,
            questions_split=QUESTIONS_SPLIT,
            databases_repo=DATABASES_REPO,
            databases_revision=DATABASES_REVISION,
            archive_sha256=DATABASES_SHA256,
            db_ids=db_ids,
        )

        try:
            for db_id in db_ids:
                dest = stage_out / _DATABASES_DIR / db_id / f"{db_id}.sqlite"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged_dbs / db_id / f"{db_id}.sqlite", dest)
            (stage_out / _EXAMPLES_FILE).write_text(
                "".join(
                    json.dumps(ex.model_dump(mode="json", exclude={"db_path"}), ensure_ascii=False)
                    + "\n"
                    for ex in examples
                ),
                encoding="utf-8",
            )
            if tables is not None:
                (stage_out / _TABLES_FILE).write_text(
                    json.dumps(tables, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            (stage_out / _MANIFEST_FILE).write_text(
                manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            _install(stage_out, data_dir)
        except OSError as exc:
            raise SpiderDataError(f"Failed to write sample to {data_dir}: {exc}") from exc

    logger.info("Built %d examples across %d databases in %s", len(examples), len(db_ids), data_dir)
    return _read_sample(data_dir)[1]


def load_spider_sample(
    data_dir: Path = DEFAULT_DATA_DIR,
    *,
    ensure: bool = True,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[SpiderExample]:
    """Load the sample, building it first if it is missing, stale, or ``force`` is set.

    With ``ensure=False`` no network access happens: a missing or mismatched
    sample raises :class:`SampleNotFoundError` / :class:`SpiderDataError`.
    """
    data_dir = data_dir.resolve()
    if not force:
        try:
            manifest, examples = _read_sample(data_dir)
        except SpiderDataError as exc:
            if not ensure:
                raise
            logger.info("Sample unavailable (%s); building.", exc)
        else:
            if manifest.sample_size == sample_size and manifest.seed == seed:
                return examples
            if not ensure:
                raise SpiderDataError(
                    f"Existing sample has size={manifest.sample_size}, seed={manifest.seed}; "
                    f"requested size={sample_size}, seed={seed}"
                )
            logger.info("Existing sample parameters differ; rebuilding.")
    elif not ensure:
        raise ValueError("force=True requires ensure=True")

    return build_spider_sample(
        data_dir, sample_size=sample_size, seed=seed, cache_dir=cache_dir
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the Spider evaluation subset.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face cache dir.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a sample exists.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        examples = load_spider_sample(
            args.data_dir,
            sample_size=args.size,
            seed=args.seed,
            cache_dir=args.cache_dir,
            force=args.force,
        )
    except (SpiderDataError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    n_dbs = len({ex.db_id for ex in examples})
    print(f"Spider sample ready: {len(examples)} examples across {n_dbs} databases in {args.data_dir}")
    return 0


# Public aliases so other pipelines (e.g. router.data) can reuse the verified
# download/extract/DDL helpers without importing underscore-prefixed names.
download_databases_archive = _download_archive
index_databases_archive = _index_archive
extract_archive_member = _extract_member
read_schema_ddl = _schema_ddl


if __name__ == "__main__":
    sys.exit(main())

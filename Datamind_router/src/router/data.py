"""Training-data preparation for LoRA fine-tuning on Spider.

Pipeline: Spider *train* split (``xlangai/spider``) + the SQLite schemas from the
pinned Spider archive -> SFT records -> split by database -> JSONL.

Design points
-------------
* **Train/serve prompt parity.** Prompts are built with the same
  ``SYSTEM_PROMPT`` / ``build_user_prompt`` and the same SQLite DDL that
  ``DataMindAgent`` sends at inference, and the target is a fenced ```sql block,
  the format ``extract_sql`` parses.
* **No leakage.** The evaluation sample comes from Spider's *validation* split,
  whose databases are disjoint from train. ``assert_no_leakage`` verifies that
  against the evaluation manifest and refuses to continue on overlap.
* **Safe targets only.** Gold queries must pass the structural AST guard so the
  tuned model is not taught SQL the agent would reject.
* **Validation split by database**, not by row, so the monitoring loss is not
  inflated by near-duplicate questions on the same schema.

Output (``data/spider_train/`` by default): ``train.jsonl``, ``val.jsonl``,
``schemas.json`` (cache), ``manifest.json``.

CLI::

    PYTHONPATH=src python -m router.data
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Protocol

from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, ValidationError

from datamind.guardrails import SQLGuard
from datamind.prompts import SYSTEM_PROMPT, build_user_prompt
from eval.loader import (
    DATABASES_REVISION,
    DATABASES_SHA256,
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    QUESTIONS_REPO,
    SampleManifest,
    SpiderDataError,
    download_databases_archive,
    extract_archive_member,
    index_databases_archive,
    read_schema_ddl,
)

__all__ = [
    "DEFAULT_TRAIN_DATA_DIR",
    "DataManifest",
    "LeakageError",
    "SFTRecord",
    "TrainingDataError",
    "assert_no_leakage",
    "build_prompt_messages",
    "build_records",
    "build_target",
    "prepare_training_data",
    "read_jsonl",
    "split_by_database",
    "tokenize_record",
    "tokenize_records",
    "write_jsonl",
]

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_DATA_DIR: Final = Path(__file__).resolve().parents[2] / "data" / "spider_train"
_TRAIN_SPLIT: Final = "train"
_IGNORE_INDEX: Final = -100


class TrainingDataError(RuntimeError):
    """Training data could not be built or read."""


class LeakageError(TrainingDataError):
    """Training databases overlap with the evaluation sample."""


class SFTRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    db_id: str
    question: str
    gold_sql: str
    schema_ddl: str


class DataManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    seed: int
    val_fraction: float
    source_dataset: str
    source_split: str
    databases_revision: str
    archive_sha256: str
    n_source_examples: int
    n_train: int
    n_val: int
    dropped: dict[str, int]
    train_db_ids: list[str]
    val_db_ids: list[str]
    leakage_checked_against: list[str]


# --------------------------------------------------------------------- record building
def build_records(
    rows: Iterable[Mapping[str, Any]],
    schemas: Mapping[str, str],
    guard: SQLGuard | None = None,
) -> tuple[list[SFTRecord], dict[str, int]]:
    """Turn raw Spider rows into SFT records.

    Drops (and counts) malformed rows, rows whose database has no schema, gold SQL
    rejected by the structural guard, and exact duplicates.
    """
    guard = guard or SQLGuard()
    dropped: Counter[str] = Counter()
    seen: set[tuple[str, str, str]] = set()
    records: list[SFTRecord] = []
    for row in rows:
        db_id = str(row.get("db_id") or "").strip()
        question = str(row.get("question") or "").strip()
        gold = str(row.get("query") or "").strip()
        if not (db_id and question and gold):
            dropped["malformed"] += 1
            continue
        schema = schemas.get(db_id)
        if schema is None:
            dropped["no_schema"] += 1
            continue
        key = (db_id, question, gold)
        if key in seen:
            dropped["duplicate"] += 1
            continue
        if not guard.validate(gold).is_valid:
            dropped["unsafe_or_unparsable"] += 1
            continue
        seen.add(key)
        records.append(SFTRecord(db_id=db_id, question=question, gold_sql=gold, schema_ddl=schema))
    return records, dict(dropped)


def split_by_database(
    records: Sequence[SFTRecord], val_fraction: float, seed: int
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Seeded split where whole databases go to either train or validation."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    db_ids = sorted({r.db_id for r in records})
    if val_fraction == 0.0 or len(db_ids) < 2:
        return list(records), []
    rng = random.Random(seed)
    rng.shuffle(db_ids)
    n_val = min(max(1, round(len(db_ids) * val_fraction)), len(db_ids) - 1)
    val_dbs = set(db_ids[:n_val])
    train = [r for r in records if r.db_id not in val_dbs]
    val = [r for r in records if r.db_id in val_dbs]
    return train, val


def assert_no_leakage(train_db_ids: Iterable[str], eval_db_ids: Iterable[str]) -> None:
    overlap = sorted(set(train_db_ids) & set(eval_db_ids))
    if overlap:
        raise LeakageError(
            f"Training data shares {len(overlap)} database(s) with the evaluation sample: "
            f"{', '.join(overlap)}"
        )


# ------------------------------------------------------------------------- tokenizing
class TokenizerLike(Protocol):
    def apply_chat_template(
        self, conversation: Any, *, tokenize: bool, add_generation_prompt: bool = False
    ) -> str: ...

    def __call__(self, text: str, add_special_tokens: bool = True) -> Any: ...


def build_prompt_messages(question: str, schema_ddl: str) -> list[dict[str, str]]:
    """Chat messages identical to what the agent sends at inference time."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, schema_ddl)},
    ]


def build_target(gold_sql: str) -> str:
    """Assistant reply in the format ``extract_sql`` expects."""
    return f"```sql\n{gold_sql.strip()}\n```"


def tokenize_record(
    record: SFTRecord, tokenizer: TokenizerLike, max_len: int
) -> dict[str, list[int]] | None:
    """Tokenize one record with loss on the assistant reply only.

    Returns None if the example exceeds ``max_len`` tokens. Prompt and completion
    are tokenized separately, so generation after the prompt matches training.
    """
    messages = build_prompt_messages(record.question, record.schema_ddl)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": build_target(record.gold_sql)}],
        tokenize=False,
    )
    if not full_text.startswith(prompt_text):
        raise TrainingDataError("Chat template is not prefix-stable; cannot mask the prompt")
    completion_text = full_text[len(prompt_text) :].rstrip("\n")

    prompt_ids = list(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
    completion_ids = list(tokenizer(completion_text, add_special_tokens=False)["input_ids"])
    total = len(prompt_ids) + len(completion_ids)
    if total > max_len:
        return None
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * total,
        "labels": [_IGNORE_INDEX] * len(prompt_ids) + completion_ids,
    }


def tokenize_records(
    records: Sequence[SFTRecord], tokenizer: TokenizerLike, max_len: int
) -> tuple[list[dict[str, list[int]]], int]:
    """Tokenize all records; returns ``(rows, n_dropped_too_long)``."""
    rows: list[dict[str, list[int]]] = []
    dropped = 0
    for record in records:
        row = tokenize_record(record, tokenizer, max_len)
        if row is None:
            dropped += 1
        else:
            rows.append(row)
    return rows, dropped


# ------------------------------------------------------------------------------- I/O
def write_jsonl(path: Path, records: Iterable[SFTRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r.model_dump(), ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[SFTRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrainingDataError(f"Cannot read {path}: {exc}") from exc
    records: list[SFTRecord] = []
    for lineno, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(SFTRecord.model_validate_json(line))
        except ValidationError as exc:
            raise TrainingDataError(f"{path}:{lineno}: invalid record: {exc}") from exc
    return records


# --------------------------------------------------------------------- source retrieval
def _fetch_source_rows() -> list[dict[str, Any]]:
    try:
        dataset = load_dataset(QUESTIONS_REPO, split=_TRAIN_SPLIT)
        return [dict(r) for r in dataset.select_columns(["db_id", "question", "query"])]
    except Exception as exc:  # noqa: BLE001 - `datasets` raises many unrelated error types
        raise TrainingDataError(
            f"Failed to load {QUESTIONS_REPO}[{_TRAIN_SPLIT}] from the Hugging Face Hub: {exc}"
        ) from exc


def _load_or_build_schemas(
    out_dir: Path, db_ids: Sequence[str], cache_dir: Path | None
) -> dict[str, str]:
    """DDL per database, from ``schemas.json`` if it covers ``db_ids``, else the archive."""
    cache_path = out_dir / "schemas.json"
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            cached = {}
        if isinstance(cached, dict) and set(db_ids) <= set(cached):
            logger.info("Using cached schemas from %s", cache_path)
            return {k: str(cached[k]) for k in db_ids}

    archive = download_databases_archive(cache_dir)
    schemas: dict[str, str] = {}
    try:
        with (
            zipfile.ZipFile(archive) as zf,
            tempfile.TemporaryDirectory(prefix=".spider_train_stage_", dir=out_dir) as tmp,
        ):
            index, _ = index_databases_archive(zf)
            for db_id in db_ids:
                member = index.get(db_id)
                if member is None:
                    logger.warning("Database '%s' missing from archive", db_id)
                    continue
                dest = Path(tmp) / f"{db_id}.sqlite"
                extract_archive_member(zf, member, dest)
                try:
                    schemas[db_id] = read_schema_ddl(dest)
                except SpiderDataError as exc:
                    logger.warning("Skipping '%s': %s", db_id, exc)
                finally:
                    dest.unlink(missing_ok=True)
    except (zipfile.BadZipFile, OSError) as exc:
        raise TrainingDataError(f"Cannot read database archive {archive}: {exc}") from exc

    cache_path.write_text(json.dumps(schemas, ensure_ascii=False), encoding="utf-8")
    return schemas


def _eval_db_ids(eval_data_dir: Path) -> set[str] | None:
    path = eval_data_dir / "manifest.json"
    if not path.is_file():
        return None
    try:
        return set(SampleManifest.model_validate_json(path.read_text(encoding="utf-8")).db_ids)
    except (ValidationError, OSError) as exc:
        raise TrainingDataError(f"Cannot read evaluation manifest {path}: {exc}") from exc


def prepare_training_data(
    out_dir: Path = DEFAULT_TRAIN_DATA_DIR,
    *,
    eval_data_dir: Path = DEFAULT_DATA_DIR,
    val_fraction: float = 0.05,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
) -> DataManifest:
    """Build ``train.jsonl`` / ``val.jsonl`` and return the manifest.

    Raises:
        LeakageError: training databases overlap with the evaluation sample.
        TrainingDataError: source data or archive problems.
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _fetch_source_rows()
    db_ids = sorted({str(r.get("db_id") or "").strip() for r in rows} - {""})
    schemas = _load_or_build_schemas(out_dir, db_ids, cache_dir)

    records, dropped = build_records(rows, schemas)
    if not records:
        raise TrainingDataError("No usable training records after filtering")

    eval_ids = _eval_db_ids(eval_data_dir)
    if eval_ids is None:
        logger.warning(
            "No evaluation manifest in %s; cannot verify train/eval database separation", eval_data_dir
        )
    else:
        assert_no_leakage({r.db_id for r in records}, eval_ids)

    train, val = split_by_database(records, val_fraction, seed)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    manifest = DataManifest(
        seed=seed,
        val_fraction=val_fraction,
        source_dataset=QUESTIONS_REPO,
        source_split=_TRAIN_SPLIT,
        databases_revision=DATABASES_REVISION,
        archive_sha256=DATABASES_SHA256,
        n_source_examples=len(rows),
        n_train=len(train),
        n_val=len(val),
        dropped=dropped,
        train_db_ids=sorted({r.db_id for r in train}),
        val_db_ids=sorted({r.db_id for r in val}),
        leakage_checked_against=sorted(eval_ids or []),
    )
    (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Spider SFT data for LoRA training.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR)
    parser.add_argument("--eval-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Hugging Face cache dir.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        manifest = prepare_training_data(
            args.out_dir,
            eval_data_dir=args.eval_data_dir,
            val_fraction=args.val_fraction,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    except (TrainingDataError, SpiderDataError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print(
        f"Prepared {manifest.n_train} train / {manifest.n_val} val examples "
        f"({len(manifest.train_db_ids)} train DBs, {len(manifest.val_db_ids)} val DBs) in {args.out_dir}\n"
        f"Dropped: {manifest.dropped or 'nothing'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Tests for router.data and TrainConfig (no network, no model downloads)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from datamind.prompts import SYSTEM_PROMPT, build_user_prompt, extract_sql
from router.data import (
    LeakageError,
    SFTRecord,
    TrainingDataError,
    assert_no_leakage,
    build_prompt_messages,
    build_records,
    build_target,
    read_jsonl,
    split_by_database,
    tokenize_record,
    tokenize_records,
    write_jsonl,
)
from router.train_lora import TrainConfig

DDL = "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"


class FakeTokenizer:
    """ChatML-style template + character-level 'tokens'."""

    def apply_chat_template(self, conversation, *, tokenize: bool, add_generation_prompt: bool = False) -> str:
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in conversation)
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return text

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


def decode(ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


def record(i: int = 0, db_id: str = "music", gold: str = "SELECT name FROM singer") -> SFTRecord:
    return SFTRecord(db_id=db_id, question=f"question {i}", gold_sql=gold, schema_ddl=DDL)


def test_prompt_matches_agent_prompt() -> None:
    messages = build_prompt_messages("How many singers?", DDL)
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert messages[1]["content"] == build_user_prompt("How many singers?", DDL)


def test_target_is_parseable_by_extract_sql() -> None:
    assert extract_sql(build_target("  SELECT 1  ")) == "SELECT 1"


def test_tokenize_masks_prompt_and_trains_on_reply() -> None:
    row = tokenize_record(record(), FakeTokenizer(), max_len=100_000)
    assert row is not None
    ids, labels = row["input_ids"], row["labels"]
    assert len(ids) == len(labels) == len(row["attention_mask"])
    n_masked = sum(1 for x in labels if x == -100)
    assert all(x == -100 for x in labels[:n_masked])
    assert decode(ids[n_masked:]) == "```sql\nSELECT name FROM singer\n```<|im_end|>"
    assert decode(labels[n_masked:]) == decode(ids[n_masked:])
    assert decode(ids[:n_masked]).endswith("<|im_start|>assistant\n")


def test_overlong_examples_are_dropped() -> None:
    rows, dropped = tokenize_records([record(0), record(1)], FakeTokenizer(), max_len=50)
    assert rows == [] and dropped == 2
    rows, dropped = tokenize_records([record(0)], FakeTokenizer(), max_len=100_000)
    assert len(rows) == 1 and dropped == 0


def test_non_prefix_stable_template_is_rejected() -> None:
    class Bad(FakeTokenizer):
        def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt=False):
            text = super().apply_chat_template(conversation, tokenize=tokenize)
            return ("X" + text) if add_generation_prompt else text

    with pytest.raises(TrainingDataError):
        tokenize_record(record(), Bad(), max_len=100_000)


def test_build_records_filters_and_counts() -> None:
    schemas = {"music": DDL}
    rows = [
        {"db_id": "music", "question": "ok", "query": "SELECT name FROM singer"},
        {"db_id": "music", "question": "ok", "query": "SELECT name FROM singer"},  # duplicate
        {"db_id": "music", "question": "bad", "query": "DELETE FROM singer"},  # unsafe
        {"db_id": "music", "question": "bad2", "query": "SELECT ("},  # unparsable
        {"db_id": "other", "question": "x", "query": "SELECT 1"},  # no schema
        {"db_id": "music", "question": "", "query": "SELECT 1"},  # malformed
        {"db_id": "music", "question": 'dq', "query": 'SELECT name FROM singer WHERE name = "A"'},
    ]
    records, dropped = build_records(rows, schemas)
    assert [r.question for r in records] == ["ok", "dq"]
    assert dropped == {"duplicate": 1, "unsafe_or_unparsable": 2, "no_schema": 1, "malformed": 1}


def test_split_by_database_is_disjoint_and_deterministic() -> None:
    records = [record(i, db_id=f"db{i % 10}") for i in range(50)]
    train, val = split_by_database(records, 0.2, seed=1)
    assert {r.db_id for r in train}.isdisjoint({r.db_id for r in val})
    assert len(train) + len(val) == 50
    assert len({r.db_id for r in val}) == 2
    assert split_by_database(records, 0.2, seed=1) == (train, val)


def test_split_edge_cases() -> None:
    records = [record(i, db_id="only") for i in range(5)]
    assert split_by_database(records, 0.5, seed=1) == (records, [])  # single DB: no val
    assert split_by_database([record(0, "a"), record(1, "b")], 0.0, seed=1)[1] == []
    with pytest.raises(ValueError):
        split_by_database(records, 1.0, seed=1)
    # Even a tiny fraction yields one validation DB, and never empties train.
    train, val = split_by_database([record(0, "a"), record(1, "b")], 0.9, seed=1)
    assert len(train) == 1 and len(val) == 1


def test_assert_no_leakage() -> None:
    assert_no_leakage({"a", "b"}, {"c"})
    with pytest.raises(LeakageError, match="b"):
        assert_no_leakage({"a", "b"}, {"b", "c"})


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [record(0), record(1, gold='SELECT "é" FROM singer')]
    write_jsonl(tmp_path / "x.jsonl", records)
    assert read_jsonl(tmp_path / "x.jsonl") == records


def test_read_jsonl_errors(tmp_path: Path) -> None:
    with pytest.raises(TrainingDataError):
        read_jsonl(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"db_id": "x"}\n', encoding="utf-8")
    with pytest.raises(TrainingDataError, match="bad.jsonl:1"):
        read_jsonl(bad)


@pytest.mark.parametrize(
    "overrides",
    [{"epochs": 0}, {"lora_r": 0}, {"max_seq_len": 8}, {"learning_rate": -1.0}, {"lora_dropout": 1.0}],
)
def test_train_config_validation(overrides: dict, tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(output_dir=tmp_path, **overrides)


def test_train_config_defaults(tmp_path: Path) -> None:
    cfg = TrainConfig(output_dir=tmp_path)
    assert cfg.base_model == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert cfg.per_device_batch_size * cfg.grad_accum_steps == 32

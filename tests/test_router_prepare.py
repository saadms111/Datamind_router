"""End-to-end tests for router.data.prepare_training_data (no network).

The Hugging Face fetch is monkeypatched and the schema cache is pre-seeded, so
only the orchestration is exercised: filtering, database-level split, leakage
refusal against the evaluation manifest, and the files written.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from router import data as router_data
from router.data import LeakageError, TrainingDataError, prepare_training_data, read_jsonl
from router.train_lora import LORA_TARGET_MODULES

DDL = "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER);"
N_DBS = 10


def _eval_manifest(directory: Path, db_ids: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_size": 50,
        "seed": 13,
        "example_count": 50,
        "questions_dataset": "xlangai/spider",
        "questions_split": "validation",
        "databases_repo": "x/y",
        "databases_revision": "r",
        "archive_sha256": "s",
        "db_ids": db_ids,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def out_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    rows = [
        {"db_id": f"db{d}", "question": f"question {d}-{i}", "query": f"SELECT name FROM singer LIMIT {i + 1}"}
        for d in range(N_DBS)
        for i in range(3)
    ]
    monkeypatch.setattr(router_data, "_fetch_source_rows", lambda: rows)
    path = tmp_path / "train"
    path.mkdir()
    (path / "schemas.json").write_text(json.dumps({f"db{d}": DDL for d in range(N_DBS)}), encoding="utf-8")
    return path


def test_writes_outputs_and_manifest(out_dir: Path, tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _eval_manifest(eval_dir, ["dev_only_db"])
    manifest = prepare_training_data(out_dir, eval_data_dir=eval_dir, val_fraction=0.2, seed=3)

    assert manifest.n_source_examples == 30
    assert manifest.n_train + manifest.n_val == 30 and manifest.n_val > 0
    assert set(manifest.train_db_ids).isdisjoint(manifest.val_db_ids)
    assert manifest.leakage_checked_against == ["dev_only_db"]
    assert len(read_jsonl(out_dir / "train.jsonl")) == manifest.n_train
    assert len(read_jsonl(out_dir / "val.jsonl")) == manifest.n_val
    on_disk = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk["n_train"] == manifest.n_train and on_disk["seed"] == 3


def test_refuses_overlap_with_eval_databases(out_dir: Path, tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    _eval_manifest(eval_dir, ["db3"])
    with pytest.raises(LeakageError, match="db3"):
        prepare_training_data(out_dir, eval_data_dir=eval_dir)
    assert not (out_dir / "train.jsonl").exists()  # nothing written on refusal


def test_runs_without_eval_manifest(out_dir: Path, tmp_path: Path) -> None:
    manifest = prepare_training_data(out_dir, eval_data_dir=tmp_path / "no_eval")
    assert manifest.leakage_checked_against == []


def test_errors_when_nothing_survives_filtering(
    monkeypatch: pytest.MonkeyPatch, out_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        router_data, "_fetch_source_rows", lambda: [{"db_id": "db0", "question": "q", "query": "DROP TABLE x"}]
    )
    with pytest.raises(TrainingDataError, match="No usable"):
        prepare_training_data(out_dir, eval_data_dir=tmp_path / "no_eval")


def test_lora_targets_cover_attention_and_mlp() -> None:
    assert set(LORA_TARGET_MODULES) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }


def test_train_lora_warmup_steps_replace_removed_warmup_ratio() -> None:
    from router.train_lora import TrainConfig, warmup_steps_for

    cfg = TrainConfig(output_dir=Path("out"))
    assert warmup_steps_for(cfg, 6657) == 13  # ceil(6657/32)=209 steps/epoch * 2 epochs * 3%
    assert warmup_steps_for(TrainConfig(output_dir=Path("out"), max_steps=5), 200) == 1
    assert warmup_steps_for(TrainConfig(output_dir=Path("out"), warmup_ratio=0.0), 6657) == 0


def test_train_lora_drops_unsupported_training_arguments(caplog: pytest.LogCaptureFixture) -> None:
    import dataclasses

    from router.train_lora import supported_kwargs

    @dataclasses.dataclass
    class FakeArgs:
        learning_rate: float = 0.1

    with caplog.at_level("WARNING"):
        kept = supported_kwargs(FakeArgs, {"learning_rate": 0.2, "warmup_ratio": 0.03})
    assert kept == {"learning_rate": 0.2}
    assert "warmup_ratio" in caplog.text

"""LoRA fine-tuning of a small Qwen2.5-Coder model on Spider text-to-SQL.

Uses plain ``transformers.Trainer`` + ``peft`` (no extra training framework).
Loss is computed on the assistant reply only (prompt tokens are masked).

Prerequisite: ``python -m router.data`` to build ``data/spider_train/``.

Example::

    PYTHONPATH=src python -m router.train_lora \\
        --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --output-dir outputs/qwen-coder-1.5b-spider

    # Quick smoke test (a few steps on a few hundred examples)
    PYTHONPATH=src python -m router.train_lora --max-steps 5 --limit-train 200 \\
        --output-dir outputs/smoke

Resume: pass ``--resume`` to continue from the latest checkpoint after a
disconnect (keep ``--output-dir`` on persistent storage such as Google Drive).

Hardware: a CUDA GPU is effectively required. As a rough guide the 1.5B model
with gradient checkpointing and bf16 fits a ~16 GB card; 3B wants ~24 GB. On CPU
this will run but be impractically slow (a warning is logged).

Outputs under ``--output-dir``: ``adapter/`` (LoRA weights + tokenizer),
``train_summary.json`` and ``checkpoints/``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from router.data import (
    DEFAULT_TRAIN_DATA_DIR,
    TrainingDataError,
    read_jsonl,
    tokenize_records,
)

__all__ = ["LORA_TARGET_MODULES", "TrainConfig", "main", "train"]

logger = logging.getLogger(__name__)

LORA_TARGET_MODULES: Final = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


class TrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    data_dir: Path = DEFAULT_TRAIN_DATA_DIR
    output_dir: Path
    max_seq_len: int = Field(default=2048, ge=64)
    epochs: float = Field(default=2.0, gt=0)
    max_steps: int = Field(default=-1, ge=-1)
    per_device_batch_size: int = Field(default=4, ge=1)
    grad_accum_steps: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    eval_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    gradient_checkpointing: bool = True
    limit_train: int | None = Field(default=None, ge=1)
    seed: int = 13
    resume: bool = False


def warmup_steps_for(cfg: TrainConfig, n_train_examples: int) -> int:
    """Warmup length in optimizer steps.

    Recent transformers releases dropped ``warmup_ratio`` in favour of ``warmup_steps``,
    which every release accepts as an integer, so the ratio is converted here instead of
    depending on the installed version. Assumes a single device (the Colab T4 case).
    """
    if cfg.max_steps > 0:
        total_steps = cfg.max_steps
    else:
        batch = cfg.per_device_batch_size * cfg.grad_accum_steps
        total_steps = math.ceil(math.ceil(n_train_examples / batch) * cfg.epochs)
    return max(1, round(total_steps * cfg.warmup_ratio)) if cfg.warmup_ratio > 0 else 0


def supported_kwargs(cls: type, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Drop (and log) keyword arguments the installed ``cls`` dataclass does not define."""
    valid = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(kwargs) - valid)
    if unknown:
        logger.warning("Ignoring %s arguments not supported by the installed version: %s", cls.__name__, unknown)
    return {k: v for k, v in kwargs.items() if k in valid}


def train(cfg: TrainConfig) -> Path:
    """Run LoRA fine-tuning; returns the adapter directory.

    Raises:
        TrainingDataError: prepared data is missing or empty.
    """
    # Heavy imports live here so TrainConfig and the CLI parser stay cheap to import.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.seed)
    train_path = cfg.data_dir / "train.jsonl"
    val_path = cfg.data_dir / "val.jsonl"
    if not train_path.is_file():
        raise TrainingDataError(f"{train_path} not found; run `python -m router.data` first")
    train_records = read_jsonl(train_path)
    if cfg.limit_train is not None:
        train_records = train_records[: cfg.limit_train]
    val_records = read_jsonl(val_path) if val_path.is_file() else []

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_rows, dropped_train = tokenize_records(train_records, tokenizer, cfg.max_seq_len)
    val_rows, dropped_val = tokenize_records(val_records, tokenizer, cfg.max_seq_len)
    if not train_rows:
        raise TrainingDataError("No training examples fit within max_seq_len")
    logger.info(
        "Train examples: %d (dropped %d too long); val examples: %d (dropped %d)",
        len(train_rows), dropped_train, len(val_rows), dropped_val,
    )

    if torch.cuda.is_available():
        # Native bf16 needs compute capability >= 8.0 (Ampere+). Newer PyTorch reports
        # bf16 as "supported" on older GPUs such as the T4 via slow emulation, so
        # ask the hardware instead and fall back to fp16 there.
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        use_fp16 = not use_bf16
    else:
        logger.warning("No CUDA device found; training on CPU will be extremely slow")
        dtype, use_bf16, use_fp16 = torch.float32, False, False

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, torch_dtype=dtype)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(LORA_TARGET_MODULES),
        ),
    )
    model.print_trainable_parameters()

    args = TrainingArguments(
        **supported_kwargs(
            TrainingArguments,
            dict(
                output_dir=str(cfg.output_dir / "checkpoints"),
                num_train_epochs=cfg.epochs,
                max_steps=cfg.max_steps,
                per_device_train_batch_size=cfg.per_device_batch_size,
                per_device_eval_batch_size=cfg.per_device_batch_size,
                gradient_accumulation_steps=cfg.grad_accum_steps,
                learning_rate=cfg.learning_rate,
                lr_scheduler_type="cosine",
                warmup_steps=warmup_steps_for(cfg, len(train_rows)),
                weight_decay=cfg.weight_decay,
                logging_steps=cfg.logging_steps,
                eval_strategy="steps" if val_rows else "no",
                eval_steps=cfg.eval_steps,
                save_strategy="steps",
                save_steps=cfg.save_steps,
                save_total_limit=2,
                bf16=use_bf16,
                fp16=use_fp16,
                gradient_checkpointing=cfg.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
                report_to="none",
                seed=cfg.seed,
                remove_unused_columns=False,
            ),
        )
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(val_rows) if val_rows else None,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8
        ),
        processing_class=tokenizer,
    )
    checkpoint_root = cfg.output_dir / "checkpoints"
    resume = cfg.resume and any(checkpoint_root.glob("checkpoint-*"))
    if cfg.resume and not resume:
        logger.warning("--resume given but no checkpoint found in %s; starting fresh", checkpoint_root)
    trainer.train(resume_from_checkpoint=True if resume else None)
    eval_metrics = trainer.evaluate() if val_rows else {}

    adapter_dir = cfg.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    summary = {
        "config": cfg.model_dump(mode="json"),
        "n_train_examples": len(train_rows),
        "n_train_dropped_too_long": dropped_train,
        "n_val_examples": len(val_rows),
        "n_val_dropped_too_long": dropped_val,
        "final_eval": eval_metrics,
    }
    (cfg.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Adapter saved to %s", adapter_dir)
    return adapter_dir


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LoRA fine-tune a Qwen2.5-Coder model on Spider.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN_DATA_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--max-seq-len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=4, dest="per_device_batch_size")
    p.add_argument("--grad-accum", type=int, default=8, dest="grad_accum_steps")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--limit-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint in --output-dir")
    p.add_argument("-v", "--verbose", action="store_true")
    ns = vars(p.parse_args(argv))
    verbose = ns.pop("verbose")
    ns["gradient_checkpointing"] = not ns.pop("no_gradient_checkpointing")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        cfg = TrainConfig(**ns)
        train(cfg)
    except (ValidationError, TrainingDataError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

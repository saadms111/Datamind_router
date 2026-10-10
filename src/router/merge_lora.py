"""Merge a LoRA adapter into its base model for simple, fast vLLM serving.

Example::

    PYTHONPATH=src python -m router.merge_lora \\
        --base-model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --adapter-dir outputs/qwen-coder-1.5b-spider/adapter \\
        --output-dir outputs/qwen-coder-1.5b-spider/merged

vLLM can also serve an unmerged adapter (``--enable-lora``), but a merged
checkpoint avoids per-request adapter overhead and works with any server.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["main", "merge_adapter"]

logger = logging.getLogger(__name__)

_DTYPES = ("bfloat16", "float16", "float32")


def merge_adapter(base_model: str, adapter_dir: Path, output_dir: Path, dtype: str = "bfloat16") -> Path:
    """Load the base model, apply the adapter, merge, and save to ``output_dir``."""
    if dtype not in _DTYPES:
        raise ValueError(f"dtype must be one of {_DTYPES}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=getattr(torch, dtype))
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(output_dir))
    logger.info("Merged model saved to %s", output_dir)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge a LoRA adapter into its base model.")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dtype", choices=_DTYPES, default="bfloat16")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        merge_adapter(args.base_model, args.adapter_dir, args.output_dir, args.dtype)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

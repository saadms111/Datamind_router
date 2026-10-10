"""CLI: evaluate DataMind on the Spider sample and write a report.

Backends:

* ``claude``  - Claude only (the "before routing" baseline). Needs ANTHROPIC_API_KEY.
* ``local``   - the fine-tuned model on a vLLM server (``--vllm-url``).
* ``routed``  - local first, escalate to Claude (needs both).
* ``oracle``  - answers with the gold SQL; must score 100% (harness self-check).

Examples::

    PYTHONPATH=src python -m eval.run_eval --backend oracle --run-name oracle_check
    PYTHONPATH=src python -m eval.run_eval --backend claude --run-name claude_baseline

    # Local model on vLLM; --gpu-hourly-usd is mandatory so the cost basis is explicit
    PYTHONPATH=src python -m eval.run_eval --backend local --gpu-hourly-usd 0.526 \\
        --run-name local_qwen15b

    PYTHONPATH=src python -m eval.run_eval --backend routed --gpu-hourly-usd 0.526 \\
        --min-confidence -0.15 --run-name routed_qwen15b

Run local/routed benchmarks on the machine hosting vLLM (e.g. the same Colab
notebook), with the default ``--concurrency 1``, so latencies are model time.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import CatalogError, DatabaseCatalog
from datamind.llm import ClaudeClient, LLMClient, VLLMClient
from datamind.pricing import GpuPricing, ModelPricing, UnknownModelPricingError, resolve_pricing
from eval.loader import (
    DEFAULT_DATA_DIR,
    DEFAULT_SAMPLE_SIZE,
    DEFAULT_SEED,
    SpiderDataError,
    load_spider_sample,
)
from eval.report import summarize, write_report
from eval.runner import ExampleResult, OracleLLM, run_eval
from router.cascade import Answerer, CascadeAgent, RoutingPolicy

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "results"
_BACKENDS = ("claude", "local", "routed", "oracle")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate DataMind on the Spider sample.")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-name", default=None)
    p.add_argument("--backend", choices=_BACKENDS, default="claude")
    p.add_argument("--model", default="claude-sonnet-5-5", help="Claude model id")
    p.add_argument("--input-price", type=float, default=None, help="Claude USD per 1M input tokens (override)")
    p.add_argument("--output-price", type=float, default=None, help="Claude USD per 1M output tokens (override)")
    p.add_argument("--max-attempts", type=int, default=2, help="LLM attempts per query for Claude")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1")
    p.add_argument("--local-model", default="datamind-qwen", help="Model name as served by vLLM")
    p.add_argument("--local-max-attempts", type=int, default=2, help="LLM attempts per query for the local model")
    p.add_argument(
        "--gpu-hourly-usd",
        type=float,
        default=None,
        help="USD per GPU-hour attributed to the local model (required for local/routed; 0 is allowed)",
    )
    p.add_argument("--gpu-concurrency", type=int, default=1, help="Requests sharing the GPU (divides cost)")
    p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Routed backend: escalate when mean token logprob is below this (<= 0, e.g. -0.15)",
    )
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--limit", type=int, default=None, help="Only evaluate the first N examples")
    p.add_argument("--size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size used when building")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _claude_pricing(args: argparse.Namespace) -> ModelPricing:
    if args.input_price is not None or args.output_price is not None:
        if args.input_price is None or args.output_price is None:
            raise ValueError("--input-price and --output-price must be given together")
        return ModelPricing(args.input_price, args.output_price)
    return resolve_pricing(args.model)


def _print_progress(done: int, total: int, result: ExampleResult) -> None:
    mark = "ok  " if result.correct else "FAIL"
    category = f" ({result.failure_category.value})" if result.failure_category else ""
    route = f" [{result.route}]" if result.route else ""
    print(f"[{done}/{total}] {mark} {result.example_id}{route}{category}", flush=True)


class _Closer:
    """Collects async clients to close at the end of a run."""

    def __init__(self) -> None:
        self._clients: list[ClaudeClient | VLLMClient] = []

    def add(self, client: ClaudeClient | VLLMClient) -> None:
        self._clients.append(client)

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()


def _build_answerer(
    args: argparse.Namespace,
    catalog: DatabaseCatalog,
    examples: Sequence[object],
    closer: _Closer,
) -> tuple[Answerer, dict[str, float | str]]:
    """Construct the answerer for ``args.backend`` plus the cost assumptions to record.

    Raises:
        ValueError: invalid or missing configuration (message is user-facing).
    """
    assumptions: dict[str, float | str] = {}

    def claude_agent() -> DataMindAgent:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ValueError("ANTHROPIC_API_KEY is not set")
        pricing = _claude_pricing(args)
        client = ClaudeClient(args.model, pricing=pricing)
        closer.add(client)
        assumptions["claude_model"] = args.model
        assumptions["claude_usd_per_mtok_in_out"] = f"{pricing.input_per_mtok} / {pricing.output_per_mtok}"
        return DataMindAgent(client, catalog, AgentConfig(max_attempts=args.max_attempts))

    def local_agent() -> DataMindAgent:
        if args.gpu_hourly_usd is None:
            raise ValueError(
                "--gpu-hourly-usd is required for local/routed runs so the cost basis is explicit "
                "(use the on-demand price of an equivalent cloud GPU; pass 0 to ignore GPU cost)"
            )
        pricing = GpuPricing(args.gpu_hourly_usd, args.gpu_concurrency)
        client = VLLMClient(args.local_model, base_url=args.vllm_url, pricing=pricing)
        closer.add(client)
        assumptions["local_model"] = args.local_model
        assumptions["gpu_hourly_usd"] = args.gpu_hourly_usd
        assumptions["gpu_concurrent_requests"] = args.gpu_concurrency
        return DataMindAgent(client, catalog, AgentConfig(max_attempts=args.local_max_attempts))

    if args.backend == "oracle":
        return DataMindAgent(OracleLLM(examples), catalog, AgentConfig(max_attempts=1)), assumptions  # type: ignore[arg-type]
    if args.backend == "claude":
        return claude_agent(), assumptions
    if args.backend == "local":
        return local_agent(), assumptions
    local = local_agent()
    remote = claude_agent()
    if args.min_confidence is not None:
        assumptions["min_confidence"] = args.min_confidence
    return CascadeAgent(local, remote, RoutingPolicy(args.min_confidence)), assumptions


async def _run(args: argparse.Namespace) -> int:
    try:
        examples = load_spider_sample(
            args.data_dir, ensure=False, sample_size=args.size, seed=args.seed
        )
        catalog = DatabaseCatalog.from_directory(args.data_dir / "databases")
    except (SpiderDataError, CatalogError, FileNotFoundError) as exc:
        logger.error("%s (build the sample first: python -m eval.loader)", exc)
        return 1
    if args.limit is not None:
        examples = examples[: args.limit]

    closer = _Closer()
    try:
        agent, assumptions = _build_answerer(args, catalog, examples, closer)
    except (UnknownModelPricingError, ValueError) as exc:
        logger.error("%s", exc)
        await closer.aclose()
        return 2

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = args.run_name or f"{args.backend}-{stamp}"
    try:
        results = await run_eval(
            agent, examples, concurrency=args.concurrency, progress=_print_progress
        )
    finally:
        await closer.aclose()

    summary = summarize(results, run_name=run_name, assumptions=assumptions)
    out_dir = args.output_dir / run_name
    write_report(out_dir, summary, results)

    print(
        f"\n{run_name}: accuracy {summary.execution_accuracy:.1%} "
        f"({summary.n_correct}/{summary.n_examples}), "
        f"p95 latency {summary.latency_ms.p95:.0f} ms, "
        f"${summary.cost_per_1k_queries_usd:.2f} per 1k queries"
    )
    if summary.accuracy_by_route:
        print("By route: " + ", ".join(f"{k} n={v.n} acc={v.accuracy:.1%}" for k, v in summary.accuracy_by_route.items()))
    print(f"Report written to {out_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

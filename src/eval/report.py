"""Aggregate per-example results into benchmark metrics and write report files."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from eval.runner import ExampleResult

__all__ = [
    "BenchmarkSummary",
    "BucketStats",
    "LatencyStats",
    "percentile",
    "render_markdown",
    "summarize",
    "write_report",
]

_MAX_FAILURES_LISTED = 25


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 100]); 0.0 for empty input."""
    if not 0.0 <= q <= 100.0:
        raise ValueError("q must be within [0, 100]")
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (q / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (rank - low) * (ordered[high] - ordered[low])


class LatencyStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    mean: float
    p50: float
    p95: float
    max: float


class BucketStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    n: int
    accuracy: float


class BenchmarkSummary(BaseModel):
    """Stable, comparable summary of one evaluation run (written to summary.json)."""

    model_config = ConfigDict(frozen=True)

    run_name: str
    created_at: str
    models: list[str]
    n_examples: int
    n_correct: int
    execution_accuracy: float
    status_counts: dict[str, int]
    failure_counts: dict[str, int]
    accuracy_by_complexity: dict[str, BucketStats]
    accuracy_by_route: dict[str, BucketStats]
    latency_ms: LatencyStats
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    cost_per_1k_queries_usd: float
    avg_llm_calls: float


def _bucket(results: Sequence[ExampleResult]) -> BucketStats:
    return BucketStats(n=len(results), accuracy=sum(r.correct for r in results) / len(results))


def summarize(
    results: Sequence[ExampleResult], *, run_name: str, created_at: str | None = None
) -> BenchmarkSummary:
    if not results:
        raise ValueError("Cannot summarize an empty result set")
    n = len(results)
    correct = sum(r.correct for r in results)

    by_complexity: dict[str, list[ExampleResult]] = defaultdict(list)
    by_route: dict[str, list[ExampleResult]] = defaultdict(list)
    for r in results:
        by_complexity[r.complexity].append(r)
        if r.route:
            by_route[r.route].append(r)

    latencies = [r.latency_ms for r in results if r.status is not None]
    total_cost = sum(r.cost_usd for r in results)
    return BenchmarkSummary(
        run_name=run_name,
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        models=sorted({r.model for r in results if r.model}),
        n_examples=n,
        n_correct=correct,
        execution_accuracy=correct / n,
        status_counts=dict(Counter(r.status.value if r.status else "harness_error" for r in results)),
        failure_counts=dict(Counter(r.failure_category.value for r in results if r.failure_category)),
        accuracy_by_complexity={k: _bucket(v) for k, v in sorted(by_complexity.items())},
        accuracy_by_route={k: _bucket(v) for k, v in sorted(by_route.items())},
        latency_ms=LatencyStats(
            mean=sum(latencies) / len(latencies) if latencies else 0.0,
            p50=percentile(latencies, 50),
            p95=percentile(latencies, 95),
            max=max(latencies, default=0.0),
        ),
        total_prompt_tokens=sum(r.prompt_tokens for r in results),
        total_completion_tokens=sum(r.completion_tokens for r in results),
        total_cost_usd=total_cost,
        cost_per_1k_queries_usd=total_cost / n * 1000.0,
        avg_llm_calls=sum(r.llm_calls for r in results) / n,
    )


def render_markdown(summary: BenchmarkSummary, results: Sequence[ExampleResult]) -> str:
    s = summary
    lines = [
        f"# Evaluation report: {s.run_name}",
        "",
        f"- Created: {s.created_at}",
        f"- Models: {', '.join(s.models) or 'n/a'}",
        f"- **Execution accuracy: {s.execution_accuracy:.1%}** ({s.n_correct}/{s.n_examples})",
        f"- Latency ms: mean {s.latency_ms.mean:.0f}, p50 {s.latency_ms.p50:.0f}, "
        f"**p95 {s.latency_ms.p95:.0f}**, max {s.latency_ms.max:.0f}",
        f"- Tokens: {s.total_prompt_tokens:,} prompt + {s.total_completion_tokens:,} completion",
        f"- Total cost: ${s.total_cost_usd:.4f}; **${s.cost_per_1k_queries_usd:.2f} per 1,000 queries**",
        f"- Average LLM calls per query: {s.avg_llm_calls:.2f}",
        "",
        "## Outcome by status",
        *(f"- {k}: {v}" for k, v in sorted(s.status_counts.items())),
        "",
        "## Failure categories",
        *(f"- {k}: {v}" for k, v in sorted(s.failure_counts.items(), key=lambda kv: -kv[1])),
        *(["- none"] if not s.failure_counts else []),
        "",
        "## Accuracy by gold-SQL complexity (heuristic buckets)",
        *(f"- {k}: {v.accuracy:.1%} (n={v.n})" for k, v in s.accuracy_by_complexity.items()),
    ]
    if s.accuracy_by_route:
        lines += [
            "",
            "## Accuracy by route",
            *(f"- {k}: {v.accuracy:.1%} (n={v.n})" for k, v in s.accuracy_by_route.items()),
        ]

    failures = [r for r in results if not r.correct]
    lines += ["", f"## Failures ({len(failures)})"]
    for r in failures[:_MAX_FAILURES_LISTED]:
        category = r.failure_category.value if r.failure_category else "unknown"
        lines += [
            "",
            f"### {r.example_id} · {r.db_id} · {category}",
            f"Question: {r.question}",
            f"Gold: `{r.gold_sql}`",
            f"Predicted: `{r.predicted_sql}`" if r.predicted_sql else "Predicted: (none)",
        ]
        if r.error:
            lines.append(f"Error: {r.error}")
        if r.feature_mismatches:
            lines.append(f"Structural differences: {', '.join(r.feature_mismatches)}")
    if len(failures) > _MAX_FAILURES_LISTED:
        lines += ["", f"... and {len(failures) - _MAX_FAILURES_LISTED} more in results.jsonl"]
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, summary: BenchmarkSummary, results: Sequence[ExampleResult]) -> None:
    """Write ``results.jsonl``, ``summary.json`` and ``report.md`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text(
        "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in results),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(render_markdown(summary, results), encoding="utf-8")

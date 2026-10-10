"""Per-token pricing used for per-query cost tracking.

Rates are USD per million tokens. Verify them against
https://platform.claude.com/docs/en/about-claude/pricing before publishing
benchmark numbers, and pass an explicit :class:`ModelPricing` to the client if
your contract rates differ.

For a self-hosted model the "price" is whatever you want to attribute to
GPU time. Use ``ModelPricing(0, 0)`` to ignore it, or convert your amortised
GPU $/hour and measured tokens/second into a $/MTok figure so the
before/after-routing cost comparison is honest.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

__all__ = [
    "DEFAULT_PRICING",
    "GpuPricing",
    "ModelPricing",
    "UnknownModelPricingError",
    "resolve_pricing",
]

_TOKENS_PER_MTOK: Final = 1_000_000


class UnknownModelPricingError(ValueError):
    """No pricing is registered for the requested model."""


@dataclass(frozen=True, slots=True)
class ModelPricing:
    input_per_mtok: float
    output_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0 or self.output_per_mtok < 0:
            raise ValueError("Pricing rates must be non-negative")

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """USD cost of one call. Token counts must be non-negative."""
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        return (
            prompt_tokens * self.input_per_mtok + completion_tokens * self.output_per_mtok
        ) / _TOKENS_PER_MTOK


@dataclass(frozen=True, slots=True)
class GpuPricing:
    """Cost of a self-hosted model, attributed by GPU time.

    A request is charged ``latency_s / 3600 * hourly_usd / concurrent_requests``.

    * ``hourly_usd`` is whatever you decide a GPU-hour is worth. A free Colab T4
      costs $0 out of pocket, so for a meaningful cost comparison use the
      on-demand price of an equivalent cloud GPU, and state it in your results.
    * ``concurrent_requests`` is how many requests share the GPU while the
      benchmark runs (1 for a sequential run, which charges the whole GPU to each
      query and so over-states cost for a busy production server).
    """

    hourly_usd: float
    concurrent_requests: int = 1

    def __post_init__(self) -> None:
        if self.hourly_usd < 0:
            raise ValueError("hourly_usd must be non-negative")
        if self.concurrent_requests < 1:
            raise ValueError("concurrent_requests must be >= 1")

    def cost(self, latency_ms: float) -> float:
        """USD attributed to one request that took ``latency_ms``."""
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        return latency_ms / 1000.0 / 3600.0 * self.hourly_usd / self.concurrent_requests


DEFAULT_PRICING: Final[Mapping[str, ModelPricing]] = MappingProxyType(
    {
        "claude-sonnet-5-5": ModelPricing(input_per_mtok=2.0, output_per_mtok=10.0),
        "claude-haiku-4-5-20251001": ModelPricing(input_per_mtok=1.0, output_per_mtok=5.0),
    }
)


def resolve_pricing(
    model: str, overrides: Mapping[str, ModelPricing] | None = None
) -> ModelPricing:
    """Look up pricing for ``model``; ``overrides`` take precedence over the defaults."""
    if overrides and model in overrides:
        return overrides[model]
    try:
        return DEFAULT_PRICING[model]
    except KeyError:
        raise UnknownModelPricingError(
            f"No pricing registered for model '{model}'. Pass an explicit ModelPricing."
        ) from None

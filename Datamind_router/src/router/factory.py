"""Build the answering backend (Claude, local vLLM, or the cascade) from settings.

Shared by the MCP server. ``eval.run_eval`` keeps its own equivalent wiring (it
also needs the oracle backend), so both stay independent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from datamind.agent import AgentConfig, DataMindAgent
from datamind.catalog import DatabaseCatalog
from datamind.llm import ClaudeClient, VLLMClient
from datamind.pricing import GpuPricing, ModelPricing, UnknownModelPricingError, resolve_pricing
from router.cascade import Answerer, CascadeAgent, RoutingPolicy

__all__ = ["BACKENDS", "BackendSettings", "Closable", "ConfigError", "build_answerer"]

BACKENDS = ("claude", "local", "routed")


class ConfigError(ValueError):
    """Invalid or incomplete backend configuration (the message is user-facing)."""


class Closable(Protocol):
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class BackendSettings:
    backend: str = "claude"
    claude_model: str = "claude-sonnet-5-5"
    claude_input_price: float | None = None  # USD per 1M tokens; both or neither
    claude_output_price: float | None = None
    claude_max_attempts: int = 2
    vllm_url: str = "http://localhost:8000/v1"
    local_model: str = "datamind-qwen"
    local_max_attempts: int = 2
    gpu_hourly_usd: float | None = None  # required for local/routed (0 is allowed)
    gpu_concurrency: int = 1
    min_confidence: float | None = None


def build_answerer(
    settings: BackendSettings, catalog: DatabaseCatalog
) -> tuple[Answerer, list[Closable]]:
    """Construct the answerer and the clients that must be closed at shutdown.

    Raises:
        ConfigError: unknown backend, missing API key or GPU rate, bad pricing.
    """
    if settings.backend not in BACKENDS:
        raise ConfigError(f"Unknown backend '{settings.backend}'; choose from {', '.join(BACKENDS)}")

    closers: list[Closable] = []

    def claude_agent() -> DataMindAgent:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise ConfigError("ANTHROPIC_API_KEY is not set")
        try:
            if settings.claude_input_price is not None or settings.claude_output_price is not None:
                if settings.claude_input_price is None or settings.claude_output_price is None:
                    raise ConfigError("Claude input and output prices must be given together")
                pricing = ModelPricing(settings.claude_input_price, settings.claude_output_price)
            else:
                pricing = resolve_pricing(settings.claude_model)
            client = ClaudeClient(settings.claude_model, pricing=pricing)
        except (UnknownModelPricingError, ValueError) as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(str(exc)) from exc
        closers.append(client)
        return DataMindAgent(client, catalog, AgentConfig(max_attempts=settings.claude_max_attempts))

    def local_agent() -> DataMindAgent:
        if settings.gpu_hourly_usd is None:
            raise ConfigError(
                "gpu_hourly_usd is required for local/routed backends so cost reporting is explicit "
                "(pass 0 to ignore GPU cost)"
            )
        try:
            pricing = GpuPricing(settings.gpu_hourly_usd, settings.gpu_concurrency)
            client = VLLMClient(settings.local_model, base_url=settings.vllm_url, pricing=pricing)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        closers.append(client)
        return DataMindAgent(client, catalog, AgentConfig(max_attempts=settings.local_max_attempts))

    if settings.backend == "claude":
        return claude_agent(), closers
    if settings.backend == "local":
        return local_agent(), closers
    try:
        policy = RoutingPolicy(settings.min_confidence)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    return CascadeAgent(local_agent(), claude_agent(), policy), closers

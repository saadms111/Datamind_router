"""Cascade router: try the local fine-tuned model first, escalate to Claude.

``CascadeAgent`` wraps two answerers (normally two :class:`DataMindAgent`
instances, one per backend) and decides per query:

1. ``route_preference`` ``local`` / ``claude`` forces that backend.
2. Otherwise the local agent answers first. Its answer is **accepted** only if
   the query succeeded (the SQL passed the AST guard and executed) *and*, when
   ``RoutingPolicy.min_confidence`` is set, the mean token log-probability of the
   generation is at least that threshold.
3. Anything else escalates to Claude. The returned response is Claude's, but its
   metrics include the wasted local attempt (tokens, GPU cost, latency, calls),
   so an escalation is honestly more expensive and slower than going straight to
   Claude.

Limits worth knowing: the guard and executor catch SQL that is invalid or unsafe,
not SQL that is valid but answers the wrong question. Only the confidence
threshold can catch some of those, and only probabilistically, so routing can
cost accuracy relative to always using Claude. Measure it; that is what the
benchmark is for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from datamind.schema import QueryRequest, QueryResponse, QueryStatus, RoutePreference

__all__ = ["Answerer", "CascadeAgent", "RoutingPolicy"]

logger = logging.getLogger(__name__)


class Answerer(Protocol):
    async def answer(self, request: QueryRequest) -> QueryResponse: ...


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """When to trust the local model's answer.

    Attributes:
        min_confidence: minimum mean token log-probability (a value <= 0; closer
            to 0 is more confident) required to accept a local answer. ``None``
            accepts every local answer that executed successfully. A local
            answer from a backend that reports no confidence is accepted.
    """

    min_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.min_confidence is not None and self.min_confidence > 0:
            raise ValueError("min_confidence is a log-probability and must be <= 0")

    def accepts(self, response: QueryResponse) -> bool:
        if response.status is not QueryStatus.SUCCESS:
            return False
        confidence = response.metrics.confidence
        if self.min_confidence is None or confidence is None:
            return True
        return confidence >= self.min_confidence


class CascadeAgent:
    """Routes each query between a local and a remote answerer."""

    def __init__(self, local: Answerer, remote: Answerer, policy: RoutingPolicy | None = None) -> None:
        self._local = local
        self._remote = remote
        self._policy = policy or RoutingPolicy()

    async def answer(self, request: QueryRequest) -> QueryResponse:
        """Answer ``request``; see the module docstring for the routing rules.

        Raises whatever the underlying agents raise for unknown databases.
        """
        if request.route_preference is RoutePreference.CLAUDE:
            return await self._remote.answer(request)
        if request.route_preference is RoutePreference.LOCAL:
            return await self._local.answer(request)

        local_response = await self._local.answer(request)
        if self._policy.accepts(local_response):
            return local_response

        logger.info(
            "escalating request_id=%s local_status=%s confidence=%s",
            request.request_id,
            local_response.status.value,
            local_response.metrics.confidence,
        )
        remote_response = await self._remote.answer(request)
        merged = local_response.metrics.merge(remote_response.metrics)
        return remote_response.model_copy(update={"metrics": merged})

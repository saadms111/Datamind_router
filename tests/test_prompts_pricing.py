"""Tests for prompt parsing and pricing."""

from __future__ import annotations

import pytest

from datamind.pricing import ModelPricing, UnknownModelPricingError, resolve_pricing
from datamind.prompts import extract_sql


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("Here you go:\n```sql\nSELECT name FROM t;\n```\nHope that helps", "SELECT name FROM t;"),
        ("```\nSELECT 2\n```", "SELECT 2"),
        ("```sql\nSELECT 3", "SELECT 3"),  # unterminated (truncated) fence
        ("SELECT 4 FROM t", "SELECT 4 FROM t"),  # bare query
        ("WITH a AS (SELECT 1) SELECT * FROM a", "WITH a AS (SELECT 1) SELECT * FROM a"),
        ("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```", "SELECT 1"),  # first block wins
    ],
)
def test_extract_sql(reply: str, expected: str) -> None:
    assert extract_sql(reply) == expected


@pytest.mark.parametrize("reply", ["I cannot help with that.", "", "```sql\n```", "   "])
def test_extract_sql_none(reply: str) -> None:
    assert extract_sql(reply) is None


def test_cost_arithmetic() -> None:
    pricing = ModelPricing(input_per_mtok=2.0, output_per_mtok=10.0)
    assert pricing.cost(1_000_000, 500_000) == pytest.approx(7.0)
    assert pricing.cost(0, 0) == 0.0


def test_invalid_pricing_and_tokens() -> None:
    with pytest.raises(ValueError):
        ModelPricing(-1.0, 1.0)
    with pytest.raises(ValueError):
        ModelPricing(1.0, 1.0).cost(-1, 0)


def test_resolve_pricing() -> None:
    assert resolve_pricing("claude-sonnet-5-5").input_per_mtok == 2.0
    custom = ModelPricing(9.0, 9.0)
    assert resolve_pricing("claude-sonnet-5-5", {"claude-sonnet-5-5": custom}) is custom
    with pytest.raises(UnknownModelPricingError):
        resolve_pricing("no-such-model")

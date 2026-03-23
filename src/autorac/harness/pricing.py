"""Model pricing helpers for eval and trace analysis."""

from __future__ import annotations

from dataclasses import dataclass

from .encoding_db import TokenUsage


@dataclass(frozen=True)
class ModelPricing:
    """Per-million token rates for a model family."""

    input_per_million: float
    output_per_million: float
    cache_read_per_million: float = 0.0
    cache_create_per_million: float = 0.0


@dataclass(frozen=True)
class UsageCostBreakdown:
    """Token-cost breakdown for a single model invocation."""

    non_cached_input_tokens: int
    input_cost_usd: float
    cache_read_cost_usd: float
    cache_creation_cost_usd: float
    output_cost_usd: float
    reasoning_output_cost_usd: float
    total_cost_usd: float


_MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-opus-4-6": ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_read_per_million=0.50,
        cache_create_per_million=6.25,
    ),
    "opus": ModelPricing(
        input_per_million=5.0,
        output_per_million=25.0,
        cache_read_per_million=0.50,
        cache_create_per_million=6.25,
    ),
    "gpt-5.4": ModelPricing(
        input_per_million=2.5,
        output_per_million=15.0,
        cache_read_per_million=0.25,
    ),
}


def get_model_pricing(model: str) -> ModelPricing | None:
    """Return pricing metadata for a supported model string."""
    normalized = (model or "").strip()
    if not normalized:
        return None

    if normalized in _MODEL_PRICING:
        return _MODEL_PRICING[normalized]

    for prefix, pricing in _MODEL_PRICING.items():
        if normalized.startswith(prefix):
            return pricing

    return None


def estimate_usage_cost_usd(model: str, usage: TokenUsage | None) -> float | None:
    """Estimate cost for a usage bundle if model pricing is known."""
    breakdown = estimate_usage_cost_breakdown(model, usage)
    if breakdown is None:
        return None
    return breakdown.total_cost_usd


def estimate_usage_cost_breakdown(
    model: str,
    usage: TokenUsage | None,
) -> UsageCostBreakdown | None:
    """Estimate a detailed cost breakdown for a usage bundle."""
    if usage is None:
        return None

    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    non_cached_input_tokens = max(
        usage.input_tokens - usage.cache_read_tokens - usage.cache_creation_tokens,
        0,
    )
    input_cost_usd = (
        non_cached_input_tokens * pricing.input_per_million / 1_000_000
    )
    cache_read_cost_usd = (
        usage.cache_read_tokens * pricing.cache_read_per_million / 1_000_000
    )
    cache_creation_cost_usd = (
        usage.cache_creation_tokens * pricing.cache_create_per_million / 1_000_000
    )
    output_cost_usd = usage.output_tokens * pricing.output_per_million / 1_000_000
    reasoning_output_cost_usd = (
        usage.reasoning_output_tokens * pricing.output_per_million / 1_000_000
    )

    return UsageCostBreakdown(
        non_cached_input_tokens=non_cached_input_tokens,
        input_cost_usd=input_cost_usd,
        cache_read_cost_usd=cache_read_cost_usd,
        cache_creation_cost_usd=cache_creation_cost_usd,
        output_cost_usd=output_cost_usd,
        reasoning_output_cost_usd=reasoning_output_cost_usd,
        total_cost_usd=(
            input_cost_usd
            + cache_read_cost_usd
            + cache_creation_cost_usd
            + output_cost_usd
        ),
    )

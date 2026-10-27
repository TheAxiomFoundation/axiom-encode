"""Smoke tests for the TOML-backed pricing rate loader."""

from __future__ import annotations

import pytest

from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.pricing import (
    ANTHROPIC_BILLED_PRE_OUTPUT_REFUSAL_CATEGORIES,
    ANTHROPIC_FREE_PRE_OUTPUT_REFUSAL_CATEGORIES,
    ModelPricing,
    PricingRates,
    _load_pricing_rates,
    anthropic_refusal_is_billed,
    estimate_usage_cost_breakdown,
    estimate_usage_cost_usd,
    get_model_pricing,
    get_pricing_rates,
)


def test_pricing_rates_load_has_expected_shape():
    rates = get_pricing_rates()
    assert isinstance(rates, PricingRates)
    assert rates.version >= 1
    assert rates.effective_date  # non-empty string
    assert rates.models, "pricing_rates.toml should declare at least one model"

    for name, pricing in rates.models.items():
        assert isinstance(name, str) and name
        assert isinstance(pricing, ModelPricing)
        assert pricing.input_per_million >= 0.0
        assert pricing.output_per_million >= 0.0
        assert pricing.cache_read_per_million >= 0.0
        assert pricing.cache_create_per_million >= 0.0
        assert pricing.max_input_tokens is None or pricing.max_input_tokens > 0


def test_known_models_resolve_via_public_api():
    # The public API must keep working after the TOML extraction.
    opus = get_model_pricing("opus")
    assert opus is not None
    assert opus.input_per_million > 0

    # Prefix matching is part of the existing contract.
    extended = get_model_pricing("claude-opus-4-6-some-variant")
    assert extended is not None

    terra = get_model_pricing("gpt-5.6-terra")
    sol = get_model_pricing("gpt-5.6-sol")
    base_alias = get_model_pricing("gpt-5.6")
    assert (
        terra.input_per_million,
        terra.output_per_million,
        terra.cache_read_per_million,
        terra.cache_create_per_million,
        terra.max_input_tokens,
    ) == (2.0, 12.0, 0.20, 2.50, 272000)
    assert (
        sol.input_per_million,
        sol.output_per_million,
        sol.cache_read_per_million,
        sol.cache_create_per_million,
        sol.max_input_tokens,
    ) == (4.0, 20.0, 0.40, 5.0, 272000)
    assert base_alias == sol
    # Every GPT-5.6 rate traces to the vendor page it was read from, on a date.
    for pricing in (terra, sol):
        assert pricing.source_url and pricing.source_url.startswith(
            "https://developers.openai.com/"
        )
        assert pricing.captured_at == "2026-09-10"
    assert sol.promotional_until == "2026-11-21"
    assert get_model_pricing("gpt-5.6-luna") is None


def test_gpt_6_encoder_pair_rates_trace_to_vendor_pages():
    luna = get_model_pricing("gpt-6-luna")
    sol = get_model_pricing("gpt-6-sol")
    assert (
        luna.input_per_million,
        luna.output_per_million,
        luna.cache_read_per_million,
        luna.cache_create_per_million,
        luna.max_input_tokens,
    ) == (0.10, 0.50, 0.01, 0.125, 272000)
    assert (
        sol.input_per_million,
        sol.output_per_million,
        sol.cache_read_per_million,
        sol.cache_create_per_million,
        sol.max_input_tokens,
    ) == (2.0, 10.0, 0.20, 2.50, 272000)
    for model, pricing in (("gpt-6-luna", luna), ("gpt-6-sol", sol)):
        assert pricing.source_url == (
            f"https://developers.openai.com/api/docs/models/{model}"
        )
        assert pricing.captured_at == "2026-09-24"
        assert pricing.promotional_until is None
    # Variant boundary: lexical siblings never inherit GPT-6 Sol pricing.
    assert get_model_pricing("gpt-6-solstice") is None
    assert get_model_pricing("gpt-6") is None


def test_claude_5_5_rates_trace_to_vendor_page():
    opus = get_model_pricing("claude-opus-5-5")
    sonnet = get_model_pricing("claude-sonnet-5-5")
    assert (
        opus.input_per_million,
        opus.output_per_million,
        opus.cache_read_per_million,
        opus.cache_create_per_million,
        opus.max_input_tokens,
    ) == (4.0, 20.0, 0.20, 5.0, None)
    assert (
        sonnet.input_per_million,
        sonnet.output_per_million,
        sonnet.cache_read_per_million,
        sonnet.cache_create_per_million,
        sonnet.max_input_tokens,
    ) == (2.0, 10.0, 0.20, 2.50, None)
    for pricing in (opus, sonnet):
        assert pricing.source_url == (
            "https://platform.claude.com/docs/en/about-claude/pricing"
        )
        assert pricing.captured_at == "2026-09-28"
        assert pricing.promotional_until is None
    # Variant boundary: lexical lookalikes never inherit the 5.5 rates.
    for lookalike in ("claude-opus-5-50", "claude-opus-5-5x", "claude-sonnet-5-50"):
        assert get_model_pricing(lookalike) is None, lookalike
    # One million output tokens on Opus 5.5 costs the published $20.
    assert (
        estimate_usage_cost_usd(
            "claude-opus-5-5", TokenUsage(input_tokens=0, output_tokens=1_000_000)
        )
        == 20.0
    )


def test_anthropic_refusal_billing_matches_vendor_table():
    # "Billed before any output" column of
    # platform.claude.com/docs/en/build-with-claude/refusals-and-fallback,
    # read 2026-10-05.
    billed = {"bio", "frontier_llm", "reasoning_extraction"}
    free = {"cyber", "general_harms"}
    assert ANTHROPIC_BILLED_PRE_OUTPUT_REFUSAL_CATEGORIES == billed
    assert ANTHROPIC_FREE_PRE_OUTPUT_REFUSAL_CATEGORIES == free
    # Before any output: billed and free by the table, a null category free.
    for category in billed:
        assert anthropic_refusal_is_billed(category, 0) is True, category
    for category in [*free, None]:
        assert anthropic_refusal_is_billed(category, 0) is False, category
    # A category the table did not list, even a near miss, has an unknown bill.
    unlisted = ["", "Bio", "chem", "a_future_category"]
    for category in unlisted:
        assert anthropic_refusal_is_billed(category, 0) is None, category
    # After any output every refusal bills the input and that output.
    for category in [*billed, *free, None, *unlisted]:
        for output_tokens in (1, 50, 64_000):
            assert anthropic_refusal_is_billed(category, output_tokens) is True


def test_prefix_fallback_requires_variant_boundary():
    # Dash-suffixed variants keep inheriting their family's pricing...
    assert get_model_pricing("gpt-5.6-terra-2") == get_model_pricing("gpt-5.6-terra")
    # ...but lexical siblings that merely share leading characters do not.
    assert get_model_pricing("gpt-5.6-solstice") is None
    assert get_model_pricing("gpt-5.6-terra2") is None


def test_gpt_6_1_sol_proxy_rates_record_unverified_local_provenance():
    pricing = get_model_pricing("gpt-6.1-sol")
    assert pricing is not None
    assert (
        pricing.input_per_million,
        pricing.output_per_million,
        pricing.cache_read_per_million,
        pricing.cache_create_per_million,
        pricing.max_input_tokens,
    ) == (2.0, 10.0, 0.20, 2.50, 272000)
    assert pricing.source_url == (
        "UNVERIFIED: src/axiom_encode/harness/pricing_rates.toml:40-47"
        "@3bf5a2afbff0 (gpt-6-sol proxy)"
    )
    assert pricing.captured_at == "2026-10-06"
    assert pricing.promotional_until is None
    assert get_pricing_rates().version >= 5
    assert get_pricing_rates().effective_date >= "2026-10-06"


def test_gpt_6_1_sol_proxy_variant_matching_requires_boundary():
    pricing = get_model_pricing("gpt-6.1-sol")
    assert pricing is not None
    assert get_model_pricing("gpt-6.1-sol-fast") == pricing
    assert get_model_pricing("gpt-6.1-solstice") is None
    assert get_model_pricing("gpt-6.1-sol2") is None
    assert get_model_pricing("gpt-6.1") is None


@pytest.mark.parametrize("model", ["gpt-6.1-sol", "gpt-6.1-sol-fast"])
@pytest.mark.parametrize("input_tokens", [1000, 272000, 272001])
def test_gpt_6_1_sol_proxy_estimators_refuse_unverified_prices(model, input_tokens):
    usage = TokenUsage(
        input_tokens=input_tokens,
        output_tokens=100,
        cache_read_tokens=500,
        cache_creation_tokens=100,
    )

    assert estimate_usage_cost_usd(model, usage) is None
    assert estimate_usage_cost_breakdown(model, usage) is None


@pytest.mark.parametrize("input_tokens", [1000, 272001])
def test_aggregated_usage_cannot_bypass_unverified_pricing(input_tokens):
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=100)

    assert (
        estimate_usage_cost_breakdown("gpt-6.1-sol", usage, enforce_context_tier=False)
        is None
    )


def test_context_tier_gate_can_be_skipped_for_aggregated_usage():
    over_boundary = TokenUsage(input_tokens=272001, output_tokens=100)

    assert estimate_usage_cost_breakdown("gpt-5.6-terra", over_boundary) is None
    breakdown = estimate_usage_cost_breakdown(
        "gpt-5.6-terra",
        over_boundary,
        enforce_context_tier=False,
    )
    assert breakdown is not None
    assert breakdown.total_cost_usd > 0


def test_gpt_5_6_standard_pricing_fails_closed_above_short_context_tier():
    at_boundary = TokenUsage(input_tokens=272000, output_tokens=100)
    over_boundary = TokenUsage(input_tokens=272001, output_tokens=100)

    assert estimate_usage_cost_usd("gpt-5.6-terra", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6-sol", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6", at_boundary) is not None
    assert estimate_usage_cost_usd("gpt-5.6-terra", over_boundary) is None
    assert estimate_usage_cost_usd("gpt-5.6-sol", over_boundary) is None
    assert estimate_usage_cost_usd("gpt-5.6", over_boundary) is None


def test_load_pricing_rates_is_reparseable():
    # _load_pricing_rates bypasses the cache, so calling twice must still work
    # and produce structurally identical output.
    first = _load_pricing_rates()
    second = _load_pricing_rates()
    assert first.version == second.version
    assert first.effective_date == second.effective_date
    assert set(first.models) == set(second.models)

"""Guidance identities cannot activate inherited manual typography masks."""

from __future__ import annotations

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

GUIDANCE_PAGE = "us-md/guidance/dhs/fia/snap-manual-214/page-5"


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "The following table lists benefit amounts in dollars.\n"
            "214.50 Monthly Allowance\n"
            "All households receive the listed allowance.",
            {214.5},
        ),
        ("Applications filed in (07/2026) are eligible.", {7, 2026}),
        ("Applicable interest rates, in percent:\n7.5 Annual Rate", {7.5}),
        (
            "The following table lists the eligible child counts.\n"
            "1-2 Eligible Children",
            {1, 2},
        ),
    ),
    ids=("amount", "date", "rate", "child-count-range"),
)
def test_guidance_complete_source_rejects_missing_operative_values(source, expected):
    result = completeness.analyze_complete_source_unit(
        yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []}),
        source,
        corpus_citation_path=GUIDANCE_PAGE,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    for value in expected:
        assert any(
            f"Authoritative corpus numeric value {value:g} has no named scalar" in issue
            for issue in result.issues
        ), f"Guidance silently accepts omission of {value}: {result.issues!r}"


def test_empty_guidance_namespace_keeps_numbered_text():
    source = "214.2 Shared Utility Costs\nThe payment is $444."
    assert (
        completeness.authoritative_numeric_recall_text(
            source,
            corpus_citation_path="us-md/guidance//snap-manual-214/page-5",
        )
        == source
    )


@given(
    cents=st.integers(min_value=1, max_value=999999),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    lower=st.integers(min_value=1, max_value=98),
    namespace=st.sampled_from(("dhs/fia", "agency", "")),
)
@settings(deadline=None)
def test_guidance_manual_shaped_paths_preserve_table_values_and_body_dates(
    cents, month, year, lower, namespace
):
    source = (
        "The following table lists benefit amounts in dollars.\n"
        f"{cents // 100}.{cents % 100:02d} Monthly Allowance\n"
        "Applicable interest rates, in percent:\n"
        "7.5 Annual Rate\n"
        "The following table lists the eligible child counts.\n"
        f"{lower}-{lower + 1} Eligible Children\n"
        f"Applications filed in ({month:02d}/{year}) are eligible."
    )
    assert (
        completeness.authoritative_numeric_recall_text(
            source,
            corpus_citation_path=f"us-md/guidance/{namespace}/snap-manual-214/page-5",
        )
        == source
    )

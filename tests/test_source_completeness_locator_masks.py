"""Statutes at Large locators cannot hide operative numeric values."""

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

AZ_SECTION = "us-az/statute/43-1072"


def _values(text: str, citation: str) -> set[float]:
    cleaned = completeness.authoritative_numeric_recall_text(
        text, corpus_citation_path=citation
    )
    return {
        item.value
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="en-US"
        )
    }


def test_numeric_demands_keep_values_equal_to_masked_components():
    content = yaml.safe_dump({"format": "rulespec/v1", "module": {}, "rules": []})
    result = completeness.analyze_complete_source_unit(
        content,
        "The amounts are $49 and $620. The provision cites (49 Stat. 620).",
        corpus_citation_path=AZ_SECTION,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    for value in (49, 620):
        assert any(
            f"Authoritative corpus numeric value {value} has no named scalar" in issue
            for issue in result.issues
        )


def test_only_complete_parenthesized_statutes_at_large_citation_is_masked():
    source = "The Social Security Act (49 Stat. 620) applies."
    assert not _values(source, AZ_SECTION) & {49, 620}
    assert {49, 620} <= _values(
        source + " The applicable amounts are $49 and $620.", AZ_SECTION
    )
    # Replacing a citation with nothing would join two operative numbers.
    assert {49, 620} <= _values(
        "The applicable amounts are $49(49 Stat. 620)620 dollars.", AZ_SECTION
    )


@pytest.mark.parametrize(
    "source",
    (
        "49 Stat. 620",
        "(49 Stat. 620",
        "(49 Stat. 620 dollars)",
        "($49 Stat. 620)",
        "(49 Stat. $620)",
        "(49 Stat. 620 - 7)",
        "(49 Stat. 620; a deduction of $444 is allowed)",
    ),
)
def test_partial_or_operative_statutes_at_large_shapes_keep_values(source):
    assert {49, 620} <= _values(source, AZ_SECTION)


@pytest.mark.parametrize("source", ("(49\nStat. 620)", "(49 Stat.\n620)"))
def test_statutes_at_large_mask_never_crosses_a_line(source):
    assert (
        completeness._STATUTES_AT_LARGE_NUMERIC_RECALL_CITATION.search(source) is None
    )


def test_manual_page_and_revision_masks_keep_operative_values():
    assert _values(
        "93 (07/2026) Chapter 2: Eligibility\nThe payment is $93 and $2026.",
        "us-or/manual/odhs/open/page-5",
    ) == {93, 2026}


@pytest.mark.parametrize(
    "body",
    (
        "is the qualifying application month.",
        "The payment is $93 in the qualifying application month shown above.",
        "The payment is $93 and $2026.\nThe month shown above determines eligibility.",
    ),
)
def test_manual_header_like_body_date_keeps_operative_values(body):
    source = f"93 (07/2026) Chapter 2: Eligibility\n{body}"
    assert {7, 2026} <= _values(source, "us-or/manual/odhs/open/page-5")


@pytest.mark.parametrize(
    "citation, caption",
    (
        ("us-or/manual/odhs/open/page-5", "Chapter 2: Unknown Caption"),
        ("us-az/manual/program/page-5", "Chapter 2: Eligibility"),
    ),
)
def test_independent_amount_sentence_cannot_authenticate_unknown_manual_header(
    citation, caption
):
    source = f"93 (07/2026) {caption}\nThe payment is $93 and $2026."
    assert {7, 93, 2026} <= _values(source, citation)


@pytest.mark.parametrize(
    "citation", ("us-or/manual/odhs/open/page-5", "us-az/manual/program/page-5")
)
def test_spaced_header_without_independent_body_keeps_parent_date_demands(citation):
    source = "93 (07/2026) Chapter 2: Eligibility"
    assert {7, 2026} <= _values(source, citation)


@given(
    page=st.integers(min_value=1, max_value=9999),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    subject=st.sampled_from(("payment", "amount", "limit")),
    spacing=st.sampled_from(("", " ")),
)
@settings(max_examples=50, deadline=None)
def test_authenticated_header_masks_only_its_metadata_occurrences(
    page, month, year, subject, spacing
):
    stamp = f"({month:02d}/{year})"
    body = f"The {subject} is ${page} and ${year}."
    source = f"{page} {stamp} Chapter 2:{spacing}Eligibility\n{body}"
    citation = "us-or/manual/odhs/open/page-5"
    assert _values(source, citation) == {page, year}
    masked = completeness._without_us_manual_locators(
        source, corpus_citation_path=citation
    )
    assert stamp not in masked
    assert masked.endswith(body)
    assert len(masked) == len(source)
    assert (
        completeness._without_us_manual_locators(masked, corpus_citation_path=citation)
        == masked
    )


@given(
    st.integers(min_value=1, max_value=999999),
    st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ", max_size=80
    ),
)
@settings(deadline=None)
def test_every_amount_outside_validated_locators_is_recalled(amount, operative_prose):
    # Equal numeric values are never globally excluded with the Stat. volume/page.
    source = (
        "The Social Security Act (49 Stat. 620) applies. "
        + f"The limit {operative_prose} is ${amount}."
    )
    assert float(amount) in _values(source, AZ_SECTION)


@given(st.integers(min_value=1, max_value=999999))
@settings(deadline=None)
def test_locator_masks_are_idempotent_and_never_lengthen_text(amount):
    source = f"(49 Stat. 620) The limit is ${amount}."
    masked = completeness.authoritative_numeric_recall_text(
        source, corpus_citation_path=AZ_SECTION
    )
    assert len(masked) == len(source)
    assert (
        completeness.authoritative_numeric_recall_text(
            masked, corpus_citation_path=AZ_SECTION
        )
        == masked
    )

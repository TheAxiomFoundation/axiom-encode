"""Complete-source regressions for conservative manual-mask context authority."""

from __future__ import annotations

import functools
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.manual_locator_metadata import MANUAL_CONTENTS_HEADINGS
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

EMPTY_RULES = "format: rulespec/v1\nmodule: {}\nrules: []\n"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)
# Only these independently reviewed procedure titles establish harmless context.
HARMLESS_PROCEDURE_TITLES = frozenset(
    {
        "Application Process",
        "Retroactive Applications",
        "Advance Notice of Adverse Action",
        "Warranties, Representations and Covenants",
    }
)
UNCERTAIN_CONTENTS_ENTRIES = tuple(
    (namespace, number, title)
    for namespace, headings in sorted(MANUAL_CONTENTS_HEADINGS.items())
    for number, title in sorted(headings)
    if title not in HARMLESS_PROCEDURE_TITLES
)


def _demanded(source: str, citation: str, profile: str) -> set[float]:
    result = sc.analyze_complete_source_unit(
        EMPTY_RULES,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        float(match[1])
        for issue in result.issues
        if (match := RECALL_VALUE.search(issue)) is not None
    }


def _raw_values(source: str, profile: str) -> set[float]:
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            source, profile=profile
        )
    }


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("shape", ("raw", "page", "root"))
def test_reviewed_financial_titles_cannot_authenticate_an_ambiguous_money_column(
    profile, shape
):
    body = (
        "Section 8701: Applications\n8701.3 Capital Gains\n"
        "8701.1 Averaging Self-Employment Income\n"
        "8701.2 Self-Employment From a New Source"
    )
    citation = "us-sd/manual/dss/snap-policy-procedure-manual/page-1"
    source = body if shape == "raw" else "Page 1\n\n" + body
    if shape == "root":
        citation = citation.rsplit("/page-", 1)[0]
    assert {8701.3, 8701.1, 8701.2} <= _demanded(source, citation, profile)


@settings(max_examples=60, deadline=None)
@given(
    entry=st.sampled_from(UNCERTAIN_CONTENTS_ENTRIES),
    profile=st.sampled_from(("en-US", "legacy")),
    shape=st.sampled_from(("raw", "page", "root")),
    newline=st.sampled_from(("\n", "\r\n")),
)
@example(
    entry=(
        "us-sd/manual/dss/snap-policy-procedure-manual",
        "8701.3",
        "Capital Gains",
    ),
    profile="en-US",
    shape="raw",
    newline="\n",
)
def test_generated_contents_titles_need_independent_harmless_context(
    entry, profile, shape, newline
):
    """A published title alone cannot establish the role of a new source body."""
    namespace, number, title = entry
    section = re.split(r"[.-]", number)[0]
    body = f"Section {section}: Applications\n{number} {title}"
    source = body if shape == "raw" else "Page 1\n\n" + body
    citation = namespace if shape == "root" else namespace + "/page-1"
    expected = _raw_values(number, profile)
    assert expected <= _demanded(source.replace("\n", newline), citation, profile)


@pytest.mark.parametrize("composed", (False, True), ids=("raw", "sole-body"))
@pytest.mark.parametrize(
    "source, expected, profile",
    (
        ("Table 1: $2,474 2 $2,909", {2474, 2909}, "en-US"),
        ("Table 1: $2,474 2 $2,909", {2474, 2909}, "legacy"),
        ("Table 1: Household size 1 100 size 2 200", {100, 200}, "en-US"),
    ),
    ids=("currency-cells-en-US", "currency-cells-legacy", "size-cells-en-US"),
)
def test_flattened_tables_keep_current_main_operative_values(
    source, expected, profile, composed
):
    """Carry forward main's value-preserving table cleaner during integration."""
    if composed:
        source = "Page 1\n\n" + source
    assert expected <= _demanded(source, "us-sc/manual/scdhhs/mppm/page-1", profile)


@settings(max_examples=16, deadline=None)
@given(
    first=st.integers(min_value=1000, max_value=9999),
    second=st.integers(min_value=1000, max_value=9999),
    table=st.integers(min_value=1, max_value=99),
    currency=st.sampled_from(("$", "£", "€")),
    composed=st.booleans(),
    profile=st.sampled_from(("en-US", "legacy")),
)
@example(
    first=2474,
    second=2909,
    table=1,
    currency="$",
    composed=False,
    profile="en-US",
)
def test_generated_flattened_currency_tables_keep_every_amount(
    first, second, table, currency, composed, profile
):
    """A table locator or adjacent row key never consumes currency cells."""
    source = f"Table {table}: {currency}{first:,} 2 {currency}{second:,}"
    if composed:
        source = "Page 1\n\n" + source
    assert {first, second} <= _demanded(
        source, "us-sc/manual/scdhhs/mppm/page-1", profile
    )

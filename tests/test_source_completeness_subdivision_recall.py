"""Legal subdivision locators and poverty geography are not benefit operands."""

from __future__ import annotations

import functools
import re

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

VA_CITATION = "us-va/statute/58.1/58.1-339.8"
EXTRACT_NUMBERS = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)


def _clean(text: str) -> str:
    return completeness.authoritative_numeric_recall_text(
        text, corpus_citation_path=VA_CITATION
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("The subtraction under subdivision 8 of § 58.1-322.02 ;", []),
        ("the credit authorized under subdivision 1 or 2", []),
        ("the 48 contiguous states", []),
        ("under section 5 of the act", []),
        ("under paragraph 5 of the act", []),
        ("a credit of $300 and 20 percent", ["300", "20"]),
        ("under subdivision B 3", []),
        ("under subdivision B3", []),
        ("a credit under subdivision 2 of this section for taxpayers", []),
        ("subdivision B3 is required for low-income families", []),
        ("under subdivision 2 b", ["2"]),
        ("under subdivision 2(b)", []),
        ("under subdivisions 2, 3, 5, 15, and 16", []),
        ("under subdivisions B 2 or B 3", []),
        ("under subdivisions 3 through 5 of this section", []),
        ("under subdivision 1 or 3 for the same taxable year", []),
        ("under subdivision 1 or 2 for the same taxable year", []),
        ("under subdivision 2 and 20 per cent", ["2", "20"]),
        ("under subdivision 2 or 3 times the credit", ["2", "3"]),
        ("under subdivision 2 and 2025 taxable years", ["2", "2025"]),
        ("under subdivision 2, 3 dependents", ["2", "3"]),
        ("under subdivision 2, 3 returns", ["2", "3"]),
        ("under subdivision 2 or 20.5 percent", ["2", "20.5"]),
        ("under subdivision 2 or 2,000 dollars", ["2", "2", "000"]),
        ("under subdivision 2 or 3 eligible nonresident married taxpayers", ["2", "3"]),
        ("under subdivision 2 and 20 annual qualifying student members", ["2", "20"]),
        ("under subdivision 2 or 14 senators", ["2", "14"]),
    ),
)
def test_issue_1841_reduced_reproduction(source, expected):
    cleaned = _clean(source)
    assert re.findall(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", cleaned) == expected
    if "20.5 percent" in source:
        assert "20.5 percent" in cleaned
        assert 20.5 in {item.value for item in EXTRACT_NUMBERS(cleaned)}


@pytest.mark.parametrize(
    "source",
    (
        "Under subdivision 2, the credit is $300 and 20 percent for 3 persons in 2025.",
        "Under subdivision 2 and 20 percent of income, the credit is $300 for 3 persons in 2025.",
        "Under subdivision 2 or 3 qualifying children, the credit is $300 and 20 percent in 2025.",
        "Under subdivision 1 or 3 for the same taxable year, the credit is $300 and 20 percent for 3 persons in 2025.",
        "Under subdivision 1 or 2 for the same taxable year, the credit is $300 and 20 percent for 3 persons in 2025.",
    ),
)
def test_subdivision_mask_preserves_operative_amounts_rates_counts_and_years(source):
    assert {300, 20, 3, 2025} <= {
        item.value for item in EXTRACT_NUMBERS(_clean(source))
    }


@pytest.mark.parametrize(
    "source",
    (
        "48 states",
        "47 contiguous states",
        "148 contiguous states",
        "48 contiguous stateside offices",
        "$48 contiguous states",
        "$ 48 contiguous states",
        "$   48 contiguous states",
        "€48 contiguous states",
        "£ 48 contiguous states",
        "¥48 contiguous states",
        "₹ 48 contiguous states",
        "48.5 contiguous states",
        "1,048 contiguous states",
    ),
)
def test_geography_mask_requires_the_exact_bounded_identifier(source):
    assert _clean(source) == source


@pytest.mark.parametrize("separator", ("\n", "\r\n"))
def test_wrapped_currency_is_not_poverty_geography(separator):
    source = f"${separator}48 contiguous states"
    assert _clean(source) == source


@pytest.mark.parametrize("prefix", ("₩", "USD ", "usd ", "yuan "))
def test_currency_geography_counts_remain_demanded(prefix):
    source = f"{prefix}48 contiguous states"
    assert _clean(source) == source
    result = completeness.analyze_complete_source_unit(
        "format: rulespec/v1\nmodule: {}\nrules: []\n",
        source,
        corpus_citation_path=VA_CITATION,
        test_cases=[],
        extract_numeric_occurrences=EXTRACT_NUMBERS,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert any(
        "Authoritative corpus numeric value 48 has no named scalar" in issue
        for issue in result.issues
    ), result.issues


@given(
    marker=st.one_of(
        st.characters(categories=("Sc",)),
        st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=3, max_size=3).filter(
            lambda word: word != "THE"
        ),
    ),
    gap=st.sampled_from(("", " ", "\t", "\n", "\r\n", "\u202f")),
)
@example(marker="USD", gap=" ")
@settings(max_examples=35, deadline=None)
def test_unknown_money_markers_never_become_geography_identifiers(marker, gap):
    source = f"{marker}{gap}48 contiguous states"
    assert _clean(source) == source
    assert tuple(item.value for item in EXTRACT_NUMBERS(_clean(source))) == tuple(
        item.value for item in EXTRACT_NUMBERS(source)
    )


@pytest.mark.parametrize(
    "source",
    (
        "subdivision 2.5",
        "subdivision 2,500",
        "subdivision 2:30",
        "subdivision 2;3",
    ),
)
def test_subdivision_mask_requires_a_complete_locator(source):
    assert _clean(source) == source


@pytest.mark.parametrize(
    "source",
    ("subdivision 2m, 3 members must qualify.", "Subdivision a3b applies."),
)
def test_ambiguous_subdivision_labels_keep_every_number(source):
    assert _clean(source) == source


@pytest.mark.parametrize("suffix", ("K", "İ", "ſ", "ı"))
@pytest.mark.parametrize("entry_point", ("recall", "complete_source"))
def test_unicode_subdivision_suffix_never_crashes_or_masks_quantity(
    suffix, entry_point
):
    source = f"Under subdivision 15{suffix} is required."
    if entry_point == "recall":
        assert _clean(source) == source
    else:
        extracted_texts = []

        def extract_numbers(text):
            extracted_texts.append(text)
            return EXTRACT_NUMBERS(text)

        result = completeness.analyze_complete_source_unit(
            "format: rulespec/v1\nmodule: {}\nrules: []\n",
            source,
            corpus_citation_path=VA_CITATION,
            test_cases=[],
            extract_numeric_occurrences=extract_numbers,
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
        # Glued Unicode suffixes are not currently typed numeric tokens. Prove
        # that the gate receives their intact text, without inventing demands.
        assert source in extracted_texts
        assert result.source_numeric_occurrence_count == len(EXTRACT_NUMBERS(source))


@given(
    quantity=st.integers(min_value=1, max_value=999),
    suffix=st.text(
        alphabet=st.characters(categories=("L", "N", "M")),
        min_size=1,
        max_size=8,
    ),
    ending=st.sampled_from((".", " is required.", " of this section;")),
)
@example(quantity=15, suffix="K", ending=" is required.")
@example(quantity=15, suffix="İ", ending=".")
@example(quantity=15, suffix="ſ", ending=" of this section;")
@example(quantity=15, suffix="ı", ending=".")
@example(quantity=15, suffix="m", ending=".")
@example(quantity=15, suffix="\u0301", ending=".")
@example(quantity=15, suffix="٢", ending=".")
@example(quantity=15, suffix="2", ending=".")
@example(quantity=1, suffix="AA", ending=".")
@settings(max_examples=100, deadline=None)
def test_unicode_subdivision_suffix_masks_only_proven_ascii_locators(
    quantity, suffix, ending
):
    label = f"{quantity}{suffix}"
    source = f"Under subdivision {label}{ending}"
    # Public recall must never raise. Check the subdivision masks separately
    # from unrelated legacy cleanup (for example, glued sentence marker 1AA).
    _clean(source)
    cleaned = completeness._mask_numeric_spans(
        source, completeness._english_subdivision_numeric_recall_spans(source)
    )
    # An unparenthesized letter can denote a unit. Only an entirely ASCII
    # numeric label with a completed reference is proven in this grammar.
    if re.fullmatch(r"[0-9]+", label) and ending != " is required.":
        assert cleaned == f"Under subdivision {' ' * len(label)}{ending}"
    else:
        assert cleaned == source


@given(
    quantity=st.integers(min_value=100, max_value=999),
    suffix=st.characters(categories=("L", "N", "M")),
    ending=st.sampled_from((".", " is required.", " of this section;")),
)
@settings(max_examples=100, deadline=None)
def test_public_recall_preserves_arbitrary_unicode_subdivision_suffixes(
    quantity, suffix, ending
):
    # Three-digit numbers avoid the independent one/two-digit German sentence
    # marker grammar. Every Unicode letter, number and mark remains eligible.
    label = f"{quantity}{suffix}"
    source = f"Under subdivision {label}{ending}"
    if suffix in "0123456789" and ending != " is required.":
        assert _clean(source) == f"Under subdivision {' ' * len(label)}{ending}"
    else:
        assert _clean(source) == source


@pytest.mark.parametrize("ending", (".", " is required.", " of this section;"))
def test_untokenizable_subdivision_citation_keeps_every_number(monkeypatch, ending):
    class NoLocatorTokens:
        def finditer(self, *args):
            return iter(())

    monkeypatch.setattr(
        completeness, "_ENGLISH_SUBDIVISION_LOCATOR_TOKEN", NoLocatorTokens()
    )
    source = f"Under subdivision 15{ending}"
    assert _clean(source) == source


def test_subdivision_continuations_keep_decimal_rates_and_grouped_thresholds():
    source = "Under subdivision 2, the credit is 20.5 percent above $2,000."
    assert "20.5 percent above $2,000" in _clean(source)
    assert {20.5, 2000} <= {item.value for item in EXTRACT_NUMBERS(_clean(source))}


def test_land_subdivision_quantity_is_not_a_legal_locator():
    source = "A land subdivision 3 acres or larger qualifies."
    assert _clean(source) == source


def test_actual_virginia_two_return_count_remains_source_content():
    source = (
        "The credit provided under this section shall be allowed against the tax "
        "for only one of such two tax returns."
    )
    assert "two tax returns" in _clean(source)


@pytest.mark.parametrize(
    "quantity",
    (
        "3-5 persons",
        "3–5 persons",
        "3 through 5 persons",
        "2025 to 2026 taxable years",
        "3-5,000 qualifying people",
        "3-5 senators",
        "3 through 5 senators",
        "3–5 below-poverty senators",
    ),
)
def test_coordinated_operative_ranges_keep_both_endpoints(quantity):
    source = f"Under subdivision 2 or {quantity}, the credit is $300."
    assert quantity in _clean(source)
    assert tuple(item.value for item in EXTRACT_NUMBERS(_clean(source))) == tuple(
        item.value for item in EXTRACT_NUMBERS(f"2 or {quantity}, the credit is $300.")
    )


def test_numeric_gate_keeps_values_equal_to_locator_and_geography_numbers():
    result = completeness.analyze_complete_source_unit(
        "format: rulespec/v1\nmodule: {}\nrules: []\n",
        "Under subdivision 8, use the guidelines for the 48 contiguous states. "
        "The credit is $8 for each of 48 persons and 20 percent of income.",
        corpus_citation_path=VA_CITATION,
        test_cases=[],
        extract_numeric_occurrences=EXTRACT_NUMBERS,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    for value in (8, 48, 20):
        assert any(
            f"Authoritative corpus numeric value {value:g} has no named scalar" in issue
            for issue in result.issues
        ), result.issues


@given(
    locator=st.sampled_from(
        ("subdivision 8", "subdivision 1 or 2", "subdivision B 3", "subdivision 2(b)")
    ),
    amount=st.integers(min_value=1, max_value=999999),
    count=st.integers(min_value=1, max_value=999),
    percent=st.integers(min_value=1, max_value=99),
    year=st.integers(min_value=2000, max_value=2099),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_only_identifier_spans_are_removed(locator, amount, count, percent, year):
    source = (
        f"Under {locator}, use the guidelines for the 48 contiguous states. "
        f"The credit is ${amount} and {percent} percent for {count} persons in {year}."
    )
    operative = source[source.index("The credit") :]
    assert tuple(item.value for item in EXTRACT_NUMBERS(_clean(source))) == tuple(
        item.value for item in EXTRACT_NUMBERS(operative)
    )
    # Preserve legal/geographic meaning as text, and make cleanup idempotent.
    assert "contiguous states" in _clean(source)
    assert _clean(_clean(source)) == _clean(source)


@given(
    count=st.integers(min_value=1, max_value=999),
    modifiers=st.lists(
        st.sampled_from(
            ("eligible", "nonresident", "married", "qualifying", "student")
        ),
        max_size=10,
    ),
    unit=st.sampled_from(
        (
            "persons",
            "children",
            "dependents",
            "taxpayers",
            "returns",
            "states",
            "acres",
            "senators",
        )
    ),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_coordinated_operative_counts_are_never_locator_continuations(
    count, modifiers, unit
):
    quantity = " ".join((str(count), *modifiers, unit))
    source = f"Under subdivision 2 or {quantity}, the credit is $300 and 20 percent."
    assert quantity in _clean(source)
    assert float(count) in {item.value for item in EXTRACT_NUMBERS(_clean(source))}


@given(
    lower=st.integers(min_value=1, max_value=999),
    width=st.integers(min_value=1, max_value=999),
    connector=st.sampled_from(("-", "–", " through ", " to ")),
    unit=st.sampled_from(("persons", "senators", "widgets")),
)
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_unit_bearing_range_endpoints_are_never_locator_numbers(
    lower, width, connector, unit
):
    quantity = f"{lower}{connector}{lower + width} eligible {unit}"
    source = f"Under subdivision 2 or {quantity}, the credit is $300."
    assert quantity in _clean(source)

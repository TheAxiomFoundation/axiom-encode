"""Generated recall invariants, including equal-value and near-miss adversaries.

The deterministic generators use only the existing test dependencies so the
suite can run in offline encoder lanes. Each property varies numeric values,
source formatting and extraction profile; regressions remain reproducible.
"""

from __future__ import annotations

import functools
import random

import pytest

from axiom_encode.harness.source_completeness import (
    _additional_numeric_recall_spans,
    _analyze_rulespec_payload,
    authoritative_numeric_recall_text,
)
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

UK = "uk/regulation/uksi/2002/1792/schedule/IIA"
US = "us/statute/26/7701"
PROFILES = ("legacy", "en-US", "en-GB")
STRUCTURES = (
    (UK, "General {n} This Schedule applies."),
    (UK, "{n}) An individual is eligible."),
    (UK, "| {n}Severe Disability Premium— | {n} |"),
    (UK, "The {year} Act applies."),
    (UK, "The Act of {year} applies."),
    (UK, "See section {n}C(1)(a)."),
    (US, "Internal Revenue Service Notice {year}–{n} applies."),
    (US, "Pub. L. {n}–369 applies."),
    (US, "{n} Stat. 792 applies."),
    (US, "{n} U.S.C. note prec. 4651 applies."),
    (US, "See {n} of the National Housing Act."),
    (US, "1 1 So in original. See {year} Amendment note below."),
)


def _inventory(source: str, citation: str, profile: str):
    cleaned = authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    occurrences = extract_typed_numeric_inventory_occurrences_from_text(
        cleaned, profile=profile
    )
    assert all(cleaned[item.start : item.end] == item.raw for item in occurrences)
    return occurrences


def _recall_issues(source: str, citation: str, profile: str, values=(), module=None):
    # Exercise the production accounting with an empty Python payload. This
    # test creates no RuleSpec YAML, encoded module or apply manifest.
    analysis = _analyze_rulespec_payload(
        {"module": module or {}, "rules": []},
        content="",
        source_text=source,
        corpus_citation_path=citation,
        test_cases=(),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=lambda _content: (),
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=values,
        artifact_numeric_bindings=None,
        authenticated_same_act_aliases=(),
        imported_symbol_contents=(),
    )
    return [issue for issue in analysis.issues if ":numeric-recall]" in issue]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation,template", STRUCTURES)
def test_generated_structures_never_exempt_equal_operative_values(
    citation, template, profile
):
    generator = random.Random(481516)
    values = [1, 6, 98, 2012, *generator.sample(range(2, 9999), 8)]
    quantities = ("£{n}", "{n} dollars", "{n} percent", "{n} months")
    for index, value in enumerate(values):
        year = generator.randrange(1900, 2100)
        structure = template.format(n=value, year=year)
        assert not _inventory(structure, citation, profile), structure
        # Years are operative controls too: no filtering by numeric value.
        amount = year if "{year}" in template else value
        quantity = quantities[index % len(quantities)].format(n=amount)
        source = f"{structure}\nThe required quantity is {quantity}."
        occurrences = _inventory(source, citation, profile)
        assert len(occurrences) == 1, (source, occurrences)
        scalar = float(amount) / 100 if "percent" in quantity else float(amount)
        assert numeric_value_is_grounded(scalar, occurrences)
        assert _recall_issues(source, citation, profile), source
        assert not _recall_issues(source, citation, profile, (scalar,)), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation,template", STRUCTURES)
def test_generated_marker_renumbering_preserves_substantive_inventory(
    citation, template, profile
):
    generator = random.Random(8675309)
    for _ in range(12):
        first, second, amount = generator.sample(range(1, 9999), 3)
        inventories = []
        for label in (first, second):
            structure = template.format(n=label, year=1900 + label % 200)
            source = f"{structure}\nThe payment is £{amount}."
            # New exclusions must retain every source coordinate.
            spans = _additional_numeric_recall_spans(
                source, corpus_citation_path=citation
            )
            for start, end in spans:
                assert 0 <= start < end <= len(source)
                assert source[start:end].strip()
                assert "£" not in source[start:end]
            inventories.append(
                [
                    (item.value, item.raw)
                    for item in _inventory(source, citation, profile)
                ]
            )
        assert inventories[0] == inventories[1] == [(float(amount), str(amount))]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source",
    (
        "1 dollar is payable.",
        "1 percent is payable.",
        "1-year period applies.",
        "1 month is required.",
        "1.5 dollars is payable.",
        "1) Dollars are required.",
        "An act of 1977 is required.",
        "A code of 2000 is required.",
        "Act of 1977 dollars is required.",
        "£2012 Act applies.",
        "£ 2012 Act applies.",
        "GBP 2012 Act applies.",
        "Minimum earnings 100 The claimant must be employed.",
        "Payment GBP 100 The award is due.",
        "| 1Person household size | 1 |",
        "| 6Severe Disability Premium— | £6 |",
        "| 6Severe Disability Premium— | 7 |",
        "General 1.5 This Schedule applies.",
    ),
)
def test_quantity_and_unconfirmed_marker_near_misses_remain_required(source, profile):
    assert _inventory(source, UK, profile), source
    assert _recall_issues(source, UK, profile), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source",
    ("General 1 This Schedule applies.", "1) An individual is eligible.", "2012 Act"),
)
def test_uk_conventions_require_authoritative_uk_citation(source, profile):
    assert _inventory(source, US, profile)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize(
    "source,value",
    (
        ("$236 of the National Housing Act subsidy is payable.", 236),
        ("£236 of the National Housing Act subsidy is payable.", 236),
        ("GBP 236 of the National Housing Act subsidy is payable.", 236),
        ("1.236 of the National Housing Act subsidy is payable.", 1.236),
        ("$98 Stat. 792 is payable.", 98),
        ("236 of the National Housing Act dollars is payable.", 236),
    ),
)
def test_bibliographic_near_misses_cannot_remove_amounts(source, value, profile):
    occurrences = _inventory(source, US, profile)
    assert any(item.value == value for item in occurrences), (source, occurrences)
    assert _recall_issues(source, US, profile), source


@pytest.mark.parametrize(
    "prefix,required",
    (
        ("$1\u00a0", 236),
        ("$1 ", 236),
        ("1/", 236),
        ("− ", 236),
        ("1 * ", 236),
        ("1,", 1236),
        ("1. ", 236),
    ),
)
def test_numeric_envelope_prefixes_cannot_hide_bibliographic_shaped_amounts(
    prefix, required
):
    source = f"{prefix}236 of the National Housing Act subsidy is payable."
    assert not _additional_numeric_recall_spans(source, corpus_citation_path=US)
    assert any(item.value == required for item in _inventory(source, US, "legacy"))
    assert _recall_issues(source, US, "legacy", (1,)), source


def test_ambiguous_multiple_quantities_cannot_be_treated_as_a_heading():
    source = "Maximum 3 10 The claimant must qualify."
    assert not _additional_numeric_recall_spans(source, corpus_citation_path=UK)
    assert _recall_issues(source, UK, "legacy")


@pytest.mark.parametrize("profile", PROFILES)
def test_currency_vocabulary_protects_citation_shaped_amounts(profile):
    prefixes = (
        "£",
        "$",
        "€",
        "¥",
        "₹",
        "pounds",
        "dollars",
        "euros",
        "gbp",
        "usd",
        "cad",
        "aud",
        "chf",
        "Canadian dollars",
        "Swiss francs",
    )
    for currency in prefixes:
        for citation, numeral, fragment in (
            (UK, 2012, "2012 Act"),
            (US, 236, "236 of the National Housing Act"),
        ):
            source = f"{currency} {fragment} subsidy is payable."
            assert any(
                item.value == numeral for item in _inventory(source, citation, profile)
            ), source
            assert _recall_issues(source, citation, profile), source


def test_new_masks_preserve_currency_codes_even_if_legacy_cleaning_does_not():
    # JPY already disappears through the pinned all-caps document-reference
    # cleaner. That pre-existing behavior must not justify a new span mask.
    assert not _additional_numeric_recall_spans(
        "JPY 2012 Act applies.", corpus_citation_path=UK
    )


@pytest.mark.parametrize("profile", PROFILES)
def test_currency_extensions_preserve_untyped_numeric_obligations(profile):
    for currency in ("jpy", "ils", "nis", "kr", "kroner", "øre"):
        for citation, numeral, fragment in (
            (UK, 2012, "2012 Act"),
            (US, 236, "236 of the National Housing Act"),
        ):
            for spacing in ("", " "):
                source = f"{currency}{spacing}{fragment} subsidy is payable."
                assert not _additional_numeric_recall_spans(
                    source, corpus_citation_path=citation
                )
                if spacing:
                    # Existing extraction does not promise numeric obligations
                    # for digits glued to unknown currency words. The new
                    # masks must still avoid claiming those tokens as citations.
                    assert any(
                        item.value == numeral
                        for item in _inventory(source, citation, profile)
                    ), source
                    assert _recall_issues(source, citation, profile), source
        source = f"Act of 1977 {currency} is payable."
        assert not _additional_numeric_recall_spans(source, corpus_citation_path=UK)
        assert any(item.value == 1977 for item in _inventory(source, UK, profile)), (
            source
        )
        assert _recall_issues(source, UK, profile), source


@pytest.mark.parametrize("profile", PROFILES)
def test_candidate_metadata_cannot_exempt_authoritative_amounts(profile):
    source = "2012 Act applies. The payment is £2012."
    module = {
        "summary": "All 2012 values are structural and have been encoded.",
        "numeric_recall_exclusions": [2012],
        "source_verification": {"corpus_citation_path": UK},
    }
    assert _recall_issues(source, UK, profile, module=module)
    assert _recall_issues(source, US, profile, module=module)


@pytest.mark.parametrize("unit,factor", (("weeks", 7), ("ans", 12)))
def test_generated_duration_conversions_cannot_hide_a_separate_payment(unit, factor):
    generator = random.Random(314159)
    for amount in (1, 4, 8, 26, 52, *generator.sample(range(2, 100), 12)):
        source = f"The duration is {amount} {unit}."
        occurrences = _inventory(source, UK, "legacy")
        assert [item.value for item in occurrences] == [float(amount)]
        grounding = extract_typed_numeric_occurrences_from_text(source)
        assert numeric_value_is_grounded(float(amount * factor), grounding)
        assert not _recall_issues(source, UK, "legacy", (float(amount),))
        assert _recall_issues(source, UK, "legacy")
        paired = f"{source}\nThe payment is £{amount * factor}."
        assert _recall_issues(paired, UK, "legacy", (float(amount),)), paired
        assert not _recall_issues(
            paired, UK, "legacy", (float(amount), float(amount * factor))
        ), paired


@pytest.mark.parametrize(
    "source,converted",
    (("deux ans", 24), ("La sixieme semaine.", 42)),
)
def test_word_durations_without_a_surviving_source_amount_keep_a_recall_obligation(
    source, converted
):
    assert converted in [item.value for item in _inventory(source, UK, "legacy")]
    assert _recall_issues(source, UK, "legacy")


@pytest.mark.parametrize(
    "source,amount,factor",
    (
        ("eight weeks", 8, 7),
        ("La periode est de six semaines.", 6, 7),
        ("Cette prolongation ne peut depasser quinze ans.", 15, 12),
    ),
)
def test_word_durations_with_a_complete_recalled_amount_do_not_require_its_conversion(
    source, amount, factor
):
    assert [item.value for item in _inventory(source, UK, "legacy")] == [amount]
    assert _recall_issues(source, UK, "legacy")
    assert not _recall_issues(source, UK, "legacy", (amount,))
    assert numeric_value_is_grounded(
        amount * factor, extract_typed_numeric_occurrences_from_text(source)
    )
    assert _recall_issues(
        f"{source} The payment is £{amount * factor}.", UK, "legacy", (amount,)
    )


@pytest.mark.parametrize("source", ("1\u00a0000 weeks", "1.000 weeks"))
def test_partial_digit_duration_cannot_clear_recall_with_its_leading_component(source):
    assert 7000 in [item.value for item in _inventory(source, UK, "legacy")]
    assert _recall_issues(source, UK, "legacy", (1,))


def test_decimal_duration_requires_the_complete_written_amount():
    source = "4.5 weeks"
    # Legacy's European conversion parser and decimal digit parser disagree
    # here. Keep both existing obligations until that ambiguity is resolved.
    assert 4.5 in [item.value for item in _inventory(source, UK, "legacy")]
    assert _recall_issues(source, UK, "legacy", (4, 5))
    assert _recall_issues(source, UK, "legacy", (315,))


@pytest.mark.parametrize("unit", ("weeks", "ans"))
def test_very_small_duration_sources_keep_a_nonzero_recall_obligation(unit):
    # Legacy does not normalize these long comma decimals accurately. Keep
    # its existing nonzero obligation rather than silently accepting zero.
    source = f"0,0000004 {unit}"
    assert _inventory(source, UK, "legacy")
    assert _recall_issues(source, UK, "legacy")
    assert _recall_issues(source, UK, "legacy", (0,))

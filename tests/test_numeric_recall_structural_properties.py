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
    SourceStructureBranch,
    _additional_numeric_recall_spans,
    _analyze_rulespec_payload,
    _source_boundary_obligations,
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
GLUED_RECALL_QUANTITIES = (
    "100EUR",
    "100eur",
    "100pence",
    "100PENCE",
    "100EUR(1)",
    "100eur(1)",
    "100.5pence",
    "1.100EUR",
    "1.100pence",
    "100.5ha",
    "100.5km",
    "100.5m",
    "100.5h",
    "100.5x",
    "100.5C",
)
# The citation recognizer must stop before each operative amount, including
# unfamiliar units and a bare threshold that has no unit to whitelist.
ADJACENT_QUANTITIES = (
    ("section 7C", "35 per cent of earnings is disregarded.", (0.35,), (35,)),
    ("subsection 7", "35 per centum of earnings is disregarded.", (0.35,), (35,)),
    ("section 7C", "100 pence is the maximum weekly payment.", (1,), (100,)),
    ("paragraph 2(1)", "100 pence is the maximum weekly payment.", (1,), (100,)),
    ("section 7C", "100,000 is the maximum annual income.", (100000,), (100000,)),
    (
        "section 7C",
        "100,000 or 150,000 is the annual income limit.",
        (100000, 150000),
        (100000, 150000),
    ),
    (
        "paragraph 2(1)",
        "250 basis points is the maximum adjustment.",
        (250,),
        (250,),
    ),
    ("section 7C", "18 hours is the maximum permitted duration.", (18,), (18,)),
    ("section 7C", "60 minutes is the minimum required duration.", (60,), (60,)),
    ("section 7C", "100 cents is the maximum weekly payment.", (100,), (100,)),
    ("section 7C", "100 households may participate in the pilot.", (100,), (100,)),
    ("section 7C", "17.5 metres is the maximum permitted height.", (17.5,), (17.5,)),
    ("section 7C", "1,000 people may participate in the pilot.", (1000,), (1000,)),
    (
        "sections 7C and 8C",
        "35 per cent of earnings is disregarded.",
        (0.35,),
        (35,),
    ),
    ("section 7", "100,000 is the maximum annual income.", (100000,), (100000,)),
)
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
@pytest.mark.parametrize(
    "quantity,value", (("1 year", 1), ("2 years", 2), ("1.5 years", 1.5))
)
def test_citation_then_duration_distinguishes_glossary_from_operative_sentence(
    profile, quantity, value
):
    tail = (
        "into the U.S. under section 212(d)(5) of the INA "
        f"for a period of at least {quantity}"
    )
    for source, expected in (
        (f"Parolees Paroled {tail}.", []),
        (f"A parolee is eligible if paroled {tail}.", [(value, str(value))]),
        (
            f"Parolees Paroled {tail} and applicants receive SNAP benefits "
            "if they are citizens.",
            [(value, str(value))],
        ),
    ):
        assert [(item.value, item.raw) for item in _inventory(source, US, profile)] == [
            (value, str(value))
        ], source
        root = SourceStructureBranch(
            (), "source-unit", "source unit", source, 0, len(source)
        )
        obligations = _source_boundary_obligations(
            (root,),
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text, profile=profile
            ),
        )
        assert [
            (occurrence.value, occurrence.raw) for _branch, occurrence in obligations
        ] == expected, source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("gate", ("missing", "present"))
@pytest.mark.parametrize(
    "reference,quantity,legacy_values,locale_values,separator",
    [
        (*case, separator)
        for case in ADJACENT_QUANTITIES
        for separator in (", ", ",")
        # A suffix/subdivision separates the citation from a following
        # no-space comma quantity. A bare target could itself be grouped.
        if separator != "," or not case[0][-1].isdigit()
    ],
)
def test_adjacent_citation_quantities_keep_their_production_recall_gate(
    reference, quantity, legacy_values, locale_values, profile, gate, separator
):
    source = f"Under {reference}{separator}{quantity}"
    values = legacy_values if profile == "legacy" else locale_values
    occurrences = _inventory(source, UK, profile)
    assert tuple(item.value for item in occurrences) == values, (source, occurrences)
    if gate == "present":
        assert not _recall_issues(source, UK, profile, values), source
    else:
        # Each alternative threshold has its own obligation. Recalling only
        # the other alternatives cannot satisfy the omitted source amount.
        for missing_index in range(len(values)):
            recalled = values[:missing_index] + values[missing_index + 1 :]
            assert _recall_issues(source, UK, profile, recalled), (source, recalled)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
def test_generated_adjacent_citations_preserve_every_operative_numeric_envelope(
    citation, profile
):
    """Structural labels never waive an adjacent or equal-valued quantity."""

    generator = random.Random(1779)
    values = (7, 35, 100, *generator.sample(range(2, 999), 8))
    separators = (", ", ",", " and ", " to ", " through ", "–")
    units = ("", "widgets", "basis points", "furlongs", "households", "metres")
    references = ("section {n}", "section {n}C", "paragraph {n}(1)")
    for index, value in enumerate(values):
        reference = references[index % len(references)].format(n=value)
        for amount, scalar in (
            (str(value), float(value)),
            (f"{value},000", float(value * 1000)),
            (f"{value}.5", value + 0.5),
        ):
            unit = units[index % len(units)]
            quantity = f"{amount} {unit} is the required limit."
            for separator in separators:
                if separator == "," and reference[-1].isdigit():
                    continue
                source = f"Under {reference}{separator}{quantity}"
                occurrences = _inventory(source, citation, profile)
                assert [(item.value, item.raw) for item in occurrences] == [
                    (scalar, amount)
                ], (source, occurrences)
                assert _recall_issues(source, citation, profile), source
                assert not _recall_issues(source, citation, profile, (scalar,)), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize(
    "source",
    (
        "Under sections 7C and 8C the claimant must qualify.",
        "Under sections 7C(1), 8C(2) and 9C(3) the claimant must qualify.",
        "Under sections 7, 8C(2) and 9C(3) the claimant must qualify.",
        "Under sections 7,100C(2) and 9C(3) the claimant must qualify.",
        "Under sections 7,100(2) the claimant must qualify.",
    ),
)
def test_complete_multi_target_citations_remain_structural(source, citation, profile):
    assert not _inventory(source, citation, profile)
    assert not _recall_issues(source, citation, profile)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize(
    "source",
    (
        "Under section\n7C(1)(a) the claimant must qualify.",
        "Under section\r\n7C(1)(a) the claimant must qualify.",
        "Under sections\n7C(1)(a) and 8C(2)(b) the claimant must qualify.",
        "Under sections 7C(1)(a)\nand 8C(2)(b) the claimant must qualify.",
        "Under sections 7C(1)(a) and\n8C(2)(b) the claimant must qualify.",
        "Under sections 7C(1)(a),\n8C(2)(b) the claimant must qualify.",
        "Under sections 7C(1)(a),\r\n8C(2)(b) the claimant must qualify.",
    ),
)
def test_wrapped_qualified_citations_preserve_structural_and_operative_obligations(
    source, citation, profile
):
    assert not _inventory(source, citation, profile)
    assert not _recall_issues(source, citation, profile)
    source += "\nThe maximum annual income is 100,000."
    assert [item.value for item in _inventory(source, citation, profile)] == [100000]
    assert _recall_issues(source, citation, profile)
    assert not _recall_issues(source, citation, profile, (100000,))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("reference", ("Section 40-18-2", "Section 40-18-2(a)"))
def test_hyphenated_section_identifier_keeps_its_entire_structural_target(
    reference, citation, profile
):
    structure = f"Under {reference} the claimant must qualify."
    assert not _inventory(structure, citation, profile)
    assert not _recall_issues(structure, citation, profile)
    source = f"Under {reference}, £40 is the required payment."
    assert [item.value for item in _inventory(source, citation, profile)] == [40]
    assert _recall_issues(source, citation, profile)
    assert not _recall_issues(source, citation, profile, (40,))
    for quantity, value in (("40", 40), ("100,000", 100000)):
        source = f"Under {reference},{quantity} is the required payment."
        assert [item.value for item in _inventory(source, citation, profile)] == [value]
        assert _recall_issues(source, citation, profile)
        assert not _recall_issues(source, citation, profile, (value,))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("gate", ("missing", "present"))
def test_grouped_first_target_near_miss_keeps_its_complete_threshold(profile, gate):
    source = "The section 100,000 is the annual income limit."
    assert [item.value for item in _inventory(source, UK, profile)] == [100000]
    if gate == "missing":
        assert _recall_issues(source, UK, profile)
    else:
        assert not _recall_issues(source, UK, profile, (100000,))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("reference", ("section 7C", "paragraph 2(1)"))
@pytest.mark.parametrize("separator", (", ", ","))
@pytest.mark.parametrize("sign", ("-", "−"))
@pytest.mark.parametrize(
    "quantity,legacy_value,locale_value",
    (
        ("100 pence is the required payment.", -1, -100),
        ("100,000 is the annual income threshold.", -100000, -100000),
    ),
)
def test_signed_adjacent_quantities_keep_their_profile_recall_obligation(
    quantity, legacy_value, locale_value, sign, separator, reference, profile
):
    source = f"Under {reference}{separator}{sign}{quantity}"
    value = legacy_value if profile == "legacy" else locale_value
    # Legacy already normalizes signed pence and mathematical-minus grouped
    # amounts as positive at the original PR head. Preserve that profile's
    # existing obligation; citation masking must not introduce another change.
    if profile == "legacy" and ("pence" in quantity or sign == "−"):
        value = abs(value)
    assert [item.value for item in _inventory(f"{sign}{quantity}", UK, profile)] == [
        value
    ]
    assert [item.value for item in _inventory(source, UK, profile)] == [value], source
    assert _recall_issues(source, UK, profile), source
    assert _recall_issues(source, UK, profile, (-value,)), source
    assert not _recall_issues(source, UK, profile, (value,)), source


@pytest.mark.parametrize("token", GLUED_RECALL_QUANTITIES)
@pytest.mark.parametrize("reference", ("section 7C", "paragraph 2(1)"))
@pytest.mark.parametrize("separator", (", ", ","))
@pytest.mark.parametrize("gate", ("missing", "present"))
def test_glued_quantity_units_keep_every_existing_legacy_recall_obligation(
    token, reference, separator, gate
):
    quantity = f"{token} is the limit."
    baseline = _inventory(f"Under this provision, {quantity}", UK, "legacy")
    assert baseline, quantity
    source = f"Under {reference}{separator}{quantity}"
    actual = _inventory(source, UK, "legacy")
    assert [(item.value, item.raw) for item in actual] == [
        (item.value, item.raw) for item in baseline
    ], source
    values = tuple(dict.fromkeys(item.value for item in baseline))
    if gate == "present":
        assert not _recall_issues(source, UK, "legacy", values), source
    else:
        for missing_index in range(len(values)):
            recalled = values[:missing_index] + values[missing_index + 1 :]
            assert _recall_issues(source, UK, "legacy", recalled), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("token", (*GLUED_RECALL_QUANTITIES, "100pence(1)"))
@pytest.mark.parametrize("reference", ("section 7C", "paragraph 2(1)"))
@pytest.mark.parametrize("separator", (", ", ","))
def test_glued_quantity_units_are_outside_citation_spans(
    token, reference, separator, profile
):
    # Strict profiles and annotated pence do not yet promise extraction of
    # glued units. The citation mask must still leave the source token intact.
    prefix = f"Under {reference}{separator}"
    source = f"{prefix}{token} is the limit."
    start, end = len(prefix), len(prefix) + len(token)
    assert all(
        span_end <= start or end <= span_start
        for span_start, span_end in _additional_numeric_recall_spans(
            source, corpus_citation_path=UK
        )
    ), source
    cleaned = authoritative_numeric_recall_text(source, corpus_citation_path=UK)
    baseline_source = f"Under this provision, {token} is the limit."
    baseline_text = authoritative_numeric_recall_text(
        baseline_source, corpus_citation_path=UK
    )
    if token in baseline_text:
        assert token in cleaned, source
    # Compare with this profile's existing quantity inventory, including an
    # empty inventory when it does not support this glued lexical form.
    assert [(item.value, item.raw) for item in _inventory(source, UK, profile)] == [
        (item.value, item.raw) for item in _inventory(baseline_source, UK, profile)
    ], source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
def test_generated_glued_units_cannot_become_implicit_citation_targets(
    citation, profile
):
    generator = random.Random(1779100)
    values = (7, 35, 100, *generator.sample(range(2, 999), 4))
    for index, value in enumerate(values):
        reference = f"section {value}C" if index % 2 else f"paragraph {value}(1)"
        for unit in ("EUR", "eur", "pence", "PENCE", "ha", "km", "m", "h", "x", "C"):
            if unit in ("ha", "km", "m", "h", "x", "C"):
                amounts = (f"{value}.5",)
            else:
                amounts = (
                    str(value),
                    f"1.{value:03d}" if unit.lower() == "eur" else f"{value}.5",
                )
            for amount in amounts:
                token = f"{amount}{unit}"
                quantity = f"{token} is the limit."
                baseline_source = f"Under this provision, {quantity}"
                baseline = _inventory(baseline_source, citation, profile)
                baseline_text = authoritative_numeric_recall_text(
                    baseline_source, corpus_citation_path=citation
                )
                for separator in (", ", ","):
                    prefix = f"Under {reference}{separator}"
                    source = f"{prefix}{quantity}"
                    start, end = len(prefix), len(prefix) + len(token)
                    assert all(
                        span_end <= start or end <= span_start
                        for span_start, span_end in _additional_numeric_recall_spans(
                            source, corpus_citation_path=citation
                        )
                    ), source
                    cleaned = authoritative_numeric_recall_text(
                        source, corpus_citation_path=citation
                    )
                    # Keep the baseline inline too: existing sentence-marker
                    # cleaning changes some line-initial glued uppercase forms.
                    if quantity in baseline_text:
                        assert quantity in cleaned, source
                    assert [
                        (item.value, item.raw)
                        for item in _inventory(source, citation, profile)
                    ] == [(item.value, item.raw) for item in baseline], source
                    if profile == "legacy" and baseline:
                        scalars = tuple(dict.fromkeys(item.value for item in baseline))
                        assert _recall_issues(source, citation, profile), source
                        assert not _recall_issues(source, citation, profile, scalars), (
                            source
                        )


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

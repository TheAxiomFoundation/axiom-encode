"""A citation continuation must parse every subdivision before masking it."""

from __future__ import annotations

import random
import re
from collections import Counter
from itertools import product

import pytest
import yaml

from axiom_encode.harness.source_completeness import (
    _additional_numeric_recall_spans,
    _analyze_rulespec_payload,
    _mask_numeric_spans,
    authoritative_numeric_recall_text,
)
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_ungrounded_numeric_issues,
    numeric_value_is_grounded,
)
from tests.test_numeric_recall_structural_properties import (
    PROFILES,
    UK,
    US,
    _inventory,
    _recall_issues,
)

EXPRESSIONS = (
    ("1 + 2", (1, 2)),
    ("1/2", (1, 2)),
    ("1.5", (1.5,)),
    ("1 + rate", (1,)),
    ("rate / 2", (2,)),
    ("rate + adjustment", ()),
)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("reference", ("section 7C", "paragraph 2(1)"))
@pytest.mark.parametrize("expression,inner_values", EXPRESSIONS)
def test_parenthesized_payment_after_citation_retains_its_entire_inventory(
    expression, inner_values, reference, citation, profile
):
    source = f"Under {reference}, 100({expression}) dollars shall be paid."
    occurrences = _inventory(source, citation, profile)
    assert [
        (item.value, item.raw)
        for item in sorted(occurrences, key=lambda item: item.start)
    ] == [
        (100, "100"),
        *((value, str(value)) for value in inner_values),
    ], (source, occurrences)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("reference", ("section 7C", "paragraph 2(1)"))
@pytest.mark.parametrize("expression,inner_values", EXPRESSIONS)
@pytest.mark.parametrize("gate", ("missing", "present"))
def test_parenthesized_payment_after_citation_requires_every_production_scalar(
    expression, inner_values, reference, citation, profile, gate
):
    source = f"Under {reference}, 100({expression}) dollars shall be paid."
    values = (100, *inner_values)
    if gate == "present":
        assert not _recall_issues(source, citation, profile, values), source
    else:
        for missing_index in range(len(values)):
            recalled = values[:missing_index] + values[missing_index + 1 :]
            assert _recall_issues(source, citation, profile, recalled), (
                source,
                recalled,
            )
        assert _recall_issues(source, citation, profile), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("subdivisions", ("(1)", "(1)(a)", "(1)(a)(i)"))
def test_complete_parenthesized_citation_continuation_remains_structural(
    subdivisions, citation, profile
):
    source = f"Under sections 7C, 100{subdivisions} the claimant must qualify."
    assert not _inventory(source, citation, profile), source
    assert not _recall_issues(source, citation, profile), source


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("payment", ("100(1)(rate / 2)", "100(rate)(1.5)"))
def test_incomplete_subdivision_chain_stays_outside_citation_mask(
    payment, citation, profile
):
    prefix = "Under section 7C, "
    quantity = f"{payment} dollars shall be paid."
    source = prefix + quantity
    start, end = len(prefix), len(prefix) + len(payment)
    assert all(
        span_end <= start or end <= span_start
        for span_start, span_end in _additional_numeric_recall_spans(
            source, corpus_citation_path=citation
        )
    ), source
    assert payment in authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    # The extractor already classifies valid leading subdivisions in these
    # chains as references. Preserve that existing inventory independently of
    # the citation mask, which must still leave the complete token intact.
    assert [
        (item.value, item.raw) for item in _inventory(source, citation, profile)
    ] == [(item.value, item.raw) for item in _inventory(quantity, citation, profile)]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
def test_generated_citation_renumbering_preserves_parenthesized_payment_literals(
    citation, profile
):
    """Removing or renumbering a citation preserves every expression literal."""

    generator = random.Random(177902)
    for expression, inner_values in EXPRESSIONS:
        coefficient = generator.randrange(50, 999)
        payment = f"{coefficient}({expression}) dollars shall be paid."
        expected = [(coefficient, str(coefficient))] + [
            (value, str(value)) for value in inner_values
        ]
        values = (coefficient, *inner_values)
        for reference in (
            "",
            f"Under section {coefficient}C, ",
            f"Under paragraph {generator.randrange(2, 999)}(1)(a),",
        ):
            source = reference + payment
            occurrences = sorted(
                _inventory(source, citation, profile), key=lambda item: item.start
            )
            assert [(item.value, item.raw) for item in occurrences] == expected, source
            assert not _recall_issues(source, citation, profile, values), source
            for missing_index in range(len(values)):
                recalled = values[:missing_index] + values[missing_index + 1 :]
                assert _recall_issues(source, citation, profile, recalled), (
                    source,
                    recalled,
                )


@pytest.mark.parametrize("factor", ("100x", "100a"))
def test_separately_grounded_adjustment_cannot_cover_a_binary_subtraction_operand(
    factor,
):
    source = (
        f"Under section 7C, {factor}-2 dollars shall be paid. "
        "A separate adjustment of -2 dollars applies."
    )
    profile = _numeric_profile_for_citation_path(US)
    assert profile == "legacy"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": US}},
        "rules": [
            {
                "name": "adjustment",
                "kind": "parameter",
                "dtype": "Float",
                "source": US,
                "versions": [{"effective_from": "2026-01-01", "formula": "-2"}],
            }
        ],
    }
    content = yaml.safe_dump(payload)
    assert not find_ungrounded_numeric_issues(content, source, source_citation_path=US)
    result = _analyze_rulespec_payload(
        payload,
        content=content,
        source_text=source,
        corpus_citation_path=US,
        test_cases=(),
        extract_numeric_occurrences=extract_typed_numeric_inventory_occurrences_from_text,
        extract_numeric_grounding_occurrences=extract_typed_numeric_occurrences_from_text,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=None,
        artifact_numeric_bindings=None,
        authenticated_same_act_aliases=(),
        imported_symbol_contents=(),
    )
    assert result.source_numeric_occurrence_count == 2
    assert result.covered_source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == 1
    assert any(
        "numeric value 2 has no named scalar" in issue for issue in result.issues
    )
    assert not _recall_issues(source, US, profile, (2, -2))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("citation", (UK, US))
@pytest.mark.parametrize("quantity", ("17.5 dollars", "100.5pence", "100,000 dollars"))
def test_introducer_only_masks_recover_the_complete_numeric_envelope(
    profile, citation, quantity
):
    source = f"Under section {quantity} is the limit."
    baseline = _inventory(
        f"Under this provision, {quantity} is the limit.", citation, profile
    )
    assert quantity in authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    assert [
        (item.value, item.raw) for item in _inventory(source, citation, profile)
    ] == [(item.value, item.raw) for item in baseline]
    if baseline:
        assert _recall_issues(source, citation, profile)
        assert not _recall_issues(
            source, citation, profile, tuple(item.value for item in baseline)
        )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("year", (1900, 1996, 2026))
def test_complete_reference_includes_only_its_own_instrument_year(profile, year):
    source = f"Under section 431 of the Act of {year}, applicants qualify."
    spans = _additional_numeric_recall_spans(source, corpus_citation_path=US)
    assert (source.index(str(year)), source.index(str(year)) + 4) in spans
    assert not _inventory(source, US, profile)
    paired = source + f" A separate payment of {year} dollars applies."
    assert [(item.value, item.raw) for item in _inventory(paired, US, profile)] == [
        (year, str(year))
    ]
    assert _recall_issues(paired, US, profile)
    assert not _recall_issues(paired, US, profile, (year,))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("suffix", (" dollars", "EUR", " percent", " years"))
def test_reference_instrument_year_shaped_quantity_remains_required(profile, suffix):
    source = f"Under section 431 of the Act of 1996{suffix} is the limit."
    start = source.index("1996")
    spans = _additional_numeric_recall_spans(source, corpus_citation_path=US)
    assert all(end <= start or start + 4 <= begin for begin, end in spans)
    baseline = _inventory(f"The limit is 1996{suffix}.", US, profile)
    occurrences = _inventory(source, US, profile)
    # If removing the section would make the extractor erase this operative
    # amount as a title year, retaining the section is the required refusal.
    # Its extra structural scalar must not conceal the separate amount.
    for expected in baseline:
        assert any(
            item.value == expected.value and item.raw == expected.raw
            for item in occurrences
        )
    if baseline:
        values = tuple(item.value for item in occurrences)
        assert not _recall_issues(source, US, profile, values)
        for expected in baseline:
            assert _recall_issues(
                source,
                US,
                profile,
                tuple(value for value in values if value != expected.value),
            )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("quantity", ("17.5 metres", "100.5pence", "100,000 dollars"))
def test_unsafe_target_cannot_disable_an_independent_quantity_recovery(
    profile, quantity
):
    prefix = "Under section 7C, 100x-2 dollars shall be paid. "
    source = (
        prefix + "A separate adjustment of -2 dollars applies. "
        f"Under section 8C(1)(a), {quantity} is the limit."
    )
    occurrences = _inventory(source, US, profile)
    operand_start = prefix.index("-2") + 1
    assert any(
        item.start == operand_start and item.value == 2 and item.raw == "2"
        for item in occurrences
    )
    baseline = _inventory(f"The limit is {quantity}.", US, profile)
    for expected in baseline:
        assert any(
            item.value == expected.value and item.raw == expected.raw
            for item in occurrences
        )
    assert not _recall_issues(
        source, US, profile, (2, -2, *(item.value for item in baseline))
    )
    if baseline:
        assert _recall_issues(source, US, profile, (2, -2))


@pytest.mark.parametrize("profile", PROFILES)
def test_generated_citation_masks_preserve_every_outside_numeric_token(profile):
    """Masks preserve span, value, sign and text, including repeated amounts.

    Hypothesis is not a repository dependency. Use the existing deterministic
    property-test tooling, with exhaustive neighbour classes and seeded values.
    Neutralize citation-introducer words in the reference view, retaining all
    target digits, suffixes and operators. This prevents the extractor's old
    citation regex from clipping an adjacent decimal before comparison.
    """

    generator = random.Random(177904)
    references = ("section 7C", "paragraph 2(1)", "sections 7C and 8C")
    neighbours = (
        "100x-2",
        "100a+2",
        "100(1)-2",
        "100(rate)-2",
        "100(1)(a)-2",
        "100(1 + 2)",
        "100(1/2)",
        "100(rate / 2)",
        "100x−2",
        "100x*2",
        "100x/2",
        "100x^2",
        "100–200",
        "100-200",
        "-2",
        "+2",
        "100EUR",
        "100 pence",
        "100,000",
        "100.5km",
        "100 or 200",
    )
    cases = [
        f"Under {reference}{separator}{neighbour} dollars shall be paid. "
        "A separate adjustment of -2 dollars applies."
        for reference, separator, neighbour in product(
            references, (", ", ",", " — ", " and "), neighbours
        )
    ]
    for _ in range(100):
        amount = generator.randrange(2, 999)
        operand = generator.randrange(1, 99)
        factor = generator.choice(("x", "a", "(1)", "(rate)", "(1)(a)"))
        operator = generator.choice(("-", "+", "−", "*", "/", "^", "–"))
        cases.append(
            f"{operand} dollars applies. Under section {amount}C, "
            f"{amount}{factor}{operator}{operand} dollars shall be paid. "
            f"A separate adjustment of -{operand} dollars applies."
        )
    for source in cases:
        spans = _additional_numeric_recall_spans(source, corpus_citation_path=US)
        masked = _mask_numeric_spans(source, spans)
        # Use a non-citation word of equal length, independently of the mask
        # implementation's blank introducers. Every numeric character and its
        # arithmetic neighbours remain in their original source coordinates.
        reference = re.sub(
            r"\b(?:sections?|paragraphs?)\b",
            lambda match: (
                "q" * len(match.group())
                if any(start <= match.start() < end for start, end in spans)
                else match.group()
            ),
            source,
        )
        outside = Counter(
            (item.start, item.end, item.value, item.raw)
            for item in extract_typed_numeric_inventory_occurrences_from_text(
                reference, profile=profile
            )
            if not any(item.start < end and start < item.end for start, end in spans)
        )
        after = Counter(
            (item.start, item.end, item.value, item.raw)
            for item in extract_typed_numeric_inventory_occurrences_from_text(
                masked, profile=profile
            )
        )
        assert outside <= after, (source, spans, outside, after)
        assert all(
            source[index] == masked[index]
            for index in range(len(source))
            if not any(start <= index < end for start, end in spans)
        )

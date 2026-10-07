"""A citation continuation must parse every subdivision before masking it."""

from __future__ import annotations

import random

import pytest

from axiom_encode.harness.source_completeness import (
    _additional_numeric_recall_spans,
    authoritative_numeric_recall_text,
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

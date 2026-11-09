"""Closed printed references affect only evidenced locator/operator spans."""

from pathlib import Path

import pytest

from axiom_encode.form_numeric_structure import form_numeric_locator_spans
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.printed_source_rows import recognize_printed_rows

P = Path(__file__).parent / "fixtures/form-numeric-locators"


def formula_rows(text):
    return s._source_formula_branches(
        text, branches=(), active_branches=(), deferred_paths=set()
    )


@pytest.mark.parametrize("profile", ["legacy", "en-US", "da-DK"])
@pytest.mark.parametrize(
    "extract",
    [
        v.extract_typed_numeric_inventory_occurrences_from_text,
        v.extract_typed_numeric_occurrences_from_text,
    ],
)
def test_actual_parenthetical_references_preserve_real_values_and_offsets(
    profile, extract
):
    text = (P / "step-2.txt").read_text()
    assert not any(x.value in (3, 4) for x in extract(text, profile=profile))
    text = text.replace("5000-R", "Deduct 1074 dollars.\n5000-R")
    found = extract(text, profile=profile)
    assert [x.value for x in found if not x.raw.isalpha()] == [1074]
    assert all(text[x.start : x.end] == x.raw for x in found)


@pytest.mark.parametrize(
    "ref",
    [
        "(complete Schedule 3.5)",
        "(complete Schedule 3,5)",
        "(complete Schedule 3 dollars)",
        "(complete Schedule 3%)",
        "(complete Schedule 3 years)",
        "(complete Schedule 3 kg)",
        "(complete Schedule 3",
        "((complete Schedule 3))",
        "(discussion (complete Schedule 3)",
        "(discussion (complete Schedule 3))",
        "discussion) (complete Schedule 3)",
        "(complete Schedule 3) discussion(",
        "(complete Schedule 3))",
        '"(complete Schedule 3)"',
        "(complete Schedule 3) + 4",
        "(complete Schedule 3) dollars",
        "(complete Schedule 3).5",
    ],
)
def test_incomplete_or_amount_reference_produces_no_new_locator_span(ref):
    text = (P / "step-2.txt").read_text().replace("(complete Schedule 3)", ref)
    start = text.index(ref)
    end = start + len(ref)
    assert not any(start <= a < b <= end for a, b in form_numeric_locator_spans(text))


def test_actual_referral_and_descriptor_rows_are_not_computations():
    for step in (2, 3):
        text = (P / f"step-{step}.txt").read_text()
        found = formula_rows(text)
        assert not any(
            "Support payments" in b.text or "Taxable FHSA income - other" in b.text
            for b in found
        )
        assert any("Add lines" in b.text for b in found)
        assert all(text[b.start : b.end] == b.text for b in found)


@pytest.mark.parametrize(
    "caption",
    [
        "income - expenses",
        "income - other",
        "Divide income by expenses",
        "income / expenses",
        "Taxable income - other + amount (see the T4FHSA slip) ^",
        "income - expenses (go to canada.ca/taxes-support-payments)",
        "Divide income by expenses (go to canada.ca/taxes-support-payments)",
    ],
)
def test_real_arithmetic_survives_valid_referral(caption):
    text = (P / "step-2.txt").read_text()
    old = "Taxable FHSA income - other (see the T4FHSA slip) ^"
    replacement = caption[0].upper() + caption[1:]
    text = text.replace(old, replacement)
    context = recognize_printed_rows(text)
    classified = list(replacement)
    start = text.index(replacement)
    for a, b in context.typography_spans:
        if start <= a < b <= start + len(replacement):
            classified[a - start : b - start] = s._TYPOGRAPHY_MASK * (b - a)
    assert s.source_states_explicit_computation("".join(classified))
    if not replacement.startswith("Divide"):
        assert any(replacement in b.text for b in formula_rows(text))


@pytest.mark.parametrize(
    "caption",
    [
        "Support payments (go to canada.ca/)",
        "Support payments (go to canada.ca/taxes + amount)",
        "Support payments ((go to canada.ca/taxes-support))",
        'Support payments "(go to canada.ca/taxes-support)"',
        "Taxable income - other (see the slip) ^",
        "Taxable income - other (see the T4FHSA slip)",
        "Taxable income - other (see the T4FHSA slip) ^ + 3",
    ],
)
def test_unknown_caption_has_no_new_typography_span(caption):
    text = (
        (P / "step-2.txt")
        .read_text()
        .replace("Taxable FHSA income - other (see the T4FHSA slip) ^", caption)
    )
    context = recognize_printed_rows(text)
    start = text.index(caption)
    end = start + len(caption)
    assert context is not None
    assert not any(start <= a < b <= end for a, b in context.typography_spans)


def test_no_grid_no_descriptor_typography():
    assert (
        recognize_printed_rows("Taxable income - other (see the T4FHSA slip) ^") is None
    )

"""In US agency manuals a slash or spaced dash between words is prose.

The arithmetic recognizer read `case/individual`, `SNAP/Cash` and
`Processing - SNAP` as division and subtraction, so manual pages carried
"explicit source computation" obligations that no encoding can satisfy (741
slash and 211 dash clauses on the Oregon/Utah SNAP dispatch pilot). On US
manual paths, the scope of #1756, those spans are now masked like other manual
typography. Statutes, regulations and non-US sources keep the generic reading.

The replay fixture is the byte-exact ``final-rejected-candidate`` of run
36877819271 (Oregon Programs Eligibility Notebook page 93), whose clause 2
(`recertification/ renewal`) asked for a computation.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness import validator_pipeline as validator_pipeline_module
from axiom_encode.harness.coverage_index import format_coverage_index
from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    source_states_explicit_computation,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
COMPUTATION = "[complete-source-unit:formula-output] Explicit source computation"
EMPTY_CANDIDATE = "format: rulespec/v1\nmodule:\n  summary: probe\nrules: []\n"
MANUAL = "us-or/manual/odhs/open/page-62"
STATUTE = "us/statute/7/2014"


def _issues(content: str, source: str, citation: str, test_cases) -> list[str]:
    return list(
        analyze_complete_source_unit(
            content,
            source,
            corpus_citation_path=citation,
            test_cases=test_cases,
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile="legacy",
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile="legacy"
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ).issues
    )


def _computations(source: str, citation: str) -> list[str]:
    return [
        issue
        for issue in _issues(EMPTY_CANDIDATE, source, citation, [])
        if issue.startswith(COMPUTATION)
    ]


def test_page_93_slash_clause_is_not_a_computation():
    fixture = FIXTURES / "signed_reencode_36877819271"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]
    source = (fixture / "source.txt").read_text()
    clauses = {
        index: clause
        for index, (_start, _end, clause) in enumerate(
            completeness_module._source_clause_spans(source, branches=()),
            start=1,
        )
    }

    issues = _issues(content, source, recorded["citation"], yaml.safe_load(tests))

    assert "recertification/ renewal" in clauses[2]
    assert not [issue for issue in issues if issue.startswith(COMPUTATION)]


@pytest.mark.parametrize(
    "source",
    (
        "ECF Electronic Case File Where documents associated to a case/individual "
        "are stored.",
        "The SNAP/Cash worker reviews the application.",
        "Either the parent or he/she signs the form.",
        "Processing - SNAP applications follow the standard timeline.",
        "Application Status – Pending is shown in the system.",
    ),
)
def test_manual_word_slash_and_dash_are_prose(source: str):
    assert not _computations(source, MANUAL)
    # Outside US manuals the generic reading is unchanged.
    assert _computations(source, STATUTE)
    assert _computations(source, "uk-harrow/manual/council-tax-reduction-2026")


@pytest.mark.parametrize(
    "source",
    (
        "Net income = gross income - 100.",
        "The monthly amount is the annual income / 12.",
        "The allotment is 100 - 20.",
        "Countable income equals gross income minus allowable deductions.",
    ),
)
def test_manual_arithmetic_stays_a_computation(source: str):
    assert _computations(source, MANUAL)


@pytest.mark.parametrize(
    "source",
    (
        # Formulas that main read as computations only through a word slash or
        # spaced dash, because `$` operands, `÷` and parentheses are not
        # recognized (corpus differential, review of this PR).
        "Family member allowance = (monthly maintenance needs standard - family "
        "member income) ÷ 3.",
        "In this situation, the actual value of the S/M is $216.67 or $300 CMRV - "
        "$83.33 client's voluntary payment = $216.67.",
        # `÷` alone also states an equation.
        "The allowance is (monthly need standard - family income) ÷ 3.",
    ),
)
def test_manual_equation_keeps_its_slashes_and_dashes(source: str):
    assert _computations(source, MANUAL)


def test_web_address_equals_sign_does_not_keep_word_slashes():
    assert not _computations(
        "See https://example.gov/forms?s=7 for the SNAP/Cash worker guide.", MANUAL
    )


@pytest.mark.parametrize(
    ("citation", "expected"),
    ((MANUAL, True), (STATUTE, False)),
)
def test_stated_conversion_hint_reads_the_source_like_the_validator(
    citation: str, expected: bool
):
    # The hint takes the artifact's corpus path when the pipeline has no
    # attested source path, as the complete-source-unit analysis does.
    content = (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {citation}\n  summary: probe\nrules: []\n"
    )
    source = (
        "The annual standard is $2,400. Converted to the monthly SNAP/Cash "
        "amount, the result is 200."
    )

    hint = validator_pipeline_module._stated_conversion_ungrounded_literal_hint(
        content,
        source,
        12.0,
        source_citation_path=None,
        require_complete_source_unit=True,
    )

    assert bool(hint) is expected


def test_word_only_manual_subtraction_is_prose():
    # Decided trade-off: a formula written only with words and a spaced dash
    # reads as prose in a US manual.
    assert not _computations("Countable income - allowable deductions applies.", MANUAL)


def test_scope_ends_with_the_analysis():
    _issues(EMPTY_CANDIDATE, "The SNAP/Cash worker reviews it.", MANUAL, [])

    assert source_states_explicit_computation("The SNAP/Cash worker reviews it.")
    assert completeness_module._US_MANUAL_PROSE_SCOPE.get() is False


def test_scope_masks_keep_offsets():
    source = "Case/individual records - Processing"
    token = completeness_module._US_MANUAL_PROSE_SCOPE.set(True)
    try:
        masked = completeness_module._without_manual_typography_operators(source)
    finally:
        completeness_module._US_MANUAL_PROSE_SCOPE.reset(token)

    assert len(masked) == len(source)
    assert "/" not in masked
    assert " - " not in masked
    assert masked.startswith("Case") and masked.endswith("Processing")
    # The sentinel, unlike a space, keeps the words apart (#1771).
    assert masked == "Case\x00individual records\x00\x00\x00Processing"


@pytest.mark.parametrize(
    ("citation", "coordinates"),
    ((MANUAL, 0), (STATUTE, 1)),
)
def test_prompt_coverage_index_reads_the_source_like_the_validator(
    citation: str, coordinates: int
):
    index = format_coverage_index("The SNAP/Cash worker reviews it.", citation)

    assert f"formula_coordinates={coordinates}" in index

"""In US agency manuals only a stated numeric limit is a source-stated control.

The complete-source-unit check asked for a derived output for every `if`,
`when` and `unless` sentence. In agency manuals those sentences are procedure
(`When child care hours are requested due to schooling, verification of
registration and current class schedule are required.`), and the demand blocked
693 of the 955 units of the Oregon/Utah SNAP dispatch pilot. Numeric limits
(`more than 50% of their meals`) remain controls everywhere, and statutes and
non-US manuals keep exception controls.

The replay fixture is the byte-exact ``final-rejected-candidate`` of run
36880457452 (Utah eligibility manual 770-1), whose only remaining
complete-source-unit issue on main was the procedural control.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
CONTROL = "[complete-source-unit:formula-output] Source-stated control"
EMPTY_CANDIDATE = "format: rulespec/v1\nmodule:\n  summary: probe\nrules: []\n"
PROCEDURE = (
    "When child care hours are requested due to schooling, verification of "
    "registration and current class schedule are required."
)
NUMERIC_LIMIT = (
    "Residents of institutions that provide more than 50% of their meals are "
    "not eligible."
)
OREGON_PAGE = "us-or/manual/odhs/open/page-644"
_LIMIT_CANDIDATE = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: {CITATION}
  summary: probe
inputs:
- name: institution_meal_share_percent
  entity: Person
  dtype: Decimal
  period: Month
rules:
- name: institution_meal_share_limit_percent
  kind: parameter
  dtype: Decimal
  source: Institutional Residents
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: parameter
        source:
          corpus_citation_path: {CITATION}
          excerpt: more than 50% of their meals
  versions:
  - effective_from: '0001-01-01'
    formula: '50'
- name: institution_resident_ineligible
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  source: Institutional Residents
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: condition
        source:
          corpus_citation_path: {CITATION}
          excerpt: Residents of institutions that provide more than 50% of their
            meals are not eligible.
  versions:
  - effective_from: '0001-01-01'
    formula: 'institution_meal_share_percent > institution_meal_share_limit_percent'
"""
_LIMIT_TESTS = """
- name: at limit
  period: 2026-05
  input:
    us-or:policies/odhs/open/page-999#input.institution_meal_share_percent: 50
  output:
    us-or:policies/odhs/open/page-999#institution_resident_ineligible: not_holds
- name: above limit
  period: 2026-05
  input:
    us-or:policies/odhs/open/page-999#input.institution_meal_share_percent: 51
  output:
    us-or:policies/odhs/open/page-999#institution_resident_ineligible: holds
"""


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


def _controls(source: str, citation: str) -> list[str]:
    return [
        issue
        for issue in _issues(EMPTY_CANDIDATE, source, citation, [])
        if issue.startswith(CONTROL)
    ]


def test_utah_770_1_rejection_has_no_procedural_control():
    fixture = FIXTURES / "signed_reencode_36880457452"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]

    issues = _issues(
        content,
        (fixture / "source.txt").read_text(),
        recorded["citation"],
        yaml.safe_load(tests),
    )

    assert any(CONTROL in issue for issue in recorded["issues"])
    assert not [issue for issue in issues if CONTROL in issue]


def test_manual_procedure_is_not_a_control():
    assert not _controls(PROCEDURE, OREGON_PAGE)


@pytest.mark.parametrize(
    "source",
    (
        NUMERIC_LIMIT,
        # Limits that only the exception cue made controls on main (review of
        # #1833): the boundary recognizer does not read these comparators.
        "Households are eligible unless income exceeds $500.",
        "Households are eligible unless resources are over $2,750.",
        "A vehicle is excluded if its fair market value is $4,650 or less.",
        "The household is ineligible if income is greater than $3,000.",
        "The household is exempt if it has fewer than 2 members.",
        "Shelter costs are deducted only if they are in excess of $744.",
        "A child is exempt if under age 18.",
        "A person is exempt if age 55 or older.",
        "A person is exempt if employed 20 or more hours a week.",
        # Shapes from the corpus (review round 2 of #1833).
        "The resident is eligible if he spends fifteen (15) or more calendar "
        "days in the shelter.",
        "The property is excluded if its value is fifty percent (50%) or less "
        "of the limit.",
        "If the minor dependent child is age 16 to 18, school attendance is required.",
        "A child is eligible if 21 years of age and under.",
        "The household is exempt if all household members are age 60 and over.",
        "A parent is exempt if the child is younger than 6.",
        "An adult is subject to the time limit if between the ages of 18 and 49.",
        "A parent is exempt if caring for a child under one year of age.",
    ),
)
def test_manual_numeric_limit_is_still_a_control(source: str):
    assert _controls(source, OREGON_PAGE)


@pytest.mark.parametrize(
    ("text", "expected"),
    (
        ("unless income exceeds $500", True),
        ("if its fair market value is $4,650 or less", True),
        ("if under age 18", True),
        ("if age 55 or older", True),
        ("if employed 20 or more hours a week", True),
        ("if s/he spends fifteen (15) or more calendar days", True),
        ("value of fifty percent (50%) or less", True),
        ("If the minor dependent child is age 16 to 18,", True),
        ("A child 21 years of age and under", True),
        ("all household members are age 60 and over", True),
        ("if the child is younger than 6", True),
        ("if between the ages of 18 and 49", True),
        ("caring for a child under one year of age", True),
        ("if two or more members apply", True),
        (PROCEDURE, False),
        ("When the household moves, update the address.", False),
        # References after a comparator are not limits.
        ("If the client calls over the phone at 1-800-555-1212, record it.", False),
        ("Eligible under subsection (1)(a) or (b) of this section.", False),
        ("The resource is excluded under WAC 388-450-0140 if verified.", False),
        ("If you did not check any box above, go to Step 3.", False),
        ("The waiver applies under section 1915(c) of the Act.", False),
        ("Studies must comply with the rules under 42 CFR 50.", False),
    ),
)
def test_stated_numeric_limit(text: str, expected: bool):
    assert bool(completeness_module._STATED_NUMERIC_LIMIT.search(text)) is expected


def test_manual_procedure_needs_no_paired_test_cases():
    # A candidate that encodes the page's numeric limit with a derived rule
    # must not be asked for positive/blocking cases for the procedure.
    source = f"Institutional Residents\n\n{NUMERIC_LIMIT}\n\n{PROCEDURE}\n"

    def paired_case_issues(citation: str) -> list[str]:
        return [
            issue
            for issue in _issues(
                _LIMIT_CANDIDATE.replace("{CITATION}", citation),
                source,
                citation,
                yaml.safe_load(_LIMIT_TESTS),
            )
            if "require paired positive/blocking cases" in issue
        ]

    assert not paired_case_issues("us-or/manual/odhs/open/page-999")
    assert paired_case_issues("us-or/regulation/461/135/9999")


@pytest.mark.parametrize(
    "citation",
    (
        "us/statute/7/2015",
        "us-or/regulation/461/135/0750",
        "uk-harrow/manual/council-tax-reduction-scheme-2026-2027",
    ),
)
def test_procedure_stays_a_control_outside_us_manuals(citation: str):
    assert _controls(PROCEDURE, citation)


@pytest.mark.parametrize(
    ("citation", "expected"),
    (
        ("us-or/manual/odhs/open/page-93", True),
        ("us-ut/manual/dws/eligibility-manual/510", True),
        ("us/manual/fns/snap-quality-control", True),
        ("us/statute/7/2015", False),
        ("uk-harrow/manual/council-tax-reduction-scheme-2026-2027", False),
        ("de/manual/x", False),
        ("", False),
    ),
)
def test_us_manual_citation_path(citation: str, expected: bool):
    assert completeness_module._is_us_manual_citation_path(citation) is expected

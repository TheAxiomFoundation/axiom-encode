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


def test_manual_numeric_limit_is_still_a_control():
    assert _controls(NUMERIC_LIMIT, OREGON_PAGE)


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

"""Agency-manual typography must not create formula-output obligations.

The generic arithmetic recognizer reads ``/``, ``-``, ``*`` and ``•`` as
operators and accepts words as operands, so ordinary manual typography became
"explicit source computation" clauses that no encoding can satisfy: the
"93 (07/2026)" header on every page of the Oregon Programs Eligibility
Notebook, list bullets between words, SharePoint and agency web addresses, and
footnote asterisks. Every Oregon page in the SNAP dispatch pilot carried at
least one such obligation.

The fixtures are the byte-exact ``final-rejected-candidate`` files from the
``targeted-reencode-failure-<run>-1`` artifacts of runs 36877819271 (page 93)
and 36791259738 (page 26), with the ``source.txt`` the encoder resolved (sha256
values of the candidates recorded in each ``issues.json``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
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
FORMULA_CLAUSE = re.compile(r"source unit formula clause (\d+) in ")


def _analyze(content: str, source: str, citation: str, test_cases):
    return analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=test_cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _replay(run_id: str):
    fixture = FIXTURES / f"signed_reencode_{run_id}"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]
    source = (fixture / "source.txt").read_text()
    result = _analyze(content, source, recorded["citation"], yaml.safe_load(tests))
    return recorded, source, list(result.issues)


def _formula_clauses(issues) -> set[int]:
    return {
        int(match.group(1))
        for issue in issues
        if "[complete-source-unit:formula-output]" in issue
        for match in [FORMULA_CLAUSE.search(issue)]
        if match
    }


def test_page_26_rejection_was_typography_only():
    recorded, _source, issues = _replay("36791259738")

    assert _formula_clauses(recorded["issues"]) == {1, 8, 11}
    assert not [
        issue for issue in issues if "[complete-source-unit:formula-output]" in issue
    ]


def test_page_93_no_longer_asks_for_typography_formulas():
    recorded, source, issues = _replay("36877819271")
    spans = {
        index: clause
        for index, (_start, _end, clause) in enumerate(
            completeness_module._source_clause_spans(source, branches=()),
            start=1,
        )
    }

    assert {1, 8, 11} <= _formula_clauses(recorded["issues"])
    # Clause 1 is the "93 (07/2026)" header plus a bullet, clause 8 a bullet
    # list of paper applications, clause 11 a footnote asterisk.
    assert "07/2026" in spans[1] and "•" in spans[1]
    assert "•" in spans[8]
    assert "*includes" in spans[11]
    assert not _formula_clauses(issues) & {1, 8, 11}


@pytest.mark.parametrize(
    "source",
    (
        "93 (07/2026) Chapter 2:Eligibility • Section 1: Application",
        "Policy effective 07/2026 for all households.",
        "Individuals can use the following paper applications: • DHS 0415F "
        "Application (for SNAP, ERDC, REF, TANF, TA-DVS), • DHS 7476 Employment "
        "Related Day Care and • OHP 7210 Application.",
        "The manual: • Serves as a simplified explanation of federal and state "
        "regulations • Includes information on programs",
        "Other resources Application Status *includes screenshots: Describes "
        "how to search for an application.",
        "Income verification* is required before approval.",
        "SNAP Staff Tools: https://dhsoha.sharepoint.com/teams/Hub-DHS-ET/"
        "SitePages/SNAP.aspx",
        "See oregon.gov/odhs/food/pages/snap-benefits.aspx for details.",
    ),
)
def test_manual_typography_is_not_a_computation(source: str):
    assert not source_states_explicit_computation(source)


@pytest.mark.parametrize(
    "source",
    (
        # The § 32a EStG tariff multiplies with a bullet-shaped operator.
        "2. von 12 349 Euro bis 17 799 Euro:(914,51 • y + 1 400) • y;",
        "4. von 69 879 Euro bis 277 825 Euro:0,42 • x – 11 135,63;",
        "The factor is 2 • 3.",
        "The allotment is 100 - 20.",
        "The benefit is 2 × 3 dollars.",
        "E = Z × F",
        "A household receives 1/2 of the amount.",
        "The result is 2 * 3.",
        "The amount is computed by dividing income by the divisor.",
    ),
)
def test_real_arithmetic_stays_a_computation(source: str):
    assert source_states_explicit_computation(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("Eligibility • Section", True),
        ("applications: • DHS 0415F", True),
        ("• Serves as a summary", True),
        ("aspx • TANF Staff Tools", True),
        ("(914,51 • y + 1 400)", False),
        (") • y", False),
        ("0,42 • x", False),
        ("2 • 3", False),
        ("Section 1 • Intent", False),
    ),
)
def test_list_bullet_is_told_from_multiplication(source: str, expected: bool):
    index = source.index("•")
    assert completeness_module._is_list_bullet(source, index, index + 1) is expected


def test_masking_keeps_offsets():
    source = (
        "93 (07/2026) Eligibility • Section, see https://example.gov/a-b/c *includes"
    )

    masked = completeness_module._without_manual_typography_operators(source)

    assert len(masked) == len(source)
    assert "07/2026" not in masked
    assert "https" not in masked
    assert "•" not in masked
    assert "*" not in masked
    assert masked.startswith("93 (")

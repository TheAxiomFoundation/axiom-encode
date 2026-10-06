"""US manual locators are not demanded as named scalars.

Two SNAP dispatch pilot runs were rejected on complete-source-unit numeric
recall for numbers that only locate text:

* run 36880457452, Utah eligibility manual 770-1: `2`, `3` and `770` from the
  `policy 770-2` / `policy 770-3` cross-references;
* run 36877819271, Oregon Programs Eligibility Notebook page 93: `93`, `7` and
  `2026` from the page header `93 (07/2026)`.

The fixtures are the byte-exact ``final-rejected-candidate`` files from the
``targeted-reencode-failure-<run>-1`` artifacts of those runs, with the
``source.txt`` the encoder resolved (sha256 values of the candidates recorded
in each ``issues.json``).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.source_completeness import (
    analyze_complete_source_unit,
    authoritative_numeric_recall_text,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURES = Path(__file__).parent / "fixtures/source_completeness"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)
OREGON_PAGE = "us-or/manual/odhs/open/page-93"


def _recall_values(issues) -> set[str]:
    return {
        match.group(1)
        for issue in issues
        for match in [RECALL_VALUE.search(issue)]
        if match
    }


def _replay(run_id: str):
    fixture = FIXTURES / f"signed_reencode_{run_id}"
    recorded = json.loads((fixture / "issues.json").read_text())
    content = (fixture / "candidate.yaml").read_text()
    tests = (fixture / "candidate.test.yaml").read_text()
    assert hashlib.sha256(content.encode()).hexdigest() == recorded["rulespec_sha256"]
    assert hashlib.sha256(tests.encode()).hexdigest() == recorded["tests_sha256"]
    result = analyze_complete_source_unit(
        content,
        (fixture / "source.txt").read_text(),
        corpus_citation_path=recorded["citation"],
        test_cases=yaml.safe_load(tests),
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return recorded, list(result.issues)


@pytest.mark.parametrize(
    ("run_id", "locator_values"),
    (
        ("36880457452", {"2", "3", "770"}),
        ("36877819271", {"7", "93", "2026"}),
    ),
)
def test_pilot_rejections_no_longer_demand_locator_values(run_id, locator_values):
    recorded, issues = _replay(run_id)

    assert locator_values <= _recall_values(recorded["issues"])
    assert not _recall_values(issues) & locator_values


@pytest.mark.parametrize(
    "source",
    (
        "93 (07/2026) Chapter 2:Eligibility • Section 1: Application Introduction",
        "26 Chapter 1: Introduction to the Oregon Programs Eligibility Notebook "
        "• Table of contents (07/2026) Introduction to the manual",
    ),
)
def test_manual_page_header_is_not_a_value(source: str):
    cleaned = authoritative_numeric_recall_text(
        source, corpus_citation_path=OREGON_PAGE
    )

    assert not re.search(r"\b(?:93|26|07|2026)\b", cleaned)


@pytest.mark.parametrize(
    ("source", "citation", "kept"),
    (
        # A leading number that is not followed by a stamp or a chapter heading
        # is content.
        ("30 days after the application date", OREGON_PAGE, "30"),
        ("5 percent of income (07/2026)", OREGON_PAGE, "5"),
        # Outside manuals the page-header shape is not assumed.
        ("93 (07/2026) Chapter 2", "us/statute/7/2014", "93"),
        ("93 (07/2026) Chapter 2", "de/statute/estg/32a", "93"),
    ),
)
def test_values_near_the_header_shape_are_kept(source, citation, kept):
    cleaned = authoritative_numeric_recall_text(source, corpus_citation_path=citation)

    assert re.search(rf"\b{kept}\b", cleaned)

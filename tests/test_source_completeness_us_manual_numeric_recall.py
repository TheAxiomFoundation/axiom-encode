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
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
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


def _legacy_values(source: str, citation: str) -> set[float]:
    cleaned = authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            cleaned, profile="legacy"
        )
    }


def test_values_after_a_masked_page_header_stay_in_recall():
    # Oregon notebook page 82 is one line; with its page number removed it
    # began with `Chapter 1:` and was dropped whole as a structural heading.
    source = (
        "82 Chapter 1: Introduction to the Oregon Programs Eligibility Notebook "
        "\u2022 Section 4: Glossary and acronyms (07/2026) Households may receive "
        "up to $3200.00 over a 90-day certification period."
    )

    values = _legacy_values(source, "us-or/manual/odhs/open/page-82")

    assert {3200, 90} <= values
    assert not values & {82, 7, 2026}


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        # A dotted policy label is structural; it must not leave `.01` behind.
        ("Policy 24.01 Definitions apply to all programs.", set()),
        # A comma-grouped amount after `policy` is a value.
        ("The insurance policy 6,740 Total assets", {6740}),
        # A range that a unit word follows is not a list continuation.
        ("Refer to policy 254, 10-15 days apply.", {10, 15}),
        # A reference never crosses a line.
        ("countable resources policy\n\n130% gross income limit", {1.3}),
        # The whole hyphen chain is the identifier.
        ("POLICY 05-1-2024 update", set()),
        ("Use Form IL-482-0634 and Form FNS-380-1.", set()),
        # Backtracking must not leave a negative remainder.
        ("See policy 770-2.5 for the rate.", {770, 2.5}),
        ("Under policy 770-2,500 applies.", {770, 2500}),
        # A unit word after a range or identifier, even two words later.
        ("Refer to policy 254, 10-15 business days apply.", {10, 15}),
        # The extractor reads `10-15%` as 10 and 15 percent in any text.
        ("Refer to policy 254, 10-15% applies.", {10, 0.15}),
        ("See policy 770-2, 3-4 household members.", {3, 4}),
        ("under this policy 30 days apply", {30}),
        # A hyphenated unit is still a unit (review of #1756).
        ("Under this policy 10-day advance notice is required.", {10}),
        ("The policy 12-month certification period applies.", {12}),
        ("Under this policy 10-calendar-day notice applies.", {10}),
    ),
)
def test_policy_and_form_references_mask_whole_identifiers_only(source, expected):
    assert _legacy_values(source, "us-wa/manual/dshs/eaz/example") == expected


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        ("24-60 MONTH TIME LIMIT", "us-la/manual/dcfs/fitap/page-7", {24, 60}),
        ("0.3 PERCENT MAP INCREASE", "us-ca/manual/cdss/acl/page-3", {0.003}),
        ("75.38 AABD cash payment", "us-il/manual/dhs/csmm/18929", {75.38}),
        (
            "44.00 Countable Earned Income (earned income disregard and "
            "recognized employment expenses were allowed)",
            "us-il/manual/dhs/csmm/18929/block-4",
            {44},
        ),
        # A cents-shaped label beside a `$` amount line is a budget row,
        # whatever the case of its label (review of #1756).
        (
            "$470.00 Supplemental Security Income (SSI)\n44.50 Countable "
            "Earned Income (earned income disregard)\n10.70 AABD cash payment",
            "us-il/manual/dhs/csmm/18929/block-4",
            {470, 44.5, 10.7},
        ),
        (
            "$313.00 Supplemental Security Income (SSI)\n\n75.38 AABD Cash Payment",
            "us-il/manual/dhs/csmm/18929/block-4",
            {313, 75.38},
        ),
        # The `$` line may also follow the row.
        (
            "44.50 Countable Earned Income\n$470.00 Supplemental Security Income (SSI)",
            "us-il/manual/dhs/csmm/18929/block-4",
            {44.5, 470},
        ),
        # Only a cents-shaped label is a budget row; a hyphenated heading
        # beside a `$` line stays a locator.
        (
            "$470.00 Supplemental Security Income (SSI)\n770-1 Advance Notice "
            "of Adverse Action",
            "us-ut/manual/dws/snap/page-1",
            {470},
        ),
        # Consecutive cents-shaped section headings stay locators.
        (
            "101.03 Application Process\n101.04 Retroactive Applications",
            "us-sc/manual/scdhhs/mppm/page-1",
            set(),
        ),
        ("1-2 Person Household", "us-xx/manual/agency/page-1", {1, 2}),
        ("3-5 Business Days", "us-xx/manual/agency/page-1", {3, 5}),
        # Outside manuals, numbered-heading masking does not apply.
        ("214.3 Telephone Allowance", "us-ca/guidance/cdss/acl-2024-24-55", {214.3}),
    ),
)
def test_table_rows_are_not_section_headings(source, citation, expected):
    assert _legacy_values(source + "\n", citation) == expected

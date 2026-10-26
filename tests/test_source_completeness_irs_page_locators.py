"""An IRS PDF page's printed page number is a locator, not a recall demand.

The excerpts are the opening bytes of the Rev. Proc. 2025-32 and Notice
2025-67 page sources in the offline US replay benchmark (#1891, fixtures at
359a2c907). Since 0.2.2217
the extractor no longer treats a number alone on a line as structural, so each
page started to demand its own page number.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

REV_PROC = "us/guidance/irs/rev-proc-2025-32"
NOTICE = "us/guidance/irs/notice-2025-67"

REPLAY_PAGES = (
    (
        f"{REV_PROC}/page-13",
        "13\n\n.03 Maximum Capital Gains Rate (§ 1(h), § 1(j)(5)). For taxable "
        "years beginning in 2026, the maximum zero rate amounts and maximum 15 "
        "percent rate amounts under § 1(j)(5)(B), as adjusted for inflation, "
        "are as follows:",
        13,
    ),
    (
        f"{REV_PROC}/page-14",
        "14\n\n(3) Refundable Portion. For taxable years beginning in 2026, the "
        "amount used in § 23(a)(4) to determine the amount of the credit under "
        "§ 23 that may be refundable is $5,120.",
        14,
    ),
    (
        f"{REV_PROC}/page-15",
        "15\n\nNumber of Qualifying Children Item One Two Three or More None "
        "Earned Income Amount $13,020 $18,290 $18,290 $8,680",
        15,
    ),
    (
        f"{REV_PROC}/page-18",
        "18\n\n2002-1 C.B. 1098. If the employer provides fuel or otherwise "
        "reimburses fuel expenses, an amount up to $14 per hour is deemed "
        "substantiated if paid in accordance with Rev. Proc. 2002-41.",
        18,
    ),
    (
        f"{NOTICE}/page-4",
        "4\n\nhousehold is increased from $35,625 to $36,375; the limitation "
        "under section 25B(b)(1)(B) is increased from $38,250 to $39,375;",
        4,
    ),
)


def _issues(source, citation, *, scalars=()):
    candidate = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "rules": [
                {"name": f"supplied_{index}", "versions": [{"formula": value}]}
                for index, value in enumerate(scalars)
            ],
        }
    )
    return sc.analyze_complete_source_unit(
        candidate,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    ).issues


def _demanded(issues):
    prefix = "[complete-source-unit:numeric-recall] Authoritative corpus numeric value "
    return {
        float(issue[len(prefix) :].split(" ", 1)[0])
        for issue in issues
        if issue.startswith(prefix)
    }


def _inventory(source, citation):
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    return [
        (item.value, item.start)
        for item in extract_typed_numeric_inventory_occurrences_from_text(
            recall, profile="en-US"
        )
    ]


@pytest.mark.parametrize(
    ("citation", "source", "page"), REPLAY_PAGES, ids=[c for c, *_ in REPLAY_PAGES]
)
def test_replay_page_number_is_not_demanded(citation, source, page):
    recall = sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    assert recall.startswith(" " * len(str(page)) + "\n")
    assert (page, 0) not in _inventory(source, citation)
    # Main's inventory without the citation keeps every other occurrence.
    unmasked = _inventory(source, "")
    assert (page, 0) in unmasked
    assert _inventory(source, citation) == [
        item for item in unmasked if item != (page, 0)
    ]
    assert page not in _demanded(_issues(source, citation))


REPLAY_PAGE_DEMANDS = {
    f"{REV_PROC}/page-13": {15},
    f"{REV_PROC}/page-14": {5120},
    f"{REV_PROC}/page-15": {8680, 13020, 18290},
    f"{REV_PROC}/page-18": {1, 14, 1098, 2002},
    f"{NOTICE}/page-4": {35625, 36375, 38250, 39375},
}


@pytest.mark.parametrize(
    ("citation", "source", "page"), REPLAY_PAGES, ids=[c for c, *_ in REPLAY_PAGES]
)
def test_replay_page_amounts_stay_demanded(citation, source, page):
    assert _demanded(_issues(source, citation)) == REPLAY_PAGE_DEMANDS[citation]


@pytest.mark.parametrize("line_break", ("\n", "\r\n"), ids=("lf", "crlf"))
@pytest.mark.parametrize(
    "continuation",
    (
        # Currency.
        "dollars per hour may be reimbursed.",
        "cents per mile is the standard rate.",
        # Percentage.
        "percent of taxable income is the maximum payment.",
        "% of taxable income is the maximum payment.",
        # Duration.
        "months is the maximum benefit duration.",
        "years after the close of the taxable year.",
        "days after the date of the notice.",
        "weeks of benefits may be paid.",
        # Units after modifier words.
        "U.S. dollars per hour may be reimbursed.",
        "taxable years is the maximum carryforward period.",
        "consecutive months of benefits may be paid.",
        # Rate or range.
        "per hour may be reimbursed.",
        "or more qualifying children.",
        "to 18 years of age.",
        "through 17 years of age.",
        "to less than 20 percent of taxable income.",
        "and older qualify for the benefit.",
        # Modifier words wrapped across lines or joined by a no-break space.
        "taxable\nyears is the maximum carryforward period.",
        "U.S.\u00a0dollars per hour may be reimbursed.",
        # Coordinated amounts.
        "and 12 percent brackets apply.",
        "or $20 per hour, whichever is greater.",
    ),
)
def test_page_index_amount_with_wrapped_unit_stays_demanded(line_break, continuation):
    source = "14" + line_break + continuation
    citation = f"{REV_PROC}/page-14"
    assert (14, 0) in _inventory(source, citation)
    assert 14 in _demanded(_issues(source, citation))


@pytest.mark.parametrize("line_break", ("\n", "\r\n"), ids=("lf", "crlf"))
def test_wrapped_unit_after_a_blank_line_stays_demanded(line_break):
    source = "14" + line_break * 2 + "dollars per hour may be reimbursed."
    assert _demanded(_issues(source, f"{REV_PROC}/page-14")) == {14}


@pytest.mark.parametrize("line_break", ("\n", "\r\n"), ids=("lf", "crlf"))
@pytest.mark.parametrize(
    "opening",
    (
        "To qualify, the taxpayer must earn at least $500.",
        "Or, if the taxpayer elects, the amount is $500.",
        "Through this notice, the amount is increased to $500.",
    ),
)
def test_page_number_before_ordinary_prose_is_not_demanded(line_break, opening):
    source = "14" + line_break * 2 + opening
    assert _demanded(_issues(source, f"{REV_PROC}/page-14")) == {500}


def test_masked_page_number_lets_later_geography_recognition_run():
    # The page-number mask runs before context-sensitive recognizers. Once the
    # page number is masked, `48 contiguous states` opens the text and is read
    # as geography; the operative amount stays demanded.
    source = "14\n48 contiguous states have an allowance of $500."
    citation = f"{REV_PROC}/page-14"
    assert _demanded(_issues(source, "")) == {14, 48, 500}
    assert _demanded(_issues(source, citation)) == {500}


@pytest.mark.parametrize(
    ("first_line", "citation", "value"),
    (
        # A real amount on line 1 keeps its demand.
        ("$14", f"{REV_PROC}/page-14", 14),
        ("14 percent", f"{REV_PROC}/page-14", 14),
        ("14 months", f"{REV_PROC}/page-14", 14),
        ("5,120", f"{REV_PROC}/page-14", 5120),
        ("14.5", f"{REV_PROC}/page-14", 14.5),
        # A bare number that is not this page's index is not proven a locator.
        ("13", f"{REV_PROC}/page-14", 13),
        ("140", f"{REV_PROC}/page-14", 140),
        ("014", f"{REV_PROC}/page-14", 14),
        # Only IRS revenue procedure and notice page sources are recognized.
        ("14", REV_PROC, 14),
        ("14", "us/guidance/irs/rev-rul-2025-32/page-14", 14),
        ("14", "us/guidance/usda/fns/snap-fy2026-cola/page-14", 14),
        ("14", "us-or/manual/odhs/open/page-14", 14),
    ),
)
def test_line_one_amount_stays_demanded(first_line, citation, value):
    source = first_line + "\n\nThe amount is described above."
    assert value in _demanded(_issues(source, citation))


@pytest.mark.parametrize("prefix", ("14\r\n\r\n", "14  \n\n", "  14\n"))
def test_page_number_line_endings_and_padding_are_recognized(prefix):
    source = prefix + "The refundable amount is $5,120."
    demanded = _demanded(_issues(source, f"{REV_PROC}/page-14"))
    assert 14 not in demanded
    assert 5120 in demanded


def test_page_number_without_a_following_line_stays_demanded():
    assert 14 in _demanded(_issues("14", f"{REV_PROC}/page-14"))


def test_page_number_on_a_later_line_stays_demanded():
    source = "Section 3.\n14\n\nThe amount is described above."
    assert 14 in _demanded(_issues(source, f"{REV_PROC}/page-14"))


def test_same_value_later_on_the_page_stays_demanded():
    source = "14\n\nThe employer may deem up to $14 per hour substantiated."
    citation = f"{REV_PROC}/page-14"
    assert _inventory(source, citation) == [(14, source.index("$14") + 1)]
    assert 14 in _demanded(_issues(source, citation))
    assert _issues(source, citation, scalars=(14,)) == ()


def test_source_map_keeps_offsets_for_the_masked_page():
    source = REPLAY_PAGES[1][1]
    mapping = sc._NumericRecallSourceMap(list(range(len(source))))
    recall = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=REPLAY_PAGES[1][0], _source_map=mapping
    )
    assert recall == sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=REPLAY_PAGES[1][0]
    )
    assert mapping.source_span(0, 2) is None
    amount = recall.index("5,120")
    assert mapping.source_span(amount, amount + 5) == (
        source.index("5,120"),
        source.index("5,120") + 5,
    )

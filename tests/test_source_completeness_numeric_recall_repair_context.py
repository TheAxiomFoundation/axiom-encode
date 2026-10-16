"""Numeric-recall retry feedback locates every missing value in the source.

Retry feedback carries one diagnostic per category, so before this change a
model repairing 26 U.S.C. 32 (36 missing values) saw only `value 0.5 has no
named scalar representation`. The first numeric-recall issue now carries a
bounded, branch-grouped locator for every missing value. Demands, issue counts
and the head of every issue are unchanged.
"""

from __future__ import annotations

import functools
import re

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

STATUTE = "us/statute/26/32"
HEAD = re.compile(
    r"^\[complete-source-unit:numeric-recall\] Authoritative corpus numeric value "
    r"(\S+) has no named scalar representation\. `module\.summary` is not consulted\."
)
SOURCE = (
    "(a) Allowance of credit. The credit percentage is 34 percent and the "
    "phaseout percentage is 15.98 percent.\n"
    "(b) Table.\n"
    "| 1 qualifying child | $6,330 | $11,610 |\n"
    "| 2 or more qualifying children | $8,890 | $11,610 |\n"
    "(c) Investment income. No credit is allowed if disqualified income "
    "exceeds $10,000.\n"
    "(d) Age. The applicable minimum age is 19, and 24 for a specified student."
)


def _analysis(source: str, rules=(), citation: str = STATUTE):
    content = yaml.safe_dump(
        {"format": "rulespec/v1", "module": {}, "rules": list(rules)}
    )
    return completeness.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _recall_issues(analysis):
    return [issue for issue in analysis.issues if HEAD.match(issue)]


def _without_context(monkeypatch, source: str, citation: str = STATUTE):
    with monkeypatch.context() as patch:
        patch.setattr(
            completeness, "_numeric_recall_repair_context", lambda *a, **k: ""
        )
        return _analysis(source, citation=citation)


def test_demands_counts_and_heads_are_unchanged(monkeypatch):
    analysis = _analysis(SOURCE)
    baseline = _without_context(monkeypatch, SOURCE)
    assert [HEAD.match(i).group(0) for i in _recall_issues(analysis)] == list(
        _recall_issues(baseline)
    )
    assert (
        analysis.source_numeric_occurrence_count,
        analysis.covered_source_numeric_occurrence_count,
        analysis.missing_source_numeric_occurrence_count,
    ) == (
        baseline.source_numeric_occurrence_count,
        baseline.covered_source_numeric_occurrence_count,
        baseline.missing_source_numeric_occurrence_count,
    )
    # Only the first numeric-recall issue changes; every other issue is
    # byte-identical, so per-issue identity and counting are preserved.
    changed = [a for a, b in zip(analysis.issues, baseline.issues) if a != b]
    assert len(analysis.issues) == len(baseline.issues)
    assert changed == [_recall_issues(analysis)[0]]
    prefixed = [f"statutes/26/32.yaml: ci: {issue}" for issue in analysis.issues]
    baseline_prefixed = [
        f"statutes/26/32.yaml: ci: {issue}" for issue in baseline.issues
    ]
    assert cli._validation_residual_obligation_count(
        prefixed
    ) == cli._validation_residual_obligation_count(baseline_prefixed)
    assert all(
        cli._validation_retry_branch_locator(item) is None
        for item in prefixed
        if "numeric-recall" in item
    )


def test_first_issue_locates_every_missing_value_by_branch():
    issues = _recall_issues(_analysis(SOURCE))
    values = [HEAD.match(issue).group(1) for issue in issues]
    first = issues[0]
    assert f"All {len(values)} values of this source without a named scalar" in first
    for value in values:
        assert re.search(
            rf"(?:^|[ ,]){re.escape(value)}(?:,| at [0-9]+:[0-9]+:)", first
        )
    # Values on one table row share one excerpt.
    assert "6330, 11610 at " in first
    assert "`| 1 qualifying child | $6,330 | $11,610 |`" in first
    assert "In us/statute/26/32(c) [Absatz c]:" in first
    assert "10000 at " in first
    assert "disqualified income exceeds $10,000.`" in first
    # A rate is named by the value the gate demands (34 percent -> 0.34).
    assert "0.34, 0.1598 at " in first
    # Excerpts are verbatim source text (whitespace-normalized).
    excerpts = re.findall(r"\d+:\d+(?: in source\.txt, [^`]*)?: `([^`]*)`", first)
    assert len(excerpts) >= 5
    for excerpt in excerpts:
        stripped = excerpt.removeprefix("... ").removesuffix(" ...")
        assert stripped in " ".join(SOURCE.split())
    assert "Displayed text is only a bounded locator" in first


def test_value_is_located_where_the_recall_view_demands_it():
    # `5` first appears as an attachment locator, which recall ignores; the
    # demanded `5` is the five years, and that is where the locator points.
    source = (
        "(a) See Exhibit B to Attachment 5 of the DOJ Interim Guidance.\n"
        "(b) The alien must be in a qualified status for 5 years before receiving "
        "benefits, or 40 qualifying quarters."
    )
    issues = _recall_issues(_analysis(source, citation="us/regulation/7/273/4"))
    first = issues[0]
    assert "qualified status for 5 years" in first
    assert "Attachment 5" not in first


def test_single_missing_value_gets_only_its_locator():
    issues = _recall_issues(_analysis("The cap is $1,500 per household."))
    assert len(issues) == 1
    assert (
        "First source location of 1500: character (code point) offsets 12:17"
        in issues[0]
    )
    assert "`The cap is $1,500 per household.`" in issues[0]
    assert "All 1 values" not in issues[0]


def test_context_is_capped_and_lists_every_value():
    rows = "\n".join(f"| row {index} | ${1000 + index * 7:,} |" for index in range(400))
    source = "(a) Table.\n" + rows
    issues = _recall_issues(_analysis(source))
    first = issues[0]
    head = HEAD.match(first).group(0)
    assert (
        len(first) - len(head) <= completeness._NUMERIC_RECALL_CONTEXT_MAX_CHARS + 1_000
    )
    assert "Not located above (" in first
    assert first.rstrip().endswith(".")
    # Every value is either located or listed (the listing itself is capped).
    listed = len(re.findall(r"\$[0-9,]+ \|", first))
    assert listed > 0


def test_retry_feedback_now_shows_every_missing_value():
    analysis = _analysis(SOURCE)
    items = [f"statutes/26/32.yaml: ci: {issue}" for issue in analysis.issues]
    shown = cli._finite_validation_retry_issue_snapshot(items)
    recall = [item for item in shown if "numeric-recall" in item]
    assert recall, "retry feedback must carry a numeric-recall diagnostic"
    for issue in _recall_issues(analysis):
        value = HEAD.match(issue).group(1)
        assert re.search(
            rf"(?:^|[ ,]){re.escape(value)}(?:,| at [0-9]+:[0-9]+:)", recall[0]
        )


@pytest.mark.parametrize(
    "citation",
    ("us/statute/26/32", "us-va/statute/58.1/58.1-322.03", "uk/legislation/ukpga/1/1"),
)
def test_context_is_deterministic(citation):
    assert (
        _analysis(SOURCE, citation=citation).issues
        == _analysis(SOURCE, citation=citation).issues
    )


@pytest.mark.parametrize(
    "citation, source, locator, operative",
    (
        # Review of #1934: the smallest missing value was often a locator.
        (
            "us/regulation/7/273/4",
            "(G) An alien child whose parent has been battered; 2 or\n"
            "(H) The household pays $1,500 for 12 months.",
            "2",
            "12",
        ),
        (
            "us-mo/statute/143.011",
            "1. The rate is 4.7 percent. 2. (1) The top rate shall be reduced by "
            "$175 each year.",
            "2",
            "0.047",
        ),
        (
            "us-va/statute/58.1/58.1-339.8",
            "A credit under subdivision 1 or 2 equals 15 percent of the credit.",
            "2",
            "0.15",
        ),
        (
            "us-az/statute/43-1072",
            "43-1072 - Earned credit for property taxes\nThe credit is $502.",
            "43",
            "502",
        ),
    ),
)
def test_locator_values_are_grouped_apart_and_never_the_first_example(
    citation, source, locator, operative
):
    issues = _recall_issues(_analysis(source, citation=citation))
    first = issues[0]
    assert f"First source location of {operative}:" in first
    assert "appear to be citation, caption or section locators" in first
    tail = first.split("appear to be citation, caption or section locators", 1)[1]
    assert re.search(rf"(?:^|[ ,]){re.escape(locator)}(?:,| at [0-9]+:[0-9]+:)", tail)


def test_excerpt_word_widening_is_bounded():
    token = "x" * 500
    source = f"{token} The cap is $1,500 per household. {token}"
    first = _recall_issues(_analysis(source))[0]
    excerpt = re.search(r"offsets \d+:\d+ in source\.txt, [^`]*: `([^`]*)`", first)
    assert excerpt is not None
    assert (
        len(excerpt.group(1))
        < 2
        * (
            completeness._NUMERIC_RECALL_EXCERPT_RADIUS
            + completeness._NUMERIC_RECALL_WORD_WIDENING
        )
        + 40
    )

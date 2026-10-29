"""Decimal subsection citations stay inside their deferral reason sentence."""

from __future__ import annotations

import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "us-co/statute/39/39-22-2003"
REASON = (
    f"Cannot compute {CITATION}(1.5)(b) without the "
    "amount of interest income from state and local bonds."
)


def test_decimal_deferral_reason_preserves_the_full_missing_input_clause():
    match = re.search(r"amount of interest income", REASON)
    assert match is not None
    start, end = sc._reason_clause_bounds(REASON, match)
    assert REASON[start:end] == REASON[:-1]
    assert sc._reason_cites_exact_current_statute_branch(
        REASON[start:end],
        corpus_citation_path=CITATION,
        path=("1.5", "b"),
        strict_terminal=True,
    )


@pytest.mark.parametrize("separator", [". ", "; ", "\n"])
def test_decimal_citation_does_not_hide_an_adjacent_sentence(separator):
    reason = "The dependency is available" + separator + REASON
    match = re.search(r"amount of interest income", reason)
    assert match is not None
    start, end = sc._reason_clause_bounds(reason, match)
    assert reason[start:end].strip() == REASON[:-1]


def test_decimal_deferral_cannot_authenticate_a_different_branch():
    assert not sc._reason_cites_exact_current_statute_branch(
        REASON,
        corpus_citation_path=CITATION,
        path=("1.5", "a"),
        strict_terminal=True,
    )


def test_decimal_citation_with_letter_suffix_stays_in_the_reason():
    reason = REASON.replace("(1.5)", "(1.5a)")
    match = re.search(r"amount of interest income", reason)
    assert match is not None
    start, end = sc._reason_clause_bounds(reason, match)
    assert reason[start:end] == reason[:-1]


def test_decimal_deferral_covers_only_its_exact_authenticated_branch():
    source = (
        "(1) Earlier definition.\n"
        "(1.5) Adjusted gross income means:\n"
        "(b) Repealed.\n"
        "(c) The claimant must submit an annual report.\n"
        "(2) A refund is allowed.\n"
    )
    reason = f"{CITATION}(1.5)(b) is repealed and supplies no operative rule."
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": "us-co:statutes/39/39-22-2003/1.5/b#interest_income",
                        "reason": reason,
                    }
                ]
            }
        },
        corpus_citation_path=CITATION,
        source_text=source,
        branches=branches,
    )
    assert issues == []
    assert covered == {("1.5", "b")}


def test_decimal_self_citation_cannot_defer_a_real_computation():
    source = (
        "(1) Earlier definition.\n"
        "(1.5) Adjusted gross income means:\n"
        "(b) The total of federal adjusted gross income and interest income.\n"
        "(2) A refund is allowed.\n"
    )
    branches = sc.recognize_source_structure(source, corpus_citation_path=CITATION)
    covered, issues = sc._deferred_coverage(
        {
            "module": {
                "deferred_outputs": [
                    {
                        "output": "us-co:statutes/39/39-22-2003/1.5/b#interest_income",
                        "reason": REASON,
                    }
                ]
            }
        },
        corpus_citation_path=CITATION,
        source_text=source,
        branches=branches,
    )
    assert ("1.5", "b") not in covered
    assert issues


@given(
    whole=st.integers(min_value=1, max_value=99),
    fractional=st.integers(min_value=1, max_value=999),
    suffix=st.sampled_from(["", "a", "b"]),
    child=st.sampled_from(["a", "b", "c"]),
    separator=st.sampled_from([". ", "; ", "\n"]),
)
@settings(max_examples=40, deadline=None)
def test_decimal_citation_punctuation_never_splits_a_reason(
    whole, fractional, suffix, child, separator
):
    clause = (
        f"Cannot compute {CITATION}({whole}.{fractional}{suffix})({child}) "
        "without the amount of interest income"
    )
    reason = "Earlier statement" + separator + clause + ". Later statement."
    match = re.search(r"amount of interest income", reason)
    assert match is not None
    start, end = sc._reason_clause_bounds(reason, match)
    assert reason[start:end].strip() == clause

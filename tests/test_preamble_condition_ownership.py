"""Structural footnotes must not add conditions to preceding worksheet rules."""

import pytest

from axiom_encode.harness import source_completeness as completeness

SOURCE = """Enter the amount from line 26000 if it is $57,375 or less.
If it is more than $57,375, enter the result of the following calculation:
amount from line 76 of their return
÷ 14.5% =
Amount from line 30000 of their return
8
Amount from line 96 of their return
+
9
Amount from line 32300 of their return
Add lines 8 to 10.

+
=

10
7
11
36100 =
Federal amounts transferred from your spouse or common-law partner

(1) If this is a new claim for the disability amount, attach a completed and certified Form T2201.
(2) The spouse qualifies if the spouse is employed and the spouse is disabled.
"""


def clauses_for(excerpt):
    clauses, ambiguous = completeness._source_condition_clauses_owned_by_excerpt(
        excerpt,
        rule={},
        source_text=SOURCE,
        branches=completeness.recognize_source_structure(SOURCE),
        corpus_citation_path="ca/policy/worksheet",
    )
    assert not ambiguous
    return clauses


@pytest.mark.parametrize(
    "excerpt",
    [
        "Add lines 8 to 10.",
        "If it is more than $57,375, enter the result of the following calculation:",
    ],
)
def test_worksheet_preamble_does_not_absorb_numbered_footnote(excerpt):
    clauses = clauses_for(excerpt)
    assert len(clauses) == 1
    assert clauses[0].end <= SOURCE.index("(1)")
    assert "T2201" not in clauses[0].text
    assert not completeness._source_conjunctive_fact_gates(clauses[0].text)


def test_footnote_own_conjunctive_obligations_remain_visible():
    clauses = clauses_for("the spouse is employed and the spouse is disabled")
    assert len(clauses) == 1
    assert clauses[0].branch_path == ("2",)
    assert len(completeness._source_conjunctive_fact_gates(clauses[0].text)) == 2


def test_excerpt_spanning_preamble_and_footnote_is_not_truncated():
    excerpt = SOURCE[SOURCE.index("Add lines") : SOURCE.index("(2)")].strip()
    clauses = clauses_for(excerpt)
    assert len(clauses) == 1
    assert "T2201" in clauses[0].text

"""Printed work charts own their conditions even in flattened PDF text."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c

SOURCE = (
    Path(__file__).parent / "fixtures/source_completeness/quebec_work_charts_prefix.txt"
).read_text()


def test_complete_unique_chart_proof_keeps_negative_condition_without_next_chart():
    start, end = SOURCE.index("105 WORK CHART"), SOURCE.index("142 WORK CHART")
    clauses, ambiguous = c._source_condition_clauses_owned_by_excerpt(
        SOURCE[start:end],
        rule={},
        source_text=SOURCE,
        branches=c.recognize_source_structure(SOURCE),
        corpus_citation_path="ca/policy/test",
    )
    assert not ambiguous
    assert clauses
    joined = " ".join(x.text for x in clauses)
    assert "If the result is negative" in joined
    assert "1997" not in joined
    assert "142 WORK CHART" not in joined
    for clause in clauses:
        assert clause.text == SOURCE[clause.start : clause.end].strip()
        assert clause.end <= end


def test_chart_clause_offsets_and_deliberately_spanning_proof():
    boundary = SOURCE.index("142 WORK CHART")
    assert boundary in c._work_chart_heading_starts(SOURCE)
    for a, b, text in c._source_clause_spans(SOURCE, branches=()):
        assert text == SOURCE[a:b]
        assert not a < boundary < b
    start = SOURCE.index("105 WORK CHART")
    end = SOURCE.index("Carry-forward") + len("Carry-forward")
    a, b = c._source_proposition_bounds(SOURCE, start, end)
    assert a <= start and b >= end
    assert "1997" in SOURCE[a:b]
    assert "If the result is negative" in SOURCE[a:b]


@pytest.mark.parametrize("prefix", ["", "\n"])
def test_structural_heading_requires_worksheet_rows(prefix):
    heading = "142 WORK CHART – Support payments received Total payments 1 Non-taxable amounts 2"
    assert c._work_chart_heading_starts(prefix + heading) == (len(prefix),)


@pytest.mark.parametrize(
    "text",
    [
        "See 142 WORK CHART – Support payments received Total payments 1 Non-taxable amounts 2",
        "The quotation is “142 WORK CHART – Support payments received Total payments 1”.",
        "142 WORK CHART Support payments received Total payments 1",
        "142 WORK CHART – Support payments received, as discussed elsewhere.",
        "An amount of 1.142 WORK CHART – Support payments received Total payments 1",
        "Previous prose = 3 142 WORK CHART – Support payments received Total payments 1",
    ],
)
def test_nonstructural_references_do_not_create_boundaries(text):
    assert c._work_chart_heading_starts(text) == ()


def test_following_chart_retains_its_conditions_and_repeated_excerpt_ambiguity():
    start, end = SOURCE.index("142 WORK CHART"), SOURCE.index("201 WORK CHART")
    clauses, ambiguous = c._source_condition_clauses_owned_by_excerpt(
        SOURCE[start:end],
        rule={},
        source_text=SOURCE,
        branches=c.recognize_source_structure(SOURCE),
        corpus_citation_path="ca/policy/test",
    )
    assert not ambiguous
    joined = " ".join(x.text for x in clauses)
    assert "1997 through 2024" in joined
    assert "for 2025" in joined
    assert "Carry-forward of non-taxable support" in joined
    _, ambiguous = c._source_condition_clauses_owned_by_excerpt(
        "Subtract line 4 from line 1.",
        rule={},
        source_text=SOURCE,
        branches=c.recognize_source_structure(SOURCE),
        corpus_citation_path="ca/policy/test",
    )
    assert ambiguous


@pytest.mark.parametrize(
    "text",
    [
        "142 WORK CHART – Reference only.\nSee the next chart.\n143 WORK CHART – Actual calculation Amount from box A 1 Total from box B 2",
        "“\n142 WORK CHART – Support payments received Total payments 1 Non-taxable amounts 2\n”",
        "Earlier layout\b Ordinary prose evaluates x = 3 142 WORK CHART – Mention Total payments 1",
    ],
)
def test_heading_does_not_borrow_layout_or_rows(text):
    starts = c._work_chart_heading_starts(text)
    assert text.index("142 WORK CHART") not in starts
    if "143 WORK CHART" in text:
        assert text.index("143 WORK CHART") in starts


def test_complete_105_condition_diagnostic_matches_its_isolated_chart():
    citation = "ca/policy/revenu-quebec/tp1-2025/main-return"
    excerpt = SOURCE[SOURCE.index("105 WORK CHART") : SOURCE.index("142 WORK CHART")]
    rule = {
        "name": "tp1_employment_income_correction_line_105",
        "kind": "derived",
        "entity": "Person",
        "dtype": "Money",
        "period": "Year",
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": citation,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
        "versions": [
            {"effective_from": "2025-01-01", "formula": "rl22_box_a - rl1_box_p_total"}
        ],
    }
    payload = {
        "rules": [rule],
        "inputs": [{"name": "rl22_box_a"}, {"name": "rl1_box_p_total"}],
    }
    combined = c._opaque_same_source_condition_input_issues(
        payload,
        source_text=SOURCE,
        branches=c.recognize_source_structure(SOURCE),
        principal_rules={rule["name"]: rule},
        corpus_citation_path=citation,
    )

    isolated = SOURCE[: SOURCE.index("142 WORK CHART")]
    own_issues = c._opaque_same_source_condition_input_issues(
        payload,
        source_text=isolated,
        branches=c.recognize_source_structure(isolated),
        principal_rules={rule["name"]: rule},
        corpus_citation_path=citation,
    )
    # Its own negative-result instruction still trips the existing fact-gate
    # classifier. This boundary fix does not waive that separate diagnostic.
    assert own_issues and combined == own_issues

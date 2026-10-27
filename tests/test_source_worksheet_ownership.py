"""Printed worksheet operations and identifiers retain their actual owners."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as completeness

SOURCE = (
    Path(__file__).parent / "fixtures/source_completeness/quebec_2025_work_charts.txt"
).read_text()


def test_actual_work_chart_headers_and_input_rows_are_corroborated():
    starts = completeness._work_chart_heading_starts(SOURCE)
    assert [SOURCE[p:].split()[0] for p in starts] == [
        "142",
        "201",
        "225",
        "297",
        "395",
        "401",
        "414",
        "458",
        "466",
    ]


@pytest.mark.parametrize(
    "excerpt,next_title",
    [
        (
            "Subtract line 2 from line 1. Carry the result to line 105 of your return. If the result is negative",
            "142 WORK CHART",
        ),
        (
            "Add lines 7, 9 and 11. Carry the result to line 395 of your return.",
            "401 WORK CHART",
        ),
        (
            "Subtract line 7 from line 6. = 8 × 6% Multiply line 8 by 6%. Carry the result to line 201 of your return (maximum $1,420).",
            "225 WORK CHART",
        ),
        (
            "Add lines 12 and 15. Carry the result to line 89 of Schedule J.",
            "466 WORK CHART",
        ),
    ],
)
def test_actual_final_arithmetic_proofs_do_not_inherit_previous_floor_or_next_chart(
    excerpt, next_title
):
    clauses, ambiguous = completeness._source_condition_clauses_owned_by_excerpt(
        excerpt,
        rule={"source": "Work Chart"},
        source_text=SOURCE,
        branches=(),
        corpus_citation_path="ca/policy/revenu-quebec/tp1-2025/main-return",
    )
    assert not ambiguous
    assert len(clauses) == 1
    clause = clauses[0]
    assert excerpt in clause.text
    assert next_title not in clause.text
    assert not clause.text.startswith("If the result is negative")
    assert completeness._source_conjunctive_fact_gates(clause.text) == ()
    assert SOURCE[clause.start : clause.end].strip(" ;,") == clause.text


def test_intentionally_spanning_chart_proof_keeps_both_sides():
    start = SOURCE.index("Add lines 7, 9 and 11.")
    end = SOURCE.index("Taxable income (line 299") + len("Taxable income (line 299")
    left, right = completeness._source_proposition_bounds(SOURCE, start, end)
    assert SOURCE[start:end] in SOURCE[left:right]
    assert "395" in SOURCE[left:right] and "401 WORK CHART" in SOURCE[left:right]
    # All original negative-floor obligations remain in the source clauses.
    spans = list(completeness._source_clause_spans(SOURCE, branches=()))
    assert sum(
        t.count("If the result is negative") for _, _, t in spans
    ) == SOURCE.count("If the result is negative")
    assert all(SOURCE[a:b] == text for a, b, text in spans)


CHART = (
    "123 WORK CHART – Test amount Amount from line 1 1 Subtract line 1 from line 2. "
)


@pytest.mark.parametrize(
    "tail",
    [
        "If the applicant is resident and disabled, × 24% 11 Add lines 7, 9 and 11.",
        "Whenever the spouse earns income, × 24% 11 Add lines 7, 9 and 11.",
        "Assuming the spouse earns income, × 24% 11 Add lines 7, 9 and 11.",
        "If the applicant is resident and disabled, e.g. × 24% 11 Add lines 7, 9 and 11.",
        "If the applicant is resident and disabled, i.e. × 24% 11 Add lines 7, 9 and 11.",
        "If the applicant is resident and disabled, e.g.11 Add lines 7, 9 and 11.",
        'If the applicant is resident and disabled, "A." × 24% 11 Add lines 7, 9 and 11.',
        "If the applicant is resident and disabled, A. Smith says × 24% 11 Add lines 7, 9 and 11.",
        "If the applicant is resident and disabled, Fig.11 Add lines 7, 9 and 11.",
        "If the applicant is resident and disabled, see illustration No.11 Add lines 7, 9 and 11.",
        "Unless the applicant is resident, – 7 Subtract line 7 from line 6.",
        'The instruction says "× 24% 11 Add lines 7, 9 and 11."',
        "The instruction says “× 24% 11 Add lines 7, 9 and 11.”",
        "The amount is 11 Add lines 7, 9 and 11.",
        "× 24% 11 Add lines 7, 9 and 10.",
        "– 7 Subtract line 6 from line 7.",
        "× 24% $11 Add lines 7, 9 and 11.",
    ],
)
def test_ambiguous_or_condition_governed_instructions_are_not_detached(tail):
    source = CHART + tail
    assert not any(
        p >= len(CHART) for p in completeness._printed_chart_arithmetic_starts(source)
    )


def test_no_arithmetic_boundary_without_work_chart():
    assert (
        completeness._printed_chart_arithmetic_starts("× 24% 11 Add lines 7, 9 and 11.")
        == ()
    )


@pytest.mark.parametrize(
    "identifier", ["G1A 1B9", "G1X 4A5", "H5B 1A4", "box 16A of your T4 slips"]
)
def test_closed_identifiers_are_not_german_sentence_markers(identifier):
    text = f"Send the form to {identifier}. The applicable amount is 500."
    assert completeness.recognize_source_structure(text) == ()
    assert list(completeness._GLUED_SENTENCE_MARKER.finditer(text)) == []


@pytest.mark.parametrize(
    "sentence",
    [
        "1Die Steuer beträgt 500.",
        "2Der Betrag beträgt 500.",
        "1Änderungen sind anzuwenden.",
        "Satz 1: Die Steuer beträgt 500.",
    ],
)
def test_real_german_sentence_labels_remain_supported(sentence):
    branches = completeness.recognize_source_structure(sentence)
    assert any(b.kind == "sentence" for b in branches)


def test_quoted_and_uncorroborated_chart_headings_are_not_owners():
    assert completeness._work_chart_heading_starts('"' + CHART + '"') == ()
    assert (
        completeness._work_chart_heading_starts(
            "See 123 WORK CHART – Test amount Amount from line 1 1."
        )
        == ()
    )
    assert (
        completeness._work_chart_heading_starts(
            "123 WORK CHART – Test amount Taxable income is discussed here."
        )
        == ()
    )

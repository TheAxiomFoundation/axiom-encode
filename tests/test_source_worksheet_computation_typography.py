"""Printed referrals do not invent computations or hide operative arithmetic."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as completeness

SOURCE = (
    Path(__file__).parent / "fixtures/source_completeness/quebec_2025_work_charts.txt"
).read_text()
REFERRAL = SOURCE[6461:6633]
CARRY = SOURCE[8138:8301]


@pytest.mark.parametrize(
    "text",
    [
        REFERRAL,
        CARRY,
        REFERRAL.replace("466", "123").replace(
            "Financial compensation", "Special credit"
        ),
        CARRY.replace("466", "123").replace("= 30", "= 18"),
        "Carry the result to line 123 of your return.\b Special credit = 18",
    ],
)
def test_corroborated_referral_typography_is_not_computation(text):
    assert not completeness.source_states_explicit_computation(text)
    assert not completeness._source_states_nonrounding_computation(text)
    masked = completeness._without_manual_typography_operators(text)
    assert len(masked) == len(text)
    changed = [i for i, (a, b) in enumerate(zip(text, masked, strict=True)) if a != b]
    # Other manual typography can include the bullet; none of the source words
    # or numbers are removed, and the original proof/source text stays intact.
    assert changed
    assert all(not text[i].isalnum() for i in changed)


@pytest.mark.parametrize(
    "text",
    [
        REFERRAL.replace("line 466 in", "line 123 in"),
        '"' + REFERRAL + '"',
        "“" + CARRY + "”",
        "See " + REFERRAL,
        "If eligible, " + CARRY,
        CARRY.replace(".\b", " (maximum $1,420).\b"),
        CARRY.replace(".\b", " (minimum $155).\b"),
        CARRY.replace("\b", " "),
        CARRY.replace("4 of 4", "page four"),
        CARRY.replace("Keep these pages for your files.", "Do another calculation."),
        "benefit = 30",
        CARRY.replace("Financial compensation", "Maximum compensation"),
        REFERRAL.replace("Financial compensation", "Maximum compensation"),
        REFERRAL + " Multiply line 9 by line 10: 9 * 10.",
        REFERRAL + " 39% of eligible costs.",
        CARRY + " Subtotal / 12.",
    ],
)
def test_ambiguous_or_computational_text_is_not_exempted(text):
    assert completeness.source_states_explicit_computation(text)
    assert completeness._source_states_nonrounding_computation(text)


@pytest.mark.parametrize(
    "start",
    ["201 WORK CHART", "Carry the result to line 201", "Carry the result to line 414"],
)
def test_existing_worded_or_capped_chart_obligations_remain(start):
    spans = completeness._source_clause_spans(SOURCE, branches=())
    clause = next(text for _, _, text in spans if text.startswith(start))
    assert completeness.source_states_explicit_computation(clause)
    assert not completeness._worksheet_referral_typography_spans(clause)


def test_source_formula_inventory_changes_only_demonstrated_referrals():
    spans = list(completeness._source_clause_spans(SOURCE, branches=()))
    affected = [
        (start, end)
        for start, end, text in spans
        if completeness._worksheet_referral_typography_spans(text)
    ]
    assert affected == [(3934, 4020), (4667, 4810), (6461, 6633), (8138, 8301)]
    assert all(SOURCE[start:end] == text for start, end, text in spans)
    assert "Multiply line 9 by line 10." in SOURCE
    assert "Add lines 28 and 29." in SOURCE

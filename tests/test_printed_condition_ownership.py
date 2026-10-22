"""Printed row containers preserve propositions without creating legal paths."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as s
from axiom_encode.printed_source_rows import recognize_printed_rows

P = Path(__file__).parent / "fixtures/form-numeric-locators"
C = "ca/policy/test"


def resolve(text, excerpt, branches=()):
    rule = {
        "name": "owner",
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {"corpus_citation_path": C, "excerpt": excerpt},
                    }
                ]
            }
        },
    }
    return s._source_condition_clauses_owned_by_excerpt(
        excerpt, rule=rule, source_text=text, branches=branches, corpus_citation_path=C
    )


def test_net_income_owner_excludes_previous_else_and_footer_but_keeps_advisory():
    text = (P / "step-3.txt").read_text()
    excerpt = 'Line 56 and line 23600: Line 54 minus line 55 (if negative, enter "0")'
    clauses, ambiguous = resolve(text, excerpt)
    assert not ambiguous and len(clauses) == 1
    clause = clauses[0]
    assert clause.text.startswith("Line 56")
    assert "If not, enter" not in clause.text and "5000-R" not in clause.text
    assert "If negative, you may have a non-capital loss" in clause.text
    assert "negative amount is to be used for certain calculations" in clause.text
    assert text[clause.start : clause.end].strip(" ;,") == clause.text
    assert clause.branch_path == ()


def test_repayment_group_retains_threshold_bullets_and_else():
    text = (P / "step-3.txt").read_text()
    context = recognize_printed_rows(text)
    a, b = context.atomic_spans[0]
    excerpt = text[a:b]
    clauses, ambiguous = resolve(text, excerpt)
    assert not ambiguous and len(clauses) == 1
    for phrase in [
        "$82,125",
        "$93,454",
        "line 11900",
        "line 11300",
        "line 14600",
        'If not, enter "0"',
    ]:
        assert phrase in clauses[0].text
    assert "Line 56" not in clauses[0].text


def test_within_row_if_otherwise_not_severed():
    text = (P / "step-4.txt").read_text()
    a, b = recognize_printed_rows(text).atomic_spans[0]
    clauses, ambiguous = resolve(text, text[a:b])
    assert not ambiguous and len(clauses) == 1
    assert "if you did not enter" in clauses[0].text
    assert "; otherwise, use Federal Worksheet)" in clauses[0].text


@pytest.mark.parametrize("step", [3, 4])
def test_explicit_owner_inside_group_prevents_printed_container(step):
    text = (P / f"step-{step}.txt").read_text()
    a, b = recognize_printed_rows(text).atomic_spans[0]
    owner = s.SourceStructureBranch(
        ("a",), "paragraph", "a", text[a + 2 : b - 2], a + 2, b - 2
    )
    assert s._unconflicted_printed_rows(text, (owner,)) is None
    excerpt = text[a + 3 : a + 20]
    clauses, ambiguous = resolve(text, excerpt, (owner,))
    assert clauses and all(c.branch_path == ("a",) for c in clauses)
    assert all(owner.start <= c.start < c.end <= owner.end for c in clauses)


def test_cross_row_excerpt_is_never_cropped():
    text = (P / "step-3.txt").read_text()
    start = text.index("Line 54 and line 23400:")
    end = text.index("Line 56 and line 23600:")
    excerpt = text[start:end]
    clauses, _ = resolve(text, excerpt)
    assert clauses and any("Line 54" in c.text and "Line 55" in c.text for c in clauses)


def test_unknown_grid_retains_existing_ownership_behavior():
    text = (P / "step-3.txt").read_text().replace("Line 37 and 20805:", "Unknown row:")
    excerpt = 'Line 56 and line 23600: Line 54 minus line 55 (if negative, enter "0")'
    clauses, _ = resolve(text, excerpt)
    assert "If not, enter" in clauses[0].text


def test_repeated_excerpt_remains_ambiguous():
    text = (P / "step-3.txt").read_text()
    excerpt = 'Line 56 and line 23600: Line 54 minus line 55 (if negative, enter "0")'
    clauses, ambiguous = resolve(
        text + "\n" + excerpt,
        text[text.index(excerpt) : text.index(excerpt) + len(excerpt)],
    )
    assert ambiguous


@pytest.mark.parametrize(
    "excerpt",
    [
        'If not, enter "0" on line 23500.',
        'Line 55 and line 23500: If not, enter "0" on line 23500.',
        "the amount on line 23400 is more than $82,125",
        "the amount on line 23400 is more than $93,454",
    ],
)
def test_short_repayment_proof_cannot_detach_inherited_conditions(excerpt):
    text = (P / "step-3.txt").read_text()
    clauses, ambiguous = resolve(text, excerpt)
    assert not ambiguous and len(clauses) == 1
    for part in [
        "Complete the chart",
        "$82,125",
        "$93,454",
        "line 11900",
        "line 11300",
        "line 14600",
        'If not, enter "0"',
    ]:
        assert part in clauses[0].text
    assert "Line 56" not in clauses[0].text


@pytest.mark.parametrize(
    "excerpt",
    [
        "if you did not enter an amount on line 14600",
        "otherwise, use Federal Worksheet",
        "enter the amount from line 14700",
    ],
)
def test_short_conditional_arm_cannot_detach_other_arm(excerpt):
    text = (P / "step-4.txt").read_text()
    clauses, ambiguous = resolve(text, excerpt)
    assert not ambiguous and len(clauses) == 1
    assert (
        "if you did not enter an amount on line 14600; otherwise, use Federal Worksheet"
        in clauses[0].text
    )

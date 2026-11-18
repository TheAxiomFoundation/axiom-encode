"""Unpunctuated bullet lists must not consume subsequent independent sections.

The regression shape comes from AZ DES FAA5 child-support verification block-4.
This is a source-parser test, not generated RuleSpec or a policy fixture.
"""

import pytest

from axiom_encode.harness.source_completeness import (
    _post_list_verification_heading_starts,
    _source_clause_spans,
    _source_condition_clauses_owned_by_excerpt,
    _source_conjunctive_fact_gates,
    recognize_source_structure,
    source_states_explicit_computation,
)

IRREGULAR = """Verification of the most recent prior three consecutive months is needed to average the expense when one or more of the following occurs regarding the child support payment:

● Paid less often than monthly

● Paid with no established frequency

● Unpredictable

● One or more months had no payment (zero paid month)"""

EVIDENCE = """Verification of the court-ordered child support amount that can be used includes the following:

● Child support order issued by the court

● Official pay record from Arizona’s Clerk of the Court showing the amount of support paid"""

SOURCE = IRREGULAR + "\n\n" + EVIDENCE


def test_formula_clause_does_not_absorb_independent_evidence_section():
    spans = tuple(
        _source_clause_spans(SOURCE, branches=recognize_source_structure(SOURCE))
    )
    computation = [text for _, _, text in spans if "needed to average" in text]
    assert len(computation) == 1
    assert "Verification of the court-ordered" not in computation[0]
    assert "zero paid month" in computation[0]
    assert [text for _, _, text in spans] == [IRREGULAR, EVIDENCE]
    for start, end, text in spans:
        assert SOURCE[start:end] == text
    assert any("Child support order issued" in text for _, _, text in spans)


def test_evidence_proof_does_not_inherit_unrelated_irregular_condition():
    excerpt = EVIDENCE.split("\n\n")[0]
    clauses, ambiguous = _source_condition_clauses_owned_by_excerpt(
        excerpt,
        rule={},
        source_text=SOURCE,
        branches=recognize_source_structure(SOURCE),
        corpus_citation_path="us-az/manual/des/faa5/na-child-support-expense/block-4",
    )
    assert not ambiguous
    assert clauses
    assert all("when one or more" not in clause.text for clause in clauses)


def test_two_conditions_inside_one_bullet_remain_required():
    source = "● Allowed when the parent is eligible and the child is disabled."
    spans = tuple(_source_clause_spans(source, branches=()))
    assert len(spans) == 1
    assert spans[0][2] == source
    assert len(_source_conjunctive_fact_gates(spans[0][2])) == 2


def test_line_wrapped_conjunction_is_not_a_new_section():
    source = "Allowed when the parent is eligible\n\nand the child is disabled."
    spans = tuple(_source_clause_spans(source, branches=()))
    assert len(spans) == 1
    assert len(_source_conjunctive_fact_gates(spans[0][2])) == 2


@pytest.mark.parametrize(
    "continuation",
    [
        "provided that the parent is eligible",
        "unless the parent is exempt",
        "and only when the child is disabled",
    ],
)
def test_post_list_proviso_stays_with_computation(continuation):
    governed = IRREGULAR + "\n\n" + continuation
    source = governed + "\n\n" + EVIDENCE
    spans = tuple(_source_clause_spans(source, branches=()))
    assert [text for _, _, text in spans] == [governed, EVIDENCE]


def test_wrapped_bullet_qualification_stays_with_computation():
    governed = IRREGULAR + "\n\nonly during the relevant history period"
    source = governed + "\n\n" + EVIDENCE
    spans = tuple(_source_clause_spans(source, branches=()))
    assert [text for _, _, text in spans] == [governed, EVIDENCE]


@pytest.mark.parametrize(
    "chapeau",
    [
        "If the parent is eligible:",
        "If the parent is eligible\nand resides in Arizona:",
        "Only when the parent is eligible:",
        "● If the parent is eligible:",
    ],
)
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_inherited_conditional_is_not_discarded_as_independent_heading(
    chapeau, newline
):
    source = chapeau + "\n\n● Submit the application\n\n" + EVIDENCE
    source = source.replace("\n", newline)
    assert _post_list_verification_heading_starts(source) == ()


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("blank", ["", "  "])
def test_source_whitespace_preserves_independent_sections(newline, blank):
    irregular = IRREGULAR.replace("\n\n", newline + blank + newline)
    evidence = EVIDENCE.replace("\n\n", newline + blank + newline)
    source = irregular + newline + blank + newline + evidence
    spans = tuple(_source_clause_spans(source, branches=()))
    assert [text for _, _, text in spans] == [irregular, evidence]


def test_payment_frequency_comparison_is_not_subtraction():
    assert not source_states_explicit_computation("Paid less often than monthly")
    assert not source_states_explicit_computation(IRREGULAR)


@pytest.mark.parametrize(
    "source",
    [
        "income less expenses",
        "income less the expenses",
        "Paid less often than monthly. Compute income less expenses.",
        "income less often",  # ambiguous symbol named often stays conservative
    ],
)
def test_real_subtraction_remains_computation(source):
    assert source_states_explicit_computation(source)


def test_physical_alternatives_are_not_conjunctive_facts():
    assert _source_conjunctive_fact_gates(IRREGULAR) == ()


@pytest.mark.parametrize(
    "change",
    [
        lambda text: text.replace("one or more", "all", 1),
        lambda text: text.replace(
            "● Unpredictable", "● The parent is eligible and the child is disabled"
        ),
        lambda text: (
            text + "\n\nprovided that the parent is eligible and the child is disabled"
        ),
        lambda text: text.replace(
            "when one or more",
            "when the parent is eligible and the child is disabled and one or more",
        ),
    ],
)
def test_alternative_list_does_not_hide_conjunctions(change):
    assert len(_source_conjunctive_fact_gates(change(IRREGULAR))) >= 2

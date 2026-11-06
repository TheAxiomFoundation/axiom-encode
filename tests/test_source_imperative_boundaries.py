"""Factual antecedents and PDF page boundaries must retain source ownership."""

import pytest

from axiom_encode.harness import source_completeness as completeness

SPECIAL_CASE = (
    "Special case If the cost of services included in your rent changed more than "
    "once in the year, do the calculation on lines 20 through 27 of Work Chart 466 "
    "for each month the cost of services changed and enter the result of your "
    "calculations on line 30 of the work chart."
)
REFUND = "80 2025 – GUIDE-V Refund 474 If you enter an amount on line 474, you are entitled to a refund."


def test_special_case_commands_are_not_conjunctive_claimant_facts():
    assert completeness._source_conjunctive_fact_gates(SPECIAL_CASE) == ()


@pytest.mark.parametrize("connector", ["and", "or"])
def test_imperative_consequent_preserves_real_antecedent_signatures(connector):
    antecedent = f"If the applicant is not disabled {connector} the spouse is resident"
    expected = completeness._source_conjunctive_fact_gates(antecedent)
    assert len(expected) == (2 if connector == "and" else 0)
    assert (
        completeness._without_worksheet_imperative_consequent(
            antecedent + ", do the calculation and enter the result."
        )
        == antecedent
    )
    assert (
        completeness._source_conjunctive_fact_gates(
            antecedent + ", do the calculation and enter the result."
        )
        == expected
    )


@pytest.mark.parametrize(
    "body",
    [
        " you are employed, do the calculation and if your spouse is disabled enter the result.",
        " you are employed, do the calculation when your spouse is disabled.",
        " you are employed, do the calculation unless your spouse is disabled.",
        " you are employed, do the calculation provided that your spouse is disabled.",
        " you are employed, do the calculation and your spouse is resident.",
        ' the notice says "employed, do the calculation and enter the result" and you are resident.',
        " the notice says “employed, do the calculation and enter the result” and you are resident.",
        " you meet the condition (employed, do the calculation and enter the result) and are resident.",
    ],
)
def test_uncertain_or_embedded_commands_remain_conservative(body):
    assert completeness._without_worksheet_imperative_consequent(body) == body


@pytest.mark.parametrize(
    "restriction",
    [
        "while the spouse earns income",
        "whenever the spouse earns income",
        "where the spouse earns income",
        "until the spouse earns income",
        "for children whose income exceeds 100",
        "for children who earn income",
        "after the spouse earns income",
        "only for employed applicants",
        "as long as the spouse earns income",
        "assuming the spouse earns income",
        "subject to residence",
        "contingent upon the spouse earning income",
        "conditional on the spouse earning income",
        "dependent on the spouse earning income",
        "for disabled spouses",
    ],
)
def test_imperative_with_restrictive_clause_is_not_trimmed(restriction):
    body = f" the applicant is disabled, enter the result {restriction} and calculate the total."
    assert completeness._without_worksheet_imperative_consequent(body) == body


@pytest.mark.parametrize(
    "antecedent",
    [
        " income exceeds 1,250 and you are resident",
        " you moved on May 1, 2025 and you are resident",
        ' the notice says "calculate, enter" and you are resident',
        " your income (including interest, dividends) exceeds 100 and you are resident",
    ],
)
def test_internal_commas_are_preserved(antecedent):
    assert (
        completeness._without_worksheet_imperative_consequent(
            antecedent + ", do the calculation and enter the result."
        )
        == antecedent
    )


def test_printed_page_header_separates_owned_propositions_without_changing_bytes():
    source = SPECIAL_CASE + "\n\n" + REFUND
    start = source.index("cost of services")
    end = start + len("cost of services")
    left, right = completeness._source_proposition_bounds(source, start, end)
    assert source[left:right] == SPECIAL_CASE
    spans = list(completeness._source_clause_spans(source, branches=()))
    assert spans[0] == (0, len(SPECIAL_CASE), SPECIAL_CASE)
    assert all(source[a:b] == text for a, b, text in spans)
    assert "Refund 474" in " ".join(text for _, _, text in spans[1:])
    assert "Refund 474" not in spans[0][2]
    # An intentional excerpt crossing the header still owns both propositions.
    left, right = completeness._source_proposition_bounds(
        source, start, source.index("entitled") + len("entitled")
    )
    assert SPECIAL_CASE in source[left:right]
    assert "entitled" in source[left:right]


@pytest.mark.parametrize(
    "source",
    [
        "Completed. 80 2025 – GUIDE-V Refund",
        "Completed.\n80 2025 – GUIDE-V Refund",
        "Incomplete\n\n80 2025 – GUIDE-V Refund",
        "Completed.\n\n80.5 2025 – GUIDE-V Refund",
        "Completed.\n\n80 2025 – guide-v Refund",
        "Completed.\n\n80 2025 – GUIDE Refund",
        "Completed.\n\n2025 – GUIDE-V Refund",
        '"Completed.\n\n80 2025 – GUIDE-V Refund"',
        "“Completed.\n\n80 2025 – GUIDE-V Refund”",
        "Completed.\n\n80 2025 – GUIDE-Very Refund",
    ],
)
def test_prose_and_quoted_page_like_text_are_not_structural_headers(source):
    assert completeness._printed_page_header_boundaries(source) == ()


def test_page_split_retains_genuine_previous_conjunction():
    first = "If the applicant is resident and the spouse is disabled, enter the result."
    source = first + "\n\n" + REFUND
    spans = list(completeness._source_clause_spans(source, branches=()))
    assert completeness._source_conjunctive_fact_gates(spans[0][2]) == (
        (frozenset({"applicant"}), frozenset({"resident"})),
        (frozenset({"spouse"}), frozenset({"disabled"})),
    )

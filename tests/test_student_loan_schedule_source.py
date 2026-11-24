"""Source-only records do not confer arithmetic or condition authority."""

import dataclasses
import hashlib
from pathlib import Path

import pytest

from axiom_encode.student_loan_schedule_source import recognize_student_loan_schedule

SOURCE = (
    Path(__file__).parent / "fixtures/student_loan_schedule/source.txt"
).read_text()
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest"


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("crlf", [False, True])
def test_complete_raw_records_and_offsets(wrapped, crlf):
    source = SOURCE if wrapped else SOURCE.split("\n\n", 1)[1]
    if crlf:
        source = source.replace("\n", "\r\n")
    bound = recognize_student_loan_schedule(source, CITATION)
    assert bound is not None
    assert bound.bound(source, CITATION)
    assert bound.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert (bound.year, bound.prior_year, bound.history_first_year) == (
        2025,
        2024,
        1998,
    )
    assert len(bound.records) == 10
    spans = sorted([r.span for r in bound.records] + list(bound.unowned_spans))
    assert spans[0][0] == 0 and spans[-1][1] == len(source)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert "".join(source[a:b] for a, b in spans) == source
    by_role = {r.role: source[slice(*r.span)] for r in bound.records}
    assert by_role["current_completion"].startswith("Complete this schedule if")
    assert "decided to carry" in by_role["prior_completion"]
    assert "minus the interest already used" in by_role["historical_reconstruction"]
    assert "or the amount shown" in by_role["entered_prior"]
    assert "line 385" in by_role["return_claim"]
    assert (
        by_role["guide_referral"]
        == "If you want to keep a copy for your records, see the guide."
    )
    assert not bound.bound(source + " ", CITATION)
    assert not bound.bound(source, CITATION + "/other")
    forged = dataclasses.replace(bound, year=2024)
    assert not forged.bound(source, CITATION)


def test_years_are_roles_not_literal_year_allowlist():
    source = SOURCE.replace("2025", "2026").replace("2024", "2025")
    bound = recognize_student_loan_schedule(source, CITATION)
    assert bound is not None and bound.prior_year == 2025


@pytest.mark.parametrize(
    "before,after",
    [
        ("Schedule M 2025 -", "Schedule M 2024 -"),
        ("for 2024,", "for 2023,"),
        ("after 1997", "after 1996"),
        ("1998 through", "1999 through"),
        ("your 2025 return", "your 2024 return"),
        ("Add lines 46 and 48.", "Add lines 48 and 46."),
        ("Subtract line 60 from line 52.", "Subtract line 52 from line 60."),
        ("line 385", "line 386"),
        ("46\n48\n52\n60\n62", "46\n52\n48\n60\n62"),
        ("minus the interest", "plus the interest"),
        ("and decided to carry", "or decided to carry"),
        ("If interest was paid", "Unless interest was paid"),
        ("Complete this schedule if", "Do not complete this schedule if"),
        (
            "Enclose this schedule with your return.",
            "Do not enclose this schedule with your return.",
        ),
        ("see the guide.", "ignore the guide."),
        ("see the guide.", "see the guide unless you owe tax."),
        ("notice of assessment)", "notice of assessment) less assistance"),
        (
            "Interest paid on a student loan =",
            "Interest paid on a student loan = if eligible",
        ),
        ("\n\nEnclose", "\n\nDo not claim this credit.\n\nEnclose"),
        ("\nIf you want", "\nUnless you owe tax.\nIf you want"),
    ],
)
def test_operative_or_inconsistent_change_refuses_whole_record(before, after):
    assert before in SOURCE
    assert (
        recognize_student_loan_schedule(SOURCE.replace(before, after), CITATION) is None
    )


@pytest.mark.parametrize(
    "extra", ["Do not:\n\n", "Never claim this credit.\n", " ", "\n"]
)
def test_prefix_or_suffix_is_not_silently_discarded(extra):
    assert recognize_student_loan_schedule(extra + SOURCE, CITATION) is None
    assert recognize_student_loan_schedule(SOURCE + extra, CITATION) is None


def test_duplicate_clause_or_document_refuses():
    assert recognize_student_loan_schedule(SOURCE + SOURCE, CITATION) is None
    clause = "Enclose this schedule with your return."
    assert (
        recognize_student_loan_schedule(
            SOURCE.replace(clause, clause + "\n" + clause), CITATION
        )
        is None
    )


def test_raw_fixture_is_exact_authenticated_parent_body():
    assert (
        hashlib.sha256(SOURCE.encode()).hexdigest()
        == "e8a905d1a3e9dd708c46ffc0090fafeeeb5c840a0c708a8c80dff2674b78c90a"
    )


@pytest.mark.parametrize(
    "citation", ["", "../ca/policy", "ca//policy", "https://example.com"]
)
def test_malformed_citation_refuses(citation):
    assert recognize_student_loan_schedule(SOURCE, citation) is None

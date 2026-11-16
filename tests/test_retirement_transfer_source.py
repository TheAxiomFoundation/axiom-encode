"""Closed raw records never confer arithmetic, fact or execution authority."""

import dataclasses
import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from axiom_encode.retirement_transfer_source import (
    recognize_retirement_transfer_schedule,
)

SOURCE = (Path(__file__).parent / "fixtures/retirement_transfer/source.txt").read_text()
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-q-retirement-income-transfer"


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize("crlf", [True, False])
def test_complete_raw_partition_dates_and_numeric_roles(wrapped, crlf):
    source = SOURCE if wrapped else SOURCE.split("\n\n", 1)[1]
    if crlf:
        source = source.replace("\n", "\r\n")
    result = recognize_retirement_transfer_schedule(source, CITATION)
    assert result is not None and result.bound(source, CITATION)
    assert result.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert result.year == 2025
    assert result.birth_cutoff == date(1961, 1, 1)
    assert result.spouse_reference_date == date(2025, 12, 31)
    assert result.cap_percent == Decimal("50")
    assert len(result.records) == 27
    spans = sorted([r.span for r in result.records] + list(result.unowned_spans))
    assert spans[0][0] == 0 and spans[-1][1] == len(source)
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert "".join(source[a:b] for a, b in spans) == source
    for number in result.numbers:
        assert source[slice(*number.span)] == number.text
    fifties = [(n.owner, n.role) for n in result.numbers if n.text == "50"]
    assert fifties == [
        ("selected_transfer", "cap_percent"),
        ("withholding_proportion", "multiplicand_row"),
    ]
    records = {r.role: source[slice(*r.span)] for r in result.records}
    assert "jointly decide" in records["permission"]
    assert "For more information" in records["informational_guide"]
    assert records["corresponding_withholding"].startswith("If you transfer a portion")
    assert records["transferor_enclosure"].startswith("Only the person transferring")
    assert "RL-1" in records["conditional_rl1_withholding"]
    assert (
        records["copy_guide"]
        == "If you want to keep a copy for your records, see the guide."
    )
    assert not result.bound(source + " ", CITATION)
    assert not result.bound(source, CITATION + "/foreign")
    assert not dataclasses.replace(result, cap_percent=Decimal("100")).bound(
        source, CITATION
    )
    assert not dataclasses.replace(result, birth_cutoff=date(1960, 1, 1)).bound(
        source, CITATION
    )


def test_birth_date_is_independent_source_literal_not_inferred_age():
    source = SOURCE.replace("2025", "2026")
    result = recognize_retirement_transfer_schedule(source, CITATION)
    assert result is not None
    assert result.birth_cutoff == date(1961, 1, 1)
    assert result.spouse_reference_date == date(2026, 12, 31)
    result = recognize_retirement_transfer_schedule(
        source.replace("1961", "1960"), CITATION
    )
    assert result is not None and result.birth_cutoff == date(1960, 1, 1)


@pytest.mark.parametrize(
    "before,after",
    [
        ("Schedule Q 2025 -", "Schedule Q 2024 -"),
        ("D.Q-V (2025-12)", "D.Q-V (2025-11)"),
        ("born before January 1", "born before July 1"),
        ("and you had a spouse", "or you had a spouse"),
        ("jointly decide", "independently decide"),
        (
            "Last name and first name of your spouse on December 31, 2025",
            "Last name and first name of your spouse on December 31, 2024",
        ),
        ("If you transfer a portion", "Unless you transfer a portion"),
        ("must also transfer", "need not transfer"),
        ("Only the person transferring", "Both persons receiving"),
        ("see “Eligible retirement income”", "ignore “Eligible retirement income”"),
        ("Add lines 12 through 16.", "Add lines 16 through 12."),
        ("Subtract line 18 from line 10.", "Subtract line 10 from line 18."),
        ("maximum: 50%", "maximum: 100%"),
        ("line 245 of your return", "line 123 of your return"),
        ("same amount on line 123", "half amount on line 123"),
        ("box J", "box E"),
        ("If the income includes", "If the income excludes"),
        ("box E of your RL-1", "box J of your RL-2"),
        ("Multiply line 50 by line 52.", "Multiply line 50 by line 53."),
        ("Divide the result by line 53.", "Divide the result by line 52."),
        ("line 451.1 of your return", "line 451.3 of your return"),
        (
            "Enclose this schedule with your return.",
            "Do not enclose this schedule with your return.",
        ),
        ("see the guide.", "see the guide unless you owe tax."),
        ("\n+ 13\n", "\n+ 14\n"),
        ("\n\n50\n52\n53\n\n58", "\n\n50\n53\n52\n\n58"),
        ("\n3\n\nA\n\n4\n", "\n4\n\nA\n\n3\n"),
        ("\n12\n", "\n12 if eligible\n"),
        ("amount on line 10 above", "amount on line 10 above less assistance"),
        (
            "Your spouse’s\nsocial insurance number",
            "Your spouse’s\nsocial insurance number unless exempt",
        ),
        ("\n\nEnclose", "\n\nDo not transfer.\n\nEnclose"),
        ("\nNOTE\n", "\nNOTE: unless eligible\n"),
    ],
)
def test_changed_operative_or_layout_text_refuses(before, after):
    assert before in SOURCE
    assert (
        recognize_retirement_transfer_schedule(SOURCE.replace(before, after), CITATION)
        is None
    )


@pytest.mark.parametrize(
    "extra", ["Do not:\n\n", "Never transfer.\n", " ", "\n", "\nUnless eligible."]
)
def test_no_prefix_suffix_or_duplicate_is_ignored(extra):
    assert recognize_retirement_transfer_schedule(extra + SOURCE, CITATION) is None
    assert recognize_retirement_transfer_schedule(SOURCE + extra, CITATION) is None


def test_duplicate_document_and_clause_refuse():
    assert recognize_retirement_transfer_schedule(SOURCE + SOURCE, CITATION) is None
    clause = "Enclose this schedule with your return."
    assert (
        recognize_retirement_transfer_schedule(
            SOURCE.replace(clause, clause + "\n" + clause), CITATION
        )
        is None
    )


@pytest.mark.parametrize(
    "citation", ["", "../ca/policy", "ca//policy", "https://example.com", None]
)
def test_malformed_citation_refuses(citation):
    assert recognize_retirement_transfer_schedule(SOURCE, citation) is None


def test_fixture_exact_authenticated_body_and_no_scalar_credit_interface():
    assert (
        hashlib.sha256(SOURCE.encode()).hexdigest()
        == "e7a0bbfd4382c5329fd53a6174e2bdc94f86e8aacf64259a6763bb811d19317b"
    )
    result = recognize_retirement_transfer_schedule(SOURCE, CITATION)
    assert result is not None
    assert set(vars(result)) == {
        "source",
        "citation",
        "source_sha256",
        "year",
        "birth_cutoff",
        "spouse_reference_date",
        "cap_percent",
        "records",
        "numbers",
        "unowned_spans",
    }

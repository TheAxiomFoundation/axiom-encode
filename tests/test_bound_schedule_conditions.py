from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from axiom_encode.bound_schedule_conditions import recognize_schedule_conditions
from axiom_encode.bound_schedule_structure import recognize_bound_schedule_structure

SOURCE = (Path(__file__).parent / "fixtures/nb_schedule/source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/nb-s12-seniors-home-renovation-credit"


def _records(source):
    form = recognize_bound_schedule_structure(source, CITATION)
    return recognize_schedule_conditions(form) if form is not None else None


@pytest.mark.parametrize("source", [SOURCE, SOURCE.replace("\n", "\r\n")])
def test_complete_raw_medical_and_residence_records(source):
    records = _records(source)
    assert records is not None and records.bound(source, CITATION)
    m = records.medical
    assert m.reference_date == date(2025, 12, 31)
    assert source[slice(*m.reference_date_span)] == "December 31, 2025"
    assert (
        source[slice(*m.relationship_span)].replace("\r\n", "\n")
        == "you and your spouse or common-law\npartner"
    )
    assert (
        source[slice(*m.separate_residences_span)]
        == "occupied separate principal residences"
    )
    assert source[slice(*m.medical_reason_span)] == "for medical reasons"
    assert source[slice(*m.permission_span)].startswith("you can")
    assert source[slice(*m.address_reporting_span)].endswith("line 60890.")
    assert source[slice(*records.residence.yes_span)] == "If yes, tick this box. ^"
    assert source[slice(*records.residence.no_span)].startswith("If not, enter")
    assert m.span[1] < records.residence.span[0]


def test_valid_other_source_year_field_and_title_are_not_allowlisted():
    source = SOURCE.replace("2025", "2032").replace("60890", "70901")
    source = source.replace(
        "New Brunswick Seniors' Home Renovation Tax Credit",
        "Other Seniors' Home Renovation Tax Credit",
    )
    records = _records(source)
    assert records is not None and records.medical.reference_date == date(2032, 12, 31)
    assert records.medical.reporting_field == 70901


@pytest.mark.parametrize(
    "before,after",
    [
        ("December 31, 2025", "December 32, 2025"),
        ("December 31, 2025", "December 31, 2024"),
        ("for medical reasons", "for other reasons"),
        (
            "occupied separate principal residences",
            "did not occupy separate principal residences",
        ),
        ("you can\nboth claim", "you cannot\nboth claim"),
        ("you can\nboth claim", "you can\nboth claim if income is low"),
        (
            "home renovation tax credit.",
            "home renovation tax credit and claim another allowance.",
        ),
        ("address on line 60890.", "address on line 70901."),
        ("Involuntary separation", "Do not:"),
        ("Involuntary separation", "Unless a doctor approves"),
        ("Address of the residence", "Do not:"),
        ("If yes, tick this box.", "If yes, do not tick this box."),
        ("expenses on line 1 relate", "expenses on line 2 relate"),
        ("See the privacy notice", "Unless income is high. See the privacy notice"),
    ],
)
def test_qualified_negated_incomplete_or_mismatched_records_refuse(before, after):
    assert _records(SOURCE.replace(before, after)) is None


def test_operative_tail_after_medical_refuses_and_is_not_discarded():
    source = SOURCE.replace(
        "\n\nAddress of the residence",
        "\n\nUnless a doctor approves.\n\nAddress of the residence",
    )
    assert _records(source) is None


@pytest.mark.parametrize(
    "suffix", ["Do not claim this credit.", "Never claim this credit."]
)
@pytest.mark.parametrize(
    "next_heading", ["Address of the residence", "See the privacy notice"]
)
def test_prohibitory_suffix_after_either_record_refuses(suffix, next_heading):
    source = SOURCE.replace(
        "\n\n" + next_heading, "\n\n" + suffix + "\n\n" + next_heading
    )
    assert _records(source) is None


@pytest.mark.parametrize(
    "heading", ["Involuntary separation", "Address of the residence"]
)
def test_enclosing_qualification_before_display_heading_refuses(heading):
    source = SOURCE.replace(heading, "Unless a doctor approves:\n\n" + heading)
    assert _records(source) is None


def test_duplicate_record_and_forged_spans_refuse():
    records = _records(SOURCE)
    assert records is not None
    span = records.residence.span
    duplicate = (
        SOURCE[: span[1]]
        + "\n\nAddress of the residence\n\n"
        + SOURCE[slice(*span)]
        + SOURCE[span[1] :]
    )
    assert _records(duplicate) is None
    assert not replace(
        records, medical=replace(records.medical, reference_date_span=(0, 4))
    ).bound(SOURCE, CITATION)
    assert not records.bound(
        SOURCE.replace("medical reasons", "financial reasons"), CITATION
    )

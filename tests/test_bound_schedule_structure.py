from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.bound_schedule_structure import recognize_bound_schedule_structure

SOURCE = (Path(__file__).parent / "fixtures/nb_schedule/source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/nb-s12-seniors-home-renovation-credit"


def test_parent_layout_and_every_reporting_condition_remains_raw():
    form = recognize_bound_schedule_structure(SOURCE, CITATION)
    assert form is not None and form.bound(SOURCE, CITATION)
    assert [r.number for r in form.rows] == list(range(1, 8))
    assert form.rows[4].field_number == 60360
    assert [r.number for r in form.reporting_fields] == [60890]
    assert len(form.entry_field_spans) == 5
    assert SOURCE[slice(*form.column_count_span)] == "5"
    assert SOURCE[slice(*form.entry_count_span)] == "15"
    assert SOURCE[slice(*form.header_year_span)] == "2025"
    assert SOURCE[slice(*form.wrapper_year_span)] == "2025"
    for phrase in (
        "CRA Schedule",
        "Complete this schedule if",
        "Do not include",
        "If you need more space",
        "if applicable",
        "Involuntary separation",
        "Line 60890:",
        "December 31, 2025",
        "If not, enter",
    ):
        start = SOURCE.index(phrase)
        assert any(a <= start < b for a, b in form.unowned_spans), phrase


def test_bare_and_crlf_absolute_spans():
    for source in (SOURCE.split("\n\n", 1)[1], SOURCE.replace("\n", "\r\n")):
        form = recognize_bound_schedule_structure(source, CITATION)
        assert form is not None and form.bound(source, CITATION)
        assert source[slice(*form.confidentiality_cue_span)] == "when"
        assert source[slice(*form.reporting_fields[0].span)].startswith("Line 60890:")
        assert source[slice(*form.reporting_fields[0].span)].endswith("^")


def test_no_literal_schedule_jurisdiction_year_or_table_size_allowlist():
    source = SOURCE.replace("NB(S12)", "XY(S34)").replace("5004-S12", "9876-S34")
    source = source.replace("2025", "2032").replace("15 entries", "18 entries")
    form = recognize_bound_schedule_structure(source, "xx/policy/other/schedule")
    assert form is not None and form.year == 2032 and form.form == "XY(S34)"


@pytest.mark.parametrize(
    "before,after",
    [
        ("CRA Schedule NB(S12) 2025", "CRA Schedule NB(S12) 2024"),
        ("CRA Schedule NB(S12)", "CRA Schedule NB(S13)"),
        ("CRA Schedule NB(S12)", "CRA Form NB(S12)"),
        ("5004-S12", "5004-S13"),
        ("5 columns", "4 columns"),
        ("15 entries", "15 dollars"),
        ("Line 3:", "Line 4:"),
        ("Line 60890:", "Line 8:"),
        ("Supplier or contractor - Name ^", "Description ^"),
        ("table above Total eligible expenses", "table above Total refundable credit"),
        ("CRA Schedule NB(S12) 2025 New", "CRA Schedule NB(S12) 2025 If New"),
    ],
)
def test_malformed_or_operative_layout_refuses(before, after):
    assert (
        recognize_bound_schedule_structure(SOURCE.replace(before, after, 1), CITATION)
        is None
    )


def test_duplicate_wrapper_and_reporting_alias_refuse():
    prefix = SOURCE.split("\n\n", 1)[0] + "\n\n"
    assert recognize_bound_schedule_structure(prefix + SOURCE, CITATION) is None
    assert (
        recognize_bound_schedule_structure(
            SOURCE.replace("line 60360", "line 60890"), CITATION
        )
        is None
    )


@pytest.mark.parametrize(
    "tail", ["If income is high, reduce the amount.", "Subtract assistance again."]
)
def test_blank_line_continuations_are_exposed_not_owned(tail):
    source = SOURCE.replace("Line 4:", tail + "\n\nLine 4:")
    form = recognize_bound_schedule_structure(source, CITATION)
    assert form is not None
    at = source.index(tail)
    assert any(a <= at < b for a, b in form.unowned_spans)


def test_body_identity_and_forged_offsets_refuse():
    form = recognize_bound_schedule_structure(SOURCE, CITATION)
    assert form is not None
    assert not form.bound(SOURCE.replace("medical reasons", "other reasons"), CITATION)
    assert not form.bound(SOURCE, CITATION + "/other")
    assert not replace(form, header_year_span=(0, 4)).bound(SOURCE, CITATION)
    assert not replace(form, reporting_fields=()).bound(SOURCE, CITATION)

"""Raw structural ownership does not assert numeric or conditional coverage."""

from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.harness.source_completeness import authoritative_numeric_recall_text

SOURCE = (
    Path(__file__).parent / "fixtures/informational_referral/source.txt"
).read_text()
CITATION = "ca/policy/cra/t1-2025/nl479-credits/primary-document"


def test_exact_source_raw_ownership_and_unconsumed_conditions():
    structure = recognize_bound_form_structure(SOURCE, CITATION)
    assert structure is not None and structure.bound(SOURCE, CITATION)
    assert SOURCE[slice(*structure.header.year_span)] == "2025"
    assert SOURCE[slice(*structure.table.column_count_span)] == "5"
    assert SOURCE[slice(*structure.table.entry_count_span)] == "14"
    assert len(structure.table.field_spans) == 5
    assert [r.number for r in structure.rows] == list(range(1, 10))
    assert structure.rows[4].field_number == 62000
    assert structure.rows[-1].field_number is None  # Do not infer online47900of.
    assert SOURCE[slice(*structure.rows[0].caption_span)] == (
        "Total amount paid from all receipts included in the table above Total\n"
        "eligible expenses = ^"
    )
    assert SOURCE[slice(*structure.rows[6].caption_span)] == (
        "Line 5 multiplied by the percentage from line 6 Physical activity tax\ncredit = ^"
    )
    owned = [
        structure.header.span,
        structure.table.note_span,
        *structure.table.field_spans,
        *(r.span for r in structure.rows),
    ]
    for phrase in (
        "even if",
        "If the total",
        "If applicable",
        "If you need more space",
    ):
        start = SOURCE.index(phrase)
        assert not any(a <= start < b for a, b in owned)
    assert structure.source == SOURCE
    # No consumer integration or numeric masking is granted by recognition.
    recall = authoritative_numeric_recall_text(SOURCE)
    assert all(n in recall for n in ("5", "14", "2025", "2,000", "17.4"))


def test_other_form_year_counts_and_citation_are_not_allowlisted():
    source = SOURCE.replace("5001-TC", "9876-AB").replace("Form NL479", "Form XY123")
    source = source.replace("2025", "2031").replace(
        "5 columns with 14", "3 columns with 27"
    )
    source = source.replace("Description of activity ^\n", "").replace(
        "Name of family member ^\n", ""
    )
    structure = recognize_bound_form_structure(source, "xx/policy/other/form")
    assert structure is not None and structure.bound(source, "xx/policy/other/form")
    assert structure.header.year == 2031
    assert (structure.table.column_count, structure.table.entry_count) == (3, 27)


def test_crlf_offsets_are_raw_not_normalized():
    text = SOURCE.replace("\n", "\r\n")
    structure = recognize_bound_form_structure(text, CITATION)
    assert structure is not None and structure.bound(text, CITATION)
    assert "\r\n" in text[slice(*structure.rows[6].caption_span)]
    assert text[slice(*structure.table.entry_count_span)] == "14"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s.replace("5 columns", "4 columns"),
        lambda s: s.replace("14\nentry lines", "14 dollars"),
        lambda s: s.replace("each column. ***", "each column. Credit is $14. ***"),
        lambda s: s.replace("Date of receipt ^", "Date of receipt ^\n5 | 14"),
        lambda s: s.replace("Date of receipt ^", "Description of activity ^"),
        lambda s: s.replace("Date of receipt ^\n", ""),
        lambda s: s.replace("Line 4:", "Line 3:"),
        lambda s: s.replace("Line 4:", "Line 40:"),
        lambda s: s.replace("Line 4:", "Line wrong:"),
        lambda s: s.replace("Line 4:", "Line 4 and line 62000:"),
        lambda s: s.replace(
            "Total amount paid from all receipts included in the table above",
            "Amount payable including assistance",
        ),
        lambda s: s.replace("Form NL479", "Form NL479\nForm XY123"),
        lambda s: s.replace("2025\n\nForm", "2025\n2026\n\nForm"),
        lambda s: s.replace("*** Transcriber", '"*** Transcriber'),
        lambda s: (
            s
            + "\n*** Transcriber's Note: In print, the table is set-up as 2 columns with 3 entry lines for each column. ***"
        ),
        lambda s: s[s.index("Line 1:") :],
        lambda s: s.replace("***", "", 1),
    ],
)
def test_malformed_conflicting_or_value_bearing_layout_refuses(mutate):
    changed = mutate(SOURCE)
    assert changed != SOURCE
    assert recognize_bound_form_structure(changed, CITATION) is None


@pytest.mark.parametrize(
    "citation", ["", "../source", CITATION + "/", CITATION + "#rule", "ca//policy"]
)
def test_noncanonical_citation_refuses(citation):
    assert recognize_bound_form_structure(SOURCE, citation) is None


def test_forged_spans_stale_body_and_wrong_citation_refuse():
    structure = recognize_bound_form_structure(SOURCE, CITATION)
    assert structure is not None
    assert not structure.bound(SOURCE.replace("$2,000", "$3,000"), CITATION)
    assert not structure.bound(SOURCE, "ca/policy/other")
    assert not replace(structure, source_sha256="0" * 64).bound(SOURCE, CITATION)
    forged = replace(structure.table, column_count_span=structure.rows[2].caption_span)
    assert not replace(structure, table=forged).bound(SOURCE, CITATION)
    clipped = replace(
        structure.rows[6],
        span=(structure.rows[6].span[0], structure.rows[6].span[1] - 10),
    )
    assert not replace(
        structure, rows=(*structure.rows[:6], clipped, *structure.rows[7:])
    ).bound(SOURCE, CITATION)


def test_operative_values_year_comparisons_and_conditions_are_not_masked():
    text = SOURCE.replace(
        "Line 6: Applicable rate 17.4%",
        "Line 6: Applicable rate 17.4% if year > 2025 and amount > 14",
    )
    structure = recognize_bound_form_structure(text, CITATION)
    assert structure is not None
    row = text[slice(*structure.rows[5].caption_span)]
    assert row == "Applicable rate 17.4% if year > 2025 and amount > 14"
    # A row boundary is not proof of its operation or condition.
    assert structure.source == text
    assert "2025" in authoritative_numeric_recall_text(text)


def test_unowned_heading_and_separated_operative_continuations_remain_explicit():
    original = recognize_bound_form_structure(SOURCE, CITATION)
    assert original is not None
    assert any(
        "Research and development tax credit (individuals)" in SOURCE[slice(*span)]
        for span in original.unowned_spans
    )
    for continuation in (
        "If income exceeds $500, do not claim this credit.",
        "Subtract $14 from the preceding result.",
    ):
        text = SOURCE.replace("\n\nLine 4:", "\n\n" + continuation + "\n\nLine 4:")
        structure = recognize_bound_form_structure(text, CITATION)
        assert structure is not None and structure.bound(text, CITATION)
        assert any(
            continuation in text[slice(*span)] for span in structure.unowned_spans
        )
        assert continuation not in text[slice(*structure.rows[2].span)]
        assert not original.bound(text, CITATION)
    assert any(
        "If applicable" in SOURCE[slice(*span)] for span in original.unowned_spans
    )
    assert any(
        "See the privacy notice" in SOURCE[slice(*span)]
        for span in original.unowned_spans
    )

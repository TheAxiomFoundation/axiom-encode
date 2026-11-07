"""Source-record tests only; the fixture grants no execution authority."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.harness.bound_capital_gift_structure import (
    recognize_capital_gift_source,
)

SOURCE = (Path(__file__).parent / "fixtures/t1170-authenticated-body.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/capital-gains-on-gifts"


def test_complete_raw_source_identity_and_character_partition():
    result = recognize_capital_gift_source(SOURCE, CITATION)
    assert result is not None
    assert result.source == SOURCE
    assert result.source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
    assert result.year == 2025
    spans = sorted(
        [record.span for record in result.records] + list(result.unowned_spans)
    )
    assert spans[0][0] == 0 and spans[-1][1] == len(SOURCE)
    assert all(left[1] == right[0] for left, right in zip(spans, spans[1:]))
    assert all(SOURCE[a:b].isspace() for a, b in result.unowned_spans)
    assert result.bound(SOURCE, CITATION)
    assert not replace(result, source_sha256="0" * 64).bound(SOURCE, CITATION)
    assert not replace(result, unowned_spans=()).bound(SOURCE, CITATION)


def test_all_three_tables_have_distinct_routes_and_exact_column_references():
    result = recognize_capital_gift_source(SOURCE)
    assert result is not None
    assert [
        (t.role, t.destination, t.proceeds_field, t.zero_inclusion_field)
        for t in result.tables
    ] == [
        ("shares_table", "13200", "68220", "68230"),
        ("bonds_table", "15300", "68240", "68250"),
        ("land_table", "13800", "68251", "68252"),
    ]
    for table in result.tables:
        assert [SOURCE[a:b] for a, b in (r.span for r in table.columns)] == [
            f"){n}(" for n in range(1, 9)
        ]
        refs = {r.role: SOURCE[slice(*r.span)] for r in table.references}
        assert refs["column_7_multiplier_column_6"] == "6\nnmuloc\nyb"
        assert refs["column_7_divisor_column_2"] == ")2\nnmuloc\nyb"
        assert refs["column_5_deductions_columns_3_4"] == ")4\ndna\n3\nsnmuloc"
        assert refs["half_rate"] == "%05"
        assert refs["destination"] == table.destination[::-1]
        assert all(
            table.span[0] <= r.span[0] < r.span[1] <= table.span[1]
            for r in (*table.columns, *table.references)
        )


def test_full_operative_clauses_and_six_alternatives_are_retained():
    result = recognize_capital_gift_source(SOURCE)
    assert result is not None
    records = {r.role: SOURCE[slice(*r.span)] for r in result.subrecords}
    assert sum(key.startswith("property_alternative_") for key in records) == 6
    assert (
        "under certain\nconditions, a personal servitude"
        in records["property_alternative_6"]
    )
    assert records["no_advantage"].rstrip().endswith("rate of zero.")
    assert records["advantage"].rstrip().endswith("rate of zero.")
    assert records["half_inclusion"].endswith("50%.")
    assert records["additional_sheet"].endswith("fI\n")
    assert records["enclosure"] == ".nruter\nruoy\nhtiw\nmrof\nsiht\nedulcnI"


@pytest.mark.parametrize(
    "old,new",
    [
        ("2025", "2026"),
        ("5202", "6202"),
        ("T1170", "T1171"),
        ("first five", "first six"),
        ("30 days", "31 days"),
        ("50%", "60%"),
        ("%05", "%06"),
        ("qualified donees", "donees"),
        ("other than prescribed interests", "including prescribed interests"),
        ("under certain\nconditions", "without any\nconditions"),
        ("03286", "13286"),
        ("05286", "15286"),
        ("25286", "35286"),
        ("00231", "00232"),
        ("00351", "00352"),
        ("00831", "00832"),
        ("13286", "23286"),
        ("52\nenil", "62\nenil"),
        ("5\nnmuloc(", "6\nnmuloc("),
        (")2\nnmuloc\nyb", ")3\nnmuloc\nyb"),
        (")4\ndna\n3\nsnmuloc", ")3\ndna\n4\nsnmuloc"),
        (".3\neludehcS", ".4\neludehcS"),
    ],
)
def test_changed_operative_scalar_locator_or_column_refuses(old, new):
    assert old in SOURCE
    assert recognize_capital_gift_source(SOURCE.replace(old, new, 1)) is None


@pytest.mark.parametrize("record_index", range(16))
@pytest.mark.parametrize(
    "mutation", ["delete", "duplicate", "qualify_before", "prohibit_after"]
)
def test_each_source_record_is_required_once_and_closed(record_index, mutation):
    result = recognize_capital_gift_source(SOURCE)
    assert result is not None
    start, end = result.records[record_index].span
    text = SOURCE[start:end]
    replacement = {
        "delete": "",
        "duplicate": text + "\n" + text,
        "qualify_before": "Unless another rule applies:\n" + text,
        "prohibit_after": text + "\nDo not claim this amount.",
    }[mutation]
    assert (
        recognize_capital_gift_source(SOURCE[:start] + replacement + SOURCE[end:])
        is None
    )


@pytest.mark.parametrize(
    "source", [None, 17, "", " " + SOURCE, SOURCE + "\n", SOURCE.replace("\n", "\r\n")]
)
def test_unknown_type_outer_text_or_extraction_layout_refuses(source):
    assert recognize_capital_gift_source(source) is None


def test_table_reordering_and_cross_citation_refuse():
    result = recognize_capital_gift_source(SOURCE)
    assert result is not None
    one, two, _ = result.tables
    first = SOURCE[slice(*one.span)]
    second = SOURCE[slice(*two.span)]
    mutated = (
        SOURCE.replace(first, "PLACEHOLDER")
        .replace(second, first)
        .replace("PLACEHOLDER", second)
    )
    assert recognize_capital_gift_source(mutated) is None
    assert recognize_capital_gift_source(SOURCE, CITATION + "/child") is None

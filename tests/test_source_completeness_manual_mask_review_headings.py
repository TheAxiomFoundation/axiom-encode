"""Round-two regressions: numeric rows cannot supply their own locator evidence."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_resolver as resolver
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

OREGON_PAGE = "us-or/manual/odhs/open/page-93"
EMPTY_RULES = "format: rulespec/v1\nmodule: {}\nrules: []\n"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)
CORPUS_LOCATOR_CONTROLS = json.loads(
    (Path(__file__).parent / "fixtures/manual_mask_review_locators.json").read_text()
)


def _values(text: str) -> set[float]:
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile="en-US"
        )
    }


def _recalled(source: str, citation: str = OREGON_PAGE) -> set[float]:
    return _values(
        sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    )


def _missing(source: str, citation: str = OREGON_PAGE) -> set[float]:
    result = sc.analyze_complete_source_unit(
        EMPTY_RULES,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        float(match[1])
        for issue in result.issues
        if (match := RECALL_VALUE.search(issue)) is not None
    }


@pytest.mark.parametrize(
    "caption",
    (
        "The following table lists payment standards.",
        "The following table lists need standards.",
        "Limits:",
        "Thresholds:",
        "Maximums:",
        "Allotments:",
        "Deductions:",
        "Benefit amounts, in dollars:\nhousehold type",
        "Income\nHousehold Type",
        "Program Schedule:\nassistance group",
        "",
    ),
)
def test_review_sibling_amounts_are_recalled_and_demanded(caption):
    source = f"{caption}\n214.50 Family Assistance\n214.75 Adult Assistance"
    expected = {214.5, 214.75}
    assert _recalled(source) == expected
    assert _missing(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("0.10 Family Assistance\n0.20 Adult Assistance", {0.1, 0.2}),
        ("1 Family Group\n1.1 Adult Assistance", {1, 1.1}),
        (
            "1.1 Family Assistance\n1.1.1 Adult Assistance",
            {1.1},
        ),
        ("214.10 Application Process\n214.50 Family Assistance", {214.1, 214.5}),
    ),
    ids=("zero-root-siblings", "numeric-parent", "numeric-descendant", "numeric-peer"),
)
def test_numeric_candidates_do_not_authenticate_each_other(source, expected):
    assert expected <= _recalled(source)
    assert expected <= _missing(source)
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    assert masked == source


@pytest.mark.parametrize(
    ("prefix", "citation"),
    (
        ("Section 214: Assistance", OREGON_PAGE),
        ("", "us-or/manual/odhs/open/214/page-93"),
        (
            "See A-214 policies for background.",
            "us-nv/manual/dwss/eligibility-payments/page-93",
        ),
        ("CONTENTS PAGES", OREGON_PAGE),
        ("", "us-or/manual/odhs/open/table-of-contents/page-93"),
    ),
    ids=(
        "section",
        "document-identity",
        "nevada-cross-reference",
        "contents",
        "contents-path",
    ),
)
def test_payment_table_overrides_every_numbering_evidence_collision(prefix, citation):
    source = (
        f"{prefix}\nThe following table lists payment standards.\n"
        "household type\n214.50 Family Assistance\n214.75 Adult Assistance"
    )
    assert {214.5, 214.75} <= _recalled(source, citation)
    assert {214.5, 214.75} <= _missing(source, citation)


def test_generic_page_prefix_does_not_interrupt_operative_table_evidence():
    source = (
        "Section 214: Assistance\nBenefit amounts, in dollars:\n"
        "Page 93 of 100 214.50 Family Assistance\n214.75 Adult Assistance"
    )
    assert {214.5, 214.75} <= _recalled(source)
    assert {214.5, 214.75} <= _missing(source)


def test_unrecognised_caption_and_column_layout_override_section_numbering():
    source = (
        "Section 214: Assistance\nProgram Schedule:\ngroup classification\n"
        "214.50 Family Assistance\n214.75 Adult Assistance"
    )
    assert {214.5, 214.75} <= _recalled(source)
    assert {214.5, 214.75} <= _missing(source)


def test_quantity_caption_on_section_anchor_keeps_matching_decimal_amount():
    source = "Section 214: Benefit Levels\n214.50 Family Assistance"
    assert 214.5 in _recalled(source)
    assert 214.5 in _missing(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Section 214: Assistance\nUnit cost in USD\n214.50 Family Assistance",
            214.5,
        ),
        (
            "Section 1: Assistance\nX = Y * coefficient\n1.198 Family Assistance",
            1.198,
        ),
        ("Section 214: Assistance\nUnits\n214.50 Family Assistance", 214.5),
        ("Section 214: Assistance\nUSD\n214.50 Family Assistance", 214.5),
        (
            "Section 214: Assistance\nCurrency (USD)\n214.50 Family Assistance",
            214.5,
        ),
        ("Section 214: Assistance\nQuantity\n214.50 Family Assistance", 214.5),
        (
            "Section 214: Assistance\nCost per member\n214.50 Family Assistance",
            214.5,
        ),
        (
            "Section 214: Assistance\nHourly rate\n214.50 Family Assistance",
            214.5,
        ),
        ("Section 1: Assistance\nCoefficient\n1.198 Family Assistance", 1.198),
        ("Section 1: Assistance\nFormula\n1.198 Family Assistance", 1.198),
        (
            "Section 1: Assistance\nAdjustment factor\n1.198 Family Assistance",
            1.198,
        ),
        (
            "Section 1: Assistance\nCoefficient = cost / quantity\n1.198 Family Assistance",
            1.198,
        ),
        (
            "Section 214: Assistance\nCost (USD)\nCategory\n214.50 Family Assistance",
            214.5,
        ),
        (
            "Section 101: Assistance\nUnit cost in USD\n101.03 Application Process",
            101.03,
        ),
        ("Section 214: Assistance\n214.50 USD Family Assistance", 214.5),
        ("Section 214: Assistance\n214.50 Family Assistance USD", 214.5),
        ("Section 214: Assistance\n214.50 Family Assistance", 214.5),
        ("Section 1: Assistance\n1.198 Family Assistance", 1.198),
    ),
    ids=(
        "reviewed-singleton-cost",
        "reviewed-singleton-coefficient-formula",
        "units-column",
        "currency-column",
        "currency-caption",
        "quantity-column",
        "cost-column",
        "rate-column",
        "coefficient-column",
        "formula-column",
        "factor-column",
        "coefficient-equation",
        "cost-caption-and-category-column",
        "procedural-title-in-cost-table",
        "currency-before-row-title",
        "currency-after-row-title",
        "uncaptioned-singleton-cost",
        "uncaptioned-singleton-coefficient",
    ),
)
def test_singleton_quantity_context_overrides_matching_section_root(source, expected):
    """A section-root collision cannot authenticate an operative quantity."""

    assert expected in _recalled(source)
    assert expected in _missing(source)
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    label = source.rsplit("\n", 1)[1].split()[0]
    assert label in masked


@settings(max_examples=40, deadline=None)
@given(
    section=st.integers(min_value=1, max_value=899),
    fraction=st.integers(min_value=1, max_value=999),
    caption=st.sampled_from(
        (
            "Unit cost in USD",
            "Units",
            "Quantity",
            "USD",
            "Coefficient",
            "Formula",
            "Adjustment factor",
            "X = Y * coefficient",
        )
    ),
    title=st.sampled_from(("Family Assistance", "Application Process")),
    separator=st.sampled_from(("\n", "\n\n")),
)
@example(
    section=214,
    fraction=500,
    caption="Unit cost in USD",
    title="Family Assistance",
    separator="\n",
)
@example(
    section=1,
    fraction=198,
    caption="X = Y * coefficient",
    title="Family Assistance",
    separator="\n",
)
def test_generated_singleton_quantity_context_keeps_operative_value(
    section, fraction, caption, title, separator
):
    label = f"{section}.{fraction:03d}"
    source = (
        f"Section {section}: Assistance{separator}{caption}{separator}{label} {title}"
    )
    expected = float(label)
    assert expected in _recalled(source)
    assert expected in _missing(source)


@pytest.mark.parametrize(
    "caption",
    (
        "Unit cost in USD",
        "Units",
        "Quantity",
        "Currency (USD)",
        "Coefficient",
        "Formula",
        "Adjustment factor",
        "X = Y * coefficient",
        "X = Y * 1.198",
        "X × Y",
        "The unit cost is in USD",
        "The coefficient is X * Y",
        "The cost is $5",
        "The rate is 5%",
        "Unit cost (2026)",
        "Unit cost per 2 members",
        "Coefficient (2026)",
        "The coefficient is 2",
        "The cost is 5",
        "The factor is 2",
    ),
)
def test_quantity_and_formula_captions_override_even_reviewed_heading_titles(caption):
    """Known heading titles become quantities under an operative column caption."""

    source = (
        f"Section 101: Applications\n{caption}\nCategory\n"
        "101.03 Application Process\n101.04 Retroactive Applications"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    assert {101.03, 101.04} <= _recalled(source, citation)
    assert {101.03, 101.04} <= _missing(source, citation)
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert "101.03 Application Process" in masked
    assert "101.04 Retroactive Applications" in masked
    assert len(masked) == len(source)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=citation) == masked
    )


@settings(max_examples=40, deadline=None)
@given(
    section=st.sampled_from((100, 101, 214)),
    year=st.integers(min_value=1900, max_value=2099),
    members=st.integers(min_value=1, max_value=20),
    caption=st.sampled_from(
        (
            "Unit cost ({year})",
            "Unit cost per {members} members",
            "Coefficient ({year})",
        )
    ),
)
@example(section=100, year=2026, members=2, caption="Unit cost ({year})")
@example(section=100, year=2026, members=2, caption="Unit cost per {members} members")
def test_generated_numeric_qualifiers_preserve_quantity_caption_priority(
    section, year, members, caption
):
    """A numeric caption qualifier cannot turn costs into procedure headings."""

    caption = caption.format(year=year, members=members)
    source = (
        f"Section {section}: Applications\n{caption}\nCategory\n"
        f"{section}.03 Application Process\n{section}.04 Retroactive Applications"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    expected = {float(f"{section}.03"), float(f"{section}.04")}
    assert expected <= _recalled(source, citation)
    assert expected <= _missing(source, citation)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("Section 214: Applications\n214.03 Application Process", {214.03}),
        (
            "Section 214: Applications\n214.03 Application Process\n"
            "Section 214: Applications\n214.03 Application Process",
            {214.03},
        ),
        (
            "Section 214: Applications\n214.03 Application Process\n"
            "Section 214: Applications\n214.003 Application Process",
            {214.03, 214.003},
        ),
        (
            "Section 214: Applications\n214.50 Application Process\n"
            "214.75 Retroactive Applications",
            {214.5, 214.75},
        ),
        (
            "Section 214: Applications\nUnit cost in USD\n"
            "214.03 Application Process\n214.04 Retroactive Applications",
            {214.03, 214.04},
        ),
    ),
    ids=(
        "singleton",
        "repeated-label",
        "zero-padded-repeated-label",
        "wrong-numbering-scheme",
        "quantity-caption",
    ),
)
def test_procedure_title_scheme_requires_distinct_siblings_outside_quantity_context(
    source, expected
):
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    assert expected <= _recalled(source, citation)
    assert expected <= _missing(source, citation)


@settings(max_examples=40, deadline=None)
@given(
    section=st.integers(min_value=100, max_value=899),
    anchor=st.sampled_from(("Section", "Chapter")),
    caption=st.sampled_from(("Benefit Levels", "Payment Standards", "Need Standards")),
)
@example(section=214, anchor="Section", caption="Benefit Levels")
def test_generated_quantity_section_anchor_cannot_authenticate_an_amount(
    section, anchor, caption
):
    source = f"{anchor} {section}: {caption}\n{section}.50 Family Assistance"
    assert section + 0.5 in _recalled(source)
    assert section + 0.5 in _missing(source)


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        (
            "Section 214: Assistance\nProgram Schedule:\n"
            "214.50 Family Assistance\n214.75 Adult Assistance",
            OREGON_PAGE,
            {214.5, 214.75},
        ),
        (
            "Section 214: Assistance\ngroup classification\n"
            "214 Family Assistance\n214.50 Adult Assistance\n214.75 Children Assistance",
            OREGON_PAGE,
            {214, 214.5, 214.75},
        ),
        (
            "Section 214: Assistance\nBenefit amounts, in dollars:\n"
            "Family group classification and household eligibility application category name\n"
            "214.50 Family Assistance\n214.75 Adult Assistance",
            OREGON_PAGE,
            {214.5, 214.75},
        ),
        (
            "Section 214: Assistance\n214.50 Family Assistance\n214.75 Adult Assistance",
            OREGON_PAGE,
            {214.5, 214.75},
        ),
        (
            "Section 214: Assistance\n214 Family Assistance\n214.50 Adult Assistance",
            OREGON_PAGE,
            {214, 214.5},
        ),
        (
            "Section 214: Assistance\n214.50 Family Assistance\n214 Adult Assistance",
            OREGON_PAGE,
            {214, 214.5},
        ),
        (
            "214.50 Family Assistance\n214.75 Adult Assistance",
            "us-or/manual/odhs/open/214/page-93",
            {214.5, 214.75},
        ),
        (
            "214.50 Family Assistance\n214.75 Adult Assistance",
            "us-or/manual/odhs/open/table-of-contents/page-93",
            {214.5, 214.75},
        ),
        (
            "Section 214: Assistance\nFY 2026 Schedule:\n214.50 Family Assistance",
            OREGON_PAGE,
            {214.5},
        ),
        (
            "Section 214: Assistance\nThe following table lists payment standards.\n"
            "How Assistance Is Calculated\n214.50 Family Assistance",
            OREGON_PAGE,
            {214.5},
        ),
        (
            "Section 214: Assistance\nThe following table lists payment standards.\n"
            "Family group classification and household eligibility application category "
            "name for adult child elderly disabled applicant assistance groups and members\n"
            "214.50 Family Assistance",
            OREGON_PAGE,
            {214.5},
        ),
    ),
    ids=(
        "unknown-caption-without-column",
        "whole-dollar-row-before-decimals",
        "ordinary-column-longer-than-eight-words",
        "uncaptioned-table-under-section",
        "integer-and-one-decimal-under-section",
        "one-decimal-and-integer-under-section",
        "uncaptioned-table-on-matching-section-citation",
        "uncaptioned-table-on-contents-citation",
        "dated-unrecognised-caption-single-decimal",
        "ordinary-question-column-single-decimal",
        "ordinary-column-longer-than-eighteen-words-single-decimal",
    ),
)
def test_reviewed_table_runs_override_heading_authentication(
    source, citation, expected
):
    assert expected <= _recalled(source, citation)
    assert expected <= _missing(source, citation)


def test_numeric_table_run_scan_inserts_each_offset_in_linear_work(monkeypatch):
    """Count insertion attempts so quadratic prefix flushing fails deterministically."""

    counters = []

    class CountingSet(set):
        def __init__(self):
            super().__init__()
            self.insertions = 0
            counters.append(self)

        def add(self, item):
            self.insertions += 1
            super().add(item)

        def update(self, *iterables):
            for iterable in iterables:
                for item in iterable:
                    self.add(item)

    row = "214.50 Family Assistance\n"
    row_count = 1000
    source = row * row_count
    monkeypatch.setattr(sc, "set", CountingSet, raising=False)
    actual = sc._manual_ambiguous_numeric_line_starts(source)
    assert actual == frozenset(index * len(row) for index in range(row_count))
    assert sum(counter.insertions for counter in counters) <= 2 * row_count


@settings(max_examples=40, deadline=None)
@given(
    section=st.integers(min_value=100, max_value=899),
    year=st.integers(min_value=2020, max_value=2099),
    table_layout=st.sampled_from(
        (
            "Program Schedule:",
            "FY {year} Schedule:",
            "The following table lists payment standards.\nHow Assistance Is Calculated",
            "The following table lists payment standards.\n"
            "Family group classification and household eligibility application category "
            "name for adult child elderly disabled applicant assistance groups and members",
        )
    ),
    identity=st.sampled_from(("section", "citation", "contents")),
)
def test_generated_single_decimal_table_evidence_overrides_numbering(
    section, year, table_layout, identity
):
    prefix = f"Section {section}: Assistance\n" if identity == "section" else ""
    citation = OREGON_PAGE
    if identity == "citation":
        citation = f"us-or/manual/odhs/open/{section}/page-93"
    elif identity == "contents":
        citation = "us-or/manual/odhs/open/table-of-contents/page-93"
    source = f"{prefix}{table_layout.format(year=year)}\n{section}.50 Family Assistance"
    assert section + 0.5 in _recalled(source, citation)


def test_mixed_numeric_table_and_reviewed_heading_preserves_values_and_idempotence():
    source = (
        "Section 101: Assistance\n101.50 Family Assistance\n"
        "101.03 Application Process\n101.75 Adult Assistance"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    assert {101.5, 101.75} <= _recalled(source, citation)
    assert {101.5, 101.75} <= _missing(source, citation)
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=citation) == masked
    )


@pytest.mark.parametrize(
    "column", ("Type.", "Category.", "household type.", "No. of dependent children")
)
def test_quantity_caption_overrides_reviewed_titles_below_punctuated_columns(column):
    source = (
        f"Benefit amounts, in dollars:\n{column}\n"
        "101.03 Application Process\n101.04 Retroactive Applications"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    assert {101.03, 101.04} <= _recalled(source, citation)
    assert {101.03, 101.04} <= _missing(source, citation)


@settings(max_examples=40, deadline=None)
@given(
    caption=st.sampled_from(
        (
            "Benefit amounts, in dollars:",
            "The following table lists payment standards.",
            "Need Standards:",
            "SNAP MANUAL UTILITY ALLOWANCES",
        )
    ),
    column=st.sampled_from(
        ("Type.", "Category.", "household type.", "No. of dependent children")
    ),
    separator=st.sampled_from(("\n", "\n\n")),
)
@example(caption="Benefit amounts, in dollars:", column="Type.", separator="\n")
@example(caption="SNAP MANUAL UTILITY ALLOWANCES", column="Type.", separator="\n")
def test_generated_quantity_tables_override_reviewed_heading_titles(
    caption, column, separator
):
    source = (
        f"{caption}{separator}{column}{separator}"
        f"101.03 Application Process{separator}101.04 Retroactive Applications"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    assert {101.03, 101.04} <= _recalled(source, citation)
    assert {101.03, 101.04} <= _missing(source, citation)


@pytest.mark.parametrize("table_caption", (False, True), ids=("caption-only", "table"))
def test_institutional_caption_cannot_hide_an_ambiguous_quantity_table(table_caption):
    source = (
        ("Benefit amounts, in dollars:\n" if table_caption else "")
        + "SNAP MANUAL UTILITY ALLOWANCES\n"
        "214.2 Shared Utility Costs (continued)\n214.3 Telephone Allowance"
    )
    citation = "us-md/manual/fia/snap/214/page-5"
    expected = {214.2, 214.3}
    assert _recalled(source, citation) == expected
    assert _missing(source, citation) == expected


def test_original_combined_manual_heading_fixture_keeps_its_locator_exclusions():
    source = (
        "SNAP MANUAL UTILITY ALLOWANCES\n\n"
        "214.2 Shared Utility Costs (continued)\n\n"
        "D. The household pays $35 of the cost.\n\n"
        "214.3 Telephone Allowance\n\n"
        "A. The telephone allowance is $27.\n\n"
        "770-1 Advance Notice of Adverse Action\n\n"
        "Advance notice is 10 days."
    )
    citation = "us-wa/manual/dshs/eaz/example"
    assert _recalled(source, citation) == {10, 27, 35}
    assert _missing(source, citation) == {10, 27, 35}


@pytest.mark.parametrize(
    "entry", CORPUS_LOCATOR_CONTROLS, ids=lambda entry: entry["citation"]
)
@settings(max_examples=2, deadline=None)
@given(
    caption=st.sampled_from(
        ("Benefit amounts, in dollars:", "Applicable rates, in percent:")
    )
)
def test_pinned_corpus_predecessor_locator_exclusion_fails_closed_on_body_edits(
    entry, caption
):
    source, citation = entry["body"], entry["citation"]
    start, end = entry["numeric_span"]
    assert source[start:end] == entry["label"]
    value = float(entry["label"])
    assert value not in _recalled(source, citation)
    assert value not in _missing(source, citation)

    edited = source[:start] + caption + "\n" + source[start:]
    assert value in _recalled(edited, citation)
    assert value in _missing(edited, citation)


@pytest.fixture(scope="module")
def real_composed_maryland_source(tmp_path_factory):
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/manual_mask_review_composed_maryland.json"
        ).read_text()
    )
    scope = resolver.ReleaseScope(**fixture["scope"])
    root = tmp_path_factory.mktemp("composed-maryland")
    provisions = root / "data/corpus/provisions"
    path = provisions / "us-md/manual" / f"{scope.version}.jsonl"
    path.parent.mkdir(parents=True)
    raw = "".join(
        json.dumps(item["record"], ensure_ascii=False) + "\n"
        for item in fixture["rows"]
    ).encode()
    path.write_bytes(raw)
    artifact = resolver._ParsedCorpusArtifact(
        path,
        hashlib.sha256(raw).hexdigest(),
        len(raw),
        tuple((item["line_number"], item["record"]) for item in fixture["rows"]),
    )
    records = resolver._read_descendant_records(
        artifact,
        citation_path=fixture["citation"],
        active_scopes=frozenset((scope,)),
        provisions_root=provisions,
    )
    body, components = resolver._compose_descendant_text(
        fixture["citation"], records, expected_scope=scope
    )
    assert len(components) == len(records) == 10
    assert hashlib.sha256(body.encode()).hexdigest() == fixture["composed_body_sha256"]
    original = next(
        entry for entry in CORPUS_LOCATOR_CONTROLS if entry["label"] == "104.3"
    )
    assert (
        next(
            record.body
            for record in records
            if record.row.citation_path == original["citation"]
        )
        == original["body"]
    )
    return {**fixture, "body": body, "records": records, "scope": scope}


def _composed_maryland_with_table_caption(source, caption):
    records = []
    for record in source["records"]:
        if record.row.citation_path.endswith("/block-2"):
            column = "Certification Periods"
            assert len(caption) <= len(column)
            body = record.body.replace(column, caption.ljust(len(column)), 1)
            assert body != record.body and len(body) == len(record.body)
            record = replace(
                record,
                body=body,
                row=replace(
                    record.row, body_sha256=hashlib.sha256(body.encode()).hexdigest()
                ),
            )
        records.append(record)
    body, _components = resolver._compose_descendant_text(
        source["citation"], records, expected_scope=source["scope"]
    )
    return body


def test_real_composed_maryland_source_keeps_predecessor_locator_exclusions(
    real_composed_maryland_source,
):
    source = real_composed_maryland_source
    values = {float(span["label"]) for span in source["new_locator_spans"]}
    assert not (values & _recalled(source["body"], source["citation"]))
    assert not (values & _missing(source["body"], source["citation"]))

    edited = _composed_maryland_with_table_caption(source, "Benefit amounts:")
    assert 104.3 in _recalled(edited, source["citation"])
    assert 104.3 in _missing(edited, source["citation"])


def test_real_composed_maryland_locator_roles_require_the_original_manual_namespace(
    real_composed_maryland_source,
):
    source = real_composed_maryland_source
    citation = "us-md/manual/dhs/snap-alternative/104-self-employed-households"
    values = {float(span["label"]) for span in source["new_locator_spans"]}
    assert values <= _recalled(source["body"], citation)
    assert values <= _missing(source["body"], citation)


@settings(max_examples=4, deadline=None)
@given(
    caption=st.sampled_from(
        ("Benefit amounts:", "Payment Standards:", "Need Standards:", "Maximums:")
    )
)
def test_generated_real_composed_maryland_body_edits_keep_table_values(
    real_composed_maryland_source, caption
):
    source = real_composed_maryland_source
    edited = _composed_maryland_with_table_caption(source, caption)
    assert len(edited) == len(source["body"])
    assert edited != source["body"]
    masked = sc._without_us_manual_locators(
        edited, corpus_citation_path=source["citation"]
    )
    span = next(
        item for item in source["new_locator_spans"] if item["label"] == "104.3"
    )
    assert edited[span["start"] : span["end"]] == "104.3"
    assert masked[span["start"] : span["end"]] == "104.3"


@pytest.mark.parametrize(
    "matching_namespace", (True, False), ids=("recorded-manual", "other-manual")
)
@settings(max_examples=30, deadline=None)
@given(
    section=st.integers(min_value=100, max_value=899),
    caption=st.sampled_from(("Program Schedule:", "Programme Categories:")),
)
def test_generated_exact_locator_role_requires_original_body_and_manual_scope(
    matching_namespace, section, caption
):
    first = f"{section}.50"
    second = f"{section}.75"
    source = f"{caption}\n{first} Family Assistance\n{second} Adult Assistance"
    start = source.index(first)
    roles = {
        "us-or/manual/odhs/open": {
            hashlib.sha256(source.rstrip().encode()).hexdigest(): (
                (start, start + len(first)),
            )
        }
    }
    citation = OREGON_PAGE if matching_namespace else "us-ut/manual/dws/snap/page-93"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sc, "REVIEWED_MANUAL_LOCATOR_SPANS", roles)
        expected = {section + 0.75}
        if not matching_namespace:
            expected.add(section + 0.5)
        assert _recalled(source, citation) == expected
        masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
        assert len(masked) == len(source)
        assert (
            sc._without_us_manual_locators(masked, corpus_citation_path=citation)
            == masked
        )
        edited = source.replace("Family Assistance", "Primary Assistance")
        assert _recalled(edited, citation) == {section + 0.5, section + 0.75}


@st.composite
def _uncaptioned_numeric_siblings(draw):
    whole = draw(st.integers(min_value=100, max_value=999))
    first = draw(st.integers(min_value=1, max_value=98))
    second = draw(st.integers(min_value=first + 1, max_value=99))
    separator = draw(st.sampled_from(("\n", "\n\n")))
    rows = f"{whole}.{first:02d} Family Assistance{separator}{whole}.{second:02d} Adult Assistance"
    return rows, {whole + first / 100, whole + second / 100}


@settings(max_examples=40, deadline=None)
@given(table=_uncaptioned_numeric_siblings())
@example(table=("214.50 Family Assistance\n214.75 Adult Assistance", {214.5, 214.75}))
def test_generated_uncaptioned_decimal_siblings_preserve_each_value(table):
    source, expected = table
    assert _recalled(source) == expected
    assert (
        sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
        == source
    )


@settings(max_examples=40, deadline=None)
@given(
    start=st.integers(min_value=1, max_value=40),
    end=st.integers(min_value=41, max_value=99),
    caption=st.sampled_from(("", "Program Schedule:", "Unclassified Values:")),
    column=st.sampled_from(("", "household type", "assistance group", "Category")),
    contents_path=st.booleans(),
)
@example(start=1, end=2, caption="", column="household type", contents_path=True)
def test_generated_uncaptioned_and_unrecognised_range_tables_preserve_endpoints(
    start, end, caption, column, contents_path
):
    source = (
        f"{caption}\n{column}\n{start}-{end} Family Assistance\n"
        f"{end + 1}-{end + 2} Adult Assistance"
    )
    citation = (
        "us-or/manual/odhs/open/table-of-contents/page-93"
        if contents_path
        else OREGON_PAGE
    )
    assert {start, end, end + 1, end + 2} <= _recalled(source, citation)
    assert (
        sc._without_us_manual_locators(source, corpus_citation_path=citation) == source
    )


@settings(max_examples=50, deadline=None)
@given(
    section=st.integers(min_value=100, max_value=899),
    caption=st.sampled_from(
        (
            "The following table lists payment standards.",
            "The following table lists need standards.",
            "Benefit amounts, in dollars:",
            "Income",
            "Program Schedule:",
            "The following table lists programme classes.",
            "",
        )
    ),
    column=st.sampled_from(
        (
            "household type",
            "Household Type",
            "assistance group",
            "group classification",
            "Eligibility Category",
            "Family group classification and household eligibility application category name",
        )
    ),
    evidence=st.sampled_from(
        ("section", "identity", "contents", "contents-path", "nevada")
    ),
)
@example(
    section=214,
    caption="The following table lists payment standards.",
    column="household type",
    evidence="section",
)
@example(
    section=214,
    caption="Program Schedule:",
    column="group classification",
    evidence="section",
)
@example(
    section=214,
    caption="",
    column="Eligibility Category",
    evidence="section",
)
def test_generated_table_evidence_overrides_numbering_coincidences(
    section, caption, column, evidence
):
    prefix = ""
    citation = OREGON_PAGE
    if evidence == "section":
        prefix = f"Section {section}: Assistance"
    elif evidence == "identity":
        citation = f"us-or/manual/odhs/open/{section}/page-93"
    elif evidence == "contents":
        prefix = "CONTENTS PAGES"
    elif evidence == "contents-path":
        citation = "us-or/manual/odhs/open/table-of-contents/page-93"
    else:
        prefix = f"See A-{section} policies for background."
        citation = "us-nv/manual/dwss/eligibility-payments/page-93"
    source = (
        f"{prefix}\n{caption}\n{column}\n{section}.50 Family Assistance\n"
        f"{section}.75 Adult Assistance"
    )
    assert {section + 0.5, section + 0.75} <= _recalled(source, citation)


def test_real_georgia_page_prefix_does_not_expose_policy_locator():
    source = (
        "CAPS Policy – Provider Rights and Responsibilities No.: CAPS/00-12 "
        "Effective Date: 10/01/2016 Revised: 03/02/2026 "
        "Revision Effective: 03/02/2026\n\nCAPS Policy Manual\n\n"
        "Page 78 of 109 12.4.18 Warranties, Representations and Covenants"
    )
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path="us-ga/manual/decal/caps/policy-manual/page-78"
    )
    assert "12.4.18" not in cleaned
    assert "Warranties, Representations and Covenants" in cleaned


@settings(max_examples=30, deadline=None)
@given(
    page=st.integers(min_value=1, max_value=99),
    total=st.integers(min_value=100, max_value=999),
    section=st.integers(min_value=1, max_value=99),
    subsection=st.integers(min_value=1, max_value=99),
    provision=st.integers(min_value=1, max_value=99),
)
@example(page=78, total=109, section=12, subsection=4, provision=18)
def test_generated_generic_page_prefix_preserves_heading_exclusion(
    page, total, section, subsection, provision
):
    locator = f"{section}.{subsection}.{provision}"
    source = (
        f"Section {section}: Policy\nPage {page} of {total} {locator} "
        "Warranties, Representations and Covenants"
    )
    cleaned = sc.authoritative_numeric_recall_text(
        source,
        corpus_citation_path=f"us-ga/manual/decal/caps/policy-manual/page-{page}",
    )
    assert locator not in cleaned
    assert "Warranties, Representations and Covenants" in cleaned

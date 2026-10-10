"""Manual locator masks must preserve operative table values and body dates.

The twelve probes reproduce #1821 at the authenticated Oregon manual page
identity. Their source bodies are synthetic; the archived page does not
assert these amounts, rates, ranges or dates.
"""

from __future__ import annotations

import functools
import re

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

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
OPERATIVE_PROBES = (
    (
        "manual_operative_budget_row",
        "The following table lists benefit amounts in dollars.\n"
        "214.50 Monthly Allowance\n"
        "All households receive the listed allowance.",
        {214.5},
    ),
    (
        "manual_operative_budget_row_minimal",
        "Benefit amounts, in dollars:\n214.50 Monthly Allowance",
        {214.5},
    ),
    (
        "manual_operative_budget_row_percent",
        "The following table lists applicable rates in percent.\n"
        "7.65 Payroll Contribution\nAll employers pay the listed rate.",
        {7.65},
    ),
    (
        "manual_operative_annual_rate",
        "Applicable interest rates, in percent:\n7.5 Annual Rate",
        {7.5},
    ),
    (
        "manual_operative_children_range",
        "The following table lists the eligible child counts.\n1-2 Eligible Children",
        {1, 2},
    ),
    (
        "manual_operative_children_range_with_amount",
        "The payment is $444 for the range listed below.\n1-2 Eligible Children",
        {444, 1, 2},
    ),
    (
        "manual_operative_age_range",
        "Applicable claimant age bands in years:\n18-25 Young Adults",
        {18, 25},
    ),
    (
        "manual_operatively_parenthesized_month",
        "Applications filed in (07/2026) are eligible.",
        {7, 2026},
    ),
    (
        "manual_operative_date_sentence",
        "The application period is (07/2026).",
        {7, 2026},
    ),
    (
        "manual_operative_revision_as_body_formula",
        "For (07/2026), the payment is $444.",
        {7, 2026, 444},
    ),
    (
        "manual_operative_comparison_date",
        "Applications filed before (07/2026) are eligible.",
        {7, 2026},
    ),
    (
        "manual_operative_date_repeated",
        "Applications filed in July 2026 (07/2026) are eligible.",
        {7, 2026},
    ),
)


def _values(text: str) -> set[float]:
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            text, profile="en-US"
        )
    }


def _recalled_values(source: str, citation: str = OREGON_PAGE) -> set[float]:
    return _values(
        sc.authoritative_numeric_recall_text(source, corpus_citation_path=citation)
    )


def _missing_values(source: str, citation: str = OREGON_PAGE) -> set[float]:
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
        float(match.group(1))
        for issue in result.issues
        if (match := RECALL_VALUE.search(issue)) is not None
    }


@pytest.mark.parametrize(
    ("source", "expected"),
    [(source, expected) for _name, source, expected in OPERATIVE_PROBES],
    ids=[name for name, _source, _expected in OPERATIVE_PROBES],
)
def test_inherited_manual_mask_probes_recall_and_demand_each_value(source, expected):
    assert _recalled_values(source) == expected
    assert _missing_values(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (OPERATIVE_PROBES[index][1], OPERATIVE_PROBES[index][2])
        for index in (0, 7, 3, 4)
    ],
    ids=("amount", "date", "rate", "child-count-range"),
)
def test_reviewers_four_operative_regressions_on_manual_path(source, expected):
    assert expected <= _recalled_values(source)
    assert expected <= _missing_values(source)


@pytest.mark.parametrize(
    ("source", "citation"),
    (
        ("93 (07/2026) Chapter 2:Eligibility", OREGON_PAGE),
        (
            "26 Chapter 1: Introduction to the Oregon Programs Eligibility Notebook "
            "• Table of contents (07/2026) Introduction to the manual",
            "us-or/manual/odhs/open/page-26",
        ),
        (
            "770-1 Advance Notice of Adverse Action",
            "us-ut/manual/dws/snap/page-1",
        ),
        (
            "101.03 Application Process\n101.04 Retroactive Applications",
            "us-sc/manual/scdhhs/mppm/page-1",
        ),
        (
            "See WAC 388-450-0185 and policies 770-2, 770-3. Use Form IL-482-0634.",
            "us-wa/manual/dshs/eaz/example",
        ),
    ),
)
def test_authenticated_manual_locators_remain_excluded(source, citation):
    assert _recalled_values(source, citation) == set()


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("214.50 Monthly Allowance", {214.5}),
        ("7.65 Payroll Contribution", {7.65}),
        ("7.5 Annual Rate", {7.5}),
        ("1-2 Eligible Children", {1, 2}),
        ("18-25 Young Adults", {18, 25}),
        ("The application period is\n(07/2026).", {7, 2026}),
    ),
)
def test_ambiguous_or_quantity_describing_lines_fail_closed(source, expected):
    assert _recalled_values(source) == expected
    assert _missing_values(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Benefit amounts, in dollars:\n"
            "101.03 Application Process\n101.04 Retroactive Applications",
            {101.03, 101.04},
        ),
        (
            "93 (07/2026) Chapter 2:Eligibility "
            "Applications filed in (07/2026) are eligible.",
            {7, 2026},
        ),
        (
            "93 (07/2026) Chapter 2:Eligibility\n"
            "Benefit amounts, in dollars:\n214.50 Monthly Allowance\n"
            "Applications filed before (07/2026) are eligible.",
            {214.5, 7, 2026},
        ),
    ),
)
def test_authenticated_locators_do_not_authorize_masking_body_values(source, expected):
    assert _recalled_values(source) == expected
    assert _missing_values(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        (
            "Income brackets:\n100-200 Low Earners\n100-300 High Earners",
            {100, 200, 300},
        ),
        (
            "Chapter 1: Employment\n1-2 Adults Covered",
            {1, 2},
        ),
        (
            "TABLE OF CONTENTS\n1 Introduction\n\nThe benefit follows.\n"
            "214.50 Child Benefit",
            {214.5},
        ),
        (
            "Income brackets:\n300-200 Low Earners\n300-100 High Earners",
            {100, 200, 300},
        ),
    ),
    ids=(
        "ascending-income-row-keys",
        "adult-count-range-under-chapter",
        "body-benefit-after-table-of-contents",
        "descending-income-row-keys",
    ),
)
def test_reviewed_quantity_contexts_override_locator_evidence(source, expected):
    assert expected <= _recalled_values(source)
    assert expected <= _missing_values(source)


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        (
            "Permitted durations in months:\n101.03 Basic Category\n"
            "101.04 Extended Category",
            "us-sc/manual/scdhhs/mppm/page-1",
            {101.03, 101.04},
        ),
        (
            "2011.50 Filing Charge",
            "us-ak/manual/dpa/snap/"
            "transmittals-previous-transmittals-2011-05-9-11/block-1",
            {2011.5},
        ),
        (
            "7 (07/2026) Chapter members are eligible.",
            OREGON_PAGE,
            {7, 2026},
        ),
        (
            "See section 214 for procedural details.\n214.50 Personal Assistance",
            OREGON_PAGE,
            {214.5},
        ),
    ),
    ids=(
        "month-unit-caption",
        "transmittal-date-is-not-section-identity",
        "chapter-word-is-not-numbered-header",
        "cross-reference-is-not-own-section-numbering",
    ),
)
def test_reviewed_metadata_shapes_do_not_hide_operative_values(
    source, citation, expected
):
    assert expected <= _recalled_values(source, citation)
    assert expected <= _missing_values(source, citation)


def test_quantity_caption_governs_rows_beyond_the_local_context_window():
    source = (
        "Benefit amounts, in dollars:\n"
        + "100.01 Basic Category\n" * 220
        + "101.03 Other Category\n101.04 Extended Category"
    )
    assert _recalled_values(source) == {100.01, 101.03, 101.04}
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    assert masked == source


@pytest.mark.parametrize("number", ("0.01", "7.65", "2026", "1-2"))
def test_month_column_header_preserves_numeric_row_text_and_mask_idempotence(number):
    source = f"Benefit amounts, in dollars:\nMay\n{number} Monthly Allowance"
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    assert masked == source
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=OREGON_PAGE)
        == masked
    )
    assert (
        sc.authoritative_numeric_recall_text(source, corpus_citation_path=OREGON_PAGE)
        == source
    )


def test_real_oregon_reversed_header_excludes_revision_stamp():
    source = (
        "(07/2026) OREGON PROGRAMS ELIGIBILITY NOTEBOOK (OPEN) 1314 "
        "Chapter 9: Summer EBT Program (SEBT)"
    )
    citation = "us-or/manual/odhs/open/page-1314"
    # Main already demands the interior page number. Preserve that inventory
    # while retaining the established exclusion of the revision stamp.
    assert _recalled_values(source, citation) == {1314}
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert len(masked) == len(source)
    assert "Chapter 9: Summer EBT Program (SEBT)" in masked


def test_real_virginia_underscored_form_footer_excludes_revision_stamp():
    source = "Other " + "_" * 80 + " " + "_" * 88 + " 032-02-0072-13-eng (09/2024)"
    citation = "us-va/manual/dss/snap/full-manual/page-548"
    assert not {9, 2024} & _recalled_values(source, citation)
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert len(masked) == len(source)
    assert masked[: source.index("(09/2024)")] == source[: source.index("(09/2024)")]


@pytest.mark.parametrize(
    ("source", "citation", "body_stamp"),
    (
        (
            "032-03-0018-34-eng (12/2023) (07/2026) applicants are eligible.",
            "us-va/manual/dss/snap/full-manual/page-522",
            "(07/2026)",
        ),
        (
            "OREGON PROGRAMS ELIGIBILITY NOTEBOOK (OPEN) "
            "100-13450_DHS 2818 (07/2026) (08/2027)",
            "us-or/manual/odhs/open/page-1",
            "(08/2027)",
        ),
    ),
)
def test_masking_publication_stamp_does_not_authenticate_following_body_date(
    source, citation, body_stamp
):
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert body_stamp in masked
    assert _values(body_stamp) <= _recalled_values(source, citation)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=citation) == masked
    )


@pytest.mark.parametrize("count", (7, 8))
@pytest.mark.parametrize("separator", (" ", "\n"))
def test_leading_count_and_month_year_are_not_authenticated_page_metadata(
    count, separator
):
    source = f"{count} (07/2026){separator}applicants are eligible."
    assert _recalled_values(source) == {count, 7, 2026}
    assert _missing_values(source) == {count, 7, 2026}
    assert (
        sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
        == source
    )


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        (
            "SNAP\n\nSNAP Standard Deductions\n\nOther SNAP Deductions\n\n"
            "SNAP Utility Standards (10/2025)\n\nSNAP Resource Limits\n\n"
            "Program Standards",
            "us-il/manual/dhs/csmm/21738/block-1",
            {10, 2025},
        ),
        (
            "Type of Home Heating\n\nPercentage Increments Natural Gas\n\n"
            "1.198 Liquefied Petroleum Gas (LPG)\n\n1.089 Coal 1.089 Wood "
            "1.089 Electricity 1.750 Fuel Oil and Kerosene 1.089",
            "us-wv/manual/bfa/income-maintenance-manual/page-1756",
            {1.198, 1.089, 1.75},
        ),
    ),
    ids=(
        "illinois-utility-standard-applicability-date",
        "west-virginia-heating-factor",
    ),
)
def test_real_manual_operative_excerpts_are_recalled_and_demanded(
    source, citation, expected
):
    assert _recalled_values(source, citation) == expected
    assert _missing_values(source, citation) == expected


@pytest.mark.parametrize(
    ("source", "citation", "stamp", "operative_body"),
    (
        (
            "Issuance/Administrative Worker\n\nDate\n\n032-03-0387-07-eng (09/2024)",
            "us-va/manual/dss/snap/full-manual/page-545",
            "(09/2024)",
            "Issuance/Administrative Worker\n\nDate",
        ),
        (
            "This institution is an equal opportunity provider "
            "032-03-0051-44-eng (09/2025) Chart 1 (Gross Income Limit 200%) "
            "Chart 2 (Gross Income Limit 130%) HH Size",
            "us-va/manual/dss/snap/full-manual/page-529",
            "(09/2025)",
            "Chart 1 (Gross Income Limit 200%) Chart 2 (Gross Income Limit 130%) HH Size",
        ),
        (
            "WISCONSIN DEPARTMENT OF HEALTH SERVICES Division of Medicaid "
            "Services P-16001 (04/2026)\n\nFoodShare Handbook Release 26-01",
            "us-wi/manual/dhs/foodshare/handbook/release-26-01/page-1",
            "(04/2026)",
            "FoodShare Handbook Release 26-01",
        ),
    ),
    ids=("virginia-form-footer", "virginia-form-stamp-before-table", "wisconsin-cover"),
)
def test_real_manual_publication_stamps_are_excluded_without_changing_body(
    source, citation, stamp, operative_body
):
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert len(masked) == len(source)
    assert stamp not in masked
    assert not _values(stamp) & _recalled_values(source, citation)
    assert operative_body in masked


TABLE_CAPTIONS_AND_LABELS = (
    ("Benefit amounts, in dollars:", "Monthly Allowance"),
    ("Applicable interest rates, in percent:", "Annual Rate"),
    ("The following table lists eligible child counts.", "Eligible Children"),
    ("Applicable claimant age bands in years:", "Young Adults"),
    ("Applicable percentages:", "Payroll Contribution"),
    ("Income brackets:", "Low Earners"),
    ("Permitted durations in months:", "Basic Category"),
)


@st.composite
def _table_rows(draw):
    kind = draw(st.sampled_from(("decimal", "integer", "range")))
    if kind == "decimal":
        cents = draw(st.integers(min_value=1, max_value=999999))
        token = f"{cents // 100}.{cents % 100:02d}"
        return token, {cents / 100}
    if kind == "range":
        start = draw(st.integers(min_value=1, max_value=98))
        end = draw(st.integers(min_value=start + 1, max_value=99))
        return f"{start}-{end}", {start, end}
    number = draw(st.integers(min_value=1, max_value=9999))
    return str(number), {number}


@st.composite
def _quantity_tables(draw, *, include_month_column=False):
    caption, label = draw(st.sampled_from(TABLE_CAPTIONS_AND_LABELS))
    label = draw(
        st.sampled_from(
            (label, "Basic Category", "Application Process", "Extended Category")
        )
    )
    rows = draw(st.lists(_table_rows(), min_size=1, max_size=8))
    headers = ("", "Category", "Description", "$", "%")
    if include_month_column:
        headers = (*headers, "May")
    column_header = draw(st.sampled_from(headers))
    source = "\n".join(
        [caption, column_header, *(f"{token} {label}" for token, _values in rows)]
    )
    expected = set().union(*(values for _token, values in rows))
    return source, expected


@settings(max_examples=100, deadline=None)
@given(table=_quantity_tables())
def test_generated_quantity_table_recalls_every_row_number(table):
    source, expected = table
    assert _recalled_values(source) == expected


@settings(max_examples=100, deadline=None)
@given(
    caption=st.sampled_from([caption for caption, _label in TABLE_CAPTIONS_AND_LABELS]),
    section=st.integers(min_value=100, max_value=899),
    separator=st.sampled_from((".", "-")),
    column_header=st.sampled_from(("", "Category", "Description", "$", "%")),
)
def test_generated_quantity_caption_overrides_matching_identity_and_heading_sequence(
    caption, section, separator, column_header
):
    source = (
        f"{caption}\n{column_header}\n{section}{separator}03 Basic Category\n"
        f"{section}{separator}04 Extended Category"
    )
    citation = f"us-sc/manual/scdhhs/mppm/{section}/page-1"
    assert _recalled_values(source, citation) == _values(source)


@settings(max_examples=100, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    sentence=st.sampled_from(
        (
            "Applications filed in ({date}) are eligible.",
            "Applications filed before ({date}) are eligible.",
            "The application period is ({date}).",
            "For ({date}), households receive the listed allowance.",
        )
    ),
)
def test_generated_parenthesized_body_month_year_is_recalled(month, year, sentence):
    source = sentence.format(date=f"{month:02d}/{year}")
    assert _recalled_values(source) == {month, year}


@settings(max_examples=100, deadline=None)
@given(
    page=st.integers(min_value=1, max_value=999),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    section=st.integers(min_value=100, max_value=899),
)
def test_generated_authenticated_header_and_heading_sequence_only_mask_locators(
    page, month, year, section
):
    stamp = f"({month:02d}/{year})"
    first_heading = f"{section}.03"
    second_heading = f"{section}.04"
    body = (
        "Households receive $444 and must apply within 30 days.\n"
        "Applications filed in (11/2027) are eligible."
    )
    source = (
        f"{page} {stamp} Chapter 2:Eligibility\n"
        f"{first_heading} Application Process\n"
        f"{second_heading} Retroactive Applications\n{body}"
    )
    citation = f"us-sc/manual/scdhhs/mppm/page-{page}"

    assert _recalled_values(source, citation) == {444, 30, 11, 2027}
    masked = sc._without_us_manual_locators(source, corpus_citation_path=citation)
    assert len(masked) == len(source)
    assert masked.endswith(body)
    masked_offsets = set(range(len(str(page))))
    for token in (stamp, first_heading, second_heading):
        start = source.index(token)
        masked_offsets.update(range(start, start + len(token)))
        assert not any(
            character.isdigit() for character in masked[start : start + len(token)]
        )
    assert all(
        masked[index] == character
        for index, character in enumerate(source)
        if index not in masked_offsets
    )


@settings(max_examples=150, deadline=None)
@given(
    source=st.one_of(
        st.text(
            alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            " .,:;!?()/$%\n\t-•",
            max_size=400,
        ),
        _quantity_tables(include_month_column=True).map(lambda table: table[0]),
        st.sampled_from(
            [source for _name, source, _expected in OPERATIVE_PROBES]
            + [
                "93 (07/2026) Chapter 2:Eligibility\n"
                "101.03 Application Process\n101.04 Retroactive Applications\n"
                "The allowance is $444.",
                "26 Chapter 1: Introduction • Table of contents (07/2026)",
            ]
        ),
    )
)
def test_manual_locator_mask_is_idempotent_and_never_lengthens(source):
    masked = sc._without_us_manual_locators(source, corpus_citation_path=OREGON_PAGE)
    assert len(masked) <= len(source)
    assert (
        sc._without_us_manual_locators(masked, corpus_citation_path=OREGON_PAGE)
        == masked
    )

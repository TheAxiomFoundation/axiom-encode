"""Generated complete-source properties for fail-safe manual locator masks."""

from __future__ import annotations

import functools
import re

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

EMPTY_RULES = "format: rulespec/v1\nmodule: {}\nrules: []\n"
RECALL_VALUE = re.compile(
    r"\[complete-source-unit:numeric-recall\] Authoritative corpus numeric "
    r"value (\S+) has no named scalar"
)
MONETARY_CORES = (
    "Monetary values",
    "Fee",
    "Fees",
    "Fee rates",
    "Payment",
    "Payments",
    "Amount",
    "Amounts",
    "Rate",
    "Rates",
    "Standard",
    "Standards",
    "Dollar",
    "Dollars",
    "Fees in dollars",
    "Payment amounts",
    "Payment standards",
    "Dollar amounts",
    "Rates in percent",
    "Monetary standards",
    "Benefit amounts",
    "Unit cost in USD",
)
QUALIFIER_DECORATIONS = (
    "words",
    "parentheses",
    "footnote",
    "units",
    "year",
    "month-year",
    "nested-parentheses",
    "line-footnote",
    "mixed",
    "slash-month-year",
)
REFERENCES = (
    "The value shown below determines eligibility.",
    "The value shown above determines eligibility.",
    "The month shown determines eligibility.",
    "The date shown determines eligibility.",
    "The year shown determines eligibility.",
    "The month shown below determines eligibility.",
    "The month shown above determines eligibility.",
    "The date shown below determines eligibility.",
    "The date shown above determines eligibility.",
    "The year shown below determines eligibility.",
    "The year shown above determines eligibility.",
    "This date determines eligibility.",
    "The effective date determines eligibility.",
    "Eligibility applies as of the printed date.",
    "Section 214: Effective Date",
    "Section 214: This Date",
)
STAMP_PREFIXES = (
    ("032-03-0445-01", "us-va/manual/dss/snap/page-1"),
    (
        "WISCONSIN DEPARTMENT OF HEALTH SERVICES Division of Medicaid Services P-16001",
        "us-wi/manual/dhs/foodshare/page-1",
    ),
    (
        "OREGON PROGRAMS ELIGIBILITY NOTEBOOK (OPEN) 100-13450_DHS 2818",
        "us-or/manual/odhs/open/page-1",
    ),
    (
        "93 Chapter 4:Working with families • Section 10: Support service payments\n",
        "us-or/manual/odhs/open/page-93",
    ),
)
QUALIFIER_WORDS = st.one_of(
    st.sampled_from(
        ("required", "eligible", "must", "shall", "may", "provided", "applies")
    ),
    st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=80),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "M")),
        min_size=1,
        max_size=80,
    ),
)
QUALIFIERS = st.lists(QUALIFIER_WORDS, min_size=1, max_size=8).map(" ".join)


def _demanded(source: str, citation: str, profile: str) -> set[float]:
    """Observe the public complete-source gate, with no candidate scalars."""
    result = sc.analyze_complete_source_unit(
        EMPTY_RULES,
        source,
        corpus_citation_path=citation,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        float(match.group(1))
        for issue in result.issues
        if (match := RECALL_VALUE.search(issue)) is not None
    }


def _raw_values(source: str, profile: str) -> set[float]:
    return {
        occurrence.value
        for occurrence in extract_typed_numeric_inventory_occurrences_from_text(
            source, profile=profile
        )
    }


def _assert_demanded(source: str, citation: str, profile: str, expected: set[float]):
    demanded = _demanded(source, citation, profile)
    assert expected <= demanded, (source, citation, profile, expected - demanded)


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "sole-body-chunk"))
@pytest.mark.parametrize(
    ("citation", "source", "expected"),
    (
        (
            "us-sc/manual/scdhhs/mppm/page-1",
            "Section 214: Applications\nFees in dollars (required)\nCategory\n"
            "214.03 Application Process\n214.04 Retroactive Applications",
            {214.03, 214.04},
        ),
        (
            "us-va/manual/dss/snap/page-1",
            "The month shown below determines eligibility.\n032-03-0445-01 (07/2026)",
            {7, 2026},
        ),
        (
            "us-sc/manual/scdhhs/mppm/page-1",
            "Section 214: Applications\nFees in dollars (May-2026)\nCategory\n"
            "214.03 Application Process\n214.04 Retroactive Applications",
            {214.03, 214.04},
        ),
    ),
    ids=("round-five-fee", "round-five-footer", "round-five-fee-month-year"),
)
def test_round_five_operative_values_are_demanded(
    citation, source, expected, profile, composed
):
    if composed:
        source = "Page 1\n\n" + source
    _assert_demanded(source, citation, profile, expected)


@pytest.mark.parametrize("core", MONETARY_CORES)
@pytest.mark.parametrize("decoration", QUALIFIER_DECORATIONS)
@settings(max_examples=4, deadline=None)
@given(
    qualifier=QUALIFIERS,
    unit=st.one_of(
        st.sampled_from(("members", "dollars", "USD", "%", "months", "m²", "kWh")),
        QUALIFIER_WORDS,
    ),
    footnote=st.sampled_from(("*", "†", "‡", "¹", "[note]")),
    year=st.integers(min_value=1900, max_value=2099),
    count=st.integers(min_value=1, max_value=20),
    section=st.sampled_from((101, 214)),
    column=st.booleans(),
    composed=st.booleans(),
    profile=st.sampled_from(("en-US", "legacy")),
)
@example(
    qualifier="required",
    unit="members",
    footnote="*",
    year=2026,
    count=2,
    section=214,
    column=False,
    composed=False,
    profile="en-US",
)
@example(
    qualifier="required",
    unit="USD",
    footnote="†",
    year=2026,
    count=2,
    section=214,
    column=False,
    composed=True,
    profile="legacy",
)
@example(
    qualifier="required " + "eligible " * 40,
    unit="months",
    footnote="‡",
    year=2026,
    count=2,
    section=214,
    column=True,
    composed=True,
    profile="en-US",
)
def test_generated_monetary_caption_and_column_qualifiers_keep_every_value(
    core,
    qualifier,
    unit,
    footnote,
    decoration,
    year,
    count,
    section,
    column,
    composed,
    profile,
):
    """An operative caption core wins over arbitrary qualifiers and heading titles."""
    suffixes = {
        "words": f" {qualifier}",
        "parentheses": f" ({qualifier})",
        "footnote": f"{footnote} [{qualifier}] [{count}]",
        "units": f" per {count} {unit} ({qualifier})",
        "year": f" ({qualifier}; {year})",
        "month-year": f" ({qualifier}; May-{year})",
        "nested-parentheses": f" (({qualifier}) [{count}]; May-{year})",
        "line-footnote": f"{footnote}\n[{count}] {qualifier} ({year})",
        "mixed": f" ({qualifier}) {footnote}[{count}] per {count} {unit} — May {year}",
        "slash-month-year": f" ({qualifier}; May/{year})",
    }
    caption = core + suffixes[decoration]
    context = (
        f"Program Inventory\nCategory\n{caption}\nClassification"
        if column
        else f"{caption}\nCategory"
    )
    rows = f"{section}.03 Application Process\n{section}.04 Retroactive Applications"
    source = f"Section {section}: Applications\n{context}\n{rows}"
    expected = _raw_values(caption, profile) | _raw_values(rows, profile)
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    shapes = (
        _whole_body_composition_shapes(source, citation)
        if composed
        else ((source, citation),)
    )
    for text, composed_citation in shapes:
        _assert_demanded(text, composed_citation, profile, expected)


@settings(max_examples=60, deadline=None)
@given(
    core=st.sampled_from(MONETARY_CORES),
    qualifier=st.one_of(st.just(""), QUALIFIERS),
    section=st.sampled_from((101, 214)),
    numbered_column=st.booleans(),
    profile=st.sampled_from(("en-US", "legacy")),
)
@example(
    core="Payment", qualifier="", section=214, numbered_column=False, profile="en-US"
)
@example(
    core="Payment",
    qualifier="required",
    section=214,
    numbered_column=True,
    profile="legacy",
)
def test_generated_monetary_anchor_and_numbered_column_keep_every_value(
    core, qualifier, section, numbered_column, profile
):
    """Numbering and a familiar title cannot authenticate an operative column."""
    suffix = f" ({qualifier})" if qualifier else ""
    context = (
        f"Section {section}: Applications\n{section}.02 {core} Entries{suffix}"
        if numbered_column
        else f"Section {section}: {core}{suffix}"
    )
    rows = f"{section}.03 Application Process\n{section}.04 Retroactive Applications"
    source = f"{context}\n{rows}"
    expected = _raw_values(rows, profile)
    if numbered_column:
        expected |= _raw_values(context.rsplit("\n", 1)[1], profile)
    for text, citation in _whole_body_composition_shapes(
        source, "us-sc/manual/scdhhs/mppm/page-1"
    ):
        _assert_demanded(text, citation, profile, expected)


@pytest.mark.parametrize(
    "core",
    ("amount", "payment", "limit", "household-payment-window", "application-date"),
)
@settings(max_examples=10, deadline=None)
@given(
    prices=st.lists(
        st.tuples(
            # Public recall diagnostics use :g; cents stay exact within six digits.
            st.integers(min_value=1, max_value=9999),
            st.integers(min_value=0, max_value=99),
        ).filter(lambda price: price[0] not in (101, 214) or price[1] not in (3, 4)),
        min_size=1,
        max_size=4,
    ),
    fractional=st.booleans(),
    section=st.sampled_from((101, 214)),
    days=st.integers(min_value=1, max_value=365),
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
)
@example(prices=[(444, 0)], fractional=False, section=214, days=45, month=7, year=2026)
@example(
    prices=[(444, 25), (555, 75)],
    fractional=True,
    section=101,
    days=60,
    month=10,
    year=2025,
)
def test_generated_declarative_monetary_context_before_rows_keeps_every_value(
    core, prices, fractional, section, days, month, year
):
    """Self-contained amounts, windows and dates can caption any following row."""
    amounts = [
        f"${whole}.{cents:02d}" if fractional else f"${whole}"
        for whole, cents in prices
    ]
    if core == "household-payment-window":
        statement = (
            f"Households receive {amounts[0]} and must apply within {days} days."
        )
    elif core == "application-date":
        statement = f"Applications filed in ({month:02d}/{year}) are eligible."
    else:
        statement = f"The {core} is " + " and ".join(amounts) + "."
    anchor = f"Section {section}: Applications"
    first = f"{section}.03 Application Process"
    last = f"{section}.04 Retroactive Applications"
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    for profile in ("en-US", "legacy"):
        row_values = _raw_values(first + "\n" + last, profile)
        stated_values = _raw_values(statement, profile)
        assert not row_values & stated_values
        for body in (
            f"{anchor}\n{statement}\n{first}\n{last}",
            f"{anchor}\n{first}\n{statement}\n{last}",
        ):
            for source, path in _whole_body_composition_shapes(body, citation):
                _assert_demanded(source, path, profile, row_values | stated_values)

        # After every heading, the fact cannot supply a following-row caption.
        body = f"{anchor}\n{first}\n{last}\n{statement}"
        root = citation.rsplit("/page-", 1)[0]
        for source, path in (
            (body, citation),
            ("Page 1\n\n" + body, citation),
            ("Page 1\n\n" + body, root),
        ):
            assert _demanded(source, path, profile) == stated_values


@pytest.mark.parametrize(
    "title_template",
    (
        "The Date (Shown) Determines Eligibility",
        "The Month ({qualifier}) Shown Determines Eligibility",
        "The Year Shown ({qualifier}) Determines Eligibility",
        "This ({qualifier}) Date",
        "Effective ({qualifier}) Date",
        "Shown (Below)",
        "Shown ({qualifier}) Above",
        "As (Of)",
        "As ({qualifier}) Of",
        "Related Policy ({qualifier})",
    ),
)
@settings(max_examples=8, deadline=None)
@given(
    qualifier=QUALIFIERS,
    section=st.sampled_from((101, 214)),
    profile=st.sampled_from(("en-US", "legacy")),
)
@example(qualifier="Printed", section=214, profile="en-US")
@example(qualifier="required", section=214, profile="legacy")
def test_generated_unknown_numbered_titles_cannot_authorize_row_masks(
    title_template, qualifier, section, profile
):
    """Numbered titles with unknown or wrapped references cannot prove safety."""
    title = title_template.format(qualifier=qualifier).title()
    rows = f"{section}.03 Application Process\n{section}.04 Retroactive Applications"
    body = f"Section {section}: Applications\n{section}.02.1 {title}\n{rows}"
    expected = _raw_values(rows, profile)
    for source, citation in _whole_body_composition_shapes(
        body, "us-sc/manual/scdhhs/mppm/page-1"
    ):
        _assert_demanded(source, citation, profile, expected)


@settings(max_examples=30, deadline=None)
@given(
    caption=QUALIFIERS,
    profile=st.sampled_from(("en-US", "legacy")),
    composed=st.booleans(),
)
@example(caption="Program Schedule", profile="en-US", composed=False)
def test_generated_unknown_caption_retains_uncertain_row_values(
    caption, profile, composed
):
    """Unclassified caption text is insufficient proof that its numbers are locators."""
    source = (
        "Section 214: Applications\n"
        f"{caption}\nCategory\n214.03 Application Process\n"
        "214.04 Retroactive Applications"
    )
    if composed:
        source = "Page 1\n\n" + source
    _assert_demanded(
        source, "us-sc/manual/scdhhs/mppm/page-1", profile, {214.03, 214.04}
    )


@settings(max_examples=12, deadline=None)
@given(
    section=st.sampled_from((101, 214)),
    profile=st.sampled_from(("en-US", "legacy")),
)
def test_generated_genuine_locator_scheme_remains_nonoperative(section, profile):
    """A closed context consisting of authenticated procedure headings still masks."""
    body = (
        f"Section {section}: Applications\n"
        f"{section}.03 Application Process\n{section}.04 Retroactive Applications"
    )
    citation = "us-sc/manual/scdhhs/mppm/page-1"
    root = citation.rsplit("/page-", 1)[0]
    for source, path in (
        (body, citation),
        ("Page 1\n\n" + body, citation),
        ("Page 1\n\n" + body, root),
    ):
        assert not (_raw_values(body, profile) & _demanded(source, path, profile))


@pytest.mark.parametrize(
    ("citation", "body", "labels", "expected_body_values"),
    (
        (
            "us-wa/manual/dshs/eaz/example",
            "SNAP MANUAL UTILITY ALLOWANCES\n\n"
            "214.2 Shared Utility Costs (continued)\n\n"
            "D. The household pays $35 of the cost.\n\n"
            "214.3 Telephone Allowance\n\n"
            "A. The telephone allowance is $27.\n\n"
            "770-1 Advance Notice of Adverse Action\n\n"
            "Advance notice is 10 days.",
            ("214.2", "214.3", "770-1"),
            {10, 27, 35},
        ),
        (
            "us-ut/manual/dws/snap/page-1",
            "$470.00 Supplemental Security Income (SSI)\n"
            "770-1 Advance Notice of Adverse Action",
            ("770-1",),
            {470},
        ),
    ),
    ids=("combined-locators", "independent-amount-neighbor"),
)
@settings(max_examples=10, deadline=None)
@given(
    core=st.sampled_from(MONETARY_CORES),
    qualifier=QUALIFIERS,
    profile=st.sampled_from(("en-US", "legacy")),
)
@example(core="Fees in dollars", qualifier="required", profile="en-US")
@example(core="Payment standards", qualifier="must", profile="legacy")
def test_generated_exact_locator_roles_fail_closed_on_context_edits(
    citation, body, labels, expected_body_values, core, qualifier, profile
):
    """An exact source role cannot authorize any altered caption, body or namespace."""
    assert _demanded(body, citation, profile) == expected_body_values
    label_values = _raw_values("\n".join(labels), profile)
    caption = f"{core} ({qualifier})"
    edited_citation = citation.rsplit("/", 1)[0] + "/unreviewed-role"
    for source, path in (
        (body + "\n" + qualifier, citation),
        (caption + "\n" + body, citation),
        (body, edited_citation),
    ):
        _assert_demanded(source, path, profile, label_values | expected_body_values)


def _whole_body_composition_shapes(body: str, citation: str):
    """Place one whole body in each resolver-style complete-source layout."""
    root, page = citation.rsplit("/page-", 1)
    previous, following = int(page) - 1, int(page) + 1
    harmless = "Applications are reviewed."
    return (
        (body, citation),
        (body, root),
        (f"Page {page}\n\n{body}", citation),
        (f"Page {page}\n\n{body}", root),
        (f"Manual contents\n\nPage {page}\n\n{body}", root),
        (f"Page {previous}\n\n{harmless}\n\nPage {page}\n\n{body}", root),
        (f"Page {page}\n\n{body}\n\nPage {following}\n\n{harmless}", root),
        (
            f"Page {previous}\n\n{harmless}\n\nPage {page}\n\n{body}"
            f"\n\nPage {following}\n\n{harmless}",
            root,
        ),
    )


def _referenced_stamp_shapes(
    reference: str, stamp: str, citation: str, preceding: bool
):
    root = citation.rsplit("/page-", 1)[0]
    page = citation.rsplit("/page-", 1)[1]
    before, after = (reference, stamp) if preceding else (stamp, reference)
    body = before + "\n" + after
    return _whole_body_composition_shapes(body, citation) + (
        (f"Page {page}\n\n{before}\n\nPage {int(page) + 1}\n\n{after}", root),
        (f"{before}\n\nPage {page}\n\n{after}", root),
        (
            f"Page {page}\n\n{before}\n\nPage {int(page) + 1}\n\n"
            f"Applications are reviewed.\n\nPage {int(page) + 2}\n\n{after}",
            root,
        ),
        (
            f"Page 90\n\nApplications are reviewed.\n\nPage {page}\n\n{body}"
            "\n\nPage 94\n\nApplications are reviewed.",
            root,
        ),
    )


@pytest.mark.parametrize("reference", REFERENCES)
@pytest.mark.parametrize("preceding", (False, True), ids=("following", "preceding"))
@settings(max_examples=10, deadline=None)
@given(
    month=st.integers(min_value=1, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    prefix_and_citation=st.sampled_from(STAMP_PREFIXES),
    newline=st.sampled_from(("\n", "\r\n")),
)
@example(month=7, year=2026, prefix_and_citation=STAMP_PREFIXES[0], newline="\n")
@example(month=7, year=2026, prefix_and_citation=STAMP_PREFIXES[1], newline="\n")
@example(month=7, year=2026, prefix_and_citation=STAMP_PREFIXES[2], newline="\n")
@example(month=7, year=2026, prefix_and_citation=STAMP_PREFIXES[3], newline="\n")
def test_generated_referenced_stamp_date_survives_every_source_shape(
    reference, preceding, month, year, prefix_and_citation, newline
):
    """Any date reference in the whole source prevents revision-date masking."""
    prefix, citation = prefix_and_citation
    stamp = f"{prefix} ({month:02d}/{year})"
    for source, composed_citation in _referenced_stamp_shapes(
        reference, stamp, citation, preceding
    ):
        source = source.replace("\n", newline)
        for profile in ("en-US", "legacy"):
            _assert_demanded(source, composed_citation, profile, {month, year})


@settings(max_examples=12, deadline=None)
@given(
    month=st.integers(min_value=7, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    prefix_and_citation=st.sampled_from(STAMP_PREFIXES[:3]),
    profile=st.sampled_from(("en-US", "legacy")),
)
def test_generated_unreferenced_stamps_remain_nonoperative(
    month, year, prefix_and_citation, profile
):
    """Fail-safe reference checks preserve the authenticated-stamp goal."""
    prefix, citation = prefix_and_citation
    body = f"{prefix} ({month:02d}/{year})"
    for source in (body, "Page 1\n\n" + body):
        assert not ({month, year} & _demanded(source, citation, profile))


@pytest.mark.parametrize("profile", ("en-US", "legacy"))
@pytest.mark.parametrize("composed", (False, True), ids=("raw", "sole-body-chunk"))
@pytest.mark.parametrize("root_citation", (False, True), ids=("page", "root"))
def test_operative_amount_colliding_with_authenticated_stamp_page_is_demanded(
    profile, composed, root_citation
):
    source = "1225 (07/2026) Chapter 2:Eligibility\nThe amount is $12.25."
    citation = "us-or/manual/odhs/open/page-1225"
    if composed:
        source = "Page 1\n\n" + source
    if root_citation:
        citation = citation.rsplit("/page-", 1)[0]
    _assert_demanded(source, citation, profile, {12.25})


@settings(max_examples=12, deadline=None)
@given(
    whole=st.integers(min_value=1, max_value=99),
    cents=st.integers(min_value=1, max_value=99),
    month=st.integers(min_value=7, max_value=12),
    year=st.integers(min_value=1900, max_value=2099),
    reference=st.one_of(st.none(), st.sampled_from(REFERENCES)),
    placement=st.sampled_from(("preceding", "following", "between")),
)
@example(whole=12, cents=25, month=7, year=2026, reference=None, placement="following")
@example(
    whole=12,
    cents=25,
    month=7,
    year=2026,
    reference="The month shown above determines eligibility.",
    placement="following",
)
@example(
    whole=12,
    cents=25,
    month=7,
    year=2026,
    reference="This date determines eligibility.",
    placement="preceding",
)
@example(
    whole=12,
    cents=25,
    month=7,
    year=2026,
    reference="The date shown above determines eligibility.",
    placement="between",
)
def test_generated_stamp_page_collisions_keep_operative_amounts(
    whole, cents, month, year, reference, placement
):
    """A genuine page locator must not displace a decimal amount in composition."""
    page = whole * 100 + cents
    amount = f"{whole}.{cents:02d}"
    body = (
        f"{page} ({month:02d}/{year}) Chapter 2:Eligibility\nThe amount is ${amount}."
    )
    if reference is not None:
        if placement == "preceding":
            body = reference + "\n" + body
        elif placement == "following":
            body += "\n" + reference
        else:
            stamp, money = body.split("\n", 1)
            body = stamp + "\n" + reference + "\n" + money
    citation = f"us-or/manual/odhs/open/page-{page}"
    shapes = _whole_body_composition_shapes(body, citation) + (
        ("Page 1\n\n" + body, citation),
        ("Page 1\n\n" + body, citation.rsplit("/page-", 1)[0]),
    )
    expected = {float(amount)}
    if reference is not None:
        expected |= {month, year}
    for source, path in shapes:
        for profile in ("en-US", "legacy"):
            _assert_demanded(source, path, profile, expected)


@settings(max_examples=30, deadline=None)
@given(
    whole=st.integers(min_value=1, max_value=99),
    cents=st.integers(min_value=1, max_value=99),
    number_first=st.booleans(),
)
@example(whole=12, cents=25, number_first=True)
@example(whole=12, cents=25, number_first=False)
def test_generated_legacy_grouped_aliases_require_same_evidence_span(
    whole, cents, number_first
):
    """A separate integer cannot suppress an operative decimal's evidence."""
    integer = str(whole * 100 + cents)
    amount = f"{whole}.{cents:02d}"
    number = f"Printed reference: {integer}."
    money = f"The amount is ${amount}."
    source = (number + "\n" + money) if number_first else (money + "\n" + number)
    occurrences = extract_typed_numeric_inventory_occurrences_from_text(
        source, profile="legacy"
    )
    expected_span = (source.index(amount), source.index(amount) + len(amount))
    assert any(
        occurrence.value == float(amount) and occurrence.span == expected_span
        for occurrence in occurrences
    ), (source, [(occurrence.value, occurrence.span) for occurrence in occurrences])

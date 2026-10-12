"""Florida ESS eligibility heading safeguards and folded 100% rates.

The Florida ESS Program Policy Manual examples below reproduce passage
locator and folded-rate demands in the complete-source-unit gate. The
sentences are verbatim from the pinned corpus rows
`us-fl/manual/dcf/ess-program-policy-manual/2000-coverage-groups/page-5` and
`.../2600-calculation-of-benefits/page-8`:

* numeric recall retains the passage numbers 2010.0202 (an inline heading) and
  2610.0103 (`Refer to passage 2610.0103`) and the `1` of `Appendix A-1`;
* formula clause 11 of page 5 runs across the inline heading `2010.0202
  Noncategorically Eligible Assistance Groups (FS)`, so the heading number
  remains an operand the executed formula has to contain;
* simplification folds a 100% rate out of `rate * base`, so the source's
  `100% of` could not be grounded by the rate that executes it.
"""

from __future__ import annotations

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

PAGE_5 = "us-fl/manual/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
PAGE_8 = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
NAMESPACE = "us-fl:policies/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
INVENTORY = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)
GROUNDING = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)

# Page 5, characters 2437:2932 (formula clause 11 as the gate segments it).
CLAUSE_ACROSS_HEADING = (
    "If the SFU does not meet the 200% of the federal poverty income limit, the "
    "SFU must meet the net income limit of 100% of the federal poverty level for "
    "the AG size and the asset limit of $4,500. 2010.0202 Noncategorically "
    "Eligible Assistance Groups (FS) If the standard filing unit does not contain "
    "all individuals who receive or are authorized to receive Temporary Cash "
    "Assistance, SSI, or services through Healthy Families Florida, the "
    "assistance group is not considered categorically eligible."
)


def _recall_values(source: str, citation: str) -> set[float]:
    cleaned = sc.authoritative_numeric_recall_text(
        source, corpus_citation_path=citation
    )
    return {occurrence.value for occurrence in INVENTORY(cleaned)}


# --- numeric recall --------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        (CLAUSE_ACROSS_HEADING, PAGE_5, {2.0, 1.0, 4500.0, 2010.0202}),
        (
            "The assistance group is responsible for providing information "
            "regarding the individual's inability to purchase and prepare meals "
            "since this is the key determining factor. Refer to Appendix A-1 for "
            "the 165% Need Standard Table.",
            PAGE_8,
            {1.0, 1.65},
        ),
        (
            "Refer to passage 2610.0103 for assistance groups not tested against "
            "the gross income limit. The gross income limits can be found in the "
            "on-line reference table or in Appendix A-1.",
            PAGE_8,
            {1.0, 2610.0103},
        ),
        (
            "Income tests for determining food stamp eligibility are presented in "
            "passages 2610.0104.02 through 2610.0104.04.",
            PAGE_8,
            set(),
        ),
    ),
)
def test_florida_inline_and_reference_locator_inventory_is_retained(
    source, citation, expected
):
    assert _recall_values(source, citation) == expected


@pytest.mark.parametrize(
    ("source", "citation", "expected"),
    (
        # A four-by-four dotted number without a titled program-code heading
        # after it is a value.
        ("The rate is 1234.5678 dollars per unit.", PAGE_5, {1234.5678}),
        ("Pay 2010.0202 Noncategorically Eligible amounts.", PAGE_5, {2010.0202}),
        # Outside manuals the heading shape is not assumed.
        (
            "2010.0202 Noncategorically Eligible Assistance Groups (FS)",
            "us/guidance/usda/fns/snap-example",
            {2010.0202},
        ),
        ("Refer to passage 2610.0103 now.", "us/statute/7/2014", {2610.0103}),
        # Values beside a locator keep their value.
        ("Appendix A-1 lists the $4,500 asset limit.", PAGE_8, {1.0, 4500.0}),
        (
            "Refer to passage 2610.0103 for income of $2,221.",
            PAGE_8,
            {2610.0103, 2221.0},
        ),
        ("See passage 12 of the act.", PAGE_8, {12.0}),
        ("Appendix A-12 shows 3 rates.", PAGE_8, {12.0, 3.0}),
    ),
)
def test_values_beside_florida_locator_shapes_are_kept(source, citation, expected):
    assert _recall_values(source, citation) == expected


def test_transport_cost_heading_keeps_first_class_passage_fare():
    source = "Transport costs (USD):\nFirst-class passage 1234.5678."
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump({"format": "rulespec/v1", "rules": []}),
        source,
        corpus_citation_path=PAGE_5,
        test_cases=[],
        extract_numeric_occurrences=INVENTORY,
        extract_numeric_grounding_occurrences=GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert any(
        "[complete-source-unit:numeric-recall]" in issue
        and "has no named scalar representation" in issue
        for issue in result.issues
    ), result.issues
    assert _recall_values(source, PAGE_5) == {1234.5678}


def _empty_encoding_analysis(source: str):
    return sc.analyze_complete_source_unit(
        yaml.safe_dump({"format": "rulespec/v1", "rules": []}),
        source,
        corpus_citation_path=PAGE_5,
        test_cases=[],
        extract_numeric_occurrences=INVENTORY,
        extract_numeric_grounding_occurrences=GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _assert_empty_encoding_demands(source: str, value: float) -> None:
    # Use the same typed inventory and grounding callbacks as the public gate.
    # The unmasked inventory is the operative-value oracle; it also guards
    # against generated examples whose spelling never exposed a numeric value.
    assert value in {occurrence.value for occurrence in INVENTORY(source)}, source
    result = _empty_encoding_analysis(source)
    expected = (
        "[complete-source-unit:numeric-recall] Authoritative corpus numeric "
        f"value {value:g} has no named scalar representation. "
        "`module.summary` is not consulted."
    )
    assert expected in result.issues, (source, result.issues)
    assert value in _recall_values(source, PAGE_5), source


@pytest.mark.parametrize(
    ("source", "value"),
    (
        ("U.S.D.\nAppendix A-100.", 100.0),
        (
            "SEK.\nOther row 50.\nEconomy free.\n"
            "1234.5678 Eligible Assistance Groups (FS).",
            1234.5678,
        ),
        (
            "U.\nS.\nD.\nOther row 50.\nEconomy free.\n"
            "1234.5678 Eligible Assistance Groups (FS).",
            1234.5678,
        ),
        (
            "1234.5678 Eligible Assistance Groups (FS) If the SFU qualifies, "
            "pay this amount in CAD.",
            1234.5678,
        ),
        (
            "1234.5678 Eligible Assistance Groups (FS) If it applies, "
            "the fare is denominated in USD.",
            1234.5678,
        ),
    ),
)
def test_reviewed_dotted_currency_headers_and_continuations_keep_public_demands(
    source, value
):
    _assert_empty_encoding_demands(source, value)


@pytest.mark.parametrize(
    "source",
    (
        "The fare for adults (e.g. first-class passage 1234.5678).",
        "The fare in U.S. currency for first-class passage 1234.5678.",
        "Item | USD¹ | Quantity |\nFirst-class passage 1234.5678.",
        "Item | $¹\nEconomy: free.\nFirst-class passage 1234.5678.",
        "Fares (USD)¹\nEconomy: free.\nFirst-class passage 1234.5678.",
    ),
)
def test_reviewed_abbreviations_and_footnoted_headers_keep_public_fare_demand(source):
    _assert_empty_encoding_demands(source, 1234.5678)


@pytest.mark.parametrize(
    ("source", "value"),
    (
        ("First-class passage 1234.5678. USD¹", 1234.5678),
        ("First-class passage 1234.5678.\nUSD¹", 1234.5678),
        ("First-class passage 1234.5678. A.\nUSD¹", 1234.5678),
        ("Appendix A-100. Sec.\nRate†", 100.0),
        ("1234.5678 Eligible Assistance Groups (FS). A.\nUSD¹", 1234.5678),
        ("Item | USD¹\nReferences:\nFirst-class passage 1234.5678.", 1234.5678),
        ("Weight (kg):\nFirst-class passage 1234.5678.", 1234.5678),
        ("Number of tickets:\nFirst-class passage 1234.5678.", 1234.5678),
        ("First-class passage 1234.5678. BPS of income.", 1234.5678),
    ),
)
def test_reviewed_abbreviations_and_footnoted_headers_keep_wrapped_suffix_demand(
    source, value
):
    _assert_empty_encoding_demands(source, value)


def test_real_page_five_paragraph_keeps_inline_heading_and_public_operands():
    source = (
        "Broad-based categorically eligible SFUs must meet the 200% gross income "
        "limits. If the SFU contains a member who is age 60 or over or meets the "
        "definition of food stamp disabled, the SFU must meet the gross income "
        "limit of 200% of the federal poverty level for the AG size. "
        + CLAUSE_ACROSS_HEADING
    )
    assert _recall_values(source, PAGE_5) == {2.0, 60.0, 1.0, 4500.0, 2010.0202}
    result = _empty_encoding_analysis(source)
    assert result.source_numeric_occurrence_count == 5
    assert result.missing_source_numeric_occurrence_count == 5
    assert {
        issue
        for issue in result.issues
        if "[complete-source-unit:numeric-recall]" in issue
    } == {
        "[complete-source-unit:numeric-recall] Authoritative corpus numeric "
        f"value {value:g} has no named scalar representation. "
        "`module.summary` is not consulted."
        for value in (2.0, 60.0, 1.0, 4500.0, 2010.0202)
    }


# These spellings cover the existing locator unit veto and monetary-context
# recognizers, including currency codes preserved by the typed inventory.
# Parametrizing the vocabulary ensures every token runs; Hypothesis composes
# its context rather than relying on a few hand-picked document shapes.
_OPERATIVE_CONTEXT_TOKENS = (
    "$",
    "€",
    "£",
    "¥",
    "₹",
    "¢",
    "₵",
    "￠",
    "₦",
    "₪",
    "₩",
    "₽",
    "USD",
    "EUR",
    "GBP",
    "CAD",
    "AUD",
    "CHF",
    "FRW",
    "RWF",
    "GHS",
    "GHP",
    "NGN",
    "UGX",
    "ZMW",
    "ETB",
    "KES",
    "TZS",
    "XAF",
    "XOF",
    "ZAR",
    "ILS",
    "NIS",
    "JPY",
    "dollar",
    "dollars",
    "cent",
    "cents",
    "euro",
    "euros",
    "pound",
    "pounds",
    "franc",
    "francs",
    "Swiss francs",
    "U.S. dollars",
    "United States dollars",
    "Canadian dollars",
    "Australian dollars",
    "yen",
    "yuan",
    "renminbi",
    "peso",
    "pesos",
    "rupee",
    "rupees",
    "ruble",
    "rubles",
    "rouble",
    "roubles",
    "won",
    "kr",
    "kroner",
    "øre",
    "שקלים חדשים",
    "שקלים",
    "שקל",
    'ש"ח',
    "דולר",
    "דולרים",
    "יורו",
    "אירו",
    'ליש"ט',
    "לירות",
    "לירה",
    "אגורות",
    "אגורה",
    "fee",
    "fees",
    "fare",
    "fares",
    "payment",
    "payments",
    "amount",
    "amounts",
    "rate",
    "rates",
    "price",
    "prices",
    "cost",
    "costs",
    "charge",
    "charges",
    "allowance",
    "allowances",
    "percent",
    "percentage",
    "percentages",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "year",
    "years",
    "hour",
    "hours",
    "member",
    "members",
    "person",
    "persons",
    "people",
    "quantity",
    "quantities",
    "unit",
    "units",
    "child",
    "children",
    "kg",
    "kilogram",
    "kilograms",
    "weight",
    "count",
    "number",
    "tickets",
)
_CONTEXT_ABBREVIATIONS = (
    "e.g.",
    "i.e.",
    "U.S.",
    "etc.",
    "No.",
    "Sec.",
    "St.",
    "a.m.",
    "p.m.",
    "A.",
    "Z.",
    "A.B.",
    "Q.R.S.",
    "approx.",
    "Assn.",
)
_CONTEXT_FOOTNOTES = (
    "",
    "¹",
    "²",
    "⁽¹⁾",
    "*",
    "**",
    "†",
    "‡",
    "[a]",
    "(a)",
    "[1]",
    "(1)",
    ".",
    ":",
    ",",
    ";",
    "!",
    "?",
    "_",
)


@st.composite
def _adversarial_operative_documents(draw, context):
    major = draw(st.integers(1000, 9999))
    minor = draw(st.integers(1, 9999))
    appendix = draw(st.integers(100, 999))
    number = f"{major}.{minor:04d}"
    abbreviation = draw(st.sampled_from(_CONTEXT_ABBREVIATIONS))
    marker = draw(st.sampled_from(_CONTEXT_FOOTNOTES))
    separator = draw(st.sampled_from((" ", "\n", "\r\n", "\n\n")))
    row_count = draw(st.integers(0, 8))
    row = draw(
        st.sampled_from(
            (
                "Economy: free.\n",
                "Other row 50.\n",
                "Economy:\nfree.\n",
                "| Economy | free | adults |\n",
                "References:\n",
                "Citations:\n",
            )
        )
    )
    layout = draw(
        st.sampled_from(
            (
                "inline-prefix",
                "wrapped-prefix",
                "plain-heading",
                "wrapped-heading",
                "table-header",
                "parenthesized-header",
                "table-column",
                "flattened-heading",
                "flattened-table-row",
                "operative-suffix",
            )
        )
    )
    token = context.upper() if draw(st.booleans()) else context
    annotated = f"{token}{marker}"
    if layout == "inline-prefix":
        prefix = f"The {annotated} for adults ({abbreviation}{separator}"
        suffix = ")."
    elif layout == "wrapped-prefix":
        prefix = f"The {annotated} for\n{abbreviation}{separator}"
        suffix = "."
    elif layout == "plain-heading":
        prefix = f"{annotated}{separator}"
        suffix = "."
    elif layout == "wrapped-heading":
        prefix = f"Transport\n{annotated}:{separator}"
        suffix = "."
    elif layout == "parenthesized-header":
        prefix = f"Transport ({token}){marker}{separator}"
        suffix = "."
    elif layout == "table-column":
        prefix = f"| Item | {annotated} | Category |\n"
        suffix = " | adults |"
    elif layout == "flattened-heading":
        prefix = f"{annotated}: Economy: free. " + row.replace("\n", " ") * row_count
        suffix = "."
    elif layout == "flattened-table-row":
        prefix = f"Item | {annotated} | Category | Economy: free. Other row 50. "
        suffix = "."
    elif layout == "operative-suffix":
        prefix = ""
        suffix = f". {abbreviation}{separator}{annotated}"
    else:
        prefix = f"Item | {annotated}\n"
        suffix = "."
    if abbreviation in {"Sec.", "No."} and layout in {
        "inline-prefix",
        "wrapped-prefix",
    }:
        # A dotted value immediately after Sec. is already an authenticated
        # statutory citation on the baseline. Generate an operative row, while
        # still exercising the abbreviation inside its monetary sentence.
        prefix += "listed item "
    if layout not in {
        "inline-prefix",
        "wrapped-prefix",
        "flattened-heading",
        "flattened-table-row",
        "operative-suffix",
    }:
        prefix += row * row_count
    return tuple(
        (prefix + body + suffix, value)
        for body, value in (
            (f"First-class passage {number}", float(number)),
            (f"Appendix A-{appendix}", float(appendix)),
            (f"{number} Eligible Assistance Groups (FS)", float(number)),
        )
    )


@pytest.mark.parametrize("context", _OPERATIVE_CONTEXT_TOKENS)
@settings(max_examples=10, deadline=None, database=None)
@given(data=st.data())
def test_adversarial_operative_context_property_keeps_public_demands(context, data):
    for source, value in data.draw(_adversarial_operative_documents(context)):
        _assert_empty_encoding_demands(source, value)


@settings(max_examples=30, deadline=None, database=None)
@given(symbol=st.characters(categories=("Sc",)), data=st.data())
def test_any_unicode_currency_symbol_keeps_public_operative_demands(symbol, data):
    for source, value in data.draw(_adversarial_operative_documents(symbol)):
        _assert_empty_encoding_demands(source, value)


@settings(max_examples=80, deadline=None, database=None)
@given(
    context=st.sampled_from(_OPERATIVE_CONTEXT_TOKENS),
    cue=st.sampled_from(("See passage", "Refer to passage", "presented in passages")),
    data=st.data(),
)
def test_reference_cues_cannot_override_generated_public_value_context(
    context, cue, data
):
    source, value = data.draw(_adversarial_operative_documents(context))[0]
    source = source.replace("First-class passage", cue)
    # A genuine References section intentionally ends the prior table in the
    # positive controls. Here the intervening labels are still operative rows.
    source = source.replace("References:", "Reference row:").replace(
        "Citations:", "Citation row:"
    )
    _assert_empty_encoding_demands(source, value)


_MONETARY_HEADINGS = (
    "Transport costs (USD)",
    "First-class fares",
    "Ticket prices (EUR)",
    "Service charges",
    "Fees",
    "Payment amounts",
    "Allowances (GBP)",
    "Amounts in dollars",
    "Item | USD",
    "Item | $",
)


@pytest.mark.parametrize("heading", _MONETARY_HEADINGS)
def test_each_monetary_heading_keeps_a_passage_fare(heading):
    assert _recall_values(f"{heading}:\nFirst-class passage 1234.5678.", PAGE_5) == {
        1234.5678
    }


@pytest.mark.parametrize(
    "source",
    (
        "Transport costs (USD)\nFirst-class passage 1234.5678.",
        "Item | USD\nFirst-class passage 1234.5678.",
        "Item | $\nFirst-class passage 1234.5678.",
        "| Item | USD |\nFirst-class passage 1234.5678.",
        "| Item | USD | Quantity |\nFirst-class passage 1234.5678.",
        "Transport costs (USD):\nEconomy: 50.\nFirst-class passage 1234.5678.",
        "Transport costs (USD):\nEconomy 50 (adult)\nFirst-class passage 1234.5678.",
        "Transport costs (USD):\nEconomy: free.\nFirst-class passage 1234.5678.",
        "Transport costs (USD):\nOther fare 50.\nEconomy (free)\n"
        "First-class passage 1234.5678.",
        "Transport costs (USD):\nEconomy:\nfree.\nFirst-class passage 1234.5678.",
        "The fare for passage 1234.5678.",
        "The fare for\npassage 1234.5678.",
        "The cost of\npassage 1234.5678.",
        "The cost of passage 1234.5678.",
        "Pay $ passage 1234.5678.",
        "Transport costs (USD):\n"
        + "Other row 50.\n" * 100
        + "First-class passage 1234.5678.",
    ),
)
def test_preceding_currency_and_cost_context_keeps_the_fare(source):
    assert 1234.5678 in _recall_values(source, PAGE_5)


@pytest.mark.parametrize("row", ("", "Other fare 50.\n", "Economy 50 (adult)\n"))
@pytest.mark.parametrize("heading", ("Policy references", "References", "Citations"))
def test_reference_heading_keeps_unmasked_passage_and_table_values(row, heading):
    source = f"Transport costs (USD):\n{row}{heading}:\nSee passage 2610.0103."
    assert _recall_values(source, PAGE_5) == ({50, 2610.0103} if row else {2610.0103})


@settings(max_examples=100, deadline=None)
@given(
    heading=st.sampled_from(_MONETARY_HEADINGS),
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    appendix=st.integers(1, 999),
    separator=st.sampled_from((" ", "\n", "\r\n", "\n\n")),
    preceding_rows=st.integers(0, 5),
    row=st.sampled_from(
        (
            "Other fare 50.\n",
            "Economy: 50.\n",
            "Economy 50 (adult)\n",
            "Economy: free.\n",
            "Economy (free)\n",
            "Economy:\nfree.\n",
        )
    ),
)
def test_monetary_heading_property_keeps_locator_shaped_values(
    heading, major, minor, appendix, separator, preceding_rows, row
):
    number = f"{major}.{minor:04d}"
    prefix = f"{heading}:{separator}" + row * preceding_rows
    for row, value in (
        (f"First-class passage {number}.", float(number)),
        (f"Appendix A-{appendix}.", appendix),
        (f"{number} Eligible Assistance Groups (FS).", float(number)),
    ):
        source = prefix + row
        assert (
            sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5)
            == ()
        )
        assert value in _recall_values(source, PAGE_5)


@pytest.mark.parametrize(
    "source",
    (
        "The maximum payment is 1234.5678 dollars for food stamps (FS).",
        "The payment equals 1234.5678 percent of income for food stamps (FS).",
        "The payment is １２３４.５６７８ dollars for Food Stamps (FS).",
        "Pay 1234.5678 Food Stamps (FS).",
        "1234.5678 Dollars for Food Stamps (FS)",
        "1234.5678 Percent of Income (FS)",
        "1234.5678 Benefit Limit in USD (FS)",
        "1234.5678 Benefit Limit in EUR (FS)",
        "1234.5678 Benefit Limit (FS) dollars per month",
        "1234.5678 Benefit Limit (FS) percent of income",
        "1234.5678 Benefit Limit (FS) USD per month",
        "1234.5678 Income Is Allowed (FS)",
        "1234.5678 Income Equals Benefits (FS)",
        "1234.5678 Income eat Benefits (FS)",
        "passage 1234.5678 dollars of income",
        "passage 1234.5678% of income",
        "passage 1234.5678 for the effective date",
        "Appendix A-50% of income is allowed.",
        "Appendix A-50 dollars of income is allowed.",
    ),
)
def test_operative_contexts_are_never_passage_or_appendix_labels(source):
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )


@pytest.mark.parametrize(
    ("source", "operative"),
    (
        ("See passage 2610.0103 and 1234.5678 dollars of income.", 1234.5678),
        ("See passage 2610.0103 through 1234.5678 percent of income.", 12.345678),
        ("See passage 2610.0103 to 1234.5678% of income.", 12.345678),
        ("See passage 2610.0103 and 1234.5678 - 50 dollars.", 1234.5678),
        ("Refer to passage 2610.0103 to 2026.1011 for the effective date.", 2026.1011),
        ("Appendix A-1-50% of income is allowed.", 0.5),
        ("Appendix A-1+50% of income is allowed.", 50.0),
        ("Appendix A-1-50 dollars of income is allowed.", 50.0),
        ("Appendix A-1*50 dollars of income is allowed.", 50.0),
    ),
)
def test_locator_continuations_keep_operative_amounts_and_rates(source, operative):
    assert operative in _recall_values(source, PAGE_8)


@pytest.mark.parametrize(
    "source",
    (
        "See passage 2610.0103 and 1234.5678 as the maximum payment.",
        "See passage 2610.0103 and 1234.5678 of income.",
        "See passage 1234.5678 as the maximum payment.",
        "See passage 1234.5678 of income.",
        "Before. 1234.5678 Earns Interest (FS).",
        "Before. 1234.5678 Income Generates Benefits (FS).",
        "Before. 1234.5678 Eligible Assistance Groups (USD).",
        "Before. 1234.5678 Eligible Assistance Groups (EUR).",
        "Before. 1234.5678 Eligible Assistance Groups (FS, USD).",
        "Before. 1234.5678 Eligible Assistance Groups (FS)\ndollars per month.",
        "passage 1234.5678\ndollars of income",
    ),
)
def test_independently_reviewed_operative_shapes_keep_their_demand(source):
    assert 1234.5678 in _recall_values(source, PAGE_5)


@pytest.mark.parametrize(
    ("source", "operative"),
    (
        ("See passage 1234.5678 U.S. dollars.", 1234.5678),
        ("Appendix A-50 U.S. dollars is allowed.", 50),
        ("See passage 1234.5678 per child.", 1234.5678),
        ("Appendix A-50 per child is allowed.", 50),
        ("See passage 1234.5678 is the maximum payment.", 1234.5678),
        ("See passage 1234.5678 maximum payment.", 1234.5678),
        ("Appendix A-50 is the maximum payment.", 50),
        ("Appendix A-50 maximum payment.", 50),
        ("See passage 1234.5678 yen.", 1234.5678),
        ("Appendix A-50 yen is allowed.", 50),
        ("See passage 1234.5678 pesos.", 1234.5678),
        ("Appendix A-50 pesos is allowed.", 50),
        ("See passage 1234.5678 (dollars per month).", 1234.5678),
        ("Appendix A-50 (U.S. dollars)", 50),
        ("See passage 1234.5678 ($50 per month).", 1234.5678),
        ("See passage 1234.5678 ^ 50 dollars.", 1234.5678),
        ("See passage 1234.5678 ± 50 dollars.", 1234.5678),
        ("Appendix A-1^50% of income is allowed.", 1),
        ("Appendix A-1±50% of income is allowed.", 1),
    ),
)
def test_explicit_operative_suffixes_keep_their_numeric_demand(source, operative):
    assert operative in _recall_values(source, PAGE_5)


@settings(max_examples=100, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    appendix=st.integers(1, 999),
    suffix=st.sampled_from(
        (
            "U.S. dollars",
            "per child",
            "is the maximum payment",
            "maximum payment",
            "yen",
            "pesos",
            "(dollars per month)",
            "(U.S. dollars)",
            "($50 per month)",
        )
    ),
)
def test_operative_suffix_property_keeps_passage_and_appendix_values(
    major, minor, appendix, suffix
):
    number = f"{major}.{minor:04d}"
    assert float(number) in _recall_values(f"See passage {number} {suffix}.", PAGE_5)
    assert appendix in _recall_values(f"Appendix A-{appendix} {suffix}.", PAGE_5)


_UNAUTHENTICATED_LOCATOR_TAILS = (
    "basis points of income",
    "bps of income",
    "shekels",
    "pennies",
    "children",
    "units",
    "credits",
    "unknownmeasure",
)


@pytest.mark.parametrize("suffix", _UNAUTHENTICATED_LOCATOR_TAILS)
@pytest.mark.parametrize("shape", ("passage", "appendix"))
def test_unrecognized_value_suffixes_cannot_authenticate_a_locator(shape, suffix):
    if shape == "passage":
        source, value = f"See passage 1234.5678 {suffix}.", 1234.5678
    else:
        source, value = f"Appendix A-50 {suffix}.", 50
    assert value in _recall_values(source, PAGE_5)


@pytest.mark.parametrize(
    ("source", "operative"),
    (
        ("See passage 1234.5678\nbasis points of income.", 1234.5678),
        ("Appendix A-50\nbasis points of income.", 50),
        ("See passage 1234.5678, paid per child.", 1234.5678),
        ("See passage 1234.5678: paid per child.", 1234.5678),
        ("Appendix A-50, paid per child.", 50),
        ("Appendix A-50: paid per child.", 50),
        (
            "Before. 1234.5678 Eligible Assistance Groups (FS) for each child.",
            1234.5678,
        ),
        (
            "Before. 1234.5678 Eligible Assistance Groups (FS) basis points of income.",
            1234.5678,
        ),
    ),
)
def test_structural_boundary_cannot_hide_an_operative_tail(source, operative):
    assert operative in _recall_values(source, PAGE_5)


@pytest.mark.parametrize(
    "suffix",
    (
        "for assistance groups per child",
        "for more information per child",
        "for the rules per child",
    ),
)
def test_reference_phrase_prefix_cannot_authenticate_operative_tail(suffix):
    assert 1234.5678 in _recall_values(f"See passage 1234.5678 {suffix}.", PAGE_5)


@pytest.mark.parametrize(
    ("source", "operative"),
    (
        ("See passage 1234.5678.basis points of income.", 1234.5678),
        ("See passage 1234.5678.BPS of income.", 1234.5678),
        ("Appendix A-50.basis points of income.", 50),
        ("Appendix A-50.per child.", 50),
    ),
)
def test_glued_punctuation_cannot_authenticate_operative_tail(source, operative):
    assert operative in _recall_values(source, PAGE_5)


@settings(max_examples=100, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    appendix=st.integers(1, 999),
    punctuation=st.sampled_from((".", ";", "!", "?")),
    suffix=st.one_of(
        st.sampled_from(_UNAUTHENTICATED_LOCATOR_TAILS),
        st.from_regex(r"unknown[a-z]{1,12}", fullmatch=True),
    ),
)
def test_glued_punctuation_property_retains_numeric_demand(
    major, minor, appendix, punctuation, suffix
):
    number = f"{major}.{minor:04d}"
    for source in (
        f"See passage {number}{punctuation}{suffix}.",
        f"Appendix A-{appendix}{punctuation}{suffix}.",
    ):
        assert (
            sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5)
            == ()
        )


@settings(max_examples=100, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    appendix=st.integers(1, 999),
    suffix=st.one_of(
        st.sampled_from(_UNAUTHENTICATED_LOCATOR_TAILS),
        st.from_regex(r"unknown[a-z]{1,12}", fullmatch=True),
    ),
)
def test_unknown_suffix_property_retains_numeric_demand(major, minor, appendix, suffix):
    number = f"{major}.{minor:04d}"
    assert float(number) in _recall_values(f"See passage {number} {suffix}.", PAGE_5)
    assert appendix in _recall_values(f"Appendix A-{appendix} {suffix}.", PAGE_5)
    assert float(number) in _recall_values(
        f"Before. {number} Eligible Assistance Groups (FS) {suffix}.", PAGE_5
    )


def test_line_wrapped_appendix_currency_value_keeps_its_demand():
    assert 50 in _recall_values("Appendix A-50\nUSD is paid", PAGE_5)


def test_separate_amount_line_vetoes_eligibility_heading_mask():
    source = "2010.0202 Eligible Assistance Groups (FS)\n$4500"
    assert _recall_values(source, PAGE_5) == {2010.0202, 4500}
    _assert_empty_encoding_demands(source, 2010.0202)
    _assert_empty_encoding_demands(source, 4500)


@pytest.mark.parametrize("operator", ("×", "÷", "−", "–", "—", "·", "•", "∗", "∙"))
def test_unicode_arithmetic_context_keeps_passage_value(operator):
    source = f"See passage 1234.5678 {operator} 50 dollars."
    assert 1234.5678 in _recall_values(source, PAGE_5)


@pytest.mark.parametrize("operator", ("×", "÷", "−", "–", "—", "·", "•", "∗", "∙"))
def test_unicode_arithmetic_context_keeps_appendix_numeric_inventory(operator):
    source = f"Appendix A-1{operator}50% of income is allowed."
    assert _recall_values(source, PAGE_5) == {
        occurrence.value for occurrence in INVENTORY(source)
    }


def test_attached_currency_suffix_cannot_authenticate_a_passage_number():
    source = "passage 1234.5678USD"
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )
    assert _recall_values(source, PAGE_5) == {
        occurrence.value for occurrence in INVENTORY(source)
    }


def test_passage_reference_range_is_unmasked():
    source = "See passages 2610.0104.02 through 2610.0104.04."
    spans = sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_8)
    assert spans == ()
    assert _recall_values(source, PAGE_8) == {
        occurrence.value for occurrence in INVENTORY(source)
    }


@settings(max_examples=100, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    unit=st.sampled_from(("dollars", "percent", "%")),
    prefix=st.sampled_from(("The payment is ", "See passage 2610.0103 and ")),
)
def test_locator_shapes_never_hide_an_operative_number(major, minor, unit, prefix):
    number = f"{major}.{minor:04d}"
    source = f"{prefix}{number} {unit} for Food Stamps (FS)."
    spans = sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5)
    start = source.index(number, len(prefix))
    assert all(end <= start or begin >= start + len(number) for begin, end in spans)


_PROGRAM = st.sampled_from(("FS", "TCA", "MA", "SFP", "RAP"))
_ELIGIBILITY_TITLES = st.sampled_from(
    (
        "Eligible Assistance Groups",
        "Categorically Eligible Assistance Groups",
        "Noncategorically Eligible Assistance Groups",
    )
)


_NEUTRAL_SURROUNDINGS = (
    "This paragraph explains eligibility.",
    "The policy explains who qualifies.",
    "The following prose describes assistance.",
)


@settings(max_examples=200, deadline=None, database=None)
@given(
    preceding=st.sampled_from(_NEUTRAL_SURROUNDINGS),
    following=st.sampled_from(_NEUTRAL_SURROUNDINGS),
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    title=_ELIGIBILITY_TITLES,
    program=_PROGRAM,
    newline=st.sampled_from(("\n", "\r\n")),
)
def test_heading_spans_cover_exactly_the_passage_number(
    preceding, following, major, minor, title, program, newline
):
    number = f"{major}.{minor:04d}"
    text = newline.join(
        (preceding, f"{number} {title} ({program})", following, "Pay $4500.")
    )
    spans = sc._us_manual_passage_locator_spans(text, corpus_citation_path=PAGE_5)
    start = text.index(number)
    assert spans == ((start, start + len(number)),)
    masked = sc._mask_numeric_spans(text, spans)
    # Masking keeps the length and changes nothing outside the heading number.
    assert len(masked) == len(text)
    assert masked[:start] == text[:start]
    assert masked[start + len(number) :] == text[start + len(number) :]
    assert _recall_values(text, PAGE_5) == {4500.0}
    result = _empty_encoding_analysis(text)
    assert result.source_numeric_occurrence_count == 1
    assert result.missing_source_numeric_occurrence_count == 1
    assert {
        issue
        for issue in result.issues
        if "[complete-source-unit:numeric-recall]" in issue
    } == {
        "[complete-source-unit:numeric-recall] Authoritative corpus numeric "
        "value 4500 has no named scalar representation. "
        "`module.summary` is not consulted."
    }


_POTENTIAL_VALUE_HEADERS = (
    "U.S.D.",
    "SEK.",
    "U.\nS.\nD.",
    "USD¹",
    "USD†",
    "XQZ‡",
    "X.Q.Z.",
    "unknowncode",
    "abcdefghijkl",
    "a b c d e f g h i j k l",
    "UNKNOWN CURRENCY HEADER",
    "X.Q.Z. U.N.I.T. H.E.A.D.E.R.",
    "Arbitrary column label",
    "Unrecognized table heading:",
    "| Mystery column |",
    "$¹",
    "₿†",
    "These fares are denominated in USD.",
    "This amount is a payment.",
    "This rate is expressed as a percentage.",
    "The unit for this row is kg.",
    "The allowance is counted per child.",
    "The quantity is listed in unknownmeasure.",
    "Other row 50.",
    "Economy free.",
    "BPS of income.",
    "fee.",
    "cost.",
    "day.",
    "rate.",
    "usd.",
    "The amt. billed is shown below.",
    "The qty. supplied is shown below.",
    "The following entries use sek.",
    "All figures are stated in quetzales.",
)


@pytest.mark.parametrize("context", _POTENTIAL_VALUE_HEADERS)
@pytest.mark.parametrize("position", ("before", "after"))
def test_exact_heading_context_veto_keeps_all_public_demands(context, position):
    heading = "1234.5678 Eligible Assistance Groups (FS)"
    lines = (context, heading) if position == "before" else (heading, context)
    source = "\n\n".join(lines)
    for value in {occurrence.value for occurrence in INVENTORY(source)}:
        _assert_empty_encoding_demands(source, value)
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )


@settings(max_examples=60, deadline=None, database=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(1, 9999),
    title=_ELIGIBILITY_TITLES,
    program=_PROGRAM,
    context=st.one_of(
        st.sampled_from(_OPERATIVE_CONTEXT_TOKENS + _POTENTIAL_VALUE_HEADERS),
        # Unknown short tokens remain ambiguous regardless of alphabet, case,
        # punctuation, or spacing between their non-space characters.
        st.text(
            alphabet=st.characters(
                blacklist_categories=("Cs", "Cc", "Cf", "Zs", "Zl", "Zp")
            ),
            min_size=1,
            max_size=12,
        ).flatmap(lambda token: st.sampled_from((token, " ".join(token)))),
        st.from_regex(r"[A-Z]{1,12}[¹†‡]?", fullmatch=True),
        st.from_regex(r"(?:[A-Za-z]\.){1,12}[¹†‡]?", fullmatch=True),
        st.from_regex(r"(?:[A-Za-z]\.\n){1,12}[A-Za-z]\.", fullmatch=True),
        st.characters(categories=("Sc",)),
    ),
    marker=st.sampled_from(_CONTEXT_FOOTNOTES),
    separator=st.sampled_from(("\n", "\r\n", "\n\n")),
)
def test_generated_exact_heading_contexts_keep_public_numeric_inventory(
    major, minor, title, program, context, marker, separator
):
    number = f"{major}.{minor:04d}"
    heading = f"{number} {title} ({program})"
    # Both sides of the heading must independently veto; blank lines and
    # annotations cannot detach a header from the candidate value.
    for lines in ((context + marker, heading), (heading, context + marker)):
        source = separator.join(lines)
        _assert_empty_encoding_demands(source, float(number))
        assert (
            sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5)
            == ()
        )


@settings(max_examples=60, deadline=None, database=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(1, 9999),
    prefix=st.sampled_from(("Before. ", "fee. ", "usd. ", " ", "\t")),
    suffix=st.sampled_from(
        (
            ".",
            " ",
            "\t",
            " If the SFU qualifies, pay this amount in CAD.",
            " If it applies, the fare is denominated in USD.",
        )
    ),
)
def test_generated_heading_prefixes_and_continuations_keep_public_demands(
    major, minor, prefix, suffix
):
    number = f"{major}.{minor:04d}"
    heading = f"{number} Eligible Assistance Groups (FS)"
    for source in (prefix + heading, heading + suffix):
        assert (
            sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5)
            == ()
        )
        _assert_empty_encoding_demands(source, float(number))


@pytest.mark.parametrize(
    "source",
    (
        "Before. 1234.5678 Eligible Assistance Groups (FS)",
        "1234.5678 Eligible Assistance Groups (FS).",
        "1234.5678 Eligible Assistance Groups (FS) ",
        " 1234.5678 Eligible Assistance Groups (FS)",
        "1234.5678 Eligible Assistance Groups (FS) If it applies.",
        "1234.5678 Eligible Assistance Groups (FS) If the SFU qualifies.",
        "1234.5678 Eligible Assistance Groups (FS, TCA)",
        "1234.5678 Eligible Assistance Groups (FS/TCA)",
        "1234.5678 Eligible Assistance Groups (USD)",
        "1234.5678 Other Assistance Groups (FS).",
    ),
)
def test_only_exact_whole_line_eligibility_heading_can_mask_number(source):
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )
    _assert_empty_encoding_demands(source, 1234.5678)


def test_other_titled_heading_keeps_existing_generic_heading_behavior():
    source = "1234.5678 Other Assistance Groups (FS)"
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )
    # The existing generic title-case heading cleaner already removes this
    # number on main; the new eligibility-only mask must stay inert here.
    assert _recall_values(source, PAGE_5) == set()


def test_three_segment_heading_keeps_unmasked_public_inventory():
    source = "2010.0202.01 Eligible Assistance Groups (FS)"
    assert (
        sc._us_manual_passage_locator_spans(source, corpus_citation_path=PAGE_5) == ()
    )
    expected = {occurrence.value for occurrence in INVENTORY(source)}
    assert _recall_values(source, PAGE_5) == expected
    result = _empty_encoding_analysis(source)
    assert result.source_numeric_occurrence_count == len(expected)
    for value in expected:
        _assert_empty_encoding_demands(source, value)


@settings(max_examples=200, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    tail=st.from_regex(r"[a-z][a-z ,]{0,60}", fullmatch=True),
)
def test_untitled_four_by_four_numbers_never_mask(major, minor, tail):
    text = f"The limit is {major}.{minor:04d} {tail}"
    assert sc._us_manual_passage_locator_spans(text, corpus_citation_path=PAGE_5) == ()


@settings(max_examples=100, deadline=None)
@given(
    major=st.integers(1000, 9999),
    minor=st.integers(0, 9999),
    appendix=st.integers(1, 999),
)
def test_locator_spans_are_inert_outside_us_manuals(major, minor, appendix):
    text = f"{major}.{minor:04d} Eligible Assistance Groups (FS)"
    # The exact heading shape exercises the sole mask on a manual citation.
    assert (
        len(sc._us_manual_passage_locator_spans(text, corpus_citation_path=PAGE_5)) == 1
    )
    references = f"See passage 2610.0103 and Appendix A-{appendix}."
    assert (
        sc._us_manual_passage_locator_spans(references, corpus_citation_path=PAGE_5)
        == ()
    )
    for citation in (
        "us/statute/7/2014",
        "us/regulation/7/273",
        "us/guidance/usda/example",
        "us/guidance/usda/manual/example",
        "us/statute/title/manual/example",
        "us-fl/guidance/dcf/manual/example",
        "de/manual/example",
        "ca/manual/example",
        "",
    ):
        assert (
            sc._us_manual_passage_locator_spans(text, corpus_citation_path=citation)
            == ()
        )


# --- formula clauses -------------------------------------------------------


def _formula_payload(rate: str, excerpt: str, formula: str) -> str:
    def atom(kind: str, text: str) -> dict:
        return {
            "kind": kind,
            "path": "versions[0].formula",
            "source": {"corpus_citation_path": PAGE_5, "excerpt": text},
        }

    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {"source_verification": {"corpus_citation_path": PAGE_5}},
            "inputs": [
                {
                    "name": name,
                    "entity": "Household",
                    "dtype": "Money",
                    "unit": "USD",
                    "period": "Month",
                }
                for name in (
                    "federal_poverty_level",
                    "sfu_net_income",
                    "sfu_countable_assets",
                )
            ],
            "rules": [
                {
                    "name": "net_income_ratio",
                    "kind": "parameter",
                    "dtype": "Rate",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                atom("amount", f"{rate} of the federal poverty level")
                            ]
                        }
                    },
                    "versions": [
                        {
                            "effective_from": "0001-01-01",
                            "formula": str(int(rate[:-1]) / 100),
                        }
                    ],
                },
                {
                    "name": "asset_limit",
                    "kind": "parameter",
                    "dtype": "Money",
                    "unit": "USD",
                    "metadata": {
                        "proof": {"atoms": [atom("amount", "asset limit of $4,500")]}
                    },
                    "versions": [{"effective_from": "0001-01-01", "formula": "4500"}],
                },
                {
                    "name": "meets_limits",
                    "kind": "derived",
                    "entity": "Household",
                    "dtype": "Judgment",
                    "period": "Month",
                    "metadata": {"proof": {"atoms": [atom("formula", excerpt)]}},
                    "versions": [{"effective_from": "0001-01-01", "formula": formula}],
                },
            ],
        },
        sort_keys=False,
    )


def _cases(limit_holds_at: float) -> list[dict]:
    return [
        {
            "name": "at_the_limits",
            "period": "2026-05",
            "input": {
                "federal_poverty_level": 1000,
                "sfu_net_income": limit_holds_at,
                "sfu_countable_assets": 4500,
            },
            "output": {f"{NAMESPACE}#meets_limits": "holds"},
        },
        {
            "name": "above_the_net_limit",
            "period": "2026-05",
            "input": {
                "federal_poverty_level": 1000,
                "sfu_net_income": limit_holds_at + 1,
                "sfu_countable_assets": 4500,
            },
            "output": {f"{NAMESPACE}#meets_limits": "not_holds"},
        },
    ]


def _formula_branch_issues(
    source: str, rate: str, excerpt: str, formula: str, at: float
):
    result = sc.analyze_complete_source_unit(
        _formula_payload(rate, excerpt, formula),
        source,
        corpus_citation_path=PAGE_5,
        test_cases=_cases(at),
        extract_numeric_occurrences=INVENTORY,
        extract_numeric_grounding_occurrences=GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [
        issue
        for issue in result.issues
        if "Companion tests do not demonstrate formula branch" in issue
    ]


_EXCERPT_130 = (
    "the SFU must meet the net income limit of 130% of the federal poverty level "
    "for the AG size and the asset limit of $4,500"
)
_FORMULA = (
    "sfu_net_income <= net_income_ratio * federal_poverty_level "
    "and sfu_countable_assets <= asset_limit"
)


def test_inline_passage_heading_inside_a_formula_clause_remains_an_operand():
    # The 130% rate is not an identity factor, so only the heading number
    # inside the clause can keep this branch undemonstrated.
    source = CLAUSE_ACROSS_HEADING.replace(
        "If the SFU does not meet the 200% of the federal poverty income limit, ",
        "If the SFU does not meet the gross income limit, ",
    ).replace("100%", "130%")
    assert _formula_branch_issues(source, "130%", _EXCERPT_130, _FORMULA, 1300)


def test_without_a_heading_the_same_clause_was_already_demonstrated():
    source = (
        "If the SFU does not meet the gross income limit, the SFU must meet the "
        "net income limit of 130% of the federal poverty level for the AG size "
        "and the asset limit of $4,500."
    )
    assert _formula_branch_issues(source, "130%", _EXCERPT_130, _FORMULA, 1300) == []


def test_a_folded_100_percent_rate_still_grounds_its_source_operand():
    source = (
        "If the SFU does not meet the gross income limit, the SFU must meet the "
        "net income limit of 100% of the federal poverty level for the AG size "
        "and the asset limit of $4,500."
    )
    excerpt = _EXCERPT_130.replace("130%", "100%")
    assert _formula_branch_issues(source, "100%", excerpt, _FORMULA, 1000) == []

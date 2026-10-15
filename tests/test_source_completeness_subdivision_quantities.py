"""A locator mask must retain every occurrence of an operative quantity."""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

import pytest
from hypothesis import HealthCheck, Phase, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)

VA_CITATION = "us-va/statute/58.1/58.1-339.8"
EMPTY_RULESPEC = "format: rulespec/v1\nmodule: {}\nrules: []\n"
EXTRACT_NUMBERS = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)


def _keyword_case(text: str, style: str) -> str:
    """Vary every generated keyword without moving numeric occurrences."""

    def transform(match: re.Match[str]) -> str:
        word = match.group()
        if style == "upper":
            return word.upper()
        if style == "title":
            return word.title()
        if style == "mixed":
            return "".join(
                letter.lower() if index % 2 == 0 else letter.upper()
                for index, letter in enumerate(word)
            )
        return word

    return re.sub(r"[A-Za-z]+", transform, text)


@dataclass(frozen=True)
class QuantitySource:
    text: str
    locator_spans: tuple[tuple[int, int], ...]

    def expected_recall(self) -> str:
        characters = list(self.text)
        for start, end in self.locator_spans:
            characters[start:end] = " " * (end - start)
        return "".join(characters)


def _assert_quantity_source(case: QuantitySource) -> None:
    cleaned = completeness.authoritative_numeric_recall_text(
        case.text, corpus_citation_path=VA_CITATION
    )
    expected = case.expected_recall()
    # This checks both exact operative tokens and every surrounding character.
    # Value sets alone cannot catch a lost occurrence equal to a locator value.
    assert cleaned == expected
    expected_occurrences = EXTRACT_NUMBERS(expected)
    assert [
        (item.value, item.start, item.end, item.raw)
        for item in EXTRACT_NUMBERS(cleaned)
    ] == [(item.value, item.start, item.end, item.raw) for item in expected_occurrences]
    result = completeness.analyze_complete_source_unit(
        EMPTY_RULESPEC,
        case.text,
        corpus_citation_path=VA_CITATION,
        test_cases=[],
        extract_numeric_occurrences=EXTRACT_NUMBERS,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    assert result.source_numeric_occurrence_count == len(expected_occurrences)
    assert result.missing_source_numeric_occurrence_count == len(expected_occurrences)
    assert result.covered_source_numeric_occurrence_count == 0
    actual_demands = {
        float(match.group(1))
        for issue in result.issues
        if (
            match := re.search(
                r"Authoritative corpus numeric value (\S+) has no named scalar", issue
            )
        )
    }
    assert actual_demands == {item.value for item in expected_occurrences}


@st.composite
def _mixed_quantity_sources(draw):
    """Every example includes each construction; values and wording vary."""
    locator = draw(st.integers(min_value=701, max_value=799))
    case_style = draw(st.sampled_from(("original", "upper", "title", "mixed")))
    space = draw(st.sampled_from((" ", "\u00a0", "\u202f")))
    first = draw(st.integers(min_value=2, max_value=49))
    second = first + draw(st.integers(min_value=1, max_value=24))
    third = second + draw(st.integers(min_value=1, max_value=24))
    decimal = f"{draw(st.integers(min_value=2, max_value=49))}.5"
    unit = draw(st.sampled_from(("children", "applicants", "kilograms", "widgets")))
    label = draw(
        st.sampled_from(
            (
                f"subdivision {locator}",
                f"subdivision B {locator}",
                f"subdivision B{locator}",
                f"subdivision {locator}(b)",
            )
        )
    )
    quantities = [
        *(
            f"{first} or {comparison} eligible {unit} must qualify"
            for comparison in ("more", "fewer", "less", "greater")
        ),
        f"{first} of every {second} {unit} qualify",
        *(
            f"{first}{connector}{second} of the taxpayer’s children must qualify"
            for connector in (" to ", "-", "–", " through ")
        ),
        *(
            f"{first} {rate} of income is allowed"
            for rate in ("percent", "per cent", "%")
        ),
        *(
            f"{decimal} {rate} of income is allowed"
            for rate in ("percent", "per cent", "%")
        ),
        f"${first} is allowed",
        f"€{second} is allowed",
        f"{first} dollars are allowed",
        f"USD {second} is allowed",
        f"{first} qualifying {unit} are allowed",
        *(
            f"{prefix} {first} {unit} must qualify"
            for prefix in ("at least", "at most", "not more than", "up to")
        ),
        f"{first} or {decimal} or {third} percent of income is allowed",
        f"{first}, {second}, and {third} {unit} must qualify",
        f"{first} and {second} and {third} per cent of income is allowed",
        f"{first} or {second} or {third}% of income is allowed",
        # An operative occurrence equal to the identifier must still survive.
        f"${locator} is allowed",
    ]
    clauses = []
    spans = []
    offset = 0
    for quantity in quantities:
        # Prove the locator independently of any shared quantity suffix.
        clause = f"Under {label}, the following applies: {quantity}. "
        number_start = clause.index(str(locator))
        spans.append((offset + number_start, offset + number_start + len(str(locator))))
        clauses.append(clause)
        offset += len(clause)

    # Quantity context attached to the first number also prevents masking.
    clauses.extend(
        (
            *(
                f"Land subdivision {first} {connector} {second}: minimum acreage range. "
                for connector in ("through", "to")
            ),
            *(
                f"Under subdivision {first} or {comparison} eligible {unit}, the credit applies. "
                for comparison in ("more", "fewer", "less", "greater")
            ),
            f"Under subdivision {first} of every {second} {unit}, the credit applies. ",
            *(
                f"Under subdivision {first} of this {legal_object}{possessive}s {second} {unit}, the credit applies. "
                for legal_object in (
                    "section",
                    "subsection",
                    "paragraph",
                    "subdivision",
                )
                for possessive in ("'", "’")
            ),
            *(
                f"Under subdivision {first}{connector}{second} of the taxpayer’s children, the credit applies. "
                for connector in (" to ", "-", "–", " through ")
            ),
            f"Under subdivision {decimal} percent of income, the credit applies. ",
            *(
                f"Under subdivision {first}{spacing}{rate} of income, the credit applies. "
                for rate in ("percent", "per cent", "%")
                for spacing in ("", " ")
            ),
            *(
                f"Under subdivision {first}{connector}{decimal}{connector}{third} {rate} of income, the credit applies. "
                for rate in ("percent", "per cent", "%")
                for connector in (" or ", " and ")
            ),
            f"Under subdivisions {first}, {second}, and {third} {unit}, the credit applies. ",
            f"Under subdivisions {first}, {decimal}, and {third} percent of income, the credit applies. ",
            *(
                f"Under subdivision {first}, {second}{connector}{third} {rate} of income, the credit applies. "
                for connector in (", or ", ", and ", " or ", " and ")
                for rate in ("percent", "per cent", "%")
            ),
            *(
                f"A land subdivision {first}, or {comparison} acres, qualifies. "
                for comparison in ("more", "fewer", "less", "greater")
            ),
            *(
                f"A land subdivision {first}{spacing}{quantity_unit}{tail} "
                for quantity_unit in ("m", "g", "l", "s")
                for spacing in ("", " ")
                for tail in (".", " is required.")
            ),
            f"Under subdivision {first} eligible {unit}, the credit applies. ",
            f"Under subdivision ${first}, the credit applies. ",
        )
    )
    # This locator has no equal-valued operative occurrence: the gate must not
    # demand it, as well as retaining the equal-valued occurrence above.
    final_label = f"subdivision {locator + 1000}"
    final_clause = f"Under {final_label}, the credit applies."
    offset = sum(map(len, clauses))
    number_start = final_clause.index(str(locator + 1000))
    spans.append(
        (offset + number_start, offset + number_start + len(str(locator + 1000)))
    )
    clauses.append(final_clause)
    # Exercise the geography phrase beside equal-valued quantities and money.
    geography = " Use the guidelines for the 48 contiguous states. "
    start = offset + len(final_clause) + geography.index("48")
    spans.append((start, start + 2))
    clauses.extend(
        (
            geography,
            "For USD 48 contiguous states; 48 children qualify. ",
            "For yuan 48 contiguous states; 48 children qualify. ",
        )
    )
    text = _keyword_case("".join(clauses), case_style).replace(" ", space)
    return QuantitySource(text, tuple(spans))


@given(case=_mixed_quantity_sources())
@example(
    case=QuantitySource("Land subdivision 3 THROUGH 5: minimum acreage range.", ())
)
@example(
    case=QuantitySource(
        "Under subdivision 2, 3 or more eligible children must qualify.", ((18, 19),)
    )
)
@example(
    case=QuantitySource(
        "Under subdivision 2, 20 of every 100 applicants qualify.", ((18, 19),)
    )
)
@example(
    case=QuantitySource(
        "Under subdivision 2, 3–5 of the taxpayer’s children must qualify.", ((18, 19),)
    )
)
@example(
    case=QuantitySource(
        "Under subdivision 2 or 20 or 30 percent of income, the credit is $300.", ()
    )
)
@settings(
    max_examples=35,
    deadline=None,
    phases=(Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink),
    suppress_health_check=[HealthCheck.too_slow],
)
def test_every_quantity_occurrence_survives_only_proven_locator_removal(case):
    _assert_quantity_source(case)


@pytest.mark.parametrize("connector", ("through", "to"))
@pytest.mark.parametrize("style", ("original", "upper", "title", "mixed"))
def test_range_keyword_case_preserves_operative_endpoints(connector, style):
    source = _keyword_case(
        f"Land subdivision 3 {connector} 5: minimum acreage range.", style
    )
    _assert_quantity_source(QuantitySource(source, ()))


@pytest.mark.parametrize("connector", ("through", "to"))
@pytest.mark.parametrize("style", ("original", "upper", "title", "mixed"))
@pytest.mark.parametrize("space", (" ", "\u00a0", "\u202f"))
def test_complete_statutory_ranges_remain_locators_in_keyword_variants(
    connector, style, space
):
    source = _keyword_case(
        f"Under subdivisions 3 {connector} 5 of this section.", style
    ).replace(" ", space)
    spans = tuple(match.span() for match in re.finditer(r"\d+", source))
    _assert_quantity_source(QuantitySource(source, spans))


@pytest.mark.parametrize("space", (" ", "\u00a0", "\u202f"))
@pytest.mark.parametrize("connector", ("-", "–", "—", "−"))
def test_typographic_statutory_ranges_preserve_original_text(space, connector):
    source = f"Under subdivisions 3{connector}5 of this section.".replace(" ", space)
    spans = tuple(match.span() for match in re.finditer(r"\d+", source))
    _assert_quantity_source(QuantitySource(source, spans))


@pytest.mark.parametrize("space", (" ", "\u00a0", "\u202f"))
@pytest.mark.parametrize("style", ("original", "upper", "title", "mixed"))
def test_geography_and_definition_typography_preserves_operative_values(space, style):
    source = _keyword_case(
        "Under subdivision 15, “business interest” means 20 percent. "
        "Use the guidelines for the 48 contiguous states; USD 48 contiguous states.",
        style,
    ).replace(" ", space)
    locator = source.index("15")
    geography = source.index("48")
    _assert_quantity_source(
        QuantitySource(source, ((locator, locator + 2), (geography, geography + 2)))
    )


@pytest.mark.parametrize("style", ("original", "upper", "title", "mixed"))
@pytest.mark.parametrize(
    "rate", ("percent", "per cent", "per\u00a0cent", "per\u202fcent")
)
def test_glued_percentage_units_do_not_become_sentence_numbers(style, rate):
    source = _keyword_case(f"Under subdivision 2 or 3{rate} of income.", style)
    _assert_quantity_source(QuantitySource(source, ()))


@pytest.mark.parametrize(
    ("source", "operative_values", "locator_is_unambiguous"),
    (
        (
            "Under subdivision 2, 3 or more eligible children must qualify.",
            {3},
            True,
        ),
        (
            "Under subdivision 2, 20 of every 100 applicants qualify.",
            {20, 100},
            True,
        ),
        (
            "Under subdivision 2, 3–5 of the taxpayer’s children must qualify.",
            {3, 5},
            True,
        ),
        (
            "Under subdivision 2 or 20 or 30 percent of income, the credit is $300.",
            {2, 20, 30, 300},
            False,
        ),
    ),
)
def test_review_quantity_probes(source, operative_values, locator_is_unambiguous):
    cleaned = completeness.authoritative_numeric_recall_text(
        source, corpus_citation_path=VA_CITATION
    )
    if locator_is_unambiguous:
        _assert_quantity_source(QuantitySource(source, ((18, 19),)))
    else:
        # A shared quantity suffix makes the entire coordination ambiguous.
        assert cleaned == source
        result = completeness.analyze_complete_source_unit(
            EMPTY_RULESPEC,
            source,
            corpus_citation_path=VA_CITATION,
            test_cases=[],
            extract_numeric_occurrences=EXTRACT_NUMBERS,
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )
        for value in operative_values:
            assert any(
                f"Authoritative corpus numeric value {value:g} has no named scalar"
                in issue
                for issue in result.issues
            ), result.issues
    assert operative_values <= {item.value for item in EXTRACT_NUMBERS(cleaned)}


@pytest.mark.parametrize(
    ("source", "locator_spans"),
    (
        (
            "Under subdivision 2, 20 of this section's 100 applicants qualify.",
            ((18, 19),),
        ),
        (
            "Under subdivision 2, 20 of this section’s 100 applicants qualify.",
            ((18, 19),),
        ),
        ("A land subdivision 3 m is required.", ()),
        ("A land subdivision 3m.", ()),
        ("A land subdivision 3 g is required.", ()),
        ("A land subdivision 3g.", ()),
        (
            "Under subdivision 2, 20, or 30 percent of income, the credit is $300.",
            (),
        ),
        ("A land subdivision 3, or more acres, qualifies.", ()),
    ),
)
def test_possessive_counts_and_single_letter_units_are_operative(source, locator_spans):
    _assert_quantity_source(QuantitySource(source, locator_spans))


@pytest.mark.parametrize(
    "source",
    (
        "A land subdivision 2 or 3 m.",
        "Under subdivision 2 or 3 m.",
    ),
)
def test_ambiguous_coordinated_single_letter_units_keep_every_number(source):
    _assert_quantity_source(QuantitySource(source, ()))


def test_singular_statutory_range_keeps_every_number():
    source = "Under subdivision 3 to 5 of this section."
    _assert_quantity_source(QuantitySource(source, ()))


def test_shared_quantity_coordination_keeps_every_number():
    source = "Under subdivision 2, 3 or 4 children are eligible."
    _assert_quantity_source(QuantitySource(source, ()))


@pytest.mark.parametrize(
    "source",
    (
        "Land subdivision 15m x.",
        "Each lot in the subdivision 15m a.",
    ),
)
def test_two_letter_suffixes_keep_the_quantity(source):
    _assert_quantity_source(QuantitySource(source, ()))


@given(
    quantity=st.integers(min_value=1, max_value=999),
    gap=st.sampled_from(("", " ", "\t")),
    first=st.sampled_from(
        tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    ),
    second=st.one_of(
        st.none(),
        st.sampled_from(tuple("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")),
    ),
)
@example(quantity=15, gap="", first="m", second="x")
@example(quantity=15, gap=" ", first="m", second=None)
@settings(max_examples=35, deadline=None)
def test_unparenthesized_letter_suffixes_keep_quantities(quantity, gap, first, second):
    suffix = first if second is None else f"{first} {second}"
    source = f"Land subdivision {quantity}{gap}{suffix}."
    _assert_quantity_source(QuantitySource(source, ()))


def test_coordinated_decimal_rate_requires_the_full_operative_token():
    source = "under subdivision 2 or 20.5 percent"
    cleaned = completeness.authoritative_numeric_recall_text(
        source, corpus_citation_path=VA_CITATION
    )
    assert "20.5 percent" in cleaned
    decimal_start = source.index("20.5")
    assert any(
        item.value == 20.5
        and item.raw == "20.5"
        and item.start == decimal_start
        and item.end == decimal_start + len("20.5")
        for item in EXTRACT_NUMBERS(cleaned)
    )


def test_long_unproven_quantity_clause_keeps_every_number():
    source = "Under subdivision 2, 3 " + "must " * 1600 + "!"
    cleaned = completeness.authoritative_numeric_recall_text(
        source, corpus_citation_path=VA_CITATION
    )
    assert cleaned == source
    assert [item.value for item in EXTRACT_NUMBERS(cleaned)] == [2, 3]


@pytest.mark.parametrize(("opening", "closing"), (('"', '"'), ("“", "”")))
def test_quoted_definition_proves_a_complete_legal_locator(opening, closing):
    source = (
        f"For purposes of subdivision 15, {opening}business interest{closing} "
        "means 20 percent of deductible interest."
    )
    start = source.index("15")
    _assert_quantity_source(QuantitySource(source, ((start, start + 2),)))

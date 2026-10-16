"""A parameter covers a source branch only when the branch is a stated value.

Main let any rule's `source:` citation cover its branch, so a parameter
holding the number 18 "covered" the exemption "a sponsored alien child under
18 years of age". A parameter now covers a branch only when the branch text
states one of its values and the branch's own text states no condition,
exception or limit around it.
"""

from __future__ import annotations

import functools

from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CFR = "us/regulation/7/273/4"
SOURCE = """\
(a) Household members meeting citizenship or alien status requirements.
(b) Reporting illegal aliens.
(c) Households containing sponsored alien members.
(3) Exempt aliens. The provisions of paragraph (c)(2) do not apply to:
(i) An alien who is a member of the sponsor's household;
(vi) A sponsored alien child under 18 years of age.
(4) Federal share. The Federal share is 85 percent.
(5) Reporting period. The State agency must report each quarter.
"""


def _structure(rules: str) -> set[str]:
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CFR}
rules:
{rules}"""
    result = analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=CFR,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return {
        issue.split(" at ", 1)[1].split(" is neither", 1)[0].split(" [", 1)[1]
        for issue in result.issues
        if issue.startswith("[complete-source-unit:structure]")
    }


def _parameter(name: str, source: str, value: str) -> str:
    return f"""\
  - name: {name}
    kind: parameter
    dtype: Decimal
    source: {source}
    versions:
      - effective_from: '2025-01-01'
        formula: '{value}'
"""


def _labels(structure: set[str], label: str) -> bool:
    return any(item.rstrip("]").endswith(f"({label})") for item in structure)


def test_parameter_holding_an_exemption_age_does_not_cover_the_exemption():
    structure = _structure(
        _parameter("sponsored_child_age_limit", "7 CFR 273.4(c)(3)(vi)", "18")
    )
    assert _labels(structure, "vi"), structure


def test_parameter_covers_a_branch_that_only_states_its_value():
    structure = _structure(_parameter("federal_share", "7 CFR 273.4(c)(4)", "0.85"))
    assert not _labels(structure, "4"), structure


def test_parameter_citing_a_branch_that_does_not_state_its_value_covers_nothing():
    structure = _structure(
        _parameter("federal_share", "7 CFR 273.4(c)(5)", "0.85")
        + _parameter("other_share", "7 CFR 273.4(c)(4)", "0.85")
    )
    assert _labels(structure, "5"), structure
    assert not _labels(structure, "4"), structure


def test_parameter_without_a_number_covers_nothing():
    structure = _structure(
        """\
  - name: sponsor_household_member_exempt
    kind: parameter
    dtype: Judgment
    source: 7 CFR 273.4(c)(3)(i)
    versions:
      - effective_from: '2025-01-01'
        formula: 'true'
"""
    )
    assert _labels(structure, "i"), structure


def test_parameter_table_values_count_as_stated_values():
    structure = _structure(
        """\
  - name: federal_share_by_year
    kind: parameter
    dtype: Decimal
    source: 7 CFR 273.4(c)(4)
    versions:
      - effective_from: '2025-01-01'
        values:
          2025: 0.85
          2026: 0.85
"""
    )
    assert not _labels(structure, "4"), structure


def test_principal_output_still_covers_the_exemption():
    structure = _structure(
        """\
  - name: sponsored_child_exempt
    kind: derived
    dtype: Judgment
    source: 7 CFR 273.4(c)(3)(vi)
    versions:
      - effective_from: '2025-01-01'
        formula: member_age < 18
"""
    )
    assert not _labels(structure, "vi"), structure


def _covers(branch_text: str, *values: str, kind: str = "parameter") -> bool:
    """Whether rules citing (1) with these formula values cover it."""

    source = f"(1) {branch_text}\n(2) The State agency must keep records.\n"
    rules = "".join(
        f"""\
  - name: value_{index}
    kind: {kind}
    dtype: Decimal
    source: 7 CFR 273.4(1)
    versions:
      - effective_from: '2025-01-01'
        formula: '{value}'
"""
        for index, value in enumerate(values)
    )
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {CFR}
rules:
{rules}"""
    result = analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path=CFR,
        test_cases=[],
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return not any(
        issue.startswith("[complete-source-unit:structure]")
        and "Source branch (1) " in issue
        for issue in result.issues
    )


def test_values_stated_in_words_or_with_a_scale_are_stated_values():
    # Review item 1: these were uncovered by the first cut (covered on main).
    assert _covers("The deduction is one-half of the remaining income.", "0.5")
    assert _covers("The credit rate is fifty percent.", "0.5")
    assert _covers("The share is two-thirds of the amount.", "0.6666666666666666")
    assert _covers("Payment is made within ten days.", "10")
    assert _covers("The annual cap is $1.5 million.", "1500000")
    assert _covers("The fee is 50 cents per return.", "0.5")
    # Negative control: a value the branch does not state still covers nothing.
    assert not _covers("The credit rate is fifty percent.", "0.25")


def test_rate_table_rows_compared_with_the_rules_own_threshold_are_stated_values():
    # Review item 2 (us-co 39-29-103): threshold + rate parameters encode the
    # table rows that compare income with that same threshold.
    assert _covers("Amount exceeding $11,000,000 2.25%", "11000000", "0.0225")
    assert _covers(
        "Over $10,000 but not over $20,000, 5 percent of the excess over $10,000.",
        "10000",
        "20000",
        "0.05",
    )
    # A cross-reference ("set forth in (b) below") is not a comparison.
    assert _covers("The limit of $2,000 is set forth in paragraph (b) below.", "2000")
    # Negative controls: a rate alone does not make a condition a table row,
    # and an eligibility limit without a rate stays operative.
    assert not _covers("Amount exceeding $11,000,000 2.25%", "0.0225")
    assert not _covers("Liquid resources that do not exceed $100.", "100")


def test_attainment_aggregate_limits_and_lesser_of_are_conditions():
    # Review item 3: these stay operative, so a parameter does not cover them.
    assert not _covers("A child who has not attained age 17.", "17")
    assert not _covers("An individual age 65 and over.", "65")
    assert not _covers("The credit is a maximum of $2,000 per return.", "2000")
    assert not _covers("The credit is the lesser of $2,000 or the tax.", "2000")
    assert not _covers("Income that does not reach $25,000.", "25000")
    # Positive control: a stated maximum amount (not "a maximum of") is a value.
    assert _covers("The maximum allotment is $975.", "975")


def test_relation_rules_keep_their_citations():
    # Review item 4: only parameters state values; relation rules are unchanged.
    assert _covers("A child who has not attained age 17.", "17", kind="source_relation")

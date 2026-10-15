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
    return any(item.rstrip("]").endswith(label) for item in structure)


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

"""A case pair witnesses an exception only through the exception's own condition.

The paired-case gate links a toggled formula selector to a source exception by
matching the selector name against the exception's condition text. That text
ran from the condition cue to the end of the clause, so it also held the
consequence. In 7 CFR 273.4(c)(5) a pair that toggles only
`sponsored_alien_information_or_verification_needed_for_deeming_is_outstanding`
counted as the witness for

    If the same sponsor is responsible for the entire household, the entire
    household is ineligible until such time as the household provides the
    needed sponsor information or verification.

because `information`, `verification` and `sponsor` appear in the consequence.
The pair never toggles whether the same sponsor is responsible for the entire
household, so the exception was not tested. Relevance is now read from the
controlling condition alone (`_source_exception_controlling_condition_text`).
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness_module
from axiom_encode.harness.source_completeness import analyze_complete_source_unit
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

PAIRED = "require paired positive/blocking cases"
CITATION = "us/regulation/7/273/9998"
FIRST_SENTENCE = (
    "Until the alien provides information or verification necessary to carry out "
    "the provisions of paragraph (c)(2) of this section, the sponsored alien is "
    "ineligible."
)
SAME_SPONSOR = (
    "If the same sponsor is responsible for the entire household, the entire "
    "household is ineligible until such time as the household provides the needed "
    "sponsor information or verification."
)
SOURCE = f"{FIRST_SENTENCE} {SAME_SPONSOR}\n"


def _candidate(selector: str, rule: str, excerpt: str) -> str:
    return f"""format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: {CITATION}
  summary: probe
inputs:
- name: {selector}
  entity: Person
  dtype: Judgment
  period: Month
rules:
- name: {rule}
  kind: derived
  entity: Person
  dtype: Judgment
  period: Month
  source: 7 CFR 273.4(c)(5)
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: condition
        source:
          corpus_citation_path: {CITATION}
          excerpt: {excerpt}
  versions:
  - effective_from: '0001-01-01'
    formula: {selector}
"""


def _cases(selector: str, rule: str) -> list[dict]:
    module = "us:regulations/7-cfr/273/9998"
    return yaml.safe_load(
        f"""
- name: selector holds
  period: 2026-05
  input:
    {module}#input.{selector}: true
  output:
    {module}#{rule}: holds
- name: selector does not hold
  period: 2026-05
  input:
    {module}#input.{selector}: false
  output:
    {module}#{rule}: not_holds
"""
    )


def _paired_issues(selector: str, rule: str, excerpt: str) -> list[str]:
    return [
        issue
        for issue in analyze_complete_source_unit(
            _candidate(selector, rule, excerpt),
            SOURCE,
            corpus_citation_path=CITATION,
            test_cases=_cases(selector, rule),
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile="legacy",
            ),
            extract_numeric_grounding_occurrences=functools.partial(
                extract_typed_numeric_occurrences_from_text, profile="legacy"
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
        ).issues
        if PAIRED in issue
    ]


def test_pair_toggling_only_the_consequence_does_not_witness_the_exception():
    # 7 CFR 273.4(c)(5): the pair toggles information outstanding, which the
    # consequence mentions, not the same-sponsor condition.
    issues = _paired_issues(
        "sponsored_alien_information_or_verification_needed_for_deeming_is_outstanding",
        "sponsored_alien_awaiting_verification_ineligible",
        FIRST_SENTENCE,
    )

    assert issues
    assert "same sponsor is responsible for the entire household" in issues[0]


def test_pair_toggling_the_exception_condition_still_witnesses_it():
    issues = _paired_issues(
        "same_sponsor_is_responsible_for_entire_household",
        "entire_household_ineligible_pending_sponsor_information",
        SAME_SPONSOR,
    )

    assert not issues


@pytest.mark.parametrize(
    ("condition_text", "expected"),
    (
        # A leading condition ends at its comma; the consequence is dropped.
        (
            "If the same sponsor is responsible for the entire household, the "
            "entire household is ineligible until such time as the household "
            "provides the needed sponsor information or verification.",
            "If the same sponsor is responsible for the entire household",
        ),
        # A list inside the condition is kept whole.
        (
            "If the household's income, resources, or household size change, the "
            "State agency shall recompute the allotment.",
            "If the household's income, resources, or household size change",
        ),
        (
            "When the household is absent on a Saturday, Sunday, or legal holiday, "
            "the household remains eligible.",
            "When the household is absent on a Saturday, Sunday, or legal holiday",
        ),
        # A comma before `and`/`but`/`or` continues the condition.
        (
            "When the group timely reported a change, but the agency did not act "
            "on the change, the overissuance is not counted against the group.",
            "When the group timely reported a change, but the agency did not act "
            "on the change",
        ),
        # A requirement consequence is the tested condition: keep the clause.
        (
            "if the claimant is married, except for a claimant who files as a head "
            "of household, the claimant shall file a joint return.",
            "if the claimant is married, except for a claimant who files as a head "
            "of household, the claimant shall file a joint return.",
        ),
        # A parenthetical condition ends at its closing parenthesis.
        (
            "if he or she has executed USCIS Form I-864 or I-864A) deemed as that "
            "of the eligible sponsored alien must be the total monthly income.",
            "if he or she has executed USCIS Form I-864 or I-864A",
        ),
        # A comma inside a parenthesis does not end the condition.
        (
            "If continued misuse occurs (identified on the report, after a "
            "warning), the cash benefit shall be suspended for one month.",
            "If continued misuse occurs (identified on the report, after a warning)",
        ),
        # A trailing condition without an end is kept whole.
        (
            "unless the household contains an elderly member.",
            "unless the household contains an elderly member.",
        ),
        # A reservation keeps the whole clause (its substance is the result).
        (
            "subject to deeming, the eligible sponsored alien is responsible for "
            "obtaining the cooperation of the sponsor.",
            "subject to deeming, the eligible sponsored alien is responsible for "
            "obtaining the cooperation of the sponsor.",
        ),
    ),
)
def test_source_exception_controlling_condition_text(
    condition_text: str, expected: str
):
    assert (
        completeness_module._source_exception_controlling_condition_text(condition_text)
        == expected
    )

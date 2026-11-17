"""Pin the analyzer behavior behind the cap-in-its-own-rule prompt guidance.

The encoder prompt tells the model to encode a source cap ("Up to $40 per
day") in its own derived rule when the same provision also has a comparison
gate ("over 50 miles"). These tests record why, on Arizona DES FAA5 NA Medical
Expenses and Deduction block 8 (public source text, quoted verbatim below).

The candidates here are test-made mutations of the shape the encoder produced
for block 8. They are not encoder outputs. The tests pin how the
complete-source analyzer treats each shape today. They do not claim that the
analyzer verifies that a cap is applied: it does not check that the helper or
the principal output is actually capped.
"""

from __future__ import annotations

import functools

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us-az/manual/des/faa5/na-medical-expenses-and-deduction/block-8"
PREFIX = "us-az:policies/des/faa5/na-medical-expenses-and-deduction/block-8"
PRINCIPAL = f"{PREFIX}#allowable_daily_medical_lodging_cost"
HELPER = f"{PREFIX}#capped_daily_lodging_cost"
FOOD = f"{PREFIX}#medical_travel_food_cost_deduction_allowable"
COST = f"{PREFIX}#input.daily_lodging_cost"
MILES = f"{PREFIX}#input.travel_distance_miles"

CAP_SENTENCE = (
    "Up to $40 per day for lodging is allowed when the participant must travel "
    "over 50 miles to obtain medical treatments or services."
)
SOURCE = f"""The cost of lodging to obtain medical treatment or services such as doctor or dental visits and supplies are allowable.

{CAP_SENTENCE} Examples of medical treatments or services include any of the following:

● Prescriptions

● Eyeglasses

● Dentures

● Hearing aids

● Sickroom equipment

NOTE The cost of food is not allowed as a deduction."""

GATE = (
    "if participant_has_lodging_cost and "
    "lodging_is_to_obtain_medical_treatment_or_services and "
    "participant_must_travel and "
    "travel_distance_miles > medical_lodging_travel_distance_threshold:"
)
CLAMP = "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)"

HELPER_RULE = f"""  - name: capped_daily_lodging_cost
    kind: derived
    entity: Person
    dtype: Money
    period: Day
    unit: USD
    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: "{CAP_SENTENCE}"
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          {CLAMP}
"""


def _candidate(*, helper_rule: str, gated_amount: str) -> str:
    return f"""format: rulespec/v1
module:
  summary: |-
    {CAP_SENTENCE}
    NOTE The cost of food is not allowed as a deduction.
  source_verification:
    corpus_citation_path: {CITATION}
    source_sha256: 79ceaa92daca31ced9fd61ad00d4cffe928290abd5facaa2f3152b539cc7fc1b
  proof_validation:
    required: true
inputs:
  - name: participant_has_lodging_cost
    entity: Person
    dtype: Boolean
    period: Day
  - name: lodging_is_to_obtain_medical_treatment_or_services
    entity: Person
    dtype: Boolean
    period: Day
  - name: participant_must_travel
    entity: Person
    dtype: Boolean
    period: Day
  - name: travel_distance_miles
    entity: Person
    dtype: Decimal
    period: Day
  - name: daily_lodging_cost
    entity: Person
    dtype: Money
    unit: USD
    period: Day
rules:
  - name: medical_lodging_daily_cap
    kind: parameter
    dtype: Money
    period: Day
    unit: USD
    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: {CITATION}
              excerpt: "Up to $40 per day for lodging is allowed"
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          40
  - name: medical_lodging_travel_distance_threshold
    kind: parameter
    dtype: Decimal
    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: parameter
            source:
              corpus_citation_path: {CITATION}
              excerpt: "the participant must travel over 50 miles"
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          50
{helper_rule}  - name: allowable_daily_medical_lodging_cost
    kind: derived
    entity: Person
    dtype: Money
    period: Day
    unit: USD
    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              corpus_citation_path: {CITATION}
              excerpt: "The cost of lodging to obtain medical treatment or services such as doctor or dental visits and supplies are allowable."
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: "{CAP_SENTENCE}"
          - path: versions[0].formula
            kind: definition
            source:
              corpus_citation_path: {CITATION}
              excerpt: |-
                Examples of medical treatments or services include any of the following:

                ● Prescriptions

                ● Eyeglasses

                ● Dentures

                ● Hearing aids

                ● Sickroom equipment
      temporal_scope: The start date is the evidence floor following the snapshot, not a claimed legal commencement.
      cost_scope: Daily lodging cost excludes food. Medical treatment or services includes the listed examples without limiting eligibility to those examples.
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          {GATE}
            {gated_amount}
          else: 0
  - name: medical_travel_food_cost_deduction_allowable
    kind: derived
    entity: Person
    dtype: Judgment
    period: Day
    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8, NOTE
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: {CITATION}
              excerpt: "NOTE The cost of food is not allowed as a deduction."
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          false
"""


# The shape the encoder produced: gate comparison and cap `min` in one formula.
SINGLE_FORMULA = _candidate(helper_rule="", gated_amount=CLAMP)
# The shape the prompt recommends: the clamp alone in a helper rule.
HELPER_SHAPE = _candidate(
    helper_rule=HELPER_RULE, gated_amount="capped_daily_lodging_cost"
)

DAY = {
    "period_kind": "custom",
    "name": "day",
    "start": "2026-08-01",
    "end": "2026-08-01",
}
AT_CAP = "qualifying_lodging_exactly_at_daily_cap"
# name: (has cost, medical purpose, must travel, miles, cost, principal output)
AMOUNT_CASES = {
    "qualifying_lodging_below_daily_cap": (True, True, True, 51, 30, 30),
    "no_lodging_cost": (False, True, True, 51, 30, 0),
    "lodging_without_medical_purpose": (True, False, True, 51, 30, 0),
    "travel_not_required": (True, True, False, 51, 30, 0),
    "exactly_fifty_miles_is_blocked": (True, True, True, 50, 30, 0),
    "below_fifty_miles_is_blocked": (True, True, True, 49, 30, 0),
    AT_CAP: (True, True, True, 51, 40, 40),
    "qualifying_lodging_above_daily_cap": (True, True, True, 51, 60, 40),
    "negative_daily_lodging_cost_is_floored": (True, True, True, 51, -10, 0),
}


def _cases(*, assert_helper_in: tuple[str, ...] = (), drop: tuple[str, ...] = ()):
    """The encoder's companion cases, optionally asserting the cap helper."""
    cases = []
    for name, (has_cost, medical, travel, miles, cost, output) in AMOUNT_CASES.items():
        if name in drop:
            continue
        outputs = {PRINCIPAL: output}
        if name in assert_helper_in:
            outputs[HELPER] = min(max(0, cost), 40)
        cases.append(
            {
                "name": name,
                "period": dict(DAY),
                "input": {
                    f"{PREFIX}#input.participant_has_lodging_cost": has_cost,
                    f"{PREFIX}#input.lodging_is_to_obtain_medical_treatment_or_services": medical,
                    f"{PREFIX}#input.participant_must_travel": travel,
                    MILES: miles,
                    COST: cost,
                },
                "output": outputs,
            }
        )
    cases.append(
        {
            "name": "food_cost_is_not_deductible",
            "period": dict(DAY),
            "input": {},
            "output": {FOOD: "not_holds"},
        }
    )
    return cases


def _issues(candidate: str, cases: list[dict]) -> list[str]:
    profile = _numeric_profile_for_citation_path(CITATION)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate, extract_named_scalars=extract_named_scalar_occurrences
    )
    analysis = sc.analyze_complete_source_unit(
        candidate,
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return [str(issue) for issue in analysis.issues]


BOUNDARY_40 = (
    "[complete-source-unit:tests] Companion tests do not exercise every "
    "source-stated boundary input; missing: source unit=40."
)
ALL_AMOUNT_CASES = tuple(AMOUNT_CASES)


def test_gate_and_cap_in_one_formula_reports_the_cap_boundary_as_unexercised():
    # A case with the amount exactly at 40 exists, and the finding stays.
    cases = _cases()
    assert any(case["input"].get(COST) == 40 for case in cases)
    assert _issues(SINGLE_FORMULA, cases) == [BOUNDARY_40]


def test_cap_helper_rule_asserted_in_every_amount_case_is_clean():
    assert _issues(HELPER_SHAPE, _cases(assert_helper_in=ALL_AMOUNT_CASES)) == []


def test_cap_helper_rule_without_the_case_at_the_limit_reports_the_boundary():
    cases = _cases(assert_helper_in=ALL_AMOUNT_CASES, drop=(AT_CAP,))
    assert not any(case["input"].get(COST) == 40 for case in cases)
    assert _issues(HELPER_SHAPE, cases) == [BOUNDARY_40]


def test_cap_helper_rule_asserted_only_in_the_limit_case_leaves_two_findings():
    issues = _issues(HELPER_SHAPE, _cases(assert_helper_in=(AT_CAP,)))
    assert len(issues) == 2
    assert issues[0].startswith(
        "[complete-source-unit:tests] Direct nonnegative clamp companion "
        "evidence is missing for `max(0, daily_lodging_cost)` in "
        "`capped_daily_lodging_cost`."
    )
    assert issues[1].startswith(
        "[complete-source-unit:tests] Source-stated exceptions or applicability "
        "conditions require paired positive/blocking cases"
    )
    assert f"`{CAP_SENTENCE}`" in issues[1]
    assert BOUNDARY_40 not in issues


def test_boundary_unexercised_fix_text_points_to_the_cap_helper_rule():
    fix = sc.CHECK_TESTS_BOUNDARY_UNEXERCISED.fix
    assert "move the clamp into its own derived rule with no comparison" in fix
    assert "assert that rule in every case that sets the capped amount" in fix

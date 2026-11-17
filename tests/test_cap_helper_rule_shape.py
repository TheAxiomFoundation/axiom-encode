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

The last four tests cover two limits of the helper shape that the guidance
also names: a capped amount that is itself computed, and a cap selected
between two source values. Their sources and candidates are test-made
synthetic variants of the block-8 source sentence. They are not Arizona policy
text and not encoder outputs.
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

CAP_EXCERPT = "Up to $40 per day for lodging is allowed"


def _money_rule(name: str, formula: str, sentence: str = CAP_SENTENCE) -> str:
    return f"""  - name: {name}
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
              excerpt: "{sentence}"
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          {formula}
"""


def _parameter(name: str, dtype: str, kind: str, excerpt: str, value: int) -> str:
    money = "    period: Day\n    unit: USD\n" if dtype == "Money" else ""
    return f"""  - name: {name}
    kind: parameter
    dtype: {dtype}
{money}    source: Arizona DES FAA5, NA Medical Expenses and Deduction, block 8
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: {kind}
            source:
              corpus_citation_path: {CITATION}
              excerpt: "{excerpt}"
    versions:
      - effective_from: '2026-08-01'
        formula: |-
          {value}
"""


HELPER_RULE = _money_rule("capped_daily_lodging_cost", CLAMP)


def _candidate(
    *,
    helper_rule: str,
    gated_amount: str,
    sentence: str = CAP_SENTENCE,
    cap_excerpt: str = CAP_EXCERPT,
    extra_inputs: str = "",
    extra_parameters: str = "",
) -> str:
    return f"""format: rulespec/v1
module:
  summary: |-
    {sentence}
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
{extra_inputs}rules:
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
              excerpt: "{cap_excerpt}"
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
{extra_parameters}{helper_rule}  - name: allowable_daily_medical_lodging_cost
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
              excerpt: "{sentence}"
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


def _issues(candidate: str, cases: list[dict], source: str = SOURCE) -> list[str]:
    profile = _numeric_profile_for_citation_path(CITATION)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate, extract_named_scalars=extract_named_scalar_occurrences
    )
    analysis = sc.analyze_complete_source_unit(
        candidate,
        source,
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


def _case(name: str, inputs: dict, outputs: dict) -> dict:
    return {
        "name": name,
        "period": dict(DAY),
        "input": {f"{PREFIX}#input.{key}": value for key, value in inputs.items()},
        "output": {f"{PREFIX}#{key}": value for key, value in outputs.items()},
    }


FOOD_CASE = _case(
    "food_cost_is_not_deductible",
    {},
    {"medical_travel_food_cost_deduction_allowable": "not_holds"},
)
QUALIFIES = {
    "participant_has_lodging_cost": True,
    "lodging_is_to_obtain_medical_treatment_or_services": True,
    "participant_must_travel": True,
}
# name: (gate overrides, miles)
GATE_ROWS = {
    "qualifying": ({}, 51),
    "no_lodging_cost": ({"participant_has_lodging_cost": False}, 51),
    "without_medical_purpose": (
        {"lodging_is_to_obtain_medical_treatment_or_services": False},
        51,
    ),
    "travel_not_required": ({"participant_must_travel": False}, 51),
    "exactly_fifty_miles": ({}, 50),
    "below_fifty_miles": ({}, 49),
}

# Synthetic variant 1: the cap applies to a computed amount (a sum).
SUM_SENTENCE = (
    "Up to $40 per day for lodging and parking combined is allowed when the "
    "participant must travel over 50 miles to obtain medical treatments or "
    "services."
)
SUM_SOURCE = SOURCE.replace(CAP_SENTENCE, SUM_SENTENCE)
PARKING_INPUT = """  - name: daily_parking_cost
    entity: Person
    dtype: Money
    unit: USD
    period: Day
"""


def _sum_candidate(*, sum_in_own_rule: bool) -> str:
    total = "daily_lodging_cost + daily_parking_cost"
    if sum_in_own_rule:
        helpers = _money_rule(
            "daily_lodging_and_parking_cost", total, SUM_SENTENCE
        ) + _money_rule(
            "capped_daily_lodging_cost",
            "min(max(0, daily_lodging_and_parking_cost), medical_lodging_daily_cap)",
            SUM_SENTENCE,
        )
    else:
        helpers = _money_rule(
            "capped_daily_lodging_cost",
            f"min(max(0, {total}), medical_lodging_daily_cap)",
            SUM_SENTENCE,
        )
    return _candidate(
        helper_rule=helpers,
        gated_amount="capped_daily_lodging_cost",
        sentence=SUM_SENTENCE,
        cap_excerpt="Up to $40 per day for lodging and parking combined is allowed",
        extra_inputs=PARKING_INPUT,
    )


def _sum_cases(*, assert_sum_rule: bool) -> list[dict]:
    """Every case asserts the clamp rule; the sum is exactly 40 in `at_cap`."""
    rows = [
        (f"{name}_below_cap", overrides, miles, 20, 10)
        for name, (overrides, miles) in GATE_ROWS.items()
    ] + [
        ("at_cap", {}, 51, 30, 10),
        ("above_cap", {}, 51, 50, 10),
        ("negative_lodging", {}, 51, -10, 5),
        ("negative_parking", {}, 51, 5, -10),
        ("negative_both", {}, 51, -5, -10),
    ]
    cases = []
    for name, overrides, miles, lodging, parking in rows:
        gates = {**QUALIFIES, **overrides}
        capped = min(max(0, lodging + parking), 40)
        outputs = {
            "allowable_daily_medical_lodging_cost": (
                capped if all(gates.values()) and miles > 50 else 0
            ),
            "capped_daily_lodging_cost": capped,
        }
        if assert_sum_rule:
            outputs["daily_lodging_and_parking_cost"] = lodging + parking
        inputs = {
            **gates,
            "travel_distance_miles": miles,
            "daily_lodging_cost": lodging,
            "daily_parking_cost": parking,
        }
        cases.append(_case(name, inputs, outputs))
    return cases + [FOOD_CASE]


def test_computed_amount_inside_the_clamp_reports_the_cap_boundary():
    cases = _sum_cases(assert_sum_rule=False)
    at_cap = next(case for case in cases if case["name"] == "at_cap")
    assert at_cap["output"][HELPER] == 40
    assert _issues(_sum_candidate(sum_in_own_rule=False), cases, SUM_SOURCE) == [
        BOUNDARY_40
    ]


def test_computed_amount_in_its_own_rule_then_clamped_is_clean():
    cases = _sum_cases(assert_sum_rule=True)
    assert _issues(_sum_candidate(sum_in_own_rule=True), cases, SUM_SOURCE) == []


# Synthetic variant 2: two source cap values, selected by a second comparison.
TWO_CAP_SENTENCE = (
    "Up to $40 per day for lodging is allowed, or up to $75 per day for a "
    "participant over age 60, when the participant must travel over 50 miles to "
    "obtain medical treatments or services."
)
TWO_CAP_SOURCE = SOURCE.replace(CAP_SENTENCE, TWO_CAP_SENTENCE)
AGE_INPUT = """  - name: participant_age
    entity: Person
    dtype: Integer
    period: Day
"""
AGE_PARAMETERS = _parameter(
    "medical_lodging_daily_cap_over_age_threshold",
    "Money",
    "amount",
    "up to $75 per day for a participant over age 60",
    75,
) + _parameter(
    "medical_lodging_higher_cap_age_threshold",
    "Decimal",
    "parameter",
    "a participant over age 60",
    60,
)
OVER_AGE = "participant_age > medical_lodging_higher_cap_age_threshold"


def _two_cap_candidate(*, one_rule_per_cap: bool) -> str:
    if one_rule_per_cap:
        helpers = _money_rule(
            "capped_daily_lodging_cost", CLAMP, TWO_CAP_SENTENCE
        ) + _money_rule(
            "capped_daily_lodging_cost_over_age_threshold",
            "min(max(0, daily_lodging_cost), "
            "medical_lodging_daily_cap_over_age_threshold)",
            TWO_CAP_SENTENCE,
        )
        gated_amount = (
            f"(if {OVER_AGE}: capped_daily_lodging_cost_over_age_threshold "
            "else: capped_daily_lodging_cost)"
        )
    else:
        helpers = _money_rule(
            "applicable_daily_lodging_cap",
            f"if {OVER_AGE}: medical_lodging_daily_cap_over_age_threshold "
            "else: medical_lodging_daily_cap",
            TWO_CAP_SENTENCE,
        ) + _money_rule(
            "capped_daily_lodging_cost",
            "min(max(0, daily_lodging_cost), applicable_daily_lodging_cap)",
            TWO_CAP_SENTENCE,
        )
        gated_amount = "capped_daily_lodging_cost"
    return _candidate(
        helper_rule=helpers,
        gated_amount=gated_amount,
        sentence=TWO_CAP_SENTENCE,
        extra_inputs=AGE_INPUT,
        extra_parameters=AGE_PARAMETERS,
    )


def _two_cap_cases(*, one_rule_per_cap: bool) -> list[dict]:
    """Block-8 cases at age 45 plus over-60 cases; every helper is asserted."""
    rows = [(f"{name}_age_45", row[:5], 45) for name, row in AMOUNT_CASES.items()] + [
        ("older_below_cap", (True, True, True, 51, 50), 61),
        ("older_at_cap", (True, True, True, 51, 75), 61),
        ("older_above_cap", (True, True, True, 51, 90), 61),
        ("age_exactly_60_uses_lower_cap", (True, True, True, 51, 60), 60),
        ("age_59_uses_lower_cap", (True, True, True, 51, 60), 59),
        ("older_negative_cost", (True, True, True, 51, -10), 61),
        ("older_exactly_fifty_miles", (True, True, True, 50, 50), 61),
    ]
    cases = []
    for name, (has_cost, medical, travel, miles, cost), age in rows:
        cap = 75 if age > 60 else 40
        allowed = has_cost and medical and travel and miles > 50
        outputs = {
            "allowable_daily_medical_lodging_cost": (
                min(max(0, cost), cap) if allowed else 0
            )
        }
        if one_rule_per_cap:
            outputs["capped_daily_lodging_cost"] = min(max(0, cost), 40)
            outputs["capped_daily_lodging_cost_over_age_threshold"] = min(
                max(0, cost), 75
            )
        else:
            outputs["capped_daily_lodging_cost"] = min(max(0, cost), cap)
            outputs["applicable_daily_lodging_cap"] = cap
        inputs = {
            "participant_has_lodging_cost": has_cost,
            "lodging_is_to_obtain_medical_treatment_or_services": medical,
            "participant_must_travel": travel,
            "travel_distance_miles": miles,
            "daily_lodging_cost": cost,
            "participant_age": age,
        }
        cases.append(_case(name, inputs, outputs))
    return cases + [FOOD_CASE]


def test_clamp_against_a_selected_cap_rule_reports_the_cap_boundary():
    cases = _two_cap_cases(one_rule_per_cap=False)
    assert any(
        case["input"].get(COST) == 40 and case["output"].get(HELPER) == 40
        for case in cases
    )
    assert _issues(
        _two_cap_candidate(one_rule_per_cap=False), cases, TWO_CAP_SOURCE
    ) == [BOUNDARY_40]


def test_one_clamp_rule_per_source_cap_selected_in_the_principal_is_clean():
    cases = _two_cap_cases(one_rule_per_cap=True)
    assert (
        _issues(_two_cap_candidate(one_rule_per_cap=True), cases, TWO_CAP_SOURCE) == []
    )


def test_boundary_unexercised_fix_text_points_to_the_cap_helper_rule():
    fix = sc.CHECK_TESTS_BOUNDARY_UNEXERCISED.fix
    assert "move the clamp into its own derived rule with no comparison" in fix
    assert "assert that rule in every case that sets the capped amount" in fix

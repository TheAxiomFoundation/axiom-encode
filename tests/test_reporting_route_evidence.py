from __future__ import annotations

import copy
import json

import pytest

from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.reporting_route_evidence import ReportingRouteCoverage
from tests.test_measurement_role_evidence import current_material, issues


def test_actual_six_sparse_routes_have_full_source_coverage_without_filler():
    doc, cases, source = current_material()
    old = copy.deepcopy(cases)
    assert issues(doc, cases, source) == ()
    assert cases == old
    assert not any("paid_1998" in k for k in cases[0]["input"])
    assert not any(
        "unused_student_loan_interest_line_46" in k for k in cases[-1]["input"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "no_history",
        "no_prior",
        "missing_selector",
        "wrong_selector",
        "missing_paid",
        "missing_prior",
        "zero_prior",
        "inactive_filler",
        "wrong_carry_assertion",
        "missing_dependency",
        "wrong_owner",
        "wrong_unit",
        "wrong_period",
        "missing_carry_proof",
        "no_current_zero",
        "current_pair_second_fact",
        "boolean_money",
    ],
)
def test_incomplete_reporting_evidence_keeps_specific_conditions(mutation):
    doc, cases, source = current_material()
    before = copy.deepcopy((doc, cases))
    prior = next(
        k for k in cases[0]["input"] if "unused_student_loan_interest_line_46" in k
    )
    selector = next(k for k in cases[0]["input"] if "carryforward_calculated" in k)
    if mutation == "no_history":
        cases = cases[:-1]
    elif mutation == "no_prior":
        cases = cases[-1:]
    elif mutation == "missing_selector":
        del cases[-1]["input"][selector]
    elif mutation == "wrong_selector":
        cases[-1]["input"][selector] = True
    elif mutation == "missing_paid":
        del cases[-1]["input"][next(k for k in cases[-1]["input"] if "paid_1998" in k)]
    elif mutation == "missing_prior":
        for case in cases[:-1]:
            del case["input"][prior]
    elif mutation == "zero_prior":
        for case in cases[:-1]:
            case["input"][prior] = 0
            current = next(
                v for k, v in case["input"].items() if "paid_2025_line_48" in k
            )
            claim_key = next(k for k in case["input"] if "claimed_line_60" in k)
            case["input"][claim_key] = 0
            for key in case["output"]:
                case["output"][key] = (
                    current
                    if key.endswith(("available_line_52", "carryforward_line_62"))
                    else 0
                )
    elif mutation == "inactive_filler":
        cases[-1]["input"][prior] = 300
    elif mutation == "wrong_carry_assertion":
        for case in cases:
            case["output"][
                next(k for k in case["output"] if k.endswith("carryforward_line_62"))
            ] += 1
    elif mutation == "missing_dependency":
        for case in cases:
            del case["output"][
                next(k for k in case["output"] if k.endswith("available_line_52"))
            ]
    elif mutation == "wrong_owner":
        doc["rules"][3]["entity"] = "Family"
    elif mutation == "wrong_unit":
        doc["rules"][3]["unit"] = "USD"
    elif mutation == "wrong_period":
        for case in cases:
            case["period"]["end"] = "2024-12-31"
    elif mutation == "missing_carry_proof":
        doc["rules"][3]["metadata"]["proof"]["atoms"] = []
    elif mutation == "no_current_zero":
        cases = [
            c
            for c in cases
            if not any(
                "paid_2025_line_48" in k and v == 0 for k, v in c["input"].items()
            )
        ]
    elif mutation == "current_pair_second_fact":
        zero = next(
            c
            for c in cases
            if any("paid_2025_line_48" in k and v == 0 for k, v in c["input"].items())
        )
        zero["input"][prior] += 1
        for key in zero["output"]:
            if not key.endswith("claim_line_385"):
                zero["output"][key] += 1
    else:
        for case in cases:
            key = next(k for k in case["input"] if "paid_2025_line_48" in k)
            if case["input"][key] == 0:
                case["input"][key] = False
    assert json.dumps((doc, cases), sort_keys=True) != json.dumps(
        before, sort_keys=True
    )
    if mutation in {"zero_prior", "current_pair_second_fact"}:
        assert_all_four_outputs_replay(doc, cases)
    assert any(
        "paired positive/blocking" in item for item in issues(doc, cases, source)
    )


@pytest.mark.parametrize(
    "mutation", ["joined", "unknown", "duplicate", "prefix", "joined_prefix", "suffix"]
)
def test_unrecognized_source_conditions_are_not_reported_as_covered(mutation):
    doc, cases, source = current_material()
    before = source
    clause = "Complete this schedule if interest was paid in 2025 on a student loan granted to you."
    if mutation == "joined":
        source = source.replace(clause, clause[:-1] + " and you are eligible.")
    elif mutation == "unknown":
        source += "\nIf the applicant is eligible, pay the credit.\n"
    elif mutation == "duplicate":
        source += "\n" + clause + "\n"
    elif mutation == "prefix":
        source = source.replace(
            clause, "Only eligible applicants may use this form. " + clause
        )
    elif mutation == "joined_prefix":
        source = source.replace(clause, "If the applicant is eligible, " + clause)
    else:
        source = source.replace(
            clause, clause[:-1] + "; retain eligibility for the credit."
        )
    assert source != before
    assert any(
        "paired positive/blocking" in item for item in issues(doc, cases, source)
    )


def test_one_physical_route_cannot_cover_duplicate_obligations():
    text = "An exact reporting clause."
    coverage = ReportingRouteCoverage(text, ((0, len(text), "prior"),), 0, 2025)
    branch = s.SourceStructureBranch((), "condition", "route", text, 0, len(text))
    assert coverage.unmatched((branch, branch)) == (branch,)
    scoped = s.replace(branch, required_principal_output="unrelated")
    assert coverage.unmatched((scoped,)) == (scoped,)


def assert_all_four_outputs_replay(doc, cases):
    rules = {rule["name"]: rule for rule in doc["rules"]}
    environment = s._constant_rule_environment(doc)
    for case in cases:
        dependencies = s._case_asserted_dependency_environment(
            rules, case, formula_environment=environment
        )
        assert rules.keys() <= dependencies.keys()
        for name, rule in rules.items():
            execution = s._case_formula_execution(
                rule,
                case,
                formula_environment=environment,
                dependency_environment=dependencies,
            )
            assert execution is not None
            assert s._asserted_formula_runtime_values_equal(
                rule,
                s._formula_execution_runtime_value(execution),
                s._test_case_asserted_output_value(case, name),
            )


@pytest.mark.parametrize(
    "mutation",
    [
        "foreign_inputs",
        "foreign_output",
        "equal_alias",
        "short",
        "derived",
        "wrong_slot",
    ],
)
def test_reporting_certificate_requires_own_canonical_fact_identity(mutation):
    doc, cases, source = current_material()
    for case in cases:
        key = next(iter(case["input"]))
        if mutation == "foreign_inputs":
            case["input"] = {
                "foreign:unrelated#" + k.split("#", 1)[1]: v
                for k, v in case["input"].items()
            }
        elif mutation == "foreign_output":
            key = next(iter(case["output"]))
            case["output"]["foreign:unrelated#" + key.split("#", 1)[1]] = case[
                "output"
            ].pop(key)
        elif mutation == "equal_alias":
            case["input"]["foreign:unrelated#" + key.split("#", 1)[1]] = case["input"][
                key
            ]
        elif mutation == "derived":
            case["input"].update(case["output"])
        else:
            replacement = (
                key.split("#input.", 1)[1]
                if mutation == "short"
                else key.replace("#input.", "#")
            )
            case["input"][replacement] = case["input"].pop(key)
    assert any(
        "paired positive/blocking" in item for item in issues(doc, cases, source)
    )


def test_reporting_consumer_refuses_listed_assertions_from_another_row():
    doc, cases, source = current_material()
    for case in cases:
        key = next(k for k in case["output"] if k.endswith("carryforward_line_62"))
        case["output"][key] = [case["output"][key]]
        row = copy.deepcopy(case["input"])
        paid = next(k for k in row if "paid_2025_line_48" in k)
        row[paid] += 1
        case["tables"] = {"Person": [row]}
    assert any(
        "paired positive/blocking" in item for item in issues(doc, cases, source)
    )

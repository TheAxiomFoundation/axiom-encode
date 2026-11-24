import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_execution import ENGINE_PIN, PREFIX
from axiom_encode.harness.investment_adjustment_records import (
    observe_investment_adjustment_record,
)
from axiom_encode.harness.investment_adjustment_refusal import (
    observe_investment_adjustment_refusal,
    pair_investment_adjustment_refusal,
)

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
ARTIFACT = json.loads((F / "schedule-n-compiled.json").read_text())
DATA = json.loads((F / "schedule-n-part-c-refusal.json").read_text())
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def observe(case, call):
    return observe_investment_adjustment_refusal(
        BOUND, CONTENT, PARENT, GUIDE, ARTIFACT, ENGINE_PIN, case, call
    )


def positive():
    return observe_investment_adjustment_record(
        BOUND,
        CONTENT,
        PARENT,
        GUIDE,
        ARTIFACT,
        ENGINE_PIN,
        DATA["positive_case"],
        DATA["positive_call"],
        caller="pipeline",
    )


def test_actual_direct_refusal_is_separate_and_one_presence_fact_pair():
    refusal = observe(DATA["negative_case"], DATA["negative_call"])
    pairs = pair_investment_adjustment_refusal([positive()], refusal)
    assert len(pairs) == 1
    assert pairs[0][2] == "limited_partnership_loss_deduction_entered"
    assert not hasattr(refusal, "outputs")


@pytest.mark.parametrize(
    "mutation",
    [
        "successful",
        "timeout",
        "bool_exit",
        "wrong_error",
        "extra_error",
        "stdout",
        "root",
        "query",
        "relation",
        "missing_input",
        "type",
        "output_map",
        "period",
        "true_selector",
    ],
)
def test_no_generic_failure_or_forged_contract_qualifies(mutation):
    case, call = (
        copy.deepcopy(DATA["negative_case"]),
        copy.deepcopy(DATA["negative_call"]),
    )
    if mutation == "successful":
        call["returncode"] = 0
    elif mutation == "timeout":
        call["returncode"] = 124
    elif mutation == "bool_exit":
        call["returncode"] = True
    elif mutation == "wrong_error":
        call["stderr"] = "missing input"
    elif mutation == "extra_error":
        call["stderr"] += "another failure\n"
    elif mutation == "stdout":
        call["stdout"] = "{}"
    elif mutation == "root":
        call["request"]["queries"][0]["entity_id"] = "unrelated"
    elif mutation == "query":
        call["request"]["queries"][0]["outputs"].append(
            PREFIX + "schedule_n_adjustment_carryforward_line_80"
        )
    elif mutation == "relation":
        call["request"]["dataset"]["relations"] = [{"name": "invented", "tuples": []}]
    elif mutation == "missing_input":
        case["input"].pop(next(iter(case["input"])))
    elif mutation == "type":
        case["input"][PREFIX + "input.limited_partnership_loss_deduction_entered"] = 0
    elif mutation == "output_map":
        case["output"] = {}
    elif mutation == "period":
        case["period"]["start"] = "2024-01-01"
    elif mutation == "true_selector":
        key = PREFIX + "input.limited_partnership_loss_deduction_entered"
        case["input"][key] = True
        for row in call["request"]["dataset"]["inputs"]:
            if row["name"] == key:
                row["value"]["value"] = True
    with pytest.raises(ValueError):
        observe(case, call)


def test_positive_dependency_without_principal_query_cannot_pair():
    p = positive()
    refusal = observe(DATA["negative_case"], DATA["negative_call"])
    p = replace(
        p,
        outputs=tuple(
            x
            for x in p.outputs
            if not x[0].endswith("#schedule_n_other_investment_expenses_line_54")
        ),
    )
    with pytest.raises(ValueError, match="positive pair"):
        pair_investment_adjustment_refusal([p], refusal)

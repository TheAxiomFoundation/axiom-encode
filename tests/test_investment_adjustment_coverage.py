import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_coverage import (
    FORMULAS,
    observe_investment_adjustment_coverage,
)
from axiom_encode.harness.investment_adjustment_execution import ENGINE_PIN

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
ARTIFACT = json.loads((F / "schedule-n-compiled.json").read_text())
RECORD = json.loads((F / "schedule-n-actual-record.json").read_text())
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def segment():
    return {
        "companion": yaml.safe_dump([RECORD["case"]], sort_keys=False),
        "calls": [copy.deepcopy(RECORD["call"])],
    }


def observe(segments):
    return observe_investment_adjustment_coverage(
        BOUND, CONTENT, PARENT, GUIDE, ARTIFACT, ENGINE_PIN, segments, caller="pipeline"
    )


def test_single_execution_does_not_fabricate_conditional_or_boundary_witnesses():
    result = observe([segment()])
    assert {"enclosure", "records_copy", "qualified_gain_contact"} <= set(
        result.missing
    )
    assert {"floor:40", "floor:60", "floor:64"} <= set(result.missing)
    assert "part_c_inconsistent_input_refusal_required" in result.pending
    assert "protected_capture_required" in result.pending
    assert "numeric_occurrence_allocation_required" in result.pending
    assert len(result.segment_companions) == 1


def test_source_formula_owners_are_joint_and_literal_copy_not_eligibility_gate():
    assert len(FORMULAS) == 6
    assert FORMULAS[-1] == (
        2419,
        2505,
        (
            "schedule_n_adjustment_available_line_76",
            "schedule_n_adjustment_used_line_252",
            "schedule_n_adjustment_carryforward_line_80",
        ),
    )
    result = observe([segment()])
    assert any(w[0] == "formula:2419:2505" for w in result.witnesses)
    assert not any("selected_carryover" in str(w) for w in result.witnesses)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_segment",
        "duplicate_case",
        "missing_call",
        "bad_call",
        "preobserved",
        "empty",
    ],
)
def test_entire_raw_evidence_is_revalidated_without_passing_subset(mutation):
    raw = segment()
    segments = [raw]
    if mutation == "duplicate_segment":
        segments.append(copy.deepcopy(raw))
    elif mutation == "duplicate_case":
        extra = copy.deepcopy(raw)
        extra["companion"] += "\n"  # Different bytes cannot multiply the same case.
        segments.append(extra)
    elif mutation == "missing_call":
        raw["calls"] = []
    elif mutation == "bad_call":
        raw["calls"][0]["returncode"] = 1
    elif mutation == "preobserved":
        raw["witnesses"] = ["all source covered"]
    elif mutation == "empty":
        segments = []
    with pytest.raises(ValueError):
        observe(segments)


@pytest.mark.parametrize(
    "span,principal",
    [
        ((78, 850), "investment_expenses_line_18"),
        ((851, 881), "investment_expense_adjustment_line_260"),
        ((1365, 1576), "investment_income_line_36"),
        ((1854, 1884), "other_investment_expense_adjustment_line_276"),
        ((1921, 2242), "other_investment_expenses_line_54"),
        ((2419, 2505), "adjustment_available_line_76"),
    ],
)
def test_reached_dependency_without_queried_principal_cannot_own_clause(
    span, principal
):
    # Coherently omit a queried principal from all three wire locations. Its
    # genuine trace still appears because other queried outputs depend on it.
    # A trace union alone must not discharge the joint source owner.
    from axiom_encode.harness.investment_adjustment_execution import PREFIX

    key = PREFIX + "schedule_n_" + principal
    case = copy.deepcopy(RECORD["case"])
    call = copy.deepcopy(RECORD["call"])
    case["output"].pop(key)
    call["request"]["queries"][0]["outputs"].remove(key)
    response = json.loads(call["stdout"])
    response["results"][0]["outputs"].pop(key)
    call["stdout"] = json.dumps(response)
    result = observe(
        [{"companion": yaml.safe_dump([case], sort_keys=False), "calls": [call]}]
    )
    role = f"formula:{span[0]}:{span[1]}"
    assert role in result.missing
    assert not any(w[0] == role for w in result.witnesses)

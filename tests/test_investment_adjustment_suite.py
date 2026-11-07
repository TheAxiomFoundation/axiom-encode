import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_execution import ENGINE_PIN
from axiom_encode.harness.investment_adjustment_suite import (
    observe_investment_adjustment_suite,
)

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
ARTIFACT = json.loads((F / "schedule-n-compiled.json").read_text())
RECORD = json.loads((F / "schedule-n-actual-record.json").read_text())
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def observe(companion, calls):
    return observe_investment_adjustment_suite(
        BOUND,
        CONTENT,
        PARENT,
        GUIDE,
        ARTIFACT,
        ENGINE_PIN,
        companion,
        calls,
        caller="pipeline",
    )


def test_entire_raw_companion_is_bound_without_granting_source_authority():
    companion = yaml.safe_dump([RECORD["case"]], sort_keys=False)
    result = observe(companion, [RECORD["call"]])
    assert result.companion_sha256 == hashlib.sha256(companion.encode()).hexdigest()
    assert len(result.records) == 1
    assert "negative_selected_claim_domain_not_source_proved" not in result.residuals
    assert "return252_is_already_computed_supplied_fact" in BOUND.upstream_boundaries
    assert "protected_capture_required" in result.residuals


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "failed",
        "duplicate_name",
        "duplicate_yaml_key",
        "not_list",
        "empty",
    ],
)
def test_no_passing_subset_or_ambiguous_case_authority(mutation):
    case = copy.deepcopy(RECORD["case"])
    companion = yaml.safe_dump([case], sort_keys=False)
    calls = [copy.deepcopy(RECORD["call"])]
    if mutation == "missing":
        calls = []
    elif mutation == "extra":
        calls = calls * 2
    elif mutation == "failed":
        calls[0]["returncode"] = 1
    elif mutation == "duplicate_name":
        companion = yaml.safe_dump([case, case], sort_keys=False)
        calls = calls * 2
    elif mutation == "duplicate_yaml_key":
        companion = "- name: one\n  name: two\n"
    elif mutation == "not_list":
        companion = yaml.safe_dump(case)
    elif mutation == "empty":
        companion = "[]"
        calls = []
    with pytest.raises(ValueError):
        observe(companion, calls)

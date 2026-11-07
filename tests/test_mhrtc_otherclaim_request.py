"""Request association/mutation controls; transport fixtures are not source proof."""

import copy

import pytest

from axiom_encode.harness.mhrtc_otherclaim_request import (
    reconstruct_mhrtc_other_claim_request,
)
from axiom_encode.harness.mhrtc_request import reconstruct_mhrtc_request
from tests.test_mhrtc_request import fixture as principal_fixture  # noqa: F401

PREFIX = "ca:policies/cra/t1-2025/multigenerational-home-renovation-tax-credit#"


@pytest.fixture
def other_fixture(principal_fixture):  # noqa: F811 - pytest fixture dependency
    case, compiled, _, _ = principal_fixture
    case["tables"]["Claim"][0][PREFIX + "input.renovation_key"] = 101
    case["tables"]["OtherClaim"] = [
        {
            "id": "other-1",
            PREFIX + "input.other_renovation_key": 101,
            PREFIX + "input.other_claimant_key": 2,
            PREFIX + "input.other_actual_claimed_expenses": 42000,
        }
    ]
    case["relations"].append(
        {
            "name": PREFIX + "relation.candidate_other_claims",
            "tuple": ["other-1", "claim-1"],
        }
    )
    for declaration in compiled["program"]["derived"]:
        if declaration["name"] in (
            "other_claim_source_eligibility",
            "other_selected_for_individual",
            "other_allocated_local_capacity",
        ):
            money = declaration["name"] == "other_allocated_local_capacity"
            declaration.update(
                dtype="decimal" if money else "judgment",
                unit="CAD" if money else None,
                period="Year",
                versions=[
                    {"effective_from": "2025-01-01", "effective_to": "2025-12-31"}
                ],
            )
    return principal_fixture


def make_plan(fixture, kind="capacity", target="other-1"):
    case, compiled, pipe, policy = fixture
    return reconstruct_mhrtc_other_claim_request(
        case,
        compiled,
        pipe=pipe,
        rules_file=policy,
        case_index=1,
        other_claim_id=target,
        witness_kind=kind,
    )


@pytest.mark.parametrize("kind,number", [("qualification", 2), ("capacity", 1)])
def test_exact_original_dataset_and_fixed_target(other_fixture, kind, number):
    case, compiled, pipe, policy = other_fixture
    before = copy.deepcopy(case)
    principal = reconstruct_mhrtc_request(
        case, compiled, pipe=pipe, rules_file=policy, case_index=1
    )
    witness = make_plan(other_fixture, kind)
    assert witness.request["dataset"] == principal.request["dataset"]
    assert witness.request["dataset"] is not principal.request["dataset"]
    assert witness.request["queries"][0]["entity_id"] == "other-1"
    assert len(witness.derived_outputs) == number
    assert witness.parameter_outputs == ()
    witness.request["dataset"]["inputs"].clear()
    assert case == before
    assert principal.request["dataset"]["inputs"]


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "duplicate",
        "other_renovation",
        "self",
        "missing_edge",
        "wrong_filer",
        "wrong_parent",
        "missing_key",
        "wrong_kind",
        "dtype",
        "unit",
        "period",
        "versions",
        "version_type",
        "end",
    ],
)
def test_target_association_and_output_mutations_reject(other_fixture, change):
    case, compiled, _, _ = other_fixture
    target = case["tables"]["OtherClaim"][0]
    declaration = next(
        r
        for r in compiled["program"]["derived"]
        if r["name"] == "other_allocated_local_capacity"
    )
    if change == "unknown":
        target["id"] = "other-2"
    elif change == "duplicate":
        case["tables"]["OtherClaim"].append(copy.deepcopy(target))
    elif change == "other_renovation":
        target[PREFIX + "input.other_renovation_key"] = 102
    elif change == "self":
        target[PREFIX + "input.other_claimant_key"] = 1
    elif change == "missing_edge":
        case["relations"].pop()
    elif change == "wrong_filer":
        case["input"][PREFIX + "input.filing_claimant_key"] = 99
    elif change == "wrong_parent":
        case["relations"][0]["tuple"][1] = "other-1"
    elif change == "missing_key":
        target.pop(PREFIX + "input.other_claimant_key")
    elif change == "wrong_kind":
        declaration["entity"] = "Claim"
    elif change in {"dtype", "unit", "period"}:
        declaration[change] = "wrong"
    elif change == "versions":
        declaration["versions"] = []
    elif change == "version_type":
        declaration["versions"] = [None]
    else:
        declaration["versions"][0]["effective_to"] = "2026-12-31"
    with pytest.raises((ValueError, TypeError)):
        make_plan(other_fixture)


def test_irrelevant_candidate_does_not_invalidate_relevant_witness(other_fixture):
    case = other_fixture[0]
    extra = copy.deepcopy(case["tables"]["OtherClaim"][0])
    extra["id"] = "irrelevant"
    extra[PREFIX + "input.other_renovation_key"] = 999
    case["tables"]["OtherClaim"].append(extra)
    case["relations"].append(
        {
            "name": PREFIX + "relation.candidate_other_claims",
            "tuple": ["irrelevant", "claim-1"],
        }
    )
    assert make_plan(other_fixture).request["queries"][0]["entity_id"] == "other-1"
    with pytest.raises(ValueError, match="not associated"):
        make_plan(other_fixture, target="irrelevant")


def test_kind_is_closed_and_principal_otherclaim_root_remains_rejected(other_fixture):
    with pytest.raises(ValueError, match="fixed OtherClaim"):
        make_plan(other_fixture, kind="arbitrary")
    case, compiled, pipe, policy = other_fixture
    case["entity_id"] = "other-1"
    case["output"] = {PREFIX + "other_allocated_local_capacity": 50000}
    with pytest.raises(ValueError, match="one supported"):
        reconstruct_mhrtc_request(
            case, compiled, pipe=pipe, rules_file=policy, case_index=1
        )


@pytest.mark.parametrize("split_row", [True, False])
def test_claim_root_fact_placement_preserves_original_dataset(other_fixture, split_row):
    case, compiled, pipe, policy = other_fixture
    case["entity_id"] = "claim-1"
    root_facts = case["tables"]["Claim"].pop()
    root_facts.pop("id")
    case["input"] = root_facts
    if split_row:
        case["tables"]["Claim"] = [
            {
                "id": "claim-1",
                PREFIX + "input.renovation_key": case["input"].pop(
                    PREFIX + "input.renovation_key"
                ),
            }
        ]
    case["relations"] = case["relations"][1:]
    case["output"] = {PREFIX + "selected_qualifying_expenditures": 8000}
    principal = reconstruct_mhrtc_request(
        case, compiled, pipe=pipe, rules_file=policy, case_index=1
    )
    assert make_plan(other_fixture).request["dataset"] == principal.request["dataset"]

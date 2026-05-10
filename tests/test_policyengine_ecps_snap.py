import shutil
from types import SimpleNamespace

import pytest

from axiom_encode.oracles.policyengine.ecps_snap import (
    JURISDICTION_CONFIGS,
    add_snapscreener_results,
    project_income_resource_inputs,
    project_utility_allowance_type,
    set_input_value,
)
from axiom_encode.oracles.snapscreener import (
    project_payload,
    run_payloads,
    snapscreener_state_key,
)


def test_set_input_value_updates_every_matching_legal_input():
    inputs = {
        "us:policies/usda/snap/fy-2026-cola/maximum-allotments#input.household_size": 1,
        "us:policies/usda/snap/fy-2026-cola/deductions#input.household_size": 1,
        "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#input.household_size": 1,
    }

    set_input_value(inputs, "household_size", 4)

    assert set(inputs.values()) == {4}


def test_new_york_projector_uses_federal_income_and_resource_inputs():
    values = {
        "snap_earned_income": [123.45],
        "snap_unearned_income": [67.89],
        "snap_assets": [999],
    }

    projected = project_income_resource_inputs(JURISDICTION_CONFIGS["us-ny"], values, 0)

    assert projected == {
        "snap_countable_earned_income": 123.45,
        "snap_countable_unearned_income": 67.89,
        "snap_countable_financial_resources": 999,
    }


def test_new_york_policyengine_utility_type_projection_sets_region_and_bua():
    projected = project_utility_allowance_type(
        JURISDICTION_CONFIGS["us-ny"], "BUA", "NY_NAS"
    )

    assert projected["household_resides_in_new_york_city"] is False
    assert projected["household_resides_in_nassau_or_suffolk_county"] is True
    assert (
        projected["household_billed_separately_for_non_telephone_standard_utility"]
        is True
    )
    assert (
        projected[
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage"
        ]
        is False
    )


def test_snapscreener_new_york_state_key_uses_dependent_care_first():
    state_key = snapscreener_state_key(
        "NY",
        {
            "snap_utility_region_str": "NY_NAS",
            "snap_dependent_care_deduction": 12,
            "snap_earned_income": 500,
            "has_usda_elderly_disabled": False,
        },
    )

    assert state_key == "NY_NAS_DC"


def test_snapscreener_new_york_state_key_uses_earned_income_tier():
    state_key = snapscreener_state_key(
        "NY",
        {
            "snap_utility_region_str": "NY_NYC",
            "snap_dependent_care_deduction": 0,
            "snap_earned_income": 500,
            "has_usda_elderly_disabled": False,
        },
    )

    assert state_key == "NY_NYC_EI"


def test_snapscreener_payload_projects_ecps_case():
    case = SimpleNamespace(
        pe_outputs={
            "snap_unit_size": 2,
            "snap_utility_region_str": "NY_ONY",
            "snap_utility_allowance_type": "SUA",
            "snap_earned_income": 0,
            "snap_unearned_income": 2400,
            "snap_assets": float("inf"),
            "snap_dependent_care_deduction": 0,
            "snap_excess_medical_expense_deduction": 100,
            "snap_child_support_deduction": 25,
            "housing_cost": 900,
            "has_usda_elderly_disabled": True,
        }
    )

    payload = project_payload(case, state="NY")

    assert payload["state_or_territory"] == "NY_ONY_DC"
    assert payload["household_size"] == 2
    assert payload["monthly_non_job_income"] == 2400
    assert payload["resources"] == 0
    assert payload["medical_expenses_for_elderly_or_disabled"] == 135
    assert payload["utility_heating"] is True


def test_snapscreener_results_mark_primary_ineligible_rows_noncomparable():
    rows = [
        {
            "pe_snap": 0.0,
            "axiom_snap_allotment": 0.0,
            "pe_snap_eligible": False,
            "axiom_snap_eligible": "not_holds",
        }
    ]

    add_snapscreener_results(
        rows,
        [{"state_or_territory": "NY_ONY_DC"}],
        [
            {
                "status": "OK",
                "estimated_eligibility": True,
                "estimated_monthly_benefit": 298,
            }
        ],
        tolerance=1.5,
    )

    assert rows[0]["snapscreener_comparable"] is False
    assert rows[0]["axiom_snapscreener_match"] is False


def test_snapscreener_runner_executes_public_api_shape(tmp_path):
    if shutil.which("node") is None:
        pytest.skip("Node.js is not installed")
    api_js = tmp_path / "api.js"
    api_js.write_text(
        """
var SnapAPI = {
  SnapEstimateEntrypoint: class {
    constructor(payload) { this.payload = payload; }
    calculate() {
      return {
        status: "OK",
        estimated_eligibility: this.payload.state_or_territory.endsWith("_DC"),
        estimated_monthly_benefit: this.payload.household_size + 10,
        gross_income_result: this.payload.monthly_job_income + this.payload.monthly_non_job_income,
        net_income_result: 42
      };
    }
  }
};
"""
    )

    results = run_payloads(
        [
            {
                "state_or_territory": "NY_NYC_DC",
                "household_size": 2,
                "monthly_job_income": 100,
                "monthly_non_job_income": 50,
            }
        ],
        api_js=api_js,
    )

    assert results == [
        {
            "status": "OK",
            "estimated_eligibility": True,
            "estimated_monthly_benefit": 12,
            "gross_income_result": 150,
            "net_income_result": 42,
            "errors": [],
        }
    ]

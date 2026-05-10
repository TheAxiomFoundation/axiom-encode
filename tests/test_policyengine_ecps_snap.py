from axiom_encode.oracles.policyengine.ecps_snap import (
    JURISDICTION_CONFIGS,
    project_income_resource_inputs,
    project_utility_allowance_type,
    set_input_value,
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

    projected = project_income_resource_inputs(
        JURISDICTION_CONFIGS["us-ny"], values, 0
    )

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

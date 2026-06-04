import argparse
import math
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import axiom_encode.oracles.policyengine.efrs_uk as efrs_uk
from axiom_encode.oracles.policyengine.efrs_uk import (
    BENEFIT_CAP_REGULATION_80A_BASE,
    BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS,
    CHILD_BENEFIT_BASE,
    CHILD_BENEFIT_OUTPUTS,
    INCOME_TAX_INCOME_BASE_COMPONENTS,
    INCOME_TAX_INCOME_BASE_OUTPUTS,
    INCOME_TAX_SECTION_10_BASE,
    INCOME_TAX_SECTION_10_OUTPUTS,
    INCOME_TAX_SECTION_11D_BASE,
    INCOME_TAX_SECTION_11D_OUTPUTS,
    INCOME_TAX_SECTION_13_BASE,
    INCOME_TAX_SECTION_13_OUTPUTS,
    INCOME_TAX_SECTION_23_ADDITION_COMPONENTS,
    INCOME_TAX_SECTION_23_BASE,
    INCOME_TAX_SECTION_23_REDUCTION_COMPONENTS,
    NATIONAL_INSURANCE_CLASS_1_OUTPUTS,
    NATIONAL_INSURANCE_SECTION_8_BASE,
    PENSION_CREDIT_BASE,
    PENSION_CREDIT_OUTPUTS,
    PERSONAL_ALLOWANCE_BASE,
    PERSONAL_ALLOWANCE_OUTPUTS,
    PERSONAL_ALLOWANCE_PROGRAM_PATH,
    STATE_PENSION_CREDIT_QUALIFYING_AGE_OUTPUTS,
    STATE_PENSION_CREDIT_SAVINGS_CREDIT_OUTPUTS,
    STATE_PENSION_CREDIT_SECTION_1_BASE,
    STATE_PENSION_CREDIT_SECTION_3_BASE,
    UNIVERSAL_CREDIT_AWARD_OUTPUTS,
    UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS,
    UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS,
    UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS,
    UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS,
    UNIVERSAL_CREDIT_LCWRA_OUTPUTS,
    UNIVERSAL_CREDIT_REGULATION_22_BASE,
    UNIVERSAL_CREDIT_REGULATION_34_BASE,
    UNIVERSAL_CREDIT_REGULATION_72_BASE,
    UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS,
    UNIVERSAL_CREDIT_TARIFF_INCOME_OUTPUTS,
    UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS,
    WEEKS_IN_YEAR,
    WELFARE_REFORM_ACT_SECTION_8_BASE,
    WELFARE_REFORM_ACT_SECTION_11_BASE,
    build_benefit_cap_relevant_amount_request,
    build_child_benefit_request,
    build_income_tax_income_base_request,
    build_income_tax_section_10_request,
    build_income_tax_section_11d_request,
    build_income_tax_section_13_request,
    build_national_insurance_class_1_request,
    build_pension_credit_request,
    build_personal_allowance_request,
    build_state_pension_credit_qualifying_age_request,
    build_state_pension_credit_savings_credit_request,
    build_uk_efrs_coverage_report,
    build_universal_credit_award_request,
    build_universal_credit_childcare_element_request,
    build_universal_credit_housing_costs_request,
    build_universal_credit_income_deduction_request,
    build_universal_credit_request,
    build_universal_credit_tariff_income_request,
    build_universal_credit_work_allowance_request,
    compare_outputs,
    compare_uk_efrs,
    normalize_policyengine_entity,
    policyengine_benunit_variables_for_surfaces,
    policyengine_person_variables_for_surfaces,
    project_benefit_cap_relevant_amount_inputs,
    project_child_benefit_inputs,
    project_income_tax_income_base_components,
    project_income_tax_section_10_inputs,
    project_income_tax_section_11d_inputs,
    project_income_tax_section_13_inputs,
    project_income_tax_section_23_inputs,
    project_pension_credit_inputs,
    project_personal_allowance_inputs,
    project_state_pension_credit_qualifying_age_inputs,
    project_state_pension_credit_savings_credit_inputs,
    project_universal_credit_award_inputs,
    project_universal_credit_childcare_element_inputs,
    project_universal_credit_housing_costs_inputs,
    project_universal_credit_income_deduction_inputs,
    project_universal_credit_tariff_income_inputs,
    project_universal_credit_work_allowance_inputs,
    require_policyengine_uk_versions,
    select_person_indices,
)


def decimal_output(value):
    return {"value": {"value": str(value)}}


def test_national_insurance_class_1_request_projects_weekly_inputs(monkeypatch):
    monkeypatch.setattr(
        efrs_uk,
        "policyengine_uk_class_1_weekly_parameters",
        lambda year: {
            "primary_threshold": 241.73,
            "upper_earnings_limit": 966.73,
        },
    )

    request = build_national_insurance_class_1_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "ni_class_1_income": 52_000,
                    "ni_class_1_employee": 3_356.12,
                    "ni_liable": True,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "custom",
                "name": "tax_week",
                "start": "2026-04-06",
                "end": "2026-04-12",
            },
            "outputs": list(
                output["axiom"]
                for output in NATIONAL_INSURANCE_CLASS_1_OUTPUTS.values()
            ),
        }
    ]
    inputs = {
        record["name"] + ":" + record["entity_id"]: record["value"]
        for record in request["dataset"]["inputs"]
    }
    assert inputs[
        f"{NATIONAL_INSURANCE_SECTION_8_BASE}#input.primary_class_1_contribution_payable_as_mentioned_in_section_6_1_a:person_7"
    ] == {"kind": "bool", "value": True}
    assert inputs[
        f"{NATIONAL_INSURANCE_SECTION_8_BASE}#input.earnings_paid_in_tax_week_in_respect_of_employment:person_7"
    ] == {"kind": "decimal", "value": "1000.0"}
    assert inputs[
        f"{NATIONAL_INSURANCE_SECTION_8_BASE}#input.current_primary_threshold_or_prescribed_equivalent:person_7"
    ] == {"kind": "decimal", "value": "241.73"}
    assert inputs[
        f"{NATIONAL_INSURANCE_SECTION_8_BASE}#input.current_upper_earnings_limit_or_prescribed_equivalent:person_7"
    ] == {"kind": "decimal", "value": "966.73"}


class FakePolicyEngineVariable:
    def __init__(self, name, entity, *, adds=None, subtracts=None):
        self.name = name
        self.entity = entity
        self.adds = adds
        self.subtracts = subtracts


class FakePolicyEngineEntity:
    key = "benunit"


def test_normalize_policyengine_entity_accepts_entity_objects():
    assert normalize_policyengine_entity(FakePolicyEngineEntity()) == "benunit"


def test_personal_allowance_projection_matches_policyengine_gift_aid_taper():
    projected = project_personal_allowance_inputs(
        {
            "adjusted_net_income": 101_000,
            "gift_aid_grossed_up": 600,
        }
    )

    assert projected == {
        "individual_makes_claim": True,
        "individual_meets_requirements_under_section_56": True,
        "adjusted_net_income": 100_400,
    }


def test_personal_allowance_request_projects_efrs_people():
    request = build_personal_allowance_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "adjusted_net_income": 20_000,
                    "gift_aid_grossed_up": 0,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": [PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"]["axiom"]],
        }
    ]
    assert {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    } == {
        f"{PERSONAL_ALLOWANCE_BASE}#input.individual_makes_claim": {
            "kind": "bool",
            "value": True,
        },
        f"{PERSONAL_ALLOWANCE_BASE}#input.individual_meets_requirements_under_section_56": {
            "kind": "bool",
            "value": True,
        },
        f"{PERSONAL_ALLOWANCE_BASE}#input.adjusted_net_income": {
            "kind": "decimal",
            "value": "20000.0",
        },
    }


def test_income_tax_income_base_projection_uses_income_components():
    components = project_income_tax_income_base_components(
        {
            "employment_income": 30_000,
            "private_pension_income": 2_000,
            "savings_interest_income": 100,
        }
    )

    assert components == [
        {
            "name": "employment_income",
            "amount_charged_to_income_tax": 30_000,
            "relief_deducted_under_section_24": 0.0,
        },
        {
            "name": "private_pension_income",
            "amount_charged_to_income_tax": 2_000,
            "relief_deducted_under_section_24": 0.0,
        },
        {
            "name": "savings_interest_income",
            "amount_charged_to_income_tax": 100,
            "relief_deducted_under_section_24": 0.0,
        },
    ]


def test_income_tax_income_base_projection_projects_net_income_relief():
    components = project_income_tax_income_base_components(
        {
            "employment_income": 30_000,
            "private_pension_income": 2_000,
            "adjusted_net_income": 31_250,
        }
    )

    assert components[0]["relief_deducted_under_section_24"] == 750
    assert components[1]["relief_deducted_under_section_24"] == 0.0


def test_income_tax_section_23_projection_uses_pe_liability_components():
    projected = project_income_tax_section_23_inputs(
        {
            "income_tax_pre_charges": 4_500,
            "CB_HITC": 300,
            "personal_pension_contributions_tax": 200,
            "capped_mcad": 100,
            "other_tax_credits": 50,
        }
    )

    assert projected == {
        "net_income_taken_as_zero_under_section_24b": False,
        "tax_calculated_at_applicable_rates_on_income_remaining_after_allowances": 5_000,
        "tax_reductions_listed_in_section_26": 150,
        "additional_tax_amounts_listed_in_section_30": 0.0,
    }


def test_income_tax_section_10_projection_uses_pe_earned_income_and_rates():
    projected = project_income_tax_section_10_inputs(
        {"earned_taxable_income": 60_000},
        parameters={
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "basic_rate": 0.2,
            "higher_rate": 0.4,
            "additional_rate": 0.45,
        },
    )

    assert projected == {
        "income_charged_under_section_10": 60_000,
        "basic_rate_limit": 37_700,
        "higher_rate_limit": 125_140,
        "basic_rate": 0.2,
        "higher_rate": 0.4,
        "additional_rate": 0.45,
    }


def test_income_tax_section_11d_projection_removes_savings_deductions():
    projected = project_income_tax_section_11d_inputs(
        {
            "earned_taxable_income": 37_000,
            "taxable_savings_interest_income": 2_500,
            "received_allowances_savings_income": 100,
            "savings_allowance": 500,
            "savings_starter_rate_income": 400,
        },
        parameters={
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "savings_basic_rate": 0.2,
            "savings_higher_rate": 0.4,
            "savings_additional_rate": 0.45,
        },
    )

    assert projected == {
        "income_already_charged_before_section_11d_savings_income": 37_900,
        "savings_income_remaining_after_sections_12_and_12a": 1_500,
        "basic_rate_limit": 37_700,
        "higher_rate_limit": 125_140,
        "savings_basic_rate": 0.2,
        "savings_higher_rate": 0.4,
        "savings_additional_rate": 0.45,
    }


def test_income_tax_section_13_projection_uses_taxed_dividends_and_rates():
    projected = project_income_tax_section_13_inputs(
        {
            "earned_taxable_income": 37_000,
            "taxable_savings_interest_income": 2_000,
            "received_allowances_savings_income": 500,
            "taxable_dividend_income": 3_000,
            "received_allowances_dividend_income": 500,
            "taxed_dividend_income": 2_000,
        },
        parameters={
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "dividend_allowance": 500,
            "dividend_ordinary_rate": 0.1075,
            "dividend_upper_rate": 0.3575,
            "dividend_additional_rate": 0.3935,
        },
    )

    assert projected == {
        "income_already_charged_before_section_13_dividend_income": 39_000,
        "dividend_income_subject_to_section_13_rates": 2_000,
        "basic_rate_limit": 37_700,
        "higher_rate_limit": 125_140,
        "dividend_ordinary_rate": 0.1075,
        "dividend_upper_rate": 0.3575,
        "dividend_additional_rate": 0.3935,
    }


def test_income_tax_net_income_output_skips_negative_component_rows():
    spec = INCOME_TAX_INCOME_BASE_OUTPUTS["net_income"]

    assert efrs_uk.output_applies(
        spec,
        {
            "total_income": 31_250,
            "adjusted_net_income": 31_250,
            "employment_income": 30_000,
            "private_pension_income": 1_250,
        },
    )
    assert not efrs_uk.output_applies(
        spec,
        {
            "total_income": 14_483.34,
            "adjusted_net_income": 17_287.67,
            "employment_income": 17_287.67,
            "self_employment_income": -2_804.33,
        },
    )


def test_income_tax_income_base_request_projects_section_23_relation():
    request = build_income_tax_income_base_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "employment_income": 30_000,
                    "private_pension_income": 2_000,
                    "adjusted_net_income": 31_250,
                    "income_tax_pre_charges": 4_500,
                    "CB_HITC": 300,
                    "personal_pension_contributions_tax": 200,
                    "capped_mcad": 100,
                    "other_tax_credits": 50,
                    "income_tax": 4_850,
                    "total_income": 32_000,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": list(
                output["axiom"] for output in INCOME_TAX_INCOME_BASE_OUTPUTS.values()
            ),
        }
    ]
    assert request["dataset"]["relations"] == [
        {
            "name": f"{INCOME_TAX_SECTION_23_BASE}#relation.income_component_of_taxpayer",
            "tuple": ["person_7_income_employment_income", "person_7"],
            "interval": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
        },
        {
            "name": f"{INCOME_TAX_SECTION_23_BASE}#relation.income_component_of_taxpayer",
            "tuple": ["person_7_income_private_pension_income", "person_7"],
            "interval": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
        },
    ]
    inputs = {
        record["name"] + ":" + record["entity_id"]: record["value"]
        for record in request["dataset"]["inputs"]
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.amount_charged_to_income_tax:person_7_income_employment_income"
    ] == {"kind": "decimal", "value": "30000.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.relief_deducted_under_section_24:person_7_income_employment_income"
    ] == {"kind": "decimal", "value": "750.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.relief_deducted_under_section_24:person_7_income_private_pension_income"
    ] == {"kind": "decimal", "value": "0.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.net_income_taken_as_zero_under_section_24b:person_7"
    ] == {"kind": "bool", "value": False}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.tax_calculated_at_applicable_rates_on_income_remaining_after_allowances:person_7"
    ] == {"kind": "decimal", "value": "5000.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.tax_reductions_listed_in_section_26:person_7"
    ] == {"kind": "decimal", "value": "150.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.additional_tax_amounts_listed_in_section_30:person_7"
    ] == {"kind": "decimal", "value": "0.0"}


def test_income_tax_section_10_request_projects_earned_income_inputs(monkeypatch):
    monkeypatch.setattr(
        efrs_uk,
        "policyengine_uk_income_tax_section_10_parameters",
        lambda year: {
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "basic_rate": 0.2,
            "higher_rate": 0.4,
            "additional_rate": 0.45,
        },
    )
    request = build_income_tax_section_10_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "earned_taxable_income": 60_000,
                    "pays_scottish_income_tax": False,
                    "basic_rate_earned_income": 37_700,
                    "higher_rate_earned_income": 22_300,
                    "add_rate_earned_income": 0,
                    "basic_rate_earned_income_tax": 7_540,
                    "higher_rate_earned_income_tax": 8_920,
                    "add_rate_earned_income_tax": 0,
                    "earned_income_tax": 16_460,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": list(
                output["axiom"] for output in INCOME_TAX_SECTION_10_OUTPUTS.values()
            ),
        }
    ]
    inputs = {
        record["name"] + ":" + record["entity_id"]: record["value"]
        for record in request["dataset"]["inputs"]
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_10_BASE}#input.income_charged_under_section_10:person_7"
    ] == {"kind": "decimal", "value": "60000.0"}
    assert inputs[f"{INCOME_TAX_SECTION_10_BASE}#input.basic_rate_limit:person_7"] == {
        "kind": "integer",
        "value": 37700,
    }
    assert inputs[f"{INCOME_TAX_SECTION_10_BASE}#input.higher_rate_limit:person_7"] == {
        "kind": "integer",
        "value": 125140,
    }
    assert inputs[f"{INCOME_TAX_SECTION_10_BASE}#input.basic_rate:person_7"] == {
        "kind": "decimal",
        "value": "0.2",
    }
    assert inputs[f"{INCOME_TAX_SECTION_10_BASE}#input.higher_rate:person_7"] == {
        "kind": "decimal",
        "value": "0.4",
    }
    assert inputs[f"{INCOME_TAX_SECTION_10_BASE}#input.additional_rate:person_7"] == {
        "kind": "decimal",
        "value": "0.45",
    }


def test_income_tax_section_11d_request_projects_savings_inputs(monkeypatch):
    monkeypatch.setattr(
        efrs_uk,
        "policyengine_uk_income_tax_section_11d_parameters",
        lambda year: {
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "savings_basic_rate": 0.2,
            "savings_higher_rate": 0.4,
            "savings_additional_rate": 0.45,
        },
    )
    request = build_income_tax_section_11d_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "earned_taxable_income": 37_000,
                    "taxable_savings_interest_income": 2_500,
                    "received_allowances_savings_income": 100,
                    "savings_allowance": 500,
                    "savings_starter_rate_income": 400,
                    "basic_rate_savings_income": 700,
                    "higher_rate_savings_income": 800,
                    "add_rate_savings_income": 0,
                    "taxed_savings_income": 1_500,
                    "savings_income_tax": 460,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": list(
                output["axiom"] for output in INCOME_TAX_SECTION_11D_OUTPUTS.values()
            ),
        }
    ]
    inputs = {
        record["name"] + ":" + record["entity_id"]: record["value"]
        for record in request["dataset"]["inputs"]
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.income_already_charged_before_section_11d_savings_income:person_7"
    ] == {"kind": "decimal", "value": "37900.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.savings_income_remaining_after_sections_12_and_12a:person_7"
    ] == {"kind": "decimal", "value": "1500.0"}
    assert inputs[f"{INCOME_TAX_SECTION_11D_BASE}#input.basic_rate_limit:person_7"] == {
        "kind": "integer",
        "value": 37700,
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.higher_rate_limit:person_7"
    ] == {
        "kind": "integer",
        "value": 125140,
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.savings_basic_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.2",
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.savings_higher_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.4",
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_11D_BASE}#input.savings_additional_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.45",
    }


def test_income_tax_section_13_request_projects_dividend_inputs(monkeypatch):
    monkeypatch.setattr(
        efrs_uk,
        "policyengine_uk_income_tax_section_13_parameters",
        lambda year: {
            "basic_rate_limit": 37_700,
            "higher_rate_limit": 125_140,
            "dividend_allowance": 500,
            "dividend_ordinary_rate": 0.1075,
            "dividend_upper_rate": 0.3575,
            "dividend_additional_rate": 0.3935,
        },
    )
    request = build_income_tax_section_13_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "earned_taxable_income": 37_000,
                    "taxable_savings_interest_income": 2_000,
                    "received_allowances_savings_income": 500,
                    "taxable_dividend_income": 3_000,
                    "received_allowances_dividend_income": 500,
                    "taxed_dividend_income": 2_000,
                    "dividend_income_tax": 715,
                }
            ],
            "person_ids": [7],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": list(
                output["axiom"] for output in INCOME_TAX_SECTION_13_OUTPUTS.values()
            ),
        }
    ]
    inputs = {
        record["name"] + ":" + record["entity_id"]: record["value"]
        for record in request["dataset"]["inputs"]
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_13_BASE}#input.income_already_charged_before_section_13_dividend_income:person_7"
    ] == {"kind": "decimal", "value": "39000.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_13_BASE}#input.dividend_income_subject_to_section_13_rates:person_7"
    ] == {"kind": "decimal", "value": "2000.0"}
    assert inputs[f"{INCOME_TAX_SECTION_13_BASE}#input.basic_rate_limit:person_7"] == {
        "kind": "integer",
        "value": 37700,
    }
    assert inputs[f"{INCOME_TAX_SECTION_13_BASE}#input.higher_rate_limit:person_7"] == {
        "kind": "integer",
        "value": 125140,
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_13_BASE}#input.dividend_ordinary_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.1075",
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_13_BASE}#input.dividend_upper_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.3575",
    }
    assert inputs[
        f"{INCOME_TAX_SECTION_13_BASE}#input.dividend_additional_rate:person_7"
    ] == {
        "kind": "decimal",
        "value": "0.3935",
    }


def test_income_tax_section_10_output_skips_scottish_income_tax_rows():
    spec = INCOME_TAX_SECTION_10_OUTPUTS["income_tax_on_section_10_income"]

    assert efrs_uk.output_applies(spec, {"pays_scottish_income_tax": False})
    assert not efrs_uk.output_applies(spec, {"pays_scottish_income_tax": True})


def test_child_benefit_projection_uses_child_index():
    projected = project_child_benefit_inputs(
        {
            "child_benefit_child_index": 2,
            "child_benefit_respective_amount": 17.90 * WEEKS_IN_YEAR,
        }
    )

    assert projected == {
        "during_subsistence_of_marriage_any_party_married_to_more_than_one_person": False,
        "marriage_ceremony_took_place_under_law_permitting_polygamy": False,
        "specified_benefit_allowance_or_increase_paid_for_week_to_person": False,
        "specified_benefit_is_in_respect_of_only_elder_or_eldest_child_for_child_benefit_entitlement": False,
        "child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee": False,
        "paragraph_2_relationship_coordination_applies": False,
        "child_or_qualifying_young_person_is_elder_or_eldest_among_paragraph_2_children": False,
        "payee_is_voluntary_organisation": False,
        "payee_resides_with_parent_otherwise_than_paragraph_2_a": False,
    }


def test_child_benefit_request_filters_to_positive_respective_amount_rows():
    request = build_child_benefit_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "child_benefit_child_index": -1,
                    "child_benefit_respective_amount": 0,
                },
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 27.05 * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [7, 8],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_8",
            "period": {
                "period_kind": "custom",
                "name": "benefit_week",
                "start": "2026-04-06",
                "end": "2026-04-12",
            },
            "outputs": [CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"]["axiom"]],
        }
    ]
    assert {record["name"]: record["value"] for record in request["dataset"]["inputs"]}[
        f"{CHILD_BENEFIT_BASE}#input.child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee"
    ] == {
        "kind": "bool",
        "value": True,
    }


def test_benefit_cap_relevant_amount_projection_uses_pe_case_inputs():
    projected = project_benefit_cap_relevant_amount_inputs(
        {
            "num_adults": 1,
            "num_children": 0,
            "benunit_region": "LONDON",
        }
    )

    assert projected == {
        "claim_is_for_joint_claimants": False,
        "responsible_for_child_or_qualifying_young_person": False,
        "award_contains_housing_costs_element": False,
        "accommodation_in_respect_of_which_claimant_meets_occupation_condition_is_in_greater_london": False,
        "claimant_receives_housing_benefit_for_dwelling_in_greater_london": False,
        "claimant_has_accommodation_normally_occupied_as_home": True,
        "accommodation_normally_occupied_as_home_is_in_greater_london": True,
        "jobcentre_plus_office_allocated_to_claim_is_in_greater_london": False,
    }

    assert (
        project_benefit_cap_relevant_amount_inputs(
            {
                "num_adults": 2,
                "num_children": 0,
                "benunit_region": "WALES",
            }
        )["claim_is_for_joint_claimants"]
        is True
    )


def test_benefit_cap_relevant_amount_request_filters_to_finite_caps():
    request = build_benefit_cap_relevant_amount_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benefit_cap": math.inf,
                    "num_adults": 1,
                    "num_children": 0,
                    "benunit_region": "LONDON",
                },
                {
                    "benunit_id": 12,
                    "benefit_cap": 22_020,
                    "num_adults": 2,
                    "num_children": 0,
                    "benunit_region": "WALES",
                },
            ],
            "benunit_ids": [11, 12],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_12",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{BENEFIT_CAP_REGULATION_80A_BASE}#input.claim_is_for_joint_claimants"
    ] == {
        "kind": "bool",
        "value": True,
    }


def test_pension_credit_projection_uses_relation_type():
    assert project_pension_credit_inputs(
        {
            "relation_type": "COUPLE",
            "is_couple": False,
            "severe_disability_minimum_guarantee_addition": 0,
            "num_carers": 1,
        }
    ) == {
        "claimant_is_prisoner": False,
        "member_of_religious_order_fully_maintained_by_order": False,
        "claimant_has_partner": True,
        "treated_as_severely_disabled_person_under_schedule_i_part_i_paragraph_1": False,
        "severe_disability_couple_rate_conditions_satisfied": False,
        "paragraph_4_of_part_ii_of_schedule_i_satisfied_for_this_partner": True,
    }


def test_state_pension_credit_qualifying_age_projection_uses_equalized_age():
    assert project_state_pension_credit_qualifying_age_inputs(
        {
            "gender": "FEMALE",
            "state_pension_age": 66,
            "age": 67,
        }
    ) == {
        "claimant_is_woman": True,
        "pensionable_age": 66,
        "pensionable_age_for_woman_born_same_day": 66,
        "claimant_age": 67,
    }
    assert (
        project_state_pension_credit_qualifying_age_inputs(
            {
                "gender": "Gender.MALE",
                "state_pension_age": 66,
                "age": 65,
            }
        )["claimant_is_woman"]
        is False
    )


def test_state_pension_credit_qualifying_age_request_projects_people():
    request = build_state_pension_credit_qualifying_age_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "gender": "FEMALE",
                    "state_pension_age": 66,
                    "age": 67,
                }
            ],
            "person_ids": [7],
            "benunits": [],
            "benunit_ids": [],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "person_7",
            "period": {
                "period_kind": "custom",
                "name": "day",
                "start": "2026-04-06",
                "end": "2026-04-06",
            },
            "outputs": list(
                output["axiom"]
                for output in STATE_PENSION_CREDIT_QUALIFYING_AGE_OUTPUTS.values()
            ),
        }
    ]
    assert {record["name"]: record["value"] for record in request["dataset"]["inputs"]}[
        f"{STATE_PENSION_CREDIT_SECTION_1_BASE}#input.claimant_is_woman"
    ] == {
        "kind": "bool",
        "value": True,
    }


def test_pension_credit_request_projects_benefit_units():
    request = build_pension_credit_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 0,
                    "carer_minimum_guarantee_addition": 0,
                    "num_carers": 0,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "custom",
                "name": "benefit_week",
                "start": "2026-04-06",
                "end": "2026-04-12",
            },
            "outputs": list(
                output["axiom"] for output in PENSION_CREDIT_OUTPUTS.values()
            ),
        }
    ]
    assert {record["name"]: record["value"] for record in request["dataset"]["inputs"]}[
        f"{PENSION_CREDIT_BASE}#input.claimant_has_partner"
    ] == {
        "kind": "bool",
        "value": False,
    }


def test_state_pension_credit_savings_credit_projection_uses_section_3_inputs():
    projected = project_state_pension_credit_savings_credit_inputs(
        {
            "relation_type": "RelationType.COUPLE",
            "is_savings_credit_eligible": True,
            "savings_credit_income": 18_000,
            "pension_credit_income": 18_600,
            "standard_minimum_guarantee": 363.25 * WEEKS_IN_YEAR,
            "minimum_guarantee": 410.50 * WEEKS_IN_YEAR,
        },
        parameters={
            "savings_credit_threshold_single": 208.07 * WEEKS_IN_YEAR,
            "savings_credit_threshold_couple": 329.75 * WEEKS_IN_YEAR,
            "phase_in_rate": 0.6,
            "phase_out_rate": 0.4,
        },
    )

    assert projected == {
        "claimant_satisfies_savings_credit_first_condition": True,
        "claimant_qualifying_income": 18_000,
        "claimant_income": 18_600,
        "savings_credit_threshold": 329.75 * WEEKS_IN_YEAR,
        "prescribed_percentage_for_amount_a": 0.6,
        "prescribed_percentage_for_amount_b": 0.4,
        "prescribed_percentage_for_maximum_savings_credit": 0.6,
        "standard_minimum_guarantee": 363.25 * WEEKS_IN_YEAR,
        "appropriate_minimum_guarantee": 410.50 * WEEKS_IN_YEAR,
        "maximum_savings_credit_taken_as_nil_by_regulations": False,
    }


def test_state_pension_credit_savings_credit_request_projects_benefit_units(
    monkeypatch,
):
    monkeypatch.setattr(
        efrs_uk,
        "policyengine_uk_savings_credit_parameters",
        lambda year: {
            "savings_credit_threshold_single": 208.07 * WEEKS_IN_YEAR,
            "savings_credit_threshold_couple": 329.75 * WEEKS_IN_YEAR,
            "phase_in_rate": 0.6,
            "phase_out_rate": 0.4,
        },
    )

    request = build_state_pension_credit_savings_credit_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 12,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_savings_credit_eligible": True,
                    "savings_credit": 12.5 * WEEKS_IN_YEAR,
                    "savings_credit_income": 11_000,
                    "pension_credit_income": 11_000,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                    "minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [12],
        },
        year=2026,
    )

    assert request["queries"] == [
        {
            "entity_id": "benunit_12",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "outputs": list(
                output["axiom"]
                for output in STATE_PENSION_CREDIT_SAVINGS_CREDIT_OUTPUTS.values()
            ),
        }
    ]
    assert {record["name"]: record["value"] for record in request["dataset"]["inputs"]}[
        f"{STATE_PENSION_CREDIT_SECTION_3_BASE}#input.savings_credit_threshold"
    ] == {
        "kind": "decimal",
        "value": str(208.07 * WEEKS_IN_YEAR),
    }


def test_universal_credit_request_queries_monthly_table_amounts():
    request = build_universal_credit_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "uc_standard_allowance": 338.58 * 12,
                    "uc_standard_allowance_claimant_type": "SINGLE_YOUNG",
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
        surface="universal-credit-standard-allowance",
    )

    assert request["mode"] == "explain"
    assert request["dataset"] == {"inputs": [], "relations": []}
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS.values()
            ),
        }
    ]


def test_universal_credit_award_projection_uses_monthly_eligible_components():
    projected = project_universal_credit_award_inputs(
        {
            "is_uc_eligible": True,
            "uc_standard_allowance": 1_200,
            "uc_child_element": 2_400,
            "uc_disability_elements": 600,
            "uc_housing_costs_element": 360,
            "uc_childcare_element": 120,
            "uc_carer_element": 60,
            "uc_income_reduction": 300,
        }
    )

    assert projected == {
        "amount_included_under_section_9_standard_allowance": 100,
        "amount_included_under_section_10_responsibility_for_children_and_young_persons": 200,
        "amount_included_under_section_11_housing_costs": 30,
        "amount_included_under_section_12_other_particular_needs_or_circumstances": 65,
        "earned_income_deduction_calculated_in_prescribed_manner": 25,
        "unearned_income_deduction_calculated_in_prescribed_manner": 0.0,
    }

    assert project_universal_credit_award_inputs(
        {
            "is_uc_eligible": False,
            "uc_standard_allowance": 1_200,
            "uc_child_element": 2_400,
            "uc_income_reduction": 300,
        }
    ) == {
        "amount_included_under_section_9_standard_allowance": 0.0,
        "amount_included_under_section_10_responsibility_for_children_and_young_persons": 0.0,
        "amount_included_under_section_11_housing_costs": 0.0,
        "amount_included_under_section_12_other_particular_needs_or_circumstances": 0.0,
        "earned_income_deduction_calculated_in_prescribed_manner": 0.0,
        "unearned_income_deduction_calculated_in_prescribed_manner": 0.0,
    }


def test_universal_credit_award_request_projects_section_8_inputs():
    request = build_universal_credit_award_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "is_uc_eligible": True,
                    "uc_standard_allowance": 1_200,
                    "uc_child_element": 2_400,
                    "uc_disability_elements": 600,
                    "uc_housing_costs_element": 360,
                    "uc_childcare_element": 120,
                    "uc_carer_element": 60,
                    "uc_income_reduction": 300,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"] for output in UNIVERSAL_CREDIT_AWARD_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{WELFARE_REFORM_ACT_SECTION_8_BASE}#input.amount_included_under_section_12_other_particular_needs_or_circumstances"
    ] == {
        "kind": "decimal",
        "value": "65.0",
    }


def test_universal_credit_childcare_element_projection_reverses_rate_and_cap():
    projected = project_universal_credit_childcare_element_inputs(
        {
            "uc_childcare_element": 1_020,
            "uc_maximum_childcare_element_amount": 1_800,
        }
    )

    assert projected == {
        "charges_paid_for_relevant_childcare_attributable_to_assessment_period": 100,
        "amount_considered_excessive_having_regard_to_paid_work_extent": 0.0,
        "amount_met_or_reimbursed_by_employer_or_some_other_person": 0.0,
        "secretary_of_state_work_transition_childcare_payment_meets_non_other_relevant_support_conditions": False,
        "amount_from_funds_provided_by_secretary_of_state_or_scottish_or_welsh_ministers_for_work_related_activity_or_training": 0.0,
        "secretary_of_state_work_transition_childcare_payment_amount": 0.0,
        "maximum_amount_specified_in_table_in_regulation_36": 150,
    }
    assert project_universal_credit_childcare_element_inputs(
        {
            "uc_childcare_element": 0,
            "uc_maximum_childcare_element_amount": 0,
        }
    ) == {
        "charges_paid_for_relevant_childcare_attributable_to_assessment_period": 0.0,
        "amount_considered_excessive_having_regard_to_paid_work_extent": 0.0,
        "amount_met_or_reimbursed_by_employer_or_some_other_person": 0.0,
        "secretary_of_state_work_transition_childcare_payment_meets_non_other_relevant_support_conditions": False,
        "amount_from_funds_provided_by_secretary_of_state_or_scottish_or_welsh_ministers_for_work_related_activity_or_training": 0.0,
        "secretary_of_state_work_transition_childcare_payment_amount": 0.0,
        "maximum_amount_specified_in_table_in_regulation_36": 0.0,
    }


def test_universal_credit_childcare_element_request_projects_regulation_34_inputs():
    request = build_universal_credit_childcare_element_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_childcare_element": 1_020,
                    "uc_maximum_childcare_element_amount": 1_800,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_34_BASE}#input.charges_paid_for_relevant_childcare_attributable_to_assessment_period"
    ] == {
        "kind": "decimal",
        "value": "100.0",
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_34_BASE}#input.maximum_amount_specified_in_table_in_regulation_36"
    ] == {
        "kind": "decimal",
        "value": "150.0",
    }


def test_universal_credit_housing_costs_projection_uses_monthly_amount():
    projected = project_universal_credit_housing_costs_inputs(
        {"uc_housing_costs_element": 360}
    )

    assert projected["payments_are_in_respect_of_accommodation_for_section_11"] is True
    assert projected["accommodation_is_in_great_britain"] is True
    assert projected["accommodation_is_residential_accommodation"] is True
    assert projected["claimant_is_liable_to_make_accommodation_payments"] is True
    assert (
        projected[
            "claimant_is_treated_as_not_liable_to_make_accommodation_payments_by_regulations"
        ]
        is False
    )
    assert (
        projected["amount_determined_or_calculated_by_regulations_under_section_11"]
        == 30
    )


def test_universal_credit_housing_costs_request_projects_section_11_inputs():
    request = build_universal_credit_housing_costs_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "uc_housing_costs_element": 360,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{WELFARE_REFORM_ACT_SECTION_11_BASE}#input.amount_determined_or_calculated_by_regulations_under_section_11"
    ] == {
        "kind": "decimal",
        "value": "30.0",
    }


def test_universal_credit_work_allowance_projection_uses_regulation_22_case_inputs():
    projected = project_universal_credit_work_allowance_inputs(
        {
            "num_adults": 1,
            "num_children": 1,
            "is_uc_work_allowance_eligible": True,
            "uc_housing_costs_element": 360,
        }
    )

    assert projected == {
        "claim_is_for_joint_claimants": False,
        "claimant_is_member_of_couple": False,
        "claimant_makes_claim_as_single_person": False,
        "joint_claimants_responsible_for_child_or_qualifying_young_person": False,
        "one_or_both_joint_claimants_have_limited_capability_for_work": False,
        "single_claimant_responsible_for_child_or_qualifying_young_person": True,
        "single_claimant_has_limited_capability_for_work": False,
        "award_contains_housing_costs_element": True,
    }

    assert (
        project_universal_credit_work_allowance_inputs(
            {
                "num_adults": 2,
                "num_children": 0,
                "is_uc_work_allowance_eligible": True,
                "uc_housing_costs_element": 0,
            }
        )["one_or_both_joint_claimants_have_limited_capability_for_work"]
        is True
    )


def test_universal_credit_work_allowance_request_projects_regulation_22_inputs():
    request = build_universal_credit_work_allowance_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "num_adults": 1,
                    "num_children": 1,
                    "is_uc_work_allowance_eligible": True,
                    "uc_housing_costs_element": 360,
                    "uc_work_allowance": 5_124,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_22_BASE}#input.award_contains_housing_costs_element"
    ] == {
        "kind": "bool",
        "value": True,
    }


def test_universal_credit_income_deduction_projection_uses_regulation_22_inputs():
    projected = project_universal_credit_income_deduction_inputs(
        {
            "num_adults": 1,
            "num_children": 1,
            "is_uc_work_allowance_eligible": True,
            "uc_housing_costs_element": 360,
            "uc_work_allowance": 5_124,
            "uc_earned_income": 6_876,
            "uc_unearned_income": 600,
        }
    )

    assert projected == {
        "claim_is_for_joint_claimants": False,
        "claimant_is_member_of_couple": False,
        "claimant_makes_claim_as_single_person": False,
        "joint_claimants_responsible_for_child_or_qualifying_young_person": False,
        "one_or_both_joint_claimants_have_limited_capability_for_work": False,
        "single_claimant_responsible_for_child_or_qualifying_young_person": True,
        "single_claimant_has_limited_capability_for_work": False,
        "award_contains_housing_costs_element": True,
        "claimant_earned_income_in_assessment_period": 1_000,
        "joint_claimants_combined_earned_income_in_assessment_period": 0.0,
        "claimant_unearned_income_in_assessment_period": 50,
        "joint_claimants_combined_unearned_income_in_assessment_period": 0.0,
    }

    joint_projection = project_universal_credit_income_deduction_inputs(
        {
            "num_adults": 2,
            "num_children": 0,
            "is_uc_work_allowance_eligible": True,
            "uc_housing_costs_element": 0,
            "uc_work_allowance": 8_520,
            "uc_earned_income": 15_480,
            "uc_unearned_income": 3_600,
        }
    )
    assert (
        joint_projection["joint_claimants_combined_earned_income_in_assessment_period"]
        == 2_000
    )
    assert (
        joint_projection[
            "joint_claimants_combined_unearned_income_in_assessment_period"
        ]
        == 300
    )
    assert (
        joint_projection["one_or_both_joint_claimants_have_limited_capability_for_work"]
        is True
    )


def test_universal_credit_income_deduction_request_projects_regulation_22_inputs():
    request = build_universal_credit_income_deduction_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "num_adults": 1,
                    "num_children": 1,
                    "is_uc_work_allowance_eligible": True,
                    "uc_housing_costs_element": 360,
                    "uc_work_allowance": 5_124,
                    "uc_earned_income": 6_876,
                    "uc_unearned_income": 600,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_22_BASE}#input.claimant_earned_income_in_assessment_period"
    ] == {
        "kind": "decimal",
        "value": "1000.0",
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_22_BASE}#input.claimant_unearned_income_in_assessment_period"
    ] == {
        "kind": "decimal",
        "value": "50.0",
    }


def test_universal_credit_tariff_income_projection_uses_regulation_72_inputs():
    assert project_universal_credit_tariff_income_inputs(
        {
            "uc_assessable_capital": 6_001,
        }
    ) == {
        "person_capital": 6_001,
        "capital_is_disregarded": False,
        "actual_income_from_capital_taken_into_account_under_regulation_66_1_i_annuity": False,
        "actual_income_from_capital_taken_into_account_under_regulation_66_1_j_trust": False,
        "actual_income_derived_from_that_capital_due_to_be_paid_to_person_on_day_amount": 0.0,
    }


def test_universal_credit_tariff_income_request_projects_regulation_72_inputs():
    request = build_universal_credit_tariff_income_request(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "is_uc_eligible": True,
                    "uc_assessable_capital": 6_001,
                    "uc_tariff_income": 52.20,
                }
            ],
            "benunit_ids": [11],
        },
        year=2026,
    )

    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "benunit_11",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_TARIFF_INCOME_OUTPUTS.values()
            ),
        }
    ]
    inputs_by_name = {
        record["name"]: record["value"] for record in request["dataset"]["inputs"]
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_72_BASE}#input.person_capital"
    ] == {
        "kind": "decimal",
        "value": "6001.0",
    }
    assert inputs_by_name[
        f"{UNIVERSAL_CREDIT_REGULATION_72_BASE}#input.capital_is_disregarded"
    ] == {
        "kind": "bool",
        "value": False,
    }


def test_universal_credit_child_element_request_filters_to_positive_rows():
    request = build_universal_credit_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "uc_child_index": -1,
                    "uc_individual_child_element": 0,
                    "uc_individual_disabled_child_element": 0,
                    "uc_individual_severely_disabled_child_element": 0,
                },
                {
                    "person_id": 8,
                    "uc_child_index": 1,
                    "uc_is_child_born_before_child_limit": True,
                    "uc_individual_child_element": 351.88 * 12,
                    "uc_individual_disabled_child_element": 0,
                    "uc_individual_severely_disabled_child_element": 0,
                },
            ],
            "person_ids": [7, 8],
        },
        year=2026,
        surface="universal-credit-child-element",
    )

    assert request["queries"] == [
        {
            "entity_id": "person_8",
            "period": {
                "period_kind": "month",
                "name": "benefit_month",
                "start": "2026-04-01",
                "end": "2026-04-30",
            },
            "outputs": list(
                output["axiom"]
                for output in UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS.values()
            ),
        }
    ]


def test_run_axiom_parameter_outputs_reads_generated_rulespec_parameters(tmp_path):
    program = tmp_path / "regulations" / "uksi" / "2013" / "376" / "36.yaml"
    program.parent.mkdir(parents=True)
    program.write_text(
        """
format: rulespec/v1
rules:
  - name: standard_allowance_single_under_25_amount
    kind: parameter
    dtype: Money
    period: Month
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          300
      - effective_from: '2026-04-01'
        formula: |-
          338.58
""".strip()
    )

    results = efrs_uk.run_axiom_parameter_outputs(
        program=program,
        rulespec_root=tmp_path,
        request={
            "queries": [
                {
                    "period": {"start": "2026-04-01"},
                    "outputs": [
                        "uk:regulations/uksi/2013/376/36#standard_allowance_single_under_25_amount"
                    ],
                }
            ]
        },
    )

    assert results == [
        {
            "outputs": {
                "uk:regulations/uksi/2013/376/36#standard_allowance_single_under_25_amount": {
                    "value": {"value": "338.58"}
                }
            }
        }
    ]


def test_run_axiom_parameter_outputs_resolves_composed_program_imports(tmp_path):
    rulespec_root = tmp_path / "rulespec-uk"
    source = rulespec_root / "regulations" / "uksi" / "2013" / "376" / "36.yaml"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
format: rulespec/v1
rules:
  - name: carer_element_amount
    kind: parameter
    dtype: Money
    period: Month
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          200
      - effective_from: '2026-04-01'
        formula: |-
          209.34
""".strip()
    )
    composed = tmp_path / "uk-uc-composed.yaml"
    composed.write_text(
        """
format: rulespec/v1
module:
  kind: composition
imports:
  - uk:regulations/uksi/2013/376/36
""".strip()
    )

    results = efrs_uk.run_axiom_parameter_outputs(
        program=composed,
        rulespec_root=rulespec_root,
        request={
            "queries": [
                {
                    "period": {"start": "2026-04-01"},
                    "outputs": ["uk:regulations/uksi/2013/376/36#carer_element_amount"],
                }
            ]
        },
    )

    assert results == [
        {
            "outputs": {
                "uk:regulations/uksi/2013/376/36#carer_element_amount": {
                    "value": {"value": "209.34"}
                }
            }
        }
    ]


def test_select_person_indices_uses_positive_weights_and_explicit_ids():
    rows = [
        {"person_id": 10, "person_weight": 0},
        {"person_id": 11, "person_weight": 2.5},
        {"person_id": 12, "person_weight": 3.5},
    ]

    assert select_person_indices(rows, sample_size=1) == [1]
    assert select_person_indices(rows, sample_size=0) == [1, 2]
    assert select_person_indices(rows, sample_size=1, person_ids=(12,)) == [2]

    with pytest.raises(SystemExit, match="not eligible"):
        select_person_indices(rows, sample_size=0, person_ids=(10,))


def test_policyengine_variables_for_surfaces_deduplicates_person_variables():
    assert policyengine_person_variables_for_surfaces(
        ["personal-allowance", "child-benefit"]
    ) == (
        "adjusted_net_income",
        "child_benefit_child_index",
        "child_benefit_respective_amount",
        "gift_aid_grossed_up",
        "personal_allowance",
    )
    assert policyengine_person_variables_for_surfaces(["income-tax-income-base"]) == (
        *sorted(
            (
                *INCOME_TAX_INCOME_BASE_COMPONENTS,
                *INCOME_TAX_SECTION_23_ADDITION_COMPONENTS,
                *INCOME_TAX_SECTION_23_REDUCTION_COMPONENTS,
                "adjusted_net_income",
                "income_tax",
                "total_income",
            )
        ),
    )
    assert policyengine_person_variables_for_surfaces(
        ["income-tax-section-13-dividend-income"]
    ) == (
        "dividend_income_tax",
        "earned_taxable_income",
        "received_allowances_dividend_income",
        "received_allowances_savings_income",
        "taxable_dividend_income",
        "taxable_savings_interest_income",
        "taxed_dividend_income",
    )
    assert policyengine_person_variables_for_surfaces(
        ["national-insurance-class-1"]
    ) == (
        "ni_class_1_employee",
        "ni_class_1_employee_additional",
        "ni_class_1_employee_primary",
        "ni_class_1_income",
        "ni_employee",
        "ni_liable",
    )
    assert policyengine_person_variables_for_surfaces(
        ["state-pension-credit-qualifying-age"]
    ) == (
        "age",
        "gender",
        "is_SP_age",
        "state_pension_age",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["personal-allowance", "pension-credit"]
    ) == (
        "carer_minimum_guarantee_addition",
        "is_couple",
        "num_carers",
        "relation_type",
        "severe_disability_minimum_guarantee_addition",
        "standard_minimum_guarantee",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["state-pension-credit-savings-credit"]
    ) == (
        "is_savings_credit_eligible",
        "minimum_guarantee",
        "pension_credit_income",
        "relation_type",
        "savings_credit",
        "savings_credit_income",
        "standard_minimum_guarantee",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["benefit-cap-relevant-amount"]
    ) == (
        "benefit_cap",
        "benunit_region",
        "num_adults",
        "num_children",
    )
    assert policyengine_person_variables_for_surfaces(
        ["universal-credit-child-element"]
    ) == (
        "uc_child_index",
        "uc_individual_child_element",
        "uc_individual_disabled_child_element",
        "uc_individual_severely_disabled_child_element",
        "uc_is_child_born_before_child_limit",
    )
    assert policyengine_benunit_variables_for_surfaces(
        [
            "universal-credit-standard-allowance",
            "universal-credit-carer-element",
        ]
    ) == (
        "uc_carer_element",
        "uc_standard_allowance",
        "uc_standard_allowance_claimant_type",
    )
    assert policyengine_benunit_variables_for_surfaces(["universal-credit-award"]) == (
        "is_uc_eligible",
        "uc_carer_element",
        "uc_child_element",
        "uc_childcare_element",
        "uc_disability_elements",
        "uc_housing_costs_element",
        "uc_income_reduction",
        "uc_maximum_amount",
        "uc_standard_allowance",
        "universal_credit_pre_benefit_cap",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["universal-credit-housing-costs"]
    ) == ("uc_housing_costs_element",)
    assert policyengine_benunit_variables_for_surfaces(
        ["universal-credit-childcare-element"]
    ) == (
        "uc_childcare_element",
        "uc_maximum_childcare_element_amount",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["universal-credit-work-allowance"]
    ) == (
        "is_uc_work_allowance_eligible",
        "num_adults",
        "num_children",
        "uc_housing_costs_element",
        "uc_work_allowance",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["universal-credit-income-deduction"]
    ) == (
        "is_uc_work_allowance_eligible",
        "num_adults",
        "num_children",
        "uc_earned_income",
        "uc_housing_costs_element",
        "uc_income_reduction",
        "uc_maximum_amount",
        "uc_unearned_income",
        "uc_work_allowance",
    )
    assert policyengine_benunit_variables_for_surfaces(
        ["universal-credit-tariff-income"]
    ) == (
        "is_uc_eligible",
        "uc_assessable_capital",
        "uc_tariff_income",
    )
    assert policyengine_person_variables_for_surfaces(
        ["income-tax-section-11d-savings-income"]
    ) == (
        "add_rate_savings_income",
        "basic_rate_savings_income",
        "earned_taxable_income",
        "higher_rate_savings_income",
        "received_allowances_savings_income",
        "savings_allowance",
        "savings_income_tax",
        "savings_starter_rate_income",
        "taxable_savings_interest_income",
        "taxed_savings_income",
    )


def test_uk_efrs_coverage_report_counts_computed_pe_backlog(tmp_path):
    source_root = tmp_path / "variables"
    gov_root = source_root / "gov" / "hmrc"
    input_root = source_root / "input"
    gov_root.mkdir(parents=True)
    input_root.mkdir(parents=True)
    (gov_root / "income_tax.py").write_text(
        """
from policyengine_uk.model_api import *

class income_tax(Variable):
    entity = Person
    def formula(person, period, parameters):
        return person("taxable_income", period)

class personal_allowance(Variable):
    entity = Person
    def formula(person, period, parameters):
        return 12570
""".strip()
    )
    (input_root / "employment_income.py").write_text(
        """
from policyengine_uk.model_api import *

class employment_income(Variable):
    entity = Person
    adds = ["employment_income_before_lsr"]

class age(Variable):
    entity = Person
""".strip()
    )

    report = build_uk_efrs_coverage_report(
        policyengine_variables=[
            FakePolicyEngineVariable("income_tax", "person"),
            FakePolicyEngineVariable("personal_allowance", "person"),
            FakePolicyEngineVariable(
                "employment_income",
                "person",
                adds=["employment_income_before_lsr"],
            ),
            FakePolicyEngineVariable("age", "person"),
        ],
        source_root=source_root,
    )

    assert report.variables_total == 4
    assert report.computed_variables_total == 3
    assert report.computed_variables_by_entity == {"person": 3}
    assert report.computed_variables_by_domain == {"gov": 2, "input": 1}
    assert report.computed_covered_variables == [
        "income_tax",
        "personal_allowance",
    ]
    assert report.missing_variables_total == 1
    assert [variable.name for variable in report.missing_variables] == [
        "employment_income",
    ]


def test_uk_efrs_coverage_report_filters_domain_and_entity(tmp_path):
    source_root = tmp_path / "variables"
    (source_root / "gov").mkdir(parents=True)
    (source_root / "household").mkdir(parents=True)
    (source_root / "gov" / "tax.py").write_text(
        """
from policyengine_uk.model_api import *

class income_tax(Variable):
    entity = Person
    def formula(person, period, parameters):
        return 1

class universal_credit(Variable):
    entity = BenUnit
    def formula(benunit, period, parameters):
        return 1
""".strip()
    )
    (source_root / "household" / "net_income.py").write_text(
        """
from policyengine_uk.model_api import *

class hbai_household_net_income(Variable):
    entity = Household
    def formula(household, period, parameters):
        return 1
""".strip()
    )

    report = build_uk_efrs_coverage_report(
        policyengine_variables=[
            FakePolicyEngineVariable("income_tax", "person"),
            FakePolicyEngineVariable("universal_credit", "benunit"),
            FakePolicyEngineVariable("hbai_household_net_income", "household"),
        ],
        source_root=source_root,
        domain_filter="gov",
        entity_filter="benunit",
    )

    assert report.variables_total == 1
    assert report.computed_variables_total == 1
    assert report.missing_variables_total == 1
    assert report.missing_variables[0].name == "universal_credit"


def test_uk_efrs_coverage_report_serializes_json(tmp_path):
    source_root = tmp_path / "variables" / "gov"
    source_root.mkdir(parents=True)
    (source_root / "income_tax.py").write_text(
        """
from policyengine_uk.model_api import *

class income_tax(Variable):
    entity = Person
    def formula(person, period, parameters):
        return 1
""".strip()
    )

    report = build_uk_efrs_coverage_report(
        policyengine_variables=[FakePolicyEngineVariable("income_tax", "person")],
        source_root=source_root.parent,
    )

    payload = report.to_json()
    assert payload["missing_variables_total"] == 0
    assert payload["missing_variables"] == []
    assert "income_tax" in payload["covered_output_variables"]
    assert "personal_allowance" in payload["covered_output_variables"]


def test_policyengine_uk_version_guard_rejects_unpinned_version(monkeypatch):
    def fake_version(package):
        versions = {
            "policyengine-core": efrs_uk.POLICYENGINE_CORE_VERSION,
            "policyengine-uk": "2.88.39",
        }
        return versions[package]

    monkeypatch.setattr(efrs_uk, "version", fake_version)

    with pytest.raises(SystemExit, match="policyengine-uk==2.88.40 required"):
        require_policyengine_uk_versions()


def test_policyengine_uk_version_guard_allows_pinned_versions(monkeypatch):
    def fake_version(package):
        versions = {
            "policyengine-core": efrs_uk.POLICYENGINE_CORE_VERSION,
            "policyengine-uk": efrs_uk.POLICYENGINE_UK_VERSION,
        }
        return versions[package]

    monkeypatch.setattr(efrs_uk, "version", fake_version)

    require_policyengine_uk_versions()


def test_policyengine_uk_version_guard_names_coverage_command(monkeypatch):
    def fake_version(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(efrs_uk, "version", fake_version)

    with pytest.raises(SystemExit, match="axiom-encode uk-efrs-coverage"):
        require_policyengine_uk_versions(command="uk-efrs-coverage")


def test_compare_outputs_reports_personal_allowance_mismatch():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
        axiom_outputs_by_surface={
            "personal-allowance": [
                {
                    "outputs": {
                        PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                            "axiom"
                        ]: decimal_output(12_569)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_persons == 1
    assert report.compared_benunits == 0
    assert report.compared_values == 1
    assert len(report.mismatches) == 1
    assert report.oracle_divergences == []
    assert report.mismatches[0].entity_id == "person_7"
    assert report.output_summary[0]["mismatches"] == 1
    assert report.skipped_surfaces == []


def test_compare_outputs_rejects_missing_axiom_rows():
    with pytest.raises(ValueError, match="personal-allowance produced 0 Axiom"):
        compare_outputs(
            pe_data={
                "persons": [
                    {
                        "person_id": 7,
                        "personal_allowance": 12_570,
                    }
                ],
                "person_ids": [7],
            },
            axiom_outputs_by_surface={"personal-allowance": []},
            tolerance=0.01,
            relative_tolerance=2e-7,
        )


def test_compare_outputs_classifies_known_policyengine_personal_allowance_rounding():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "personal_allowance": 12_569.5,
                }
            ],
            "person_ids": [7],
        },
        axiom_outputs_by_surface={
            "personal-allowance": [
                {
                    "outputs": {
                        PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                            "axiom"
                        ]: decimal_output(12_570)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1738")
    assert report.output_summary[0]["oracle_divergences"] == 1


def test_compare_outputs_compares_child_benefit_weekly_rate():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "child_benefit_child_index": -1,
                    "child_benefit_respective_amount": 0,
                },
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 27.05 * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [7, 8],
        },
        axiom_outputs_by_surface={
            "child-benefit": [
                {
                    "outputs": {
                        CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"][
                            "axiom"
                        ]: decimal_output(27.05)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_persons == 2
    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []
    assert report.output_summary[0]["compared"] == 1


def test_compare_outputs_transforms_benefit_cap_relevant_amount_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benefit_cap": 22_020,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "benefit-cap-relevant-amount": [
                {
                    "outputs": {
                        BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS[
                            "benefit_cap_relevant_amount"
                        ]["axiom"]: decimal_output(1_835),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []


def test_compare_outputs_compares_applicable_universal_credit_standard_allowance():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_standard_allowance": 338.58 * 12,
                    "uc_standard_allowance_claimant_type": "SINGLE_YOUNG",
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-standard-allowance": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS[
                            "standard_allowance_single_under_25"
                        ]["axiom"]: decimal_output(338.58),
                        UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS[
                            "standard_allowance_single_25_or_over"
                        ]["axiom"]: decimal_output(424.90),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []
    assert [
        (item["output"], item["compared"])
        for item in report.output_summary
        if item["compared"]
    ] == [("standard_allowance_single_under_25", 1)]


def test_compare_outputs_compares_raw_protected_universal_credit_lcwra_amount():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_LCWRA_element": 429.80 * 12,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-lcwra-element": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_LCWRA_OUTPUTS[
                            "lcwra_element_standard_lcwra_claimant"
                        ]["axiom"]: decimal_output(217.26),
                        UNIVERSAL_CREDIT_LCWRA_OUTPUTS[
                            "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant"
                        ]["axiom"]: decimal_output(429.80),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []
    assert [
        (item["output"], item["compared"])
        for item in report.output_summary
        if item["compared"]
    ] == [
        (
            "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant",
            1,
        )
    ]


def test_compare_outputs_transforms_universal_credit_award_outputs_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_maximum_amount": 12_000,
                    "uc_income_reduction": 3_600,
                    "universal_credit_pre_benefit_cap": 8_400,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-award": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_AWARD_OUTPUTS[
                            "universal_credit_maximum_amount"
                        ]["axiom"]: decimal_output(1_000),
                        UNIVERSAL_CREDIT_AWARD_OUTPUTS[
                            "universal_credit_amounts_to_be_deducted"
                        ]["axiom"]: decimal_output(300),
                        UNIVERSAL_CREDIT_AWARD_OUTPUTS["universal_credit_award_amount"][
                            "axiom"
                        ]: decimal_output(700),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 3
    assert report.mismatches == []


def test_compare_outputs_transforms_universal_credit_housing_costs_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_housing_costs_element": 360,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-housing-costs": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS[
                            "section_11_amount_for_accommodation_payments"
                        ]["axiom"]: decimal_output(30),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_transforms_universal_credit_childcare_element_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_childcare_element": 1_020,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-childcare-element": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS[
                            "childcare_costs_element_amount"
                        ]["axiom"]: decimal_output(85),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_transforms_universal_credit_work_allowance_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_work_allowance": 5_124,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-work-allowance": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS[
                            "applicable_work_allowance_amount"
                        ]["axiom"]: decimal_output(427),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []


def test_compare_outputs_transforms_universal_credit_income_deduction_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_earned_income": 6_876,
                    "uc_unearned_income": 600,
                    "uc_income_reduction": 4_381.8,
                    "uc_maximum_amount": 12_000,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-income-deduction": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "earned_income_amount_subject_to_taper"
                        ]["axiom"]: decimal_output(573),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "unearned_income_for_deduction"
                        ]["axiom"]: decimal_output(50),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "universal_credit_award_deduction_from_maximum_amount"
                        ]["axiom"]: decimal_output(365.15),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 3
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_matches_income_tax_section_10_earned_income():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "pays_scottish_income_tax": False,
                    "basic_rate_earned_income": 37_700,
                    "higher_rate_earned_income": 22_300,
                    "add_rate_earned_income": 0,
                    "basic_rate_earned_income_tax": 7_540,
                    "higher_rate_earned_income_tax": 8_920,
                    "add_rate_earned_income_tax": 0,
                    "earned_income_tax": 16_460,
                }
            ],
            "person_ids": [7],
            "benunits": [],
            "benunit_ids": [],
        },
        axiom_outputs_by_surface={
            "income-tax-section-10-earned-income": [
                {
                    "outputs": {
                        INCOME_TAX_SECTION_10_OUTPUTS["income_charged_at_basic_rate"][
                            "axiom"
                        ]: decimal_output(37_700),
                        INCOME_TAX_SECTION_10_OUTPUTS["income_charged_at_higher_rate"][
                            "axiom"
                        ]: decimal_output(22_300),
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "income_charged_at_additional_rate"
                        ]["axiom"]: decimal_output(0),
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "tax_on_income_charged_at_basic_rate"
                        ]["axiom"]: decimal_output(7_540),
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "tax_on_income_charged_at_higher_rate"
                        ]["axiom"]: decimal_output(8_920),
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "tax_on_income_charged_at_additional_rate"
                        ]["axiom"]: decimal_output(0),
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "income_tax_on_section_10_income"
                        ]["axiom"]: decimal_output(16_460),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 7
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_matches_income_tax_section_11d_savings_income():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "basic_rate_savings_income": 700,
                    "higher_rate_savings_income": 800,
                    "add_rate_savings_income": 0,
                    "taxed_savings_income": 1_500,
                    "savings_income_tax": 460,
                }
            ],
            "person_ids": [7],
            "benunits": [],
            "benunit_ids": [],
        },
        axiom_outputs_by_surface={
            "income-tax-section-11d-savings-income": [
                {
                    "outputs": {
                        INCOME_TAX_SECTION_11D_OUTPUTS[
                            "savings_income_charged_at_savings_basic_rate"
                        ]["axiom"]: decimal_output(700),
                        INCOME_TAX_SECTION_11D_OUTPUTS[
                            "savings_income_charged_at_savings_higher_rate"
                        ]["axiom"]: decimal_output(800),
                        INCOME_TAX_SECTION_11D_OUTPUTS[
                            "savings_income_charged_at_savings_additional_rate"
                        ]["axiom"]: decimal_output(0),
                        INCOME_TAX_SECTION_11D_OUTPUTS[
                            "savings_income_charged_under_section_11d"
                        ]["axiom"]: decimal_output(1_500),
                        INCOME_TAX_SECTION_11D_OUTPUTS[
                            "income_tax_on_section_11d_savings_income"
                        ]["axiom"]: decimal_output(460),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 5
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_skips_scottish_income_tax_section_10_rows():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "pays_scottish_income_tax": True,
                    "earned_income_tax": 17_000,
                }
            ],
            "person_ids": [7],
            "benunits": [],
            "benunit_ids": [],
        },
        axiom_outputs_by_surface={
            "income-tax-section-10-earned-income": [
                {
                    "outputs": {
                        INCOME_TAX_SECTION_10_OUTPUTS[
                            "income_tax_on_section_10_income"
                        ]["axiom"]: decimal_output(16_460),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 0
    assert report.mismatches == []


def test_compare_outputs_skips_capped_universal_credit_income_deduction_final():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_earned_income": 12_000,
                    "uc_unearned_income": 0,
                    "uc_income_reduction": 3_000,
                    "uc_maximum_amount": 3_000,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-income-deduction": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "earned_income_amount_subject_to_taper"
                        ]["axiom"]: decimal_output(1_000),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "unearned_income_for_deduction"
                        ]["axiom"]: decimal_output(0),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "universal_credit_award_deduction_from_maximum_amount"
                        ]["axiom"]: decimal_output(550),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 2
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_skips_negative_unearned_income_deduction_final():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_earned_income": 24_000,
                    "uc_unearned_income": -1_200,
                    "uc_income_reduction": 12_000,
                    "uc_maximum_amount": 20_000,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-income-deduction": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "earned_income_amount_subject_to_taper"
                        ]["axiom"]: decimal_output(2_000),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "unearned_income_for_deduction"
                        ]["axiom"]: decimal_output(-100),
                        UNIVERSAL_CREDIT_INCOME_DEDUCTION_OUTPUTS[
                            "universal_credit_award_deduction_from_maximum_amount"
                        ]["axiom"]: decimal_output(1_100),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 2
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_transforms_universal_credit_tariff_income_monthly():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "is_uc_eligible": True,
                    "uc_assessable_capital": 6_001,
                    "uc_tariff_income": 52.20,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-tariff-income": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_TARIFF_INCOME_OUTPUTS[
                            "capital_tariff_monthly_income"
                        ]["axiom"]: decimal_output(4.35),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_skips_undefined_universal_credit_tariff_income():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "is_uc_eligible": False,
                    "uc_assessable_capital": 20_000,
                    "uc_tariff_income": 0,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-tariff-income": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_TARIFF_INCOME_OUTPUTS[
                            "capital_tariff_monthly_income"
                        ]["axiom"]: decimal_output(243.60),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 0
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_skips_rebalanced_universal_credit_lcwra_health_element():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "uc_LCWRA_element": 426.5063 * 12,
                },
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "universal-credit-lcwra-element": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_LCWRA_OUTPUTS[
                            "lcwra_element_standard_lcwra_claimant"
                        ]["axiom"]: decimal_output(217.26),
                        UNIVERSAL_CREDIT_LCWRA_OUTPUTS[
                            "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant"
                        ]["axiom"]: decimal_output(429.80),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 0
    assert report.mismatches == []
    assert report.oracle_divergences == []


def test_compare_outputs_classifies_known_policyengine_universal_credit_rates():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 8,
                    "uc_child_index": 1,
                    "uc_is_child_born_before_child_limit": True,
                    "uc_individual_child_element": 350.526123 * 12,
                    "uc_individual_disabled_child_element": 0,
                    "uc_individual_severely_disabled_child_element": 0,
                },
            ],
            "person_ids": [8],
        },
        axiom_outputs_by_surface={
            "universal-credit-child-element": [
                {
                    "outputs": {
                        UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS[
                            "child_element_first_child_or_qualifying_young_person"
                        ]["axiom"]: decimal_output(351.88)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1741")


def test_compare_outputs_classifies_known_policyengine_child_benefit_amounts():
    report = compare_outputs(
        pe_data={
            "persons": [
                {
                    "person_id": 8,
                    "child_benefit_child_index": 1,
                    "child_benefit_respective_amount": 26.935709699992934
                    * WEEKS_IN_YEAR,
                },
            ],
            "person_ids": [8],
        },
        axiom_outputs_by_surface={
            "child-benefit": [
                {
                    "outputs": {
                        CHILD_BENEFIT_OUTPUTS["child_benefit_weekly_rate"][
                            "axiom"
                        ]: decimal_output(27.05)
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1739")


def test_compare_outputs_classifies_known_policyengine_pension_credit_rates():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 0,
                    "carer_minimum_guarantee_addition": 0,
                    "num_carers": 0,
                    "standard_minimum_guarantee": 229.3929826081932 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "pension-credit": [
                {
                    "outputs": {
                        PENSION_CREDIT_OUTPUTS["standard_minimum_guarantee"][
                            "axiom"
                        ]: decimal_output(238.00),
                        PENSION_CREDIT_OUTPUTS["severe_disability_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                        PENSION_CREDIT_OUTPUTS["carer_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].entity_id == "benunit_11"
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1740")


def test_compare_outputs_classifies_policyengine_savings_credit_maximum_bug():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 79791,
                    "savings_credit": 4022.257080078125,
                    "standard_minimum_guarantee": 18889.0,
                    "minimum_guarantee": 27838.2,
                }
            ],
            "benunit_ids": [79791],
        },
        axiom_outputs_by_surface={
            "state-pension-credit-savings-credit": [
                {
                    "outputs": {
                        f"{STATE_PENSION_CREDIT_SECTION_3_BASE}#savings_credit": decimal_output(
                            1045.2
                        )
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=0,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].entity_id == "benunit_79791"
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1753")


def test_compare_outputs_classifies_known_policyengine_pension_credit_additions():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 0,
                    "carer_minimum_guarantee_addition": 47.9466 * WEEKS_IN_YEAR,
                    "num_carers": 1,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "pension-credit": [
                {
                    "outputs": {
                        PENSION_CREDIT_OUTPUTS["standard_minimum_guarantee"][
                            "axiom"
                        ]: decimal_output(238.00),
                        PENSION_CREDIT_OUTPUTS["severe_disability_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                        PENSION_CREDIT_OUTPUTS["carer_additional_amount"][
                            "axiom"
                        ]: decimal_output(48.15),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].output == "carer_additional_amount"
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1742")


def test_compare_outputs_classifies_known_policyengine_pension_credit_severe_addition():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 85.682
                    * WEEKS_IN_YEAR,
                    "carer_minimum_guarantee_addition": 0,
                    "num_carers": 0,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "pension-credit": [
                {
                    "outputs": {
                        PENSION_CREDIT_OUTPUTS["standard_minimum_guarantee"][
                            "axiom"
                        ]: decimal_output(238.00),
                        PENSION_CREDIT_OUTPUTS["severe_disability_additional_amount"][
                            "axiom"
                        ]: decimal_output(86.05),
                        PENSION_CREDIT_OUTPUTS["carer_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.mismatches == []
    assert len(report.oracle_divergences) == 1
    assert report.oracle_divergences[0].output == "severe_disability_additional_amount"
    assert report.oracle_divergences[0].issue_url.endswith("/issues/1742")


def test_compare_outputs_does_not_hide_wrong_pension_credit_additional_rate():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 0,
                    "carer_minimum_guarantee_addition": 47.9466 * WEEKS_IN_YEAR,
                    "num_carers": 1,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "pension-credit": [
                {
                    "outputs": {
                        PENSION_CREDIT_OUTPUTS["standard_minimum_guarantee"][
                            "axiom"
                        ]: decimal_output(238.00),
                        PENSION_CREDIT_OUTPUTS["severe_disability_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                        PENSION_CREDIT_OUTPUTS["carer_additional_amount"][
                            "axiom"
                        ]: decimal_output(44.00),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.oracle_divergences == []
    assert len(report.mismatches) == 1
    assert report.mismatches[0].output == "carer_additional_amount"


def test_compare_outputs_does_not_hide_wrong_pension_credit_severe_addition_rate():
    report = compare_outputs(
        pe_data={
            "persons": [],
            "person_ids": [],
            "benunits": [
                {
                    "benunit_id": 11,
                    "benunit_weight": 1,
                    "relation_type": "SINGLE",
                    "is_couple": False,
                    "severe_disability_minimum_guarantee_addition": 85.682
                    * WEEKS_IN_YEAR,
                    "carer_minimum_guarantee_addition": 0,
                    "num_carers": 0,
                    "standard_minimum_guarantee": 238.00 * WEEKS_IN_YEAR,
                }
            ],
            "benunit_ids": [11],
        },
        axiom_outputs_by_surface={
            "pension-credit": [
                {
                    "outputs": {
                        PENSION_CREDIT_OUTPUTS["standard_minimum_guarantee"][
                            "axiom"
                        ]: decimal_output(238.00),
                        PENSION_CREDIT_OUTPUTS["severe_disability_additional_amount"][
                            "axiom"
                        ]: decimal_output(82.00),
                        PENSION_CREDIT_OUTPUTS["carer_additional_amount"][
                            "axiom"
                        ]: decimal_output(0),
                    }
                }
            ]
        },
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.oracle_divergences == []
    assert len(report.mismatches) == 1
    assert report.mismatches[0].output == "severe_disability_additional_amount"


def test_compare_uk_efrs_runs_axiom_personal_allowance(
    monkeypatch,
    tmp_path,
):
    rulespec_root = tmp_path / "rulespec-uk"
    program = rulespec_root / PERSONAL_ALLOWANCE_PROGRAM_PATH
    program.parent.mkdir(parents=True)
    program.write_text("format: rulespec/v1\n")
    axiom_rules = tmp_path / "axiom-rules-engine"
    axiom_rules.mkdir()
    captured = {}

    monkeypatch.setattr(
        efrs_uk,
        "load_policyengine_uk_data",
        lambda **_: {
            "persons": [
                {
                    "person_id": 7,
                    "person_weight": 1,
                    "adjusted_net_income": 20_000,
                    "gift_aid_grossed_up": 0,
                    "personal_allowance": 12_570,
                }
            ],
            "person_ids": [7],
        },
    )

    def fake_run_axiom_program(**kwargs):
        captured.update(kwargs)
        return [
            {
                "outputs": {
                    PERSONAL_ALLOWANCE_OUTPUTS["personal_allowance"][
                        "axiom"
                    ]: decimal_output(12_570)
                }
            }
        ]

    monkeypatch.setattr(efrs_uk, "run_axiom_surface", fake_run_axiom_program)

    report = compare_uk_efrs(
        workspace_root=tmp_path,
        rulespec_root=None,
        axiom_rules_path=None,
        year=2026,
        sample_size=100,
        surface="personal-allowance",
        dataset="enhanced_frs_2023_24",
        data_folder=Path(".axiom/policyengine-data"),
        tolerance=0.01,
        relative_tolerance=2e-7,
    )

    assert report.compared_values == 1
    assert report.mismatches == []
    assert captured["program"] == program
    assert captured["rulespec_root"] == rulespec_root.resolve()
    assert captured["axiom_rules_path"] == axiom_rules.resolve()
    assert captured["request"]["queries"][0]["entity_id"] == "person_7"


def test_main_returns_nonzero_when_requested_for_mismatches(monkeypatch, tmp_path):
    monkeypatch.setattr(efrs_uk, "resolve_workspace_root", lambda root: tmp_path)
    monkeypatch.setattr(
        efrs_uk,
        "compare_uk_efrs",
        lambda **_: efrs_uk.UKEFRSComparisonReport(
            compared_persons=1,
            compared_benunits=0,
            compared_values=1,
            mismatches=[
                efrs_uk.UKEFRSComparisonRow(
                    surface="personal-allowance",
                    entity_id="person_7",
                    output="personal_allowance",
                    axiom=0,
                    policyengine=1,
                    diff=-1,
                )
            ],
            oracle_divergences=[],
            output_summary=[],
            skipped_surfaces=[],
            projection_notes=[],
        ),
    )

    assert (
        efrs_uk.main(
            argparse.Namespace(
                root=None,
                rulespec_root=None,
                axiom_rules_engine_path=None,
                year=2026,
                sample_size=100,
                surface="all",
                dataset="enhanced_frs_2023_24",
                data_folder=Path(".axiom/policyengine-data"),
                tolerance=0.01,
                relative_tolerance=2e-7,
                person_ids=None,
                json=True,
                fail_on_mismatch=True,
            )
        )
        == 1
    )

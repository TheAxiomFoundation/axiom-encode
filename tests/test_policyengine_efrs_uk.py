import argparse
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import pytest

import axiom_encode.oracles.policyengine.efrs_uk as efrs_uk
from axiom_encode.oracles.policyengine.efrs_uk import (
    CHILD_BENEFIT_BASE,
    CHILD_BENEFIT_OUTPUTS,
    INCOME_TAX_INCOME_BASE_COMPONENTS,
    INCOME_TAX_INCOME_BASE_OUTPUTS,
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
    STATE_PENSION_CREDIT_SECTION_1_BASE,
    UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS,
    UNIVERSAL_CREDIT_LCWRA_OUTPUTS,
    UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS,
    WEEKS_IN_YEAR,
    build_child_benefit_request,
    build_income_tax_income_base_request,
    build_national_insurance_class_1_request,
    build_pension_credit_request,
    build_personal_allowance_request,
    build_state_pension_credit_qualifying_age_request,
    build_uk_efrs_coverage_report,
    build_universal_credit_request,
    compare_outputs,
    compare_uk_efrs,
    normalize_policyengine_entity,
    policyengine_benunit_variables_for_surfaces,
    policyengine_person_variables_for_surfaces,
    project_child_benefit_inputs,
    project_income_tax_income_base_components,
    project_income_tax_section_23_inputs,
    project_pension_credit_inputs,
    project_personal_allowance_inputs,
    project_state_pension_credit_qualifying_age_inputs,
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
        },
        {
            "name": "private_pension_income",
            "amount_charged_to_income_tax": 2_000,
        },
        {
            "name": "savings_interest_income",
            "amount_charged_to_income_tax": 100,
        },
    ]


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
        "tax_calculated_at_applicable_rates_on_income_remaining_after_allowances": 5_000,
        "tax_reductions_listed_in_section_26": 150,
        "additional_tax_amounts_listed_in_section_30": 0.0,
    }


def test_income_tax_income_base_request_projects_section_23_relation():
    request = build_income_tax_income_base_request(
        pe_data={
            "persons": [
                {
                    "person_id": 7,
                    "employment_income": 30_000,
                    "private_pension_income": 2_000,
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
        f"{INCOME_TAX_SECTION_23_BASE}#input.tax_calculated_at_applicable_rates_on_income_remaining_after_allowances:person_7"
    ] == {"kind": "decimal", "value": "5000.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.tax_reductions_listed_in_section_26:person_7"
    ] == {"kind": "decimal", "value": "150.0"}
    assert inputs[
        f"{INCOME_TAX_SECTION_23_BASE}#input.additional_tax_amounts_listed_in_section_30:person_7"
    ] == {"kind": "decimal", "value": "0.0"}


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
                "income_tax",
                "total_income",
            )
        ),
    )
    assert policyengine_person_variables_for_surfaces(
        ["national-insurance-class-1"]
    ) == (
        "ni_class_1_employee",
        "ni_class_1_employee_additional",
        "ni_class_1_employee_primary",
        "ni_class_1_income",
        "ni_liable",
    )
    assert policyengine_person_variables_for_surfaces(
        ["state-pension-credit-qualifying-age"]
    ) == (
        "age",
        "gender",
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

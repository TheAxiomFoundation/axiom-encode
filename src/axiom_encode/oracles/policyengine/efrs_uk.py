"""Compare UK RuleSpec output against PolicyEngine Enhanced FRS.

The UK counterpart to the ECPS tax comparators covers mapped surfaces whose
RuleSpec inputs can be projected from PolicyEngine's Enhanced FRS, plus scalar
parameter surfaces whose generated RuleSpec values can be compared against
PolicyEngine component outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml

from .ecps_tax import (
    POLICYENGINE_CORE_VERSION,
    input_record,
    money,
    output_number,
    relative_diff,
    run_axiom_program,
    within_tolerance,
)

DEFAULT_DATASET = "enhanced_frs_2023_24"
WEEKS_IN_YEAR = 52
MONTHS_IN_YEAR = 12
POLICYENGINE_UK_VERSION = "2.88.40"

NATIONAL_INSURANCE_SECTION_8_PROGRAM_PATH = Path("statutes/ukpga/1992/4/8.yaml")
NATIONAL_INSURANCE_SECTION_8_BASE = "uk:statutes/ukpga/1992/4/8"
PERSONAL_ALLOWANCE_PROGRAM_PATH = Path("statutes/ukpga/2007/3/35.yaml")
PERSONAL_ALLOWANCE_BASE = "uk:statutes/ukpga/2007/3/35"
INCOME_TAX_SECTION_23_PROGRAM_PATH = Path("statutes/ukpga/2007/3/23.yaml")
INCOME_TAX_SECTION_23_BASE = "uk:statutes/ukpga/2007/3/23"
CHILD_BENEFIT_PROGRAM_PATH = Path("regulations/uksi/2006/965/2.yaml")
CHILD_BENEFIT_BASE = "uk:regulations/uksi/2006/965/2"
BENEFIT_CAP_REGULATION_80A_PROGRAM_PATH = Path("regulations/uksi/2013/376/80A.yaml")
BENEFIT_CAP_REGULATION_80A_BASE = "uk:regulations/uksi/2013/376/80A"
STATE_PENSION_CREDIT_SECTION_1_PROGRAM_PATH = Path("statutes/ukpga/2002/16/1.yaml")
STATE_PENSION_CREDIT_SECTION_1_BASE = "uk:statutes/ukpga/2002/16/1"
PENSION_CREDIT_PROGRAM_PATH = Path("regulations/uksi/2002/1792/6.yaml")
PENSION_CREDIT_BASE = "uk:regulations/uksi/2002/1792/6"
UNIVERSAL_CREDIT_PROGRAM_PATH = Path("regulations/uksi/2013/376/36.yaml")
UNIVERSAL_CREDIT_BASE = "uk:regulations/uksi/2013/376/36"
UNIVERSAL_CREDIT_REGULATION_22_PROGRAM_PATH = Path("regulations/uksi/2013/376/22.yaml")
UNIVERSAL_CREDIT_REGULATION_22_BASE = "uk:regulations/uksi/2013/376/22"
UNIVERSAL_CREDIT_REGULATION_34_PROGRAM_PATH = Path("regulations/uksi/2013/376/34.yaml")
UNIVERSAL_CREDIT_REGULATION_34_BASE = "uk:regulations/uksi/2013/376/34"
WELFARE_REFORM_ACT_SECTION_8_PROGRAM_PATH = Path("statutes/ukpga/2012/5/8.yaml")
WELFARE_REFORM_ACT_SECTION_8_BASE = "uk:statutes/ukpga/2012/5/8"
WELFARE_REFORM_ACT_SECTION_11_PROGRAM_PATH = Path("statutes/ukpga/2012/5/11.yaml")
WELFARE_REFORM_ACT_SECTION_11_BASE = "uk:statutes/ukpga/2012/5/11"

PERSONAL_ALLOWANCE_OUTPUTS = {
    "personal_allowance": {
        "axiom": f"{PERSONAL_ALLOWANCE_BASE}#personal_allowance",
        "pe": "personal_allowance",
    },
}

NATIONAL_INSURANCE_CLASS_1_OUTPUTS = {
    "main_primary_class_1_contribution": {
        "axiom": f"{NATIONAL_INSURANCE_SECTION_8_BASE}#main_primary_class_1_contribution",
        "pe": "ni_class_1_employee_primary",
        "pe_transform": "annual_to_weekly",
        "applies": ("ni_liable", True),
    },
    "additional_primary_class_1_contribution": {
        "axiom": f"{NATIONAL_INSURANCE_SECTION_8_BASE}#additional_primary_class_1_contribution",
        "pe": "ni_class_1_employee_additional",
        "pe_transform": "annual_to_weekly",
        "applies": ("ni_liable", True),
    },
    "primary_class_1_contribution": {
        "axiom": f"{NATIONAL_INSURANCE_SECTION_8_BASE}#primary_class_1_contribution",
        "pe": "ni_class_1_employee",
        "pe_transform": "annual_to_weekly",
    },
    "employee_national_insurance": {
        "axiom": f"{NATIONAL_INSURANCE_SECTION_8_BASE}#primary_class_1_contribution",
        "pe": "ni_employee",
        "pe_transform": "annual_to_weekly",
    },
}

INCOME_TAX_INCOME_BASE_COMPONENTS = (
    "employment_income",
    "private_pension_income",
    "social_security_income",
    "self_employment_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "miscellaneous_income",
)

INCOME_TAX_SECTION_23_ADDITION_COMPONENTS = (
    "income_tax_pre_charges",
    "CB_HITC",
    "personal_pension_contributions_tax",
)

INCOME_TAX_SECTION_23_REDUCTION_COMPONENTS = (
    "capped_mcad",
    "other_tax_credits",
)

INCOME_TAX_INCOME_BASE_OUTPUTS = {
    "total_income": {
        "axiom": f"{INCOME_TAX_SECTION_23_BASE}#total_income",
        "pe": "total_income",
    },
    "net_income": {
        "axiom": f"{INCOME_TAX_SECTION_23_BASE}#net_income",
        "pe": "adjusted_net_income",
        "applies": "income_tax_net_income_comparable",
    },
    "income_tax_liability": {
        "axiom": f"{INCOME_TAX_SECTION_23_BASE}#income_tax_liability",
        "pe": "income_tax",
    },
}

CHILD_BENEFIT_OUTPUTS = {
    "child_benefit_weekly_rate": {
        "axiom": f"{CHILD_BENEFIT_BASE}#child_benefit_weekly_rate",
        "pe": "child_benefit_respective_amount",
        "pe_transform": "annual_to_weekly",
    },
}

BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS = {
    "benefit_cap_relevant_amount": {
        "axiom": f"{BENEFIT_CAP_REGULATION_80A_BASE}#benefit_cap_relevant_amount",
        "pe": "benefit_cap",
        "pe_transform": "annual_to_monthly",
    },
}

STATE_PENSION_CREDIT_QUALIFYING_AGE_OUTPUTS = {
    "qualifying_age": {
        "axiom": f"{STATE_PENSION_CREDIT_SECTION_1_BASE}#qualifying_age",
        "pe": "state_pension_age",
    },
    "claimant_has_attained_qualifying_age": {
        "axiom": (
            f"{STATE_PENSION_CREDIT_SECTION_1_BASE}"
            "#claimant_has_attained_qualifying_age"
        ),
        "pe": "is_SP_age",
    },
}

PENSION_CREDIT_OUTPUTS = {
    "standard_minimum_guarantee": {
        "axiom": f"{PENSION_CREDIT_BASE}#standard_minimum_guarantee",
        "pe": "standard_minimum_guarantee",
        "pe_transform": "annual_to_weekly",
    },
    "severe_disability_additional_amount": {
        "axiom": f"{PENSION_CREDIT_BASE}#severe_disability_additional_amount",
        "pe": "severe_disability_minimum_guarantee_addition",
        "pe_transform": "annual_to_weekly",
    },
    "carer_additional_amount": {
        "axiom": f"{PENSION_CREDIT_BASE}#carer_additional_amount",
        "pe": "carer_minimum_guarantee_addition",
        "pe_transform": "annual_to_weekly_per_carer",
    },
}

UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS = {
    "standard_allowance_single_under_25": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_single_under_25_amount",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "SINGLE_YOUNG"),
    },
    "standard_allowance_single_25_or_over": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_single_25_or_over_amount",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "SINGLE_OLD"),
    },
    "standard_allowance_joint_both_under_25": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_joint_both_under_25_amount",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "COUPLE_YOUNG"),
    },
    "standard_allowance_joint_either_25_or_over": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#standard_allowance_joint_either_25_or_over_amount",
        "pe": "uc_standard_allowance",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_standard_allowance_claimant_type", "COUPLE_OLD"),
    },
}

UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS = {
    "child_element_first_child_or_qualifying_young_person": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#first_child_element_amount",
        "pe": "uc_individual_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_first_child_element",
    },
    "child_element_second_and_each_subsequent_child_or_qualifying_young_person": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#second_and_subsequent_child_element_amount",
        "pe": "uc_individual_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_subsequent_child_element",
    },
    "disabled_child_additional_amount_lower_rate": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#disabled_child_lower_rate_amount",
        "pe": "uc_individual_disabled_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
    "disabled_child_additional_amount_higher_rate": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#disabled_child_higher_rate_amount",
        "pe": "uc_individual_severely_disabled_child_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
}

UNIVERSAL_CREDIT_LCWRA_OUTPUTS = {
    "lcwra_element_standard_lcwra_claimant": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#lcwra_ordinary_claimant_amount",
        "pe": "uc_LCWRA_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_lcwra_standard_amount",
    },
    "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#lcwra_protected_claimant_amount",
        "pe": "uc_LCWRA_element",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_lcwra_higher_amount",
    },
}

UNIVERSAL_CREDIT_CARER_OUTPUTS = {
    "carer_element": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#carer_element_amount",
        "pe": "uc_carer_element",
        "pe_transform": "annual_to_monthly",
        "applies": "positive_pe_output",
    },
}

UNIVERSAL_CREDIT_CHILDCARE_OUTPUTS = {
    "childcare_costs_element_maximum_one_child": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#childcare_costs_element_one_child_maximum_amount",
        "pe": "uc_maximum_childcare_element_amount",
        "pe_transform": "annual_to_monthly",
        "applies": ("uc_childcare_element_eligible_children", 1),
    },
    "childcare_costs_element_maximum_two_or_more_children": {
        "axiom": f"{UNIVERSAL_CREDIT_BASE}#childcare_costs_element_two_or_more_children_maximum_amount",
        "pe": "uc_maximum_childcare_element_amount",
        "pe_transform": "annual_to_monthly",
        "applies": "uc_childcare_two_or_more_children",
    },
}

UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS = {
    "childcare_costs_element_amount": {
        "axiom": f"{UNIVERSAL_CREDIT_REGULATION_34_BASE}#childcare_costs_element_amount",
        "pe": "uc_childcare_element",
        "pe_transform": "annual_to_monthly",
    },
}

UNIVERSAL_CREDIT_AWARD_OUTPUTS = {
    "universal_credit_maximum_amount": {
        "axiom": f"{WELFARE_REFORM_ACT_SECTION_8_BASE}#universal_credit_maximum_amount",
        "pe": "uc_maximum_amount",
        "pe_transform": "annual_to_monthly",
    },
    "universal_credit_amounts_to_be_deducted": {
        "axiom": (
            f"{WELFARE_REFORM_ACT_SECTION_8_BASE}"
            "#universal_credit_amounts_to_be_deducted"
        ),
        "pe": "uc_income_reduction",
        "pe_transform": "annual_to_monthly",
    },
    "universal_credit_award_amount": {
        "axiom": f"{WELFARE_REFORM_ACT_SECTION_8_BASE}#universal_credit_award_amount",
        "pe": "universal_credit_pre_benefit_cap",
        "pe_transform": "annual_to_monthly",
    },
}

UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS = {
    "section_11_amount_for_accommodation_payments": {
        "axiom": (
            f"{WELFARE_REFORM_ACT_SECTION_11_BASE}"
            "#section_11_amount_for_accommodation_payments"
        ),
        "pe": "uc_housing_costs_element",
        "pe_transform": "annual_to_monthly",
    },
}

UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS = {
    "applicable_work_allowance_amount": {
        "axiom": (
            f"{UNIVERSAL_CREDIT_REGULATION_22_BASE}#applicable_work_allowance_amount"
        ),
        "pe": "uc_work_allowance",
        "pe_transform": "annual_to_monthly",
    },
}

UNIVERSAL_CREDIT_2026_RULESPEC_RATES = {
    "standard_allowance_single_under_25": 338.58,
    "standard_allowance_single_25_or_over": 424.90,
    "standard_allowance_joint_both_under_25": 528.34,
    "standard_allowance_joint_either_25_or_over": 666.97,
    "child_element_first_child_or_qualifying_young_person": 351.88,
    "child_element_second_and_each_subsequent_child_or_qualifying_young_person": 303.94,
    "disabled_child_additional_amount_lower_rate": 164.79,
    "disabled_child_additional_amount_higher_rate": 514.71,
    "lcwra_element_standard_lcwra_claimant": 217.26,
    "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant": 429.80,
    "carer_element": 209.34,
    "childcare_costs_element_maximum_one_child": 1071.09,
    "childcare_costs_element_maximum_two_or_more_children": 1836.16,
    "childcare_costs_element_reimbursement_rate": 0.85,
}


@dataclass(frozen=True)
class UKEFRSSurfaceSpec:
    program: Path
    entity: str
    outputs: dict[str, dict[str, Any]]
    pe_variables: tuple[str, ...]


SURFACE_SPECS = {
    "national-insurance-class-1": UKEFRSSurfaceSpec(
        program=NATIONAL_INSURANCE_SECTION_8_PROGRAM_PATH,
        entity="person",
        outputs=NATIONAL_INSURANCE_CLASS_1_OUTPUTS,
        pe_variables=(
            "ni_class_1_employee",
            "ni_class_1_employee_additional",
            "ni_class_1_employee_primary",
            "ni_class_1_income",
            "ni_employee",
            "ni_liable",
        ),
    ),
    "personal-allowance": UKEFRSSurfaceSpec(
        program=PERSONAL_ALLOWANCE_PROGRAM_PATH,
        entity="person",
        outputs=PERSONAL_ALLOWANCE_OUTPUTS,
        pe_variables=(
            "adjusted_net_income",
            "gift_aid_grossed_up",
            "personal_allowance",
        ),
    ),
    "income-tax-income-base": UKEFRSSurfaceSpec(
        program=INCOME_TAX_SECTION_23_PROGRAM_PATH,
        entity="person",
        outputs=INCOME_TAX_INCOME_BASE_OUTPUTS,
        pe_variables=(
            "adjusted_net_income",
            *INCOME_TAX_INCOME_BASE_COMPONENTS,
            *INCOME_TAX_SECTION_23_ADDITION_COMPONENTS,
            *INCOME_TAX_SECTION_23_REDUCTION_COMPONENTS,
            "income_tax",
            "total_income",
        ),
    ),
    "child-benefit": UKEFRSSurfaceSpec(
        program=CHILD_BENEFIT_PROGRAM_PATH,
        entity="person",
        outputs=CHILD_BENEFIT_OUTPUTS,
        pe_variables=(
            "child_benefit_child_index",
            "child_benefit_respective_amount",
        ),
    ),
    "benefit-cap-relevant-amount": UKEFRSSurfaceSpec(
        program=BENEFIT_CAP_REGULATION_80A_PROGRAM_PATH,
        entity="benunit",
        outputs=BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS,
        pe_variables=(
            "benefit_cap",
            "benunit_region",
            "num_adults",
            "num_children",
        ),
    ),
    "state-pension-credit-qualifying-age": UKEFRSSurfaceSpec(
        program=STATE_PENSION_CREDIT_SECTION_1_PROGRAM_PATH,
        entity="person",
        outputs=STATE_PENSION_CREDIT_QUALIFYING_AGE_OUTPUTS,
        pe_variables=(
            "age",
            "gender",
            "is_SP_age",
            "state_pension_age",
        ),
    ),
    "pension-credit": UKEFRSSurfaceSpec(
        program=PENSION_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=PENSION_CREDIT_OUTPUTS,
        pe_variables=(
            "carer_minimum_guarantee_addition",
            "is_couple",
            "num_carers",
            "relation_type",
            "severe_disability_minimum_guarantee_addition",
            "standard_minimum_guarantee",
        ),
    ),
    "universal-credit-standard-allowance": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_STANDARD_ALLOWANCE_OUTPUTS,
        pe_variables=(
            "uc_standard_allowance",
            "uc_standard_allowance_claimant_type",
        ),
    ),
    "universal-credit-child-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="person",
        outputs=UNIVERSAL_CREDIT_CHILD_ELEMENT_OUTPUTS,
        pe_variables=(
            "uc_child_index",
            "uc_individual_child_element",
            "uc_individual_disabled_child_element",
            "uc_individual_severely_disabled_child_element",
            "uc_is_child_born_before_child_limit",
        ),
    ),
    "universal-credit-lcwra-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_LCWRA_OUTPUTS,
        pe_variables=("uc_LCWRA_element",),
    ),
    "universal-credit-carer-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_CARER_OUTPUTS,
        pe_variables=("uc_carer_element",),
    ),
    "universal-credit-childcare-cap": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_CHILDCARE_OUTPUTS,
        pe_variables=(
            "uc_childcare_element_eligible_children",
            "uc_maximum_childcare_element_amount",
        ),
    ),
    "universal-credit-childcare-element": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_REGULATION_34_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS,
        pe_variables=(
            "uc_childcare_element",
            "uc_maximum_childcare_element_amount",
        ),
    ),
    "universal-credit-award": UKEFRSSurfaceSpec(
        program=WELFARE_REFORM_ACT_SECTION_8_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_AWARD_OUTPUTS,
        pe_variables=(
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
        ),
    ),
    "universal-credit-housing-costs": UKEFRSSurfaceSpec(
        program=WELFARE_REFORM_ACT_SECTION_11_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS,
        pe_variables=("uc_housing_costs_element",),
    ),
    "universal-credit-work-allowance": UKEFRSSurfaceSpec(
        program=UNIVERSAL_CREDIT_REGULATION_22_PROGRAM_PATH,
        entity="benunit",
        outputs=UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS,
        pe_variables=(
            "is_uc_work_allowance_eligible",
            "num_adults",
            "num_children",
            "uc_housing_costs_element",
            "uc_work_allowance",
        ),
    ),
}

UNIVERSAL_CREDIT_REGULATION_36_SURFACES = frozenset(
    surface
    for surface, spec in SURFACE_SPECS.items()
    if spec.program == UNIVERSAL_CREDIT_PROGRAM_PATH
)

SKIPPED_SURFACES: list[dict[str, str]] = []

ENTITY_WEIGHT_COLUMNS = {
    "person": "person_weight",
    "benunit": "benunit_weight",
    "household": "household_weight",
}

ENTITY_ID_COLUMNS = {
    "person": "person_id",
    "benunit": "benunit_id",
    "household": "household_id",
}


@dataclass(frozen=True)
class UKEFRSVariableSource:
    name: str
    entity: str
    domain: str
    path: str
    has_formula: bool
    has_aggregate: bool

    @property
    def computation_kind(self) -> str:
        if self.has_formula and self.has_aggregate:
            return "formula+aggregate"
        if self.has_formula:
            return "formula"
        if self.has_aggregate:
            return "aggregate"
        return "input"


@dataclass(frozen=True)
class UKEFRSVariableMetadata:
    name: str
    entity: str
    domain: str
    path: str
    computed: bool
    computation_kind: str
    covered_output: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity": self.entity,
            "domain": self.domain,
            "path": self.path,
            "computed": self.computed,
            "computation_kind": self.computation_kind,
            "covered_output": self.covered_output,
        }


@dataclass(frozen=True)
class UKEFRSVariableActivity:
    name: str
    entity: str
    nonzero_count: int
    finite_count: int
    nonfinite_count: int
    weighted_abs_total: float
    max_abs_value: float

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "entity": self.entity,
            "nonzero_count": self.nonzero_count,
            "finite_count": self.finite_count,
            "nonfinite_count": self.nonfinite_count,
            "weighted_abs_total": self.weighted_abs_total,
            "max_abs_value": self.max_abs_value,
        }


@dataclass(frozen=True)
class UKEFRSCoverageReport:
    policyengine_versions: dict[str, str]
    source_root: str | None
    domain_filter: str | None
    entity_filter: str | None
    variables_total: int
    computed_variables_total: int
    computed_variables_by_entity: dict[str, int]
    computed_variables_by_domain: dict[str, int]
    covered_output_variables: list[str]
    projection_support_variables: list[str]
    computed_covered_variables: list[str]
    missing_variables_total: int
    missing_variables: list[UKEFRSVariableMetadata]
    activity: list[UKEFRSVariableActivity]
    activity_errors: list[dict[str, str]]

    def to_json(self) -> dict[str, Any]:
        return {
            "policyengine_versions": self.policyengine_versions,
            "source_root": self.source_root,
            "domain_filter": self.domain_filter,
            "entity_filter": self.entity_filter,
            "variables_total": self.variables_total,
            "computed_variables_total": self.computed_variables_total,
            "computed_variables_by_entity": self.computed_variables_by_entity,
            "computed_variables_by_domain": self.computed_variables_by_domain,
            "covered_output_variables": self.covered_output_variables,
            "projection_support_variables": self.projection_support_variables,
            "computed_covered_count": len(self.computed_covered_variables),
            "computed_covered_variables": self.computed_covered_variables,
            "missing_variables_total": self.missing_variables_total,
            "missing_variables": [item.to_json() for item in self.missing_variables],
            "activity": [item.to_json() for item in self.activity],
            "activity_errors": self.activity_errors,
        }


@dataclass(frozen=True)
class UKEFRSComparisonRow:
    surface: str
    entity_id: str
    output: str
    axiom: float
    policyengine: float
    diff: float


@dataclass(frozen=True)
class UKEFRSOracleDivergence(UKEFRSComparisonRow):
    reason: str
    issue_url: str


@dataclass(frozen=True)
class UKEFRSComparisonReport:
    compared_persons: int
    compared_benunits: int
    compared_values: int
    mismatches: list[UKEFRSComparisonRow]
    oracle_divergences: list[UKEFRSOracleDivergence]
    output_summary: list[dict[str, Any]]
    skipped_surfaces: list[dict[str, str]]
    projection_notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "compared_persons": self.compared_persons,
            "compared_benunits": self.compared_benunits,
            "compared_values": self.compared_values,
            "mismatch_count": len(self.mismatches),
            "mismatches": [row.__dict__ for row in self.mismatches],
            "oracle_divergence_count": len(self.oracle_divergences),
            "oracle_divergences": [row.__dict__ for row in self.oracle_divergences],
            "output_summary": self.output_summary,
            "skipped_surfaces": self.skipped_surfaces,
            "projection_notes": self.projection_notes,
        }


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Workspace root containing rulespec-uk and axiom-rules-engine",
    )
    parser.add_argument(
        "--rulespec-root",
        type=Path,
        default=None,
        help="rulespec-uk checkout; defaults to <root>/rulespec-uk",
    )
    parser.add_argument(
        "--axiom-rules-engine-path",
        type=Path,
        default=None,
        help="axiom-rules-engine checkout; defaults to <root>/axiom-rules-engine",
    )
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Number of positive-weight EFRS people to compare; "
            "0 compares all eligible people"
        ),
    )
    parser.add_argument(
        "--person-id",
        type=int,
        action="append",
        default=None,
        dest="person_ids",
        help=(
            "Compare a specific EFRS person_id. Repeat to compare multiple "
            "known residual cases; when provided this bypasses --sample-size."
        ),
    )
    parser.add_argument(
        "--surface",
        choices=["all", *SURFACE_SPECS],
        default="all",
        help="UK EFRS surface to compare; defaults to all implemented surfaces",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "PolicyEngine UK dataset logical name, HuggingFace URI, or local .h5 "
            "path. Defaults to enhanced_frs_2023_24 and prefers local managed "
            "mirrors when available."
        ),
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(".axiom") / "policyengine-data",
        help="PolicyEngine dataset cache folder",
    )
    parser.add_argument(
        "--universal-credit-program",
        type=Path,
        default=None,
        help=(
            "Optional RuleSpec program to use for Universal Credit surfaces. "
            "This lets oracle runs compare a composed axiom-programs package "
            "while keeping non-UC UK surfaces on their source RuleSpec files."
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Absolute tolerance for matching PolicyEngine outputs",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=2e-7,
        help=(
            "Relative tolerance for large floating PolicyEngine intermediates; "
            "ordinary pound outputs remain controlled by --tolerance"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit nonzero when any compared value differs beyond tolerance",
    )


def main(args: argparse.Namespace) -> int:
    report = compare_uk_efrs(
        workspace_root=resolve_workspace_root(args.root),
        rulespec_root=args.rulespec_root,
        axiom_rules_path=args.axiom_rules_engine_path,
        year=args.year,
        sample_size=args.sample_size,
        surface=args.surface,
        dataset=args.dataset,
        data_folder=args.data_folder,
        tolerance=args.tolerance,
        relative_tolerance=args.relative_tolerance,
        universal_credit_program=getattr(args, "universal_credit_program", None),
        person_ids=tuple(args.person_ids or ()),
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print_report(
            report,
            tolerance=args.tolerance,
            relative_tolerance=args.relative_tolerance,
        )
    if args.fail_on_mismatch and report.mismatches:
        return 1
    return 0


def configure_coverage_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument(
        "--domain",
        default=None,
        help=(
            "Restrict computed-variable backlog to a PolicyEngine source domain "
            "such as gov, household, input, contrib, or misc"
        ),
    )
    parser.add_argument(
        "--entity",
        choices=sorted(ENTITY_WEIGHT_COLUMNS),
        default=None,
        help="Restrict computed-variable backlog to one PolicyEngine entity",
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=(
            "PolicyEngine UK dataset logical name, HuggingFace URI, or local .h5 "
            "path for --with-efrs-activity"
        ),
    )
    parser.add_argument(
        "--data-folder",
        type=Path,
        default=Path(".axiom") / "policyengine-data",
        help="PolicyEngine dataset cache folder",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=40,
        help="Maximum missing variables to print in text mode",
    )
    parser.add_argument(
        "--with-efrs-activity",
        action="store_true",
        help=(
            "Run PolicyEngine over EFRS for missing variables and rank by "
            "observed activity"
        ),
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")


def main_coverage(args: argparse.Namespace) -> int:
    report = build_uk_efrs_coverage_report(
        year=args.year,
        dataset=args.dataset,
        data_folder=args.data_folder,
        domain_filter=args.domain,
        entity_filter=args.entity,
        include_efrs_activity=args.with_efrs_activity,
    )
    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    else:
        print_uk_efrs_coverage_report(report, top=args.top)
    return 0


def compare_uk_efrs(
    *,
    workspace_root: Path,
    rulespec_root: Path | None,
    axiom_rules_path: Path | None,
    year: int,
    sample_size: int,
    surface: str,
    dataset: str,
    data_folder: Path,
    tolerance: float,
    relative_tolerance: float,
    universal_credit_program: Path | None = None,
    person_ids: tuple[int, ...] = (),
) -> UKEFRSComparisonReport:
    resolved_rulespec_root = (rulespec_root or workspace_root / "rulespec-uk").resolve()
    resolved_axiom_rules_path = (
        axiom_rules_path or workspace_root / "axiom-rules-engine"
    ).resolve()
    surfaces = list(SURFACE_SPECS) if surface == "all" else [surface]
    pe_data = load_policyengine_uk_data(
        year=year,
        sample_size=sample_size,
        dataset=dataset,
        data_folder=data_folder,
        person_ids=person_ids,
        person_variables=policyengine_person_variables_for_surfaces(surfaces),
        benunit_variables=policyengine_benunit_variables_for_surfaces(surfaces),
    )
    surface_results: dict[str, list[dict[str, Any]]] = {}
    for selected_surface in surfaces:
        spec = SURFACE_SPECS[selected_surface]
        if (
            selected_surface in UNIVERSAL_CREDIT_REGULATION_36_SURFACES
            and universal_credit_program is not None
        ):
            program = universal_credit_program.resolve()
        else:
            program = resolved_rulespec_root / spec.program
        if not program.exists():
            raise SystemExit(f"{selected_surface} RuleSpec not found: {program}")
        request = build_axiom_request(
            pe_data=pe_data,
            year=year,
            surface=selected_surface,
        )
        surface_results[selected_surface] = run_axiom_surface(
            program=program,
            request=request,
            rulespec_root=resolved_rulespec_root,
            axiom_rules_path=resolved_axiom_rules_path,
            surface=selected_surface,
        )
    return compare_outputs(
        pe_data=pe_data,
        axiom_outputs_by_surface=surface_results,
        tolerance=tolerance,
        relative_tolerance=relative_tolerance,
    )


def load_policyengine_uk_data(
    *,
    year: int,
    sample_size: int,
    dataset: str,
    data_folder: Path,
    person_ids: tuple[int, ...] = (),
    person_variables: tuple[str, ...] = SURFACE_SPECS[
        "personal-allowance"
    ].pe_variables,
    benunit_variables: tuple[str, ...] = (),
) -> dict[str, Any]:
    local_dataset = local_policyengine_uk_dataset_path(dataset)
    if local_dataset is not None:
        return load_local_policyengine_uk_data(
            local_path=local_dataset,
            year=year,
            sample_size=sample_size,
            person_ids=person_ids,
            person_variables=person_variables,
            benunit_variables=benunit_variables,
        )

    raise SystemExit(
        "uk-efrs-compare with current PolicyEngine UK requires a local .h5 "
        f"--dataset path. {policyengine_uk_install_message()}"
    )


def load_local_policyengine_uk_data(
    *,
    local_path: Path,
    year: int,
    sample_size: int,
    person_ids: tuple[int, ...],
    person_variables: tuple[str, ...],
    benunit_variables: tuple[str, ...],
) -> dict[str, Any]:
    require_policyengine_uk_versions()
    try:
        import pandas as pd
        from policyengine_uk import Microsimulation
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    log("Loading local PolicyEngine UK EFRS...")
    pe_dataset = UKSingleYearDataset(file_path=str(local_path))
    sim = Microsimulation(dataset=pe_dataset)

    with pd.HDFStore(local_path, mode="r") as store:
        raw_person = store["person"].copy()
        raw_benunit = store["benunit"].copy()
        household = store["household"].copy()

    person = add_policyengine_uk_person_weights(raw_person, household)
    person_columns = ["person_id", "person_weight"]
    if "person_benunit_id" in person.columns:
        person_columns.append("person_benunit_id")
    merged = person[person_columns].copy()
    log("Running PolicyEngine UK person outputs...")
    for variable in person_variables:
        merged[variable] = sim.calculate(
            variable,
            period=year,
            map_to="person",
        ).values
    records = table_records(merged)
    selected_indices = select_person_indices(
        records,
        sample_size=sample_size,
        person_ids=person_ids,
    )
    selected = [records[index] for index in selected_indices]
    selected_benunits: list[dict[str, Any]] = []
    if benunit_variables:
        benunit = add_policyengine_uk_benunit_weights(
            raw_benunit,
            raw_person,
            household,
        )
        merged_benunits = benunit[["benunit_id", "benunit_weight"]].copy()
        log("Running PolicyEngine UK benefit-unit outputs...")
        for variable in benunit_variables:
            merged_benunits[variable] = sim.calculate(
                variable,
                period=year,
                map_to="benunit",
            ).values
        benunit_records = table_records(merged_benunits)
        selected_benunit_ids = ()
        if person_ids:
            selected_benunit_ids = tuple(
                dict.fromkeys(
                    int(row_value(row, "person_benunit_id"))
                    for row in selected
                    if row_value(row, "person_benunit_id") is not None
                )
            )
        selected_benunit_indices = select_benunit_indices(
            benunit_records,
            sample_size=sample_size,
            benunit_ids=selected_benunit_ids,
        )
        selected_benunits = [
            benunit_records[index] for index in selected_benunit_indices
        ]
    return {
        "persons": selected,
        "person_ids": [int(row_value(row, "person_id")) for row in selected],
        "benunits": selected_benunits,
        "benunit_ids": [int(row_value(row, "benunit_id")) for row in selected_benunits],
    }


def add_policyengine_uk_person_weights(raw_person: Any, household: Any) -> Any:
    person = raw_person.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    return person.rename(columns={"household_weight": "person_weight"}).drop(
        columns=["household_id"],
    )


def add_policyengine_uk_benunit_weights(
    raw_benunit: Any,
    raw_person: Any,
    household: Any,
) -> Any:
    benunit_household_map = raw_person[
        ["person_benunit_id", "person_household_id"]
    ].drop_duplicates()
    benunit = raw_benunit.merge(
        benunit_household_map,
        left_on="benunit_id",
        right_on="person_benunit_id",
        how="left",
    )
    benunit = benunit.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    return benunit.rename(columns={"household_weight": "benunit_weight"}).drop(
        columns=[
            "person_benunit_id",
            "person_household_id",
            "household_id",
        ],
        errors="ignore",
    )


def resolve_policyengine_uk_dataset_reference(dataset: str) -> str:
    if "://" in dataset:
        return dataset
    try:
        from policyengine.provenance.manifest import resolve_dataset_reference
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc
    return resolve_dataset_reference("uk", dataset)


def local_policyengine_uk_dataset(
    *,
    dataset: str,
    year: int,
) -> Any | None:
    local_path = local_policyengine_uk_dataset_path(dataset)
    if local_path is None:
        return None
    try:
        import pandas as pd
        from microdf import MicroDataFrame
        from policyengine.tax_benefit_models.uk.datasets import (
            PolicyEngineUKDataset,
            UKYearData,
        )
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    with pd.HDFStore(local_path, mode="r") as store:
        raw_person = store["person"].copy()
        raw_benunit = store["benunit"].copy()
        household = store["household"].copy()

    person = raw_person.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    person = person.rename(columns={"household_weight": "person_weight"}).drop(
        columns=["household_id"]
    )
    benunit_household_map = person[
        ["person_benunit_id", "person_household_id"]
    ].drop_duplicates()
    benunit = raw_benunit.merge(
        benunit_household_map,
        left_on="benunit_id",
        right_on="person_benunit_id",
        how="left",
    )
    benunit = benunit.merge(
        household[["household_id", "household_weight"]],
        left_on="person_household_id",
        right_on="household_id",
        how="left",
    )
    benunit = benunit.rename(columns={"household_weight": "benunit_weight"}).drop(
        columns=[
            "person_benunit_id",
            "person_household_id",
            "household_id",
        ],
        errors="ignore",
    )

    dataset_id = Path(local_path).stem
    return PolicyEngineUKDataset(
        id=f"{dataset_id}_local_{year}",
        name=f"{dataset_id}-local-{year}",
        description=f"Local UK Dataset for year {year} based on {dataset_id}",
        filepath=str(Path(".axiom") / "policyengine-data" / f"{dataset_id}_{year}.h5"),
        year=int(year),
        data=UKYearData(
            person=MicroDataFrame(person, weights="person_weight"),
            benunit=MicroDataFrame(benunit, weights="benunit_weight"),
            household=MicroDataFrame(household, weights="household_weight"),
        ),
    )


def local_policyengine_uk_dataset_path(dataset: str) -> Path | None:
    direct_path = Path(dataset).expanduser()
    if direct_path.exists():
        return direct_path.resolve()
    try:
        from policyengine.provenance.manifest import (
            resolve_dataset_reference,
            resolve_local_managed_dataset_source,
        )
    except ImportError:
        return None

    resolved_dataset = dataset
    if "://" not in dataset:
        try:
            resolved_dataset = resolve_dataset_reference("uk", dataset)
        except ValueError:
            return None
    local_source = resolve_local_managed_dataset_source("uk", resolved_dataset)
    local_path = Path(local_source).expanduser()
    if local_path.exists():
        return local_path.resolve()
    return None


def policyengine_person_variables_for_surfaces(
    surfaces: list[str],
) -> tuple[str, ...]:
    return policyengine_variables_for_surfaces(surfaces, entity="person")


def policyengine_benunit_variables_for_surfaces(
    surfaces: list[str],
) -> tuple[str, ...]:
    return policyengine_variables_for_surfaces(surfaces, entity="benunit")


def policyengine_variables_for_surfaces(
    surfaces: list[str],
    *,
    entity: str,
) -> tuple[str, ...]:
    variables: set[str] = set()
    for surface in surfaces:
        spec = SURFACE_SPECS[surface]
        if spec.entity == entity:
            variables.update(spec.pe_variables)
    return tuple(sorted(variables))


def build_uk_efrs_coverage_report(
    *,
    year: int = 2026,
    dataset: str = DEFAULT_DATASET,
    data_folder: Path = Path(".axiom") / "policyengine-data",
    domain_filter: str | None = None,
    entity_filter: str | None = None,
    include_efrs_activity: bool = False,
    policyengine_variables: list[Any] | None = None,
    source_root: Path | None = None,
) -> UKEFRSCoverageReport:
    versions = policyengine_uk_versions()
    variables = discover_policyengine_uk_variables(
        policyengine_variables=policyengine_variables,
        source_root=source_root,
    )
    scoped_variables = [
        variable
        for variable in variables
        if (domain_filter is None or variable.domain == domain_filter)
        and (entity_filter is None or variable.entity == entity_filter)
    ]
    computed_variables = [
        variable for variable in scoped_variables if variable.computed
    ]
    covered_outputs = covered_uk_policyengine_output_variables()
    support_variables = uk_policyengine_projection_support_variables()
    computed_covered = sorted(
        variable.name
        for variable in computed_variables
        if variable.name in set(covered_outputs)
    )
    missing_variables = [
        variable
        for variable in computed_variables
        if variable.name not in set(covered_outputs)
    ]
    activity: list[UKEFRSVariableActivity] = []
    activity_errors: list[dict[str, str]] = []
    if include_efrs_activity and missing_variables:
        activity, activity_errors = policyengine_uk_efrs_activity(
            missing_variables,
            year=year,
            dataset=dataset,
            data_folder=data_folder,
        )

    source_root_label = None
    if source_root is not None:
        source_root_label = str(source_root.resolve())
    else:
        resolved_source_root = policyengine_uk_variables_source_root(required=False)
        if resolved_source_root is not None:
            source_root_label = str(resolved_source_root)

    return UKEFRSCoverageReport(
        policyengine_versions=versions,
        source_root=source_root_label,
        domain_filter=domain_filter,
        entity_filter=entity_filter,
        variables_total=len(scoped_variables),
        computed_variables_total=len(computed_variables),
        computed_variables_by_entity=dict(
            sorted(Counter(variable.entity for variable in computed_variables).items())
        ),
        computed_variables_by_domain=dict(
            sorted(Counter(variable.domain for variable in computed_variables).items())
        ),
        covered_output_variables=covered_outputs,
        projection_support_variables=support_variables,
        computed_covered_variables=computed_covered,
        missing_variables_total=len(missing_variables),
        missing_variables=sorted(
            missing_variables,
            key=lambda variable: (
                variable.domain,
                variable.entity,
                variable.path,
                variable.name,
            ),
        ),
        activity=sorted(
            activity,
            key=lambda item: (
                item.nonzero_count > 0,
                item.weighted_abs_total,
                item.nonzero_count,
            ),
            reverse=True,
        ),
        activity_errors=activity_errors,
    )


def discover_policyengine_uk_variables(
    *,
    policyengine_variables: list[Any] | None = None,
    source_root: Path | None = None,
) -> list[UKEFRSVariableMetadata]:
    raw_variables = (
        policyengine_variables
        if policyengine_variables is not None
        else load_policyengine_uk_variables()
    )
    resolved_source_root = source_root or policyengine_uk_variables_source_root()
    source_index = parse_policyengine_uk_variable_sources(resolved_source_root)
    covered_outputs = set(covered_uk_policyengine_output_variables())
    metadata: list[UKEFRSVariableMetadata] = []
    for raw_variable in raw_variables:
        name = str(variable_attribute(raw_variable, "name"))
        source = source_index.get(name)
        entity = normalize_policyengine_entity(
            variable_attribute(raw_variable, "entity", source.entity if source else "")
        )
        has_aggregate = bool(variable_attribute(raw_variable, "adds", None)) or bool(
            variable_attribute(raw_variable, "subtracts", None)
        )
        if source is not None:
            has_aggregate = has_aggregate or source.has_aggregate
            has_formula = source.has_formula
            domain = source.domain
            path = source.path
            computation_kind = source.computation_kind
        else:
            has_formula = False
            domain = "unknown"
            path = ""
            computation_kind = "aggregate" if has_aggregate else "input"
        computed = has_formula or has_aggregate
        metadata.append(
            UKEFRSVariableMetadata(
                name=name,
                entity=entity,
                domain=domain,
                path=path,
                computed=computed,
                computation_kind=computation_kind if computed else "input",
                covered_output=name in covered_outputs,
            )
        )
    return sorted(metadata, key=lambda variable: variable.name)


def load_policyengine_uk_variables() -> list[Any]:
    require_policyengine_uk_versions(command="uk-efrs-coverage")
    try:
        from policyengine_uk import CountryTaxBenefitSystem
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message("uk-efrs-coverage")) from exc
    return list(CountryTaxBenefitSystem().variables.values())


def parse_policyengine_uk_variable_sources(
    source_root: Path,
) -> dict[str, UKEFRSVariableSource]:
    sources: dict[str, UKEFRSVariableSource] = {}
    for source_file in sorted(source_root.rglob("*.py")):
        if source_file.name == "__init__.py":
            continue
        relative_path = source_file.relative_to(source_root)
        domain = relative_path.parts[0] if relative_path.parts else "unknown"
        try:
            tree = ast.parse(source_file.read_text())
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            if not class_extends_variable(node):
                continue
            entity = ""
            has_aggregate = False
            has_formula = any(
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name.startswith("formula")
                for item in node.body
            )
            for item in node.body:
                targets: list[ast.expr] = []
                value: ast.expr | None = None
                if isinstance(item, ast.Assign):
                    targets = list(item.targets)
                    value = item.value
                elif isinstance(item, ast.AnnAssign):
                    targets = [item.target]
                    value = item.value
                for target in targets:
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id == "entity" and value is not None:
                        entity = normalize_policyengine_entity(ast_value_name(value))
                    if target.id in {"adds", "subtracts"}:
                        has_aggregate = True
            sources[node.name] = UKEFRSVariableSource(
                name=node.name,
                entity=entity,
                domain=domain,
                path=relative_path.as_posix(),
                has_formula=has_formula,
                has_aggregate=has_aggregate,
            )
    return sources


def class_extends_variable(node: ast.ClassDef) -> bool:
    return any(ast_value_name(base) == "Variable" for base in node.bases)


def ast_value_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Subscript):
        return ast_value_name(node.value)
    return ""


def variable_attribute(raw_variable: Any, name: str, default: Any = None) -> Any:
    if isinstance(raw_variable, dict):
        return raw_variable.get(name, default)
    return getattr(raw_variable, name, default)


def normalize_policyengine_entity(value: Any) -> str:
    if hasattr(value, "key"):
        return str(value.key).strip().lower()
    text = str(value or "").strip()
    if "." in text:
        text = text.rsplit(".", 1)[-1]
    normalized = {
        "Person": "person",
        "BenUnit": "benunit",
        "BenefitUnit": "benunit",
        "Household": "household",
    }.get(text, text)
    return normalized.lower()


def covered_uk_policyengine_output_variables() -> list[str]:
    return sorted(
        {
            str(output["pe"])
            for spec in SURFACE_SPECS.values()
            for output in spec.outputs.values()
        }
    )


def uk_policyengine_projection_support_variables() -> list[str]:
    covered_outputs = set(covered_uk_policyengine_output_variables())
    return sorted(
        {
            variable
            for spec in SURFACE_SPECS.values()
            for variable in spec.pe_variables
            if variable not in covered_outputs
        }
    )


def policyengine_uk_variables_source_root(*, required: bool = True) -> Path | None:
    try:
        import policyengine_uk
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        if required:
            raise SystemExit(
                policyengine_uk_install_message("uk-efrs-coverage")
            ) from exc
        return None
    source_root = Path(policyengine_uk.__file__).resolve().parent / "variables"
    if not source_root.exists():
        if required:
            raise SystemExit(
                f"PolicyEngine UK variable source root not found: {source_root}"
            )
        return None
    return source_root


def policyengine_uk_versions() -> dict[str, str]:
    packages = ("policyengine", "policyengine-core", "policyengine-uk")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def policyengine_uk_efrs_activity(
    variables: list[UKEFRSVariableMetadata],
    *,
    year: int,
    dataset: str,
    data_folder: Path,
) -> tuple[list[UKEFRSVariableActivity], list[dict[str, str]]]:
    require_policyengine_uk_versions(command="uk-efrs-coverage")
    try:
        import numpy as np
        import pandas as pd
        from policyengine_uk import Microsimulation
        from policyengine_uk.data import UKSingleYearDataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message("uk-efrs-coverage")) from exc

    local_dataset = local_policyengine_uk_dataset_path(dataset)
    if local_dataset is None:
        raise SystemExit(
            "uk-efrs-coverage --with-efrs-activity with current PolicyEngine UK "
            f"requires a local .h5 --dataset path. "
            f"{policyengine_uk_install_message('uk-efrs-coverage')}"
        )

    log("Loading local PolicyEngine UK EFRS for coverage activity...")
    pe_dataset = UKSingleYearDataset(file_path=str(local_dataset))
    sim = Microsimulation(dataset=pe_dataset)
    with pd.HDFStore(local_dataset, mode="r") as store:
        raw_person = store["person"].copy()
        raw_benunit = store["benunit"].copy()
        household = store["household"].copy()
    raw_tables = {
        "person": add_policyengine_uk_person_weights(raw_person, household),
        "benunit": add_policyengine_uk_benunit_weights(
            raw_benunit,
            raw_person,
            household,
        ),
        "household": household,
    }

    extra_variables: dict[str, list[str]] = defaultdict(list)
    variables_by_name = {variable.name: variable for variable in variables}
    for variable in variables:
        if variable.entity in ENTITY_WEIGHT_COLUMNS:
            extra_variables[variable.entity].append(variable.name)

    log("Running PolicyEngine UK outputs for coverage activity...")

    activity: list[UKEFRSVariableActivity] = []
    errors: list[dict[str, str]] = []
    for entity, names in sorted(extra_variables.items()):
        raw_table = raw_tables[entity]
        weight_column = ENTITY_WEIGHT_COLUMNS[entity]
        id_column = ENTITY_ID_COLUMNS[entity]
        if weight_column not in raw_table.columns:
            errors.append(
                {
                    "entity": entity,
                    "variable": "*",
                    "error": f"missing weight column {weight_column}",
                }
            )
            continue
        if id_column not in raw_table.columns:
            errors.append(
                {
                    "entity": entity,
                    "variable": "*",
                    "error": f"missing id column {id_column}",
                }
            )
            continue
        output_table = raw_table[[id_column]].copy()
        for name in sorted(names):
            try:
                output_table[name] = sim.calculate(
                    name,
                    period=year,
                    map_to=entity,
                ).values
            except Exception as exc:  # pragma: no cover - depends on PE variable graph
                errors.append(
                    {
                        "entity": entity,
                        "variable": name,
                        "error": str(exc),
                    }
                )
        raw_weights = raw_table[[id_column, weight_column]].copy()
        output_ids = output_table[[id_column]].copy()
        aligned_weights = output_ids.merge(
            raw_weights,
            on=id_column,
            how="left",
            validate="one_to_one",
        )
        weights = pd.Series(
            pd.to_numeric(aligned_weights[weight_column], errors="coerce")
        )
        weight_values = weights.to_numpy(dtype=float, na_value=np.nan)
        weight_values = np.where(np.isfinite(weight_values), weight_values, 0.0)
        for name in sorted(names):
            if name not in output_table.columns:
                errors.append(
                    {
                        "entity": entity,
                        "variable": name,
                        "error": "missing PolicyEngine output column",
                    }
                )
                continue
            raw_values = pd.Series(pd.to_numeric(output_table[name], errors="coerce"))
            values = raw_values.to_numpy(dtype=float, na_value=np.nan)
            finite = np.isfinite(values)
            nonzero = finite & (np.abs(values) > 1e-9)
            weighted_abs_total = float(
                np.sum(np.abs(values[finite]) * weight_values[finite])
            )
            max_abs_value = float(np.max(np.abs(values[finite]))) if finite.any() else 0
            variable = variables_by_name[name]
            activity.append(
                UKEFRSVariableActivity(
                    name=variable.name,
                    entity=variable.entity,
                    nonzero_count=int(np.sum(nonzero)),
                    finite_count=int(np.sum(finite)),
                    nonfinite_count=int(len(values) - np.sum(finite)),
                    weighted_abs_total=weighted_abs_total,
                    max_abs_value=max_abs_value,
                )
            )
    return activity, errors


def print_uk_efrs_coverage_report(
    report: UKEFRSCoverageReport,
    *,
    top: int,
) -> None:
    print("PolicyEngine UK EFRS coverage")
    print(
        "PolicyEngine versions: "
        + ", ".join(
            f"{package}=={version}"
            for package, version in report.policyengine_versions.items()
        )
    )
    if report.source_root:
        print(f"Variable source root: {report.source_root}")
    if report.domain_filter or report.entity_filter:
        print(
            "Scope: "
            f"domain={report.domain_filter or 'all'}, "
            f"entity={report.entity_filter or 'all'}"
        )
    print(f"Variables in scope: {report.variables_total:,}")
    print(f"Computed/aggregate variables in scope: {report.computed_variables_total:,}")
    print(
        "Direct Axiom-covered PE outputs in scope: "
        f"{len(report.computed_covered_variables):,}"
    )
    print(f"Missing computed PE outputs in scope: {report.missing_variables_total:,}")
    print()
    print("Computed variables by entity:")
    for entity, count in report.computed_variables_by_entity.items():
        print(f"  - {entity}: {count:,}")
    print("Computed variables by source domain:")
    for domain, count in report.computed_variables_by_domain.items():
        print(f"  - {domain}: {count:,}")
    print()
    print("Currently direct-covered PE output variables:")
    for name in report.covered_output_variables:
        marker = "*" if name in report.computed_covered_variables else "-"
        print(f"  {marker} {name}")
    if report.projection_support_variables:
        print()
        print("PE variables used only to project current RuleSpec inputs:")
        for name in report.projection_support_variables:
            print(f"  - {name}")
    if report.activity:
        print()
        print(f"Top missing EFRS-active PE variables (first {top:,}):")
        activity_by_name = {item.name: item for item in report.activity}
        missing_by_name = {item.name: item for item in report.missing_variables}
        for activity in report.activity[:top]:
            variable = missing_by_name.get(activity.name)
            location = variable.path if variable else activity.entity
            print(
                f"  - {activity.name} ({location}): "
                f"nonzero={activity.nonzero_count:,}, "
                f"weighted_abs_total={activity.weighted_abs_total:.2f}, "
                f"max_abs={activity.max_abs_value:.2f}, "
                f"nonfinite={activity.nonfinite_count:,}"
            )
        inactive_count = sum(1 for item in report.activity if item.nonzero_count == 0)
        print(f"Missing variables with zero nonzero EFRS rows: {inactive_count:,}")
        if activity_by_name:
            print()
    print(f"Missing computed PE variables (first {top:,}):")
    for variable in report.missing_variables[:top]:
        print(
            f"  - {variable.name} "
            f"({variable.entity}, {variable.domain}, {variable.computation_kind}) "
            f"{variable.path}"
        )
    if report.activity_errors:
        print()
        print("Activity errors:")
        for error in report.activity_errors[:top]:
            print(
                f"  - {error.get('entity', '?')}:{error.get('variable', '?')}: "
                f"{error.get('error', '')}"
            )


def select_person_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    person_ids: tuple[int, ...] = (),
) -> list[int]:
    return select_entity_indices(
        rows,
        sample_size=sample_size,
        requested_ids=person_ids,
        id_column="person_id",
        weight_column="person_weight",
        entity_label="EFRS person_id",
    )


def select_benunit_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    benunit_ids: tuple[int, ...] = (),
) -> list[int]:
    return select_entity_indices(
        rows,
        sample_size=sample_size,
        requested_ids=benunit_ids,
        id_column="benunit_id",
        weight_column="benunit_weight",
        entity_label="EFRS benunit_id",
    )


def select_entity_indices(
    rows: list[dict[str, Any]],
    *,
    sample_size: int,
    requested_ids: tuple[int, ...] = (),
    id_column: str,
    weight_column: str,
    entity_label: str,
) -> list[int]:
    eligible = [
        index
        for index, row in enumerate(rows)
        if money(row_value(row, weight_column, 0)) > 0
    ]
    if not requested_ids:
        return eligible if sample_size <= 0 else eligible[:sample_size]

    requested_ids = tuple(dict.fromkeys(int(value) for value in requested_ids))
    index_by_id = {
        int(row_value(row, id_column)): index for index, row in enumerate(rows)
    }
    eligible_set = set(eligible)
    selected: list[int] = []
    missing: list[int] = []
    filtered: list[int] = []
    for person_id in requested_ids:
        index = index_by_id.get(person_id)
        if index is None:
            missing.append(person_id)
        elif index not in eligible_set:
            filtered.append(person_id)
        else:
            selected.append(index)
    if missing:
        raise SystemExit(
            f"Requested {entity_label} not found: "
            + ", ".join(str(value) for value in missing)
        )
    if filtered:
        raise SystemExit(
            f"Requested {entity_label} is not eligible for this comparison: "
            + ", ".join(str(value) for value in filtered)
        )
    return selected


def run_axiom_surface(
    *,
    program: Path,
    request: dict[str, Any],
    rulespec_root: Path,
    axiom_rules_path: Path,
    surface: str,
) -> list[dict[str, Any]]:
    if surface in UNIVERSAL_CREDIT_REGULATION_36_SURFACES:
        return run_axiom_parameter_outputs(
            program=program,
            request=request,
            rulespec_root=rulespec_root,
        )
    return run_axiom_program(
        program=program,
        request=request,
        rulespec_root=rulespec_root,
        axiom_rules_path=axiom_rules_path,
    )


def run_axiom_parameter_outputs(
    *,
    program: Path,
    request: dict[str, Any],
    rulespec_root: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    parameter_cache: dict[str, dict[str, float]] = {}
    for query in request.get("queries", []):
        period_start = str((query.get("period") or {}).get("start") or "")
        if period_start not in parameter_cache:
            parameter_cache[period_start] = rulespec_scalar_parameter_values(
                program,
                rulespec_root=rulespec_root,
                period_start=period_start,
            )
        parameter_values = parameter_cache[period_start]
        outputs: dict[str, dict[str, Any]] = {}
        for output in query.get("outputs", []):
            if output not in parameter_values:
                raise SystemExit(f"unknown RuleSpec scalar parameter: {output}")
            outputs[output] = {"value": {"value": str(parameter_values[output])}}
        results.append({"outputs": outputs})
    return results


def rulespec_scalar_parameter_values(
    program: Path, *, rulespec_root: Path, period_start: str
) -> dict[str, float]:
    return _rulespec_scalar_parameter_values(
        program.resolve(),
        rulespec_root=rulespec_root.resolve(),
        period_start=period_start,
        seen=set(),
    )


def _rulespec_scalar_parameter_values(
    program: Path,
    *,
    rulespec_root: Path,
    period_start: str,
    seen: set[Path],
) -> dict[str, float]:
    if program in seen:
        return {}
    seen.add(program)
    payload = yaml.safe_load(program.read_text()) or {}
    values: dict[str, float] = {}
    try:
        base = rule_base_from_program(program)
    except ValueError:
        base = None
    for rule in payload.get("rules") or []:
        if str(rule.get("kind") or "").strip() != "parameter":
            continue
        if base is None:
            continue
        version = effective_parameter_version(rule.get("versions") or [], period_start)
        if version is None:
            continue
        formula = str(version.get("formula") or "").strip().replace("_", "")
        try:
            value = float(formula)
        except ValueError:
            continue
        values[f"{base}#{rule['name']}"] = value
    for import_ref in payload.get("imports") or []:
        imported = resolve_rulespec_import(import_ref, rulespec_root=rulespec_root)
        if imported is None:
            continue
        values.update(
            _rulespec_scalar_parameter_values(
                imported,
                rulespec_root=rulespec_root,
                period_start=period_start,
                seen=seen,
            )
        )
    return values


def resolve_rulespec_import(import_ref: Any, *, rulespec_root: Path) -> Path | None:
    raw = str(import_ref or "").strip()
    if not raw:
        return None
    if ":" in raw:
        repo_prefix, path_ref = raw.split(":", 1)
        if repo_prefix != "uk":
            return None
    else:
        path_ref = raw
    candidate = rulespec_root / f"{path_ref}.yaml"
    if candidate.exists():
        return candidate.resolve()
    return None


def effective_parameter_version(
    versions: list[dict[str, Any]],
    period_start: str,
) -> dict[str, Any] | None:
    eligible = [
        version
        for version in versions
        if str(version.get("effective_from") or "") <= period_start
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda version: str(version.get("effective_from") or ""))


def rule_base_from_program(program: Path) -> str:
    parts = program.with_suffix("").parts
    if "regulations" in parts:
        index = parts.index("regulations")
        return "uk:" + "/".join(parts[index:])
    if "statutes" in parts:
        index = parts.index("statutes")
        return "uk:" + "/".join(parts[index:])
    raise ValueError(f"cannot infer UK RuleSpec base from {program}")


def build_axiom_request(
    *,
    pe_data: dict[str, Any],
    year: int,
    surface: str = "personal-allowance",
) -> dict[str, Any]:
    if surface == "national-insurance-class-1":
        return build_national_insurance_class_1_request(pe_data=pe_data, year=year)
    if surface == "personal-allowance":
        return build_personal_allowance_request(pe_data=pe_data, year=year)
    if surface == "income-tax-income-base":
        return build_income_tax_income_base_request(pe_data=pe_data, year=year)
    if surface == "child-benefit":
        return build_child_benefit_request(pe_data=pe_data, year=year)
    if surface == "benefit-cap-relevant-amount":
        return build_benefit_cap_relevant_amount_request(
            pe_data=pe_data,
            year=year,
        )
    if surface == "state-pension-credit-qualifying-age":
        return build_state_pension_credit_qualifying_age_request(
            pe_data=pe_data,
            year=year,
        )
    if surface == "pension-credit":
        return build_pension_credit_request(pe_data=pe_data, year=year)
    if surface == "universal-credit-childcare-element":
        return build_universal_credit_childcare_element_request(
            pe_data=pe_data,
            year=year,
        )
    if surface == "universal-credit-award":
        return build_universal_credit_award_request(pe_data=pe_data, year=year)
    if surface == "universal-credit-housing-costs":
        return build_universal_credit_housing_costs_request(
            pe_data=pe_data,
            year=year,
        )
    if surface == "universal-credit-work-allowance":
        return build_universal_credit_work_allowance_request(
            pe_data=pe_data,
            year=year,
        )
    if surface in UNIVERSAL_CREDIT_REGULATION_36_SURFACES:
        return build_universal_credit_request(
            pe_data=pe_data,
            year=year,
            surface=surface,
        )
    raise ValueError(f"unsupported UK EFRS surface: {surface}")


def build_national_insurance_class_1_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = tax_week_interval(year)
    parameters = policyengine_uk_class_1_weekly_parameters(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "national-insurance-class-1"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        projected = {
            "primary_class_1_contribution_payable_as_mentioned_in_section_6_1_a": bool(
                row_value(row, "ni_liable", True)
            ),
            "regulations_under_section_6_6_do_not_displace_calculation": True,
            "regulations_under_sections_116_to_120_do_not_displace_calculation": True,
            "earnings_paid_in_tax_week_in_respect_of_employment": money(
                row_value(row, "ni_class_1_income", 0)
            )
            / WEEKS_IN_YEAR,
            "current_primary_threshold_or_prescribed_equivalent": parameters[
                "primary_threshold"
            ],
            "current_upper_earnings_limit_or_prescribed_equivalent": parameters[
                "upper_earnings_limit"
            ],
        }
        for name, value in projected.items():
            inputs.append(
                input_record(
                    f"{NATIONAL_INSURANCE_SECTION_8_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in NATIONAL_INSURANCE_CLASS_1_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_personal_allowance_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = tax_year_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "personal-allowance"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        for name, value in project_personal_allowance_inputs(row).items():
            inputs.append(
                input_record(
                    f"{PERSONAL_ALLOWANCE_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in PERSONAL_ALLOWANCE_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_income_tax_income_base_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = tax_year_interval(year)
    inputs: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "income-tax-income-base"):
        person_id = int(row_value(row, "person_id"))
        entity_id = person_entity_id(person_id)
        for component in project_income_tax_income_base_components(row):
            payment_id = income_tax_component_entity_id(
                person_id,
                str(component["name"]),
            )
            relations.append(
                {
                    "name": f"{INCOME_TAX_SECTION_23_BASE}#relation.income_component_of_taxpayer",
                    "tuple": [payment_id, entity_id],
                    "interval": interval,
                }
            )
            inputs.append(
                input_record(
                    f"{INCOME_TAX_SECTION_23_BASE}#input.amount_charged_to_income_tax",
                    payment_id,
                    interval,
                    money(component["amount_charged_to_income_tax"]),
                )
            )
            inputs.append(
                input_record(
                    f"{INCOME_TAX_SECTION_23_BASE}#input.relief_deducted_under_section_24",
                    payment_id,
                    interval,
                    money(component["relief_deducted_under_section_24"]),
                )
            )
        for name, value in project_income_tax_section_23_inputs(row).items():
            inputs.append(
                input_record(
                    f"{INCOME_TAX_SECTION_23_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in INCOME_TAX_INCOME_BASE_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": relations},
        "queries": queries,
    }


def build_child_benefit_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_week_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "child-benefit"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        for name, value in project_child_benefit_inputs(row).items():
            inputs.append(
                input_record(
                    f"{CHILD_BENEFIT_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [spec["axiom"] for spec in CHILD_BENEFIT_OUTPUTS.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_benefit_cap_relevant_amount_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "benefit-cap-relevant-amount"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_benefit_cap_relevant_amount_inputs(row).items():
            inputs.append(
                input_record(
                    f"{BENEFIT_CAP_REGULATION_80A_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in BENEFIT_CAP_RELEVANT_AMOUNT_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_state_pension_credit_qualifying_age_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = day_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "state-pension-credit-qualifying-age"):
        entity_id = person_entity_id(int(row_value(row, "person_id")))
        for name, value in project_state_pension_credit_qualifying_age_inputs(
            row
        ).items():
            inputs.append(
                input_record(
                    f"{STATE_PENSION_CREDIT_SECTION_1_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in STATE_PENSION_CREDIT_QUALIFYING_AGE_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_pension_credit_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_week_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "pension-credit"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_pension_credit_inputs(row).items():
            inputs.append(
                input_record(
                    f"{PENSION_CREDIT_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [spec["axiom"] for spec in PENSION_CREDIT_OUTPUTS.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_universal_credit_request(
    *, pe_data: dict[str, Any], year: int, surface: str
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    spec = SURFACE_SPECS[surface]
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, surface):
        if spec.entity == "benunit":
            entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        else:
            entity_id = person_entity_id(int(row_value(row, "person_id")))
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [output["axiom"] for output in spec.outputs.values()],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": [], "relations": []},
        "queries": queries,
    }


def build_universal_credit_childcare_element_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "universal-credit-childcare-element"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_universal_credit_childcare_element_inputs(
            row
        ).items():
            inputs.append(
                input_record(
                    f"{UNIVERSAL_CREDIT_REGULATION_34_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in UNIVERSAL_CREDIT_CHILDCARE_ELEMENT_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_universal_credit_award_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "universal-credit-award"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_universal_credit_award_inputs(row).items():
            inputs.append(
                input_record(
                    f"{WELFARE_REFORM_ACT_SECTION_8_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"] for spec in UNIVERSAL_CREDIT_AWARD_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_universal_credit_housing_costs_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "universal-credit-housing-costs"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_universal_credit_housing_costs_inputs(row).items():
            inputs.append(
                input_record(
                    f"{WELFARE_REFORM_ACT_SECTION_11_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in UNIVERSAL_CREDIT_HOUSING_COSTS_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def build_universal_credit_work_allowance_request(
    *, pe_data: dict[str, Any], year: int
) -> dict[str, Any]:
    interval = benefit_month_interval(year)
    inputs: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    for row in rows_for_surface(pe_data, "universal-credit-work-allowance"):
        entity_id = benunit_entity_id(int(row_value(row, "benunit_id")))
        for name, value in project_universal_credit_work_allowance_inputs(row).items():
            inputs.append(
                input_record(
                    f"{UNIVERSAL_CREDIT_REGULATION_22_BASE}#input.{name}",
                    entity_id,
                    interval,
                    value,
                )
            )
        queries.append(
            {
                "entity_id": entity_id,
                "period": interval,
                "outputs": [
                    spec["axiom"]
                    for spec in UNIVERSAL_CREDIT_WORK_ALLOWANCE_OUTPUTS.values()
                ],
            }
        )

    return {
        "mode": "explain",
        "dataset": {"inputs": inputs, "relations": []},
        "queries": queries,
    }


def project_personal_allowance_inputs(row: Any) -> dict[str, Any]:
    adjusted_net_income = money(row_value(row, "adjusted_net_income"))
    gift_aid_grossed_up = money(row_value(row, "gift_aid_grossed_up", 0))
    return {
        "individual_makes_claim": True,
        "individual_meets_requirements_under_section_56": True,
        "adjusted_net_income": max(0.0, adjusted_net_income - gift_aid_grossed_up),
    }


def project_income_tax_income_base_components(row: Any) -> list[dict[str, Any]]:
    components = [
        {
            "name": name,
            "amount_charged_to_income_tax": money(row_value(row, name, 0)),
            "relief_deducted_under_section_24": 0.0,
        }
        for name in INCOME_TAX_INCOME_BASE_COMPONENTS
    ]
    nonzero_components = [
        component
        for component in components
        if money(component["amount_charged_to_income_tax"])
    ]
    if not nonzero_components:
        return []
    total_income = sum(
        money(component["amount_charged_to_income_tax"])
        for component in nonzero_components
    )
    adjusted_net_income = money(row_value(row, "adjusted_net_income", total_income))
    nonzero_components[0]["relief_deducted_under_section_24"] = max(
        0.0,
        total_income - adjusted_net_income,
    )
    return nonzero_components


def project_income_tax_section_23_inputs(row: Any) -> dict[str, Any]:
    additions = sum(
        money(row_value(row, name, 0))
        for name in INCOME_TAX_SECTION_23_ADDITION_COMPONENTS
    )
    reductions = sum(
        money(row_value(row, name, 0))
        for name in INCOME_TAX_SECTION_23_REDUCTION_COMPONENTS
    )
    return {
        "net_income_taken_as_zero_under_section_24b": False,
        "tax_calculated_at_applicable_rates_on_income_remaining_after_allowances": additions,
        "tax_reductions_listed_in_section_26": reductions,
        "additional_tax_amounts_listed_in_section_30": 0.0,
    }


def project_child_benefit_inputs(row: Any) -> dict[str, Any]:
    child_index = int(row_value(row, "child_benefit_child_index", -1))
    is_eldest = child_index == 1
    return {
        "during_subsistence_of_marriage_any_party_married_to_more_than_one_person": False,
        "marriage_ceremony_took_place_under_law_permitting_polygamy": False,
        "specified_benefit_allowance_or_increase_paid_for_week_to_person": False,
        "specified_benefit_is_in_respect_of_only_elder_or_eldest_child_for_child_benefit_entitlement": False,
        "child_or_qualifying_young_person_is_only_elder_or_eldest_for_payee": is_eldest,
        "paragraph_2_relationship_coordination_applies": False,
        "child_or_qualifying_young_person_is_elder_or_eldest_among_paragraph_2_children": is_eldest,
        "payee_is_voluntary_organisation": False,
        "payee_resides_with_parent_otherwise_than_paragraph_2_a": False,
    }


def project_benefit_cap_relevant_amount_inputs(row: Any) -> dict[str, Any]:
    num_adults = int(money(row_value(row, "num_adults", 0)))
    num_children = int(money(row_value(row, "num_children", 0)))
    in_london = enum_name(row_value(row, "benunit_region", "")).upper() == "LONDON"
    return {
        "claim_is_for_joint_claimants": num_adults > 1,
        "responsible_for_child_or_qualifying_young_person": num_children > 0,
        "award_contains_housing_costs_element": False,
        "accommodation_in_respect_of_which_claimant_meets_occupation_condition_is_in_greater_london": False,
        "claimant_receives_housing_benefit_for_dwelling_in_greater_london": False,
        "claimant_has_accommodation_normally_occupied_as_home": True,
        "accommodation_normally_occupied_as_home_is_in_greater_london": in_london,
        "jobcentre_plus_office_allocated_to_claim_is_in_greater_london": False,
    }


def project_state_pension_credit_qualifying_age_inputs(row: Any) -> dict[str, Any]:
    state_pension_age = money(row_value(row, "state_pension_age", 0))
    return {
        "claimant_is_woman": enum_name(row_value(row, "gender", "")).upper()
        == "FEMALE",
        "pensionable_age": state_pension_age,
        "pensionable_age_for_woman_born_same_day": state_pension_age,
        "claimant_age": money(row_value(row, "age", 0)),
    }


def project_universal_credit_award_inputs(row: Any) -> dict[str, Any]:
    is_uc_eligible = bool(row_value(row, "is_uc_eligible", False))

    def monthly(name: str) -> float:
        return money(row_value(row, name, 0)) / MONTHS_IN_YEAR

    def eligible_monthly(name: str) -> float:
        return monthly(name) if is_uc_eligible else 0.0

    return {
        "amount_included_under_section_9_standard_allowance": eligible_monthly(
            "uc_standard_allowance"
        ),
        "amount_included_under_section_10_responsibility_for_children_and_young_persons": eligible_monthly(
            "uc_child_element"
        ),
        "amount_included_under_section_11_housing_costs": eligible_monthly(
            "uc_housing_costs_element"
        ),
        "amount_included_under_section_12_other_particular_needs_or_circumstances": (
            eligible_monthly("uc_disability_elements")
            + eligible_monthly("uc_childcare_element")
            + eligible_monthly("uc_carer_element")
        ),
        "earned_income_deduction_calculated_in_prescribed_manner": eligible_monthly(
            "uc_income_reduction"
        ),
        "unearned_income_deduction_calculated_in_prescribed_manner": 0.0,
    }


def project_universal_credit_childcare_element_inputs(row: Any) -> dict[str, Any]:
    monthly_childcare_element = money(row_value(row, "uc_childcare_element", 0)) / (
        MONTHS_IN_YEAR
    )
    monthly_childcare_maximum = (
        money(row_value(row, "uc_maximum_childcare_element_amount", 0)) / MONTHS_IN_YEAR
    )
    relevant_charges = (
        monthly_childcare_element
        / UNIVERSAL_CREDIT_2026_RULESPEC_RATES[
            "childcare_costs_element_reimbursement_rate"
        ]
        if monthly_childcare_element
        else 0.0
    )
    return {
        "charges_paid_for_relevant_childcare_attributable_to_assessment_period": relevant_charges,
        "amount_considered_excessive_having_regard_to_paid_work_extent": 0.0,
        "amount_met_or_reimbursed_by_employer_or_some_other_person": 0.0,
        "secretary_of_state_work_transition_childcare_payment_meets_non_other_relevant_support_conditions": False,
        "amount_from_funds_provided_by_secretary_of_state_or_scottish_or_welsh_ministers_for_work_related_activity_or_training": 0.0,
        "secretary_of_state_work_transition_childcare_payment_amount": 0.0,
        "maximum_amount_specified_in_table_in_regulation_36": max(
            0.0,
            monthly_childcare_maximum,
        ),
    }


def project_universal_credit_housing_costs_inputs(row: Any) -> dict[str, Any]:
    monthly_housing_costs = money(row_value(row, "uc_housing_costs_element", 0)) / (
        MONTHS_IN_YEAR
    )
    return {
        "payments_are_in_respect_of_accommodation_for_section_11": True,
        "accommodation_is_in_great_britain": True,
        "accommodation_is_residential_accommodation": True,
        "claimant_is_liable_to_make_accommodation_payments": True,
        "claimant_is_treated_as_liable_to_make_accommodation_payments_by_regulations": False,
        "claimant_is_treated_as_not_liable_to_make_accommodation_payments_by_regulations": False,
        "claimant_occupies_accommodation_as_home": True,
        "claimant_is_treated_as_occupying_accommodation_as_home_by_regulations": False,
        "claimant_is_treated_as_not_occupying_accommodation_as_home_by_regulations": False,
        "section_11_exception_to_accommodation_amount_applies_by_regulations": False,
        "section_11_inclusion_has_ended_at_prescribed_time_by_regulations": False,
        "section_11_inclusion_has_not_started_until_prescribed_time_by_regulations": False,
        "amount_determined_or_calculated_by_regulations_under_section_11": monthly_housing_costs,
    }


def project_universal_credit_work_allowance_inputs(row: Any) -> dict[str, Any]:
    num_adults = int(money(row_value(row, "num_adults", 0)))
    num_children = int(money(row_value(row, "num_children", 0)))
    joint_claim = num_adults > 1
    has_child = num_children > 0
    eligible = bool(row_value(row, "is_uc_work_allowance_eligible", False))
    has_limited_capability = eligible and not has_child
    return {
        "claim_is_for_joint_claimants": joint_claim,
        "claimant_is_member_of_couple": False,
        "claimant_makes_claim_as_single_person": False,
        "joint_claimants_responsible_for_child_or_qualifying_young_person": joint_claim
        and has_child,
        "one_or_both_joint_claimants_have_limited_capability_for_work": joint_claim
        and has_limited_capability,
        "single_claimant_responsible_for_child_or_qualifying_young_person": (
            not joint_claim and has_child
        ),
        "single_claimant_has_limited_capability_for_work": (
            not joint_claim and has_limited_capability
        ),
        "award_contains_housing_costs_element": money(
            row_value(row, "uc_housing_costs_element", 0)
        )
        > 0,
    }


def project_pension_credit_inputs(row: Any) -> dict[str, Any]:
    relation_type = str(row_value(row, "relation_type", "")).upper()
    is_couple = bool(row_value(row, "is_couple", False)) or relation_type == "COUPLE"
    severe_disability_addition = (
        money(row_value(row, "severe_disability_minimum_guarantee_addition", 0))
        / WEEKS_IN_YEAR
    )
    num_carers = int(money(row_value(row, "num_carers", 0)))
    return {
        "claimant_is_prisoner": False,
        "member_of_religious_order_fully_maintained_by_order": False,
        "claimant_has_partner": is_couple,
        "treated_as_severely_disabled_person_under_schedule_i_part_i_paragraph_1": severe_disability_addition
        > 0,
        "severe_disability_couple_rate_conditions_satisfied": severe_disability_addition
        > 100,
        "paragraph_4_of_part_ii_of_schedule_i_satisfied_for_this_partner": num_carers
        > 0,
    }


def rows_for_surface(pe_data: dict[str, Any], surface: str) -> list[dict[str, Any]]:
    persons = pe_data["persons"]
    if surface == "child-benefit":
        return [
            row
            for row in persons
            if money(row_value(row, "child_benefit_respective_amount", 0)) > 0
        ]
    benunits = pe_data.get("benunits", [])
    if surface == "benefit-cap-relevant-amount":
        return [
            row
            for row in benunits
            if math.isfinite(money(row_value(row, "benefit_cap", math.inf)))
        ]
    if surface == "universal-credit-child-element":
        return [
            row
            for row in persons
            if money(row_value(row, "uc_individual_child_element", 0)) > 0
            or money(row_value(row, "uc_individual_disabled_child_element", 0)) > 0
            or money(row_value(row, "uc_individual_severely_disabled_child_element", 0))
            > 0
        ]
    if surface == "universal-credit-standard-allowance":
        return [
            row
            for row in benunits
            if money(row_value(row, "uc_standard_allowance", 0)) > 0
        ]
    if surface == "universal-credit-lcwra-element":
        return [
            row for row in benunits if money(row_value(row, "uc_LCWRA_element", 0)) > 0
        ]
    if surface == "universal-credit-carer-element":
        return [
            row for row in benunits if money(row_value(row, "uc_carer_element", 0)) > 0
        ]
    if surface == "universal-credit-childcare-cap":
        return [
            row
            for row in benunits
            if money(row_value(row, "uc_maximum_childcare_element_amount", 0)) > 0
        ]
    if SURFACE_SPECS[surface].entity == "benunit":
        return benunits
    return persons


def compare_outputs(
    *,
    pe_data: dict[str, Any],
    axiom_outputs_by_surface: dict[str, list[dict[str, Any]]],
    tolerance: float,
    relative_tolerance: float,
) -> UKEFRSComparisonReport:
    mismatches: list[UKEFRSComparisonRow] = []
    oracle_divergences: list[UKEFRSOracleDivergence] = []
    summary: dict[str, dict[str, Any]] = {
        f"{surface}:{name}": {
            "surface": surface,
            "output": name,
            "compared": 0,
            "mismatches": 0,
            "oracle_divergences": 0,
            "max_abs_diff": 0.0,
            "max_relative_diff": 0.0,
        }
        for surface, spec in SURFACE_SPECS.items()
        if surface in axiom_outputs_by_surface
        for name in spec.outputs
    }
    compared_values = 0
    for surface, axiom_outputs in axiom_outputs_by_surface.items():
        output_specs = SURFACE_SPECS[surface].outputs
        persons = rows_for_surface(pe_data, surface)
        if len(axiom_outputs) != len(persons):
            raise ValueError(
                f"{surface} produced {len(axiom_outputs):,} Axiom result rows "
                f"for {len(persons):,} PolicyEngine rows"
            )
        for index, result in enumerate(axiom_outputs):
            pe_row = persons[index]
            entity_id = entity_id_for_surface(surface, pe_row)
            outputs = result.get("outputs") or {}
            for name, spec in output_specs.items():
                if not output_applies(spec, pe_row):
                    continue
                axiom_value = output_number(outputs.get(spec["axiom"]))
                pe_value = policyengine_output_value(spec, pe_row)
                diff = axiom_value - pe_value
                abs_diff = abs(diff)
                compared_values += 1
                summary_key = f"{surface}:{name}"
                summary[summary_key]["compared"] += 1
                summary[summary_key]["max_abs_diff"] = max(
                    summary[summary_key]["max_abs_diff"], abs_diff
                )
                summary[summary_key]["max_relative_diff"] = max(
                    summary[summary_key]["max_relative_diff"],
                    relative_diff(axiom_value, pe_value),
                )
                if not within_tolerance(
                    axiom_value,
                    pe_value,
                    absolute_tolerance=tolerance,
                    relative_tolerance=relative_tolerance,
                ):
                    divergence = known_policyengine_divergence(
                        surface=surface,
                        output=name,
                        entity_id=entity_id,
                        axiom_value=axiom_value,
                        policyengine_value=pe_value,
                        diff=diff,
                    )
                    if divergence is not None:
                        summary[summary_key]["oracle_divergences"] += 1
                        oracle_divergences.append(divergence)
                    else:
                        summary[summary_key]["mismatches"] += 1
                        mismatches.append(
                            UKEFRSComparisonRow(
                                surface=surface,
                                entity_id=entity_id,
                                output=name,
                                axiom=axiom_value,
                                policyengine=pe_value,
                                diff=diff,
                            )
                        )
    return UKEFRSComparisonReport(
        compared_persons=len(pe_data["person_ids"]),
        compared_benunits=len(pe_data.get("benunit_ids", [])),
        compared_values=compared_values,
        mismatches=mismatches,
        oracle_divergences=oracle_divergences,
        output_summary=list(summary.values()),
        skipped_surfaces=SKIPPED_SURFACES,
        projection_notes=[
            "Personal allowance projection supplies EFRS adjusted net income "
            "net of PolicyEngine's gift_aid_grossed_up taper adjustment, because "
            "PolicyEngine UK applies that subtraction inside its personal "
            "allowance formula.",
            "The current projection treats EFRS people as making a claim and "
            "meeting the Section 56 residence/citizenship-condition boundary "
            "facts, matching the usual PolicyEngine UK EFRS personal allowance "
            "surface until those upstream legal predicates are encoded.",
            "National Insurance Class 1 comparison projects annual PolicyEngine "
            "NI Class 1 income into a representative tax week, supplies the "
            "PolicyEngine weekly primary threshold and upper earnings limit, "
            "compares RuleSpec's weekly section 8 aggregate output against "
            "PolicyEngine's annual ni_class_1_employee divided by 52, and "
            "also compares the same aggregate against PolicyEngine's "
            "ni_employee wrapper because PE-UK defines ni_employee as an "
            "aggregate containing only ni_class_1_employee. It "
            "compares the main/additional component outputs on ni_liable rows "
            "because PolicyEngine's component formulas are not masked by that "
            "liability predicate.",
            "Child Benefit comparison filters to positive PolicyEngine "
            "child_benefit_respective_amount rows, divides that annualized "
            "PolicyEngine output by 52 to compare against the RuleSpec weekly "
            "rate, and projects the eldest-child branch from "
            "child_benefit_child_index.",
            "Child Benefit relationship-coordination, voluntary-organisation, "
            "specified-benefit, and polygamous-marriage branches are projected "
            "false because PolicyEngine UK's child_benefit_respective_amount "
            "does not expose those legal predicates separately.",
            "Universal Credit Regulation 80A benefit-cap relevant-amount "
            "comparison filters to finite PolicyEngine benefit_cap rows, "
            "divides the annual PE cap by 12, and projects the annual-limit "
            "case from PolicyEngine's num_adults, num_children, and "
            "benunit_region. PolicyEngine uses infinity for exempt benunits, "
            "which are outside the finite relevant-amount comparison.",
            "Pension Credit standard minimum guarantee comparison runs at "
            "benefit-unit level, divides PolicyEngine's annual output by 52, "
            "and projects claimant_has_partner from PolicyEngine's relation_type "
            "or is_couple. Prisoner and fully-maintained religious-order branches "
            "are projected false because those legal predicates are not exposed "
            "in the EFRS oracle data.",
            "Pension Credit carer additions compare RuleSpec's per-partner "
            "amount against PolicyEngine's annual aggregate carer addition "
            "divided by num_carers and 52. The EFRS oracle has no positive "
            "severe-disability addition rows, so that branch is currently a "
            "zero-row guard rather than a positive-eligibility validation.",
            "State Pension Credit Act section 1 qualifying-age comparison "
            "queries RuleSpec's day-level qualifying_age on a representative "
            "day and supplies PolicyEngine's annual state_pension_age for both "
            "the pensionable-age leaf and the woman-born-same-day leaf. The "
            "same projection compares the attained-age judgment against "
            "PolicyEngine's is_SP_age boolean. Current PolicyEngine UK EFRS "
            "data exposes the modern equalized-age surface rather than "
            "historical sex-specific age transitions.",
            "Universal Credit Regulation 36 comparisons treat the generated "
            "RuleSpec outputs as component table amounts. PolicyEngine annual "
            "EFRS component outputs are divided by 12, and EFRS category "
            "variables select the matching standard-allowance, child-element, "
            "carer, LCWRA, and childcare-cap rows.",
            "Welfare Reform Act 2012 section 8 Universal Credit award "
            "comparison projects annual PolicyEngine component outputs into "
            "monthly section 8 maximum-amount buckets, gates those buckets by "
            "PolicyEngine's is_uc_eligible predicate to match uc_maximum_amount, "
            "and projects PolicyEngine's capped uc_income_reduction as the "
            "prescribed income deduction because PolicyEngine exposes the final "
            "deduction rather than the exact statutory earned/unearned split.",
            "Welfare Reform Act 2012 section 11 Universal Credit housing-costs "
            "comparison projects PolicyEngine's annual uc_housing_costs_element "
            "into the monthly amount determined or calculated by regulations, "
            "with ordinary Great Britain residential-accommodation eligibility "
            "predicates projected true and exception/timing predicates false.",
            "Universal Credit Regulation 22 work-allowance comparison projects "
            "PolicyEngine's is_uc_work_allowance_eligible, num_adults, "
            "num_children, and uc_housing_costs_element into the statutory "
            "higher/lower work-allowance case split, then compares the monthly "
            "RuleSpec allowance against PolicyEngine's annual uc_work_allowance "
            "divided by 12.",
            "Universal Credit Regulation 34 childcare-costs element comparison "
            "projects PolicyEngine's annual uc_childcare_element into monthly "
            "relevant childcare charges by reversing the statutory 85 percent "
            "reimbursement rate, supplies PolicyEngine's annual "
            "uc_maximum_childcare_element_amount divided by 12 as the regulation "
            "36 maximum, and projects excluded, reimbursed, and other-support "
            "amounts to zero.",
            "The protected LCWRA Regulation 36 table amount is compared only "
            "when PolicyEngine exposes the raw protected amount. Current "
            "PolicyEngine UK EFRS outputs apply the July 2025 UC rebalancing "
            "scenario for existing claimants, so those rebalanced health-element "
            "values are not treated as the same surface.",
            "Income Tax Act 2007 section 23 net-income comparison projects "
            "PolicyEngine UK's adjusted_net_income into section 24 reliefs only "
            "for rows with non-negative income components where adjusted net "
            "income does not exceed total income. Rows with loss semantics are "
            "skipped because RuleSpec section 23 net_income cannot exceed the "
            "section 23 total_income output.",
            "Income Tax Act 2007 section 23 final-liability comparison collapses "
            "PolicyEngine's income_tax_additions into the RuleSpec step-4 tax "
            "input and PolicyEngine's income_tax_subtractions into the section "
            "26 reductions input, because PolicyEngine exposes final income_tax "
            "and aggregate additive/subtractive components rather than exact "
            "section 23 step buckets.",
        ],
    )


def entity_id_for_surface(surface: str, row: Any) -> str:
    if SURFACE_SPECS[surface].entity == "benunit":
        return benunit_entity_id(int(row_value(row, "benunit_id")))
    return person_entity_id(int(row_value(row, "person_id")))


def policyengine_output_value(spec: dict[str, Any], row: Any) -> float:
    raw_value = money(row_value(row, spec["pe"]))
    if spec.get("pe_transform") == "annual_to_weekly":
        return raw_value / WEEKS_IN_YEAR
    if spec.get("pe_transform") == "annual_to_weekly_per_carer":
        return (
            raw_value
            / WEEKS_IN_YEAR
            / max(1, int(money(row_value(row, "num_carers", 0))))
        )
    if spec.get("pe_transform") == "annual_to_monthly":
        return raw_value / MONTHS_IN_YEAR
    return raw_value


def output_applies(spec: dict[str, Any], row: Any) -> bool:
    applies = spec.get("applies")
    if applies is None:
        return True
    if applies == "positive_pe_output":
        return policyengine_output_value(spec, row) > 0
    if applies == "income_tax_net_income_comparable":
        return all(
            money(row_value(row, name, 0)) >= 0
            for name in INCOME_TAX_INCOME_BASE_COMPONENTS
        ) and money(row_value(row, "adjusted_net_income", 0)) <= money(
            row_value(row, "total_income", 0)
        )
    if applies == "uc_first_child_element":
        return (
            policyengine_output_value(spec, row) > 0
            and int(row_value(row, "uc_child_index", -1)) == 1
            and bool(row_value(row, "uc_is_child_born_before_child_limit", False))
        )
    if applies == "uc_subsequent_child_element":
        return policyengine_output_value(spec, row) > 0 and not output_applies(
            {**spec, "applies": "uc_first_child_element"},
            row,
        )
    if applies == "uc_lcwra_standard_amount":
        monthly_value = policyengine_output_value(spec, row)
        return 0 < monthly_value < 300
    if applies == "uc_lcwra_higher_amount":
        monthly_value = policyengine_output_value(spec, row)
        expected = UNIVERSAL_CREDIT_2026_RULESPEC_RATES[
            "lcwra_element_pre_2026_severe_conditions_or_terminally_ill_claimant"
        ]
        return math.isclose(monthly_value, expected, abs_tol=0.01)
    if applies == "uc_childcare_two_or_more_children":
        return int(row_value(row, "uc_childcare_element_eligible_children", 0)) >= 2
    if isinstance(applies, tuple) and len(applies) == 2:
        name, expected = applies
        value = row_value(row, name)
        if isinstance(expected, str):
            return enum_name(value) == expected
        return value == expected
    raise ValueError(f"unsupported output applicability rule: {applies!r}")


def enum_name(value: Any) -> str:
    if hasattr(value, "name"):
        return str(value.name)
    text = str(value)
    if "." in text:
        return text.rsplit(".", 1)[-1]
    return text


def known_policyengine_divergence(
    *,
    surface: str,
    output: str,
    entity_id: str,
    axiom_value: float,
    policyengine_value: float,
    diff: float,
) -> UKEFRSOracleDivergence | None:
    if (
        surface == "personal-allowance"
        and output == "personal_allowance"
        and 0 < diff < 1
        and math.isclose(axiom_value, math.ceil(policyengine_value), abs_tol=1e-9)
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently returns fractional tapered personal "
                "allowances instead of rounding up to the nearest pound under "
                "ITA 2007 s.35(3)."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1738",
        )
    if (
        surface == "child-benefit"
        and output == "child_benefit_weekly_rate"
        and 0 < diff < 0.2
        and (
            math.isclose(axiom_value, 27.05, abs_tol=1e-9)
            or math.isclose(axiom_value, 17.90, abs_tol=1e-9)
        )
        and (
            math.isclose(policyengine_value, 26.935709699992934, abs_tol=0.005)
            or math.isclose(policyengine_value, 17.836506423219888, abs_tol=0.005)
        )
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 Child "
                "Benefit amounts instead of the published 2026-27 weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1739",
        )
    if (
        surface == "pension-credit"
        and output == "standard_minimum_guarantee"
        and 0 < diff < 15
        and (
            math.isclose(axiom_value, 238.00, abs_tol=1e-9)
            or math.isclose(axiom_value, 363.25, abs_tol=1e-9)
        )
        and (
            math.isclose(policyengine_value, 229.3929826081932, abs_tol=0.01)
            or math.isclose(policyengine_value, 350.12286608501375, abs_tol=0.01)
        )
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 Pension "
                "Credit guarantee amounts instead of the published 2026-27 "
                "weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1740",
        )
    if (
        surface == "pension-credit"
        and output in {"severe_disability_additional_amount", "carer_additional_amount"}
        and pension_credit_additional_amount_matches_published_rate(output, axiom_value)
        and policyengine_value > 0
        and 0 < abs(diff) < 5
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 "
                "Pension Credit additional amounts instead of the published "
                "2026-27 weekly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1742",
        )
    expected_uc_rate = UNIVERSAL_CREDIT_2026_RULESPEC_RATES.get(output)
    if (
        surface.startswith("universal-credit-")
        and expected_uc_rate is not None
        and math.isclose(axiom_value, expected_uc_rate, abs_tol=1e-9)
        and policyengine_value > 0
        and 0 < abs(diff) < 75
    ):
        return UKEFRSOracleDivergence(
            surface=surface,
            entity_id=entity_id,
            output=output,
            axiom=axiom_value,
            policyengine=policyengine_value,
            diff=diff,
            reason=(
                "PolicyEngine UK currently uses forecast-indexed 2026 "
                "Universal Credit Regulation 36 amounts instead of the "
                "published 2026-27 monthly rates."
            ),
            issue_url="https://github.com/PolicyEngine/policyengine-uk/issues/1741",
        )
    return None


def pension_credit_additional_amount_matches_published_rate(
    output: str,
    axiom_value: float,
) -> bool:
    if output == "carer_additional_amount":
        return math.isclose(axiom_value, 48.15, abs_tol=1e-9)
    if output == "severe_disability_additional_amount":
        return math.isclose(axiom_value, 86.05, abs_tol=1e-9) or math.isclose(
            axiom_value,
            172.10,
            abs_tol=1e-9,
        )
    return False


def print_report(
    report: UKEFRSComparisonReport,
    *,
    tolerance: float,
    relative_tolerance: float,
) -> None:
    print("PolicyEngine UK EFRS comparison")
    print(f"Compared persons: {report.compared_persons:,}")
    print(f"Compared benefit units: {report.compared_benunits:,}")
    print(f"Compared values: {report.compared_values:,}")
    print(f"Tolerance: {tolerance:g}")
    print(f"Relative tolerance: {relative_tolerance:g}")
    print(f"Mismatches: {len(report.mismatches):,}")
    print(f"Known PolicyEngine oracle divergences: {len(report.oracle_divergences):,}")
    print()
    print("By output:")
    for item in report.output_summary:
        print(
            f"  - {item['surface']}:{item['output']}: "
            f"{item['mismatches']:,}/{item['compared']:,} mismatch, "
            f"{item['oracle_divergences']:,} known PE divergence, "
            f"max_abs_diff={item['max_abs_diff']:.2f}, "
            f"max_rel_diff={item['max_relative_diff']:.2g}"
        )
    if report.mismatches:
        print()
        print("Top mismatches:")
        for row in sorted(
            report.mismatches, key=lambda item: abs(item.diff), reverse=True
        )[:20]:
            print(
                f"  - entity={row.entity_id} {row.surface}:{row.output}: "
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} "
                f"diff={row.diff:.2f}"
            )
    if report.oracle_divergences:
        print()
        print("Known PolicyEngine oracle divergences:")
        for row in sorted(
            report.oracle_divergences, key=lambda item: abs(item.diff), reverse=True
        )[:20]:
            print(
                f"  - entity={row.entity_id} {row.surface}:{row.output}: "
                f"axiom={row.axiom:.2f} pe={row.policyengine:.2f} "
                f"diff={row.diff:.2f}; {row.issue_url}"
            )
    if report.skipped_surfaces:
        print()
        print("Skipped mapped UK surfaces:")
        for item in report.skipped_surfaces:
            print(f"  - {item['surface']}: {item['reason']}")
    print()
    print("Projection notes:")
    for note in report.projection_notes:
        print(f"  - {note}")


def table_records(table: Any) -> list[dict[str, Any]]:
    if isinstance(table, list):
        return [dict(row) for row in table]
    if isinstance(table, dict):
        keys = list(table)
        length = len(table[keys[0]]) if keys else 0
        return [{key: table[key][index] for key in keys} for index in range(length)]
    if hasattr(table, "to_dict"):
        return table.to_dict("records")
    raise TypeError(f"unsupported table type: {type(table).__name__}")


def row_value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, dict):
        return row.get(name, default)
    if hasattr(row, "get"):
        return row.get(name, default)
    return getattr(row, name, default)


def tax_year_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "tax_year",
        "start": f"{year:04d}-01-01",
        "end": f"{year:04d}-12-31",
    }


def day_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "custom",
        "name": "day",
        "start": f"{year:04d}-04-06",
        "end": f"{year:04d}-04-06",
    }


def benefit_week_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "custom",
        "name": "benefit_week",
        "start": f"{year:04d}-04-06",
        "end": f"{year:04d}-04-12",
    }


def tax_week_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "custom",
        "name": "tax_week",
        "start": f"{year:04d}-04-06",
        "end": f"{year:04d}-04-12",
    }


def benefit_month_interval(year: int) -> dict[str, str]:
    return {
        "period_kind": "month",
        "name": "benefit_month",
        "start": f"{year:04d}-04-01",
        "end": f"{year:04d}-04-30",
    }


def person_entity_id(person_id: int) -> str:
    return f"person_{person_id}"


def income_tax_component_entity_id(person_id: int, component: str) -> str:
    return f"{person_entity_id(person_id)}_income_{component}"


def benunit_entity_id(benunit_id: int) -> str:
    return f"benunit_{benunit_id}"


def resolve_workspace_root(root: Path | None) -> Path:
    if root is not None:
        return root.resolve()
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents, Path.home() / "TheAxiomFoundation"]:
        if (candidate / "rulespec-uk").exists() and (
            candidate / "axiom-rules-engine"
        ).exists():
            return candidate
    return cwd


def require_policyengine_uk_versions(command: str = "uk-efrs-compare") -> None:
    try:
        policyengine_core_version = version("policyengine-core")
        policyengine_uk_version = version("policyengine-uk")
    except PackageNotFoundError as exc:
        raise SystemExit(policyengine_uk_install_message(command)) from exc
    if policyengine_core_version != POLICYENGINE_CORE_VERSION:
        raise SystemExit(
            f"policyengine-core=={POLICYENGINE_CORE_VERSION} required; found "
            f"{policyengine_core_version}. {policyengine_uk_install_message(command)}"
        )
    if policyengine_uk_version != POLICYENGINE_UK_VERSION:
        raise SystemExit(
            f"policyengine-uk=={POLICYENGINE_UK_VERSION} required; found "
            f"{policyengine_uk_version}. {policyengine_uk_install_message(command)}"
        )


def policyengine_uk_class_1_weekly_parameters(year: int) -> dict[str, float]:
    require_policyengine_uk_versions()
    try:
        from policyengine_uk import CountryTaxBenefitSystem
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(policyengine_uk_install_message()) from exc

    class_1 = (
        CountryTaxBenefitSystem()
        .parameters(f"{year:04d}-04-06")
        .gov.hmrc.national_insurance.class_1
    )
    return {
        "primary_threshold": money(class_1.thresholds.primary_threshold),
        "upper_earnings_limit": money(class_1.thresholds.upper_earnings_limit),
    }


def policyengine_uk_install_message(command: str = "uk-efrs-compare") -> str:
    return (
        "Run with: uv run "
        f"--with policyengine-uk=={POLICYENGINE_UK_VERSION} "
        f"axiom-encode {command}"
    )


def log(message: str) -> None:
    print(message, file=sys.stderr)

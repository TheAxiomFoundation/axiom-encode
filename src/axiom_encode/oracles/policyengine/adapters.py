"""PolicyEngine adapter metadata for oracle scenario construction."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyEngineUSVarAdapter:
    """Declarative mapping/config for PE-US replay of an encoded variable."""

    rule_names: tuple[str, ...]
    pe_var: str
    monthly: bool = False
    spm: bool = False
    annualized_person_inputs: tuple[tuple[str, str], ...] = ()
    boolean_person_inputs: tuple[tuple[str, str], ...] = ()
    monthly_boolean_person_inputs: tuple[tuple[str, str], ...] = ()
    direct_spm_overrides: tuple[tuple[str, str], ...] = ()
    derived_spm_overrides: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    annual_direct_spm_overrides: tuple[tuple[str, str], ...] = ()
    annual_derived_spm_overrides: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    unsupported_input_keys: tuple[str, ...] = ()
    unsupported_input_patterns: tuple[str, ...] = ()
    unsupported_input_reason: str | None = None
    default_state_code: str | None = None
    state_code_from_boolean_input: tuple[str, str, str] | None = None
    parameter_path: str | None = None
    parameter_value_mode: str = "bool"


def normalize_state_code_from_utility_region(region: str) -> str:
    """Map sub-state SNAP utility region codes back to their parent state code."""
    match = re.match(r"^([A-Z]{2})_", region)
    if match:
        return match.group(1)
    return region


PE_US_VAR_ADAPTERS = (
    PolicyEngineUSVarAdapter(
        rule_names=("snap", "snap_benefits"),
        pe_var="snap",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_allotment", "snap_normal_allotment"),
        pe_var="snap_normal_allotment",
        monthly=True,
        spm=True,
        unsupported_input_keys=(
            "snap_max_allotment",
            "snap_expected_contribution",
            "snap_min_allotment",
            "is_snap_eligible",
        ),
        unsupported_input_reason=(
            "RuleSpec test supplies intermediate SNAP allotment inputs that "
            "PolicyEngine US does not expose as scenario inputs"
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_expected_contribution",),
        pe_var="snap_expected_contribution",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_earned_income_deduction",),
        pe_var="snap_earned_income_deduction",
        monthly=True,
        spm=True,
        direct_spm_overrides=(("snap_earned_income", "snap_earned_income"),),
        derived_spm_overrides=(
            (
                "snap_earned_income",
                "difference_floor_zero",
                (
                    "snap_earned_income_before_exclusions",
                    "snap_child_earned_income_exclusion",
                    "snap_other_earned_income_exclusions",
                    "snap_work_support_public_assistance_income",
                ),
            ),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_min_allotment", "minimum_allotment"),
        pe_var="snap_min_allotment",
        monthly=True,
        spm=True,
        unsupported_input_keys=("snap_one_person_thrifty_food_plan_cost",),
        unsupported_input_patterns=("thrifty_food_plan_cost",),
        unsupported_input_reason=(
            "RuleSpec test supplies a thrifty-food-plan cost input that "
            "PolicyEngine US treats as an internal parameter, not a scenario input"
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_net_income", "snap_net_income_calculation"),
        pe_var="snap_net_income",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_earned_income", "snap_earned_income"),
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
        ),
        derived_spm_overrides=(
            (
                "snap_net_income",
                "difference",
                ("snap_household_income", "snap_deductions"),
            ),
        ),
        annual_derived_spm_overrides=(
            ("housing_cost", "monthly_to_annual", ("housing_cost",)),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_net_income_pre_shelter",),
        pe_var="snap_net_income_pre_shelter",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            (
                "snap_monthly_household_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_monthly_household_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_household_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_household_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_income_after_all_other_applicable_deductions",
                "snap_net_income_pre_shelter",
            ),
            (
                "snap_income_after_all_other_applicable_deductions_have_been_allowed",
                "snap_net_income_pre_shelter",
            ),
            ("snap_gross_income", "snap_gross_income"),
            ("snap_earned_income", "snap_earned_income"),
            ("snap_standard_deduction", "snap_standard_deduction"),
            ("snap_earned_income_deduction", "snap_earned_income_deduction"),
            ("snap_child_support_deduction", "snap_child_support_deduction"),
            (
                "snap_excess_medical_expense_deduction",
                "snap_excess_medical_expense_deduction",
            ),
        ),
        derived_spm_overrides=(
            (
                "snap_earned_income",
                "difference_floor_zero",
                (
                    "snap_earned_income_before_exclusions",
                    "snap_child_earned_income_exclusion",
                    "snap_other_earned_income_exclusions",
                    "snap_work_support_public_assistance_income",
                ),
            ),
        ),
        annual_derived_spm_overrides=(
            (
                "spm_unit_pre_subsidy_childcare_expenses",
                "difference_floor_zero_annualized",
                (
                    "snap_dependent_care_actual_costs",
                    "snap_dependent_care_excluded_expenses",
                ),
            ),
            (
                "spm_unit_pre_subsidy_childcare_expenses",
                "monthly_to_annual",
                ("snap_dependent_care_deduction",),
            ),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_utility_allowance",),
        pe_var="snap_standard_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_limited_utility_allowance",),
        pe_var="snap_limited_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_individual_utility_allowance",),
        pe_var="snap_individual_utility_allowance",
        monthly=True,
        spm=True,
        direct_spm_overrides=(
            ("snap_utility_allowance_type", "snap_utility_allowance_type"),
            ("spm_unit_size", "spm_unit_size"),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_state_using_standard_utility_allowance",),
        pe_var="snap_state_using_standard_utility_allowance",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_state_uses_child_support_deduction",),
        pe_var="snap_state_uses_child_support_deduction",
        default_state_code="TN",
        parameter_path="gov.usda.snap.income.deductions.child_support",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_self_employment_expense_based_deduction_applies",),
        pe_var="snap_self_employment_expense_based_deduction_applies",
        default_state_code="CA",
        parameter_path=(
            "gov.usda.snap.income.deductions.self_employment."
            "expense_based_deduction_applies"
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_self_employment_simplified_deduction_rate",),
        pe_var="snap_self_employment_simplified_deduction_rate",
        default_state_code="MD",
        parameter_path="gov.usda.snap.income.deductions.self_employment.rate",
        parameter_value_mode="float",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_medical_expense_deduction",),
        pe_var="snap_standard_medical_expense_deduction",
        parameter_path="gov.usda.snap.income.deductions.excess_medical_expense.standard",
        parameter_value_mode="float",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_homeless_shelter_deduction_available",),
        pe_var="snap_homeless_shelter_deduction_available",
        parameter_path=(
            "gov.usda.snap.income.deductions.excess_shelter_expense.homeless.available"
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_tanf_non_cash_gross_income_limit_fpg_ratio",),
        pe_var="snap_tanf_non_cash_gross_income_limit_fpg_ratio",
        default_state_code="TX",
        parameter_path="gov.hhs.tanf.non_cash.income_limit.gross",
        parameter_value_mode="float",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_tanf_non_cash_asset_limit",),
        pe_var="snap_tanf_non_cash_asset_limit",
        default_state_code="TX",
        parameter_path="gov.hhs.tanf.non_cash.asset_limit",
        parameter_value_mode="float",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_asset_test",),
        pe_var="meets_snap_asset_test",
        monthly=True,
        spm=True,
        annual_direct_spm_overrides=(
            ("snap_assets", "snap_assets"),
            ("snap_countable_resources", "snap_assets"),
            ("snap_countable_financial_resources", "snap_assets"),
            ("snap_financial_resources", "snap_assets"),
            (
                "snap_household_has_elderly_or_disabled_member",
                "has_usda_elderly_disabled",
            ),
        ),
        annual_derived_spm_overrides=(
            (
                "snap_assets",
                "difference",
                (
                    "snap_total_resources_before_exclusions",
                    "snap_mandatory_retirement_account_resource_exclusion",
                    "snap_discretionary_retirement_account_resource_exclusion",
                    "snap_mandatory_education_account_resource_exclusion",
                    "snap_discretionary_education_account_resource_exclusion",
                    "snap_other_resource_exclusions_under_g",
                ),
            ),
        ),
        unsupported_input_keys=(
            "snap_statutory_asset_limit",
            "snap_applicable_asset_limit",
            "snap_asset_limit",
            "snap_asset_limit_with_elderly_or_disabled_member",
        ),
        unsupported_input_reason=(
            "RuleSpec test restates the SNAP asset-test threshold with local "
            "limit/resource abstractions that PolicyEngine US does not expose "
            "as scenario inputs"
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_gross_income_test",),
        pe_var="meets_snap_gross_income_test",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("meets_snap_net_income_test",),
        pe_var="meets_snap_net_income_test",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("is_snap_eligible",),
        pe_var="is_snap_eligible",
        monthly=True,
        spm=True,
        boolean_person_inputs=(
            ("is_snap_ineligible_student", "is_snap_ineligible_student"),
        ),
        monthly_boolean_person_inputs=(
            (
                "is_snap_immigration_status_eligible",
                "is_snap_immigration_status_eligible",
            ),
        ),
        direct_spm_overrides=(
            ("meets_snap_gross_income_test", "meets_snap_gross_income_test"),
            ("meets_snap_net_income_test", "meets_snap_net_income_test"),
            ("meets_snap_asset_test", "meets_snap_asset_test"),
            (
                "meets_snap_categorical_eligibility",
                "meets_snap_categorical_eligibility",
            ),
            ("meets_snap_work_requirements", "meets_snap_work_requirements"),
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_standard_deduction",),
        pe_var="snap_standard_deduction",
        monthly=True,
        spm=True,
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_child_support_deduction",),
        pe_var="snap_child_support_gross_income_deduction",
        monthly=True,
        spm=True,
        annualized_person_inputs=(
            ("snap_child_support_payments_made", "child_support_expense"),
        ),
        state_code_from_boolean_input=(
            "snap_state_uses_child_support_deduction",
            "TX",
            "CA",
        ),
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_excess_medical_expense_deduction",),
        pe_var="snap_excess_medical_expense_deduction",
        monthly=True,
        spm=True,
        annualized_person_inputs=(
            (
                "snap_allowable_medical_expenses_before_threshold",
                "medical_out_of_pocket_expenses",
            ),
        ),
        boolean_person_inputs=(
            (
                "snap_household_has_elderly_or_disabled_member",
                "is_usda_disabled",
            ),
        ),
        default_state_code="NY",
    ),
    PolicyEngineUSVarAdapter(
        rule_names=("snap_maximum_allotment",),
        pe_var="snap_max_allotment",
        monthly=True,
        spm=True,
    ),
)

PE_US_VAR_ADAPTERS_BY_NAME = {
    name: adapter
    for adapter in PE_US_VAR_ADAPTERS
    for name in (adapter.pe_var, *adapter.rule_names)
}

PE_US_MONTHLY_VAR_NAMES = {
    name
    for adapter in PE_US_VAR_ADAPTERS
    if adapter.monthly
    for name in (adapter.pe_var, *adapter.rule_names)
}

PE_US_SPM_VAR_NAMES = {
    name
    for adapter in PE_US_VAR_ADAPTERS
    if adapter.spm
    for name in (adapter.pe_var, *adapter.rule_names)
}


def get_pe_us_var_adapter(name: str) -> PolicyEngineUSVarAdapter | None:
    """Return a PE-US adapter row for a mapped PE variable or adapter alias."""
    return PE_US_VAR_ADAPTERS_BY_NAME.get(name)

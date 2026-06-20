from pathlib import Path

import pytest

from axiom_encode.oracles.policyengine.coverage import (
    build_policyengine_candidate_report,
    build_policyengine_coverage_report,
    build_policyengine_program_surface_report,
)
from axiom_encode.oracles.policyengine.registry import PolicyEngineMapping


class _ProgramSurfaceRegistry:
    def __init__(self, mappings_by_variable):
        self.mappings_by_variable = mappings_by_variable

    def mappings_for_policyengine_variable(
        self, policyengine_variable, *, country=None
    ):
        return [
            mapping
            for mapping in self.mappings_by_variable.get(policyengine_variable, [])
            if country is None or mapping.country == country
        ]


def _write_rulespec_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_policyengine_coverage_classifies_executable_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/7/2014/e/2.yaml",
        """format: rulespec/v1
rules:
  - name: snap_earned_income_deduction
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: earned_income * 0.2
  - name: snap_earned_income_subject_to_deduction
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: earned_income
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/10-ccr-2506-1/4.999.yaml",
        """format: rulespec/v1
rules:
  - name: snap_local_helper
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: local_input
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/7/9999.yaml",
        """format: rulespec/v1
rules:
  - name: snap_unclassified_new_output
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 1
  - name: documentary_relation
    kind: source_relation
    source_relation:
      type: cites
      target: us:statutes/7/2014/e/2#snap_earned_income_deduction
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 4
    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 2,
        "unmapped": 1,
    }
    assert report["untested_comparable"] == 1
    statuses_by_id = {item["legal_id"]: item["status"] for item in report["items"]}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        statuses_by_id["us:statutes/7/2014/e/2#snap_earned_income_deduction"]
        == "comparable"
    )
    assert (
        items_by_id["us:statutes/7/2014/e/2#snap_earned_income_deduction"]["tested"]
        is False
    )
    assert (
        statuses_by_id["us:statutes/7/2014/e/2#snap_earned_income_subject_to_deduction"]
        == "known_not_comparable"
    )
    assert (
        statuses_by_id["us-co:regulations/10-ccr-2506-1/4.999#snap_local_helper"]
        == "known_not_comparable"
    )
    assert (
        statuses_by_id["us:statutes/7/9999#snap_unclassified_new_output"] == "unmapped"
    )


def test_policyengine_program_surface_report_overlays_legal_registry(tmp_path):
    manifest = tmp_path / "program_surfaces.yaml"
    manifest.write_text(
        """source:
  repository: PolicyEngine/policyengine-us
  ref: test
  path: policyengine_us/programs.yaml
surfaces:
  - country: us
    program_id: federal_income_tax
    program_name: Federal income taxes
    category: Taxes
    policyengine_status: complete
    coverage: US
    variable: income_tax
    axiom_status: pending_rulespec_encoding
    priority: P1
    rationale: Should be overridden by the legal-ID registry.
  - country: us
    program_id: wic
    program_name: WIC
    category: Benefits
    policyengine_status: complete
    coverage: US
    variable: wic
    axiom_status: pending_rulespec_encoding
    priority: P1
    rationale: Needs RuleSpec encoding.
  - country: us
    program_id: aca_subsidies
    program_name: ACA subsidies
    category: Healthcare
    policyengine_status: complete
    coverage: US
    variable: aca_ptc
    axiom_status: pending_rulespec_encoding
    priority: P1
    rationale: Encoded but not comparable to PE's aggregate legal boundary.
  - country: us
    program_id: tanf
    program_name: Colorado TANF
    category: Benefits
    policyengine_status: complete
    coverage: CO
    variable: co_tanf
    axiom_status: known_not_comparable
    priority: P1
    rationale: Encoded legal slices are not the same boundary as PE's final program variable.
  - country: us
    program_id: payroll_taxes
    program_name: Payroll taxes
    category: Taxes
    policyengine_status: complete
    coverage: US
    variable: employee_payroll_tax
    axiom_status: out_of_scope
    priority: P1
    rationale: Aggregate surface should not be treated as direct RuleSpec work.
  - country: us
    program_id: fdpir
    program_name: FDPIR
    category: Benefits
    policyengine_status: partial
    coverage: US
    variable: fdpir
    axiom_status: input_only
    priority: P1
    rationale: PE treats this as input-only.
""",
        encoding="utf-8",
    )
    registry = _ProgramSurfaceRegistry(
        {
            "income_tax": [
                PolicyEngineMapping(
                    legal_id="us:statutes/26/6401#income_tax",
                    country="us",
                    mapping_type="direct_variable",
                    policyengine_variable="income_tax",
                )
            ],
            "aca_ptc": [
                PolicyEngineMapping(
                    legal_id="us:statutes/26/36B/b#premium_assistance_credit_amount",
                    country="us",
                    mapping_type="not_comparable",
                    policyengine_variable="aca_ptc",
                )
            ],
            "employee_payroll_tax": [
                PolicyEngineMapping(
                    legal_id="us:statutes/26/3102/a#employee_payroll_tax",
                    country="us",
                    mapping_type="not_comparable",
                    policyengine_variable="employee_payroll_tax",
                )
            ],
        }
    )

    report = build_policyengine_program_surface_report(
        manifest_path=manifest,
        registry=registry,
    )

    assert report["total_surfaces"] == 6
    assert report["status_counts"] == {
        "input_only": 1,
        "known_not_comparable": 2,
        "out_of_scope": 1,
        "pending_rulespec_encoding": 1,
        "wired": 1,
    }
    assert report["pending_surfaces"] == 1
    items_by_variable = {item["variable"]: item for item in report["items"]}
    assert items_by_variable["income_tax"]["axiom_status"] == "wired"
    assert items_by_variable["income_tax"]["mapping_count"] == 1
    assert items_by_variable["wic"]["axiom_status"] == "pending_rulespec_encoding"
    assert items_by_variable["aca_ptc"]["axiom_status"] == "known_not_comparable"
    assert items_by_variable["aca_ptc"]["mapping_count"] == 1
    assert items_by_variable["aca_ptc"]["comparable_mapping_count"] == 0
    assert items_by_variable["co_tanf"]["axiom_status"] == "known_not_comparable"
    assert items_by_variable["co_tanf"]["mapping_count"] == 0
    assert items_by_variable["employee_payroll_tax"]["axiom_status"] == "out_of_scope"
    assert items_by_variable["employee_payroll_tax"]["mapping_count"] == 1
    assert items_by_variable["employee_payroll_tax"]["comparable_mapping_count"] == 0
    assert items_by_variable["fdpir"]["axiom_status"] == "input_only"


def test_policyengine_program_surface_marks_colorado_ccap_final_subsidy_known_not_comparable():
    report = build_policyengine_program_surface_report(program="ccap")

    items_by_variable = {item["variable"]: item for item in report["items"]}
    colorado_ccap = items_by_variable["co_ccap_subsidy"]

    assert colorado_ccap["program_id"] == "ccdf"
    assert colorado_ccap["state"] == "CO"
    assert colorado_ccap["axiom_status"] == "known_not_comparable"
    assert colorado_ccap["mapping_count"] == 0
    assert "final modeled subsidy" in colorado_ccap["rationale"]


def test_policyengine_program_surface_marks_arizona_ccap_pending_source_ingestion():
    report = build_policyengine_program_surface_report(program="ccap")

    items_by_variable = {item["variable"]: item for item in report["items"]}
    arizona_ccap = items_by_variable["az_ccap"]

    assert arizona_ccap["program_id"] == "ccdf"
    assert arizona_ccap["state"] == "AZ"
    assert arizona_ccap["axiom_status"] == "pending_source_ingestion"
    assert arizona_ccap["mapping_count"] == 0
    assert "official DES/AZSOS sources" in arizona_ccap["rationale"]


def test_policyengine_program_surface_marks_colorado_oap_wired():
    report = build_policyengine_program_surface_report(program="co_oap")

    items_by_variable = {item["variable"]: item for item in report["items"]}
    colorado_oap = items_by_variable["co_oap"]

    assert colorado_oap["axiom_status"] == "wired"
    assert colorado_oap["mapping_count"] >= 1
    assert colorado_oap["comparable_mapping_count"] >= 1
    assert (
        "us-co:regulations/9-ccr-2503-5/3.532#oap_authorized_grant_payment_for_month"
        in colorado_oap["legal_ids"]
    )


def test_policyengine_program_surface_marks_colorado_ssp_wired():
    report = build_policyengine_program_surface_report(program="co_state_supplement")

    items_by_variable = {item["variable"]: item for item in report["items"]}
    colorado_ssp = items_by_variable["co_state_supplement"]

    assert colorado_ssp["program_id"] == "ssi_state_supplement"
    assert colorado_ssp["state"] == "CO"
    assert colorado_ssp["axiom_status"] == "wired"
    assert colorado_ssp["mapping_count"] >= 1
    assert colorado_ssp["comparable_mapping_count"] >= 1
    assert (
        "us-co:regulations/9-ccr-2503-5/3.548#and_cs_authorized_grant_payment"
        in colorado_ssp["legal_ids"]
    )
    assert (
        "us-co:regulations/9-ccr-2503-5/3.548#and_cs_authorized_grant_payment_for_month"
        in colorado_ssp["legal_ids"]
    )


def test_policyengine_coverage_classifies_nz_outputs_outside_policyengine(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-nz"
        / "nz/statutes/income_tax/schedule_1/individual_income_tax.yaml",
        """format: rulespec/v1
rules:
  - name: individual_income_tax_bracket_rates
    kind: parameter
    indexed_by: bracket
    versions:
      - effective_from: '2025-04-01'
        values:
          1: 0.105
  - name: individual_income_tax_bracket_thresholds
    kind: parameter
    indexed_by: bracket
    versions:
      - effective_from: '2025-04-01'
        values:
          1: 15600
  - name: individual_income_tax_before_credits
    kind: derived
    versions:
      - effective_from: '2025-04-01'
        formula: taxable_income * individual_income_tax_bracket_rates[1]
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-nz" / "nz/statutes/social_security/example.yaml",
        """format: rulespec/v1
rules:
  - name: jobseeker_support_placeholder
    kind: derived
    versions:
      - effective_from: '2025-04-01'
        formula: 1
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["total_outputs"] == 4
    assert report["status_counts"] == {"known_not_comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    income_tax_item = items_by_id[
        "nz:statutes/income_tax/schedule_1/individual_income_tax#individual_income_tax_before_credits"
    ]
    assert income_tax_item["mapping_type"] == "not_comparable"
    assert income_tax_item["policyengine_variable"] is None
    assert income_tax_item["policyengine_parameter"] is None
    assert income_tax_item["program"] == "tax"
    assert (
        items_by_id[
            "nz:statutes/social_security/example#jobseeker_support_placeholder"
        ]["program"]
        == "unknown"
    )

    tax_report = build_policyengine_coverage_report(tmp_path, program="tax")
    assert tax_report["total_outputs"] == 3
    assert tax_report["status_counts"] == {"known_not_comparable": 3}


def test_policyengine_coverage_classifies_7_cfr_275_admin_prefix(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/7-cfr/275/23/e/1.yaml",
        """format: rulespec/v1
rules:
  - name: investment_liability_cap_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2000-01-01'
        formula: 0.5
  - name: at_risk_repayment_liability_cap_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2000-01-01'
        formula: 0.5
  - name: new_investment_federal_match_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2000-01-01'
        formula: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    for item in report["items"]:
        assert item["mapping_type"] == "not_comparable"
        assert "7 CFR part 275" in str(item["rationale"])


def test_policyengine_coverage_classifies_conclusive_agency_determinations(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/5/5566.yaml",
        """format: rulespec/v1
rules:
  - name: agency_determination_conclusive_as_to_death
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
  - name: agency_determination_conclusive_as_to_dependency
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
  - name: agency_determination_conclusive_as_to_essential_date
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
  - name: agency_determination_conclusive_as_to_official_report_of_death
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
  - name: agency_determination_conclusive_as_to_other_covered_status
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
  - name: agency_determination_conclusive_under_subchapter
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: agency_determination_made
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/37/556.yaml",
        """format: rulespec/v1
rules:
  - name: secretary_determination_conclusive
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: secretary_determination_made
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["status_counts"] == {"known_not_comparable": 7}
    assert {item["program"] for item in report["items"]} == {"unknown"}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["mapping_type"] for item in report["items"]} == {"not_comparable"}


def test_policyengine_coverage_classifies_colorado_tanf_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/9-ccr-2503-6/3.606.1/F.yaml",
        """format: rulespec/v1
rules:
  - name: basic_cash_assistance_grant_standard
    kind: derived
    versions:
      - effective_from: '2025-07-01'
        formula: grant_table_amount
  - name: basic_cash_assistance_need_standard
    kind: derived
    versions:
      - effective_from: '2025-07-01'
        formula: need_table_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/9-ccr-2503-6/3.606.1/K.yaml",
        """format: rulespec/v1
rules:
  - name: basic_cash_assistance_authorized_grant_for_eligible_assistance_unit
    kind: derived
    versions:
      - effective_from: '2025-07-01'
        formula: grant_standard - net_countable_income
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tanf")

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    assert report["untested_comparable"] == 0
    assert {item["program"] for item in report["items"]} == {"tanf"}
    assert {item["mapping_type"] for item in report["items"]} == {"not_comparable"}
    assert {item["candidate_priority"] for item in report["items"]} == {"P4"}
    assert all(
        "PolicyEngine-US exposes adjacent Colorado TANF variables"
        in str(item["rationale"])
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_arizona_tanf_exact_parameters(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-payment-standard-a1-2fa2.yaml",
        """format: rulespec/v1
rules:
  - name: a1_annual_payment_standard_percentage_of_base_year_fpl
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-09-02'
        formula: 0.36
  - name: a1_payment_standard_applies
    kind: derived
    entity: TanfUnit
    dtype: Judgment
    versions:
      - effective_from: '2025-09-02'
        formula: budgetary_unit_has_obligation_to_pay_shelter_cost
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-payment-standard-a1-2fa2.test.yaml",
        """- name: a1_rate
  output:
    us-az:policies/des/faa5/ca-payment-standard-a1-2fa2#a1_annual_payment_standard_percentage_of_base_year_fpl: 0.36
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/earned-income-deduction.yaml",
        """format: rulespec/v1
rules:
  - name: ca_earned_income_deduction_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-09-02'
        formula: 0.30
  - name: ca_earned_income_deduction
    kind: derived
    entity: Person
    dtype: Money
    versions:
      - effective_from: '2025-09-02'
        formula: ca_earned_income_deduction_rate * countable_earned_income
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/earned-income-deduction.test.yaml",
        """- name: earned_income_rate
  output:
    us-az:policies/des/faa5/ca-benefit-determination/earned-income-deduction#ca_earned_income_deduction_rate: 0.3
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/cost-of-employment-deduction.yaml",
        """format: rulespec/v1
rules:
  - name: cost_of_employment_monthly_deduction_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2025-09-02'
        formula: 90
  - name: cost_of_employment_deduction_applies
    kind: derived
    entity: Person
    dtype: Judgment
    versions:
      - effective_from: '2025-09-02'
        formula: participant_is_employed
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/cost-of-employment-deduction.test.yaml",
        """- name: cost_of_employment_amount
  output:
    us-az:policies/des/faa5/ca-benefit-determination/cost-of-employment-deduction#cost_of_employment_monthly_deduction_amount: 90
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/needy-family-test.yaml",
        """format: rulespec/v1
rules:
  - name: npcr_child_only_needy_family_fpl_limit_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-09-02'
        formula: 1.30
  - name: parent_or_npcr_self_and_child_needy_family_fpl_limit_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-09-02'
        formula: 1.00
  - name: needy_family_criteria_met
    kind: derived
    entity: TanfUnit
    dtype: Judgment
    versions:
      - effective_from: '2025-09-02'
        formula: caretaker_relative_family_income <= current_federal_poverty_level
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/ca-benefit-determination/needy-family-test.test.yaml",
        """- name: needy_family_rates
  output:
    us-az:policies/des/faa5/ca-benefit-determination/needy-family-test#npcr_child_only_needy_family_fpl_limit_rate: 1.3
    us-az:policies/des/faa5/ca-benefit-determination/needy-family-test#parent_or_npcr_self_and_child_needy_family_fpl_limit_rate: 1.0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tanf")
    items_by_id = {item["legal_id"]: item for item in report["items"]}

    exact_ids = {
        "us-az:policies/des/faa5/ca-payment-standard-a1-2fa2#a1_annual_payment_standard_percentage_of_base_year_fpl",
        "us-az:policies/des/faa5/ca-benefit-determination/earned-income-deduction#ca_earned_income_deduction_rate",
        "us-az:policies/des/faa5/ca-benefit-determination/cost-of-employment-deduction#cost_of_employment_monthly_deduction_amount",
        "us-az:policies/des/faa5/ca-benefit-determination/needy-family-test#npcr_child_only_needy_family_fpl_limit_rate",
        "us-az:policies/des/faa5/ca-benefit-determination/needy-family-test#parent_or_npcr_self_and_child_needy_family_fpl_limit_rate",
    }

    assert report["total_outputs"] == 9
    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 4,
    }
    assert report["untested_comparable"] == 0
    assert {items_by_id[legal_id]["mapping_type"] for legal_id in exact_ids} == {
        "parameter_value"
    }
    assert {items_by_id[legal_id]["tested"] for legal_id in exact_ids} == {True}
    assert (
        items_by_id[
            "us-az:policies/des/faa5/ca-benefit-determination/earned-income-deduction#ca_earned_income_deduction"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us-az:policies/des/faa5/ca-payment-standard-a1-2fa2#a1_payment_standard_applies"
        ]["candidate_priority"]
        == "P4"
    )


def test_policyengine_coverage_classifies_new_york_tanf_state_plan_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-ny"
        / "policies/otda/tanf-state-plan-2024-2026"
        / "financial-eligibility-and-income-disregards.yaml",
        """format: rulespec/v1
rules:
  - name: application_resource_limit
    kind: parameter
    versions:
      - effective_from: '2024-01-01'
        value: 2000
  - name: earned_income_disregard_rate
    kind: parameter
    versions:
      - effective_from: '2024-01-01'
        value: 0.5
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-ny"
        / "policies/otda/tanf-state-plan-2024-2026"
        / "standard-of-need-and-monthly-grant.yaml",
        """format: rulespec/v1
rules:
  - name: regular_recurring_monthly_need
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: need_schedule_amount
  - name: monthly_grant_and_allowance_with_home_energy
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: grant_and_allowance_amount
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tanf")

    assert report["total_outputs"] == 4
    assert report["status_counts"] == {"known_not_comparable": 4}
    assert report["untested_comparable"] == 0
    assert {item["program"] for item in report["items"]} == {"tanf"}
    assert {item["mapping_type"] for item in report["items"]} == {"not_comparable"}
    assert {item["candidate_priority"] for item in report["items"]} == {"P4"}
    assert all(
        "New York TANF State Plan outputs are source-specific OTDA"
        in str(item["rationale"])
        for item in report["items"]
    )


def test_policyengine_coverage_infers_health_programs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/hcpf/health-coverage.yaml",
        """format: rulespec/v1
rules:
  - name: is_medicaid_eligible
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: true
  - name: is_chip_eligible
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: true
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/36B.yaml",
        """format: rulespec/v1
rules:
  - name: aca_ptc
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 1
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "policies/cdhs/snap/vacation.yaml",
        """format: rulespec/v1
rules:
  - name: snap_sick_vacation_bonus_earned_income
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 1
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/n/5.yaml",
        """format: rulespec/v1
rules:
  - name: activity_secondarily_treated_woody_fuels_by_lopping_scattering_piling_chipping_removing_from_site
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: true
""",
    )

    medicaid = build_policyengine_coverage_report(tmp_path, program="medicaid")
    chip = build_policyengine_coverage_report(tmp_path, program="chip")
    aca = build_policyengine_coverage_report(tmp_path, program="aca_ptc")

    assert medicaid["total_outputs"] == 1
    assert medicaid["items"][0]["rule_name"] == "is_medicaid_eligible"
    assert chip["total_outputs"] == 1
    assert chip["items"][0]["rule_name"] == "is_chip_eligible"
    assert aca["total_outputs"] == 1
    assert aca["items"][0]["rule_name"] == "aca_ptc"
    health = build_policyengine_coverage_report(tmp_path, program="health")
    assert {item["rule_name"] for item in health["items"]} == {
        "is_medicaid_eligible",
        "is_chip_eligible",
        "aca_ptc",
    }


def test_policyengine_coverage_splits_combined_medicaid_chip_sources_by_rule_name(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-co"
        / "policies/cms/medicaid-chip-bhp-eligibility-levels.yaml",
        """format: rulespec/v1
rules:
  - name: children_separate_chip_income_standard
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 2.60
  - name: adult_expansion_medicaid_income_standard
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 1.33
  - name: magi_fpl_disregard_equivalent
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 0.05
""",
    )

    chip = build_policyengine_coverage_report(tmp_path, program="chip")
    medicaid = build_policyengine_coverage_report(tmp_path, program="medicaid")
    health = build_policyengine_coverage_report(tmp_path, program="health")

    assert [item["rule_name"] for item in chip["items"]] == [
        "children_separate_chip_income_standard"
    ]
    assert [item["rule_name"] for item in medicaid["items"]] == [
        "adult_expansion_medicaid_income_standard"
    ]
    assert {item["rule_name"] for item in health["items"]} == {
        "children_separate_chip_income_standard",
        "adult_expansion_medicaid_income_standard",
        "magi_fpl_disregard_equivalent",
    }


def test_policyengine_coverage_classifies_colorado_medicaid_chip_thresholds(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-co"
        / "policies/cms/colorado-medicaid-chip-bhp-eligibility-levels.yaml",
        """format: rulespec/v1
rules:
  - name: magi_fpl_disregard_rate
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 0.05
  - name: colorado_children_medicaid_ages_0_to_1_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 1.42
  - name: colorado_children_medicaid_ages_1_to_5_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 1.42
  - name: colorado_children_medicaid_ages_6_to_18_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 1.42
  - name: colorado_children_separate_chip_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 2.60
  - name: colorado_pregnant_women_medicaid_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 1.95
  - name: colorado_pregnant_women_chip_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 2.60
  - name: colorado_parent_caretaker_adults_medicaid_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 0.68
  - name: colorado_adult_medicaid_expansion_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2023-12-01'
        formula: 1.33
""",
    )

    medicaid = build_policyengine_coverage_report(tmp_path, program="medicaid")
    chip = build_policyengine_coverage_report(tmp_path, program="chip")
    health = build_policyengine_coverage_report(tmp_path, program="health")

    assert medicaid["total_outputs"] == 6
    assert chip["total_outputs"] == 2
    assert health["total_outputs"] == 9
    assert health["status_counts"] == {"known_not_comparable": 9}
    assert health["untested_comparable"] == 0
    assert {item["mapping_type"] for item in health["items"]} == {"not_comparable"}
    assert {item["candidate_priority"] for item in health["items"]} == {"P4"}

    items_by_name = {item["rule_name"]: item for item in health["items"]}
    assert items_by_name["magi_fpl_disregard_rate"]["program"] == "health"
    assert (
        items_by_name["colorado_adult_medicaid_expansion_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.medicaid.eligibility.categories.adult.income_limit"
    )
    assert (
        items_by_name["colorado_children_medicaid_ages_0_to_1_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.medicaid.eligibility.categories.infant.income_limit"
    )
    assert (
        items_by_name["colorado_children_separate_chip_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.chip.child.income_limit"
    )
    assert all(
        "5% MAGI FPL disregard" in str(item["rationale"]) for item in health["items"]
    )


def test_policyengine_coverage_classifies_georgia_snap_medicaid_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-ga"
        / "policies/cms/georgia-medicaid-chip-bhp-eligibility-levels.yaml",
        """format: rulespec/v1
rules:
  - name: magi_fpl_disregard_rate
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 0.05
  - name: georgia_children_medicaid_ages_0_to_1_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 2.05
  - name: georgia_children_medicaid_ages_1_to_5_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 1.49
  - name: georgia_children_medicaid_ages_6_to_18_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 1.33
  - name: georgia_children_separate_chip_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 2.47
  - name: georgia_pregnant_women_medicaid_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 2.20
  - name: georgia_children_medicaid_ages_0_to_1_effective_fpl_limit
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: georgia_children_medicaid_ages_0_to_1_fpl_limit + magi_fpl_disregard_rate
  - name: georgia_children_medicaid_ages_1_to_5_effective_fpl_limit
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: georgia_children_medicaid_ages_1_to_5_fpl_limit + magi_fpl_disregard_rate
  - name: georgia_children_medicaid_ages_6_to_18_effective_fpl_limit
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: georgia_children_medicaid_ages_6_to_18_fpl_limit + magi_fpl_disregard_rate
  - name: georgia_children_separate_chip_effective_fpl_limit
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: georgia_children_separate_chip_fpl_limit + magi_fpl_disregard_rate
  - name: georgia_pregnant_women_medicaid_effective_fpl_limit
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: georgia_pregnant_women_medicaid_fpl_limit + magi_fpl_disregard_rate
  - name: georgia_parent_caretaker_adults_medicaid_fpl_limit
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: 0.28
  - name: georgia_parent_caretaker_standard_uses_dollar_amounts
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: true
  - name: georgia_pregnant_women_chip_available
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: false
  - name: georgia_adult_medicaid_expansion_available
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: false
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-ga"
        / "policies/cms/georgia-medicaid-chip-bhp-eligibility-levels.test.yaml",
        """- name: georgia_effective_magi_limits
  output:
    us-ga:policies/cms/georgia-medicaid-chip-bhp-eligibility-levels#georgia_children_medicaid_ages_0_to_1_effective_fpl_limit: 2.10
    us-ga:policies/cms/georgia-medicaid-chip-bhp-eligibility-levels#georgia_children_medicaid_ages_1_to_5_effective_fpl_limit: 1.54
    us-ga:policies/cms/georgia-medicaid-chip-bhp-eligibility-levels#georgia_children_medicaid_ages_6_to_18_effective_fpl_limit: 1.38
    us-ga:policies/cms/georgia-medicaid-chip-bhp-eligibility-levels#georgia_children_separate_chip_effective_fpl_limit: 2.52
    us-ga:policies/cms/georgia-medicaid-chip-bhp-eligibility-levels#georgia_pregnant_women_medicaid_effective_fpl_limit: 2.25
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-ga" / "policies/dfcs/snap/3210/block-2.yaml",
        """format: rulespec/v1
rules:
  - name: assistance_unit_member_receives_tanf_wsp_or_ssi
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: receives_tanf or receives_wsp or receives_ssi
  - name: categorically_eligible_for_snap
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: all_au_members_receive_tanf_wsp_or_ssi or receives_or_authorized_for_tcos
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["total_outputs"] == 17
    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 12,
    }
    assert report["untested_comparable"] == 0
    assert report["program_counts"] == {
        "chip": 3,
        "health": 1,
        "medicaid": 11,
        "snap": 2,
    }

    items_by_name = {item["rule_name"]: item for item in report["items"]}
    assert (
        items_by_name["georgia_children_medicaid_ages_0_to_1_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.medicaid.eligibility.categories.infant.income_limit"
    )
    assert (
        items_by_name["georgia_children_separate_chip_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.chip.child.income_limit"
    )
    effective_infant = items_by_name[
        "georgia_children_medicaid_ages_0_to_1_effective_fpl_limit"
    ]
    assert effective_infant["status"] == "comparable"
    assert effective_infant["mapping_type"] == "parameter_value"
    assert effective_infant["policyengine_parameter"] == (
        "gov.hhs.medicaid.eligibility.categories.infant.income_limit"
    )
    assert effective_infant["tested"] is True
    assert (
        items_by_name["georgia_pregnant_women_medicaid_effective_fpl_limit"][
            "policyengine_parameter"
        ]
        == "gov.hhs.medicaid.eligibility.categories.pregnant.income_limit"
    )
    assert (
        items_by_name["georgia_pregnant_women_chip_available"]["policyengine_parameter"]
        == "gov.hhs.chip.pregnant.income_limit"
    )
    assert (
        items_by_name["georgia_pregnant_women_chip_available"]["status"]
        == "known_not_comparable"
    )
    assert items_by_name["categorically_eligible_for_snap"]["program"] == "snap"
    assert (
        items_by_name["categorically_eligible_for_snap"]["policyengine_variable"]
        is None
    )
    assert (
        items_by_name["assistance_unit_member_receives_tanf_wsp_or_ssi"][
            "policyengine_variable"
        ]
        is None
    )


def test_policyengine_coverage_classifies_aca_ptc_percentage_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "policies/irs/rev-proc-2025-25/aca-ptc.yaml",
        """format: rulespec/v1
rules:
  - name: applicable_percentage_band
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: 0
  - name: applicable_initial_percentage_table
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.021
  - name: applicable_final_percentage_table
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.021
  - name: applicable_initial_percentage
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: applicable_initial_percentage_table[applicable_percentage_band]
  - name: applicable_final_percentage
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: applicable_final_percentage_table[applicable_percentage_band]
  - name: required_contribution_percentage
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 0.0996
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "policies/irs/rev-proc-2025-25/aca-ptc.test.yaml",
        """- name: required_contribution_percentage_for_2026_plan_year
  output:
    us:policies/irs/rev-proc-2025-25/aca-ptc#required_contribution_percentage: 0.0996
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/36B/b/3/A.yaml",
        """format: rulespec/v1
rules:
  - name: applicable_percentage_income_tier
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0
  - name: applicable_percentage_tier_lower_bound
    kind: parameter
    indexed_by: applicable_percentage_income_tier
    versions:
      - effective_from: '1990-01-01'
        values:
          1: 1.33
  - name: applicable_percentage_tier_upper_bound
    kind: parameter
    indexed_by: applicable_percentage_income_tier
    versions:
      - effective_from: '1990-01-01'
        values:
          0: 1.33
  - name: initial_premium_percentage_by_income_tier
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        values:
          0: 0.02
  - name: final_premium_percentage_by_income_tier
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        values:
          0: 0.02
  - name: applicable_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0.02
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="aca_ptc")

    assert report["total_outputs"] == 12
    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 11,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    required = items_by_id[
        "us:policies/irs/rev-proc-2025-25/aca-ptc#required_contribution_percentage"
    ]
    assert required["status"] == "comparable"
    assert (
        required["policyengine_parameter"]
        == "gov.aca.required_contribution_percentage.final"
    )
    assert required["tested"] is True
    statute_applicable = items_by_id["us:statutes/26/36B/b/3/A#applicable_percentage"]
    assert statute_applicable["status"] == "known_not_comparable"
    assert (
        statute_applicable["policyengine_variable"]
        == "aca_required_contribution_percentage"
    )
    lower_bound = items_by_id[
        "us:statutes/26/36B/b/3/A#applicable_percentage_tier_lower_bound"
    ]
    upper_bound = items_by_id[
        "us:statutes/26/36B/b/3/A#applicable_percentage_tier_upper_bound"
    ]
    assert lower_bound["status"] == "known_not_comparable"
    assert upper_bound["status"] == "known_not_comparable"
    assert lower_bound["mapping_type"] == "not_comparable"
    assert upper_bound["mapping_type"] == "not_comparable"
    assert "Indexed interval-bound helper" in str(lower_bound["rationale"])


def test_policyengine_coverage_classifies_alabama_snap_manual_prefix(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-al"
        / "policies/dhr/poe/chapter-07-work-requirements/710.yaml",
        """format: rulespec/v1
rules:
  - name: person_subject_to_abawd_provision
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: person_age >= 18
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-al"
        / "policies/dhr/poe/chapter-09-income-and-deductions/900.yaml",
        """format: rulespec/v1
rules:
  - name: household_meets_snap_income_eligibility_standards
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: household_income <= income_limit
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 2
    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    manual_output = items_by_id[
        "us-al:policies/dhr/poe/chapter-07-work-requirements/710#person_subject_to_abawd_provision"
    ]
    exact_output = items_by_id[
        "us-al:policies/dhr/poe/chapter-09-income-and-deductions/900#household_meets_snap_income_eligibility_standards"
    ]
    assert manual_output["mapping_type"] == "not_comparable"
    assert "Alabama DHR POE SNAP manual sections" in str(manual_output["rationale"])
    assert exact_output["mapping_type"] == "not_comparable"
    assert "state income bridge" in str(exact_output["rationale"])


def test_policyengine_coverage_classifies_medicaid_work_requirement_prefixes(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/42/1396a/xx.yaml",
        """format: rulespec/v1
rules:
  - name: medicaid_community_engagement_condition_applies
    kind: derived
    versions:
      - effective_from: '2026-12-31'
        formula: applicable_individual and not exempt
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/552.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_community_engagement_hours_requirement
    kind: parameter
    versions:
      - effective_from: '2026-12-31'
        value: 80
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/558.yaml",
        """format: rulespec/v1
rules:
  - name: disenrollment_after_noncompliance_period
    kind: derived
    versions:
      - effective_from: '2026-12-31'
        formula: noncompliance_months >= 3
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="medicaid")

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    statute_output = items_by_id[
        "us:statutes/42/1396a/xx#medicaid_community_engagement_condition_applies"
    ]
    hours_output = items_by_id[
        "us:regulations/42-cfr/435/552#monthly_community_engagement_hours_requirement"
    ]
    disenrollment_output = items_by_id[
        "us:regulations/42-cfr/435/558#disenrollment_after_noncompliance_period"
    ]
    assert statute_output["mapping_type"] == "not_comparable"
    assert hours_output["mapping_type"] == "not_comparable"
    assert disenrollment_output["mapping_type"] == "not_comparable"
    assert "work-requirement" in str(statute_output["rationale"])
    assert "community-engagement" in str(disenrollment_output["rationale"])


def test_policyengine_coverage_classifies_medicaid_magi_prefixes(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/110.yaml",
        """format: rulespec/v1
rules:
  - name: parent_or_caretaker_relative_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: parent_or_caretaker_relative and household_income_within_limit
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/116.yaml",
        """format: rulespec/v1
rules:
  - name: pregnant_woman_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: is_pregnant and income_within_standard
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/118.yaml",
        """format: rulespec/v1
rules:
  - name: infants_and_children_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: child_under_age_19 and income_within_standard
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/42-cfr/435/119.yaml",
        """format: rulespec/v1
rules:
  - name: adult_group_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: adult_group_age_eligible and income_within_limit
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="medicaid")

    assert report["total_outputs"] == 4
    assert report["status_counts"] == {"known_not_comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    expected_legal_ids = [
        "us:regulations/42-cfr/435/110#parent_or_caretaker_relative_eligible",
        "us:regulations/42-cfr/435/116#pregnant_woman_eligible",
        "us:regulations/42-cfr/435/118#infants_and_children_eligible",
        "us:regulations/42-cfr/435/119#adult_group_eligible",
    ]
    for legal_id in expected_legal_ids:
        output = items_by_id[legal_id]
        assert output["mapping_type"] == "not_comparable"
        assert "MAGI" in str(output["rationale"])


def test_policyengine_coverage_classifies_nc_and_sc_snap_manual_prefixes(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-nc"
        / "policies/dhhs/fns/fns-600-simplified-nutritional-assistance-program-snap/page-1.yaml",
        """format: rulespec/v1
rules:
  - name: person_eligible_for_snap
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: person_receives_ssi
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-sc" / "policies/dss/snap-policy-manual/page-100.yaml",
        """format: rulespec/v1
rules:
  - name: snap_et_referral_required
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: person_subject_to_work_requirement
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 2
    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    nc_output = items_by_id[
        "us-nc:policies/dhhs/fns/fns-600-simplified-nutritional-assistance-program-snap/page-1#person_eligible_for_snap"
    ]
    sc_output = items_by_id[
        "us-sc:policies/dss/snap-policy-manual/page-100#snap_et_referral_required"
    ]
    assert nc_output["mapping_type"] == "not_comparable"
    assert "North Carolina DHHS FNS manual sections" in str(nc_output["rationale"])
    assert sc_output["mapping_type"] == "not_comparable"
    assert "South Carolina DSS SNAP manual sections" in str(sc_output["rationale"])


def test_policyengine_coverage_ignores_nested_axiom_dependency_checkout(tmp_path):
    content = """format: rulespec/v1
rules:
  - name: snap_earned_income_deduction
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: earned_income * 0.2
"""
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/7/2014/e/2.yaml",
        content,
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us"
        / "_axiom"
        / "rulespec-us"
        / "statutes/7/2014/e/2.yaml",
        content,
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 1
    assert report["status_counts"] == {"comparable": 1}
    assert report["untested_comparable"] == 1


def test_policyengine_coverage_counts_derived_relation_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "regulations/7-cfr/273/1.yaml",
        """format: rulespec/v1
rules:
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    assert report["total_outputs"] == 1
    assert report["items"][0]["legal_id"] == "us:regulations/7-cfr/273/1#snap_unit"
    assert report["items"][0]["kind"] == "derived_relation"


def test_policyengine_coverage_uses_uk_registry_mappings(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/35.yaml",
        """format: rulespec/v1
rules:
  - name: personal_allowance
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 12570
  - name: allowance_rounding_multiple
    kind: parameter
    dtype: Money
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 1
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/35.test.yaml",
        """- name: personal allowance
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:statutes/ukpga/2007/3/35#personal_allowance: 12570
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    personal_allowance = items_by_id["uk:statutes/ukpga/2007/3/35#personal_allowance"]
    rounding_multiple = items_by_id[
        "uk:statutes/ukpga/2007/3/35#allowance_rounding_multiple"
    ]
    assert personal_allowance["policyengine_variable"] == "personal_allowance"
    assert personal_allowance["tested"] is True
    assert rounding_multiple["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_uk_uc_regulation_18_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/376/18.yaml",
        """format: rulespec/v1
rules:
  - name: claimant_capital_for_prescribed_capital_limit
    kind: derived
    entity: Family
    dtype: Money
    period: Day
    unit: GBP
    versions:
      - effective_from: '2013-04-29'
        formula: claimant_capital
  - name: prescribed_capital_limit_for_claim
    kind: derived
    entity: Family
    dtype: Money
    period: Day
    unit: GBP
    versions:
      - effective_from: '2013-04-29'
        formula: prescribed_capital_limit_for_single_claimant
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/376/18.test.yaml",
        """- name: capital
  period:
    period_kind: custom
    name: day
    start: '2026-04-06'
    end: '2026-04-06'
  input: {}
  output:
    uk:regulations/uksi/2013/376/18#claimant_capital_for_prescribed_capital_limit: 12000
    uk:regulations/uksi/2013/376/18#prescribed_capital_limit_for_claim: 16000
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="universal_credit")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    claimant_capital = items_by_id[
        "uk:regulations/uksi/2013/376/18#claimant_capital_for_prescribed_capital_limit"
    ]
    prescribed_limit = items_by_id[
        "uk:regulations/uksi/2013/376/18#prescribed_capital_limit_for_claim"
    ]
    assert claimant_capital["policyengine_variable"] == "uc_assessable_capital"
    assert claimant_capital["status"] == "comparable"
    assert claimant_capital["mapping_type"] == "direct_variable"
    assert claimant_capital["tested"] is True
    assert prescribed_limit["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_uk_universal_credit_schedule_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/4/paragraph/36.yaml",
        """format: rulespec/v1
rules:
  - name: under_occupancy_deduction_amount
    kind: derived
    entity: BenefitUnit
    dtype: Money
    period: Month
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/10/paragraph/1.yaml",
        """format: rulespec/v1
rules:
  - name: premises_treated_as_persons_home_for_paragraphs_1_to_5
    kind: derived
    entity: BenefitUnit
    dtype: Bool
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: false
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/5/paragraph/9.yaml",
        """format: rulespec/v1
rules:
  - name: owner_occupier_housing_costs_element_amount
    kind: derived
    entity: BenefitUnit
    dtype: Money
    period: Month
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="universal_credit")

    assert report["status_counts"] == {"known_not_comparable": 3}
    statuses = {item["legal_id"]: item["status"] for item in report["items"]}
    assert (
        statuses[
            "uk:regulations/uksi/2013/376/schedule/4/paragraph/36#under_occupancy_deduction_amount"
        ]
        == "known_not_comparable"
    )
    assert (
        statuses[
            "uk:regulations/uksi/2013/376/schedule/10/paragraph/1#premises_treated_as_persons_home_for_paragraphs_1_to_5"
        ]
        == "known_not_comparable"
    )
    assert (
        statuses[
            "uk:regulations/uksi/2013/376/schedule/5/paragraph/9#owner_occupier_housing_costs_element_amount"
        ]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_income_tax_section_23_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/23.yaml",
        """format: rulespec/v1
rules:
  - name: total_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: income
  - name: net_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, total_income - reliefs)
  - name: income_tax_liability
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: income_tax
  - name: income_remaining_after_allowances
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income
  - name: tax_left_after_reductions
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: tax_after_reductions
  - name: net_income_zero_amount
    kind: parameter
    dtype: Money
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0
  - name: future_section_23_output
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/23.test.yaml",
        """- name: income tax steps
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:statutes/ukpga/2007/3/23#total_income: 40000
    uk:statutes/ukpga/2007/3/23#net_income: 38500
    uk:statutes/ukpga/2007/3/23#income_tax_liability: 4300
    uk:statutes/ukpga/2007/3/23#income_remaining_after_allowances: 25430
    uk:statutes/ukpga/2007/3/23#tax_left_after_reductions: 4200
    uk:statutes/ukpga/2007/3/23#net_income_zero_amount: 0
    uk:statutes/ukpga/2007/3/23#future_section_23_output: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 3,
        "known_not_comparable": 3,
        "unmapped": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    total_income = items_by_id["uk:statutes/ukpga/2007/3/23#total_income"]
    net_income = items_by_id["uk:statutes/ukpga/2007/3/23#net_income"]
    liability = items_by_id["uk:statutes/ukpga/2007/3/23#income_tax_liability"]
    remaining = items_by_id[
        "uk:statutes/ukpga/2007/3/23#income_remaining_after_allowances"
    ]
    future_output = items_by_id["uk:statutes/ukpga/2007/3/23#future_section_23_output"]

    assert total_income["policyengine_variable"] == "total_income"
    assert total_income["tested"] is True
    assert net_income["policyengine_variable"] == "adjusted_net_income"
    assert net_income["tested"] is True
    assert liability["policyengine_variable"] == "income_tax"
    assert liability["tested"] is True
    assert remaining["status"] == "known_not_comparable"
    assert future_output["status"] == "unmapped"


def test_policyengine_coverage_classifies_uk_income_tax_section_11d_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/11D.yaml",
        """format: rulespec/v1
rules:
  - name: savings_income_charged_at_savings_basic_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 1
  - name: savings_income_charged_at_savings_higher_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: savings_income_charged_at_savings_additional_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: savings_income_charged_under_section_11d
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 6
  - name: tax_on_savings_income_charged_at_savings_basic_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0.2
  - name: tax_on_savings_income_charged_at_savings_higher_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0.8
  - name: tax_on_savings_income_charged_at_savings_additional_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 1.35
  - name: income_tax_on_section_11d_savings_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 2.35
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/11D.test.yaml",
        """- name: savings income tax
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_basic_rate: 1
    uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_higher_rate: 2
    uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_additional_rate: 3
    uk:statutes/ukpga/2007/3/11D#savings_income_charged_under_section_11d: 6
    uk:statutes/ukpga/2007/3/11D#tax_on_savings_income_charged_at_savings_basic_rate: 0.2
    uk:statutes/ukpga/2007/3/11D#tax_on_savings_income_charged_at_savings_higher_rate: 0.8
    uk:statutes/ukpga/2007/3/11D#tax_on_savings_income_charged_at_savings_additional_rate: 1.35
    uk:statutes/ukpga/2007/3/11D#income_tax_on_section_11d_savings_income: 2.35
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 3,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_basic_rate"
        ]["policyengine_variable"]
        == "basic_rate_savings_income"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_higher_rate"
        ]["policyengine_variable"]
        == "higher_rate_savings_income"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#savings_income_charged_at_savings_additional_rate"
        ]["policyengine_variable"]
        == "add_rate_savings_income"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#savings_income_charged_under_section_11d"
        ]["policyengine_variable"]
        == "taxed_savings_income"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#income_tax_on_section_11d_savings_income"
        ]["policyengine_variable"]
        == "savings_income_tax"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/11D#tax_on_savings_income_charged_at_savings_basic_rate"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_income_tax_section_13_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/13.yaml",
        """format: rulespec/v1
rules:
  - name: dividend_income_charged_at_dividend_ordinary_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 1
  - name: dividend_income_charged_at_dividend_upper_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 2
  - name: dividend_income_charged_at_dividend_additional_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 3
  - name: dividend_income_charged_under_section_13
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 6
  - name: tax_on_dividend_income_charged_at_dividend_ordinary_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0.1075
  - name: tax_on_dividend_income_charged_at_dividend_upper_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 0.715
  - name: tax_on_dividend_income_charged_at_dividend_additional_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 1.1805
  - name: income_tax_on_section_13_dividend_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 2.003
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2007/3/13.test.yaml",
        """- name: dividend income tax
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:statutes/ukpga/2007/3/13#dividend_income_charged_at_dividend_ordinary_rate: 1
    uk:statutes/ukpga/2007/3/13#dividend_income_charged_at_dividend_upper_rate: 2
    uk:statutes/ukpga/2007/3/13#dividend_income_charged_at_dividend_additional_rate: 3
    uk:statutes/ukpga/2007/3/13#dividend_income_charged_under_section_13: 6
    uk:statutes/ukpga/2007/3/13#tax_on_dividend_income_charged_at_dividend_ordinary_rate: 0.1075
    uk:statutes/ukpga/2007/3/13#tax_on_dividend_income_charged_at_dividend_upper_rate: 0.715
    uk:statutes/ukpga/2007/3/13#tax_on_dividend_income_charged_at_dividend_additional_rate: 1.1805
    uk:statutes/ukpga/2007/3/13#income_tax_on_section_13_dividend_income: 2.003
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 6,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/13#dividend_income_charged_under_section_13"
        ]["policyengine_variable"]
        == "taxed_dividend_income"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/13#income_tax_on_section_13_dividend_income"
        ]["policyengine_variable"]
        == "dividend_income_tax"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2007/3/13#dividend_income_charged_at_dividend_ordinary_rate"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_class_1_ni_section_8_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/8.yaml",
        """format: rulespec/v1
rules:
  - name: main_primary_percentage
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2024-04-06'
        formula: '0.08'
  - name: additional_primary_percentage
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2011-04-06'
        formula: '0.02'
  - name: primary_class_1_contribution
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2024-04-06'
        formula: 68
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/8.test.yaml",
        """- name: class 1 employee ni
  period:
    period_kind: custom
    name: tax_week
    start: '2024-04-08'
    end: '2024-04-14'
  input: {}
  output:
    uk:statutes/ukpga/1992/4/8#main_primary_percentage: 0.08
    uk:statutes/ukpga/1992/4/8#additional_primary_percentage: 0.02
    uk:statutes/ukpga/1992/4/8#primary_class_1_contribution: 68
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 3}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    main_rate = items_by_id["uk:statutes/ukpga/1992/4/8#main_primary_percentage"]
    additional_rate = items_by_id[
        "uk:statutes/ukpga/1992/4/8#additional_primary_percentage"
    ]
    contribution = items_by_id[
        "uk:statutes/ukpga/1992/4/8#primary_class_1_contribution"
    ]

    assert (
        main_rate["policyengine_parameter"]
        == "gov.hmrc.national_insurance.class_1.rates.employee.main"
    )
    assert main_rate["tested"] is True
    assert (
        additional_rate["policyengine_parameter"]
        == "gov.hmrc.national_insurance.class_1.rates.employee.additional"
    )
    assert additional_rate["tested"] is True
    assert contribution["status"] == "comparable"
    assert contribution["policyengine_variable"] == "ni_class_1_employee"
    assert contribution["tested"] is True


def test_policyengine_coverage_classifies_uk_pension_credit_section_1_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2002/16/1.yaml",
        """format: rulespec/v1
rules:
  - name: qualifying_age
    kind: derived
    entity: Person
    dtype: Count
    period: Day
    unit: year
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          if claimant_is_woman: pensionable_age else: pensionable_age_for_woman_born_same_day
  - name: claimant_has_attained_qualifying_age
    kind: derived
    entity: Person
    dtype: Judgment
    period: Day
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          claimant_age >= qualifying_age
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2002/16/1.test.yaml",
        """- name: claimant at qualifying age
  period:
    period_kind: custom
    name: day
    start: '2026-01-01'
    end: '2026-01-01'
  input: {}
  output:
    uk:statutes/ukpga/2002/16/1#qualifying_age: 66
    uk:statutes/ukpga/2002/16/1#claimant_has_attained_qualifying_age: holds
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pension_credit")

    assert report["status_counts"] == {"comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    qualifying_age = items_by_id["uk:statutes/ukpga/2002/16/1#qualifying_age"]
    attained_age = items_by_id[
        "uk:statutes/ukpga/2002/16/1#claimant_has_attained_qualifying_age"
    ]

    assert qualifying_age["policyengine_variable"] == "state_pension_age"
    assert qualifying_age["status"] == "comparable"
    assert qualifying_age["mapping_type"] == "direct_variable"
    assert qualifying_age["candidate_priority"] == "P3"
    assert attained_age["policyengine_variable"] == "is_SP_age"
    assert attained_age["status"] == "comparable"
    assert attained_age["mapping_type"] == "direct_variable"
    assert attained_age["candidate_priority"] == "P3"


def test_policyengine_coverage_classifies_uk_pension_credit_section_3_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2002/16/3.yaml",
        """format: rulespec/v1
rules:
  - name: maximum_savings_credit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          10
  - name: amount_a_for_savings_credit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          9
  - name: amount_b_for_savings_credit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          2
  - name: savings_credit_second_condition_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          amount_a_for_savings_credit > amount_b_for_savings_credit
  - name: savings_credit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '0001-01-01'
        formula: |-
          amount_a_for_savings_credit - amount_b_for_savings_credit
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2002/16/3.test.yaml",
        """- name: savings credit
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:statutes/ukpga/2002/16/3#maximum_savings_credit: 10
    uk:statutes/ukpga/2002/16/3#amount_a_for_savings_credit: 9
    uk:statutes/ukpga/2002/16/3#amount_b_for_savings_credit: 2
    uk:statutes/ukpga/2002/16/3#savings_credit_second_condition_satisfied: holds
    uk:statutes/ukpga/2002/16/3#savings_credit: 7
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pension_credit")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 4,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    savings_credit = items_by_id["uk:statutes/ukpga/2002/16/3#savings_credit"]

    assert savings_credit["policyengine_variable"] == "savings_credit"
    assert savings_credit["status"] == "comparable"
    assert savings_credit["mapping_type"] == "direct_variable"
    assert savings_credit["candidate_priority"] == "P2"
    assert (
        items_by_id["uk:statutes/ukpga/2002/16/3#amount_a_for_savings_credit"]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_pension_credit_regulation_15_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/1792/15.yaml",
        """format: rulespec/v1
rules:
  - name: capital_treated_as_yielding_weekly_income
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2009-11-02'
        formula: claimant_capital > capital_deemed_income_lower_threshold
  - name: capital_deemed_weekly_income
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2009-11-02'
        formula: capital_deemed_income_weekly_amount_per_band
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/1792/15.test.yaml",
        """- name: deemed income
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2002/1792/15#capital_treated_as_yielding_weekly_income: holds
    uk:regulations/uksi/2002/1792/15#capital_deemed_weekly_income: 2
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pension_credit")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    deemed_income = items_by_id[
        "uk:regulations/uksi/2002/1792/15#capital_deemed_weekly_income"
    ]
    helper = items_by_id[
        "uk:regulations/uksi/2002/1792/15#capital_treated_as_yielding_weekly_income"
    ]

    assert deemed_income["policyengine_variable"] == "pension_credit_deemed_income"
    assert deemed_income["status"] == "comparable"
    assert deemed_income["mapping_type"] == "direct_variable"
    assert deemed_income["tested"] is True
    assert helper["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_uk_pension_credit_schedule_iia_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/1792/schedule/IIA.yaml",
        """format: rulespec/v1
rules:
  - name: child_or_qualifying_young_person_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 69.98
  - name: first_child_or_qualifying_young_person_born_before_6_april_2017_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 81.07
  - name: disabled_child_or_qualifying_young_person_further_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 37.93
  - name: severely_disabled_child_or_qualifying_young_person_further_weekly_amount
    kind: parameter
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 118.46
  - name: schedule_applies_to_claimant
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: claimant_responsible_child_or_qualifying_young_person_count > 0
  - name: additional_amount_applicable
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: child_or_qualifying_young_person_weekly_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/1792/schedule/IIA.test.yaml",
        """- name: child addition
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2002/1792/schedule/IIA#child_or_qualifying_young_person_weekly_amount: 69.98
    uk:regulations/uksi/2002/1792/schedule/IIA#first_child_or_qualifying_young_person_born_before_6_april_2017_weekly_amount: 81.07
    uk:regulations/uksi/2002/1792/schedule/IIA#disabled_child_or_qualifying_young_person_further_weekly_amount: 37.93
    uk:regulations/uksi/2002/1792/schedule/IIA#severely_disabled_child_or_qualifying_young_person_further_weekly_amount: 118.46
    uk:regulations/uksi/2002/1792/schedule/IIA#additional_amount_applicable: 69.98
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pension_credit")

    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    additional_amount = items_by_id[
        "uk:regulations/uksi/2002/1792/schedule/IIA#additional_amount_applicable"
    ]
    child_amount = items_by_id[
        "uk:regulations/uksi/2002/1792/schedule/IIA#child_or_qualifying_young_person_weekly_amount"
    ]
    helper = items_by_id[
        "uk:regulations/uksi/2002/1792/schedule/IIA#schedule_applies_to_claimant"
    ]

    assert (
        additional_amount["policyengine_variable"] == "child_minimum_guarantee_addition"
    )
    assert additional_amount["status"] == "comparable"
    assert additional_amount["mapping_type"] == "direct_variable"
    assert additional_amount["tested"] is True
    assert (
        child_amount["policyengine_parameter"]
        == "gov.dwp.pension_credit.guarantee_credit.child.addition"
    )
    assert child_amount["mapping_type"] == "parameter_value"
    assert helper["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_uk_wtc_schedule_2_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/2005/schedule/2.yaml",
        """format: rulespec/v1
rules:
  - name: wtc_basic_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 2435
  - name: wtc_disabled_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 3935
  - name: schedule_2_row_3_maximum_annual_rate
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 1015
  - name: wtc_second_adult_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 2500
  - name: wtc_lone_parent_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 2500
  - name: wtc_severely_disabled_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 1705
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/2005/schedule/2.test.yaml",
        """- name: wtc schedule 2
  period:
    period_kind: tax_year
    start: '2024-04-06'
    end: '2025-04-05'
  input: {}
  output:
    uk:regulations/uksi/2002/2005/schedule/2#wtc_basic_element_amount: 2435
    uk:regulations/uksi/2002/2005/schedule/2#wtc_disabled_element_amount: 3935
    uk:regulations/uksi/2002/2005/schedule/2#schedule_2_row_3_maximum_annual_rate: 1015
    uk:regulations/uksi/2002/2005/schedule/2#wtc_second_adult_element_amount: 2500
    uk:regulations/uksi/2002/2005/schedule/2#wtc_lone_parent_element_amount: 2500
    uk:regulations/uksi/2002/2005/schedule/2#wtc_severely_disabled_element_amount: 1705
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="working_tax_credit")

    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    basic = items_by_id[
        "uk:regulations/uksi/2002/2005/schedule/2#wtc_basic_element_amount"
    ]
    second_adult = items_by_id[
        "uk:regulations/uksi/2002/2005/schedule/2#wtc_second_adult_element_amount"
    ]
    row_3 = items_by_id[
        "uk:regulations/uksi/2002/2005/schedule/2#schedule_2_row_3_maximum_annual_rate"
    ]

    assert (
        basic["policyengine_parameter"]
        == "gov.dwp.tax_credits.working_tax_credit.elements.basic"
    )
    assert basic["mapping_type"] == "parameter_value"
    assert basic["tested"] is True
    assert (
        second_adult["policyengine_parameter"]
        == "gov.dwp.tax_credits.working_tax_credit.elements.couple"
    )
    assert row_3["status"] == "known_not_comparable"
    assert (
        row_3["policyengine_parameter"]
        == "gov.dwp.tax_credits.working_tax_credit.elements.worker"
    )


def test_policyengine_coverage_classifies_uk_child_tax_credit_element_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/2007/7.yaml",
        """format: rulespec/v1
rules:
  - name: ctc_family_element_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 545
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2002/2007/7.test.yaml",
        """- name: ctc family element
  period:
    period_kind: tax_year
    start: '2024-04-06'
    end: '2025-04-05'
  input: {}
  output:
    uk:regulations/uksi/2002/2007/7#ctc_family_element_amount: 545
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2024/247/3.yaml",
        """format: rulespec/v1
rules:
  - name: ctc_individual_element_substituted_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 3455
  - name: ctc_disabled_child_element_substituted_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 4170
  - name: ctc_severely_disabled_child_rate_substituted_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: 5850
  - name: ctc_individual_element_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: ctc_individual_element_substituted_amount
  - name: ctc_disabled_child_element_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: ctc_disabled_child_element_substituted_amount
  - name: ctc_severely_disabled_child_rate_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: ctc_severely_disabled_child_rate_substituted_amount
  - name: ctc_severely_disabled_child_additional_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2024-04-06'
        formula: ctc_severely_disabled_child_rate_amount - ctc_disabled_child_element_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2024/247/3.test.yaml",
        """- name: ctc uprating
  period:
    period_kind: tax_year
    start: '2024-04-06'
    end: '2025-04-05'
  input: {}
  output:
    uk:regulations/uksi/2024/247/3#ctc_individual_element_substituted_amount: 3455
    uk:regulations/uksi/2024/247/3#ctc_disabled_child_element_substituted_amount: 4170
    uk:regulations/uksi/2024/247/3#ctc_severely_disabled_child_rate_substituted_amount: 5850
    uk:regulations/uksi/2024/247/3#ctc_individual_element_amount: 3455
    uk:regulations/uksi/2024/247/3#ctc_disabled_child_element_amount: 4170
    uk:regulations/uksi/2024/247/3#ctc_severely_disabled_child_rate_amount: 5850
    uk:regulations/uksi/2024/247/3#ctc_severely_disabled_child_additional_amount: 1680
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="child_tax_credit")

    assert report["status_counts"] == {
        "comparable": 6,
        "known_not_comparable": 2,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    family = items_by_id["uk:regulations/uksi/2002/2007/7#ctc_family_element_amount"]
    child = items_by_id["uk:regulations/uksi/2024/247/3#ctc_individual_element_amount"]
    severe_rate = items_by_id[
        "uk:regulations/uksi/2024/247/3#ctc_severely_disabled_child_rate_amount"
    ]
    severe_additional = items_by_id[
        "uk:regulations/uksi/2024/247/3#ctc_severely_disabled_child_additional_amount"
    ]

    assert (
        family["policyengine_parameter"]
        == "gov.dwp.tax_credits.child_tax_credit.elements.family_element"
    )
    assert (
        child["policyengine_parameter"]
        == "gov.dwp.tax_credits.child_tax_credit.elements.child_element"
    )
    assert severe_rate["status"] == "known_not_comparable"
    assert (
        severe_additional["policyengine_parameter"]
        == "gov.dwp.tax_credits.child_tax_credit.elements.severe_dis_child_element"
    )
    assert severe_additional["mapping_type"] == "parameter_value"
    assert severe_additional["tested"] is True


def test_policyengine_coverage_classifies_uk_pip_rate_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/377/24.yaml",
        """format: rulespec/v1
rules:
  - name: pip_daily_living_standard_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 76.70
  - name: pip_daily_living_enhanced_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 114.60
  - name: pip_mobility_standard_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 30.30
  - name: pip_mobility_enhanced_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 80.00
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/377/24.test.yaml",
        """- name: pip rates
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2013/377/24#pip_daily_living_standard_weekly_rate: 76.70
    uk:regulations/uksi/2013/377/24#pip_daily_living_enhanced_weekly_rate: 114.60
    uk:regulations/uksi/2013/377/24#pip_mobility_standard_weekly_rate: 30.30
    uk:regulations/uksi/2013/377/24#pip_mobility_enhanced_weekly_rate: 80.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pip")

    assert report["status_counts"] == {"comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    daily_standard = items_by_id[
        "uk:regulations/uksi/2013/377/24#pip_daily_living_standard_weekly_rate"
    ]
    daily_enhanced = items_by_id[
        "uk:regulations/uksi/2013/377/24#pip_daily_living_enhanced_weekly_rate"
    ]
    mobility_standard = items_by_id[
        "uk:regulations/uksi/2013/377/24#pip_mobility_standard_weekly_rate"
    ]
    mobility_enhanced = items_by_id[
        "uk:regulations/uksi/2013/377/24#pip_mobility_enhanced_weekly_rate"
    ]

    assert (
        daily_standard["policyengine_parameter"] == "gov.dwp.pip.daily_living.standard"
    )
    assert (
        daily_enhanced["policyengine_parameter"] == "gov.dwp.pip.daily_living.enhanced"
    )
    assert (
        mobility_standard["policyengine_parameter"] == "gov.dwp.pip.mobility.standard"
    )
    assert (
        mobility_enhanced["policyengine_parameter"] == "gov.dwp.pip.mobility.enhanced"
    )
    assert daily_standard["mapping_type"] == "parameter_value"
    assert daily_standard["tested"] is True


def test_policyengine_coverage_classifies_uk_pip_component_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/78.yaml",
        """format: rulespec/v1
rules:
  - name: pip_daily_living_enhanced_rate_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: enhanced_condition
  - name: pip_daily_living_standard_rate_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: standard_condition
  - name: pip_daily_living_component_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 114.60
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/79.yaml",
        """format: rulespec/v1
rules:
  - name: pip_mobility_standard_rate_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: standard_condition
  - name: pip_mobility_enhanced_rate_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: enhanced_condition
  - name: pip_mobility_component_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 30.30
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/77.yaml",
        """format: rulespec/v1
rules:
  - name: personal_independence_payment_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 144.90
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/78.test.yaml",
        """- name: daily living amount
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input:
    enhanced_condition: true
  output:
    uk:statutes/ukpga/2012/5/78#pip_daily_living_enhanced_rate_entitlement: holds
    uk:statutes/ukpga/2012/5/78#pip_daily_living_component_weekly_amount: 114.60
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/79.test.yaml",
        """- name: mobility amount
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:statutes/ukpga/2012/5/79#pip_mobility_component_weekly_amount: 30.30
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/77.test.yaml",
        """- name: total amount
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:statutes/ukpga/2012/5/77#personal_independence_payment_weekly_amount: 144.90
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pip")

    assert report["status_counts"] == {
        "comparable": 4,
        "known_not_comparable": 3,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/78#pip_daily_living_enhanced_rate_entitlement"
        ]["policyengine_variable"]
        == "receives_enhanced_pip_dl"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/78#pip_daily_living_component_weekly_amount"
        ]["policyengine_variable"]
        == "pip_dl"
    )
    assert (
        items_by_id["uk:statutes/ukpga/2012/5/79#pip_mobility_component_weekly_amount"][
            "policyengine_variable"
        ]
        == "pip_m"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/77#personal_independence_payment_weekly_amount"
        ]["policyengine_variable"]
        == "pip"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/78#pip_daily_living_standard_rate_entitlement"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/79#pip_mobility_standard_rate_entitlement"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "uk:statutes/ukpga/2012/5/79#pip_mobility_enhanced_rate_entitlement"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_tax_free_childcare_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2014/28/1.yaml",
        """format: rulespec/v1
rules:
  - name: tax_free_childcare_top_up_payment_rate
    kind: parameter
    dtype: Decimal
    period: Year
    versions:
      - effective_from: '2017-04-21'
        formula: '0.25'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2014/28/21.yaml",
        """format: rulespec/v1
rules:
  - name: tax_free_childcare_top_up_element_rate
    kind: derived
    dtype: Decimal
    period: Year
    versions:
      - effective_from: '2017-04-21'
        formula: '0.2'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2015/448/15.yaml",
        """format: rulespec/v1
rules:
  - name: tax_free_childcare_maximum_adjusted_net_income
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2015-01-01'
        formula: 100000
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2014/28/1.test.yaml",
        """- name: section_1_top_up_payment_rate
  output:
    uk:statutes/ukpga/2014/28/1#tax_free_childcare_top_up_payment_rate: 0.25
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2014/28/21.test.yaml",
        """- name: section_21_top_up_element_rate
  output:
    uk:statutes/ukpga/2014/28/21#tax_free_childcare_top_up_element_rate: 0.2
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2015/448/15.test.yaml",
        """- name: regulation_15_maximum_adjusted_net_income
  output:
    uk:regulations/uksi/2015/448/15#tax_free_childcare_maximum_adjusted_net_income: 100000
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="tax_free_childcare",
    )

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 1,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    payment_rate = items_by_id[
        "uk:statutes/ukpga/2014/28/1#tax_free_childcare_top_up_payment_rate"
    ]
    element_rate = items_by_id[
        "uk:statutes/ukpga/2014/28/21#tax_free_childcare_top_up_element_rate"
    ]
    income_limit = items_by_id[
        "uk:regulations/uksi/2015/448/15#tax_free_childcare_maximum_adjusted_net_income"
    ]

    assert payment_rate["status"] == "known_not_comparable"
    assert (
        element_rate["policyengine_parameter"]
        == "gov.hmrc.tax_free_childcare.contribution.rate"
    )
    assert (
        income_limit["policyengine_parameter"]
        == "gov.hmrc.tax_free_childcare.income.income_limit"
    )
    assert element_rate["mapping_type"] == "parameter_value"
    assert income_limit["tested"] is True


def test_policyengine_coverage_classifies_uk_sure_start_maternity_grant_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2005/3061/5.yaml",
        """format: rulespec/v1
rules:
  - name: sure_start_maternity_grant_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2005-12-05'
        formula: 500
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2005/3061/5.test.yaml",
        """- name: sure_start_maternity_grant_2026_amount
  output:
    uk:regulations/uksi/2005/3061/5#sure_start_maternity_grant_amount: 500
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="ssmg")

    assert report["status_counts"] == {"comparable": 1}
    assert report["untested_comparable"] == 0
    item = report["items"][0]
    assert item["policyengine_parameter"] == "gov.dwp.ssmg.rate"
    assert item["mapping_type"] == "parameter_value"
    assert item["tested"] is True


def test_policyengine_coverage_classifies_uk_scottish_child_payment_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2020/351/20.yaml",
        """format: rulespec/v1
rules:
  - name: scottish_child_payment_weekly_amount
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-04-01'
        formula: 28.20
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2020/351/18.yaml",
        """format: rulespec/v1
rules:
  - name: scottish_child_payment_maximum_child_age
    kind: parameter
    dtype: Integer
    period: Year
    unit: year
    versions:
      - effective_from: '2022-11-14'
        formula: 16
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2020/351/20.test.yaml",
        """- name: scottish_child_payment_2026_weekly_amount
  output:
    uk:regulations/ssi/2020/351/20#scottish_child_payment_weekly_amount: 28.20
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2020/351/18.test.yaml",
        """- name: scottish_child_payment_maximum_child_age
  output:
    uk:regulations/ssi/2020/351/18#scottish_child_payment_maximum_child_age: 16
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="scottish_child_payment",
    )

    assert report["status_counts"] == {"comparable": 2}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    weekly_amount = items_by_id[
        "uk:regulations/ssi/2020/351/20#scottish_child_payment_weekly_amount"
    ]
    maximum_age = items_by_id[
        "uk:regulations/ssi/2020/351/18#scottish_child_payment_maximum_child_age"
    ]

    assert (
        weekly_amount["policyengine_parameter"]
        == "gov.social_security_scotland.scottish_child_payment.amount"
    )
    assert (
        maximum_age["policyengine_parameter"]
        == "gov.social_security_scotland.scottish_child_payment.max_age"
    )
    assert weekly_amount["mapping_type"] == "parameter_value"
    assert maximum_age["mapping_type"] == "parameter_value"
    assert weekly_amount["tested"] is True
    assert maximum_age["tested"] is True


def test_policyengine_coverage_classifies_uk_carer_support_payment_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2023/302/5.yaml",
        """format: rulespec/v1
rules:
  - name: carer_support_payment_minimum_weekly_care_hours
    kind: parameter
    dtype: Count
    period: Week
    unit: hour
    versions:
      - effective_from: '2024-11-01'
        formula: 35
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2023/302/16.yaml",
        """format: rulespec/v1
rules:
  - name: carer_support_payment_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-04-05'
        formula: 86.45
  - name: scottish_carer_supplement_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-04-05'
        formula: 11.70
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2023/302/5.test.yaml",
        """- name: carer_support_payment_minimum_weekly_care_hours
  output:
    uk:regulations/ssi/2023/302/5#carer_support_payment_minimum_weekly_care_hours: 35
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/ssi/2023/302/16.test.yaml",
        """- name: carer_support_payment_2026_weekly_rates
  output:
    uk:regulations/ssi/2023/302/16#carer_support_payment_weekly_rate: 86.45
    uk:regulations/ssi/2023/302/16#scottish_carer_supplement_weekly_rate: 11.70
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="carer_support_payment",
    )

    assert report["status_counts"] == {"comparable": 3}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    min_hours = items_by_id[
        "uk:regulations/ssi/2023/302/5#carer_support_payment_minimum_weekly_care_hours"
    ]
    rate = items_by_id[
        "uk:regulations/ssi/2023/302/16#carer_support_payment_weekly_rate"
    ]
    supplement = items_by_id[
        "uk:regulations/ssi/2023/302/16#scottish_carer_supplement_weekly_rate"
    ]

    assert (
        min_hours["policyengine_parameter"]
        == "gov.social_security_scotland.carer_support_payment.min_hours"
    )
    assert (
        rate["policyengine_parameter"]
        == "gov.social_security_scotland.carer_support_payment.rate"
    )
    assert (
        supplement["policyengine_parameter"]
        == "gov.social_security_scotland.carer_support_payment.supplement"
    )
    assert min_hours["mapping_type"] == "parameter_value"
    assert rate["mapping_type"] == "parameter_value"
    assert supplement["mapping_type"] == "parameter_value"
    assert min_hours["tested"] is True
    assert rate["tested"] is True
    assert supplement["tested"] is True


def test_policyengine_coverage_classifies_uk_cost_of_living_support_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2022/38/1.yaml",
        """format: rulespec/v1
rules:
  - name: first_means_tested_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2022-06-28'
        formula: 326
  - name: second_means_tested_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2022-06-28'
        formula: 324
  - name: means_tested_additional_payment_total
    kind: derived
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2022-06-28'
        formula: first_means_tested_additional_payment_amount + second_means_tested_additional_payment_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2022/38/5.yaml",
        """format: rulespec/v1
rules:
  - name: disability_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2022-06-28'
        formula: 150
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2023/7/1.yaml",
        """format: rulespec/v1
rules:
  - name: first_means_tested_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2023-03-23'
        formula: 301
  - name: second_means_tested_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2023-03-23'
        formula: 300
  - name: third_means_tested_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2023-03-23'
        formula: 299
  - name: means_tested_additional_payment_total
    kind: derived
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2023-03-23'
        formula: first_means_tested_additional_payment_amount + second_means_tested_additional_payment_amount + third_means_tested_additional_payment_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2023/7/5.yaml",
        """format: rulespec/v1
rules:
  - name: disability_additional_payment_amount
    kind: parameter
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2023-03-23'
        formula: 150
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2022/38/1.test.yaml",
        """- name: means_tested_additional_payment_total_2022
  output:
    uk:statutes/ukpga/2022/38/1#first_means_tested_additional_payment_amount: 326
    uk:statutes/ukpga/2022/38/1#second_means_tested_additional_payment_amount: 324
    uk:statutes/ukpga/2022/38/1#means_tested_additional_payment_total: 650
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2022/38/5.test.yaml",
        """- name: disability_additional_payment_amount_2022
  output:
    uk:statutes/ukpga/2022/38/5#disability_additional_payment_amount: 150
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2023/7/1.test.yaml",
        """- name: means_tested_additional_payment_total_2023
  output:
    uk:statutes/ukpga/2023/7/1#first_means_tested_additional_payment_amount: 301
    uk:statutes/ukpga/2023/7/1#second_means_tested_additional_payment_amount: 300
    uk:statutes/ukpga/2023/7/1#third_means_tested_additional_payment_amount: 299
    uk:statutes/ukpga/2023/7/1#means_tested_additional_payment_total: 900
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2023/7/5.test.yaml",
        """- name: disability_additional_payment_amount_2023
  output:
    uk:statutes/ukpga/2023/7/5#disability_additional_payment_amount: 150
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="cost_of_living_support_payment",
    )

    assert report["status_counts"] == {
        "comparable": 4,
        "known_not_comparable": 5,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}

    assert (
        items_by_id[
            "uk:statutes/ukpga/2022/38/1#means_tested_additional_payment_total"
        ]["policyengine_parameter"]
        == "gov.treasury.cost_of_living_support.means_tested_households.amount"
    )
    assert (
        items_by_id["uk:statutes/ukpga/2023/7/1#means_tested_additional_payment_total"][
            "policyengine_parameter"
        ]
        == "gov.treasury.cost_of_living_support.means_tested_households.amount"
    )
    assert (
        items_by_id["uk:statutes/ukpga/2022/38/5#disability_additional_payment_amount"][
            "policyengine_parameter"
        ]
        == "gov.treasury.cost_of_living_support.disabled.amount"
    )
    assert (
        items_by_id["uk:statutes/ukpga/2023/7/5#disability_additional_payment_amount"][
            "policyengine_parameter"
        ]
        == "gov.treasury.cost_of_living_support.disabled.amount"
    )
    comparable_items = [
        item for item in items_by_id.values() if item["status"] == "comparable"
    ]
    helper_items = [
        item
        for item in items_by_id.values()
        if item["status"] == "known_not_comparable"
    ]
    assert {item["mapping_type"] for item in comparable_items} == {"parameter_value"}
    assert {item["mapping_type"] for item in helper_items} == {"not_comparable"}
    assert all(item["tested"] is True for item in comparable_items)


def test_policyengine_coverage_classifies_uk_schedule_1_benefit_rate_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/schedule/1.yaml",
        """format: rulespec/v1
rules:
  - name: attendance_allowance_higher_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 114.60
  - name: attendance_allowance_lower_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 76.70
  - name: severe_disablement_allowance_basic_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 103.85
  - name: sda_age_related_addition_higher_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 15.50
  - name: sda_age_related_addition_middle_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 8.60
  - name: sda_age_related_addition_lower_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 8.60
  - name: severe_disablement_allowance_maximum_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: severe_disablement_allowance_basic_weekly_rate + sda_age_related_addition_higher_weekly_rate
  - name: carers_allowance_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 86.45
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/schedule/1.test.yaml",
        """- name: schedule 1 benefit rates
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2026/148/schedule/1#attendance_allowance_higher_weekly_rate: 114.60
    uk:regulations/uksi/2026/148/schedule/1#attendance_allowance_lower_weekly_rate: 76.70
    uk:regulations/uksi/2026/148/schedule/1#severe_disablement_allowance_basic_weekly_rate: 103.85
    uk:regulations/uksi/2026/148/schedule/1#sda_age_related_addition_higher_weekly_rate: 15.50
    uk:regulations/uksi/2026/148/schedule/1#sda_age_related_addition_middle_weekly_rate: 8.60
    uk:regulations/uksi/2026/148/schedule/1#sda_age_related_addition_lower_weekly_rate: 8.60
    uk:regulations/uksi/2026/148/schedule/1#severe_disablement_allowance_maximum_weekly_rate: 119.35
    uk:regulations/uksi/2026/148/schedule/1#carers_allowance_weekly_rate: 86.45
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["status_counts"] == {
        "comparable": 4,
        "known_not_comparable": 4,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    attendance_higher = items_by_id[
        "uk:regulations/uksi/2026/148/schedule/1#attendance_allowance_higher_weekly_rate"
    ]
    attendance_lower = items_by_id[
        "uk:regulations/uksi/2026/148/schedule/1#attendance_allowance_lower_weekly_rate"
    ]
    sda_basic = items_by_id[
        "uk:regulations/uksi/2026/148/schedule/1#severe_disablement_allowance_basic_weekly_rate"
    ]
    sda_maximum = items_by_id[
        "uk:regulations/uksi/2026/148/schedule/1#severe_disablement_allowance_maximum_weekly_rate"
    ]
    carers_allowance = items_by_id[
        "uk:regulations/uksi/2026/148/schedule/1#carers_allowance_weekly_rate"
    ]

    assert (
        attendance_higher["policyengine_parameter"]
        == "gov.dwp.attendance_allowance.higher"
    )
    assert (
        attendance_lower["policyengine_parameter"]
        == "gov.dwp.attendance_allowance.lower"
    )
    assert sda_basic["status"] == "known_not_comparable"
    assert sda_maximum["policyengine_parameter"] == "gov.dwp.sda.maximum"
    assert carers_allowance["policyengine_parameter"] == "gov.dwp.carers_allowance.rate"
    assert attendance_higher["tested"] is True
    assert sda_maximum["tested"] is True
    assert carers_allowance["tested"] is True


def test_policyengine_coverage_classifies_uk_winter_fuel_payment_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2025/969/3.yaml",
        """format: rulespec/v1
rules:
  - name: winter_fuel_payment_under_80_standard_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 200
  - name: winter_fuel_payment_under_80_shared_or_residential_care_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 100
  - name: winter_fuel_payment_under_80_partner_80_relevant_benefit_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 300
  - name: winter_fuel_payment_age_80_standard_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 300
  - name: winter_fuel_payment_age_80_shared_under_80_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 200
  - name: winter_fuel_payment_age_80_shared_80_or_residential_care_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 150
  - name: winter_fuel_payment_higher_age_requirement
    kind: parameter
    dtype: Count
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: 80
  - name: winter_fuel_payment_lower_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: winter_fuel_payment_under_80_standard_amount
  - name: winter_fuel_payment_higher_amount
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2025-09-15'
        formula: winter_fuel_payment_age_80_standard_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2025/969/3.test.yaml",
        """- name: winter fuel payment amounts
  period:
    period_kind: custom
    name: calendar_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_under_80_standard_amount: 200
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_under_80_shared_or_residential_care_amount: 100
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_under_80_partner_80_relevant_benefit_amount: 300
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_age_80_standard_amount: 300
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_age_80_shared_under_80_amount: 200
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_age_80_shared_80_or_residential_care_amount: 150
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_higher_age_requirement: 80
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_lower_amount: 200
    uk:regulations/uksi/2025/969/3#winter_fuel_payment_higher_amount: 300
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path, program="winter_fuel_allowance"
    )

    assert report["status_counts"] == {
        "comparable": 5,
        "known_not_comparable": 4,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    lower = items_by_id[
        "uk:regulations/uksi/2025/969/3#winter_fuel_payment_lower_amount"
    ]
    higher = items_by_id[
        "uk:regulations/uksi/2025/969/3#winter_fuel_payment_higher_amount"
    ]
    age = items_by_id[
        "uk:regulations/uksi/2025/969/3#winter_fuel_payment_higher_age_requirement"
    ]
    shared = items_by_id[
        "uk:regulations/uksi/2025/969/3#winter_fuel_payment_under_80_shared_or_residential_care_amount"
    ]

    assert lower["policyengine_parameter"] == "gov.dwp.winter_fuel_payment.amount.lower"
    assert (
        higher["policyengine_parameter"] == "gov.dwp.winter_fuel_payment.amount.higher"
    )
    assert (
        age["policyengine_parameter"]
        == "gov.dwp.winter_fuel_payment.eligibility.higher_age_requirement"
    )
    assert shared["status"] == "known_not_comparable"
    assert lower["tested"] is True
    assert higher["tested"] is True
    assert age["tested"] is True


def test_policyengine_coverage_classifies_uk_dla_rate_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/14.yaml",
        """format: rulespec/v1
rules:
  - name: dla_self_care_higher_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 114.60
  - name: dla_self_care_middle_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 76.70
  - name: dla_self_care_lower_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 30.30
  - name: dla_mobility_higher_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 80.00
  - name: dla_mobility_lower_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 30.30
  - name: dla_self_care_higher_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: dla_self_care_higher_substituted_amount
  - name: dla_self_care_middle_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: dla_self_care_middle_substituted_amount
  - name: dla_self_care_lower_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: dla_self_care_lower_substituted_amount
  - name: dla_mobility_higher_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: dla_mobility_higher_substituted_amount
  - name: dla_mobility_lower_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: dla_mobility_lower_substituted_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/14.test.yaml",
        """- name: article 14 dla rates
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2026/148/article/14#dla_self_care_higher_substituted_amount: 114.60
    uk:regulations/uksi/2026/148/article/14#dla_self_care_middle_substituted_amount: 76.70
    uk:regulations/uksi/2026/148/article/14#dla_self_care_lower_substituted_amount: 30.30
    uk:regulations/uksi/2026/148/article/14#dla_mobility_higher_substituted_amount: 80.00
    uk:regulations/uksi/2026/148/article/14#dla_mobility_lower_substituted_amount: 30.30
    uk:regulations/uksi/2026/148/article/14#dla_self_care_higher_weekly_rate: 114.60
    uk:regulations/uksi/2026/148/article/14#dla_self_care_middle_weekly_rate: 76.70
    uk:regulations/uksi/2026/148/article/14#dla_self_care_lower_weekly_rate: 30.30
    uk:regulations/uksi/2026/148/article/14#dla_mobility_higher_weekly_rate: 80.00
    uk:regulations/uksi/2026/148/article/14#dla_mobility_lower_weekly_rate: 30.30
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="dla")

    assert report["status_counts"] == {"comparable": 10}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "uk:regulations/uksi/2026/148/article/14#dla_self_care_higher_weekly_rate"
        ]["policyengine_parameter"]
        == "gov.dwp.dla.self_care.higher"
    )
    assert (
        items_by_id[
            "uk:regulations/uksi/2026/148/article/14#dla_self_care_middle_weekly_rate"
        ]["policyengine_parameter"]
        == "gov.dwp.dla.self_care.middle"
    )
    assert (
        items_by_id[
            "uk:regulations/uksi/2026/148/article/14#dla_self_care_lower_weekly_rate"
        ]["policyengine_parameter"]
        == "gov.dwp.dla.self_care.lower"
    )
    assert (
        items_by_id[
            "uk:regulations/uksi/2026/148/article/14#dla_mobility_higher_weekly_rate"
        ]["policyengine_parameter"]
        == "gov.dwp.dla.mobility.higher"
    )
    assert (
        items_by_id[
            "uk:regulations/uksi/2026/148/article/14#dla_mobility_lower_weekly_rate"
        ]["policyengine_parameter"]
        == "gov.dwp.dla.mobility.lower"
    )
    assert all(item["mapping_type"] == "parameter_value" for item in report["items"])
    assert all(item["tested"] is True for item in report["items"])


def test_policyengine_coverage_classifies_uk_tv_licence_fee_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2004/692/schedule/1.yaml",
        """format: rulespec/v1
rules:
  - name: colour_tv_licence_general_form_issue_fee
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-04-01'
        formula: 180.00
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2004/692/schedule/1.test.yaml",
        """- name: tv licence fee
  period:
    period_kind: custom
    name: licence_year
    start: '2026-04-01'
    end: '2027-03-31'
  input: {}
  output:
    uk:regulations/uksi/2004/692/schedule/1#colour_tv_licence_general_form_issue_fee: 180.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tv_licence")

    assert report["status_counts"] == {"comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "uk:regulations/uksi/2004/692/schedule/1#colour_tv_licence_general_form_issue_fee"
    )
    assert item["policyengine_parameter"] == "gov.dcms.bbc.tv_licence.colour"
    assert item["mapping_type"] == "parameter_value"
    assert item["tested"] is True


def test_policyengine_coverage_classifies_uk_state_pension_rate_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/4.yaml",
        """format: rulespec/v1
rules:
  - name: category_a_basic_retirement_pension_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 184.90
  - name: category_a_basic_retirement_pension_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: category_a_basic_retirement_pension_substituted_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/4.test.yaml",
        """- name: article 4 basic state pension rate
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2026/148/article/4#category_a_basic_retirement_pension_substituted_amount: 184.90
    uk:regulations/uksi/2026/148/article/4#category_a_basic_retirement_pension_weekly_rate: 184.90
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/6.yaml",
        """format: rulespec/v1
rules:
  - name: full_new_state_pension_substituted_amount
    kind: parameter
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 241.30
  - name: full_new_state_pension_weekly_rate
    kind: derived
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: full_new_state_pension_substituted_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2026/148/article/6.test.yaml",
        """- name: article 6 new state pension rate
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:regulations/uksi/2026/148/article/6#full_new_state_pension_substituted_amount: 241.30
    uk:regulations/uksi/2026/148/article/6#full_new_state_pension_weekly_rate: 241.30
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="state_pension")

    assert report["status_counts"] == {"comparable": 4}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    basic_rate = items_by_id[
        "uk:regulations/uksi/2026/148/article/4#category_a_basic_retirement_pension_weekly_rate"
    ]
    new_rate = items_by_id[
        "uk:regulations/uksi/2026/148/article/6#full_new_state_pension_weekly_rate"
    ]
    assert (
        basic_rate["policyengine_parameter"]
        == "gov.dwp.state_pension.basic_state_pension.amount"
    )
    assert (
        new_rate["policyengine_parameter"]
        == "gov.dwp.state_pension.new_state_pension.amount"
    )
    assert all(item["mapping_type"] == "parameter_value" for item in report["items"])
    assert all(item["tested"] is True for item in report["items"])


def test_policyengine_coverage_classifies_uk_state_pension_final_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/state-pension.yaml",
        """format: rulespec/v1
rules:
  - name: current_state_pension_flat_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 184.90
  - name: additional_state_pension_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: 15.10
  - name: state_pension_weekly_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    versions:
      - effective_from: '2026-04-06'
        formula: current_state_pension_flat_weekly_amount + additional_state_pension_weekly_amount
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/state-pension.test.yaml",
        """- name: state pension final wrapper
  period:
    period_kind: week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {}
  output:
    uk:policies/govuk/state-pension#additional_state_pension_weekly_amount: 15.10
    uk:policies/govuk/state-pension#state_pension_weekly_amount: 200.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="state_pension")

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 1,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "uk:policies/govuk/state-pension#additional_state_pension_weekly_amount"
        ]["policyengine_variable"]
        == "additional_state_pension"
    )
    assert (
        items_by_id["uk:policies/govuk/state-pension#state_pension_weekly_amount"][
            "policyengine_variable"
        ]
        == "state_pension"
    )
    assert (
        items_by_id[
            "uk:policies/govuk/state-pension#current_state_pension_flat_weekly_amount"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_uk_national_insurance_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/1.yaml",
        """format: rulespec/v1
rules:
  - name: national_insurance_contribution
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-04-06'
        formula: primary_class_1_contribution_for_year + class_4_contribution_for_year
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/1.test.yaml",
        """- name: national insurance final aggregate
  period:
    period_kind: tax_year
    start: '2026-04-06'
    end: '2027-04-05'
  input: {}
  output:
    uk:statutes/ukpga/1992/4/1#national_insurance_contribution: 2900.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 1}
    assert report["untested_comparable"] == 0
    item = report["items"][0]
    assert item["legal_id"] == (
        "uk:statutes/ukpga/1992/4/1#national_insurance_contribution"
    )
    assert item["policyengine_variable"] == "national_insurance"
    assert item["tested"] is True


def test_policyengine_coverage_classifies_uk_universal_credit_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/universal-credit.yaml",
        """format: rulespec/v1
rules:
  - name: universal_credit_annual_amount
    kind: derived
    entity: Family
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, universal_credit_pre_benefit_cap_for_year - benefit_cap_reduction_for_year)
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/universal-credit.test.yaml",
        """- name: universal credit final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/universal-credit#input.universal_credit_pre_benefit_cap_for_year: 12000.00
    uk:policies/govuk/universal-credit#input.benefit_cap_reduction_for_year: 1250.00
  output:
    uk:policies/govuk/universal-credit#universal_credit_annual_amount: 10750.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="universal_credit")

    assert report["status_counts"] == {"comparable": 1}
    assert report["untested_comparable"] == 0
    item = report["items"][0]
    assert item["legal_id"] == (
        "uk:policies/govuk/universal-credit#universal_credit_annual_amount"
    )
    assert item["policyengine_variable"] == "universal_credit"
    assert item["mapping_type"] == "direct_variable"
    assert item["tested"] is True


def test_policyengine_coverage_classifies_uk_carers_allowance_final_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/carers-allowance.yaml",
        """format: rulespec/v1
rules:
  - name: carers_allowance_minimum_weekly_care_hours
    kind: parameter
    entity: Person
    dtype: Count
    period: Week
    unit: hour
    versions:
      - effective_from: '2026-01-01'
        formula: 35
  - name: carers_allowance_weeks_in_year
    kind: parameter
    entity: Person
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 52
  - name: carers_allowance_annual_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if not (person_is_in_scotland and carer_support_payment_replaces_carers_allowance) and (weekly_care_hours >= carers_allowance_minimum_weekly_care_hours or reported_carers_allowance_for_year > 0): carers_allowance_weekly_rate * carers_allowance_weeks_in_year else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/carers-allowance.test.yaml",
        """- name: carers allowance final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/carers-allowance#input.person_is_in_scotland: false
    uk:policies/govuk/carers-allowance#input.carer_support_payment_replaces_carers_allowance: true
    uk:policies/govuk/carers-allowance#input.weekly_care_hours: 35
    uk:policies/govuk/carers-allowance#input.reported_carers_allowance_for_year: 0
  output:
    uk:policies/govuk/carers-allowance#carers_allowance_minimum_weekly_care_hours: 35
    uk:policies/govuk/carers-allowance#carers_allowance_annual_amount: 4495.40
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="carers_allowance")

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 1,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    min_hours = items_by_id[
        "uk:policies/govuk/carers-allowance#carers_allowance_minimum_weekly_care_hours"
    ]
    weeks_in_year = items_by_id[
        "uk:policies/govuk/carers-allowance#carers_allowance_weeks_in_year"
    ]
    annual_amount = items_by_id[
        "uk:policies/govuk/carers-allowance#carers_allowance_annual_amount"
    ]

    assert min_hours["policyengine_parameter"] == "gov.dwp.carers_allowance.min_hours"
    assert min_hours["mapping_type"] == "parameter_value"
    assert min_hours["tested"] is True
    assert weeks_in_year["status"] == "known_not_comparable"
    assert weeks_in_year["mapping_type"] == "not_comparable"
    assert annual_amount["policyengine_variable"] == "carers_allowance"
    assert annual_amount["mapping_type"] == "direct_variable"
    assert annual_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_pension_credit_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/pension-credit.yaml",
        """format: rulespec/v1
rules:
  - name: pension_credit_annual_amount
    kind: derived
    entity: Family
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if person_or_partner_would_claim_pension_credit: pension_credit_entitlement_for_year else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/pension-credit.test.yaml",
        """- name: pension credit final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/pension-credit#input.pension_credit_entitlement_for_year: 3400.00
    uk:policies/govuk/pension-credit#input.person_or_partner_would_claim_pension_credit: true
  output:
    uk:policies/govuk/pension-credit#pension_credit_annual_amount: 3400.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="pension_credit")

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    final_amount = items_by_id[
        "uk:policies/govuk/pension-credit#pension_credit_annual_amount"
    ]

    assert final_amount["policyengine_variable"] == "pension_credit"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_esa_income_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/esa-income.yaml",
        """format: rulespec/v1
rules:
  - name: income_related_esa_annual_amount
    kind: derived
    entity: Family
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if income_related_esa_eligible: max(0, reported_income_related_esa_for_year - income_related_esa_tariff_income_for_year) else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/esa-income.test.yaml",
        """- name: income related ESA final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/esa-income#input.reported_income_related_esa_for_year: 2600.00
    uk:policies/govuk/esa-income#input.income_related_esa_tariff_income_for_year: 520.00
    uk:policies/govuk/esa-income#input.income_related_esa_eligible: true
  output:
    uk:policies/govuk/esa-income#income_related_esa_annual_amount: 2080.00
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="employment_and_support_allowance",
    )

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    final_amount = items_by_id[
        "uk:policies/govuk/esa-income#income_related_esa_annual_amount"
    ]

    assert final_amount["policyengine_variable"] == "esa_income"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_housing_benefit_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/housing-benefit.yaml",
        """format: rulespec/v1
rules:
  - name: housing_benefit_annual_amount
    kind: derived
    entity: Family
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if would_claim_housing_benefit: max(0, housing_benefit_pre_benefit_cap_for_year - benefit_cap_reduction_for_year) else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/housing-benefit.test.yaml",
        """- name: housing benefit final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/housing-benefit#input.housing_benefit_pre_benefit_cap_for_year: 5200.00
    uk:policies/govuk/housing-benefit#input.benefit_cap_reduction_for_year: 1040.00
    uk:policies/govuk/housing-benefit#input.would_claim_housing_benefit: true
  output:
    uk:policies/govuk/housing-benefit#housing_benefit_annual_amount: 4160.00
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="housing_benefit")

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    final_amount = items_by_id[
        "uk:policies/govuk/housing-benefit#housing_benefit_annual_amount"
    ]

    assert final_amount["policyengine_variable"] == "housing_benefit"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_carer_support_payment_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/carer-support-payment.yaml",
        """format: rulespec/v1
rules:
  - name: carer_support_payment_weeks_in_year
    kind: parameter
    entity: Person
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 52
  - name: carer_support_payment_annual_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if person_is_in_scotland and carer_support_payment_in_effect and (weekly_care_hours >= 35 or reported_carers_allowance_for_year > 0): 98.15 * carer_support_payment_weeks_in_year else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/carer-support-payment.test.yaml",
        """- name: carer support payment final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/carer-support-payment#input.person_is_in_scotland: true
    uk:policies/govuk/carer-support-payment#input.carer_support_payment_in_effect: true
    uk:policies/govuk/carer-support-payment#input.weekly_care_hours: 35
    uk:policies/govuk/carer-support-payment#input.reported_carers_allowance_for_year: 0
  output:
    uk:policies/govuk/carer-support-payment#carer_support_payment_weeks_in_year: 52
    uk:policies/govuk/carer-support-payment#carer_support_payment_annual_amount: 5103.80
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="carer_support_payment",
    )

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    weeks_in_year = items_by_id[
        "uk:policies/govuk/carer-support-payment#carer_support_payment_weeks_in_year"
    ]
    final_amount = items_by_id[
        "uk:policies/govuk/carer-support-payment#carer_support_payment_annual_amount"
    ]

    assert weeks_in_year["status"] == "known_not_comparable"
    assert weeks_in_year["mapping_type"] == "not_comparable"
    assert final_amount["policyengine_variable"] == "carer_support_payment"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_scottish_child_payment_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/scottish-child-payment.yaml",
        """format: rulespec/v1
rules:
  - name: scottish_child_payment_weeks_in_year
    kind: parameter
    entity: Person
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 52
  - name: scottish_child_payment_annual_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if is_scottish_child_payment_eligible and would_claim_scottish_child_payment: 28.20 * scottish_child_payment_weeks_in_year else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/scottish-child-payment.test.yaml",
        """- name: scottish child payment final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/scottish-child-payment#input.is_scottish_child_payment_eligible: true
    uk:policies/govuk/scottish-child-payment#input.would_claim_scottish_child_payment: true
  output:
    uk:policies/govuk/scottish-child-payment#scottish_child_payment_weeks_in_year: 52
    uk:policies/govuk/scottish-child-payment#scottish_child_payment_annual_amount: 1466.40
""",
    )

    report = build_policyengine_coverage_report(
        tmp_path,
        program="scottish_child_payment",
    )

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    weeks_in_year = items_by_id[
        "uk:policies/govuk/scottish-child-payment#scottish_child_payment_weeks_in_year"
    ]
    final_amount = items_by_id[
        "uk:policies/govuk/scottish-child-payment#scottish_child_payment_annual_amount"
    ]

    assert weeks_in_year["status"] == "known_not_comparable"
    assert weeks_in_year["mapping_type"] == "not_comparable"
    assert final_amount["policyengine_variable"] == "scottish_child_payment"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


def test_policyengine_coverage_classifies_uk_sda_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/severe-disablement-allowance.yaml",
        """format: rulespec/v1
rules:
  - name: severe_disablement_allowance_weeks_in_year
    kind: parameter
    entity: Person
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 52
  - name: severe_disablement_allowance_annual_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if reported_severe_disablement_allowance_for_year > 0: 119.35 * severe_disablement_allowance_weeks_in_year else: 0
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "policies/govuk/severe-disablement-allowance.test.yaml",
        """- name: severe disablement allowance final annual amount
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    uk:policies/govuk/severe-disablement-allowance#input.reported_severe_disablement_allowance_for_year: 6206.20
  output:
    uk:policies/govuk/severe-disablement-allowance#severe_disablement_allowance_weeks_in_year: 52
    uk:policies/govuk/severe-disablement-allowance#severe_disablement_allowance_annual_amount: 6206.20
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="sda")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    weeks_in_year = items_by_id[
        "uk:policies/govuk/severe-disablement-allowance#severe_disablement_allowance_weeks_in_year"
    ]
    final_amount = items_by_id[
        "uk:policies/govuk/severe-disablement-allowance#severe_disablement_allowance_annual_amount"
    ]

    assert weeks_in_year["status"] == "known_not_comparable"
    assert weeks_in_year["mapping_type"] == "not_comparable"
    assert final_amount["policyengine_variable"] == "sda"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True


@pytest.mark.parametrize(
    (
        "path",
        "legal_base",
        "program",
        "policyengine_variable",
        "helper_name",
    ),
    [
        (
            "regulations/uksi/2008/794/118",
            "uk:regulations/uksi/2008/794/118",
            "employment_and_support_allowance",
            "esa_income_tariff_income",
            "capital_treated_as_yielding_weekly_income",
        ),
        (
            "regulations/uksi/1996/207/116",
            "uk:regulations/uksi/1996/207/116",
            "jobseekers_allowance",
            "jsa_income_tariff_income",
            "capital_treated_as_yielding_weekly_income",
        ),
        (
            "regulations/uksi/1987/1967/53",
            "uk:regulations/uksi/1987/1967/53",
            "income_support",
            "income_support_tariff_income",
            "capital_treated_as_yielding_weekly_income",
        ),
        (
            "regulations/uksi/2006/213/52",
            "uk:regulations/uksi/2006/213/52",
            "housing_benefit",
            "housing_benefit_tariff_income",
            "capital_treated_as_yielding_weekly_tariff_income",
        ),
        (
            "regulations/uksi/2006/214/29",
            "uk:regulations/uksi/2006/214/29",
            "housing_benefit",
            "housing_benefit_tariff_income",
            "capital_treated_as_yielding_weekly_income",
        ),
    ],
)
def test_policyengine_coverage_classifies_uk_legacy_tariff_income_outputs(
    tmp_path,
    path,
    legal_base,
    program,
    policyengine_variable,
    helper_name,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / f"{path}.yaml",
        f"""format: rulespec/v1
rules:
  - name: {helper_name}
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-01-01'
        formula: claimant_capital > capital_tariff_income_lower_threshold
  - name: capital_tariff_weekly_income
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: capital_tariff_income_weekly_amount_per_band
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / f"{path}.test.yaml",
        f"""- name: tariff income
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-04-06'
    end: '2026-04-12'
  input: {{}}
  output:
    {legal_base}#{helper_name}: holds
    {legal_base}#capital_tariff_weekly_income: 2
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program=program)

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    tariff_income = items_by_id[f"{legal_base}#capital_tariff_weekly_income"]
    helper = items_by_id[f"{legal_base}#{helper_name}"]

    assert tariff_income["policyengine_variable"] == policyengine_variable
    assert tariff_income["status"] == "comparable"
    assert tariff_income["mapping_type"] == "direct_variable"
    assert tariff_income["tested"] is True
    assert helper["status"] == "known_not_comparable"


def test_policyengine_coverage_counts_uk_aliases_as_tested(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2006/965/2.yaml",
        """format: rulespec/v1
rules:
  - name: child_benefit_enhanced_weekly_rate
    kind: parameter
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: 27.05
  - name: child_benefit_weekly_rate
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: child_benefit_enhanced_weekly_rate
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2006/965/2.test.yaml",
        """- name: eldest child rate
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-01-05'
    end: '2026-01-11'
  input: {}
  output:
    uk:regulations/uksi/2006/965/2#child_benefit_weekly_rate: 27.05
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="child_benefit")

    assert report["status_counts"] == {"comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    enhanced_rate = items_by_id[
        "uk:regulations/uksi/2006/965/2#child_benefit_enhanced_weekly_rate"
    ]
    assert (
        enhanced_rate["policyengine_parameter"]
        == "gov.hmrc.child_benefit.amount.eldest"
    )
    assert enhanced_rate["tested"] is True
    assert enhanced_rate["test_output_count"] == 1


def test_policyengine_coverage_classifies_uk_child_benefit_entitlement_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/141.yaml",
        """format: rulespec/v1
rules:
  - name: entitled_to_child_benefit_for_week
    kind: derived
    entity: Family
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-01-01'
        formula: child_count > 0
  - name: child_benefit_weekly_rate_for_responsible_child_or_qualifying_young_person
    kind: derived
    entity: Person
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: child_benefit_weekly_rate
  - name: child_or_qualifying_young_person_counts_for_child_benefit_weekly_entitlement
    kind: derived
    entity: Person
    dtype: Judgment
    period: Week
    versions:
      - effective_from: '2026-01-01'
        formula: is_child_or_qualifying_young_person
  - name: child_benefit_weekly_entitlement
    kind: derived
    entity: Family
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: weekly_rate_sum
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/1992/4/141.test.yaml",
        """- name: child benefit entitlement
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-01-05'
    end: '2026-01-11'
  input: {}
  output:
    uk:statutes/ukpga/1992/4/141#entitled_to_child_benefit_for_week: holds
    uk:statutes/ukpga/1992/4/141#child_benefit_weekly_rate_for_responsible_child_or_qualifying_young_person: 27.05
    uk:statutes/ukpga/1992/4/141#child_or_qualifying_young_person_counts_for_child_benefit_weekly_entitlement: holds
    uk:statutes/ukpga/1992/4/141#child_benefit_weekly_entitlement: 44.95
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="child_benefit")

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 2,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    entitlement = items_by_id[
        "uk:statutes/ukpga/1992/4/141#child_benefit_weekly_entitlement"
    ]
    rate = items_by_id[
        "uk:statutes/ukpga/1992/4/141#child_benefit_weekly_rate_for_responsible_child_or_qualifying_young_person"
    ]
    entitled = items_by_id[
        "uk:statutes/ukpga/1992/4/141#entitled_to_child_benefit_for_week"
    ]

    assert entitlement["status"] == "comparable"
    assert entitlement["policyengine_variable"] == "child_benefit_entitlement"
    assert entitlement["mapping_type"] == "direct_variable"
    assert entitlement["tested"] is True
    assert rate["status"] == "comparable"
    assert rate["policyengine_variable"] == "child_benefit_respective_amount"
    assert entitled["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_uk_child_benefit_final_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/child-benefit.yaml",
        """format: rulespec/v1
rules:
  - name: child_benefit_weekly_payment_periods_in_year
    kind: parameter
    dtype: Number
    versions:
      - effective_from: '2026-01-01'
        formula: 52
  - name: child_benefit_weekly_amount
    kind: derived
    entity: Family
    dtype: Money
    period: Week
    unit: GBP
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if would_claim_child_benefit: child_benefit_weekly_entitlement else: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/child-benefit.test.yaml",
        """- name: child benefit final amount
  period:
    period_kind: custom
    name: benefit_week
    start: '2026-01-05'
    end: '2026-01-11'
  input:
    uk:policies/govuk/child-benefit#input.would_claim_child_benefit: true
  output:
    uk:policies/govuk/child-benefit#child_benefit_weekly_payment_periods_in_year: 52
    uk:policies/govuk/child-benefit#child_benefit_weekly_amount: 44.95
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="child_benefit")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 1,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    final_amount = items_by_id[
        "uk:policies/govuk/child-benefit#child_benefit_weekly_amount"
    ]
    periods = items_by_id[
        "uk:policies/govuk/child-benefit#child_benefit_weekly_payment_periods_in_year"
    ]

    assert final_amount["status"] == "comparable"
    assert final_amount["policyengine_variable"] == "child_benefit"
    assert final_amount["mapping_type"] == "direct_variable"
    assert final_amount["tested"] is True
    assert periods["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_arizona_snap_medical_and_child_support(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-child-support-expense/allowable-deductions.yaml",
        """format: rulespec/v1
rules:
  - name: allowable_child_support_expense_amount
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: verified_amount_paid
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction.yaml",
        """format: rulespec/v1
rules:
  - name: medical_expense_disregard
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 35
  - name: standard_medical_deduction_net_amount
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 145
  - name: standard_medical_deduction_gross_amount
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 180
  - name: medical_deduction
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, medical_expenses - medical_expense_disregard)
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction.test.yaml",
        """- name: medical_outputs_are_tested
  period: 2026-01
  input: {}
  output:
    us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#medical_expense_disregard: 35
    us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#standard_medical_deduction_net_amount: 145
    us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#medical_deduction: 145
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    child_support = items_by_id[
        "us-az:policies/des/faa5/na-child-support-expense/allowable-deductions#allowable_child_support_expense_amount"
    ]
    disregard = items_by_id[
        "us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#medical_expense_disregard"
    ]
    standard = items_by_id[
        "us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#standard_medical_deduction_net_amount"
    ]
    gross = items_by_id[
        "us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#standard_medical_deduction_gross_amount"
    ]
    deduction = items_by_id[
        "us-az:policies/des/faa5/na-medical-expenses-and-deduction/medical-deduction#medical_deduction"
    ]

    assert child_support["status"] == "known_not_comparable"
    assert disregard["status"] == "comparable"
    assert (
        disregard["policyengine_parameter"]
        == "gov.usda.snap.income.deductions.excess_medical_expense.disregard"
    )
    assert standard["status"] == "comparable"
    assert (
        standard["policyengine_parameter"]
        == "gov.usda.snap.income.deductions.excess_medical_expense.standard"
    )
    assert gross["status"] == "known_not_comparable"
    assert deduction["status"] == "comparable"
    assert deduction["policyengine_variable"] == "snap_excess_medical_expense_deduction"
    assert deduction["tested"] is True


def test_policyengine_coverage_classifies_california_income_resource_bridge(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-ca"
        / "policies/cdss/snap/fy-2026-benefit-calculation.yaml",
        """format: rulespec/v1
rules:
  - name: calfresh_income_and_resource_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_categorically_eligible_for_resource_exemption or (snap_resource_eligible and snap_standard_income_eligible)
""",
    )

    coverage = build_policyengine_coverage_report(tmp_path, program="snap")
    candidates = build_policyengine_candidate_report(tmp_path, program="snap")

    assert coverage["status_counts"] == {"known_not_comparable": 1}
    item = coverage["items"][0]
    assert item["legal_id"] == (
        "us-ca:policies/cdss/snap/fy-2026-benefit-calculation"
        "#calfresh_income_and_resource_eligible"
    )
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == "is_snap_eligible"
    assert item["candidate_priority"] == "P4"
    assert candidates["category_counts"] == {"known_adjacent_target": 1}
    assert candidates["priority_counts"] == {"P4": 1}


def test_policyengine_coverage_classifies_arizona_snap_composition_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation.yaml",
        """format: rulespec/v1
rules:
  - name: snap_gross_monthly_income
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: earned_income + unearned_income
  - name: snap_net_income
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: max(0, snap_gross_monthly_income - deductions)
  - name: snap_eligible
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: income_eligible and resource_eligible
  - name: snap_maximum_allotment
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: thrift_food_plan_amount
  - name: snap_excess_shelter_deduction
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: shelter_deduction
  - name: snap_regular_month_allotment
    kind: derived
    versions:
      - effective_from: '2025-10-01'
        formula: max(0, snap_maximum_allotment - expected_contribution)
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation.test.yaml",
        """- name: composition_outputs_are_tested
  period: 2026-01
  input: {}
  output:
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_gross_monthly_income: 2000
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_net_income: 1200
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_eligible: holds
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_maximum_allotment: 768
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_excess_shelter_deduction: 350
    us-az:policies/des/faa5/na-eligibility-and-benefit-determination/fy-2026-benefit-calculation#snap_regular_month_allotment: 408
""",
    )

    coverage = build_policyengine_coverage_report(tmp_path, program="snap")

    assert coverage["status_counts"] == {"known_not_comparable": 6}
    composition_items = {
        item["rule_name"]: item
        for item in coverage["items"]
        if item["file"].endswith("fy-2026-benefit-calculation.yaml")
    }
    assert set(composition_items) == {
        "snap_gross_monthly_income",
        "snap_net_income",
        "snap_eligible",
        "snap_maximum_allotment",
        "snap_excess_shelter_deduction",
        "snap_regular_month_allotment",
    }
    assert {item["status"] for item in composition_items.values()} == {
        "known_not_comparable"
    }
    assert {item["candidate_priority"] for item in composition_items.values()} == {"P4"}
    assert {item["tested"] for item in composition_items.values()} == {True}
    assert (
        composition_items["snap_gross_monthly_income"]["policyengine_variable"]
        == "snap_gross_income"
    )
    assert (
        composition_items["snap_net_income"]["policyengine_variable"]
        == "snap_net_income"
    )
    assert (
        composition_items["snap_eligible"]["policyengine_variable"]
        == "is_snap_eligible"
    )
    assert (
        composition_items["snap_maximum_allotment"]["policyengine_variable"]
        == "snap_max_allotment"
    )
    assert (
        composition_items["snap_excess_shelter_deduction"]["policyengine_variable"]
        == "snap_excess_shelter_expense_deduction"
    )
    assert (
        composition_items["snap_regular_month_allotment"]["policyengine_variable"]
        == "snap"
    )


def test_policyengine_coverage_classifies_arizona_snap_utility_eligibility(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility.yaml",
        """format: rulespec/v1
rules:
  - name: liheap_minimum_annual_payment_amount
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 20
  - name: liheap_application_month_lookback_months
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 12
  - name: utility_allowance_prerequisites_met
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: billed_separately and obligated and verified
  - name: qualifying_liheap_sua_condition
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: liheap_annual_payment_amount >= liheap_minimum_annual_payment_amount
  - name: snap_standard_utility_allowance
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: utility_allowance_prerequisites_met and has_heating_or_cooling
  - name: snap_limited_utility_allowance
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: utility_allowance_prerequisites_met and has_two_non_heating_utilities
  - name: snap_telephone_utility_allowance_eligible
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: utility_allowance_prerequisites_met and has_only_telephone
  - name: snap_individual_utility_allowance
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: snap_telephone_utility_allowance_eligible
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility.test.yaml",
        """- name: utility_outputs_are_tested
  period: 2026-01
  input: {}
  output:
    us-az:policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility#liheap_minimum_annual_payment_amount: 20
    us-az:policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility#snap_standard_utility_allowance: holds
    us-az:policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility#snap_limited_utility_allowance: not_holds
    us-az:policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility#snap_telephone_utility_allowance_eligible: not_holds
    us-az:policies/des/faa5/na-utility-expenses-and-allowances/utility-allowance-eligibility#snap_individual_utility_allowance: not_holds
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    utility_items = {
        item["rule_name"]: item
        for item in report["items"]
        if item["file"].endswith("utility-allowance-eligibility.yaml")
    }
    assert set(utility_items) == {
        "liheap_minimum_annual_payment_amount",
        "liheap_application_month_lookback_months",
        "utility_allowance_prerequisites_met",
        "qualifying_liheap_sua_condition",
        "snap_standard_utility_allowance",
        "snap_limited_utility_allowance",
        "snap_telephone_utility_allowance_eligible",
        "snap_individual_utility_allowance",
    }
    assert {item["status"] for item in utility_items.values()} == {
        "known_not_comparable"
    }
    assert (
        utility_items["snap_standard_utility_allowance"]["policyengine_variable"]
        == "snap_standard_utility_allowance"
    )
    assert (
        utility_items["snap_limited_utility_allowance"]["policyengine_variable"]
        == "snap_limited_utility_allowance"
    )
    assert (
        utility_items["snap_individual_utility_allowance"]["policyengine_variable"]
        == "snap_individual_utility_allowance"
    )
    assert (
        utility_items["snap_telephone_utility_allowance_eligible"][
            "policyengine_variable"
        ]
        == "snap_utility_allowance_type"
    )
    assert utility_items["snap_standard_utility_allowance"]["tested"] is True


def test_policyengine_coverage_classifies_arizona_snap_shelter_deduction(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/shelter-expenses-and-deduction/shelter-deduction.yaml",
        """format: rulespec/v1
rules:
  - name: shelter_deduction_net_income_share
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 0.50
  - name: allowable_shelter_expense
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, shelter_expense - vendor_payment)
  - name: shelter_costs_with_utility_allowance
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: allowable_shelter_expense + utility_allowance
  - name: uncapped_shelter_deduction
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, shelter_costs_with_utility_allowance - shelter_deduction_net_income_share * net_income)
  - name: shelter_deduction_limit_for_budgetary_unit_without_elderly_or_disabled_participant
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: max(maximum_shelter_deduction_amount, homeless_shelter_deduction)
  - name: shelter_deduction
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: min(uncapped_shelter_deduction, shelter_deduction_limit_for_budgetary_unit_without_elderly_or_disabled_participant)
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/shelter-expenses-and-deduction/shelter-deduction.test.yaml",
        """- name: shelter_outputs_are_tested
  period: 2026-01
  input: {}
  output:
    us-az:policies/des/faa5/shelter-expenses-and-deduction/shelter-deduction#shelter_deduction_net_income_share: 0.5
    us-az:policies/des/faa5/shelter-expenses-and-deduction/shelter-deduction#allowable_shelter_expense: 800
    us-az:policies/des/faa5/shelter-expenses-and-deduction/shelter-deduction#shelter_deduction: 500
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    shelter_items = {
        item["rule_name"]: item
        for item in report["items"]
        if item["file"].endswith("shelter-deduction.yaml")
    }
    assert set(shelter_items) == {
        "allowable_shelter_expense",
        "shelter_costs_with_utility_allowance",
        "shelter_deduction",
        "shelter_deduction_limit_for_budgetary_unit_without_elderly_or_disabled_participant",
        "shelter_deduction_net_income_share",
        "uncapped_shelter_deduction",
    }
    assert shelter_items["shelter_deduction_net_income_share"]["status"] == "comparable"
    assert (
        shelter_items["shelter_deduction_net_income_share"]["policyengine_parameter"]
        == "gov.usda.snap.income.deductions.excess_shelter_expense.income_share_disregard"
    )
    assert shelter_items["shelter_deduction_net_income_share"]["tested"] is True
    assert shelter_items["shelter_deduction"]["status"] == "known_not_comparable"
    assert (
        shelter_items["shelter_deduction"]["policyengine_variable"]
        == "snap_excess_shelter_expense_deduction"
    )
    assert (
        shelter_items["allowable_shelter_expense"]["status"] == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_arizona_snap_dependent_care(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/dependent-care-expense/na-dependent-care.yaml",
        """format: rulespec/v1
rules:
  - name: dependent_child_under_age_limit
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: 18
  - name: dependent_care_expense_recipient_qualifies
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: care_recipient_is_dependent_child and care_recipient_age < dependent_child_under_age_limit
  - name: dependent_care_expense_necessary_for_allowable_activity
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: necessary_to_seek_accept_or_continue_employment
  - name: dependent_care_expense_type_is_allowable
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: billed_for_dependent_care or required_registration_fee
  - name: out_of_home_care_disallowed_due_to_available_parent
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: care_provided_out_of_home and parent_physically_capable_of_caring_for_dependent
  - name: dependent_care_expense_not_disallowed
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: not cost_paid_by_reimbursement and not out_of_home_care_disallowed_due_to_available_parent
  - name: dependent_care_expense_prorated_or_billed_amount
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: billed_dependent_care_expense_amount
  - name: snap_allowable_monthly_dependent_care_expenses
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: dependent_care_expense_prorated_or_billed_amount
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-az"
        / "policies/des/faa5/dependent-care-expense/na-dependent-care.test.yaml",
        """- name: dependent_care_outputs_are_tested
  period: 2026-01
  input: {}
  output:
    us-az:policies/des/faa5/dependent-care-expense/na-dependent-care#dependent_child_under_age_limit: 18
    us-az:policies/des/faa5/dependent-care-expense/na-dependent-care#dependent_care_expense_recipient_qualifies: holds
    us-az:policies/des/faa5/dependent-care-expense/na-dependent-care#snap_allowable_monthly_dependent_care_expenses: 300
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="snap")

    dependent_care_items = {
        item["rule_name"]: item
        for item in report["items"]
        if item["file"].endswith("na-dependent-care.yaml")
    }
    assert set(dependent_care_items) == {
        "dependent_care_expense_necessary_for_allowable_activity",
        "dependent_care_expense_not_disallowed",
        "dependent_care_expense_prorated_or_billed_amount",
        "dependent_care_expense_recipient_qualifies",
        "dependent_care_expense_type_is_allowable",
        "dependent_child_under_age_limit",
        "out_of_home_care_disallowed_due_to_available_parent",
        "snap_allowable_monthly_dependent_care_expenses",
    }
    assert {item["status"] for item in dependent_care_items.values()} == {
        "known_not_comparable"
    }
    assert (
        dependent_care_items["snap_allowable_monthly_dependent_care_expenses"][
            "policyengine_variable"
        ]
        == "snap_dependent_care_deduction"
    )
    assert (
        dependent_care_items["snap_allowable_monthly_dependent_care_expenses"]["tested"]
        is True
    )


def test_policyengine_coverage_classifies_tax_parameter_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3101/a.yaml",
        """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: '0.062'
  - name: oasdi_wage_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages * oasdi_wage_tax_rate
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/45A/a.yaml",
        """format: rulespec/v1
rules:
  - name: indian_employment_credit_rate
    kind: parameter
    versions:
      - effective_from: '1994-01-01'
        formula: '0.20'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 1,
    }
    statuses_by_id = {item["legal_id"]: item["status"] for item in report["items"]}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert statuses_by_id["us:statutes/26/3101/a#oasdi_wage_tax"] == "comparable"
    assert (
        items_by_id["us:statutes/26/3101/a#oasdi_wage_tax_rate"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.social_security.rate.employee"
    )
    assert (
        items_by_id["us:statutes/26/3101/a#oasdi_wage_tax_rate"]["test_output_count"]
        == 0
    )
    assert statuses_by_id["us:statutes/26/3101/a#oasdi_wage_tax_rate"] == "comparable"
    assert (
        statuses_by_id["us:statutes/26/45A/a#indian_employment_credit_rate"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_maps_colorado_income_tax_rate_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/c.yaml",
        """format: rulespec/v1
rules:
  - name: individual_estate_trust_income_tax_rate
    kind: parameter
    versions:
      - effective_from: '2022-01-01'
        formula: '0.044'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/c.test.yaml",
        """- name: rate for tax year beginning after january 2022
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/1.7/c#individual_estate_trust_income_tax_rate: 0.044
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us-co:statutes/39/39-22-104/1.7/c#individual_estate_trust_income_tax_rate"
    )
    assert item["policyengine_parameter"] == "gov.states.co.tax.income.rate"
    assert item["tested"] is True
    assert item["test_output_count"] == 1


def test_policyengine_coverage_maps_colorado_1999_income_tax_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.5.yaml",
        """format: rulespec/v1
rules:
  - name: subsection_1_5_individual_income_tax_rate
    kind: parameter
    versions:
      - effective_from: '1999-01-01'
        formula: '0.0475'
  - name: subsection_1_5_individual_income_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1999-01-01'
        formula: 'if taxable_year_commences_in_subsection_1_5_window: max(0, federal_taxable_income_after_subsection_2_modifications) * subsection_1_5_individual_income_tax_rate else: 0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.5.test.yaml",
        """- name: rate applies to positive modified income
  period:
    period_kind: tax_year
    start: '1999-01-01'
    end: '1999-12-31'
  input:
    us-co:statutes/39/39-22-104/1.5#input.taxable_year_commences_in_subsection_1_5_window: true
    us-co:statutes/39/39-22-104/1.5#input.federal_taxable_income_after_subsection_2_modifications: 100000
  output:
    us-co:statutes/39/39-22-104/1.5#subsection_1_5_individual_income_tax_rate: 0.0475
    us-co:statutes/39/39-22-104/1.5#subsection_1_5_individual_income_tax: 4750
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    rate = items_by_id[
        "us-co:statutes/39/39-22-104/1.5#subsection_1_5_individual_income_tax_rate"
    ]
    tax = items_by_id[
        "us-co:statutes/39/39-22-104/1.5#subsection_1_5_individual_income_tax"
    ]
    assert rate["policyengine_parameter"] == "gov.states.co.tax.income.rate"
    assert rate["tested"] is True
    assert rate["test_output_count"] == 1
    assert tax["policyengine_variable"] == "co_income_tax_before_non_refundable_credits"
    assert tax["tested"] is True
    assert tax["test_output_count"] == 1


def test_policyengine_coverage_maps_colorado_ccap_smi_limit(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-co"
        / "regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_state_median_income_85_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: state_median_income_85_monthly_by_family_size[family_size]
  - name: gross_income_within_state_median_income_ceiling
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: household_monthly_gross_income <= monthly_state_median_income_85_limit
  - name: income_and_asset_eligible_for_ccap_low_income_guidelines
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: gross_income_within_state_median_income_ceiling and assets_within_ccap_limit
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us-co"
        / "regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines.test.yaml",
        """- name: family size four limit
  period: 2025-10
  input:
    us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#input.family_size: 4
  output:
    us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#monthly_state_median_income_85_limit: 9828.83
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="ccap")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 2,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    smi_limit = items_by_id[
        "us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#monthly_state_median_income_85_limit"
    ]
    income_predicate = items_by_id[
        "us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#gross_income_within_state_median_income_ceiling"
    ]
    combined_predicate = items_by_id[
        "us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#income_and_asset_eligible_for_ccap_low_income_guidelines"
    ]
    assert smi_limit["policyengine_variable"] == "co_ccap_smi"
    assert smi_limit["tested"] is True
    assert smi_limit["test_output_count"] == 1
    assert income_predicate["status"] == "known_not_comparable"
    assert income_predicate["policyengine_variable"] == "co_ccap_smi_eligible"
    assert combined_predicate["status"] == "known_not_comparable"
    assert (
        combined_predicate["policyengine_variable"] == "co_ccap_entry_income_eligible"
    )


def test_policyengine_coverage_maps_colorado_taxable_income_base(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/2.yaml",
        """format: rulespec/v1
rules:
  - name: federal_taxable_income_after_subsection_2_modifications
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1987-01-01'
        formula: 'max(0, federal_taxable_income + additions - subtractions)'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/2.test.yaml",
        """- name: taxable income base is tested
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/2#federal_taxable_income_after_subsection_2_modifications: 90000
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us-co:statutes/39/39-22-104/2#federal_taxable_income_after_subsection_2_modifications"
    )
    assert item["policyengine_variable"] == "co_taxable_income"
    assert item["tested"] is True
    assert item["test_output_count"] == 1


def test_policyengine_coverage_classifies_remaining_colorado_104_adjustments(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/a.yaml",
        """format: rulespec/v1
rules:
  - name: federal_net_operating_loss_carryover_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1987-01-01'
        formula: federal_net_operating_loss_carryover
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/z.yaml",
        """format: rulespec/v1
rules:
  - name: retroactive_cares_act_subtraction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2021-01-01'
        formula: retroactive_cares_act_subtraction_calculated_before_limitation
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    addition = items_by_id[
        "us-co:statutes/39/39-22-104/3/a#federal_net_operating_loss_carryover_addition_to_federal_taxable_income"
    ]
    subtraction = items_by_id[
        "us-co:statutes/39/39-22-104/4/z#retroactive_cares_act_subtraction"
    ]
    assert addition["policyengine_variable"] == "co_additions"
    assert addition["candidate_priority"] == "P3"
    assert subtraction["policyengine_variable"] == "co_subtractions"
    assert subtraction["candidate_priority"] == "P3"


def test_policyengine_coverage_classifies_colorado_base_rates_not_comparable(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/a.yaml",
        """format: rulespec/v1
rules:
  - name: individual_estate_trust_income_tax_rate_before_2020
    kind: parameter
    versions:
      - effective_from: '2000-01-01'
        formula: '0.0463'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/a.test.yaml",
        """- name: rate for tax year beginning in 2019
  period:
    period_kind: tax_year
    start: '2019-01-01'
    end: '2019-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/1.7/a#individual_estate_trust_income_tax_rate_before_2020: 0.0463
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/b.yaml",
        """format: rulespec/v1
rules:
  - name: individual_estate_trust_income_tax_rate_before_2022
    kind: parameter
    versions:
      - effective_from: '2020-01-01'
        formula: '0.0455'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/1.7/b.test.yaml",
        """- name: rate for tax year beginning in 2021
  period:
    period_kind: tax_year
    start: '2021-01-01'
    end: '2021-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/1.7/b#individual_estate_trust_income_tax_rate_before_2022: 0.0455
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["legal_id"] for item in report["items"]} == {
        "us-co:statutes/39/39-22-104/1.7/a#individual_estate_trust_income_tax_rate_before_2020",
        "us-co:statutes/39/39-22-104/1.7/b#individual_estate_trust_income_tax_rate_before_2022",
    }
    assert {item["program"] for item in report["items"]} == {"tax"}
    assert {item["tested"] for item in report["items"]} == {True}


def test_policyengine_coverage_maps_colorado_state_tax_and_us_interest_adjustments(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/d.yaml",
        """format: rulespec/v1
rules:
  - name: state_income_tax_deduction_addition_limit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1992-01-01'
        formula: itemized_deductions - standard_deduction
  - name: state_income_tax_deduction_addition_to_federal_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1992-01-01'
        formula: min(state_income_tax_deduction, state_income_tax_deduction_addition_limit)
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/d.test.yaml",
        """- name: state income tax deduction addback outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/3/d#state_income_tax_deduction_addition_limit: 4000
    us-co:statutes/39/39-22-104/3/d#state_income_tax_deduction_addition_to_federal_taxable_income: 500
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/a.yaml",
        """format: rulespec/v1
rules:
  - name: united_states_possessions_obligations_interest_income_subtraction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1989-01-01'
        formula: max(0, us_obligation_interest_included_in_federal_taxable_income)
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/a.test.yaml",
        """- name: us obligations interest subtraction output
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/4/a#united_states_possessions_obligations_interest_income_subtraction: 1200
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 1,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    state_addback = items_by_id[
        "us-co:statutes/39/39-22-104/3/d#state_income_tax_deduction_addition_to_federal_taxable_income"
    ]
    state_addback_limit = items_by_id[
        "us-co:statutes/39/39-22-104/3/d#state_income_tax_deduction_addition_limit"
    ]
    us_interest = items_by_id[
        "us-co:statutes/39/39-22-104/4/a#united_states_possessions_obligations_interest_income_subtraction"
    ]
    assert state_addback["policyengine_variable"] == "co_state_addback"
    assert state_addback["status"] == "comparable"
    assert state_addback["tested"] is True
    assert state_addback_limit["status"] == "known_not_comparable"
    assert state_addback_limit["policyengine_variable"] == "co_state_addback"
    assert us_interest["policyengine_variable"] == "us_govt_interest"
    assert us_interest["status"] == "comparable"
    assert us_interest["tested"] is True


def test_policyengine_coverage_classifies_colorado_deduction_addbacks(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/o.yaml",
        """format: rulespec/v1
rules:
  - name: single_return_adjusted_gross_income_threshold_for_section_199a_addition
    kind: parameter
    versions:
      - effective_from: '2021-01-01'
        formula: '500000'
  - name: joint_return_adjusted_gross_income_threshold_for_section_199a_addition
    kind: parameter
    versions:
      - effective_from: '2021-01-01'
        formula: '1000000'
  - name: taxpayer_subject_to_section_199a_deduction_addition
    kind: derived
    versions:
      - effective_from: '2021-01-01'
        formula: agi_above_threshold and not schedule_f_exception
  - name: section_199a_deduction_addition_to_federal_taxable_income
    kind: derived
    versions:
      - effective_from: '2021-01-01'
        formula: 'if taxpayer_subject_to_section_199a_deduction_addition: section_199a_deduction else: 0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/o.test.yaml",
        """- name: qbi addback outputs
  period:
    period_kind: tax_year
    start: '2021-01-01'
    end: '2021-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/3/o#single_return_adjusted_gross_income_threshold_for_section_199a_addition: 500000
    us-co:statutes/39/39-22-104/3/o#joint_return_adjusted_gross_income_threshold_for_section_199a_addition: 1000000
    us-co:statutes/39/39-22-104/3/o#taxpayer_subject_to_section_199a_deduction_addition: holds
    us-co:statutes/39/39-22-104/3/o#section_199a_deduction_addition_to_federal_taxable_income: 1000
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p.yaml",
        """format: rulespec/v1
rules:
  - name: federal_adjusted_gross_income_threshold_for_itemized_deduction_addition
    kind: parameter
    versions:
      - effective_from: '2022-01-01'
        formula: '400000'
  - name: single_return_itemized_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2022-01-01'
        formula: '30000'
  - name: joint_return_itemized_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2022-01-01'
        formula: '60000'
  - name: taxpayer_subject_to_itemized_deduction_addition
    kind: derived
    versions:
      - effective_from: '2022-01-01'
        formula: itemizes and agi_above_threshold and not subsection_p_5_displaced
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p.test.yaml",
        """- name: itemized deduction addback outputs
  period:
    period_kind: tax_year
    start: '2022-01-01'
    end: '2022-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/3/p#federal_adjusted_gross_income_threshold_for_itemized_deduction_addition: 400000
    us-co:statutes/39/39-22-104/3/p#single_return_itemized_deduction_threshold: 30000
    us-co:statutes/39/39-22-104/3/p#joint_return_itemized_deduction_threshold: 60000
    us-co:statutes/39/39-22-104/3/p#taxpayer_subject_to_itemized_deduction_addition: holds
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p/5.yaml",
        """format: rulespec/v1
rules:
  - name: federal_adjusted_gross_income_threshold
    kind: parameter
    versions:
      - effective_from: '2023-01-01'
        formula: '300000'
  - name: single_return_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2023-01-01'
        formula: '12000'
  - name: joint_return_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2023-01-01'
        formula: '16000'
  - name: itemized_or_standard_deduction_addition_applies
    kind: derived
    versions:
      - effective_from: '2023-01-01'
        formula: claims_deduction and agi_above_threshold and not healthy_school_meals_repealed
  - name: initial_window_addition_to_federal_taxable_income
    kind: derived
    versions:
      - effective_from: '2023-01-01'
        formula: 'if itemized_or_standard_deduction_addition_applies: deduction_excess else: 0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p/5.test.yaml",
        """- name: subsection p5 deduction addback outputs
  period:
    period_kind: tax_year
    start: '2023-01-01'
    end: '2023-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/3/p/5#federal_adjusted_gross_income_threshold: 300000
    us-co:statutes/39/39-22-104/3/p/5#single_return_deduction_threshold: 12000
    us-co:statutes/39/39-22-104/3/p/5#joint_return_deduction_threshold: 16000
    us-co:statutes/39/39-22-104/3/p/5#itemized_or_standard_deduction_addition_applies: holds
    us-co:statutes/39/39-22-104/3/p/5#initial_window_addition_to_federal_taxable_income: 8000
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p/7.yaml",
        """format: rulespec/v1
rules:
  - name: ongoing_federal_adjusted_gross_income_threshold
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: '300000'
  - name: ongoing_single_return_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: '1000'
  - name: ongoing_joint_return_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: '2000'
  - name: ongoing_return_deduction_threshold
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: 'if single: ongoing_single_return_deduction_threshold else: ongoing_joint_return_deduction_threshold'
  - name: ongoing_itemized_or_standard_deduction_addition_applies
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: claims_deduction and not healthy_school_meals_repealed
  - name: ongoing_itemized_or_standard_deduction_addition_to_federal_taxable_income
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: 'if ongoing_itemized_or_standard_deduction_addition_applies: deduction_excess else: 0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/3/p/7.test.yaml",
        """- name: subsection p7 deduction addback outputs
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/3/p/7#ongoing_federal_adjusted_gross_income_threshold: 300000
    us-co:statutes/39/39-22-104/3/p/7#ongoing_single_return_deduction_threshold: 1000
    us-co:statutes/39/39-22-104/3/p/7#ongoing_joint_return_deduction_threshold: 2000
    us-co:statutes/39/39-22-104/3/p/7#ongoing_return_deduction_threshold: 1000
    us-co:statutes/39/39-22-104/3/p/7#ongoing_itemized_or_standard_deduction_addition_applies: holds
    us-co:statutes/39/39-22-104/3/p/7#ongoing_itemized_or_standard_deduction_addition_to_federal_taxable_income: 8000
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/m.yaml",
        """format: rulespec/v1
rules:
  - name: charitable_contribution_subtraction_floor
    kind: parameter
    versions:
      - effective_from: '2001-01-01'
        formula: '500'
  - name: standard_deduction_charitable_contribution_subtraction
    kind: derived
    entity: Person
    versions:
      - effective_from: '2001-01-01'
        formula: charitable_deduction_if_itemized - charitable_contribution_subtraction_floor
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/m.test.yaml",
        """- name: charitable contribution subtraction outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/4/m#charitable_contribution_subtraction_floor: 500
    us-co:statutes/39/39-22-104/4/m#standard_deduction_charitable_contribution_subtraction: 1000
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 21
    assert report["status_counts"] == {
        "comparable": 10,
        "known_not_comparable": 11,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    qbi_threshold = items_by_id[
        "us-co:statutes/39/39-22-104/3/o#single_return_adjusted_gross_income_threshold_for_section_199a_addition"
    ]
    qbi_addback = items_by_id[
        "us-co:statutes/39/39-22-104/3/o#section_199a_deduction_addition_to_federal_taxable_income"
    ]
    p5_threshold = items_by_id[
        "us-co:statutes/39/39-22-104/3/p/5#joint_return_deduction_threshold"
    ]
    p5_addback = items_by_id[
        "us-co:statutes/39/39-22-104/3/p/5#initial_window_addition_to_federal_taxable_income"
    ]
    p7_agi_threshold = items_by_id[
        "us-co:statutes/39/39-22-104/3/p/7#ongoing_federal_adjusted_gross_income_threshold"
    ]
    p7_single_threshold = items_by_id[
        "us-co:statutes/39/39-22-104/3/p/7#ongoing_single_return_deduction_threshold"
    ]
    p7_addback = items_by_id[
        "us-co:statutes/39/39-22-104/3/p/7#ongoing_itemized_or_standard_deduction_addition_to_federal_taxable_income"
    ]
    charitable_floor = items_by_id[
        "us-co:statutes/39/39-22-104/4/m#charitable_contribution_subtraction_floor"
    ]
    charitable_subtraction = items_by_id[
        "us-co:statutes/39/39-22-104/4/m#standard_deduction_charitable_contribution_subtraction"
    ]
    assert (
        qbi_threshold["policyengine_parameter"]
        == "gov.states.co.tax.income.additions.qualified_business_income_deduction.agi_threshold"
    )
    assert qbi_threshold["tested"] is True
    assert qbi_addback["status"] == "known_not_comparable"
    assert (
        qbi_addback["policyengine_variable"]
        == "co_qualified_business_income_deduction_addback"
    )
    assert (
        p5_threshold["policyengine_parameter"]
        == "gov.states.co.tax.income.additions.federal_deductions.exemption"
    )
    assert p5_addback["status"] == "known_not_comparable"
    assert p5_addback["policyengine_variable"] == "co_federal_deduction_addback"
    assert (
        p7_agi_threshold["policyengine_parameter"]
        == "gov.states.co.tax.income.additions.federal_deductions.agi_threshold"
    )
    assert (
        p7_single_threshold["policyengine_parameter"]
        == "gov.states.co.tax.income.additions.federal_deductions.exemption"
    )
    assert p7_single_threshold["status"] == "known_not_comparable"
    assert p7_single_threshold["candidate_priority"] == "P2"
    assert p7_single_threshold["tested"] is True
    assert p7_addback["status"] == "known_not_comparable"
    assert p7_addback["policyengine_variable"] == "co_federal_deduction_addback"
    assert (
        charitable_floor["policyengine_parameter"]
        == "gov.states.co.tax.income.subtractions.charitable_contribution.adjustment"
    )
    assert charitable_subtraction["status"] == "known_not_comparable"
    assert (
        charitable_subtraction["policyengine_variable"]
        == "co_charitable_contribution_subtraction"
    )


def test_policyengine_coverage_classifies_colorado_military_retirement_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/y.yaml",
        """format: rulespec/v1
rules:
  - name: military_retirement_benefits_cap_initial_phase
    kind: parameter
    versions:
      - effective_from: '2019-01-01'
        formula: '4500'
  - name: military_retirement_benefits_cap_second_phase
    kind: parameter
    versions:
      - effective_from: '2020-01-01'
        formula: '7500'
  - name: military_retirement_benefits_cap_third_phase
    kind: parameter
    versions:
      - effective_from: '2021-01-01'
        formula: '10000'
  - name: military_retirement_benefits_cap_final_phase
    kind: parameter
    versions:
      - effective_from: '2022-01-01'
        formula: '15000'
  - name: qualified_individual_for_military_retirement_benefits_subtraction
    kind: derived
    versions:
      - effective_from: '2019-01-01'
        formula: individual_under_fifty_five_at_close_of_taxable_year
  - name: military_retirement_benefits_subtraction
    kind: derived
    versions:
      - effective_from: '2019-01-01'
        formula: military_retirement_benefits
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/y.test.yaml",
        """- name: military retirement cap outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_cap_initial_phase: 4500
    us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_cap_second_phase: 7500
    us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_cap_third_phase: 10000
    us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_cap_final_phase: 15000
    us-co:statutes/39/39-22-104/4/y#qualified_individual_for_military_retirement_benefits_subtraction: holds
    us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_subtraction: 15000
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 4,
        "known_not_comparable": 2,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    initial_cap = items_by_id[
        "us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_cap_initial_phase"
    ]
    subtraction = items_by_id[
        "us-co:statutes/39/39-22-104/4/y#military_retirement_benefits_subtraction"
    ]
    age_predicate = items_by_id[
        "us-co:statutes/39/39-22-104/4/y#qualified_individual_for_military_retirement_benefits_subtraction"
    ]
    assert (
        initial_cap["policyengine_parameter"]
        == "gov.states.co.tax.income.subtractions.military_retirement.max_amount"
    )
    assert initial_cap["tested"] is True
    assert subtraction["status"] == "known_not_comparable"
    assert subtraction["policyengine_variable"] == "co_military_retirement_subtraction"
    assert age_predicate["status"] == "known_not_comparable"
    assert (
        age_predicate["policyengine_parameter"]
        == "gov.states.co.tax.income.subtractions.military_retirement.age_threshold"
    )


def test_policyengine_coverage_classifies_colorado_pension_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/f.yaml",
        """format: rulespec/v1
rules:
  - name: pension_annuity_subtraction_cap_for_age_fifty_five_to_sixty_four
    kind: parameter
    versions:
      - effective_from: '1989-01-01'
        formula: '20000'
  - name: pension_annuity_subtraction_cap_for_age_sixty_five_or_older
    kind: parameter
    versions:
      - effective_from: '1989-01-01'
        formula: '24000'
  - name: individual_filing_agi_limit_for_age_fifty_five_to_sixty_four_social_security_cap_increase
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: '75000'
  - name: joint_filing_agi_limit_for_age_fifty_five_to_sixty_four_social_security_cap_increase
    kind: parameter
    versions:
      - effective_from: '2025-01-01'
        formula: '95000'
  - name: individual_qualifies_for_pension_annuity_subtraction
    kind: derived
    versions:
      - effective_from: '1989-01-01'
        formula: individual_is_fifty_five_or_older
  - name: pension_annuity_subtraction_applicable_cap
    kind: derived
    versions:
      - effective_from: '1989-01-01'
        formula: pension_annuity_subtraction_cap_for_age_fifty_five_to_sixty_four
  - name: pension_annuity_subtraction
    kind: derived
    versions:
      - effective_from: '1989-01-01'
        formula: pension_annuity_subtraction_applicable_cap
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-104/4/f.test.yaml",
        """- name: pension outputs
  period:
    period_kind: tax_year
    start: '2025-01-01'
    end: '2025-12-31'
  input: {}
  output:
    us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction_cap_for_age_fifty_five_to_sixty_four: 20000
    us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction_cap_for_age_sixty_five_or_older: 24000
    us-co:statutes/39/39-22-104/4/f#individual_filing_agi_limit_for_age_fifty_five_to_sixty_four_social_security_cap_increase: 75000
    us-co:statutes/39/39-22-104/4/f#joint_filing_agi_limit_for_age_fifty_five_to_sixty_four_social_security_cap_increase: 95000
    us-co:statutes/39/39-22-104/4/f#individual_qualifies_for_pension_annuity_subtraction: holds
    us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction_applicable_cap: 20000
    us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction: 20000
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 2,
        "known_not_comparable": 5,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    younger_cap = items_by_id[
        "us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction_cap_for_age_fifty_five_to_sixty_four"
    ]
    older_cap = items_by_id[
        "us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction_cap_for_age_sixty_five_or_older"
    ]
    individual_agi_limit = items_by_id[
        "us-co:statutes/39/39-22-104/4/f#individual_filing_agi_limit_for_age_fifty_five_to_sixty_four_social_security_cap_increase"
    ]
    subtraction = items_by_id[
        "us-co:statutes/39/39-22-104/4/f#pension_annuity_subtraction"
    ]
    assert (
        younger_cap["policyengine_parameter"]
        == "gov.states.co.tax.income.subtractions.pension.cap.younger"
    )
    assert (
        older_cap["policyengine_parameter"]
        == "gov.states.co.tax.income.subtractions.pension.cap.older"
    )
    assert individual_agi_limit["status"] == "known_not_comparable"
    assert individual_agi_limit["program"] == "tax"
    assert subtraction["status"] == "known_not_comparable"
    assert subtraction["policyengine_variable"] == "co_pension_subtraction"


def test_policyengine_coverage_infers_unmapped_colorado_title_39_as_tax(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-999.yaml",
        """format: rulespec/v1
rules:
  - name: unmapped_colorado_tax_output
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 1
    # The jurisdiction-wide `us-co:` prefix mapping classifies the output as
    # not comparable (no exact mapping exists for it); program inference
    # still runs because prefix entries carry no program.
    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == (
        "us-co:statutes/39/39-22-999#unmapped_colorado_tax_output"
    )
    assert item["program"] == "tax"


def test_policyengine_coverage_classifies_colorado_amt_outputs(tmp_path):
    output_names = (
        "alternative_minimum_tax_rate",
        "minimum_tax_credit_percentage",
        "individual_alternative_minimum_tax_amount_before_regular_tax_offset",
        "individual_minimum_tax_credit_before_nonresident_apportionment",
        "nonresident_apportionment_ratio",
        "individual_minimum_tax_credit",
    )
    rules = "\n".join(
        f"""  - name: {name}
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in output_names
    )
    outputs = "\n".join(
        f"    us-co:statutes/39/39-22-105#{name}: 1" for name in output_names
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-105.yaml",
        f"""format: rulespec/v1
rules:
{rules}
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-105.test.yaml",
        f"""- name: colorado amt outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {{}}
  output:
{outputs}
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == len(output_names)
    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 5,
    }
    assert report["untested_comparable"] == 0
    assert {item["tested"] for item in report["items"]} == {True}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    rate = items_by_id["us-co:statutes/39/39-22-105#alternative_minimum_tax_rate"]
    tentative_tax = items_by_id[
        "us-co:statutes/39/39-22-105#individual_alternative_minimum_tax_amount_before_regular_tax_offset"
    ]
    credit = items_by_id["us-co:statutes/39/39-22-105#individual_minimum_tax_credit"]
    assert rate["policyengine_parameter"] == "gov.states.co.tax.income.amt.rate"
    assert rate["status"] == "comparable"
    assert tentative_tax["policyengine_variable"] == "co_tentative_minimum_tax"
    assert tentative_tax["status"] == "known_not_comparable"
    assert credit["status"] == "known_not_comparable"
    assert credit["policyengine_variable"] is None


def test_policyengine_coverage_classifies_colorado_ctc_45_outputs(tmp_path):
    output_names = (
        "single_return_low_income_upper_threshold",
        "single_return_middle_income_lower_threshold",
        "single_return_middle_income_upper_threshold",
        "single_return_high_income_lower_threshold",
        "single_return_high_income_upper_threshold",
        "joint_return_low_income_upper_threshold",
        "joint_return_middle_income_lower_threshold",
        "joint_return_middle_income_upper_threshold",
        "joint_return_high_income_lower_threshold",
        "joint_return_high_income_upper_threshold",
        "low_income_child_credit_amount",
        "middle_income_child_credit_amount",
        "high_income_child_credit_amount",
        "child_tax_credit_per_eligible_child",
    )
    rules = "\n".join(
        f"""  - name: {name}
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in output_names
    )
    outputs = "\n".join(
        f"    us-co:statutes/39/39-22-129/4.5#{name}: 1" for name in output_names
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-129/4.5.yaml",
        f"""format: rulespec/v1
rules:
{rules}
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-129/4.5.test.yaml",
        f"""- name: colorado ctc current-law outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {{}}
  output:
{outputs}
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == len(output_names)
    assert report["status_counts"] == {
        "comparable": len(output_names) - 1,
        "known_not_comparable": 1,
    }
    assert report["untested_comparable"] == 0
    assert {item["tested"] for item in report["items"]} == {True}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    per_child = items_by_id[
        "us-co:statutes/39/39-22-129/4.5#child_tax_credit_per_eligible_child"
    ]
    threshold = items_by_id[
        "us-co:statutes/39/39-22-129/4.5#single_return_low_income_upper_threshold"
    ]
    amount = items_by_id[
        "us-co:statutes/39/39-22-129/4.5#low_income_child_credit_amount"
    ]
    assert per_child["status"] == "known_not_comparable"
    assert per_child["policyengine_variable"] == "co_ctc"
    assert threshold["status"] == "comparable"
    assert (
        threshold["policyengine_parameter"]
        == "gov.states.co.tax.income.credits.ctc.amount.single"
    )
    assert amount["status"] == "comparable"
    assert (
        amount["policyengine_parameter"]
        == "gov.states.co.tax.income.credits.ctc.amount.single"
    )


def test_policyengine_coverage_classifies_colorado_cdcc_outputs(tmp_path):
    section_119_names = (
        "annual_federal_adjusted_gross_income_limit_after_inflation_adjustment",
        "applicable_federal_adjusted_gross_income_limit",
        "child_and_dependent_care_credit_percentage",
        "child_and_dependent_care_expenses_credit_allowed",
        "child_and_dependent_care_expenses_credit_before_part_year_apportionment",
        "child_and_dependent_care_expenses_credit_refundable_excess_before_part_year_apportionment",
        "federal_child_and_dependent_care_credit_base_after_child_care_assistance_limitation",
        "federal_child_and_dependent_care_credit_base_before_assistance_limitation",
        "federal_credit_base_uses_allowed_before_federal_tax_liability_limitation_indicator",
        "income_limit_adjustment_minimum_increase",
        "income_limit_inflation_adjustment_applies_indicator",
        "income_limit_rounding_increment",
        "rounded_federal_adjusted_gross_income_limit_after_cumulative_inflation",
        "statutory_federal_adjusted_gross_income_limit",
    )
    section_1195_comparable_parameters = (
        "dependent_age_exclusive_limit",
        "low_income_adjusted_gross_income_limit",
        "low_income_child_care_expenses_credit_percentage",
        "multiple_dependents_credit_cap",
        "multiple_dependents_credit_cap_minimum_dependents",
        "single_dependent_credit_cap",
        "single_dependent_credit_cap_dependent_count",
    )
    section_1195_remaining_names = (
        "care_provider_information_requirement_satisfied",
        "child_care_expense_basis_after_earned_income_limit",
        "earned_income_limit_for_child_care_expense_basis",
        "low_income_child_care_credit_tax_year_indicator",
        "low_income_child_care_expenses_credit_allowed",
        "low_income_child_care_expenses_credit_before_part_year_apportionment",
        "low_income_child_care_expenses_credit_dependent_cap",
        "low_income_child_care_expenses_credit_refundable_excess_before_part_year_apportionment",
        "person_is_under_dependent_age_limit_for_credit",
    )

    section_119_rules = "\n".join(
        f"""  - name: {name}
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in section_119_names
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-119.yaml",
        f"""format: rulespec/v1
rules:
{section_119_rules}
""",
    )
    section_119_outputs = "\n".join(
        f"    us-co:statutes/39/39-22-119#{name}: 1" for name in section_119_names
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-119.test.yaml",
        f"""- name: colorado cdcc outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {{}}
  output:
{section_119_outputs}
""",
    )

    section_1195_parameter_rules = "\n".join(
        f"""  - name: {name}
    kind: parameter
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in section_1195_comparable_parameters
    )
    section_1195_remaining_rules = "\n".join(
        f"""  - name: {name}
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in section_1195_remaining_names
    )
    section_1195_outputs = "\n".join(
        f"    us-co:statutes/39/39-22-119.5#{name}: 1"
        for name in (section_1195_comparable_parameters + section_1195_remaining_names)
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-119.5.yaml",
        f"""format: rulespec/v1
rules:
{section_1195_parameter_rules}
{section_1195_remaining_rules}
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-119.5.test.yaml",
        f"""- name: colorado low-income cdcc outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {{}}
  output:
{section_1195_outputs}
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == (
        len(section_119_names)
        + len(section_1195_comparable_parameters)
        + len(section_1195_remaining_names)
    )
    assert report["status_counts"] == {
        "comparable": len(section_1195_comparable_parameters),
        "known_not_comparable": len(section_119_names)
        + len(section_1195_remaining_names),
    }
    assert report["untested_comparable"] == 0
    assert {item["tested"] for item in report["items"]} == {True}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    regular_credit = items_by_id[
        "us-co:statutes/39/39-22-119#child_and_dependent_care_expenses_credit_before_part_year_apportionment"
    ]
    low_income_rate = items_by_id[
        "us-co:statutes/39/39-22-119.5#low_income_child_care_expenses_credit_percentage"
    ]
    single_cap = items_by_id[
        "us-co:statutes/39/39-22-119.5#single_dependent_credit_cap"
    ]
    low_income_credit = items_by_id[
        "us-co:statutes/39/39-22-119.5#low_income_child_care_expenses_credit_before_part_year_apportionment"
    ]
    assert regular_credit["status"] == "known_not_comparable"
    assert regular_credit["policyengine_variable"] == "co_cdcc"
    assert low_income_rate["status"] == "comparable"
    assert (
        low_income_rate["policyengine_parameter"]
        == "gov.states.co.tax.income.credits.cdcc.low_income.rate"
    )
    assert single_cap["status"] == "comparable"
    assert (
        single_cap["policyengine_parameter"]
        == "gov.states.co.tax.income.credits.cdcc.low_income.max_amount"
    )
    assert low_income_credit["status"] == "known_not_comparable"
    assert low_income_credit["policyengine_variable"] == "co_low_income_cdcc"


def test_policyengine_coverage_classifies_colorado_eitc_outputs(tmp_path):
    output_names = (
        "base_credit_percentage",
        "temporary_credit_percentage_increase",
        "revenue_adjustment_regime",
        "adjustment_factor_threshold_for_single_year_increase",
        "single_year_revenue_growth_percentage_point_increase",
        "adjustment_factor_lower_bound_for_small_increase",
        "small_revenue_growth_percentage_point_increase",
        "adjustment_factor_lower_bound_for_moderate_increase",
        "moderate_revenue_growth_percentage_point_increase",
        "adjustment_factor_lower_bound_for_large_increase",
        "large_revenue_growth_percentage_point_increase",
        "adjustment_factor_lower_bound_for_larger_increase",
        "larger_revenue_growth_percentage_point_increase",
        "adjustment_factor_lower_bound_for_maximum_increase",
        "maximum_revenue_growth_percentage_point_increase",
        "revenue_growth_adjustment_percentage_points",
        "credit_percentage_after_revenue_adjustment",
        "regular_federal_return_eitc_branch_before_part_year_apportionment",
        "missing_valid_ssn_eitc_branch_before_part_year_apportionment",
        "regular_federal_return_eitc_branch_refundable_excess",
        "missing_valid_ssn_eitc_branch_refundable_excess",
        "regular_federal_return_eitc_branch_excluded_from_public_benefit_income_or_resources",
        "missing_valid_ssn_eitc_branch_excluded_from_public_benefit_income_or_resources",
    )
    rules = "\n".join(
        f"""  - name: {name}
    kind: derived
    versions:
      - effective_from: '2024-01-01'
        formula: '1'"""
        for name in output_names
    )
    outputs = "\n".join(
        f"    us-co:statutes/39/39-22-123.5#{name}: 1" for name in output_names
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-123.5.yaml",
        f"""format: rulespec/v1
rules:
{rules}
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "statutes/39/39-22-123.5.test.yaml",
        f"""- name: colorado eitc outputs
  period:
    period_kind: tax_year
    start: '2024-01-01'
    end: '2024-12-31'
  input: {{}}
  output:
{outputs}
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == len(output_names)
    assert report["status_counts"] == {"known_not_comparable": len(output_names)}
    assert report["untested_comparable"] == 0
    assert {item["program"] for item in report["items"]} == {"tax"}
    assert {item["tested"] for item in report["items"]} == {True}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    rate = items_by_id[
        "us-co:statutes/39/39-22-123.5#credit_percentage_after_revenue_adjustment"
    ]
    regular_branch = items_by_id[
        "us-co:statutes/39/39-22-123.5#regular_federal_return_eitc_branch_before_part_year_apportionment"
    ]
    missing_ssn_branch = items_by_id[
        "us-co:statutes/39/39-22-123.5#missing_valid_ssn_eitc_branch_before_part_year_apportionment"
    ]
    refundable_excess = items_by_id[
        "us-co:statutes/39/39-22-123.5#regular_federal_return_eitc_branch_refundable_excess"
    ]
    assert (
        rate["policyengine_parameter"] == "gov.states.co.tax.income.credits.eitc.match"
    )
    assert rate["candidate_priority"] == "P2"
    assert regular_branch["policyengine_variable"] == "co_eitc"
    assert missing_ssn_branch["policyengine_variable"] == "co_eitc"
    assert refundable_excess["policyengine_variable"] is None


def test_policyengine_coverage_classifies_3102a_collection_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3102/a.yaml",
        """format: rulespec/v1
rules:
  - name: paragraph_7C_or_10_cash_remuneration_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 100
  - name: paragraph_8B_cash_remuneration_deduction_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 150
  - name: employer_required_to_collect_section_3101_tax_by_wage_deduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employer_is_employer_of_taxpayer and wages_are_paid
  - name: paragraph_7B_cash_remuneration_tax_deduction_permitted_when_below_applicable_threshold
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: cash_remuneration < applicable_dollar_threshold
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "employee_payroll_tax"
    }


def test_policyengine_coverage_classifies_3102b_collection_liability_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3102/b.yaml",
        """format: rulespec/v1
rules:
  - name: employer_liable_for_payment_of_deducted_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tax_deducted_under_section_3102
  - name: employer_indemnified_against_claims_for_deducted_tax_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employer_paid_tax_deducted_under_section_3102
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "employee_payroll_tax"
    }


def test_policyengine_coverage_maps_3306_b_1_futa_wage_base(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/1.yaml",
        """format: rulespec/v1
rules:
  - name: annual_remuneration_wage_base_limit
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 7000
  - name: successor_predecessor_remuneration_considered_paid
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: predecessor_remuneration
  - name: remuneration_excluded_from_wages_after_annual_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, wages - annual_remuneration_wage_base_limit)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 2,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    wage_base = items_by_id[
        "us:statutes/26/3306/b/1#annual_remuneration_wage_base_limit"
    ]
    assert wage_base["status"] == "comparable"
    assert (
        wage_base["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.taxable_wage_base"
    )
    assert (
        items_by_id[
            "us:statutes/26/3306/b/1#successor_predecessor_remuneration_considered_paid"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3306/b/1#remuneration_excluded_from_wages_after_annual_limit"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_3306_b_2_employer_plan_payment_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/2.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: employer_plan_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if payment_made_to_or_on_behalf_of_employee_or_dependent
          and employer_plan_or_system_makes_provision_for_employees_or_classes_and_dependents
          and (
            payment_on_account_of_medical_or_hospitalization_expenses_in_connection_with_sickness_or_accident_disability
            or payment_on_account_of_death
            or (
              payment_on_account_of_sickness_or_accident_disability
              and payment_received_under_workmens_compensation_law
            )
          ): payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/2#employer_plan_payment_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_4_post_work_sickness_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/4.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: sickness_accident_disability_payment_waiting_period_calendar_months
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 6
  - name: post_work_sickness_accident_or_medical_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          (payment_on_account_of_sickness_or_accident_disability
          or payment_on_account_of_medical_or_hospitalization_expenses_in_connection_with_sickness_or_accident_disability)
          and (payment_made_by_employer_to_employee
          or payment_made_by_employer_on_behalf_of_employee)
          and full_calendar_months_following_last_work_month_expired_before_payment >= sickness_accident_disability_payment_waiting_period_calendar_months
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    waiting_period = items_by_id[
        "us:statutes/26/3306/b/4#sickness_accident_disability_payment_waiting_period_calendar_months"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/4#post_work_sickness_accident_or_medical_payment_excluded_from_wages"
    ]
    assert waiting_period["status"] == "known_not_comparable"
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_5_qualified_plan_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/5.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: qualified_plan_cafeteria_or_deferred_compensation_exclusion_category_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          (
            payment_from_or_to_trust_described_in_section_401_a_exempt_under_section_501_a_at_time_of_payment
            and not payment_made_to_employee_of_trust_as_remuneration_for_services_rendered_as_employee_and_not_as_beneficiary
          )
          or payment_under_or_to_annuity_plan_described_in_section_403_a_at_time_of_payment
  - name: qualified_plan_cafeteria_or_deferred_compensation_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if payment_made_to_or_on_behalf_of_employee_or_beneficiary
          and qualified_plan_cafeteria_or_deferred_compensation_exclusion_category_applies: payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/5#qualified_plan_cafeteria_or_deferred_compensation_exclusion_category_applies"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/5#qualified_plan_cafeteria_or_deferred_compensation_payment_excluded_from_wages"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert exclusion["policyengine_variable"] == (
        "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_6_state_unemployment_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/6.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: state_unemployment_payment_exclusion_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          payment_made_by_employer_without_deduction_from_employee_remuneration
          and payment_required_from_employee_under_state_unemployment_compensation_law
          and (
            remuneration_paid_to_employee_for_domestic_service_in_private_home_of_employer
            or remuneration_paid_to_employee_for_agricultural_labor
          )
  - name: employer_state_unemployment_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if (
            payment_made_by_employer_without_deduction_from_employee_remuneration
            and payment_required_from_employee_under_state_unemployment_compensation_law
            and (
              remuneration_paid_to_employee_for_domestic_service_in_private_home_of_employer
              or remuneration_paid_to_employee_for_agricultural_labor
            )
          ): payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/6#state_unemployment_payment_exclusion_applies"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/6#employer_state_unemployment_payment_excluded_from_wages"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert exclusion["policyengine_variable"] == (
        "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_7_noncash_nonbusiness_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/7.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: noncash_nonbusiness_service_remuneration_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if (
            remuneration_paid_to_employee
            and remuneration_paid_in_medium_other_than_cash
            and remuneration_for_service_not_in_course_of_employers_trade_or_business
          ): payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/7#noncash_nonbusiness_service_remuneration_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_9_section_217_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/9.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: corresponding_section_217_deduction_reasonably_believed_allowable_without_section_274_n
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          (remuneration_paid_to_employee
          or remuneration_paid_on_behalf_of_employee)
          and reasonable_to_believe_at_time_of_payment_corresponding_deduction_allowable_under_section_217_determined_without_regard_to_section_274_n
  - name: section_217_deduction_remuneration_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if corresponding_section_217_deduction_reasonably_believed_allowable_without_section_274_n:
            min(
              max(0, payment_amount),
              max(0, corresponding_deduction_amount_reasonably_believed_allowable_under_section_217_determined_without_regard_to_section_274_n)
            )
          else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/9#corresponding_section_217_deduction_reasonably_believed_allowable_without_section_274_n"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/9#section_217_deduction_remuneration_excluded_from_wages"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_10_death_disability_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/10.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: death_or_disability_retirement_termination_plan_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if (
            payment_or_series_paid_by_employer_to_employee_or_dependent
            and payment_paid_upon_or_after_termination_of_employment_relationship
            and (
              employment_relationship_terminated_because_of_death
              or employment_relationship_terminated_because_of_retirement_for_disability
            )
            and employer_established_plan_makes_provision_for_employees_generally_or_classes_and_dependents
            and not payment_or_series_would_have_been_paid_if_employment_relationship_had_not_been_terminated
          ): payment_or_series_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/10#death_or_disability_retirement_termination_plan_payment_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_11_noncash_agricultural_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/11.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: noncash_agricultural_labor_remuneration_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if (
            remuneration_for_agricultural_labor
            and remuneration_paid_in_medium_other_than_cash
          ): remuneration_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/11#noncash_agricultural_labor_remuneration_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_13_income_exclusion_benefits(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/13.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: cited_income_exclusion_reasonably_expected
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          reasonable_to_believe_employee_can_exclude_payment_or_benefit_from_income_under_section_127
          or reasonable_to_believe_employee_can_exclude_payment_or_benefit_from_income_under_section_129
          or reasonable_to_believe_employee_can_exclude_payment_or_benefit_from_income_under_section_134_b_4
          or reasonable_to_believe_employee_can_exclude_payment_or_benefit_from_income_under_section_134_b_5
  - name: payment_or_benefit_excluded_from_wages_due_to_expected_income_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    unit: USD
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if payment_made_or_benefit_furnished_to_or_for_benefit_of_employee and cited_income_exclusion_reasonably_expected: payment_or_benefit_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/13#cited_income_exclusion_reasonably_expected"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/13#payment_or_benefit_excluded_from_wages_due_to_expected_income_exclusion"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_14_meals_lodging_exclusion(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/14.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: meals_or_lodging_section_119_income_exclusion_reasonably_expected_at_furnishing
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          meals_or_lodging_furnished_by_or_on_behalf_of_employer_to_employee
          and reasonable_at_time_of_furnishing_to_believe_employee_can_exclude_meals_or_lodging_from_income_under_section_119
  - name: meals_or_lodging_value_excluded_from_wages_due_to_expected_section_119_income_exclusion
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if meals_or_lodging_section_119_income_exclusion_reasonably_expected_at_furnishing: value_of_meals_or_lodging_furnished_to_employee else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/14#meals_or_lodging_section_119_income_exclusion_reasonably_expected_at_furnishing"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/14#meals_or_lodging_value_excluded_from_wages_due_to_expected_section_119_income_exclusion"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_15_survivor_estate_payment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/15.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: survivor_or_estate_payment_after_employee_death_year_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if (
            (
              payment_made_by_employer_to_survivor_of_former_employee
              or payment_made_by_employer_to_estate_of_former_employee
            )
            and payment_made_after_calendar_year_in_which_former_employee_died
          ): payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/15#survivor_or_estate_payment_after_employee_death_year_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_16_income_exclusion_benefits(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/16.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: benefit_income_exclusion_reasonably_expected_under_cited_sections
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          reasonable_at_time_benefit_provided_to_believe_employee_can_exclude_benefit_from_income_under_section_74_c
          or reasonable_at_time_benefit_provided_to_believe_employee_can_exclude_benefit_from_income_under_section_108_f_4
          or reasonable_at_time_benefit_provided_to_believe_employee_can_exclude_benefit_from_income_under_section_117
          or reasonable_at_time_benefit_provided_to_believe_employee_can_exclude_benefit_from_income_under_section_132
  - name: benefit_excluded_from_wages_due_to_expected_income_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    unit: USD
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if benefit_provided_to_or_on_behalf_of_employee and benefit_income_exclusion_reasonably_expected_under_cited_sections: benefit_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/16#benefit_income_exclusion_reasonably_expected_under_cited_sections"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/16#benefit_excluded_from_wages_due_to_expected_income_exclusion"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_17_section_106_b_payments(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/17.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: payment_made_to_employee_with_expected_section_106_b_income_exclusion
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          payment_made_to_or_for_benefit_of_employee
          and reasonable_at_time_of_payment_to_believe_employee_can_exclude_payment_from_income_under_section_106_b
  - name: payment_excluded_from_wages_due_to_expected_section_106_b_income_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    unit: USD
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if payment_made_to_employee_with_expected_section_106_b_income_exclusion: payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/17#payment_made_to_employee_with_expected_section_106_b_income_exclusion"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/17#payment_excluded_from_wages_due_to_expected_section_106_b_income_exclusion"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_18_section_106_d_payments(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/18.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: section_106_d_income_exclusion_reasonably_expected_at_payment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          payment_made_to_or_for_benefit_of_employee
          and reasonable_at_time_of_payment_to_believe_employee_can_exclude_payment_from_income_under_section_106_d
  - name: payment_excluded_from_wages_due_to_expected_section_106_d_income_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    unit: USD
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if section_106_d_income_exclusion_reasonably_expected_at_payment: payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/18#section_106_d_income_exclusion_reasonably_expected_at_payment"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/18#payment_excluded_from_wages_due_to_expected_section_106_d_income_exclusion"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_19_stock_remuneration(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/19.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: stock_option_or_employee_stock_purchase_plan_stock_remuneration_exclusion_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          (
            remuneration_on_account_of_transfer_of_share_of_stock_to_individual
            and (
              stock_transfer_pursuant_to_exercise_of_incentive_stock_option_as_defined_in_section_422_b
              or stock_transfer_under_employee_stock_purchase_plan_as_defined_in_section_423_b
            )
          )
          or remuneration_on_account_of_disposition_by_individual_of_stock_transferred_pursuant_to_exercise_of_incentive_stock_option_or_under_employee_stock_purchase_plan
  - name: stock_option_or_employee_stock_purchase_plan_stock_remuneration_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    unit: USD
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if stock_option_or_employee_stock_purchase_plan_stock_remuneration_exclusion_applies: remuneration_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    predicate = items_by_id[
        "us:statutes/26/3306/b/19#stock_option_or_employee_stock_purchase_plan_stock_remuneration_exclusion_applies"
    ]
    exclusion = items_by_id[
        "us:statutes/26/3306/b/19#stock_option_or_employee_stock_purchase_plan_stock_remuneration_excluded_from_wages"
    ]
    assert predicate["status"] == "known_not_comparable"
    assert (
        predicate["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert exclusion["status"] == "known_not_comparable"
    assert (
        exclusion["policyengine_variable"]
        == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_b_20_third_party_employer_treatment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/b/20.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: third_party_treated_as_employer_for_this_chapter_and_chapter_22_wages
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          third_party_makes_payment
          and payment_included_in_wages_solely_by_reason_of_parenthetical_matter_contained_in_paragraph_2_subparagraph_A
          and not regulations_prescribed_by_secretary_provide_otherwise_for_third_party_payment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3306/b/20#third_party_treated_as_employer_for_this_chapter_and_chapter_22_wages"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3306_c_1_agricultural_labor_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/1.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: agricultural_labor_cash_remuneration_quarter_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: 20000
  - name: agricultural_labor_days_different_weeks_threshold
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 20
  - name: agricultural_labor_individuals_per_day_threshold
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 10
  - name: agricultural_labor_cash_remuneration_test_satisfied
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          maximum_cash_remuneration_paid_to_agricultural_labor_individuals_in_any_calendar_quarter >= agricultural_labor_cash_remuneration_quarter_threshold
  - name: agricultural_labor_day_count_test_satisfied
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          counted_days_during_calendar_year_or_preceding_calendar_year_each_in_different_calendar_week >= agricultural_labor_days_different_weeks_threshold
          and minimum_agricultural_labor_individuals_employed_for_some_portion_of_each_counted_day >= agricultural_labor_individuals_per_day_threshold
  - name: agricultural_labor_employer_test_satisfied
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          agricultural_labor_cash_remuneration_test_satisfied
          or agricultural_labor_day_count_test_satisfied
  - name: agricultural_labor_excepted_from_employment
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          labor_is_agricultural_labor
          and (
            not agricultural_labor_employer_test_satisfied
            or labor_performed_by_immigration_and_nationality_act_agricultural_worker_alien
          )
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 7}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_2_domestic_service_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/2.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: domestic_service_cash_remuneration_quarter_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: 1000
  - name: domestic_service_cash_remuneration_test_satisfied
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          maximum_cash_remuneration_paid_to_individuals_employed_in_such_domestic_service_in_any_calendar_quarter >= domestic_service_cash_remuneration_quarter_threshold
  - name: domestic_service_excepted_from_employment
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_is_domestic_service_in_private_home_local_college_club_or_local_chapter_of_college_fraternity_or_sorority
          and not domestic_service_cash_remuneration_test_satisfied
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_3_nonbusiness_service_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/3.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: nonbusiness_service_cash_remuneration_quarter_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: 50
  - name: nonbusiness_service_regular_employment_days_threshold
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 24
  - name: nonbusiness_service_cash_remuneration_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          cash_remuneration_paid_for_nonbusiness_service_in_calendar_quarter >= nonbusiness_service_cash_remuneration_quarter_threshold
  - name: nonbusiness_service_current_quarter_regular_employment_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          counted_days_in_calendar_quarter_person_performed_nonbusiness_service_for_employer >= nonbusiness_service_regular_employment_days_threshold
  - name: nonbusiness_service_regular_employment_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          nonbusiness_service_current_quarter_regular_employment_test_satisfied
          or person_was_regularly_employed_by_employer_for_nonbusiness_service_during_preceding_calendar_quarter
  - name: nonbusiness_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          person_performed_service_not_in_course_of_employer_trade_or_business_as_employee_in_calendar_quarter
          and not (
            nonbusiness_service_cash_remuneration_test_satisfied
            and nonbusiness_service_regular_employment_test_satisfied
          )
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 6}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_4_vessel_aircraft_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/4.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: non_american_vessel_or_aircraft_service_excepted_from_employment
    kind: derived
    entity: Asset
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_on_or_in_connection_with_vessel_or_aircraft
          and not (american_vessel or american_aircraft)
          and employee_employed_on_and_in_connection_with_vessel_or_aircraft_when_outside_united_states
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_5_family_employment(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/5.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: family_employment_child_age_limit
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 21
  - name: service_performer_under_family_employment_child_age_limit
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performer_age < family_employment_child_age_limit
  - name: family_employment_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_by_individual_in_employ_of_individuals_son
          or service_performed_by_individual_in_employ_of_individuals_daughter
          or service_performed_by_individual_in_employ_of_individuals_spouse
          or (
            service_performer_under_family_employment_child_age_limit
            and (
              service_performed_by_child_in_employ_of_childs_father
              or service_performed_by_child_in_employ_of_childs_mother
            )
          )
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_6_federal_government_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/6.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: service_in_employ_of_qualifying_united_states_instrumentality
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_instrumentality_of_united_states
          and (
            instrumentality_wholly_or_partially_owned_by_united_states
            or instrumentality_exempt_from_federal_unemployment_excise_tax_by_specific_reference_to_section_3301_or_prior_law
          )
  - name: federal_government_or_qualifying_instrumentality_service_excepted_from_employment
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_united_states_government
          or service_in_employ_of_qualifying_united_states_instrumentality
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_7_state_tribal_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/7.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: service_in_employ_of_state_political_subdivision_or_indian_tribe
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_state
          or service_performed_in_employ_of_political_subdivision_of_state
          or service_performed_in_employ_of_indian_tribe
  - name: service_in_employ_of_wholly_owned_state_political_subdivision_or_indian_tribe_instrumentality
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_instrumentality_of_state_political_subdivision_or_indian_tribe
          and instrumentality_wholly_owned_by_one_or_more_states_political_subdivisions_or_indian_tribes
  - name: service_in_employ_of_constitutionally_immune_state_or_political_subdivision_instrumentality
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_instrumentality_of_one_or_more_states_or_political_subdivisions
          and instrumentality_is_with_respect_to_service_immune_under_constitution_from_federal_unemployment_excise_tax
  - name: state_political_subdivision_indian_tribe_or_immune_instrumentality_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_in_employ_of_state_political_subdivision_or_indian_tribe
          or service_in_employ_of_wholly_owned_state_political_subdivision_or_indian_tribe_instrumentality
          or service_in_employ_of_constitutionally_immune_state_or_political_subdivision_instrumentality
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_8_exempt_organization_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/8.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: tax_exempt_section_501_c_3_organization_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_religious_charitable_educational_or_other_organization_described_in_section_501_c_3
          and employing_organization_exempt_from_income_tax_under_section_501_a
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_9_railroad_employment(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/9.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: railroad_employee_or_employee_representative_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_by_individual_as_employee_defined_in_section_1_of_railroad_unemployment_insurance_act
          or service_performed_by_individual_as_employee_representative_defined_in_section_1_of_railroad_unemployment_insurance_act
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_10_educational_health_and_exempt_organization_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/10.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: exempt_organization_service_remuneration_quarter_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: 50
  - name: exempt_organization_service_remuneration_below_threshold
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: remuneration_for_service_in_calendar_quarter < exempt_organization_service_remuneration_quarter_threshold
  - name: service_in_employ_of_qualifying_exempt_organization
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_employ_of_organization and employer_is_section_501_a_income_tax_exempt_organization and not employer_is_section_401_a_organization
  - name: exempt_organization_low_remuneration_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_in_employ_of_qualifying_exempt_organization and exempt_organization_service_remuneration_below_threshold
  - name: student_school_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_employ_of_school_college_or_university and service_performed_by_student_enrolled_and_regularly_attending_classes_at_employing_school_college_or_university
  - name: student_spouse_school_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_by_spouse_of_student_enrolled_and_regularly_attending_classes_at_employing_school_college_or_university
  - name: school_student_or_spouse_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: student_school_service_excepted_from_employment or student_spouse_school_service_excepted_from_employment
  - name: work_experience_education_program_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: individual_enrolled_at_qualifying_nonprofit_or_public_educational_institution_as_full_time_credit_student
  - name: hospital_patient_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_employ_of_hospital and service_performed_by_patient_of_hospital
  - name: educational_health_and_exempt_organization_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: exempt_organization_low_remuneration_service_excepted_from_employment or school_student_or_spouse_service_excepted_from_employment or work_experience_education_program_service_excepted_from_employment or hospital_patient_service_excepted_from_employment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 10}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_11_foreign_government_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/11.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: foreign_government_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_foreign_government
          or service_performed_as_consular_or_other_officer_or_employee_of_foreign_government
          or service_performed_as_nondiplomatic_representative_of_foreign_government
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_12_foreign_government_instrumentality_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/12.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: foreign_government_wholly_owned_instrumentality_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          service_performed_in_employ_of_instrumentality_wholly_owned_by_foreign_government
          and service_character_similar_to_service_performed_in_foreign_countries_by_united_states_government_or_instrumentality_employees
          and secretary_of_state_certifies_foreign_government_grants_equivalent_exemption_for_similar_service_by_united_states_government_or_instrumentality_employees
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_13_health_training_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/13.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: student_nurse_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_as_student_nurse
  - name: hospital_intern_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_as_intern and service_performed_in_employ_of_hospital
  - name: student_nurse_or_hospital_intern_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: student_nurse_service_excepted_from_employment or hospital_intern_service_excepted_from_employment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_14_insurance_commission_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/14.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: insurance_agent_or_solicitor_commission_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_for_person_as_insurance_agent_or_insurance_solicitor
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_15_newspaper_service_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/15.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: newspaper_delivery_distribution_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: individual_under_age_18
  - name: ultimate_consumer_newspaper_or_magazine_sale_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_and_at_time_of_sale_of_newspapers_or_magazines_to_ultimate_consumers
  - name: newspaper_delivery_or_sales_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: newspaper_delivery_distribution_service_excepted_from_employment or ultimate_consumer_newspaper_or_magazine_sale_service_excepted_from_employment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_16_international_organization_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/16.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: international_organization_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_employ_of_international_organization
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_17_aquatic_life_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/17.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: large_vessel_net_tons_threshold
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '1990-01-01'
        formula: 10
  - name: qualifying_aquatic_life_service
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_performed_in_catching_taking_harvesting_cultivating_or_farming_aquatic_life
  - name: large_vessel_service_carveout_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: vessel_net_tons > large_vessel_net_tons_threshold
  - name: aquatic_life_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: qualifying_aquatic_life_service and not large_vessel_service_carveout_applies
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_c_18_fishing_boat_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/18.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: fishing_boat_service_excepted_from_employment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: fishing_boat_service_excluded_from_employment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_19_nonresident_nonimmigrant_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/19.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: nonresident_alien_nonimmigrant_purpose_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: individual_is_nonresident_alien and individual_temporarily_present_in_united_states_as_immigration_and_nationality_act_subparagraph_f_nonimmigrant
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_variable"] == "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_c_20_organized_camp_employment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/c/20.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: organized_camp_operation_month_limit
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 7
  - name: organized_camp_gross_receipts_percentage_limit
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: 0.3333333333333333
  - name: full_time_student_camp_service_week_limit
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 13
  - name: camp_employing_student_short_season_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: months <= organized_camp_operation_month_limit
  - name: camp_employing_student_gross_receipts_seasonality_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: selected_receipts <= organized_camp_gross_receipts_percentage_limit * other_receipts
  - name: camp_employing_student_paragraph_a_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: camp_employing_student_short_season_test_satisfied or camp_employing_student_gross_receipts_seasonality_test_satisfied
  - name: full_time_student_camp_service_week_test_satisfied
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: weeks < full_time_student_camp_service_week_limit
  - name: full_time_student_organized_camp_service_excepted_from_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: student and camp_employing_student_paragraph_a_test_satisfied and full_time_student_camp_service_week_test_satisfied
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 8}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_a_employer_definition(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/a.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: general_calendar_quarter_wage_threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '1990-01-01'
        formula: 1500
  - name: domestic_service_cash_wage_threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '1990-01-01'
        formula: 1000
  - name: employer_under_general_rule
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: wages >= general_calendar_quarter_wage_threshold
  - name: employer_for_domestic_service
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: domestic_wages >= domestic_service_cash_wage_threshold
  - name: employer
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: employer_under_general_rule or employer_for_domestic_service
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_d_pay_period_deeming(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/d.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: pay_period_for_included_and_excluded_service
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '1990-01-01'
        formula: pay_period_consecutive_days <= pay_period_max_consecutive_days
  - name: local_all_services_for_pay_period_deemed_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '1990-01-01'
        formula: pay_period_for_included_and_excluded_service and employment_fraction >= one_half_services_threshold
  - name: local_no_services_for_pay_period_deemed_employment
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '1990-01-01'
        formula: pay_period_for_included_and_excluded_service and nonemployment_fraction > one_half_services_threshold
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_f_unemployment_fund(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/f.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: unemployment_fund
    kind: derived
    entity: Asset
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: fund_is_special_fund_established_under_state_law
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3306/f#unemployment_fund"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == (
        "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_g_contributions(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/g.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: contributions
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if payment_required_by_state_law: payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3306/g#contributions"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == "employer_federal_unemployment_tax"


def test_policyengine_coverage_classifies_3306_h_compensation(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/h.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: compensation
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: payment_is_cash_benefit
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3306/h#compensation"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == (
        "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_i_employee_definition(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/i.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: employee
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: officer_of_corporation or common_law_employee_status
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3306/i#employee"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == (
        "taxable_earnings_for_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3306_j_definitions(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/j.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: local_american_employer
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: individual_employer_is_resident_of_united_states
  - name: puerto_rico_or_virgin_islands_citizen_considered_united_states_citizen_for_section
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: individual_is_citizen_of_puerto_rico_or_virgin_islands
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3306_k_agricultural_labor(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3306/k.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3306
rules:
  - name: chapter_group_operator_unmanufactured_commodity_handling_service
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: service_for_group_of_farm_operators and group_produced_more_than_half
  - name: local_agricultural_labor
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: section_3121_g_agricultural_labor or chapter_group_operator_unmanufactured_commodity_handling_service
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3307_deduction_payment_treatment(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3307.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3307
rules:
  - name: remuneration_deduction_payment_treatment_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: employer_required_to_deduct and amount_paid_to_government
  - name: remuneration_deduction_considered_paid_to_employee_for_chapter
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if remuneration_deduction_payment_treatment_applies: amount_deducted else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "taxable_earnings_for_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3308_instrumentality_exemption(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3308.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3308
rules:
  - name: other_law_specific_section_3301_exemption_requirement_met
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: section_3301_specific_exemption or prior_law_specific_exemption
  - name: united_states_instrumentality_section_3301_tax_exemption_precluded
    kind: derived
    entity: Employer
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          employer_is_instrumentality_of_united_states
          and general_instrumentality_tax_exemption
          and not other_law_specific_section_3301_exemption_requirement_met
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {
        "employer_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3311_short_title_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3311.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3311
rules:
  - name: federal_unemployment_tax_act_short_title
    kind: parameter
    dtype: String
    versions:
      - effective_from: '1990-01-01'
        formula: '"Federal Unemployment Tax Act"'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == (
        "us:statutes/26/3311#federal_unemployment_tax_act_short_title"
    )
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None
    assert "short-title" in item["rationale"]


def test_policyengine_coverage_classifies_3310_review_deadlines(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3310.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3310
rules:
  - name: state_petition_for_review_deadline_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '1990-01-01'
        formula: '60'
  - name: secretary_certification_withholding_notice_period_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '1990-01-01'
        formula: '60'
  - name: judicial_proceedings_stay_period_days
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '1990-01-01'
        formula: '30'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all("timing rules" in item["rationale"] for item in report["items"])


def test_policyengine_coverage_classifies_3303_state_reduced_rate_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3303.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3303
rules:
  - name: pooled_or_partially_pooled_minimum_experience_years
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '1990-01-01'
        formula: '3'
  - name: pooled_fund
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: unemployment_fund and all_contributions_are_mingled
  - name: employer_account_not_relieved_due_to_fault_pattern
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: employer_fault and pattern
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all("State-law" in item["rationale"] for item in report["items"])


def test_policyengine_coverage_classifies_3304_state_law_approval_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3304.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3304
rules:
  - name: institution_of_higher_education
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: educational_institution and nonprofit
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == ("us:statutes/26/3304#institution_of_higher_education")
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None
    assert "State-law approval" in item["rationale"]


def test_policyengine_coverage_classifies_3305_state_law_compliance_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3305.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3305
rules:
  - name: state_unemployment_fund_payment_compliance_not_relieved_by_commerce_ground
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: required_to_pay and interstate_commerce
  - name: state_unemployment_compensation_compliance_not_relieved_by_federal_property_services
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: subject_to_state_law and federal_property_services
  - name: state_unemployment_contribution_credits_denied_for_permission_condition_failure
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: required_to_contribute and certified_failure
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all("State-law compliance" in item["rationale"] for item in report["items"])


def test_policyengine_coverage_classifies_3309_coverage_predicate_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3309.yaml",
        """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/3309
rules:
  - name: policymaking_advisory_position_weekly_hours_limit
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: 8
  - name: election_official_worker_remuneration_annual_limit
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '1990-01-01'
        formula: 1000
  - name: service_category_excludes_section_application
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: religious_service or inmate_service
  - name: organization_has_minimum_employment_for_section_application
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: counted_days >= 20 and workers >= 4
  - name: tribal_uncorrected_failure_prevents_governmental_service_exception
    kind: derived
    dtype: Judgment
    versions:
      - effective_from: '1990-01-01'
        formula: tribal_delinquency and not corrected
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all("coverage thresholds" in item["rationale"] for item in report["items"])


def test_policyengine_coverage_classifies_3301_gross_futa_tax(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3301.yaml",
        """format: rulespec/v1
rules:
  - name: federal_unemployment_excise_tax_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.06
  - name: federal_unemployment_excise_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: federal_unemployment_excise_tax_rate * wages
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    rate = items_by_id["us:statutes/26/3301#federal_unemployment_excise_tax_rate"]
    tax = items_by_id["us:statutes/26/3301#federal_unemployment_excise_tax"]
    assert rate["status"] == "known_not_comparable"
    assert (
        rate["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert tax["status"] == "known_not_comparable"
    assert tax["policyengine_variable"] == "employer_federal_unemployment_tax"


def test_policyengine_coverage_classifies_3302_a_late_credit_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/a.yaml",
        """format: rulespec/v1
rules:
  - name: late_paid_contributions_credit_percentage
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.9
  - name: title11_trustee_no_fault_late_paid_contributions_credit_percentage
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 1
  - name: applicable_late_paid_contributions_credit_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: late_paid_contributions_credit_percentage
  - name: credit_for_late_paid_contributions_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: applicable_late_paid_contributions_credit_percentage * late_credit
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id["us:statutes/26/3302/a#late_paid_contributions_credit_percentage"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/a#title11_trustee_no_fault_late_paid_contributions_credit_percentage"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/a#applicable_late_paid_contributions_credit_percentage"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id["us:statutes/26/3302/a#credit_for_late_paid_contributions_limit"][
            "policyengine_variable"
        ]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_b_additional_credit_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/b.yaml",
        """format: rulespec/v1
rules:
  - name: additional_credit_comparison_rate_cap
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.054
  - name: applicable_additional_credit_comparison_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: min(state_rate, additional_credit_comparison_rate_cap)
  - name: additional_credit_rate_differential_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, comparison_contributions - required_contributions)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id["us:statutes/26/3302/b#additional_credit_comparison_rate_cap"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/b#applicable_additional_credit_comparison_rate"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id["us:statutes/26/3302/b#additional_credit_rate_differential_amount"][
            "policyengine_variable"
        ]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_c_1_credit_limit_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/c/1.yaml",
        """format: rulespec/v1
rules:
  - name: total_credits_allowed_percentage_limit
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.9
  - name: total_credits_allowed_under_section_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: total_credits_allowed_percentage_limit * tax
  - name: total_credits_allowed_under_section_after_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: min(credits_before_limit, total_credits_allowed_under_section_limit)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id["us:statutes/26/3302/c/1#total_credits_allowed_percentage_limit"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/1#total_credits_allowed_under_section_limit"
        ]["policyengine_variable"]
        == "employer_federal_unemployment_tax"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/1#total_credits_allowed_under_section_after_limit"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_3302_e_successor_credit_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/e.yaml",
        """format: rulespec/v1
rules:
  - name: successor_employer_credit_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: acquisition_conditions_hold
  - name: successor_employer_credit_before_subsection_c_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if successor_employer_credit_applies: other_person_credit else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert {item["status"] for item in items_by_id.values()} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in items_by_id.values()} == {
        "employer_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3302_f_credit_reduction_limitation_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/f.yaml",
        """format: rulespec/v1
rules:
  - name: credit_reduction_limitation_wage_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.006
  - name: state_meets_credit_reduction_limitation_requirements
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: secretary_determination
  - name: credit_reduction_after_partial_limitation
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, credit_reduction - partial_limitation)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert {item["status"] for item in items_by_id.values()} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in items_by_id.values()} == {
        "employer_federal_unemployment_tax"
    }


def test_policyengine_coverage_classifies_3202_collection_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3202.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_tip_collection_deadline_day
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 10
  - name: tier_2_employee_tax_collectible_by_employer_deduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: min(compensation, tier_2_employee_tax)
  - name: employer_section_3201_tax_payment_liability
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tax_required_to_deduct
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert {item["status"] for item in items_by_id.values()} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in items_by_id.values()} == {None}


def test_policyengine_coverage_classifies_3201_employee_rrta_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3201.yaml",
        """format: rulespec/v1
rules:
  - name: tier_1_tax_tier_number
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 1
  - name: tier_2_tax_tier_number
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 2
  - name: tier_2_employee_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tier_2_compensation * tier_2_rate
  - name: tier_2_applicable_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: applicable_percentage
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3212_compensation_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3212.yaml",
        """format: rulespec/v1
rules:
  - name: employee_representative_compensation_for_tax_ascertainment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: compensation_paid_by_employee_organization
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == (
        "us:statutes/26/3212#employee_representative_compensation_for_tax_ascertainment"
    )
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None


def test_policyengine_coverage_classifies_3221_employer_rrta_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3221.yaml",
        """format: rulespec/v1
rules:
  - name: tier_1_applicable_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tier_1_rate
  - name: tier_1_tax_tier_identifier
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: tier_1
  - name: tier_1_applicable_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tier_1_percentage
  - name: tier_2_applicable_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tier_2_rate
  - name: tier_2_tax_tier_identifier
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: tier_2
  - name: tier_2_employer_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tier_2_compensation * tier_2_rate
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 6}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3241_a_applicable_percentage_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3241/a.yaml",
        """format: rulespec/v1
rules:
  - name: applicable_percentage_for_section_3201_b_for_purposes_of_subsection_a
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: applicable_percentage_for_section_3201_b
  - name: applicable_percentage_for_sections_3211_b_and_3221_b_for_purposes_of_subsection_a
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: applicable_percentage_for_sections_3211_b_and_3221_b
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3241_c_account_ratio_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3241/c.yaml",
        """format: rulespec/v1
rules:
  - name: most_recent_fiscal_year_count_for_average_account_benefits_ratio
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 10
  - name: average_account_benefits_ratio_rounding_multiple
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.1
  - name: unrounded_average_account_benefits_ratio
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: sum_of_account_benefits_ratios / 10
  - name: average_account_benefits_ratio
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: ceil(unrounded_average_account_benefits_ratio / 0.1) * 0.1
  - name: account_benefits_ratio
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: assets / benefits_and_administrative_expenses
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_tip_timing_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231.yaml",
        """format: rulespec/v1
rules:
  - name: tips_compensation_deemed_paid_on_day_for_section_3201_taxes
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tips_received_day
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == (
        "us:statutes/26/3231#tips_compensation_deemed_paid_on_day_for_section_3201_taxes"
    )
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None


def test_policyengine_coverage_classifies_3231_a_employer_definition_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/a.yaml",
        """format: rulespec/v1
rules:
  - name: carrier_or_controlled_service_company_included
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: carrier_as_defined_in_subsection_g
  - name: receiver_or_trustee_of_employer_included
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: receiver_or_trustee
  - name: railroad_association_or_organization_included
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: railroad_association
  - name: railway_labor_organization_included
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: railway_labor_organization
  - name: electric_railway_exception_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: electric_railway_exception
  - name: coal_company_exception_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: coal_company_exception
  - name: employer
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: carrier_or_controlled_service_company_included
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 7}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_b_employee_definition_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/b.yaml",
        """format: rulespec/v1
rules:
  - name: coal_physical_operations_exclusion_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: engaged_in_physical_operations_consisting_of_mining_of_coal
  - name: employee
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: individual_in_service_of_one_or_more_employers_for_compensation
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3401_withholding_definitions(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3401/b.yaml",
        """format: rulespec/v1
rules:
  - name: payroll_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_of_wages_is_ordinarily_made_to_employee_by_employer_for_period
  - name: miscellaneous_payroll_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payroll_period and not payroll_period_is_daily
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3401/c.yaml",
        """format: rulespec/v1
rules:
  - name: employee
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_is_officer_of_corporation
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3401/d.yaml",
        """format: rulespec/v1
rules:
  - name: employer_for_subsection_a
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_for_whom_individual_performs_or_performed_service_as_employee
  - name: employer
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_has_control_of_payment_of_wages_for_services
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3401/f.yaml",
        """format: rulespec/v1
rules:
  - name: tips_included_in_wages_for_subsection_a
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if tips_received_by_employee_in_course_of_employment: tips_received_amount else: 0
  - name: tips_deemed_paid_when_written_statement_furnished
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tips_received_by_employee_in_course_of_employment and written_statement_furnished
  - name: tips_deemed_paid_when_received
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tips_received_by_employee_in_course_of_employment and not written_statement_furnished
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3401/h.yaml",
        """format: rulespec/v1
rules:
  - name: active_duty_period_minimum_days_for_differential_wage_payment
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 30
  - name: differential_wage_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_is_made_by_employer and active_duty_period_days > active_duty_period_minimum_days_for_differential_wage_payment
  - name: differential_wage_payment_treated_as_wages_for_subsection_a
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if differential_wage_payment: differential_wage_payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 11}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402a_withholding_table_wages(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/a.yaml",
        """format: rulespec/v1
rules:
  - name: amount_of_wages_for_withholding_tables
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, wages - taxpayer_withholding_allowance)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402b_withholding_period_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/b.yaml",
        """format: rulespec/v1
rules:
  - name: miscellaneous_allowance_period_days_for_nonpayroll_period_wages
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: days_in_period_with_respect_to_which_wages_are_paid
  - name: miscellaneous_allowance_period_days_for_wages_paid_without_regard_to_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: days_elapsed_since_later_reference_date
  - name: secretary_may_authorize_weekly_aggregate_computation
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: period_is_less_than_one_week
  - name: wages_computed_for_withholding_under_percentage_method
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_computed_to_nearest_dollar
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402c_wage_bracket_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/c.yaml",
        """format: rulespec/v1
rules:
  - name: wage_bracket_miscellaneous_period_days_for_nonpayroll_period_wages
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: days_in_period_with_respect_to_which_wages_are_paid
  - name: wage_bracket_miscellaneous_period_days_for_wages_paid_without_regard_to_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: days_elapsed_since_later_reference_date
  - name: wage_bracket_weekly_aggregate_computation_authorized
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: period_is_less_than_one_week
  - name: wages_computed_for_wage_bracket_withholding
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_computed_to_nearest_dollar
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402d_withholding_liability_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/d.yaml",
        """format: rulespec/v1
rules:
  - name: required_withholding_tax_not_collected_from_employer
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: recipient_tax_paid
  - name: employer_remains_liable_for_penalties_or_additions_for_withholding_failure
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: penalties_or_additions_apply
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402e_wage_deeming_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/e.yaml",
        """format: rulespec/v1
rules:
  - name: maximum_consecutive_days_for_payroll_period_deeming_rule
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 31
  - name: payroll_period_within_deeming_rule_day_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payroll_period_consecutive_days <= maximum_consecutive_days_for_payroll_period_deeming_rule
  - name: all_remuneration_deemed_wages_for_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: half_or_more_services_constitute_wages
  - name: no_remuneration_deemed_wages_for_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: more_than_half_services_do_not_constitute_wages
  - name: remuneration_amount_deemed_wages_for_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: remuneration_paid_by_employer_to_employee_for_period
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402f_withholding_certificate_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/f.yaml",
        """format: rulespec/v1
rules:
  - name: change_status_new_certificate_due_days
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 10
  - name: replacement_certificate_default_wait_days
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 30
  - name: nonresident_alien_withholding_exemption_limit
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 1
  - name: commencement_certificate_requirement_satisfied
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: signed_certificate_furnished
  - name: excess_allowance_new_certificate_violation
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: not new_certificate_furnished
  - name: first_certificate_effective_for_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: first_payroll_period_after_certificate
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 6}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402g_special_wage_withholding_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/g.yaml",
        """format: rulespec/v1
rules:
  - name: special_wage_payment_regulatory_withholding_rule_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: supplemental_wage_payment_condition
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402h_alternative_withholding_methods(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/h.yaml",
        """format: rulespec/v1
rules:
  - name: average_wage_method_quarterly_adjustment_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: quarterly_required_withholding - actual_withholding
  - name: annualized_wages_for_withholding_method
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_for_period * payroll_periods_in_year
  - name: annualized_wage_method_tax_to_withhold_on_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: annualized_tax / payroll_periods_in_year
  - name: cumulative_wage_method_tax_to_withhold_on_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: cumulative_wage_method_excess_tax
  - name: other_substantially_same_method_authorizable
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: secretary_regulations_authorize_other_method
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402i_requested_increased_withholding(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/i.yaml",
        """format: rulespec/v1
rules:
  - name: employee_requested_increased_withholding_regulatory_authority
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employee_requests_increased_withholding
  - name: increased_withholding_treated_as_required_withholding_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: increased_withholding_deducted_and_withheld
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402j_retail_commission_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/j.yaml",
        """format: rulespec/v1
rules:
  - name: retail_commission_noncash_remuneration_withholding_not_required
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: noncash_remuneration and retail_salesman_commission_service
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402k_tip_withholding_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/k.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_tip_statement_threshold_for_paragraph_16_b_permission
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 20
  - name: tips_withholding_under_subsection_a_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tips_constitute_wages
  - name: maximum_tip_tax_deduction_and_withholding_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, employer_controlled_wages - section_3102_or_3202_tax)
  - name: low_monthly_tip_statement_withholding_permitted_for_paragraph_16_b_tips
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: monthly_tips < monthly_tip_statement_threshold_for_paragraph_16_b_permission
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402l_marital_status_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/l.yaml",
        """format: rulespec/v1
rules:
  - name: employee_considered_not_married_for_married_certificate_disclosure
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: legally_separated or nonresident_alien_status
  - name: employee_considered_married_after_current_year_spouse_death
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: spouse_died and not subparagraph_a_disqualified
  - name: married_to_single_new_certificate_requirement_satisfied
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: not certificate_due or new_certificate_furnished
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402m_withholding_allowance_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/m.yaml",
        """format: rulespec/v1
rules:
  - name: employee_entitled_to_additional_withholding_adjustment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employee and secretary_regulations_prescribe_adjustment
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402n_no_liability_certificate_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/n.yaml",
        """format: rulespec/v1
rules:
  - name: employer_withholding_not_required_for_no_liability_certificate_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_is_wages and no_liability_certificate_in_effect
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402o_nonwage_withholding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/o.yaml",
        """format: rulespec/v1
rules:
  - name: supplemental_unemployment_compensation_benefit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: supplemental_unemployment_compensation_benefit_amount > 0
  - name: request_specified_withholding_amount_for_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: requested_amount
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402p_voluntary_withholding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/p.yaml",
        """format: rulespec/v1
rules:
  - name: unemployment_compensation_voluntary_withholding_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.10
  - name: specified_federal_payment_voluntary_withholding_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_amount * requested_rate
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402q_gambling_withholding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/q.yaml",
        """format: rulespec/v1
rules:
  - name: withholding_winnings_proceeds_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 5000
  - name: winnings_subject_to_withholding
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wager_proceeds > withholding_winnings_proceeds_threshold
  - name: payment_of_winnings_treated_as_wages_for_section_3403
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: winnings_subject_to_withholding
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402r_indian_casino_profit_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/r.yaml",
        """format: rulespec/v1
rules:
  - name: indian_casino_profit_payment_withholding_predicate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payer_makes_payment and payment_is_to_member_of_indian_tribe
  - name: tax_to_deduct_and_withhold_from_indian_casino_profit_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_proportionate_share_of_annualized_tax
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402s_vehicle_fringe_benefit_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/s.yaml",
        """format: rulespec/v1
rules:
  - name: vehicle_fringe_benefit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_is_fringe_benefit and fringe_benefit_constitutes_wages
  - name: vehicle_fringe_benefit_treated_as_wages_for_section_6051
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: vehicle_fringe_benefit
  - name: employer_vehicle_fringe_benefit_nonwithholding_election_available
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: vehicle_fringe_benefit and employee_notified_by_employer
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3402t_qualified_stock_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3402/t.yaml",
        """format: rulespec/v1
rules:
  - name: qualified_stock_with_section_83_i_election
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: stock_is_qualified_stock and section_83_i_election_made
  - name: section_1_maximum_rate_floor_applies_to_qualified_stock_with_section_83_i_election
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: qualified_stock_with_section_83_i_election
  - name: qualified_stock_treated_as_noncash_fringe_benefit_for_section_3501_b
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: qualified_stock_with_section_83_i_election
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3403_withholding_liability(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3403.yaml",
        """format: rulespec/v1
rules:
  - name: employer_liability_for_chapter_withholding_tax_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: tax_required_to_be_deducted_and_withheld_under_this_chapter
  - name: employer_liability_to_any_person_for_chapter_withholding_tax_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3404_government_return_maker(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3404.yaml",
        """format: rulespec/v1
rules:
  - name: government_employer_withholding_return_maker_authorized
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_is_officer_or_employee and person_has_control_of_wages
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    [item] = report["items"]
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None
    assert "return-filing delegation" in item["rationale"]


def test_policyengine_coverage_classifies_3405b_nonperiodic_distribution_withholding(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3405/b.yaml",
        """format: rulespec/v1
rules:
  - name: nonperiodic_distribution_withholding_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.10
  - name: nonperiodic_distribution_withholding
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: nonperiodic_distribution_amount * nonperiodic_distribution_withholding_rate
  - name: election_scope_for_current_nonperiodic_distribution
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: individual_elects_no_withholding_for_nonperiodic_distribution
  - name: election_scope_for_subsequent_nonperiodic_distributions
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: election_scope_for_current_nonperiodic_distribution and same_arrangement
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "distribution withholding administration" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3405_child_withholding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3405/c.yaml",
        """format: rulespec/v1
rules:
  - name: eligible_rollover_distribution_withholding_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.20
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    [item] = report["items"]
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None
    assert "withholding administration" in item["rationale"]


def test_policyengine_coverage_classifies_3406_child_backup_withholding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3406/a.yaml",
        """format: rulespec/v1
rules:
  - name: backup_withholding_requirement_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: reportable_payment and payee_fails_to_furnish_tin_to_payor
  - name: underreporting_or_certification_failure_applies_to_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: reportable_interest_or_dividend_payment and notified_payee_underreporting
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "backup withholding administration" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3127_religious_exemption_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3127.yaml",
        """format: rulespec/v1
rules:
  - name: employer_application_meets_statutory_approval_prerequisites
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employer_application_contains_required_evidence
  - name: employee_application_meets_statutory_approval_prerequisites
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employee_application_contains_required_evidence
  - name: employer_meets_religious_exemption_requirements
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employer_application_meets_statutory_approval_prerequisites
  - name: employee_meets_religious_exemption_requirements
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employee_application_meets_statutory_approval_prerequisites
  - name: wages_paid_during_section_3127_effective_period
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_paid_after_application_effective_date
  - name: wages_exempt_from_section_3111_taxes_under_section_3127
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_paid_to_employee_by_employer
  - name: wages_exempt_from_section_3101_taxes_under_section_3127
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages_paid_to_employee_by_employer
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 7}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3504_payroll_agent_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3504.yaml",
        """format: rulespec/v1
rules:
  - name: secretary_may_designate_wage_control_person_to_perform_employer_acts
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_has_control_receipt_custody_or_disposal_of_employee_wages
  - name: designated_person_subject_to_employer_law_provisions
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_is_designated_by_secretary_under_section_3504
  - name: employer_remains_subject_to_employer_law_provisions
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: designated_person_acts_for_employer
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3502_deduction_disallowance_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3502.yaml",
        """format: rulespec/v1
rules:
  - name: chapter_21_and_22_employment_taxes_allowed_as_subtitle_a_deduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0
  - name: employer_chapter_24_withheld_tax_allowed_as_subtitle_a_deduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0
  - name: income_recipient_chapter_24_withheld_tax_allowed_as_subtitle_a_deduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3503_cross_chapter_refund_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3503.yaml",
        """format: rulespec/v1
rules:
  - name: chapter_21_or_22_tax_paid_for_period_without_liability
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: if tax_paid_under_chapter_21_or_22_for_period_with_no_liability_under_that_chapter then max(0, tax_paid_under_chapter_21_or_22_for_period) else 0
  - name: credit_against_tax_imposed_by_other_chapter
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: min(chapter_21_or_22_tax_paid_for_period_without_liability, max(0, tax_imposed_by_other_chapter_on_taxpayer))
  - name: refund_balance_after_other_chapter_credit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, chapter_21_or_22_tax_paid_for_period_without_liability - credit_against_tax_imposed_by_other_chapter)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3505_third_party_liability_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3505.yaml",
        """format: rulespec/v1
rules:
  - name: supplied_funds_liability_limit_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.25
  - name: direct_wage_payment_third_party_liability_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: third_party_pays_wages_directly_to_employee_group_or_agent
  - name: direct_wage_payment_third_party_liability
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: taxes_together_with_interest_required_to_be_deducted_and_withheld_from_directly_paid_wages
  - name: supplied_funds_third_party_liability_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: funds_supplied_for_specific_purpose_of_paying_employer_wages
  - name: supplied_funds_third_party_liability_before_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: taxes_together_with_interest_not_paid_over_by_employer
  - name: supplied_funds_third_party_liability_limit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: supplied_funds_liability_limit_rate * amount_supplied
  - name: supplied_funds_third_party_liability
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: min(supplied_funds_third_party_liability_before_limit, supplied_funds_third_party_liability_limit)
  - name: employer_liability_credit_for_section_3505_payments
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: amounts_paid_to_united_states_pursuant_to_section_3505
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 8}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "third-party legal-liability" in item["rationale"] for item in report["items"]
    )


def test_policyengine_coverage_classifies_3506_sitter_placement_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3506.yaml",
        """format: rulespec/v1
rules:
  - name: sitters
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: individual_furnishes_personal_attendance_companionship_or_household_care_services
  - name: sitter_placement_person_not_treated_as_employer
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: person_engaged_in_trade_or_business_of_putting_sitters_in_touch_with_individuals_who_wish_to_employ_them
  - name: sitter_not_treated_as_employee_of_placement_person
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: sitters and not placement_person_pays_or_receives_salary_or_wages_of_sitters
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "placement-service employer-status classification" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3508_worker_classification_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3508.yaml",
        """format: rulespec/v1
rules:
  - name: qualified_real_estate_agent
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: individual_is_licensed_real_estate_agent
  - name: direct_seller_engaged_trade_or_business
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: engaged_in_consumer_products_home_or_nonpermanent_retail_trade_or_business
  - name: direct_seller
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: direct_seller_engaged_trade_or_business
  - name: services_performed_as_qualified_real_estate_agent_or_direct_seller
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: qualified_real_estate_agent or direct_seller
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "worker-classification predicates" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3131_paid_leave_credit_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3131/a.yaml",
        """format: rulespec/v1
rules:
  - name: qualified_sick_leave_wages_credit_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 1
  - name: qualified_sick_leave_wages_credit_against_applicable_employment_taxes
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: qualified_sick_leave_wages_credit_rate * qualified_sick_leave_wages
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "employer paid-leave credit surfaces" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3509_misclassification_liability_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3509.yaml",
        """format: rulespec/v1
rules:
  - name: default_withholding_liability_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.015
  - name: default_employee_social_security_tax_liability_percentage
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.20
  - name: increased_withholding_liability_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.03
  - name: increased_employee_social_security_tax_liability_percentage
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.40
  - name: section_applies_to_misclassified_employee_withholding_failure
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: employer_failed_to_deduct_and_withhold_tax
  - name: section_applies_to_employee_social_security_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: section_applies_to_misclassified_employee_withholding_failure
  - name: employee_tax_liability_unaffected_by_section_assessment_or_collection
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: amount_of_liability_for_tax_determined_under_this_section
  - name: employer_not_entitled_to_recover_section_tax_from_employee
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: amount_of_liability_for_tax_determined_under_this_section
  - name: section_3402_d_and_section_6521_do_not_apply_to_section_determined_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: amount_of_liability_for_tax_determined_under_this_section
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 9}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "misclassification assessment surface" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3510_domestic_service_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3510.yaml",
        """format: rulespec/v1
rules:
  - name: amount_withheld_from_domestic_service_remuneration_under_section_3402_p_agreement
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, amount_withheld_from_remuneration_for_domestic_service_in_private_home_pursuant_to_section_3402_p_agreement)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}
    assert all(
        "domestic-service withholding-administration component" in item["rationale"]
        for item in report["items"]
    )


def test_policyengine_coverage_classifies_3511_cpeo_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3511.yaml",
        """format: rulespec/v1
rules:
  - name: related_party_nonapplication_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: customer_bears_relationship_to_cpeo
  - name: cpeo_exclusive_employer_for_work_site_employee_remuneration
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: organization_is_certified_professional_employer_organization
  - name: employer_type_based_rules_apply_to_cpeo_remitted_work_site_remuneration
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: cpeo_exclusive_employer_for_work_site_employee_remuneration
  - name: cpeo_successor_customer_predecessor_during_service_contract
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: organization_entering_into_service_contract_with_customer
  - name: customer_successor_cpeo_predecessor_after_service_contract_termination
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: customer_service_contract_with_cpeo_terminated
  - name: individual_not_work_site_employee_due_to_self_employment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: individual_has_net_earnings_from_self_employment
  - name: specified_credit_applies_to_customer_not_cpeo
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: credit_is_specified_in_subsection_d
  - name: customer_takes_cpeo_paid_wages_and_employment_taxes_into_account_for_specified_credit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: specified_credit_applies_to_customer_not_cpeo
  - name: cpeo_information_furnishing_required_for_customer_credit
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: specified_credit_applies_to_customer_not_cpeo
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 9}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_c_employee_representative_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/c.yaml",
        """format: rulespec/v1
rules:
  - name: regularly_assigned_or_employed_individual_qualifies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: regular_assignment_or_employment_for_employee_representation
  - name: railway_labor_officer_representative_qualifies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: officer_or_official_representing_employees
  - name: employee_representative
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: regularly_assigned_or_employed_individual_qualifies
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_d_service_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/d.yaml",
        """format: rulespec/v1
rules:
  - name: basic_service_conditions_satisfied
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: subject_to_continuing_authority
  - name: general_chairman_minimum_compensation_percentage
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.75
  - name: general_chairman_service_remuneration_regarded_as_compensation
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: remuneration_for_service
  - name: general_committee_service_qualifies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: general_committee_service
  - name: local_lodge_or_division_service_qualifies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: local_lodge_service
  - name: non_us_principal_non_labor_organization_employer_service_qualifies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: non_us_principal_service
  - name: noncitizen_nonresident_foreign_required_hire_exception_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: foreign_required_hire
  - name: service
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: basic_service_conditions_satisfied
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 8}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_e_compensation_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/e.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_cash_tip_inclusion_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 20
  - name: local_lodge_monthly_disregard_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 25
  - name: money_remuneration_for_employee_services
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: payment_is_money_remuneration
  - name: cash_tips_included_as_compensation_for_section_3201_taxes
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: cash_tip_amount
  - name: employer_sickness_accident_medical_death_plan_exclusion_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: sickness_plan_payment
  - name: identified_business_expense_reimbursement_exclusion_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: business_expense_reimbursement
  - name: qualifying_nonresident_alien_service_exclusion_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: nonresident_alien_service
  - name: convention_delegate_compensation_disregard_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: convention_delegate_service
  - name: local_lodge_low_monthly_compensation_disregard_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: local_lodge_low_monthly_compensation
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 9}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_e_2_contribution_base_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/e/2.yaml",
        """format: rulespec/v1
rules:
  - name: successor_employer_compensation_base_continuity_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: successor_acquisition and immediate_service
  - name: predecessor_compensation_counted_for_successor_applicable_base
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: predecessor_compensation
  - name: compensation_counted_toward_applicable_base_before_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: compensation_before_payment
  - name: remaining_applicable_base_before_payment
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: applicable_base - compensation_before_payment
  - name: compensation_excess_excluded_before_hospital_insurance_carveout
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, remuneration - remaining_applicable_base_before_payment)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3231_f_company_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/f.yaml",
        """format: rulespec/v1
rules:
  - name: company
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: corporation or association or joint_stock_company
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3231/f#company"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None


def test_policyengine_coverage_classifies_3231_g_carrier_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3231/g.yaml",
        """format: rulespec/v1
rules:
  - name: carrier
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: subject_to_part_i_of_interstate_commerce_act
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3231/g#carrier"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None


def test_policyengine_coverage_classifies_3232_court_jurisdiction_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3232.yaml",
        """format: rulespec/v1
rules:
  - name: district_court_jurisdiction_to_compel_employee_or_other_person
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: attorney_general_application and person_resides_in_jurisdiction
  - name: district_court_jurisdiction_to_compel_employer
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: attorney_general_application and employer_subject_to_service
  - name: conferred_federal_court_jurisdiction_not_exclusive
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: otherwise_possessed_jurisdiction and enforcement_civil_action
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    assert {item["status"] for item in report["items"]} == {"known_not_comparable"}
    assert {item["policyengine_variable"] for item in report["items"]} == {None}


def test_policyengine_coverage_classifies_3233_short_title_output(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3233.yaml",
        """format: rulespec/v1
rules:
  - name: chapter_short_title
    kind: parameter
    dtype: String
    versions:
      - effective_from: '1990-01-01'
        formula: '"Railroad Retirement Tax Act"'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3233#chapter_short_title"
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] is None
    assert "short-title" in item["rationale"]


def test_policyengine_coverage_classifies_3302_c_2_a_advance_reduction_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/c/2/A.yaml",
        """format: rulespec/v1
rules:
  - name: second_consecutive_january1_advances_balance_reduction_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.05
  - name: succeeding_consecutive_january1_advances_balance_additional_reduction_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.05
  - name: advances_credit_reduction_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: second_consecutive_january1_advances_balance_reduction_rate
  - name: advances_credit_reduction_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: advances_credit_reduction_rate * tax
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/A#second_consecutive_january1_advances_balance_reduction_rate"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/A#succeeding_consecutive_january1_advances_balance_additional_reduction_rate"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id["us:statutes/26/3302/c/2/A#advances_credit_reduction_rate"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id["us:statutes/26/3302/c/2/A#advances_credit_reduction_amount"][
            "policyengine_variable"
        ]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_c_2_b_third_fourth_year_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/c/2/B.yaml",
        """format: rulespec/v1
rules:
  - name: federal_unemployment_credit_reduction_benchmark_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.027
  - name: taxable_year_begins_with_third_or_fourth_consecutive_january1_with_balance_of_advances
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: third_year or fourth_year
  - name: third_or_fourth_consecutive_january1_advances_balance_excess_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: federal_unemployment_credit_reduction_benchmark_rate - average_rate
  - name: third_or_fourth_consecutive_january1_advances_balance_reduction_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: third_or_fourth_consecutive_january1_advances_balance_excess_percentage
  - name: third_or_fourth_consecutive_january1_advances_balance_credit_reduction_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: third_or_fourth_consecutive_january1_advances_balance_reduction_rate * wages
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 5}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/B#federal_unemployment_credit_reduction_benchmark_rate"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/B#taxable_year_begins_with_third_or_fourth_consecutive_january1_with_balance_of_advances"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/B#third_or_fourth_consecutive_january1_advances_balance_excess_percentage"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/B#third_or_fourth_consecutive_january1_advances_balance_credit_reduction_amount"
        ]["policyengine_variable"]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_c_2_c_fifth_succeeding_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/c/2/C.yaml",
        """format: rulespec/v1
rules:
  - name: fifth_or_succeeding_benefit_cost_floor_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.027
  - name: fifth_or_succeeding_advances_balance_reduction_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: fifth_or_succeeding_balance and not secretary_exception
  - name: fifth_or_succeeding_advances_balance_subparagraph_b_applies_instead
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: fifth_or_succeeding_balance and secretary_exception
  - name: fifth_or_succeeding_benefit_cost_rate_after_floor
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(five_year_rate, fifth_or_succeeding_benefit_cost_floor_rate)
  - name: fifth_or_succeeding_advances_balance_excess_percentage
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: fifth_or_succeeding_benefit_cost_rate_after_floor - average_rate
  - name: fifth_or_succeeding_advances_balance_reduction_rate
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: fifth_or_succeeding_advances_balance_excess_percentage
  - name: fifth_or_succeeding_advances_balance_credit_reduction_amount
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: fifth_or_succeeding_advances_balance_reduction_rate * wages
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 7}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/C#fifth_or_succeeding_benefit_cost_floor_rate"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/C#fifth_or_succeeding_advances_balance_reduction_applies"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/C#fifth_or_succeeding_advances_balance_subparagraph_b_applies_instead"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/C#fifth_or_succeeding_benefit_cost_rate_after_floor"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/2/C#fifth_or_succeeding_advances_balance_credit_reduction_amount"
        ]["policyengine_variable"]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_c_3_trade_act_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/c/3.yaml",
        """format: rulespec/v1
rules:
  - name: trade_act_agreement_credit_reduction_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.075
  - name: trade_act_agreement_credit_reduction_applies
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: taxpayer_subject and secretary_determination
  - name: trade_act_agreement_credit_reduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: trade_act_agreement_credit_reduction_rate * tax
  - name: total_credits_allowed_under_section_after_trade_act_agreement_reduction
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: max(0, credits_before_reduction - trade_act_agreement_credit_reduction)
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 4}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/c/3#trade_act_agreement_credit_reduction_rate"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/3#trade_act_agreement_credit_reduction_applies"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id["us:statutes/26/3302/c/3#trade_act_agreement_credit_reduction"][
            "policyengine_variable"
        ]
        == "employer_federal_unemployment_tax"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/c/3#total_credits_allowed_under_section_after_trade_act_agreement_reduction"
        ]["policyengine_variable"]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_d_1_subsection_c_tax_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/d/1.yaml",
        """format: rulespec/v1
rules:
  - name: subsection_c_tax_computation_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.06
  - name: tax_imposed_by_section_3301_for_applying_subsection_c
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: federal_unemployment_excise_tax
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id["us:statutes/26/3302/d/1#subsection_c_tax_computation_rate"][
            "policyengine_parameter"
        ]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/d/1#tax_imposed_by_section_3301_for_applying_subsection_c"
        ]["policyengine_variable"]
        == "employer_federal_unemployment_tax"
    )


def test_policyengine_coverage_classifies_3302_d_2_state_attribution_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/d/2.yaml",
        """format: rulespec/v1
rules:
  - name: wages_attributable_to_particular_state_for_subsection_c
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: state_law_applies or secretary_rules_attribute_state
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3302/d/2#wages_attributable_to_particular_state_for_subsection_c"
    )
    assert item["status"] == "known_not_comparable"
    assert item["policyengine_variable"] == "employer_federal_unemployment_tax"


def test_policyengine_coverage_classifies_3302_d_4_state_rate_threshold_output(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/d/4.yaml",
        """format: rulespec/v1
rules:
  - name: average_employer_contribution_rate_employee_payment_adjustment_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.027
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3302/d/4#average_employer_contribution_rate_employee_payment_adjustment_threshold"
    )
    assert item["status"] == "known_not_comparable"
    assert (
        item["policyengine_parameter"]
        == "gov.irs.payroll.federal_unemployment.effective_rate"
    )


def test_policyengine_coverage_classifies_3302_d_5_benefit_cost_rate_scalars(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/d/5.yaml",
        """format: rulespec/v1
rules:
  - name: benefit_cost_rate_compensation_lookback_years
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 5
  - name: benefit_cost_rate_compensation_averaging_fraction
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 1 / 5
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/d/5#benefit_cost_rate_compensation_lookback_years"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/d/5#benefit_cost_rate_compensation_averaging_fraction"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_3302_d_6_rounding_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3302/d/6.yaml",
        """format: rulespec/v1
rules:
  - name: subparagraph_b_or_c_percentage_rounding_multiple
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 0.001
  - name: subparagraph_b_or_c_percentage_after_rounding
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: floor((rate / subparagraph_b_or_c_percentage_rounding_multiple) + 0.5) * subparagraph_b_or_c_percentage_rounding_multiple
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 2}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        items_by_id[
            "us:statutes/26/3302/d/6#subparagraph_b_or_c_percentage_rounding_multiple"
        ]["status"]
        == "known_not_comparable"
    )
    assert (
        items_by_id[
            "us:statutes/26/3302/d/6#subparagraph_b_or_c_percentage_after_rounding"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_classifies_legacy_tax_procedural_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/68/b.yaml",
        """format: rulespec/v1
rules:
  - name: section_68_applied_after_other_itemized_deduction_limitations
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: true
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/443/a/1.yaml",
        """format: rulespec/v1
rules:
  - name: annual_accounting_period_change_with_secretary_approval
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: taxpayer_changes_annual_accounting_period
  - name: return_made_for_required_short_period_after_accounting_period_change
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: return_is_made_for_short_period
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 3}
    statuses_by_id = {item["legal_id"]: item["status"] for item in report["items"]}
    assert (
        statuses_by_id[
            "us:statutes/26/68/b#section_68_applied_after_other_itemized_deduction_limitations"
        ]
        == "known_not_comparable"
    )
    assert (
        statuses_by_id[
            "us:statutes/26/443/a/1#annual_accounting_period_change_with_secretary_approval"
        ]
        == "known_not_comparable"
    )
    assert (
        statuses_by_id[
            "us:statutes/26/443/a/1#return_made_for_required_short_period_after_accounting_period_change"
        ]
        == "known_not_comparable"
    )


def test_policyengine_coverage_maps_section_1401_rate_leaf_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/a/rate.yaml",
        """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: '0.124'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/a/rate.test.yaml",
        """- name: section_1401_a_rate
  output:
    us:statutes/26/1401/a/rate#old_age_survivors_and_disability_insurance_tax_rate: 0.124
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/b/1/rate.yaml",
        """format: rulespec/v1
rules:
  - name: self_employment_income_tax_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: '0.029'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/b/1/rate.test.yaml",
        """- name: section_1401_b_1_rate
  output:
    us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate: 0.029
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 2
    assert report["status_counts"] == {"comparable": 2}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    oasdi_rate = items_by_id[
        "us:statutes/26/1401/a/rate#old_age_survivors_and_disability_insurance_tax_rate"
    ]
    assert oasdi_rate["status"] == "comparable"
    assert (
        oasdi_rate["policyengine_parameter"]
        == "gov.irs.self_employment.rate.social_security"
    )
    assert oasdi_rate["tested"] is True
    medicare_rate = items_by_id[
        "us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate"
    ]
    assert medicare_rate["status"] == "comparable"
    assert (
        medicare_rate["policyengine_parameter"]
        == "gov.irs.self_employment.rate.medicare"
    )
    assert medicare_rate["tested"] is True


def test_policyengine_coverage_maps_section_1401_child_tax_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/a.yaml",
        """format: rulespec/v1
rules:
  - name: old_age_survivors_and_disability_insurance_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: '0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/a.test.yaml",
        """- name: section_1401_a_tax
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 0
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/b/1.yaml",
        """format: rulespec/v1
rules:
  - name: self_employment_income_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: '0'
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1401/b/1.test.yaml",
        """- name: section_1401_b_1_tax
  output:
    us:statutes/26/1401/b/1#self_employment_income_tax: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 2
    assert report["status_counts"] == {"comparable": 2}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    oasdi_tax = items_by_id[
        "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax"
    ]
    assert oasdi_tax["status"] == "comparable"
    assert oasdi_tax["policyengine_variable"] == "self_employment_social_security_tax"
    assert oasdi_tax["tested"] is True
    medicare_tax = items_by_id["us:statutes/26/1401/b/1#self_employment_income_tax"]
    assert medicare_tax["status"] == "comparable"
    assert medicare_tax["policyengine_variable"] == "self_employment_medicare_tax"
    assert medicare_tax["tested"] is True


def test_policyengine_coverage_maps_section_32_earned_income_to_adjusted_earnings(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/32/c/2.yaml",
        """format: rulespec/v1
rules:
  - name: earned_income
    kind: derived
    entity: TaxUnit
    versions:
      - effective_from: '2026-01-01'
        formula: wages + net_earnings_from_self_employment_after_164_f
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    item = items_by_id["us:statutes/26/32/c/2#earned_income"]
    assert item["status"] == "comparable"
    assert item["policyengine_variable"] == "filer_adjusted_earnings"


def test_policyengine_coverage_classifies_section_1402_b_self_employment_outputs(
    tmp_path,
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1402/b.yaml",
        """format: rulespec/v1
rules:
  - name: self_employment_income_inclusion_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: '400'
  - name: self_employment_income
    kind: derived
    entity: TaxUnit
    versions:
      - effective_from: '1990-01-01'
        formula: net_earnings_from_self_employment
  - name: self_employment_income_for_section_1401_a
    kind: derived
    entity: TaxUnit
    versions:
      - effective_from: '1990-01-01'
        formula: min(net_earnings_from_self_employment, contribution_base - wages)
  - name: self_employment_income_excluded_for_taxpayer
    kind: derived
    entity: TaxUnit
    versions:
      - effective_from: '1990-01-01'
        formula: individual_is_nonresident_alien
  - name: individual_excluded_from_self_employment_income_definition
    kind: derived
    entity: Person
    versions:
      - effective_from: '1990-01-01'
        formula: individual_is_nonresident_alien
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/1402/b.test.yaml",
        """- name: section_1402_b
  output:
    us:statutes/26/1402/b#self_employment_income_inclusion_threshold: 400
    us:statutes/26/1402/b#self_employment_income: 923.5
    us:statutes/26/1402/b#self_employment_income_for_section_1401_a: 923.5
    us:statutes/26/1402/b#self_employment_income_excluded_for_taxpayer: false
    us:statutes/26/1402/b#individual_excluded_from_self_employment_income_definition: false
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 4,
    }
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    threshold = items_by_id[
        "us:statutes/26/1402/b#self_employment_income_inclusion_threshold"
    ]
    assert threshold["status"] == "comparable"
    assert (
        threshold["policyengine_parameter"]
        == "gov.irs.self_employment.net_earnings_exemption"
    )
    assert threshold["tested"] is True
    assert (
        items_by_id["us:statutes/26/1402/b#self_employment_income"][
            "policyengine_variable"
        ]
        == "taxable_self_employment_income"
    )
    assert (
        items_by_id["us:statutes/26/1402/b#self_employment_income_for_section_1401_a"][
            "policyengine_variable"
        ]
        == "social_security_taxable_self_employment_income"
    )
    assert (
        items_by_id[
            "us:statutes/26/1402/b#individual_excluded_from_self_employment_income_definition"
        ]["status"]
        == "known_not_comparable"
    )


def test_policyengine_coverage_treats_ssa_policy_parameters_as_tax(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-us"
        / "policies/ssa/contribution-and-benefit-base/2024.yaml",
        """format: rulespec/v1
rules:
  - name: contribution_and_benefit_base
    kind: parameter
    versions:
      - effective_from: '2024-01-01'
        formula: '168600'
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-us"
        / "policies/ssa/contribution-and-benefit-base/2026.yaml",
        """format: rulespec/v1
rules:
  - name: base_year_contribution_and_benefit_base
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: '60600'
  - name: contribution_and_benefit_base_under_section_230_of_social_security_act
    kind: parameter
    versions:
      - effective_from: '2026-01-01'
        formula: '184500'
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["total_outputs"] == 3
    statuses_by_id = {item["legal_id"]: item["status"] for item in report["items"]}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert (
        statuses_by_id[
            "us:policies/ssa/contribution-and-benefit-base/2024#contribution_and_benefit_base"
        ]
        == "comparable"
    )
    assert (
        items_by_id[
            "us:policies/ssa/contribution-and-benefit-base/2024#contribution_and_benefit_base"
        ]["policyengine_parameter"]
        == "gov.irs.payroll.social_security.cap"
    )
    assert (
        statuses_by_id[
            "us:policies/ssa/contribution-and-benefit-base/2026#base_year_contribution_and_benefit_base"
        ]
        == "known_not_comparable"
    )
    assert (
        statuses_by_id[
            "us:policies/ssa/contribution-and-benefit-base/2026#contribution_and_benefit_base_under_section_230_of_social_security_act"
        ]
        == "comparable"
    )


@pytest.mark.parametrize(
    ("subsection", "rule_name"),
    [
        ("2", "employer_plan_sickness_medical_death_payment_excluded_from_wages"),
        ("4", "post_work_sickness_disability_medical_payment_excluded_from_wages"),
        (
            "6",
            "state_unemployment_domestic_or_agricultural_payment_exclusion_applies",
        ),
        (
            "6",
            "state_unemployment_domestic_or_agricultural_payment_excluded_from_wages",
        ),
        (
            "7",
            "nontrade_or_domestic_service_remuneration_excluded_from_wages",
        ),
        ("8", "agricultural_labor_remuneration_excluded_from_wages"),
        ("10", "home_worker_low_cash_remuneration_exclusion_applies"),
        ("10", "home_worker_low_cash_remuneration_excluded_from_wages"),
        ("11", "moving_expense_deduction_reasonable_belief_exclusion_applies"),
        ("11", "moving_expense_deduction_reasonable_belief_excluded_from_wages"),
        ("12", "noncash_tip_exclusion_applies"),
        ("12", "noncash_tips_excluded_from_wages"),
        ("12", "low_monthly_cash_tip_exclusion_applies"),
        ("12", "low_monthly_cash_tips_excluded_from_wages"),
        ("12", "tips_excluded_from_wages"),
        ("13", "termination_plan_payment_excluded_from_wages"),
        ("14", "survivor_or_estate_post_death_year_payment_excluded_from_wages"),
        (
            "15",
            "social_security_disability_insurance_prior_year_no_services_payment_excluded_from_wages",
        ),
        ("16", "qualifying_exempt_organization_for_low_remuneration_exclusion"),
        ("16", "exempt_organization_low_remuneration_exclusion_applies"),
        ("16", "exempt_organization_low_remuneration_excluded_from_wages"),
        ("18", "employee_benefit_income_exclusion_reasonable_belief_applies"),
        ("18", "employee_benefit_income_exclusion_excluded_from_wages"),
        ("19", "meals_or_lodging_section_119_reasonable_belief_exclusion_applies"),
        ("19", "meals_or_lodging_section_119_reasonable_belief_excluded_from_wages"),
        ("20", "benefit_income_exclusion_reasonable_belief_applies"),
        ("20", "benefit_income_exclusion_reasonable_belief_excluded_from_wages"),
        ("21", "indian_fishing_rights_remuneration_exclusion_applies"),
        ("21", "indian_fishing_rights_remuneration_excluded_from_wages"),
        (
            "22",
            "stock_option_or_purchase_plan_share_transfer_remuneration_exclusion_applies",
        ),
        ("22", "stock_disposition_remuneration_exclusion_applies"),
        ("22", "stock_option_transfer_or_disposition_remuneration_exclusion_applies"),
        ("22", "stock_option_transfer_or_disposition_remuneration_excluded_from_wages"),
    ],
)
def test_policyengine_coverage_classifies_3121_wage_exclusions(
    tmp_path, subsection, rule_name
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / f"statutes/26/3121/a/{subsection}.yaml",
        f"""format: rulespec/v1
rules:
  - name: {rule_name}
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: payment_amount
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == f"us:statutes/26/3121/a/{subsection}#{rule_name}"
    assert item["status"] == "known_not_comparable"


@pytest.mark.parametrize(
    ("path", "legal_id", "rule_name"),
    [
        (
            "statutes/26/3121/b.yaml",
            "us:statutes/26/3121/b#service_excluded_from_employment",
            "service_excluded_from_employment",
        ),
        (
            "statutes/26/3121/b/1.yaml",
            "us:statutes/26/3121/b/1#foreign_agricultural_worker_service_excluded_from_employment",
            "foreign_agricultural_worker_service_excluded_from_employment",
        ),
        (
            "statutes/26/3121/c.yaml",
            "us:statutes/26/3121/c#all_services_for_pay_period_deemed_employment",
            "all_services_for_pay_period_deemed_employment",
        ),
        (
            "statutes/26/3121/d.yaml",
            "us:statutes/26/3121/d#employee",
            "employee",
        ),
        (
            "statutes/26/3121/e.yaml",
            "us:statutes/26/3121/e#citizen_of_united_states_for_section",
            "citizen_of_united_states_for_section",
        ),
        (
            "statutes/26/3121/f.yaml",
            "us:statutes/26/3121/f#american_vessel",
            "american_vessel",
        ),
        (
            "statutes/26/3121/g.yaml",
            "us:statutes/26/3121/g#agricultural_labor",
            "agricultural_labor",
        ),
        (
            "statutes/26/3121/h.yaml",
            "us:statutes/26/3121/h#american_employer",
            "american_employer",
        ),
        (
            "statutes/26/3121/i.yaml",
            "us:statutes/26/3121/i#uniformed_service_remuneration_included_before_subsection_a_1_limit",
            "uniformed_service_remuneration_included_before_subsection_a_1_limit",
        ),
        (
            "statutes/26/3121/j.yaml",
            "us:statutes/26/3121/j#covered_transportation_service",
            "covered_transportation_service",
        ),
        (
            "statutes/26/3121/l.yaml",
            "us:statutes/26/3121/l#foreign_affiliate",
            "foreign_affiliate",
        ),
        (
            "statutes/26/3121/n.yaml",
            "us:statutes/26/3121/n#member_of_uniformed_service",
            "member_of_uniformed_service",
        ),
        (
            "statutes/26/3121/o.yaml",
            "us:statutes/26/3121/o#crew_leader",
            "crew_leader",
        ),
        (
            "statutes/26/3121/p.yaml",
            "us:statutes/26/3121/p#peace_corps_volunteer_service_constitutes_employment",
            "peace_corps_volunteer_service_constitutes_employment",
        ),
        (
            "statutes/26/3121/q.yaml",
            "us:statutes/26/3121/q#tips_considered_remuneration_for_employment",
            "tips_considered_remuneration_for_employment",
        ),
        (
            "statutes/26/3121/r.yaml",
            "us:statutes/26/3121/r#member",
            "member",
        ),
        (
            "statutes/26/3121/s.yaml",
            "us:statutes/26/3121/s#common_paymaster_remuneration_allocated_to_disbursing_corporation",
            "common_paymaster_remuneration_allocated_to_disbursing_corporation",
        ),
        (
            "statutes/26/3121/u.yaml",
            "us:statutes/26/3121/u#medicare_qualified_government_employment",
            "medicare_qualified_government_employment",
        ),
        (
            "statutes/26/3121/v.yaml",
            "us:statutes/26/3121/v#nonqualified_deferred_compensation_taken_into_account_as_wages",
            "nonqualified_deferred_compensation_taken_into_account_as_wages",
        ),
        (
            "statutes/26/3121/w.yaml",
            "us:statutes/26/3121/w#services_excluded_from_employment_by_church_election",
            "services_excluded_from_employment_by_church_election",
        ),
        (
            "statutes/26/3121/x.yaml",
            "us:statutes/26/3121/x#applicable_dollar_threshold",
            "applicable_dollar_threshold",
        ),
        (
            "statutes/26/3121/y.yaml",
            "us:statutes/26/3121/y#transferred_federal_employee_international_organization_service_is_employment",
            "transferred_federal_employee_international_organization_service_is_employment",
        ),
        (
            "statutes/26/3121/z.yaml",
            "us:statutes/26/3121/z#foreign_person_treated_as_american_employer_for_government_contract_service",
            "foreign_person_treated_as_american_employer_for_government_contract_service",
        ),
    ],
)
def test_policyengine_coverage_classifies_3121_employment_exclusions(
    tmp_path, path, legal_id, rule_name
):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / path,
        f"""format: rulespec/v1
rules:
  - name: {rule_name}
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: service_condition
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == legal_id
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_c_pay_period_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/c.yaml",
        """format: rulespec/v1
rules:
  - name: pay_period_max_consecutive_days
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 31
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3121/c#pay_period_max_consecutive_days"
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_7_threshold_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/7.yaml",
        """format: rulespec/v1
rules:
  - name: cash_nontrade_service_annual_remuneration_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 100
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3121/a/7#cash_nontrade_service_annual_remuneration_threshold"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_8_threshold_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/8.yaml",
        """format: rulespec/v1
rules:
  - name: agricultural_labor_cash_remuneration_employee_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 150
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3121/a/8#agricultural_labor_cash_remuneration_employee_threshold"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_10_threshold_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/10.yaml",
        """format: rulespec/v1
rules:
  - name: home_worker_cash_remuneration_annual_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 100
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3121/a/10#home_worker_cash_remuneration_annual_threshold"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_5_subparts(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/5/D.yaml",
        """format: rulespec/v1
rules:
  - name: section_403_b_annuity_contract_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '1990-01-01'
        formula: |
          if section_403_b_annuity_contract_payment_exclusion_applies: payment_amount else: 0
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3121/a/5/D#section_403_b_annuity_contract_payment_excluded_from_wages"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_408_p_subparts(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/408/p/2/A/i.yaml",
        """format: rulespec/v1
rules:
  - name: employee_election_to_have_employer_make_payments_available
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: employee_eligible_to_participate_in_arrangement
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/408/p/2/A/i#employee_election_to_have_employer_make_payments_available"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_12_tip_threshold_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/12.yaml",
        """format: rulespec/v1
rules:
  - name: monthly_cash_tip_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 20
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert item["legal_id"] == "us:statutes/26/3121/a/12#monthly_cash_tip_threshold"
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_classifies_3121_a_16_threshold_parameter(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3121/a/16.yaml",
        """format: rulespec/v1
rules:
  - name: exempt_organization_remuneration_annual_threshold
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: 100
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"known_not_comparable": 1}
    item = report["items"][0]
    assert (
        item["legal_id"]
        == "us:statutes/26/3121/a/16#exempt_organization_remuneration_annual_threshold"
    )
    assert item["status"] == "known_not_comparable"


def test_policyengine_coverage_tracks_comparable_test_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3101/a.yaml",
        """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    versions:
      - effective_from: '1990-01-01'
        formula: '0.062'
  - name: oasdi_wage_tax
    kind: derived
    versions:
      - effective_from: '1990-01-01'
        formula: wages * oasdi_wage_tax_rate
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/3101/a.test.yaml",
        """- name: oasdi
  input:
    us:statutes/26/3101/a#input.wages: 100000
  output:
    us:statutes/26/3101/a#oasdi_wage_tax_rate: 0.062
    us:statutes/26/3101/a#oasdi_wage_tax: 6200
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {"comparable": 2}
    assert report["untested_comparable"] == 0
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    assert items_by_id["us:statutes/26/3101/a#oasdi_wage_tax_rate"]["tested"] is True
    assert items_by_id["us:statutes/26/3101/a#oasdi_wage_tax"]["tested"] is True


def test_policyengine_coverage_tracks_mapping_alias_test_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/32.yaml",
        """format: rulespec/v1
rules:
  - name: eitc_phase_in_rates
    kind: parameter
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0765
          1: 0.34
  - name: eitc_phase_in_rate
    kind: derived
    versions:
      - effective_from: '2026-01-01'
        formula: eitc_phase_in_rates[eitc_child_count]
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/26/32.test.yaml",
        """- name: selected_rate
  output:
    us:statutes/26/32#eitc_phase_in_rate: 0.34
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    items_by_id = {item["legal_id"]: item for item in report["items"]}
    table_item = items_by_id["us:statutes/26/32#eitc_phase_in_rates"]
    assert table_item["status"] == "comparable"
    assert table_item["tested"] is True
    assert table_item["test_output_count"] == 1
    assert report["untested_comparable"] == 0


def test_policyengine_candidates_prioritize_exact_unmapped_outputs(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/7/9999.yaml",
        """format: rulespec/v1
rules:
  - name: snap_new_exact_variable
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 1
  - name: snap_other_unmapped_helper
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 1
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us" / "statutes/7/9999.test.yaml",
        """- name: base
  output:
    us:statutes/7/9999#snap_new_exact_variable: 1
    us:statutes/7/9999#snap_other_unmapped_helper: 1
""",
    )

    report = build_policyengine_candidate_report(
        tmp_path,
        program="snap",
        policyengine_variables={"snap_new_exact_variable"},
    )

    assert report["category_counts"]["exact_variable_unmapped"] == 1
    assert report["category_counts"]["tested_unmapped_pe_like"] == 1
    first = report["items"][0]
    assert first["legal_id"] == "us:statutes/7/9999#snap_new_exact_variable"
    assert first["priority"] == "P1"
    assert first["policyengine_variable"] == "snap_new_exact_variable"


def test_policyengine_candidates_report_known_adjacent_targets(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/10-ccr-2506-1/4.408.yaml",
        """format: rulespec/v1
rules:
  - name: passes_resource_test
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: resources <= limit
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/10-ccr-2506-1/4.408.test.yaml",
        """- name: resources
  output:
    us-co:regulations/10-ccr-2506-1/4.408#passes_resource_test: holds
""",
    )

    report = build_policyengine_candidate_report(
        tmp_path,
        program="snap",
        policyengine_variables=set(),
    )

    candidate = report["items"][0]
    assert candidate["category"] == "known_adjacent_target"
    assert candidate["priority"] == "P2"
    assert candidate["policyengine_variable"] == "meets_snap_asset_test"


def test_policyengine_candidates_honor_registry_priority_overrides(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/10-ccr-2506-1/4.407.31.yaml",
        """format: rulespec/v1
rules:
  - name: snap_individual_utility_allowance
    kind: derived
    versions:
      - effective_from: '2025-01-01'
        formula: 97
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-us-co" / "regulations/10-ccr-2506-1/4.407.31.test.yaml",
        """- name: phone_only
  output:
    us-co:regulations/10-ccr-2506-1/4.407.31#snap_individual_utility_allowance: 97
""",
    )

    report = build_policyengine_candidate_report(
        tmp_path,
        program="snap",
        policyengine_variables={"snap_individual_utility_allowance"},
    )

    candidate = report["items"][0]
    assert candidate["category"] == "known_adjacent_target"
    assert candidate["priority"] == "P4"
    assert candidate["policyengine_variable"] == "snap_individual_utility_allowance"


def test_universal_credit_parameter_alias_counts_branch_output_test(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/376/36.yaml",
        """format: rulespec/v1
rules:
  - name: standard_allowance_single_under_25_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-04-01'
        value: 338.58
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "regulations/uksi/2013/376/36.test.yaml",
        """- name: branch_selected_standard_allowance
  period: 2026-04
  output:
    uk:regulations/uksi/2013/376/36#standard_allowance_amount: 338.58
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    item = report["items"][0]
    assert item["legal_id"] == (
        "uk:regulations/uksi/2013/376/36#standard_allowance_single_under_25_amount"
    )
    assert item["status"] == "comparable"
    assert item["tested"] is True
    assert item["test_output_count"] == 1


def test_universal_credit_source_helper_prefix_is_classified(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "statutes/ukpga/2012/5/2.yaml",
        """format: rulespec/v1
rules:
  - name: universal_credit_claim_may_be_made
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: claim_submitted
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    item = report["items"][0]
    assert item["legal_id"] == (
        "uk:statutes/ukpga/2012/5/2#universal_credit_claim_may_be_made"
    )
    assert item["program"] == "universal_credit"
    assert item["status"] == "known_not_comparable"
    assert item["mapping_type"] == "not_comparable"


def test_council_tax_reduction_policy_surface_is_classified(tmp_path):
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/council-tax-reduction.yaml",
        """format: rulespec/v1
rules:
  - name: council_tax_reduction_annual_amount
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: simulated_council_tax_reduction_annual_amount
  - name: simulated_council_tax_reduction_annual_amount
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: council_tax_liability_for_year
  - name: council_tax_reduction_capital_limit
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-04-01'
        value: 16000
""",
    )
    _write_rulespec_file(
        tmp_path / "rulespec-uk" / "policies/govuk/council-tax-reduction.test.yaml",
        """- name: council_tax_reduction_award
  period: 2026
  output:
    uk:policies/govuk/council-tax-reduction#council_tax_reduction_annual_amount: 1200
    uk:policies/govuk/council-tax-reduction#simulated_council_tax_reduction_annual_amount: 1200
    uk:policies/govuk/council-tax-reduction#council_tax_reduction_capital_limit: 16000
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 2,
    }
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    final = items_by_id[
        "uk:policies/govuk/council-tax-reduction#council_tax_reduction_annual_amount"
    ]
    assert final["status"] == "comparable"
    assert final["mapping_type"] == "direct_variable"
    assert final["policyengine_variable"] == "council_tax_reduction"
    assert final["tested"] is True
    simulated = items_by_id[
        "uk:policies/govuk/council-tax-reduction#simulated_council_tax_reduction_annual_amount"
    ]
    capital_limit = items_by_id[
        "uk:policies/govuk/council-tax-reduction#council_tax_reduction_capital_limit"
    ]
    assert simulated["status"] == "known_not_comparable"
    assert simulated["mapping_type"] == "not_comparable"
    assert capital_limit["status"] == "known_not_comparable"
    assert capital_limit["mapping_type"] == "not_comparable"


def test_kingston_council_tax_reduction_policy_surface_is_classified(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk-kingston-upon-thames"
        / "policies/kingston-upon-thames/council-tax-reduction.yaml",
        """format: rulespec/v1
rules:
  - name: kingston_upon_thames_council_tax_reduction_annual_amount
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: simulated_kingston_upon_thames_council_tax_reduction_annual_amount
  - name: simulated_kingston_upon_thames_council_tax_reduction_annual_amount
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: council_tax_liability_for_year
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["status_counts"] == {"known_not_comparable": 2}
    assert {item["program"] for item in report["items"]} == {"council_tax_reduction"}
    assert {item["mapping_type"] for item in report["items"]} == {"not_comparable"}


def test_universal_credit_housing_schedule_prefixes_are_classified(tmp_path):
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/4/paragraph/22.yaml",
        """format: rulespec/v1
rules:
  - name: renters_housing_costs_element_calculated_under_this_part
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: renters_lower_rent_amount
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/5/paragraph/9.yaml",
        """format: rulespec/v1
rules:
  - name: owner_occupier_housing_costs_element_amount
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: relevant_payment_amount
""",
    )
    _write_rulespec_file(
        tmp_path
        / "rulespec-uk"
        / "regulations/uksi/2013/376/schedule/10/paragraph/1.yaml",
        """format: rulespec/v1
rules:
  - name: premises_treated_as_persons_home_for_paragraphs_1_to_5
    kind: derived
    versions:
      - effective_from: '2026-04-01'
        formula: claimant_occupies_premises
""",
    )

    report = build_policyengine_coverage_report(tmp_path)

    assert report["total_outputs"] == 3
    assert report["status_counts"] == {"known_not_comparable": 3}
    items_by_id = {item["legal_id"]: item for item in report["items"]}
    schedule_4 = items_by_id[
        "uk:regulations/uksi/2013/376/schedule/4/paragraph/22#renters_housing_costs_element_calculated_under_this_part"
    ]
    schedule_5 = items_by_id[
        "uk:regulations/uksi/2013/376/schedule/5/paragraph/9#owner_occupier_housing_costs_element_amount"
    ]
    schedule_10 = items_by_id[
        "uk:regulations/uksi/2013/376/schedule/10/paragraph/1#premises_treated_as_persons_home_for_paragraphs_1_to_5"
    ]
    assert schedule_4["program"] == "universal_credit"
    assert schedule_4["mapping_type"] == "not_comparable"
    assert "Schedule 4 housing-cost calculation helpers" in str(
        schedule_4["rationale"],
    )
    assert schedule_5["program"] == "universal_credit"
    assert schedule_5["mapping_type"] == "not_comparable"
    assert "Schedule 5 owner-occupier housing-cost helpers" in str(
        schedule_5["rationale"],
    )
    assert schedule_10["program"] == "universal_credit"
    assert schedule_10["mapping_type"] == "not_comparable"
    assert "Schedule 10 premises predicates" in str(schedule_10["rationale"])

from pathlib import Path

import pytest

from axiom_encode.oracles.policyengine.coverage import (
    build_policyengine_candidate_report,
    build_policyengine_coverage_report,
)


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
""",
    )

    report = build_policyengine_coverage_report(tmp_path, program="tax")

    assert report["status_counts"] == {
        "comparable": 1,
        "known_not_comparable": 3,
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


def test_policyengine_coverage_treats_ssa_policy_parameters_as_tax(tmp_path):
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

    assert report["total_outputs"] == 2
    statuses_by_id = {item["legal_id"]: item["status"] for item in report["items"]}
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

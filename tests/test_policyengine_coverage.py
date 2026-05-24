from pathlib import Path

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

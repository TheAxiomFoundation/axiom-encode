import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.proof_validator import (
    find_rulespec_proof_issues,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import (
    OracleSubprocessResult,
    ValidatorPipeline,
    _extract_json_object,
    _infer_us_state_code_from_rulespec_path,
    _load_applied_encoding_manifest_source_metadata,
    _normalize_us_tax_filing_status,
    _policyengine_expected_float,
    _policyengine_period_string,
    _policyengine_us_snap_input_aliases,
    _tax_unit_member_aged_flags,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    find_aggregate_exception_predicate_issues,
    find_anaphoric_scope_omission_issues,
    find_broad_application_passthrough_issues,
    find_child_fragment_reencoding_issues,
    find_copied_cross_reference_source_issues,
    find_current_purpose_placeholder_issues,
    find_current_year_final_amount_table_issues,
    find_deferred_output_issues,
    find_deferred_purpose_specific_limitation_issues,
    find_deprecated_source_url_issues,
    find_employer_scoped_entity_issues,
    find_empty_rules_module_issues,
    find_entity_limited_aggregation_order_issues,
    find_exception_test_coverage_issues,
    find_filtered_entity_dependency_issues,
    find_formula_absolute_reference_issues,
    find_formula_date_literal_issues,
    find_helper_only_definition_issues,
    find_import_shape_issues,
    find_judgment_conditional_formula_issues,
    find_judgment_positive_companion_output_issues,
    find_missing_child_exception_import_issues,
    find_missing_derived_companion_output_issues,
    find_missing_same_section_subsection_import_issues,
    find_nonnegative_amount_reduction_issues,
    find_partial_extent_zeroing_issues,
    find_person_scoped_definition_unit_issues,
    find_person_scoped_rate_base_unit_issues,
    find_proof_import_hash_consistency_issues,
    find_proof_import_reference_issues,
    find_relation_aggregate_syntax_issues,
    find_role_limited_relation_scope_issues,
    find_rule_name_path_suffix_issues,
    find_rule_source_metadata_issues,
    find_scoped_exception_category_gate_issues,
    find_shared_statutory_rate_entity_suffix_name_issues,
    find_sibling_rule_name_collision_issues,
    find_source_claim_reference_issues,
    find_source_condition_coverage_issues,
    find_source_limitation_application_issues,
    find_source_scope_consistency_issues,
    find_source_subparagraph_coverage_issues,
    find_source_table_row_scalar_parameter_issues,
    find_source_verification_issues,
    find_tax_filing_status_enum_representation_issues,
    find_tax_filing_status_local_input_issues,
    find_tax_filing_status_surviving_spouse_issues,
    find_tax_filing_status_test_input_issues,
    find_tax_status_component_local_input_issues,
    find_temporal_value_fact_name_issues,
    find_test_input_assignment_issues,
    find_unconsumed_local_exception_output_issues,
    find_ungrounded_numeric_issues,
    find_unused_import_issues,
    find_unused_modifier_parameter_issues,
    find_upstream_placement_issues,
    find_versioned_derived_formula_issues,
    find_zero_branch_test_coverage_issues,
    repair_copied_cross_reference_summary,
    repair_current_year_final_amount_tables,
    repair_nonnegative_amount_reductions,
    repair_source_table_band_scalar_parameters,
    repair_source_table_interval_row_alignment,
    repair_source_table_interval_tests,
)
from axiom_encode.oracles.policyengine.registry import (
    PolicyEngineMapping,
    PolicyEngineOracleRegistry,
    load_policyengine_registry,
)

AXIOM_RULES_PATH = Path("/Users/maxghenis/TheAxiomFoundation/axiom-rules-engine")
AXIOM_RULES_ENGINE_BINARY = AXIOM_RULES_PATH / "target" / "debug" / "axiom-rules-engine"


def test_rulespec_compile_env_exposes_policy_repo_roots(monkeypatch, tmp_path):
    repo_parent = tmp_path / "repos"
    policy_repo = repo_parent / "rulespec-us-ny"
    policy_repo.mkdir(parents=True)
    existing_root = tmp_path / "existing-roots"
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(existing_root))

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )

    roots = pipeline._rulespec_compile_env()["AXIOM_RULESPEC_REPO_ROOTS"].split(
        os.pathsep
    )
    assert roots[:2] == [str(policy_repo), str(repo_parent)]
    assert str(existing_root) in roots


def test_rulespec_validation_run_compiled_uses_current_repo_env(monkeypatch, tmp_path):
    repo_parent = tmp_path / "repos"
    policy_repo = repo_parent / "rulespec-us"
    policy_repo.mkdir(parents=True)
    stale_root = tmp_path / "stale-rulespec-us"
    stale_root.mkdir()
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_root))

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )
    captured_env: dict[str, str] | None = None

    def fake_run(cmd, **kwargs):
        nonlocal captured_env
        captured_env = kwargs.get("env")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "outputs": {
                                "us:statutes/1/1#benefit": {
                                    "kind": "scalar",
                                    "id": "us:statutes/1/1#benefit",
                                    "value": {"kind": "integer", "value": 6},
                                }
                            }
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={"input": {}},
        case_name="computes_benefit",
        case_index=0,
        period={
            "period_kind": "month",
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        output_names=["us:statutes/1/1#benefit"],
        derived_by_key={"us:statutes/1/1#benefit": {"entity": "Household"}},
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        module_target=None,
    )

    assert issues == []
    assert outputs is not None
    assert outputs["us:statutes/1/1#benefit"]["value"]["value"] == 6
    assert captured_env is not None
    roots = captured_env["AXIOM_RULESPEC_REPO_ROOTS"].split(os.pathsep)
    assert roots[:2] == [str(policy_repo), str(repo_parent)]
    assert str(stale_root) in roots


def test_unrelated_same_section_term_import_rejects_other_section_standin(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_checks_policy_repo_for_temp_output(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    generated_section_root = tmp_path / "run/codex/statutes/26/3134"
    generated_section_root.mkdir(parents=True)
    rules_file = generated_section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_rejects_file_level_standin(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    other_root = repo / "statutes/26/45A"
    other_root.mkdir(parents=True)
    (other_root / "b.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Section 45A wage amount.
rules:
  - name: qualified_wages_and_health_costs_taken_into_account
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: wages
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b` overlaps same-section term `qualified_wages` "
        "defined or deferred in `statutes/26/3134/c.yaml`. Import the "
        "same-section output or defer the dependent output instead of using an "
        "unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_rejects_exclusion_citation(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages exclude wages taken into account under section 45A.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_allows_same_section_definition(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
rules:
  - name: qualified_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: wages
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3134/c#qualified_wages
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_unrelated_same_section_term_import_allows_defined_cross_reference(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages have the meaning given by section 45A.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_unrelated_same_section_term_import_allows_narrower_cross_reference_credit(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3131/e#qualified_sick_leave_wages_credit_with_collectively_bargained_contributions_increase
module:
  summary: Applicable employment taxes are reduced by credits allowed under sections 3131 and 3132.
rules:
  - name: employment_tax_limit_for_credit
    kind: derived
    entity: Employer
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: max(0, applicable_employment_taxes - qualified_sick_leave_wages_credit_with_collectively_bargained_contributions_increase)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_rulespec_companion_runner_uses_rows_for_absolute_list_outputs(
    monkeypatch, tmp_path
):
    repo_parent = tmp_path / "repos"
    policy_repo = repo_parent / "rulespec-us"
    policy_repo.mkdir(parents=True)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )
    captured_request: dict[str, object] | None = None

    def fake_run(cmd, **kwargs):
        nonlocal captured_request
        if "input" not in kwargs:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        captured_request = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            "outputs": {
                                "excluded_from_wages": {
                                    "kind": "scalar",
                                    "id": "us:statutes/26/3121/a/6#excluded_from_wages",
                                    "value": {"kind": "money", "value": 300},
                                }
                            }
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={
            "input": {},
            "tables": {
                "Payment": [
                    {
                        "payment_amount": 300,
                    }
                ]
            },
            "output": {
                "us:statutes/26/3121/a/6#excluded_from_wages": [300],
            },
        },
        case_name="excluded_payment",
        case_index=1,
        period={
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        output_names=["excluded_from_wages"],
        output_runtime_keys={
            "us:statutes/26/3121/a/6#excluded_from_wages": "excluded_from_wages",
        },
        derived_by_key={"excluded_from_wages": {"entity": "Payment"}},
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        module_target=None,
    )

    assert issues == []
    assert outputs is not None
    assert captured_request is not None
    assert captured_request["queries"][0]["entity_id"] == "payment-1"


def test_cross_statute_definition_import_check_uses_cited_title_and_existing_targets(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    target = repo / "statutes" / "7" / "2014" / "e.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    rules_file = repo / "statutes" / "7" / "2015" / "e.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Eligibility is described in section 2014(e). A controlled substance is
    defined in section 802 of title 21.
rules: []
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_statute_definition_imports(rules_file)

    assert issues == [
        "Cross-statute definition import missing: source text references "
        "section 2014(e) but file does not import from 7/2014/e"
    ]

    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/7/2014/e#snap_earned_income_deduction
module:
  summary: |-
    Eligibility is described in section 2014(e).
rules: []
"""
    )

    assert pipeline._check_cross_statute_definition_imports(rules_file) == []


def test_formula_absolute_reference_rejects_import_targets_in_formula():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
imports:
  - us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies
"""

    issues = find_formula_absolute_reference_issues(content)

    assert issues == [
        "Formula absolute import reference: "
        "`person_exempt_from_paragraph_1_work_requirements` contains "
        "`us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies` "
        "inside a formula. Add that target to `imports:` and reference the "
        "imported rule by bare local name in formula text."
    ]


def test_versioned_derived_formula_rejects_multiple_formula_versions():
    content = """format: rulespec/v1
rules:
  - name: savers_credit_gross_contributions
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: qualified_retirement_contributions
      - effective_from: '2027-01-01'
        formula: able_account_contributions
  - name: inflation_adjusted_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '50000'
      - effective_from: '2027-01-01'
        formula: '52000'
"""

    issues = find_versioned_derived_formula_issues(content)

    assert len(issues) == 1
    assert "savers_credit_gross_contributions has 2 formula versions" in issues[0]
    assert "Versioned derived formula unsupported" in issues[0]


def test_rule_source_metadata_rejects_executable_rules_without_rule_source():
    content = """format: rulespec/v1
module:
  summary: The standard is 20 hours.
  source_verification:
    corpus_citation_path: us/statute/7/2015
rules:
  - name: minimum_hours
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2026-01-01'
        formula: '20'
  - name: runtime_membership
    kind: data_relation
    data_relation:
      arity: 2
  - name: restates_upstream
    kind: source_relation
    source_relation:
      type: restates
      target: us:statutes/7/2015#minimum_hours
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source metadata required: `minimum_hours` is an executable rule "
        "and must include `source:` with the legal citation/span supporting it."
    ]


def test_rule_source_metadata_rejects_derived_relation_without_rule_source():
    content = """format: rulespec/v1
module:
  summary: Household members who meet the eligibility tests form the SNAP unit.
  source_verification:
    corpus_citation_path: us/regulation/7/273/1
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
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source metadata required: `snap_unit` is an executable rule "
        "and must include `source:` with the legal citation/span supporting it."
    ]


def test_rule_source_metadata_rejects_missing_module_source_locator():
    content = """format: rulespec/v1
module:
  summary: The standard is 20 hours.
rules:
  - name: minimum_hours
    kind: parameter
    dtype: Count
    source: 7 USC 2015(e)(4)
    versions:
      - effective_from: '2026-01-01'
        formula: '20'
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source locator required: module.source_verification must include "
        "`corpus_citation_path` or `corpus_citation_paths` when executable "
        "rules are present."
    ]


def test_missing_derived_companion_output_rejects_uncovered_derived_rule(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    rules_file = policy_repo / "statutes" / "7" / "2015" / "e.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: student_age_exception_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2015(e)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: person_age_years < 18
  - name: student_single_parent_exception_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2015(e)(8)
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_single_parent
"""
    cases = [
        {
            "name": "age_exception",
            "period": "2026-01",
            "input": {"us:statutes/7/2015/e#input.person_age_years": 17},
            "output": {"us:statutes/7/2015/e#student_age_exception_applies": "holds"},
        }
    ]

    issues = find_missing_derived_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Derived rule missing companion output coverage: "
        "`us:statutes/7/2015/e#student_single_parent_exception_applies` "
        "is not asserted by the companion `.test.yaml` file."
    ]


def test_missing_derived_companion_output_uses_origin_repo_prefix_for_temp_checkout(
    tmp_path,
):
    policy_repo = tmp_path / "rulespec-us-clean.abcd"
    rules_file = policy_repo / "statutes" / "26" / "63" / "f.yaml"
    rules_file.parent.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=policy_repo, check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
        ],
        cwd=policy_repo,
        check=True,
        capture_output=True,
    )
    content = """format: rulespec/v1
rules:
  - name: blind_under_subsection_f
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    source: 26 USC 63(f)
    versions:
      - effective_from: '2026-01-01'
        formula: taxpayer_is_blind
"""
    cases = [
        {
            "name": "empty",
            "period": "2026",
            "input": {},
            "output": {},
        }
    ]

    issues = find_missing_derived_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Derived rule missing companion output coverage: "
        "`us:statutes/26/63/f#blind_under_subsection_f` "
        "is not asserted by the companion `.test.yaml` file."
    ]


def test_judgment_positive_companion_output_rejects_never_holds(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    rules_file = policy_repo / "statutes" / "26" / "3102" / "f" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3102(f)(1)
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          tax_is_imposed_by_section_3101_b_2
          and wages_from_employer_in_excess_of_additional_medicare_collection_threshold > 0
"""
    cases = [
        {
            "name": "below_threshold",
            "period": "2026",
            "input": {},
            "output": {
                "us:statutes/26/3102/f/1#subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold": "not_holds"
            },
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Judgment rule missing positive companion output coverage: "
        "`us:statutes/26/3102/f/1#subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold` "
        "is not asserted as `holds` by the companion `.test.yaml` file."
    ]


def test_judgment_positive_companion_output_allows_holds_case(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    rules_file = policy_repo / "statutes" / "26" / "3102" / "f" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: additional_medicare_collection_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3102(f)(1)
    versions:
      - effective_from: '2013-01-01'
        formula: wages_above_threshold > 0
"""
    cases = [
        {
            "name": "above_threshold",
            "period": "2026",
            "input": {},
            "output": {
                "us:statutes/26/3102/f/1#additional_medicare_collection_applies": [
                    "holds"
                ]
            },
        }
    ]

    assert (
        find_judgment_positive_companion_output_issues(
            content,
            cases,
            rules_file=rules_file,
            policy_repo_path=policy_repo,
        )
        == []
    )


def test_test_input_assignment_scopes_inputs_to_asserted_outputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: table_value
    kind: parameter
    dtype: Money
    indexed_by: household_size
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 209
          2: 209
  - name: deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: table_value[household_size]
  - name: asset_limit_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: '3000'
  - name: asset_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: asset_limit_amount
"""
    cases = [
        {
            "name": "asset_limit_does_not_need_household_size",
            "period": "2026-01",
            "input": {},
            "output": {
                "us:policies/usda/snap/fy-2026-cola/deductions#asset_limit": 3000
            },
        }
    ]

    assert find_test_input_assignment_issues(content, cases) == []


def test_same_section_subsection_import_accepts_transitive_child_import(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    child = policy_repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        "format: rulespec/v1\nimports:\n  - us:statutes/7/2015/e\nrules: []\n"
    )
    parent = policy_repo / "statutes" / "7" / "2015" / "d" / "2.yaml"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text("placeholder\n")
    content = """format: rulespec/v1
module:
  summary: |-
    A person shall be exempt if the person is a student, except that a person
    enrolled in an institution of higher education is ineligible unless the
    person meets the requirements of subsection (e) of this section.
imports:
  - us:statutes/7/2015/d/2/C
rules: []
"""

    issues = find_missing_same_section_subsection_import_issues(
        content,
        rules_file=parent,
        policy_repo_path=policy_repo,
    )

    assert issues == []


def test_unused_imports_are_rejected():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/63/c#standard_deduction
  - us:statutes/26/163/a#interest_deduction
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income - standard_deduction
"""

    issues = find_unused_import_issues(content)

    assert issues == [
        "Unused import `us:statutes/26/163/a#interest_deduction`: imported "
        "symbol `interest_deduction` is not referenced by any formula or proof "
        "import."
    ]


def test_proof_imports_must_be_referenced_by_rule_formula():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/151#section_151_exemption_deduction
rules:
  - name: section_931_disallowed_deductions_excluding_section_151
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/151#section_151_exemption_deduction
              output: section_151_exemption_deduction
              hash: sha256:abc
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          min(deductions_before_denial, allocable_deductions)
"""

    issues = find_proof_import_reference_issues(content)

    assert issues == [
        "Proof import not referenced: "
        "`section_931_disallowed_deductions_excluding_section_151` proof imports "
        "`section_151_exemption_deduction`, but the rule formula does not "
        "reference that imported symbol."
    ]


def test_import_shape_rejects_map_entries():
    content = """format: rulespec/v1
imports:
  - target: us:statutes/26/45A/a
    symbols:
      - base_year_1993_indian_employment_costs
rules: []
"""

    issues = find_import_shape_issues(content)

    assert len(issues) == 1
    assert "Import shape invalid" in issues[0]
    assert "imports[0]" in issues[0]
    assert "scalar string" in issues[0]


def test_import_shape_rejects_unprefixed_targets():
    content = """format: rulespec/v1
imports:
  - statutes/26/24/h#ctc_refundable_maximum_under_subsection_h
rules: []
"""

    issues = find_import_shape_issues(content)

    assert len(issues) == 1
    assert "Import target invalid" in issues[0]
    assert "imports[0]" in issues[0]
    assert "absolute RuleSpec targets" in issues[0]


def _mock_corpus_source_text(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline._fetch_corpus_source_text",
        lambda _citation_path: text,
    )


def _write_local_corpus_provision(
    repo_parent: Path,
    citation_path: str,
    body: str = "Authoritative source text.",
) -> None:
    parts = citation_path.split("/")
    provisions_dir = repo_parent / "axiom-corpus" / "data" / "corpus" / "provisions"
    provisions_dir = provisions_dir / parts[0] / parts[1]
    provisions_dir.mkdir(parents=True, exist_ok=True)
    (provisions_dir / "test.jsonl").write_text(
        json.dumps({"citation_path": citation_path, "body": body}) + "\n",
        encoding="utf-8",
    )


def _write_local_source_claim(repo_parent: Path, record: dict) -> None:
    claims_dir = repo_parent / "axiom-corpus" / "data" / "corpus" / "claims" / "us"
    claims_dir.mkdir(parents=True, exist_ok=True)
    (claims_dir / "test.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )


def test_promoted_stub_check_uses_corpus_provisions(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")

    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_promoted_stub_file(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


def test_imported_stub_dependency_check_uses_corpus_provisions(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "root.yaml"
    target_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        "format: rulespec/v1\n"
        "imports:\n"
        "  - statutes/7/2014/e/4#snap_state_uses_child_support_deduction\n"
        "rules: []\n",
        encoding="utf-8",
    )
    _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")

    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_imported_stub_dependencies(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


def test_rulespec_compile_ci_and_grounding(tmp_path, monkeypatch):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    _mock_corpus_source_text(
        monkeypatch,
        "The official source states the standard utility allowance is $451.",
    )

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
  - name: standard_utility_allowance
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          standard_utility_allowance_value
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: base
  period: 2024-01
  input: {}
  output:
    standard_utility_allowance: 451
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_compile_check(rules_file).passed is True
    assert pipeline._run_ci(rules_file).passed is True
    assert extract_embedded_source_text(rules_file.read_text()).startswith(
        "The standard utility allowance"
    )
    assert extract_grounding_values(rules_file.read_text()) == [(1, "451", 451.0)]
    assert [
        (item.name, item.value)
        for item in extract_named_scalar_occurrences(rules_file.read_text())
    ] == [("standard_utility_allowance_value", 451.0)]


def test_rulespec_ci_rejects_repo_backed_friendly_output_keys(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    household_allotment_input: 298
  output:
    snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "must use legal RuleSpec id" in issue
        and "us:statutes/7/2017/a#snap_regular_month_allotment" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_unresolved_output_reference_path(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.household_allotment_input: 298
  output:
    us:statutes/7/9999/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/9999/a#snap_regular_month_allotment` points to a RuleSpec file that could not be resolved"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_input_reference_in_output_position(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.household_allotment_input: 298
  output:
    us:statutes/7/2017/a#input.household_allotment_input: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/2017/a#input.household_allotment_input` resolves to an input slot, which is not allowed here"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_friendly_input_keys(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `snap_maximum_allotment` must use an absolute legal RuleSpec id" in issue
        for issue in result.issues
    )


def test_rulespec_ci_executes_repo_backed_absolute_input_keys(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_input_reference_path(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/9999/a#input.snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `us:statutes/7/9999/a#input.snap_maximum_allotment` points to a RuleSpec file that could not be resolved"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_unresolved_input_reference_fragment(
    tmp_path,
):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2017" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `us:statutes/7/2017/a#snap_maximum_allotment` does not resolve to an input slot, derived rule, or parameter"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_friendly_relation_child_input_keys(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2012" / "j.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `snap_member_is_elderly_or_disabled` must use an absolute legal RuleSpec id"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_executes_repo_backed_absolute_relation_child_input_keys(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2012" / "j.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_relation_reference(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rulespec-us" / "statutes" / "7" / "2012" / "j.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.not_member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "relation input `us:statutes/7/2012/j#relation.not_member_of_household` does not resolve to a declared relation"
        in issue
        for issue in result.issues
    )


def test_oracle_test_extraction_normalizes_legal_output_ids(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: base
  period: 2026-01
  input:
    household_size: 1
  output:
    us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction: 209
"""
    )

    assert tests == [
        {
            "variable": "snap_standard_deduction",
            "raw_variable": "us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction",
            "name": "base",
            "period": "2026-01",
            "inputs": {"household_size": 1},
            "expect": 209,
        }
    ]


def test_oracle_test_extraction_aliases_legal_input_ids(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: wage_tax
  period: 2026
  input:
    us:statutes/26/3101/a#input.wages: 100000
  output:
    us:statutes/26/3101/a#oasdi_wage_tax: 6200
"""
    )

    assert tests[0]["inputs"]["us:statutes/26/3101/a#input.wages"] == 100000
    assert tests[0]["inputs"]["wages"] == 100000


def test_oracle_test_extraction_preserves_policyengine_only_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: utility_region
  period: 2026-01
  input:
    us-ny:regulations/18-nycrr/387/12/f/3/v/a#input.household_resides_in_new_york_city: false
  oracle_inputs:
    policyengine:
      snap_utility_region_str: NY_NAS
  output:
    us-ny:regulations/18-nycrr/387/12/f/3/v/a#snap_standard_utility_allowance: 988
"""
    )

    assert tests[0]["inputs"] == {
        "us-ny:regulations/18-nycrr/387/12/f/3/v/a#input.household_resides_in_new_york_city": False,
        "household_resides_in_new_york_city": False,
    }
    assert tests[0]["oracle_inputs"] == {
        "policyengine": {"snap_utility_region_str": "NY_NAS"}
    }


def test_policyengine_expected_float_normalizes_judgment_expectations():
    assert _policyengine_expected_float("holds") == 1.0
    assert _policyengine_expected_float("not_holds") == 0.0
    assert _policyengine_expected_float(209) == 209.0


def test_oracle_test_extraction_preserves_relation_list_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: relation_case
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    assert tests[0]["inputs"]["us:statutes/7/2012/j#relation.member_of_household"] == [
        {"us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled": True}
    ]


def test_policyengine_oracle_does_not_score_unmapped_outputs(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: mapped
  period: 2026-01
  input: {}
  output:
    us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction: 209
- name: unmapped
  period: 2026-01
  input: {}
  output:
    us:test/fake#unmapped_var: 99
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._should_compare_pe_test_output = lambda *_args, **_kwargs: True
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)
    pipeline._build_pe_scenario_script = lambda *_args, **_kwargs: ""
    pipeline._run_pe_subprocess_detailed = lambda *_args, **_kwargs: (
        OracleSubprocessResult(returncode=0, stdout="RESULT:209\n")
    )

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["passed"] == 1
    assert result.details["coverage"]["unmapped"] == 1


def test_policyengine_oracle_skips_unprojectable_legal_inputs(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: statutory_taxable_income_proof
  period: 2026
  input:
    us:statutes/26/63#input.gross_income: 100000
    us:statutes/26/63#input.deductions_allowed_by_this_chapter_other_than_standard_deduction: 15000
  output:
    us:statutes/26/63#taxable_income: 85000
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")

    ran_policyengine = False

    def run_policyengine(*_args, **_kwargs):
        nonlocal ran_policyengine
        ran_policyengine = True
        return OracleSubprocessResult(returncode=0, stdout="RESULT:0\n")

    pipeline._run_pe_subprocess_detailed = run_policyengine

    result = pipeline._run_policyengine(rules_file)

    assert ran_policyengine is False
    assert result.passed is True
    assert result.score is None
    assert result.details["coverage"]["unsupported"] == 1
    assert any("unprojectable RuleSpec legal input" in issue for issue in result.issues)


def test_policyengine_registry_is_legal_id_keyed():
    registry = load_policyengine_registry()

    mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/e/2#snap_earned_income_deduction",
        country="us",
    )

    assert mapping is not None
    assert mapping.policyengine_variable == "snap_earned_income_deduction"
    assert registry.mapping_for_legal_id("snap_earned_income_deduction") is None
    earned_income_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/e/2#snap_earned_income_deduction_rate",
        country="us",
    )
    assert earned_income_rate_mapping.mapping_type == "parameter_value"
    assert (
        earned_income_rate_mapping.policyengine_parameter
        == "gov.usda.snap.income.deductions.earned_income"
    )
    maximum_allotment_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment_table",
        country="us",
    )
    assert maximum_allotment_mapping.mapping_type == "parameter_value"
    assert (
        maximum_allotment_mapping.policyengine_parameter
        == "gov.usda.snap.max_allotment.main.CONTIGUOUS_US"
    )
    assert maximum_allotment_mapping.parameter_key_input == "household_size"
    gross_income_limit_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc",
        country="us",
    )
    assert gross_income_limit_mapping.policyengine_variable == "snap_fpg"
    assert gross_income_limit_mapping.result_multiplier == 1.3
    section_2014c_net_failure_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_net_income_exceeds_poverty_line",
        country="us",
    )
    assert section_2014c_net_failure_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_net_failure_mapping.policyengine_variable
        == "meets_snap_net_income_test"
    )
    section_2014c_gross_failure_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#household_fails_gross_income_standard",
        country="us",
    )
    assert section_2014c_gross_failure_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_gross_failure_mapping.policyengine_variable
        == "meets_snap_gross_income_test"
    )
    section_2014c_income_ineligible_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#household_ineligible_to_participate_due_to_income_standards",
        country="us",
    )
    assert section_2014c_income_ineligible_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_income_ineligible_mapping.policyengine_variable
        == "is_snap_eligible"
    )
    section_2014c_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_gross_income_excess_rate_over_poverty_line",
        country="us",
    )
    assert section_2014c_rate_mapping.mapping_type == "not_comparable"
    section_2014c_poverty_line_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_income_standard_poverty_line_with_territory_cap",
        country="us",
    )
    assert section_2014c_poverty_line_mapping.mapping_type == "not_comparable"
    assert section_2014c_poverty_line_mapping.policyengine_variable == "snap_fpg"
    shelter_cap_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/deductions#snap_maximum_excess_shelter_deduction_alaska",
        country="us",
    )
    assert shelter_cap_mapping.parameter_keys == (
        "AK_URBAN",
        "AK_RURAL_1",
        "AK_RURAL_2",
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.207.2#snap_initial_month_prorated_allotment",
            country="us",
        ).mapping_type
        == "not_comparable"
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.403.1#manual_specific_output",
            country="us",
        ).match_type
        == "prefix"
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.407.31#snap_standard_utility_allowance",
            country="us",
        ).mapping_type
        == "direct_variable"
    )
    phone_allowance_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/10-ccr-2506-1/4.407.31#snap_individual_utility_allowance",
        country="us",
    )
    assert phone_allowance_mapping.mapping_type == "not_comparable"
    assert phone_allowance_mapping.policyengine_variable == (
        "snap_individual_utility_allowance"
    )
    assert phone_allowance_mapping.candidate_priority == "P4"
    ny_standard_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/a#snap_standard_utility_allowance",
        country="us",
    )
    assert ny_standard_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_standard_allowance_mapping.policyengine_variable
        == "snap_standard_utility_allowance"
    )
    ny_limited_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/b#snap_limited_utility_allowance",
        country="us",
    )
    assert ny_limited_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_limited_allowance_mapping.policyengine_variable
        == "snap_limited_utility_allowance"
    )
    ny_phone_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance",
        country="us",
    )
    assert ny_phone_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_phone_allowance_mapping.policyengine_variable
        == "snap_individual_utility_allowance"
    )
    ny_composition_mapping = registry.mapping_for_legal_id(
        "us-ny:policies/otda/snap/fy-2026-benefit-calculation#snap_allotment",
        country="us",
    )
    assert ny_composition_mapping.mapping_type == "not_comparable"
    ny_initial_proration_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/14/a/1#snap_initial_month_prorated_allotment",
        country="us",
    )
    assert ny_initial_proration_mapping.mapping_type == "not_comparable"
    assert (
        registry.mapping_for_legal_id(
            "us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_telephone_allowance_eligible",
            country="us",
        ).match_type
        == "prefix"
    )
    regular_allotment_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2017/a#snap_regular_month_allotment",
        country="us",
    )
    assert regular_allotment_mapping.mapping_type == "not_comparable"
    assert regular_allotment_mapping.policyengine_variable == "snap_normal_allotment"
    assert regular_allotment_mapping.candidate_priority == "P4"
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/3101/a#oasdi_wage_tax",
            country="us",
        ).policyengine_variable
        == "employee_social_security_tax"
    )
    section_1222_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1222#net_capital_gain",
        country="us",
    )
    assert section_1222_mapping.mapping_type == "not_comparable"
    assert section_1222_mapping.match_type == "prefix"
    assert section_1222_mapping.candidate_priority == "P4"
    section_1211_limit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_loss_limit",
        country="us",
    )
    assert section_1211_limit_mapping.mapping_type == "parameter_value"
    assert (
        section_1211_limit_mapping.policyengine_parameter
        == "gov.irs.ald.loss.capital.max"
    )
    assert section_1211_limit_mapping.parameter_key == "SINGLE"
    section_1211_selected_limit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_loss_limit_by_filing_status",
        country="us",
    )
    assert section_1211_selected_limit_mapping.parameter_key_input == "filing_status"
    section_1211_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_losses_allowed",
        country="us",
    )
    assert section_1211_formula_mapping.mapping_type == "not_comparable"
    assert section_1211_formula_mapping.candidate_priority == "P4"
    section_1212_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1212/a/1#corporation_capital_loss_carryback_to_taxable_year",
        country="us",
    )
    assert section_1212_mapping.mapping_type == "not_comparable"
    assert section_1212_mapping.match_type == "prefix"
    assert section_1212_mapping.candidate_priority == "P4"
    oasdi_wage_base_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3121/a/1#oasdi_wage_base_excess_excluded_remuneration",
        country="us",
    )
    assert oasdi_wage_base_mapping.mapping_type == "not_comparable"
    assert (
        oasdi_wage_base_mapping.policyengine_variable
        == "taxable_earnings_for_social_security"
    )
    oasdi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3101/a#oasdi_wage_tax_rate",
        country="us",
    )
    assert oasdi_rate_mapping.mapping_type == "parameter_value"
    assert (
        oasdi_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.social_security.rate.employee"
    )
    assert oasdi_rate_mapping.comparable is True
    self_employment_oasdi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/a/rate#old_age_survivors_and_disability_insurance_tax_rate",
        country="us",
    )
    assert self_employment_oasdi_rate_mapping.mapping_type == "parameter_value"
    assert (
        self_employment_oasdi_rate_mapping.policyengine_parameter
        == "gov.irs.self_employment.rate.social_security"
    )
    self_employment_oasdi_tax_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax",
        country="us",
    )
    assert self_employment_oasdi_tax_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_oasdi_tax_mapping.policyengine_variable
        == "self_employment_social_security_tax"
    )
    employee_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3101/b/1#hospital_insurance_wage_tax_rate",
        country="us",
    )
    assert employee_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        employee_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.medicare.rate.employee"
    )
    self_employment_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate",
        country="us",
    )
    assert self_employment_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        self_employment_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.self_employment.rate.medicare"
    )
    self_employment_medicare_tax_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/b/1#self_employment_income_tax",
        country="us",
    )
    assert self_employment_medicare_tax_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_medicare_tax_mapping.policyengine_variable
        == "self_employment_medicare_tax"
    )
    rrta_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3201#tier_2_employee_tax",
        country="us",
    )
    assert rrta_tier_2_mapping.mapping_type == "not_comparable"
    rrta_employee_representative_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3211#tier_2_employee_representative_tax",
        country="us",
    )
    assert rrta_employee_representative_tier_2_mapping.mapping_type == "not_comparable"
    rrta_employer_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3221#tier_2_employer_tax",
        country="us",
    )
    assert rrta_employer_tier_2_mapping.mapping_type == "not_comparable"
    employer_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/b#hospital_insurance_employer_tax_rate",
        country="us",
    )
    assert employer_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        employer_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.medicare.rate.employer"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/3111/b#hospital_insurance_employer_tax",
            country="us",
        ).policyengine_variable
        == "employer_medicare_tax"
    )
    qualified_veteran_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/e#veteran_employment_credit_against_subsection_a_tax",
        country="us",
    )
    assert qualified_veteran_credit_mapping.mapping_type == "not_comparable"
    assert qualified_veteran_credit_mapping.match_type == "prefix"
    research_payroll_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/f#research_credit_allowed_against_subsection_a_tax",
        country="us",
    )
    assert research_payroll_credit_mapping.mapping_type == "not_comparable"
    assert research_payroll_credit_mapping.match_type == "prefix"
    assert (
        registry.mapping_for_legal_id(
            "us:policies/irs/rev-proc-2025-32/standard-deduction#basic_standard_deduction_amount",
            country="us",
        ).policyengine_variable
        == "basic_standard_deduction"
    )
    standard_deduction_single_mapping = registry.mapping_for_legal_id(
        "us:policies/irs/rev-proc-2025-32/standard-deduction#standard_deduction_single",
        country="us",
    )
    assert standard_deduction_single_mapping.mapping_type == "parameter_value"
    assert (
        standard_deduction_single_mapping.policyengine_parameter
        == "gov.irs.deductions.standard.amount"
    )
    assert standard_deduction_single_mapping.parameter_key == "SINGLE"
    married_additional_mapping = registry.mapping_for_legal_id(
        "us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse",
        country="us",
    )
    assert married_additional_mapping.mapping_type == "parameter_value"
    assert married_additional_mapping.parameter_keys == (
        "JOINT",
        "SEPARATE",
        "SURVIVING_SPOUSE",
    )
    assert (
        registry.mapping_for_legal_id(
            "us:policies/irs/rev-proc-2025-32/standard-deduction#standard_deduction",
            country="us",
        ).policyengine_variable
        == "standard_deduction"
    )
    basic_standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#basic_standard_deduction",
        country="us",
    )
    assert basic_standard_deduction_mapping.mapping_type == "not_comparable"
    assert (
        basic_standard_deduction_mapping.policyengine_variable
        == "basic_standard_deduction"
    )
    assert basic_standard_deduction_mapping.candidate_priority == "P4"
    additional_standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#additional_standard_deduction",
        country="us",
    )
    assert additional_standard_deduction_mapping.mapping_type == "not_comparable"
    assert (
        additional_standard_deduction_mapping.policyengine_variable
        == "additional_standard_deduction"
    )
    assert additional_standard_deduction_mapping.candidate_priority == "P4"
    standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#standard_deduction",
        country="us",
    )
    assert standard_deduction_mapping.mapping_type == "not_comparable"
    assert standard_deduction_mapping.policyengine_variable == "standard_deduction"
    assert standard_deduction_mapping.candidate_priority == "P4"
    ctc_phase_out_threshold_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/24/h#ctc_phase_out_threshold",
        country="us",
    )
    assert ctc_phase_out_threshold_mapping.mapping_type == "direct_variable"
    assert (
        ctc_phase_out_threshold_mapping.policyengine_variable
        == "ctc_phase_out_threshold"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1401#self_employment_tax",
            country="us",
        ).mapping_type
        == "not_comparable"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1402/a#self_employment_tax_equivalent_deduction_fraction",
            country="us",
        ).policyengine_parameter
        == "gov.irs.ald.misc.employer_share"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1402/a#net_earnings_from_self_employment",
            country="us",
        ).match_type
        == "prefix"
    )
    qualified_tips_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/224#qualified_tips_deduction",
        country="us",
    )
    assert qualified_tips_mapping.mapping_type == "not_comparable"
    assert qualified_tips_mapping.match_type == "prefix"
    qualified_overtime_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/225#qualified_overtime_deduction",
        country="us",
    )
    assert qualified_overtime_mapping.mapping_type == "not_comparable"
    assert qualified_overtime_mapping.match_type == "prefix"
    nonitemizer_charitable_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/170/p#nonitemizer_charitable_deduction",
        country="us",
    )
    assert nonitemizer_charitable_mapping.mapping_type == "not_comparable"
    assert nonitemizer_charitable_mapping.match_type == "prefix"
    senior_deduction_amount_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/151#senior_deduction_base_amount",
        country="us",
    )
    assert senior_deduction_amount_mapping.mapping_type == "parameter_value"
    assert (
        senior_deduction_amount_mapping.policyengine_parameter
        == "gov.irs.deductions.senior_deduction.amount"
    )
    assert senior_deduction_amount_mapping.match_type == "exact"
    section_151_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/151#senior_deduction",
        country="us",
    )
    assert section_151_formula_mapping.mapping_type == "not_comparable"
    assert section_151_formula_mapping.match_type == "prefix"
    qbi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qbi_deduction_rate",
        country="us",
    )
    assert qbi_rate_mapping.mapping_type == "parameter_value"
    assert qbi_rate_mapping.policyengine_parameter == "gov.irs.deductions.qbi.max.rate"
    assert qbi_rate_mapping.match_type == "exact"
    qbi_phasein_other_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qbi_phasein_range_other",
        country="us",
    )
    assert qbi_phasein_other_mapping.mapping_type == "parameter_value"
    assert qbi_phasein_other_mapping.parameter_keys == (
        "SINGLE",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
    )
    qbi_floor_amount_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#minimum_active_qbi_deduction_base",
        country="us",
    )
    assert qbi_floor_amount_mapping.mapping_type == "parameter_value"
    assert qbi_floor_amount_mapping.parameter_key_path == (
        "brackets",
        1,
        "amount",
    )
    qbi_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qualified_business_income_deduction",
        country="us",
    )
    assert qbi_formula_mapping.mapping_type == "not_comparable"
    assert qbi_formula_mapping.match_type == "prefix"
    medical_floor_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/213#medical_expense_agi_floor_rate",
        country="us",
    )
    assert medical_floor_mapping.mapping_type == "parameter_value"
    assert (
        medical_floor_mapping.policyengine_parameter
        == "gov.irs.deductions.itemized.medical.floor"
    )
    assert medical_floor_mapping.match_type == "exact"
    medical_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/213#medical_care_deduction",
        country="us",
    )
    assert medical_formula_mapping.mapping_type == "not_comparable"
    assert medical_formula_mapping.match_type == "prefix"
    filing_requirement_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/6012#individual_income_tax_return_required_under_2018_2025_rule",
        country="us",
    )
    assert filing_requirement_mapping.mapping_type == "not_comparable"
    assert filing_requirement_mapping.match_type == "prefix"
    self_employment_tax_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/164/f#self_employment_tax_deduction",
        country="us",
    )
    assert self_employment_tax_deduction_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_tax_deduction_mapping.policyengine_variable
        == "self_employment_tax_ald"
    )
    social_security_taxable_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/86#social_security_benefits_included_in_gross_income",
        country="us",
    )
    assert social_security_taxable_mapping.mapping_type == "direct_variable"
    assert (
        social_security_taxable_mapping.policyengine_variable
        == "tax_unit_taxable_social_security"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/86#social_security_base_amount_joint",
            country="us",
        ).policyengine_parameter
        == "gov.irs.social_security.taxability.threshold.base.main"
    )
    savers_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit",
        country="us",
    )
    assert savers_credit_mapping.mapping_type == "not_comparable"
    assert savers_credit_mapping.policyengine_variable == "savers_credit_potential"
    savers_credit_cap_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_contribution_cap",
        country="us",
    )
    assert savers_credit_cap_mapping.mapping_type == "parameter_value"
    assert (
        savers_credit_cap_mapping.policyengine_parameter
        == "gov.irs.credits.retirement_saving.contributions_cap"
    )
    savers_credit_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_middle_rate",
        country="us",
    )
    assert savers_credit_rate_mapping.parameter_key_path == ("amounts", 1)
    threshold_multiplier_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_threshold_multiplier",
        country="us",
    )
    assert threshold_multiplier_mapping.mapping_type == "parameter_value"
    assert (
        threshold_multiplier_mapping.policyengine_parameter
        == "gov.irs.credits.retirement_saving.rate.threshold_adjustment"
    )
    assert threshold_multiplier_mapping.parameter_key_input == "filing_status"
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/25B#savers_credit_applicable_percentage",
            country="us",
        ).match_type
        == "prefix"
    )


def test_policyengine_oracle_tracks_not_comparable_without_issue_noise(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: mapped
  period: 2026-01
  input: {}
  output:
    us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction: 209
- name: classified_unsupported
  period: 2026-01
  input: {}
  output:
    us:test/fake#classified_unsupported: 99
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    base_registry = load_policyengine_registry()
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            **base_registry.mappings_by_legal_id,
            "us:test/fake#classified_unsupported": PolicyEngineMapping(
                legal_id="us:test/fake#classified_unsupported",
                country="us",
                mapping_type="not_comparable",
                rationale="synthetic unsupported oracle mapping",
            ),
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)
    pipeline._build_pe_scenario_script = lambda *_args, **_kwargs: ""
    pipeline._run_pe_subprocess_detailed = lambda *_args, **_kwargs: (
        OracleSubprocessResult(returncode=0, stdout="RESULT:209\n")
    )

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["unsupported"] == 1
    assert result.details["coverage"]["unmapped"] == 0


def test_policyengine_oracle_has_no_issue_noise_for_unsupported_only(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: classified_unsupported
  period: 2026-01
  input: {}
  output:
    us:test/fake#classified_unsupported: 99
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        prefix_mappings=(
            PolicyEngineMapping(
                legal_id="us:test/fake#",
                country="us",
                mapping_type="not_comparable",
                match_type="prefix",
                rationale="synthetic unsupported oracle prefix",
            ),
        )
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._should_compare_pe_test_output = lambda *_args, **_kwargs: True

    result = pipeline._run_policyengine(rules_file)

    assert result.score is None
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["unsupported"] == 1
    assert result.details["coverage"]["unmapped"] == 0


def test_policyengine_oracle_uses_policyengine_only_inputs_for_legal_facts(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: section_1401_a_oracle_case
  period: 2024
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1200
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
    us:statutes/26/1402/a#input.partnership_section_702_a_8_income_or_loss: 0
    us:statutes/26/1402/b#input.contribution_and_benefit_base_under_section_230_of_social_security_act: 5000
    us:statutes/26/1402/b#input.wages_paid_to_individual_for_section_1401_a: 100
  oracle_inputs:
    policyengine:
      self_employment_income: 1000
      employment_income: 100
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 114.514
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax": PolicyEngineMapping(
                legal_id="us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax",
                country="us",
                mapping_type="direct_variable",
                policyengine_variable="self_employment_social_security_tax",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:114.514\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["passed"] == 1
    assert "'self_employment_income': {'2024': 1000}" in scripts[0]
    assert "'employment_income': {'2024': 100}" in scripts[0]


def test_policyengine_oracle_rejects_untranslated_legal_facts(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: section_1401_a_untranslated_case
  period: 2024
  input:
    us:statutes/26/1402/a#input.self_employment_trade_or_business_gross_income: 1200
    us:statutes/26/1402/a#input.self_employment_trade_or_business_deductions: 200
  output:
    us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax: 114.514
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax": PolicyEngineMapping(
                legal_id="us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax",
                country="us",
                mapping_type="direct_variable",
                policyengine_variable="self_employment_social_security_tax",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")

    result = pipeline._run_policyengine(rules_file)

    assert result.score is None
    assert result.passed is True
    assert result.details["coverage"]["unsupported"] == 1
    assert result.details["coverage"]["comparable"] == 0
    assert "unprojectable RuleSpec legal input" in result.issues[0]


def test_policyengine_oracle_compares_parameter_value_mapping(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: joint_threshold
  period: 2026
  input:
    us:statutes/26/3101/b/2#input.filing_status: 1
  output:
    us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold: 250000
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold": PolicyEngineMapping(
                legal_id="us:statutes/26/3101/b/2#additional_medicare_wage_tax_threshold",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.payroll.medicare.additional.exclusion",
                parameter_key_input="filing_status",
                parameter_key_map={"0": "SINGLE", "1": "JOINT"},
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:250000\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["passed"] == 1
    assert "gov.irs.payroll.medicare.additional.exclusion" in scripts[0]
    assert 'keys = ["JOINT"]' in scripts[0]


def test_policyengine_oracle_compares_multi_key_parameter_value_mapping(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: married_additional_deduction
  period: 2026
  output:
    us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse: 1650
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse": PolicyEngineMapping(
                legal_id="us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.deductions.standard.aged_or_blind.amount",
                parameter_keys=("JOINT", "SEPARATE", "SURVIVING_SPOUSE"),
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:1650\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert "gov.irs.deductions.standard.aged_or_blind.amount" in scripts[0]
    assert 'keys = ["JOINT", "SEPARATE", "SURVIVING_SPOUSE"]' in scripts[0]


def test_policyengine_oracle_passes_through_parameter_key_input(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: one_person_max_allotment
  period: 2026-01
  input:
    us:policies/usda/snap/fy-2026-cola/maximum-allotments#input.household_size: 1
  output:
    us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment_table: 298
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment_table": PolicyEngineMapping(
                legal_id="us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment_table",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.usda.snap.max_allotment.main.CONTIGUOUS_US",
                parameter_key_input="household_size",
                period="month",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:298\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["passed"] == 1
    assert "gov.usda.snap.max_allotment.main.CONTIGUOUS_US" in scripts[0]
    assert 'keys = ["1"]' in scripts[0]


def test_policyengine_oracle_compares_nested_parameter_key_path(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: single_first_bracket_threshold
  period: 2026
  input:
    us:policies/irs/rev-proc-2025-32/income-tax-brackets#input.filing_status: 0
  output:
    us:policies/irs/rev-proc-2025-32/income-tax-brackets#income_tax_bracket_1_threshold: 12400
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:policies/irs/rev-proc-2025-32/income-tax-brackets#income_tax_bracket_1_threshold": PolicyEngineMapping(
                legal_id="us:policies/irs/rev-proc-2025-32/income-tax-brackets#income_tax_bracket_1_threshold",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.income.bracket.thresholds",
                parameter_key_path=(
                    "1",
                    {
                        "input": "filing_status",
                        "key_map": {"0": "SINGLE", "1": "JOINT"},
                    },
                ),
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:12400\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert "gov.irs.income.bracket.thresholds" in scripts[0]
    assert 'key_paths = [["1", "SINGLE"]]' in scripts[0]


def test_policyengine_oracle_compares_integer_parameter_key_path(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: amt_lower_rate
  period: 2026
  input: {}
  output:
    us:statutes/26/55#amt_lower_rate: 0.26
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:statutes/26/55#amt_lower_rate": PolicyEngineMapping(
                legal_id="us:statutes/26/55#amt_lower_rate",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.income.amt.brackets.rates",
                parameter_key_path=(0,),
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:0.26\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert "gov.irs.income.amt.brackets.rates" in scripts[0]
    assert "key_paths = [[0]]" in scripts[0]


def test_policyengine_oracle_compares_parameter_attribute_key_path(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: saver_credit_middle_rate
  period: 2026
  input: {}
  output:
    us:statutes/26/25B#savers_credit_middle_rate: 0.2
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:statutes/26/25B#savers_credit_middle_rate": PolicyEngineMapping(
                legal_id="us:statutes/26/25B#savers_credit_middle_rate",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.credits.retirement_saving.rate.joint",
                parameter_key_path=("amounts", 1),
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:0.2\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert "gov.irs.credits.retirement_saving.rate.joint" in scripts[0]
    assert 'key_paths = [["amounts", 1]]' in scripts[0]
    assert "getattr(selected, key)" in scripts[0]


def test_policyengine_oracle_compares_parameter_scale_calc_input(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: child_ctc_amount
  period: 2026
  input:
    us:statutes/26/24/h#input.age: 8
  output:
    us:policies/irs/rev-proc-2025-32/child-tax-credit#ctc_child_amount: 2200
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:policies/irs/rev-proc-2025-32/child-tax-credit#ctc_child_amount": PolicyEngineMapping(
                legal_id="us:policies/irs/rev-proc-2025-32/child-tax-credit#ctc_child_amount",
                country="us",
                mapping_type="parameter_value",
                policyengine_parameter="gov.irs.credits.ctc.amount.base",
                parameter_calc_input="age",
                period="year",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:2200\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert "gov.irs.credits.ctc.amount.base" in scripts[0]
    assert "calc_values = [8]" in scripts[0]
    assert "value.calc(calc_value)" in scripts[0]


def test_policyengine_oracle_applies_result_multiplier(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: gross_income_limit
  period: 2026-01
  input:
    household_size: 1
  output:
    us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc: 130
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline.policyengine_registry = PolicyEngineOracleRegistry(
        {
            "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc": PolicyEngineMapping(
                legal_id="us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc",
                country="us",
                mapping_type="direct_variable",
                policyengine_variable="snap_fpg",
                result_multiplier=1.3,
                period="month",
            )
        }
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)

    scripts = []

    def fake_run(script, *_args, **_kwargs):
        scripts.append(script)
        return OracleSubprocessResult(returncode=0, stdout="RESULT:100\n")

    pipeline._run_pe_subprocess_detailed = fake_run

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert "snap_fpg" in scripts[0]


def test_policyengine_resolver_rejects_friendly_us_names(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )

    assert pipeline._resolve_pe_variable("us", "snap_earned_income_deduction") is None
    assert (
        pipeline._resolve_pe_variable(
            "us", "us:statutes/7/2014/e/2#snap_earned_income_deduction"
        )
        == "snap_earned_income_deduction"
    )


def test_policyengine_us_state_inference_uses_rulespec_repo_path(tmp_path):
    rules_file = tmp_path / "rulespec-us-co" / "regulations" / "rules.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\n")

    assert _infer_us_state_code_from_rulespec_path(rules_file) == "CO"
    assert (
        _infer_us_state_code_from_rulespec_path(
            tmp_path / "rules.yaml",
            "imports:\n  - us-ny:regulations/example\n",
        )
        == "NY"
    )


def test_policyengine_snap_input_aliases_derive_standard_pe_inputs():
    aliases = _policyengine_us_snap_input_aliases(
        {
            "employee_wages_received": 1000,
            "household_shelter_costs_incurred": 500,
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage": True,
        }
    )

    assert aliases == {
        "snap_earned_income": 1000.0,
        "snap_gross_income": 1000.0,
        "housing_cost": 500.0,
        "snap_utility_allowance_type": "SUA",
    }


def test_policyengine_snap_input_aliases_derive_upstream_legal_inputs():
    aliases = _policyengine_us_snap_input_aliases(
        {
            "snap_countable_earned_income": 1000,
            "snap_countable_unearned_income": 200,
            "work_supplementation_earned_income": 250,
            "snap_monthly_household_income": 1200,
            "snap_standard_deduction": 209,
            "snap_allowable_shelter_costs": 500,
            "dependent_care_deduction": 50,
            "child_support_deduction": 25,
            "medical_deduction": 10,
            "excess_shelter_deduction": 100,
        }
    )

    assert aliases == {
        "snap_earned_income": 750.0,
        "snap_unearned_income": 200.0,
        "snap_gross_income": 1200.0,
        "housing_cost": 500.0,
        "snap_utility_allowance_type": "NONE",
        "snap_dependent_care_deduction": 50.0,
        "snap_child_support_deduction": 25.0,
        "snap_excess_medical_expense_deduction": 10.0,
        "snap_excess_shelter_expense_deduction": 100.0,
    }


def test_policyengine_snap_input_aliases_read_legal_rule_keys():
    aliases = _policyengine_us_snap_input_aliases(
        {
            "us:regulations/7-cfr/273/10#input.snap_countable_earned_income": 1000,
            "us:regulations/7-cfr/273/10#input.snap_countable_unearned_income": 0,
            "us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction": 209,
            "us:regulations/7-cfr/273/10#input.snap_allowable_shelter_costs": 1094,
        }
    )

    assert aliases == {
        "snap_earned_income": 1000.0,
        "snap_unearned_income": 0.0,
        "snap_gross_income": 1000.0,
        "snap_standard_deduction": 209.0,
        "housing_cost": 1094.0,
        "snap_utility_allowance_type": "NONE",
    }


def test_policyengine_snap_input_aliases_read_relation_list_member_facts():
    aliases = _policyengine_us_snap_input_aliases(
        {
            "us:statutes/7/2012/j#relation.member_of_household": [
                {"us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled": True}
            ],
        }
    )

    assert aliases == {
        "snap_household_has_elderly_or_disabled_member": True,
        "has_usda_elderly_disabled": True,
    }


def test_policyengine_snap_input_aliases_derive_utility_allowance_type():
    aliases = _policyengine_us_snap_input_aliases(
        {
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage": False,
            "household_pays_electricity_utility_cost": True,
            "household_pays_water_utility_cost": True,
            "household_pays_telephone_service_cost": False,
        }
    )

    assert aliases["snap_utility_allowance_type"] == "LUA"
    aliases = _policyengine_us_snap_input_aliases(
        {
            "household_incurred_or_anticipated_heating_or_cooling_costs_separate_from_rent_or_mortgage": False,
            "household_pays_electricity_utility_cost": False,
            "household_pays_water_utility_cost": False,
            "household_pays_telephone_service_cost": True,
        }
    )
    assert aliases["snap_utility_allowance_type"] == "NONE"


def test_policyengine_snap_net_income_annualizes_housing_cost(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "snap_net_income",
        {
            "period": "2026-01",
            "employee_wages_received": 1000,
            "household_shelter_costs_incurred": 500,
        },
        "2026",
    )

    assert "'housing_cost': {'2026': 6000}" in script
    assert "'housing_cost': {'2026-01':" not in script


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "SINGLE"),
        (1, "JOINT"),
        (2, "SEPARATE"),
        (3, "HEAD_OF_HOUSEHOLD"),
        ("married_filing_jointly", "JOINT"),
        ("HOH", "HEAD_OF_HOUSEHOLD"),
    ],
)
def test_policyengine_tax_filing_status_normalization(value, expected):
    assert _normalize_us_tax_filing_status(value) == expected


def test_policyengine_tax_scenario_builds_net_investment_income_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "net_investment_income_tax",
        {
            "period": "2026",
            "filing_status": 0,
            "adjusted_gross_income": 205000,
            "taxable_interest_income": 1000,
            "dividend_income": 2000,
            "rental_income": 3000,
            "taxable_net_gain_from_dispositions": 4000,
        },
        "2026",
    )

    assert "'taxable_interest_income': {'2026': 1000}" in script
    assert "'dividend_income': {'2026': 2000}" in script
    assert "'rental_income': {'2026': 3000}" in script
    assert "'loss_limited_net_capital_gains': {'2026': 4000}" in script


def test_policyengine_tax_scenario_builds_capital_gains_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "capital_gains_tax",
        {
            "period": "2026",
            "filing_status": 0,
            "taxable_income": 100000,
            "long_term_capital_gains": 40000,
            "short_term_capital_gains": 0,
            "qualified_dividend_income": 5000,
            "unrecaptured_section_1250_gain": 10000,
            "capital_gains_28_percent_rate_gain": 2000,
        },
        "2026",
    )

    assert "'taxable_income': {'2026': 100000}" in script
    assert "'unrecaptured_section_1250_gain': {'2026': 10000}" in script
    assert "'capital_gains_28_percent_rate_gain': {'2026': 2000}" in script
    assert "'long_term_capital_gains': {'2026': 40000}" in script
    assert "'short_term_capital_gains': {'2026': 0}" in script
    assert "'qualified_dividend_income': {'2026': 5000}" in script


def test_policyengine_tax_scenario_builds_taxable_income_deduction_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "taxable_income",
        {
            "period": "2026",
            "filing_status": 0,
            "adjusted_gross_income": 80000,
            "exemptions": 0,
            "tax_unit_itemizes": True,
            "itemized_taxable_income_deductions": 30000,
            "standard_deduction": 15000,
            "qualified_business_income_deduction": 2000,
            "wagering_losses_deduction": 500,
            "charitable_deduction_for_non_itemizers": 1000,
            "tip_income_deduction": 500,
            "overtime_income_deduction": 600,
            "additional_senior_deduction": 700,
            "auto_loan_interest_deduction": 800,
        },
        "2026",
    )

    assert "'adjusted_gross_income': {'2026': 80000}" in script
    assert "'exemptions': {'2026': 0}" in script
    assert "'tax_unit_itemizes': {'2026': True}" in script
    assert "'itemized_taxable_income_deductions': {'2026': 30000}" in script
    assert "'qualified_business_income_deduction': {'2026': 2000}" in script
    assert "'wagering_losses_deduction': {'2026': 500}" in script
    assert "'tip_income_deduction': {'2026': 500}" in script


def test_policyengine_tax_scenario_builds_amt_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "alternative_minimum_tax",
        {
            "period": "2026",
            "filing_status": 2,
            "taxable_income": 650000,
            "standard_deduction": 16100,
            "tax_unit_itemizes": False,
            "exemptions": 0,
            "income_tax_main_rates": 150000,
            "regular_tax_before_credits": 160000,
            "capital_gains_tax": 0,
            "amt_part_iii_required": False,
            "amt_tax_including_capital_gains": 0,
            "alternative_minimum_tax_foreign_tax_credit": 0,
            "form_4972_lumpsum_distributions": 0,
            "amt_kiddie_tax_applies": False,
        },
        "2026",
    )

    assert "'taxable_income': {'2026': 650000}" in script
    assert "'standard_deduction': {'2026': 16100}" in script
    assert "'tax_unit_itemizes': {'2026': False}" in script
    assert "'income_tax_main_rates': {'2026': 150000}" in script
    assert "'regular_tax_before_credits': {'2026': 160000}" in script
    assert "'capital_gains_tax': {'2026': 0}" in script
    assert "'amt_part_iii_required': {'2026': False}" in script
    assert "'amt_tax_including_cg': {'2026': 0}" in script
    assert "'foreign_tax_credit_potential': {'2026': 0}" in script
    assert "'form_4972_lumpsum_distributions': {'2026': 0}" in script
    assert "'amt_kiddie_tax_applies': {'2026': False}" in script


def test_policyengine_tax_scenario_builds_nonrefundable_credit_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "income_tax_capped_non_refundable_credits",
        {
            "period": "2026",
            "income_tax_before_credits": 1200,
            "foreign_tax_credit": 100,
            "cdcc": 500,
            "non_refundable_american_opportunity_credit": 1500,
            "lifetime_learning_credit": 600,
            "savers_credit": 200,
            "residential_clean_energy_credit": 300,
            "energy_efficient_home_improvement_credit": 100,
            "elderly_disabled_credit": 750,
            "new_clean_vehicle_credit": 1000,
            "used_clean_vehicle_credit": 400,
            "non_refundable_ctc": 2000,
            "net_investment_income_tax": 380,
            "recapture_of_investment_credit": 50,
            "unreported_payroll_tax": 20,
            "qualified_retirement_penalty": 100,
        },
        "2026",
    )

    assert "'income_tax_before_credits': {'2026': 1200}" in script
    assert "'foreign_tax_credit': {'2026': 100}" in script
    assert "'cdcc': {'2026': 500}" in script
    assert "'non_refundable_american_opportunity_credit': {'2026': 1500}" in script
    assert "'lifetime_learning_credit': {'2026': 600}" in script
    assert "'savers_credit': {'2026': 200}" in script
    assert "'residential_clean_energy_credit': {'2026': 300}" in script
    assert "'energy_efficient_home_improvement_credit': {'2026': 100}" in script
    assert "'elderly_disabled_credit': {'2026': 750}" in script
    assert "'new_clean_vehicle_credit': {'2026': 1000}" in script
    assert "'used_clean_vehicle_credit': {'2026': 400}" in script
    assert "'non_refundable_ctc': {'2026': 2000}" in script
    assert "'net_investment_income_tax': {'2026': 380}" in script
    assert "'recapture_of_investment_credit': {'2026': 50}" in script
    assert "'unreported_payroll_tax': {'2026': 20}" in script
    assert "'qualified_retirement_penalty': {'2026': 100}" in script


def test_policyengine_tax_scenario_builds_refundable_credit_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "income_tax",
        {
            "period": "2026",
            "income_tax_before_refundable_credits": 3100,
            "eitc": 1000,
            "refundable_american_opportunity_credit": 500,
            "refundable_ctc": 1200,
            "recovery_rebate_credit": 0,
            "refundable_payroll_tax_credit": 100,
        },
        "2026",
    )

    assert "'income_tax_before_refundable_credits': {'2026': 3100}" in script
    assert "'eitc': {'2026': 1000}" in script
    assert "'refundable_american_opportunity_credit': {'2026': 500}" in script
    assert "'refundable_ctc': {'2026': 1200}" in script
    assert "'recovery_rebate_credit': {'2026': 0}" in script
    assert "'refundable_payroll_tax_credit': {'2026': 100}" in script


def test_policyengine_tax_scenario_skips_unmodelled_niit_components(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    mappable, reason = pipeline._is_pe_test_mappable(
        "us",
        "net_investment_income_tax",
        {"annuity_income": 100},
        pe_var="net_investment_income_tax",
    )

    assert not mappable
    assert "section 1411(c)/(d)" in (reason or "")


def test_policyengine_period_string_normalizes_rulespec_period_dicts():
    assert (
        _policyengine_period_string(
            {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            }
        )
        == "2026"
    )
    assert (
        _policyengine_period_string(
            {
                "period_kind": "month",
                "start": "2026-01-01",
                "end": "2026-01-31",
            }
        )
        == "2026-01"
    )


def test_policyengine_tax_unit_member_aged_flags_accept_bool_shapes():
    assert _tax_unit_member_aged_flags({"member_of_tax_unit": False}) == [False]
    assert _tax_unit_member_aged_flags({"member_of_tax_unit": [True, False]}) == [
        True,
        False,
    ]
    assert _tax_unit_member_aged_flags(
        {
            "member_of_tax_unit": [
                {"is_aged_65_or_over": True},
                {"is_aged_65_or_over": False},
            ]
        }
    ) == [True, False]
    assert _tax_unit_member_aged_flags(
        {
            "us:statutes/26/22#relation.elderly_disabled_member_of_tax_unit": [
                {"us:statutes/26/22#input.age": 70},
                {"us:statutes/26/22#input.is_aged_65_or_over": False},
            ]
        }
    ) == [True, False]


def test_policyengine_tax_scenario_uses_tax_unit_status_and_aged_flags(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "standard_deduction",
        {
            "period": "2026",
            "filing_status": 1,
            "member_of_tax_unit": [True, False],
        },
        "2026",
    )

    assert "'filing_status': {'2026': 'JOINT'}" in script
    assert "'adult': {'age': {'2026': 65}}" in script
    assert "'spouse': {'age': {'2026': 30}}" in script


def test_policyengine_tax_scenario_uses_relation_rows_for_filer_and_spouse(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "section_22_income",
        {
            "period": "2026",
            "filing_status": 1,
            "us:statutes/26/22#input.pension_annuity_disability_benefits_received": 1000,
            "us:statutes/26/22#input.taxable_pension_annuity_disability_benefits_included": 400,
            "us:statutes/26/22#relation.elderly_disabled_member_of_tax_unit": [
                {
                    "us:statutes/26/22#input.age": 70,
                    "us:statutes/26/22#input.section_22_disability_income": 0,
                },
                {
                    "us:statutes/26/22#input.age": 60,
                    "us:statutes/26/22#input.section_22_disability_income": 2000,
                    "us:statutes/26/22#input.retired_on_disability_before_year_end": True,
                    "us:statutes/26/22#input.unable_to_engage_substantial_gainful_activity": True,
                    "us:statutes/26/22#input.medically_determinable_impairment": True,
                    "us:statutes/26/22#input.impairment_expected_to_result_in_death": False,
                    "us:statutes/26/22#input.impairment_duration_months": 12,
                    "us:statutes/26/22#input.disability_proof_furnished": True,
                },
            ],
        },
        "2026",
    )

    assert "'adult': {'age': {'2026': 70}" in script
    assert "'spouse': {'age': {'2026': 60}" in script
    assert "'pension_income': {'2026': 1000}" in script
    assert "'taxable_pension_income': {'2026': 400}" in script
    assert "'total_disability_payments': {'2026': 2000}" in script
    assert "'retired_on_total_disability': {'2026': True}" in script


def test_policyengine_tax_scenario_applies_tax_unit_overrides(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "regular_tax_before_credits",
        {
            "period": "2026",
            "filing_status": 0,
            "taxable_income": 50000,
            "adjusted_gross_income": 65000,
        },
        "2026",
    )

    assert "'taxable_income': {'2026': 50000}" in script
    assert "'adjusted_gross_income': {'2026': 65000}" in script
    assert "'regular_tax_before_credits':" not in script


def test_policyengine_tax_scenario_applies_cdcc_overrides(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "cdcc",
        {
            "period": "2026",
            "filing_status": 0,
            "tax_unit_childcare_expenses": 5000,
            "min_head_spouse_earned": 20000,
            "income_tax_before_credits": 4000,
            "foreign_tax_credit": 1000,
        },
        "2026",
    )

    assert "'tax_unit_childcare_expenses': {'2026': 5000}" in script
    assert "'min_head_spouse_earned': {'2026': 20000}" in script
    assert "'income_tax_before_credits': {'2026': 4000}" in script
    assert "'foreign_tax_credit': {'2026': 1000}" in script
    assert "'cdcc':" not in script


def test_policyengine_tax_scenario_applies_ctc_refundability_overrides(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "refundable_ctc",
        {
            "period": "2026",
            "filer_adjusted_earnings": 20000,
            "ctc_limiting_tax_liability": 1000,
            "employee_social_security_tax": 1200,
            "employee_medicare_tax": 300,
            "self_employment_tax_ald": 500,
            "excess_payroll_tax_withheld": 100,
        },
        "2026",
    )

    assert "'filer_adjusted_earnings': {'2026': 20000}" in script
    assert "'ctc_limiting_tax_liability': {'2026': 1000}" in script
    assert "'employee_social_security_tax': {'2026': 1200}" in script
    assert "'employee_medicare_tax': {'2026': 300}" in script
    assert "'self_employment_tax_ald': {'2026': 500}" in script
    assert "'excess_payroll_tax_withheld': {'2026': 100}" in script


def test_policyengine_tax_scenario_preserves_relation_member_ages(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "count_cdcc_eligible",
        {
            "period": "2026",
            "filing_status": 0,
            "us:statutes/26/21#relation.cdcc_member_of_tax_unit": [
                {
                    "us:statutes/26/21#input.is_tax_unit_dependent": True,
                    "us:statutes/26/21#input.age": 14,
                },
                {
                    "us:statutes/26/21#input.is_tax_unit_dependent": True,
                    "us:statutes/26/21#input.age": 30,
                    "us:statutes/26/21#input.is_incapable_of_self_care": True,
                },
            ],
        },
        "2026",
    )

    assert "'child0': {'age': {'2026': 14}" in script
    assert "'adult_dep0': {'age': {'2026': 30}" in script
    assert "'is_incapable_of_self_care': {'2026': True}" in script


def test_policyengine_tax_scenario_builds_ctc_dependents_from_relation_rows(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "ctc",
        {
            "period": "2026",
            "filing_status": 0,
            "adjusted_gross_income": 50000,
            "us:statutes/26/24#relation.member_of_tax_unit": [
                {
                    "us:statutes/26/24#input.is_tax_unit_dependent": True,
                    "us:statutes/26/24#input.age": 8,
                },
                {
                    "us:statutes/26/24#input.is_tax_unit_dependent": True,
                    "us:statutes/26/24#input.age": 19,
                },
            ],
        },
        "2026",
    )

    assert (
        "'child0': {'age': {'2026': 8}, 'is_tax_unit_dependent': {'2026': True}}"
        in script
    )
    assert (
        "'adult_dep0': {'age': {'2026': 19}, 'is_tax_unit_dependent': {'2026': True}}"
        in script
    )
    assert "'adjusted_gross_income': {'2026': 50000}" in script


def test_policyengine_tax_scenario_builds_education_credit_students_from_relation_rows(
    tmp_path,
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    script = pipeline._build_pe_us_scenario_script(
        "american_opportunity_credit",
        {
            "period": "2026",
            "filing_status": 0,
            "modified_adjusted_gross_income": 50000,
            "us:statutes/26/25A#relation.education_credit_member_of_tax_unit": [
                {
                    "us:statutes/26/25A#input.is_tax_unit_dependent": True,
                    "us:statutes/26/25A#input.is_taxpayer": False,
                    "us:statutes/26/25A#input.is_spouse": False,
                    "us:statutes/26/25A#input.qualified_tuition_and_related_expenses": 5000,
                    "us:statutes/26/25A#input.excludable_educational_assistance": 500,
                    "us:statutes/26/25A#input.meets_higher_education_act_student_requirements": True,
                    "us:statutes/26/25A#input.at_least_half_time_student": True,
                    "us:statutes/26/25A#input.aotc_prior_year_election_count": 0,
                    "us:statutes/26/25A#input.completed_first_four_years_postsecondary_before_year": False,
                    "us:statutes/26/25A#input.has_felony_drug_conviction": False,
                    "us:statutes/26/25A#input.aotc_election_in_effect": True,
                    "us:statutes/26/25A#input.education_credit_identification_requirements_met": True,
                    "us:statutes/26/25A#input.institution_employer_identification_number_included": True,
                    "us:statutes/26/25A#input.payee_statement_received": True,
                    "us:statutes/26/25A#input.aotc_disallowance_period_applies": False,
                }
            ],
        },
        "2026",
    )

    assert "'qualified_tuition_expenses': {'2026': 4500.0}" in script
    assert "'is_eligible_for_american_opportunity_credit': {'2026': True}" in script
    assert "'adjusted_gross_income': {'2026': 50000}" in script


def test_reviewer_score_below_threshold_fails_even_if_declared_passed(
    monkeypatch, tmp_path
):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    def fake_run_claude_code(*_args, **_kwargs):
        return ('{"score": 2.0, "passed": true, "issues": []}', 0)

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.run_claude_code",
        fake_run_claude_code,
    )

    result = pipeline._run_reviewer("Formula Reviewer", rules_file)

    assert result.score == 2.0
    assert result.passed is False
    assert any(
        "reviewer_score_below_pass_threshold" in issue for issue in result.issues
    )


def test_rulespec_grounding_tolerates_decimal_percentage_float_noise():
    content = """format: rulespec/v1
module:
  summary: The tax is 2.9 percent of self-employment income.
rules:
  - name: hospital_insurance_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: '0.029'
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="The tax is 2.9 percent of self-employment income.",
        )
        == []
    )


def test_rulespec_grounding_does_not_trust_module_summary():
    content = """format: rulespec/v1
module:
  summary: The standard deduction is 16100.
rules:
  - name: standard_deduction_single
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '16100'
"""

    issues = find_ungrounded_numeric_issues(content)

    assert len(issues) == 1
    assert "Numeric source required" in issues[0]
    assert "`module.summary` is not accepted" in issues[0]


def test_rulespec_grounding_uses_declared_corpus_source_text(monkeypatch):
    content = """format: rulespec/v1
module:
  summary: A human summary is not numeric source text.
  source_verification:
    corpus_citation_path: us/guidance/example/source
rules:
  - name: standard_deduction_single
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '16100'
"""

    def fake_fetch(citation_path: str) -> str | None:
        assert citation_path == "us/guidance/example/source"
        return "The official source states $16,100 for this amount."

    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline._fetch_corpus_source_text",
        fake_fetch,
    )

    assert find_ungrounded_numeric_issues(content) == []


def test_rulespec_grounding_accepts_decimal_rates_in_percentage_table_context():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/example/rates
rules:
  - name: phase_in_rates
    kind: parameter
    dtype: Rate
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0765
          1: 0.34
"""

    source_text = (
        "The credit percentage and the phaseout percentage are determined as "
        "follows. The no-children row appears later in the table at 7.65. "
        "The one-child row is 34."
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []


def test_rulespec_grounding_allows_generated_band_selector_keys():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3241
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if ratio < 2.5: 0 else: if ratio < 3.0: 4 else: 10
  - name: rate_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0
          4: 1
          10: 2
"""

    source_text = (
        "The table includes average account benefits ratio cutoffs 2.5 and 3.0."
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []


def test_rulespec_grounding_rejects_ungrounded_index_like_integer_outputs():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/example/index
rules:
  - name: cost_of_living_index
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if eligible: 10 else: 0
"""

    issues = find_ungrounded_numeric_issues(
        content,
        source_text="The source describes eligibility but does not state the index value.",
    )

    assert len(issues) == 1
    assert "10" in issues[0]


def test_rulespec_grounding_accepts_slash_separated_source_measure_denominator():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/63
rules:
  - name: blindness_central_visual_acuity_denominator
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2026-01-01'
        formula: '200'
"""

    source_text = "Central visual acuity does not exceed 20/200."

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert {20.0, 200.0}.issubset(extract_numbers_from_text(source_text))


def test_rulespec_rejects_legacy_source_url_metadata():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/source
rules:
  - name: standard_deduction_single
    kind: parameter
    dtype: Money
    unit: USD
    source_url: https://example.gov/source
    versions:
      - effective_from: '2026-01-01'
        formula: '16100'
"""

    issues = find_deprecated_source_url_issues(content)

    assert len(issues) == 1
    assert "Legacy source URL metadata not allowed" in issues[0]
    assert "rules.standard_deduction_single.source_url" in issues[0]


def test_rulespec_accepts_accepted_source_claim_reference(tmp_path, monkeypatch):
    repo_parent = tmp_path / "repos"
    corpus_repo = repo_parent / "axiom-corpus"
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(corpus_repo))
    validator_pipeline._fetch_local_source_claim_record.cache_clear()
    validator_pipeline._fetch_corpus_source_text.cache_clear()
    validator_pipeline._fetch_local_corpus_source_text.cache_clear()

    _write_local_corpus_provision(
        repo_parent,
        "us/guidance/example/page-1",
        "Table 1 sets the monthly maximum allotment for household size 1 at $298.",
    )
    _write_local_source_claim(
        repo_parent,
        {
            "id": "claims:us/guidance/example/page-1#sets-maximum-allotment",
            "kind": "sets",
            "status": "accepted",
            "subject": {
                "type": "statutory_rule_slot",
                "id": "us:statutes/7/2017/a#snap_allotment_before_minimum.input.snap_maximum_allotment",
                "statutory_reference": "7 USC 2017(a)",
                "corpus_citation_path": "us/statute/7/2017",
            },
            "object": {
                "type": "parameter_table",
                "unit": "USD",
                "effective_from": "2025-10-01",
                "effective_to": "2026-09-30",
            },
            "evidence": [
                {
                    "corpus_citation_path": "us/guidance/example/page-1",
                    "quote": "Table 1 sets the monthly maximum allotment",
                }
            ],
            "provenance": {"method": "manual"},
        },
    )

    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
  source_claims:
    - claims:us/guidance/example/page-1#sets-maximum-allotment
rules: []
"""

    assert find_source_claim_reference_issues(content) == []


def test_rulespec_rejects_executable_or_unaccepted_source_claim(tmp_path, monkeypatch):
    repo_parent = tmp_path / "repos"
    corpus_repo = repo_parent / "axiom-corpus"
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(corpus_repo))
    validator_pipeline._fetch_local_source_claim_record.cache_clear()

    _write_local_source_claim(
        repo_parent,
        {
            "id": "claims:us/guidance/example/page-1#sets-maximum-allotment",
            "kind": "sets",
            "status": "proposed",
            "subject": {
                "type": "statutory_rule_slot",
                "id": "us:statutes/7/2017/a#snap_allotment_before_minimum.input.snap_maximum_allotment",
                "statutory_reference": "7 USC 2017(a)",
                "corpus_citation_path": "us/statute/7/2017",
            },
            "formula": "if household_size == 1: 298 else: 0",
            "evidence": [
                {"corpus_citation_path": "us/guidance/example/page-1"},
            ],
        },
    )

    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
  source_claims:
    - claims:us/guidance/example/page-1#sets-maximum-allotment
rules: []
"""

    issues = find_source_claim_reference_issues(content)

    assert any("Source claim not accepted" in issue for issue in issues)
    assert any("Source claim is executable" in issue for issue in issues)


def test_rulespec_rejects_source_claim_placeholder_subject(tmp_path, monkeypatch):
    repo_parent = tmp_path / "repos"
    corpus_repo = repo_parent / "axiom-corpus"
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(corpus_repo))
    validator_pipeline._fetch_local_source_claim_record.cache_clear()

    _write_local_source_claim(
        repo_parent,
        {
            "id": "claims:us/guidance/example/page-1#sets-maximum-allotment",
            "kind": "sets",
            "status": "accepted",
            "subject": {"type": "concept", "id": "snap.maximum_allotment"},
            "evidence": [
                {"corpus_citation_path": "us/guidance/example/page-1"},
            ],
        },
    )

    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
  source_claims:
    - claims:us/guidance/example/page-1#sets-maximum-allotment
rules: []
"""

    issues = find_source_claim_reference_issues(content)

    assert any("Source claim subject target invalid" in issue for issue in issues)
    assert any(
        "Source claim subject placeholder not allowed" in issue for issue in issues
    )


def test_rulespec_proof_validator_accepts_direct_source_and_claim_atom():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      snap_maximum_allotment_table:
        1: 298
  source_claims:
    - claims:us/guidance/example/page-1#sets-maximum-allotment
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us/guidance/example/page-1
              value_key: snap_maximum_allotment_table
              table:
                header: Maximum Allotment
                row_key: household_size
                column_key: amount
            claim:
              id: claims:us/guidance/example/page-1#sets-maximum-allotment
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
"""

    result = validate_rulespec_proofs(content)

    assert result.passed is True
    assert result.proof_required is True
    assert result.atoms_checked == 1
    assert result.issues == []


def test_rulespec_proof_validator_checks_declared_source_claim_records(
    tmp_path, monkeypatch
):
    repo_parent = tmp_path / "repos"
    corpus_repo = repo_parent / "axiom-corpus"
    monkeypatch.setenv("AXIOM_CORPUS_REPO", str(corpus_repo))
    validator_pipeline._fetch_local_source_claim_record.cache_clear()
    validator_pipeline._fetch_corpus_source_text.cache_clear()
    validator_pipeline._fetch_local_corpus_source_text.cache_clear()

    _write_local_corpus_provision(
        repo_parent,
        "us/guidance/example/page-1",
        "Table 1 sets the monthly maximum allotment for household size 1 at $298.",
    )
    _write_local_source_claim(
        repo_parent,
        {
            "id": "claims:us/guidance/example/page-1#sets-maximum-allotment",
            "kind": "sets",
            "status": "accepted",
            "subject": {
                "type": "statutory_rule_slot",
                "id": "us:statutes/7/2017/a#snap_allotment_before_minimum.input.snap_maximum_allotment",
            },
            "evidence": [
                {
                    "corpus_citation_path": "us/guidance/example/page-1",
                    "quote": "Table 1 sets the monthly maximum allotment",
                }
            ],
        },
    )

    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      snap_maximum_allotment_table:
        1: 298
  source_claims:
    - claims:us/guidance/example/page-1#sets-maximum-allotment
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    metadata:
      proof:
        atoms:
          - path: versions[0].values
            kind: parameter_table
            source:
              corpus_citation_path: us/guidance/example/page-1
              value_key: snap_maximum_allotment_table
              table:
                header: Maximum Allotment
                row_key: household_size
                column_key: amount
            claim:
              id: claims:us/guidance/example/page-1#sets-maximum-allotment
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
"""

    result = validate_rulespec_proofs(content, validate_claim_records=True)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_proof_validator_rejects_missing_source_claim_record():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      source_amount: 298
  source_claims:
    - claims:us/guidance/example/page-1#missing
rules:
  - name: amount
    kind: parameter
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              corpus_citation_path: us/guidance/example/page-1
              value_key: source_amount
            claim:
              id: claims:us/guidance/example/page-1#missing
    versions:
      - effective_from: '2025-10-01'
        formula: '298'
"""

    result = validate_rulespec_proofs(content, validate_claim_records=True)

    assert result.passed is False
    assert any("Source claim missing" in issue for issue in result.issues)


def test_rulespec_proof_validator_rejects_missing_and_unscoped_proofs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      source_amount: 298
  source_claims:
    - claims:us/guidance/example/page-1#sets-amount
rules:
  - name: missing_proof_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2025-10-01'
        formula: '298'
  - name: malformed_proof_amount
    kind: parameter
    dtype: Money
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: table_cell
            source:
              corpus_citation_path: us/guidance/example/page-2
              value_key: absent_amount
              table:
                header: Amount table
                row: household size 1
            claim:
              id: claims:us/guidance/example/page-1#sets-other-amount
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/7/2017/a#snap_regular_month_allotment
              output: snap_regular_month_allotment
              hash: compiled-export
    versions:
      - effective_from: '2025-10-01'
        formula: '298'
"""

    issues = find_rulespec_proof_issues(content)

    assert any("Proof missing" in issue for issue in issues)
    assert any("Proof source outside RuleSpec source" in issue for issue in issues)
    assert any("Proof source value key missing" in issue for issue in issues)
    assert any("Proof table cell provenance incomplete" in issue for issue in issues)
    assert any("Proof claim outside declared claims" in issue for issue in issues)
    assert any("Proof import hash invalid" in issue for issue in issues)


def test_proof_import_hash_consistency_rejects_same_file_content_hash(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes/26/22.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/22#section_22_age_threshold
              output: section_22_age_threshold
              hash: sha256:abc123
    versions:
      - effective_from: '2026-01-01'
        formula: age >= section_22_age_threshold
"""
    rules_file.write_text(content)

    issues = find_proof_import_hash_consistency_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert any("expected `sha256:local`" in issue for issue in issues)


def test_proof_import_hash_consistency_accepts_same_file_local_hash(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes/26/22.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/22#section_22_age_threshold
              output: section_22_age_threshold
              hash: sha256:local
    versions:
      - effective_from: '2026-01-01'
        formula: age >= section_22_age_threshold
"""
    rules_file.write_text(content)

    assert (
        find_proof_import_hash_consistency_issues(
            content,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_proof_import_hash_consistency_accepts_generated_same_target_local_hash(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    repo_file = repo / "statutes/26/22.yaml"
    generated_file = tmp_path / "generated/openai/statutes/26/22.yaml"
    repo_file.parent.mkdir(parents=True)
    generated_file.parent.mkdir(parents=True)
    repo_file.write_text("format: rulespec/v1\nrules: []\n")
    content = """format: rulespec/v1
rules:
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/22#section_22_age_threshold
              output: section_22_age_threshold
              hash: sha256:local
    versions:
      - effective_from: '2026-01-01'
        formula: age >= section_22_age_threshold
"""
    generated_file.write_text(content)

    assert (
        find_proof_import_hash_consistency_issues(
            content,
            rules_file=generated_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_proof_import_hash_consistency_accepts_external_file_hash(tmp_path):
    repo = tmp_path / "rulespec-us"
    current_file = repo / "statutes/26/22.yaml"
    target_file = repo / "statutes/26/1.yaml"
    current_file.parent.mkdir(parents=True)
    target_file.write_text("format: rulespec/v1\nrules: []\n")
    target_hash = target_file.read_bytes()
    import_hash = hashlib.sha256(target_hash).hexdigest()
    content = f"""format: rulespec/v1
rules:
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/1#section_1_tax
              output: section_1_tax
              hash: sha256:{import_hash}
    versions:
      - effective_from: '2026-01-01'
        formula: section_1_tax > 0
"""
    current_file.write_text(content)

    assert (
        find_proof_import_hash_consistency_issues(
            content,
            rules_file=current_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_proof_import_hash_consistency_rejects_external_file_hash_mismatch(tmp_path):
    repo = tmp_path / "rulespec-us"
    current_file = repo / "statutes/26/22.yaml"
    target_file = repo / "statutes/26/1.yaml"
    current_file.parent.mkdir(parents=True)
    target_file.write_text("format: rulespec/v1\nrules: []\n")
    content = """format: rulespec/v1
rules:
  - name: section_22_aged_individual
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/1#section_1_tax
              output: section_1_tax
              hash: sha256:abc123
    versions:
      - effective_from: '2026-01-01'
        formula: section_1_tax > 0
"""
    current_file.write_text(content)

    issues = find_proof_import_hash_consistency_issues(
        content,
        rules_file=current_file,
        policy_repo_path=repo,
    )

    assert any("Proof import hash mismatch" in issue for issue in issues)


def test_rulespec_grounding_accepts_source_leading_decimal():
    content = """format: rulespec/v1
rules:
  - name: annual_income_conversion_factor
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-10-01'
        formula: '0.083'
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="Annual income: multiply average by .083.",
        )
        == []
    )


def test_rulespec_grounding_accepts_cardinal_words_above_twelve():
    content = """format: rulespec/v1
rules:
  - name: minimum_hours
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2026-01-01'
        formula: 30
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="Employed a minimum of thirty hours per week.",
        )
        == []
    )


def test_rulespec_grounding_accepts_large_cardinal_word_amounts():
    content = """format: rulespec/v1
rules:
  - name: single_return_adjusted_gross_income_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2021-01-01'
        formula: '500000'
  - name: joint_return_adjusted_gross_income_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2021-01-01'
        formula: '1000000'
"""

    source_text = (
        "for a taxpayer who files a single return and whose adjusted gross income "
        "is greater than five hundred thousand dollars, and for taxpayers who file "
        "a joint return and whose adjusted gross income is greater than one million "
        "dollars"
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    source_values = extract_numbers_from_text(source_text)
    assert 500000 in source_values
    assert 1000000 in source_values
    occurrences = extract_numeric_occurrences_from_text(source_text)
    assert 500000 in occurrences
    assert 1000000 in occurrences


def test_rulespec_grounding_treats_household_size_match_keys_as_structural():
    content = """format: rulespec/v1
module:
  summary: The deduction amounts are 209, 223, 261, and 299.
rules:
  - name: standard_deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          match household_size:
              4 => 223
              5 => 261
              6 => 299
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="The deduction amounts are 209, 223, 261, and 299.",
        )
        == []
    )


def test_rulespec_grounding_treats_table_lookup_keys_as_structural():
    content = """format: rulespec/v1
module:
  summary: The tax rates are 10%, 12%, 22%, 24%, 32%, 35%, and 37%.
rules:
  - name: income_tax_bracket_rates
    kind: parameter
    dtype: Rate
    indexed_by: bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.10
          2: 0.12
          3: 0.22
          4: 0.24
          5: 0.32
          6: 0.35
          7: 0.37
  - name: income_tax_bracket_5_rate
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: income_tax_bracket_rates[5]
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="The tax rates are 10%, 12%, 22%, 24%, 32%, 35%, and 37%.",
        )
        == []
    )


def test_rulespec_grounding_treats_filing_status_codes_as_structural():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      additional_standard_deduction_married: 1650
rules:
  - name: additional_standard_deduction_married
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '1650'
  - name: additional_standard_deduction_per_condition_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 4:
              additional_standard_deduction_married
          else:
              0
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="The additional standard deduction amount is $1,650.",
        )
        == []
    )


def test_numeric_occurrence_extraction_ignores_nested_subsection_references():
    text = (
        "Notwithstanding any other provisions except subsections (b), (d)(2), "
        "(g), and (r) of section 2015 and section 2012(m)(4). "
        "The criteria are comparable to those under subsection (c)(2). "
        "A controlled substance is defined in section 802 of title 21."
    )

    assert extract_numeric_occurrences_from_text(text) == []


def test_numeric_occurrence_extraction_ignores_comma_conjoined_section_references():
    text = (
        "Adjusted gross income shall be determined without regard to "
        "sections 911, 931, and 933."
    )

    assert extract_numeric_occurrences_from_text(text) == []


def test_numeric_occurrence_extraction_ignores_parenthetical_subdivision_labels():
    text = (
        "(b) Fraud and misrepresentation; disqualification penalties. "
        "(1) Any person shall become ineligible "
        "(i) for a period of 1 year upon the first occasion."
    )

    assert extract_numeric_occurrences_from_text(text) == [1.0]


def test_numeric_occurrence_extraction_ignores_decimal_subsection_label():
    text = (
        "(1.5) Subject to subsection (2) of this section, a tax of four and "
        "three-quarters percent is imposed."
    )

    assert extract_numeric_occurrences_from_text(text) == [0.0475]


def test_broad_application_passthrough_rejects_furnishing_output():
    content = """format: rulespec/v1
module:
  summary: |-
    Assistance under this program shall be furnished to all eligible households who make application for such participation.
rules:
  - name: snap_assistance_furnished_to_applicant_household
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: |-
          household_is_eligible_to_participate_in_snap
          and household_makes_application_for_snap_participation
"""

    issues = find_broad_application_passthrough_issues(content)

    assert len(issues) == 1
    assert "Broad application pass-through" in issues[0]
    assert "snap_assistance_furnished_to_applicant_household" in issues[0]


def test_exception_test_coverage_requires_each_negated_exception_input():
    content = """format: rulespec/v1
module:
  summary: |-
    Notwithstanding section 1 and section 2, qualifying households shall be eligible.
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_qualifies
          and not section_1_exception_applies
          and not section_2_exception_applies
"""
    test_cases = [
        {
            "name": "section_1_positive_companion",
            "input": {
                "us:statutes/7/2014/a#input.household_qualifies": True,
                "us:statutes/7/2014/a#input.section_1_exception_applies": False,
                "us:statutes/7/2014/a#input.section_2_exception_applies": False,
            },
            "output": {
                "us:statutes/7/2014/a#eligibility": "holds",
            },
        },
        {
            "name": "section_1_blocks",
            "input": {
                "us:statutes/7/2014/a#input.household_qualifies": True,
                "us:statutes/7/2014/a#input.section_1_exception_applies": True,
                "us:statutes/7/2014/a#input.section_2_exception_applies": False,
            },
            "output": {
                "us:statutes/7/2014/a#eligibility": "not_holds",
            },
        },
    ]

    issues = find_exception_test_coverage_issues(content, test_cases)

    assert len(issues) == 1
    assert "section_2_exception_applies" in issues[0]
    assert "section_1_exception_applies" not in issues[0]


def test_exception_test_coverage_rejects_vacuous_blocking_test():
    content = """format: rulespec/v1
module:
  summary: |-
    Except section 1, qualifying households shall be eligible.
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_qualifies
          and not section_1_exception_applies
"""
    test_cases = [
        {
            "name": "positive_path",
            "input": {
                "us:statutes/7/2014/a#input.household_qualifies": True,
                "us:statutes/7/2014/a#input.section_1_exception_applies": False,
            },
            "output": {
                "us:statutes/7/2014/a#eligibility": "holds",
            },
        },
        {
            "name": "vacuous_exception_case",
            "input": {
                "us:statutes/7/2014/a#input.household_qualifies": False,
                "us:statutes/7/2014/a#input.section_1_exception_applies": True,
            },
            "output": {
                "us:statutes/7/2014/a#eligibility": "not_holds",
            },
        },
    ]

    issues = find_exception_test_coverage_issues(content, test_cases)

    assert len(issues) == 1
    assert "section_1_exception_applies" in issues[0]


def test_exception_test_coverage_ignores_defined_exception_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    A household is ineligible unless an exception applies.
rules:
  - name: exception_applies
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: exception_fact
  - name: ineligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_subject_to_rule
          and not exception_applies
"""
    test_cases = [
        {
            "name": "no_exception",
            "input": {
                "us:statutes/7/2015/e#input.exception_fact": False,
                "us:statutes/7/2015/e#input.household_subject_to_rule": True,
            },
            "output": {
                "us:statutes/7/2015/e#exception_applies": "not_holds",
                "us:statutes/7/2015/e#ineligible": "holds",
            },
        },
        {
            "name": "exception",
            "input": {
                "us:statutes/7/2015/e#input.exception_fact": True,
                "us:statutes/7/2015/e#input.household_subject_to_rule": True,
            },
            "output": {
                "us:statutes/7/2015/e#exception_applies": "holds",
                "us:statutes/7/2015/e#ineligible": "not_holds",
            },
        },
    ]

    assert find_exception_test_coverage_issues(content, test_cases) == []


def test_test_input_assignment_requires_all_local_formula_inputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
  summary: |-
    A household qualifies if it has income and no disqualifying condition.
rules:
  - name: household_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_has_income
          and not disqualifying_condition
"""
    test_cases = [
        {
            "name": "eligible",
            "input": {
                "us:statutes/7/2014/a#input.household_has_income": True,
            },
            "output": {
                "us:statutes/7/2014/a#household_eligible": "holds",
            },
        },
    ]

    issues = find_test_input_assignment_issues(content, test_cases)

    assert len(issues) == 1
    assert "disqualifying_condition" in issues[0]


def test_test_input_assignment_ignores_match_keyword():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: credit_rate
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match child_count:
              0 => 0.0765
              1 => 0.34
"""
    test_cases = [
        {
            "name": "one_child",
            "input": {
                "us:statutes/26/32#input.child_count": 1,
            },
            "output": {
                "us:statutes/26/32#credit_rate": 0.34,
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_test_input_assignment_treats_local_indexed_by_selector_as_dependency():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: income_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: if income < 100: 0 else: 4
  - name: rate_by_income_band
    kind: parameter
    dtype: Rate
    indexed_by: income_band
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.1
          4: 0.2
  - name: rate
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: rate_by_income_band[income_band]
"""
    test_cases = [
        {
            "name": "low_income",
            "input": {
                "us:statutes/example#input.income": 50,
            },
            "output": {
                "us:statutes/example#rate": 0.1,
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_test_input_assignment_counts_relation_child_inputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: taxpayer_or_spouse_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: taxpayer_or_spouse_of_tax_unit
      arity: 2
      arguments:
        - TaxUnit
        - Person
  - name: person_qualified
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: age >= 65 and not disqualifying_condition
  - name: qualified_person_count
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: count_where(taxpayer_or_spouse_of_tax_unit, person_qualified)
"""
    test_cases = [
        {
            "name": "tax_unit_with_aged_taxpayer",
            "input": {
                "us:statutes/26/22#relation.taxpayer_or_spouse_of_tax_unit": [
                    {
                        "us:statutes/26/22#input.age": 65,
                        "us:statutes/26/22#input.disqualifying_condition": False,
                    }
                ]
            },
            "output": {
                "us:statutes/26/22#qualified_person_count": 1,
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_test_input_assignment_counts_table_row_inputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: net_payment
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: payment_amount - excluded_amount
"""
    test_cases = [
        {
            "name": "multiple_payment_rows",
            "tables": {
                "Payment": [
                    {
                        "us:statutes/26/example#input.payment_amount": 100,
                        "us:statutes/26/example#input.excluded_amount": 40,
                    }
                ]
            },
            "output": {
                "us:statutes/26/example#net_payment": [60],
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_test_input_assignment_ignores_imported_rule_outputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
imports:
  - us:statutes/7/2012/j
rules:
  - name: household_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          imported_snap_household_has_elderly_or_disabled_member
          and household_has_income
"""
    test_cases = [
        {
            "name": "eligible",
            "input": {
                "us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member": "holds",
                "us:statutes/7/2014/a#input.household_has_income": True,
            },
            "output": {
                "us:statutes/7/2014/a#household_eligible": "holds",
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_test_input_assignment_ignores_imported_fragment_even_if_bad_placeholder_assigned():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
imports:
  - us:statutes/26/931#amount_excluded_from_gross_income_under_section_931
rules:
  - name: modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          adjusted_gross_income
          + amount_excluded_from_gross_income_under_section_931
"""
    test_cases = [
        {
            "name": "bad_placeholder_present",
            "input": {
                "us:statutes/26/151#input.adjusted_gross_income": 100000,
                "us:statutes/26/151#input.amount_excluded_from_gross_income_under_section_931": 0,
            },
            "output": {
                "us:statutes/26/151#modified_adjusted_gross_income": 100000,
            },
        },
        {
            "name": "no_import_placeholder",
            "input": {
                "us:statutes/26/151#input.adjusted_gross_income": 100000,
            },
            "output": {
                "us:statutes/26/151#modified_adjusted_gross_income": 100000,
            },
        },
    ]

    assert find_test_input_assignment_issues(content, test_cases) == []


def test_aggregate_exception_predicate_rejects_compressed_exception_list():
    content = """format: rulespec/v1
module:
  summary: |-
    Notwithstanding any other provisions except subsections (b), (d)(2), (g), and (r) of section 2015 and section 2012(m)(4), qualifying households shall be eligible.
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_qualifies
          and section_2015_b_d_2_g_r_and_2012_m_4_do_not_preclude_eligibility
"""

    issues = find_aggregate_exception_predicate_issues(content)

    assert len(issues) == 1
    assert "Aggregate exception predicate" in issues[0]
    assert (
        "section_2015_b_d_2_g_r_and_2012_m_4_do_not_preclude_eligibility" in issues[0]
    )


def test_unconsumed_local_exception_output_flags_matching_applies_rule():
    content = """format: rulespec/v1
rules:
  - name: readily_tradable_instrument_subsection_a_1_D_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          reportable_interest_or_dividend_payment
          and payment_on_readily_tradable_instrument
  - name: existing_account_exception_to_subsection_d_and_a_1_D
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: This subsection and subsection (a)(1)(D) shall not apply.
    versions:
      - effective_from: '2026-01-01'
        formula: account_established_before_transition_date
"""

    issues = find_unconsumed_local_exception_output_issues(content)

    assert len(issues) == 1
    assert "Unconsumed local exception output" in issues[0]
    assert "existing_account_exception_to_subsection_d_and_a_1_D" in issues[0]
    assert "readily_tradable_instrument_subsection_a_1_D_applies" in issues[0]


def test_unconsumed_local_exception_output_allows_negated_exception():
    content = """format: rulespec/v1
rules:
  - name: readily_tradable_instrument_subsection_a_1_D_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          reportable_interest_or_dividend_payment
          and payment_on_readily_tradable_instrument
          and not existing_account_exception_to_subsection_d_and_a_1_D
  - name: existing_account_exception_to_subsection_d_and_a_1_D
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: This subsection and subsection (a)(1)(D) shall not apply.
    versions:
      - effective_from: '2026-01-01'
        formula: account_established_before_transition_date
"""

    assert find_unconsumed_local_exception_output_issues(content) == []


def test_unconsumed_local_exception_output_flags_imported_exception_helper():
    content = """format: rulespec/v1
rules:
  - name: existing_account_or_instrument_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: This subsection and subsection (a)(1)(D) shall not apply.
    versions:
      - effective_from: '2026-01-01'
        formula: payment_paid_or_credited
  - name: subsection_a_1_D_applies_to_readily_tradable_instrument_payment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/3406/d#existing_account_or_instrument_exception_applies
              output: existing_account_or_instrument_exception_applies
              hash: sha256:local
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          reportable_interest_or_dividend_payment
          and payment_to_payee_on_readily_tradable_instrument
"""

    issues = find_unconsumed_local_exception_output_issues(content)

    assert len(issues) == 1
    assert "existing_account_or_instrument_exception_applies" in issues[0]
    assert (
        "subsection_a_1_D_applies_to_readily_tradable_instrument_payment" in issues[0]
    )


def test_unconsumed_local_exception_output_allows_imported_negated_helper():
    content = """format: rulespec/v1
rules:
  - name: existing_brokerage_account_exception_to_broker_notice_applies
    kind: derived
    entity: Asset
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: Subparagraph (B) of paragraph (2) shall not apply.
    versions:
      - effective_from: '2026-01-01'
        formula: broker_account_established_before_pre_cutoff_date
  - name: broker_must_notify_payor_of_backup_withholding_status
    kind: derived
    entity: Asset
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/3406/d#existing_brokerage_account_exception_to_broker_notice_applies
              output: existing_brokerage_account_exception_to_broker_notice_applies
              hash: sha256:local
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          payee_acquires_readily_tradable_instrument_through_broker
          and not existing_brokerage_account_exception_to_broker_notice_applies
"""

    assert find_unconsumed_local_exception_output_issues(content) == []


def test_anaphoric_scope_omission_rejects_broad_condition_predicate():
    source_text = (
        "Subparagraph (B) of paragraph (2) shall not apply with respect to a "
        "readily tradable instrument which was acquired through an account with "
        "a broker if- (A) such account was established before January 1, 1984, "
        "and (B) during 1983, such broker bought or sold instruments for the "
        "payee (or acted as a nominee for the payee) through such account."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3406
rules:
  - name: existing_brokerage_account_exception_to_broker_notification
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: "during 1983, such broker bought or sold instruments for the payee"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          readily_tradable_instrument_acquired_through_broker_account
          and broker_bought_or_sold_instruments_or_acted_as_nominee_for_payee_during_transition_year
"""

    issues = find_anaphoric_scope_omission_issues(
        content,
        source_texts={"us/statute/26/3406": source_text},
    )

    assert len(issues) == 1
    assert "Anaphoric scope omitted" in issues[0]
    assert "through such account" in issues[0]
    assert (
        "broker_bought_or_sold_instruments_or_acted_as_nominee_for_payee_during_transition_year"
        in issues[0]
    )


def test_anaphoric_scope_omission_accepts_same_account_predicate():
    source_text = (
        "Subparagraph (B) of paragraph (2) shall not apply with respect to a "
        "readily tradable instrument which was acquired through an account with "
        "a broker if- (A) such account was established before January 1, 1984, "
        "and (B) during 1983, such broker bought or sold instruments for the "
        "payee (or acted as a nominee for the payee) through such account."
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3406
rules:
  - name: existing_brokerage_account_exception_to_broker_notification
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              corpus_citation_path: us/statute/26/3406
              excerpt: "during 1983, such broker bought or sold instruments for the payee"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          broker_bought_or_sold_instruments_or_acted_as_nominee_for_payee_through_such_account_during_transition_year
"""

    assert (
        find_anaphoric_scope_omission_issues(
            content,
            source_texts={"us/statute/26/3406": source_text},
        )
        == []
    )


def test_parent_exception_list_requires_child_exception_imports(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "163" / "h" / "4" / "B.yaml"
    child_file = repo / "statutes" / "26" / "163" / "h" / "4" / "B" / "ii" / "I.yaml"
    child_file.parent.mkdir(parents=True)
    child_file.write_text(
        """format: rulespec/v1
module:
  summary: Such term shall not include a loan to finance fleet sales.
rules:
  - name: fleet_sales_loan_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
    versions:
      - effective_from: '2025-01-01'
        formula: loan_finances_fleet_sales
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term qualified passenger vehicle loan interest means interest paid on qualifying indebtedness.
    Such term shall not include any amount paid or incurred on any of the following:
rules:
  - name: interest_is_qualified_passenger_vehicle_loan_interest
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: interest_paid_on_qualifying_indebtedness
"""
    )

    issues = find_missing_child_exception_import_issues(
        rules_file.read_text(),
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Parent exception-list child import missing" in issues[0]
    assert (
        "us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies" in issues[0]
    )


def test_parent_exception_list_allows_child_exception_imports(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "163" / "h" / "4" / "B.yaml"
    child_file = repo / "statutes" / "26" / "163" / "h" / "4" / "B" / "ii" / "I.yaml"
    child_file.parent.mkdir(parents=True)
    child_file.write_text(
        """format: rulespec/v1
module:
  summary: Such term shall not include a loan to finance fleet sales.
rules:
  - name: fleet_sales_loan_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
    versions:
      - effective_from: '2025-01-01'
        formula: loan_finances_fleet_sales
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies
module:
  summary: |-
    The term qualified passenger vehicle loan interest means interest paid on qualifying indebtedness.
    Such term shall not include any amount paid or incurred on any of the following:
rules:
  - name: interest_is_qualified_passenger_vehicle_loan_interest
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          interest_paid_on_qualifying_indebtedness
          and not fleet_sales_loan_exception_applies
"""
    )

    assert (
        find_missing_child_exception_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_parent_exception_list_ignores_empty_wrapper_modules(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "163" / "h" / "4.yaml"
    child_file = repo / "statutes" / "26" / "163" / "h" / "4" / "B" / "ii" / "I.yaml"
    child_file.parent.mkdir(parents=True)
    child_file.write_text(
        """format: rulespec/v1
rules:
  - name: fleet_sales_loan_exception_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: loan_finances_fleet_sales
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  status: deferred
  summary: |-
    Special rules include the following exception list:
rules: []
"""
    )

    assert (
        find_missing_child_exception_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_parent_exception_list_does_not_treat_carveout_definition_as_exception(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "163" / "h" / "4.yaml"
    child_file = repo / "statutes" / "26" / "163" / "h" / "4" / "D.yaml"
    child_file.parent.mkdir(parents=True)
    child_file.write_text(
        """format: rulespec/v1
rules:
  - name: asset_is_applicable_passenger_vehicle
    kind: derived
    entity: Asset
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
    versions:
      - effective_from: '2025-01-01'
        formula: vehicle_final_assembly_occurred_within_united_states
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Special rules include the following exception list:
rules:
  - name: vehicle_interest_rule_applies
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: taxpayer_has_vehicle_interest
"""
    )

    assert (
        find_missing_child_exception_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_exception_test_coverage_accepts_imported_judgment_table_inputs():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies
module:
  summary: |-
    Such term shall not include any amount paid or incurred on any of the following:
rules:
  - name: interest_is_qualified_passenger_vehicle_loan_interest
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          interest_paid_on_qualifying_indebtedness
          and not fleet_sales_loan_exception_applies
"""
    test_cases = [
        {
            "name": "positive_path",
            "tables": {
                "Payment": [
                    {
                        "us:statutes/26/163/h/4/B#input.interest_paid_on_qualifying_indebtedness": True,
                        "us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies": "not_holds",
                    }
                ]
            },
            "output": {
                "us:statutes/26/163/h/4/B#interest_is_qualified_passenger_vehicle_loan_interest": [
                    "holds"
                ]
            },
        },
        {
            "name": "fleet_sales_exception",
            "tables": {
                "Payment": [
                    {
                        "us:statutes/26/163/h/4/B#input.interest_paid_on_qualifying_indebtedness": True,
                        "us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies": "holds",
                    }
                ]
            },
            "output": {
                "us:statutes/26/163/h/4/B#interest_is_qualified_passenger_vehicle_loan_interest": [
                    "not_holds"
                ]
            },
        },
    ]

    assert find_exception_test_coverage_issues(content, test_cases) == []


def test_exception_test_coverage_accepts_imported_exception_underlying_inputs():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/163/h/4/B/ii/I#fleet_sales_loan_exception_applies
module:
  summary: |-
    Such term shall not include any amount paid or incurred on any of the following:
rules:
  - name: interest_is_qualified_passenger_vehicle_loan_interest
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2025-01-01'
        formula: |-
          interest_paid_on_qualifying_indebtedness
          and not fleet_sales_loan_exception_applies
"""
    test_cases = [
        {
            "name": "positive_path",
            "tables": {
                "Payment": [
                    {
                        "us:statutes/26/163/h/4/B#input.interest_paid_on_qualifying_indebtedness": True,
                        "us:statutes/26/163/h/4/B/ii/I#input.amount_paid_or_incurred_on_loan": True,
                        "us:statutes/26/163/h/4/B/ii/I#input.loan_finances_fleet_sales": False,
                    }
                ]
            },
            "output": {
                "us:statutes/26/163/h/4/B#interest_is_qualified_passenger_vehicle_loan_interest": [
                    "holds"
                ]
            },
        },
        {
            "name": "fleet_sales_exception",
            "tables": {
                "Payment": [
                    {
                        "us:statutes/26/163/h/4/B#input.interest_paid_on_qualifying_indebtedness": True,
                        "us:statutes/26/163/h/4/B/ii/I#input.amount_paid_or_incurred_on_loan": True,
                        "us:statutes/26/163/h/4/B/ii/I#input.loan_finances_fleet_sales": True,
                    }
                ]
            },
            "output": {
                "us:statutes/26/163/h/4/B#interest_is_qualified_passenger_vehicle_loan_interest": [
                    "not_holds"
                ]
            },
        },
    ]

    assert find_exception_test_coverage_issues(content, test_cases) == []


def test_cross_reference_exception_placeholder_requires_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2014" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Notwithstanding section 2015(b), qualifying households shall be eligible.
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_qualifies
          and not section_2015_b_exception_applies
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert len(issues) == 1
    assert "Cross-reference placeholder" in issues[0]
    assert "statutes/7/2015/b" in issues[0]


def test_cross_reference_exception_placeholder_allows_category_label_boundary(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3121" / "b" / "7.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Paragraph (7) shall not apply in the case of service performed by any
    individual as an employee included under section 5351(2) of title 5,
    other than as a medical or dental intern or resident.
rules:
  - name: student_hospital_employee_exception_branch
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3121(b)(7)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: "by any individual as an employee included under section 5351(2) of title 5, United States Code (relating to certain interns, student nurses, and other student employees of hospitals)"
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: "other than as a medical or dental intern or as a medical or dental resident in training"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          individual_is_employee_included_under_section_5351_2_of_title_5_for_hospitals
          and not service_performed_as_medical_or_dental_intern
          and not service_performed_as_medical_or_dental_resident_in_training
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert issues == []


def test_cross_reference_placeholder_uses_explicit_usc_title(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "2" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    If the determination is made under section 556 of title 37 of the
    United States Code, the date is treated as a death date.
rules:
  - name: spouse_death_treated_within_preceding_years
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          section_556_death_determination_applies
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_encoded_cross_reference_placeholders(rules_file)

    assert len(issues) == 1
    assert "statutes/37/556" in issues[0]
    assert "statutes/26/556" not in issues[0]


def test_cross_reference_placeholder_does_not_infer_current_title_for_named_act(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "1402" / "b.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term self-employment income means net earnings derived by an individual
    other than a nonresident alien individual, except as provided by an
    agreement under section 233 of the Social Security Act.
rules:
  - name: self_employment_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if (
            nonresident_alien_individual
            and not agreement_under_section_233_of_social_security_act_applies
          ): 0 else: net_earnings_from_self_employment
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert issues == []


def test_cross_reference_placeholder_allows_current_section_helpers(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "22.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Section 22 provides a credit except as limited by the section 22 amount.
rules:
  - name: is_aged_65_or_over
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: age >= section_22_age_threshold
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_cross_reference_exception_placeholders(rules_file) == []


def test_cross_reference_placeholder_requires_same_section_subsection_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    A student enrolled in an institution of higher education is ineligible unless the student meets the requirements of subsection (e) of this section.
rules:
  - name: higher_education_student_exempt
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          person_enrolled_in_institution_of_higher_education
          and subsection_e_requirements_met
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert len(issues) == 1
    assert "Cross-reference" in issues[0]
    assert "statutes/7/2015/e" in issues[0]


def test_cross_reference_exception_placeholder_rejects_semantic_to_which_section(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "45A" / "d.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Paragraph (1) shall not apply to a transaction to which section 381(a)
    applies if the employee continues to be employed by the acquiring
    corporation.
rules:
  - name: early_termination_recapture_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          employment_terminated_by_taxpayer_before_one_year
          and not transaction_to_which_section_381_a_applies_with_employee_continuing
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert len(issues) == 1
    assert "Cross-reference placeholder" in issues[0]
    assert "transaction_to_which_section_381_a_applies" in issues[0]
    assert "statutes/26/381/a" in issues[0]
    assert "deferred" in issues[0]


def test_copied_cross_reference_source_rejects_cited_subsection_body(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    (2)(C) A bona fide student is exempt except that a higher education student is ineligible unless the student meets the requirements of subsection (e) of this section.
    (e) Students No individual enrolled at least half-time in an institution of higher education shall be eligible unless an exception applies.
rules:
  - name: copied_subsection_e_locally
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: subsection_e_exception_applies
"""
    )

    issues = find_copied_cross_reference_source_issues(
        rules_file.read_text(),
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Copied cross-reference source" in issues[0]
    assert "statutes/7/2015/e" in issues[0]


def test_copied_cross_reference_summary_repair_removes_cited_body(tmp_path):
    repo = tmp_path / "rulespec-us-co"
    rules_file = repo / "statutes" / "39" / "39-22-104" / "1.5.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
module:
  summary: |-
    (1.5) Subject to subsection (2) of this section, a tax of four and three-quarters percent is imposed.

    (2) Prior to the application of the rate of tax prescribed in subsection (1), (1.5), or (1.7) of this section, federal taxable income shall be modified.
rules:
  - name: individual_estate_trust_income_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1999-01-01'
        formula: '0.0475'
"""

    repaired, repairs = repair_copied_cross_reference_summary(
        content,
        rules_file=rules_file,
    )

    assert repairs == ["statutes/39/39-22-104/2"]
    summary = yaml.safe_load(repaired)["module"]["summary"]
    assert "four and three-quarters percent" in summary
    assert "Prior to the application" not in summary
    assert (
        find_copied_cross_reference_source_issues(
            repaired,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_copied_cross_reference_source_allows_bare_subsection_citation(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    A higher education student is ineligible unless the student meets the requirements of subsection (e) of this section.
rules:
  - name: local_rule
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_student
"""
    )

    assert (
        find_copied_cross_reference_source_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_same_section_subsection_reference_requires_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    cited_file = repo / "statutes" / "7" / "2015" / "e.yaml"
    cited_file.parent.mkdir(parents=True)
    cited_file.write_text("format: rulespec/v1\nrules: []\n")
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    A higher education student is ineligible unless the student meets the requirements of subsection (e) of this section.
rules:
  - name: higher_education_student_ineligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          person_is_higher_education_student
          and not person_meets_higher_education_student_eligibility_requirements
"""
    )

    issues = find_missing_same_section_subsection_import_issues(
        rules_file.read_text(),
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Same-section subsection import missing" in issues[0]
    assert "statutes/7/2015/e" in issues[0]


def test_same_section_under_subsection_reference_requires_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    cited_file = repo / "statutes" / "26" / "3121" / "y.yaml"
    cited_file.parent.mkdir(parents=True)
    cited_file.write_text("format: rulespec/v1\nrules: []\n")
    rules_file = repo / "statutes" / "26" / "3121" / "b" / "15.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Service performed in the employ of an international organization, except
    service which constitutes employment under subsection (y).
rules:
  - name: international_organization_service_excluded_from_employment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          service_performed_in_employ_of_international_organization
          and not service_constitutes_employment_under_subsection_y
"""
    )

    issues = find_missing_same_section_subsection_import_issues(
        rules_file.read_text(),
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Same-section subsection import missing" in issues[0]
    assert "statutes/26/3121/y" in issues[0]


def test_same_section_notwithstanding_override_does_not_require_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    cited_dir = repo / "statutes" / "26" / "3121" / "b"
    cited_dir.mkdir(parents=True)
    (cited_dir / "1.yaml").write_text("format: rulespec/v1\nrules: []\n")
    rules_file = repo / "statutes" / "26" / "3121" / "m.yaml"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term employment shall, notwithstanding the provisions of subsection (b)
    of this section, include service performed by an individual as a member of a
    uniformed service on active duty. Active duty means active duty as described
    in paragraph (21) of section 101 of title 38, except that it shall also
    include active duty for training as described in paragraph (22) of that
    section.
rules:
  - name: uniformed_service_included_in_employment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: service_performed_by_individual_as_member_of_uniformed_service_on_active_duty
"""
    )

    assert (
        find_missing_same_section_subsection_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_same_section_subsection_reference_allows_missing_source(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3121" / "i.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Service applies unless the provisions of subsection (m)(1) are unavailable.
rules:
  - name: service_applies_until_missing_subsection_boundary
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: service_applies_and_subsection_m_boundary_not_met
"""
    )

    assert (
        find_missing_same_section_subsection_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_same_section_subsection_reference_allows_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    A higher education student is ineligible unless the student meets the requirements of subsection (e) of this section.
imports:
  - us:statutes/7/2015/e
rules:
  - name: higher_education_student_ineligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          person_is_higher_education_student
          and not student_exception_to_higher_education_ineligibility_applies
"""
    )

    assert (
        find_missing_same_section_subsection_import_issues(
            rules_file.read_text(),
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_rule_name_path_suffix_rejects_citation_fragments(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("")
    content = """format: rulespec/v1
rules:
  - name: person_exempt_from_work_requirements_2_C
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
"""

    issues = find_rule_name_path_suffix_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Rule name includes citation suffix" in issues[0]
    assert "_2_c" in issues[0]


def test_rule_name_path_suffix_allows_semantic_numbers(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2015" / "b" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("")
    content = """format: rulespec/v1
rules:
  - name: first_occasion_ineligibility_period_years
    kind: parameter
    dtype: Count
"""

    assert (
        find_rule_name_path_suffix_issues(
            content,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_sibling_rule_name_collision_rejects_duplicate_exports(tmp_path):
    rules_file = (
        tmp_path / "rulespec-us" / "statutes" / "7" / "2015" / "d" / "2" / "A.yaml"
    )
    sibling = rules_file.with_name("B.yaml")
    rules_file.parent.mkdir(parents=True)
    sibling.write_text(
        """format: rulespec/v1
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
"""
    )
    content = """format: rulespec/v1
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
"""

    issues = find_sibling_rule_name_collision_issues(content, rules_file)

    assert len(issues) == 1
    assert "Sibling rule name collision" in issues[0]
    assert "B.yaml" in issues[0]


def test_sibling_rule_name_collision_allows_unique_exports(tmp_path):
    rules_file = (
        tmp_path / "rulespec-us" / "statutes" / "7" / "2015" / "d" / "2" / "A.yaml"
    )
    sibling = rules_file.with_name("B.yaml")
    rules_file.parent.mkdir(parents=True)
    sibling.write_text(
        """format: rulespec/v1
rules:
  - name: care_responsibility_exemption_applies
    kind: derived
"""
    )
    content = """format: rulespec/v1
rules:
  - name: title_iv_or_unemployment_work_registration_exemption_applies
    kind: derived
"""

    assert find_sibling_rule_name_collision_issues(content, rules_file) == []


def test_child_fragment_reencoding_rejects_parent_copying_child_inputs(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "63" / "c.yaml"
    child = repo / "statutes" / "26" / "63" / "c" / "5.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
imports:
  - us:policies/irs/rev-proc-2025-32/standard-deduction
rules:
  - name: dependent_standard_deduction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: earned_income + dependent_earned_income_addition
"""
    )
    content = """format: rulespec/v1
imports:
  - us:policies/irs/rev-proc-2025-32/standard-deduction
rules:
  - name: basic_standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if deduction_allowable_to_another_taxpayer_under_section_151:
              earned_income + dependent_earned_income_addition
          else:
              basic_standard_deduction_amount
"""

    issues = find_child_fragment_reencoding_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Child fragment re-encoded" in issues[0]
    assert "earned_income" in issues[0]
    assert "statutes/26/63/c/5" in issues[0]


def test_child_fragment_reencoding_allows_imported_child_output(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "63" / "c.yaml"
    child = repo / "statutes" / "26" / "63" / "c" / "5.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: dependent_standard_deduction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: earned_income
"""
    )
    content = """format: rulespec/v1
imports:
  - us:statutes/26/63/c/5
rules:
  - name: basic_standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if deduction_allowable_to_another_taxpayer_under_section_151:
              dependent_standard_deduction
          else:
              basic_standard_deduction_amount
"""

    assert (
        find_child_fragment_reencoding_issues(
            content,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_child_fragment_reencoding_allows_shared_input_with_terminal_child_import(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3121" / "a" / "5.yaml"
    child = repo / "statutes" / "26" / "3121" / "a" / "5" / "C.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: simplified_employee_pension_payment_branch_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: payment_made_under_simplified_employee_pension

  - name: simplified_employee_pension_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if simplified_employee_pension_payment_branch_applies: payment_amount else: 0
"""
    )
    content = """format: rulespec/v1
imports:
  - us:statutes/26/3121/a/5/C#simplified_employee_pension_payment_excluded_from_wages
rules:
  - name: executable_paragraph_5_branch_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          simplified_employee_pension_payment_excluded_from_wages
          + (if annuity_plan_403a_payment_exclusion_branch_applies: payment_amount else: 0)
"""

    assert (
        find_child_fragment_reencoding_issues(
            content,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_child_fragment_reencoding_points_to_terminal_child_output(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3101.yaml"
    child = repo / "statutes" / "26" / "3101" / "b" / "2.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: 0.009

  - name: additional_medicare_wage_tax_threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: 200000

  - name: additional_medicare_excess_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: max(0, wages - additional_medicare_wage_tax_threshold)

  - name: additional_medicare_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: additional_medicare_excess_wages * additional_medicare_tax_rate
"""
    )
    content = """format: rulespec/v1
imports:
  - us:statutes/26/3101/b/2#additional_medicare_tax_rate
rules:
  - name: section_3101_additional_medicare_component
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2013-01-01'
        formula: max(0, wages - additional_medicare_wage_tax_threshold) * additional_medicare_tax_rate
"""

    issues = find_child_fragment_reencoding_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Child fragment re-encoded" in issues[0]
    assert "us:statutes/26/3101/b/2#additional_medicare_tax" in issues[0]


def test_child_fragment_reencoding_partial_extent_guides_to_defer(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3101.yaml"
    child = repo / "statutes" / "26" / "3101" / "a.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1990-01-01'
        formula: 0.062

  - name: oasdi_wage_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: wages * oasdi_wage_tax_rate
"""
    )
    content = """format: rulespec/v1
module:
  summary: |-
    Wages shall be exempt from the taxes imposed by this section to the extent
    such wages are subject exclusively to another country's social security laws.
imports:
  - us:statutes/26/3101/a#oasdi_wage_tax_rate
rules:
  - name: section_3101_taxable_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, wages - wages_exempt_under_international_agreement)

  - name: section_3101_oasdi_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: section_3101_taxable_wages * oasdi_wage_tax_rate
"""

    issues = find_child_fragment_reencoding_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Child fragment re-encoded" in issues[0]
    assert "entity_not_supported" in issues[0]
    assert "deferred" in issues[0]


def test_partial_extent_exemption_rejects_all_or_nothing_zeroing():
    content = """format: rulespec/v1
module:
  summary: |-
    Wages shall be exempt from the taxes imposed by this section to the extent
    such wages are subject exclusively to another country's social security laws.
rules:
  - name: section_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if exempt_wages > 0: 0 else: gross_tax
"""

    issues = find_partial_extent_zeroing_issues(content)

    assert len(issues) == 1
    assert "Partial extent exemption collapsed" in issues[0]
    assert "section_tax" in issues[0]


def test_child_fragment_reencoding_rejects_parent_copying_child_numeric_output(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "24.yaml"
    child = repo / "statutes" / "26" / "24" / "h.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: ctc_joint_phase_out_threshold_under_subsection_h
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2018-01-01'
        formula: 400000

  - name: ctc_other_phase_out_threshold_under_subsection_h
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2018-01-01'
        formula: 200000
"""
    )
    content = """format: rulespec/v1
imports:
  - us:statutes/26/24/h#ctc_maximum_before_phase_out_under_subsection_h
rules:
  - name: ctc_phaseout_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    source: 26 USC 24(b)(2), 24(h)(3)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxable_year_begins_after_2017:
              if filing_status == 1 or filing_status == 4:
                  400000
              else:
                  200000
          else:
              ctc_joint_threshold_before_subsection_h
"""

    issues = find_child_fragment_reencoding_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert len(issues) == 1
    assert "Child fragment numeric output re-encoded" in issues[0]
    assert "400000" in issues[0]
    assert "200000" in issues[0]
    assert "statutes/26/24/h.yaml" in issues[0]
    assert "ctc_joint_phase_out_threshold_under_subsection_h" in issues[0]


def test_child_fragment_reencoding_allows_parent_using_imported_child_numeric_output(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "24.yaml"
    child = repo / "statutes" / "26" / "24" / "h.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        """format: rulespec/v1
rules:
  - name: ctc_joint_phase_out_threshold_under_subsection_h
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2018-01-01'
        formula: 400000
"""
    )
    content = """format: rulespec/v1
imports:
  - us:statutes/26/24/h#ctc_joint_phase_out_threshold_under_subsection_h
rules:
  - name: ctc_phaseout_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    source: 26 USC 24(b)(2), 24(h)(3)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxable_year_begins_after_2017:
              ctc_joint_phase_out_threshold_under_subsection_h
          else:
              ctc_joint_threshold_before_subsection_h
"""

    assert (
        find_child_fragment_reencoding_issues(
            content,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_cross_reference_exception_placeholder_allows_covering_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2014" / "a.yaml"
    imported_file = repo / "statutes" / "7" / "2015" / "b.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: section_2015_b_exception_applies
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: household_is_subject_to_section_2015_b_exception
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Notwithstanding section 2015(b), qualifying households shall be eligible.
imports:
  - us:statutes/7/2015/b
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_qualifies
          and not section_2015_b_exception_applies
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_cross_reference_exception_placeholders(rules_file) == []


def test_cross_reference_placeholder_rejects_covering_import_without_export(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "63.yaml"
    imported_file = repo / "statutes" / "26" / "163" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: interest_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: interest_paid_or_accrued_on_indebtedness
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/163/a#interest_deduction
module:
  summary: |-
    Except as provided in subsection (b), taxable income subtracts so much of
    the deduction allowed by section 163(a) as is attributable to the exception
    under section 163(h)(4)(A).
rules:
  - name: subsection_b_deductions_for_nonitemizer
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          standard_deduction
          + section_163_a_deduction_attributable_to_section_163_h_4_A_exception
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_exception_placeholders(rules_file)

    assert len(issues) == 1
    assert "Cross-reference placeholder" in issues[0]
    assert "interest_deduction" not in issues[0]
    assert "section_163_a_deduction_attributable" in issues[0]


def test_cross_reference_placeholder_allows_deeper_import_for_semantic_tail(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "1411.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/911/d/6#section_911_disallowed_deductions_and_exclusions
module:
  summary: |-
    Modified adjusted gross income is adjusted gross income increased by the
    excess of the amount excluded from gross income under section 911(a)(1)
    over deductions or exclusions disallowed under section 911(d)(6).
rules:
  - name: niit_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          adjusted_gross_income
          + gross_income_excluded_under_section_911_a_1
          - section_911_disallowed_deductions_and_exclusions
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_cross_reference_exception_placeholders(rules_file) == []


def test_cross_reference_placeholder_allows_relation_field_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "151.yaml"
    imported_file = repo / "statutes" / "26" / "911" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: section_911_amount_excluded_from_gross_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: foreign_earned_income_exclusion
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/911/a#section_911_amount_excluded_from_gross_income
module:
  summary: |-
    Modified adjusted gross income means adjusted gross income increased by any
    amount excluded from gross income under section 911, except as otherwise
    provided.
rules:
  - name: section_911_excluded_individual_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: section_911_excluded_individual_of_tax_unit
      arity: 2
      arguments:
        - TaxUnit
        - Person
  - name: senior_deduction_modified_adjusted_gross_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          adjusted_gross_income
          + sum(section_911_excluded_individual_of_tax_unit.section_911_amount_excluded_from_gross_income)
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_cross_reference_exception_placeholders(rules_file) == []
    assert pipeline._check_encoded_cross_reference_placeholders(rules_file) == []


def test_encoded_cross_reference_placeholder_rejects_under_section_input_when_source_exists(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "1222.yaml"
    imported_file = repo / "statutes" / "26" / "1211.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: other_taxpayer_capital_losses_allowed
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: allowed_capital_losses
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term net capital loss means the excess of the losses from sales or
    exchanges of capital assets over the sum allowed under section 1211.
rules:
  - name: net_capital_loss
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, losses_from_sales_or_exchanges_of_capital_assets - sum_allowed_under_section_1211)
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_encoded_cross_reference_placeholders(rules_file)

    assert len(issues) == 1
    assert "Encoded cross-reference placeholder" in issues[0]
    assert "sum_allowed_under_section_1211" in issues[0]
    assert "statutes/26/1211" in issues[0]


def test_encoded_cross_reference_placeholder_rejects_provided_in_section_input_when_source_exists(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "63.yaml"
    imported_file = repo / "statutes" / "26" / "170" / "p.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: qualified_charitable_contribution_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: qualified_charitable_contributions
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Taxable income means adjusted gross income minus any deduction provided
    in section 170(p).
rules:
  - name: deductions_referred_to_in_subsection_b
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          standard_deduction
          + deduction_provided_in_section_170_p
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_encoded_cross_reference_placeholders(rules_file)

    assert len(issues) == 1
    assert "Encoded cross-reference placeholder" in issues[0]
    assert "deduction_provided_in_section_170_p" in issues[0]
    assert "statutes/26/170/p" in issues[0]


def test_encoded_cross_reference_placeholder_rejects_in_effect_under_section_input_when_source_exists(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "151.yaml"
    imported_file = repo / "statutes" / "26" / "68" / "b.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: applicable_amount
    kind: parameter
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 100000
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The exemption amount is reduced when adjusted gross income exceeds the
    applicable amount in effect under section 68(b).
rules:
  - name: exemption_phaseout_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, adjusted_gross_income - applicable_amount_in_effect_under_section_68_b)
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_encoded_cross_reference_placeholders(rules_file)

    assert len(issues) == 1
    assert "Encoded cross-reference placeholder" in issues[0]
    assert "applicable_amount_in_effect_under_section_68_b" in issues[0]
    assert "statutes/26/68/b" in issues[0]


def test_cross_reference_numeric_placeholder_rejects_locally_supplied_rates(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3201.yaml"
    imported_file = repo / "statutes" / "26" / "3101.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: oasdi_wage_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: 0.062
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    (a) Tier 1 tax In addition to other taxes, there is hereby imposed on
    the income of each employee a tax equal to the applicable percentage of
    the compensation received during any calendar year by such employee. For
    purposes of the preceding sentence, the term "applicable percentage" means
    the percentage equal to the sum of the rates of tax in effect under
    subsections (a) and (b) of section 3101 for the calendar year.

    (b) Tier 2 tax In addition to other taxes, there is hereby imposed on the
    income of each employee a tax equal to the percentage determined under
    section 3241 for any calendar year of the compensation received during
    such calendar year by such employee.
rules:
  - name: tier_1_employee_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    source: 26 USC 3201(a)
    versions:
      - effective_from: '2026-01-01'
        formula: tier_1_applicable_percentage * compensation
  - name: tier_2_employee_tax
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    source: 26 USC 3201(b)
    versions:
      - effective_from: '2026-01-01'
        formula: tier_2_applicable_percentage * compensation
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_numeric_placeholders(rules_file)

    assert len(issues) == 2
    assert all("Cross-reference numeric placeholder" in issue for issue in issues)
    assert any("tier_1_applicable_percentage" in issue for issue in issues)
    assert any("statutes/26/3101" in issue for issue in issues)
    assert any("tier_2_applicable_percentage" in issue for issue in issues)
    assert any("statutes/26/3241" in issue for issue in issues)


def test_flattened_thresholded_imported_rate_is_rejected(tmp_path):
    repo = tmp_path / "rulespec-us"
    imported_file = repo / "statutes" / "26" / "3101" / "b" / "2.yaml"
    rules_file = repo / "statutes" / "26" / "3201.yaml"
    imported_file.parent.mkdir(parents=True)
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: 0.009
  - name: additional_medicare_wage_tax_threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: 200000
  - name: additional_medicare_excess_wages
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: max(0, wages - additional_medicare_wage_tax_threshold)
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3101/b/2#additional_medicare_tax_rate
rules:
  - name: tier_1_applicable_percentage
    kind: derived
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: base_rate + additional_medicare_tax_rate
  - name: tier_1_employee_tax
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: compensation * tier_1_applicable_percentage
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_flattened_thresholded_imported_rates(rules_file)

    assert len(issues) == 1
    assert "Flattened thresholded imported rate" in issues[0]
    assert "additional_medicare_tax_rate" in issues[0]


def test_thresholded_imported_rate_allows_excess_amount_formula(tmp_path):
    repo = tmp_path / "rulespec-us"
    imported_file = repo / "statutes" / "26" / "3101" / "b" / "2.yaml"
    rules_file = repo / "statutes" / "26" / "3201.yaml"
    imported_file.parent.mkdir(parents=True)
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: additional_medicare_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2013-01-01'
        formula: 0.009
  - name: additional_medicare_wage_tax_threshold
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: 200000
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3101/b/2#additional_medicare_tax_rate
rules:
  - name: additional_medicare_component_tax
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2013-01-01'
        formula: additional_medicare_excess_wages * additional_medicare_tax_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_flattened_thresholded_imported_rates(rules_file)

    assert issues == []


def test_cross_reference_base_mechanics_raw_compensation_tax_is_rejected(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3201.yaml"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    (a) Tier 1 tax In addition to other taxes, there is hereby imposed on the income
    of each employee a tax equal to the applicable percentage of the compensation
    received during any calendar year by such employee for services rendered.

    (b) Tier 2 tax In addition to other taxes, there is hereby imposed on the income
    of each employee a tax equal to the percentage determined under section 3241
    for any calendar year of the compensation received during such calendar year.

    (c) Cross reference For application of different contribution bases with respect
    to the taxes imposed by subsections (a) and (b), see section 3231(e)(2).
rules:
  - name: tier_2_employee_tax
    kind: derived
    entity: Person
    dtype: Money
    source: 26 USC 3201(b)
    versions:
      - effective_from: '2026-01-01'
        formula: compensation_received_for_services * tier_2_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unapplied_cross_reference_base_mechanics(rules_file)

    assert len(issues) == 1
    assert "Cross-reference base mechanics omitted" in issues[0]
    assert "tier_2_employee_tax" in issues[0]
    assert "section 3231(e)(2)" in issues[0]


def test_cross_reference_base_mechanics_allows_cited_base_formula(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3201.yaml"
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3231/e/2#remaining_applicable_base_before_payment
module:
  summary: |-
    (a) Tier 1 tax In addition to other taxes, there is hereby imposed on the income
    of each employee a tax equal to the applicable percentage of the compensation
    received during any calendar year by such employee for services rendered.

    (c) Cross reference For application of different contribution bases with respect
    to the taxes imposed by subsection (a), see section 3231(e)(2).
rules:
  - name: tier_1_employee_tax
    kind: derived
    entity: Person
    dtype: Money
    source: 26 USC 3201(a)
    versions:
      - effective_from: '2026-01-01'
        formula: min(compensation_received_for_services, remaining_applicable_base_before_payment) * tier_1_rate
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unapplied_cross_reference_base_mechanics(rules_file)

    assert issues == []


def test_cross_reference_numeric_placeholder_does_not_infer_named_act_title(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3121" / "b" / "7.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The exception applies for an election worker if remuneration is less than
    the adjusted amount determined under section 218(c)(8)(B) of the Social
    Security Act for any calendar year commencing on or after January 1, 2000.
rules:
  - name: election_worker_low_remuneration_exception
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3121(b)(7)(F)(iv)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          service_performed_by_election_worker
          and remuneration < adjusted_amount_determined_under_social_security_act_section_218_c_8_B
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_numeric_placeholders(rules_file)

    assert issues == []


def test_cross_reference_numeric_placeholder_accepts_top_level_import_sequence(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "3211.yaml"
    imported_file = repo / "statutes" / "26" / "3241" / "b.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
- name: section_3211_and_3221_applicable_percentage_for_tax_unit
  kind: derived
  entity: TaxUnit
  dtype: Rate
  period: Year
  versions:
  - effective_from: '2026-01-01'
    formula: 0.181
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
- us:statutes/26/3241/b#section_3211_and_3221_applicable_percentage_for_tax_unit
module:
  summary: |-
    (b) Tier 2 tax In addition to other taxes, there is hereby imposed on the
    income of each employee representative a tax equal to the percentage
    determined under section 3241 for any calendar year of the compensation
    received during such calendar year by such employee representative.
rules:
- name: employee_representative_tier_2_tax
  kind: derived
  entity: TaxUnit
  dtype: Money
  period: Year
  source: 26 USC 3211(b)
  versions:
  - effective_from: '2026-01-01'
    formula: compensation * section_3211_and_3221_applicable_percentage_for_tax_unit
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_reference_numeric_placeholders(rules_file)

    assert issues == []


def test_encoded_cross_reference_placeholder_allows_under_section_when_unencoded(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "1222.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term net capital loss means the excess of the losses from sales or
    exchanges of capital assets over the sum allowed under section 1211.
rules:
  - name: net_capital_loss
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, losses_from_sales_or_exchanges_of_capital_assets - sum_allowed_under_section_1211)
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_encoded_cross_reference_placeholders(rules_file) == []


def test_encoded_cross_reference_placeholder_rejects_missing_definition_dependency(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "45A" / "e.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term wages has the same meaning given to such term in section 51.
rules:
  - name: wages_definition_proxy
    kind: derived
    entity: Business
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: wages_have_same_meaning_under_section_51
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_encoded_cross_reference_placeholders(rules_file)

    assert len(issues) == 1
    assert "Encoded cross-reference placeholder" in issues[0]
    assert "wages_have_same_meaning_under_section_51" in issues[0]
    assert "statutes/26/51" in issues[0]
    assert "deferred" in issues[0]


def test_encoded_cross_reference_placeholder_allows_covering_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "1222.yaml"
    imported_file = repo / "statutes" / "26" / "1211.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: other_taxpayer_capital_losses_allowed
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: allowed_capital_losses
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The term net capital loss means the excess of the losses from sales or
    exchanges of capital assets over the sum allowed under section 1211.
imports:
  - us:statutes/26/1211#other_taxpayer_capital_losses_allowed
rules:
  - name: net_capital_loss
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, losses_from_sales_or_exchanges_of_capital_assets - other_taxpayer_capital_losses_allowed)
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_encoded_cross_reference_placeholders(rules_file) == []


def test_encoded_cross_reference_placeholder_allows_local_helper_using_import(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "26" / "32" / "c" / "2.yaml"
    imported_file = repo / "statutes" / "26" / "112.yaml"
    rules_file.parent.mkdir(parents=True)
    imported_file.parent.mkdir(parents=True, exist_ok=True)
    imported_file.write_text(
        """format: rulespec/v1
rules:
  - name: amount_excluded_from_gross_income_by_reason_of_section_112
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: combat_zone_compensation
"""
    )
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/112#amount_excluded_from_gross_income_by_reason_of_section_112
module:
  summary: |-
    A taxpayer may elect to treat amounts excluded from gross income by reason
    of section 112 as earned income.
rules:
  - name: section_112_excluded_amounts_treated_as_earned_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxpayer_elects_to_treat_section_112_excluded_amounts_as_earned_income:
            amount_excluded_from_gross_income_by_reason_of_section_112
          else: 0
  - name: earned_income
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: wages + section_112_excluded_amounts_treated_as_earned_income
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_encoded_cross_reference_placeholders(rules_file) == []


def test_validate_rulespec_proofs_can_require_policy_proofs_without_module_flag():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
rules:
  - name: eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: household_qualifies
"""

    result = validate_rulespec_proofs(content, require_policy_proofs=True)

    assert result.proof_required is True
    assert any("Proof missing" in issue for issue in result.issues)


def _write_rulespec_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    validator_pipeline._rulespec_executable_index_for_roots.cache_clear()
    return path


def test_upstream_placement_flags_duplicate_upstream_executable_rule(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rulespec-us" / "policies/example/fy-2026.yaml",
        """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us-co" / "regulations/example/benefit.yaml",
        """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )

    issues = find_upstream_placement_issues(
        rules_file.read_text(encoding="utf-8"),
        rules_file=rules_file,
    )

    assert len(issues) == 1
    assert "duplicates existing RuleSpec target" in issues[0]
    assert "us:policies/example/fy-2026#benefit_limit" in issues[0]


def test_upstream_placement_allows_subsection_extraction_from_ancestor(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rulespec-us" / "statutes/26/151.yaml",
        """format: rulespec/v1
rules:
  - name: exemption_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us" / "statutes/26/151/d.yaml",
        """format: rulespec/v1
rules:
  - name: exemption_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )

    issues = find_upstream_placement_issues(
        rules_file.read_text(encoding="utf-8"),
        rules_file=rules_file,
    )

    assert issues == []


def test_upstream_placement_rejects_ancestor_after_subsection_extraction(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rulespec-us" / "statutes/26/151/d.yaml",
        """format: rulespec/v1
rules:
  - name: exemption_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us" / "statutes/26/151.yaml",
        """format: rulespec/v1
rules:
  - name: exemption_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )

    issues = find_upstream_placement_issues(
        rules_file.read_text(encoding="utf-8"),
        rules_file=rules_file,
    )

    assert len(issues) == 1
    assert "duplicates existing RuleSpec target" in issues[0]
    assert "us:statutes/26/151/d#exemption_amount" in issues[0]


def test_upstream_placement_allows_distinct_local_rule_with_same_name(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rulespec-us" / "policies/example/fy-2026.yaml",
        """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
""",
    )
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us-co" / "regulations/example/benefit.yaml",
        """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '525'
""",
    )

    assert (
        find_upstream_placement_issues(
            rules_file.read_text(encoding="utf-8"),
            rules_file=rules_file,
        )
        == []
    )


def test_upstream_placement_ignores_nested_axiom_dependency_checkout(tmp_path):
    repo_parent = tmp_path / "repos"
    canonical_content = """format: rulespec/v1
rules:
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
"""
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us" / "policies/example/fy-2026.yaml",
        canonical_content,
    )
    _write_rulespec_file(
        repo_parent
        / "rulespec-us"
        / "_axiom"
        / "rulespec-us"
        / "policies/example/fy-2026.yaml",
        canonical_content,
    )

    assert (
        find_upstream_placement_issues(
            rules_file.read_text(encoding="utf-8"),
            rules_file=rules_file,
        )
        == []
    )


def test_upstream_placement_ignores_sibling_jurisdiction_duplicates(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rulespec-us-ny" / "regulations/example/benefit.yaml",
        """format: rulespec/v1
rules:
  - name: state_allowance
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
""",
    )
    rules_file = _write_rulespec_file(
        repo_parent / "rulespec-us-co" / "regulations/example/benefit.yaml",
        """format: rulespec/v1
rules:
  - name: state_allowance
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
""",
    )

    assert (
        find_upstream_placement_issues(
            rules_file.read_text(encoding="utf-8"),
            rules_file=rules_file,
        )
        == []
    )


def test_upstream_placement_rejects_executable_copy_of_restated_target():
    content = """format: rulespec/v1
rules:
  - name: benefit_limit_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/example/fy-2026#benefit_limit
  - name: benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "Restated upstream target copied as executable RuleSpec" in issues[0]
    assert "us:policies/example/fy-2026#benefit_limit" in issues[0]


def test_upstream_placement_rejects_executable_copy_of_verified_restatement_value():
    content = """format: rulespec/v1
rules:
  - name: benefit_table_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/example/fy-2026#benefit_amount
    verification:
      values:
        benefit_amount_table:
          1: 500
          2: 750
  - name: benefit_amount_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 500
          2: 750
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "benefit_amount_table" in issues[0]
    assert "us:policies/example/fy-2026#benefit_amount" in issues[0]


def test_upstream_placement_allows_pure_source_relation_restatement():
    content = """format: rulespec/v1
rules:
  - name: benefit_limit_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/example/fy-2026#benefit_limit
      authority: federal
    verification:
      values:
        benefit_limit: 500
"""

    assert find_upstream_placement_issues(content) == []


def test_upstream_placement_requires_source_relation_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: local_benefit_limit
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '500'
"""

    issues = find_upstream_placement_issues(
        content,
        source_metadata={
            "relations": [
                {
                    "relation": "restates",
                    "target": "us:policies/example/fy-2026#benefit_limit",
                }
            ]
        },
    )

    assert len(issues) == 1
    assert "Source metadata upstream relation requires source_relation" in issues[0]
    assert "us:policies/example/fy-2026#benefit_limit" in issues[0]


def test_upstream_placement_allows_source_relation_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: benefit_limit_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/example/fy-2026#benefit_limit
      authority: federal
"""

    assert (
        find_upstream_placement_issues(
            content,
            source_metadata={
                "relations": [
                    {
                        "relation": "restates",
                        "target": "us:policies/example/fy-2026#benefit_limit",
                    }
                ]
            },
        )
        == []
    )


def test_upstream_placement_requires_metadata_sets_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: local_standard_allowance
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
"""

    issues = find_upstream_placement_issues(
        content,
        source_metadata={
            "relations": [
                {
                    "relation": "sets",
                    "target": "us:regulation/7-cfr/273/9/d/6/iii#standard_allowance",
                }
            ]
        },
    )

    assert len(issues) == 1
    assert "Source metadata upstream relation not recorded" in issues[0]
    assert "source_relation.type: sets" in issues[0]
    assert "us:regulation/7-cfr/273/9/d/6/iii#standard_allowance" in issues[0]


def test_upstream_placement_allows_source_relation_sets_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: standard_allowance_setting
    kind: source_relation
    source_relation:
      type: sets
      target: us:regulation/7-cfr/273/9/d/6/iii#standard_allowance
"""

    assert (
        find_upstream_placement_issues(
            content,
            source_metadata={
                "relations": [
                    {
                        "relation": "sets",
                        "target": "us:regulation/7-cfr/273/9/d/6/iii#standard_allowance",
                    }
                ]
            },
        )
        == []
    )


def test_upstream_placement_requires_metadata_amends_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: updated_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '100'
"""

    issues = find_upstream_placement_issues(
        content,
        source_metadata={
            "relations": [
                {
                    "relation": "amends",
                    "target": "us:statutes/7/2014/c#income_threshold",
                }
            ]
        },
    )

    assert len(issues) == 1
    assert "source_relation.type: amends" in issues[0]
    assert "us:statutes/7/2014/c#income_threshold" in issues[0]


def test_upstream_placement_rejects_source_relation_metadata_on_executable_rule():
    content = """format: rulespec/v1
rules:
  - name: local_standard_allowance
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      sets: us:regulation/7-cfr/273/9/d/6/iii#standard_allowance
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "source-relation metadata is not allowed" in issues[0]
    assert "metadata.sets" in issues[0]


def test_upstream_placement_rejects_metadata_source_relation_on_executable_rule():
    content = """format: rulespec/v1
rules:
  - name: local_copy
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      source_relation: copies
      sets: us:regulation/example#amount
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "source-relation metadata is not allowed" in issues[0]
    assert "metadata.source_relation" in issues[0]


def test_upstream_placement_rejects_metadata_source_relation_without_target():
    content = """format: rulespec/v1
rules:
  - name: local_update
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      source_relation: amends
    versions:
      - effective_from: '2026-01-01'
        formula: '100'
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "source-relation metadata is not allowed" in issues[0]
    assert "metadata.source_relation" in issues[0]


def test_upstream_placement_requires_metadata_implements_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: state_mechanics
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: household_income
"""

    issues = find_upstream_placement_issues(
        content,
        source_metadata={
            "relations": [
                {
                    "relation": "implements",
                    "target": "us:statutes/7/2014/e#deduction_mechanics",
                }
            ]
        },
    )

    assert len(issues) == 1
    assert "source_relation.type: implements" in issues[0]
    assert "us:statutes/7/2014/e#deduction_mechanics" in issues[0]


def test_upstream_placement_allows_source_relation_implements_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: deduction_mechanics_implementation
    kind: source_relation
    source_relation:
      type: implements
      target: us:statutes/7/2014/e#deduction_mechanics
"""

    assert (
        find_upstream_placement_issues(
            content,
            source_metadata={
                "relations": [
                    {
                        "relation": "implements",
                        "target": "us:statutes/7/2014/e#deduction_mechanics",
                    }
                ]
            },
        )
        == []
    )


def test_upstream_placement_rejects_metadata_defines_relation():
    content = """format: rulespec/v1
rules:
  - name: canonical_income_rule
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    metadata:
      source_relation: defines
    versions:
      - effective_from: '2026-01-01'
        formula: household_income
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "source-relation metadata is not allowed" in issues[0]
    assert "metadata.source_relation" in issues[0]


def test_upstream_placement_rejects_concept_id_placeholder():
    content = """format: rulespec/v1
rules:
  - name: canonical_income_rule
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    metadata:
      concept_id: snap.income
    versions:
      - effective_from: '2026-01-01'
        formula: household_income
"""

    issues = find_upstream_placement_issues(content)

    assert len(issues) == 1
    assert "metadata.concept_id" in issues[0]
    assert "absolute RuleSpec or corpus target" in issues[0]


def test_extract_json_object_accepts_literal_newline_in_reviewer_string():
    output = """{
  "score": 9.0,
  "passed": true,
  "blocking_issues": [],
  "non_blocking_issues": [
    "self_employment_income is treated as an external input
rather than imported from a canonical definition"
  ],
  "reasoning": "safe to promote"
}"""

    data = _extract_json_object(output)

    assert data["score"] == 9.0
    assert data["passed"] is True
    assert "external input\nrather than" in data["non_blocking_issues"][0]


def test_extract_json_object_prefers_reviewer_payload_over_cli_metadata():
    output = """{"type":"thread.started"}
{"score":8.5,"passed":true,"issues":[],"reasoning":"ok"}"""

    data = _extract_json_object(output)

    assert data == {
        "score": 8.5,
        "passed": True,
        "issues": [],
        "reasoning": "ok",
    }


def test_extract_json_object_repairs_trailing_commas():
    output = """```json
{
  "score": 8,
  "passed": true,
  "issues": [],
}
```"""

    data = _extract_json_object(output)

    assert data["score"] == 8
    assert data["passed"] is True


def test_extract_json_object_accepts_fullwidth_space_from_reviewer_output():
    output = """```json
{
  "score": 7,
  "passed": true,
  "issues": [
    "first issue",
　"Inconsistent decomposition"
  ],
  "reasoning": "ok"
}
```"""

    data = _extract_json_object(output)

    assert data["passed"] is True
    assert data["issues"] == ["first issue", "Inconsistent decomposition"]


def test_extract_json_object_repairs_missing_terminal_object_brace():
    output = """{
  "score": 8.5,
  "passed": true,
  "blocking_issues": [],
  "non_blocking_issues": [
    "self_employment_income should eventually import IRC 1402"
  ],
  "reasoning": "Suitable for promotion."
"""

    data = _extract_json_object(output)

    assert data["score"] == 8.5
    assert data["passed"] is True
    assert data["non_blocking_issues"] == [
        "self_employment_income should eventually import IRC 1402"
    ]


def test_rulespec_ci_executes_companion_test_outputs(tmp_path, monkeypatch):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    citation_path = "us/guidance/example/sua"
    _write_local_corpus_provision(
        tmp_path,
        citation_path,
        body="The standard utility allowance is $451.",
    )
    monkeypatch.setenv(
        "AXIOM_CORPUS_ARTIFACT_ROOT",
        str(tmp_path / "axiom-corpus" / "data" / "corpus"),
    )
    validator_pipeline._fetch_corpus_source_text.cache_clear()
    validator_pipeline._fetch_local_corpus_source_text.cache_clear()

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: '451'
  - name: snap_standard_utility_allowance
    kind: derived
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: snap_standard_utility_allowance_value
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: catches_wrong_expected_value
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance: 452
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "snap_standard_utility_allowance" in issue
        and "expected integer 452, got integer 451" in issue
        for issue in result.issues
    )


def test_rulespec_output_lookup_rejects_friendly_name_aliases(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    runtime_output = {
        "us:statutes/7/2017/a#snap_regular_month_allotment": {
            "kind": "scalar",
            "name": "snap_regular_month_allotment",
            "id": "us:statutes/7/2017/a#snap_regular_month_allotment",
            "value": {"kind": "decimal", "value": "268"},
        }
    }

    outputs = pipeline._rulespec_outputs_by_reference(runtime_output)

    assert "snap_regular_month_allotment" not in outputs
    assert (
        outputs["us:statutes/7/2017/a#snap_regular_month_allotment"]
        is runtime_output["us:statutes/7/2017/a#snap_regular_month_allotment"]
    )


def test_rulespec_ci_rejects_mixed_derived_output_entities(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )
    compiled_payload = {
        "program": {
            "derived": [
                {
                    "name": "is_aged_65_or_over",
                    "id": "us:statutes/26/22#is_aged_65_or_over",
                    "entity": "Person",
                },
                {
                    "name": "elderly_disabled_credit",
                    "id": "us:statutes/26/22#elderly_disabled_credit",
                    "entity": "TaxUnit",
                },
            ],
            "parameters": [],
        }
    }
    cases = [
        {
            "name": "mixed_person_and_tax_unit_outputs",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "input": {},
            "output": {
                "us:statutes/26/22#is_aged_65_or_over": "holds",
                "us:statutes/26/22#elderly_disabled_credit": 750,
            },
        }
    ]

    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "statutes/26/22.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload=compiled_payload,
        cases=cases,
    )

    assert issues == [
        "Test case `mixed_person_and_tax_unit_outputs` mixes derived output "
        "entities (Person, TaxUnit); put outputs for each entity in separate "
        "test cases."
    ]


def test_rulespec_ci_allows_scalar_parameters_with_entity_outputs(
    tmp_path, monkeypatch
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )
    compiled_payload = {
        "program": {
            "derived": [
                {
                    "name": "aged_additional_amount_age_threshold",
                    "id": "us:statutes/26/63/f#aged_additional_amount_age_threshold",
                    "entity": "Scalar",
                },
                {
                    "name": "blind_under_subsection_f",
                    "id": "us:statutes/26/63/f#blind_under_subsection_f",
                    "entity": "Person",
                },
            ],
            "parameters": [
                {
                    "name": "aged_additional_amount_age_threshold",
                    "id": "us:statutes/26/63/f#aged_additional_amount_age_threshold",
                    "versions": [
                        {
                            "effective_from": "2026-01-01",
                            "values": {
                                "0": {
                                    "kind": "integer",
                                    "value": "65",
                                }
                            },
                        }
                    ],
                }
            ],
        }
    }
    cases = [
        {
            "name": "person_output_with_scalar_parameters",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "input": {},
            "output": {
                "us:statutes/26/63/f#aged_additional_amount_age_threshold": 65,
                "us:statutes/26/63/f#blind_under_subsection_f": "holds",
            },
        }
    ]

    monkeypatch.setattr(
        pipeline,
        "_axiom_rules_binary",
        lambda: tmp_path / "missing-rules-engine",
    )
    monkeypatch.setattr(
        pipeline,
        "_run_rulespec_derived_test_case",
        lambda **_kwargs: (
            {
                "us:statutes/26/63/f#blind_under_subsection_f": {
                    "kind": "judgment",
                    "outcome": "holds",
                }
            },
            [],
        ),
    )

    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "statutes/26/63/f.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload=compiled_payload,
        cases=cases,
    )

    assert issues == []


def test_rulespec_ci_allows_absolute_outputs_for_generated_local_names(
    tmp_path, monkeypatch
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )
    compiled_payload = {
        "program": {
            "derived": [
                {
                    "name": "blind_under_subsection_f",
                    "entity": "Person",
                },
            ],
            "parameters": [
                {
                    "name": "aged_or_blind_additional_amount",
                    "versions": [
                        {
                            "effective_from": "2026-01-01",
                            "values": {
                                "0": {
                                    "kind": "integer",
                                    "value": "600",
                                }
                            },
                        }
                    ],
                }
            ],
        }
    }
    cases = [
        {
            "name": "absolute_outputs_on_generated_artifact",
            "period": {
                "period_kind": "tax_year",
                "start": "2026-01-01",
                "end": "2026-12-31",
            },
            "input": {},
            "output": {
                "us:statutes/26/63/f#aged_or_blind_additional_amount": 600,
                "us:statutes/26/63/f#blind_under_subsection_f": "holds",
            },
        }
    ]

    monkeypatch.setattr(
        pipeline,
        "_axiom_rules_binary",
        lambda: tmp_path / "missing-rules-engine",
    )
    monkeypatch.setattr(
        pipeline,
        "_run_rulespec_derived_test_case",
        lambda **_kwargs: (
            {
                "blind_under_subsection_f": {
                    "kind": "judgment",
                    "outcome": "holds",
                }
            },
            [],
        ),
    )

    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "generated/statutes/26/63/f.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload=compiled_payload,
        cases=cases,
    )

    assert issues == []


def test_rulespec_dataset_uses_local_input_names_for_generated_artifacts(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes/26/63/f.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: blind_under_subsection_f
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: central_visual_acuity_in_better_eye_with_correcting_lenses <= 0.1
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )

    dataset = pipeline._build_rulespec_dataset(
        {
            "us:statutes/26/63/f#input.central_visual_acuity_in_better_eye_with_correcting_lenses": 0.1
        },
        period={
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        query_entity="Person",
        query_entity_id="person-1",
        require_legal_input_keys=False,
    )

    assert dataset["inputs"][0]["name"] == (
        "central_visual_acuity_in_better_eye_with_correcting_lenses"
    )


def test_rulespec_dataset_preserves_legal_input_names_for_repo_artifacts(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes/26/63/f.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: blind_under_subsection_f
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: central_visual_acuity_in_better_eye_with_correcting_lenses <= 0.1
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )

    input_key = (
        "us:statutes/26/63/f#input."
        "central_visual_acuity_in_better_eye_with_correcting_lenses"
    )
    dataset = pipeline._build_rulespec_dataset(
        {input_key: 0.1},
        period={
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        query_entity="Person",
        query_entity_id="person-1",
        require_legal_input_keys=True,
    )

    assert dataset["inputs"][0]["name"] == input_key


def test_rulespec_ci_rejects_computed_imported_outputs_as_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "missing-rules-engine",
        enable_oracles=False,
    )
    compiled_payload = {
        "program": {
            "derived": [
                {
                    "name": "snap_net_income",
                    "id": "us:statutes/7/2014/e/6/A#snap_net_income",
                    "entity": "Household",
                },
                {
                    "name": "snap_regular_month_allotment",
                    "id": "us:statutes/7/2017/a#snap_regular_month_allotment",
                    "entity": "Household",
                },
            ],
            "parameters": [],
        }
    }
    cases = [
        {
            "name": "stubs_imported_net_income",
            "period": "2026-01",
            "input": {"us:statutes/7/2014/e/6/A#snap_net_income": 100},
            "output": {"us:statutes/7/2017/a#snap_regular_month_allotment": 268},
        }
    ]

    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "statutes/7/2017/a.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload=compiled_payload,
        cases=cases,
    )

    assert issues == [
        "Test case `stubs_imported_net_income` assigns computed RuleSpec "
        "output(s) as input: `us:statutes/7/2014/e/6/A#snap_net_income`. "
        "Imported parameters and derived outputs are computed by the compiled "
        "program; assign their upstream `#input.*` or `#relation.*` facts instead."
    ]


def test_rulespec_ci_executes_relation_list_inputs(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: Household size is the number of household members.
rules:
  - name: household_size
    kind: derived
    entity: Household
    dtype: Integer
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: len(member_of_household)
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: two_members
  period: 2024-01
  input:
    member_of_household:
      - {}
      - {}
  output:
    household_size: 2
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_executes_table_entity_list_outputs(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: Net payment is payment amount less the excluded amount.
rules:
  - name: net_payment
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: payment_amount - excluded_amount
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: multiple_payment_rows
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  tables:
    Payment:
      - payment_amount: 100
        excluded_amount: 40
      - payment_amount: 20
        excluded_amount: 50
  output:
    net_payment:
      - 60
      - -30
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_compares_parameter_only_outputs(tmp_path, monkeypatch):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    _mock_corpus_source_text(
        monkeypatch, "The official source states the policy rate is 0.2."
    )

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The policy rate is 0.2.
  source_verification:
    corpus_citation_path: us/guidance/example/rate
rules:
  - name: policy_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: base_rate
  period:
    period_kind: tax_year
    start: 2024-04-06
    end: 2025-04-05
  input: {}
  output:
    policy_rate: 0.2
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_executes_indexed_parameter_table_lookup(tmp_path, monkeypatch):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    _mock_corpus_source_text(
        monkeypatch,
        "The official source lists $298 and $546 for sizes 1 and 2, plus $218.",
    )

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The maximum monthly allotments are 298 and 546 for household sizes 1 and 2,
    plus 218 for each additional person.
  source_verification:
    corpus_citation_path: us/guidance/example/allotments
rules:
  - name: benefit_amount_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: benefit_additional_member
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '218'
  - name: max_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if household_size > 2:
              benefit_amount_table[2] + ((household_size - 2) * benefit_additional_member)
          else: benefit_amount_table[household_size]
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: third_household_member_uses_additional_member_amount
  period: 2026-01
  input:
    household_size: 3
  output:
    max_allotment: 764
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_compares_indexed_parameter_outputs(tmp_path, monkeypatch):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    _mock_corpus_source_text(
        monkeypatch,
        "The official source lists $298 and $546 for household sizes 1 and 2.",
    )

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The maximum monthly allotments are 298 and 546.
  source_verification:
    corpus_citation_path: us/guidance/example/allotments
rules:
  - name: benefit_amount_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: second_household_member_table_value
  period: 2026-01
  input:
    household_size: 2
  output:
    benefit_amount_table: 546
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_legal_input_reference_accepts_parameter_index_slot(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "policies" / "irs" / "brackets.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: income_tax_bracket_rates
    kind: parameter
    dtype: Rate
    indexed_by: bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.10
"""
    )

    issue = validator_pipeline._rulespec_absolute_test_reference_issue(
        "us:policies/irs/brackets#input.bracket",
        label="input",
        policy_repo_path=repo,
        allow_input_slots=True,
        allow_relations=False,
        allow_outputs=False,
    )

    assert issue is None


def test_rulespec_test_reference_prefers_current_repo_over_stale_env_root(
    monkeypatch,
    tmp_path,
):
    stale_repo = tmp_path / "canonical" / "rulespec-us"
    stale_file = stale_repo / "regulations" / "7-cfr" / "273" / "10.yaml"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_net_income
"""
    )

    current_repo = tmp_path / "workspace" / "rulespec-us"
    current_file = current_repo / "regulations" / "7-cfr" / "273" / "10.yaml"
    current_file.parent.mkdir(parents=True)
    current_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if household_size <= 2: snap_minimum_benefit else: snap_net_income
"""
    )
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_repo))

    issue = validator_pipeline._rulespec_absolute_test_reference_issue(
        "us:regulations/7-cfr/273/10#input.household_size",
        label="input",
        policy_repo_path=current_repo,
        allow_input_slots=True,
        allow_relations=False,
        allow_outputs=False,
    )

    assert issue is None


def test_rulespec_ci_rejects_scale_tables_encoded_as_match_literals(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The maximum monthly allotments are 298 and 546 for household sizes 1 and 2.
rules:
  - name: max_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          match household_size:
              1 => 298
              2 => 546
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: one_person
  period: 2026-01
  input:
    household_size: 1
  output:
    max_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "Structured parameter table required" in issue and "max_allotment" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_parameter_values_without_indexed_by(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: placeholder
  period: 2026-01
  input: {}
  output: {}
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any("does not declare `indexed_by`" in issue for issue in result.issues)


def test_source_table_row_scalar_parameters_are_rejected():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
    | 3.0 | 3.5 | 4.9 |
rules:
  - name: avg_ratio_threshold_row_0_upper_2_5
    kind: parameter
    dtype: Rate
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: applicable_percentage_3201_row_0
    kind: parameter
    dtype: Rate
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 0.049
  - name: applicable_percentage_3201_row_1
    kind: parameter
    dtype: Rate
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 0.049
"""

    issues = find_source_table_row_scalar_parameter_issues(content)

    assert len(issues) == 1
    assert "Source table row/band scalar parameters" in issues[0]
    assert "avg_ratio_threshold_row_0_upper_2_5" in issues[0]
    assert "`indexed_by`" in issues[0]


def test_source_table_named_band_threshold_parameters_are_rejected():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
    | 3.0 | 3.5 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band_threshold_2_5
    kind: parameter
    dtype: Float
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_threshold_3_0
    kind: parameter
    dtype: Float
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band_threshold_3_5
    kind: parameter
    dtype: Float
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.5
"""

    issues = find_source_table_row_scalar_parameter_issues(content)

    assert len(issues) == 1
    assert "Source table row/band scalar parameters" in issues[0]
    assert "average_account_benefits_ratio_band_threshold_2_5" in issues[0]


def test_scoped_exception_category_amount_requires_category_gate():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_payment_qualifies_for_wage_exclusion
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: payment_qualifies
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_qualifies_for_wage_exclusion: max(0, payment_amount - group_term_life_insurance_payment_amount_includible_in_employee_gross_income) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]
    assert "group-term life insurance" in issues[0]


def test_scoped_exception_category_amount_allows_helper_gate():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_payment_qualifies_for_wage_exclusion
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: payment_qualifies
  - name: group_term_life_insurance_includible_carveout_from_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_is_for_group_term_life_insurance: min(payment_amount, group_term_life_insurance_payment_amount_includible_in_employee_gross_income) else: 0
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_qualifies_for_wage_exclusion: max(0, payment_amount - group_term_life_insurance_includible_carveout_from_exclusion) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_scoped_exception_category_amount_name_with_for_is_not_predicate():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_qualifies_for_wage_exclusion: max(0, payment_amount - payment_for_group_term_life_insurance_amount_includible_in_employee_gross_income) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_predicate_must_gate_amount_branch():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_qualifies_for_wage_exclusion: max(0, payment_amount - group_term_life_insurance_payment_amount_includible_in_employee_gross_income) else: 0 * payment_is_for_group_term_life_insurance
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_rejects_wrong_polarity_branch():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_is_for_group_term_life_insurance: payment_amount else: max(0, payment_amount - group_term_life_insurance_payment_amount_includible_in_employee_gross_income)
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_allows_positive_polarity_branch():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_is_for_group_term_life_insurance: max(0, payment_amount - group_term_life_insurance_payment_amount_includible_in_employee_gross_income) else: payment_amount
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_scoped_exception_category_amount_rejects_wrong_polarity_helper():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_includible_carveout_from_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_is_for_group_term_life_insurance: 0 else: group_term_life_insurance_payment_amount_includible_in_employee_gross_income
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_qualifies_for_wage_exclusion: max(0, payment_amount - group_term_life_insurance_includible_carveout_from_exclusion) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_allows_nested_gate_expression():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - (if payment_for_group_term_life_insurance: min(payment_amount, max(0, group_term_life_insurance_payment_includible_in_employee_gross_income)) else: 0)) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_scoped_exception_category_amount_rejects_nested_wrong_polarity_expression():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - (if payment_for_group_term_life_insurance: 0 else: group_term_life_insurance_payment_includible_in_employee_gross_income)) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_rejects_rate_for_as_gate():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if rate_for_group_term_life_insurance > 0: max(0, payment_amount - group_term_life_insurance_payment_includible_in_employee_gross_income) else: payment_amount
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


@pytest.mark.parametrize(
    "predicate_name",
    [
        "payment_for_group_term_life_insurance_percentage",
        "payment_for_group_term_life_insurance_total",
        "payment_for_group_term_life_insurance_sum",
        "payment_for_group_term_life_insurance_usd",
        "payment_for_group_term_life_insurance_dollars",
    ],
)
def test_scoped_exception_category_amount_rejects_payment_for_quantity_as_gate(
    predicate_name,
):
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if {predicate_name} > 0: max(0, payment_amount - group_term_life_insurance_payment_includible_in_employee_gross_income) else: payment_amount
""".format(predicate_name=predicate_name)

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_allows_nested_gated_amount_helper():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, (if payment_for_group_term_life_insurance: raw_group_term_life_insurance_payment_includible else: 0))
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_scoped_exception_category_amount_rejects_ungated_amount_helper_alias():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: raw_includible_amount
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_allows_chained_gated_amount_helper():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: group_term_life_insurance_includible_carveout_from_exclusion
  - name: group_term_life_insurance_includible_carveout_from_exclusion
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_for_group_term_life_insurance: raw_includible_amount else: 0
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_scoped_exception_category_amount_rejects_unrelated_nested_gate_in_helper():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          raw_includible_amount + (if payment_for_group_term_life_insurance: 1 else: 0)
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


def test_scoped_exception_category_amount_rejects_gated_helper_plus_constant():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: group_term_life_insurance_gated_raw_amount + 1
  - name: group_term_life_insurance_gated_raw_amount
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_for_group_term_life_insurance: raw_includible_amount else: 0
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
"""

    issues = find_scoped_exception_category_gate_issues(content)

    assert len(issues) == 1
    assert "Scoped exception category not gated" in issues[0]


@pytest.mark.parametrize(
    "helper_formula",
    [
        "min(max(0, payment_amount), group_term_life_insurance_gated_raw_amount)",
        "(group_term_life_insurance_gated_raw_amount) + (0)",
        "(if payment_for_group_term_life_insurance: raw_includible_amount else: 0) + (0)",
    ],
)
def test_scoped_exception_category_amount_allows_zero_preserving_helper(
    helper_formula,
):
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: group_term_life_insurance_payment_includible_carveout
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          {helper_formula}
  - name: group_term_life_insurance_gated_raw_amount
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if payment_for_group_term_life_insurance: raw_includible_amount else: 0
  - name: employer_plan_benefit_payment_excluded_from_wages
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              corpus_citation_path: us/statute/26/3121
              excerpt: except that this paragraph does not apply to a payment for group-term life insurance to the extent that such payment is includible in gross income
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employer_plan_payment_exclusion_applies: max(0, payment_amount - group_term_life_insurance_payment_includible_carveout) else: 0
""".format(helper_formula=helper_formula)

    issues = find_scoped_exception_category_gate_issues(content)

    assert issues == []


def test_source_table_band_bound_scalar_parameters_are_rejected():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
    | 3.0 | 3.5 | 4.9 |
    | 3.5 | 4.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_lower_bound_band_0
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_upper_bound_band_0
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_lower_bound_band_1
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
"""

    issues = find_source_table_row_scalar_parameter_issues(content)

    assert len(issues) == 1
    assert "Source table row/band scalar parameters" in issues[0]
    assert "average_account_benefits_ratio_lower_bound_band_0" in issues[0]
    assert "structural bounds inline" in issues[0]


def test_repair_source_table_band_bound_scalars_inlines_selector_bounds():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_lower_bound_band_0
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_upper_bound_band_0
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_lower_bound_band_0:
            -1
          else if average_account_benefits_ratio < average_account_benefits_ratio_upper_bound_band_0:
            0
          else:
            1
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.049
          1: 0
"""

    repaired, repaired_rules = repair_source_table_band_scalar_parameters(content)

    assert "average_account_benefits_ratio_lower_bound_band_0" not in repaired
    assert "average_account_benefits_ratio_upper_bound_band_0" not in repaired
    assert "average_account_benefits_ratio < 2.5" in repaired
    assert "< 3.0" in repaired
    assert "else if" not in repaired
    assert "average_account_benefits_ratio_band" in repaired_rules
    assert find_source_table_row_scalar_parameter_issues(repaired) == []


def test_repair_source_table_named_band_thresholds_inlines_selector_bounds():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band_threshold_2_5
    kind: parameter
    dtype: Float
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_threshold_3_0
    kind: parameter
    dtype: Float
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_2_5: 1
          else: if average_account_benefits_ratio < average_account_benefits_ratio_band_threshold_3_0: 2
          else: 3
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0
"""

    repaired, repaired_rules = repair_source_table_band_scalar_parameters(content)

    assert "average_account_benefits_ratio_band_threshold_2_5" not in repaired
    assert "average_account_benefits_ratio_band_threshold_3_0" not in repaired
    assert "average_account_benefits_ratio < 2.5" in repaired
    assert "average_account_benefits_ratio < 3.0" in repaired
    assert "average_account_benefits_ratio_band" in repaired_rules
    assert find_source_table_row_scalar_parameter_issues(repaired) == []


def test_repair_source_table_interval_alignment_parses_compact_rows():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b) | | | ------------------------------ | ------------------------------------------------------ | ----------------------------------------- | --- | | At least | But less than | | | | .............. | 2.5 | 22.1 | 4.9 | | 2.5 | 3.0 | 18.1 | 4.9 | | 3.0 | 3.5 | 15.1 | 4.9 | | 3.5 | 4.0 | 14.1 | 4.9 | | 4.0 | 6.1 | 13.1 | 4.9 | | 6.1 | 6.5 | 12.6 | 4.4 | | 6.5 | 7.0 | 12.1 | 3.9 | | 7.0 | 7.5 | 11.6 | 3.4 | | 7.5 | 8.0 | 11.1 | 2.9 | | 8.0 | 8.5 | 10.1 | 1.9 | | 8.5 | 9.0 | 9.1 | 0.9 | | 9.0 | .............. | 8.2 | 0 |
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if average_account_benefits_ratio >= 2.5 and average_account_benefits_ratio < 3.0: 1
          else: if average_account_benefits_ratio >= 3.0 and average_account_benefits_ratio < 3.5: 2
          else: if average_account_benefits_ratio >= 3.5 and average_account_benefits_ratio < 4.0: 3
          else: if average_account_benefits_ratio >= 4.0 and average_account_benefits_ratio < 6.1: 4
          else: if average_account_benefits_ratio >= 6.1 and average_account_benefits_ratio < 6.5: 5
          else: if average_account_benefits_ratio >= 6.5 and average_account_benefits_ratio < 7.0: 6
          else: if average_account_benefits_ratio >= 7.0 and average_account_benefits_ratio < 7.5: 7
          else: if average_account_benefits_ratio >= 7.5 and average_account_benefits_ratio < 8.0: 8
          else: if average_account_benefits_ratio >= 8.0 and average_account_benefits_ratio < 8.5: 9
          else: if average_account_benefits_ratio >= 8.5 and average_account_benefits_ratio < 9.0: 10
          else: if average_account_benefits_ratio >= 9.0: 11 else: 0
  - name: applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '1990-01-01'
        values:
          1: 0.221
          2: 0.181
          3: 0.151
          4: 0.141
          5: 0.131
          6: 0.126
          7: 0.121
          8: 0.116
          9: 0.111
          10: 0.101
          11: 0.082
  - name: applicable_percentage_for_section_3201_b_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '1990-01-01'
        values:
          1: 0.049
          2: 0.049
          3: 0.049
          4: 0.049
          5: 0.044
          6: 0.039
          7: 0.034
          8: 0.029
          9: 0.019
          10: 0.009
          11: 0.0
"""

    repaired, repaired_rules = repair_source_table_interval_row_alignment(content)

    assert "average_account_benefits_ratio_band" in repaired_rules
    payload = yaml.safe_load(repaired)
    selector = next(
        rule
        for rule in payload["rules"]
        if rule["name"] == "average_account_benefits_ratio_band"
    )
    assert selector["versions"][0]["formula"].startswith(
        "if average_account_benefits_ratio < 2.5: 1 else: "
        "if average_account_benefits_ratio >= 2.5"
    )
    assert selector["versions"][0]["formula"].endswith(
        "if average_account_benefits_ratio >= 8.5 and "
        "average_account_benefits_ratio < 9.0: 11 else: 12"
    )
    sections_3211_3221 = next(
        rule
        for rule in payload["rules"]
        if rule["name"]
        == "applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band"
    )
    section_3201 = next(
        rule
        for rule in payload["rules"]
        if rule["name"] == "applicable_percentage_for_section_3201_b_by_ratio_band"
    )
    assert sections_3211_3221["versions"][0]["values"][11] == 0.091
    assert sections_3211_3221["versions"][0]["values"][12] == 0.082
    assert section_3201["versions"][0]["values"][5] == 0.049
    assert section_3201["versions"][0]["values"][11] == 0.009
    assert section_3201["versions"][0]["values"][12] == 0.0


def test_repair_source_table_interval_tests_updates_guarded_lookup_outputs():
    rulespec_content = """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b) | | | ------------------------------ | ------------------------------------------------------ | ----------------------------------------- | --- | | At least | But less than | | | | .............. | 2.5 | 22.1 | 4.9 | | 2.5 | .............. | 18.1 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if average_account_benefits_ratio < 2.5: 1 else: 2
  - name: applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '1990-01-01'
        values:
          1: 0.221
          2: 0.181
  - name: applicable_percentage_for_section_3201_b_by_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    versions:
      - effective_from: '1990-01-01'
        values:
          1: 0.049
          2: 0.049
  - name: applicable_percentage_for_sections_3211_b_and_3221_b
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if average_account_benefits_ratio_band == 0: 0 else: applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band[average_account_benefits_ratio_band]
  - name: applicable_percentage_for_section_3201_b
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          if average_account_benefits_ratio_band == 0: 0 else: applicable_percentage_for_section_3201_b_by_ratio_band[average_account_benefits_ratio_band]
"""
    test_content = """- name: ratio_under_first_band
  period:
    period_kind: tax_year
    start: '2026-01-01'
    end: '2026-12-31'
  input:
    us:statutes/26/3241/b#input.average_account_benefits_ratio: 2.4
  output:
    us:statutes/26/3241/b#average_account_benefits_ratio_band: 1
    us:statutes/26/3241/b#applicable_percentage_for_sections_3211_b_and_3221_b: 0
    us:statutes/26/3241/b#applicable_percentage_for_section_3201_b: 0
"""

    repaired, repaired_cases = repair_source_table_interval_tests(
        test_content,
        rulespec_content=rulespec_content,
    )

    assert repaired_cases == ["ratio_under_first_band"]
    cases = yaml.safe_load(repaired)
    outputs = cases[0]["output"]
    assert (
        outputs[
            "us:statutes/26/3241/b#applicable_percentage_for_sections_3211_b_and_3221_b"
        ]
        == 0.221
    )
    assert (
        outputs["us:statutes/26/3241/b#applicable_percentage_for_section_3201_b"]
        == 0.049
    )


def test_repair_source_table_band_bound_scalars_inlines_adjacent_min_max():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band_1_max
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_2_min
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_2_max
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band_3_min
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_1_max:
            1
          else: if average_account_benefits_ratio < average_account_benefits_ratio_band_2_max and average_account_benefits_ratio >= average_account_benefits_ratio_band_2_min:
            2
          else:
            3
"""

    repaired, repaired_rules = repair_source_table_band_scalar_parameters(content)

    assert "average_account_benefits_ratio_band_1_max" not in repaired
    assert "average_account_benefits_ratio_band_2_min" not in repaired
    assert "average_account_benefits_ratio_band_2_max" not in repaired
    assert "average_account_benefits_ratio_band_3_min" not in repaired
    assert "average_account_benefits_ratio < 2.5" in repaired
    assert "< 3.0" in repaired
    assert ">= 2.5" in repaired
    assert "average_account_benefits_ratio_band" in repaired_rules
    assert find_source_table_row_scalar_parameter_issues(repaired) == []


def test_repair_source_table_band_bound_scalars_inlines_adjacent_upper_aliases():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: average_account_benefits_ratio_band_1_upper
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: average_account_benefits_ratio_band_2_lower
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: average_account_benefits_ratio_band_1_upper
  - name: average_account_benefits_ratio_band_2_upper
    kind: parameter
    dtype: Decimal
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: average_account_benefits_ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if average_account_benefits_ratio < average_account_benefits_ratio_band_1_upper: 1
          elif average_account_benefits_ratio >= average_account_benefits_ratio_band_2_lower and average_account_benefits_ratio < average_account_benefits_ratio_band_2_upper: 2
          else: 3
"""

    repaired, repaired_rules = repair_source_table_band_scalar_parameters(content)

    assert "average_account_benefits_ratio_band_1_upper" not in repaired
    assert "average_account_benefits_ratio_band_2_lower" not in repaired
    assert "average_account_benefits_ratio_band_2_upper" not in repaired
    assert "average_account_benefits_ratio < 2.5" in repaired
    assert ">= 2.5" in repaired
    assert "average_account_benefits_ratio < 3.0" in repaired
    assert "average_account_benefits_ratio_band" in repaired_rules
    assert find_source_table_row_scalar_parameter_issues(repaired) == []


def test_repair_source_table_band_bound_scalars_uses_external_table_text():
    content = """format: rulespec/v1
module:
  summary: Section defines applicable percentages by benefits ratio.
rules:
  - name: ratio_lower_bound_band_0
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 2.5
  - name: ratio_upper_bound_band_0
    kind: parameter
    dtype: Decimal
    versions:
      - effective_from: '2026-01-01'
        formula: 3.0
  - name: ratio_band
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if ratio < ratio_lower_bound_band_0:
            -1
          elif ratio < ratio_upper_bound_band_0:
            0
          else:
            1
"""

    repaired, _repaired_rules = repair_source_table_band_scalar_parameters(
        content,
        source_text="Tax rate schedule | Average account benefits ratio | 2.5 | 3.0",
    )

    assert "ratio_lower_bound_band_0" not in repaired
    assert "ratio < 2.5" in repaired
    assert "elif" not in repaired


def test_source_table_row_scalar_parameters_allows_indexed_table():
    content = """format: rulespec/v1
module:
  summary: |-
    Tax rate schedule | Average account benefits ratio | Applicable percentage
    | At least | But less than | Section 3201(b) |
    | .............. | 2.5 | 4.9 |
    | 2.5 | 3.0 | 4.9 |
rules:
  - name: applicable_percentage_3201_by_average_account_benefits_ratio_band
    kind: parameter
    dtype: Rate
    indexed_by: average_account_benefits_ratio_band
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.049
          1: 0.049
  - name: applicable_percentage_3201
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    source: 26 USC 3241(a), 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          applicable_percentage_3201_by_average_account_benefits_ratio_band[
              average_account_benefits_ratio_band
          ]
"""

    assert find_source_table_row_scalar_parameter_issues(content) == []


def test_source_verification_accepts_values_in_ingested_source_text():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/usda/fns/snap-fy2026-cola/page-1
    values:
      snap_maximum_allotment_table:
        1: 298
        2: 546
      snap_maximum_allotment_additional_member: 218
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: snap_maximum_allotment_additional_member
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '218'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/usda/fns/snap-fy2026-cola/page-1": (
                "Household Size 48 States & District of Columbia "
                "1 $298 2 $546 Each Additional Member $218"
            )
        },
    )

    assert issues == []


def test_source_verification_reads_local_corpus_artifact(
    tmp_path,
    monkeypatch,
):
    provisions_dir = tmp_path / "provisions" / "us" / "guidance"
    provisions_dir.mkdir(parents=True)
    (provisions_dir / "test-source.jsonl").write_text(
        json.dumps(
            {
                "citation_path": "us/guidance/example/page-1",
                "body": "The official normalized corpus source states the amount is $123.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AXIOM_CORPUS_ARTIFACT_ROOT", str(tmp_path))
    validator_pipeline._fetch_corpus_source_text.cache_clear()
    validator_pipeline._fetch_local_corpus_source_text.cache_clear()

    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      official_amount: 123
rules:
  - name: official_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '123'
"""

    assert find_source_verification_issues(content) == []
    assert find_ungrounded_numeric_issues(content) == []


def test_source_verification_accepts_values_in_corpus_source_text():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/irs/rev-proc-2025-32/page-18
    values:
      standard_deduction_single: 16100
rules:
  - name: standard_deduction_single
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '16100'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/irs/rev-proc-2025-32/page-18": (
                "For taxable years beginning in 2026, the standard deduction "
                "for unmarried individuals is $16,100."
            )
        },
    )

    assert issues == []


def test_source_verification_prefers_corpus_source_over_module_summary():
    content = """format: rulespec/v1
module:
  summary: The summary intentionally omits the exact official dollar amount.
  source_verification:
    corpus_citation_path: us/guidance/example/page-1
    values:
      official_amount: 140200
rules:
  - name: official_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '140200'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/example/page-1": (
                "Joint Returns or Surviving Spouses $140,200"
            )
        },
    )

    assert issues == []


def test_source_verification_accepts_decimal_rate_values_as_percentages():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/irs/rev-proc-2025-32/page-10
    values:
      income_tax_bracket_rates:
        1: 0.10
        2: 0.12
rules:
  - name: income_tax_bracket_rates
    kind: parameter
    dtype: Rate
    indexed_by: bracket
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 0.10
          2: 0.12
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/irs/rev-proc-2025-32/page-10": (
                "The applicable rates are 10% and 12%."
            )
        },
    )

    assert issues == []


def test_filing_status_branch_rejects_missing_surviving_spouse_code():
    content = """format: rulespec/v1
module:
  summary: The basic standard deduction is doubled for a joint return or surviving spouse.
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => standard_deduction_joint
              2 => standard_deduction_separate
              3 => standard_deduction_head_of_household
              0 => standard_deduction_single
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any("status code 4" in issue for issue in issues)


def test_filing_status_enum_rejects_string_formula():
    content = """format: rulespec/v1
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == "married_filing_jointly": standard_deduction_joint else:
          if filing_status == "surviving_spouse": standard_deduction_joint else:
          standard_deduction_single
"""

    issues = find_tax_filing_status_enum_representation_issues(content)

    assert any("Filing status must use numeric enum" in issue for issue in issues)


def test_filing_status_enum_rejects_named_match_arm():
    content = """format: rulespec/v1
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              married_filing_jointly => standard_deduction_joint
              surviving_spouse => standard_deduction_joint
              single => standard_deduction_single
"""

    issues = find_tax_filing_status_enum_representation_issues(content)

    assert any("Filing status must use numeric enum" in issue for issue in issues)


def test_filing_status_enum_rejects_quoted_named_match_arm():
    content = """format: rulespec/v1
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              "married_filing_jointly" => standard_deduction_joint
              "surviving_spouse" => standard_deduction_joint
              "single" => standard_deduction_single
"""

    issues = find_tax_filing_status_enum_representation_issues(content)

    assert any("Filing status must use numeric enum" in issue for issue in issues)


def test_filing_status_enum_rejects_inline_named_match_arm():
    content = """format: rulespec/v1
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status: married_filing_jointly => standard_deduction_joint; surviving_spouse => standard_deduction_joint; single => standard_deduction_single
"""

    issues = find_tax_filing_status_enum_representation_issues(content)

    assert any("Filing status must use numeric enum" in issue for issue in issues)


def test_filing_status_enum_allows_named_arms_in_unrelated_match_block():
    content = """format: rulespec/v1
rules:
  - name: household_type_adjusted_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match household_type:
              single => if filing_status == 1: joint_household_amount else: single_household_amount
              family => family_household_amount
"""

    assert find_tax_filing_status_enum_representation_issues(content) == []


def test_filing_status_local_input_rejects_formula_without_import():
    content = """format: rulespec/v1
rules:
  - name: joint_return_bonus
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 1: 100 else: 0
"""

    issues = find_tax_filing_status_local_input_issues(content)

    assert any(
        "Filing status is a derived legal classification" in issue for issue in issues
    )


def test_empty_rules_module_rejects_missing_status():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules: []
"""

    issues = find_empty_rules_module_issues(content)

    assert any("Empty RuleSpec module invalid" in issue for issue in issues)


def test_empty_rules_module_allows_explicit_deferred_status():
    content = """format: rulespec/v1
module:
  status: deferred
rules: []
"""

    assert find_empty_rules_module_issues(content) == []


def test_source_scope_consistency_rejects_person_source_as_household_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    No individual who refuses to provide a required Social Security number
    shall be eligible to participate as a member of any household.
rules:
  - name: snap_ssn_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 CFR 273.6
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          count_where(member_of_household, member_has_provided_ssn) == len(member_of_household)
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `snap_ssn_eligible` is declared on "
        "`Household`, but the embedded source states an "
        "individual/person/member-scoped eligibility or disqualification. "
        "Encode the rule at the person/member scope or cite source text that "
        "states the unit-level test."
    ]


def test_filtered_entity_dependency_rejects_snapunit_without_relation():
    content = """format: rulespec/v1
rules:
  - name: household_entitled_to_expedited_service
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: 7 CFR 273.2
    versions:
      - effective_from: '2026-01-01'
        formula: expedited_service_conditions_met
"""

    issues = find_filtered_entity_dependency_issues(content)

    assert issues == [
        "Filtered entity dependency missing: "
        "`household_entitled_to_expedited_service` uses `entity: SnapUnit`, "
        "but this RuleSpec file does not declare `SnapUnit` with a "
        "`kind: derived_relation` rule or import its declaring relation "
        "(`snap_unit`)."
    ]


def test_filtered_entity_dependency_allows_local_snapunit_relation():
    content = """format: rulespec/v1
rules:
  - name: household_member_eligible_for_snap_unit
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: household_member_meets_snap_unit_rules
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    source: 7 CFR 273.1(a)
    versions:
      - effective_from: '2026-01-01'
        formula: household_member_eligible_for_snap_unit
  - name: snap_unit_entitled_to_expedited_service
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: synthetic source
    versions:
      - effective_from: '2026-01-01'
        formula: expedited_service_conditions_met
"""

    assert find_filtered_entity_dependency_issues(content) == []


def test_filtered_entity_dependency_allows_imported_snapunit_relation():
    content = """format: rulespec/v1
imports:
  - us:regulations/7-cfr/273/1#snap_unit
rules:
  - name: snap_unit_entitled_to_expedited_service
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: synthetic source
    versions:
      - effective_from: '2026-01-01'
        formula: expedited_service_conditions_met
"""

    assert find_filtered_entity_dependency_issues(content) == []


def test_source_scope_consistency_accepts_person_source_as_person_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    No individual who refuses to provide a required Social Security number
    shall be eligible to participate as a member of any household.
rules:
  - name: snap_member_ssn_requirement_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.6
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_has_provided_ssn
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_rejects_household_source_as_person_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    Household resources are tested against the applicable resource limit for
    household eligibility.
rules:
  - name: snap_member_resource_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.8
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_resources <= resource_limit
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `snap_member_resource_eligible` is declared on "
        "`Person`, but the embedded source states a household/unit-scoped test. "
        "Encode the rule at the source-stated unit scope or cite source text "
        "that states the person-level test."
    ]


def test_source_scope_consistency_allows_household_source_as_snapunit_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    Household resources are tested against the applicable resource limit for
    household eligibility.
rules:
  - name: snap_unit_resource_eligible
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: 7 CFR 273.8
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          snap_unit_resources <= resource_limit
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_rejects_taxunit_source_as_household_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    The tax unit income must be below the eligibility standard.
rules:
  - name: household_tax_unit_income_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Year
    source: tax manual
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          household_income <= tax_unit_income_standard
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `household_tax_unit_income_eligible` is "
        "declared on `Household`, but the embedded source states a `TaxUnit` "
        "unit-scoped test. Encode the rule at the source-stated unit scope or "
        "cite source text that states the declared unit scope."
    ]


def test_source_scope_consistency_accepts_matching_snapunit_source():
    content = """format: rulespec/v1
module:
  summary: |-
    The SNAP unit income is tested against the applicable standard.
rules:
  - name: snap_unit_income_eligible
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: state manual
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          snap_unit_income <= snap_unit_income_standard
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_does_not_treat_family_member_as_family_unit():
    content = """format: rulespec/v1
module:
  summary: |-
    A qualifying family member is eligible for food assistance.
rules:
  - name: qualifying_family_member_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: state manual
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          is_qualifying_family_member
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_recognizes_state_manual_person_wording():
    content = """format: rulespec/v1
module:
  summary: |-
    An applicant who fails to cooperate with identity verification is not
    eligible for food assistance.
rules:
  - name: household_identity_verification_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: state manual
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          all_applicants_cooperated_with_identity_verification
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `household_identity_verification_eligible` "
        "is declared on `Household`, but the embedded source states an "
        "individual/person/member-scoped eligibility or disqualification. "
        "Encode the rule at the person/member scope or cite source text that "
        "states the unit-level test."
    ]


def test_source_scope_consistency_recognizes_state_manual_unit_wording():
    content = """format: rulespec/v1
module:
  summary: |-
    The assistance unit income must be below the eligibility standard.
rules:
  - name: applicant_income_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: state manual
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          applicant_income <= eligibility_standard
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `applicant_income_eligible` is declared on "
        "`Person`, but the embedded source states a household/unit-scoped test. "
        "Encode the rule at the source-stated unit scope or cite source text "
        "that states the person-level test."
    ]


def test_source_scope_consistency_does_not_guess_mixed_source_scope():
    content = """format: rulespec/v1
module:
  summary: |-
    Household eligibility depends on each household member meeting an
    individual condition.
rules:
  - name: snap_member_condition_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: mixed source
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_condition_met
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_uses_rule_proof_excerpt_before_mixed_summary():
    content = """format: rulespec/v1
module:
  summary: |-
    Household eligibility depends on income, resources, and whether each
    household member satisfies several person-level disqualification rules.
rules:
  - name: snap_sponsored_alien_verification_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 CFR 273.4(c)(5)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: "Until the alien provides information or verification necessary to carry out the provisions of paragraph (c)(2), the sponsored alien is ineligible."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          all_members_have_sponsor_information
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `snap_sponsored_alien_verification_eligible` "
        "is declared on `Household`, but the embedded source states an "
        "individual/person/member-scoped eligibility or disqualification. "
        "Encode the rule at the person/member scope or cite source text that "
        "states the unit-level test."
    ]


def test_source_scope_consistency_rejects_definite_person_subject_at_unit_scope():
    content = """format: rulespec/v1
module:
  summary: |-
    Until the alien provides information or verification necessary to carry out
    the provisions of paragraph (c)(2), the sponsored alien is ineligible.
rules:
  - name: sponsored_alien_ineligible_while_verification_missing
    kind: derived
    entity: SnapUnit
    dtype: Judgment
    period: Month
    source: 7 CFR 273.4(c)(5)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          not sponsored_alien_provided_verification
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: "
        "`sponsored_alien_ineligible_while_verification_missing` is declared "
        "on `SnapUnit`, but the embedded source states an "
        "individual/person/member-scoped eligibility or disqualification. "
        "Encode the rule at the person/member scope or cite source text that "
        "states the unit-level test."
    ]


def test_source_scope_consistency_skips_rule_with_mixed_proof_excerpt():
    content = """format: rulespec/v1
module:
  summary: |-
    Household eligibility depends on each household member meeting an
    individual condition.
rules:
  - name: snap_member_condition_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: mixed source
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: "Household eligibility depends on each household member meeting an individual condition."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          all_member_conditions_met
"""

    assert find_source_scope_consistency_issues(content) == []


def test_source_scope_consistency_skips_no_household_member_counterfactual_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    No household shall receive more benefits than it would have received if no
    household member was rendered ineligible.
rules:
  - name: household_benefit_after_ineligible_member_cap
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    source: state statute
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: "No household shall receive more benefits than it would have received if no household member was rendered ineligible."
    versions:
      - effective_from: '1998-09-01'
        formula: min(benefit_calculated_under_section, benefit_if_no_member_ineligible)
"""

    assert find_source_scope_consistency_issues(content) == []


def test_person_scoped_rate_base_unit_rejects_unit_level_rate_base():
    content = """format: rulespec/v1
module:
  summary: |-
    In addition to other taxes, there shall be imposed for each taxable year, on
    the self-employment income of every individual, a tax equal to 12.4 percent
    of the amount of the self-employment income for such taxable year.
rules:
  - name: self_employment_oasdi_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: 0.124
  - name: self_employment_oasdi_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 1401(a)
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_self_employment_income_for_section_1401 * self_employment_oasdi_tax_rate
"""

    issues = find_person_scoped_rate_base_unit_issues(content)

    assert any("Person-scoped rate base at unit scope" in issue for issue in issues)
    assert "self_employment_oasdi_tax" in issues[0]
    assert "TaxUnit" in issues[0]


def test_person_scoped_rate_base_unit_accepts_relation_rollup():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, a payroll contribution is 6.2 percent of taxable wages.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: payroll_contribution_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: 0.062
  - name: employee_payroll_contribution
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    source: payroll contribution rule
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_wages * payroll_contribution_rate
  - name: tax_unit_payroll_contribution
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: payroll contribution rule
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_payroll_contribution, employee_has_taxable_wages)
"""

    assert find_person_scoped_rate_base_unit_issues(content) == []


def test_person_scoped_definition_unit_rejects_individual_income_definition():
    content = """format: rulespec/v1
module:
  summary: |-
    The term self-employment income means the net earnings from self-employment
    derived by an individual during any taxable year; except that such term
    shall not include net earnings below $400.
rules:
  - name: self_employment_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 1402(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if net_earnings_from_self_employment < 400:
            0
          else:
            net_earnings_from_self_employment
"""

    issues = find_person_scoped_definition_unit_issues(content)

    assert any("Person-scoped definition at unit scope" in issue for issue in issues)
    assert "self_employment_income" in issues[0]
    assert "TaxUnit" in issues[0]


def test_person_scoped_definition_unit_accepts_relation_rollup():
    content = """format: rulespec/v1
module:
  summary: |-
    Each employee's covered wages are wages paid to such employee. The tax unit
    amount is the sum of covered wages for members of the tax unit.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: covered_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    source: source
    versions:
      - effective_from: '2026-01-01'
        formula: wages_paid_to_employee
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: source
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, covered_wages, member_has_wages)
"""

    assert find_person_scoped_definition_unit_issues(content) == []


def test_employer_scoped_entity_rejects_tax_unit():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tier 2 tax In addition to other taxes, there is hereby imposed on every
    employer an excise tax equal to the percentage determined under section 3241.
rules:
  - name: tier_2_employer_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3221(b)
    versions:
      - effective_from: '2026-01-01'
        formula: compensation_paid * applicable_percentage
"""

    issues = find_employer_scoped_entity_issues(content)

    assert any(
        "Employer-scoped rule at non-employer scope" in issue for issue in issues
    )
    assert "tier_2_employer_tax" in issues[0]
    assert "TaxUnit" in issues[0]


def test_employer_scoped_entity_ignores_employee_paid_tax_not_collected_by_employer():
    content = """format: rulespec/v1
module:
  summary: |-
    (2) Collection of amounts not withheld To the extent that the amount of any
    tax imposed by section 3101(b)(2) is not collected by the employer, such tax
    shall be paid by the employee.
rules:
  - name: employee_payment_responsibility_for_uncollected_additional_medicare_tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3102(f)(2)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              excerpt: tax imposed by section 3101(b)(2) is not collected by the employer
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, additional_medicare_tax - additional_medicare_tax_collected_by_employer)
"""

    assert find_employer_scoped_entity_issues(content) == []


def test_employer_scoped_entity_accepts_employer():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tier 2 tax In addition to other taxes, there is hereby imposed on every
    employer an excise tax equal to the percentage determined under section 3241.
rules:
  - name: tier_2_employer_tax
    kind: derived
    entity: Employer
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 3221(b)
    versions:
      - effective_from: '2026-01-01'
        formula: compensation_paid * applicable_percentage
"""

    assert find_employer_scoped_entity_issues(content) == []


def test_shared_statutory_rate_name_rejects_tax_unit_suffix():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage
    for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
rules:
  - name: section_3211_and_3221_applicable_percentage_for_tax_unit
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: section_3211_3221_applicable_percentage_by_ratio_band[average_account_benefits_ratio_band]
"""

    issues = find_shared_statutory_rate_entity_suffix_name_issues(content)

    assert any(
        "Shared statutory rate name should use source-stated application" in issue
        for issue in issues
    )
    assert "section_3211_and_3221_applicable_percentage_for_tax_unit" in issues[0]


def test_shared_statutory_rate_name_rejects_section_prefix_name():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage
    for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
rules:
  - name: section_3201_applicable_percentage
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: section_3201_applicable_percentage_by_ratio_band[average_account_benefits_ratio_band]
"""

    issues = find_shared_statutory_rate_entity_suffix_name_issues(content)

    assert any(
        "section-prefixed local cross-reference name" in issue for issue in issues
    )
    assert "section_3201_applicable_percentage" in issues[0]


def test_shared_statutory_rate_name_accepts_source_stated_section_name():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Tax rate schedule | Average account benefits ratio | Applicable percentage
    for sections 3211(b) and 3221(b) | Applicable percentage for section 3201(b)
rules:
  - name: applicable_percentage_for_sections_3211_b_and_3221_b
    kind: derived
    entity: TaxUnit
    dtype: Rate
    period: Year
    source: 26 USC 3241(b)
    versions:
      - effective_from: '2026-01-01'
        formula: applicable_percentage_for_sections_3211_b_and_3221_b_by_ratio_band[average_account_benefits_ratio_band]
"""

    assert find_shared_statutory_rate_entity_suffix_name_issues(content) == []


def test_source_scope_consistency_checks_each_rule_independently():
    content = """format: rulespec/v1
module:
  summary: |-
    Household eligibility depends on resources and member-level alien status.
rules:
  - name: snap_member_alien_status_eligible
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 CFR 273.4
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: "No person is eligible to participate in the Program unless that person meets a listed citizenship or alien status condition."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          member_has_eligible_alien_status
  - name: snap_sponsored_alien_verification_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 CFR 273.4(c)(5)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: condition
            source:
              excerpt: "Until the alien provides information or verification necessary to carry out paragraph (c)(2), the sponsored alien is ineligible."
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          all_sponsored_aliens_provided_verification
"""

    issues = find_source_scope_consistency_issues(content)

    assert issues == [
        "Source scope mismatch: `snap_sponsored_alien_verification_eligible` "
        "is declared on `Household`, but the embedded source states an "
        "individual/person/member-scoped eligibility or disqualification. "
        "Encode the rule at the person/member scope or cite source text that "
        "states the unit-level test."
    ]


def test_filing_status_local_input_allows_imported_formula():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/6013#filing_status
rules:
  - name: joint_return_bonus
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 1: joint_amount else: other_amount
"""

    assert find_tax_filing_status_local_input_issues(content) == []


def test_tax_status_component_local_input_rejects_surviving_spouse_fact():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/63
rules:
  - name: additional_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxpayer_is_surviving_spouse: regular_amount else: higher_amount
"""

    issues = find_tax_status_component_local_input_issues(content)

    assert any(
        "Tax filing-status component is a derived legal classification" in issue
        for issue in issues
    )


def test_tax_status_component_local_input_allows_snap_elderly_disabled_rule():
    """The tax-status validator should not fire on Title 7 (SNAP) rules.

    7 USC 2012(j) defines "elderly or disabled member" partly by reference
    to a person who is the surviving spouse of a veteran with specified
    Title 38 status. The `surviving_spouse` substring in the input slot
    name overlaps tax-filing-status vocabulary but the legal context is
    SNAP demographic eligibility, not income-tax filing status. Validator
    must restrict to Title 26 sources.
    """
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: elderly_or_disabled_member
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: |-
          person_is_sixty_years_of_age_or_older
          or person_is_surviving_spouse_of_veteran_with_specified_title_38_status
"""

    issues = find_tax_status_component_local_input_issues(content)

    assert issues == [], (
        f"Title 7 SNAP rules must not trigger the tax-status validator: {issues}"
    )


def test_tax_status_component_local_input_rejects_compound_status_fact():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/63
rules:
  - name: applicable_aged_or_blind_additional_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_is_not_married_and_is_not_surviving_spouse: higher_amount else: regular_amount
"""

    issues = find_tax_status_component_local_input_issues(content)

    assert any(
        "individual_is_not_married_and_is_not_surviving_spouse" in issue
        for issue in issues
    )


def test_tax_status_component_local_input_allows_imported_surviving_spouse():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/2/a#taxpayer_is_surviving_spouse
rules:
  - name: additional_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if taxpayer_is_surviving_spouse: regular_amount else: higher_amount
"""

    assert find_tax_status_component_local_input_issues(content) == []


def test_tax_status_component_local_input_allows_3121_b_3_family_service_context():
    content = """format: rulespec/v1
module:
  summary: |-
    (3) domestic service in a private home of the employer, except that the
    provisions of this subparagraph shall not be applicable to such domestic
    service performed by an individual in the employ of his son or daughter if
    the employer is a surviving spouse or a divorced individual and has not
    remarried.
rules:
  - name: family_employment_service_excluded_from_employment
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3121(b)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: exception
            source:
              excerpt: "the employer is a surviving spouse or a divorced individual and has not remarried"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          domestic_service_in_private_home_of_employer
          and employer_is_surviving_spouse_or_divorced_individual_and_has_not_remarried
"""

    assert find_tax_status_component_local_input_issues(content) == []


def test_unused_modifier_parameter_rejects_ignored_substitution_amount():
    content = """format: rulespec/v1
rules:
  - name: regular_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(1)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "additional amount of $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 600
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
  - name: additional_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          regular_additional_amount
"""

    issues = find_unused_modifier_parameter_issues(content)

    assert any(
        "`unmarried_not_surviving_spouse_additional_amount`" in issue
        for issue in issues
    )


def test_unused_modifier_parameter_allows_substitution_amount_use():
    content = """format: rulespec/v1
rules:
  - name: regular_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: 600
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
  - name: additional_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if special_branch: unmarried_not_surviving_spouse_additional_amount else: regular_additional_amount
"""

    assert find_unused_modifier_parameter_issues(content) == []


def test_unused_modifier_parameter_rejects_no_affected_numeric_output():
    content = """format: rulespec/v1
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
  - name: blind_under_subsection
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: visual_acuity <= 0.1
"""

    issues = find_unused_modifier_parameter_issues(content)

    assert any(
        "has no affected numeric derived output" in issue
        and "`unmarried_not_surviving_spouse_additional_amount`" in issue
        for issue in issues
    )


def test_unused_modifier_parameter_allows_explicit_deferred_output():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: us:statutes/26/63/f#additional_standard_deduction_amount_under_subsection_f
      reason: Requires upstream surviving-spouse status before selecting the substituted amount.
      blocked_by:
        - us:statutes/26/2/a#surviving_spouse
      source_values:
        - us:statutes/26/63/f#unmarried_not_surviving_spouse_additional_amount
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
  - name: blind_under_subsection
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: visual_acuity <= 0.1
"""

    assert find_deferred_output_issues(content) == []
    assert find_unused_modifier_parameter_issues(content) == []


def test_unused_modifier_parameter_rejects_unlinked_deferred_output():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: us:statutes/26/63/f#additional_standard_deduction_amount_under_subsection_f
      reason: Requires upstream surviving-spouse status before selecting the substituted amount.
      blocked_by:
        - us:statutes/26/2/a#surviving_spouse
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
"""

    issues = find_unused_modifier_parameter_issues(content)

    assert any("module.deferred_outputs[].source_values" in issue for issue in issues)


def test_deferred_output_rejects_bare_output_and_blocker():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: additional_standard_deduction_amount_under_subsection_f
      reason: Missing upstream status.
      blocked_by:
        - surviving_spouse
      source_values:
        - unmarried_not_surviving_spouse_additional_amount
rules: []
"""

    issues = find_deferred_output_issues(content)

    assert any("must use an absolute RuleSpec output" in issue for issue in issues)
    assert any(
        "blocked_by" in issue and "absolute RuleSpec target" in issue
        for issue in issues
    )
    assert any(
        "source_values" in issue and "absolute RuleSpec target" in issue
        for issue in issues
    )


def test_deferred_output_rejects_absolute_blocker_without_rule_fragment():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: us:regulations/7-cfr/273/4#qualified_alien_eligible_to_receive_snap_benefits
      reason: Missing upstream INA withholding-of-removal rule.
      blocked_by:
        - us:statutes/us/241/b/3
rules: []
"""

    issues = find_deferred_output_issues(content)

    assert issues == [
        "module.deferred_outputs[0].blocked_by entry "
        "`us:statutes/us/241/b/3` must be an absolute RuleSpec target with a "
        "rule fragment."
    ]


def test_deferred_output_allows_unknown_blockers_in_reason_without_blocked_by():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: us-ca:statutes/wic/18901/5#calfresh_categorical_eligibility
      reason: Requires California General Assistance rules under WIC 17000 and SNAP categorical eligibility rules under WIC 18930, but no exact RuleSpec outputs were available in context.
rules: []
"""

    assert find_deferred_output_issues(content) == []


def test_deferred_output_rejects_embedded_jurisdiction_blocker_path():
    content = """format: rulespec/v1
module:
  deferred_outputs:
    - output: us-ca:statutes/wic/18901/5#calfresh_categorical_eligibility
      reason: Missing upstream eligibility rules.
      blocked_by:
        - us:statutes/us-ca/17000#general_assistance_eligibility
rules: []
"""

    issues = find_deferred_output_issues(content)

    assert issues == [
        "module.deferred_outputs[0].blocked_by entry "
        "`us:statutes/us-ca/17000#general_assistance_eligibility` embeds a "
        "jurisdiction in the path; use the target jurisdiction prefix instead."
    ]


def test_source_subparagraph_coverage_rejects_high_signal_omission():
    source_text = """Definitions
(a) Benefit means the amount payable under this program.
(b) Effective date. This section applies after October 1.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
  summary: Definitions
rules: []
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2012": source_text},
    )

    assert issues == [
        "Source sub-paragraph coverage missing: 7 USC 2012(a) "
        "('Benefit means the amount payable under this program.') has no rule "
        "citing it and no entry in `module.deferred_outputs`. Either encode a "
        "rule with `source: 7 USC 2012(a)` or add a deferred_outputs entry "
        "naming the blocker."
    ]


def test_source_subparagraph_coverage_allows_repealed_empty_slice(tmp_path):
    source_text = """Wages
(a) Wages means all remuneration for employment.
"""
    content = """format: rulespec/v1
module:
  status: deferred
  source_verification:
    corpus_citation_path: us/statute/26/3121
  summary: |-
    (3) Repealed. Pub. L. 98-21, title III, section 324(a)(3)(B).
rules: []
"""
    rules_file = tmp_path / "rulespec-us" / "statutes" / "26" / "3121" / "a" / "3.yaml"

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            rules_file=rules_file,
            source_texts={"us/statute/26/3121": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_allows_rule_citing_child():
    source_text = """Definitions
(a) Benefit means the amount payable under this program.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    source: 7 USC 2012(a)
    versions:
      - effective_from: '2026-01-01'
        formula: benefit_amount
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_rejects_sibling_omission_for_top_level_rule():
    source_text = """Definitions
(a) Benefit means the amount payable under this program.
(b) Household means an individual or group.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    source: 7 USC 2012(a)
    versions:
      - effective_from: '2026-01-01'
        formula: benefit_amount
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2012": source_text},
    )

    assert len(issues) == 1
    assert "7 USC 2012(b)" in issues[0]


def test_source_subparagraph_coverage_allows_rule_citing_descendant():
    source_text = """Definitions
(m) Household means an individual who lives alone or a group of individuals who live together.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: snap_household_member_condition
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2012(m)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: lives_with_household
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_scopes_nested_rulespec_file_path(tmp_path):
    source_text = """Eligibility disqualifications
(a) Additional specific conditions rendering individuals ineligible.
(d) Work requirement (1) In general. (2) Exemptions.
(e) Students means individuals enrolled in higher education.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2015
rules:
  - name: title_iv_work_registration_exemption_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2015(d)(2)(A)
    versions:
      - effective_from: '2026-01-01'
        formula: complying_with_title_iv_work_registration
"""
    rules_file = (
        tmp_path / "rulespec-us" / "statutes" / "7" / "2015" / "d" / "2" / "A.yaml"
    )

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            rules_file=rules_file,
            source_texts={"us/statute/7/2015": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_scopes_state_statute_file_path(tmp_path):
    source_text = """Modifications to federal taxable income
(4) Subtractions from federal taxable income.
    (y) Military retirement benefits may be subtracted subject to stated dollar limits.
(f) Pensions or annuities from federal adjusted gross income may be subtracted.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/statute/39/39-22-104
rules:
  - name: military_retirement_benefits_subtraction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    source: 39-22-104(4)(y)(I)
    versions:
      - effective_from: '2019-01-01'
        formula: military_retirement_benefits
"""
    rules_file = (
        tmp_path / "rulespec-us-co" / "statutes" / "39" / "39-22-104" / "4" / "y.yaml"
    )

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            rules_file=rules_file,
            source_texts={"us-co/statute/39/39-22-104": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_accepts_state_rulespec_target_requested_source():
    source_text = """Modifications to federal taxable income
(4) Subtractions from federal taxable income.
    (y) Military retirement benefits may be subtracted subject to stated dollar limits.
(f) Pensions or annuities from federal adjusted gross income may be subtracted.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/statute/39/39-22-104
rules:
  - name: military_retirement_benefits_subtraction
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    source: 39-22-104(4)(y)(I)
    versions:
      - effective_from: '2019-01-01'
        formula: military_retirement_benefits
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us-co/statute/39/39-22-104": source_text},
            requested_source="us-co:statutes/39/39-22-104/4/y",
        )
        == []
    )


def test_source_subparagraph_coverage_scopes_to_requested_source_under_parent_fallback(
    tmp_path,
):
    """When the corpus serves a parent-fallback source slice for a sub-subsection
    encoding target, the validator must scope subparagraph coverage to the
    *requested* target rather than the entire parent statute. Otherwise an
    encoder asked to produce a rule for `7 USC 2014(e)(2)(B)` would have to
    defer or encode every top-level subparagraph of § 2014, which is out of
    its scope.

    Surfaced live by us_snap_earned_income_deduction_refresh.yaml on
    2026-05-27: corpus lacked the (e)(2)(B) slice, so it served the whole
    § 2014; validator then demanded coverage for (a), (c), (d), (g), (h),
    (k), (l) — none of which the encoder was asked to touch.
    """
    source_text = """Eligibility disqualifications
(a) Income standards. Households with income above thresholds are ineligible.
(c) Gross income standard. Adjusted October 1 each year.
(d) Exclusions from income. Various items excluded.
(e) Deductions from income.
    (1) Standard deduction.
    (2) (B) Earned income deduction of 20 percent.
(g) Allowable financial resources. Asset limits apply.
"""
    content = """format: rulespec/v1
module:
  status: deferred
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: Earned income deduction under 7 USC 2014(e)(2)(B).
  deferred_outputs:
    - output: us:statutes/7/2014/e/2/B#snap_earned_income_deduction
      reason: Requires the (e)(2)(C) exception which is not in scope.
rules: []
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2014": source_text},
        requested_source="us/statute/7/2014/e/2/B",
    )

    # All seven sibling subparagraphs (a, c, d, g) are out of scope when the
    # request targets (e)(2)(B). The encoder must not be held responsible
    # for them.
    assert issues == [], (
        "Validator complained about subparagraphs the encoder was never "
        f"asked to cover: {issues}"
    )


def test_source_subparagraph_coverage_uses_applied_manifest_requested_source(tmp_path):
    source_text = """Eligibility disqualifications
(a) Income standards. Households with income above thresholds are ineligible.
(c) Gross income standard. Adjusted October 1 each year.
(d) Exclusions from income. Various items excluded.
(e) Deductions from income.
    (1) Standard deduction.
    (2) (B) Earned income deduction of 20 percent.
(g) Allowable financial resources. Asset limits apply.
"""
    policy_repo = tmp_path / "rulespec-us"
    rules_file = policy_repo / "statutes/7/2014/e/2/B.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
module:
  status: deferred
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: Earned income deduction under 7 USC 2014(e)(2)(B).
  deferred_outputs:
    - output: us:statutes/7/2014/e/2/B#snap_earned_income_deduction
      reason: Requires the (e)(2)(C) exception which is not in scope.
rules: []
"""
    rules_file.write_text(content)
    manifest_path = policy_repo / ".axiom/encoding-manifests/statutes/7/2014/e/2/B.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "citation": "us/statute/7/2014/e/2/B",
                "schema_version": "axiom-encode/applied-rulespec/v1",
            }
        )
    )

    source_metadata = _load_applied_encoding_manifest_source_metadata(
        rules_file,
        policy_repo,
    )
    assert source_metadata is not None
    issues = find_source_subparagraph_coverage_issues(
        content,
        rules_file=rules_file,
        source_texts={"us/statute/7/2014": source_text},
        requested_source=source_metadata["requested_source"],
    )

    assert issues == []


def test_source_subparagraph_coverage_without_requested_source_keeps_strict_scope():
    """Sanity check: when no requested_source is supplied, behaviour is
    unchanged — the validator demands coverage of every top-level
    subparagraph as before."""
    source_text = """Definitions
(a) Benefit means the amount payable.
(b) Household means an individual or group.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
  summary: Definitions
rules: []
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2012": source_text},
    )
    assert len(issues) == 2
    assert any("7 USC 2012(a)" in i for i in issues)
    assert any("7 USC 2012(b)" in i for i in issues)


def test_source_subparagraph_coverage_accepts_human_readable_requested_source():
    """When the eval workspace writes requested_source in human form
    ('7 USC 2014(c)') rather than corpus-path form ('us/statute/7/2014/c'),
    the validator must still recognize it and scope subparagraph coverage to
    the requested fragment. Surfaced live on 7 USC 2014(c) encode 2026-05-28:
    workspace stored requested_source as the human form, scope function did
    not match, and all six sibling subparagraphs were flagged as missing.
    """
    source_text = """Eligibility disqualifications
(a) Income standards. Households with income above thresholds are ineligible.
(c) Gross income standard. Adjusted October 1 each year.
(d) Exclusions from income.
(g) Allowable financial resources.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2014
  summary: Gross and net income standards under 7 USC 2014(c).
rules:
  - name: snap_net_income_exceeds_income_standard
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 USC 2014(c)
    versions:
      - effective_from: '2008-10-01'
        formula: "snap_net_income > applicable_poverty_line"
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2014": source_text},
        requested_source="7 USC 2014(c)",
    )
    assert issues == [], (
        f"Validator should scope to (c) but flagged out-of-scope siblings: {issues}"
    )


def test_source_subparagraph_coverage_matches_irc_section_citation(tmp_path):
    source_text = """Standard deduction
(a) Rule for taxable years.
(c) Standard deduction means the sum of the basic standard deduction and the additional standard deduction.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/63
rules:
  - name: dependent_basic_standard_deduction_limit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: IRC section 63(c)(5)
    versions:
      - effective_from: '2026-01-01'
        formula: dependent_standard_deduction_limit
"""
    rules_file = tmp_path / "rulespec-us" / "statutes" / "26" / "63" / "c" / "5.yaml"

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            rules_file=rules_file,
            source_texts={"us/statute/26/63": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_rejects_broad_parent_citation():
    source_text = """Definitions
(m) Household means an individual who lives alone or a group of individuals who live together.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: snap_household_note
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    source: 7 USC 2012
    versions:
      - effective_from: '2026-01-01'
        formula: household_definition_applies
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2012": source_text},
    )

    assert len(issues) == 1
    assert "7 USC 2012(m)" in issues[0]


def test_source_subparagraph_coverage_allows_deferred_child_path():
    source_text = """Definitions
(m) Household means an individual who lives alone or a group of individuals who live together.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
  deferred_outputs:
    - output: us:statutes/7/2012/m#snap_household
      reason: Requires a base source relation not yet available.
rules: []
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_rejects_deferred_parent_path_only():
    source_text = """Definitions
(m) Household means an individual who lives alone or a group of individuals who live together.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
  deferred_outputs:
    - output: us:statutes/7/2012#snap_household
      reason: Parent path is too broad to cover subsection m.
rules: []
"""

    issues = find_source_subparagraph_coverage_issues(
        content,
        source_texts={"us/statute/7/2012": source_text},
    )

    assert len(issues) == 1
    assert "7 USC 2012(m)" in issues[0]


def test_source_subparagraph_coverage_ignores_low_signal_children():
    source_text = """Definitions
(a) Effective date. This section applies after October 1.
(b) Severability. If any provision is held invalid, the rest remains in effect.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules: []
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_skips_missing_source_verification():
    content = """format: rulespec/v1
module:
  summary: |-
    (m) Household means an individual who lives alone.
rules: []
"""

    assert find_source_subparagraph_coverage_issues(content) == []


def test_source_subparagraph_coverage_skips_multiple_source_verification_paths():
    source_text = """Definitions
(m) Household means an individual who lives alone.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us/statute/7/2012
      - us/statute/7/2014
rules: []
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={
                "us/statute/7/2012": source_text,
                "us/statute/7/2014": source_text,
            },
        )
        == []
    )


def test_source_subparagraph_coverage_skips_missing_source_text(monkeypatch):
    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline._fetch_corpus_source_text",
        lambda citation_path: None,
    )
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules: []
"""

    assert find_source_subparagraph_coverage_issues(content) == []


def test_source_subparagraph_coverage_ignores_indented_nested_markers():
    source_text = """Definitions
(m) Household means one of the following:
  (i) an individual who lives alone.
  (ii) a group of individuals who live together.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
  deferred_outputs:
    - output: us:statutes/7/2012/m#snap_household
      reason: Requires a base source relation not yet available.
rules: []
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_source_subparagraph_coverage_ignores_column_zero_nested_roman_i():
    source_text = """Definitions
(a) Application process.
(1) Special criteria.
(i) Eligible alien means an alien satisfying this nested condition.
(b) Benefit means the amount payable under this program.
"""
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/7/2012
rules:
  - name: benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    source: 7 USC 2012(a), 7 USC 2012(b)
    versions:
      - effective_from: '2026-01-01'
        formula: benefit_amount
"""

    assert (
        find_source_subparagraph_coverage_issues(
            content,
            source_texts={"us/statute/7/2012": source_text},
        )
        == []
    )


def test_unused_modifier_parameter_ignores_judgment_names_with_amount_word():
    content = """format: rulespec/v1
rules:
  - name: unmarried_not_surviving_spouse_additional_amount
    kind: parameter
    dtype: Money
    source: IRC section 63(f)(3)
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: amount
            source:
              excerpt: "applied by substituting $750 for $600"
    versions:
      - effective_from: '2026-01-01'
        formula: 750
  - name: taxpayer_aged_additional_amount_entitlement
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: taxpayer_has_attained_age_65_before_close_of_taxable_year
  - name: additional_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: unmarried_not_surviving_spouse_additional_amount
"""

    assert find_unused_modifier_parameter_issues(content) == []


def test_filing_status_local_input_allows_numeric_test_fixture():
    content = """format: rulespec/v1
rules:
  - name: filing_status_sensitive_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: 0
"""
    test_cases = [
        {
            "name": "joint_status_code",
            "input": {"us:statutes/26/63/c#input.filing_status": 1},
            "output": {},
        }
    ]

    issues = find_tax_filing_status_local_input_issues(content, test_cases)

    assert not any(
        "assigns filing status as a local input" in issue for issue in issues
    )


def test_filing_status_test_input_rejects_string_value():
    test_cases = [
        {
            "name": "joint_status_string",
            "input": {
                "us:policies/irs/rev-proc-2025-32/standard-deduction#input.filing_status": "married_filing_jointly"
            },
            "output": {},
        }
    ]

    issues = find_tax_filing_status_test_input_issues(test_cases)

    assert any(
        "Filing status is a derived legal classification" in issue for issue in issues
    )


def test_filing_status_test_input_allows_numeric_enum_fixture():
    test_cases = [
        {
            "name": "joint_status_code",
            "input": {
                "us:policies/irs/rev-proc-2025-32/standard-deduction#input.filing_status": 1
            },
            "output": {},
        }
    ]

    issues = find_tax_filing_status_test_input_issues(test_cases)

    assert issues == []


def test_filing_status_test_input_rejects_out_of_range_numeric_value():
    test_cases = [
        {
            "name": "bad_status_code",
            "input": {
                "us:policies/irs/rev-proc-2025-32/standard-deduction#input.filing_status": 9
            },
            "output": {},
        }
    ]

    issues = find_tax_filing_status_test_input_issues(test_cases)

    assert any(
        "Filing status is a derived legal classification" in issue for issue in issues
    )


def test_filing_status_test_input_rejects_tax_filing_status_alias():
    test_cases = [
        {
            "name": "joint_status_code",
            "input": {
                "us:policies/irs/rev-proc-2025-32/standard-deduction#input.tax_filing_status": 1
            },
            "output": {},
        }
    ]

    issues = find_tax_filing_status_test_input_issues(test_cases)

    assert any(
        "Filing status is a derived legal classification" in issue for issue in issues
    )


def test_filing_status_branch_allows_surviving_spouse_code():
    content = """format: rulespec/v1
module:
  summary: The basic standard deduction is doubled for a joint return or surviving spouse.
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => standard_deduction_joint
              4 => standard_deduction_joint
              2 => standard_deduction_separate
              3 => standard_deduction_head_of_household
              0 => standard_deduction_single
"""

    assert find_tax_filing_status_surviving_spouse_issues(content) == []


def test_filing_status_branch_rejects_surviving_spouse_different_result():
    content = """format: rulespec/v1
module:
  summary: The basic standard deduction is doubled for a joint return or surviving spouse.
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => standard_deduction_joint
              4 => standard_deduction_single
              2 => standard_deduction_separate
              3 => standard_deduction_head_of_household
              0 => standard_deduction_single
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any("different result" in issue for issue in issues)


def test_filing_status_branch_rejects_comparison_surviving_spouse_different_result():
    content = """format: rulespec/v1
module:
  summary: The basic standard deduction is doubled for a joint return or surviving spouse.
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 1: standard_deduction_joint else:
          if filing_status == 4: standard_deduction_single else:
          standard_deduction_single
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any("different result" in issue for issue in issues)


def test_filing_status_branch_scopes_surviving_spouse_group_to_rule_source():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Threshold amount means $110,000 in the case of a joint return, $75,000 in the case of an individual who is not married, and $55,000 in the case of a married individual filing a separate return.
    (j) Applicable income threshold means $60,000 in the case of a joint return or surviving spouse, $50,000 in the case of a head of household, and $40,000 in any other case.
rules:
  - name: ctc_phaseout_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 24(b)(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => joint_threshold
              4 => unmarried_threshold
              2 => separate_threshold
              3 => unmarried_threshold
              0 => unmarried_threshold

  - name: ctc_excess_advance_applicable_income_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 24(j)(2)(B)(iii)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => joint_or_surviving_spouse_threshold
              4 => joint_or_surviving_spouse_threshold
              3 => head_of_household_threshold
              2 => other_threshold
              0 => other_threshold
"""

    assert find_tax_filing_status_surviving_spouse_issues(content) == []


def test_filing_status_branch_allows_joint_return_exclusion_without_surviving_spouse():
    content = """format: rulespec/v1
module:
  summary: The source mentions surviving spouse elsewhere, but this rule excludes joint returns from the unmarried individual exception.
rules:
  - name: unmarried_individual_filing_exception
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          taxpayer_is_individual
          and filing_status != 1
          and filing_status != 2
          and gross_income <= standard_deduction
"""

    assert find_tax_filing_status_surviving_spouse_issues(content) == []


def test_filing_status_branch_rejects_surviving_spouse_for_joint_only_any_other_case():
    content = """format: rulespec/v1
module:
  summary: The threshold is $250,000 in the case of a joint return and $200,000 in any other case.
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 3101(b)(2)(A)-(C)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => joint_threshold
              4 => joint_threshold
              2 => separate_threshold
              3 => other_threshold
              0 => other_threshold
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any(
        "incorrectly treats surviving spouse as joint return" in issue
        for issue in issues
    )


def test_filing_status_branch_uses_subparagraph_range_source_context():
    content = """format: rulespec/v1
module:
  summary: |-
    (a) Old-age, survivors, and disability insurance.
    (b) Hospital insurance (1) In general. (2) Additional tax
    The tax applies to wages which are in excess of--
    (A) in the case of a joint return, $250,000,
    (B) in the case of a married taxpayer filing a separate return, one-half
    of the dollar amount determined under subparagraph (A), and
    (C) in any other case, $200,000.
    (c) Relief from taxes.
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 3101(b)(2)(A)-(C)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => joint_threshold
              4 => joint_threshold
              2 => separate_threshold
              3 => other_threshold
              0 => other_threshold
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any(
        "incorrectly treats surviving spouse as joint return" in issue
        for issue in issues
    )


def test_filing_status_branch_allows_surviving_spouse_as_other_case_for_joint_only():
    content = """format: rulespec/v1
module:
  summary: The threshold is $250,000 in the case of a joint return and $200,000 in any other case.
rules:
  - name: additional_medicare_wage_tax_threshold
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    source: 26 USC 3101(b)(2)(A)-(C)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => joint_threshold
              4 => other_threshold
              2 => separate_threshold
              3 => other_threshold
              0 => other_threshold
"""

    assert find_tax_filing_status_surviving_spouse_issues(content) == []


def test_filing_status_branch_rejects_unrelated_surviving_spouse_code():
    content = """format: rulespec/v1
module:
  summary: The basic standard deduction is doubled for a joint return or surviving spouse.
rules:
  - name: basic_standard_deduction_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match filing_status:
              1 => standard_deduction_joint
              2 => standard_deduction_separate
              0 => standard_deduction_single
          match unrelated_enum:
              4 => unrelated_result
"""

    issues = find_tax_filing_status_surviving_spouse_issues(content)

    assert any("status code 4" in issue for issue in issues)


def test_nonnegative_amount_reduction_rejects_unfloored_allotment():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate)
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_rejects_unfloored_maximum_name_without_prefix():
    content = """format: rulespec/v1
rules:
  - name: monthly_benefit_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          maximum_benefit - income_reduction
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_rejects_unfloored_max_name_without_prefix():
    content = """format: rulespec/v1
rules:
  - name: monthly_benefit_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          max_benefit - income_reduction
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_allows_zero_floor():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          max(0, snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate))
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_nonnegative_amount_reduction_rejects_unfloored_taxable_income_branch():
    content = """format: rulespec/v1
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: taxable_income_for_individual_who_does_not_itemize else: taxable_income_general_rule
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any("Nonnegative taxable income missing floor" in issue for issue in issues)


def test_repair_nonnegative_amount_reductions_floors_taxable_income_branches():
    content = """format: rulespec/v1
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: taxable_income_for_individual_who_does_not_itemize else: taxable_income_general_rule
"""

    repaired, rules = repair_nonnegative_amount_reductions(content)

    assert rules == ["taxable_income"]
    assert (
        "if individual_who_does_not_elect_to_itemize_deductions_for_taxable_year: "
        "max(0, taxable_income_for_individual_who_does_not_itemize) "
        "else: max(0, taxable_income_general_rule)" in repaired
    )
    assert find_nonnegative_amount_reduction_issues(repaired) == []


def test_repair_nonnegative_amount_reductions_does_not_replace_identifier_substrings():
    content = """format: rulespec/v1
rules:
  - name: capital_gains_excluded_from_taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          taxable_income
          - max(
              taxable_income - net_capital_gain,
              min(
                  min(max(taxable_income, 0), capital_gains_zero_rate_threshold),
                  taxable_income - adjusted_net_capital_gain
              )
          )
"""

    repaired, rules = repair_nonnegative_amount_reductions(content)

    assert rules == ["capital_gains_excluded_from_taxable_income"]
    assert "name: capital_gains_excluded_from_taxable_income" in repaired
    assert "capital_gains_excluded_from_max" not in repaired
    assert "max(0, taxable_income" in repaired
    assert find_nonnegative_amount_reduction_issues(repaired) == []


def test_nonnegative_amount_reduction_allows_zero_floor_for_limit_minus_reduction():
    content = """format: rulespec/v1
rules:
  - name: qualified_passenger_vehicle_interest_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          max(0, passenger_vehicle_interest_after_dollar_limit - passenger_vehicle_interest_phaseout_reduction)
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_nonnegative_amount_reduction_allows_zero_floor_with_trailing_zero_argument():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          max(snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate), 0)
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_formula_date_literal_rejects_iso_dates_in_formulas():
    content = """format: rulespec/v1
rules:
  - name: passenger_vehicle_loan_interest_period_start
    kind: parameter
    dtype: String
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          2025-01-01
"""

    issues = find_formula_date_literal_issues(content)

    assert any("Formula date literal unsupported" in issue for issue in issues)
    assert any(
        "taxable_year_begins_after_termination_date" in issue for issue in issues
    )


def test_formula_date_literal_rejects_iso_dates_in_derived_relation_formulas():
    content = """format: rulespec/v1
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
        formula: |-
          member_entry_date >= 2025-01-01
"""

    issues = find_formula_date_literal_issues(content)

    assert any("Formula date literal unsupported" in issue for issue in issues)


def test_temporal_value_fact_name_rejects_year_embedded_taxable_year_input():
    content = """format: rulespec/v1
rules:
  - name: section_applies_before_termination
    kind: derived
    entity: Business
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1994-01-01'
        formula: |-
          not taxable_year_begins_after_december_31_2021
"""

    issues = find_temporal_value_fact_name_issues(content)

    assert any(
        "Temporal fact name embeds legal date value" in issue for issue in issues
    )
    assert any(
        "taxable_year_begins_after_termination_date" in issue for issue in issues
    )


def test_temporal_value_fact_name_allows_semantic_taxable_year_input():
    content = """format: rulespec/v1
rules:
  - name: section_applies_before_termination
    kind: derived
    entity: Business
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1994-01-01'
        formula: |-
          not taxable_year_begins_after_termination_date
"""

    assert find_temporal_value_fact_name_issues(content) == []


def test_helper_only_definition_rejects_missing_final_defined_term():
    content = """format: rulespec/v1
module:
  summary: |-
    (a) Definition of surviving spouse (1) In general For purposes of section 1,
    the term "surviving spouse" means a taxpayer whose spouse died.
rules:
  - name: surviving_spouse_limitations_satisfied
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: not taxpayer_remarried_before_close_of_taxable_year
"""

    issues = find_helper_only_definition_issues(content)

    assert any(
        "Definition provision missing final defined term" in issue for issue in issues
    )
    assert any("surviving_spouse" in issue for issue in issues)


def test_helper_only_definition_allows_final_defined_term():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Definition of head of household (1) In general For purposes of this
    subtitle, an individual shall be considered a head of household if conditions
    are met.
rules:
  - name: head_of_household
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: household_requirements_satisfied
"""

    assert find_helper_only_definition_issues(content) == []


def test_helper_only_definition_rejects_final_not_encoded_note_with_rules():
    content = """format: rulespec/v1
module:
  summary: |-
    (b) Definition of head of household (1) In general.
    The final head-of-household status surface is not encoded here because
    an upstream source is unavailable.
rules:
  - name: head_household_status_prerequisites_satisfied
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: not taxpayer_is_nonresident_alien
"""

    issues = find_helper_only_definition_issues(content)

    assert any("helper-only" in issue for issue in issues)


def test_judgment_conditional_formula_rejects_if_else_returning_judgments():
    content = """format: rulespec/v1
rules:
  - name: snap_income_eligible_for_month
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_household_has_elderly_or_disabled_member: snap_net_monthly_income <= monthly_net_income_eligibility_standard else: snap_net_monthly_income <= monthly_net_income_eligibility_standard and snap_countable_gross_monthly_income <= monthly_gross_income_eligibility_standard
"""

    issues = find_judgment_conditional_formula_issues(content)

    assert any("Judgment conditional formula unsupported" in issue for issue in issues)
    assert any("boolean expression" in issue for issue in issues)


def test_nonnegative_amount_reduction_allows_zero_branch_with_floored_else():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if ineligible: 0 else: max(0, snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate))
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_nonnegative_amount_reduction_allows_nested_inline_zero_branch_with_floored_inner_else():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if ineligible: 0 else: if has_utility_cost: max(0, snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate)) else: 0
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_nonnegative_amount_reduction_allows_downstream_min_wrapper_after_zero_floor():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          min(snap_maximum_allotment_for_household_size, max(0, snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate)))
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_nonnegative_amount_reduction_rejects_unfloored_sibling_next_to_zero_floor():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          min(snap_maximum_allotment_for_household_size - snap_net_monthly_income_reduction, max(0, snap_maximum_allotment_for_household_size - snap_net_monthly_income_reduction))
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_rejects_unfloored_sibling_after_leading_zero_floor():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          max(0, snap_maximum_allotment_for_household_size - snap_net_monthly_income_reduction) + (snap_maximum_allotment_for_household_size - snap_net_monthly_income_reduction)
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_rejects_unfloored_income_base_in_credit():
    content = """format: rulespec/v1
rules:
  - name: eitc_phased_in
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * min(earned_income, eitc_earned_income_amount)
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount income base missing floor" in issue for issue in issues
    )


def test_nonnegative_amount_reduction_accepts_zero_floored_income_base_in_credit():
    content = """format: rulespec/v1
rules:
  - name: eitc_phased_in
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * min(max(0, earned_income), eitc_earned_income_amount)
"""

    assert find_nonnegative_amount_reduction_issues(content) == []


def test_repair_nonnegative_amount_reductions_floors_income_base_in_credit():
    content = """format: rulespec/v1
rules:
  - name: eitc_phased_in
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * min(earned_income, eitc_earned_income_amount)
"""

    repaired, rules = repair_nonnegative_amount_reductions(content)

    assert rules == ["eitc_phased_in"]
    assert (
        "eitc_phase_in_rate * min(max(0, earned_income), eitc_earned_income_amount)"
        in repaired
    )
    assert find_nonnegative_amount_reduction_issues(repaired) == []


def test_repair_nonnegative_amount_reductions_floors_multiline_income_base_in_credit():
    content = """format: rulespec/v1
rules:
  - name: qualified_family_leave_wages_credit_limited_to_employment_taxes
    kind: derived
    entity: Employer
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          min(
            qualified_family_leave_wages_credit_against_applicable_employment_taxes,
            applicable_employment_taxes_after_section_3131_credits
          )
"""

    repaired, rules = repair_nonnegative_amount_reductions(content)

    assert rules == ["qualified_family_leave_wages_credit_limited_to_employment_taxes"]
    assert (
        "min(max(0, "
        "qualified_family_leave_wages_credit_against_applicable_employment_taxes),"
        in repaired
    )
    assert find_nonnegative_amount_reduction_issues(repaired) == []


def test_current_year_final_amount_table_rejects_recomputed_maximum(tmp_path):
    repo = tmp_path / "rulespec-us"
    imported = repo / "policies/irs/rev-proc-2025-32/earned-income-credit.yaml"
    imported.parent.mkdir(parents=True)
    imported.write_text(
        """format: rulespec/v1
rules:
  - name: eitc_maximum_credit_amounts
    kind: parameter
    dtype: Money
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 664
          1: 4427
"""
    )
    rules_file = repo / "statutes/26/32.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
imports:
  - us:policies/irs/rev-proc-2025-32/earned-income-credit
rules:
  - name: eitc_capped_child_count
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(eitc_child_count, 3)
  - name: eitc_maximum
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/32
              text: "credit percentage of the earned income amount"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * eitc_earned_income_amount
"""

    issues = find_current_year_final_amount_table_issues(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert any("Current-year final amount table ignored" in issue for issue in issues)
    assert any("eitc_maximum_credit_amounts" in issue for issue in issues)


def test_repair_current_year_final_amount_tables_uses_imported_table(tmp_path):
    repo = tmp_path / "rulespec-us"
    imported = repo / "policies/irs/rev-proc-2025-32/earned-income-credit.yaml"
    imported.parent.mkdir(parents=True)
    imported.write_text(
        """format: rulespec/v1
rules:
  - name: eitc_maximum_credit_amounts
    kind: parameter
    dtype: Money
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 664
          1: 4427
"""
    )
    rules_file = repo / "statutes/26/32.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
imports:
  - us:policies/irs/rev-proc-2025-32/earned-income-credit
rules:
  - name: eitc_capped_child_count
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(eitc_child_count, 3)
  - name: eitc_maximum
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: formula
            source:
              corpus_citation_path: us/statute/26/32
              text: "credit percentage of the earned income amount"
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * eitc_earned_income_amount
"""

    repaired, rules = repair_current_year_final_amount_tables(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert rules == ["eitc_maximum"]
    assert "match eitc_capped_child_count:" in repaired
    assert "0 => eitc_maximum_credit_amounts[0]" in repaired
    assert "1 => eitc_maximum_credit_amounts[1]" in repaired
    assert (
        "target: us:policies/irs/rev-proc-2025-32/earned-income-credit#eitc_maximum_credit_amounts"
        in repaired
    )
    assert (
        find_current_year_final_amount_table_issues(
            repaired,
            rules_file=rules_file,
            policy_repo_path=repo,
        )
        == []
    )


def test_repair_current_year_final_amount_tables_caps_phased_in_by_maximum(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes/26/32.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: eitc_maximum
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: '4427'
  - name: eitc_phased_in
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          eitc_phase_in_rate * min(max(0, earned_income), eitc_earned_income_amount)
"""

    repaired, rules = repair_current_year_final_amount_tables(
        content,
        rules_file=rules_file,
        policy_repo_path=repo,
    )

    assert rules == ["eitc_phased_in"]
    assert (
        "if earned_income >= eitc_earned_income_amount: eitc_maximum else: "
        "eitc_phase_in_rate * min(max(0, earned_income), eitc_earned_income_amount)"
        in repaired
    )


def test_repair_nonnegative_amount_reductions_floors_conditional_rounding_branches():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if state_agency_rounds_thirty_percent_net_income_up: snap_maximum_allotment_for_household_size - ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate) else: floor(snap_maximum_allotment_for_household_size - (snap_net_monthly_income * snap_allotment_net_income_reduction_rate))
"""

    repaired, rules = repair_nonnegative_amount_reductions(content)

    assert rules == ["snap_calculated_monthly_allotment_before_minimums"]
    assert (
        "if state_agency_rounds_thirty_percent_net_income_up: "
        "max(0, snap_maximum_allotment_for_household_size - ceil("
        "snap_net_monthly_income * snap_allotment_net_income_reduction_rate)) "
        "else: max(0, floor(snap_maximum_allotment_for_household_size - "
        "(snap_net_monthly_income * snap_allotment_net_income_reduction_rate)))"
        in repaired
    )
    assert find_nonnegative_amount_reduction_issues(repaired) == []


def test_nonnegative_amount_reduction_rejects_intermediate_zero_floor():
    content = """format: rulespec/v1
rules:
  - name: snap_calculated_monthly_allotment_before_minimums
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          snap_maximum_allotment_for_household_size - max(0, ceil(snap_net_monthly_income * snap_allotment_net_income_reduction_rate))
"""

    issues = find_nonnegative_amount_reduction_issues(content)

    assert any(
        "Nonnegative amount reduction missing floor" in issue for issue in issues
    )


def test_zero_branch_test_coverage_rejects_untested_zero_output():
    content = """format: rulespec/v1
rules:
  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if household_initial_month and snap_amount < snap_minimum_issuance: 0 else: snap_amount
"""
    cases = [
        {
            "name": "above_threshold_initial_month",
            "output": {"us:regulations/7-cfr/273/10#snap_monthly_allotment": 90},
        }
    ]

    issues = find_zero_branch_test_coverage_issues(content, cases)

    assert any("Zero branch test coverage missing" in issue for issue in issues)


def test_zero_branch_test_coverage_allows_zero_output_case():
    content = """format: rulespec/v1
rules:
  - name: snap_monthly_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if household_initial_month and snap_amount < snap_minimum_issuance: 0 else: snap_amount
"""
    cases = [
        {
            "name": "below_threshold_initial_month",
            "output": {"us:regulations/7-cfr/273/10#snap_monthly_allotment": 0},
        }
    ]

    assert find_zero_branch_test_coverage_issues(content, cases) == []


def test_zero_branch_test_coverage_allows_table_zero_output_case():
    content = """format: rulespec/v1
rules:
  - name: predecessor_remuneration_considered_paid_by_successor
    kind: derived
    entity: Payment
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if successor_employer_wage_base_continuity_applies:
              predecessor_remuneration_before_acquisition
          else:
              0
"""
    cases = [
        {
            "name": "no_successor_continuity",
            "output": {
                "us:statutes/26/3121/a/1#predecessor_remuneration_considered_paid_by_successor": [
                    0
                ]
            },
        }
    ]

    assert find_zero_branch_test_coverage_issues(content, cases) == []


def test_zero_branch_test_coverage_rejects_untested_else_zero_output():
    content = """format: rulespec/v1
rules:
  - name: refundable_credit_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if credit_eligible: tentative_credit else: 0
"""
    cases = [
        {
            "name": "eligible_credit",
            "output": {"us:statutes/26/24#refundable_credit_amount": 500},
        }
    ]

    issues = find_zero_branch_test_coverage_issues(content, cases)

    assert any("Zero branch test coverage missing" in issue for issue in issues)


def test_zero_branch_test_coverage_rejects_untested_match_zero_output():
    content = """format: rulespec/v1
rules:
  - name: benefit_amount
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match eligibility_status:
              0 => 0
              1 => maximum_benefit
"""
    cases = [
        {
            "name": "eligible_benefit",
            "output": {"us:regulations/example#benefit_amount": 100},
        }
    ]

    issues = find_zero_branch_test_coverage_issues(content, cases)

    assert any("Zero branch test coverage missing" in issue for issue in issues)


def test_source_condition_coverage_rejects_cost_availability_as_only_exclusions():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us-ny/regulation/18-nycrr/387/12/f/3/v
      - us-ny/regulation/18-nycrr/387/12/f/3/v/c
rules:
  - name: snap_telephone_allowance_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          not snap_heating_cooling_standard_allowance_eligible
          and not snap_utilities_standard_allowance_eligible
"""

    issues = find_source_condition_coverage_issues(
        content,
        source_texts={
            "us-ny/regulation/18-nycrr/387/12/f/3/v": (
                "Standard allowances are available to households billed separately "
                "and on a recurring basis for heating/cooling costs, other utility "
                "costs and/or telephone costs."
            ),
            "us-ny/regulation/18-nycrr/387/12/f/3/v/c": (
                "The standard allowance for telephone is $32 per month for households "
                "that do not qualify for the heating/cooling or utilities allowances."
            ),
        },
    )

    assert any("Source condition coverage missing" in issue for issue in issues)
    assert "snap_telephone_allowance_eligible" in issues[0]


def test_source_condition_coverage_accepts_positive_cost_fact_predicate():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us-ny/regulation/18-nycrr/387/12/f/3/v
      - us-ny/regulation/18-nycrr/387/12/f/3/v/c
rules:
  - name: snap_telephone_allowance_eligible
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_heating_cooling_standard_allowance_eligible
             or snap_utilities_standard_allowance_eligible:
              false
          else: household_billed_separately_for_telephone_service
"""

    issues = find_source_condition_coverage_issues(
        content,
        source_texts={
            "us-ny/regulation/18-nycrr/387/12/f/3/v": (
                "Standard allowances are available to households billed separately "
                "and on a recurring basis for heating/cooling costs, other utility "
                "costs and/or telephone costs."
            ),
            "us-ny/regulation/18-nycrr/387/12/f/3/v/c": (
                "The standard allowance for telephone is $32 per month for households "
                "that do not qualify for the heating/cooling or utilities allowances."
            ),
        },
    )

    assert issues == []


def test_source_condition_coverage_uses_module_summary_for_sliced_source():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/45A
  summary: |-
    (f) Termination This section shall not apply to taxable years beginning after December 31, 2021.
rules:
  - name: section_45A_applies_before_termination
    kind: derived
    entity: Business
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '1994-01-01'
        formula: not taxable_year_begins_after_december_31_2021
"""

    issues = find_source_condition_coverage_issues(
        content,
        source_texts={
            "us/statute/26/45A": (
                "The credit is allowed for qualified employee health insurance costs "
                "paid or incurred by the employer."
            )
        },
    )

    assert issues == []


def test_relation_aggregate_syntax_rejects_expression_sum_over_relation():
    content = """format: rulespec/v1
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: additional_condition_count
    kind: derived
    entity: TaxUnit
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          sum(member_of_tax_unit, (if is_aged_65_or_over: 1 else: 0) + (if is_blind: 1 else: 0))
"""

    issues = find_relation_aggregate_syntax_issues(content)

    assert any("Unsupported relation aggregate syntax" in issue for issue in issues)
    assert "sum(member_of_tax_unit, ...)" in issues[0]


def test_relation_aggregate_syntax_rejects_sum_of_local_derived_relation_field():
    content = """format: rulespec/v1
rules:
  - name: capital_asset_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: capital_asset_of_tax_unit
      arity: 2
  - name: short_term_capital_gain
    kind: derived
    entity: Asset
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if asset_sale_or_exchange_is_of_capital_asset and asset_held_one_year_or_less:
              asset_gain
          else: 0
  - name: short_term_capital_gains
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: sum(capital_asset_of_tax_unit.short_term_capital_gain)
"""

    issues = find_relation_aggregate_syntax_issues(content)

    assert any("local executable output" in issue for issue in issues)
    assert "sum(capital_asset_of_tax_unit.short_term_capital_gain)" in issues[0]
    assert "sum_where(capital_asset_of_tax_unit, short_term_capital_gain" in issues[0]


def test_relation_aggregate_syntax_accepts_sum_of_relation_row_fact():
    content = """format: rulespec/v1
rules:
  - name: capital_asset_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: capital_asset_of_tax_unit
      arity: 2
  - name: short_term_capital_gains
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: sum(capital_asset_of_tax_unit.asset_gain)
"""

    assert find_relation_aggregate_syntax_issues(content) == []


def test_role_limited_relation_scope_rejects_broad_container_count():
    content = """format: rulespec/v1
module:
  summary: |-
    The additional standard deduction is the sum of each additional amount to
    which the taxpayer is entitled. If the taxpayer is married and files a
    joint return, the spouse may also be entitled to the additional amount.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: additional_condition_count
    kind: derived
    entity: TaxUnit
    dtype: Count
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          count_where(member_of_tax_unit, is_aged_65_or_over) + count_where(member_of_tax_unit, is_blind)
"""

    issues = find_role_limited_relation_scope_issues(content)

    assert any("Role-limited relation scope" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]
    assert "taxpayer" in issues[0]


def test_role_limited_relation_scope_accepts_source_stated_household_members():
    content = """format: rulespec/v1
module:
  summary: |-
    Each household member who is elderly counts toward the household allowance.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: elderly_member_count
    kind: derived
    entity: Household
    dtype: Count
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: count_where(member_of_household, is_elderly)
"""

    assert find_role_limited_relation_scope_issues(content) == []


def test_entity_limited_aggregation_order_rejects_cap_after_relation_sum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_tax_unit.covered_wages), annual_base - wages_already_paid_to_employee)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "tax_unit_covered_wages" in issues[0]
    assert "member_of_tax_unit" in issues[0]
    assert "employee" in issues[0]


def test_entity_limited_aggregation_order_rejects_limit_on_aggregate_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: raw_tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum(member_of_tax_unit.covered_wages)
  - name: tax_unit_covered_wages_after_base_limit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(raw_tax_unit_covered_wages, annual_base - wages_already_paid_to_employee)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert any("tax_unit_covered_wages_after_base_limit" in issue for issue in issues)
    assert any("member_of_tax_unit" in issue for issue in issues)


def test_entity_limited_aggregation_order_accepts_per_entity_limited_sum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, annual_base - wages_already_paid_to_employee)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_reversed_per_entity_minimum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(annual_base - wages_already_paid_to_employee, covered_wages)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_missing_entity_cap_before_sum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum(member_of_tax_unit.covered_wages)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_sum_where_with_spaced_comma():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit , covered_wages, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_unrelated_helper_minimum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, unrelated_program_cap)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_lesser_of_amount_with_unrelated_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages are limited to the lesser of covered wages
    and annual base.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, unrelated_program_cap)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_generic_lesser_of_subject_with_unrelated_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, the employee benefit is limited to the lesser of covered
    wages and annual base.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, unrelated_program_cap)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_accepts_generic_lesser_of_entity_limit():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, the employee benefit is limited to the lesser of covered
    wages and annual base.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, annual_base)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_cap_applied_to_wrong_amount():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(other_income, annual_base)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_accepts_predicate_factored_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_exceed_base
    kind: derived
    entity: Person
    dtype: Boolean
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: covered_wages > annual_base
  - name: employee_covered_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if employee_wages_exceed_base:
              annual_base
          else:
              covered_wages
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum_where(member_of_tax_unit, employee_covered_wages, employee_counts_for_tax_unit), tax_unit_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_predicate_without_capping_branch():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_above_base_kept
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if covered_wages > annual_base:
              covered_wages
          else:
              0
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_above_base_kept, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_unrelated_predicate_with_cap_branch():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if covered_wages > unrelated_program_cap:
              annual_base
          else:
              covered_wages
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_swapped_conditional_branches():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if covered_wages > annual_base:
              covered_wages
          else:
              annual_base
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_accepts_only_if_sum_where_predicate():
    content = """format: rulespec/v1
module:
  summary: |-
    For each child, the allowance applies only if the child is eligible.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_child_allowance
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_household, child_allowance, child_is_eligible)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_only_if_without_predicate():
    content = """format: rulespec/v1
module:
  summary: |-
    For each child, the allowance applies only if the child is eligible.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_child_allowance
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum(member_of_household.child_allowance)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_household" in issues[0]


def test_entity_limited_aggregation_order_accepts_standalone_per_entity_reduction():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages are reduced by excluded wages.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_covered_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: covered_wages - excluded_wages
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum_where(member_of_tax_unit, employee_covered_wages, employee_counts_for_tax_unit)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_semantic_limited_helper_name():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
    The tax unit amount shall not exceed the tax unit maximum.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_covered_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, annual_base - wages_already_paid_to_employee)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum_where(member_of_tax_unit, employee_covered_wages, employee_counts_for_tax_unit), tax_unit_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_source_stated_unit_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    Each household member has a monthly allowance. The household benefit shall
    not exceed the household maximum.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_allowance), household_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_unit_cap_with_member_context():
    content = """format: rulespec/v1
module:
  summary: |-
    The household benefit shall not exceed the maximum allotment for a
    household of the same size, based on the number of household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_allowance), household_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_unit_cap_with_member_condition():
    content = """format: rulespec/v1
module:
  summary: |-
    The household benefit for a household with an elderly member shall not
    exceed the household maximum.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_allowance), household_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_unit_cap_with_prefixed_amount():
    content = """format: rulespec/v1
module:
  summary: |-
    The maximum allotment for a household with an elderly member shall not
    exceed the household maximum.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_allotment), household_maximum)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_accepts_unrelated_limit_and_sum():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the employee cap.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_and_other_tax_unit_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(covered_wages, employee_cap) + sum(member_of_tax_unit.other_income)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_mixed_entity_and_unit_caps():
    content = """format: rulespec/v1
module:
  summary: |-
    Each household member amount shall not exceed the member maximum. The
    household benefit shall not exceed the household maximum.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_amount), household_maximum)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_household" in issues[0]


def test_entity_limited_aggregation_order_rejects_same_sentence_mixed_caps():
    content = """format: rulespec/v1
module:
  summary: |-
    The household benefit shall not exceed the household maximum, and each
    household member amount shall not exceed the member maximum.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      predicate: member_of_household
      arity: 2
  - name: household_benefit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_household.member_amount), household_maximum)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_household" in issues[0]


def test_entity_limited_aggregation_order_accepts_cap_side_unrelated_aggregate():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_annual_base_adjustment
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: sum(member_of_tax_unit.annual_base_adjustment)
"""

    assert find_entity_limited_aggregation_order_issues(content) == []


def test_entity_limited_aggregation_order_rejects_conditional_aggregate_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if sum(member_of_tax_unit.covered_wages) > annual_base:
              annual_base
          else:
              sum(member_of_tax_unit.covered_wages)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_adjacent_such_amount_cap():
    content = """format: rulespec/v1
module:
  summary: |-
    The amount for each employee is covered wages. Such amount shall not exceed
    the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum(member_of_tax_unit.covered_wages), annual_base)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "employee" in issues[0]


def test_entity_limited_aggregation_order_rejects_misleading_limited_helper_name():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: covered_wages
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit), annual_base)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_identifier_only_limited_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: covered_wages + annual_base_adjustment
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit), annual_base)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_entity_limited_aggregation_order_rejects_floor_only_limited_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    For each employee, covered wages taken into account for the employee shall
    not exceed the annual base reduced by wages already paid to the employee.
rules:
  - name: member_of_tax_unit
    kind: data_relation
    data_relation:
      predicate: member_of_tax_unit
      arity: 2
  - name: employee_wages_after_annual_base_limit
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, covered_wages)
  - name: tax_unit_covered_wages
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: min(sum_where(member_of_tax_unit, employee_wages_after_annual_base_limit, employee_counts_for_tax_unit), annual_base)
"""

    issues = find_entity_limited_aggregation_order_issues(content)

    assert any("Entity-limited aggregation order" in issue for issue in issues)
    assert "member_of_tax_unit" in issues[0]


def test_source_limitation_application_rejects_final_amount_without_limit():
    content = """format: rulespec/v1
module:
  summary: |-
    The standard deduction means the sum of the basic standard deduction and
    the additional standard deduction. Limitation on basic standard deduction
    in the case of certain dependents: the basic standard deduction shall not
    exceed the greater of $500 or earned income plus $250.
rules:
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: basic_standard_deduction_amount + additional_standard_deduction_amount
"""

    issues = find_source_limitation_application_issues(content)

    assert any("Source limitation not applied" in issue for issue in issues)
    assert "standard_deduction" in issues[0]


def test_source_limitation_application_ignores_judgment_predicates():
    content = """format: rulespec/v1
module:
  summary: |-
    A deduction is allowed subject to a limitation. An applicable taxpayer
    means a taxpayer whose active qualified business income is at least $1,000.
rules:
  - name: applicable_taxpayer_for_minimum_active_qbi_deduction
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    source: example
    versions:
      - effective_from: '2026-01-01'
        formula: active_qbi >= active_qbi_threshold
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_limitation_application_accepts_final_amount_with_limit_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    The standard deduction means the sum of the basic standard deduction and
    the additional standard deduction. Limitation on basic standard deduction
    in the case of certain dependents: the basic standard deduction shall not
    exceed the greater of $500 or earned income plus $250.
rules:
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: dependent_limited_basic_standard_deduction + additional_standard_deduction_amount
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_limitation_application_accepts_indirect_limited_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    The standard deduction means the sum of the basic standard deduction and
    the additional standard deduction. Limitation on basic standard deduction
    in the case of certain dependents: the basic standard deduction shall not
    exceed the greater of $500 or earned income plus $250.
rules:
  - name: basic_standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if deduction_under_section_151_allowable_to_another_taxpayer:
              dependent_standard_deduction
          else:
              basic_standard_deduction_amount
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: basic_standard_deduction + additional_standard_deduction_amount
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_limitation_application_accepts_transitive_limited_helper():
    content = """format: rulespec/v1
module:
  summary: |-
    A credit equals 15 percent of the section 22 amount. The section 22 amount
    is reduced by pension benefits and by the adjusted gross income limitation.
rules:
  - name: section_22_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, amount_after_benefit_reduction - agi_phaseout_reduction)
  - name: elderly_disabled_credit_potential
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: section_22_amount * credit_rate
  - name: elderly_disabled_credit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if eligible:
              elderly_disabled_credit_potential
          else:
              0
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_limitation_application_scopes_subsection_special_rule():
    content = """format: rulespec/v1
module:
  summary: |-
    (a) Assessment and collection after limitation period. The term
    overpayment includes the payment assessed after the expiration of the
    period of limitation.

    (b) Excessive credits (1) In general If refundable credits exceed the tax
    imposed by subtitle A, reduced by nonrefundable credits, the excess is an
    overpayment. (2) Special rule for credit under section 33 The credit is
    treated as refundable only if a section 6013 election is in effect. The
    preceding sentence shall not apply to a credit allowed by reason of section
    1446.
rules:
  - name: excessive_refundable_credits_overpayment
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 6401(b)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, refundable_credits - tax_reduced_by_nonrefundable_credits)
  - name: section_33_credit_treated_as_refundable_credit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 6401(b)(2)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if section_6013_election_in_effect or credit_allowed_by_reason_of_section_1446:
              section_33_credit_allowed
          else:
              0
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_limitation_application_scopes_lowercase_subsection_before_uppercase_subparagraph():
    content = """format: rulespec/v1
module:
  summary: |-
    (a) In general The tax equals 3.8 percent of the lesser of (A) net
    investment income or (B) the excess of modified adjusted gross income over
    the threshold amount.

    (b) Threshold amount The term threshold amount means (1) $250,000 for a
    joint return, (2) one-half of that amount for a separate return, and (3)
    $200,000 in any other case.

    (c) Net investment income The term net investment income means the excess
    of gross investment income over allocable deductions.
rules:
  - name: niit_threshold_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 1411(b)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          if filing_status == 1:
              niit_threshold_joint
          else:
              niit_threshold_other
"""

    assert find_source_limitation_application_issues(content) == []


def test_source_verification_accepts_decimal_rate_values_as_word_percentages():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/25A
    values:
      aotc_refundable_rate: 0.40
rules:
  - name: aotc_refundable_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: '0.40'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/statute/26/25A": (
                "Forty percent of so much of the credit shall be treated as refundable."
            )
        },
    )

    assert issues == []


def test_source_verification_accepts_fractional_decimal_rate_values_as_word_percentages():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1411
    values:
      niit_rate: 0.038
rules:
  - name: niit_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: '0.038'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/statute/26/1411": (
                "There is hereby imposed a tax equal to 3.8 percent of the lesser of "
                "net investment income or the excess modified adjusted gross income."
            )
        },
    )

    assert issues == []


def test_source_verification_accepts_decimal_rate_values_as_hyphenated_percentages():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
    values:
      capital_gains_twenty_percent_rate: 0.20
rules:
  - name: capital_gains_twenty_percent_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: '0.20'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/statute/26/1": (
                "The amount of tax shall be increased by 20-percent of the "
                "adjusted net capital gain above the applicable threshold."
            )
        },
    )

    assert issues == []


def test_numeric_grounding_accepts_decimal_rate_values_as_hyphenated_percentages():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1
rules:
  - name: capital_gains_twenty_percent_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: '0.20'
"""

    issues = find_ungrounded_numeric_issues(
        content,
        source_text="The tax is increased by 20-percent of the applicable amount.",
    )

    assert issues == []


def test_source_verification_accepts_transposed_table_values():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/usda/fns/snap-fy2026-cola/page-2
    values:
      snap_standard_deduction_48_states_dc_table:
        1: 209
        2: 209
        3: 209
        4: 223
        5: 261
        6: 299
rules:
  - name: snap_standard_deduction_48_states_dc_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 209
          2: 209
          3: 209
          4: 223
          5: 261
          6: 299
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/usda/fns/snap-fy2026-cola/page-2": (
                "Deductions Household Size 1 2 3 4 5 6+ "
                "48 States & District of Columbia "
                "$209 $209 $209 $223 $261 $299"
            )
        },
    )

    assert issues == []


def test_source_verification_accepts_bare_percentage_table_values():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/example/rates
    values:
      eitc_phase_in_rates:
        0: 0.0765
        1: 0.34
rules:
  - name: eitc_phase_in_rates
    kind: parameter
    dtype: Rate
    indexed_by: qualifying_child_count
    versions:
      - effective_from: '2026-01-01'
        values:
          0: 0.0765
          1: 0.34
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/statute/example/rates": (
                "The credit percentage is determined as follows: "
                "1 qualifying child 34; no qualifying children 7.65."
            )
        },
    )

    assert issues == []


def test_source_condition_coverage_ignores_credit_allowed_paid_tax_language():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/24
rules:
  - name: ctc_refundable_foreign_income_eligible
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: not excludes_foreign_earned_income
"""

    issues = find_source_condition_coverage_issues(
        content,
        source_texts={
            "us/statute/26/24": (
                "The aggregate credits allowed to a taxpayer shall be increased "
                "by the lesser of the credit which would be allowed under this "
                "section or social security taxes paid during the taxable year. "
                "Paragraph (1) shall not apply if the taxpayer elects to exclude "
                "foreign earned income."
            )
        },
    )

    assert issues == []


def test_source_verification_accepts_multiple_corpus_source_paths():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_paths:
      - us/guidance/example/page-1
      - us/guidance/example/page-2
    values:
      page_one_amount: 100
      page_two_amount: 200
rules:
  - name: page_one_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '100'
  - name: page_two_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '200'
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/example/page-1": "Page 1 source states $100.",
            "us/guidance/example/page-2": "Page 2 source states $200.",
        },
    )

    assert issues == []


def test_source_verification_rejects_source_value_mismatch():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/usda/fns/snap-fy2026-cola/page-1
    values:
      snap_maximum_allotment_table:
        1: 298
        2: 546
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/usda/fns/snap-fy2026-cola/page-1": (
                "Household Size 48 States & District of Columbia 1 $298 2 $545"
            )
        },
    )

    assert any("Source verification value missing" in issue for issue in issues)
    assert any("snap_maximum_allotment_table[2]" in issue for issue in issues)


def test_source_verification_rejects_rulespec_value_mismatch():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/usda/fns/snap-fy2026-cola/page-1
    values:
      snap_maximum_allotment_table:
        1: 298
        2: 546
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 545
"""

    issues = find_source_verification_issues(
        content,
        source_texts={
            "us/guidance/usda/fns/snap-fy2026-cola/page-1": (
                "Household Size 48 States & District of Columbia 1 $298 2 $546"
            )
        },
    )

    assert any("Source verification RuleSpec mismatch" in issue for issue in issues)
    assert any("snap_maximum_allotment_table[2]" in issue for issue in issues)


def test_rulespec_ci_accepts_source_relation_without_tests(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: maximum_allotment_restatement
    kind: source_relation
    source: 10 CCR 2506-1 section 4.207.3(D)
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_verifies_source_relation_values_against_target(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    us_root = tmp_path / "rulespec-us"
    target_file = us_root / "policies/usda/snap/fy-2026-cola.yaml"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: snap_maximum_allotment_additional_member
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '218'
  - name: snap_maximum_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment_table[household_size]
"""
    )

    co_root = tmp_path / "rulespec-us-co"
    rules_file = co_root / "regulations/10-ccr-2506-1/4.207.3.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: maximum_allotment_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
    verification:
      values:
        snap_maximum_allotment_table:
          1: 298
          2: 546
        snap_maximum_allotment_additional_member: 218
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=co_root,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_rejects_source_relation_value_mismatch(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    us_root = tmp_path / "rulespec-us"
    target_file = us_root / "policies/usda/snap/fy-2026-cola.yaml"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_maximum_allotment_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 298
          2: 546
  - name: snap_maximum_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment_table[household_size]
"""
    )

    co_root = tmp_path / "rulespec-us-co"
    rules_file = co_root / "regulations/10-ccr-2506-1/4.207.3.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: maximum_allotment_restatement
    kind: source_relation
    source_relation:
      type: restates
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
    verification:
      values:
        snap_maximum_allotment_table:
          1: 298
          2: 545
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=co_root,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "Source relation verification mismatch" in issue
        and "snap_maximum_allotment_table[2]" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_source_relation_without_target(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: maximum_allotment_restatement
    kind: source_relation
    source: 10 CCR 2506-1 section 4.207.3(D)
    source_relation:
      type: restates
      authority: federal
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any("Source relation target required" in issue for issue in result.issues)


def test_rulespec_ci_rejects_scalar_kind_mismatches(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The code is 1 and the count is 1.
rules:
  - name: code_text
    kind: derived
    entity: Case
    dtype: Text
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: '"1"'
  - name: count
    kind: derived
    entity: Case
    dtype: Integer
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: '1'
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: kind_mismatch
  period: 2024-01
  input: {}
  output:
    code_text: 1
    count: "1"
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "code_text" in issue and "expected integer 1, got text 1" in issue
        for issue in result.issues
    )
    assert any(
        "count" in issue and "expected text 1, got integer 1" in issue
        for issue in result.issues
    )


def test_rulespec_ci_accepts_holds_for_boolean_scalar_outputs(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The formula applies.
rules:
  - name: formula_applies
    kind: parameter
    dtype: Judgment
    versions:
      - effective_from: '2026-01-01'
        formula: 'true'
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: formula_applies
  period:
    period_kind: custom
    name: calendar_year
    start: '2026-01-01'
    end: '2026-12-31'
  input: {}
  output:
    formula_applies: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is True


def test_rulespec_ci_rejects_malformed_period_mapping(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The flag is true.
rules:
  - name: flag
    kind: derived
    entity: Case
    dtype: Bool
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: 'true'
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: missing_dates
  period:
    period_kind: month
  input: {}
  output:
    flag: true
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "period mapping missing required field(s)" in issue for issue in result.issues
    )


def test_rulespec_ci_rejects_bare_year_periods(tmp_path):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The flag is true.
rules:
  - name: flag
    kind: derived
    entity: Case
    dtype: Bool
    period: Month
    versions:
      - effective_from: '2024-01-01'
        formula: 'true'
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: ambiguous_year
  period: 2024
  input: {}
  output:
    flag: true
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )
    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any("bare year periods are ambiguous" in issue for issue in result.issues)


def test_rulespec_ci_rejects_ungrounded_generated_numeric_literal(
    tmp_path, monkeypatch
):
    if not AXIOM_RULES_ENGINE_BINARY.exists():
        pytest.skip("local axiom-rules-engine binary is not built")

    _mock_corpus_source_text(
        monkeypatch,
        "The official source states the standard utility allowance is $451.",
    )

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          452
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: base
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance_value: 452
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "Ungrounded generated numeric literal" in issue and "452" in issue
        for issue in result.issues
    )


def test_ungrounded_numeric_accepts_source_unicode_fraction():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/1411
rules:
  - name: married_separate_threshold_fraction
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          0.5
"""

    assert (
        find_ungrounded_numeric_issues(
            content,
            source_text="The amount is ½ of the dollar amount determined elsewhere.",
        )
        == []
    )
    assert 0.5 in extract_numeric_occurrences_from_text("The amount is ½.")


def test_ungrounded_numeric_accepts_source_ordinal_word():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3510
rules:
  - name: return_filing_deadline_months_after_employer_taxable_year_close
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          4
"""
    source_text = (
        "The return shall be filed on or before the 15th day of the "
        "fourth month following the close of the employer's taxable year."
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert 4 in extract_numeric_occurrences_from_text(source_text)


def test_ungrounded_numeric_preserves_substantive_parenthetical_days():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3406
rules:
  - name: broker_notice_deadline_days
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '1990-01-01'
        formula: |-
          15
"""
    source_text = (
        "such broker shall, within such period as the Secretary may prescribe by "
        "regulations (but not later than 15 days after such acquisition), notify "
        "the payor"
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert 15 in extract_numeric_occurrences_from_text(source_text)


def test_ungrounded_numeric_accepts_source_mixed_unicode_fraction_percentage():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3302
rules:
  - name: trade_act_agreement_noncompliance_reduction_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          0.075
"""

    source_text = "The total credits shall be reduced by 7½ percent of the tax."

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert 0.075 in extract_numeric_occurrences_from_text(source_text)


def test_ungrounded_numeric_accepts_spelled_hundredths_percentage():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/statute/39/39-22-104/1.7/c
rules:
  - name: individual_estate_trust_income_tax_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2022-01-01'
        formula: |-
          0.044
"""

    source_text = (
        "a tax of four and forty one-hundredths percent is imposed on the "
        "federal taxable income"
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert any(
        math.isclose(value, 0.044) for value in extract_numbers_from_text(source_text)
    )
    assert any(
        math.isclose(value, 0.044)
        for value in extract_numeric_occurrences_from_text(source_text)
    )


def test_ungrounded_numeric_accepts_spelled_fractional_percentage():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us-co/statute/39/39-22-104/1.5
rules:
  - name: individual_estate_trust_income_tax_rate_1999
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '1999-01-01'
        formula: |-
          0.0475
"""

    source_text = (
        "a tax of four and three-quarters percent is imposed on the "
        "federal taxable income"
    )

    assert find_ungrounded_numeric_issues(content, source_text=source_text) == []
    assert any(
        math.isclose(value, 0.0475) for value in extract_numbers_from_text(source_text)
    )
    assert any(
        math.isclose(value, 0.0475)
        for value in extract_numeric_occurrences_from_text(source_text)
    )


def test_non_rulespec_yaml_artifact_is_rejected(tmp_path):
    rules_file = tmp_path / "not-rulespec.yaml"
    rules_file.write_text("rules:\n  - name: missing_format_header\n")
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_compile_check(rules_file)

    assert result.passed is False
    assert "RuleSpec YAML artifacts are required" in result.issues[0]


def test_current_purpose_placeholder_input_is_rejected():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3231
rules:
  - name: remaining_applicable_base_before_payment
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, applicable_base_for_current_purpose - compensation_paid_before_payment)
"""

    issues = find_current_purpose_placeholder_issues(content)

    assert len(issues) == 1
    assert "applicable_base_for_current_purpose" in issues[0]
    assert "Current-purpose placeholder input" in issues[0]


def test_deferred_purpose_specific_limitation_rejects_generic_output():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3231
  deferred_outputs:
    - output: us:statutes/26/3231/e/2#compensation_excess_base_exclusion_for_section_3201_a_hospital_insurance_rate_portion
      reason: Clause (iii) provides that the clause (i) base exclusion shall not apply to the hospital-insurance rate portion.
rules:
  - name: compensation_excess_applicable_base_excluded
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, remuneration_paid - remaining_applicable_base_before_payment)
  - name: compensation_excess_base_exclusion_for_section_3201_a_non_hospital_insurance_rate_portion
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: max(0, remuneration_paid - remaining_applicable_base_before_payment)
"""

    issues = find_deferred_purpose_specific_limitation_issues(content)

    assert len(issues) == 1
    assert "Generic output with deferred purpose-specific limitation" in issues[0]
    assert "compensation_excess_applicable_base_excluded" in issues[0]


def test_deferred_purpose_specific_limitation_allows_named_tier_output():
    content = """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/statute/26/3231
  deferred_outputs:
    - output: us:statutes/26/3231/e/2#applicable_base_for_tier_2_taxes_and_average_monthly_compensation
      reason: Clause (ii) defines a purpose-specific base for tier 2 taxes, but section 230(c) mechanics are not yet executable.
rules:
  - name: applicable_base_for_tier_1_taxes
    kind: derived
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: contribution_and_benefit_base_for_calendar_year
"""

    assert find_deferred_purpose_specific_limitation_issues(content) == []

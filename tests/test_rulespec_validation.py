import json
import os
from pathlib import Path

import pytest

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
    _normalize_us_tax_filing_status,
    _policyengine_expected_float,
    _policyengine_period_string,
    _policyengine_us_snap_input_aliases,
    _tax_unit_member_aged_flags,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numeric_occurrences_from_text,
    find_aggregate_exception_predicate_issues,
    find_broad_application_passthrough_issues,
    find_child_fragment_reencoding_issues,
    find_copied_cross_reference_source_issues,
    find_deprecated_source_url_issues,
    find_exception_test_coverage_issues,
    find_formula_absolute_reference_issues,
    find_missing_derived_companion_output_issues,
    find_missing_same_section_subsection_import_issues,
    find_relation_aggregate_syntax_issues,
    find_role_limited_relation_scope_issues,
    find_rule_name_path_suffix_issues,
    find_rule_source_metadata_issues,
    find_sibling_rule_name_collision_issues,
    find_source_claim_reference_issues,
    find_source_condition_coverage_issues,
    find_source_limitation_application_issues,
    find_source_verification_issues,
    find_test_input_assignment_issues,
    find_ungrounded_numeric_issues,
    find_upstream_placement_issues,
    find_versioned_derived_formula_issues,
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
  - name: snap_standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
  - name: snap_standard_utility_allowance
    kind: derived
    entity: SnapUnit
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          snap_standard_utility_allowance_value
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: base
  period: 2024-01
  input: {}
  output:
    snap_standard_utility_allowance: 451
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
    ] == [("snap_standard_utility_allowance_value", 451.0)]


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
    employee_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3101/b/1#hospital_insurance_wage_tax_rate",
        country="us",
    )
    assert employee_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        employee_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.medicare.rate.employee"
    )
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
            "us:statutes/26/63/c#standard_deduction",
            country="us",
        ).policyengine_variable
        == "standard_deduction"
    )
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


def test_cross_reference_exception_placeholder_allows_covering_import(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = repo / "statutes" / "7" / "2014" / "a.yaml"
    rules_file.parent.mkdir(parents=True)
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

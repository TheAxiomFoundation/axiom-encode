import json
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
    _policyengine_period_string,
    _policyengine_us_snap_input_aliases,
    _tax_unit_member_aged_flags,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    find_deprecated_source_url_issues,
    find_source_claim_reference_issues,
    find_source_verification_issues,
    find_ungrounded_numeric_issues,
    find_upstream_placement_issues,
)
from axiom_encode.oracles.policyengine.registry import (
    PolicyEngineMapping,
    PolicyEngineOracleRegistry,
    load_policyengine_registry,
)

AXIOM_RULES_PATH = Path("/Users/maxghenis/TheAxiomFoundation/axiom-rules")
AXIOM_RULES_BINARY = AXIOM_RULES_PATH / "target" / "debug" / "axiom-rules"


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
    rules_repo = repo_parent / "rules-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")

    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules",
        enable_oracles=False,
    )

    issues = pipeline._check_promoted_stub_file(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


def test_imported_stub_dependency_check_uses_corpus_provisions(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rules-us"
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
        axiom_rules_path=repo_parent / "axiom-rules",
        enable_oracles=False,
    )

    issues = pipeline._check_imported_stub_dependencies(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


def test_rulespec_compile_ci_and_grounding(tmp_path, monkeypatch):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "must use legal RuleSpec id" in issue
        and "us:statutes/7/2017/a#snap_regular_month_allotment" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_unresolved_output_reference_path(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/9999/a#snap_regular_month_allotment` points to a RuleSpec file that could not be resolved"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_input_reference_in_output_position(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/2017/a#input.household_allotment_input` resolves to an input slot, which is not allowed here"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_friendly_input_keys(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_input_reference_path(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2017" / "a.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2012" / "j.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2012" / "j.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_relation_reference(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules-us" / "statutes" / "7" / "2012" / "j.yaml"
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
        policy_repo_path=tmp_path / "rules-us",
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
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1401#self_employment_tax",
            country="us",
        ).mapping_type
        == "not_comparable"
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
    rules_file = tmp_path / "rules-us-co" / "regulations" / "rules.yaml"
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
            "work_supplementation_earned_income": 250,
            "snap_monthly_household_income": 1200,
            "dependent_care_deduction": 50,
            "child_support_deduction": 25,
            "medical_deduction": 10,
            "excess_shelter_deduction": 100,
        }
    )

    assert aliases == {
        "snap_earned_income": 750.0,
        "snap_gross_income": 1200.0,
        "snap_dependent_care_deduction": 50.0,
        "snap_child_support_deduction": 25.0,
        "snap_excess_medical_expense_deduction": 10.0,
        "snap_excess_shelter_expense_deduction": 100.0,
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


def _write_rulespec_file(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    validator_pipeline._rulespec_executable_index_for_roots.cache_clear()
    return path


def test_upstream_placement_flags_duplicate_upstream_executable_rule(tmp_path):
    repo_parent = tmp_path / "repos"
    _write_rulespec_file(
        repo_parent / "rules-us" / "policies/example/fy-2026.yaml",
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
        repo_parent / "rules-us-co" / "regulations/example/benefit.yaml",
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
        repo_parent / "rules-us" / "policies/example/fy-2026.yaml",
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
        repo_parent / "rules-us-co" / "regulations/example/benefit.yaml",
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
        repo_parent / "rules-us" / "policies/example/fy-2026.yaml",
        canonical_content,
    )
    _write_rulespec_file(
        repo_parent
        / "rules-us"
        / "_axiom"
        / "rules-us"
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
        repo_parent / "rules-us-ny" / "regulations/example/benefit.yaml",
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
        repo_parent / "rules-us-co" / "regulations/example/benefit.yaml",
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


def test_rulespec_ci_executes_companion_test_outputs(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The standard utility allowance is $451.
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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


def test_rulespec_ci_rejects_scale_tables_encoded_as_match_literals(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    us_root = tmp_path / "rules-us"
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

    co_root = tmp_path / "rules-us-co"
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    us_root = tmp_path / "rules-us"
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

    co_root = tmp_path / "rules-us-co"
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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

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

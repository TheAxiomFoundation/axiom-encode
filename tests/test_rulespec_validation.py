from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
)

AXIOM_RULES_PATH = Path("/Users/maxghenis/TheAxiomFoundation/axiom-rules")
AXIOM_RULES_BINARY = AXIOM_RULES_PATH / "target" / "debug" / "axiom-rules"


def test_rulespec_compile_ci_and_grounding(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
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


def test_rulespec_ci_compares_parameter_only_outputs(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: The policy rate is 0.2.
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


def test_rulespec_ci_rejects_ungrounded_generated_numeric_literal(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
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

from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    _extract_json_object,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    find_source_verification_issues,
    find_ungrounded_numeric_issues,
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

    assert find_ungrounded_numeric_issues(content) == []


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

    assert find_ungrounded_numeric_issues(content) == []


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


def test_rulespec_ci_executes_indexed_parameter_table_lookup(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The maximum monthly allotments are 298 and 546 for household sizes 1 and 2,
    plus 218 for each additional person.
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
              snap_maximum_allotment_table[2] + ((household_size - 2) * snap_maximum_allotment_additional_member)
          else: snap_maximum_allotment_table[household_size]
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


def test_rulespec_ci_accepts_reiteration_without_tests(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: co_snap_maximum_allotment_reiterates_usda_fy_2026
    kind: reiteration
    source: 10 CCR 2506-1 section 4.207.3(D)
    source_url: https://example.test/co-snap
    reiterates:
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
      relationship: restates
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    assert pipeline._run_ci(rules_file).passed is True


def test_rulespec_ci_verifies_reiteration_values_against_target(tmp_path):
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
  - name: co_snap_maximum_allotment_reiterates_usda_fy_2026
    kind: reiteration
    reiterates:
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
      relationship: restates
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


def test_rulespec_ci_rejects_reiteration_value_mismatch(tmp_path):
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
  - name: co_snap_maximum_allotment_reiterates_usda_fy_2026
    kind: reiteration
    reiterates:
      target: us:policies/usda/snap/fy-2026-cola#snap_maximum_allotment
      authority: federal
      relationship: restates
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
        "Reiteration verification mismatch" in issue
        and "snap_maximum_allotment_table[2]" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_reiteration_without_target(tmp_path):
    if not AXIOM_RULES_BINARY.exists():
        pytest.skip("local axiom-rules binary is not built")

    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: co_snap_maximum_allotment_reiterates_usda_fy_2026
    kind: reiteration
    source: 10 CCR 2506-1 section 4.207.3(D)
    reiterates:
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
    assert any("Reiteration target required" in issue for issue in result.issues)


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

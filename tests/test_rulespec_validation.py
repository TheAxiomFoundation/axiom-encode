from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import (
    OracleSubprocessResult,
    ValidatorPipeline,
    _extract_json_object,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    find_source_verification_issues,
    find_ungrounded_numeric_issues,
    find_upstream_placement_issues,
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


def test_policyengine_oracle_does_not_score_unmapped_outputs(tmp_path):
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: mapped
  period: 2026-01
  input: {}
  output:
    mapped_var: 10
- name: unmapped
  period: 2026-01
  input: {}
  output:
    unmapped_var: 99
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
    pipeline._resolve_pe_variable = lambda _country, name: (
        "pe_mapped_var" if name == "mapped_var" else None
    )
    pipeline._build_pe_scenario_script = lambda *_args, **_kwargs: ""
    pipeline._run_pe_subprocess_detailed = lambda *_args, **_kwargs: (
        OracleSubprocessResult(returncode=0, stdout="RESULT:10\n")
    )

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert len(result.issues) == 1
    assert "no PE mapping" in result.issues[0]
    assert "unmapped_var" in result.issues[0]


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


def test_upstream_placement_flags_snap_elderly_disabled_definition_in_cola():
    content = """format: rulespec/v1
relations:
  - name: member_of_household
    arity: 2
rules:
  - name: snap_household_has_usda_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_usda_elderly_or_disabled) > 0
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("policies/usda/snap/fy-2026-cola/deductions.yaml"),
    )

    assert len(issues) == 1
    assert "7 USC 2012(j)" in issues[0]
    assert "us:statutes/7/2012/j" in issues[0]


def test_upstream_placement_allows_snap_elderly_disabled_import_in_cola():
    content = """format: rulespec/v1
imports:
  - us:statutes/7/2012/j
rules:
  - name: snap_asset_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_household_has_elderly_or_disabled_member:
              snap_asset_limit_elderly_or_disabled_member
          else: snap_asset_limit_other_households
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("policies/usda/snap/fy-2026-cola/deductions.yaml"),
        )
        == []
    )


def test_upstream_placement_requires_snap_elderly_disabled_import():
    content = """format: rulespec/v1
rules:
  - name: snap_resource_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_household_has_elderly_or_disabled_member:
              snap_asset_limit_elderly_or_disabled_member
          else: snap_asset_limit_other_households
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.408.yaml"),
    )

    assert any("Upstream import required" in issue for issue in issues)
    assert any("us:statutes/7/2012/j" in issue for issue in issues)


def test_upstream_placement_allows_canonical_snap_elderly_disabled_definition():
    content = """format: rulespec/v1
relations:
  - name: member_of_household
    arity: 2
rules:
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2008-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("statutes/7/2012/j.yaml"),
        )
        == []
    )


def test_upstream_placement_flags_state_manual_snap_allotment_formula():
    content = """format: rulespec/v1
rules:
  - name: snap_household_food_contribution_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-10-01'
        formula: '0.30'
  - name: household_food_contribution
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: net_income_for_benefit_formula * snap_household_food_contribution_rate
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_eligible:
              max(snap_allotment_before_minimum, snap_minimum_monthly_allotment)
          else: 0
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
    )

    assert len(issues) == 3
    assert all("7 USC 2017(a)" in issue for issue in issues)
    assert any("snap_household_food_contribution_rate" in issue for issue in issues)
    assert any("snap_household_food_contribution" in issue for issue in issues)
    assert any("snap_regular_month_allotment" in issue for issue in issues)


def test_upstream_placement_requires_snap_allotment_formula_import():
    content = """format: rulespec/v1
rules:
  - name: snap_application_denied_for_zero_benefit
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: snap_allotment_before_minimum == 0
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
    )

    assert len(issues) == 1
    assert "Upstream import required" in issues[0]
    assert "us:statutes/7/2017/a" in issues[0]


def test_upstream_placement_flags_state_manual_snap_standard_deduction():
    content = """format: rulespec/v1
rules:
  - name: standard_deduction_table
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
  - name: standard_deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: standard_deduction_table[household_size]
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.407.1.yaml"),
    )

    assert len(issues) == 2
    assert all("USDA COLA policy file" in issue for issue in issues)
    assert any("standard_deduction_table" in issue for issue in issues)
    assert any("standard_deduction" in issue for issue in issues)


def test_upstream_placement_does_not_flag_tax_standard_deduction():
    content = """format: rulespec/v1
rules:
  - name: standard_deduction
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    source: 26 USC 63(c)
    versions:
      - effective_from: '2026-01-01'
        formula: basic_standard_deduction_amount + additional_std_ded_amount
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("statutes/26/63/c.yaml"),
        )
        == []
    )


def test_upstream_placement_flags_state_manual_snap_earned_income_deduction():
    content = """format: rulespec/v1
rules:
  - name: snap_earned_income_deduction_rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2025-10-01'
        formula: '0.20'
  - name: earned_income_deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_earned_income_subject_to_deduction * snap_earned_income_deduction_rate
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.407.2.yaml"),
    )

    assert len(issues) == 2
    assert all("7 USC 2014(e)(2)" in issue for issue in issues)
    assert any("snap_earned_income_deduction_rate" in issue for issue in issues)
    assert any("snap_earned_income_deduction" in issue for issue in issues)


def test_upstream_placement_flags_state_manual_snap_income_standards():
    content = """format: rulespec/v1
rules:
  - name: gross_income_limit_table
    kind: parameter
    dtype: Money
    unit: USD
    indexed_by: household_size
    versions:
      - effective_from: '2025-10-01'
        values:
          1: 1696
          2: 2292
  - name: net_income_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: net_income_limit_table[household_size]
  - name: snap_gross_income_limit_165_percent_fpl
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_gross_income_limit_165_percent_fpl_table[household_size]
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.401.1.yaml"),
    )

    assert len(issues) == 3
    assert all("USDA income eligibility standards" in issue for issue in issues)
    assert any("gross_income_limit_table" in issue for issue in issues)
    assert any("net_income_limit" in issue for issue in issues)
    assert any("snap_gross_income_limit_165_percent_fpl" in issue for issue in issues)


def test_upstream_placement_requires_snap_income_standards_import():
    content = """format: rulespec/v1
rules:
  - name: passes_gross_income_test
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: gross_income <= snap_gross_income_limit_130_percent_fpl_48_states_dc
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.401.yaml"),
    )

    assert len(issues) == 1
    assert "Upstream import required" in issues[0]
    assert (
        "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards" in issues[0]
    )


def test_upstream_placement_flags_friendly_snap_allotment_alias_references():
    content = """format: rulespec/v1
rules:
  - name: state_snap_uses_food_contribution
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: household_food_contribution > 0
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
    )

    assert len(issues) == 1
    assert "household_food_contribution" in issues[0]
    assert "us:statutes/7/2017/a" in issues[0]


def test_upstream_placement_allows_canonical_snap_allotment_formula():
    content = """format: rulespec/v1
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2008-10-01'
        formula: |-
          if snap_eligible:
              max(snap_allotment_before_minimum, snap_minimum_monthly_allotment)
          else: 0
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("statutes/7/2017/a.yaml"),
        )
        == []
    )


def test_upstream_placement_allows_state_manual_reiteration_of_snap_allotment_formula():
    content = """format: rulespec/v1
rules:
  - name: co_snap_regular_allotment_reiterates_7_usc_2017_a
    kind: reiteration
    reiterates:
      target: us:statutes/7/2017/a#snap_regular_month_allotment
      authority: federal
      relationship: restates
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
        )
        == []
    )


def test_upstream_placement_flags_state_manual_federal_cola_values():
    content = """format: rulespec/v1
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

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
    )

    assert len(issues) == 1
    assert "federal SNAP annual COLA value" in issues[0]
    assert "reiteration" in issues[0]


def test_upstream_placement_flags_state_manual_federal_cola_aliases():
    content = """format: rulespec/v1
rules:
  - name: excess_shelter_deduction_cap
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '744'
  - name: snap_homeless_shelter_deduction_amount
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: '198.99'
  - name: snap_resource_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          if snap_household_has_elderly_or_disabled_member:
              4500
          else: 3000
"""

    issues = find_upstream_placement_issues(
        content,
        rules_file=Path("regulations/10-ccr-2506-1/4.407.3.yaml"),
    )

    assert len(issues) == 4
    assert any("snap_maximum_excess_shelter_deduction" in issue for issue in issues)
    assert any("snap_homeless_shelter_deduction" in issue for issue in issues)
    assert any("snap_asset_limit" in issue for issue in issues)
    assert any("us:statutes/7/2012/j" in issue for issue in issues)


def test_upstream_placement_allows_state_manual_reiteration_of_federal_cola_values():
    content = """format: rulespec/v1
rules:
  - name: co_snap_maximum_allotment_reiterates_usda_fy_2026
    kind: reiteration
    reiterates:
      target: us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment
      authority: federal
      relationship: restates
    verification:
      values:
        snap_maximum_allotment_table:
          1: 298
          2: 546
"""

    assert (
        find_upstream_placement_issues(
            content,
            rules_file=Path("regulations/10-ccr-2506-1/4.207.3.yaml"),
        )
        == []
    )


def test_upstream_placement_requires_reiteration_from_source_metadata():
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
    assert "Source metadata upstream relation requires reiteration" in issues[0]
    assert "us:policies/example/fy-2026#benefit_limit" in issues[0]


def test_upstream_placement_allows_reiteration_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: local_benefit_limit_reiterates_example_policy
    kind: reiteration
    reiterates:
      target: us:policies/example/fy-2026#benefit_limit
      authority: federal
      relationship: restates
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
    assert "metadata.sets" in issues[0]
    assert "us:regulation/7-cfr/273/9/d/6/iii#standard_allowance" in issues[0]


def test_upstream_placement_allows_metadata_sets_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: local_standard_allowance
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      source_relation: sets
      sets: us:regulation/7-cfr/273/9/d/6/iii#standard_allowance
    versions:
      - effective_from: '2026-01-01'
        formula: '451'
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
    assert "metadata.amends" in issues[0]
    assert "us:statutes/7/2014/c#income_threshold" in issues[0]


def test_upstream_placement_requires_source_relation_for_metadata_target():
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
    assert "metadata.source_relation" in issues[0]
    assert "metadata.sets" in issues[0]


def test_upstream_placement_rejects_unknown_source_relation():
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
    assert "unknown source relation" in issues[0]
    assert "copies" in issues[0]


def test_upstream_placement_requires_metadata_target_for_declared_relation():
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
    assert "metadata.source_relation: amends" in issues[0]
    assert "metadata.amends" in issues[0]


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
    assert "metadata.source_relation: implements" in issues[0]
    assert "metadata.implements" in issues[0]


def test_upstream_placement_allows_metadata_implements_from_source_metadata():
    content = """format: rulespec/v1
rules:
  - name: state_mechanics
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    metadata:
      source_relation: implements
      implements:
        - target: us:statutes/7/2014/e#deduction_mechanics
    versions:
      - effective_from: '2026-01-01'
        formula: household_income
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


def test_upstream_placement_requires_concept_for_defines_relation():
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
    assert "metadata.source_relation: defines" in issues[0]
    assert "metadata.defines" in issues[0]
    assert "metadata.concept_id" in issues[0]


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

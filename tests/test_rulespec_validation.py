from pathlib import Path

import pytest

from autorac.harness.validator_pipeline import (
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
        rac_us_path=tmp_path,
        rac_path=AXIOM_RULES_PATH,
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


def test_legacy_rac_artifact_is_rejected(tmp_path):
    rac_file = tmp_path / "legacy.rac"
    rac_file.write_text("legacy_amount:\n    from 2024-01-01: 451\n")
    pipeline = ValidatorPipeline(
        rac_us_path=tmp_path,
        rac_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    result = pipeline._run_compile_check(rac_file)

    assert result.passed is False
    assert "Legacy .rac artifacts are no longer supported" in result.issues[0]

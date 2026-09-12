"""Synthetic v2 law/date boundary tests through the actual Rust CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import ValidatorPipeline

PREFIX = "us:statutes/99/1#"


def year(y):
    return {"period_kind": "tax_year", "start": f"{y}-01-01", "end": f"{y}-12-31"}


@pytest.fixture
def execution(tmp_path, monkeypatch):
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip("set AXIOM_LIFETIME_TEST_ENGINE to the real v2-capable Rust CLI")
    binary = Path(configured).resolve(strict=True)
    root = tmp_path / "rulespec-us"
    rules = root / "us/statutes/99/1.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text("""format: rulespec/v1
rules:
  - name: factor
    kind: parameter
    dtype: Integer
    versions:
      - effective_from: '2020-01-01'
        effective_to: '2021-12-31'
        formula: '2'
      - effective_from: '2026-01-01'
        formula: '3'
  - name: total
    kind: derived
    entity: Person
    dtype: Money
    versions:
      - effective_from: '2020-01-01'
        effective_to: '2021-12-31'
        formula: 'sum_over_periods(amount * factor)'
      - effective_from: '2026-01-01'
        formula: '2 * sum_over_periods(amount * factor)'
""")
    compiled = tmp_path / "compiled.json"
    result = subprocess.run(
        [
            str(binary),
            "compile",
            "--program",
            str(rules),
            "--rulespec-root",
            str(root),
            "--output",
            str(compiled),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(compiled.read_text())
    pipeline = ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=binary.parent,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)
    case = {
        "name": "later legal calculation over completed earnings",
        "period": year(2026),
        "output": {PREFIX + "total": "22.50"},
        "lifetime": {
            "entity": "Person",
            "calculation_period": year(2026),
            "periods": [year(1990), year(1991)],
            "batches": [
                {
                    "row_count": 1,
                    "entity_ids": ["001"],
                    "inputs": {
                        PREFIX + "input.amount": {"kind": "decimal", "values": [amount]}
                    },
                }
                for amount in ["1.25", "2.50"]
            ],
        },
    }

    def run():
        return pipeline._run_rulespec_test_cases(
            rules_file=rules,
            compiled_path=compiled,
            compiled_payload=payload,
            cases=[case],
        )

    return case, run


@pytest.mark.parametrize("calculation, expected", [(2021, "7.50"), (2026, "22.50")])
def test_same_history_selects_actual_formula_and_parameter_version(
    execution, calculation, expected
):
    case, run = execution
    case["period"] = year(calculation)
    case["lifetime"]["calculation_period"] = year(calculation)
    case["output"][PREFIX + "total"] = expected
    assert run() == []


def test_calculation_start_selects_law_even_when_interval_crosses_change(execution):
    case, run = execution
    calculation = {
        "period_kind": "custom",
        "name": "assessment",
        "start": "2021-01-01",
        "end": "2026-12-31",
    }
    case["period"] = calculation
    case["lifetime"]["calculation_period"] = calculation
    case["output"][PREFIX + "total"] = "7.50"
    assert run() == []


def test_missing_calculation_law_fails_without_falling_back_to_observation_law(
    execution,
):
    case, run = execution
    case["period"] = year(2022)
    case["lifetime"]["calculation_period"] = year(2022)
    issues = run()
    assert len(issues) == 1 and "2022-01-01" in issues[0] and "version" in issues[0]


def test_observation_touching_calculation_is_refused(execution):
    case, run = execution
    case["lifetime"]["periods"][-1]["end"] = "2026-01-01"
    issues = run()
    assert len(issues) == 1 and "end before calculation" in issues[0]


def test_actual_v2_output_comparison_preserves_sub_float_decimal_difference(execution):
    case, run = execution
    case["output"][PREFIX + "total"] = "22.500000000000000001"
    issues = run()
    assert len(issues) == 1 and "expected" in issues[0] and "001" in issues[0]

"""Known runtime date operations must not become phantom factual input slots."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals, validator_pipeline


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("date_add_days(base_date, days_offset)", {"base_date", "days_offset"}),
        ("date_add_months(base_date, months_offset)", {"base_date", "months_offset"}),
        ("date_add_years(base_date, years_offset)", {"base_date", "years_offset"}),
        ("days_between(start_date, end_date)", {"start_date", "end_date"}),
        ("period_start", set()),
        ("period_end", set()),
        ("unsupported_date_shift(base_date)", {"unsupported_date_shift", "base_date"}),
    ],
)
def test_cli_context_and_validation_collectors_preserve_only_actual_arguments(
    tmp_path, formula, expected
):
    rules = tmp_path / "dates.yaml"
    rules.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    {
                        "name": "result",
                        "kind": "derived",
                        "versions": [{"formula": formula}],
                    }
                ],
            }
        )
    )
    assert (
        cli._local_factual_input_names_from_rules_content(rules.read_text()) == expected
    )
    # A context-free bare period name may also be a Boolean input. The full
    # scalar rule above, and nested date-call arguments, supply exact context.
    generic_expected = expected | (
        {formula} if formula in {"period_start", "period_end"} else set()
    )
    assert cli._formula_identifiers(formula) == generic_expected
    assert (
        cli._generated_rule_formula_identifiers({"versions": [{"formula": formula}]})
        == expected
    )
    assert evals._context_file_local_inputs(str(rules)) == expected
    assert validator_pipeline._formula_local_identifiers(formula) == generic_expected


def test_actual_v2_runtime_evaluates_date_operators_inside_history_reduction(
    tmp_path, monkeypatch
):
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip("set AXIOM_LIFETIME_TEST_ENGINE to the real v2-capable Rust CLI")
    binary = Path(configured).resolve(strict=True)
    root = tmp_path / "rulespec-us"
    rules = root / "us/statutes/99/4.yaml"
    rules.parent.mkdir(parents=True)
    formula = (
        "sum_over_periods("
        "days_between(period_start, date_add_days(period_start, days_offset)) + "
        "days_between(period_start, date_add_months(period_start, months_offset)) + "
        "days_between(period_start, date_add_years(period_start, years_offset)) + "
        "days_between(period_start, period_end))"
    )
    rules.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    {
                        "name": "total",
                        "kind": "derived",
                        "entity": "Person",
                        "dtype": "Number",
                        "versions": [
                            {"effective_from": "2026-01-01", "formula": formula}
                        ],
                    }
                ],
            }
        )
    )
    facts = cli._local_factual_input_names_from_rules_content(rules.read_text())
    assert facts == {"days_offset", "months_offset", "years_offset"}
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
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=binary.parent,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)

    def year(value):
        return {
            "period_kind": "tax_year",
            "start": f"{value}-01-01",
            "end": f"{value}-12-31",
        }

    prefix = "us:statutes/99/4#"
    case = {
        "name": "observation date operators use each historical period",
        "period": year(2026),
        # 2020: 1 + 31 + 366 + 365; 2021: 1 + 31 + 365 + 364.
        "output": {prefix + "total": "1524"},
        "lifetime": {
            "entity": "Person",
            "calculation_period": year(2026),
            "periods": [year(2020), year(2021)],
            "batches": [
                {
                    "row_count": 1,
                    "entity_ids": ["synthetic-calendar"],
                    "inputs": {
                        prefix + "input." + name: {"kind": "integer", "values": [1]}
                        for name in facts
                    },
                }
                for _ in range(2)
            ],
        },
    }
    assert (
        pipeline._run_rulespec_test_cases(
            rules_file=rules,
            compiled_path=compiled,
            compiled_payload=json.loads(compiled.read_text()),
            cases=[case],
        )
        == []
    )

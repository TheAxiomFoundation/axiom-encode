"""CLI fixture repairs must distinguish lifetime operators from factual inputs."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

FORMULAS = [
    "sum_over_periods(net_value)",
    "max_over_periods(net_value)",
    "count_over_periods(is_eligible)",
    "sum_top_n_over_periods(net_value, selected_years)",
    "calendar_years_to_months(selected_years) + sum_over_periods(net_value)",
]
PREFIX = "us:statutes/99/3#"


@pytest.fixture(params=FORMULAS)
def generated_zero_case(tmp_path, request):
    root = tmp_path / "rulespec-us"
    rules = root / "us/statutes/99/3.yaml"
    rules.parent.mkdir(parents=True)
    payload = {
        "format": "rulespec/v1",
        "rules": [
            {
                "name": name,
                "kind": "derived",
                "entity": "Person",
                "dtype": dtype,
                "versions": [{"effective_from": "2000-01-01", "formula": formula}],
            }
            for name, dtype, formula in [
                ("net_value", "Decimal", "gross_amount - adjustment_amount"),
                ("selected_years", "Integer", "selection_count + 1"),
                ("history_value", "Number", request.param),
                (
                    "zero_value",
                    "Decimal",
                    "if is_eligible:\n    net_value\nelse:\n    0",
                ),
            ]
        ],
    }
    rules.write_text(yaml.safe_dump(payload, sort_keys=False))
    tests = rules.with_name("3.test.yaml")
    tests.write_text("[]\n")
    repaired = cli._append_generic_zero_branch_tests_if_missing(
        rules_file=rules,
        test_file=tests,
        repo_path=root / "us",
        relative_output=Path("statutes/99/3.yaml"),
        issues=["Zero branch test coverage missing: `zero_value` returns 0."],
    )
    assert repaired == ["auto_zero_zero_value"]
    return root, rules, yaml.safe_load(tests.read_text())[0]


def test_auto_zero_inputs_include_arguments_but_not_lifetime_functions(
    generated_zero_case,
):
    _, _, case = generated_zero_case
    assert case["input"] == {
        PREFIX + "input.gross_amount": 0,
        PREFIX + "input.adjustment_amount": 0,
        PREFIX + "input.selection_count": 0,
        PREFIX + "input.is_eligible": False,
    }


@pytest.mark.parametrize("formula", FORMULAS)
def test_nearby_cli_repair_collectors_ignore_only_known_operators(formula):
    expected = {"net_value"}
    if formula.startswith("count_over_periods"):
        expected = {"is_eligible"}
    if "sum_top_n" in formula or "calendar_" in formula:
        expected.add("selected_years")
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._generated_rule_formula_identifiers({"versions": [{"formula": formula}]})
        == expected
    )


def test_unknown_function_and_its_arguments_remain_visible():
    formula = "unrecognized_reduction(gross_amount, selection_count)"
    expected = {"unrecognized_reduction", "gross_amount", "selection_count"}
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._generated_rule_formula_identifiers({"versions": [{"formula": formula}]})
        == expected
    )
    assert (
        cli._local_factual_input_names_from_rules_content(
            yaml.safe_dump(
                {
                    "rules": [
                        {
                            "name": "total",
                            "kind": "derived",
                            "versions": [{"formula": formula}],
                        }
                    ]
                }
            )
        )
        == expected
    )


def test_real_rust_executes_generated_zero_fixture_and_lifetime_companion(
    generated_zero_case, tmp_path, monkeypatch
):
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip("set AXIOM_LIFETIME_TEST_ENGINE to the real Rust lifetime CLI")
    binary = Path(configured).resolve(strict=True)
    root, rules, case = generated_zero_case
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
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=binary.parent,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)
    payload = json.loads(compiled.read_text())
    assert (
        pipeline._run_rulespec_test_cases(
            rules_file=rules,
            compiled_path=compiled,
            compiled_payload=payload,
            cases=[case],
        )
        == []
    )

    # Reuse the generated factual inputs in an actual typed lifetime companion.
    # Values are deliberately all zero so each reduction is zero; the whole-year
    # conversion uses selected_years = selection_count + 1 and therefore yields 12.
    period = {"period_kind": "tax_year", "start": "2020-01-01", "end": "2020-12-31"}
    function = yaml.safe_load(rules.read_text())["rules"][2]["versions"][0]["formula"]
    companion = {
        "name": "generated factual inputs execute over history",
        "period": period,
        "output": {
            PREFIX + "history_value": "12" if function.startswith("calendar_") else "0"
        },
        "lifetime": {
            "entity": "Person",
            "periods": [period],
            "batches": [
                {
                    "row_count": 1,
                    "entity_ids": ["synthetic-zero"],
                    "inputs": {
                        name: {
                            "kind": "bool" if isinstance(value, bool) else "integer",
                            "values": [value],
                        }
                        for name, value in case["input"].items()
                    },
                }
            ],
        },
    }
    assert (
        pipeline._run_rulespec_test_cases(
            rules_file=rules,
            compiled_path=compiled,
            compiled_payload=payload,
            cases=[companion],
        )
        == []
    )

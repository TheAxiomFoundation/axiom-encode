"""Opt-in synthetic integration with an actual run-lifetime-capable Rust CLI."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import ValidatorPipeline


@pytest.fixture
def execution(tmp_path, monkeypatch):
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip("set AXIOM_LIFETIME_TEST_ENGINE to the real Rust CLI")
    binary = Path(configured).resolve(strict=True)
    root = tmp_path / "rulespec-us"
    rules = root / "us/statutes/99/1.yaml"
    rules.parent.mkdir(parents=True)
    rules.write_text(
        "format: rulespec/v1\nrules:\n"
        + "".join(
            f"  - name: {name}\n    kind: derived\n    entity: Person\n"
            f"    dtype: {dtype}\n    versions:\n"
            f"      - effective_from: '2000-01-01'\n        formula: '{formula}'\n"
            for name, dtype, formula in [
                ("total", "Money", "sum_over_periods(amount)"),
                ("highest", "Money", "max_over_periods(amount)"),
                ("count", "Integer", "count_over_periods(amount)"),
                ("top", "Money", "sum_top_n_over_periods(amount, 1)"),
            ]
        )
    )
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
    # Only binary discovery is injected. Compilation, subprocess execution,
    # artifact admission, formulas and Decimal evaluation use the actual engine.
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)
    periods = [
        {"period_kind": "tax_year", "start": f"{year}-01-01", "end": f"{year}-12-31"}
        for year in (2020, 2021)
    ]
    prefix = "us:statutes/99/1#"
    case = {
        "name": "actual Rust lifetime reduction",
        "period": periods[-1],
        "output": {
            prefix + "total": ["3.000000000000000001", "7.50"],
            prefix + "highest": ["2", "4.25"],
            prefix + "count": [2, 2],
            prefix + "top": ["2", "4.25"],
        },
        "lifetime": {
            "entity": "Person",
            "periods": periods,
            "batches": [
                {
                    "row_count": 2,
                    "entity_ids": ["first", "second"],
                    "inputs": {
                        prefix + "input.amount": {"kind": "decimal", "values": values}
                    },
                }
                for values in (["1.000000000000000001", "3.25"], ["2", "4.25"])
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


def test_pipeline_executes_all_four_reductions_exactly(execution):
    _, run = execution
    assert run() == []


def test_pipeline_reports_exact_decimal_mismatch(execution):
    case, run = execution
    case["output"]["us:statutes/99/1#total"][0] = "3.000000000000000002"
    issues = run()
    assert len(issues) == 1 and "first" in issues[0] and "expected" in issues[0]


def test_pipeline_cannot_supply_a_computed_output_as_history_input(execution):
    case, run = execution
    case["lifetime"]["batches"][0]["inputs"]["us:statutes/99/1#total"] = {
        "kind": "decimal",
        "values": ["3", "7.5"],
    }
    issues = run()
    assert len(issues) == 1 and "unknown public input" in issues[0]

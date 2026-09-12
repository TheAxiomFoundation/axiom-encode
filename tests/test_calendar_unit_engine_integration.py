"""Opt-in transport checks against a real calendar-unit-capable Rust engine."""

import json
import os
import subprocess
from pathlib import Path

import pytest

from axiom_encode.harness.validator_pipeline import ValidatorPipeline


@pytest.fixture
def calendar_execution(tmp_path, monkeypatch):
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip(
            "set AXIOM_LIFETIME_TEST_ENGINE to a calendar-unit-capable Rust CLI"
        )
    binary = Path(configured).resolve(strict=True)

    def execute(increment="1"):
        root = tmp_path / "rulespec-us"
        rules = root / "us/statutes/99/2.yaml"
        rules.parent.mkdir(parents=True, exist_ok=True)
        rules.write_text(
            "format: rulespec/v1\nrules:\n"
            + "".join(
                f"  - name: {name}\n    kind: derived\n    entity: Person\n"
                f"    dtype: {dtype}\n    versions:\n"
                f"      - effective_from: '2000-01-01'\n        formula: '{formula}'\n"
                for name, dtype, formula in [
                    ("selected_years", "Integer", f"count_seed + {increment}"),
                    (
                        "monthly_value",
                        "Decimal",
                        "sum_top_n_over_periods(amount, selected_years) / "
                        "calendar_years_to_months(selected_years)",
                    ),
                    (
                        "observed_months",
                        "Integer",
                        "calendar_years_to_months(count_over_periods(True))",
                    ),
                ]
            )
        )
        compiled = tmp_path / "compiled.json"
        result = subprocess.run(
            [
                binary,
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
            policy_repo_path=root,
            axiom_rules_path=root,
            local_corpus_release=None,
            enable_oracles=False,
        )
        monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)
        periods = [
            {
                "period_kind": "tax_year",
                "start": f"{year}-01-01",
                "end": f"{year}-12-31",
            }
            for year in (2020, 2021, 2024)
        ]
        prefix = "us:statutes/99/2#"
        case = {
            "name": "synthetic complete-year unit conversion",
            "period": periods[-1],
            "output": {
                prefix + "monthly_value": ["0.25", "1"],
                prefix + "observed_months": [36, 36],
            },
            "lifetime": {
                "entity": "Person",
                "periods": periods,
                "batches": [
                    {
                        "row_count": 2,
                        "entity_ids": ["zero-year", "tied-years"],
                        "inputs": {
                            prefix + "input.amount": {
                                "kind": "decimal",
                                "values": values,
                            },
                            prefix + "input.count_seed": {
                                "kind": "integer",
                                "values": [1, 1],
                            },
                        },
                    }
                    for values in (["0", "12"], ["4", "12"], ["2", "12"])
                ],
            },
        }
        return pipeline._run_rulespec_test_cases(
            rules_file=rules,
            compiled_path=compiled,
            compiled_payload=json.loads(compiled.read_text()),
            cases=[case],
        )

    return execute


def test_real_calendar_conversion_handles_derived_counts_and_lifetime_arguments(
    calendar_execution,
):
    assert calendar_execution() == []


def test_real_calendar_conversion_does_not_truncate_fractional_derived_years(
    calendar_execution,
):
    issues = calendar_execution("0.5")
    assert issues
    assert any(
        "calendar" in issue.lower()
        and ("integer" in issue.lower() or "integral" in issue.lower())
        for issue in issues
    )

"""Regressions for callable spellings that are also valid bare RuleSpec facts."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals, validator_pipeline
from axiom_encode.rulespec_formula_identifiers import formula_reference_identifiers

BARE_NAMES = [
    "count",
    "all",
    "any",
    "in",
    "calendar_years_to_months",
    "sum_over_periods",
    "max_over_periods",
    "count_over_periods",
    "sum_top_n_over_periods",
    "date_add_days",
    "date_add_months",
    "date_add_years",
    "days_between",
    "exactly_one",
]
PREFIX = "us:statutes/99/5#"
PERIOD = {"period_kind": "tax_year", "start": "2020-01-01", "end": "2020-12-31"}


def write_rules(tmp_path, formula, helpers=()):
    root = tmp_path / "rulespec-us"
    rules = root / "us/statutes/99/5.yaml"
    rules.parent.mkdir(parents=True, exist_ok=True)
    rules.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "rules": [
                    {
                        "name": name,
                        "kind": "derived",
                        "entity": "Person",
                        "dtype": "Integer",
                        "versions": [
                            {"effective_from": "2000-01-01", "formula": value}
                        ],
                    }
                    for name, value in [*helpers, ("result", formula)]
                ],
            }
        )
    )
    return root, rules


@pytest.mark.parametrize("name", BARE_NAMES)
def test_bare_callable_spelling_survives_all_affected_collectors_and_auto_zero(
    tmp_path, name
):
    formula = f"if is_eligible:\n    {name} + 1\nelse:\n    0"
    root, rules = write_rules(tmp_path, formula)
    expected = {"is_eligible", name}
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._generated_rule_formula_identifiers({"versions": [{"formula": formula}]})
        == expected
    )
    assert (
        cli._local_factual_input_names_from_rules_content(rules.read_text()) == expected
    )
    assert evals._context_file_local_inputs(str(rules)) == expected
    assert validator_pipeline._formula_local_identifiers(formula) == expected
    assert validator_pipeline._rulespec_reference_summary(rules).input_slots == expected
    tests = rules.with_name("5.test.yaml")
    tests.write_text("[]\n")
    assert cli._append_generic_zero_branch_tests_if_missing(
        rules_file=rules,
        test_file=tests,
        repo_path=root / "us",
        relative_output=Path("statutes/99/5.yaml"),
        issues=["Zero branch test coverage missing: `result` returns 0."],
    ) == ["auto_zero_result"]
    assert yaml.safe_load(tests.read_text())[0]["input"] == {
        PREFIX + "input.is_eligible": False,
        PREFIX + "input." + name: 0,
    }


@pytest.mark.parametrize(
    "formula, expected",
    [
        (
            "calendar_years_to_months(calendar_years_to_months) + calendar_years_to_months",
            {"calendar_years_to_months"},
        ),
        ("sum_over_periods(sum_over_periods) + sum_over_periods", {"sum_over_periods"}),
        (
            "days_between(period_start, date_add_days(period_start, date_add_days)) + days_between",
            {"date_add_days", "days_between"},
        ),
        (
            "calendar_years_to_months \n ( count ) + all + any + in",
            {"count", "all", "any", "in"},
        ),
        (
            "date_add_days + calendar_years_to_months('date_add_days') # count()",
            {"date_add_days"},
        ),
        ("days_between(period_start, period_end)", set()),
        (
            "unrecognized(calendar_years_to_months) + count(all)",
            {"unrecognized", "calendar_years_to_months", "count", "all"},
        ),
    ],
)
def test_each_occurrence_preserves_bare_arguments_and_unknown_calls(
    tmp_path, formula, expected
):
    _, rules = write_rules(tmp_path, formula)
    assert formula_reference_identifiers(formula) == expected
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._generated_rule_formula_identifiers({"versions": [{"formula": formula}]})
        == expected
    )
    assert (
        cli._local_factual_input_names_from_rules_content(rules.read_text()) == expected
    )
    assert evals._context_file_local_inputs(str(rules)) == expected


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("date_add_days + 1", ["result"]),
        ("date_add_days(period_start, offset)", ["date_add_days", "result"]),
        (
            "date_add_days + days_between(period_start, date_add_days(period_start, offset))",
            ["result"],
        ),
    ],
)
def test_terminal_exports_distinguish_bare_helper_dependencies_from_calls(
    tmp_path, formula, expected
):
    _, rules = write_rules(tmp_path, formula, helpers=[("date_add_days", "amount + 1")])
    assert evals._context_file_terminal_exports(str(rules)) == expected


@pytest.fixture
def actual_engine():
    configured = os.environ.get("AXIOM_LIFETIME_TEST_ENGINE")
    if not configured:
        pytest.skip("set AXIOM_LIFETIME_TEST_ENGINE to the actual Rust CLI")
    return Path(configured).resolve(strict=True)


def compile_actual(binary, root, rules, compiled):
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


@pytest.mark.parametrize("name", BARE_NAMES)
def test_actual_rust_bare_names_are_factual_inputs(actual_engine, tmp_path, name):
    root, rules = write_rules(tmp_path, f"{name} + 1")
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == {
        name
    }
    compiled = tmp_path / "compiled.json"
    compile_actual(actual_engine, root, rules, compiled)
    assert run_scalar(actual_engine, compiled, {name: 7}) == {
        "kind": "decimal",
        "value": "8",
    }


@pytest.mark.parametrize("name", BARE_NAMES)
def test_actual_fixture_pipeline_accepts_bare_callable_inputs(
    actual_engine, tmp_path, monkeypatch, name
):
    root, rules = write_rules(tmp_path, f"{name} + 1")
    compiled = tmp_path / "compiled.json"
    compile_actual(actual_engine, root, rules, compiled)
    assert (
        run_pipeline(actual_engine, root, rules, compiled, {name: 7}, 8, monkeypatch)
        == []
    )


def run_pipeline(binary, root, rules, compiled, facts, expected, monkeypatch):
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=binary.parent,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: binary)
    return pipeline._run_rulespec_test_cases(
        rules_file=rules,
        compiled_path=compiled,
        compiled_payload=json.loads(compiled.read_text()),
        cases=[
            {
                "name": "actual bare-name fixture preflight and execution",
                "period": PERIOD,
                "input": {
                    PREFIX + "input." + name: value for name, value in facts.items()
                },
                "output": {PREFIX + "result": expected},
            }
        ],
    )


def run_scalar(binary, compiled, facts):
    request = {
        "mode": "explain",
        "dataset": {
            "inputs": [
                {
                    "name": PREFIX + "input." + name,
                    "entity": "Person",
                    "entity_id": "synthetic-bare",
                    "interval": {"start": PERIOD["start"], "end": PERIOD["end"]},
                    "value": {"kind": "integer", "value": value},
                }
                for name, value in facts.items()
            ],
            "relations": [],
        },
        "queries": [
            {
                "entity_id": "synthetic-bare",
                "period": PERIOD,
                "outputs": [PREFIX + "result"],
            }
        ],
    }
    result = subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(compiled)],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)["results"][0]["outputs"][PREFIX + "result"][
        "value"
    ]


@pytest.mark.parametrize(
    "formula, facts, expected",
    [
        (
            "calendar_years_to_months(calendar_years_to_months) + calendar_years_to_months",
            {"calendar_years_to_months": 2},
            "26",
        ),
        (
            "days_between(period_start, date_add_days(period_start, date_add_days)) + date_add_days",
            {"date_add_days": 7},
            "14",
        ),
    ],
)
def test_actual_rust_mixed_call_and_bare_argument(
    actual_engine, tmp_path, formula, facts, expected
):
    root, rules = write_rules(tmp_path, formula)
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == set(
        facts
    )
    compiled = tmp_path / "compiled.json"
    compile_actual(actual_engine, root, rules, compiled)
    assert run_scalar(actual_engine, compiled, facts) == {
        "kind": "decimal",
        "value": expected,
    }


def test_actual_rust_unknown_call_is_not_silently_dropped(actual_engine, tmp_path):
    root, rules = write_rules(tmp_path, "unrecognized_shift(amount)")
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == {
        "unrecognized_shift",
        "amount",
    }
    result = subprocess.run(
        [
            str(actual_engine),
            "compile",
            "--program",
            str(rules),
            "--rulespec-root",
            str(root),
            "--output",
            str(tmp_path / "compiled.json"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode != 0
    assert "unrecognized_shift" in result.stderr


@pytest.mark.parametrize(
    "formula, expected",
    [
        (
            "if period_start and not period_end:\n    1\nelse:\n    0",
            {"period_start", "period_end"},
        ),
        ("if period_start < cutoff_date:\n    1\nelse:\n    0", {"cutoff_date"}),
        (
            "if exactly_one(period_start, period_end):\n    1\nelse:\n    0",
            {"period_start", "period_end"},
        ),
        ("sum_over_periods(period_start)", set()),
        ("sum(members.period_start)", {"members", "period_start"}),
        ("count_where(members, period_end)", {"members", "period_end"}),
        ("period_start[lookup_key]", {"period_start", "lookup_key"}),
    ],
)
def test_period_names_follow_reference_context(tmp_path, formula, expected):
    _, rules = write_rules(tmp_path, formula)
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._local_factual_input_names_from_rules_content(rules.read_text()) == expected
    )
    assert evals._context_file_local_inputs(str(rules)) == expected
    assert validator_pipeline._rulespec_reference_summary(rules).input_slots == expected


def test_unsupported_syntax_conservatively_retains_possible_period_inputs():
    formula = "match selector:\n    0 => period_start\n    _ => period_end"
    assert {"period_start", "period_end"} <= formula_reference_identifiers(formula)


@pytest.mark.parametrize("name", ["period_start", "period_end"])
def test_unknown_root_context_and_judgment_repairs_preserve_period_facts(name):
    assert formula_reference_identifiers(name) == {name}
    assert formula_reference_identifiers(name, judgment=False) == set()
    assert formula_reference_identifiers(f"  {name} \n", judgment=False) == set()
    assert formula_reference_identifiers(name, judgment=True) == {name}
    payload = {
        "rules": [
            {
                "name": "eligible",
                "kind": "derived",
                "dtype": "Judgment",
                "versions": [{"formula": name}],
            }
        ]
    }
    assignments, protected = (
        cli._positive_judgment_formula_input_assignments_for_formula(
            formula=name,
            rules_payload=payload,
            rules_by_name={},
            imported_outputs=set(),
            seen_rules=set(),
        )
    )
    assert assignments == {name: True}
    assert protected == {name}
    assert cli._negated_expression_false_assignments(
        expression=name,
        rules_payload=payload,
        rules_by_name={},
        imported_outputs=set(),
        protected_positive_inputs=set(),
    ) == {name: False}
    assert cli._neutral_unassigned_formula_input_assignments(
        formula=name,
        rules_payload=payload,
        rules_by_name={},
        imported_outputs=set(),
        assigned_inputs=set(),
    ) == {name: False}


@pytest.mark.parametrize("name", ["period_start", "period_end"])
def test_declared_judgment_period_name_is_an_input(tmp_path, name):
    _, rules = write_rules(tmp_path, name)
    payload = yaml.safe_load(rules.read_text())
    rule = payload["rules"][0]
    rule["dtype"] = "Judgment"
    rules.write_text(yaml.safe_dump(payload))
    assert cli._generated_rule_formula_identifiers(rule) == {name}
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == {
        name
    }
    assert evals._context_file_local_inputs(str(rules)) == {name}
    assert validator_pipeline._rulespec_reference_summary(rules).input_slots == {name}


@pytest.mark.parametrize(
    "formula, facts",
    [
        ("if period_start:\n    1\nelse:\n    0", {"period_start": True}),
        ("if period_end:\n    1\nelse:\n    0", {"period_end": True}),
        (
            "if exactly_one(exactly_one, other):\n    1\nelse:\n    0",
            {"exactly_one": True, "other": False},
        ),
        (
            "if period_start and not period_end:\n    1\nelse:\n    0",
            {"period_start": True, "period_end": False},
        ),
    ],
)
def test_actual_fixture_pipeline_preserves_boolean_period_inputs_and_exactly_one(
    actual_engine, tmp_path, monkeypatch, formula, facts
):
    root, rules = write_rules(tmp_path, formula)
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == set(
        facts
    )
    compiled = tmp_path / "compiled.json"
    compile_actual(actual_engine, root, rules, compiled)
    assert (
        run_pipeline(actual_engine, root, rules, compiled, facts, 1, monkeypatch) == []
    )

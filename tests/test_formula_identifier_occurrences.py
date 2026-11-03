"""Regressions for callable spellings that are also valid bare RuleSpec facts."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals, validator_pipeline
from axiom_encode.rulespec_formula_identifiers import formula_reference_identifiers
from tests.lifetime_engine import lifetime_capable_binary

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
def actual_engine(real_engine):
    """Any real engine: these tests only compile and run scalar programs."""
    return real_engine.binary


@pytest.fixture
def calendar_engine(real_engine):
    """An engine with the calendar-unit conversion that ships with run-lifetime."""
    return lifetime_capable_binary(real_engine)


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


def test_actual_rust_calendar_call_and_bare_argument(calendar_engine, tmp_path):
    formula = (
        "calendar_years_to_months(calendar_years_to_months) + calendar_years_to_months"
    )
    facts = {"calendar_years_to_months": 2}
    root, rules = write_rules(tmp_path, formula)
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == set(
        facts
    )
    compiled = tmp_path / "compiled.json"
    compile_actual(calendar_engine, root, rules, compiled)
    assert run_scalar(calendar_engine, compiled, facts) == {
        "kind": "decimal",
        "value": "26",
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
        ("if period_start and not period_end:\n    1\nelse:\n    0", set()),
        ("if period_start < cutoff_date:\n    1\nelse:\n    0", {"cutoff_date"}),
        ("if exactly_one(period_start, period_end):\n    1\nelse:\n    0", set()),
        ("sum_over_periods(period_start)", set()),
        ("sum(members.period_start)", {"members", "period_start"}),
        ("sum(members . period_start)", {"members", "period_start"}),
        ("sum(members.\n    period_end)", {"members", "period_end"}),
        ("count_where(members, period_end)", {"members"}),
        ("period_start[lookup_key]", {"lookup_key"}),
        ("if period_start: 1 else: 0", set()),
    ],
)
def test_period_names_follow_reference_context(tmp_path, formula, expected):
    """A bare period name is the engine's date; only a field access is a fact.

    Every engine refuses to test ``period_start`` or ``period_end`` as a
    judgment ("is a date, so it cannot be tested as a judgment"), so no
    collector may report one as an input to supply.
    """
    _, rules = write_rules(tmp_path, formula)
    assert cli._formula_identifiers(formula) == expected
    assert (
        cli._local_factual_input_names_from_rules_content(rules.read_text()) == expected
    )
    assert evals._context_file_local_inputs(str(rules)) == expected
    assert validator_pipeline._rulespec_reference_summary(rules).input_slots == expected


def test_unsupported_syntax_keeps_field_access_but_not_bare_period_names():
    formula = "match selector:\n    0 => period_start\n    _ => members.period_end"
    references = formula_reference_identifiers(formula)
    assert {"selector", "members", "period_end"} <= references
    assert "period_start" not in references


@pytest.mark.parametrize("name", ["period_start", "period_end"])
@pytest.mark.parametrize("judgment", [None, False, True])
def test_bare_period_names_are_never_facts_to_supply(name, judgment):
    assert formula_reference_identifiers(name, judgment=judgment) == set()
    assert formula_reference_identifiers(f"  {name} \n", judgment=judgment) == set()
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
    assignments, _protected = (
        cli._positive_judgment_formula_input_assignments_for_formula(
            formula=name,
            rules_payload=payload,
            rules_by_name={},
            imported_outputs=set(),
            seen_rules=set(),
        )
    )
    assert name not in assignments
    assert name not in cli._negated_expression_false_assignments(
        expression=name,
        rules_payload=payload,
        rules_by_name={},
        imported_outputs=set(),
        protected_positive_inputs=set(),
    )
    assert name not in cli._neutral_unassigned_formula_input_assignments(
        formula=name,
        rules_payload=payload,
        rules_by_name={},
        imported_outputs=set(),
        assigned_inputs=set(),
    )


@pytest.mark.parametrize("name", ["period_start", "period_end"])
def test_declared_judgment_period_name_is_not_an_input(tmp_path, name):
    _, rules = write_rules(tmp_path, name)
    payload = yaml.safe_load(rules.read_text())
    rule = payload["rules"][0]
    rule["dtype"] = "Judgment"
    rules.write_text(yaml.safe_dump(payload))
    assert cli._generated_rule_formula_identifiers(rule) == set()
    assert cli._local_factual_input_names_from_rules_content(rules.read_text()) == set()
    assert evals._context_file_local_inputs(str(rules)) == set()
    assert validator_pipeline._rulespec_reference_summary(rules).input_slots == set()


@pytest.mark.parametrize(
    "formula, facts",
    [
        (
            "if exactly_one(exactly_one, other):\n    1\nelse:\n    0",
            {"exactly_one": True, "other": False},
        ),
    ],
)
def test_actual_fixture_pipeline_preserves_boolean_inputs_named_like_calls(
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

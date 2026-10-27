from decimal import Decimal

import pytest

from axiom_encode.harness import source_completeness as s


def _rules(mode="half_up", precision=0):
    payload = {
        "units": [{"name": "USD", "kind": "currency", "minor_units": precision}],
        "rules": [
            {
                "name": "rounded",
                "kind": "derived",
                "dtype": "Money",
                "unit": "USD",
                "rounding": mode,
                "versions": [{"effective_from": "2026-01-01", "formula": "amount"}],
            },
            {
                "name": "consumer",
                "kind": "derived",
                "dtype": "Money",
                "unit": "USD",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "rounded * 2"}
                ],
            },
        ],
    }
    bound = s._bind_currency_rounding_rules(payload)
    return payload, {r["name"]: r for r in bound["rules"]}


@pytest.mark.parametrize(
    "mode,value,expected",
    [
        ("half_up", "100.49", "100"),
        ("half_up", "100.50", "101"),
        ("half_up", "-100.50", "-101"),
        ("half_up", "-100.49", "-100"),
        ("half_even", "100.50", "100"),
        ("half_even", "101.50", "102"),
        ("floor", "100.99", "100"),
        ("floor", "-100.01", "-101"),
        ("ceil", "100.01", "101"),
        ("ceil", "-100.99", "-100"),
    ],
)
def test_currency_rounding_executes_native_decimal_output_boundary(
    mode, value, expected
):
    original, rules = _rules(mode)
    case = {"period": "2026-01", "input": {"amount": Decimal(value)}}
    execution = s._case_formula_execution(rules["rounded"], case)
    assert execution is not None
    assert execution.leaf == "amount"
    assert execution.unrounded_value == Decimal(value)
    assert s._formula_execution_runtime_value(execution) == Decimal(expected)
    assert type(original["rules"][0]) is dict


def test_currency_rounding_precedes_dependent_execution_and_assertion_validation():
    _, rules = _rules()
    case = {
        "period": "2026-01",
        "input": {"amount": Decimal("100.5")},
        "output": {"rounded": 101, "consumer": 202},
    }
    assert s._case_asserted_dependency_environment(
        rules, case, formula_environment={}
    ) == {
        "rounded": Decimal("101"),
        "consumer": Decimal("202"),
    }
    case["output"]["rounded"] = 100.5
    assert (
        s._case_asserted_dependency_environment(rules, case, formula_environment={})
        == {}
    )


def test_currency_rounding_precision_is_not_assumed_whole_dollars():
    _, rules = _rules(precision=2)
    execution = s._case_formula_execution(
        rules["rounded"],
        {
            "period": "2026-01",
            "input": {"amount": Decimal("100.505")},
        },
    )
    assert execution.currency_rounding.minor_units == 2
    assert s._formula_execution_runtime_value(execution) == Decimal("100.51")


@pytest.mark.parametrize(
    "mode,precision",
    [("bogus", 0), ([], 0), ("half_up", True), ("half_up", -1), ("half_up", 29)],
)
def test_invalid_currency_rounding_cannot_be_used_as_unrounded_evidence(
    mode, precision
):
    _, rules = _rules(mode, precision)
    assert (
        s._case_formula_execution(
            rules["rounded"], {"period": "2026-01", "input": {"amount": 100.5}}
        )
        is None
    )


def test_unresolved_currency_and_forged_descriptor_are_rejected():
    payload, _ = _rules()
    rule = payload["rules"][0]
    rule["resolved_currency_rounding"] = {"mode": "half_up", "minor_units": 0}
    case = {"period": "2026-01", "input": {"amount": 100.5}}
    assert s._case_formula_execution(rule, case) is None
    for units in ([], payload["units"] * 2):
        payload["units"] = units
        bound = s._bind_currency_rounding_rules(payload)
        assert s._case_formula_execution(bound["rules"][0], case) is None


def test_currency_rounding_does_not_resolve_unknown_import_from_expected_output():
    _, rules = _rules()
    case = {"period": "2026-01", "input": {}, "output": {"rounded": 101}}
    assert (
        s._case_asserted_dependency_environment(rules, case, formula_environment={})
        == {}
    )


def test_currency_rounding_preserves_temporal_formula_selection():
    _, rules = _rules()
    rules["rounded"]["versions"] = [
        {
            "effective_from": "2025-01-01",
            "effective_to": "2025-12-31",
            "formula": "amount",
        },
        {"effective_from": "2026-01-01", "formula": "amount * 2"},
    ]
    for period, expected in [("2025-06", 2), ("2026-06", 5)]:
        execution = s._case_formula_execution(
            rules["rounded"], {"period": period, "input": {"amount": Decimal("2.49")}}
        )
        assert s._formula_execution_runtime_value(execution) == expected


def _metadata_rounding_analysis(
    *, mode="half_up", precision=0, tamper=False, stage="benefit"
):
    import functools

    import yaml

    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded,
    )

    citation = "us/guidance/example/benefit"
    source = (
        "The benefit equals 50 percent of income, rounded to the nearest whole dollar."
    )

    def rule(name, formula, kind="derived"):
        return {
            "name": name,
            "kind": kind,
            "dtype": "Money" if kind == "derived" else "Rate",
            "entity": "Person",
            "period": "Month",
            "unit": "USD",
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "formula",
                            "source": {
                                "corpus_citation_path": citation,
                                "excerpt": source,
                            },
                        }
                    ]
                }
            },
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }

    rules = [
        rule("rate", "0.5", "parameter"),
        rule("unrounded_benefit", "income * rate"),
        rule("benefit", "unrounded_benefit"),
    ]
    next(r for r in rules if r["name"] == stage)["rounding"] = mode
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "units": [{"name": "USD", "kind": "currency", "minor_units": precision}],
        "inputs": [
            {"name": "income", "entity": "Person", "dtype": "Money", "period": "Month"}
        ],
        "rules": rules,
    }
    cases = [
        {
            "name": name,
            "period": "2026-01",
            "input": {"income": income},
            "output": {"unrounded_benefit": raw, "benefit": rounded},
        }
        for name, income, raw, rounded in [
            ("below", 200.98, 100.49, 100),
            ("tie", 201, 100.5, 101),
        ]
    ]
    if tamper:
        cases[0]["output"]["benefit"] = 101
        cases[1]["output"]["benefit"] = 100
    return s.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_complete_source_unit_accepts_executed_native_rounding_with_raw_intermediate():
    result = _metadata_rounding_analysis()
    assert not result.issues, result.issues


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "half_even"},
        {"mode": "floor"},
        {"precision": 2},
        {"tamper": True},
        {"stage": "unrounded_benefit"},
    ],
)
def test_complete_source_unit_rejects_wrong_native_rounding_evidence(kwargs):
    result = _metadata_rounding_analysis(**kwargs)
    assert any("rounding" in issue for issue in result.issues), result.issues


@pytest.mark.parametrize("amount,first,second", [("1.51", 2, 3), ("-1.51", -2, -3)])
def test_successive_rounding_occurs_at_each_owning_output_boundary(
    amount, first, second
):
    payload, _ = _rules()
    payload["rules"][1]["rounding"] = "half_up"
    payload["rules"][1]["versions"][0]["formula"] = "rounded * 1.5"
    bound = s._bind_currency_rounding_rules(payload)
    rules = {r["name"]: r for r in bound["rules"]}
    case = {
        "period": "2026-01",
        "input": {"amount": Decimal(amount)},
        "output": {"rounded": first, "consumer": second},
    }
    dependencies = s._case_asserted_dependency_environment(
        rules, case, formula_environment={}
    )
    assert dependencies == {"rounded": Decimal(first), "consumer": Decimal(second)}
    execution = s._case_formula_execution(
        rules["consumer"], case, dependency_environment=dependencies
    )
    assert execution.unrounded_value == Decimal(first) * Decimal("1.5")
    # Rounding only the final product would produce +/-2 instead of +/-3.
    assert s._formula_execution_runtime_value(execution) == second


def test_source_dependency_expansion_preserves_rounded_output_boundary():
    _, rules = _rules()
    case = {
        "period": "2026-01",
        "input": {"amount": Decimal("100.5")},
        "output": {"rounded": 101, "consumer": 202},
    }
    dependencies = s._case_asserted_dependency_environment(
        rules, case, formula_environment={}
    )
    expanded = s._expand_reached_formula_dependencies(
        "rounded * 2",
        principal_rules=rules,
        case=case,
        formula_environment={},
        dependency_environment=dependencies,
    )
    assert "amount" not in expanded
    assert s._evaluate_formula_selector(expanded, dependencies) == 202


def test_invalid_rounding_parameter_does_not_seed_alias_environment():
    payload, _ = _rules()
    payload["rules"][0]["kind"] = "parameter"
    payload["rules"][0]["versions"][0]["formula"] = "1.5"
    bound = s._bind_currency_rounding_rules(payload)
    assert "rounded" not in s._constant_rule_environment(bound)

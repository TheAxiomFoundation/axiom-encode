"""Direct relation sums evaluate derived values rather than supplied output facts."""

import json
import subprocess

import pytest
import yaml

from axiom_encode.harness.validator_pipeline import (
    find_relation_aggregate_syntax_issues,
)
from axiom_encode.prompts.encoder import ENCODER_PROMPT
from tests.test_companion_declared_entities import _companions

TARGET = "us:policies/example/relation_sum"


def declaration(field="computed"):
    return {
        "format": "rulespec/v1",
        "units": [{"name": "CAD", "kind": "currency", "minor_units": 2}],
        "inputs": [
            {"name": "amount", "entity": "Person", "dtype": "Money", "unit": "CAD"}
        ],
        "rules": [
            {
                "name": "members",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Household", "Person"]},
            },
            {
                "name": "computed",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Month",
                "versions": [{"effective_from": "2026-01-01", "formula": "amount * 2"}],
            },
            {
                "name": "total",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Money",
                "unit": "CAD",
                "period": "Month",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": f"sum(members.{field})"}
                ],
            },
        ],
    }


def test_encoder_prompt_distinguishes_fact_and_computed_sums():
    assert (
        "`sum(relation.local_derived)` evaluates that local derived rule"
        in ENCODER_PROMPT
    )
    assert "do not invent a predicate fact for an unfiltered sum" in ENCODER_PROMPT
    assert "Do not use `sum(relation.local_output)`" not in ENCODER_PROMPT


def test_direct_derived_sum_is_supported_syntax():
    assert find_relation_aggregate_syntax_issues(yaml.safe_dump(declaration())) == []


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize(
    "field,expected,error",
    [("amount", 30, None), ("computed", 60, None), ("unknown", 0, "missing input")],
)
def test_direct_sum_actual_companions(
    tmp_path, real_engine, monkeypatch, executor, field, expected, error
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/relation_sum.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(yaml.safe_dump(declaration(field)))
    module.with_suffix(".test.yaml").write_text(
        yaml.safe_dump(
            [
                {
                    "name": "distinct_children",
                    "period": "2026-01",
                    "input": {
                        f"{TARGET}#relation.members": [
                            {f"{TARGET}#input.amount": 10},
                            {f"{TARGET}#input.amount": 20},
                        ]
                    },
                    "output": {f"{TARGET}#total": expected},
                }
            ]
        )
    )
    calls = []
    original = subprocess.run

    def capture(command, **kwargs):
        result = original(command, **kwargs)
        if "run-compiled" in command:
            calls.append((json.loads(kwargs["input"]), result))
        return result

    monkeypatch.setattr(subprocess, "run", capture)
    issues = _companions(
        executor,
        binary=real_engine.binary,
        engine_root=real_engine.root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )
    assert len(calls) == 1
    request, response = calls[0]
    facts = request["dataset"]["inputs"]
    assert len(facts) == 2
    assert {fact["name"] for fact in facts} == {f"{TARGET}#input.amount"}
    assert len({fact["entity_id"] for fact in facts}) == 2
    assert {fact["entity"] for fact in facts} == {"Person"}
    if error:
        assert issues
        assert error in (response.stderr + response.stdout).lower()
    else:
        assert issues == []
        assert response.returncode == 0

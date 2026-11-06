"""A unique Boolean projection uses explicit related facts in both callers."""

import json
import os
import subprocess

import pytest
import yaml

from axiom_encode.cli import (
    _formula_identifiers,
    _formula_references_relation,
    _local_factual_input_names_from_rules_content,
)
from axiom_encode.companion_relations import executable_relation_directions
from axiom_encode.harness.validator_pipeline import (
    _relation_aggregate_helper_identifiers,
    _rulespec_formula_identifiers,
)
from axiom_encode.rules_engine_compat import run_rulespec_compile
from tests.environment_assertions import Names, hides_environment
from tests.test_companion_declared_entities import _companions

TARGET = "us:policies/example/projection"
FORMULA = "only_related(selected_subject, decision) == true"


def declaration(owner_first=True, formula=FORMULA):
    return {
        "format": "rulespec/v1",
        "inputs": [{"name": "decision", "entity": "Person", "dtype": "Boolean"}],
        "rules": [
            {
                "name": "selected_subject",
                "kind": "data_relation",
                "data_relation": {
                    "arity": 2,
                    "arguments": ["Household", "Person"]
                    if owner_first
                    else ["Person", "Household"],
                },
            },
            {
                "name": "projected",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Judgment",
                "period": "Month",
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            },
        ],
    }


def test_builtin_recognition_preserves_real_references():
    payload = declaration()
    assert _local_factual_input_names_from_rules_content(yaml.safe_dump(payload)) == {
        "decision"
    }
    assert _formula_identifiers(FORMULA) == {"selected_subject", "decision"}
    assert _rulespec_formula_identifiers(payload) == {"selected_subject", "decision"}


def test_unique_projection_does_not_infer_relation_or_all_member_source_scope():
    assert not _formula_references_relation(FORMULA, "selected_subject")
    assert _relation_aggregate_helper_identifiers(FORMULA) == set()
    aggregate = "count_where(selected_subject, decision)"
    assert _formula_references_relation(aggregate, "selected_subject")
    assert _relation_aggregate_helper_identifiers(aggregate) == {"decision"}


@pytest.mark.parametrize("owner_first", [True, False])
def test_only_related_executable_direction_uses_declared_input_kind(owner_first):
    current = 0 if owner_first else 1
    rules = {
        "projected": {
            "id": "projected",
            "entity": "Household",
            "expr": {
                "kind": "only_related",
                "relation": "selected_subject",
                "current_slot": current,
                "related_slot": 1 - current,
                "value": {"kind": "input", "name": "decision"},
            },
        }
    }
    slots = ("Household", "Person") if owner_first else ("Person", "Household")
    assert executable_relation_directions(
        rules,
        ["projected"],
        {"start": "2026-01-01", "end": "2026-01-31"},
        "Household",
        {"selected_subject": slots},
    ) == {"selected_subject": (current, "Person")}


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("owner_first", [True, False])
@pytest.mark.parametrize(
    "values,expected,error",
    [
        ([True], "holds", None),
        ([False], "not_holds", None),
        ([], "holds", "missing related entity"),
        ([True, False], "holds", "ambiguous relation"),
        ([True, True], "holds", "ambiguous relation"),
        ([None], "holds", "missing input"),
    ],
)
def test_only_related_actual_companions(
    tmp_path, real_engine, monkeypatch, executor, owner_first, values, expected, error
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/projection.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(yaml.safe_dump(declaration(owner_first)))
    rows = [
        {} if value is None else {f"{TARGET}#input.decision": value} for value in values
    ]
    cases = [
        {
            "name": "unique_subject",
            "period": "2026-01",
            "input": {f"{TARGET}#relation.selected_subject": rows},
            "output": {f"{TARGET}#projected": expected},
        }
    ]
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    calls = []
    original = subprocess.run

    @hides_environment
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
    (tmp_path / "actual-receipts.json").write_text(
        json.dumps(
            [
                {
                    "request": request,
                    "returncode": response.returncode,
                    "stdout": response.stdout,
                    "stderr": response.stderr,
                }
                for request, response in calls
            ],
            indent=2,
        )
    )
    assert len(calls) == 1, issues
    request, response = calls[0]
    assert all(fact["entity"] == "Person" for fact in request["dataset"]["inputs"])
    current = 0 if owner_first else 1
    root_id = request["queries"][0]["entity_id"]
    records = request["dataset"]["relations"]
    assert len(records) == len(rows)
    assert all(record["tuple"][current] == root_id for record in records)
    related_ids = {record["tuple"][1 - current] for record in records}
    assert root_id not in related_ids
    assert all(
        fact["entity_id"] in related_ids for fact in request["dataset"]["inputs"]
    )
    payload = json.loads((tmp_path / "compiled.json").read_text())

    def projection_nodes(node):
        if isinstance(node, dict):
            if node.get("kind") == "only_related":
                yield node
            for value in node.values():
                yield from projection_nodes(value)
        elif isinstance(node, list):
            for value in node:
                yield from projection_nodes(value)

    projections = list(projection_nodes(payload))
    assert projections
    for projection in projections:
        assert projection["current_slot"] == (0 if owner_first else 1)
        assert projection["related_slot"] == (1 if owner_first else 0)
        assert projection["value"]["kind"] == "input"
    if error:
        assert issues
        assert response.returncode != 0
        assert error in response.stderr.lower(), response.stderr
    else:
        assert not issues
        assert response.returncode == 0
        output = json.loads(response.stdout)["results"][0]["outputs"]
        assert output[f"{TARGET}#projected"]["outcome"] == expected


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("reverse", [False, True])
def test_independent_subject_contexts_and_lazy_branch(
    tmp_path, real_engine, executor, reverse
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/projection.yaml"
    module.parent.mkdir(parents=True)
    payload = declaration(formula="bypass or " + FORMULA)
    payload["inputs"].append(
        {"name": "bypass", "entity": "Household", "dtype": "Boolean"}
    )
    module.write_text(yaml.safe_dump(payload))
    cases = []
    for value in (True, False):
        cases.append(
            {
                "name": f"subject_{value}",
                "period": "2026-01",
                "input": {
                    f"{TARGET}#input.bypass": False,
                    f"{TARGET}#relation.selected_subject": [
                        {f"{TARGET}#input.decision": value}
                    ],
                },
                "output": {f"{TARGET}#projected": "holds" if value else "not_holds"},
            }
        )
    cases.append(
        {
            "name": "unreached_projection",
            "period": "2026-01",
            "input": {f"{TARGET}#input.bypass": True},
            "output": {f"{TARGET}#projected": "holds"},
        }
    )
    if reverse:
        cases.reverse()
        for case in cases:
            case["input"] = dict(reversed(list(case["input"].items())))
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases, sort_keys=False))
    assert not _companions(
        executor,
        binary=real_engine.binary,
        engine_root=real_engine.root,
        root=root,
        module=module,
        artifact=tmp_path / "compiled.json",
    )


@pytest.mark.parametrize("invalid", ["undeclared", "numeric", "wrong_owner"])
def test_only_related_declaration_errors_are_not_hidden(tmp_path, real_engine, invalid):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/projection.yaml"
    module.parent.mkdir(parents=True)
    payload = declaration()
    if invalid == "undeclared":
        payload["inputs"] = []
    elif invalid == "numeric":
        payload["inputs"][0]["dtype"] = "Decimal"
    else:
        payload["inputs"][0]["entity"] = "Household"
    module.write_text(yaml.safe_dump(payload))
    result = run_rulespec_compile(
        binary=real_engine.binary,
        program=module,
        rulespec_roots=[root],
        output=tmp_path / "compiled.json",
        cwd=real_engine.root,
        env=Names(os.environ),
    )
    assert result.returncode != 0
    assert "only_related" in result.stderr

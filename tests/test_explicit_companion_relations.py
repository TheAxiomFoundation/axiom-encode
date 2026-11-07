"""Explicit graph fixtures share one typed carrier across both executors."""

import copy
import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from axiom_encode.checks import ValidationIssue
from axiom_encode.cli import _execute_rulespec_test_case, _execute_rulespec_test_file
from axiom_encode.harness.evals import _normalize_test_periods_to_effective_dates
from axiom_encode.harness.same_case_execution import case_key
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.rules_engine_compat import run_rulespec_compile

PREFIX = "us:policies/graph#"
PERIOD = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
CATALOG = [
    {
        "name": PREFIX + "relation.claims",
        "arity": 2,
        "slot_entities": ["Claim", "Person"],
    },
    {
        "name": PREFIX + "relation.expenses",
        "arity": 2,
        "slot_entities": ["Expense", "Claim"],
    },
]


def graph_case():
    return {
        "name": "two_claims",
        "entity_id": "person-A",
        "period": PERIOD,
        "input": {},
        "tables": {
            "Claim": [{"id": "claim-1"}, {"id": "claim-2"}],
            "Expense": [
                {"id": "expense-1", PREFIX + "input.amount": 100},
                {"id": "expense-2", PREFIX + "input.amount": 250},
                {"id": "unrelated", PREFIX + "input.amount": 9000},
            ],
        },
        "relations": [
            {"name": CATALOG[0]["name"], "tuple": ["claim-1", "person-A"]},
            {"name": CATALOG[0]["name"], "tuple": ["claim-2", "person-A"]},
            {"name": CATALOG[1]["name"], "tuple": ["expense-1", "claim-1"]},
            {"name": CATALOG[1]["name"], "tuple": ["expense-2", "claim-2"]},
        ],
        "output": {PREFIX + "total": 350},
    }


def build_request(tmp_path, case, executor, catalog=None, execution_capture=None):
    catalog = CATALOG if catalog is None else catalog
    root = tmp_path / "rulespec-us"
    module = root / "us" / "policies" / "graph.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "inputs": [
                    {
                        "name": "amount",
                        "entity": "Expense",
                        "dtype": "Money",
                        "unit": "USD",
                        "period": "Year",
                    }
                ],
                "rules": [
                    {
                        "name": "expense_amount",
                        "kind": "derived",
                        "versions": [{"formula": "amount"}],
                    }
                ],
            }
        )
    )
    derived = {
        PREFIX + "total": {
            "id": "total",
            "entity": "Person",
            "expr": {"kind": "literal"},
        }
    }
    if executor == "cli":
        return _execute_rulespec_test_case(
            module.with_suffix(".test.yaml"),
            case,
            case_name="graph",
            compiled_path=tmp_path / "unused.json",
            binary=tmp_path / "unused",
            axiom_rules_path=tmp_path,
            env={},
            parameter_by_id={},
            derived_ids=set(derived),
            derived_by_id=derived,
            declared_relation_names={r["name"] for r in catalog},
            declared_relation_slots={
                r["name"]: tuple(r["slot_entities"]) for r in catalog
            },
            compiled_relations=catalog,
            policy_repo_path=root / "us",
            rulespec_roots=[root],
            request_only=True,
            execution_capture=execution_capture,
        )
    pipeline = ValidatorPipeline(
        root / "us",
        tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    root_id = pipeline._rulespec_case_query_entity_id(case, "Person", 1)
    dataset = pipeline._build_rulespec_dataset(
        case["input"],
        case_tables=case["tables"],
        explicit_relations_case=case,
        compiled_relations=catalog,
        period=PERIOD,
        query_entity="Person",
        query_entity_id=root_id,
    )
    return {"dataset": dataset, "root_id": root_id}


@pytest.mark.parametrize("executor", ["cli", "pipeline"])
def test_explicit_graph_request_and_empty_opt_in(tmp_path, executor):
    case = graph_case()
    request = build_request(tmp_path, case, executor)
    assert request["dataset"]["relations"] == [
        {**edge, "interval": {"start": PERIOD["start"], "end": PERIOD["end"]}}
        for edge in case["relations"]
    ]
    if executor == "cli":
        assert request["queries"][0]["entity_id"] == "person-A"
    else:
        assert request["root_id"] == "person-A"
    case["relations"] = []
    assert build_request(tmp_path, case, executor)["dataset"]["relations"] == []


@pytest.mark.parametrize("executor", ["cli", "pipeline"])
@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "derived",
        "arity",
        "kind",
        "unknown_id",
        "anonymous",
        "bad_id",
        "no_root",
        "null",
        "extra",
        "alias",
        "conflicting_id",
    ],
)
def test_explicit_graph_refusals(tmp_path, executor, mutation):
    case = graph_case()
    catalog = copy.deepcopy(CATALOG)
    edge = case["relations"][0]
    if mutation == "unknown":
        edge["name"] = PREFIX + "relation.unknown"
    elif mutation == "derived":
        catalog[0]["derivation"] = {}
    elif mutation == "arity":
        edge["tuple"].append("person-A")
    elif mutation == "kind":
        edge["tuple"].reverse()
    elif mutation == "unknown_id":
        edge["tuple"][0] = "missing"
    elif mutation == "anonymous":
        del case["tables"]["Claim"][0]["id"]
        edge["tuple"][0] = "claim-1"
    elif mutation == "bad_id":
        edge["tuple"][0] = True
    elif mutation == "no_root":
        del case["entity_id"]
    elif mutation == "null":
        case["relations"] = None
    elif mutation == "extra":
        edge["interval"] = PERIOD
    elif mutation == "alias":
        edge["name"] = "claims"
    elif mutation == "conflicting_id":
        case["tables"]["Claim"][0]["entity_id"] = "other"
    messages = {
        "unknown": "not a declared data_relation",
        "derived": "not a declared data_relation",
        "arity": "arity",
        "kind": "must have entity",
        "unknown_id": "no explicit identity",
        "anonymous": "no explicit identity",
        "bad_id": "nonempty strings",
        "no_root": "case.entity_id",
        "null": "must be a list",
        "extra": "exactly name and tuple",
        "alias": "canonical relation reference",
        "conflicting_id": "conflicting explicit table IDs",
    }
    with pytest.raises(ValueError, match=messages[mutation]):
        build_request(tmp_path, case, executor, catalog)


def test_relations_survive_normalization_and_bind_case_identity():
    case = graph_case()
    normalized = yaml.safe_load(
        _normalize_test_periods_to_effective_dates(
            yaml.safe_dump([case]),
            rulespec_content="format: rulespec/v1\nrules: []\n",
        )
    )[0]
    assert normalized["relations"] == case["relations"]
    assert normalized["entity_id"] == case["entity_id"]
    original_key = case_key(case)
    case["relations"][0]["tuple"][0] = "claim-2"
    assert case_key(case) != original_key


@pytest.mark.real_engine
@pytest.mark.parametrize("executor", ["cli", "pipeline"])
@pytest.mark.parametrize(
    "control",
    [
        "positive",
        "duplicate",
        "missing_fact",
        "null_relations",
        "no_root",
        "conflicting_id",
        "id_aliases",
    ],
)
def test_two_level_graph_executes(tmp_path, real_engine, executor, control):
    root = tmp_path / "rulespec-us"
    module = root / "us" / "policies" / "graph.yaml"
    module.parent.mkdir(parents=True)
    rules = [
        {
            "name": r["name"].split(".")[-1],
            "kind": "data_relation",
            "data_relation": {"arity": 2, "arguments": r["slot_entities"]},
        }
        for r in CATALOG
    ]
    for name, entity, formula in [
        ("expense_amount", "Expense", "amount"),
        ("claim_total", "Claim", "sum(expenses.expense_amount)"),
        ("total", "Person", "sum(claims.claim_total)"),
    ]:
        rules.append(
            {
                "name": name,
                "kind": "derived",
                "entity": entity,
                "dtype": "Money",
                "unit": "USD",
                "period": "Year",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "formula": formula,
                    }
                ],
            }
        )
    module.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "units": [{"name": "USD", "kind": "currency", "minor_units": 2}],
                "rules": rules,
            }
        )
    )
    case = graph_case()
    if control == "duplicate":
        case["relations"].append(copy.deepcopy(case["relations"][2]))
    if control == "missing_fact":
        del case["tables"]["Expense"][0][PREFIX + "input.amount"]
    if control == "null_relations":
        case["relations"] = None
    if control == "no_root":
        del case["entity_id"]
    if control == "conflicting_id":
        case["tables"]["Claim"][0]["entity_id"] = "different"
    if control == "id_aliases":
        row = case["tables"]["Claim"][0]
        row["entity_id"] = row["claim_id"] = row["id"]
        expense = case["tables"]["Expense"][0]
        expense["expense_id"] = expense.pop("id")
    expected_failure = control in {
        "missing_fact",
        "null_relations",
        "no_root",
        "conflicting_id",
    }
    test_file = module.with_suffix(".test.yaml")
    test_file.write_text(yaml.safe_dump([case]))
    artifact = tmp_path / "compiled.json"
    result = run_rulespec_compile(
        binary=real_engine.binary,
        program=module,
        rulespec_roots=[root],
        output=artifact,
        cwd=real_engine.root,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(artifact.read_text())
    if executor == "cli":
        result = _execute_rulespec_test_file(
            test_file,
            binary=real_engine.binary,
            axiom_rules_path=real_engine.root,
            env=os.environ.copy(),
            rulespec_roots=[root],
            tmp_path=tmp_path,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=root / "us",
        )
        failures = result["failures"]
    else:
        pipeline = ValidatorPipeline(
            root / "us",
            real_engine.root,
            axiom_rules_engine_ref=os.environ.get("AXIOM_RULES_ENGINE_REF"),
            local_corpus_release=None,
            enable_oracles=False,
            enforce_repository_layout=False,
        )
        capture = SimpleNamespace(records=[], stage=Mock())
        failures = pipeline._run_rulespec_test_cases(
            rules_file=module,
            compiled_path=artifact,
            compiled_payload=payload,
            cases=[case],
            execution_capture=capture,
        )
        if not expected_failure:
            captured_case, request = capture.stage.call_args.args[:2]
            assert case_key(captured_case) == case_key(case)
            assert request["queries"][0]["entity_id"] == case["entity_id"]
            assert [edge["tuple"] for edge in request["dataset"]["relations"]] == [
                edge["tuple"] for edge in case["relations"]
            ]
    assert bool(failures) is expected_failure, failures
    if control in {"null_relations", "no_root", "conflicting_id"}:
        expected = {
            "null_relations": "relations must be a list",
            "no_root": "case.entity_id",
            "conflicting_id": "conflicting explicit table IDs",
        }[control]
        assert expected in str(failures)


def test_tests_only_overlay_cannot_change_graph(tmp_path):
    from axiom_encode.harness.evals import (
        ValidationRetryCandidate,
        _materialize_eval_artifact,
        _preserves_companion_test_cases,
    )

    case = graph_case()
    text = yaml.safe_dump([case])
    contracts = (
        {
            "name": case["name"],
            "period": PERIOD,
            "input": {},
            "required_output": case["output"],
        },
    )
    assert _preserves_companion_test_cases(
        text, text, required_test_case_contracts=contracts
    )
    changed = copy.deepcopy(case)
    changed["relations"][0]["tuple"][0] = "claim-2"
    assert not _preserves_companion_test_cases(
        text, yaml.safe_dump([changed]), required_test_case_contracts=contracts
    )
    policy = "format: rulespec/v1\nrules: []\n"
    output = tmp_path / "graph.yaml"
    assert _materialize_eval_artifact(
        "=== FILE: graph.test.yaml ===\n" + text,
        output,
        artifact_root=tmp_path,
        repair_candidate=ValidationRetryCandidate(policy, text),
        required_test_case_contracts=contracts,
    )
    assert output.read_text() == policy
    assert yaml.safe_load(output.with_suffix(".test.yaml").read_text()) == [case]


@pytest.mark.parametrize("executor", ["cli", "pipeline"])
def test_parameter_only_graph_case_is_not_silently_ignored(tmp_path, executor):
    module = tmp_path / "rulespec-us/us/policies/graph.yaml"
    module.parent.mkdir(parents=True)
    module.write_text("format: rulespec/v1\nrules: []\n")
    case = graph_case()
    case["relations"] = []
    case["output"] = {PREFIX + "threshold": 1}
    payload = {
        "program": {
            "parameters": [{"id": PREFIX + "threshold", "name": "threshold"}],
            "derived": [],
            "relations": CATALOG,
        }
    }
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(payload))
    test_file = module.with_suffix(".test.yaml")
    test_file.write_text(yaml.safe_dump([case]))
    if executor == "cli":
        result = _execute_rulespec_test_file(
            test_file,
            binary=tmp_path / "absent",
            axiom_rules_path=tmp_path,
            env={},
            rulespec_roots=[tmp_path / "rulespec-us"],
            tmp_path=tmp_path,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=module.parent.parent,
        )
        failures = result["failures"]
    else:
        pipeline = ValidatorPipeline(
            module.parent.parent,
            tmp_path,
            local_corpus_release=None,
            enable_oracles=False,
            enforce_repository_layout=False,
        )
        failures = pipeline._run_rulespec_test_cases(
            rules_file=module,
            compiled_path=artifact,
            compiled_payload=payload,
            cases=[case],
        )
    assert "relations requires a derived query output" in str(failures)
    if executor == "pipeline":
        assert len(failures) == 1
        issue = failures[0]
        assert isinstance(issue, ValidationIssue)
        assert issue.check == "tests:companion-relations/derived-query-required"
        assert issue.definition().requires == (
            "Each companion case containing `relations`, including an empty list, "
            "asserts at least one compiled derived query output."
        )
        assert issue.subject.module == str(module)
        assert issue.subject.test_case == case["name"]
        assert issue.subject.field == "output"
        assert issue.expected == "at least one derived query output"
        assert issue.found == "no derived query output"


@pytest.mark.parametrize(
    "explicit,captured,expected_root",
    [
        (True, True, "person-A"),
        (True, False, "person-A"),
        (False, True, "person-A"),
        (False, False, "case"),
    ],
)
def test_cli_relation_and_scoped_root_composition(
    tmp_path, explicit, captured, expected_root
):
    case = graph_case()
    if not explicit:
        case.pop("relations")
        case.pop("tables")
    request = build_request(
        tmp_path, case, "cli", execution_capture=object() if captured else None
    )
    assert request["queries"][0]["entity_id"] == expected_root


def test_cli_captured_explicit_root_still_rejects_missing_identity(tmp_path):
    case = graph_case()
    case.pop("entity_id")
    with pytest.raises(ValueError, match="case.entity_id"):
        build_request(tmp_path, case, "cli", execution_capture=object())

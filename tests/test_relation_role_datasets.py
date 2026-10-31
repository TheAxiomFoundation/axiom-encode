"""Compare synthetic encoder test requests; these inputs are not RuleSpec modules."""

from __future__ import annotations

import copy
import json
import os
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _execute_rulespec_test_case
from axiom_encode.companion_relations import relation_role_query_entity_id
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.validation_waivers import fingerprint_outcome
from tests.test_companion_declared_entities import _compile

TARGET = "us:policies/example/dataset_roles"
OUTPUT = f"{TARGET}#member_count"
PAIRS = f"{TARGET}#relation.pairs"
TIES = f"{TARGET}#relation.ties"
MEMBERS = f"{TARGET}#relation.members"
AGE = f"{TARGET}#input.age"
SCALAR_VALUES = st.one_of(
    st.integers(min_value=-100, max_value=100),
    st.booleans(),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
    st.sampled_from(["ordinary", "2026-01-02", date(2026, 1, 2)]),
)


def _fixture(tmp_path: Path, *, predicate=True):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/dataset_roles.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    declaration = {
        "format": "rulespec/v1",
        "inputs": {"age": {"dtype": "Integer", "entity": "Person"}},
        "rules": [
            {
                "name": name,
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
            }
            for name in ("pairs", "ties")
        ]
        + [
            {
                "name": "members",
                "kind": "derived_relation",
                "derived_relation": {
                    "arity": 2,
                    "slot_entities": ["Person", "Person"],
                    "current_slot": 0,
                    "related_slot": 1,
                    "source_relation": "pairs",
                },
                "versions": [{"effective_from": "2026-01-01", "formula": predicate}],
            },
            {
                "name": "member_count",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Month",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(members)"}
                ],
            },
        ],
    }
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration)
    )
    return root, module


def _program(current_slot=0):
    relations = [
        {"name": name, "arity": 2, "slot_entities": ["Person", "Person"]}
        for name in (PAIRS, TIES, MEMBERS)
    ]
    relations[-1]["derivation"] = {
        "source_relation": PAIRS,
        "slot_entities": ["Person", "Person"],
        "current_slot": current_slot,
        "related_slot": 1 - current_slot,
        "predicate": {
            "kind": "comparison",
            "left": {"kind": "role_input", "term": "candidate", "name": "age"},
            "op": "lt",
            "right": {"kind": "literal", "value": {"kind": "integer", "value": 22}},
        },
    }
    derived = {
        "id": OUTPUT,
        "name": "member_count",
        "entity": "Person",
        "expr": {
            "kind": "count_related",
            "relation": MEMBERS,
            "current_slot": current_slot,
            "related_slot": 1 - current_slot,
        },
    }
    return {"program": {"derived": [derived], "relations": relations}}


def _requests(tmp_path, case, payload=None):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/dataset_roles.yaml"
    if not module.exists():
        _fixture(tmp_path, predicate="candidate.age < 22")
    payload = payload or _program()
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=tmp_path / "absent-engine",
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    derived, parameters = pipeline._rulespec_program_maps(payload)
    relations = payload["program"]["relations"]
    names = {relation["name"] for relation in relations}
    slots = {
        relation["name"]: tuple(relation["slot_entities"]) for relation in relations
    }
    cli = _execute_rulespec_test_case(
        module.with_suffix(".test.yaml"),
        case,
        case_name="comparison",
        compiled_path=tmp_path / "compiled.json",
        binary=tmp_path / "engine",
        axiom_rules_path=tmp_path,
        env={},
        parameter_by_id=parameters,
        derived_ids=set(derived),
        derived_by_id=derived,
        declared_relation_names=names,
        declared_relation_slots=slots,
        compiled_relations=relations,
        policy_repo_path=root / "us",
        rulespec_roots=[root],
        request_only=True,
    )
    assert cli.pop("_companion_parameters") == {}
    captured = {}

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        captured.update(json.loads(kwargs["input"]))
        raise Captured

    with patch("axiom_encode.harness.validator_pipeline.subprocess.run", capture):
        try:
            result = pipeline._run_rulespec_derived_test_case(
                binary=tmp_path / "engine",
                compiled_path=tmp_path / "compiled.json",
                case=case,
                case_name="comparison",
                case_index=7,
                period=pipeline._coerce_rulespec_period(case.get("period", "2026-01")),
                output_names=[OUTPUT],
                derived_by_key=derived,
                require_legal_input_keys=True,
                legal_ids_by_friendly_name={},
                module_target=TARGET,
                declared_relation_names=names,
                declared_relation_slots=slots,
                compiled_relations=relations,
            )
        except Captured:
            pass
        else:
            pytest.fail(f"pipeline stopped before execution: {result}")
    return cli, captured


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({}, "case"),
        ({"entity_id": "alice", "id": "other"}, "alice"),
        ({"id": "alice", "person_id": "other"}, "alice"),
        ({"person_id": "alice", "other_id": "other"}, "alice"),
        ({"other_id": "alice"}, "alice"),
        ({"other_id": ["ignored"]}, "case"),
    ],
)
def test_b4_explicit_tuple_companions_share_root_identity(tmp_path, metadata, expected):
    case = {
        **metadata,
        "input": {PAIRS: [{"tuple": [expected, "bob"]}], TIES: []},
        "output": {OUTPUT: 1},
    }
    cli, pipeline = _requests(tmp_path, case)
    assert cli == pipeline
    assert cli["queries"][0]["entity_id"] == expected


@pytest.mark.parametrize("id_key", ["id", "entity_id"])
def test_b5_compact_row_identity_never_becomes_a_factual_input(tmp_path, id_key):
    case = {
        "input": {
            PAIRS: [{id_key: "alice", AGE: 20}],
            TIES: [{"tuple": ["case", "alice"]}],
        },
        "output": {OUTPUT: 1},
    }
    cli, pipeline = _requests(tmp_path, case)
    assert cli == pipeline
    assert cli["dataset"]["inputs"][0]["entity_id"] == "alice"
    assert {record["name"] for record in cli["dataset"]["inputs"]} == {AGE}
    assert cli["dataset"]["relations"][0]["tuple"] == ["case", "alice"]


@pytest.mark.parametrize("id_key", ["id", "entity_id"])
@pytest.mark.parametrize("member_id", ["alice", 0, ""])
def test_compact_row_metadata_applies_to_legacy_programs(tmp_path, id_key, member_id):
    payload = _program()
    derivation = payload["program"]["relations"][-1]["derivation"]
    derivation["entity"] = "Person"
    derivation["predicate"] = {
        "kind": "literal",
        "value": {"kind": "bool", "value": True},
    }
    case = {
        "entity_id": "case",
        "input": {PAIRS: [{id_key: member_id, AGE: 20}], TIES: []},
        "output": {OUTPUT: 1},
    }
    cli, pipeline = _requests(tmp_path, case, payload)
    assert cli == pipeline
    assert cli["dataset"]["relations"][0]["tuple"] == ["case", str(member_id)]
    assert {record["name"] for record in cli["dataset"]["inputs"]} == {AGE}


@pytest.mark.parametrize("row_ordered", [False, True])
def test_relation_role_tables_change_query_ids_only_for_row_ordered_outputs(
    tmp_path, row_ordered
):
    case = {
        "entity_id": "root",
        "input": {PAIRS: [], TIES: []},
        "tables": {"Person": [{"id": "table-person", AGE: 20}]},
        "output": {OUTPUT: [0] if row_ordered else 0},
    }
    cli, pipeline = _requests(tmp_path, case)
    assert cli == pipeline
    assert cli["queries"][0]["entity_id"] == ("table-person" if row_ordered else "root")


@settings(
    max_examples=70,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    root_key=st.sampled_from([None, "entity_id", "id", "person_id", "owner_id"]),
    root_id=st.sampled_from(["alice", "root-2", "case"]),
    identity_key=st.sampled_from([None, "id", "entity_id"]),
    ages=st.lists(SCALAR_VALUES, max_size=4),
    root_fact=SCALAR_VALUES,
    current_slot=st.integers(min_value=0, max_value=1),
    explicit=st.booleans(),
    table_mode=st.sampled_from(["none", "scalar", "list"]),
    month=st.integers(min_value=1, max_value=12),
)
def test_generated_relation_role_requests_match_exactly(
    tmp_path,
    root_key,
    root_id,
    identity_key,
    ages,
    root_fact,
    current_slot,
    explicit,
    table_mode,
    month,
):
    metadata = {root_key: root_id} if root_key else {}
    root_id = relation_role_query_entity_id(metadata, "Person")
    rows = []
    for index, age in enumerate(ages):
        member_id = f"member-{index}"
        if explicit:
            pair = [root_id, member_id]
            if current_slot == 1:
                pair.reverse()
            rows.append({"tuple": pair})
        else:
            row = {AGE: age}
            if identity_key:
                row[identity_key] = member_id
            rows.append(row)
    case = {
        **metadata,
        "period": f"2026-{month:02}",
        "input": {PAIRS: rows, TIES: [], AGE: root_fact},
        "output": {OUTPUT: 0},
    }
    if table_mode != "none":
        case["tables"] = {"Person": [{"id": "table-person", AGE: root_fact}]}
        if table_mode == "list":
            case["output"][OUTPUT] = [0]
    cli, pipeline = _requests(tmp_path, case, _program(current_slot))
    assert cli == pipeline


def test_legacy_root_ids_and_outcome_fingerprints_stay_unchanged(tmp_path):
    case = {"entity_id": "alice", "input": {}, "output": {OUTPUT: 1}}
    payload = _program()
    payload["program"]["relations"] = []
    payload["program"]["derived"][0]["expr"] = {
        "kind": "literal",
        "value": {"kind": "integer", "value": 1},
    }
    cli, pipeline = _requests(tmp_path, case, payload)
    assert cli["queries"][0]["entity_id"] == "case"
    assert pipeline["queries"][0]["entity_id"] == "alice"
    pipeline_instance = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path / "absent-engine",
        local_corpus_release=None,
        enable_oracles=False,
    )
    assert pipeline_instance._rulespec_case_query_entity_id({}, "Person", 7) == "case-7"
    outcome = {"present": True, "passed": True, "cases": 1, "failures": []}
    with_requests = {**outcome, "request": cli, "dataset": pipeline["dataset"]}
    assert fingerprint_outcome({"passed": True}, outcome) == fingerprint_outcome(
        {"passed": True}, with_requests
    )


@pytest.mark.parametrize(
    "case_kind", ["named-root", "default-root", "compact-id", "compact-entity-id"]
)
def test_b4_b5_both_companion_paths_execute_the_same_dataset(
    tmp_path, real_engine, case_kind
):
    compact = case_kind.startswith("compact")
    root, module = _fixture(
        tmp_path,
        predicate="ties(anchor, candidate) and candidate.age < 22" if compact else True,
    )
    root_id = "alice" if case_kind == "named-root" else "case"
    metadata = {"entity_id": root_id} if case_kind == "named-root" else {}
    row = {"tuple": [root_id, "bob"]}
    if compact:
        row = {
            "entity_id" if case_kind == "compact-entity-id" else "id": "bob",
            AGE: 20,
        }
    case = {
        "name": case_kind,
        "period": "2026-01",
        **metadata,
        "input": {PAIRS: [row], TIES: [{"tuple": [root_id, "bob"]}]},
        "output": {OUTPUT: 1},
    }
    artifact = tmp_path / "compiled.json"
    payload = _compile(real_engine.binary, real_engine.root, root, module, artifact)
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    derived, parameters = pipeline._rulespec_program_maps(payload)
    relations = payload["program"]["relations"]
    cli_issues = _execute_rulespec_test_case(
        module.with_suffix(".test.yaml"),
        case,
        case_name=case_kind,
        compiled_path=artifact,
        binary=real_engine.binary,
        axiom_rules_path=real_engine.root,
        env=os.environ.copy(),
        parameter_by_id=parameters,
        derived_ids=set(derived),
        derived_by_id=derived,
        declared_relation_names={r["name"] for r in relations},
        declared_relation_slots={
            r["name"]: tuple(r["slot_entities"]) for r in relations
        },
        compiled_relations=relations,
        policy_repo_path=root / "us",
        rulespec_roots=[root],
    )
    assert cli_issues == []
    assert (
        pipeline._run_rulespec_test_cases(
            rules_file=module,
            compiled_path=artifact,
            compiled_payload=payload,
            cases=[case],
        )
        == []
    )
    cli_request, pipeline_request = _requests(tmp_path, copy.deepcopy(case), payload)
    assert cli_request == pipeline_request

"""Companion request shape regressions; execution is tested with a real engine separately."""

import copy
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _execute_rulespec_test_case
from axiom_encode.companion_relations import (
    LEGACY_LAYOUT,
    CompanionRelations,
    RelationLayout,
)

MODULE = "us:statutes/1/companion"
RELATION = f"{MODULE}#relation.member_of_unit"
OUTPUT = f"{MODULE}#credit"
ELIGIBLE = f"{MODULE}#eligible"
ROOT_INPUT = f"{MODULE}#input.base"
MEMBER_INPUT = f"{MODULE}#input.is_eligible"
INTERVAL = {"start": "2026-01-01", "end": "2026-12-31"}


def _aggregate(relation=RELATION, current_slot=0, kind="count_related"):
    node = {
        "kind": kind,
        "relation": relation,
        "current_slot": current_slot,
        "related_slot": 1 - current_slot,
        "where": {"kind": "derived", "name": ELIGIBLE},
    }
    if kind == "sum_related":
        node["value"] = {"kind": "input", "name": MEMBER_INPUT}
    return node


def _program(*, owner="TaxUnit", typed=True, declared_slot=0, current_slot=0):
    relation = {"name": RELATION, "arity": 2}
    if typed:
        kinds = ["Person", "Person"]
        kinds[declared_slot] = owner
        relation["slot_entities"] = kinds
    return {
        "relations": [relation],
        "derived": [
            {
                "id": OUTPUT,
                "entity": owner,
                "semantics": "scalar",
                "expr": _aggregate(current_slot=current_slot),
            },
            {
                "id": ELIGIBLE,
                "entity": "Person",
                "semantics": "judgment",
                "expr": {"kind": "input", "name": MEMBER_INPUT},
            },
        ],
    }


def _capture_request(
    program, rows, *, relation_key=RELATION, extra_inputs=None, extra_outputs=None
):
    """Stop at the process boundary; do not substitute an evaluator."""

    class RequestCaptured(Exception):
        pass

    captured = []

    def capture_run(_command, **kwargs):
        captured.append(json.loads(kwargs["input"]))
        raise RequestCaptured

    derived_by_id = {rule["id"]: rule for rule in program["derived"]}
    with patch("axiom_encode.cli.subprocess.run", side_effect=capture_run):
        with pytest.raises(RequestCaptured):
            _execute_rulespec_test_case(
                Path("companion.test.yaml"),
                {
                    "period": 2026,
                    "input": {
                        ROOT_INPUT: 0,
                        relation_key: [{MEMBER_INPUT: row} for row in rows],
                        **(extra_inputs or {}),
                    },
                    "output": {OUTPUT: 0, **(extra_outputs or {})},
                },
                case_name="request_shape",
                compiled_path=Path("compiled.json"),
                binary=Path("engine"),
                axiom_rules_path=Path("."),
                env={},
                parameter_by_id={},
                derived_ids=set(derived_by_id),
                derived_by_id=derived_by_id,
                declared_relation_names={
                    relation["name"] for relation in program["relations"]
                },
                policy_repo_path=Path("."),
                program=program,
            )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize("current_slot", [0, 1])
@pytest.mark.parametrize("kind", ["count_related", "sum_related"])
def test_typed_artifact_uses_executable_slots(current_slot, kind):
    program = _program(declared_slot=current_slot, current_slot=current_slot)
    program["derived"][0]["expr"] = _aggregate(current_slot=current_slot, kind=kind)
    layout = CompanionRelations(program).layout([RELATION], "TaxUnit")
    assert layout == RelationLayout(current_slot, "TaxUnit", "Person")
    assert layout.tuple("unit", "child")[current_slot] == "unit"
    assert layout.tuple("unit", "child")[1 - current_slot] == "child"


@pytest.mark.parametrize("declared_slot", [0, 1])
def test_executable_usage_overrides_opposite_declared_order(declared_slot):
    program = _program(declared_slot=declared_slot, current_slot=1 - declared_slot)
    assert CompanionRelations(program).layout([RELATION], "TaxUnit") == RelationLayout(
        1 - declared_slot, "TaxUnit", "Person"
    )


def test_executable_short_derived_name_supplies_actual_related_kind():
    program = _program()
    program["derived"][1]["name"] = "eligible_child"
    program["derived"][1]["entity"] = "Child"
    program["derived"][0]["expr"]["where"] = {
        "kind": "derived",
        "name": "eligible_child",
    }
    assert CompanionRelations(program).layout([RELATION], "TaxUnit") == RelationLayout(
        0, "TaxUnit", "Child"
    )
    request = _capture_request(program, [True])
    assert [record["entity"] for record in request["dataset"]["inputs"]] == [
        "TaxUnit",
        "Child",
    ]


@pytest.mark.parametrize("declared_slot", [0, 1])
def test_unused_relation_falls_back_to_declared_order(declared_slot):
    program = _program(declared_slot=declared_slot)
    program["derived"] = []
    assert CompanionRelations(program).layout([RELATION], "TaxUnit") == RelationLayout(
        declared_slot, "TaxUnit", "Person"
    )


def test_unused_same_kind_relation_preserves_legacy_orientation():
    program = _program(owner="Person")
    program["derived"] = []
    assert CompanionRelations(program).layout([RELATION], "Person") == RelationLayout(
        1, "Person", "Person"
    )


def test_empty_typed_relation_needs_no_owner_inference():
    program = _program()
    del program["derived"][0]["entity"]
    request = _capture_request(program, [])
    assert request["dataset"]["relations"] == []


@pytest.mark.parametrize("empty_declaration", [False, True])
def test_old_artifact_preserves_related_first_and_generic_labels(empty_declaration):
    program = _program(typed=False, current_slot=1)
    if empty_declaration:
        program["relations"][0]["slot_entities"] = []
    layout = CompanionRelations(program).layout([RELATION], "TaxUnit")
    assert layout == LEGACY_LAYOUT
    assert layout.tuple("unit", "child") == ["child", "unit"]


def test_versioned_rule_ignores_nonexecutable_base_semantics():
    program = _program(current_slot=1)
    program["derived"][0]["versions"] = [
        {
            "effective_from": "2026-01-01",
            "semantics": "scalar",
            "expr": _aggregate(current_slot=0),
        }
    ]
    assert CompanionRelations(program).layout([RELATION], "TaxUnit") == RelationLayout(
        0, "TaxUnit", "Person"
    )


def test_nested_predicate_membership_has_related_entity_context():
    employer_relation = f"{MODULE}#relation.employment"
    program = _program()
    program["relations"].append(
        {
            "name": employer_relation,
            "arity": 2,
            # Deliberately opposite to the executable membership node.
            "slot_entities": ["Person", "Employer"],
        }
    )
    program["derived"][0]["expr"]["where"] = {
        "kind": "and",
        "items": [
            {"kind": "derived", "name": ELIGIBLE},
            {
                "kind": "not",
                "item": {
                    "kind": "relation_member",
                    "relation": employer_relation,
                    "current_slot": 1,
                    "related_slot": 0,
                },
            },
        ],
    }
    resolver = CompanionRelations(program)
    assert resolver.layout([RELATION], "TaxUnit") == RelationLayout(
        0, "TaxUnit", "Person"
    )
    assert resolver.layout([employer_relation], "Person") == RelationLayout(
        1, "Person", "Employer"
    )


def test_membership_relation_derivation_resolves_source_slots():
    filtered_relation = f"{MODULE}#relation.eligible_member_of_unit"
    program = _program()
    program["relations"].append(
        {
            "name": filtered_relation,
            "arity": 2,
            "slot_entities": ["TaxUnit", "Person"],
            "derivation": {
                "source_relation": RELATION,
                "current_slot": 1,
                "related_slot": 0,
                "predicate": {"kind": "derived", "name": ELIGIBLE},
            },
        }
    )
    program["derived"][0]["expr"] = _aggregate(relation=filtered_relation)
    resolver = CompanionRelations(program)
    assert resolver.layout([filtered_relation], "TaxUnit") == RelationLayout(
        0, "TaxUnit", "Person"
    )
    assert resolver.layout([RELATION], "TaxUnit") == RelationLayout(
        1, "TaxUnit", "Person"
    )


def test_imported_relation_resolution_prefers_exact_canonical_name():
    short_name = "member_of_unit"
    program = _program()
    program["relations"].append(
        {"name": short_name, "arity": 2, "slot_entities": ["Person", "Household"]}
    )
    assert CompanionRelations(program).layout([RELATION, short_name], "TaxUnit") == (
        RelationLayout(0, "TaxUnit", "Person")
    )
    request = _capture_request(program, [True])
    assert [record["name"] for record in request["dataset"]["relations"]] == [RELATION]


def test_old_short_relation_alias_still_resolves_typed_declaration():
    short_name = "member_of_unit"
    program = _program()
    program["relations"][0]["name"] = short_name
    program["derived"][0]["expr"]["relation"] = short_name
    assert CompanionRelations(program).layout([RELATION, short_name], "TaxUnit") == (
        RelationLayout(0, "TaxUnit", "Person")
    )


def test_conflicting_executable_usages_report_relation_name():
    program = _program()
    conflicting_rule = copy.deepcopy(program["derived"][0])
    conflicting_rule["id"] = f"{MODULE}#other_credit"
    conflicting_rule["expr"] = _aggregate(current_slot=1)
    program["derived"].append(conflicting_rule)
    with pytest.raises(
        ValueError, match=f"conflicting executable relation slots for {RELATION}"
    ):
        CompanionRelations(program).layout([RELATION], "TaxUnit")


@pytest.mark.parametrize("typed", [False, True])
def test_companion_request_uses_artifact_order_and_entity_kinds(typed):
    program = _program(typed=typed, current_slot=0 if typed else 1)
    request = _capture_request(program, [True, False])
    assert "relation_binding" not in request
    assert [record["entity"] for record in request["dataset"]["inputs"]] == (
        ["TaxUnit", "Person", "Person"] if typed else ["Entity"] * 3
    )
    assert [record["tuple"] for record in request["dataset"]["relations"]] == (
        [["case", "related_0_0"], ["case", "related_0_1"]]
        if typed
        else [["related_0", "case"], ["related_1", "case"]]
    )


@settings(max_examples=150, deadline=None)
@given(
    owner=st.sampled_from(["TaxUnit", "Household"]),
    typed=st.booleans(),
    declared_slot=st.integers(min_value=0, max_value=1),
    executable_slot=st.integers(min_value=0, max_value=1),
    rows=st.lists(st.booleans(), min_size=0, max_size=40),
)
def test_generated_companion_requests_preserve_slot_kinds_and_legacy_bytes(
    owner, typed, declared_slot, executable_slot, rows
):
    # Untyped release artifacts aggregate from slot 1. Typed declarations may
    # disagree with executable nodes, which remain authoritative for binding.
    current_slot = executable_slot if typed else 1
    program = _program(
        owner=owner,
        typed=typed,
        declared_slot=declared_slot,
        current_slot=current_slot,
    )
    request = _capture_request(program, rows)
    assert "relation_binding" not in request
    assert len(request["dataset"]["relations"]) == len(rows)
    if typed:
        labels = {
            record["entity_id"]: record["entity"]
            for record in request["dataset"]["inputs"]
        }
        expected_kinds = ["Person", "Person"]
        expected_kinds[current_slot] = owner
        assert labels["case"] == owner
        for index, relation in enumerate(request["dataset"]["relations"]):
            assert relation["tuple"][current_slot] == "case"
            assert relation["tuple"][1 - current_slot] == f"related_0_{index}"
            assert [
                labels[entity_id] for entity_id in relation["tuple"]
            ] == expected_kinds
    else:
        legacy_request = {
            "mode": "explain",
            "dataset": {
                "inputs": [
                    {
                        "name": ROOT_INPUT,
                        "entity": "Entity",
                        "entity_id": "case",
                        "interval": INTERVAL,
                        "value": {"kind": "integer", "value": 0},
                    },
                    *[
                        {
                            "name": MEMBER_INPUT,
                            "entity": "Entity",
                            "entity_id": f"related_{index}",
                            "interval": INTERVAL,
                            "value": {"kind": "bool", "value": row},
                        }
                        for index, row in enumerate(rows)
                    ],
                ],
                "relations": [
                    {
                        "name": RELATION,
                        "tuple": [f"related_{index}", "case"],
                        "interval": INTERVAL,
                    }
                    for index in range(len(rows))
                ],
            },
            "queries": [
                {
                    "entity_id": "case",
                    "period": {"period_kind": "tax_year", **INTERVAL},
                    "outputs": [OUTPUT],
                }
            ],
        }
        assert json.dumps(request) == json.dumps(legacy_request)


def test_typed_relations_keep_distinct_member_ids_and_unambiguous_kinds():
    employer_relation = f"{MODULE}#relation.employer_of_unit"
    employer_rule = f"{MODULE}#eligible_employer"
    employer_input = f"{MODULE}#input.is_employer"
    program = _program()
    program["relations"].append(
        {
            "name": employer_relation,
            "arity": 2,
            "slot_entities": ["Employer", "TaxUnit"],
        }
    )
    program["derived"].append(
        {
            "id": employer_rule,
            "entity": "Employer",
            "semantics": "judgment",
            "expr": {"kind": "input", "name": employer_input},
        }
    )
    employer_aggregate = _aggregate(relation=employer_relation, current_slot=1)
    employer_aggregate["where"] = {"kind": "derived", "name": employer_rule}
    program["derived"][0]["expr"] = {
        "kind": "add",
        "items": [program["derived"][0]["expr"], employer_aggregate],
    }
    request = _capture_request(
        program, [True], extra_inputs={employer_relation: [{employer_input: True}]}
    )
    inputs = request["dataset"]["inputs"]
    assert len({record["entity_id"] for record in inputs}) == 3
    kinds_by_id = {record["entity_id"]: record["entity"] for record in inputs}
    assert set(kinds_by_id.values()) == {"TaxUnit", "Person", "Employer"}
    expected_kinds = {
        relation["name"]: relation["slot_entities"] for relation in program["relations"]
    }
    assert len(request["dataset"]["relations"]) == 2
    for relation in request["dataset"]["relations"]:
        assert [kinds_by_id[entity_id] for entity_id in relation["tuple"]] == (
            expected_kinds[relation["name"]]
        )


def test_scalar_output_does_not_obscure_concrete_companion_owner():
    program = _program()
    scalar_output = f"{MODULE}#constant"
    program["derived"].append(
        {
            "id": scalar_output,
            "entity": "Scalar",
            "semantics": "scalar",
            "expr": {"kind": "literal", "value": {"kind": "integer", "value": 0}},
        }
    )
    request = _capture_request(program, [True], extra_outputs={scalar_output: 0})
    assert [record["entity"] for record in request["dataset"]["inputs"]] == [
        "TaxUnit",
        "Person",
    ]
    assert request["dataset"]["relations"][0]["tuple"] == ["case", "related_0_0"]


def test_membership_keeps_enclosing_kind_when_derivation_entity_matches():
    program = _program()
    source_relation = f"{MODULE}#relation.source_member_of_unit"
    program["relations"].append(
        {
            "name": source_relation,
            "arity": 2,
            "slot_entities": ["TaxUnit", "Person"],
        }
    )
    program["relations"][0]["derivation"] = {
        "source_relation": source_relation,
        "current_slot": 0,
        "related_slot": 1,
        "entity": "TaxUnit",
        "slot_entities": ["TaxUnit", "Person"],
        "predicate": {"kind": "derived", "name": ELIGIBLE},
    }
    program["derived"][0]["semantics"] = "judgment"
    program["derived"][0]["expr"] = {
        "kind": "relation_member",
        "relation": RELATION,
        "current_slot": 1,
        "related_slot": 0,
    }
    assert CompanionRelations(program).layout([RELATION], "TaxUnit") == RelationLayout(
        1, "TaxUnit", "Person"
    )

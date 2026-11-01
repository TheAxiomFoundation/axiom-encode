"""Keep filtered companion facts on their runtime IDs; these are not RuleSpec modules.

A filtered alias takes its runtime ID from one source's current ID, so a
`SnapUnit` query's root ID also identifies its `Person` anchor. The engine keys
inputs by name and ID, so a fact on either kind belongs on that root ID. Facts
on any other kind stay apart.
"""

from __future__ import annotations

import copy
import json
import os

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.cli import _execute_rulespec_test_case
from axiom_encode.companion_entities import (
    assign_generated_fixture_ids,
    flat_input_entity_id,
    require_distinct_entity_ids,
    reserve_explicit_tuple_ids,
)
from axiom_encode.companion_relations import (
    _slot_kinds,
    filtered_entity_kinds,
    query_entity_kinds,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_datasets import (
    AGE,
    OUTPUT,
    PAIRS,
    TARGET,
    TIES,
    _fixture,
    _program,
    _requests,
)

BONUS = f"{TARGET}#input.bonus"
OTHER = f"{TARGET}#input.other"
SIZE = f"{TARGET}#input.size"
ANCHOR_PREDICATE = "candidate.age == anchor.age"
CANDIDATE_PREDICATE = "candidate.age == candidate.age"
BOB = {"id": "bob", AGE: "01"}
EXPLICIT_PAIR = [{"tuple": ["case", "bob"]}]


def _filtered(tmp_path, *, formula="len(selected)", predicate=ANCHOR_PREDICATE):
    root, module = _fixture(tmp_path, predicate=predicate, dtype="Text")
    declaration = yaml.safe_load(module.read_text())
    declaration["inputs"] = [
        {"name": "age", "dtype": "Text", "entity": "Person"},
        {"name": "bonus", "dtype": "Integer", "entity": "SnapUnit"},
        {"name": "other", "dtype": "Integer", "entity": "Household"},
    ]
    declaration["rules"][2]["derived_relation"].update(
        entity="SnapUnit", member_relation="selected"
    )
    declaration["rules"][3]["entity"] = "SnapUnit"
    declaration["rules"][3]["versions"][0]["formula"] = formula
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration)
    )
    return root, module


def _run_both(tmp_path, root, module, case, real_engine):
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
    cli = _execute_rulespec_test_case(
        module.with_suffix(".test.yaml"),
        case,
        case_name=case["name"],
        compiled_path=artifact,
        binary=real_engine.binary,
        axiom_rules_path=real_engine.root,
        env=os.environ.copy(),
        parameter_by_id=parameters,
        derived_ids=set(derived),
        derived_by_id=derived,
        declared_relation_names={relation["name"] for relation in relations},
        declared_relation_slots={
            relation["name"]: tuple(relation["slot_entities"]) for relation in relations
        },
        compiled_relations=relations,
        policy_repo_path=root / "us",
        rulespec_roots=[root],
    )
    other = pipeline._run_rulespec_test_cases(
        rules_file=module,
        compiled_path=artifact,
        compiled_payload=payload,
        cases=[case],
    )
    return cli, other, payload


def _ids_by_name(request):
    return {
        (record["name"], record["entity_id"]) for record in request["dataset"]["inputs"]
    }


@pytest.mark.parametrize(
    "formula,predicate,facts,tables,expected",
    [
        pytest.param(
            "len(selected)",
            ANCHOR_PREDICATE,
            {AGE: "01"},
            {"Person": [BOB]},
            1,
            id="anchor-fact",
        ),
        pytest.param(
            "len(selected) + bonus",
            CANDIDATE_PREDICATE,
            {BONUS: 1},
            {"Person": [BOB]},
            2,
            id="own-fact",
        ),
        pytest.param(
            "len(selected) + bonus",
            ANCHOR_PREDICATE,
            {AGE: "01", BONUS: 1},
            {"Person": [BOB]},
            2,
            id="both-kinds",
        ),
        pytest.param(
            "len(selected)",
            ANCHOR_PREDICATE,
            {},
            {"Person": [{"id": "case", AGE: "01"}, BOB]},
            1,
            id="anchor-table-row",
        ),
    ],
)
def test_filtered_query_facts_execute_on_the_root_id(
    tmp_path, real_engine, formula, predicate, facts, tables, expected
):
    root, module = _filtered(tmp_path, formula=formula, predicate=predicate)
    case = {
        "name": "filtered",
        "period": "2026-01",
        "input": {PAIRS: EXPLICIT_PAIR, TIES: [], **facts},
        "tables": tables,
        "output": {OUTPUT: expected},
    }
    cli, other, _ = _run_both(tmp_path, root, module, case, real_engine)
    assert (cli, other) == ([], [])


def test_filtered_query_reads_compact_rows_like_explicit_tuples(tmp_path, real_engine):
    root, module = _filtered(tmp_path)
    compact = {
        "name": "compact",
        "period": "2026-01",
        "input": {PAIRS: [BOB], TIES: [], AGE: "01"},
        "output": {OUTPUT: 1},
    }
    cli, other, _ = _run_both(tmp_path, root, module, compact, real_engine)
    assert (cli, other) == ([], [])


def test_filtered_compact_rows_keep_their_declared_kind(tmp_path, real_engine):
    root, module = _filtered(tmp_path)
    payload = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    case = {
        "period": "2026-01",
        "input": {PAIRS: [BOB], TIES: [], AGE: "01"},
        "output": {OUTPUT: 1},
    }
    cli, pipeline = _requests(tmp_path, copy.deepcopy(case), payload)
    assert cli == pipeline
    assert {(r["entity"], r["entity_id"]) for r in cli["dataset"]["inputs"]} == {
        ("Person", "bob"),
        ("Person", "case"),
    }
    assert [r["tuple"] for r in cli["dataset"]["relations"]] == [["case", "bob"]]


@pytest.mark.parametrize(
    "expected,passes", [(1, True), (0, False)], ids=["right", "wrong"]
)
def test_generated_row_keeps_a_root_id_the_alias_shares(
    tmp_path, real_engine, expected, passes
):
    root, module = _filtered(tmp_path, predicate="candidate == anchor")
    case = {
        "name": "generated",
        "period": "2026-01",
        "entity_id": "related_0",
        "input": {PAIRS: [{}], TIES: []},
        "output": {OUTPUT: expected},
    }
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert (cli == [], other == []) == (passes, passes)
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert [r["tuple"] for r in requests[0]["dataset"]["relations"]] == [
        ["related_0", "related_0"]
    ]


def test_generated_table_row_keeps_a_root_id_the_alias_shares(tmp_path, real_engine):
    root, module = _filtered(tmp_path, predicate="anchor.age == anchor.age")
    case = {
        "name": "table",
        "period": "2026-01",
        "entity_id": "person-1",
        "input": {PAIRS: [{"tuple": ["person-1", "bob"]}], TIES: []},
        "tables": {"Person": [{AGE: "01"}]},
        "output": {OUTPUT: 1},
    }
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert (cli, other) == ([], [])
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert _ids_by_name(requests[0]) == {(AGE, "person-1")}


def test_filtered_query_refuses_a_relation_row_without_a_direction(tmp_path):
    _filtered(tmp_path, predicate="candidate.age == anchor.age")
    payload = _program()
    payload["program"]["derived"][0]["entity"] = "SnapUnit"
    derivation = payload["program"]["relations"][-1]["derivation"]
    derivation.update(entity="SnapUnit", member_relation="selected")
    payload["program"]["derived"][0]["expr"] = {
        "kind": "literal",
        "value": {"kind": "integer", "value": 0},
    }
    case = {"input": {PAIRS: [BOB], TIES: []}, "output": {OUTPUT: 0}}
    with pytest.raises(ValueError, match="no unambiguous query-anchor direction"):
        _requests(tmp_path, case, payload)


def test_pipeline_dataset_refuses_a_filtered_row_without_a_direction(tmp_path):
    root, _ = _filtered(tmp_path)
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=tmp_path / "absent-engine",
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    with pytest.raises(ValueError, match="no unambiguous"):
        pipeline._build_rulespec_dataset(
            {"pairs": [BOB]},
            period={"start": "2026-01-01", "end": "2026-01-31"},
            query_entity="SnapUnit",
            query_entity_id="case",
            declared_relation_names={"pairs"},
            declared_relation_slots={"pairs": ("Person", "Person")},
            relation_role_program=True,
            query_kinds=frozenset({"SnapUnit", "Person"}),
        )


def test_a_foreign_kind_stays_off_the_filtered_root(tmp_path, real_engine):
    root, module = _filtered(tmp_path, formula="len(selected) + other")
    case = {
        "name": "foreign",
        "period": "2026-01",
        "input": {PAIRS: EXPLICIT_PAIR, TIES: [], AGE: "01", OTHER: 1},
        "tables": {"Person": [BOB]},
        "output": {OUTPUT: 2},
    }
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert "missing input" in cli[0]["message"]
    assert any("missing input" in issue for issue in other)
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert _ids_by_name(requests[0]) == {
        (AGE, "case"),
        (AGE, "bob"),
        (OTHER, "case:entity:Household"),
    }


def test_filtered_alias_follows_the_declared_current_slot(tmp_path, real_engine):
    root, module = _filtered(
        tmp_path,
        predicate="candidate.age == anchor.size",
        formula="len(selected)",
    )
    declaration = yaml.safe_load(module.read_text())
    declaration["inputs"] = [
        {"name": "age", "dtype": "Text", "entity": "Person"},
        {"name": "size", "dtype": "Text", "entity": "Household"},
    ]
    declaration["rules"][0]["data_relation"]["arguments"] = ["Household", "Person"]
    declaration["rules"][2]["derived_relation"]["slot_entities"] = [
        "Household",
        "Person",
    ]
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration)
    )
    case = {
        "name": "mixed",
        "period": "2026-01",
        "input": {PAIRS: EXPLICIT_PAIR, TIES: [], SIZE: "01", AGE: "01"},
        "tables": {"Person": [BOB]},
        "output": {OUTPUT: 1},
    }
    payload = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    assert _query_kinds("SnapUnit", payload["program"]["relations"]) == {
        "SnapUnit",
        "Household",
    }
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert _ids_by_name(requests[0]) == {
        (SIZE, "case"),
        (AGE, "case:entity:Person"),
        (AGE, "bob"),
    }


def _legacy_filtered_program():
    payload = _program()
    payload["program"]["derived"][0]["entity"] = "SnapUnit"
    derivation = payload["program"]["relations"][-1]["derivation"]
    derivation.update(entity="SnapUnit", member_relation="selected")
    derivation["predicate"] = {
        "kind": "literal",
        "value": {"kind": "bool", "value": True},
    }
    return payload


def _legacy_module(tmp_path):
    _, module = _fixture(tmp_path, predicate="age > 0")
    declaration = yaml.safe_load(module.read_text())
    declaration["inputs"] = [{"name": "age", "dtype": "Integer", "entity": "Person"}]
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration)
    )


def test_legacy_filtered_requests_keep_their_bytes(tmp_path):
    _legacy_module(tmp_path)
    case = {"entity_id": "case", "input": {AGE: 1}, "output": {OUTPUT: 0}}
    cli, pipeline = _requests(tmp_path, case, _legacy_filtered_program())
    assert cli == pipeline
    assert json.dumps(cli) == json.dumps(
        {
            "mode": "explain",
            "dataset": {
                "inputs": [
                    {
                        "name": AGE,
                        "entity": "Person",
                        "entity_id": "case:entity:Person",
                        "interval": {"start": "2026-01-01", "end": "2026-01-31"},
                        "value": {"kind": "integer", "value": 1},
                    }
                ],
                "relations": [],
            },
            "queries": [
                {
                    "entity_id": "case",
                    "period": {
                        "period_kind": "month",
                        "start": "2026-01-01",
                        "end": "2026-01-31",
                    },
                    "outputs": [OUTPUT],
                }
            ],
        }
    )


def test_legacy_filtered_compact_rows_keep_their_refusal(tmp_path):
    _legacy_module(tmp_path)
    case = {"input": {PAIRS: [{"id": "bob"}]}, "output": {OUTPUT: 0}}
    with pytest.raises(ValueError, match="neither of which matches query entity"):
        _requests(tmp_path, case, _legacy_filtered_program())


SNAP_ON_PERSON = {"SnapUnit": "Person"}
BOTH_ON_PERSON = {"SnapUnit": "Person", "OtherUnit": "Person"}
OTHER_LINEAGE = {"SnapUnit": "Household", "OtherUnit": "Person"}


@pytest.mark.parametrize(
    "occupied,entity_kinds",
    [
        ({"case": {"SnapUnit", "Household"}}, SNAP_ON_PERSON),
        ({"case": {"SnapUnit", "Person", "Household"}}, SNAP_ON_PERSON),
        ({"case": {"Household", "Tax"}}, SNAP_ON_PERSON),
        ({"case": {"SnapUnit", "Person"}}, None),
        ({"case": {"SnapUnit", "OtherUnit"}}, OTHER_LINEAGE),
        ({"case": {"OtherUnit", "Household"}}, OTHER_LINEAGE),
    ],
)
def test_distinct_id_check_refuses_kinds_outside_the_alias(occupied, entity_kinds):
    with pytest.raises(ValueError, match="is shared by"):
        require_distinct_entity_ids(occupied, entity_kinds=entity_kinds)


@pytest.mark.parametrize(
    "occupied,entity_kinds",
    [
        ({"case": {"SnapUnit", "Person"}, "bob": {"Household"}}, SNAP_ON_PERSON),
        ({"case": {"SnapUnit", "OtherUnit"}}, BOTH_ON_PERSON),
        ({"case": {"SnapUnit", "OtherUnit", "Person"}}, BOTH_ON_PERSON),
        ({"case": {"OtherUnit", "Person"}, "unit": {"SnapUnit"}}, OTHER_LINEAGE),
    ],
)
def test_distinct_id_check_accepts_every_name_of_one_kind(occupied, entity_kinds):
    require_distinct_entity_ids(occupied, entity_kinds=entity_kinds)


def _allocated(entity, occupied, entity_kinds):
    ids, facts = ["case"], [{"entity_id": "case"}]
    assign_generated_fixture_ids(
        [(entity, ids, 0, facts)], occupied, entity_kinds=entity_kinds
    )
    assert facts[0]["entity_id"] == ids[0]
    return ids[0]


@pytest.mark.parametrize(
    "entity,occupied,entity_kinds,expected",
    [
        ("Person", {"SnapUnit"}, SNAP_ON_PERSON, "case"),
        ("SnapUnit", {"Person"}, SNAP_ON_PERSON, "case"),
        ("OtherUnit", {"SnapUnit"}, BOTH_ON_PERSON, "case"),
        ("Person", {"OtherUnit"}, OTHER_LINEAGE, "case"),
        ("Household", {"SnapUnit"}, SNAP_ON_PERSON, "case:entity:Household"),
        ("Person", {"Household"}, SNAP_ON_PERSON, "case:entity:Person"),
        ("Person", {"SnapUnit"}, None, "case:entity:Person"),
        ("Person", {"SnapUnit"}, OTHER_LINEAGE, "case:entity:Person"),
    ],
)
def test_generated_ids_share_an_id_only_across_one_kind(
    entity, occupied, entity_kinds, expected
):
    assert _allocated(entity, {"case": occupied}, entity_kinds) == expected


def _query_kinds(query, relations):
    return query_entity_kinds(query, filtered_entity_kinds(relations))


def _relation(alias, slots, current=0, **extra):
    return {
        "name": f"{alias}-{len(slots)}",
        "slot_entities": slots,
        "derivation": {
            "entity": alias,
            "current_slot": current,
            "slot_entities": slots,
            "predicate": {"kind": "role_input", "term": "candidate", "name": "x"},
            **extra,
        },
    }


class _BoundedSources(dict):
    """Stop a walk that misses a cycle, which would otherwise never end."""

    reads = 0

    def get(self, key, default=None):
        self.reads += 1
        assert self.reads <= 6, "the slot-kind walk did not stop at its cycle"
        return super().get(key, default)


def test_a_source_cycle_settles_no_slot_kinds():
    first, second = _inheriting("One", "two"), _inheriting("Two", "one")
    first["name"], second["name"] = "one", "two"
    assert _slot_kinds(first, _BoundedSources(one=first, two=second)) == []
    assert filtered_entity_kinds([first, second]) == {}


@pytest.mark.parametrize(
    "relation",
    [
        pytest.param(
            {
                "derivation": {
                    "entity": "SnapUnit",
                    "current_slot": "0",
                    "slot_entities": ["Person", "Person"],
                }
            },
            id="text-slot",
        ),
        pytest.param(
            {
                "derivation": {
                    "entity": "SnapUnit",
                    "current_slot": 2,
                    "slot_entities": ["Person", "Person"],
                }
            },
            id="slot-past-the-end",
        ),
        pytest.param(
            {"derivation": {"entity": "SnapUnit", "current_slot": 0}},
            id="no-slot-kinds",
        ),
    ],
)
def test_unusable_alias_coordinates_add_no_kind(relation):
    relations = [relation, _relation("Other", ["Person", "Person"])]
    assert _query_kinds("SnapUnit", relations) == {"SnapUnit"}


def test_alias_reads_slot_kinds_from_the_relation_when_the_derivation_omits_them():
    relation = _relation("SnapUnit", ["Person", "Person"])
    del relation["derivation"]["slot_entities"]
    assert _query_kinds("SnapUnit", [relation]) == {"SnapUnit", "Person"}


def _inheriting(alias, source):
    relation = _relation(alias, [], source_relation=source)
    del relation["slot_entities"], relation["derivation"]["slot_entities"]
    return relation


def test_alias_inherits_slot_kinds_through_its_sources():
    base = {"name": "pairs", "slot_entities": ["Household", "Person"]}
    middle = {**_inheriting("Unused", "pairs"), "name": "middle"}
    assert filtered_entity_kinds([base, middle, _inheriting("SnapUnit", "middle")]) == {
        "Unused": "Household",
        "SnapUnit": "Household",
    }


def test_a_derived_declaration_settles_slot_kinds_before_its_source():
    base = {"name": "pairs", "slot_entities": ["Household", "Person"]}
    alias = _relation("SnapUnit", ["Person", "Household"], source_relation="pairs")
    assert filtered_entity_kinds([base, alias]) == {"SnapUnit": "Person"}


def test_a_derivation_settles_slot_kinds_before_its_own_relation():
    alias = _relation("SnapUnit", ["Person", "Person"])
    alias["slot_entities"] = ["Household", "Household"]
    assert filtered_entity_kinds([alias]) == {"SnapUnit": "Person"}


def test_a_path_source_never_inherits_slot_kinds():
    base = {"name": "pairs", "slot_entities": ["Household", "Person"]}
    alias = _inheriting("SnapUnit", "pairs")
    alias["derivation"]["source_path"] = [{"relation": "pairs"}]
    assert filtered_entity_kinds([base, alias]) == {}


def test_a_filter_chain_ends_at_its_innermost_kind():
    relations = [
        _relation("Outer", ["Inner", "Inner"]),
        _relation("Inner", ["Person", "Person"]),
    ]
    assert filtered_entity_kinds(relations) == {"Outer": "Person", "Inner": "Person"}


def test_a_self_filter_is_its_own_kind():
    relations = [
        _relation("Outer", ["Household", "Household"]),
        _relation("Household", ["Household", "Household"]),
    ]
    assert filtered_entity_kinds(relations) == {"Outer": "Household"}


def test_an_alias_with_disagreeing_relations_stays_unmapped():
    relations = [
        _relation("SnapUnit", ["Person", "Person"]),
        _relation("SnapUnit", ["Household", "Household"]),
        _relation("Outer", ["SnapUnit", "SnapUnit"]),
    ]
    assert filtered_entity_kinds(relations) == {"Outer": "SnapUnit"}


def test_an_alias_cycle_stays_unmapped():
    relations = [
        _relation("One", ["Two", "Two"]),
        _relation("Two", ["One", "One"]),
        _relation("Three", ["One", "One"]),
    ]
    assert filtered_entity_kinds(relations) == {}


def test_query_kinds_cover_every_name_of_the_query_kind():
    relations = [
        _relation("SnapUnit", ["Person", "Person"]),
        _relation("OtherUnit", ["Person", "Person"]),
        _relation("HouseUnit", ["Household", "Household"]),
    ]
    kinds = {"SnapUnit", "OtherUnit", "Person"}
    assert _query_kinds("SnapUnit", relations) == kinds
    assert _query_kinds("OtherUnit", relations) == kinds
    assert _query_kinds("Person", relations) == kinds
    assert _query_kinds("Household", relations) == {"HouseUnit", "Household"}


def test_query_kinds_ignore_missing_relations():
    assert filtered_entity_kinds(None) == {}
    assert filtered_entity_kinds([]) == {}
    assert query_entity_kinds("Person", {}) == {"Person"}


def _two_aliases(
    tmp_path, *, query_slots, other_slots, predicate, formula, input_dtype
):
    """Two filtered entities, each over a source its own slot kinds name."""
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/dataset_roles.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)

    def filtered(name, source, slots, entity, member, condition):
        return {
            "name": name,
            "kind": "derived_relation",
            "derived_relation": {
                "arity": 2,
                "slot_entities": slots,
                "current_slot": 0,
                "related_slot": 1,
                "source_relation": source,
                "entity": entity,
                "member_relation": member,
            },
            "versions": [{"effective_from": "2026-01-01", "formula": condition}],
        }

    def derived(name, entity, condition, dtype):
        return {
            "name": name,
            "kind": "derived",
            "entity": entity,
            "dtype": dtype,
            "versions": [{"effective_from": "2026-01-01", "formula": condition}],
        }

    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "inputs": [
                    {"name": "age", "entity": "OtherUnit", "dtype": input_dtype}
                ],
                "rules": [
                    {
                        "name": "pairs",
                        "kind": "data_relation",
                        "data_relation": {"arity": 2, "arguments": query_slots},
                    },
                    {
                        "name": "ties",
                        "kind": "data_relation",
                        "data_relation": {"arity": 2, "arguments": other_slots},
                    },
                    filtered(
                        "members",
                        "pairs",
                        query_slots,
                        "SnapUnit",
                        "selected",
                        predicate,
                    ),
                    derived("member_count", "SnapUnit", formula, "Integer"),
                    filtered(
                        "others",
                        "ties",
                        other_slots,
                        "OtherUnit",
                        "other_selected",
                        True,
                    ),
                    derived("other_value", "OtherUnit", "age", input_dtype),
                ],
            }
        )
    )
    return root, module


PERSON_PAIR = ["Person", "Person"]


@pytest.mark.parametrize("surface", ["table", "flat"])
def test_a_sibling_alias_fact_stays_on_the_shared_id(tmp_path, real_engine, surface):
    root, module = _two_aliases(
        tmp_path,
        query_slots=PERSON_PAIR,
        other_slots=PERSON_PAIR,
        predicate="candidate == anchor",
        formula="len(selected) + other_value",
        input_dtype="Integer",
    )
    case = {
        "name": "sibling",
        "entity_id": "related_0",
        "period": "2026-01",
        "input": {PAIRS: [{}], TIES: []},
        "output": {OUTPUT: 3},
    }
    if surface == "table":
        case["tables"] = {"OtherUnit": [{"id": "related_0", AGE: 2}]}
    else:
        case["input"][AGE] = 2
    assert _run_both(tmp_path, root, module, case, real_engine)[:2] == ([], [])
    case["output"] = {OUTPUT: 2}
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert cli and other
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert requests[0]["dataset"]["relations"][0]["tuple"] == ["related_0", "related_0"]
    assert _ids_by_name(requests[0]) == {(AGE, "related_0")}


def test_a_generated_id_joins_an_alias_outside_the_query_kind(tmp_path, real_engine):
    root, module = _two_aliases(
        tmp_path,
        query_slots=["Household", "Person"],
        other_slots=PERSON_PAIR,
        predicate="candidate.age == candidate.age",
        formula="len(selected)",
        input_dtype="Text",
    )
    case = {
        "name": "outside",
        "period": "2026-01",
        "input": {PAIRS: [{}], TIES: []},
        "tables": {"OtherUnit": [{"id": "related_0", AGE: "x"}]},
        "output": {OUTPUT: 1},
    }
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert (cli, other) == ([], [])
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert requests[0]["dataset"]["relations"][0]["tuple"] == ["case", "related_0"]


def test_an_alias_of_another_kind_stays_off_a_shared_id(tmp_path, real_engine):
    root, module = _two_aliases(
        tmp_path,
        query_slots=["Household", "Person"],
        other_slots=PERSON_PAIR,
        predicate="candidate.age == candidate.age",
        formula="len(selected)",
        input_dtype="Text",
    )
    case = {
        "name": "foreign",
        "entity_id": "shared",
        "period": "2026-01",
        "input": {PAIRS: EXPLICIT_PAIR, TIES: []},
        "tables": {"OtherUnit": [{"id": "shared", AGE: "x"}]},
        "output": {OUTPUT: 0},
    }
    payload = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    with pytest.raises(ValueError, match="is shared by"):
        _requests(tmp_path, case, payload)


KINDS = st.sampled_from(["A", "B", "C", "D", "E"])


FIXTURE_IDS = st.sampled_from(["a", "b", "c"])
OCCUPIED = st.dictionaries(FIXTURE_IDS, KINDS, max_size=3)
GENERATED = st.lists(st.tuples(KINDS, FIXTURE_IDS), max_size=6)


ENTITY_KINDS = st.dictionaries(KINDS, st.sampled_from(["P", "Q"]), max_size=5)


def _allocate_all(occupied, rows, entity_kinds):
    placed = {entity_id: {kind} for entity_id, kind in occupied.items()}
    generated = [(kind, [preferred], 0, []) for kind, preferred in rows]
    assign_generated_fixture_ids(generated, placed, entity_kinds=entity_kinds)
    return placed, [ids[0] for _, ids, _, _ in generated]


def _base_allocation(occupied, rows):
    """Literal copy of the allocator on main, which keeps every kind apart."""
    placed = {entity_id: {kind} for entity_id, kind in occupied.items()}
    allocated = []
    for entity, preferred in rows:
        reserved = {
            entity_id for entity_id, kinds in placed.items() if kinds - {entity}
        }
        entity_id = preferred
        if preferred in reserved:
            base = entity_id = f"{preferred}:entity:{entity}"
            suffix = 1
            while entity_id in reserved:
                entity_id, suffix = f"{base}:{suffix}", suffix + 1
        allocated.append(entity_id)
        placed.setdefault(entity_id, set()).add(entity)
    return placed, allocated


@settings(deadline=None)
@given(OCCUPIED, GENERATED, ENTITY_KINDS)
def test_allocation_never_makes_an_id_the_check_refuses(occupied, rows, entity_kinds):
    placed, _ = _allocate_all(occupied, rows, entity_kinds)
    require_distinct_entity_ids(placed, entity_kinds=entity_kinds)


@settings(deadline=None)
@given(OCCUPIED, GENERATED)
def test_no_alias_map_keeps_the_base_allocation(occupied, rows):
    assert _allocate_all(occupied, rows, {}) == _base_allocation(occupied, rows)


@settings(deadline=None)
@given(st.dictionaries(st.sampled_from("BCDE"), KINDS, max_size=4), KINDS)
def test_query_kinds_match_the_innermost_kind(parents, query):
    forest = {alias: parent for alias, parent in parents.items() if parent < alias}
    relations = [_relation(alias, [parent, parent]) for alias, parent in forest.items()]

    def innermost(kind):
        while kind in forest:
            kind = forest[kind]
        return kind

    assert filtered_entity_kinds(relations) == {
        alias: innermost(alias) for alias in forest
    }
    assert _query_kinds(query, relations) == {
        query,
        innermost(query),
        *(alias for alias in forest if innermost(alias) == innermost(query)),
    }


@settings(deadline=None)
@given(st.lists(st.tuples(KINDS, KINDS), max_size=8), KINDS)
def test_legacy_programs_keep_one_query_kind(edges, query):
    relations = [
        {
            "derivation": {
                "entity": alias,
                "current_slot": 0,
                "slot_entities": [underlying, underlying],
            }
        }
        for alias, underlying in edges
    ]
    assert filtered_entity_kinds(relations) == {}
    assert _query_kinds(query, relations) == {query}


@settings(deadline=None)
@given(KINDS, st.frozensets(KINDS), st.sampled_from(["case", "alice"]))
def test_alias_kinds_share_the_root_and_others_never_do(entity, query_kinds, root):
    identifier = flat_input_entity_id(
        entity, "SnapUnit", root, query_kinds=query_kinds | {"SnapUnit"}
    )
    assert (identifier == root) == (entity in query_kinds)


@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    root_id=st.sampled_from(["case", "alice"]),
    facts=st.dictionaries(
        st.sampled_from([AGE, BONUS, OTHER]),
        st.integers(min_value=0, max_value=9),
        max_size=3,
    ),
)
def test_both_runners_place_each_kind_alike(tmp_path_factory, root_id, facts):
    tmp_path = tmp_path_factory.mktemp("filtered")
    _filtered(tmp_path, formula="len(selected) + bonus + other")
    payload = _program()
    payload["program"]["derived"][0]["entity"] = "SnapUnit"
    payload["program"]["relations"][-1]["derivation"].update(
        entity="SnapUnit", member_relation="selected"
    )
    case = {
        "entity_id": root_id,
        "input": {PAIRS: [{"tuple": [root_id, "bob"]}], TIES: [], **facts},
        "output": {OUTPUT: 1},
    }
    cli, pipeline = _requests(tmp_path, case, payload)
    assert cli == pipeline
    placed = dict(_ids_by_name(cli))
    for name in {AGE, BONUS}.intersection(facts):
        assert placed[name] == root_id
    if OTHER in facts:
        assert placed[OTHER] == f"{root_id}:entity:Household"


def _household_people(tmp_path):
    """Give a household two relations, one to households and one to people."""
    root, module = _fixture(tmp_path, predicate=CANDIDATE_PREDICATE)
    declaration = yaml.safe_load(module.read_text())
    declaration["rules"][0]["data_relation"]["arguments"] = ["Household", "Person"]
    declaration["rules"][1]["data_relation"]["arguments"] = ["Household", "Household"]
    declaration["rules"][2]["derived_relation"]["slot_entities"] = [
        "Household",
        "Person",
    ]
    declaration["rules"][3]["entity"] = "Household"
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(declaration)
    )
    return root, module


def test_an_explicit_tuple_reserves_its_ids_by_slot_kind():
    occupied = {"case": {"Household"}}
    reserve_explicit_tuple_ids(
        occupied, [(["case", "related_0"], ("Household", "Household"))]
    )
    assert occupied == {"case": {"Household"}, "related_0": {"Household"}}
    row, records = ["related_0"], [{"entity_id": "related_0"}]
    assign_generated_fixture_ids([("Person", row, 0, records)], occupied)
    assert row == ["related_0:entity:Person"]
    assert records == [{"entity_id": "related_0:entity:Person"}]


@pytest.mark.parametrize("surface", ["compact", "table"])
def test_a_generated_id_avoids_a_foreign_explicit_tuple(tmp_path, real_engine, surface):
    root, module = _household_people(tmp_path)
    person = "related_0" if surface == "compact" else "person-1"
    case = {
        "name": "reserved",
        "period": "2026-01",
        "input": {
            TIES: [{"tuple": ["case", person]}],
            PAIRS: [{AGE: 7}] if surface == "compact" else [],
        },
        "output": {OUTPUT: 1 if surface == "compact" else 0},
    }
    if surface == "table":
        case["tables"] = {"Person": [{AGE: 7}]}
    cli, other, payload = _run_both(tmp_path, root, module, case, real_engine)
    assert (cli, other) == ([], [])
    requests = _requests(tmp_path, copy.deepcopy(case), payload)
    assert requests[0] == requests[1]
    assert _ids_by_name(requests[0]) == {(AGE, f"{person}:entity:Person")}


def test_a_generated_id_joins_a_same_kind_explicit_tuple(tmp_path, real_engine):
    root, module = _household_people(tmp_path)
    case = {
        "name": "joined",
        "period": "2026-01",
        "input": {PAIRS: [{"tuple": ["case", "person-1"]}], TIES: []},
        "tables": {"Person": [{AGE: 7}]},
        "output": {OUTPUT: 1},
    }
    assert _run_both(tmp_path, root, module, case, real_engine)[:2] == ([], [])


def test_an_id_two_explicit_tuples_share_stays_the_engines_call(tmp_path, real_engine):
    root, module = _household_people(tmp_path)
    case = {
        "name": "shared",
        "period": "2026-01",
        "input": {
            PAIRS: [{"tuple": ["case", "shared"]}],
            TIES: [{"tuple": ["case", "shared"]}],
        },
        "tables": {"Person": [{"id": "shared", AGE: 7}]},
        "output": {OUTPUT: 1},
    }
    cli, other, _ = _run_both(tmp_path, root, module, case, real_engine)
    assert "strict dataset relation entity validation failed" in cli[0]["message"]
    assert "strict dataset relation entity validation failed" in other[0]

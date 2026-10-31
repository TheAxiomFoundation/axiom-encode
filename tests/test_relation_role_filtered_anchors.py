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
)
from axiom_encode.companion_relations import query_entity_kinds
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
    assert query_entity_kinds("SnapUnit", payload["program"]["relations"]) == {
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


@pytest.mark.parametrize(
    "occupied,query_kinds",
    [
        ({"case": {"SnapUnit", "Household"}}, {"SnapUnit", "Person"}),
        ({"case": {"SnapUnit", "Person", "Household"}}, {"SnapUnit", "Person"}),
        ({"case": {"Household", "Tax"}}, {"SnapUnit", "Person"}),
        ({"case": {"SnapUnit", "Person"}}, ()),
    ],
)
def test_distinct_id_check_refuses_kinds_outside_the_alias(occupied, query_kinds):
    with pytest.raises(ValueError, match="is shared by"):
        require_distinct_entity_ids(occupied, query_kinds=query_kinds)


def test_distinct_id_check_accepts_a_filtered_alias_and_its_anchor():
    require_distinct_entity_ids(
        {"case": {"SnapUnit", "Person"}, "bob": {"Household"}},
        query_kinds={"SnapUnit", "Person"},
    )


def _allocated(entity, occupied, query_kinds):
    ids, facts = ["case"], [{"entity_id": "case"}]
    assign_generated_fixture_ids(
        [(entity, ids, 0, facts)], occupied, query_kinds=query_kinds
    )
    assert facts[0]["entity_id"] == ids[0]
    return ids[0]


@pytest.mark.parametrize(
    "entity,occupied,query_kinds,expected",
    [
        ("Person", {"SnapUnit"}, {"SnapUnit", "Person"}, "case"),
        ("SnapUnit", {"Person"}, {"SnapUnit", "Person"}, "case"),
        ("Household", {"SnapUnit"}, {"SnapUnit", "Person"}, "case:entity:Household"),
        ("Person", {"Household"}, {"SnapUnit", "Person"}, "case:entity:Person"),
        ("Person", {"SnapUnit"}, (), "case:entity:Person"),
        ("Person", {"SnapUnit"}, {"SnapUnit"}, "case:entity:Person"),
    ],
)
def test_generated_ids_share_a_root_only_across_query_kinds(
    entity, occupied, query_kinds, expected
):
    assert _allocated(entity, {"case": occupied}, query_kinds) == expected


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
    assert query_entity_kinds("SnapUnit", relations) == {"SnapUnit"}


def test_alias_reads_slot_kinds_from_the_relation_when_the_derivation_omits_them():
    relation = _relation("SnapUnit", ["Person", "Person"])
    del relation["derivation"]["slot_entities"]
    assert query_entity_kinds("SnapUnit", [relation]) == {"SnapUnit", "Person"}


def test_query_kinds_ignore_missing_relations():
    assert query_entity_kinds("Person", None) == {"Person"}
    assert query_entity_kinds("Person", []) == {"Person"}


KINDS = st.sampled_from(["A", "B", "C", "D", "E"])


FIXTURE_IDS = st.sampled_from(["a", "b", "c"])
OCCUPIED = st.dictionaries(FIXTURE_IDS, KINDS, max_size=3)
GENERATED = st.lists(st.tuples(KINDS, FIXTURE_IDS), max_size=6)


def _allocate_all(occupied, rows, query_kinds):
    placed = {entity_id: {kind} for entity_id, kind in occupied.items()}
    generated = [(kind, [preferred], 0, []) for kind, preferred in rows]
    assign_generated_fixture_ids(generated, placed, query_kinds=query_kinds)
    return placed, [ids[0] for _, ids, _, _ in generated]


@given(OCCUPIED, GENERATED, st.frozensets(KINDS))
def test_allocation_never_makes_an_id_the_check_refuses(occupied, rows, query_kinds):
    placed, _ = _allocate_all(occupied, rows, query_kinds)
    require_distinct_entity_ids(placed, query_kinds=query_kinds)


@given(OCCUPIED, GENERATED, KINDS)
def test_one_query_kind_keeps_the_legacy_allocation(occupied, rows, query):
    assert _allocate_all(occupied, rows, {query}) == _allocate_all(occupied, rows, ())


@given(st.lists(st.tuples(KINDS, KINDS), max_size=8), KINDS)
def test_query_kinds_equal_alias_reachability(edges, query):
    relations = [
        _relation(alias, [underlying, underlying]) for alias, underlying in edges
    ]
    reachable = {query}
    for _ in edges:
        reachable |= {b for a, b in edges if a in reachable}
    assert query_entity_kinds(query, relations) == reachable


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
    assert query_entity_kinds(query, relations) == {query}


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

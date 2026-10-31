import pytest

from axiom_encode.companion_relations import (
    executable_relation_directions,
    explicit_relation_tuples,
    pair_relation_assignment_issues,
)


def fixture(current=1):
    return {
        "count": {
            "id": "count",
            "entity": "TaxUnit",
            "expr": {
                "kind": "count_related",
                "relation": "members",
                "current_slot": current,
                "related_slot": 1 - current,
                "where": {"kind": "derived", "name": "child"},
            },
        },
        "child": {"id": "child", "entity": "Person", "expr": {"kind": "literal"}},
    }


def resolve(d, outputs=None):
    return executable_relation_directions(
        d,
        outputs or ["count"],
        {"start": "2024-01-01", "end": "2024-12-31"},
        "TaxUnit",
        {"members": ("TaxUnit", "Person")},
    )


@pytest.mark.parametrize("current", [0, 1])
def test_executable_direction_overrides_declaration(current):
    assert resolve(fixture(current)) == {"members": (current, "Person")}


def test_malformed_coordinates_fail_closed():
    d = fixture()
    d["count"]["expr"]["current_slot"] = True
    with pytest.raises(ValueError, match="coordinates"):
        resolve(d)


def test_conflicting_reachable_directions_fail_closed():
    d = fixture()
    d["other"] = fixture(0)["count"]
    d["other"]["id"] = "other"
    with pytest.raises(ValueError, match="conflicting executable"):
        resolve(d, ["count", "other"])


def test_inactive_version_does_not_override_active_direction():
    d = fixture()
    d["count"]["versions"] = [
        {
            "effective_from": "2020-01-01",
            "effective_to": "2023-12-31",
            "expr": fixture(0)["count"]["expr"],
        },
        {"effective_from": "2024-01-01", "expr": fixture(1)["count"]["expr"]},
    ]
    assert resolve(d) == {"members": (1, "Person")}


def test_unreachable_conflict_is_ignored():
    d = fixture()
    d["other"] = fixture(0)["count"]
    d["other"]["id"] = "other"
    assert resolve(d) == {"members": (1, "Person")}


def test_latest_open_ended_version_wins_at_period_start():
    d = fixture()
    d["count"]["versions"] = [
        {"effective_from": "2020-01-01", "expr": fixture(0)["count"]["expr"]},
        {"effective_from": "2024-01-01", "expr": fixture(1)["count"]["expr"]},
        {"effective_from": "2024-06-01", "expr": fixture(0)["count"]["expr"]},
    ]
    assert resolve(d) == {"members": (1, "Person")}


def test_equal_start_is_ambiguous():
    d = fixture()
    d["count"]["versions"] = [
        {"effective_from": "2024-01-01", "expr": fixture(i)["count"]["expr"]}
        for i in [0, 1]
    ]
    with pytest.raises(ValueError, match="ambiguous effective"):
        resolve(d)


def test_untyped_unconditional_count_retains_entity_fallback():
    d = fixture()
    del d["count"]["expr"]["where"]
    assert executable_relation_directions(
        d, ["count"], {"start": "2024-01-01", "end": "2024-12-31"}, "TaxUnit", {}
    ) == {"members": (1, None)}


def test_same_entity_uses_explicit_direction():
    d = fixture(0)
    d["child"]["entity"] = "TaxUnit"
    assert executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01", "end": "2024-12-31"},
        "TaxUnit",
        {"members": ("TaxUnit", "TaxUnit")},
    ) == {"members": (0, "TaxUnit")}


def test_qualified_relation_uses_exact_identity():
    d = fixture()
    d["count"]["expr"]["relation"] = "us:statutes/example#relation.members"
    assert resolve(d) == {"us:statutes/example#relation.members": (1, "Person")}


def test_nested_aggregation_does_not_become_root_relation():
    d = fixture()
    d["child"]["expr"] = {
        "kind": "count_related",
        "relation": "children",
        "current_slot": 0,
        "related_slot": 1,
        "where": {"kind": "derived", "name": "leaf"},
    }
    d["leaf"] = {"id": "leaf", "entity": "Child", "expr": {"kind": "literal"}}
    assert resolve(d) == {"members": (1, "Person")}


def test_cli_qualified_companion_uses_bare_executable_alias(monkeypatch, tmp_path):
    import json

    from axiom_encode.cli import _execute_rulespec_test_case

    class Captured(Exception):
        pass

    def capture(*args, **kwargs):
        dataset = json.loads(kwargs["input"])["dataset"]
        assert all(
            row["tuple"] == ["related_0", "case"] for row in dataset["relations"]
        )
        assert dataset["inputs"][0]["entity"] == "Person"
        raise Captured

    monkeypatch.setattr("axiom_encode.cli.subprocess.run", capture)
    with pytest.raises(Captured):
        _execute_rulespec_test_case(
            tmp_path / "case.test.yaml",
            {
                "period": "2024-01",
                "input": {"us:statutes/example#relation.members": [{"age": 5}]},
                "output": {"count": 1},
            },
            case_name="alias",
            compiled_path=tmp_path / "compiled.json",
            binary=tmp_path / "engine",
            axiom_rules_path=tmp_path,
            env={},
            parameter_by_id={},
            derived_ids={"count", "child"},
            derived_by_id=fixture(),
            declared_relation_names={"members"},
            declared_relation_slots={"members": ("TaxUnit", "Person")},
            policy_repo_path=tmp_path / "rulespec-us" / "us",
        )


def scalar_half():
    # af6 emits formula-valued SSI parameter 42/1382a/b/4 as this Scalar IR.
    return {
        "id": "half",
        "entity": "Scalar",
        "expr": {
            "kind": "div",
            "left": {"kind": "literal", "value": {"kind": "decimal", "value": "1"}},
            "right": {"kind": "literal", "value": {"kind": "decimal", "value": "2"}},
        },
    }


def test_scalar_arithmetic_does_not_change_entity_or_relation_evidence():
    d = fixture()
    d["half"] = scalar_half()
    d["child"]["expr"] = {"kind": "derived", "name": "half"}
    d["count"]["expr"]["where"] = {
        "kind": "add",
        "left": {"kind": "derived", "name": "child"},
        "right": {"kind": "derived", "name": "half"},
    }
    assert resolve(d) == {"members": (1, "Person")}


def test_scalar_proof_uses_active_version_and_recursive_helpers():
    d = fixture()
    d["half"] = scalar_half()
    d["wrapper"] = {
        "id": "wrapper",
        "entity": "Scalar",
        "expr": {"kind": "derived", "name": "half"},
    }
    d["child"]["expr"] = {"kind": "derived", "name": "wrapper"}
    d["half"]["versions"] = [
        {"effective_from": "2020-01-01", "expr": d["half"]["expr"]},
        {"effective_from": "2025-01-01", "expr": {"kind": "input", "name": "income"}},
    ]
    assert resolve(d) == {"members": (1, "Person")}


def test_scalar_add_uses_compiled_items_shape():
    d = fixture()
    d["half"] = scalar_half()
    d["half"]["expr"] = {"kind": "add", "items": [d["half"]["expr"], d["half"]["expr"]]}
    d["child"]["expr"] = {"kind": "derived", "name": "half"}
    assert resolve(d) == {"members": (1, "Person")}


def test_cross_kind_reference_preserves_root_anchor_and_traverses_relations():
    d = fixture()
    d["outer"] = {
        "id": "outer",
        "entity": "Person",
        "expr": {"kind": "derived", "name": "count"},
    }
    assert resolve(d, ["outer"]) == {"members": (1, "Person")}


def test_scalar_relation_dependency_is_traversed_and_cycles_terminate():
    d = fixture()
    d["outer"] = {
        "id": "outer",
        "entity": "Scalar",
        "expr": {"kind": "derived", "name": "count"},
    }
    d["child"]["expr"] = {"kind": "derived", "name": "outer"}
    assert resolve(d, ["outer"]) == {"members": (1, "Person")}


def test_nested_count_does_not_remap_cross_kind_reference_to_root():
    d = fixture()
    d["inner"] = fixture(0)["count"]
    d["inner"]["id"] = "inner"
    d["inner"]["expr"]["where"] = {"kind": "literal"}
    d["child"]["expr"] = {"kind": "derived", "name": "inner"}
    assert resolve(d) == {"members": (1, "Person")}


def test_derived_relation_predicate_maps_current_anchor_and_drops_context_in_body():
    d = fixture()
    d["extra"] = fixture(0)["count"]
    d["extra"]["id"] = "extra"
    d["extra"]["expr"]["relation"] = "extra_members"
    d["extra"]["expr"]["where"] = {"kind": "literal"}
    relations = [
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw_members",
                "current_slot": 1,
                "related_slot": 0,
                "slot_entities": ["Person", "TaxUnit"],
                "predicate": {"kind": "derived", "name": "extra"},
            },
        }
    ]
    assert executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "TaxUnit",
        {"members": ("TaxUnit", "Person")},
        relations,
    ) == {
        "members": (1, "Person"),
        "raw_members": (1, "Person"),
        "extra_members": (0, None),
    }


def test_derived_relation_same_kind_prefers_current_anchor():
    d = fixture()
    d["count"]["entity"] = "Person"
    d["extra"] = {
        "id": "extra",
        "entity": "Person",
        "expr": {
            "kind": "count_related",
            "relation": "extra",
            "current_slot": 0,
            "related_slot": 1,
            "where": {"kind": "literal"},
        },
    }
    schemas = [
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": {"kind": "derived", "name": "extra"},
            },
        }
    ]
    result = executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person")},
        schemas,
    )
    assert result["extra"] == (0, None)


def test_derived_body_does_not_inherit_relation_predicate_context():
    d = fixture()
    d["extra"] = {
        "id": "extra",
        "entity": "TaxUnit",
        "expr": {
            "kind": "count_related",
            "relation": "extra",
            "current_slot": 0,
            "related_slot": 1,
            "where": {"kind": "literal"},
        },
    }
    d["child"]["expr"] = {"kind": "derived", "name": "extra"}
    schemas = [
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 1,
                "related_slot": 0,
                "slot_entities": ["Person", "TaxUnit"],
                "predicate": {"kind": "derived", "name": "child"},
            },
        }
    ]
    result = executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "TaxUnit",
        {"members": ("TaxUnit", "Person")},
        schemas,
    )
    assert "extra" not in result


def test_relation_member_uses_context_current_anchor():
    d = fixture()
    schemas = [
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 1,
                "related_slot": 0,
                "slot_entities": ["Person", "TaxUnit"],
                "predicate": {
                    "kind": "relation_member",
                    "relation": "verified",
                    "current_slot": 0,
                    "related_slot": 1,
                },
            },
        }
    ]
    result = executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "TaxUnit",
        {"members": ("TaxUnit", "Person"), "verified": ("Person", "TaxUnit")},
        schemas,
    )
    assert result["verified"] == (0, "Person")


@pytest.mark.parametrize("outputs", [["count", "bare"], ["bare", "count"]])
def test_explicit_child_type_precedes_bare_count_fallback(outputs):
    d = fixture()
    d["count"]["entity"] = "Person"
    d["bare"] = {
        "id": "bare",
        "entity": "Person",
        "expr": {
            "kind": "count_related",
            "relation": "members",
            "current_slot": 1,
            "related_slot": 0,
        },
    }
    result = executable_relation_directions(
        d,
        outputs,
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Household")},
    )
    assert result == {"members": (1, "Person")}


def test_bare_count_keeps_declaration_fallback_without_explicit_child():
    d = fixture()
    d["count"]["expr"].pop("where")
    assert resolve(d) == {"members": (1, "Person")}


def test_conflicting_explicit_children_still_fail():
    d = fixture()
    d["second"] = {"id": "second", "entity": "Household", "expr": {"kind": "literal"}}
    d["other"] = fixture()["count"]
    d["other"]["id"] = "other"
    d["other"]["expr"]["where"] = {"kind": "derived", "name": "second"}
    with pytest.raises(ValueError, match="conflicting related entities"):
        resolve(d, ["count", "other"])


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_sum_value_and_predicate_use_fallback_and_traverse_both(reverse):
    d = fixture()
    d["value"] = {"id": "value", "entity": "TaxUnit", "expr": {"kind": "literal"}}
    parts = [
        ("value", {"kind": "derived", "name": "value"}),
        ("where", {"kind": "derived", "name": "child"}),
    ]
    if reverse:
        parts.reverse()
    d["count"]["expr"] = {
        "kind": "sum_related",
        "relation": "members",
        "current_slot": 1,
        "related_slot": 0,
        **dict(parts),
    }
    assert resolve(d) == {"members": (1, "Person")}
    # Both sides remain traversed: malformed nested coordinates cannot hide.
    for key in ("value", "child"):
        original = d[key]["expr"]
        d[key]["expr"] = {
            "kind": "count_related",
            "relation": "nested",
            "current_slot": 0,
            "related_slot": 0,
        }
        with pytest.raises(ValueError, match="coordinates"):
            resolve(d)
        d[key]["expr"] = original


def test_path_source_keeps_only_first_step_at_query_anchor():
    d = fixture(0)
    d["count"]["entity"] = "Person"
    schemas = [
        {
            "name": "members",
            "derivation": {
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "source_path": [
                    {"relation": "roster", "from_slot": 0, "to_slot": 1},
                    {"relation": "roster", "from_slot": 1, "to_slot": 0},
                ],
                "predicate": {"kind": "literal"},
            },
        }
    ]
    assert executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person"), "roster": ("Person", "Household")},
        schemas,
    ) == {"members": (0, "Person"), "roster": (0, "Household")}


@pytest.mark.parametrize("term,at_root", [("anchor", True), ("candidate", False)])
def test_qualified_derived_reads_follow_term_in_same_kind_predicate(term, at_root):
    d = fixture(0)
    d["count"]["entity"] = "Person"
    d["extra"] = {
        "id": "extra",
        "entity": "Person",
        "expr": {
            "kind": "count_related",
            "relation": "extra_members",
            "current_slot": 0,
            "related_slot": 1,
        },
    }
    schemas = [
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": {"kind": "role_derived", "name": "extra", "term": term},
            },
        }
    ]
    result = executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person")},
        schemas,
    )
    assert ("extra_members" in result) is at_root


def test_exists_variable_derived_read_does_not_become_root_relation():
    d = fixture(0)
    d["count"]["entity"] = "Person"
    d["extra"] = {
        "id": "extra",
        "entity": "Person",
        "expr": {
            "kind": "count_related",
            "relation": "extra_members",
            "current_slot": 0,
            "related_slot": 1,
        },
    }
    schemas = [
        {"name": "domain", "arity": 2, "slot_entities": ["Person", "Person"]},
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": {
                    "kind": "exists",
                    "relation": "domain",
                    "terms": ["anchor", {"var": 0}],
                    "variable_slot": 1,
                    "body": {
                        "kind": "role_derived",
                        "name": "extra",
                        "term": {"var": 0},
                    },
                },
            },
        },
    ]
    result = executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person")},
        schemas,
    )
    assert "extra_members" not in result


def role_relations():
    return [
        {"name": "parents", "arity": 2, "slot_entities": ["Person", "Person"]},
        {
            "name": "members",
            "arity": 2,
            "derivation": {
                "predicate": {"kind": "relation_atom", "relation": "parents"},
            },
        },
    ]


def test_relation_roles_require_explicit_empty_pair_facts():
    assert pair_relation_assignment_issues(
        role_relations(), {"input": {}}, "empty"
    ) == [
        "Test case `empty` must assign pair relation `parents`, including `[]` when no tuples hold."
    ]
    assert (
        pair_relation_assignment_issues(
            role_relations(),
            {"input": {"us:statutes/example#relation.parents": []}},
            "empty",
        )
        == []
    )


def test_legacy_filtered_relations_keep_pair_assignment_findings():
    relations = role_relations()
    relations[1]["derivation"]["predicate"] = {"kind": "derived", "name": "eligible"}
    assert pair_relation_assignment_issues(relations, {"input": {}}, "empty") == []


def test_tuple_rows_preserve_declared_order_and_shared_ids():
    assert explicit_relation_tuples(
        [{"tuple": ["child", "parent"]}], relation="parents", slots=("Person", "Person")
    ) == [["child", "parent"]]
    assert (
        explicit_relation_tuples([{"age": 10}], relation="parents", slots=None) is None
    )


@pytest.mark.parametrize(
    "rows",
    [
        [{"tuple": ["child"]}],
        [{"tuple": ["child", 3]}],
        [{"tuple": "child,parent"}],
        [{"tuple": ["child", "parent"], "age": 10}],
        [{"tuple": ["child", "parent"]}, {"age": 10}],
    ],
)
def test_malformed_tuple_rows_fail_closed(rows):
    with pytest.raises(ValueError, match="tuple row"):
        explicit_relation_tuples(rows, relation="parents", slots=("Person", "Person"))


@pytest.mark.parametrize("kind", ["relation_atom", "exists"])
@pytest.mark.parametrize("root_slot", [0, 1])
def test_data_atoms_and_generators_supply_query_anchor_direction(kind, root_slot):
    d = fixture(0)
    d["count"]["entity"] = "Person"
    terms = ["candidate", "candidate"]
    terms[root_slot] = "anchor"
    predicate = {"kind": kind, "relation": "parents", "terms": terms}
    if kind == "exists":
        predicate["terms"][1 - root_slot] = {"var": 0}
        predicate["variable_slot"] = 1 - root_slot
        predicate["body"] = {"kind": "literal"}
    schemas = [
        {"name": "parents", "arity": 2, "slot_entities": ["Person", "Person"]},
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": predicate,
            },
        },
    ]
    assert executable_relation_directions(
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person"), "parents": ("Person", "Person")},
        schemas,
    )["parents"] == (root_slot, "Person")


def test_explicit_tuples_do_not_need_one_root_direction_for_pair_atoms():
    d = fixture(0)
    d["count"]["entity"] = "Person"
    schemas = [
        {"name": "parents", "arity": 2, "slot_entities": ["Person", "Person"]},
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": {
                    "kind": "and",
                    "items": [
                        {
                            "kind": "relation_atom",
                            "relation": "parents",
                            "terms": ["anchor", "candidate"],
                        },
                        {
                            "kind": "relation_atom",
                            "relation": "parents",
                            "terms": ["candidate", "anchor"],
                        },
                    ],
                },
            },
        },
    ]
    args = (
        d,
        ["count"],
        {"start": "2024-01-01"},
        "Person",
        {"members": ("Person", "Person"), "parents": ("Person", "Person")},
        schemas,
    )
    with pytest.raises(
        ValueError, match="conflicting executable directions for parents"
    ):
        executable_relation_directions(*args)
    assert "parents" not in executable_relation_directions(
        *args, explicit_relations={"parents"}
    )


@pytest.mark.parametrize("anchor_term", [True, False])
def test_compact_same_kind_pair_rows_use_atom_direction_or_refuse(
    tmp_path, anchor_term
):
    from axiom_encode.cli import _execute_rulespec_test_case

    d = fixture(0)
    d["count"]["entity"] = "Person"
    schemas = [
        {"name": "parents", "arity": 2, "slot_entities": ["Person", "Person"]},
        {"name": "raw", "arity": 2, "slot_entities": ["Person", "Person"]},
        {
            "name": "members",
            "derivation": {
                "source_relation": "raw",
                "current_slot": 0,
                "related_slot": 1,
                "slot_entities": ["Person", "Person"],
                "predicate": {
                    "kind": "relation_atom",
                    "relation": "parents",
                    "terms": ["anchor" if anchor_term else "candidate", "candidate"],
                },
            },
        },
    ]
    kwargs = dict(
        case_name="pair",
        compiled_path=tmp_path / "compiled.json",
        binary=tmp_path / "engine",
        axiom_rules_path=tmp_path,
        env={},
        parameter_by_id={},
        derived_ids=set(d),
        derived_by_id=d,
        declared_relation_names={"parents", "raw", "members"},
        declared_relation_slots={
            name: ("Person", "Person") for name in ("parents", "raw", "members")
        },
        compiled_relations=schemas,
        policy_repo_path=tmp_path / "rulespec-us/us",
        request_only=True,
    )
    case = {
        "period": "2024-01",
        "input": {
            "us:statutes/example#relation.parents": [
                {"us:statutes/example#input.age": 10}
            ],
            "us:statutes/example#relation.raw": [{"us:statutes/example#input.age": 10}],
        },
        "output": {"count": 1},
    }
    if not anchor_term:
        with pytest.raises(ValueError) as error:
            _execute_rulespec_test_case(tmp_path / "case.test.yaml", case, **kwargs)
        assert str(error.value) == (
            "relation `us:statutes/example#relation.parents` compact rows have no "
            "unambiguous query-anchor direction; use explicit `tuple` rows"
        )
        return
    request = _execute_rulespec_test_case(tmp_path / "case.test.yaml", case, **kwargs)
    tuples = [
        row["tuple"]
        for row in request["dataset"]["relations"]
        if row["name"] == "parents"
    ]
    assert tuples and all(pair[0] == "case" for pair in tuples)

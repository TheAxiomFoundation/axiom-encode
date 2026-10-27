import pytest

from axiom_encode.companion_relations import executable_relation_directions


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

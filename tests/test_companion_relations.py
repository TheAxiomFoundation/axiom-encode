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

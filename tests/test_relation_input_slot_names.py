"""Keep synthetic relation imports distinct from local scalar input slots."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as validation
from tests.test_companion_declared_entities import _compile


def relation(name, *, derived=False, formula="true"):
    declaration = {
        "name": name,
        "kind": "derived_relation" if derived else "data_relation",
    }
    if derived:
        declaration["derived_relation"] = {
            "arity": 2,
            "slot_entities": ["Person", "Person"],
            "current_slot": 0,
            "related_slot": 1,
            "source_relation": "pairs",
        }
        declaration["versions"] = [{"effective_from": "2026-01-01", "formula": formula}]
    else:
        declaration["data_relation"] = {"arity": 2, "arguments": ["Person", "Person"]}
    return declaration


def write_input(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(payload)
    )


def material(
    tmp_path, *, provider_kind="data_relation", formula="provided > 0", prefix="us"
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/consumer.yaml"
    provider = root / prefix / "statutes/provider.yaml"
    if provider_kind == "data_relation":
        exported = relation("provided")
    else:
        exported = {
            "name": "provided",
            "kind": provider_kind,
            "entity": "Person",
            "dtype": "Integer",
            "versions": [{"effective_from": "2026-01-01", "formula": "1"}],
        }
    write_input(provider, {"format": "rulespec/v1", "rules": [exported]})
    fragment = "relation.provided" if provider_kind == "data_relation" else "provided"
    payload = {
        "format": "rulespec/v1",
        "imports": [f"{prefix}:statutes/provider#{fragment}"],
        "rules": [
            relation("pairs"),
            relation("selected", derived=True, formula=formula),
            {
                "name": "size",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "len(selected)"}
                ],
            },
        ],
    }
    write_input(module, payload)
    return root, module, payload


def local_slots(root, module, payload):
    return validation._rulespec_local_input_slots(
        payload,
        imports=set(payload["imports"]),
        rules_file=module,
        policy_repo_path=root / "us",
    )


def absolute_input_issue(root):
    return validation._rulespec_absolute_test_reference_issue(
        "us:policies/consumer#input.provided",
        label="Test input",
        policy_repo_path=root / "us",
        allow_input_slots=True,
        allow_relations=False,
        allow_outputs=False,
    )


def test_imported_relation_value_read_remains_a_local_input(tmp_path):
    root, module, payload = material(tmp_path)
    assert local_slots(root, module, payload) == {"provided"}
    assert validation._rulespec_reference_summary(module).input_slots == {"provided"}
    assert absolute_input_issue(root) is None
    assert validation._rulespec_import_export_names(
        set(payload["imports"]), rules_file=module, policy_repo_path=root / "us"
    ) == {"provided"}


@pytest.mark.parametrize(
    "formula", ["provided(anchor, candidate)", "len(provided) > 0"]
)
def test_imported_relation_atom_or_argument_is_not_a_local_input(tmp_path, formula):
    root, module, payload = material(tmp_path, formula=formula)
    assert local_slots(root, module, payload) == set()


@pytest.mark.parametrize("provider_kind", ["derived", "parameter"])
@pytest.mark.parametrize("local_relation", [False, True])
def test_imported_scalar_keeps_its_namespace_when_a_relation_has_the_same_name(
    tmp_path, provider_kind, local_relation
):
    root, module, payload = material(tmp_path, provider_kind=provider_kind)
    if local_relation:
        payload["rules"].insert(0, relation("provided"))
    write_input(module, payload)
    assert local_slots(root, module, payload) == set()
    if local_relation:
        assert validation._rulespec_reference_summary(module).input_slots == set()
        assert "does not resolve to an input slot" in absolute_input_issue(root)


def test_reference_summary_reads_changed_import_kinds_outside_its_target_cache(
    tmp_path,
):
    root, module, payload = material(tmp_path, provider_kind="derived")
    payload["rules"].insert(0, relation("provided"))
    write_input(module, payload)
    consumer_bytes = module.read_bytes()
    assert validation._rulespec_reference_summary(module).input_slots == set()
    provider = root / "us/statutes/provider.yaml"
    write_input(provider, {"format": "rulespec/v1", "rules": [relation("provided")]})
    assert validation._rulespec_reference_summary(module).input_slots == {"provided"}
    assert absolute_input_issue(root) is None
    write_input(
        provider,
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "provided",
                    "kind": "parameter",
                    "versions": [{"effective_from": "2026-01-01", "formula": "1"}],
                }
            ],
        },
    )
    assert validation._rulespec_reference_summary(module).input_slots == set()
    assert "does not resolve to an input slot" in absolute_input_issue(root)
    assert module.read_bytes() == consumer_bytes


@pytest.mark.parametrize("keep_relation", [False, True])
def test_scalar_reference_summary_does_not_traverse_imports(
    tmp_path, monkeypatch, keep_relation
):
    _root, module, payload = material(tmp_path, provider_kind="derived")
    payload["rules"] = [payload["rules"][-1]]
    payload["rules"][0]["versions"][0]["formula"] = "provided + 1"
    if keep_relation:
        payload["rules"].insert(0, relation("pairs"))
    write_input(module, payload)

    def refuse_traversal(*args, **kwargs):
        pytest.fail("A scalar summary must not traverse imported declarations")

    monkeypatch.setattr(validation, "_rulespec_import_namespaces", refuse_traversal)
    assert validation._rulespec_reference_summary(module).input_slots == {"provided"}


def test_import_namespace_uses_the_canonical_jurisdiction_owner(tmp_path):
    root, module, payload = material(tmp_path, prefix="us-ca")
    write_input(
        root / "us/statutes/provider.yaml",
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "provided",
                    "kind": "parameter",
                    "versions": [{"effective_from": "2026-01-01", "formula": "1"}],
                }
            ],
        },
    )
    assert local_slots(root, module, payload) == {"provided"}


def test_real_imported_relation_value_read_matches_compiled_input_owner(
    tmp_path, real_engine
):
    root, module, payload = material(tmp_path)
    artifact = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    target = "us:policies/consumer"
    reads, issues = validation.compiled_relation_input_reads(
        artifact,
        source_rule_ids={"selected": f"{target}#selected"},
        relation_roots={f"{target}#relation.selected"},
    )
    assert issues == []
    assert reads[f"{target}#relation.selected"] == {(target, "provided")}
    assert local_slots(root, module, payload) == {"provided"}

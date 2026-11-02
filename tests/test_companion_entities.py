"""Source declaration lookup and diagnostics for compact companion facts."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.companion_entities import (
    cross_entity_input_findings,
    declared_input_entities,
    flat_input_entity_id,
)


def _module(path: Path, *, inputs=(), imports=()) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "inputs": [{"name": name, "entity": entity} for name, entity in inputs],
                "imports": list(imports),
            }
        )
    )
    return path


@pytest.mark.parametrize(
    "reference",
    [
        "us:policies/person-facts#input.claimant_is_eligible",
        "us:policies/person-facts#eligibility.input.claimant_is_eligible",
    ],
)
def test_absolute_input_reads_declaring_module_instead_of_importer(tmp_path, reference):
    imported = _module(
        tmp_path / "person-facts.yaml", inputs=[("claimant_is_eligible", "Person")]
    )
    importer = _module(
        tmp_path / "claim.yaml", inputs=[("claimant_is_eligible", "Household")]
    )
    assert declared_input_entities(
        importer,
        [reference],
        resolve_reference=lambda key: imported if key == reference else None,
        resolve_import=lambda _target, _current: None,
    ) == {reference: "Person"}


def test_local_bare_input_declaration_takes_precedence(tmp_path):
    imported = _module(tmp_path / "imported.yaml", inputs=[("claimant", "Person")])
    importer = _module(
        tmp_path / "local.yaml",
        inputs=[("claimant", "Household")],
        imports=["us:policies/imported"],
    )
    assert declared_input_entities(
        importer,
        ["claimant"],
        resolve_reference=lambda _key: None,
        resolve_import=lambda _target, _current: imported,
    ) == {"claimant": "Household"}


def test_consumer_alias_searches_referenced_modules_imports_only(tmp_path):
    alias = "us:policies/consumer#input.claimant"
    producer = _module(tmp_path / "producer.yaml", inputs=[("claimant", "Person")])
    unrelated = _module(tmp_path / "unrelated.yaml", inputs=[("claimant", "TaxUnit")])
    consumer = _module(tmp_path / "consumer.yaml", imports=["us:policies/producer"])
    root = _module(tmp_path / "root.yaml", imports=["us:policies/unrelated"])
    targets = {"us:policies/producer": producer, "us:policies/unrelated": unrelated}
    assert declared_input_entities(
        root,
        [alias],
        resolve_reference=lambda _key: consumer,
        resolve_import=lambda target, _current: targets.get(target),
    ) == {alias: "Person"}


def test_consumer_alias_keeps_its_local_declaration_authority(tmp_path):
    alias = "us:policies/consumer#eligibility.input.claimant"
    producer = _module(tmp_path / "producer.yaml", inputs=[("claimant", "Person")])
    consumer = _module(
        tmp_path / "consumer.yaml",
        inputs=[("claimant", "Household")],
        imports=["us:policies/producer"],
    )
    assert declared_input_entities(
        consumer,
        [alias],
        resolve_reference=lambda _key: consumer,
        resolve_import=lambda _target, _current: producer,
    ) == {alias: "Household"}


def test_consumer_alias_rejects_ambiguous_imported_declarations(tmp_path):
    alias = "us:policies/consumer#input.claimant"
    first = _module(tmp_path / "first.yaml", inputs=[("claimant", "Person")])
    second = _module(tmp_path / "second.yaml", inputs=[("claimant", "TaxUnit")])
    consumer = _module(
        tmp_path / "consumer.yaml", imports=["us:policies/first", "us:policies/second"]
    )
    targets = {"us:policies/first": first, "us:policies/second": second}
    with pytest.raises(ValueError, match="ambiguous imported declarations.*absolute"):
        declared_input_entities(
            consumer,
            [alias],
            resolve_reference=lambda _key: consumer,
            resolve_import=lambda target, _current: targets.get(target),
        )


def test_bare_input_resolves_transitive_imports_without_revisiting_cycles(tmp_path):
    producer = _module(
        tmp_path / "producer.yaml",
        inputs=[("claimant", "Person")],
        imports=["us:policies/root"],
    )
    bridge = _module(tmp_path / "bridge.yaml", imports=["us:policies/producer"])
    importer = _module(tmp_path / "root.yaml", imports=["us:policies/bridge"])
    targets = {
        "us:policies/root": importer,
        "us:policies/bridge": bridge,
        "us:policies/producer": producer,
    }
    assert declared_input_entities(
        importer,
        ["claimant", "legacy_undeclared"],
        resolve_reference=lambda _key: None,
        resolve_import=lambda target, _current: targets.get(target),
    ) == {"claimant": "Person"}


def test_ambiguous_bare_import_requires_absolute_reference(tmp_path):
    first = _module(tmp_path / "first.yaml", inputs=[("claimant", "Person")])
    second = _module(tmp_path / "second.yaml", inputs=[("claimant", "Household")])
    importer = _module(
        tmp_path / "root.yaml", imports=["us:policies/first", "us:policies/second"]
    )
    targets = {"us:policies/first": first, "us:policies/second": second}
    with pytest.raises(ValueError, match="ambiguous imported declarations.*absolute"):
        declared_input_entities(
            importer,
            ["claimant"],
            resolve_reference=lambda _key: None,
            resolve_import=lambda target, _current: targets.get(target),
        )


def test_entity_ids_preserve_owner_and_distinguish_case_collisions():
    owner_id = "case"
    assert flat_input_entity_id("Household", "Household", owner_id) == owner_id
    acronym_id = flat_input_entity_id("AB", "Household", owner_id)
    titlecase_id = flat_input_entity_id("Ab", "Household", owner_id)
    assert len({owner_id, acronym_id, titlecase_id}) == 3


def test_foreign_entity_ids_avoid_all_reserved_query_ids():
    assert (
        flat_input_entity_id(
            "Person",
            "Household",
            "case",
            reserved_entity_ids={"case:entity:Person", "case:entity:Person:1"},
        )
        == "case:entity:Person:2"
    )
    assert (
        flat_input_entity_id(
            "Household", "Household", "case", reserved_entity_ids={"case"}
        )
        == "case"
    )


@pytest.mark.parametrize(
    "detail",
    [
        "execution succeeded",
        "missing input owner_input for entity owner-1",
        "unknown parameter claimant for entity owner-1",
    ],
)
def test_unread_foreign_fact_has_no_missing_relation_finding(detail):
    assert not cross_entity_input_findings(
        detail,
        {"claimant": True},
        {"claimant": "Person"},
        {"eligibility": "Household"},
        {},
    )


def test_missing_relation_finding_names_input_output_and_entities():
    key = "us:policies/person-facts#eligibility.input.claimant"
    findings = cross_entity_input_findings(
        "missing input claimant for entity owner-1",
        {key: True},
        {key: "Person"},
        {"us:policies/claim#eligibility": "Household"},
        {},
    )
    assert len(findings) == 1
    for required in (key, "eligibility", "Person", "Household", "relation", "traverse"):
        assert required in findings[0]


def test_declared_relation_leaves_failed_traversal_to_engine_diagnostic():
    assert not cross_entity_input_findings(
        "missing input claimant for entity owner-1",
        {"claimant": True},
        {"claimant": "Person"},
        {"eligibility": "Household"},
        {"member_of_household": ("Person", "Household")},
    )


@pytest.mark.parametrize("missing_name", ["wage", "`wage`", '"wage"'])
def test_missing_input_name_does_not_match_another_inputs_substring(missing_name):
    assert not cross_entity_input_findings(
        f"missing input {missing_name} for entity case",
        {"age": 65, "wage": 100},
        {"age": "Person", "wage": "Household"},
        {"income": "Household"},
        {},
    )


def test_multiple_outputs_have_one_contextual_missing_relation_finding():
    findings = cross_entity_input_findings(
        "missing input `claimant` for entity case",
        {"claimant": True},
        {"claimant": "Person"},
        {"eligibility": "Household", "constant": "Household"},
        {},
    )
    assert len(findings) == 1
    assert "`eligibility`, `constant`" in findings[0]
    assert "when querying" in findings[0]
    assert "reads it" not in findings[0]


@pytest.mark.parametrize("module", ["person-facts", "other-facts"])
def test_missing_input_absolute_alias_keeps_declaring_module_authority(module):
    key = "us:policies/person-facts#eligibility.input.claimant"
    findings = cross_entity_input_findings(
        f"missing input `us:policies/{module}#input.claimant` for entity case",
        {key: True},
        {key: "Person"},
        {"eligibility": "Household"},
        {},
    )
    assert bool(findings) == (module == "person-facts")


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_transitive_bare_import_uses_declaring_jurisdiction(
    tmp_path, monkeypatch, executor
):
    from axiom_encode.cli import _execute_rulespec_test_case
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    root = tmp_path / "rulespec-us"
    us = root / "us" / "policies"
    nm = root / "us-nm" / "policies"
    us.mkdir(parents=True)
    nm.mkdir(parents=True)
    consumer = _module(us / "consumer.yaml", imports=["us-nm:policies/bridge"])
    _module(nm / "bridge.yaml", imports=["policies/person-facts"])
    _module(
        nm / "person-facts.yaml",
        inputs=[("claimant", "Person")],
        imports=["us:policies/federal-facts"],
    )
    _module(us / "person-facts.yaml", inputs=[("claimant", "Household")])
    _module(us / "federal-facts.yaml", inputs=[("federal_fact", "TaxUnit")])
    if executor == "validator":
        pipeline = ValidatorPipeline(
            policy_repo_path=us.parent,
            axiom_rules_path=tmp_path,
            local_corpus_release=None,
            enable_oracles=False,
            enforce_repository_layout=False,
        )
        assert pipeline._rulespec_companion_input_entities(
            consumer, {"claimant": True, "federal_fact": True}
        ) == {"claimant": "Person", "federal_fact": "TaxUnit"}
        return

    requests = []
    original_run = subprocess.run

    def capture_run(command, **kwargs):
        if "run-compiled" not in command:
            return original_run(command, **kwargs)
        requests.append(json.loads(kwargs["input"]))
        return subprocess.CompletedProcess(
            command, 1, "", "missing input `claimant` for entity case"
        )

    monkeypatch.setattr("axiom_encode.cli.subprocess.run", capture_run)
    failures = _execute_rulespec_test_case(
        consumer.with_suffix(".test.yaml"),
        {
            "input": {"claimant": True, "federal_fact": True},
            "output": {"eligibility": "holds"},
        },
        case_name="declaring-jurisdiction",
        compiled_path=tmp_path / "compiled.json",
        binary=tmp_path / "engine",
        axiom_rules_path=tmp_path,
        env={},
        parameter_by_id={},
        derived_ids={"eligibility"},
        derived_by_id={"eligibility": {"id": "eligibility", "entity": "Household"}},
        declared_relation_names=set(),
        declared_relation_slots={},
        policy_repo_path=us.parent,
        rulespec_roots=[root],
    )
    fact = requests[0]["dataset"]["inputs"][0]
    assert fact["entity"] == "Person"
    assert fact["entity_id"] != requests[0]["queries"][0]["entity_id"]
    assert (
        len(
            {requests[0]["queries"][0]["entity_id"]}
            | {item["entity_id"] for item in requests[0]["dataset"]["inputs"]}
        )
        == 3
    )
    assert "Person" in failures[0]["message"]

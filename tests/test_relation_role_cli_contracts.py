"""Keep relation predicates inside ordinary input, dependency and proof guards."""

import copy

import pytest
import yaml

from axiom_encode.cli import (
    _GENERATED_PROOF_RULE_KINDS,
    _derived_custom_proof_atom_kind,
    _generated_rule_formula_identifiers,
    _local_factual_input_names_from_rules_content,
    _parsed_rule_requires_generated_source_proof,
    _reachable_local_formulas_for_rule,
    _referenced_local_rule_names,
    _repair_missing_source_proof_atoms,
)
from axiom_encode.harness.strict_compiled_inputs import (
    compiled_program_has_strict_input_reads,
)
from tests.test_companion_declared_entities import _compile
from tests.test_relation_role_companions import FIXTURES, SOURCE, reference_fixture
from tests.test_strict_compiled_inputs import format_three_artifact


def relation_rule(formula):
    return {
        "name": "members",
        "kind": "derived_relation",
        "derived_relation": {
            "source_path": [
                {"relation": "roster", "from_slot": 0, "to_slot": 1},
                {"relation": "roster", "from_slot": 1, "to_slot": 0},
            ]
        },
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def test_cli_dependencies_keep_atoms_paths_and_unknown_names_inside_quantifiers():
    rule = relation_rule(
        "exists(ties(anchor, p), p.age < 22 and "
        "other(p, candidate) and candidate.undeclared)"
    )
    expected = {"roster", "ties", "age", "other", "undeclared"}
    assert _generated_rule_formula_identifiers(rule) == expected
    assert _referenced_local_rule_names([rule]) == expected


def test_cli_factual_inputs_exclude_imported_atoms_and_bound_variables():
    payload = {
        "imports": ["us:policies/example/pairs#relation.ties"],
        "rules": [
            relation_rule("exists(ties(anchor, p), p.age < 22 and candidate.unknown)")
        ],
    }
    assert _local_factual_input_names_from_rules_content(yaml.safe_dump(payload)) == {
        "age",
        "unknown",
    }


def test_cli_predicate_inputs_preserve_legal_builtin_spellings():
    rule = relation_rule("candidate.min < min(anchor.age, 22) and exists")
    assert _generated_rule_formula_identifiers(rule) == {"roster", "min", "age", "exists"}


def test_true_view_predicate_keeps_source_relation_dependency_walk():
    filtered = relation_rule("candidate.age < 22")
    filtered["name"] = "filtered"
    view = relation_rule(True)
    view["derived_relation"] = {"source_relation": "filtered"}
    payload = {
        "rules": [
            {"name": "count", "kind": "derived", "versions": [{"formula": "len(members)"}]},
            view,
            filtered,
        ]
    }
    assert _reachable_local_formulas_for_rule(
        output_name="count", rules_payload=payload
    ) == ["len(members)", "candidate.age < 22"]


@pytest.mark.parametrize(
    "rule_name",
    [
        "lives_with",
        "snap_tie",
        "snap_within_two_ties",
        "snap_household_member",
        "snap_participating_member",
    ],
)
def test_generated_proof_selection_and_repair_cover_every_predicate_form(rule_name):
    payload = yaml.safe_load((FIXTURES / "guarded_household.yaml").read_text())
    rule = next(rule for rule in payload["rules"] if rule["name"] == rule_name)
    assert rule["kind"] in _GENERATED_PROOF_RULE_KINDS
    assert not _parsed_rule_requires_generated_source_proof(rule)
    atom = rule["metadata"]["proof"]["atoms"][0]
    assert _derived_custom_proof_atom_kind(rule, atom).kind == "formula"
    del rule["metadata"]["proof"]
    assert _parsed_rule_requires_generated_source_proof(rule)
    payload["rules"] = [rule]
    payload["module"] = {"source_verification": {"corpus_citation_path": SOURCE}}
    repaired, repairs = _repair_missing_source_proof_atoms(yaml.safe_dump(payload))
    assert repairs == [rule_name]
    repaired_rule = yaml.safe_load(repaired)["rules"][0]
    assert not _parsed_rule_requires_generated_source_proof(repaired_rule)
    repaired_atom = repaired_rule["metadata"]["proof"]["atoms"][0]
    assert repaired_atom["kind"] == "formula"
    assert repaired_atom["source"]["corpus_citation_path"] == SOURCE


def predicate_artifact():
    payload = format_three_artifact()
    predicate = {
        "kind": "exists",
        "relation": "ties",
        "terms": ["anchor", {"var": 0}],
        "variable_slot": 1,
        "body": {"kind": "role_input", "name": "selected", "term": {"var": 0}},
    }
    payload["program"]["relations"] = [
        {
            "name": "members",
            "arity": 2,
            "slot_entities": ["Person", "Person"],
            "derivation": {
                "source_relation": "ties",
                "current_slot": 0,
                "related_slot": 1,
                "predicate": predicate,
            },
        }
    ]
    return payload


def test_strict_input_walk_accepts_scoped_predicate_reads():
    assert compiled_program_has_strict_input_reads(predicate_artifact())


@pytest.mark.parametrize("mutation", ["default", "unknown", "unbound", "bool_variable"])
def test_strict_input_walk_refuses_invalid_reads_inside_exists(mutation):
    payload = predicate_artifact()
    predicate = payload["program"]["relations"][0]["derivation"]["predicate"]
    if mutation == "default":
        predicate["body"]["default"] = False
    elif mutation == "unknown":
        predicate["body"]["kind"] = "future_role_input"
    elif mutation == "unbound":
        predicate["body"]["term"] = {"var": 1}
    else:
        predicate["terms"][1] = {"var": False}
    assert not compiled_program_has_strict_input_reads(payload)


def test_strict_input_walk_does_not_leak_roles_into_aggregate_children():
    payload = predicate_artifact()
    payload["program"]["relations"][0]["derivation"]["predicate"] = {
        "kind": "count_related",
        "relation": "ties",
        "current_slot": 0,
        "related_slot": 1,
        "where": {"kind": "role_input", "name": "selected", "term": "candidate"},
    }
    assert not compiled_program_has_strict_input_reads(payload)


def test_real_compiled_guarded_fixture_keeps_strict_sparse_execution(tmp_path, real_engine):
    root, module = reference_fixture(tmp_path)
    payload = _compile(
        real_engine.binary, real_engine.root, root, module, tmp_path / "compiled.json"
    )
    assert compiled_program_has_strict_input_reads(payload)
    for relation in payload["program"]["relations"]:
        if relation.get("derivation"):
            changed = copy.deepcopy(payload)
            derived = next(
                value
                for value in changed["program"]["relations"]
                if value["name"] == relation["name"]
            )
            derived["derivation"]["predicate"] = {
                "kind": "input_or_else",
                "name": "selected",
                "default": {"kind": "bool", "value": False},
            }
            assert not compiled_program_has_strict_input_reads(changed)

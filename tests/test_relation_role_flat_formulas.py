"""Reject shorthand formulas and preserve checks on versioned relations."""

import textwrap

import pytest
import yaml

from axiom_encode.cli import (
    _derived_custom_proof_atom_kind,
    _generated_rule_formula_identifiers,
    _local_factual_input_names_from_rules_content,
    _parsed_rule_requires_generated_source_proof,
    _reachable_local_formulas_for_rule,
    _repair_missing_source_proof_atoms,
    _rulespec_rule_formula_texts_for_generated_repair,
    _safe_load_unique_keys,
)
from axiom_encode.harness import validator_pipeline as validation
from axiom_encode.harness.proof_validator import validate_rulespec_proofs
from tests.test_companion_declared_entities import _companions
from tests.test_relation_role_companions import FIXTURES, SOURCE, reference_fixture


def declaration():
    """Build synthetic encoder input, never a corpus module."""
    return {
        "format": "rulespec/v1",
        "inputs": {"age": {"dtype": "Integer"}},
        "rules": [
            {
                "name": "pairs",
                "kind": "data_relation",
                "data_relation": {"arity": 2, "arguments": ["Person", "Person"]},
            },
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
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "candidate.age < 22"}
                ],
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


def flatten(rule):
    (version,) = rule.pop("versions")
    rule.update(version)
    for atom in rule.get("metadata", {}).get("proof", {}).get("atoms", []):
        if atom.get("path") == "versions[0].formula":
            atom["path"] = "formula"


def test_versioned_predicate_keeps_default_numeric_and_dependency_checks():
    payload = declaration()
    rule = payload["rules"][1]
    payload["inputs"]["age"]["default"] = 0
    content = yaml.safe_dump(payload)
    assert validation.find_relation_role_issues(content) == [
        "[relation-input-default] Input `age` read by derived relation `members` must not declare a default. Supply the fact explicitly."
    ]
    assert validation.find_ungrounded_numeric_issues(
        content, source_text="The limit is 18."
    ) == [
        "Ungrounded generated numeric literal: 22 does not appear as a substantive numeric value in the source text."
    ]
    assert validation._rule_grounding_values_by_path(rule) == [
        ("versions[0].formula", "22", 22.0)
    ]
    assert _local_factual_input_names_from_rules_content(content) == {"age"}
    assert _generated_rule_formula_identifiers(rule) == {"age", "pairs"}
    assert _rulespec_rule_formula_texts_for_generated_repair(rule) == [
        "candidate.age < 22"
    ]
    assert _reachable_local_formulas_for_rule(
        output_name="member_count", rules_payload=payload
    ) == ["len(members)", "candidate.age < 22"]
    rule["values"] = {1: 999}
    assert validation._rule_grounding_values_by_path(rule) == [
        ("versions[0].formula", "22", 22.0)
    ]


def test_versioned_predicate_keeps_import_and_date_checks():
    payload = declaration()
    payload["imports"] = ["us:policies/example/external_pairs#relation.external_pairs"]
    payload["rules"][1]["versions"][0]["formula"] = "external_pairs(anchor, candidate)"
    assert validation.find_unused_import_issues(yaml.safe_dump(payload)) == []
    assert validation._rulespec_formula_identifiers(payload) >= {"external_pairs"}
    payload["rules"][1]["versions"][0]["formula"] = "candidate.started > 2025-01-01"
    assert validation.find_formula_date_literal_issues(yaml.safe_dump(payload)) == [
        "Formula date literal unsupported: `members` uses `2025-01-01` inside an executable formula. Axiom formula syntax has no date literal type; encode effective windows with `effective_from` metadata or source-stated boolean predicates such as `taxable_year_begins_after_termination_date`."
    ]


@pytest.mark.parametrize("indented", [False, True])
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
def test_versioned_proof_selection_and_repair_use_actual_formula_anchor(
    indented, rule_name
):
    payload = yaml.safe_load((FIXTURES / "guarded_household.yaml").read_text())
    rule = next(rule for rule in payload["rules"] if rule["name"] == rule_name)
    assert (
        _derived_custom_proof_atom_kind(
            rule, rule["metadata"]["proof"]["atoms"][0]
        ).kind
        == "formula"
    )
    del rule["metadata"]["proof"]
    assert _parsed_rule_requires_generated_source_proof(rule)
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": SOURCE}},
        "rules": [rule],
    }
    content = yaml.safe_dump(payload, sort_keys=False)
    if indented:
        prefix, _, rules = content.partition("rules:\n")
        content = prefix + "rules:\n" + textwrap.indent(rules, "  ")
    repaired, repairs = _repair_missing_source_proof_atoms(content)
    assert repairs == [rule_name]
    repaired_rule = _safe_load_unique_keys(repaired)["rules"][0]
    assert (
        repaired_rule["metadata"]["proof"]["atoms"][0]["path"] == "versions[0].formula"
    )
    assert not _parsed_rule_requires_generated_source_proof(repaired_rule)
    assert (
        validate_rulespec_proofs(
            repaired,
            require_policy_proofs=True,
            source_texts={SOURCE: (FIXTURES / "source.txt").read_text()},
        ).issues
        == []
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_versioned_guarded_reference_keeps_native_companion_memberships(
    tmp_path, real_engine, executor
):
    root, module = reference_fixture(tmp_path)
    payload = yaml.safe_load(module.read_text())
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump(payload)
    )
    cases = yaml.safe_load(module.with_suffix(".test.yaml").read_text())
    assert (
        validation.find_missing_derived_companion_output_issues(
            module.read_text(), cases, rules_file=module, policy_repo_path=root / "us"
        )
        == []
    )
    assert (
        _companions(
            executor,
            binary=real_engine.binary,
            engine_root=real_engine.root,
            root=root,
            module=module,
            artifact=tmp_path / "compiled.json",
        )
        == []
    )


@pytest.mark.parametrize("metadata", [None, {"private": True}, {"proof": {}}, []])
def test_indented_versioned_proof_repair_preserves_unique_metadata(metadata):
    payload = declaration()
    rule = payload["rules"][1]
    if metadata is not None:
        rule["metadata"] = metadata
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": SOURCE}},
        "rules": [rule],
    }
    content = yaml.safe_dump(payload, sort_keys=False)
    prefix, _, rules = content.partition("rules:\n")
    content = prefix + "rules:\n" + textwrap.indent(rules, "  ")
    repaired, repairs = _repair_missing_source_proof_atoms(content)
    assert repairs == ["members"]
    result = _safe_load_unique_keys(repaired)["rules"][0]
    assert result["metadata"]["proof"]["atoms"][0]["path"] == "versions[0].formula"
    if metadata == {"private": True}:
        assert result["metadata"]["private"] is True


@pytest.mark.parametrize("inline_module", [False, True])
@pytest.mark.parametrize("trailing_inputs", [False, True])
def test_versioned_proof_repair_keeps_atom_on_rule_and_requires_module_proofs(
    inline_module, trailing_inputs
):
    rule = declaration()["rules"][1]
    rule["source"] = {"span": "Synthetic bounded membership evidence."}
    module = {"source_verification": {"corpus_citation_path": SOURCE}}
    module_text = (
        f"module: {{source_verification: {{corpus_citation_path: {SOURCE}}}}}\n"
        if inline_module
        else yaml.safe_dump({"module": module}, sort_keys=False)
    )
    content = (
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        "format: rulespec/v1\n"
        + module_text
        + "rules:\n"
        + textwrap.indent(yaml.safe_dump([rule], sort_keys=False), "  ")
    )
    if trailing_inputs:
        content += "inputs:\n  age:\n    dtype: Integer\n"
    repaired, repairs = _repair_missing_source_proof_atoms(content)
    assert repairs == ["members"]
    result = _safe_load_unique_keys(repaired)
    assert result["module"]["proof_validation"]["required"] is True
    assert result["rules"][0]["metadata"]["proof"]["atoms"] == [
        {
            "path": "versions[0].formula",
            "kind": "formula",
            "source": {
                "corpus_citation_path": SOURCE,
                "span": "Synthetic bounded membership evidence.",
            },
        }
    ]
    if trailing_inputs:
        assert result["inputs"] == {"age": {"dtype": "Integer"}}


@pytest.mark.parametrize(
    "content",
    [
        "format: rulespec/v1\nmodule: [broken\n",
        """format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: us/guidance/axiom/relation-roles
rules:
  - name: members
    kind: derived_relation
    metadata: {}
    metadata: {private: true}
    formula: candidate.age < 22
""",
    ],
)
def test_source_proof_repair_leaves_invalid_or_duplicate_input_unchanged(content):
    with pytest.raises((yaml.YAMLError, ValueError)):
        _safe_load_unique_keys(content)
    assert _repair_missing_source_proof_atoms(content) == (content, [])


@pytest.mark.parametrize("kind", ["derived", "derived_relation"])
@pytest.mark.parametrize("empty_versions", [False, True])
def test_flat_generated_rule_refuses_before_proof_repair(kind, empty_versions):
    payload = declaration()
    rule = payload["rules"][1]
    rule["kind"] = kind
    flatten(rule)
    if empty_versions:
        rule["versions"] = []
    content = yaml.safe_dump(payload)
    from axiom_encode.rule_version_form import find_rule_version_form_issues

    assert find_rule_version_form_issues(content) == [
        "[rule-version-form] Rule `members` must put `formula` and its dates under `versions:`."
    ]
    assert _repair_missing_source_proof_atoms(content) == (content, [])


@pytest.mark.parametrize("versions", [{"bad": {}}, "bad", 3])
def test_versioned_readers_keep_malformed_version_guards(versions):
    rule = declaration()["rules"][1]
    rule["versions"] = versions
    assert validation._rulespec_rule_formula_texts(rule) == []
    assert validation._rulespec_rule_formulas({"rules": [rule]}) == []
    assert validation._rulespec_formula_identifiers({"rules": [rule]}) == set()
    assert not _parsed_rule_requires_generated_source_proof(rule)
    assert _rulespec_rule_formula_texts_for_generated_repair(rule) == []

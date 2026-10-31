"""Return a finding when nested native relation analysis exhausts Python's stack."""

import json

from axiom_encode.companion_relations import (
    executable_relation_directions,
    pair_relation_assignment_issues,
)
from axiom_encode.formula_grammar import parse_formula
from axiom_encode.formula_references import analyze_relation_predicate
from axiom_encode.harness.relation_input_reads import compiled_relation_input_reads
from axiom_encode.harness.strict_compiled_inputs import (
    compiled_program_has_strict_input_reads,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"
FINDING = "[relation-input-analysis] Cannot inventory native relation roles: expression nesting exceeds Python analysis capacity."
DEPTH_INPUT = "format: rulespec/v1\ninputs:\n- name: fact\n  entity: Person\n  dtype: Integer\nrules:\n- name: pairs\n  kind: data_relation\n  data_relation:\n    arity: 2\n    arguments:\n    - Person\n    - Household\n- name: members\n  kind: derived_relation\n  derived_relation:\n    arity: 2\n    slot_entities:\n    - Person\n    - Household\n    current_slot: 1\n    related_slot: 0\n    source_relation: pairs\n    entity: SnapUnit\n    member_relation: selected\n  versions:\n  - effective_from: '2026-01-01'\n    formula: __deep_formula__\n- name: size\n  kind: derived\n  entity: SnapUnit\n  dtype: Integer\n  versions:\n  - effective_from: '2026-01-01'\n    formula: len(selected)\n"


def wrapper_stack(call, frames=40):
    if frames:
        return wrapper_stack(call, frames - 1)
    return call()


def compiled_depth_payload():
    predicate = {
        "kind": "comparison",
        "left": {"kind": "literal", "value": {"kind": "bool", "value": True}},
        "op": "eq",
        "right": {"kind": "literal", "value": {"kind": "bool", "value": True}},
    }
    for _ in range(491):
        predicate = {"kind": "not", "item": predicate}
    prefix = "us:policies/depth_min/boundary#"
    relation = {
        "name": prefix + "relation.members",
        "arity": 2,
        "slot_entities": ["Person", "Household"],
        "derivation": {
            "source_relation": prefix + "relation.pairs",
            "current_slot": 1,
            "related_slot": 0,
            "entity": "SnapUnit",
            "member_relation": "selected",
            "slot_entities": ["Person", "Household"],
            "predicate": predicate,
        },
    }
    base = {
        "name": prefix + "relation.pairs",
        "arity": 2,
        "slot_entities": ["Person", "Household"],
    }
    output = {
        "id": prefix + "size",
        "name": "size",
        "entity": "SnapUnit",
        "dtype": "integer",
        "semantics": "scalar",
        "expr": {
            "kind": "count_related",
            "relation": prefix + "relation.members",
            "current_slot": 1,
            "related_slot": 0,
        },
    }
    return {
        "artifact_format_version": 3,
        "program": {
            "derived": [output],
            "relations": [base, relation],
            "parameters": [],
        },
    }


def test_depth_491_pair_inventory_returns_analysis_finding():
    payload = compiled_depth_payload()
    assert wrapper_stack(
        lambda: pair_relation_assignment_issues(
            payload["program"]["relations"], {"input": {}}, "depth"
        )
    ) == [FINDING]


def test_depth_491_other_walkers_complete_or_refuse_without_raising():
    payload = compiled_depth_payload()
    relations = payload["program"]["relations"]
    formula = "not " * 491 + "true"
    assert wrapper_stack(lambda: parse_formula(formula)).kind == "unary"
    assert (
        wrapper_stack(lambda: analyze_relation_predicate(formula)).parse_error is None
    )
    roots = {relations[1]["name"]}
    reads, issues = wrapper_stack(
        lambda: compiled_relation_input_reads(
            payload, source_rule_ids={}, relation_roots=roots
        )
    )
    assert reads == {relations[1]["name"]: set()}
    assert issues == []
    assert isinstance(
        wrapper_stack(lambda: compiled_program_has_strict_input_reads(payload)), bool
    )
    output = payload["program"]["derived"][0]
    directions = wrapper_stack(
        lambda: executable_relation_directions(
            {output["id"]: output},
            [output["id"]],
            {"start": "2026-01-01"},
            "SnapUnit",
            {
                relation["name"]: tuple(relation["slot_entities"])
                for relation in relations
            },
            relations,
        )
    )
    assert directions[relations[1]["name"]] == (1, None)
    assert directions[relations[0]["name"]] == (1, "Person")


def test_depth_491_public_rulespec_ci_returns_analysis_finding(tmp_path, real_engine):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/depth_min/boundary.yaml"
    module.parent.mkdir(parents=True)
    module.write_text(
        LABEL + DEPTH_INPUT.replace("__deep_formula__", "not " * 491 + "true")
    )
    target = "us:policies/depth_min/boundary"
    case = [
        {
            "name": "depth",
            "entity_id": "case",
            "period": "2026-01",
            "input": {target + "#relation.pairs": [{"tuple": ["bob", "case"]}]},
            "output": {target + "#size": 1},
        }
    ]
    module.with_suffix(".test.yaml").write_text(LABEL + json.dumps(case))
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    result = pipeline._run_rulespec_ci(module)
    assert result.passed is False
    assert FINDING in result.issues

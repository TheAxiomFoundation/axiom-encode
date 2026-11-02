"""Keep every version visible to the callers of the formula grammar."""

from __future__ import annotations

import yaml

from axiom_encode.cli import _generated_rule_formula_identifiers
from axiom_encode.harness import source_completeness
from axiom_encode.harness import validator_pipeline as validation


def versioned_relation():
    """Build a synthetic encoder test input, never a corpus module."""
    return {
        "name": "members",
        "kind": "derived_relation",
        "derived_relation": {"source_relation": "people"},
        "versions": [
            {
                "effective_from": "2025-01-01",
                "formula": "candidate.first_read < first_limit",
            },
            {
                "effective_from": "2026-01-01",
                "formula": "candidate.second_read < second_limit",
            },
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {"import": {"output": "first_limit"}},
                    {"import": {"output": "second_limit"}},
                ]
            }
        },
    }


READS = {"first_read", "first_limit", "second_read", "second_limit"}


def test_proof_imports_can_be_consumed_by_different_predicate_versions():
    payload = {"format": "rulespec/v1", "rules": [versioned_relation()]}
    assert validation.find_proof_import_reference_issues(yaml.safe_dump(payload)) == []
    payload["rules"][0]["metadata"]["proof"]["atoms"].append(
        {"import": {"output": "absent"}}
    )
    assert validation.find_proof_import_reference_issues(yaml.safe_dump(payload)) == [
        "Proof import not referenced: `members` proof imports `absent`, but the rule formula does not reference that imported symbol."
    ]


def test_source_dependency_walk_unions_each_version_and_selects_one_when_requested():
    rule = versioned_relation()
    assert source_completeness._rule_formula_identifiers(rule) == READS | {"people"}
    assert source_completeness._rule_formula_identifiers(
        rule, rule["versions"][0]["formula"]
    ) == {"first_read", "first_limit", "people"}
    assert _generated_rule_formula_identifiers(rule) == READS | {"people"}


def test_cross_reference_blocks_read_both_versions_without_joining_formulas(tmp_path):
    pipeline = validation.ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
    )
    payload = {"format": "rulespec/v1", "rules": [versioned_relation()]}
    content = yaml.safe_dump(payload)
    assert pipeline._rulespec_formula_identifiers(content) == READS
    block = pipeline._extract_definition_blocks(content)[0]
    assert pipeline._definition_block_formula_identifiers(block) == READS
    assert pipeline._local_definition_resolves_cross_reference_placeholder(
        {"members": block},
        ["us:statutes/26/52#second_limit"],
        "statutes/26/52",
        "members",
        rulespec_file=tmp_path / "example.yaml",
    )


def test_source_evidence_walk_keeps_string_literals_as_grammar_tokens():
    rule = versioned_relation()
    rule["versions"][0]["formula"] = (
        'candidate.code == "A" and candidate.first_read < first_limit'
    )
    parameters = {
        name: {
            "name": name,
            "kind": "parameter",
            "versions": [{"effective_from": "2025-01-01", "formula": 22}],
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "source": {
                                "corpus_citation_path": "us/statute/example",
                                "excerpt": f"The {name} is twenty-two.",
                            },
                        }
                    ]
                }
            },
        }
        for name in ("first_limit", "second_limit")
    }
    assert set(
        source_completeness._formula_dependency_source_excerpts(
            rule, named_rules=parameters
        )
    ) == {
        ("us/statute/example", "The first_limit is twenty-two."),
        ("us/statute/example", "The second_limit is twenty-two."),
    }

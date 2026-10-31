"""Apply dependency and proof-source checks to native top-level formulas."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from axiom_encode.harness import same_case_execution, source_completeness

CITATION = "us/guidance/axiom/relation-roles"
EXCERPT = "Add: reported + local_amount."


def formula_rule(name, formula, *, kind="derived_relation", flat=True):
    rule = {"name": name, "kind": kind}
    version = {"effective_from": "2026-01-01", "formula": formula}
    if flat:
        rule.update(version)
        path = "formula"
    else:
        rule["versions"] = [version]
        path = "versions[0].formula"
    rule["metadata"] = {
        "proof": {
            "atoms": [
                {
                    "kind": "formula",
                    "path": path,
                    "source": {"corpus_citation_path": CITATION, "excerpt": EXCERPT},
                }
            ]
        }
    }
    return rule


@pytest.mark.parametrize("empty_versions", [False, True])
def test_flat_predicate_dependency_names_match_versioned_formula(empty_versions):
    text = "exists(pair(anchor, witness), witness.age < limit and missing_fact(candidate, witness))"
    flat = formula_rule("selected", text)
    if empty_versions:
        flat["versions"] = []
    versioned = formula_rule("selected", text, flat=False)
    expected = {"pair", "age", "limit", "missing_fact"}
    assert source_completeness._rule_formula_text(flat) == text
    assert source_completeness._rule_formula_identifiers(flat) == expected
    assert source_completeness._rule_formula_identifiers(versioned) == expected
    rules = {"limit": formula_rule("limit", "22", kind="parameter")}
    assert source_completeness._local_rule_dependency_closure(flat, principal_rules=rules) == {"limit"}


@pytest.mark.parametrize("versions", [{"formula": "age"}, "age", [None]])
def test_flat_dependency_readers_preserve_malformed_version_handling(versions):
    rule = formula_rule("selected", "candidate.age < limit")
    rule["versions"] = versions
    assert source_completeness._rule_formula_text(rule) == ""
    assert source_completeness._formula_dependency_source_excerpts(rule, named_rules={}) == ()


@pytest.mark.parametrize("flat", [False, True])
def test_formula_dependency_evidence_reads_actual_proof_paths(flat):
    rule = formula_rule("selected", "candidate.age < limit", flat=flat)
    parameter = formula_rule("limit", "22", kind="parameter", flat=flat)
    assert source_completeness._formula_dependency_source_excerpts(
        rule, named_rules={"limit": parameter}
    ) == ((CITATION, EXCERPT), (CITATION, EXCERPT))


def test_explicit_versions_override_flat_dependency_formula_and_proof():
    rule = formula_rule("selected", "candidate.ignored < ignored_limit")
    rule["versions"] = [{"effective_from": "2026-01-01", "formula": "candidate.age < limit"}]
    assert source_completeness._rule_formula_identifiers(rule) == {"age", "limit"}
    assert source_completeness._rule_formula_source_excerpts(rule) == ()


@pytest.mark.parametrize("flat", [False, True])
def test_native_consumer_discovery_follows_predicate_dependencies(flat):
    relation = formula_rule("selected", "candidate.reported > 0", flat=flat)
    relation["derived_relation"] = {"source_relation": "pairs"}
    output = formula_rule("count", "len(selected)", kind="derived", flat=flat)
    pipe = SimpleNamespace(
        _source_texts_for_rulespec_content=lambda _: {CITATION: EXCERPT},
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {CITATION: EXCERPT},
    )
    payload = {"rules": [relation, output]}
    assert same_case_execution._has_consumer(pipe, "", payload, {"reported"})
    unrelated = deepcopy(payload)
    if flat:
        unrelated["rules"][0]["formula"] = "candidate.age > 0"
    else:
        unrelated["rules"][0]["versions"][0]["formula"] = "candidate.age > 0"
    assert not same_case_execution._has_consumer(pipe, "", unrelated, {"reported"})

"""Keep companion analysis linear and cache immutable formula trees."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from axiom_encode import formula_grammar as grammar
from axiom_encode.formula_references import analyze_relation_predicate
from axiom_encode.harness import validator_pipeline as validation


@pytest.mark.parametrize("with_relation", [False, True])
def test_companion_coverage_kind_lookups_grow_linearly(monkeypatch, with_relation):
    """Count inventory work independently of machine and sandbox speed."""
    lookups = 0

    class Rule(dict):
        def get(self, name, default=None):
            nonlocal lookups
            if name == "kind":
                lookups += 1
            return super().get(name, default)

    monkeypatch.setattr(
        validation,
        "_canonical_rulespec_file_target",
        lambda *, symbol, **kwargs: "us:policies/synthetic#" + symbol,
    )

    def no_relation_graph(*args, **kwargs):
        pytest.fail("scalar companion coverage built a relation-only graph")

    if not with_relation:
        monkeypatch.setattr(validation, "analyze_relation_predicate", no_relation_graph)
    counts = []
    for size in (8, 128):
        payload = {
            "format": "rulespec/v1",
            "imports": [],
            "rules": [
                Rule(
                    name=f"value_{i}",
                    kind="derived",
                    entity="Person",
                    dtype="Integer",
                    versions=[
                        {"effective_from": "2026-01-01", "formula": "amount + 1"}
                    ],
                )
                for i in range(size)
            ],
        }
        if with_relation:
            payload["rules"][0]["versions"][0]["formula"] = "len(selected)"
            payload["rules"][:0] = [
                Rule(
                    name="pairs",
                    kind="data_relation",
                    data_relation={"arity": 2, "arguments": ["Person", "Person"]},
                ),
                Rule(
                    name="selected",
                    kind="derived_relation",
                    derived_relation={
                        "arity": 2,
                        "slot_entities": ["Person", "Person"],
                        "current_slot": 0,
                        "related_slot": 1,
                        "source_relation": "pairs",
                    },
                    versions=[{"effective_from": "2026-01-01", "formula": "true"}],
                ),
            ]
        monkeypatch.setattr(validation.yaml, "safe_load", lambda _: payload)
        cases = [
            {"output": {f"us:policies/synthetic#value_{i}": 0 for i in range(size)}}
        ]
        lookups = 0
        assert (
            validation.find_missing_derived_companion_output_issues(
                "# Synthetic encoder test input; not a RuleSpec module.\n",
                cases,
                rules_file=Path("synthetic.yaml"),
            )
            == []
        )
        counts.append(lookups)
    assert counts[1] <= 20 * counts[0]
    assert counts[1] <= 8 * (128 + 2)


def test_empty_imports_skip_formula_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("analyzed a formula without imports to check")

    monkeypatch.setattr(validation, "analyze_relation_predicate", forbidden)
    content = """# Synthetic encoder test input; not a RuleSpec module.
format: rulespec/v1
imports: []
rules:
  - name: result
    kind: derived
    versions:
      - effective_from: 2026-01-01
        formula: amount + 1
"""
    assert validation.find_unused_import_issues(content) == []


@pytest.mark.parametrize(
    "proof",
    [None, {"atoms": []}, {"atoms": [{"source": {"excerpt": "amount"}}]}],
)
def test_proof_without_import_atoms_skips_formula_analysis(monkeypatch, proof):
    def forbidden(*args, **kwargs):
        pytest.fail("analyzed a formula without proof import atoms to check")

    monkeypatch.setattr(validation, "analyze_relation_predicate", forbidden)
    rule = {
        "name": "result",
        "kind": "derived",
        "versions": [{"effective_from": "2026-01-01", "formula": "amount + 1"}],
    }
    if proof is not None:
        rule["metadata"] = {"proof": proof}
    content = (
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + yaml.safe_dump({"format": "rulespec/v1", "rules": [rule]})
    )
    assert validation.find_proof_import_reference_issues(content) == []


def test_formula_cache_shares_only_immutable_syntax(monkeypatch):
    grammar.parse_formula.cache_clear()
    original = grammar.tokenise
    calls = 0

    def counted(source):
        nonlocal calls
        calls += 1
        return original(source)

    monkeypatch.setattr(grammar, "tokenise", counted)
    source = "value + 1"
    input_read = analyze_relation_predicate(
        source, predicate_terms=False, lowering_entry="rule"
    )
    rule_read = analyze_relation_predicate(
        source,
        predicate_terms=False,
        lowering_entry="rule",
        rule_names=frozenset({"value"}),
    )
    tree = grammar.parse_formula(source)
    assert calls == 1
    assert input_read.input_reads == {"value"}
    assert rule_read.rule_reads == {"value"}
    assert rule_read.input_reads == set()
    assert isinstance(tree.children, tuple)
    with pytest.raises(FrozenInstanceError):
        tree.value = "changed"
    grammar.parse_formula.cache_clear()

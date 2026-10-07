"""Closed worksheet operand grammar must not swallow claimant conditions."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness

FIXTURES = Path(__file__).parent / "fixtures" / "t2036_operand_gates"
CONTROLS = json.loads((FIXTURES / "controls.json").read_text())


@pytest.mark.parametrize("case", CONTROLS, ids=lambda case: case["name"])
def test_reviewed_typed_operation_controls(case):
    text = case["input"]
    prefix = completeness._worksheet_monetary_operation_antecedent(text)
    assert (prefix is not None) is case["accepted"]
    if prefix is None:
        assert completeness._without_worksheet_imperative_consequent(text) == text
    if prefix is not None:
        assert text.startswith(prefix)
        assert text[len(prefix)] == ","
        assert completeness._without_worksheet_imperative_consequent(text) == prefix
        assert completeness._source_conjunctive_fact_gates(text) == (
            completeness._source_conjunctive_fact_gates(prefix)
        )
        if case["name"].endswith("_compound"):
            assert "and you are disabled" in prefix


@pytest.mark.parametrize("name", ["part_year", "ontario", "amt_other"])
@pytest.mark.parametrize("whitespace", [" \n\t", "\r\n"])
def test_whitespace_preserves_original_prefix(name, whitespace):
    text = next(case["input"] for case in CONTROLS if case["name"] == name)
    text = text.replace(" ", whitespace)
    prefix = completeness._worksheet_monetary_operation_antecedent(text)
    assert prefix is not None
    assert prefix == text[: text.index(",")]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: "(" + text,
        lambda text: ")" + text,
        lambda text: "[" + text,
        lambda text: "[)" + text,
        lambda text: '"' + text,
        lambda text: "“" + text,
        lambda text: "”" + text,
        lambda text: text + '"',
        lambda text: text + " and enter another amount.",
        lambda text: text.replace("115(1)(a) to (c)", "115(1)(d) to (a)"),
    ],
)
def test_malformed_delimiters_and_extra_operations_reject(mutation):
    text = next(case["input"] for case in CONTROLS if case["name"] == "part_year")
    assert completeness._worksheet_monetary_operation_antecedent(mutation(text)) is None


def test_factual_antecedent_is_not_reinterpreted():
    operation = next(case["input"] for case in CONTROLS if case["name"] == "amt_other")
    tail = operation[operation.index(",") :]
    antecedent = "If the applicant is disabled and the spouse is resident"
    gates = completeness._source_conjunctive_fact_gates(antecedent)
    assert len(gates) == 2
    assert completeness._source_conjunctive_fact_gates(antecedent + tail) == gates


def test_full_unchanged_candidate_condition_findings_and_source_ownership(monkeypatch):
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for name, expected in provenance["hashes"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == expected
    source = (FIXTURES / "source.txt").read_text()
    payload = yaml.safe_load((FIXTURES / "candidate.txt").read_text())
    branches = completeness.recognize_source_structure(source)
    principal = {
        rule["name"]: rule
        for rule in payload["rules"]
        if rule.get("kind") in {"derived", "derived_relation"}
    }
    issues = completeness._opaque_same_source_condition_input_issues(
        payload,
        source_text=source,
        branches=branches,
        principal_rules=principal,
        corpus_citation_path=provenance["citation"],
    )
    assert issues == []
    with monkeypatch.context() as prior:
        prior.setattr(
            completeness, "_worksheet_monetary_operation_antecedent", lambda _: None
        )
        original_issues = completeness._opaque_same_source_condition_input_issues(
            payload,
            source_text=source,
            branches=branches,
            principal_rules=principal,
            corpus_citation_path=provenance["citation"],
        )
    assert len(original_issues) == 1
    for name in (
        "net_income_for_part_year_resident",
        "provincial_or_territorial_tax_otherwise_payable",
        "provincial_territorial_2025_t2036_federal_credit_line_2",
        "t2036_net_income_for_credit",
    ):
        assert f"`{name}` versions[0]" in original_issues[0]
    # The separate short proof still owns two different operations; this patch
    # must not manufacture a unique owner or clear unrelated proof obligations.
    rule = principal["t2036_net_income_for_credit"]
    excerpts = [atom[2] for atom in completeness._rule_source_excerpt_atoms(rule)]
    excerpt = next(
        text for text in excerpts if text.strip().startswith("If you paid tax")
    )
    owned, ambiguous = completeness._source_condition_clauses_owned_by_excerpt(
        excerpt,
        rule=rule,
        source_text=source,
        branches=branches,
        corpus_citation_path=provenance["citation"],
        narrow_conjunctive_excerpt=False,
    )
    assert ambiguous
    assert {(clause.start, clause.end) for clause in owned} == {
        (4219, 4333),
        (5287, 5472),
    }

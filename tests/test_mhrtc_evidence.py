"""Private-owner controls: mocked lifecycle boundary, real closed source inventories.

No synthetic fixture here claims source/runtime authentication.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import mhrtc_capture as capture
from axiom_encode.harness import mhrtc_evidence as m
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.mhrtc_coverage import MhrtcCoverage
from axiom_encode.harness.mhrtc_program_contract import CITATION, SOURCE_TEXT
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests.test_bound_mhrtc_program import policy  # noqa: F401
from tests.test_mhrtc_source_inventory import inventory  # noqa: F401


@pytest.fixture
def bound(inventory, monkeypatch):  # noqa: F811
    _, content, branches = inventory
    context = SimpleNamespace(source=SOURCE_TEXT)
    evidence = object()
    state = {"live": True, "calls": 0, "missing": ()}

    def bind(actual, text, source, cases):
        state["calls"] += 1
        assert actual is evidence and text == content and source == SOURCE_TEXT
        return context, object(), MhrtcCoverage((), state["missing"])

    monkeypatch.setattr(capture, "source_coverage_binding", bind)
    monkeypatch.setattr(
        capture,
        "source_binding_live",
        lambda ev, ctx: ev is evidence and ctx is context and state["live"],
    )
    owner = m.certify_mhrtc(content, SOURCE_TEXT, CITATION, [], evidence)
    assert owner.live()
    return owner, state, content, branches, evidence


def test_public_constructed_clone_and_missing_capture_have_no_authority():
    forged = m.MhrtcEvidence(True, SOURCE_TEXT, True)
    assert not forged.live() and not forged.owns_numeric(None)
    absent = m.certify_mhrtc("", SOURCE_TEXT, CITATION, [], None)
    assert absent.recognized and not absent.live() and absent.missing()
    assert absent.partition((object(),), "formula")[0] == ()
    assert absent.owns_numeric(None) and absent.numeric_owner(None) is None
    unknown = m.certify_mhrtc("", SOURCE_TEXT, "foreign", [], None)
    marker = object()
    assert unknown.partition((marker,), "formula") == ((marker,), ())
    assert not unknown.missing()


def test_exact_inventories_and_only_one_analysis(bound):
    owner, state, _, branches, _ = bound
    for category, kind in [
        ("formulas", "formula"),
        ("controls", "control"),
        ("exceptions", "exception"),
    ]:
        assert owner.partition(branches[category], kind) == ((), ())
        assert all(owner.covers_branch(b, kind) for b in branches[category])
    recall = s.authoritative_numeric_recall_text(
        SOURCE_TEXT, corpus_citation_path=CITATION
    )
    numbers = tuple(
        extract_typed_numeric_inventory_occurrences_from_text(recall, profile="legacy")
    )
    assert len(numbers) == 32 and owner.partition_numeric(numbers) == ((), ())
    assert all(owner.numeric_owner(o) is not None for o in numbers)
    assert state["calls"] == 1
    assert not replace(owner).live()
    state["live"] = False
    assert not owner.live() and owner.missing()
    assert owner.partition(branches["formulas"], "formula") == (
        (),
        branches["formulas"],
    )
    assert owner.numeric_owner(numbers[0]) is None
    assert state["calls"] == 1


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "metadata", "category", "bool"]
)
def test_branch_partition_is_full_typed_multiset(bound, mutation):
    owner, _, _, branches, _ = bound
    rows = branches["formulas"]
    kind = "formula"
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows = rows + (rows[0],)
    elif mutation == "metadata":
        rows = (replace(rows[0], label="forged"), *rows[1:])
    elif mutation == "category":
        kind = "control"
    else:
        rows = (replace(rows[0], start=True), *rows[1:])
    remaining, missing = owner.partition(rows, kind)
    assert remaining == () and missing[: len(rows)] == rows and missing


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "flag", "bool", "dict"])
def test_numeric_partition_is_full_typed_multiset(bound, mutation):
    owner, _, _, _, _ = bound
    recall = s.authoritative_numeric_recall_text(
        SOURCE_TEXT, corpus_citation_path=CITATION
    )
    rows = tuple(
        extract_typed_numeric_inventory_occurrences_from_text(recall, profile="legacy")
    )
    if mutation == "missing":
        rows = rows[:-1]
    elif mutation == "duplicate":
        rows = rows + (rows[0],)
    elif mutation == "flag":
        rows = (replace(rows[0], requires_month_context=True), *rows[1:])
    elif mutation == "bool":
        rows = (replace(rows[0], has_rate_context=1), *rows[1:])
    else:
        rows = ({}, *rows[1:])
    remaining, missing = owner.partition_numeric(rows)
    assert remaining == () and missing[: len(rows)] == rows and missing


def test_actual_gate_extractor_exact_map_and_no_containment(bound):
    owner, _, content, _, _ = bound
    payload = yaml.safe_load(content)
    seen = set()

    class Inspect:
        def covers_gate(self, name, rule, index, formula, start, end, clause):
            assert owner.covers_gate(name, rule, index, formula, start, end, clause)
            seen.add((name, clause.start, clause.end, clause.text))
            assert not owner.covers_gate(name, rule, True, formula, start, end, clause)
            assert not owner.covers_gate(
                name,
                rule,
                index,
                formula,
                start,
                end,
                replace(clause, start=clause.start + 1),
            )
            return True

    branches = s.recognize_source_structure(SOURCE_TEXT, corpus_citation_path=CITATION)
    s._opaque_same_source_condition_input_issues(
        payload,
        source_text=SOURCE_TEXT,
        consumer_content=content,
        branches=branches,
        corpus_citation_path=CITATION,
        principal_rules={
            r["name"]: r
            for r in payload["rules"]
            if r["kind"] in ("derived", "derived_relation")
        },
        care_source_text=SOURCE_TEXT,
        investment_evidence=Inspect(),
    )
    expected = {
        (name, a, b, SOURCE_TEXT[t:b])
        for name, spans in m._GATES.items()
        for a, b, t in spans
    }
    assert seen == expected and len(seen) == 103


@pytest.mark.parametrize(
    "mutation", ["rule", "formula", "period", "path", "text", "type"]
)
def test_gate_requires_exact_bound_rule_version_and_clause(bound, mutation):
    owner, _, content, _, _ = bound
    name = "qualifying_individual"
    rule = next(r for r in yaml.safe_load(content)["rules"] if r["name"] == name)
    a, b, t = m._GATES[name][0]
    clause = s._SourceConditionClause((), a, b, SOURCE_TEXT[t:b])
    formula = rule["versions"][0]["formula"]
    start = "2025-01-01"
    if mutation == "rule":
        rule["entity"] = "OtherClaim"
    elif mutation == "formula":
        formula = "true"
    elif mutation == "period":
        start = "2024-01-01"
    elif mutation == "path":
        clause = replace(clause, branch_path=("invented",))
    elif mutation == "text":
        clause = replace(clause, text=clause.text + " ")
    else:
        clause = SimpleNamespace(**vars(clause))
    assert not owner.covers_gate(name, rule, 0, formula, start, "2025-12-31", clause)


@pytest.mark.parametrize("mode", ["missing", "stale_after", "inventory", "source"])
def test_failed_certification_remains_reserved(bound, monkeypatch, mode):
    _, state, content, _, evidence = bound
    if mode == "missing":
        state["missing"] = ("required",)
    elif mode == "stale_after":
        state["live"] = False
    elif mode == "inventory":

        def invalid(*args):
            raise ValueError("inventory changed")

        monkeypatch.setattr(m, "_inventory", invalid)
    source = SOURCE_TEXT + "changed" if mode == "source" else SOURCE_TEXT
    result = m.certify_mhrtc(content, source, CITATION, [], evidence)
    assert not result.live() and result.missing() and result.owns_numeric(None)


def test_complete_inventory_omission_reports_exact_expected_objects(bound):
    owner, _, _, branches, _ = bound
    for category, kind in (
        ("formulas", "formula"),
        ("controls", "control"),
        ("exceptions", "exception"),
    ):
        assert owner.partition((), kind) == ((), branches[category])
    recall = s.authoritative_numeric_recall_text(
        SOURCE_TEXT, corpus_citation_path=CITATION
    )
    numeric = tuple(
        extract_typed_numeric_inventory_occurrences_from_text(recall, profile="legacy")
    )
    assert owner.partition_numeric(()) == ((), numeric)
    # Partial omission retains the exact absent item, not only surviving input.
    assert (
        owner.partition(branches["formulas"][:-1], "formula")[1][-1]
        == branches["formulas"][-1]
    )
    assert owner.partition_numeric(numeric[:-1])[1][-1] == numeric[-1]

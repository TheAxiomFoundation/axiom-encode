"""Closed policy source hooks with miniature mocked lifecycle transport.

The policy fixture covers all 99 declarations. The two companion cases are
transport fixtures, not source witnesses: private coverage is explicitly mocked.
Actual 120-case coverage remains separately reviewed runtime evidence.
"""

import copy
import json
from dataclasses import replace
from functools import partial
from importlib.resources import files
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import mhrtc_capture as c
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as v
from axiom_encode.harness.mhrtc_coverage import MhrtcCoverage
from axiom_encode.harness.mhrtc_program_contract import (
    CITATION,
    SOURCE_TEXT,
)
from tests.test_bound_mhrtc_program import policy as native_policy  # noqa: F401
from tests.test_mhrtc_request import fixture as transport_fixture  # noqa: F401

CHECK = "complete-source-unit:mhrtc-evidence/missing"


@pytest.fixture
def actual(monkeypatch, native_policy, transport_fixture):  # noqa: F811
    content = yaml.safe_dump(native_policy, sort_keys=False)
    case, _, _, _ = transport_fixture
    frozen = [copy.deepcopy(case), copy.deepcopy(case)]
    frozen[1]["name"] = "second frozen transport case"
    evidence = SimpleNamespace(bound=lambda *args: True)
    context = SimpleNamespace(source=SOURCE_TEXT)
    state = {"live": True, "calls": 0, "frozen": frozen, "evidence": evidence}

    def binding(e, text, source, cases):
        state["calls"] += 1
        if (
            not state["live"]
            or e is not evidence
            or text != content
            or source != SOURCE_TEXT
            or cases != frozen
        ):
            return None
        return context, object(), MhrtcCoverage((), ())

    monkeypatch.setattr(c, "source_coverage_binding", binding)
    monkeypatch.setattr(
        c,
        "source_binding_live",
        lambda e, ctx: state["live"] and e is evidence and ctx is context,
    )

    def analyze(cases=None, execution=evidence, extract=None):
        return s.analyze_complete_source_unit(
            content,
            SOURCE_TEXT,
            corpus_citation_path=CITATION,
            test_cases=copy.deepcopy(frozen) if cases is None else cases,
            extract_numeric_occurrences=extract
            or partial(
                v.extract_typed_numeric_inventory_occurrences_from_text,
                profile="legacy",
            ),
            extract_numeric_grounding_occurrences=v.extract_typed_numeric_occurrences_from_text,
            extract_named_scalars=v.extract_named_scalar_occurrences,
            numeric_value_is_grounded=v.numeric_value_is_grounded,
            source_context={CITATION: SOURCE_TEXT},
            execution_evidence=execution,
        )

    return analyze, state


def test_actual_closed_candidate_single_binding_and_full_inventories(
    actual, monkeypatch
):
    analyze, state = actual
    original = s._source_exception_branches
    counts = []

    def observe(*args, **kwargs):
        counts.append(len(kwargs["formula_branches"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(s, "_source_exception_branches", observe)
    result = analyze()
    assert not result.issues
    assert (
        result.source_numeric_occurrence_count,
        result.covered_source_numeric_occurrence_count,
        result.missing_source_numeric_occurrence_count,
    ) == (32, 32, 0)
    assert state["calls"] == 1 and len(counts) >= 2 and set(counts) == {9}


@pytest.mark.parametrize(
    "mutation", ["absent", "stale", "clone", "omitted", "altered", "empty", "collision"]
)
def test_failed_private_binding_cannot_use_generic_fallback(actual, mutation):
    analyze, state = actual
    cases = copy.deepcopy(state["frozen"])
    execution = state["evidence"]
    if mutation == "absent":
        execution = None
    elif mutation == "stale":
        state["live"] = False
    elif mutation == "clone":
        execution = SimpleNamespace(bound=lambda *a: True)
    elif mutation == "omitted":
        cases.pop()
    elif mutation == "altered":
        cases[0]["output"][next(iter(cases[0]["output"]))] = 999
    elif mutation == "empty":
        cases = []
    else:
        cases[0]["input"]["multigenerational_home_renovation_tax_credit_line_45355"] = 1
    result = analyze(cases, execution)
    assert any(getattr(i, "check", None) == CHECK for i in result.issues)
    assert result.covered_source_numeric_occurrence_count == 0
    if mutation == "empty":
        assert any(
            getattr(i, "check", None) == s.CHECK_TESTS_NO_SUITE.id
            for i in result.issues
        )
    if mutation == "collision":
        assert any(
            getattr(i, "check", None) == s.CHECK_TESTS_INPUT_SHADOWS_OUTPUT.id
            for i in result.issues
        )


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "metadata"])
def test_complete_numeric_partition_precedes_generic_grounding(actual, mutation):
    analyze, _ = actual

    def extract(text):
        rows = tuple(
            v.extract_typed_numeric_inventory_occurrences_from_text(
                text, profile="legacy"
            )
        )
        if mutation == "empty":
            return ()
        if mutation == "duplicate":
            return rows + (rows[0],) if rows else rows
        return (
            (replace(rows[0], requires_month_context=True), *rows[1:]) if rows else rows
        )

    result = analyze(extract=extract)
    assert any(getattr(i, "check", None) == CHECK for i in result.issues)
    assert result.covered_source_numeric_occurrence_count == 0
    assert result.missing_source_numeric_occurrence_count > 0


@pytest.mark.parametrize(
    "category,mutation",
    [
        ("formula", "empty"),
        ("control", "malformed"),
        ("exception", "empty"),
        ("exception", "duplicate"),
    ],
)
def test_reserved_branch_malformed_and_omitted_inventory(
    actual, monkeypatch, category, mutation
):
    analyze, _ = actual
    name = {
        "formula": "_source_formula_branches",
        "control": "_source_control_branches",
        "exception": "_source_exception_branches",
    }[category]
    original = getattr(s, name)
    calls = []

    def changed(*args, **kwargs):
        rows = tuple(original(*args, **kwargs))
        calls.append(True)
        if len(calls) == 1:
            return rows  # owner independently recomputes genuine inventory
        if mutation == "empty":
            return ()
        if mutation == "malformed":
            return ({"start": "invented"}, *rows[1:])
        return rows + (rows[0],)

    monkeypatch.setattr(s, name, changed)
    result = analyze()
    assert any(getattr(i, "check", None) == CHECK for i in result.issues)


def test_registry_declares_exact_typed_mandatory_source_evidence_contract():
    expected = {
        "category": "complete-source-unit:mhrtc-evidence",
        "cause": "encoder_output",
        "fix": "Correct the closed source-bound program and complete factual companion, then rerun ordinary validation with the authenticated release and receipted pinned engine; do not weaken predicates or substitute generic witnesses.",
        "gate": "ci",
        "id": "complete-source-unit:mhrtc-evidence/missing",
        "requires": "Every exact MHRtC source occurrence and supplemental role has complete fresh private principal, parameter and auxiliary evidence.",
        "severity": "important",
        "site": "axiom_encode/harness/source_completeness.py:analyze_complete_source_unit",
        "title": "MHRtC lacks complete private source execution evidence",
        "why": "Generic scalar assertions, missing or duplicated inventories, and stale or detached execution cannot establish the source-bound joint conditions.",
    }
    registry = json.loads(
        files("axiom_encode").joinpath("data/check_registry.json").read_text()
    )
    matching = [row for row in registry["checks"] if row["id"] == CHECK]
    assert matching == [expected]
    assert s.CHECK_MHRTC_EXECUTION_EVIDENCE.to_dict() == expected


def test_caller_built_owner_cannot_delegate_companion_checks():
    from axiom_encode.harness.mhrtc_evidence import MhrtcEvidence

    issues = s._companion_test_issues(
        {},
        parameter_rules={},
        principal_rule_paths={},
        principal_formula_clause_rules={},
        formula_branches=(),
        branches=(),
        source_text=SOURCE_TEXT,
        corpus_citation_path=CITATION,
        deferred_paths=set(),
        test_cases=[{"name": "synthetic", "input": {}, "output": {}}],
        extract_numeric_occurrences=lambda _: (),
        numeric_value_is_grounded=lambda *a: False,
        formula_environment={},
        source_bound_constant_occurrences={},
        declared_input_names=set(),
        mhrtc_evidence=MhrtcEvidence(True, SOURCE_TEXT, True),
        mhrtc_source_formula_branches=(),
    )
    assert any(getattr(i, "check", None) == CHECK for i in issues)

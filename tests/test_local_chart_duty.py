"""Chart-specific source owner and saved-execution seam; no admission claim."""

import ast
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness import source_completeness as source
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from axiom_encode.harness.local_pair_source_evidence import (
    bind_chart_duty,
    reserves_chart_duty,
)
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

NAME = "t936_should_first_complete_chart_a"
SELECTOR = "t936_other_than_qffp_qsbcs_gains_2025"
BRANCH = source.SourceStructureBranch(
    (), "condition", "chart", BODY[775:1075], 775, 1075
)
RECORDS = json.loads(
    (Path(__file__).parent / "fixtures/t936_chart_duty_cases.json").read_text()
)["rows"]


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "formula",
        "bytesformula",
        "astformula",
        "proof",
        "proofkind",
        "default",
        "unit",
        "entity",
        "year",
        "description",
        "inputtype",
        "duplicate",
        "rounding",
    ],
)
def test_closed_chart_binding(change):
    p = deepcopy(DATA)
    r = next(r for r in p["rules"] if r["name"] == NAME)
    d = next(d for d in p["inputs"] if d["name"] == SELECTOR)
    if change == "formula":
        r["versions"][0]["formula"] = "t936_had_investment_income_2025"
    elif change == "bytesformula":
        r["versions"][0]["formula"] = SELECTOR.encode()
    elif change == "astformula":
        r["versions"][0]["formula"] = ast.parse(SELECTOR)
    elif change == "proof":
        r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "first complete Chart A"
        )
    elif change == "proofkind":
        r["metadata"]["proof"]["atoms"][0]["kind"] = "comment"
    elif change == "default":
        d["default"] = False
    elif change == "unit":
        r["unit"] = "CAD"
    elif change == "entity":
        r["entity"] = "Other"
    elif change == "year":
        r["versions"][0]["effective_to"] = "2026-12-31"
    elif change == "description":
        d["description"] = "Had gains"
    elif change == "inputtype":
        d["dtype"] = "Integer"
    elif change == "duplicate":
        p["inputs"].append(deepcopy(d))
    elif change == "rounding":
        r["resolved_currency_rounding"] = "nearest"
    assert reserves_chart_duty(BRANCH, BODY, CITE)
    assert bool(bind_chart_duty(p, BRANCH, BODY, CITE)) is (change == "none")


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "missingcase",
        "query",
        "assertion",
        "principal",
        "unrelated",
        "expired",
        "finalexpiry",
        "period",
        "midyear",
        "extraperiod",
        "source",
        "casecontent",
    ],
)
def test_chart_uses_own_complete_live_pair(monkeypatch, change):
    records = deepcopy(RECORDS)
    if change == "missingcase":
        records = records[:1]
    raw = [r["case"] for r in records]
    policy, cases = adapter._normalized_source_inputs(deepcopy(DATA), raw)
    obs = [
        (
            frozenset({adapter._source_key(c)}),
            r["request"],
            r["response"],
            _replay_values(DATA, r["request"], r["response"]),
        )
        for r, c in zip(records, cases)
    ]
    bound = SimpleNamespace(
        policy=policy,
        cases=raw,
        content="immutable",
        sources={CITE: BODY},
        observations=obs,
        fresh=lambda: True,
    )
    context = object()
    monkeypatch.setitem(adapter._contexts, context, bound)
    rules = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    source = BODY
    if change == "query":
        for o in obs:
            o[1]["queries"][0]["outputs"] = []
    elif change == "assertion":
        for c in cases:
            c["output"] = {}
    elif change in ("principal", "unrelated"):
        rules = deepcopy(rules)
        name = NAME if change == "principal" else "t936_should_complete_form"
        rules[name]["versions"][0]["formula"] = "False"
    elif change == "expired":
        bound.fresh = lambda: False
    elif change == "finalexpiry":
        original = adapter.boolean_pair_witnesses

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            bound.fresh = lambda: False
            return result

        monkeypatch.setattr(adapter, "boolean_pair_witnesses", expire)
    elif change in ("period", "midyear", "extraperiod"):
        for o in obs:
            p = o[1]["queries"][0]["period"]
            if change == "period":
                p["start"] = "2024-01-01"
            elif change == "midyear":
                p["start"] = "2025-06-01"
            else:
                p["extra"] = True
    elif change == "source":
        source += "changed"
    elif change == "casecontent":
        cases[0]["name"] = "altered"
    result = adapter.chart_duty_witnesses(
        context,
        BRANCH,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=source,
        citation=CITE,
    )
    assert bool(result) is (change == "none")
    assert not adapter._witnesses
    if result:
        assert len(result) == 1
        assert next(iter(result)) == (NAME, tuple(sorted(map(id, cases))))
        from axiom_encode.harness.source_completeness import (
            _unmatched_evidence_obligations,
        )

        assert (
            len(_unmatched_evidence_obligations({"one": result, "duplicate": result}))
            == 1
        )


@pytest.mark.parametrize("context_present", [False, True])
@pytest.mark.parametrize("composite", [False, True])
def test_chart_duty_bypasses_permissive_fallbacks(
    monkeypatch, context_present, composite
):
    seen = []

    def generic(branches, **kwargs):
        seen.extend(branches)
        return ()  # Deliberately accepts everything it sees.

    def unmatched(branches):
        seen.extend(branches)
        return ()

    monkeypatch.setattr(source, "_source_exception_branches", lambda *a, **k: (BRANCH,))
    monkeypatch.setattr(
        source, "_source_exception_requires_paired_witness", lambda *a, **k: True
    )
    monkeypatch.setattr(source, "_unwitnessed_exception_branches", generic)
    monkeypatch.setattr(
        source,
        "certify_reporting_routes",
        lambda *a, **k: SimpleNamespace(unmatched=unmatched),
    )
    monkeypatch.setattr(
        source,
        "certify_partnership_reporting",
        lambda *a, **k: SimpleNamespace(unmatched=unmatched),
    )
    monkeypatch.setattr(adapter, "numeric_pair_witnesses", lambda *a, **k: set())
    monkeypatch.setattr(adapter, "boolean_pair_witnesses", lambda *a, **k: set())
    signed_seen = []

    def signed(context, branch, **kwargs):
        signed_seen.append(branch)
        return {("owned", (1, 2, 3))} if composite else set()

    monkeypatch.setattr(adapter, "chart_duty_witnesses", signed)
    issues = source._companion_test_issues(
        {},
        parameter_rules={},
        principal_rule_paths={},
        principal_formula_clause_rules={},
        formula_branches=(),
        branches=(),
        source_text=BODY,
        corpus_citation_path=CITE,
        deferred_paths=set(),
        test_cases=[{"name": "seam", "input": {}, "output": {}}],
        extract_numeric_occurrences=lambda *a, **k: (),
        numeric_value_is_grounded=lambda *a, **k: False,
        formula_environment={},
        source_bound_constant_occurrences={},
        declared_input_names=set(),
        partnership_content="unchanged",
        local_relation_context=object() if context_present else None,
    )
    assert BRANCH not in seen
    assert signed_seen == ([BRANCH] if context_present else [])
    assert any("paired positive/blocking" in str(i) for i in issues) is not (
        context_present and composite
    )

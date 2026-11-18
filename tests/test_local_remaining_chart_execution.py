"""Injected saved-execution seams are regression evidence, not admission."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

RECORDS = json.loads(
    (Path(__file__).parent / "fixtures/t936_remaining_chart_cases.json").read_text()
)["rows"]


@pytest.mark.parametrize(
    "span", [(7150, 7346), (8017, 8067), (8068, 8159), (8160, 8308)]
)
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "expired",
        "query",
        "assertion",
        "case",
        "source",
        "declaration",
        "period",
        "entity",
        "secondfact",
    ],
)
def test_complete_roles(monkeypatch, span, mutation):
    records = deepcopy(RECORDS)
    raw = [r["case"] for r in records]
    policy, cases = adapter._normalized_source_inputs(deepcopy(DATA), raw)
    observations = [
        (
            frozenset({adapter._source_key(c)}),
            r["request"],
            r["response"],
            _replay_values(DATA, r["request"], r["response"]),
        )
        for c, r in zip(cases, records)
    ]
    bound = SimpleNamespace(
        policy=policy,
        cases=raw,
        content="immutable",
        sources={CITE: BODY},
        observations=observations,
        fresh=lambda: True,
    )
    context = object()
    monkeypatch.setitem(adapter._contexts, context, bound)
    rules = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    source = BODY
    if mutation == "expired":
        bound.fresh = lambda: False
    elif mutation == "query":
        for o in observations:
            o[1]["queries"][0]["outputs"] = []
    elif mutation == "assertion":
        for c in cases:
            c["output"] = {}
    elif mutation in ("period", "entity", "rate", "secondfact"):
        for o in observations:
            if mutation == "period":
                o[1]["queries"][0]["period"]["start"] = "2025-06-01"
            elif mutation == "entity":
                o[1]["queries"][0]["entity_id"] = "other"
            elif mutation == "secondfact":
                o[1]["dataset"]["inputs"][1]["value"]["value"] += 99
            else:
                for result in o[2]["results"]:
                    for node in result["trace"].values():
                        for read in node.get("parameter_reads", []):
                            read["index"] = 99
    elif mutation in ("only19", "only20"):
        name = (
            "federal_2025_t936_chart_a_line_19"
            if mutation == "only19"
            else "federal_2025_t936_chart_a_additional_investment_income_line_20"
        )
        for o in observations:
            o[1]["queries"][0]["outputs"] = [
                x for x in o[1]["queries"][0]["outputs"] if x.split("#")[-1] == name
            ]
    elif mutation == "foreignids":
        from dataclasses import replace

        original = adapter.boolean_pair_witnesses

        def foreign(*a, **kw):
            return {replace(w, case_pair_identity=(1, 2)) for w in original(*a, **kw)}

        monkeypatch.setattr(adapter, "boolean_pair_witnesses", foreign)
    elif mutation == "borrowedshortcut":
        from dataclasses import replace

        original = adapter.corroborated_execution

        def borrowed(*a, **kw):
            result = original(*a, **kw)
            if (
                result
                and result.leaf == "0"
                and result.trace
                and result.trace[0].choice == 1
            ):
                result = replace(
                    result,
                    trace=(
                        replace(
                            result.trace[0],
                            choice=0,
                            selectors=result.trace[0].selectors[:1],
                        ),
                    ),
                )
            return result

        monkeypatch.setattr(adapter, "corroborated_execution", borrowed)
    elif mutation == "case":
        cases[0]["name"] = "altered"
    elif mutation == "source":
        source += "changed"
    elif mutation == "declaration":
        rules = deepcopy(rules)
        rules["t936_should_complete_form"]["versions"][0]["formula"] = "False"
    branch = SimpleNamespace(start=span[0], end=span[1], text=BODY[span[0] : span[1]])

    def run():
        return adapter.remaining_chart_witnesses(
            context,
            branch,
            content="immutable",
            cases=cases,
            principal_rules=rules,
            source_text=source,
            citation=CITE,
        )

    if mutation == "finalexpiry":
        count = [0]

        def counting():
            count[0] += 1
            return True

        bound.fresh = counting
        assert run()
        last = count[0]
        count[0] = 0

        def final_expiry():
            count[0] += 1
            return count[0] < last

        bound.fresh = final_expiry
    found = run()
    assert bool(found) == (mutation == "none")
    assert len(found) <= 1
    assert not adapter._witnesses


def test_t3_requires_actual_selected_rate(monkeypatch):
    test_complete_roles(monkeypatch, (8160, 8308), "rate")


@pytest.mark.parametrize("mutation", ["only19", "only20", "foreignids"])
def test_t3_both_own_principals_original_ids(monkeypatch, mutation):
    test_complete_roles(monkeypatch, (8160, 8308), mutation)


def test_second_shortcut_cannot_borrow_first(monkeypatch):
    test_complete_roles(monkeypatch, (8068, 8159), "borrowedshortcut")


@pytest.mark.parametrize(
    "span", [(7150, 7346), (8017, 8067), (8068, 8159), (8160, 8308)]
)
def test_final_expiry_refuses_each_role(monkeypatch, span):
    test_complete_roles(monkeypatch, span, "finalexpiry")

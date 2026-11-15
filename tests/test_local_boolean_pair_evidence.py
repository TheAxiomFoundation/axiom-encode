"""Captured pair and negative adapter seams, not live evidence capabilities."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness import source_completeness as source
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

RECORDS = json.loads(
    (Path(__file__).parent / "fixtures/t936_boolean_pair_cases.json").read_text()
)


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "source",
        "assertion",
        "query",
        "entity",
        "period",
        "twofacts",
        "numeric",
        "forged",
        "expired",
        "finalexpiry",
        "principal",
        "derived",
    ],
)
def test_boolean_pair_private_scope(monkeypatch, mutation):
    records = deepcopy(RECORDS)
    raw = [r["case"] for r in records]
    policy, cases = adapter._normalized_source_inputs(deepcopy(DATA), raw)
    obs = [
        (
            frozenset({adapter._source_key(case)}),
            r["request"],
            r["response"],
            _replay_values(DATA, r["request"], r["response"]),
        )
        for r, case in zip(records, cases)
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
    text = BODY
    if mutation == "source":
        text += " altered"
    elif mutation == "assertion":
        cases[0]["output"] = {}
    elif mutation == "query":
        obs[0][1]["queries"][0]["outputs"] = []
    elif mutation == "entity":
        obs[0][1]["queries"][0]["entity_id"] = "other"
    elif mutation == "period":
        obs[0][1]["queries"][0]["period"]["end"] = "2026-12-31"
    elif mutation == "twofacts":
        obs[0][1]["dataset"]["inputs"][0]["value"]["value"] = 101
    elif mutation == "numeric":
        for o in obs:
            for fact in o[1]["dataset"]["inputs"]:
                if fact["name"].endswith("#input.t936_chart_a_completed"):
                    fact["value"] = {
                        "kind": "integer",
                        "value": int(fact["value"]["value"]),
                    }
    elif mutation == "forged":
        context = object()
    elif mutation == "expired":
        bound.fresh = lambda: False
    elif mutation == "finalexpiry":
        original = source._exception_witness_for_case_pair

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            bound.fresh = lambda: False
            return result

        monkeypatch.setattr(source, "_exception_witness_for_case_pair", expire)
    elif mutation == "derived":
        original = source._rule_exception_selector_names

        def derived_only(rule):
            return {"t936_should_complete_form"} if original(rule) else set()

        monkeypatch.setattr(source, "_rule_exception_selector_names", derived_only)
    elif mutation == "principal":
        rules = deepcopy(rules)
        for r in rules.values():
            for v in r.get("versions", []):
                v["formula"] = "0"
    result = adapter.boolean_pair_witnesses(
        context,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=text,
        citation=CITE,
    )
    assert bool(result) is (mutation == "none")
    if mutation == "none":
        assert len({w.rule_name for w in result}) == 2
        assert (
            adapter._pair_changed_fact(records[0]["request"], records[1]["request"])
            is None
        )
        assert (
            adapter._pair_changed_fact(
                records[0]["request"], records[1]["request"], boolean=True
            )
            == "t936_chart_a_completed"
        )
    assert not adapter._witnesses


def test_boolean_mode_is_explicit_and_not_numeric_truthiness():
    left, right = [deepcopy(r["request"]) for r in RECORDS]
    for mode in (1, 0, "true", None):
        assert adapter._pair_changed_fact(left, right, boolean=mode) is None
    assert adapter._pair_changed_fact(left, deepcopy(left), boolean=True) is None

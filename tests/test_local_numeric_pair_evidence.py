"""Private adapter seam controls; saved captures are not fresh authority."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

RECORDS = json.loads(
    (Path(__file__).parent / "fixtures/t936_numeric_pair_cases.json").read_text()
)


@pytest.mark.parametrize("indices", [(61, 70), (61, 71), (0, 68), (0, 69)])
def test_actual_request_shape_one_root_value(indices):
    a, b = [deepcopy(next(r for r in RECORDS if r["index"] == i)) for i in indices]
    assert adapter._pair_changed_fact(a["request"], b["request"])


@pytest.mark.parametrize(
    "mutation",
    ["period", "entity", "relations", "extra_fact", "two", "boolean", "provider"],
)
def test_request_shape_refusals(mutation):
    a, b = [
        deepcopy(next(r for r in RECORDS if r["index"] == i)["request"])
        for i in (61, 70)
    ]
    if mutation == "period":
        b["queries"][0]["period"]["end"] = "2026-12-31"
    elif mutation == "entity":
        b["queries"][0]["entity_id"] = "other"
    elif mutation == "relations":
        b["dataset"]["relations"].append({"bad": "edge"})
    elif mutation == "extra_fact":
        b["dataset"]["inputs"].append(deepcopy(b["dataset"]["inputs"][0]))
    elif mutation == "two":
        b["dataset"]["inputs"][0]["value"]["value"] = 101
    else:
        ix = next(
            i
            for i, (x, y) in enumerate(
                zip(a["dataset"]["inputs"], b["dataset"]["inputs"])
            )
            if x != y
        )
        if mutation == "boolean":
            b["dataset"]["inputs"][ix]["value"] = {"kind": "boolean", "value": True}
        else:
            for req in (a, b):
                req["dataset"]["inputs"][ix]["entity_id"] = "provider"
    assert adapter._pair_changed_fact(a, b) is None


@pytest.mark.parametrize(
    "indices,expect",
    [
        ((61, 70), True),
        ((61, 71), True),
        ((0, 68), True),
        ((0, 69), True),
        ((66, 77), True),
        ((66, 67), False),
        ((68, 69), False),
        ((70, 71), False),
    ],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "source",
        "principal",
        "query",
        "wholecase",
        "finalexpiry",
        "assertion",
        "forged",
    ],
)
def test_private_pair_causal_witness(monkeypatch, indices, expect, mutation):
    records = [deepcopy(next(r for r in RECORDS if r["index"] == i)) for i in indices]
    raw = [r["case"] for r in records]
    policy, cases = adapter._normalized_source_inputs(deepcopy(DATA), raw)
    obs = []
    for r, c in zip(records, cases):
        obs.append(
            (
                frozenset({adapter._source_key(c)}),
                r["request"],
                r["response"],
                _replay_values(DATA, r["request"], r["response"]),
            )
        )
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
    source_text = BODY
    if mutation == "source":
        source_text += " changed"
    elif mutation == "principal":
        for rule in rules.values():
            for version in rule.get("versions", []):
                version["formula"] = "0"
        # Keep authoritative policy independent from mutated caller declarations.
        bound.policy, _ = adapter._normalized_source_inputs(deepcopy(DATA), raw)
    elif mutation == "query":
        for observation in obs:
            observation[1]["queries"][0]["outputs"] = []
    elif mutation == "assertion":
        for case in cases:
            case["output"] = {}
    elif mutation == "forged":
        context = object()
    elif mutation == "wholecase":
        cases[0]["input"][next(iter(cases[0]["input"]))] = 999999
    elif mutation == "finalexpiry":
        from axiom_encode.harness import source_completeness as source

        original = source._formula_relational_transitions

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            bound.fresh = lambda: False
            return result

        monkeypatch.setattr(source, "_formula_relational_transitions", expire)
    result = adapter.numeric_pair_witnesses(
        context,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=source_text,
        citation=CITE,
    )
    assert bool(result) is (expect and mutation == "none")
    if expect and mutation == "none":
        assert any(w.zeroes for w in result)
        assert any(not w.zeroes for w in result)
    assert not adapter._witnesses
    bound.fresh = lambda: False
    assert not adapter.numeric_pair_witnesses(
        context,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=BODY,
        citation=CITE,
    )

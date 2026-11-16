"""Saved real fixture responses under an injected seam, not live authority."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from axiom_encode.harness.local_pair_source_evidence import _SIGNED_PROFILES
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

RECORDS = json.loads(
    (Path(__file__).parent / "fixtures/t936_signed_pair_cases.json").read_text()
)


@pytest.mark.parametrize("name", _SIGNED_PROFILES)
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "alternative_positive",
        "no_negative",
        "no_gate",
        "no_positive",
        "source",
        "query",
        "assertion",
        "declaration",
        "expiry",
        "finalexpiry",
        "twofacts",
        "identity",
        "numericbool",
    ],
)
def test_signed_composite_requires_each_obligation(monkeypatch, name, mutation):
    records = deepcopy(RECORDS)
    if mutation == "alternative_positive":
        extra = deepcopy(records[0])
        extra["index"] = 80
        extra["case"]["name"] += "_alternative"
        records.append(extra)
    elif mutation == "no_negative":
        records = [r for r in records if r["index"] != 78]
    elif mutation == "no_gate":
        records = [r for r in records if r["index"] != 79]
    elif mutation == "no_positive":
        records = [r for r in records if r["index"] != 0]
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
    span, _, _ = _SIGNED_PROFILES[name]
    branch = SimpleNamespace(start=span[0], end=span[1], text=BODY[span[0] : span[1]])
    text = BODY
    if mutation == "source":
        text += " altered"
    elif mutation == "query":
        for o in obs:
            o[1]["queries"][0]["outputs"] = []
    elif mutation == "assertion":
        for c in cases:
            c["output"] = {}
    elif mutation == "declaration":
        rules = deepcopy(rules)
        rules[name]["versions"][0]["formula"] = "0"
    elif mutation == "expiry":
        bound.fresh = lambda: False
    elif mutation == "finalexpiry":
        original = adapter.corroborated_execution

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            bound.fresh = lambda: False
            return result

        monkeypatch.setattr(adapter, "corroborated_execution", expire)
    elif mutation == "identity":
        obs[1][1]["queries"][0]["entity_id"] = "other"
    elif mutation == "twofacts":
        obs[2][1]["dataset"]["inputs"].append(
            deepcopy(obs[2][1]["dataset"]["inputs"][0])
        )
    elif mutation == "numericbool":
        for o in obs:
            for f in o[1]["dataset"]["inputs"]:
                if f["name"].endswith("schedule_3_line_19200_reported"):
                    f["value"] = {"kind": "integer", "value": int(f["value"]["value"])}
    result = adapter.signed_branch_witnesses(
        context,
        branch,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=text,
        citation=CITE,
    )
    # This injected seam bypasses the real seal; only row8 owns the raw gate.
    # Whole-record mutation rejection is separately enforced by live custody.
    expected = mutation in ("none", "alternative_positive") or (
        mutation in ("no_gate", "twofacts", "numericbool")
        and name != "t936_chart_a_line_8"
    )
    assert bool(result) is expected
    assert not adapter._witnesses
    if result:
        from axiom_encode.harness.source_completeness import (
            _unmatched_evidence_obligations,
        )

        assert (
            len(_unmatched_evidence_obligations({"a": result, "duplicate": result}))
            == 1
        )
        assert all(
            len(token[1]) == (3 if name == "t936_chart_a_line_8" else 2)
            for token in result
        )

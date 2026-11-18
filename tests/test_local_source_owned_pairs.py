"""Saved actual records exercise an injected private seam, not live authority."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from axiom_encode.harness.local_pair_source_evidence import _PROFILES
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

FIXTURES = Path(__file__).parent / "fixtures"
NUMERIC = json.loads((FIXTURES / "t936_numeric_pair_cases.json").read_text())
BOOLEAN = json.loads((FIXTURES / "t936_boolean_pair_cases.json").read_text())
for record, ordinal in zip(BOOLEAN, (8, 76), strict=True):
    record["index"] = ordinal


@pytest.mark.parametrize(
    "role,indices,expected",
    [
        ("chart_completed_expense", (8, 76), True),
        ("chart_completed_income", (8, 76), True),
        ("cnil_floor", (61, 71), True),
        ("cnil_floor", (61, 70), False),
        ("chart20_floor", (0, 69), True),
        ("chart20_floor", (0, 68), False),
        ("chart11_floor", (66, 77), True),
        ("chart11_floor", (66, 67), False),
    ],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "source",
        "assertion",
        "query",
        "declaration",
        "expiry",
        "finalexpiry",
        "twofacts",
    ],
)
def test_source_owned_pairs(monkeypatch, role, indices, expected, mutation):
    records = [
        deepcopy(next(r for r in NUMERIC + BOOLEAN if r["index"] == i)) for i in indices
    ]
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
    p = _PROFILES[role]
    branch = SimpleNamespace(start=p[0], end=p[1], text=p[5])
    text = BODY
    if mutation == "source":
        text += " altered"
    elif mutation == "assertion":
        for c in cases:
            c["output"] = {}
    elif mutation == "query":
        for o in obs:
            o[1]["queries"][0]["outputs"] = []
    elif mutation == "declaration":
        rules = deepcopy(rules)
        rules[p[2]]["versions"][0]["formula"] = "0"
    elif mutation == "expiry":
        bound.fresh = lambda: False
    elif mutation == "finalexpiry":
        original = adapter._pair_changed_fact

        def expire(*args, **kwargs):
            result = original(*args, **kwargs)
            bound.fresh = lambda: False
            return result

        monkeypatch.setattr(adapter, "_pair_changed_fact", expire)
    elif mutation == "twofacts":
        obs[1][1]["dataset"]["inputs"].append(
            deepcopy(obs[1][1]["dataset"]["inputs"][0])
        )
    found = adapter.paired_branch_witnesses(
        context,
        branch,
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=text,
        citation=CITE,
    )
    assert bool(found) is (expected and mutation == "none")
    assert not adapter._witnesses
    if found:
        from axiom_encode.harness.source_completeness import (
            _unmatched_evidence_obligations,
        )

        assert (
            len(_unmatched_evidence_obligations({"first": found, "duplicate": found}))
            == 1
        )

"""Captured seams exercise private roles; fixtures do not grant admission."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

ROWS = json.loads(
    (Path(__file__).parent / "fixtures/t936_expense_group_cases.json").read_text()
)["rows"]


@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "expired",
        "case",
        "no_timber",
        "no_category",
        "no_aggregate",
        "no_selector",
        "no_included",
        "provider_id",
        "provider_edge",
        "provider_interval",
        "provider_amount",
        "no_permutation",
        "no_aggregate_equality",
        "final_expiry",
        "aggregate_below_cap",
        *[
            f"band{cap}"
            for cap in (30000, 34000, 36000, 37000, 38000, 55000, 59000, 61000)
        ],
    ],
)
def test_whole_group(monkeypatch, mutation):
    rows = deepcopy(ROWS)
    if mutation == "no_timber":
        rows.pop()
    if mutation == "no_permutation":
        rows.pop(49)
    if mutation == "no_aggregate_equality":
        rows.pop(50)
    if mutation.startswith("band"):
        cap = int(mutation[4:])
        rows = [
            r
            for r in rows
            if not any(
                read.get("name") == f"t936_vehicle_cost_{cap}"
                for node in r["response"]["results"][0]["trace"].values()
                if node["entity_id"] == r["request"]["queries"][0]["entity_id"]
                for read in node.get("parameter_reads", [])
            )
        ]
    policy, cases = adapter._normalized_source_inputs(
        deepcopy(DATA), [r["case"] for r in rows]
    )
    observations = [
        (
            frozenset({adapter._source_key(c)}),
            r["request"],
            r["response"],
            _replay_values(DATA, r["request"], r["response"]),
        )
        for c, r in zip(cases, rows, strict=True)
    ]
    bound = SimpleNamespace(
        policy=policy,
        cases=[r["case"] for r in rows],
        content="immutable",
        sources={CITE: BODY},
        observations=observations,
        fresh=lambda: True,
    )
    ctx = object()
    monkeypatch.setitem(adapter._contexts, ctx, bound)
    rules = {r["name"]: r for r in policy["rules"] if r.get("kind") == "derived"}
    if mutation == "expired":
        bound.fresh = lambda: False
    if mutation == "case":
        cases[0]["name"] = "changed"
    output = {
        "no_category": "other_investment_expenses_line_68080",
        "no_aggregate": "t936_uncollectible_dispositions_total",
        "no_selector": "t936_ordinary_vehicle_cost_excluded",
        "no_included": "t936_asset_included_uncollectible_portion",
    }.get(mutation)
    if output:
        for o in observations:
            o[1]["queries"][0]["outputs"] = [
                x for x in o[1]["queries"][0]["outputs"] if x.split("#")[-1] != output
            ]
    if mutation == "aggregate_below_cap":
        # Both equality and a lower cost produce170, but only equality satisfies
        # the required aggregation boundary control. Preserve that distinction.
        for o in observations:
            if any(
                x.endswith("#t936_uncollectible_dispositions_total")
                for x in o[1]["queries"][0]["outputs"]
            ):
                for fact in o[1]["dataset"]["inputs"]:
                    if fact["name"].endswith(
                        "#input.t936_asset_acquisition_cost"
                    ) and fact["value"] == {"kind": "integer", "value": 38000}:
                        fact["value"]["value"] = 37999
    if mutation.startswith("provider_"):
        for o in observations:
            for node in o[2]["results"][0]["trace"].values():
                if (
                    node["name"] == "t936_asset_included_uncollectible_portion"
                    and node["entity_id"] != o[1]["queries"][0]["entity_id"]
                ):
                    if mutation == "provider_id":
                        node["entity_id"] = "foreign"
                    if mutation == "provider_amount":
                        node["value"] = {"kind": "decimal", "value": "999"}
            if o[1]["dataset"]["relations"]:
                if mutation == "provider_edge":
                    o[1]["dataset"]["relations"].pop()
                if mutation == "provider_interval":
                    o[1]["dataset"]["relations"][0]["interval"]["start"] = "2025-06-01"
    if mutation == "final_expiry":
        calls = [0]

        def count():
            calls[0] += 1
            return True

        bound.fresh = count
        baseline = adapter.expense_group_witnesses(
            ctx,
            SimpleNamespace(start=2877, end=3984, text=BODY[2877:3984]),
            content="immutable",
            cases=cases,
            principal_rules=rules,
            source_text=BODY,
            citation=CITE,
        )
        assert baseline
        final = calls[0]
        calls[0] = 0

        def expire():
            calls[0] += 1
            return calls[0] < final

        bound.fresh = expire
    result = adapter.expense_group_witnesses(
        ctx,
        SimpleNamespace(start=2877, end=3984, text=BODY[2877:3984]),
        content="immutable",
        cases=cases,
        principal_rules=rules,
        source_text=BODY,
        citation=CITE,
    )
    assert bool(result) == (mutation == "none")
    if result:
        assert len(result) == 1

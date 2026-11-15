"""Closed static binding controls, never runtime or source admission evidence."""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.local_vehicle_date_evidence import vehicle_date_role_binding

DATA = yaml.safe_load(
    json.loads(
        (Path(__file__).parent / "fixtures/t936_vehicle_dates.json").read_text()
    )["policy_yaml"]
)
BODY = (Path(__file__).parent / "fixtures/multipage_t936_source.txt").read_text()
YEAR = {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"}
CITE = "ca/policy/cra/t1-2025/cumulative-net-investment-loss"
EXPECTED = [
    (3630, None, "2022-01-01", 30000),
    (3669, "2022-01-01", "2023-01-01", 34000),
    (3708, "2023-01-01", "2024-01-01", 36000),
    (3747, "2024-01-01", "2025-01-01", 37000),
    (3937, "2022-01-01", "2023-01-01", 59000),
    (3979, "2023-01-01", None, 61000),
]


@pytest.mark.parametrize("start,lower,upper,cap", EXPECTED)
def test_whole_ordered_interval_not_year_scalar(start, lower, upper, cap):
    result = vehicle_date_role_binding(
        DATA, BODY, CITE, {"period": YEAR}, (start, start + 4)
    )
    assert result is not None
    assert (result.lower, result.upper, result.cap) == (
        date.fromisoformat(lower) if lower else None,
        date.fromisoformat(upper) if upper else None,
        cap,
    )


@pytest.mark.parametrize("start", [5792, 5836, 6189, 6234, 3631])
def test_prior_form_or_relocated_same_year_not_represented(start):
    assert (
        vehicle_date_role_binding(
            DATA, BODY, CITE, {"period": YEAR}, (start, start + 4)
        )
        is None
    )


@pytest.mark.parametrize("period", [None, "2024-01-01", "2026-01-01", "not-a-date"])
def test_no_missing_or_outside_query_period(period):
    case = {} if period is None else {"period": period}
    assert vehicle_date_role_binding(DATA, BODY, CITE, case, (3630, 3634)) is None


@pytest.mark.parametrize("start", [3630, 3979])
@pytest.mark.parametrize(
    "change",
    [
        "date_day",
        "date_year",
        "date_string_index",
        "index_bool",
        "overlap",
        "date_default",
        "date_proof",
        "header",
        "cost_value",
        "cost_unit",
        "cost_proof",
        "selector_operator",
        "selector_cap",
        "selector_bound",
        "selector_proof",
        "propagation",
        "timber",
        "principal_unit",
        "input_date",
        "input_meaning",
        "input_default",
        "duplicate",
        "wrong_source",
        "wrong_citation",
    ],
)
def test_misbound_semantics_refused(start, change):
    p = deepcopy(DATA)
    r = {x["name"]: x for x in p["rules"]}
    inputs = {x["name"]: x for x in p["inputs"]}
    body = BODY
    cite = CITE
    d = r["t936_start_2023"]
    v = d["versions"][0]
    selector = r[
        "t936_ordinary_vehicle_cost_excluded"
        if start == 3630
        else "t936_zepv_cost_excluded"
    ]
    cap = r["t936_vehicle_cost_61000"]
    principal = r["t936_asset_included_uncollectible_portion"]
    if change == "date_day":
        v["values"][2025] = "2023-01-02"
    elif change == "date_year":
        v["values"][2025] = "2022-01-01"
    elif change == "date_string_index":
        v["values"] = {"2025": "2023-01-01"}
    elif change == "index_bool":
        r["t936_source_year"]["versions"][0]["formula"] = "true"
    elif change == "overlap":
        d["versions"].append(deepcopy(v))
    elif change == "date_default":
        d["default"] = "2023-01-01"
    elif change == "date_proof":
        d["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "2023"
    elif change == "header":
        d["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = "2025"
    elif change == "cost_value":
        cap["versions"][0]["formula"] = "61001"
    elif change == "cost_unit":
        cap["unit"] = "USD"
    elif change == "cost_proof":
        cap["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "61000"
    elif change == "selector_operator":
        selector["versions"][0]["formula"] = selector["versions"][0]["formula"].replace(
            "cost > ", "cost >= "
        )
    elif change == "selector_cap":
        selector["versions"][0]["formula"] = (
            selector["versions"][0]["formula"]
            .replace("cost_30000", "cost_34000")
            .replace("cost_61000", "cost_59000")
        )
    elif change == "selector_bound":
        selector["versions"][0]["formula"] = selector["versions"][0]["formula"].replace(
            "start_2023", "start_2022"
        )
    elif change == "selector_proof":
        selector["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "vehicles"
    elif change == "propagation":
        principal["versions"][0]["formula"] = principal["versions"][0][
            "formula"
        ].replace("== true", "== false")
    elif change == "timber":
        principal["versions"][0]["formula"] = principal["versions"][0][
            "formula"
        ].replace("if t936_asset_is_timber_resource:", "if false:")
    elif change == "principal_unit":
        principal["unit"] = "USD"
    elif change == "input_date":
        inputs["t936_asset_acquisition_date"]["dtype"] = "String"
    elif change == "input_meaning":
        inputs["t936_asset_acquisition_date"]["description"] = "Disposition date"
    elif change == "input_default":
        inputs["t936_asset_acquisition_cost"]["default"] = 0
    elif change == "duplicate":
        p["inputs"].append(deepcopy(inputs["t936_asset_acquisition_cost"]))
    elif change == "wrong_source":
        body = body.replace("prior to 2022", "prior to 2023")
    elif change == "wrong_citation":
        cite += "-other"
    assert (
        vehicle_date_role_binding(p, body, cite, {"period": YEAR}, (start, start + 4))
        is None
    )


@pytest.mark.parametrize(
    "period",
    [
        "2025-01-01",
        "2025-06-01",
        {"period_kind": "month", "start": "2025-01-01", "end": "2025-01-31"},
        {"period_kind": "tax_year", "start": "2025-01-01", "end": "2026-12-31"},
    ],
)
def test_full_annual_frame_required(period):
    assert (
        vehicle_date_role_binding(DATA, BODY, CITE, {"period": period}, (3630, 3634))
        is None
    )


@pytest.mark.parametrize(
    "name,field,value",
    [
        ("t936_asset_acquisition_date", "unit", "CAD"),
        ("t936_asset_is_timber_resource", "unit", "CAD"),
        ("t936_ordinary_vehicle_cost_excluded", "unit", "CAD"),
        ("t936_asset_acquisition_cost", "indexed_by", "t936_source_year"),
        ("t936_asset_acquisition_cost", "formula", "0"),
    ],
)
def test_closed_input_and_nonmoney_unit(name, field, value):
    p = deepcopy(DATA)
    next(r for r in p["inputs"] + p["rules"] if r["name"] == name)[field] = value
    assert (
        vehicle_date_role_binding(p, BODY, CITE, {"period": YEAR}, (3630, 3634)) is None
    )


@pytest.mark.parametrize("start", [r[0] for r in EXPECTED])
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "expired",
        "source",
        "span",
        "missing_controls",
        "assertion",
        "root",
        "expires_during_execution",
        "complex_value",
        "integer_value",
        "source_value",
        "raw",
        "temporal_flag",
        "rate_flag",
        "structural_flag",
        "word_flag",
        "alternative",
        "bool_offset",
        "gapped_origin",
        "shifted_origin",
        "float_origin",
        "float_raw_span",
    ],
)
def test_private_route_seam_preserves_root_and_liveness(monkeypatch, start, mutation):
    # Deliberately injected registry seam with saved observations. These tests
    # do not exercise the authenticated factory or confer source authority.
    from types import SimpleNamespace

    from axiom_encode.harness import local_relation_source_evidence as adapter
    from axiom_encode.harness.actual_local_relation_evidence import _replay_values

    data = json.loads(
        (Path(__file__).parent / "fixtures/t936_vehicle_dates.json").read_text()
    )
    raw = [r["case"] for r in data["diagnostic_records"]]
    policy, cases = adapter._normalized_source_inputs(DATA, raw)
    observations = []
    for r, c in zip(data["diagnostic_records"], cases, strict=True):
        request, response = r["request"], r["response"]
        values = _replay_values(DATA, request, response)
        if mutation == "root":
            request["queries"][0]["entity_id"] = "foreign"
        observations.append(
            (frozenset({adapter._source_key(c)}), request, response, values)
        )
    if mutation == "missing_controls":
        raw = raw[::3]
        cases = cases[::3]
        observations = observations[::3]
    if mutation == "assertion":
        for c in cases:
            c["output"] = {}
    live = mutation != "expired"
    original_execution = adapter.corroborated_execution
    if mutation == "expires_during_execution":

        def expires(*args):
            nonlocal live
            result = original_execution(*args)
            live = False
            return result

        monkeypatch.setattr(adapter, "corroborated_execution", expires)
    bound = SimpleNamespace(
        policy=policy,
        cases=raw,
        content="immutable",
        sources={CITE: BODY},
        observations=observations,
        fresh=lambda: live,
    )
    context = object()
    monkeypatch.setitem(adapter._contexts, context, bound)
    from dataclasses import replace

    occurrence = occurrence_at(start)
    recall_map = identity_map()
    changed = {
        "complex_value": {"value": complex(occurrence.value)},
        "integer_value": {"value": int(occurrence.value)},
        "source_value": {"source_value": None},
        "raw": {"raw": "2025"},
        "temporal_flag": {"has_temporal_context": True},
        "rate_flag": {"has_rate_context": True},
        "structural_flag": {"has_structural_context": True},
        "word_flag": {"is_word_number": True},
        "alternative": {"alternative_values": (2022.0,)},
        "bool_offset": {"start": True},
    }.get(mutation, {})
    occurrence = replace(occurrence, **changed)
    if mutation == "gapped_origin":
        recall_map.offsets[start + 1] = None
    if mutation == "shifted_origin":
        recall_map.offsets[start : start + 4] = list(range(start + 1, start + 5))
    if mutation == "float_origin":
        recall_map.offsets[start : start + 4] = [
            float(i) for i in range(start, start + 4)
        ]
    raw_span = (start, start + 4) if mutation != "span" else (5792, 5796)
    if mutation == "float_raw_span":
        raw_span = tuple(float(i) for i in raw_span)
    result = adapter.vehicle_year_occurrence(
        context,
        content="immutable",
        cases=cases,
        source_text=BODY if mutation != "source" else BODY + "other",
        citation=CITE,
        raw_span=raw_span,
        occurrence=occurrence,
        recall_text=BODY,
        recall_map=recall_map,
    )
    assert result is (mutation == "none")
    assert not adapter._witnesses


@pytest.mark.parametrize("context", [None, object(), {}, []])
def test_unregistered_vehicle_context_never_credits(context):
    from axiom_encode.harness.local_relation_source_evidence import (
        vehicle_year_occurrence,
    )

    assert not vehicle_year_occurrence(
        context,
        content="candidate",
        cases=[],
        source_text=BODY,
        citation=CITE,
        raw_span=(3630, 3634),
        occurrence=occurrence_at(3630),
        recall_text=BODY,
        recall_map=identity_map(),
    )


def occurrence_at(start):
    from axiom_encode.harness.validator_pipeline import NumericOccurrence

    return NumericOccurrence(
        float(BODY[start : start + 4]),
        start,
        start + 4,
        BODY[start : start + 4],
        source_value=float(BODY[start : start + 4]),
    )


def identity_map():
    from axiom_encode.harness.source_completeness import _NumericRecallSourceMap

    return _NumericRecallSourceMap(list(range(len(BODY))))

"""Static reference-frame and injected private seam controls; not admission."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness import local_relation_source_evidence as adapter
from axiom_encode.harness.actual_local_relation_evidence import _replay_values
from axiom_encode.harness.local_prior_form_evidence import prior_form_reference_binding
from tests.test_local_vehicle_date_evidence import (
    BODY,
    CITE,
    DATA,
    YEAR,
    identity_map,
    occurrence_at,
)

STARTS = [5792, 5836, 6189, 6234]


@pytest.mark.parametrize("start", STARTS)
def test_exact_reference_and_full_historical_roster(start):
    role = prior_form_reference_binding(
        DATA, BODY, CITE, {"period": YEAR}, (start, start + 4)
    )
    assert role is not None and role.reference_year == 2024
    assert len(role.annual_inputs) == len(set(role.annual_inputs)) == 37
    assert role.annual_inputs[0].startswith("t936_completed_1988_")
    assert role.annual_inputs[-1].startswith("t936_completed_2024_")
    assert role.copy_input.endswith("_16" if start < 6000 else "_19")


@pytest.mark.parametrize("start", [5792, 6189])
@pytest.mark.parametrize(
    "change",
    [
        "period",
        "sourceyear",
        "wrongspan",
        "line",
        "invert",
        "drop",
        "duplicate",
        "crosspart",
        "historycopy",
        "note",
        "row",
        "header",
        "earlycontract",
        "historyyear",
        "copycontract",
        "default",
        "unit",
        "booleanunit",
        "computedinput",
        "inputindex",
        "rounding",
        "version",
        "duplicateinput",
    ],
)
def test_static_conflicts_refuse(start, change):
    p = deepcopy(DATA)
    text = BODY
    case = {"period": YEAR}
    part = "a" if start < 6000 else "b"
    line = 16 if part == "a" else 19
    r = next(r for r in p["rules"] if r["name"] == f"t936_previous_{part}")
    v = r["versions"][0]
    inputs = {i["name"]: i for i in p["inputs"]}
    term = f"t936_completed_2006_part_{part}"
    copy = f"t936_2024_part3_line_{line}"
    if change == "period":
        case = {"period": "2025-06-01"}
    elif change == "sourceyear":
        text = text[:start] + "2023" + text[start + 4 :]
    elif change == "wrongspan":
        start = 3747
    elif change == "line":
        v["formula"] = v["formula"].replace(copy, copy[:-2] + "21")
    elif change == "invert":
        v["formula"] = v["formula"].replace(
            "if t936_completed_2024:", "if not t936_completed_2024:"
        )
    elif change == "drop":
        v["formula"] = v["formula"].replace(" + " + term, "")
    elif change == "duplicate":
        v["formula"] = v["formula"].replace(term, f"t936_completed_2005_part_{part}")
    elif change == "crosspart":
        v["formula"] = v["formula"].replace(
            term, f"t936_completed_2006_part_{'b' if part == 'a' else 'a'}"
        )
    elif change == "historycopy":
        v["formula"] = f"if t936_completed_2024:\n  {copy}\nelse:\n  {copy}"
    elif change in ["note", "row", "header"]:
        r["metadata"]["proof"]["atoms"][{"row": 0, "note": 1, "header": 2}[change]][
            "source"
        ]["excerpt"] = "2024"
    elif change == "earlycontract":
        inputs[f"t936_completed_1988_part_{part}"]["description"] = "Completed total"
    elif change == "historyyear":
        inputs[term]["description"] = inputs[term]["description"].replace(
            "2006", "2007"
        )
    elif change == "copycontract":
        inputs[copy]["description"] = "Any prior balance"
    elif change == "default":
        inputs[term]["default"] = 0
    elif change == "unit":
        inputs[term]["unit"] = "USD"
    elif change == "booleanunit":
        inputs["t936_completed_2024"]["unit"] = "CAD"
    elif change == "computedinput":
        inputs[term]["formula"] = "0"
    elif change == "inputindex":
        inputs[term]["indexed_by"] = "year"
    elif change == "rounding":
        r["rounding"] = "half_up"
    elif change == "version":
        v["effective_to"] = "2026-12-31"
    elif change == "duplicateinput":
        p["inputs"].append(deepcopy(inputs[term]))
    assert prior_form_reference_binding(p, text, CITE, case, (start, start + 4)) is None


@pytest.mark.parametrize("start", STARTS)
@pytest.mark.parametrize(
    "mutation",
    [
        "none",
        "onlycopy",
        "onlyhistory",
        "nointerior",
        "wrongroot",
        "assertion",
        "expired",
        "expires",
        "float_origin",
    ],
)
def test_private_both_routes_and_lifetime(monkeypatch, start, mutation):
    records = json.loads(
        (Path(__file__).parent / "fixtures/t936_prior_form_cases.json").read_text()
    )["records"]
    if mutation == "onlycopy":
        records = records[1:2]
    elif mutation == "onlyhistory":
        records = [records[0], records[2]]
    elif mutation == "nointerior":
        records = records[:2]
    raw = [r["case"] for r in records]
    policy, cases = adapter._normalized_source_inputs(DATA, raw)
    obs = []
    for r, c in zip(records, cases, strict=True):
        request, response = r["request"], r["response"]
        values = _replay_values(DATA, request, response)
        if mutation == "wrongroot":
            request["queries"][0]["entity_id"] = "different"
        obs.append((frozenset({adapter._source_key(c)}), request, response, values))
    if mutation == "assertion":
        for c in cases:
            c["output"] = {}
    live = mutation != "expired"
    original = adapter.corroborated_execution
    if mutation == "expires":

        def expire(*args):
            nonlocal live
            r = original(*args)
            live = False
            return r

        monkeypatch.setattr(adapter, "corroborated_execution", expire)
    bound = SimpleNamespace(
        policy=policy,
        cases=raw,
        content="immutable",
        sources={CITE: BODY},
        observations=obs,
        fresh=lambda: live,
    )
    context = object()
    monkeypatch.setitem(adapter._contexts, context, bound)
    mapping = identity_map()
    if mutation == "float_origin":
        mapping.offsets[start : start + 4] = [float(i) for i in range(start, start + 4)]
    result = adapter.prior_form_year_occurrence(
        context,
        content="immutable",
        cases=cases,
        source_text=BODY,
        citation=CITE,
        raw_span=(start, start + 4),
        occurrence=occurrence_at(start),
        recall_text=BODY,
        recall_map=mapping,
    )
    assert result is (mutation == "none")
    assert not adapter._witnesses

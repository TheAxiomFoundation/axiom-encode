"""Exact diagnostic candidate fixture: source binding, not runtime authority."""

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.actual_floor_evidence import bind_same_page_floors

P = Path(__file__).parent / "fixtures"
BODY = (P / "detached_schedule2_source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"
CONTENT = (P / "same_page_floor_candidate.yaml").read_text()
PAYLOAD = yaml.safe_load(CONTENT)


def bind(payload, body=BODY):
    return bind_same_page_floors(yaml.safe_dump(payload), {CITATION: body})


def test_actual_candidate_immediate_floor_roles_are_unique():
    result = bind(PAYLOAD)
    assert set(result) == {6, 7, 11, 12, 13}
    assert result[12]["name"].endswith("_line_12")
    assert result[13]["name"] == "federal_amounts_transferred_from_spouse_line_32600"


@pytest.mark.parametrize("row", [6, 7, 11, 12, 13])
@pytest.mark.parametrize("mutation", ["duplicate", "unit", "version", "proof", "dates"])
def test_participating_role_mutation_refuses(row, mutation):
    a = deepcopy(PAYLOAD)
    name = bind(PAYLOAD)[row]["name"]
    r = next(r for r in a["rules"] if r["name"] == name)
    if mutation == "duplicate":
        copy = deepcopy(r)
        copy["name"] += "_shadow"
        a["rules"].append(copy)
    elif mutation == "unit":
        r["unit"] = "USD"
    elif mutation == "version":
        r["versions"].append(deepcopy(r["versions"][0]))
    elif mutation == "proof":
        r["metadata"]["proof"]["atoms"] = []
    else:
        r["versions"][0]["effective_to"] = "2026-12-31"
    assert bind(a) == {}


@pytest.mark.parametrize("row", [12, 13])
def test_reversed_floor_order_is_not_sorted_into_validity(row):
    a = deepcopy(PAYLOAD)
    bound = bind(PAYLOAD)
    r = next(r for r in a["rules"] if r["name"] == bound[row]["name"])
    left, right = (7, 11) if row == 12 else (6, 12)
    f = r["versions"][0]["formula"]
    r["versions"][0]["formula"] = f.replace(
        bound[left]["name"] + " - " + bound[right]["name"],
        bound[right]["name"] + " - " + bound[left]["name"],
    )
    assert bind(a) == {}


def test_changed_source_even_with_updated_hash_is_not_recognized():
    a = deepcopy(PAYLOAD)
    body = BODY.replace("90 days", "89 days")
    a["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        body.encode()
    ).hexdigest()
    assert bind(a, body) == {}


@pytest.mark.parametrize("versions", [[], None, [None], "bad"])
def test_malformed_local_child_refuses_without_exception(versions):
    a = deepcopy(PAYLOAD)
    child = next(r for r in a["rules"] if r["name"] == "schedule_2_spouse_base_line_8")
    child["versions"] = versions
    assert bind(a) == {}


def test_whole_source_cannot_replace_exact_aggregate_proof():
    a = deepcopy(PAYLOAD)
    name = bind(PAYLOAD)[6]["name"]
    r = next(r for r in a["rules"] if r["name"] == name)
    atom = next(x for x in r["metadata"]["proof"]["atoms"] if x["kind"] == "formula")
    atom["source"]["excerpt"] = BODY
    assert bind(a) == {}


def test_unknown_gate_and_unowned_gate_refuse():
    a = deepcopy(PAYLOAD)
    name = bind(PAYLOAD)[13]["name"]
    r = next(r for r in a["rules"] if r["name"] == name)
    r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
        "source_relationship_allows_transfer", "unrelated_gate"
    )
    assert bind(a) == {}
    a = deepcopy(PAYLOAD)
    gate = next(
        r for r in a["rules"] if r["name"] == "source_relationship_allows_transfer"
    )
    gate["metadata"]["proof"]["atoms"] = []
    assert bind(a) == {}


def test_child_currency_and_shadow_version_refuse():
    for mutation in ("unit", "version"):
        a = deepcopy(PAYLOAD)
        child = next(
            r for r in a["rules"] if r["name"] == "schedule_2_spouse_base_line_8"
        )
        if mutation == "unit":
            child["unit"] = "USD"
        else:
            child["versions"].append(deepcopy(child["versions"][0]))
        assert bind(a) == {}


@pytest.mark.parametrize("mutation", ["gate_unit", "gate_frame", "child_bool"])
def test_strict_selector_and_copy_contract(mutation):
    a = deepcopy(PAYLOAD)
    gate = next(
        r for r in a["rules"] if r["name"] == "source_relationship_allows_transfer"
    )
    if mutation == "gate_unit":
        gate["unit"] = "CAD"
    elif mutation == "gate_frame":
        gate["metadata"]["proof"]["atoms"] = [
            x
            for x in gate["metadata"]["proof"]["atoms"]
            if x["kind"] != "effective_period"
        ]
    else:
        child = next(
            r for r in a["rules"] if r["name"] == "schedule_2_spouse_base_line_8"
        )
        child["versions"][0]["formula"] = True
    assert bind(a) == {}

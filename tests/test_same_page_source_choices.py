"""Source choice binding is static ownership, never execution authority."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.actual_floor_evidence import bind_same_page_choices

F = Path(__file__).parent / "fixtures"
BODY = (F / "detached_schedule2_source.txt").read_text()
PAYLOAD = yaml.safe_load((F / "same_page_floor_candidate.yaml").read_text())
CITATION = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"


def bind(payload=PAYLOAD):
    return bind_same_page_choices(yaml.safe_dump(payload), {CITATION: BODY})


def test_closed_source_choices_bind_without_issuing_witnesses():
    found = bind()
    assert set(found) == {1, 7}
    assert found[1]["name"] == "schedule_2_age_amount_line_1"


@pytest.mark.parametrize("row", [1, 7])
@pytest.mark.parametrize(
    "mutation",
    ["operator", "copy", "rounding", "proof", "date", "extra_version", "duplicate"],
)
def test_changed_source_choice_refuses(row, mutation):
    p = deepcopy(PAYLOAD)
    name = bind()[row]["name"]
    r = next(r for r in p["rules"] if r["name"] == name)
    v = r["versions"][0]
    if mutation == "operator":
        v["formula"] = v["formula"].replace("<=", "<")
    elif mutation == "copy":
        v["formula"] = "spouse_age_amount_line_30100"
    elif mutation == "rounding":
        r["rounding"] = "floor"
    elif mutation == "proof":
        r["metadata"]["proof"]["atoms"] = []
    elif mutation == "date":
        v["effective_to"] = "2026-12-31"
    elif mutation == "extra_version":
        r["versions"].append(deepcopy(v))
    else:
        other = deepcopy(r)
        other["name"] += "_duplicate"
        p["rules"].append(other)
    assert not bind(p)


@pytest.mark.parametrize(
    "name",
    [
        "federal_2025_schedule_2_taxable_income_threshold",
        "federal_2025_schedule_2_latest_birth_year_for_age_amount",
        "federal_2025_schedule_2_lowest_rate",
    ],
)
def test_wrong_parameter_proof_or_value_refuses(name):
    for mutation in ("value", "proof", "type"):
        p = deepcopy(PAYLOAD)
        r = next(r for r in p["rules"] if r["name"] == name)
        if mutation == "value":
            r["versions"][0]["formula"] = "0"
        elif mutation == "proof":
            r["metadata"]["proof"]["atoms"] = []
        else:
            r["dtype"] = "Judgment"
        assert not bind(p)


def test_detached_age_projection_cannot_prove_propagation():
    p = deepcopy(PAYLOAD)
    age_name = bind()[1]["name"]
    age = next(r for r in p["rules"] if r["name"] == age_name)
    unrelated = deepcopy(age)
    unrelated["name"] = "unrelated_amount"
    unrelated["versions"][0]["formula"] = "0"
    p["rules"].append(unrelated)
    for r in p["rules"]:
        if r.get("kind") == "derived" and age_name in r["versions"][0]["formula"]:
            r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
                age_name, "unrelated_amount"
            )
    assert not bind(p)


@pytest.mark.parametrize(
    "name",
    [
        "schedule_2_age_amount_line_1",
        "federal_2025_schedule_2_latest_birth_year_for_age_amount",
    ],
)
def test_missing_choice_formula_refuses(name):
    p = deepcopy(PAYLOAD)
    next(r for r in p["rules"] if r["name"] == name)["versions"][0].pop("formula")
    assert not bind(p)

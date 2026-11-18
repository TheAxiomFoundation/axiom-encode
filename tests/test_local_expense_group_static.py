"""Whole expense static contract; positive fixtures confer no runtime authority."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from axiom_encode.harness.local_pair_source_evidence import (
    _EXPENSE_CATEGORIES,
    bind_expense_group,
)
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA


def bind(data):
    return bind_expense_group(
        data, SimpleNamespace(start=2877, end=3984, text=BODY[2877:3984]), BODY, CITE
    )


def test_complete_static_contract():
    assert bind(deepcopy(DATA)) == "other_investment_expenses_line_68080"


@pytest.mark.parametrize("name", list(_EXPENSE_CATEGORIES))
@pytest.mark.parametrize(
    "mutation", ["missing", "exclusion", "unit", "default", "duplicate"]
)
def test_category_closure(name, mutation):
    d = deepcopy(DATA)
    r = next(x for x in d["inputs"] if x["name"] == name)
    if mutation == "missing":
        d["inputs"].remove(r)
    elif mutation == "exclusion":
        r["description"] = r["description"].split(" Excludes")[0]
    elif mutation == "duplicate":
        d["inputs"].append(deepcopy(r))
    else:
        r[mutation] = "invalid"
    assert bind(d) is None


@pytest.mark.parametrize(
    "name",
    [
        "t936_source_year",
        "t936_start_2022",
        "t936_march_18_2019",
        "t936_vehicle_cost_38000",
        "t936_vehicle_cost_55000",
        "t936_ordinary_vehicle_cost_excluded",
        "t936_zepv_cost_excluded",
        "t936_asset_included_uncollectible_portion",
        "t936_uncollectible_dispositions_total",
        "other_investment_expenses_line_68080",
    ],
)
@pytest.mark.parametrize(
    "mutation",
    ["proofkind", "proofpath", "proofbody", "default", "rounding", "version", "entity"],
)
def test_envelope_closure(name, mutation):
    d = deepcopy(DATA)
    r = next(x for x in d["rules"] if x["name"] == name)
    if mutation.startswith("proof"):
        a = r["metadata"]["proof"]["atoms"][0]
        if mutation == "proofkind":
            a["kind"] = "irrelevant"
        elif mutation == "proofpath":
            a["path"] = "description"
        else:
            a["source"]["excerpt"] = BODY
    elif mutation == "version":
        r["versions"].append(deepcopy(r["versions"][0]))
    elif mutation == "rounding":
        r["resolved_currency_rounding"] = "none"
    else:
        r[mutation] = "invalid"
    assert bind(d) is None


@pytest.mark.parametrize("mutation", ["reverse", "boolarity", "formula", "proofkind"])
def test_relation_contract(mutation):
    d = deepcopy(DATA)
    r = next(
        x for x in d["rules"] if x["name"] == "t936_depreciable_disposition_for_return"
    )
    if mutation == "reverse":
        r["data_relation"]["arguments"].reverse()
    elif mutation == "boolarity":
        r["data_relation"]["arity"] = True
    elif mutation == "proofkind":
        r["metadata"]["proof"]["atoms"][0]["kind"] = "formula"
    else:
        r["formula"] = "0"
    assert bind(d) is None


@pytest.mark.parametrize(
    "name",
    ["t936_uncollectible_dispositions_total", "other_investment_expenses_line_68080"],
)
def test_literal_replacement(name):
    d = deepcopy(DATA)
    next(x for x in d["rules"] if x["name"] == name)["versions"][0]["formula"] = "70"
    assert bind(d) is None

"""Static contracts only; no test fixture grants runtime/source admission."""

import ast
from copy import deepcopy
from types import SimpleNamespace

import pytest

from axiom_encode.harness.local_pair_source_evidence import bind_remaining_chart_role
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA

SPANS = [(7150, 7346), (8017, 8067), (8068, 8159), (8160, 8308)]


def bind(data, span=(8017, 8067)):
    return bind_remaining_chart_role(
        data,
        SimpleNamespace(start=span[0], end=span[1], text=BODY[span[0] : span[1]]),
        BODY,
        CITE,
    )


@pytest.mark.parametrize("span", SPANS)
def test_exact_closed_contract(span):
    assert bind(deepcopy(DATA), span) is not None


@pytest.mark.parametrize(
    "name",
    [
        "federal_2025_t936_chart_a_line_3",
        "federal_2025_t936_chart_a_line_13",
        "federal_2025_t936_chart_a_line_17",
        "federal_2025_t936_chart_a_line_7",
        "t936_chart_a_line_8",
        "federal_2025_t936_chart_a_line_19",
        "federal_2025_t936_chart_a_additional_investment_income_line_20",
    ],
)
@pytest.mark.parametrize(
    "mutation", ["zero", "bytes", "ast", "proof", "default", "period"]
)
def test_closed_formula_mutants(name, mutation):
    data = deepcopy(DATA)
    r = next(x for x in data["rules"] if x["name"] == name)
    v = r["versions"][0]
    if mutation == "zero":
        v["formula"] = "0"
    elif mutation == "bytes":
        v["formula"] = v["formula"].encode()
    elif mutation == "ast":
        v["formula"] = ast.parse(v["formula"])
    elif mutation == "proof":
        r["metadata"]["proof"]["atoms"] = []
    elif mutation == "default":
        r["default"] = 0
    else:
        v["effective_to"] = "2026-12-31"
    assert bind(data) is None


@pytest.mark.parametrize(
    "name",
    [
        "schedule_3_line_19900",
        "t2048_line_17215",
        "t2048_line_17216",
        "t3_box_21_capital_gains",
        "t3_box_30_capital_gains_eligible_for_deduction",
        "t936_chart_a_line_14_includes_t3",
    ],
)
@pytest.mark.parametrize("mutation", ["description", "dtype", "default", "duplicate"])
def test_completed_input_contract(name, mutation):
    data = deepcopy(DATA)
    r = next(x for x in data["inputs"] if x["name"] == name)
    if mutation == "duplicate":
        data["inputs"].append(deepcopy(r))
    elif mutation == "default":
        r["default"] = 0
    else:
        r[mutation] = "wrong"
    assert bind(data) is None

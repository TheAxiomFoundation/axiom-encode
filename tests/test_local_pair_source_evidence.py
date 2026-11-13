"""Static role binding rejects source, operand and type substitutions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from axiom_encode.harness.local_pair_source_evidence import _PROFILES, bind_pair_role
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA


@pytest.mark.parametrize("role", _PROFILES)
def test_complete_printed_owner(role):
    p = _PROFILES[role]
    branch = SimpleNamespace(start=p[0], end=p[1], text=p[5])
    result = bind_pair_role(DATA, branch, BODY, CITE, p[2])
    assert result is not None and result.role == role
    assert result.operands == p[4]


@pytest.mark.parametrize("role", _PROFILES)
@pytest.mark.parametrize(
    "change",
    [
        "citation",
        "branch",
        "source",
        "proof",
        "proofkind",
        "period",
        "unit",
        "duplicate",
        "shadowversion",
        "operand",
        "default",
        "rounding",
    ],
)
def test_conflicts_refuse(role, change):
    policy = deepcopy(DATA)
    p = _PROFILES[role]
    branch = SimpleNamespace(start=p[0], end=p[1], text=p[5])
    text, cite = BODY, CITE
    r = next(r for r in policy["rules"] if r["name"] == p[2])
    if change == "citation":
        cite += "/foreign"
    elif change == "branch":
        branch.start += 1
    elif change == "source":
        text = text[: p[0]] + "!" + text[p[0] + 1 :]
    elif change == "proof":
        r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = BODY
    elif change == "proofkind":
        r["metadata"]["proof"]["atoms"][0]["kind"] = "parameter"
    elif change == "period":
        r["versions"][0]["effective_from"] = "2024-01-01"
    elif change == "unit":
        r["unit"] = "USD"
    elif change == "duplicate":
        policy["rules"].append(deepcopy(r))
    elif change == "shadowversion":
        r["versions"].append(deepcopy(r["versions"][0]))
    elif change == "operand":
        r["versions"][0]["formula"] = "0"
    elif change == "default":
        r["default"] = 0
    elif change == "rounding":
        r["rounding"] = "floor"
    assert bind_pair_role(policy, branch, text, cite, p[2]) is None


@pytest.mark.parametrize("role", ["cnil_floor", "chart11_floor", "chart20_floor"])
def test_reversed_order_refused(role):
    policy = deepcopy(DATA)
    p = _PROFILES[role]
    r = next(r for r in policy["rules"] if r["name"] == p[2])
    a, b = p[4]
    r["versions"][0]["formula"] = (
        r["versions"][0]["formula"]
        .replace(a, "REPLACEMENT")
        .replace(b, a)
        .replace("REPLACEMENT", b)
    )
    assert (
        bind_pair_role(
            policy, SimpleNamespace(start=p[0], end=p[1], text=p[5]), BODY, CITE, p[2]
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "rateunit",
        "rateentity",
        "rateperiod",
        "ratevalue",
        "rateversion",
        "inputrate",
        "resolvedrounding",
    ],
)
@pytest.mark.parametrize(
    "role", ["chart11_floor", "chart20_floor", "chart_completed_expense"]
)
def test_reviewed_dependency_boundary_mutations(role, change):
    policy = deepcopy(DATA)
    by_name = {r["name"]: r for r in [*policy["rules"], *policy["inputs"]]}
    rate = by_name["federal_2025_t936_chart_a_inclusion_rate"]
    if change == "rateunit":
        rate["unit"] = "USD"
    elif change == "rateentity":
        rate["entity"] = "Alien"
    elif change == "rateperiod":
        rate["period"] = "Day"
    elif change == "ratevalue":
        rate["versions"][0]["formula"] = "0.25"
    elif change == "rateversion":
        rate["versions"].append(deepcopy(rate["versions"][0]))
    elif change == "inputrate":
        if role != "chart_completed_expense":
            return
        by_name["t1_line_25300_net_capital_losses_of_other_years"]["dtype"] = "Rate"
    else:
        target = (
            "federal_2025_t936_chart_a_line_9"
            if role == "chart11_floor"
            else "federal_2025_t936_chart_a_line_19"
        )
        by_name[target]["resolved_currency_rounding"] = "floor"
    p = _PROFILES[role]
    assert (
        bind_pair_role(
            policy, SimpleNamespace(start=p[0], end=p[1], text=p[5]), BODY, CITE, p[2]
        )
        is None
    )

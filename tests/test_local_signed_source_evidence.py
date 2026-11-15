"""Closed signed-row bindings retain the reporting condition and ordered sum."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from axiom_encode.harness.local_pair_source_evidence import (
    _SIGNED_PROFILES,
    bind_signed_role,
)
from tests.test_local_vehicle_date_evidence import BODY, CITE, DATA


@pytest.mark.parametrize("name", _SIGNED_PROFILES)
def test_exact_signed_owner(name):
    span, _, _ = _SIGNED_PROFILES[name]
    branch = SimpleNamespace(start=span[0], end=span[1], text=BODY[span[0] : span[1]])
    role = bind_signed_role(DATA, branch, BODY, CITE, name)
    assert role is not None
    assert (role.selector is not None) == (name == "t936_chart_a_line_8")
    if name.endswith("line_7"):
        assert role.operands == (
            "schedule_3_line_10700",
            "schedule_3_line_11000",
            "schedule_3_line_12400",
        )


@pytest.mark.parametrize("name", _SIGNED_PROFILES)
@pytest.mark.parametrize(
    "change",
    [
        "clamp",
        "abs",
        "reverse",
        "partialproof",
        "proofkind",
        "foreign",
        "year",
        "shadow",
        "default",
        "rounding",
        "unit",
        "inputtype",
        "inputcontract",
        "childformula",
        "span",
        "source",
    ],
)
def test_signed_conflicts_refuse(name, change):
    p = deepcopy(DATA)
    text = BODY
    cite = CITE
    span, _, _ = _SIGNED_PROFILES[name]
    branch = SimpleNamespace(start=span[0], end=span[1], text=BODY[span[0] : span[1]])
    r = next(r for r in p["rules"] if r["name"] == name)
    v = r["versions"][0]
    if change == "clamp":
        v["formula"] = "max(0, 1)"
    elif change == "abs":
        v["formula"] = "abs(1)"
    elif change == "reverse":
        if name.endswith("line_7"):
            v["formula"] = (
                "schedule_3_line_12400 + schedule_3_line_11000 + schedule_3_line_10700"
            )
        elif name.endswith("line_9"):
            v["formula"] = "t936_chart_a_line_8 + federal_2025_t936_chart_a_line_7"
        else:
            v["formula"] = v["formula"].replace(
                "if schedule_3_line_19200_reported:",
                "if not schedule_3_line_19200_reported:",
            )
    elif change == "partialproof":
        r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "(if negative, show in brackets)"
        )
    elif change == "proofkind":
        r["metadata"]["proof"]["atoms"][0]["kind"] = "parameter"
    elif change == "foreign":
        cite += "/other"
    elif change == "year":
        v["effective_from"] = "2024-01-01"
    elif change == "shadow":
        r["versions"].append(deepcopy(v))
    elif change == "default":
        r["default"] = 0
    elif change == "rounding":
        r["resolved_currency_rounding"] = "floor"
    elif change == "unit":
        r["unit"] = "USD"
    elif change in ("inputtype", "inputcontract"):
        key = (
            "t2017_line_13_reserves_qffp_qsbcs"
            if name == "t936_chart_a_line_8"
            else "schedule_3_line_10700"
        )
        d = next(d for d in p["inputs"] if d["name"] == key)
        d["dtype" if change == "inputtype" else "description"] = (
            "Rate" if change == "inputtype" else "Any amount"
        )
    elif change == "childformula":
        if name.endswith("line_9"):
            next(d for d in p["rules"] if d["name"] == "t936_chart_a_line_8")[
                "versions"
            ][0]["formula"] = "0"
        else:
            v["formula"] = "0"
    elif change == "span":
        branch.start = float(branch.start)
    elif change == "source":
        text = text[: span[0]] + "!" + text[span[0] + 1 :]
    assert bind_signed_role(p, branch, text, cite, name) is None

"""Mutation and numeric identity controls, not execution or source coverage."""

# ruff: noqa: F811 -- imported pytest fixture is injected by parameter name

import copy
from decimal import Decimal

import pytest
import yaml

from axiom_encode.harness.mhrtc_case_identity import frozen_mhrtc_case
from tests.test_mhrtc_request import CREDIT, PREFIX, fixture  # noqa: F401


def test_exact_decimal_survives_ordinary_float_identity(fixture):
    case, *_ = fixture
    raw = (
        yaml.safe_dump([case])
        .replace("0.5", "0.50000000000000000000000000001")
        .encode()
    )
    caller = yaml.safe_load(raw)[0]
    result = frozen_mhrtc_case(raw, 1, caller)
    assert result.case["tables"]["Claim"][0][
        PREFIX + "input.dwelling_land_area_hectares"
    ] == Decimal("0.50000000000000000000000000001")
    assert (
        type(
            result.case["tables"]["Claim"][0][
                PREFIX + "input.dwelling_land_area_hectares"
            ]
        )
        is Decimal
    )
    assert result.ordinal == 1


@pytest.mark.parametrize("ordinal", [True, False, 0, -1, 1.0, "1", 2])
def test_invalid_ordinal_rejected(fixture, ordinal):
    case, *_ = fixture
    with pytest.raises(ValueError):
        frozen_mhrtc_case(yaml.safe_dump([case]).encode(), ordinal, case)


@pytest.mark.parametrize(
    "mutation", ["bool", "float", "expected", "name", "extra", "row"]
)
def test_changed_caller_rejected(fixture, mutation):
    case, *_ = fixture
    raw = yaml.safe_dump([case]).encode()
    caller = copy.deepcopy(case)
    if mutation == "bool":
        caller["input"][PREFIX + "input.filing_claimant_key"] = True
    elif mutation == "float":
        caller["input"][PREFIX + "input.filing_claimant_key"] = 1.0
    elif mutation == "expected":
        caller["output"][CREDIT] += 1
    elif mutation == "name":
        caller["name"] += " changed"
    elif mutation == "extra":
        caller["unused"] = 1
    else:
        caller["tables"]["Claim"].append(copy.deepcopy(caller["tables"]["Claim"][0]))
    with pytest.raises(ValueError, match="caller case differs"):
        frozen_mhrtc_case(raw, 1, caller)


def test_mutating_returned_case_does_not_change_frozen_source(fixture):
    case, *_ = fixture
    raw = yaml.safe_dump([case]).encode()
    first = frozen_mhrtc_case(raw, 1, case)
    first.case["output"][CREDIT] = 0
    second = frozen_mhrtc_case(raw, 1, case)
    assert second.case["output"][CREDIT] == 5800
    assert second.companion_sha256 == first.companion_sha256


def test_duplicate_raw_mapping_rejected_before_caller_view(fixture):
    case, *_ = fixture
    text = yaml.safe_dump([case]).replace(
        "name: explicit transport", "name: first\n  name: explicit transport"
    )
    with pytest.raises((ValueError, yaml.YAMLError)):
        frozen_mhrtc_case(text.encode(), 1, yaml.safe_load(text)[0])


def test_negative_zero_and_positive_zero_are_distinct_caller_views(fixture):
    case, *_ = fixture
    key = PREFIX + "input.dwelling_land_area_hectares"
    case["tables"]["Claim"][0][key] = -0.0
    raw = yaml.safe_dump([case]).encode()
    caller = copy.deepcopy(case)
    caller["tables"]["Claim"][0][key] = 0.0
    with pytest.raises(ValueError):
        frozen_mhrtc_case(raw, 1, caller)

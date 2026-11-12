from copy import deepcopy
from datetime import date

import pytest

from axiom_encode.harness import source_completeness as sc


def payload():
    return {
        "rules": [
            {
                "name": "cutoff",
                "kind": "parameter",
                "dtype": "Date",
                "indexed_by": "key",
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "values": {0: "1960-12-31"},
                    }
                ],
            }
        ]
    }


def env(p=None, case=None):
    return sc._formula_environment_for_case(
        sc._constant_rule_environment(p or payload()),
        case if case is not None else {"period": "2025-06"},
    )


@pytest.mark.parametrize(
    "expression,expected",
    [("cutoff[0]", date(1960, 12, 31)), ("birth <= cutoff[0]", True)],
)
def test_typed_cells(expression, expected):
    assert (
        sc._evaluate_rulespec_formula(
            expression, environment={**env(), "birth": date(1960, 12, 31)}
        )
        == expected
    )


@pytest.mark.parametrize(
    "index", ["false", "true", "0.0", '"0"', "1", "-1", "9223372036854775808"]
)
def test_bad_indices(index):
    assert (
        sc._evaluate_rulespec_formula(f"cutoff[{index}]", environment=env())
        is sc._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize(
    "case", [{}, {"period": "2024-12"}, {"period": "2026-01"}, {"period": "invalid"}]
)
def test_period_required_and_bounded(case):
    assert "cutoff" not in env(case=case)


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "Text"},
        {"default": "1960-12-31"},
        {"rounding": None},
        {"indexed_by": ""},
    ],
)
def test_reject_declaration_ambiguity(change):
    p = payload()
    p["rules"][0].update(change)
    assert "cutoff" not in env(p)


@pytest.mark.parametrize(
    "values",
    [{False: "1960-12-31"}, {0: "1960-02-30"}, {0: 123}, {}, {2**63: "1960-12-31"}],
)
def test_invalid_cells(values):
    p = payload()
    p["rules"][0]["versions"][0]["values"] = values
    assert "cutoff" not in env(p)


def test_duplicate_name_and_equal_start_ambiguity():
    p = payload()
    p["rules"].append(deepcopy(p["rules"][0]))
    assert "cutoff" not in env(p)
    p = payload()
    p["rules"][0]["versions"] *= 2
    assert "cutoff" not in env(p)


def test_latest_applicable_version():
    p = payload()
    v = deepcopy(p["rules"][0]["versions"][0])
    v.update(effective_from="2025-07-01", values={0: "1961-12-31"})
    p["rules"][0]["versions"].append(v)
    assert env(p)["cutoff"][0] == date(1960, 12, 31)
    assert env(p, {"period": "2025-08"})["cutoff"][0] == date(1961, 12, 31)


@pytest.mark.parametrize("bad", [None, 2025, [], {}, "2025-02-30"])
def test_invalid_period_metadata_fails_closed(bad):
    p = payload()
    p["rules"][0]["versions"][0]["effective_from"] = bad
    assert "cutoff" not in env(p)


@pytest.mark.parametrize("bad", [None, {}, "input"])
def test_invalid_input_declarations_fail_closed(bad):
    p = payload()
    p["inputs"] = bad
    assert "cutoff" not in env(p)

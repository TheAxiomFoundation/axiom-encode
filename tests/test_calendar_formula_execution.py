"""Calendar execution preserves engine types and fails closed on bad dates."""

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal

import pytest

from axiom_encode.harness import source_completeness as sc


@pytest.mark.parametrize(
    "function,base,offset,expected",
    [
        ("date_add_days", "2024-02-28", 1, "2024-02-29"),
        ("date_add_days", "2025-03-01", -1, "2025-02-28"),
        ("date_add_months", "2025-01-31", 1, "2025-02-28"),
        ("date_add_months", "2024-01-31", 1, "2024-02-29"),
        ("date_add_months", "2025-03-31", -1, "2025-02-28"),
        ("date_add_months", "2025-08-31", 6, "2026-02-28"),
        ("date_add_months", "2025-01-31", 2, "2025-03-31"),
        ("date_add_months", "2025-01-15", -13, "2023-12-15"),
        ("date_add_months", "2025-01-31", 0, "2025-01-31"),
        ("date_add_years", "2024-02-29", 1, "2025-02-28"),
        ("date_add_years", "2024-02-29", 4, "2028-02-29"),
        ("date_add_years", "2000-02-29", 100, "2100-02-28"),
        ("date_add_years", "2000-02-29", -100, "1900-02-28"),
        ("date_add_years", "2025-06-01", -18, "2007-06-01"),
        ("date_add_years", "2024-02-29", 0, "2024-02-29"),
    ],
)
def test_calendar_formula_matches_pinned_engine_calendar_vectors(
    function, base, offset, expected
):
    # Month/year cases are the af6e4ea engine tests/calendar.rs vectors.
    actual = sc._evaluate_rulespec_formula(
        f"{function}(base, offset)",
        environment={"base": date.fromisoformat(base), "offset": offset},
    )
    assert type(actual) is date
    assert actual == date.fromisoformat(expected)


@pytest.mark.parametrize(
    "offset",
    [True, "1", Decimal("1.5"), Decimal("NaN"), float("inf"), 2**63, -(2**63) - 1],
)
@pytest.mark.parametrize(
    "function", ["date_add_days", "date_add_months", "date_add_years"]
)
def test_calendar_rejects_non_integral_or_invalid_engine_offsets(function, offset):
    assert (
        sc._evaluate_calendar_call(function, [date(2025, 1, 1), offset])
        is sc._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize("base", ["2025-01-01", datetime(2025, 1, 1), None, 2025])
def test_calendar_requires_typed_date(base):
    assert (
        sc._evaluate_calendar_call("date_add_years", [base, 1])
        is sc._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize(
    "function,base,offset",
    [
        ("date_add_days", date.min, -1),
        ("date_add_days", date.max, 1),
        ("date_add_months", date.min, -1),
        ("date_add_years", date.max, 1),
        ("date_add_months", date(2025, 1, 1), 2**32),
        ("date_add_years", date(2025, 1, 1), 2**63 - 1),
    ],
)
def test_calendar_overflow_stays_unresolved(function, base, offset):
    assert (
        sc._evaluate_calendar_call(function, [base, offset])
        is sc._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize(
    "arguments", [[], [date(2025, 1, 1)], [date(2025, 1, 1), 1, 2]]
)
def test_calendar_arity(arguments):
    assert (
        sc._evaluate_calendar_call("date_add_days", arguments)
        is sc._UNRESOLVED_CONDITION_VALUE
    )


def test_declared_dates_decode_without_mutating_cases_or_string_fields():
    payload = {
        "inputs": [
            {"name": "birth", "dtype": "Date"},
            {"name": "label", "dtype": "String"},
        ],
        "rules": [{"name": "birthday", "dtype": "Date"}],
    }
    cases = [
        {
            "period": "2025-01",
            "input": {"birth": "2000-01-02", "label": "2000-01-02"},
            "output": {"birthday": "2025-01-02"},
        }
    ]
    original = deepcopy(cases)
    result = sc._typed_date_cases(cases, payload)
    assert cases == original
    assert result[0]["input"] == {"birth": date(2000, 1, 2), "label": "2000-01-02"}
    assert result[0]["output"] == {"birthday": date(2025, 1, 2)}


@pytest.mark.parametrize(
    "value", ["2025-02-29", "2025-1-1", " 2025-01-01", "2025-01-01T00:00:00"]
)
def test_invalid_dates_are_not_invented(value):
    case = {"input": {"birth": value}}
    assert sc._typed_date_cases(
        [case], {"inputs": [{"name": "birth", "dtype": "Date"}]}
    ) == [case]


def test_duplicate_declarations_are_not_typed():
    case = {"input": {"birth": "2000-01-01"}}
    payload = {
        "inputs": [{"name": "birth", "dtype": "Date"}],
        "rules": [{"name": "birth", "dtype": "Date"}],
    }
    assert sc._typed_date_cases([case], payload) == [case]


@pytest.mark.parametrize(
    "period,expected",
    [
        (
            {"period_kind": "tax_year", "start": "2025-01-01", "end": "2025-12-31"},
            date(2025, 1, 1),
        ),
        ("2025-02", date(2025, 2, 1)),
        (date(2025, 2, 3), date(2025, 2, 3)),
        (
            {"period_kind": "benefit_week", "start": "2025-02-03", "end": "2025-02-08"},
            date(2025, 2, 3),
        ),
    ],
)
def test_period_start_is_typed(period, expected):
    assert (
        sc._formula_environment_for_case({}, {"period": period})["period_start"]
        == expected
    )


@pytest.mark.parametrize("period", [None, "", "2025-02-29", "2025-13"])
def test_missing_or_invalid_period_cannot_supply_calendar_start(period):
    assert (
        sc._formula_environment_for_case({}, {"period": period})["period_start"]
        is sc._UNRESOLVED_CONDITION_VALUE
    )


def test_conflicting_period_start_cannot_override_case_period():
    result = sc._formula_environment_for_case(
        {"period_start": date(2024, 1, 1)}, {"period": "2025"}
    )
    assert result["period_start"] is sc._UNRESOLVED_CONDITION_VALUE


def test_calendar_execution_uses_inputs_not_expected_assertion():
    rule = {
        "name": "birthday",
        "dtype": "Date",
        "versions": [
            {"effective_from": "2025-01-01", "formula": "date_add_years(birth, 25)"}
        ],
    }
    payload = {"inputs": [{"name": "birth", "dtype": "Date"}], "rules": [rule]}
    case = {
        "period": "2025-06",
        "input": {"birth": "2000-02-29"},
        "output": {"birthday": "2025-03-01"},
    }
    typed = sc._typed_date_cases([case], payload)[0]
    execution = sc._case_formula_execution(rule, typed)
    assert execution is not None
    actual = sc._formula_execution_runtime_value(execution)
    assert actual == date(2025, 2, 28)
    assert actual != typed["output"]["birthday"]


@pytest.mark.parametrize(
    "period",
    [
        "2025",
        2025,
        "2025-02-03",
        {"start": "2025-01-01", "end": "2025-01-02"},
        {"period_kind": "nonsense", "start": "2025-01-01", "end": "2025-01-02"},
        {"period_kind": "month", "start": "2025-02-01", "end": "2025-01-01"},
        {"period_kind": "month", "start": "2025-01-01", "end": "2025-02-29"},
        {"period_kind": "custom", "start": "2025-01-01", "end": "2025-01-01"},
        {
            "period_kind": "custom",
            "name": 1,
            "start": "2025-01-01",
            "end": "2025-01-01",
        },
        {
            "period_kind": "custom",
            "name": " ",
            "start": "2025-01-01",
            "end": "2025-01-01",
        },
    ],
)
def test_invalid_companion_period_is_not_a_runtime_coordinate(period):
    assert (
        sc._case_runtime_period_start({"period": period})
        is sc._UNRESOLVED_CONDITION_VALUE
    )


@pytest.mark.parametrize("location", ["input", "dependency", "formula"])
def test_missing_period_cannot_be_supplied_by_a_caller_alias(location):
    rule = {"name": "shifted", "formula": "date_add_days(period_start, 1)"}
    case = {"input": {}, "output": {"shifted": date(2025, 1, 2)}}
    kwargs = {}
    alias = {"period_start": date(2025, 1, 1)}
    if location == "input":
        case["input"] = alias
    else:
        kwargs[location + "_environment"] = alias
    execution = sc._case_formula_execution(rule, case, **kwargs)
    assert (
        execution is None
        or sc._formula_execution_runtime_value(execution)
        is sc._UNRESOLVED_CONDITION_VALUE
    )

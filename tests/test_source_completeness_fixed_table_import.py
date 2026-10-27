import copy
from datetime import date

import pytest
import yaml

from axiom_encode.harness import source_completeness as completeness


def _provider():
    window = {"effective_from": "2025-10-01", "effective_to": "2026-09-30"}
    return {
        "format": "rulespec/v1",
        "inputs": [{"name": "household_size", "dtype": "Integer"}],
        "rules": [
            {
                "name": "one_person_cost",
                "kind": "derived",
                "dtype": "Money",
                "unit": "USD",
                "entity": "Household",
                "period": "Month",
                "versions": [{**window, "formula": "allotments[1]"}],
            },
            {
                "name": "allotments",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "USD",
                "indexed_by": "household_size",
                "versions": [{**window, "values": {1: 298, 2: 546}}],
            },
        ],
    }


def _resolve(provider):
    content = provider if isinstance(provider, str) else yaml.safe_dump(provider)
    return completeness._resolved_imported_parameter_rules(
        {"imports": ["us:provider#one_person_cost"], "rules": []},
        imported_symbol_contents=[("one_person_cost", content)],
    )


def test_fixed_import_preserves_provider_and_exact_temporal_window():
    provider = _provider()
    original = copy.deepcopy(provider)
    resolved = _resolve(provider)
    rule = resolved["one_person_cost"]
    assert rule["versions"] == [
        {"effective_from": "2025-10-01", "effective_to": "2026-09-30", "formula": "298"}
    ]
    assert rule["kind"] == "parameter"
    assert provider == original
    environment = completeness._constant_rule_environment({"rules": [rule]})
    value = environment["one_person_cost"]
    assert isinstance(value, completeness._TemporalFormulaValue)
    assert value.versions == (("2025-10-01", "2026-09-30", 298),)


@pytest.mark.parametrize(
    "mutation",
    [
        "dynamic",
        "boolean_index",
        "arithmetic",
        "missing_cell",
        "imports",
        "duplicate_rule",
        "input_export",
        "input_table",
        "rounding",
        "table_rounding",
        "mismatched_dates",
        "multiple_versions",
        "unit",
        "dtype",
        "entity",
        "period",
        "nested_index",
        "default",
        "formula_table",
        "boolean_key",
        "string_key",
        "nan",
        "infinity",
        "oversized",
        "float",
        "missing_end",
        "reversed_dates",
        "top_level_date",
        "date_object",
        "nested_value",
    ],
)
def test_fixed_import_rejects_unresolved_or_ambiguous_provider(mutation):
    provider = _provider()
    export, table = provider["rules"]
    version, table_version = export["versions"][0], table["versions"][0]
    if mutation in {"dynamic", "boolean_index", "arithmetic", "missing_cell"}:
        version["formula"] = {
            "dynamic": "allotments[household_size]",
            "boolean_index": "allotments[True]",
            "arithmetic": "allotments[1] + 0",
            "missing_cell": "allotments[3]",
        }[mutation]
    elif mutation == "imports":
        provider["imports"] = ["us:other#value"]
    elif mutation == "duplicate_rule":
        provider["rules"].append(copy.deepcopy(table))
    elif mutation.startswith("input_"):
        provider["inputs"].append(
            {"name": export["name"] if mutation == "input_export" else table["name"]}
        )
    elif mutation == "rounding":
        export["rounding"] = "half_up"
    elif mutation == "table_rounding":
        table["rounding"] = "floor"
    elif mutation == "mismatched_dates":
        table_version["effective_from"] = "2026-01-01"
    elif mutation == "multiple_versions":
        export["versions"].append(copy.deepcopy(version))
    elif mutation in {"unit", "dtype", "entity", "period"}:
        table[mutation] = "Other"
    elif mutation == "nested_index":
        table["indexed_by"] = ["household_size", "state"]
    elif mutation == "default":
        table["default"] = 298
    elif mutation == "formula_table":
        table_version["formula"] = "other_table"
    elif mutation == "boolean_key":
        table_version["values"] = {True: 298}
    elif mutation == "string_key":
        table_version["values"] = {"1": 298}
    elif mutation in {"nan", "infinity", "oversized", "float", "nested_value"}:
        table_version["values"][1] = {
            "nan": "NaN",
            "infinity": "Infinity",
            "oversized": 10**30,
            "float": 298.0,
            "nested_value": {1: 298},
        }[mutation]
    elif mutation == "missing_end":
        version.pop("effective_to")
    elif mutation == "reversed_dates":
        version["effective_to"] = "2024-01-01"
    elif mutation == "top_level_date":
        export["effective_from"] = "2020-01-01"
    elif mutation == "date_object":
        version["effective_from"] = date(2025, 10, 1)
    assert _resolve(provider) == {}


@pytest.mark.parametrize("key", ["1", "true"])
def test_fixed_import_rejects_duplicate_yaml_cells_before_construction(key):
    content = yaml.safe_dump(_provider())
    assert "1: 298" in content
    content = content.replace("1: 298", f"1: 298\n      {key}: 999")
    assert _resolve(content) == {}


@pytest.mark.parametrize(
    "period,expected",
    [
        ("2025-09", {}),
        ("2025-10", {"one_person_cost": 298}),
        ("2026-09", {"one_person_cost": 298}),
        ("2026-10", {}),
    ],
)
def test_fixed_import_does_not_escape_provider_window_or_trust_expected_values(
    period, expected
):
    rule = _resolve(_provider())["one_person_cost"]
    environment = completeness._constant_rule_environment({"rules": [rule]})
    case = {"period": period, "output": {"one_person_cost": 999}}
    assert completeness._formula_environment_for_case(environment, case) == {
        **expected,
        "period_start": date.fromisoformat(period + "-01"),
    }


def test_fixed_import_cannot_be_shadowed_by_a_consumer_input():
    assert (
        completeness._resolved_imported_parameter_rules(
            {
                "imports": ["us:provider#one_person_cost"],
                "inputs": [{"name": "one_person_cost"}],
            },
            imported_symbol_contents=[("one_person_cost", yaml.safe_dump(_provider()))],
        )
        == {}
    )

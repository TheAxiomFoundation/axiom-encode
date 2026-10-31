"""Check synthetic encoder companion inputs; these are not RuleSpec modules."""

from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.companion_scalars import (
    companion_input_keys,
    companion_scalar_value,
    compiled_input_dtypes,
    declared_input_dtypes,
)


@pytest.mark.parametrize("dtype", ["Text", "String", "text", "string"])
@pytest.mark.parametrize("value", ["01", "1", " 1 ", "-1.50", "2026-01-02"])
def test_declared_text_retains_quoted_numeric_and_date_values(dtype, value):
    assert companion_scalar_value(value, dtype=dtype) == {
        "kind": "text",
        "value": value,
    }


@settings(max_examples=50, deadline=None, derandomize=True)
@given(st.text(), st.text())
def test_declared_text_never_merges_distinct_strings(first, second):
    assert (
        companion_scalar_value(first, dtype="Text")
        == companion_scalar_value(second, dtype="Text")
    ) == (first == second)


@pytest.mark.parametrize(
    "value,dtype,roles,expected",
    [
        ("01", "Integer", True, {"kind": "integer", "value": 1}),
        ("1.50", "Decimal", True, {"kind": "decimal", "value": "1.50"}),
        ("2026-01-02", "Date", True, {"kind": "date", "value": "2026-01-02"}),
        ("01", None, True, {"kind": "text", "value": "01"}),
        ("01", "unknown", True, {"kind": "text", "value": "01"}),
        ("01", " Integer ", True, {"kind": "integer", "value": 1}),
        ("2026-01-02", None, True, {"kind": "date", "value": "2026-01-02"}),
        ("01", None, False, {"kind": "integer", "value": 1}),
        (True, "Text", True, {"kind": "bool", "value": True}),
        (1, "Text", True, {"kind": "integer", "value": 1}),
        (date(2026, 1, 2), "Text", True, {"kind": "date", "value": "2026-01-02"}),
    ],
)
def test_scalar_conversion_uses_dtype_and_keeps_legacy_inference(
    value, dtype, roles, expected
):
    assert (
        companion_scalar_value(value, dtype=dtype, relation_role_program=roles)
        == expected
    )


def test_compiled_input_types_include_metadata_request_aliases():
    payload = {
        "program": {
            "interface": {
                "inputs": [
                    {
                        "name": "age",
                        "legal_id": "us:fact#input.age",
                        "dtype": "integer",
                    },
                    {"name": "unknown", "dtype": None},
                    {"name": "unrecognized", "dtype": "unknown"},
                ]
            }
        },
        "metadata": {
            "input_catalog": [
                {
                    "slot": "age",
                    "canonical_request_name": "us:fact#input.age",
                    "request_names": ["us:consumer#input.age"],
                },
            ]
        },
    }
    assert compiled_input_dtypes(payload) == {
        "age": "integer",
        "us:fact#input.age": "integer",
        "us:consumer#input.age": "integer",
    }


@pytest.mark.parametrize("mapping", [False, True])
def test_declared_input_types_follow_imports_and_local_authority(tmp_path, mapping):
    provider = tmp_path / "provider.yaml"
    provider.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        + (
            "inputs: {fact: {dtype: Text}}\n"
            if mapping
            else "inputs: [{name: fact, dtype: Text}]\n"
        )
    )
    consumer = tmp_path / "consumer.yaml"
    consumer.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\nimports: ['us:provider']\n"
        "inputs: [{name: local_fact, dtype: Integer}]\n"
    )
    assert declared_input_dtypes(
        consumer,
        ["fact", "local_fact", "us:provider#input.fact"],
        resolve_reference=lambda _: provider,
        resolve_import=lambda *_: provider,
    ) == {"fact": "Text", "local_fact": "Integer", "us:provider#input.fact": "Text"}


def test_input_key_inventory_excludes_row_identity_and_tuple_metadata():
    assert companion_input_keys(
        {
            "input": {
                "flat": "01",
                "relation": [
                    {"id": "a", "entity_id": "a", "nested": "1"},
                    {"tuple": ["a", "b"]},
                ],
            },
            "tables": {
                "TaxUnit": [{"id": "a", "tax_unit_id": "a", "fact": "2026-01-01"}]
            },
        }
    ) == {"flat", "nested", "fact"}


@pytest.mark.parametrize("value", [" 2026-01-02 ", "\t2026-01-02", "2026-01-02\n"])
def test_unknown_role_type_keeps_padded_dates_as_distinct_text(value):
    assert companion_scalar_value(value, relation_role_program=True) == {
        "kind": "text",
        "value": value,
    }
    assert companion_scalar_value(value, dtype="Date", relation_role_program=True) == {
        "kind": "date",
        "value": "2026-01-02",
    }
    assert companion_scalar_value(value) == {"kind": "date", "value": "2026-01-02"}


@pytest.mark.parametrize("dtype_fields", [{}, {"dtype": None}, {"dtype": ""}])
def test_local_untyped_declaration_blocks_imported_same_name_dtype(
    tmp_path, dtype_fields
):
    import yaml

    provider = tmp_path / "provider.yaml"
    consumer = tmp_path / "consumer.yaml"
    label = "# Synthetic encoder test input; not a RuleSpec module.\n"
    provider.write_text(
        label + yaml.safe_dump({"inputs": [{"name": "fact", "dtype": "Text"}]})
    )
    consumer.write_text(
        label
        + yaml.safe_dump(
            {"imports": ["provider.yaml"], "inputs": [{"name": "fact", **dtype_fields}]}
        )
    )
    assert (
        declared_input_dtypes(
            consumer,
            ["fact"],
            resolve_reference=lambda _: consumer,
            resolve_import=lambda *_: provider,
        )
        == {}
    )

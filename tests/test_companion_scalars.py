"""Check synthetic encoder companion inputs; these are not RuleSpec modules."""

import json
import re
from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.companion_scalars import (
    companion_input_keys,
    companion_scalar_value,
    compiled_input_dtypes,
    compiled_input_readers,
    declared_input_dtypes,
)


@pytest.mark.parametrize("dtype", ["Text", "String", "text", "string"])
@pytest.mark.parametrize("value", ["01", "1", " 1 ", "-1.50", "2026-01-02"])
def test_declared_text_retains_quoted_numeric_and_date_values(dtype, value):
    assert companion_scalar_value(value, dtype=dtype, infer_numeric_text=True) == {
        "kind": "text",
        "value": value,
    }


@settings(max_examples=50, deadline=None, derandomize=True)
@given(st.text(), st.text())
def test_declared_text_never_merges_distinct_strings(first, second):
    assert (
        companion_scalar_value(first, dtype="Text", infer_numeric_text=True)
        == companion_scalar_value(second, dtype="Text", infer_numeric_text=True)
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
        ("1", "Boolean", True, {"kind": "text", "value": "1"}),
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
        companion_scalar_value(value, dtype=dtype, infer_numeric_text=not roles)
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
        "age": "number",
        "us:fact#input.age": "number",
        "us:consumer#input.age": "number",
    }


@pytest.mark.parametrize("mapping", [False, True])
def test_declared_input_types_follow_readers_and_local_authority(tmp_path, mapping):
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
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        "inputs: [{name: local_fact, dtype: Integer}]\n"
    )
    readers = compiled_input_readers(
        {
            "metadata": {
                "input_catalog": [
                    {
                        "slot": "fact",
                        "request_names": ["us:provider#input.fact"],
                    }
                ]
            }
        }
    )
    assert declared_input_dtypes(
        consumer,
        ["fact", "local_fact", "us:provider#input.fact"],
        input_readers=readers,
        resolve_reference=lambda _: provider,
    ) == {"fact": "text", "local_fact": "Integer", "us:provider#input.fact": "Text"}


def test_the_first_declaration_of_a_name_decides_its_dtype(tmp_path):
    module = tmp_path / "module.yaml"
    module.write_text(
        "# Synthetic encoder test input; not a RuleSpec module.\n"
        "inputs: [{name: fact, dtype: Text}, {name: fact, dtype: Integer}]\n"
    )
    assert declared_input_dtypes(
        module,
        ["fact"],
        input_readers={},
        resolve_reference=lambda _: None,
    ) == {"fact": "Text"}


@pytest.mark.parametrize(
    "inputs",
    [
        "[{name: []}, {name: fact, dtype: Text}]",
        "[{name: {other: 1}}, {name: fact, dtype: Text}]",
        "[{name: 1}, {name: fact, dtype: Text}]",
        "{1: {dtype: Integer}, fact: {dtype: Text}}",
    ],
)
def test_an_entry_without_a_usable_name_describes_no_input(tmp_path, inputs):
    module = tmp_path / "module.yaml"
    module.write_text(
        f"# Synthetic encoder test input; not a RuleSpec module.\ninputs: {inputs}\n"
    )
    assert declared_input_dtypes(
        module,
        ["fact", "1"],
        input_readers={},
        resolve_reference=lambda _: None,
    ) == {"fact": "Text"}


@pytest.mark.parametrize("inputs", ["5", "text", "null", "true"])
def test_a_scalar_inputs_block_declares_nothing(tmp_path, inputs):
    module = tmp_path / "module.yaml"
    module.write_text(
        f"# Synthetic encoder test input; not a RuleSpec module.\ninputs: {inputs}\n"
    )
    assert (
        declared_input_dtypes(
            module,
            ["fact"],
            input_readers={},
            resolve_reference=lambda _: None,
        )
        == {}
    )


def _reader_modules(tmp_path, **declared):
    label = "# Synthetic encoder test input; not a RuleSpec module.\n"
    modules = {}
    for name, dtype in declared.items():
        modules[name] = tmp_path / f"{name}.yaml"
        fields = {"name": "fact", "dtype": dtype} if dtype else {"name": "other"}
        modules[name].write_text(label + f"inputs: [{json.dumps(fields)}]\n")
    return modules


def _resolver(modules):
    return lambda key: modules.get(key.split(":", 1)[1].split("#", 1)[0])


def test_an_absolute_key_never_takes_a_dtype_from_a_module_that_does_not_read_it(
    tmp_path,
):
    modules = _reader_modules(tmp_path, consumer=None, one="Text", two="Integer")
    key = "us:consumer#input.fact"
    assert (
        declared_input_dtypes(
            modules["consumer"],
            [key],
            input_readers={key: (key,), "fact": (key,)},
            resolve_reference=_resolver(modules),
        )
        == {}
    )


def test_disagreeing_readers_leave_the_input_untyped(tmp_path):
    modules = _reader_modules(tmp_path, consumer=None, one="Text", two="Integer")
    names = ("us:one#input.fact", "us:two#input.fact", "us:consumer#input.fact")
    assert (
        declared_input_dtypes(
            modules["consumer"],
            ["fact", names[2]],
            input_readers=dict.fromkeys(("fact", *names), names),
            resolve_reference=_resolver(modules),
        )
        == {}
    )


def test_agreeing_readers_share_a_slot_dtype(tmp_path):
    modules = _reader_modules(tmp_path, consumer=None, one="Text", two="text")
    names = ("us:one#input.fact", "us:two#input.fact", "us:consumer#input.fact")
    assert declared_input_dtypes(
        modules["consumer"],
        [names[2]],
        input_readers=dict.fromkeys(("fact", *names), names),
        resolve_reference=_resolver(modules),
    ) == {names[2]: "text"}


def _two_reader_dtypes(tmp_path, one, two):
    modules = _reader_modules(tmp_path, consumer=None, one=one, two=two)
    names = ("us:one#input.fact", "us:two#input.fact", "us:consumer#input.fact")
    return declared_input_dtypes(
        modules["consumer"],
        [names[2]],
        input_readers=dict.fromkeys(("fact", *names), names),
        resolve_reference=_resolver(modules),
    ).get(names[2])


@pytest.mark.parametrize(
    "one,two,expected",
    [
        ("int", "Integer", "number"),
        ("Integer", "Decimal", "number"),
        ("Money", "float", "number"),
        ("Rate", "Number", "number"),
        ("Text", "String", "text"),
        ("Bool", "Judgment", "bool"),
        ("boolean", "bool", "bool"),
        ("Date", "date", "date"),
    ],
)
def test_readers_agree_when_they_convert_a_quoted_value_alike(
    tmp_path, one, two, expected
):
    assert _two_reader_dtypes(tmp_path, one, two) == expected
    assert _two_reader_dtypes(tmp_path, two, one) == expected


@pytest.mark.parametrize(
    "one,two",
    [
        ("Text", "Integer"),
        ("String", "int"),
        ("Date", "Text"),
        ("Date", "Decimal"),
        ("Bool", "Integer"),
        ("boolean", "Text"),
        ("Judgment", "Text"),
    ],
)
def test_readers_disagree_when_they_convert_a_quoted_value_differently(
    tmp_path, one, two
):
    assert _two_reader_dtypes(tmp_path, one, two) is None
    assert _two_reader_dtypes(tmp_path, two, one) is None


def test_a_rule_qualified_key_finds_its_readers(tmp_path):
    modules = _reader_modules(tmp_path, consumer=None, one="Text")
    names = ("us:one#input.fact", "us:consumer#input.fact")
    key = "us:consumer#result.input.fact"
    assert declared_input_dtypes(
        modules["consumer"],
        [key],
        input_readers=dict.fromkeys(("fact", *names), names),
        resolve_reference=_resolver(modules),
    ) == {key: "text"}


def test_compiled_readers_read_both_slots_and_request_names():
    payload = {
        "metadata": {
            "input_catalog": [
                {"slot": "age", "request_names": ["us:a#input.age", "us:b#input.age"]},
                "skipped",
            ]
        }
    }
    names = ("us:a#input.age", "us:b#input.age")
    assert compiled_input_readers(payload) == dict.fromkeys(("age", *names), names)
    assert compiled_input_readers({}) == {}


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
    assert companion_scalar_value(value, infer_numeric_text=False) == {
        "kind": "text",
        "value": value,
    }
    assert companion_scalar_value(value, dtype="Date", infer_numeric_text=False) == {
        "kind": "date",
        "value": "2026-01-02",
    }
    assert companion_scalar_value(value, infer_numeric_text=True) == {
        "kind": "date",
        "value": "2026-01-02",
    }


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
            input_readers={"fact": ("us:provider#input.fact",)},
            resolve_reference=lambda _: provider,
        )
        == {}
    )


def base_cli_value(value):
    """Copy the CLI converter from before the shared one, as a reference."""
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "integer", "value": value}
    if isinstance(value, float):
        return {"kind": "decimal", "value": str(value)}
    if isinstance(value, date):
        return {"kind": "date", "value": value.isoformat()}
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return {"kind": "date", "value": value}
    return {"kind": "text", "value": value}


def base_pipeline_value(value):
    """Copy the pipeline converter from before the shared one, as a reference."""
    if not isinstance(value, str):
        return base_cli_value(value)
    stripped = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped):
        return {"kind": "date", "value": stripped}
    if re.fullmatch(r"-?\d+", stripped):
        return {"kind": "integer", "value": int(stripped)}
    if re.fullmatch(r"-?(?:\d+\.\d*|\d*\.\d+)", stripped):
        return {"kind": "decimal", "value": stripped}
    return {"kind": "text", "value": value}


SCALARS = st.one_of(
    st.booleans(),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.dates(),
    st.text(max_size=12),
    st.from_regex(r"\s?-?[0-9]{1,4}(\.[0-9]{0,2})?\s?", fullmatch=True),
    st.from_regex(r"\s?-?\.[0-9]{1,2}\s?", fullmatch=True),
    st.from_regex(r"\s?[0-9]{4}-[0-9]{2}-[0-9]{2}\s?", fullmatch=True),
)


@given(SCALARS)
def test_an_untyped_value_keeps_each_runners_own_conversion(value):
    assert companion_scalar_value(value, infer_numeric_text=False) == base_cli_value(
        value
    )
    assert companion_scalar_value(
        value, infer_numeric_text=True
    ) == base_pipeline_value(value)


@given(
    SCALARS,
    st.sampled_from(["Integer", "Decimal", "Money", "Rate", "Text", "String", "Date"]),
)
def test_a_declared_type_converts_alike_in_both_runners(value, dtype):
    assert companion_scalar_value(
        value, dtype=dtype, infer_numeric_text=False
    ) == companion_scalar_value(value, dtype=dtype, infer_numeric_text=True)


@pytest.mark.parametrize(
    "dtype", ["Integer", "Int", "Decimal", "Float", "Money", "Rate", "Number"]
)
def test_every_numeric_type_converts_a_quoted_number(dtype):
    assert companion_scalar_value("01", dtype=dtype, infer_numeric_text=False) == {
        "kind": "integer",
        "value": 1,
    }


@given(st.text(max_size=12), st.sampled_from(["Text", "String", " text "]))
def test_declared_text_keeps_its_exact_value(value, dtype):
    for infer in (False, True):
        assert companion_scalar_value(value, dtype=dtype, infer_numeric_text=infer) == {
            "kind": "text",
            "value": value,
        }

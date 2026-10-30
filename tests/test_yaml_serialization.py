"""String preservation across the encoder's YAML 1.1/1.2 boundary."""

from __future__ import annotations

import io
import re

import pytest
import yaml
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.yaml_serialization import safe_dump_yaml


class _Yaml12CoreLoader(yaml.BaseLoader):
    """Independent YAML 1.2 core-schema resolver, without PyYAML 1.1 rules."""

    yaml_implicit_resolvers = {}


# YAML 1.2.2 section 10.3.2: these are the non-string core-schema tags.
# Keep the resolver independent of the production representer's expression.
for _tag, _expression in (
    ("null", r"^(?:~|null|Null|NULL|)$"),
    ("bool", r"^(?:true|True|TRUE|false|False|FALSE)$"),
    ("int", r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
    (
        "float",
        r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$",
    ),
):
    _Yaml12CoreLoader.add_implicit_resolver(
        f"tag:yaml.org,2002:{_tag}", re.compile(_expression), None
    )


@settings(max_examples=500, deadline=None)
@given(st.text())
@example("1e5")
@example("2e10")
@example("-0o17")
@example("+.5")
@example("0x1a")
@example(".inf")
@example(".nan")
@example("true")
@example("null")
@example("")
@example("\x85")
@example("\x85\n")
@example("\x00\r\n\ufeff\u2028\u2029\"'")
@example("\ud800\udfff")
def test_dump_preserves_arbitrary_string_in_yaml11_and_yaml12(value: str):
    dumped = safe_dump_yaml(value, allow_unicode=True)
    loaded = yaml.safe_load(dumped)
    assert isinstance(loaded, str)
    assert loaded == value
    node = yaml.compose(dumped, Loader=_Yaml12CoreLoader)
    assert isinstance(node, yaml.ScalarNode)
    assert node.tag == "tag:yaml.org,2002:str"
    assert node.value == value


@pytest.mark.parametrize(
    "value",
    [
        "0",
        "-17",
        "+17",
        "0017",
        "0o17",
        "-0o17",
        "+0o17",
        "0x1a",
        "-0x1a",
        "+0x1a",
        "1e5",
        "2e10",
        "-1E-5",
        "+1E+5",
        "0.",
        ".5",
        "+.5",
        "-.5",
        ".inf",
        "+.Inf",
        "-.INF",
        ".nan",
        ".NaN",
        ".NAN",
        "true",
        "True",
        "TRUE",
        "false",
        "False",
        "FALSE",
        "null",
        "Null",
        "NULL",
        "~",
        "",
    ],
)
def test_dump_quotes_all_core_non_string_forms(value: str):
    dumped = safe_dump_yaml({"value": value})
    node = yaml.compose(dumped, Loader=_Yaml12CoreLoader).value[0][1]
    assert node.style in {"'", '"'}
    assert yaml.safe_load(dumped) == {"value": value}


def test_dump_preserves_stream_and_existing_options():
    stream = io.StringIO()
    assert (
        safe_dump_yaml(
            {"z": "1e5", "a": "é"},
            stream=stream,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=True,
        )
        is None
    )
    assert stream.getvalue() == '{z: "1e5", a: é}\n'


def test_dump_does_not_reconfigure_global_pyyaml():
    assert yaml.compose(yaml.safe_dump("1e5")).style is None
    assert yaml.compose(safe_dump_yaml("1e5")).style == '"'


@pytest.mark.parametrize("name", ["1e5", "a\x85b\n"])
def test_repair_dumpers_preserve_numeric_strings_and_literal_blocks(name):
    from axiom_encode.cli import (
        _dump_rulespec_repair_yaml,
        _render_rulespec_rule_block,
    )

    payload = {"name": name, "description": "first\nsecond\n"}
    rendered = _dump_rulespec_repair_yaml(payload)
    assert yaml.safe_load(rendered) == payload
    assert 'name: "' in rendered
    assert "description: |" in rendered
    block = _render_rulespec_rule_block(payload)
    assert yaml.safe_load(block) == [payload]
    assert 'name: "' in block

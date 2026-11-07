"""Closed MHRtC static contract matching, without source/coverage authority.

The reference roles and formulas remain subject to independent source review.
A match cannot mint execution evidence or suppress any completeness finding.
No candidate expression is evaluated and no diagnostic fixture is read here.
"""

from __future__ import annotations

import ast
import hashlib
import io
import re
import tokenize
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import yaml

from .mhrtc_native_envelope import (
    INPUT_DESCRIPTIONS,
    MODULE,
    RULE_METADATA,
    RULE_SOURCES,
)
from .mhrtc_program_contract import (
    CITATION,
    INPUTS,
    POLICY_PATH,
    RULE_GROUPS,
    RULES,
    SOURCE_GROUP_SPANS,
    SOURCE_TEXT,
    UNITS,
)


@dataclass(frozen=True)
class MhrtcProgramBinding:
    """Static review evidence only; publicly constructible, never authority."""

    content_sha256: str
    source_sha256: str
    expressions: tuple[tuple[str, str], ...]
    rule_groups: tuple[tuple[str, tuple[str, ...]], ...]
    source_spans: tuple[tuple[str, int, int, str], ...]
    unresolved_obligations: tuple[str, ...] = (
        "independent source semantic review",
        "fresh compiler/runtime binding",
        "complete-source occurrence/witness coverage",
    )


class _UniqueLoader(yaml.SafeLoader):
    """Reject duplicate keys, aliases, anchors and YAML merge semantics."""


def _exact_decimal(loader: _UniqueLoader, node: yaml.ScalarNode) -> Decimal:
    """Preserve YAML decimal precision before checking the semantic reference."""
    try:
        value = Decimal(loader.construct_scalar(node))
    except InvalidOperation as exc:
        raise ValueError("ordinary finite decimal required") from exc
    if not value.is_finite():
        raise ValueError("ordinary finite decimal required")
    return value


_UniqueLoader.add_constructor("tag:yaml.org,2002:float", _exact_decimal)


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict:
    result = {}
    for key_node, value_node in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise ValueError("merge keys outside contract")
        key = loader.construct_object(key_node)
        if type(key) not in (str, int, Decimal, bool) or key in result:
            raise ValueError("duplicate or unsupported mapping key")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


def _typed_equal(left: object, right: object) -> bool:
    """Compare mapping keys as well as values without bool/int coercion."""
    # The reviewed Python reference spells its decimal constants explicitly.
    # Candidate YAML is parsed as Decimal, never rounded through a Python float.
    if type(left) is Decimal and type(right) is float:
        return left == Decimal(str(right))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        if len(left) != len(right):
            return False
        for key, value in left.items():
            matching = [other for other in right if _typed_equal(key, other)]
            if len(matching) != 1 or not _typed_equal(value, right[matching[0]]):
                return False
        return True
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _typed_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _closed_expression(text: str) -> str:
    """Normalize only the bounded, non-evaluating M formula AST grammar."""
    if type(text) is not str or not text or len(text) > 12000:
        raise ValueError("bounded formula required")
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    if any(token.type == tokenize.COMMENT for token in tokens):
        raise ValueError("formula comments outside grammar")
    numbers = []
    for token in tokens:
        if token.type != tokenize.NUMBER:
            continue
        if not re.fullmatch(
            r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", token.string
        ):
            raise ValueError("ordinary decimal numeric literal required")
        is_decimal = any(mark in token.string for mark in (".", "e", "E"))
        numbers.append(("decimal" if is_decimal else "integer", Decimal(token.string)))
    inline = re.fullmatch(r"if ([^\n]+?): ([^\n]+?) else: ([^\n]+)", text)
    if inline:
        text = f"if {inline[1]}:\n    {inline[2]}\nelse:\n    {inline[3]}"
    tree = ast.parse(text.strip(), mode="exec")
    allowed = (
        ast.Module,
        ast.Expr,
        ast.If,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.UnaryOp,
        ast.Not,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Attribute,
        ast.Subscript,
        ast.Call,
    )
    nodes = list(ast.walk(tree))
    if len(nodes) > 2048 or any(type(node) not in allowed for node in nodes):
        raise ValueError("node outside bounded grammar")
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.Expr, ast.If)):
        raise ValueError("one expression required")
    rule_by_name = {rule["name"]: rule for rule in RULES}
    names = set(rule_by_name) | {item["name"] for item in INPUTS}
    names |= {"period_start", "period_end", "true", "false", "min", "sum"}
    relation_names = {
        name
        for name, rule in rule_by_name.items()
        if rule["kind"] in {"data_relation", "derived_relation"}
    }
    for node in nodes:
        if isinstance(node, ast.Name) and node.id not in names:
            raise ValueError("unknown formula symbol")
        if isinstance(node, ast.Constant) and type(node.value) not in (int, float, str):
            raise ValueError("unsupported literal type")
        if isinstance(node, ast.If) and (
            len(node.body) != 1
            or len(node.orelse) != 1
            or not isinstance(node.body[0], ast.Expr)
            or not isinstance(node.orelse[0], ast.Expr)
        ):
            raise ValueError("expression arms required")
        if isinstance(node, ast.Attribute) and (
            not isinstance(node.value, ast.Name)
            or node.value.id not in relation_names
            or node.attr not in names
        ):
            raise ValueError("declared relation member required")
        if isinstance(node, ast.Subscript) and (
            not isinstance(node.value, ast.Name)
            or "indexed_by" not in rule_by_name.get(node.value.id, {})
            or not isinstance(node.slice, ast.Constant)
            or type(node.slice.value) is not int
            or node.slice.value != 0
        ):
            raise ValueError("fixed integer parameter key required")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.keywords:
                raise ValueError("closed call required")
            if node.func.id == "min" and len(node.args) == 2:
                continue
            if (
                node.func.id == "sum"
                and len(node.args) == 1
                and isinstance(node.args[0], ast.Attribute)
            ):
                continue
            raise ValueError("only binary min or relation-member sum allowed")
    # AST float constants alone collapse distinct exact decimal tokens. Keep
    # the exact values and their integer/decimal kind alongside the AST shape.
    return ast.dump(tree, include_attributes=False) + repr(numbers)


def _named_items(items: object) -> dict[str, dict]:
    if type(items) is not list:
        raise ValueError("declaration list required")
    result = {}
    for item in items:
        if (
            type(item) is not dict
            or type(item.get("name")) is not str
            or item["name"] in result
        ):
            raise ValueError("unique named declarations required")
        result[item["name"]] = item
    return result


def _match_rule(rule: dict, reference: dict) -> str | None:
    if set(rule) != set(reference):
        raise ValueError("closed rule fields required")
    for key in rule:
        if key != "versions" and not _typed_equal(rule[key], reference[key]):
            raise ValueError("rule declaration differs")
    if "versions" not in reference:
        return None
    versions = rule["versions"]
    if type(versions) is not list or len(versions) != 1:
        raise ValueError("one exact 2025 version required")
    version, expected = versions[0], reference["versions"][0]
    if type(version) is not dict or set(version) != set(expected):
        raise ValueError("closed version fields required")
    expression = None
    for key in version:
        if key == "formula":
            expression = _closed_expression(version[key])
            if expression != _closed_expression(expected[key]):
                raise ValueError("semantic reference formula differs")
        elif not _typed_equal(version[key], expected[key]):
            raise ValueError("typed version value differs")
    return expression


def bind_mhrtc_program(
    content: str,
    source: str,
    *,
    citation: str = CITATION,
    policy_path: str = POLICY_PATH,
) -> MhrtcProgramBinding | None:
    """Match the frozen static reference, retaining every authority obligation.

    Whole source literals and live excerpt coordinates are checked, but this
    does not authenticate the supplied source or certify the reference's legal
    interpretation. Future private owners must independently establish both.
    """
    try:
        if (
            type(content) is not str
            or len(content) > 1024 * 1024
            or type(source) is not str
            or source != SOURCE_TEXT
            or type(citation) is not str
            or citation != CITATION
            or type(policy_path) is not str
            or policy_path != POLICY_PATH
        ):
            return None
        for token in yaml.scan(content):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                return None
        doc = yaml.load(content, Loader=_UniqueLoader)
        if type(doc) is not dict or set(doc) != {
            "format",
            "imports",
            "module",
            "units",
            "inputs",
            "rules",
        }:
            return None
        if (
            doc["format"] != "rulespec/v1"
            or not _typed_equal(doc["imports"], [])
            or not _typed_equal(doc["module"], MODULE)
            or not _typed_equal(doc["units"], UNITS)
        ):
            return None
        inputs = _named_items(doc["inputs"])
        expected_inputs = _named_items(
            [
                {**item, "description": INPUT_DESCRIPTIONS[item["name"]]}
                for item in INPUTS
            ]
        )
        rules = _named_items(doc["rules"])
        if (
            not _typed_equal(inputs, expected_inputs)
            or set(rules) != {rule["name"] for rule in RULES}
            or set(rules) != set(RULE_GROUPS)
        ):
            return None
        expressions = []
        for reference in RULES:
            name = reference["name"]
            expression = _match_rule(
                rules[name],
                {
                    **reference,
                    "source": RULE_SOURCES[name],
                    "metadata": RULE_METADATA[name],
                },
            )
            if expression is not None:
                expressions.append((name, expression))
        spans = []
        for group, excerpts in SOURCE_GROUP_SPANS.items():
            for start, end, excerpt in excerpts:
                if source[start:end] != excerpt:
                    return None
                spans.append((group, start, end, excerpt))
        if set(SOURCE_GROUP_SPANS) != {
            group for groups in RULE_GROUPS.values() for group in groups
        }:
            return None
        return MhrtcProgramBinding(
            hashlib.sha256(content.encode()).hexdigest(),
            hashlib.sha256(source.encode()).hexdigest(),
            tuple(expressions),
            tuple((name, RULE_GROUPS[name]) for name in rules),
            tuple(spans),
        )
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        SyntaxError,
        RecursionError,
        yaml.YAMLError,
        tokenize.TokenError,
    ):
        return None

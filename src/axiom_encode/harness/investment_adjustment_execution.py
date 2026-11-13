"""Pinned Schedule N compiled/record observation, never capture authority."""

from __future__ import annotations

import ast
import json
import re
from decimal import Decimal

from .bound_investment_adjustment_operands import (
    CITATION,
    FLAGS,
    INPUT_ROWS,
    RATE,
    InvestmentAdjustmentBinding,
    _contracts,
    _expression,
)
from .bound_investment_adjustment_operands import (
    PREFIX as LOCAL_PREFIX,
)

ENGINE_PIN = "89571cc2a938707fd60a5489b951135345697296"
PREFIX = "ca:policies/revenu-quebec/tp1-2025/schedule-n-investment-expense-adjustment#"


def _same(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _units():
    # Official895 formula.rs1125-1189, with the declared CAD unit appended.
    return [
        {
            "name": n,
            "kind": "currency",
            "minor_units": 0 if n in {"GHp", "UGX", "RWF"} else 2,
        }
        for n in (
            "GBP",
            "USD",
            "EUR",
            "CHF",
            "NOK",
            "GHS",
            "GHp",
            "NGN",
            "UGX",
            "ZMW",
            "ETB",
            "RWF",
            "TZS",
            "DKK",
            "AMD",
            "ILS",
        )
    ] + [
        {"name": "count", "kind": "count"},
        {"name": "person", "kind": "count"},
        {"name": "ratio", "kind": "ratio"},
        {"name": "status", "kind": "custom", "label": "judgment"},
        {"name": "CAD", "kind": "currency", "minor_units": 2},
    ]


def _literal(value, decimal=False):
    if type(value) is bool:
        wire = {"kind": "bool", "value": value}
    elif decimal or type(value) is float:
        wire = {"kind": "decimal", "value": str(Decimal(str(value)).normalize())}
    elif type(value) is int:
        wire = {"kind": "integer", "value": value}
    else:
        raise ValueError("unsupported literal")
    return {"kind": "literal", "value": wire}


def _lower(node, inputs, derived, decimal=False):
    if isinstance(node, ast.Name):
        if node.id in ("true", "false"):
            return _literal(node.id == "true")
        if node.id == RATE:
            return {"kind": "parameter_lookup", "parameter": RATE, "index": _literal(0)}
        if node.id in inputs:
            return {"kind": "input", "name": node.id}
        if node.id in derived:
            return {"kind": "derived", "name": node.id}
        raise ValueError("unknown operand")
    if isinstance(node, ast.Constant):
        return _literal(node.value, decimal)
    if isinstance(node, ast.BinOp):
        left = _lower(node.left, inputs, derived, decimal)
        right = _lower(node.right, inputs, derived, decimal)
        if type(node.op) is ast.Add:
            return {"kind": "add", "items": [left, right]}
        op = {ast.Sub: "sub", ast.Mult: "mul"}[type(node.op)]
        return {"kind": op, "left": left, "right": right}
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        return {
            "kind": "comparison",
            "left": _lower(node.left, inputs, derived),
            "op": {ast.Eq: "eq", ast.Gt: "gt", ast.GtE: "gte", ast.LtE: "lte"}[
                type(node.ops[0])
            ],
            "right": _lower(node.comparators[0], inputs, derived),
        }
    if isinstance(node, ast.BoolOp):
        items = [_lower(v, inputs, derived) for v in node.values]
        result = items[0]
        for item in items[1:]:
            result = {
                "kind": {ast.And: "and", ast.Or: "or"}[type(node.op)],
                "items": [result, item],
            }
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("min", "max")
        and len(node.args) == 2
        and not node.keywords
    ):
        return {
            "kind": node.func.id,
            "items": [_lower(a, inputs, derived, decimal) for a in node.args],
        }
    if isinstance(node, ast.If):
        return {
            "kind": "if",
            "condition": _lower(node.test, inputs, derived),
            "then_expr": _lower(node.body[0].value, inputs, derived, decimal),
            "else_expr": _lower(node.orelse[0].value, inputs, derived, decimal),
        }
    raise ValueError("unsupported lowering node")


def _lower_formula(text, inputs, derived, decimal=False):
    _expression(text)  # Strict independently reviewed scalar grammar.
    match = re.fullmatch(r"match ([A-Za-z_]\w*):\n +true => ([^\n]+)\n?", text)
    if match:
        subject = _lower(ast.Name(id=match[1]), inputs, derived)
        return {
            "kind": "if",
            "condition": {
                "kind": "comparison",
                "left": subject,
                "op": "eq",
                "right": _literal(True),
            },
            "then_expr": _lower(
                ast.parse(match[2], mode="eval").body, inputs, derived, decimal
            ),
            "else_expr": {
                "kind": "no_match",
                "subject": subject,
                "patterns": [_literal(True)],
            },
        }
    inline = re.fullmatch(r"if ([^\n]+?): ([^\n]+?) else: ([^\n]+)", text)
    if inline:
        text = f"if {inline[1]}:\n    {inline[2]}\nelse:\n    {inline[3]}"
    node = ast.parse(text, mode="exec").body[0]
    return _lower(
        node.value if isinstance(node, ast.Expr) else node, inputs, derived, decimal
    )


def _dependencies(expr):
    if isinstance(expr, dict):
        if expr.get("kind") == "derived":
            yield expr["name"]
        for value in expr.values():
            yield from _dependencies(value)
    elif isinstance(expr, list):
        for value in expr:
            yield from _dependencies(value)


def verify_investment_adjustment_compiled(
    bound: InvestmentAdjustmentBinding,
    content: str,
    parent: str,
    guide: str,
    artifact: dict,
    engine_pin: str,
):
    """Independently lower every source-owned formula and compare the entire IR."""
    from .validator_pipeline import _safe_load_unique_keys

    if (
        type(bound) is not InvestmentAdjustmentBinding
        or not bound.bound(content, parent, guide)
        or engine_pin != ENGINE_PIN
    ):
        raise ValueError("unbound source or engine")
    doc = _safe_load_unique_keys(content)
    inputs = {LOCAL_PREFIX + v for v in INPUT_ROWS.values()} | {
        LOCAL_PREFIX + "resource_deduction_recovery_base",
        *FLAGS,
    }
    contracts = _contracts(parent)
    names = {name for name, *_ in contracts if name != RATE}
    derived = []
    parameters = []
    for name, dtype, formula, proofs in contracts:
        common = {
            "id": PREFIX + name,
            "name": name,
            "unit": "CAD" if dtype == "Money" else None,
            "source": proofs[0][1],
            "source_url": None,
            "corpus_citation_path": CITATION,
        }
        if name == RATE:
            parameters.append(
                {
                    **common,
                    "indexed_by": None,
                    "versions": [
                        {
                            "effective_from": "2025-01-01",
                            "effective_to": "2025-12-31",
                            "values": {"0": {"kind": "decimal", "value": "0.5"}},
                        }
                    ],
                }
            )
            continue
        expr = _lower_formula(formula, inputs, names, dtype == "Money")
        semantics = "judgment" if dtype == "Judgment" else "scalar"
        derived.append(
            {
                **common,
                "entity": "Person",
                "dtype": {
                    "Money": "decimal",
                    "Boolean": "bool",
                    "Judgment": "judgment",
                }[dtype],
                "period": "Year",
                "semantics": semantics,
                "expr": expr,
                "versions": [
                    {
                        "effective_from": "2025-01-01",
                        "effective_to": "2025-12-31",
                        "semantics": semantics,
                        "expr": expr,
                    }
                ],
            }
        )
    expected_program = {
        "module": {
            "summary": doc["module"]["summary"],
            "source_verification": doc["module"]["source_verification"],
        },
        "units": _units(),
        "relations": [],
        "parameters": parameters,
        "derived": derived,
    }
    if (
        set(artifact)
        != {"artifact_format_version", "engine_version", "program", "metadata"}
        or type(artifact["artifact_format_version"]) is not int
        or artifact["artifact_format_version"] != 2
        or artifact["engine_version"] != "0.2.2"
        or not _same(artifact["program"], expected_program)
    ):
        raise ValueError("compiled program differs from full source contract")
    expected_catalog = [
        {
            "slot": n,
            "canonical_request_name": PREFIX + "input." + n,
            "request_names": [PREFIX + "input." + n],
        }
        for n in sorted(inputs)
    ]
    metadata = artifact["metadata"]
    if (
        set(metadata) != {"evaluation_order", "fast_path", "input_catalog"}
        or not _same(metadata["input_catalog"], expected_catalog)
        or not _same(
            metadata["fast_path"],
            {"strategy": "generic_bulk", "compatible": True, "blockers": []},
        )
    ):
        raise ValueError("compiled metadata differs")
    order = metadata["evaluation_order"]
    if type(order) is not list or len(order) != len(names) or set(order) != names:
        raise ValueError("incomplete evaluation order")
    positions = {name: index for index, name in enumerate(order)}
    for rule in derived:
        if any(
            positions[dep] >= positions[rule["name"]]
            for dep in _dependencies(rule["expr"])
        ):
            raise ValueError("dependency order differs")
    return {rule["name"]: rule for rule in derived}

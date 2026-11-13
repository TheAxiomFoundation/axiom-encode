"""Closed conditional excess syntax; not final-refund entitlement or admission."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Mapping

from axiom_encode.bound_form_structure import BoundFormStructure

from .bound_form_obligations import _proofs


@dataclass(frozen=True)
class ExcessExpression:
    credit: str
    taxes: str
    tree: str


@dataclass(frozen=True)
class CreditExcessRequirement:
    rule_name: str
    input_name: str
    credit_owner: str
    return_field: str
    source_span: tuple[int, int]
    source_excerpt: str
    entity: str
    unit: str
    frame: tuple[str, str]
    formula_ast: str


_CLAUSE = re.compile(
    r"(?m)^You can claim these refundable tax credits even if you do not have to pay any\r?\n"
    r"tax\. (?P<excess>If the total of this credit is more than the amount of taxes you have to\r?\n"
    r"pay, you may get a refund for the difference\.)(?=\r?$)"
)
_TAX_NAME = re.compile(r"taxes_to_pay(?:_for_[a-z][a-z0-9_]*_comparison)?")
_TAX_DESCRIPTION = re.compile(
    r"The amount of taxes the claimant has to pay for the [A-Za-z0-9 -]+ credit comparison, "
    r"explicitly supplied; not a computed return settlement\."
)


def bind_credit_excess(
    form: BoundFormStructure,
    rule: Mapping[str, Any],
    fact: Mapping[str, Any],
    *,
    credit_owner: str,
    return_field: str,
    entity: str,
    unit: str,
) -> CreditExcessRequirement | None:
    """Bind a clause to an already certified terminal row, never an input alias."""
    if not form.bound(form.source, form.citation):
        return None
    clauses = tuple(_CLAUSE.finditer(form.source))
    if len(clauses) != 1:
        return None
    clause = clauses[0]
    before, after = form.source[: clause.start()], form.source[clause.end() :]
    if (
        (before and not before.endswith(("\n\n", "\r\n\r\n")))
        or (after and not after.startswith(("\n\n", "\r\n\r\n")))
        or before.rstrip().endswith(":")
        or re.match(
            r"(?:unless|except|provided|subject\s+to|but|however)\b",
            after.lstrip(),
            re.I,
        )
        or not any(
            a <= clause.start() and clause.end() <= b for a, b in form.unowned_spans
        )
        or not _proofs(rule, form, clause.group("excess"))
        or entity != "Person"
        or unit != "CAD"
    ):
        return None
    versions = rule.get("versions")
    frame = (f"{form.header.year}-01-01", f"{form.header.year}-12-31")
    if (
        set(rule)
        - {
            "name",
            "kind",
            "dtype",
            "entity",
            "period",
            "unit",
            "source",
            "description",
            "versions",
            "metadata",
        }
        or any(
            rule.get(k) != v
            for k, v in {
                "kind": "derived",
                "dtype": "Money",
                "entity": entity,
                "unit": unit,
                "period": "Year",
            }.items()
        )
        or rule.get("source") != clause.group("excess")
        or not isinstance(versions, list)
        or len(versions) != 1
        or not isinstance(versions[0], dict)
        or set(versions[0]) != {"formula", "effective_from", "effective_to"}
        or tuple(versions[0][k] for k in ("effective_from", "effective_to")) != frame
    ):
        return None
    parsed = parse_credit_excess(versions[0]["formula"])
    if (
        parsed is None
        or parsed.credit != credit_owner
        or parsed.taxes != fact.get("name")
        or not _TAX_NAME.fullmatch(parsed.taxes)
        or set(fact) != {"name", "dtype", "entity", "period", "unit", "description"}
        or any(
            fact.get(k) != v
            for k, v in {
                "dtype": "Money",
                "entity": entity,
                "unit": unit,
                "period": "Year",
            }.items()
        )
        or not isinstance(fact.get("description"), str)
        or not _TAX_DESCRIPTION.fullmatch(fact["description"])
    ):
        return None
    return CreditExcessRequirement(
        rule["name"],
        parsed.taxes,
        credit_owner,
        return_field,
        clause.span("excess"),
        clause.group("excess"),
        entity,
        unit,
        frame,
        parsed.tree,
    )


def compiled_credit_excess(requirement: CreditExcessRequirement) -> dict[str, Any]:
    credit = {"kind": "derived", "name": requirement.credit_owner}
    taxes = {"kind": "input", "name": requirement.input_name}
    return {
        "kind": "if",
        "condition": {"kind": "comparison", "left": credit, "op": "gt", "right": taxes},
        "then_expr": {"kind": "sub", "left": credit, "right": taxes},
        "else_expr": {"kind": "literal", "value": {"kind": "decimal", "value": "0"}},
    }


def parse_credit_excess(formula: object) -> ExcessExpression | None:
    """Accept one strict comparison/subtraction with zero otherwise.

    RuleSpec allows its else clause on the same line. Only that clause separator
    is normalized before parsing; row arithmetic parsing is unaffected.
    """
    if not isinstance(formula, str) or len(formula) > 2048 or "#" in formula:
        return None
    source = re.sub(r"[ \t]+else[ \t]*:", "\nelse:", formula.strip())
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, ValueError, RecursionError):
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.If):
        return None
    branch = module.body[0]
    predicate = branch.test
    if (
        not isinstance(predicate, ast.Compare)
        or len(predicate.ops) != 1
        or not isinstance(predicate.ops[0], ast.Gt)
        or not isinstance(predicate.left, ast.Name)
        or len(predicate.comparators) != 1
        or not isinstance(predicate.comparators[0], ast.Name)
        or len(branch.body) != 1
        or len(branch.orelse) != 1
        or not isinstance(branch.body[0], ast.Expr)
        or not isinstance(branch.orelse[0], ast.Expr)
    ):
        return None
    positive, otherwise = branch.body[0].value, branch.orelse[0].value
    credit, taxes = predicate.left.id, predicate.comparators[0].id
    if (
        credit == taxes
        or not isinstance(positive, ast.BinOp)
        or not isinstance(positive.op, ast.Sub)
        or not isinstance(positive.left, ast.Name)
        or not isinstance(positive.right, ast.Name)
        or (positive.left.id, positive.right.id) != (credit, taxes)
        or not isinstance(otherwise, ast.Constant)
        or type(otherwise.value) not in (int, float)
        or otherwise.value != 0
    ):
        return None
    return ExcessExpression(credit, taxes, ast.dump(branch))

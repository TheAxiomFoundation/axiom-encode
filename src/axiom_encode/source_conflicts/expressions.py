"""A small, safe expression form for link constraints.

A constraint is one comparison of two linear terms over the link's facts::

    monthly_income >= work_hours * 7.25
    work_hours + community_service_hours <= 744
    agency_lacks_hours_documentation == true

Grammar::

    constraint := side CMP side
    CMP        := ">=" | "<=" | ">" | "<" | "==" | "!="
    side       := term (("+" | "-") term)*
    term       := ["-"] atom ["*" atom]
    atom       := NUMBER | FACT | "true" | "false"

A product may hold at most one fact, so every side is linear. ``true`` and
``false`` and bool facts may only stand alone on a side of ``==`` or ``!=``.
Nothing is evaluated by Python: the text is tokenized, parsed into this
module's own tree and evaluated with ``Decimal`` arithmetic.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

_TOKEN = re.compile(
    r"\s*(?:(?P<number>\d+(?:\.\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>>=|<=|==|!=|>|<|\+|-|\*))"
)
_COMPARATORS = (">=", "<=", "==", "!=", ">", "<")
_ORDERING = (">=", "<=", ">", "<")


class ExpressionError(ValueError):
    """The constraint text is outside the supported expression form."""


@dataclass(frozen=True)
class Linear:
    """``sum(coefficient * fact) + constant``."""

    coefficients: tuple[tuple[str, Decimal], ...]
    constant: Decimal

    def value(self, case: Mapping[str, object]) -> Decimal:
        total = self.constant
        for fact, coefficient in self.coefficients:
            total += coefficient * Decimal(str(case[fact]))
        return total

    @property
    def single_fact(self) -> str | None:
        if self.constant == 0 and len(self.coefficients) == 1:
            fact, coefficient = self.coefficients[0]
            if coefficient == 1:
                return fact
        return None


@dataclass(frozen=True)
class BoolLiteral:
    value: bool


@dataclass(frozen=True)
class Constraint:
    """One parsed constraint, checked against the kinds of the facts."""

    text: str
    left: Linear | BoolLiteral
    op: str
    right: Linear | BoolLiteral
    boolean: bool

    @property
    def facts(self) -> tuple[str, ...]:
        names: list[str] = []
        for side in (self.left, self.right):
            if isinstance(side, Linear):
                names.extend(fact for fact, _ in side.coefficients)
        return tuple(dict.fromkeys(names))

    def holds(self, case: Mapping[str, object]) -> bool:
        if self.boolean:
            left, right = (
                side.value if isinstance(side, BoolLiteral) else case[side.single_fact]
                for side in (self.left, self.right)
            )
            return (left == right) == (self.op == "==")
        left, right = self.left.value(case), self.right.value(case)
        return {
            ">=": left >= right,
            "<=": left <= right,
            ">": left > right,
            "<": left < right,
            "==": left == right,
            "!=": left != right,
        }[self.op]


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    position = 0
    stripped = text.rstrip()
    while position < len(stripped):
        match = _TOKEN.match(stripped, position)
        if match is None:
            raise ExpressionError(
                f"unsupported text at column {position + 1} of constraint {text!r}"
            )
        kind = match.lastgroup
        tokens.append((kind, match.group(kind)))
        position = match.end()
    return tokens


def _parse_side(tokens: list[tuple[str, str]], text: str) -> Linear | BoolLiteral:
    if not tokens:
        raise ExpressionError(f"a side of constraint {text!r} is empty")
    if len(tokens) == 1 and tokens[0] in (("name", "true"), ("name", "false")):
        return BoolLiteral(tokens[0][1] == "true")
    coefficients: dict[str, Decimal] = {}
    constant = Decimal(0)
    index = 0
    first = True
    while index < len(tokens):
        sign = Decimal(1)
        kind, value = tokens[index]
        if kind == "op" and value in "+-":
            if value == "-":
                sign = Decimal(-1)
            elif first:
                raise ExpressionError(f"a term of constraint {text!r} starts with '+'")
            index += 1
        elif not first:
            raise ExpressionError(f"constraint {text!r} lacks '+' or '-' between terms")
        first = False
        atoms: list[tuple[str, str]] = []
        while True:
            if index >= len(tokens) or tokens[index][0] == "op":
                raise ExpressionError(f"constraint {text!r} lacks a number or a fact")
            atoms.append(tokens[index])
            index += 1
            if index < len(tokens) and tokens[index] == ("op", "*"):
                index += 1
                continue
            break
        if len(atoms) > 2:
            raise ExpressionError(
                f"a term of constraint {text!r} multiplies more than two operands"
            )
        factor = sign
        fact: str | None = None
        for atom_kind, atom in atoms:
            if atom_kind == "number":
                factor *= Decimal(atom)
            elif atom in ("true", "false"):
                raise ExpressionError(
                    f"'{atom}' must stand alone on a side of constraint {text!r}"
                )
            elif fact is not None:
                raise ExpressionError(
                    f"constraint {text!r} multiplies two facts; only linear terms are supported"
                )
            else:
                fact = atom
        if fact is None:
            constant += factor
        else:
            coefficients[fact] = coefficients.get(fact, Decimal(0)) + factor
    return Linear(tuple(coefficients.items()), constant)


def parse_constraint(text: str, fact_kinds: Mapping[str, str]) -> Constraint:
    """Parse ``text`` and check every name against the link's facts."""

    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("a constraint rule must be a non-empty string")
    tokens = _tokenize(text)
    comparators = [
        index
        for index, token in enumerate(tokens)
        if token[0] == "op" and token[1] in _COMPARATORS
    ]
    if len(comparators) != 1:
        raise ExpressionError(
            f"constraint {text!r} must hold exactly one comparison operator"
        )
    split = comparators[0]
    op = tokens[split][1]
    left = _parse_side(tokens[:split], text)
    right = _parse_side(tokens[split + 1 :], text)
    for side in (left, right):
        if isinstance(side, Linear):
            for fact, _ in side.coefficients:
                if fact not in fact_kinds:
                    raise ExpressionError(
                        f"constraint {text!r} names '{fact}', which is not a fact of the link"
                    )

    def is_bool(side: Linear | BoolLiteral) -> bool:
        if isinstance(side, BoolLiteral):
            return True
        return side.single_fact is not None and fact_kinds[side.single_fact] == "bool"

    def uses_bool_fact(side: Linear | BoolLiteral) -> bool:
        return isinstance(side, Linear) and any(
            fact_kinds[fact] == "bool" for fact, _ in side.coefficients
        )

    boolean = is_bool(left) and is_bool(right)
    if boolean:
        if op in _ORDERING:
            raise ExpressionError(
                f"constraint {text!r} orders bool values; use '==' or '!='"
            )
    elif any(
        isinstance(side, BoolLiteral) or uses_bool_fact(side) for side in (left, right)
    ):
        raise ExpressionError(f"constraint {text!r} mixes bool and numeric operands")
    if not any(
        isinstance(side, Linear) and side.coefficients for side in (left, right)
    ):
        raise ExpressionError(f"constraint {text!r} names no fact")
    return Constraint(text=text, left=left, op=op, right=right, boolean=boolean)

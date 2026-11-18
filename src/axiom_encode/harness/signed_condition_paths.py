"""Bounded, signed Boolean paths for source-condition corroboration.

Unlike a selector-name inventory, a false prefix of an OR arm cannot provide
positive evidence. This is a closed AST reader, not a general evaluator.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True, order=True)
class SignedFact:
    name: str
    positive: bool
    relation: str = "boolean"
    threshold: int | None = None


@dataclass(frozen=True)
class SignedPath:
    facts: frozenset[SignedFact]
    start: str | None
    end: str | None


def signed_granting_paths(
    expression: ast.expr,
    *,
    start: str | None,
    end: str | None,
    resolve: Callable[
        [str, str | None, str | None],
        tuple[tuple[ast.expr, str | None, str | None], ...] | None,
    ],
    intersect: Callable[
        [str | None, str | None, str | None, str | None],
        tuple[str | None, str | None] | None,
    ],
    limit: int,
    depth_limit: int,
    numeric_names: frozenset[str],
) -> tuple[SignedPath, ...] | None:
    """Enumerate true paths, preserving negative facts and temporal overlap.

    ``resolve`` returns None for an opaque Boolean terminal, or all applicable
    version expressions for a local derived name. Unknown references must be
    rejected by the caller before treating them as terminals. No subset
    minimization is valid here: larger paths may have different polarities.
    """
    work = 0

    def bounded(paths):
        unique = tuple(dict.fromkeys(paths))
        if len(unique) > limit:
            raise ValueError("signed path cap")
        return unique

    def product(left, right):
        result = []
        if len(left) * len(right) > limit:
            raise ValueError("signed path product cap")
        for a in left:
            for b in right:
                interval = intersect(a.start, a.end, b.start, b.end)
                if interval is None:
                    continue
                facts = a.facts | b.facts
                if any(
                    SignedFact(f.name, not f.positive, f.relation, f.threshold) in facts
                    for f in facts
                ):
                    continue
                result.append(SignedPath(facts, *interval))
        return bounded(result)

    def paths(node, wanted, lo, hi, stack, depth):
        nonlocal work
        work += 1
        if depth > depth_limit or work > limit * limit or len(stack) > depth_limit:
            raise ValueError("signed path traversal cap")
        identity = (SignedPath(frozenset(), lo, hi),)
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return identity if node.value == wanted else ()
        if isinstance(node, ast.Name):
            if node.id in stack:
                raise ValueError("cyclic signed path")
            versions = resolve(node.id, lo, hi)
            if versions is None:
                return (SignedPath(frozenset({SignedFact(node.id, wanted)}), lo, hi),)
            if not versions:
                raise ValueError("missing signed path version")
            if len(versions) > limit:
                raise ValueError("signed path version cap")
            result = []
            covered = []
            for child, child_start, child_end in versions:
                interval = intersect(lo, hi, child_start, child_end)
                if interval is not None:
                    covered.append(interval)
                    result.extend(
                        paths(child, wanted, *interval, stack | {node.id}, depth + 1)
                    )
            covered.sort(key=lambda value: value[0] or "")
            if not covered or covered[0][0] != lo or covered[-1][1] != hi:
                raise ValueError("unresolved signed path interval")
            for previous, following in zip(covered, covered[1:]):
                if (
                    previous[1] is None
                    or following[0] is None
                    or date.fromisoformat(previous[1]) + timedelta(days=1)
                    != date.fromisoformat(following[0])
                ):
                    raise ValueError("signed path version gap or overlap")
            return bounded(result)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return paths(node.operand, not wanted, lo, hi, stack, depth + 1)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"holds", "not_holds"}
            and len(node.args) == 1
            and not node.keywords
        ):
            return paths(
                node.args[0],
                wanted == (node.func.id == "holds"),
                lo,
                hi,
                stack,
                depth + 1,
            )
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == len(node.comparators) == 1
        ):
            left, right = node.left, node.comparators[0]
            relations = {
                ast.Gt: "gt",
                ast.GtE: "ge",
                ast.Lt: "lt",
                ast.LtE: "le",
                ast.Eq: "eq",
                ast.NotEq: "ne",
            }
            relation = relations.get(type(node.ops[0]))
            if (
                isinstance(left, ast.Name)
                and left.id in numeric_names
                and isinstance(right, ast.Constant)
                and type(right.value) is int
                and relation is not None
                and resolve(left.id, lo, hi) is None
            ):
                return (
                    SignedPath(
                        frozenset({SignedFact(left.id, wanted, relation, right.value)}),
                        lo,
                        hi,
                    ),
                )
            raise ValueError("unread signed comparison")
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            # AND true / OR false require every operand. Other outcomes retain
            # each short-circuit branch and its false/true prefix respectively.
            all_operands = wanted == isinstance(node.op, ast.And)
            prefix = identity
            alternatives = []
            for child in node.values:
                if all_operands:
                    prefix = product(
                        prefix, paths(child, wanted, lo, hi, stack, depth + 1)
                    )
                else:
                    alternatives.extend(
                        product(prefix, paths(child, wanted, lo, hi, stack, depth + 1))
                    )
                    bounded(alternatives)
                    prefix = product(
                        prefix, paths(child, not wanted, lo, hi, stack, depth + 1)
                    )
            return prefix if all_operands else bounded(alternatives)
        raise ValueError("unread signed Boolean expression")

    try:
        return paths(expression, True, start, end, frozenset(), 0)
    except (ValueError, RecursionError, OverflowError):
        return None

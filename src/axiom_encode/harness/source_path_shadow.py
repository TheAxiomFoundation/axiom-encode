"""Bounded structural path extraction, not independent source certification.

Source contracts are explicit reviewed raw-span records, and unproved bindings
stay visible. Formula paths come from the real candidate, not an asserted path
table. The separate source_path_evidence layer validates selected proofs and
executions before acceptance consumers can use its certified evidence. Structural
matches here alone cannot discharge obligations or mutate annual coverage.
"""

from __future__ import annotations

import ast
import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Sequence

from axiom_encode.harness import source_completeness as sc

_MAX_NODES = 128
_MAX_PATHS = 64
_MAX_PARTITIONS = 64


@dataclass(frozen=True)
class RawSpan:
    start: int
    end: int
    text: str

    def valid(self, source: str) -> bool:
        return (
            0 <= self.start < self.end <= len(source)
            and source[self.start : self.end] == self.text
        )


@dataclass(frozen=True)
class Predicate:
    name: str
    entity: str
    period: str
    operator: str
    value: str | Decimal | bool
    kind: str


@dataclass(frozen=True)
class InputContract:
    """Independent reviewed source-role contract, never inferred from candidate.

    Raw source anchors are retained, but this record does not claim that its
    semantic mapping has been certified by an automated source parser.
    """

    name: str
    dtype: str
    entity: str
    period: str
    unit: str | None
    source_spans: tuple[RawSpan, ...]


@dataclass(frozen=True)
class SourceObligation:
    """Diagnostic source binding, with explicit automation/provenance limits."""

    identity: str
    owner: str
    source_sha256: str
    citation: str
    spans: tuple[RawSpan, ...]
    predicates: tuple[Predicate, ...]
    expected_expression: str
    entity: str
    start: str
    end: str
    unresolved_bindings: tuple[str, ...] = ()
    binding_provenance: str = "reviewed diagnostic mapping; not automated source proof"
    input_contracts: tuple[InputContract, ...] = ()
    output_dtype: str = "Money"
    output_unit: str | None = "CAD"


@dataclass(frozen=True)
class SelectedDependency:
    name: str
    version: int
    entity: str
    period: str
    start: str
    end: str
    formula: str


@dataclass(frozen=True)
class FormulaPath:
    start: str
    end: str
    predicates: tuple[Predicate, ...]
    result: str | None
    dependencies: tuple[SelectedDependency, ...]
    unresolved: tuple[str, ...]


@dataclass(frozen=True)
class ShadowResult:
    source_sha256: str
    owner: str
    paths: tuple[FormulaPath, ...]
    rows: tuple[dict[str, Any], ...]
    unresolved: tuple[str, ...]
    # An invariant, not an optional caller switch.
    acceptance_activated: bool = False


def _expression(value: Any) -> ast.expr | None:
    if isinstance(value, bool):
        return ast.Constant(value=value)
    if isinstance(value, (int, float)):
        return ast.Constant(value=value)
    return sc._parse_formula_expression(str(value))


def _date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _index(items: Any) -> dict[str, dict[str, Any]] | None:
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            return None
        if item["name"] in result:
            return None
        result[item["name"]] = item
    return result


@dataclass(frozen=True)
class _ValueType:
    kind: str
    unit: str | None = None
    literal: bool = False


def _declared_type(item: Mapping[str, Any]) -> _ValueType | None:
    dtype, unit = item.get("dtype"), item.get("unit")
    if dtype == "Money":
        return (
            _ValueType("money", unit)
            if isinstance(unit, str) and re.fullmatch(r"[A-Z]{3}", unit)
            else None
        )
    if unit is not None:
        return None
    if dtype in {"Integer", "Count", "Decimal", "Rate"}:
        return _ValueType("number")
    if dtype in {"Boolean", "Judgment"}:
        return _ValueType("boolean")
    if dtype == "Text":
        return _ValueType("text")
    return None


def _assignable(actual: _ValueType, declared: _ValueType) -> bool:
    return (actual.kind == declared.kind and actual.unit == declared.unit) or (
        actual.kind == "number" and actual.literal and declared.kind == "money"
    )


def _common_type(a: _ValueType, b: _ValueType) -> _ValueType | None:
    if a.kind == b.kind and a.unit == b.unit:
        return _ValueType(a.kind, a.unit, a.literal and b.literal)
    if a.kind == "money" and b.kind == "number" and b.literal:
        return _ValueType("money", a.unit)
    if b.kind == "money" and a.kind == "number" and a.literal:
        return _ValueType("money", b.unit)
    return None


def _arithmetic_type(
    a: _ValueType, b: _ValueType, op: ast.operator
) -> _ValueType | None:
    if a.kind not in {"number", "money"} or b.kind not in {"number", "money"}:
        return None
    if isinstance(op, (ast.Add, ast.Sub)):
        return _common_type(a, b)
    if isinstance(op, ast.Mult):
        if a.kind == b.kind == "number":
            return _ValueType("number", literal=a.literal and b.literal)
        if a.kind == "money" and b.kind == "number":
            return _ValueType("money", a.unit)
        if b.kind == "money" and a.kind == "number":
            return _ValueType("money", b.unit)
    if isinstance(op, ast.Div):
        if a.kind == b.kind == "money" and a.unit == b.unit:
            return _ValueType("number")
        if b.kind == "number":
            return _ValueType(a.kind, a.unit, a.literal and b.literal)
    return None


class _Resolver:
    def __init__(
        self,
        payload: Mapping[str, Any],
        owner: str,
        start: str,
        end: str,
        source: str,
        citation: str,
    ):
        self.rules = _index(payload.get("rules"))
        self.inputs = _index(payload.get("inputs"))
        self.source, self.citation = source, citation
        self.owner = owner
        self.start, self.end = start, end
        self.visited: set[str] = set()
        self.dependencies: dict[str, SelectedDependency] = {}
        self.errors: set[str] = set()
        self.types: dict[ast.AST, _ValueType] = {}
        rule = (self.rules or {}).get(owner, {})
        self.entity = rule.get("entity")
        self.period = rule.get("period")

    def declaration(self, name: str) -> dict[str, Any] | None:
        item = (self.inputs or {}).get(name)
        if (
            item is None
            or item.get("entity") != self.entity
            or item.get("period") != self.period
        ):
            self.errors.add("input identity/entity/period unresolved: " + name)
            return None
        return item

    def select(self, name: str) -> tuple[dict[str, Any], str] | None:
        rule = (self.rules or {}).get(name)
        if rule is None:
            self.errors.add("unknown or imported dependency: " + name)
            return None
        if rule.get("kind") != "parameter" and (
            rule.get("entity") != self.entity or rule.get("period") != self.period
        ):
            self.errors.add("dependency entity/period mismatch: " + name)
            return None
        intervals = [
            (i, str(f), a, b)
            for i, f, a, b in sc._effective_formula_version_intervals(rule)
            if a is not None and b is not None and a <= self.start and b >= self.end
        ]
        if len(intervals) != 1:
            self.errors.add("no unique selected interval: " + name)
            return None
        index, formula, a, b = intervals[0]
        atoms = tuple(sc._rule_source_excerpt_atoms(rule))
        for path in (
            f"versions[{index}].formula",
            f"versions[{index}].effective_from",
            f"versions[{index}].effective_to",
        ):
            if not any(
                re.sub(r"\s+", "", atom_path) == path
                and citation.strip("/").casefold()
                == self.citation.strip("/").casefold()
                and sc._collapse_text(excerpt) in sc._collapse_text(self.source)
                and excerpt.strip()
                for atom_path, citation, excerpt in atoms
            ):
                self.errors.add("unbound selected proof " + name + ":" + path)
        if rule.get("kind") == "parameter":
            parsed = _expression(formula)
            if (
                not isinstance(parsed, ast.Constant)
                or type(parsed.value) not in {int, float}
                or rule.get("dtype")
                not in {"Count", "Integer", "Money", "Rate", "Decimal"}
            ):
                self.errors.add("nonliteral/unsupported parameter: " + name)
                return None
        dep = SelectedDependency(
            name,
            index,
            str(rule.get("entity") or ""),
            str(rule.get("period") or ""),
            a,
            b,
            formula,
        )
        self.dependencies[name] = dep
        if len(self.dependencies) > _MAX_NODES:
            self.errors.add("dependency node limit")
            return None
        return rule, formula

    def mark(self, node: ast.expr, value_type: _ValueType) -> ast.expr:
        self.types[node] = value_type
        return node

    def expand(self, node: ast.expr, stack: tuple[str, ...] = ()) -> ast.expr | None:
        if len(stack) > _MAX_NODES:
            self.errors.add("dependency depth limit")
            return None
        if isinstance(node, ast.Name):
            if node.id in (self.inputs or {}):
                declaration = self.declaration(node.id)
                value_type = _declared_type(declaration or {})
                if declaration is None or value_type is None:
                    self.errors.add("unsupported input type/unit: " + node.id)
                    return None
                return self.mark(node, value_type)
            if node.id in stack:
                self.errors.add("cyclic dependency: " + node.id)
                return None
            selected = self.select(node.id)
            if selected is None:
                return None
            rule, formula = selected
            if rule.get("kind") not in {"derived", "parameter"}:
                self.errors.add("unsupported dependency kind: " + node.id)
                return None
            parsed = _expression(formula)
            if parsed is None:
                self.errors.add("conditional/unsupported result dependency: " + node.id)
                return None
            result = self.expand(parsed, (*stack, node.id))
            declared = _declared_type(rule)
            if (
                result is None
                or declared is None
                or not _assignable(self.types[result], declared)
            ):
                self.errors.add("dependency expression dtype/unit mismatch: " + node.id)
                return None
            return self.mark(result, declared)
        if isinstance(node, ast.Constant) and type(node.value) in {
            bool,
            int,
            float,
            str,
        }:
            kind = (
                "boolean"
                if type(node.value) is bool
                else "text"
                if isinstance(node.value, str)
                else "number"
            )
            return self.mark(node, _ValueType(kind, None, kind == "number"))
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left, right = self.expand(node.left, stack), self.expand(node.right, stack)
            if left is not None and right is not None:
                value_type = _arithmetic_type(
                    self.types[left], self.types[right], node.op
                )
                if value_type is not None:
                    return self.mark(
                        ast.BinOp(left=left, op=node.op, right=right), value_type
                    )
            self.errors.add("arithmetic dtype/unit mismatch")
            return None
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = (
                self.expand(node.left, stack),
                self.expand(node.comparators[0], stack),
            )
            if (
                left is not None
                and right is not None
                and _common_type(self.types[left], self.types[right]) is not None
            ):
                return self.mark(
                    ast.Compare(left=left, ops=node.ops, comparators=[right]),
                    _ValueType("boolean"),
                )
            self.errors.add("comparison dtype/unit mismatch")
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            operand = self.expand(node.operand, stack)
            if operand is not None:
                value_type = self.types[operand]
                if (isinstance(node.op, ast.Not) and value_type.kind == "boolean") or (
                    isinstance(node.op, ast.USub)
                    and value_type.kind in {"number", "money"}
                ):
                    return self.mark(
                        ast.UnaryOp(op=node.op, operand=operand), value_type
                    )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"min", "max"}
            and not node.keywords
        ):
            args = [self.expand(x, stack) for x in node.args]
            if args and all(x is not None for x in args):
                value_type = self.types[args[0]]
                for arg in args[1:]:
                    value_type = (
                        _common_type(value_type, self.types[arg])
                        if value_type is not None
                        else None
                    )
                if value_type is not None and value_type.kind in {"number", "money"}:
                    return self.mark(
                        ast.Call(func=node.func, args=args, keywords=[]), value_type
                    )
        self.errors.add("unsupported expression or dtype/unit: " + type(node).__name__)
        return None

    def predicate(self, text: str, truth: bool) -> Predicate | None:
        node = _expression(text)
        if node is None:
            self.errors.add("unparsed selector")
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.predicate(ast.unparse(node.operand), not truth)
        if isinstance(node, ast.Name) and node.id in (self.inputs or {}):
            item = self.declaration(node.id)
            if item is not None and item.get("dtype") == "Boolean":
                return Predicate(
                    node.id, str(self.entity), str(self.period), "==", truth, "boolean"
                )
            self.errors.add("not a direct declared Boolean: " + node.id)
            return None
        if isinstance(node, ast.Name) and node.id in (self.rules or {}):
            selected = self.select(node.id)
            dep = self.dependencies.get(node.id)
            if (
                selected is None
                or dep is None
                or not _residence_selector_is_source_bound(
                    selected[0],
                    dep.version,
                    self.source,
                    self.citation,
                    self.inputs or {},
                )
            ):
                self.errors.add("unproved derived selector identity: " + node.id)
                return None
        expanded = self.expand(node)
        ops = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        if (
            not isinstance(expanded, ast.Compare)
            or len(expanded.ops) != 1
            or type(expanded.ops[0]) not in ops
        ):
            self.errors.add("unsupported signed selector")
            return None
        left, right = expanded.left, expanded.comparators[0]
        if not isinstance(left, ast.Name) or not isinstance(right, ast.Constant):
            self.errors.add(
                "selector is not input compared with selected literal constant"
            )
            return None
        declaration = self.declaration(left.id)
        if declaration is None:
            return None
        value = right.value
        kind = "text" if isinstance(value, str) else "numeric"
        if kind == "text":
            if declaration.get("dtype") != "Text":
                self.errors.add("text declaration mismatch")
                return None
        elif type(value) not in {int, float} or declaration.get("dtype") not in {
            "Integer",
            "Count",
            "Money",
            "Decimal",
            "Rate",
        }:
            self.errors.add("numeric declaration mismatch")
            return None
        else:
            value = Decimal(str(value))
        op = ops[type(expanded.ops[0])]
        if not truth:
            op = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[
                op
            ]
        return Predicate(left.id, str(self.entity), str(self.period), op, value, kind)


def _raw_paths(
    formula: str, depth: int = 0
) -> list[tuple[tuple[tuple[str, bool], ...], str]] | None:
    if depth > 32:
        return None
    node = sc._first_formula_branch_node(formula.strip())
    if node is None:
        return [((), formula)] if _expression(formula) is not None else None
    if (
        node.kind != "if"
        or node.start != 0
        or formula.strip()[node.end :].strip()
        or len(node.selectors) != 1
        or len(node.choices) != 2
    ):
        return None
    paths = []
    for truth, choice in zip((True, False), node.choices, strict=True):
        children = _raw_paths(choice, depth + 1)
        if children is None:
            return None
        paths.extend(
            (((node.selectors[0], truth), *p), result) for p, result in children
        )
    return paths if len(paths) <= _MAX_PATHS else None


def _contradiction(predicates: Sequence[Predicate]) -> str | None:
    keys = {(p.name, p.entity, p.period, p.kind) for p in predicates}
    for key in keys:
        group = [p for p in predicates if (p.name, p.entity, p.period, p.kind) == key]
        if key[-1] in {"text", "boolean"}:
            equal = {p.value for p in group if p.operator == "=="}
            excluded = {p.value for p in group if p.operator == "!="}
            if len(equal) > 1 or equal & excluded:
                return "same declared identity contradiction"
        if key[-1] == "numeric":
            lows = [
                (p.value, p.operator in {">=", "=="})
                for p in group
                if p.operator in {">", ">=", "=="}
            ]
            highs = [
                (p.value, p.operator in {"<=", "=="})
                for p in group
                if p.operator in {"<", "<=", "=="}
            ]
            if lows and highs:
                lo = max(lows, key=lambda x: (x[0], not x[1]))
                hi = min(highs, key=lambda x: (x[0], x[1]))
                if lo[0] > hi[0] or lo[0] == hi[0] and not (lo[1] and hi[1]):
                    return "empty numeric interval"
    return None


def _entails(path: Sequence[Predicate], expected: Predicate) -> bool:
    opposite = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[
        expected.operator
    ]
    negated = Predicate(
        expected.name,
        expected.entity,
        expected.period,
        opposite,
        expected.value,
        expected.kind,
    )
    # Numeric != is not an interval; equality entailment is deliberately exact.
    if expected.operator == "==" and expected.kind == "numeric":
        return expected in path
    return _contradiction((*path, negated)) is not None


def _obligation_contract_errors(
    obligation: SourceObligation, payload: Mapping[str, Any], source: str
) -> tuple[str, ...]:
    rules = _index(payload.get("rules")) or {}
    inputs = _index(payload.get("inputs")) or {}
    owner = rules.get(obligation.owner, {})
    errors: set[str] = set()
    if owner.get("entity") != obligation.entity or owner.get("period") != "Year":
        errors.add("source owner entity/period mismatch")
    if (
        owner.get("dtype") != obligation.output_dtype
        or owner.get("unit") != obligation.output_unit
    ):
        errors.add("source owner dtype/unit mismatch")
    contracts = {c.name: c for c in obligation.input_contracts}
    if len(contracts) != len(obligation.input_contracts):
        errors.add("duplicate independent input contract")
    expected = _expression(obligation.expected_expression)
    needed = {p.name for p in obligation.predicates}
    if expected is not None:
        needed.update(
            n.id
            for n in ast.walk(expected)
            if isinstance(n, ast.Name) and n.id not in {"min", "max"}
        )
    for name in needed:
        c = contracts.get(name)
        if c is None:
            errors.add("missing independent source operand contract: " + name)
            continue
        if not c.source_spans or not all(span.valid(source) for span in c.source_spans):
            errors.add("unbound operand source span: " + name)
        if c.entity != obligation.entity or c.period != "Year":
            errors.add("independent operand entity/period mismatch: " + name)
        item = inputs.get(name)
        if item is None or any(
            item.get(field) != value
            for field, value in (
                ("dtype", c.dtype),
                ("entity", c.entity),
                ("period", c.period),
                ("unit", c.unit),
            )
        ):
            errors.add("candidate operand type/unit/identity mismatch: " + name)
    for predicate in obligation.predicates:
        c = contracts.get(predicate.name)
        if c is None or predicate.entity != c.entity or predicate.period != c.period:
            errors.add("source predicate identity mismatch: " + predicate.name)
        elif (
            (predicate.kind == "text" and c.dtype != "Text")
            or (predicate.kind == "boolean" and c.dtype != "Boolean")
            or (
                predicate.kind == "numeric"
                and c.dtype not in {"Money", "Integer", "Count", "Rate", "Decimal"}
            )
        ):
            errors.add("source predicate type mismatch: " + predicate.name)
    return tuple(sorted(errors))


def inspect_source_paths(
    payload: Mapping[str, Any],
    *,
    source_text: str,
    owner: str,
    obligations: Sequence[SourceObligation],
) -> ShadowResult:
    """Extract actual paths and compare reviewed mappings without granting proof.

    Caller mappings remain visibly unautomated. Even a fully matching table is
    NOT an accepted source witness, numeric binding, or annual representation.
    """
    digest = hashlib.sha256(source_text.encode()).hexdigest()
    rules, inputs = _index(payload.get("rules")), _index(payload.get("inputs"))
    if (
        rules is None
        or inputs is None
        or owner not in rules
        or set(rules) & set(inputs)
    ):
        return ShadowResult(
            digest, owner, (), (), ("invalid/duplicate candidate declarations",)
        )
    selected_obligations = [o for o in obligations if o.owner == owner]
    if not selected_obligations:
        return ShadowResult(digest, owner, (), (), ("no owned source obligations",))
    citations = {o.citation.strip("/").casefold() for o in selected_obligations}
    verification = (payload.get("module") or {}).get("source_verification") or {}
    if (
        len(citations) != 1
        or str(verification.get("corpus_citation_path") or "").strip("/").casefold()
        not in citations
        or verification.get("source_sha256") != digest
    ):
        return ShadowResult(
            digest,
            owner,
            (),
            (),
            ("candidate/obligation citation or source identity mismatch",),
        )
    if any(
        o.source_sha256 != digest
        or not o.spans
        or not all(s.valid(source_text) for s in o.spans)
        for o in selected_obligations
    ):
        return ShadowResult(digest, owner, (), (), ("source hash/span mismatch",))
    try:
        starts = {_date(o.start) for o in selected_obligations}
        ends = {_date(o.end) + dt.timedelta(days=1) for o in selected_obligations}
        boundaries = starts | ends
        # Include all candidate dependency boundaries, not merely owner dates.
        # A later reachability optimization must preserve this sound partition.
        for rule in rules.values():
            for _, _, a, b in sc._effective_formula_version_intervals(rule):
                if a is not None and b is not None:
                    boundaries.update((_date(a), _date(b) + dt.timedelta(days=1)))
        lo, hi = min(starts), max(ends)
        points = sorted(x for x in boundaries if lo <= x <= hi)
    except (ValueError, TypeError, OverflowError):
        return ShadowResult(digest, owner, (), (), ("invalid temporal interval",))
    if len(points) > _MAX_PARTITIONS:
        return ShadowResult(digest, owner, (), (), ("temporal partition limit",))
    paths = []
    errors = set()
    for a, b in zip(points, points[1:]):
        start, end = a.isoformat(), (b - dt.timedelta(days=1)).isoformat()
        resolver = _Resolver(
            payload, owner, start, end, source_text, selected_obligations[0].citation
        )
        selected = resolver.select(owner)
        raw = _raw_paths(selected[1]) if selected is not None else None
        if raw is None:
            errors.update(resolver.errors or {"unparsed owner paths"})
            continue
        for selectors, leaf in raw:
            r = _Resolver(
                payload,
                owner,
                start,
                end,
                source_text,
                selected_obligations[0].citation,
            )
            r.select(owner)
            predicates = tuple(
                p
                for text, truth in selectors
                if (p := r.predicate(text, truth)) is not None
            )
            node = _expression(leaf)
            expanded = r.expand(node) if node is not None else None
            declared_owner = _declared_type((r.rules or {}).get(owner, {}))
            if expanded is None:
                r.errors.add("unresolved result")
            elif declared_owner is None or not _assignable(
                r.types[expanded], declared_owner
            ):
                r.errors.add("owner result dtype/unit mismatch")
            result = (
                ast.dump(expanded, include_attributes=False)
                if expanded is not None
                else None
            )
            paths.append(
                FormulaPath(
                    start,
                    end,
                    predicates,
                    result,
                    tuple(r.dependencies.values()),
                    tuple(sorted(r.errors)),
                )
            )
    rows = []
    for obligation in selected_obligations:
        expected = _expression(obligation.expected_expression)
        expected_dump = (
            ast.dump(expected, include_attributes=False)
            if expected is not None
            else None
        )
        compatible = []
        contract_errors = _obligation_contract_errors(obligation, payload, source_text)
        errors.update(contract_errors)
        for i, path in enumerate(paths):
            if path.end < obligation.start or path.start > obligation.end:
                continue
            contradiction = (
                _contradiction((*obligation.predicates, *path.predicates))
                if not path.unresolved
                and not contract_errors
                and not obligation.unresolved_bindings
                else None
            )
            if contradiction is not None:
                rows.append(
                    {
                        "obligation": obligation.identity,
                        "path": i,
                        "status": "excluded",
                        "reason": contradiction,
                    }
                )
                continue
            compatible.append(i)
            entails = all(_entails(path.predicates, p) for p in obligation.predicates)
            matches = expected_dump is not None and path.result == expected_dump
            rows.append(
                {
                    "obligation": obligation.identity,
                    "path": i,
                    "status": "diagnostic_match"
                    if entails
                    and matches
                    and not path.unresolved
                    and not obligation.unresolved_bindings
                    and not contract_errors
                    else "unresolved",
                    "predicates_entailed": entails,
                    "independent_expression_matches": matches,
                    "operation_proof": "unresolved: separate source operation certificate required",
                    "source_binding_limits": obligation.unresolved_bindings,
                    "unresolved": (*path.unresolved, *contract_errors),
                    "acceptance": False,
                }
            )
        if not compatible:
            errors.add("no nonempty compatible path set: " + obligation.identity)
    return ShadowResult(digest, owner, tuple(paths), tuple(rows), tuple(sorted(errors)))


@dataclass(frozen=True)
class SourceFrame:
    """Closed source syntax and raw ownership, independent of candidate names."""

    kind: str
    owner: RawSpan
    body: RawSpan
    fields: tuple[tuple[str, str], ...]
    interpretation: str


def _layout_pattern(text: str) -> str:
    # Templates contain explicit named regex fields; only publisher whitespace
    # varies. Never strip punctuation, quote delimiters, operators or numbers.
    return r"\s+".join(text.strip().split())


_OTHER_FORM = _layout_pattern(r"""
If you were a resident of a province or territory other than
(?P<excluded>Ontario, Alberta, or British Columbia), enter the amount of tax
calculated before determining the provincial or territorial foreign tax credit
by using the appropriate Form (?P<form>428)\. However, if you have to pay tax to
more than one jurisdiction, complete the applicable part of Section
(?P<section>428MJ) of Form (?P<container>T[0-9]+) for the province or territory
in which you resided at the end of the year\.
""")
_ON_FORM = _layout_pattern(r"""
If you were a resident of (?P<place>Ontario), calculate this amount by entering
"0" on lines (?P<zero_a>[1-9][0-9]*) and (?P<zero_b>[1-9][0-9]*) of Form
(?P<form>ON428) and continue the calculation\. The result from line
(?P<result_a>[1-9][0-9]*) is your provincial or territorial tax otherwise payable\.
If you paid tax to more than one jurisdiction in (?P<year>20[0-9]{2}), calculate
this amount by entering "0" on lines (?P<zero_c>[1-9][0-9]*) and
(?P<zero_d>[1-9][0-9]*) in Part (?P<part>[1-9][0-9]*) of Section
(?P<section>ON428MJ) of Form (?P<container>T[0-9]+) and continue the calculation\.
The amount from line (?P<result_b>[1-9][0-9]*) is your provincial or territorial tax
otherwise payable\.
""")
_AB_FORM = _layout_pattern(r"""
If you were a resident of (?P<place>Alberta), calculate your provincial or
territorial tax otherwise payable by adding the amount from line
(?P<left_a>[1-9][0-9]*) to the amount on line (?P<right_a>[1-9][0-9]*) of Form
(?P<form>AB428) or by adding the amount from line (?P<left_b>[1-9][0-9]*) to the
amount from line (?P<right_b>[1-9][0-9]*) in Part (?P<part>[1-9][0-9]*) of Section
(?P<section>AB428MJ) of Form (?P<container>T[0-9]+)\.
""")
_BC_FORM = _layout_pattern(r"""
If you were a resident of (?P<place>British Columbia) at the end of the tax year,
your provincial or territorial tax otherwise payable is the amount of tax
excluding the provincial and territorial foreign tax credit and any British
Columbia additional tax for minimum tax purposes from Form (?P<form>BC428) or
Section (?P<section>BC428MJ) of Form (?P<container>T[0-9]+)\.
""")


def closed_result_frames(source: str) -> tuple[SourceFrame, ...]:
    """Recognize one complete note, not arbitrary adjacent If statements.

    This first bounded syntax family identifies ordinary/override and applicable
    form alternatives. It is diagnostic extraction, not an eligibility parser.
    Unconsumed normative material invalidates the entire group. Source roles
    such as the AMT sibling group remain separate unresolved bindings for now.
    """
    headers = tuple(
        re.finditer(
            r"(?m)^\([1-9][0-9]*\) Provincial or territorial tax otherwise payable\n",
            source,
        )
    )
    quotes = sc._annual_source_quoted_spans(source)
    if len(headers) != 1 or quotes is None:
        return ()
    header = headers[0]
    if any(a < header.end() and b > header.start() for a, b in quotes):
        return ()
    cursor = header.end()
    found = []
    for kind, pattern, meaning in (
        (
            "same_output_however",
            _OTHER_FORM,
            "same-output ordinary instruction refined by explicit However; complement only within this closed frame",
        ),
        (
            "same_output_annual_override",
            _ON_FORM,
            "both zero-line procedures AND line81/58 result-identification sentences are required; later explicit conditional refines same output",
        ),
        (
            "applicable_form_alternative",
            _AB_FORM,
            "source OR operation set; caller's applicable-form fact is not monetary eligibility or document admission",
        ),
        (
            "excluded_components",
            _BC_FORM,
            "typed upstream form-result fact excluding both named components",
        ),
    ):
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        match = re.compile(pattern).match(source, cursor)
        if match is None:
            return ()
        # Quoted numeric zeros are grammatical operands. A quote wrapping a
        # whole instruction is not source authority for this extractor.
        if any(a <= match.start() < b for a, b in quotes):
            return ()
        found.append((kind, match, meaning))
        cursor = match.end()
    tail = source[cursor:]
    if not re.fullmatch(
        r"\s*See the privacy notice on your return\.\s+[A-Z][0-9]+ E \([0-9]{2}\) Page [0-9]+ of [0-9]+\s*",
        tail,
    ):
        return ()
    owner = RawSpan(header.start(), cursor, source[header.start() : cursor])
    return tuple(
        SourceFrame(
            kind,
            owner,
            RawSpan(match.start(), match.end(), match.group()),
            tuple(match.groupdict().items()),
            meaning,
        )
        for kind, match, meaning in found
    )


_AMT_PATTERN = _layout_pattern(r"""
\((?P<note>[1-9][0-9]*)\) If you must pay minimum tax, follow the instructions below:
\(cid:129\) If you were a resident of (?P<first>British Columbia) at the end of the
year, enter the amount from line (?P<federal_line>[1-9][0-9]*) of Form
(?P<federal_form>T[0-9]+) on line (?P<target>[1-9][0-9]*)\.
\(cid:129\) If you were a resident of (?P<second>Ontario) at the end of the year,
follow the instructions that apply to your situation:
– If the total non-business income taxes you paid to all foreign countries is
\$(?P<threshold>[1-9][0-9]*) or less:
1\. Take the amount from line (?P<base>[1-9][0-9]*) of this form\.
2\. Divide it by the sum of line (?P=base) of this form and the amount on line
(?P<tax_line>[1-9][0-9]*) of Part (?P<part>[1-9][0-9]*) of Form
(?P<amt_form>T[0-9]+), Alternative Minimum Tax\.
3\. Multiply that result by the special foreign tax credit on line
(?P<credit_line>[1-9][0-9]*) of Part (?P=part) of Form (?P=amt_form)\.
4\. Enter the result on line (?P=target)\.
– If the total non-business income taxes you paid to all foreign countries is
more than \$(?P=threshold), you must calculate for each country:
1\. Take the amount from line (?P=base) of this form for that country\.
2\. Divide it by the total foreign taxes paid for (?P<year>20[0-9]{2})
\(to get this total, add the amount on line (?P<additional_line>[1-9][0-9]*)
of Part (?P=part) of Form (?P=amt_form), Alternative Minimum Tax, divided by
(?P<rate>[0-9]+\.[0-9]+)% and the amount on line (?P=tax_line) of Part (?P=part)
of Form (?P=amt_form)\)\.
3\. Multiply that result by the special foreign tax credit on line
(?P=credit_line) of Part (?P=part) of Form (?P=amt_form)\.
4\. Enter the result on line (?P=target) of the sheet for that country\.
\(cid:129\) If you were a resident of another province or territory at the end
of the year, enter the part of special foreign tax credit
\(line (?P=credit_line) of Part (?P=part) of Form (?P=amt_form)\) that relates to
non-business income taxes you paid to a foreign country for (?P=year) on
line (?P=target)\.
""")


def closed_amt_frame(source: str) -> SourceFrame | None:
    headers = tuple(
        re.finditer(
            r"(?m)^\([1-9][0-9]*\) If you must pay minimum tax, follow the instructions below:",
            source,
        )
    )
    quotes = sc._annual_source_quoted_spans(source)
    if len(headers) != 1 or quotes is None:
        return None
    start = headers[0].start()
    following = re.search(r"(?m)^\([1-9][0-9]*\) ", source[headers[0].end() :])
    if following is None:
        return None
    end = headers[0].end() + following.start()
    body = source[start:end].rstrip()
    match = re.fullmatch(_AMT_PATTERN, body)
    if match is None or any(a < start + len(body) and b > start for a, b in quotes):
        return None
    span = RawSpan(start, start + len(body), body)
    return SourceFrame(
        "closed_amt_siblings",
        span,
        span,
        tuple(match.groupdict().items()),
        "another is the complement of the two named siblings within this complete same-line-result note; not a global province assumption",
    )


def _residence_selector_is_source_bound(
    rule: dict[str, Any],
    selected: int,
    source: str,
    citation: str,
    inputs: Mapping[str, dict[str, Any]],
) -> bool:
    """Reuse the reviewed typed definition, with closed-frame predicate scope.

    This only certifies the direct residence definition used in a path. It does
    not borrow the definition proof to certify the monetary result or its gates.
    """
    expression = _expression(rule["versions"][selected].get("formula"))
    if (
        not isinstance(expression, ast.Compare)
        or len(expression.ops) != 1
        or not isinstance(expression.ops[0], ast.Eq)
    ):
        return False
    a, b = expression.left, expression.comparators[0]
    if isinstance(a, ast.Constant):
        a, b = b, a
    if (
        not isinstance(a, ast.Name)
        or not isinstance(b, ast.Constant)
        or not isinstance(b.value, str)
    ):
        return False
    place = b.value
    intro_pattern = re.compile(
        r"^Use this form to calculate the [A-Za-z -]{1,100} (?:credit|tax|refund) "
        r"for (?P<year>(?:19|20)[0-9]{2}) that you can deduct from the income tax\s+"
        r"payable to the province or territory you resided in at the end of the tax year\.",
        re.MULTILINE,
    )
    intros = tuple(intro_pattern.finditer(source))
    quoted = sc._annual_source_quoted_spans(source)
    if len(intros) != 1 or quoted is None:
        return False
    intro = intros[0]
    header = re.fullmatch(
        r"(?:[A-Z][A-Za-z]* )?Form [A-Z]*[1-9]\d* "
        + re.escape(intro["year"])
        + r" (?P<title>[A-Za-z -]{1,120})\n\s*(?:Protected [A-Z] when completed\s+)?(?P=title)\s*",
        source[: intro.start()],
    )
    if header is None or any(a < intro.end() and b > 0 for a, b in quoted):
        return False
    permitted = []
    amt = closed_amt_frame(source)
    if amt is not None and amt.body.start > intro.end():
        fields = dict(amt.fields)
        if place == fields["first"]:
            permitted.append(
                f"If you were a resident of {place} at the end of the year, enter the amount from line {fields['federal_line']} of Form {fields['federal_form']} on line {fields['target']}."
            )
        if place == fields["second"]:
            permitted.append(
                f"If you were a resident of {place} at the end of the year, follow the instructions that apply to your situation:"
            )
    for frame in closed_result_frames(source):
        if dict(frame.fields).get("place") == place and frame.body.start > intro.end():
            permitted.append(sc._collapse_text(frame.body.text))
    intro_text = sc._collapse_text(intro.group())
    return any(
        sc._source_owned_residence_definition(
            rule,
            selected=selected,
            place=place,
            year=intro["year"],
            source_text=source,
            predicate_text=text,
            annual_header=intro_text.split(" that you can deduct", 1)[0],
            intro_text=intro_text,
            citation=citation.strip("/").casefold(),
            input_declarations=inputs,
        )
        is not None
        for text in permitted
    )


@dataclass(frozen=True)
class WorksheetShadowContract:
    obligations: tuple[SourceObligation, ...]
    frames: tuple[SourceFrame, ...]
    unresolved: tuple[str, ...]
    limits: tuple[str, ...] = (
        "Existing upstream fact aliases are an explicit reviewed contract, not a general alias inference.",
        "Closed source syntax and typed expression matching do not certify formula/numeric/date proof validity.",
        "No acceptance, condition coverage, annual coverage, or publication authorization is produced.",
    )


def worksheet_shadow_contract(
    source: str, *, citation: str, line2_owner: str, tax_otherwise_owner: str
) -> WorksheetShadowContract:
    """Bind the closed worksheet family to its existing factual API contract.

    Owner identities are explicit caller-selected targets, never inferred from
    arbitrary sibling rules. Operations and field coordinates come exclusively
    from source frames. Candidate formulas are not an argument to this builder.
    The existing fact aliases below are deliberately narrow and reviewable.
    """
    amt = closed_amt_frame(source)
    frames = closed_result_frames(source)
    if amt is None or len(frames) != 4:
        return WorksheetShadowContract(
            (), (), ("unrecognized/incomplete closed worksheet source frames",)
        )
    a = dict(amt.fields)
    by_kind = {f.kind: f for f in frames}
    on = dict(by_kind["same_output_annual_override"].fields)
    ab = dict(by_kind["applicable_form_alternative"].fields)
    year = a["year"]
    # Current upstream fact aliases do not encode container or part. Their
    # reviewed contract is exactly T2203, Part4; another coordinate must not
    # silently acquire the same alias merely because line numbers match.
    if (
        any(dict(frame.fields).get("container") != "T2203" for frame in frames)
        or on["part"] != "4"
        or ab["part"] != "4"
    ):
        return WorksheetShadowContract(
            (), (amt, *frames), ("unreviewed upstream form container/part identity",)
        )
    header = re.search(
        r"(?m)^Use this form to calculate the foreign non-business income tax credit\s+for (?P<year>20[0-9]{2}) that you can deduct from the income tax",
        source,
    )
    if header is None or header["year"] != year or on["year"] != year:
        return WorksheetShadowContract(
            (), (amt, *frames), ("annual source identity unresolved",)
        )
    # Verify the source's local line1 and ordinary line2 aliases. These are
    # source coordinates, not names recovered from the mutated candidate.
    line1_pattern = rf"(?m)^Enter the amount from line (?P<input>[1-9][0-9]*) of Form (?P<form>T[0-9]+)\. {re.escape(a['base'])}$"
    line1 = tuple(re.finditer(line1_pattern, source))
    ordinary_pattern = rf"(?m)^Enter the amount from line {re.escape(a['federal_line'])} of Form {re.escape(a['federal_form'])}, unless you have to pay minimum tax\.\({re.escape(a['note'])}\) – {re.escape(a['target'])}$"
    ordinary = tuple(re.finditer(ordinary_pattern, source))
    if (
        len(line1) != 1
        or len(ordinary) != 1
        or line1[0].end() >= amt.body.start
        or ordinary[0].end() >= amt.body.start
    ):
        return WorksheetShadowContract(
            (), (amt, *frames), ("local and ordinary form result aliases unresolved",)
        )
    quoted = sc._annual_source_quoted_spans(source)
    if quoted is None or any(
        a0 < m.end() and b0 > m.start()
        for m in (*line1, *ordinary)
        for a0, b0 in quoted
    ):
        return WorksheetShadowContract(
            (), (amt, *frames), ("quoted or malformed local form alias",)
        )
    digest = hashlib.sha256(source.encode()).hexdigest()
    text = "province_or_territory_of_residence_at_year_end"
    minimum = "minimum_tax_is_payable"
    total = "total_non_business_income_taxes_paid_to_all_foreign_countries"
    x = f"{line1[0]['form'].lower()}_non_business_income_tax_paid_to_foreign_country_line_{line1[0]['input']}"
    credit = f"{a['federal_form'].lower()}_federal_non_business_foreign_tax_credit_line_{a['federal_line']}"
    y = f"{a['amt_form'].lower()}_foreign_taxes_line_{a['tax_line']}_part_{a['part']}"
    z = f"{a['amt_form'].lower()}_special_foreign_tax_credit_line_{a['credit_line']}_part_{a['part']}"
    additional = f"{a['amt_form'].lower()}_foreign_taxes_line_{a['additional_line']}_part_{a['part']}"
    rate = Decimal(a["rate"]) / 100
    if not rate.is_finite() or not 0 < rate <= 1:
        return WorksheetShadowContract((), (amt, *frames), ("unsupported source rate",))
    threshold = Decimal(a["threshold"])
    obligations = []

    def predicate(name: str, op: str, value: Any, kind: str) -> Predicate:
        return Predicate(name, "Person", "Year", op, value, kind)

    def residence(place: str, positive: bool = True) -> Predicate:
        return predicate(text, "==" if positive else "!=", place, "text")

    def add(
        identity: str,
        owner: str,
        predicates: Sequence[Predicate],
        expression: str,
        spans: tuple[RawSpan, ...],
    ) -> None:
        names = {p.name for p in predicates}
        names.update(
            n.id
            for n in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(n, ast.Name)
        )
        predicate_types = {
            p.name: "Text"
            if p.kind == "text"
            else "Boolean"
            if p.kind == "boolean"
            else "Integer"
            if p.name.startswith("jurisdictions_tax_")
            else "Money"
            for p in predicates
        }
        contracts = tuple(
            InputContract(
                name,
                predicate_types.get(name, "Money"),
                "Person",
                "Year",
                "CAD" if predicate_types.get(name, "Money") == "Money" else None,
                spans,
            )
            for name in sorted(names)
        )
        obligations.append(
            SourceObligation(
                identity,
                owner,
                digest,
                citation,
                spans,
                tuple(predicates),
                expression,
                "Person",
                f"{year}-01-01",
                f"{year}-12-31",
                binding_provenance="closed source frame and independent operation descriptor; explicitly reviewed existing factual API aliases; semantic proof validators still mandatory",
                input_contracts=contracts,
            )
        )

    base = predicate(minimum, "==", True, "boolean")
    ordinary_span = RawSpan(ordinary[0].start(), ordinary[0].end(), ordinary[0].group())
    add(
        "ordinary_line2",
        line2_owner,
        [predicate(minimum, "==", False, "boolean")],
        credit,
        (ordinary_span,),
    )
    add("bc_amt", line2_owner, [base, residence(a["first"])], credit, (amt.body,))
    add(
        "ontario_small",
        line2_owner,
        [base, residence(a["second"]), predicate(total, "<=", threshold, "numeric")],
        f"({x} / ({x} + {y})) * {z}",
        (amt.body,),
    )
    add(
        "ontario_large",
        line2_owner,
        [base, residence(a["second"]), predicate(total, ">", threshold, "numeric")],
        f"({x} / ({additional} / {rate} + {y})) * {z}",
        (amt.body,),
    )
    add(
        "other_amt",
        line2_owner,
        [base, residence(a["first"], False), residence(a["second"], False)],
        "special_foreign_tax_credit_related_to_non_business_income",
        (amt.body,),
    )
    other = by_kind["same_output_however"]
    exclusion = [
        residence(place, False) for place in ("Ontario", "Alberta", "British Columbia")
    ]
    count = "jurisdictions_tax_payable_count"
    add(
        "other_single",
        tax_otherwise_owner,
        [*exclusion, predicate(count, "<=", Decimal(1), "numeric")],
        "form_428_tax_before_provincial_or_territorial_foreign_tax_credit",
        (other.owner, other.body),
    )
    add(
        "other_multi",
        tax_otherwise_owner,
        [*exclusion, predicate(count, ">", Decimal(1), "numeric")],
        "section_428mj_tax_before_foreign_tax_credit_for_residence_jurisdiction",
        (other.owner, other.body),
    )
    count = "jurisdictions_tax_paid_count"
    on_frame = by_kind["same_output_annual_override"]
    add(
        "ontario_single",
        tax_otherwise_owner,
        [residence(on["place"]), predicate(count, "<=", Decimal(1), "numeric")],
        f"{on['form'].lower()}_line_{on['result_a']}_after_entering_zero_on_lines_{on['zero_a']}_and_{on['zero_b']}",
        (on_frame.owner, on_frame.body),
    )
    add(
        "ontario_multi",
        tax_otherwise_owner,
        [residence(on["place"]), predicate(count, ">", Decimal(1), "numeric")],
        f"{on['section'].lower()}_line_{on['result_b']}_after_entering_zero_on_lines_{on['zero_c']}_and_{on['zero_d']}",
        (on_frame.owner, on_frame.body),
    )
    ab_frame = by_kind["applicable_form_alternative"]
    choice = "alberta_applicable_form_is_428mj"
    for selected, form, left, right in (
        (False, ab["form"], ab["left_a"], ab["right_a"]),
        (True, ab["section"], ab["left_b"], ab["right_b"]),
    ):
        add(
            f"alberta_{selected}",
            tax_otherwise_owner,
            [residence(ab["place"]), predicate(choice, "==", selected, "boolean")],
            f"{form.lower()}_line_{left} + {form.lower()}_line_{right}",
            (ab_frame.owner, ab_frame.body),
        )
    bc_frame = by_kind["excluded_components"]
    add(
        "bc_otherwise",
        tax_otherwise_owner,
        [residence("British Columbia")],
        "british_columbia_tax_excluding_foreign_tax_credit_and_additional_minimum_tax",
        (bc_frame.owner, bc_frame.body),
    )
    return WorksheetShadowContract(tuple(obligations), (amt, *frames), ())


def corroborated_case_links(
    payload: Mapping[str, Any],
    result: ShadowResult,
    cases: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Actual existing-case links, not paired-witness or coverage certification."""
    rules = _index(payload.get("rules")) or {}
    rule = rules.get(result.owner)
    if rule is None:
        return ()
    # Match the established complete-source analyzer boundary; do not invent
    # numeric coercion in the strict runtime comparator. Text stays Text.
    cases = sc._typed_numeric_expected_cases(cases, rules) or ()
    constants = sc._constant_rule_environment(payload)
    links = []
    for case_index, case in enumerate(cases):
        asserted = sc._test_case_asserted_output_value(case, result.owner)
        if asserted is sc._UNRESOLVED_CONDITION_VALUE:
            continue
        period = case.get("period")
        if not isinstance(period, dict) or period.get("period_kind") != "tax_year":
            continue
        start, end = period.get("start"), period.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            continue
        if sc._case_runtime_period_start(case) is sc._UNRESOLVED_CONDITION_VALUE:
            continue
        if start[5:] != "01-01" or end != start[:4] + "-12-31":
            continue
        environment = sc._case_input_formula_environment(case)
        if environment is None:
            continue
        for index, path in enumerate(result.paths):
            if path.unresolved or not path.start <= start <= end <= path.end:
                continue
            if any(
                sc._selected_rule_formula_version_index(rules[d.name], case)
                != d.version
                for d in path.dependencies
            ):
                continue
            if not all(
                _predicate_value(p, environment) is True for p in path.predicates
            ):
                continue
            dependencies = sc._case_asserted_dependency_environment(
                rules, case, formula_environment=constants
            )
            execution = sc._case_formula_execution(
                rule,
                case,
                formula_environment=constants,
                dependency_environment=dependencies,
            )
            actual = (
                sc._formula_execution_runtime_value(execution)
                if execution is not None
                else sc._UNRESOLVED_CONDITION_VALUE
            )
            resolved = actual is not sc._UNRESOLVED_CONDITION_VALUE
            links.append(
                {
                    "case_index": case_index,
                    "case_name": case.get("name"),
                    "path": index,
                    "owner": result.owner,
                    "asserted": asserted,
                    "actual": actual if resolved else None,
                    "resolved": resolved,
                    "corroborated": resolved
                    and sc._asserted_formula_runtime_values_equal(
                        rule, actual, asserted
                    ),
                    "selected_dependencies": tuple(
                        (d.name, d.version, d.start, d.end) for d in path.dependencies
                    ),
                    "scope": "actual asserted owner execution; not a same-world pair or source witness assignment",
                }
            )
    return tuple(links)


def _predicate_value(
    predicate: Predicate, environment: Mapping[str, Any]
) -> bool | None:
    if predicate.name not in environment:
        return None
    value = environment[predicate.name]
    if predicate.kind == "text":
        if not isinstance(value, str):
            return None
    elif predicate.kind == "boolean":
        if type(value) is not bool:
            return None
    elif predicate.kind == "numeric":
        if type(value) not in {int, float, Decimal}:
            return None
        value = Decimal(str(value))
        if not value.is_finite():
            return None
    else:
        return None
    if predicate.operator == "==":
        return value == predicate.value
    if predicate.operator == "!=":
        return value != predicate.value
    if predicate.kind != "numeric":
        return None
    if predicate.operator == "<":
        return value < predicate.value
    if predicate.operator == "<=":
        return value <= predicate.value
    if predicate.operator == ">":
        return value > predicate.value
    if predicate.operator == ">=":
        return value >= predicate.value
    return None


def cap_transfer_frames(source: str) -> tuple[SourceFrame, ...]:
    """Separate an explicit monetary upper bound from output routing."""
    labels = sc._corroborated_form_output_label_spans(source)
    if not labels:
        return ()
    cap = tuple(
        re.finditer(
            r"The amount on line (?P<line>[1-9][0-9]*) should not be more than the amount entered Provincial or territorial\s+"
            r"on the line for provincial or territorial tax otherwise payable\. foreign tax credit (?P=line)",
            source,
        )
    )
    transfer = tuple(
        re.finditer(
            r"Enter the total from line (?P<line>[1-9][0-9]*) \(for each country if applicable\) on the line for the provincial or territorial foreign tax credit of\s+"
            r"Form (?P<form>428)\. If you have to pay tax to more than one jurisdiction, enter the amount from line (?P=line) on the applicable line in Part (?P<part>[1-9][0-9]*),\s+"
            r"Section (?P<section>428MJ) of Form (?P<container>T[0-9]+), Provincial and Territorial Taxes for Multiple Jurisdictions, only for the province or territory\s+"
            r"you resided in on the last day of the tax year\.",
            source,
        )
    )
    if len(cap) != 1 or len(transfer) != 1 or cap[0]["line"] != transfer[0]["line"]:
        return ()
    quoted = sc._annual_source_quoted_spans(source)
    if quoted is None or any(
        a < transfer[0].end() and b > cap[0].start() for a, b in quoted
    ):
        return ()
    return tuple(
        SourceFrame(
            kind,
            RawSpan(m.start(), m.end(), m.group()),
            RawSpan(m.start(), m.end(), m.group()),
            tuple(m.groupdict().items()),
            meaning,
        )
        for kind, m, meaning in (
            (
                "numeric_upper_bound",
                cap[0],
                "amount must not exceed tax otherwise payable; operation identity, not a factual Boolean toggle",
            ),
            (
                "output_transfer",
                transfer[0],
                "route the same computed line to applicable form; no required amount change implied",
            ),
        )
    )


def inspect_cap_dependency(
    payload: Mapping[str, Any], *, source_text: str, principal: str, bound_owner: str
) -> dict[str, Any]:
    """Structural cap observation only; full operation/proof checks remain due."""
    frames = cap_transfer_frames(source_text)
    rules = _index(payload.get("rules")) or {}
    owner = rules.get(principal)
    target = rules.get(bound_owner)
    if not frames or owner is None or target is None:
        return {"unresolved": "missing source or output identity", "acceptance": False}
    observations = []
    for index, formula, start, end in sc._effective_formula_version_intervals(owner):
        raw = _raw_paths(str(formula))
        if raw is None:
            observations.append(
                {"version": index, "unresolved": "unsupported principal paths"}
            )
            continue
        for selectors, leaf in raw:
            node = _expression(leaf)
            # The exact bound dependency must be an argument of the reached
            # min operation. Mentioning it inside an unrelated subexpression
            # or unused helper is not this operation identity.
            has_bound = (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "min"
                and not node.keywords
                and any(
                    isinstance(arg, ast.Name) and arg.id == bound_owner
                    for arg in node.args
                )
            )
            observations.append(
                {
                    "version": index,
                    "start": start,
                    "end": end,
                    "selectors": selectors,
                    "leaf": leaf,
                    "direct_min_bound_operand": has_bound,
                }
            )
    return {
        "observations": observations,
        "source_roles": tuple(f.kind for f in frames),
        "acceptance": False,
        "limit": "cap operation identity observation; no path/witness or transfer discharge",
    }

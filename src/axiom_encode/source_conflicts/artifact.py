"""Read-only view of one compiled engine artifact.

The compare step needs four things from an artifact, and takes all of them
from the artifact's own expression trees, never from the module text:

- the inputs it accepts (``metadata.input_catalog``);
- the rules and inputs that a set of output rules depends on;
- whether each input is used as a bool or as a number;
- the constant thresholds that fact-dependent values are compared with.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, DivisionByZero, InvalidOperation
from typing import Any

BOOL = "bool"
NUMERIC = "numeric"
_BOOL_NODES = frozenset({"and", "or", "not", "comparison"})
_NUMERIC_NODES = frozenset(
    {"add", "sub", "mul", "div", "min", "max", "neg", "floor", "ceil", "round", "abs"}
)
_BOOL_DTYPES = frozenset({"judgment", "bool", "boolean"})
_NUMERIC_VALUE_KINDS = frozenset({"integer", "decimal"})


class ArtifactError(ValueError):
    """The artifact does not have the shape this view reads."""


@dataclass(frozen=True)
class Threshold:
    """A constant that a fact-dependent value is compared with."""

    value: Decimal
    rule: str
    inputs: frozenset[str]


def _children(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for value in node.values():
        if isinstance(value, dict) and "kind" in value:
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "kind" in item:
                    yield item


class Artifact:
    """One compiled module, read at the start date of the compared period."""

    def __init__(self, payload: Mapping[str, Any], period_start: str):
        try:
            program = payload["program"]
            catalog = payload["metadata"]["input_catalog"]
            self._derived = {rule["name"]: rule for rule in program["derived"]}
            self._parameters = {rule["name"]: rule for rule in program["parameters"]}
            self.request_name_by_slot = {
                entry["slot"]: entry["canonical_request_name"] for entry in catalog
            }
        except (KeyError, TypeError) as error:
            raise ArtifactError(
                f"the compiled artifact lacks the expected field {error}"
            ) from error
        self.engine_version = payload.get("engine_version")
        self.period_start = period_start
        self.slot_by_request_name = {
            name: slot for slot, name in self.request_name_by_slot.items()
        }
        self._derived_by_id = {
            rule["id"]: rule for rule in self._derived.values() if rule.get("id")
        }
        self._inputs_memo: dict[str, frozenset[str]] = {}

    # -- lookup -----------------------------------------------------------

    def derived(self, rule_id: str) -> Mapping[str, Any] | None:
        return self._derived_by_id.get(rule_id)

    def rule_id(self, name: str) -> str:
        rule = self._derived.get(name) or self._parameters.get(name) or {}
        return rule.get("id") or name

    def rule(self, rule_id: str) -> Mapping[str, Any] | None:
        for group in (self._derived, self._parameters):
            for rule in group.values():
                if rule.get("id") == rule_id:
                    return rule
        return None

    def _effective(self, rule: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The version of ``rule`` in force at the period start."""

        chosen = None
        for version in rule.get("versions") or []:
            start = version.get("effective_from") or "0001-01-01"
            end = version.get("effective_to")
            if start <= self.period_start and (end is None or self.period_start <= end):
                if chosen is None or start >= (chosen.get("effective_from") or ""):
                    chosen = version
        return chosen

    def expression(self, rule: Mapping[str, Any]) -> Mapping[str, Any] | None:
        """The formula in force at the period start; ``None`` when there is none.

        A rule with versions but none in force has no formula on that date,
        as in the engine; its top-level formula is not used then.
        """

        if rule.get("versions"):
            version = self._effective(rule)
            if version is None:
                return None
            if isinstance(version.get("expr"), dict):
                return version["expr"]
        return rule.get("expr") if isinstance(rule.get("expr"), dict) else None

    def without_version(self, derived_names: Iterable[str]) -> list[str]:
        """Ids of the derived rules with no version in force at the period start."""

        return [
            self.rule_id(name)
            for name in derived_names
            if (self._derived.get(name) or {}).get("versions")
            and self._effective(self._derived[name]) is None
        ]

    # -- dependencies -----------------------------------------------------

    def _walk(self, node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
        yield node
        if node.get("kind") == "literal":
            return
        for child in _children(node):
            yield from self._walk(child)

    def closure(
        self, rule_ids: Iterable[str]
    ) -> tuple[list[str], list[str], list[str]]:
        """Derived names, parameter names and input slots the rules depend on.

        Each list is in first-reached order, so it is stable for one artifact.
        """

        derived: dict[str, None] = {}
        parameters: dict[str, None] = {}
        slots: dict[str, None] = {}
        pending = [
            self._derived_by_id[rule_id]["name"]
            for rule_id in rule_ids
            if rule_id in self._derived_by_id
        ]
        while pending:
            name = pending.pop(0)
            if name in derived or name not in self._derived:
                continue
            derived[name] = None
            expression = self.expression(self._derived[name])
            if expression is None:
                continue
            for node in self._walk(expression):
                kind = node.get("kind")
                if kind == "derived":
                    pending.append(node.get("name"))
                elif kind == "input":
                    slots[node.get("name")] = None
                elif kind == "parameter_lookup":
                    parameters[node.get("parameter")] = None
        return list(derived), list(parameters), list(slots)

    def direct_dependencies(self, name: str) -> list[str]:
        """Ids of the rules that the derived rule ``name`` reads directly."""

        found: dict[str, None] = {}
        expression = self.expression(self._derived.get(name) or {})
        if expression is not None:
            for node in self._walk(expression):
                if node.get("kind") == "derived":
                    found[self.rule_id(node.get("name"))] = None
                elif node.get("kind") == "parameter_lookup":
                    found[self.rule_id(node.get("parameter"))] = None
        return sorted(found)

    def _expression_inputs(self, node: Mapping[str, Any]) -> frozenset[str]:
        slots: set[str] = set()
        for item in self._walk(node):
            if item.get("kind") == "input":
                slots.add(item.get("name"))
            elif item.get("kind") == "derived":
                slots |= self._rule_inputs(item.get("name"))
        return frozenset(slots)

    def _rule_inputs(self, name: str) -> frozenset[str]:
        if name not in self._inputs_memo:
            self._inputs_memo[name] = frozenset()  # a cycle reads nothing new
            expression = self.expression(self._derived.get(name) or {})
            if expression is not None:
                self._inputs_memo[name] = self._expression_inputs(expression)
        return self._inputs_memo[name]

    # -- kinds ------------------------------------------------------------

    def _static_class(self, node: Mapping[str, Any]) -> str | None:
        kind = node.get("kind")
        if kind in _BOOL_NODES:
            return BOOL
        if kind in _NUMERIC_NODES:
            return NUMERIC
        if kind == "literal":
            value = node.get("value")
            value_kind = value.get("kind") if isinstance(value, dict) else None
            if value_kind == "bool":
                return BOOL
            return NUMERIC if value_kind in _NUMERIC_VALUE_KINDS else None
        if kind == "derived":
            dtype = str(
                (self._derived.get(node.get("name")) or {}).get("dtype")
            ).lower()
            if dtype in _BOOL_DTYPES:
                return BOOL
            return (
                NUMERIC if dtype in {"decimal", "integer", "money", "count"} else None
            )
        if kind == "parameter_lookup":
            value = self._parameter_value(node)
            if isinstance(value, bool):
                return BOOL
            return NUMERIC if isinstance(value, Decimal) else None
        return None

    def input_classes(self, derived_names: Iterable[str]) -> dict[str, set[str]]:
        """How each input slot is used: as a bool, as a number, or both."""

        classes: dict[str, set[str]] = {}

        def visit(node: Mapping[str, Any], context: str | None) -> None:
            kind = node.get("kind")
            if kind == "input":
                if context is not None:
                    classes.setdefault(node.get("name"), set()).add(context)
                return
            if kind == "literal":
                return
            if kind == "comparison":
                left, right = node.get("left"), node.get("right")
                if isinstance(left, dict) and isinstance(right, dict):
                    visit(left, self._static_class(right))
                    visit(right, self._static_class(left))
                return
            if kind == "if":
                for key, value in node.items():
                    if isinstance(value, dict) and "kind" in value:
                        visit(value, BOOL if key == "condition" else context)
                return
            if kind in ("and", "or", "not"):
                child_context = BOOL
            elif kind in _NUMERIC_NODES:
                child_context = NUMERIC
            else:
                child_context = None
            for child in _children(node):
                visit(child, child_context)

        for name in derived_names:
            expression = self.expression(self._derived.get(name) or {})
            if expression is not None:
                visit(expression, None)
        return classes

    # -- constants --------------------------------------------------------

    def _parameter_value(
        self, node: Mapping[str, Any], fixed: Mapping[str, object] | None = None
    ) -> Decimal | bool | None:
        parameter = self._parameters.get(node.get("parameter"))
        if parameter is None:
            return None
        version = self._effective(parameter)
        index = node.get("index")
        key = self.fold(index, fixed or {}) if isinstance(index, dict) else Decimal(0)
        if version is None or not isinstance(key, Decimal) or isinstance(key, bool):
            return None
        value = (version.get("values") or {}).get(str(int(key)))
        return _literal(value) if isinstance(value, dict) else None

    def fold(
        self,
        node: Mapping[str, Any],
        fixed: Mapping[str, object],
        _memo: dict[str, Decimal | bool | None] | None = None,
    ) -> Decimal | bool | None:
        """The constant value of ``node``, or ``None`` when it depends on a fact.

        ``fixed`` maps input slots to the link's fixed values. Only literals,
        parameters, fixed inputs, derived rules over those, and plain
        arithmetic fold; anything else is not a constant here.
        """

        memo: dict[str, Decimal | bool | None] = {} if _memo is None else _memo
        kind = node.get("kind")
        if kind == "literal":
            value = node.get("value")
            return _literal(value) if isinstance(value, dict) else None
        if kind == "parameter_lookup":
            return self._parameter_value(node, fixed)
        if kind == "input":
            return _plain(fixed.get(node.get("name")))
        if kind == "derived":
            return self._fold_rule(node.get("name"), fixed, memo)
        if kind in ("sub", "mul", "div"):
            operands = [
                self.fold(node[key], fixed, memo)
                if isinstance(node.get(key), dict)
                else None
                for key in ("left", "right")
            ]
        elif kind in ("add", "min", "max", "neg"):
            operands = [self.fold(child, fixed, memo) for child in _children(node)]
        else:
            return None
        if not operands or any(
            isinstance(operand, bool) or not isinstance(operand, Decimal)
            for operand in operands
        ):
            return None
        try:
            if kind == "add":
                return sum(operands, Decimal(0))
            if kind == "min":
                return min(operands)
            if kind == "max":
                return max(operands)
            if kind == "neg":
                return -operands[0] if len(operands) == 1 else None
            left, right = operands
            if kind == "sub":
                return left - right
            if kind == "mul":
                return left * right
            return left / right
        except (DivisionByZero, InvalidOperation):
            return None

    def _fold_rule(
        self,
        name: str,
        fixed: Mapping[str, object],
        memo: dict[str, Decimal | bool | None],
    ) -> Decimal | bool | None:
        """Fold one derived rule once; rules it reads fold first, without recursion depth.

        A chain of rules is walked with an explicit stack, so neither a long
        chain nor a rule that many rules read costs more than one fold each.
        """

        stack = [name]
        while stack:
            current = stack[-1]
            if current in memo:
                stack.pop()
                continue
            expression = self.expression(self._derived.get(current) or {})
            if expression is None:
                memo[current] = None
                stack.pop()
                continue
            waiting = [
                item.get("name")
                for item in self._walk(expression)
                if item.get("kind") == "derived"
                and item.get("name") not in memo
                and item.get("name") not in stack
            ]
            if waiting:
                stack.extend(dict.fromkeys(waiting))
                continue
            memo[current] = None  # a rule that reads itself is not a constant
            memo[current] = self.fold(expression, fixed, memo)
            stack.pop()
        return memo[name]

    def thresholds(
        self, derived_names: Iterable[str], fixed: Mapping[str, object]
    ) -> list[Threshold]:
        """Constants that the rules compare a fact-dependent value with."""

        found: list[Threshold] = []
        memo: dict[str, Decimal | bool | None] = {}
        for name in derived_names:
            rule = self._derived.get(name) or {}
            expression = self.expression(rule)
            if expression is None:
                continue
            for node in self._walk(expression):
                if node.get("kind") != "comparison":
                    continue
                left, right = node.get("left"), node.get("right")
                if not isinstance(left, dict) or not isinstance(right, dict):
                    continue
                for constant, other in ((left, right), (right, left)):
                    value = self.fold(constant, fixed, memo)
                    if (
                        not isinstance(value, Decimal)
                        or self.fold(other, fixed, memo) is not None
                    ):
                        continue
                    inputs = self._expression_inputs(other) - set(fixed)
                    if inputs:
                        found.append(
                            Threshold(value, rule.get("id") or name, frozenset(inputs))
                        )
        return found


def _plain(value: object) -> Decimal | bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, str)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    return None


def _literal(value: Mapping[str, Any]) -> Decimal | bool | None:
    kind = value.get("kind")
    if kind == "bool":
        return value.get("value") if isinstance(value.get("value"), bool) else None
    if kind in _NUMERIC_VALUE_KINDS:
        return _plain(value.get("value"))
    return None

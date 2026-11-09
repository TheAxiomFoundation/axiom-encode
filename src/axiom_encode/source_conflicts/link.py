"""The link file: which rules of two modules decide the same question.

A link names the output rules on each side, maps each shared fact to each
side's input, defines inputs that one side takes and the other computes,
states the constraints a real household meets, and fixes the remaining inputs.
``load_link`` reads JSON or YAML and validates the whole document; every
problem is reported in one :class:`LinkError`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import yaml

from .expressions import Constraint, ExpressionError, parse_constraint

LINK_FORMAT = "axiom-source-conflict-link/v1"
SIDES = ("a", "b")
KINDS = ("bool", "integer", "decimal")
COMBINE = ("any", "all", "value")
VALUE = "value"
DERIVED_OPERATORS = ("sum",)
DEFAULT_MAX_COMBINATIONS = 20_000
DEFAULT_SEED = 0
MAX_COMBINATIONS_LIMIT = 1_000_000
MAX_LINK_BYTES = 1_000_000

_ID = re.compile(r"[a-z0-9][a-z0-9._/-]*(?::[a-z0-9][a-z0-9._/-]*)+")
_FACT = re.compile(r"[a-z][a-z0-9_]*")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_NUMBER = re.compile(r"-?[0-9]+(?:\.[0-9]+)?")
_RESERVED_FACTS = ("true", "false")

_TOP_KEYS = {
    "format": True,
    "id": True,
    "question": False,
    "entity": True,
    "period": True,
    "a": True,
    "b": True,
    "facts": True,
    "derived": False,
    "fixed": False,
    "constraints": False,
    "guards": False,
    "sampling": False,
    "tolerance": False,
}
_SIDE_KEYS = {
    "label": True,
    "module": True,
    "outputs": True,
    "combine": True,
    "note": False,
}
_FACT_KEYS = {
    "kind": True,
    "a": False,
    "b": False,
    "values": False,
    "step": False,
    "min": False,
    "max": False,
    "note": False,
}
_DERIVED_KEYS = {"kind": True, "op": True, "of": True, "note": False}
_FIXED_KEYS = {"kind": True, "value": True, "note": False}
_CONSTRAINT_KEYS = {"rule": True, "note": False}
_PERIOD_KEYS = {"period_kind": True, "start": True, "end": True}
_SAMPLING_KEYS = {"max_combinations": False, "seed": False}


class LinkError(ValueError):
    """The link file is malformed. ``problems`` lists every finding."""

    heading = "invalid source-conflict link"

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__(
            f"{self.heading}:\n"
            + "\n".join(f"- {problem}" for problem in self.problems)
        )


def canonical_decimal(value: Decimal) -> str:
    """One spelling per number: no exponent, no trailing zeros."""

    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def to_decimal(value: object) -> Decimal | None:
    """A link number as ``Decimal``; ``None`` when it is not a plain number."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        value = repr(value)
    if isinstance(value, str) and _NUMBER.fullmatch(value):
        try:
            return Decimal(value)
        except InvalidOperation:  # pragma: no cover - the pattern admits only numbers
            return None
    return None


def normalize_value(kind: str, value: object) -> bool | int | str | None:
    """The canonical form of one value of ``kind``, or ``None`` when it does not fit."""

    if kind == "bool":
        return value if isinstance(value, bool) else None
    number = to_decimal(value)
    if number is None:
        return None
    if kind == "integer":
        return int(number) if number == number.to_integral_value() else None
    return canonical_decimal(number)


@dataclass(frozen=True)
class Side:
    key: str
    label: str
    module: str
    outputs: tuple[str, ...]
    combine: str
    guards: tuple[str, ...]
    note: str | None


@dataclass(frozen=True)
class Fact:
    name: str
    kind: str
    inputs: Mapping[str, str]
    values: tuple[bool | int | str, ...] | None
    step: Decimal
    minimum: Decimal | None
    maximum: Decimal | None
    note: str | None


@dataclass(frozen=True)
class DerivedInput:
    input: str
    kind: str
    op: str
    of: tuple[str, ...]
    note: str | None


@dataclass(frozen=True)
class FixedInput:
    input: str
    kind: str
    value: bool | int | str
    note: str | None


@dataclass(frozen=True)
class LinkConstraint:
    rule: str
    note: str | None
    expression: Constraint


@dataclass(frozen=True)
class Link:
    id: str
    question: str | None
    entity: str
    period: Mapping[str, str]
    sides: Mapping[str, Side]
    facts: Mapping[str, Fact]
    derived: Mapping[str, DerivedInput]
    fixed: Mapping[str, FixedInput]
    constraints: tuple[LinkConstraint, ...]
    max_combinations: int
    seed: int
    sha256: str
    tolerance: Decimal = Decimal(0)


def _keys(
    value: object, allowed: Mapping[str, bool], where: str, problems: list[str]
) -> dict[str, Any] | None:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        problems.append(f"{where} must be a mapping")
        return None
    for key in value:
        if key not in allowed:
            problems.append(f"{where} has the unknown key '{key}'")
    for key, required in allowed.items():
        if required and key not in value:
            problems.append(f"{where} lacks the required key '{key}'")
    return value


def _text(value: object, where: str, problems: list[str]) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    problems.append(f"{where} must be a non-empty string")
    return None


def _note(mapping: dict[str, Any], where: str, problems: list[str]) -> str | None:
    if "note" not in mapping:
        return None
    return _text(mapping["note"], f"{where}.note", problems)


def _names(value: object, where: str, problems: list[str]) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        problems.append(f"{where} must be a non-empty list of names")
        return ()
    if len(set(value)) != len(value):
        problems.append(f"{where} repeats a name")
    return tuple(value)


def _side(
    key: str, document: dict[str, Any], guards: tuple[str, ...], problems: list[str]
) -> Side | None:
    raw = _keys(document.get(key), _SIDE_KEYS, key, problems)
    if raw is None:
        return None
    label = _text(raw.get("label"), f"{key}.label", problems)
    module = _text(raw.get("module"), f"{key}.module", problems)
    if module is not None:
        parts = Path(module).parts
        if (
            Path(module).is_absolute()
            or ".." in parts
            or "\\" in module
            or not module.endswith(".yaml")
        ):
            problems.append(
                f"{key}.module must be a relative '.yaml' path inside the RuleSpec checkout"
            )
            module = None
    outputs = _names(raw.get("outputs"), f"{key}.outputs", problems)
    combine = raw.get("combine")
    if combine not in COMBINE:
        problems.append(f"{key}.combine must be one of {', '.join(COMBINE)}")
    if combine == VALUE and len(outputs) > 1:
        problems.append(
            f"{key}.combine is 'value', which compares one number: {key}.outputs "
            "must name exactly one rule"
        )
    if label is None or module is None or not outputs or combine not in COMBINE:
        return None
    return Side(
        key=key,
        label=label,
        module=module,
        outputs=outputs,
        combine=combine,
        guards=guards,
        note=_note(raw, key, problems),
    )


def _bound(raw: dict[str, Any], key: str, where: str, problems: list[str]):
    if key not in raw:
        return None
    number = to_decimal(raw[key])
    if number is None:
        problems.append(f"{where}.{key} must be a number")
    return number


def _fact(name: str, value: object, problems: list[str]) -> Fact | None:
    where = f"facts.{name}"
    if not _FACT.fullmatch(name) or name in _RESERVED_FACTS:
        problems.append(
            f"{where}: a fact name must match [a-z][a-z0-9_]* and not be 'true' or 'false'"
        )
        return None
    raw = _keys(value, _FACT_KEYS, where, problems)
    if raw is None:
        return None
    kind = raw.get("kind")
    if kind not in KINDS:
        problems.append(f"{where}.kind must be one of {', '.join(KINDS)}")
        return None
    inputs = {}
    for side in SIDES:
        if side in raw:
            text = _text(raw[side], f"{where}.{side}", problems)
            if text is not None:
                inputs[side] = text
    values = None
    if "values" in raw:
        listed = raw["values"]
        if not isinstance(listed, list) or not listed:
            problems.append(f"{where}.values must be a non-empty list")
        else:
            normalized = [normalize_value(kind, item) for item in listed]
            if any(item is None for item in normalized):
                problems.append(f"{where}.values must all be {kind} values")
            elif len(set(normalized)) != len(normalized):
                problems.append(f"{where}.values repeats a value")
            else:
                values = tuple(normalized)
    step = Decimal(1)
    minimum = maximum = None
    if kind == "bool":
        for key in ("step", "min", "max"):
            if key in raw:
                problems.append(f"{where}.{key} does not apply to a bool fact")
    else:
        found = _bound(raw, "step", where, problems)
        if found is not None:
            if found <= 0 or (kind == "integer" and found != found.to_integral_value()):
                problems.append(f"{where}.step must be a positive {kind} number")
            else:
                step = found
        minimum = _bound(raw, "min", where, problems)
        maximum = _bound(raw, "max", where, problems)
        if minimum is None:
            minimum = Decimal(0)
        if maximum is not None and maximum < minimum:
            problems.append(f"{where}.max is below its min")
    return Fact(
        name=name,
        kind=kind,
        inputs=inputs,
        values=values,
        step=step,
        minimum=minimum,
        maximum=maximum,
        note=_note(raw, where, problems),
    )


def _canonical(link_fields: dict[str, Any]) -> str:
    # Key order is kept: the order of the facts sets the grid order, the
    # sample and the witness that is chosen, so it is part of the link.
    return json.dumps(link_fields, separators=(",", ":"))


def parse_link(document: object) -> Link:
    """Validate a parsed link document and return the typed link."""

    problems: list[str] = []
    raw = _keys(document, _TOP_KEYS, "link", problems)
    if raw is None:
        raise LinkError(problems)
    if raw.get("format") != LINK_FORMAT:
        problems.append(f"link.format must be '{LINK_FORMAT}'")
    link_id = raw.get("id")
    if not isinstance(link_id, str) or not _ID.fullmatch(link_id):
        problems.append(
            "link.id must be a stable lowercase id with a jurisdiction prefix, "
            "such as 'us:medicaid-community-engagement-statute-vs-ifr'"
        )
        link_id = None
    question = None
    if "question" in raw:
        question = _text(raw["question"], "link.question", problems)
    entity = _text(raw.get("entity"), "link.entity", problems)

    period = _keys(raw.get("period"), _PERIOD_KEYS, "period", problems) or {}
    for key in ("start", "end"):
        if key in period and not (
            isinstance(period[key], str) and _DATE.fullmatch(period[key])
        ):
            problems.append(f"period.{key} must be a YYYY-MM-DD date")
    if "period_kind" in period:
        _text(period["period_kind"], "period.period_kind", problems)
    if (
        isinstance(period.get("start"), str)
        and isinstance(period.get("end"), str)
        and period["end"] <= period["start"]
    ):
        problems.append("period.end must be after period.start")

    guards: dict[str, tuple[str, ...]] = {side: () for side in SIDES}
    if "guards" in raw:
        raw_guards = _keys(raw["guards"], {"a": False, "b": False}, "guards", problems)
        for side, names in (raw_guards or {}).items():
            if side in SIDES:
                guards[side] = _names(names, f"guards.{side}", problems)

    sides = {side: _side(side, raw, guards[side], problems) for side in SIDES}

    facts: dict[str, Fact] = {}
    raw_facts = raw.get("facts")
    if not isinstance(raw_facts, dict) or not raw_facts:
        problems.append("facts must be a non-empty mapping")
        raw_facts = {}
    for name, value in raw_facts.items():
        fact = _fact(str(name), value, problems)
        if fact is not None:
            facts[fact.name] = fact

    derived: dict[str, DerivedInput] = {}
    raw_derived = raw.get("derived", {})
    if not isinstance(raw_derived, dict):
        problems.append("derived must be a mapping from input name to definition")
        raw_derived = {}
    for name, value in raw_derived.items():
        where = f"derived.{name}"
        entry = _keys(value, _DERIVED_KEYS, where, problems)
        if entry is None:
            continue
        kind, op = entry.get("kind"), entry.get("op")
        if kind not in ("integer", "decimal"):
            problems.append(f"{where}.kind must be integer or decimal")
        if op not in DERIVED_OPERATORS:
            problems.append(f"{where}.op must be one of {', '.join(DERIVED_OPERATORS)}")
        of = _names(entry.get("of"), f"{where}.of", problems)
        for fact_name in of:
            if fact_name not in raw_facts:
                problems.append(f"{where}.of names '{fact_name}', which is not a fact")
            elif fact_name in facts and facts[fact_name].kind == "bool":
                problems.append(f"{where}.of names the bool fact '{fact_name}'")
            elif (
                fact_name in facts
                and kind == "integer"
                and facts[fact_name].kind != "integer"
            ):
                problems.append(
                    f"{where} is an integer but sums the decimal fact '{fact_name}'"
                )
        if kind in ("integer", "decimal") and op in DERIVED_OPERATORS and of:
            derived[str(name)] = DerivedInput(
                input=str(name),
                kind=kind,
                op=op,
                of=of,
                note=_note(entry, where, problems),
            )

    summed = {fact_name for entry in derived.values() for fact_name in entry.of}
    for fact in facts.values():
        if not fact.inputs and fact.name not in summed:
            problems.append(
                f"facts.{fact.name} must map to an input on side a, side b or both, "
                "or be a fact of a derived input"
            )

    fixed: dict[str, FixedInput] = {}
    raw_fixed = raw.get("fixed", {})
    if not isinstance(raw_fixed, dict):
        problems.append("fixed must be a mapping from input name to value")
        raw_fixed = {}
    for name, value in raw_fixed.items():
        where = f"fixed.{name}"
        entry = _keys(value, _FIXED_KEYS, where, problems)
        if entry is None:
            continue
        kind = entry.get("kind")
        if kind not in KINDS:
            problems.append(f"{where}.kind must be one of {', '.join(KINDS)}")
            continue
        normalized = normalize_value(kind, entry.get("value"))
        if normalized is None:
            problems.append(f"{where}.value must be a {kind} value")
            continue
        fixed[str(name)] = FixedInput(
            input=str(name),
            kind=kind,
            value=normalized,
            note=_note(entry, where, problems),
        )

    constraints: list[LinkConstraint] = []
    raw_constraints = raw.get("constraints", [])
    if not isinstance(raw_constraints, list):
        problems.append("constraints must be a list")
        raw_constraints = []
    fact_kinds = {name: fact.kind for name, fact in facts.items()}
    for index, value in enumerate(raw_constraints):
        where = f"constraints[{index}]"
        entry = _keys(value, _CONSTRAINT_KEYS, where, problems)
        if entry is None or "rule" not in entry:
            continue
        try:
            expression = parse_constraint(entry["rule"], fact_kinds)
        except ExpressionError as error:
            problems.append(f"{where}: {error}")
            continue
        constraints.append(
            LinkConstraint(
                rule=entry["rule"],
                note=_note(entry, where, problems),
                expression=expression,
            )
        )

    combines = {side.combine for side in sides.values() if side is not None}
    value_mode = combines == {VALUE}
    if VALUE in combines and len(combines) > 1:
        problems.append(
            "a.combine and b.combine must both be 'value', or both be 'any' or 'all': "
            "a number cannot be compared with a yes/no result"
        )
    tolerance = Decimal(0)
    if "tolerance" in raw:
        found = to_decimal(raw["tolerance"])
        if found is None or found < 0:
            problems.append("link.tolerance must be a number that is not negative")
        elif not value_mode:
            problems.append(
                "link.tolerance applies only when both sides have combine 'value'"
            )
        else:
            tolerance = found

    max_combinations, seed = DEFAULT_MAX_COMBINATIONS, DEFAULT_SEED
    if "sampling" in raw:
        sampling = _keys(raw["sampling"], _SAMPLING_KEYS, "sampling", problems) or {}
        if "max_combinations" in sampling:
            value = sampling["max_combinations"]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 1 <= value <= MAX_COMBINATIONS_LIMIT
            ):
                problems.append(
                    "sampling.max_combinations must be an integer from 1 to "
                    f"{MAX_COMBINATIONS_LIMIT}"
                )
            else:
                max_combinations = value
        if "seed" in sampling:
            value = sampling["seed"]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                problems.append("sampling.seed must be a non-negative integer")
            else:
                seed = value

    # One input has one supplier: a fact, a derived input or a fixed input.
    suppliers: dict[tuple[str, str], str] = {}
    for fact in facts.values():
        for side, name in fact.inputs.items():
            key = (side, name)
            if key in suppliers:
                problems.append(
                    f"input '{name}' on side {side} is mapped by both "
                    f"{suppliers[key]} and facts.{fact.name}"
                )
            suppliers[key] = f"facts.{fact.name}"
    for group, entries in (("derived", derived), ("fixed", fixed)):
        for name in entries:
            taken = [
                suppliers[(side, name)] for side in SIDES if (side, name) in suppliers
            ]
            if taken:
                problems.append(
                    f"input '{name}' is supplied by both {taken[0]} and {group}"
                )
            for side in SIDES:
                suppliers[(side, name)] = group

    if problems:
        raise LinkError(problems)

    normalized = {
        "format": LINK_FORMAT,
        "id": link_id,
        "question": question,
        "entity": entity,
        "period": {key: period[key] for key in ("period_kind", "start", "end")},
        "sides": {
            key: {
                "label": side.label,
                "module": side.module,
                "outputs": list(side.outputs),
                "combine": side.combine,
                "guards": list(side.guards),
                "note": side.note,
            }
            for key, side in sides.items()
        },
        "facts": {
            name: {
                "kind": fact.kind,
                "inputs": dict(fact.inputs),
                "values": None if fact.values is None else list(fact.values),
                "step": canonical_decimal(fact.step),
                "min": None
                if fact.minimum is None
                else canonical_decimal(fact.minimum),
                "max": None
                if fact.maximum is None
                else canonical_decimal(fact.maximum),
                "note": fact.note,
            }
            for name, fact in facts.items()
        },
        "derived": {
            name: {"kind": d.kind, "op": d.op, "of": list(d.of), "note": d.note}
            for name, d in derived.items()
        },
        "fixed": {
            name: {"kind": f.kind, "value": f.value, "note": f.note}
            for name, f in fixed.items()
        },
        "constraints": [{"rule": c.rule, "note": c.note} for c in constraints],
        "sampling": {"max_combinations": max_combinations, "seed": seed},
    }
    if value_mode:
        # Only a value link has this key, so the hash of a yes/no link is unchanged.
        normalized["tolerance"] = canonical_decimal(tolerance)
    return Link(
        id=link_id,
        question=question,
        entity=entity,
        period={key: period[key] for key in ("period_kind", "start", "end")},
        sides=sides,
        facts=facts,
        derived=derived,
        fixed=fixed,
        constraints=tuple(constraints),
        max_combinations=max_combinations,
        seed=seed,
        sha256=hashlib.sha256(_canonical(normalized).encode()).hexdigest(),
        tolerance=tolerance,
    )


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    # A repeated key would replace a fact or a constraint without a trace.
    mapping: dict[str, Any] = {}
    for key, value in pairs:
        if key in mapping:
            raise ValueError(f"the key '{key}' appears twice in one mapping")
        mapping[key] = value
    return mapping


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that refuses a key that appears twice in a mapping."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise ValueError(f"the key '{key}' appears twice in one mapping")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def read_document(path: Path, what: str = "link") -> object:
    """Parse a ``.json``, ``.yaml`` or ``.yml`` file; refuse a repeated key."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise LinkError(
            [f"cannot read the {what} file {path}: {error.strerror}"]
        ) from error
    if len(raw) > MAX_LINK_BYTES:
        raise LinkError(
            [f"the {what} file {path} is larger than {MAX_LINK_BYTES} bytes"]
        )
    try:
        text = raw.decode("utf-8")
        if path.suffix == ".json":
            return json.loads(text, object_pairs_hook=_unique_pairs)
        if path.suffix in (".yaml", ".yml"):
            return yaml.load(text, Loader=_UniqueKeyLoader)  # noqa: S506 - a SafeLoader
    except (
        UnicodeError,
        ValueError,
        TypeError,
        RecursionError,
        yaml.YAMLError,
    ) as error:
        raise LinkError([f"cannot parse the {what} file {path}: {error}"]) from error
    raise LinkError([f"the {what} file {path} must end in .json, .yaml or .yml"])


def load_link(path: Path) -> Link:
    """Read and validate a link file (``.json``, ``.yaml`` or ``.yml``)."""

    return parse_link(read_document(path))

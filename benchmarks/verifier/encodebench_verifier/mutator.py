"""Versioned, seeded single-edit mutator over known-good RuleSpec artifacts.

Every mutation is one edit applied to the *parsed* YAML document, after which
both the untouched document (the control) and the edited one (the defective
artifact) are re-serialised through the same canonical dumper. That gives two
guarantees the encoder-side text splice in the 2026-09-17 pilot could not:
the two artifacts differ in exactly one leaf, and the defective artifact is
always well-formed YAML (a line deleted from a quoted multi-line scalar is
not).

What is and is not touched:

* ``amount_changed``, ``boundary_flipped``, ``conjunct_dropped`` and
  ``polarity_swapped`` edit only ``formula`` / ``value`` strings under
  ``rules``.
* ``date_or_period_wrong`` edits ``rules[i].versions[j].effective_from`` or
  ``rules[i].period`` — the two fields that *are* the effective date and the
  period. It never touches proof excerpts, source hashes, citations,
  ``source_verification`` or any other metadata date.
* ``entity_wrong`` edits ``rules[i].entity``.

Detectability guards: a planted defect must be visible to a reader of the
provision window plus the artifact. Amounts must occur verbatim in the
window; effective dates only move when the window states the original year;
periods and entities only change when the window mentions the original.

Bump :data:`MUTATOR_VERSION` for any change to candidate selection, edit
arithmetic, guards or the canonical dump — boards refuse to fold runs whose
suites disagree on it.
"""

from __future__ import annotations

import copy
import datetime
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from . import DEFECT_KINDS
from .canonical import dump_yaml_document, load_yaml_document
from .cases import Locator

MUTATOR_VERSION = "1.0.0"

_YEAR_RE = re.compile(r"^(19|20)\d\d$")
_NUMBER_RE = re.compile(r"(?<![\w.#:/-])(\d{2,}(?:\.\d+)?|0\.\d+)(?![\w/-])")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AND_RE = re.compile(r"\band\b")
_OR_RE = re.compile(r"\bor\b")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_BOUNDARY_RE = re.compile(r">=|<=|(?<![-=<>])>(?!=)|(?<![<>])<(?![=>-])")

_BOUNDARY_FLIP = {">=": ">", ">": ">=", "<=": "<", "<": "<="}
_PERIOD_SWAP = {"Year": "Month", "Month": "Year", "Week": "Month", "Day": "Month"}
_PERIOD_WORDS = {
    "Year": ("year", "annual", "annually", "yearly"),
    "Month": ("month", "monthly"),
    "Week": ("week", "weekly"),
    "Day": ("day", "daily"),
}
_ENTITY_WORDS = {
    "Person": ("individual", "person", "taxpayer", "employee", "child", "applicant"),
    "Household": ("household",),
    "TaxUnit": ("taxpayer", "joint return", "tax unit", "spouse"),
    "Family": ("family",),
    "Employer": ("employer",),
    "Business": ("business", "trade"),
    "Asset": ("asset", "property", "vehicle", "resource"),
    "Payment": ("payment",),
    "TanfUnit": ("assistance unit", "tanf", "assistance group"),
    "AssistanceUnit": ("assistance unit", "assistance group"),
    "Corporation": ("corporation",),
}
_CANONICAL_ENTITIES = ("Person", "Household", "TaxUnit", "Family", "Employer")


class MutationError(ValueError):
    """The artifact could not be parsed as a RuleSpec document with rules."""


@dataclass(frozen=True)
class Mutation:
    kind: str
    locator: Locator
    control_document: Any
    defective_document: Any

    @property
    def control_text(self) -> str:
        return dump_yaml_document(self.control_document)

    @property
    def defective_text(self) -> str:
        return dump_yaml_document(self.defective_document)


# -- document walking -------------------------------------------------------


def parse_artifact(artifact_text: str) -> dict[str, Any]:
    try:
        document = load_yaml_document(artifact_text)
    except Exception as exc:  # noqa: BLE001 - reported as a mutation error
        raise MutationError(f"artifact is not valid YAML: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("rules"), list):
        raise MutationError("artifact has no `rules` list")
    if not any(isinstance(rule, dict) for rule in document["rules"]):
        raise MutationError("artifact has no rule objects")
    return document


def _rule_meta(document: dict[str, Any], rule_index: int) -> tuple[int, Optional[str]]:
    rule = document["rules"][rule_index]
    name = rule.get("name") if isinstance(rule, dict) else None
    return rule_index, str(name) if name is not None else None


def iter_formula_targets(
    document: dict[str, Any],
) -> Iterator[tuple[int, str, list[Any], Any]]:
    """Yield ``(rule_index, path, container_and_key, text)`` for every
    ``formula`` / ``value`` leaf under ``rules``.

    ``container_and_key`` is ``[container, key]`` so a caller can assign back
    into the (deep-copied) document.
    """

    def walk(node: Any, rule_index: int, path: str) -> Iterator:
        if isinstance(node, dict):
            for key, value in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                if (
                    key in ("formula", "value")
                    and isinstance(value, (str, int, float))
                    and not isinstance(value, bool)
                ):
                    yield rule_index, child_path, [node, key], value
                else:
                    yield from walk(value, rule_index, child_path)
        elif isinstance(node, list):
            for position, value in enumerate(node):
                yield from walk(value, rule_index, f"{path}[{position}]")

    for rule_index, rule in enumerate(document["rules"]):
        if isinstance(rule, dict):
            yield from walk(rule, rule_index, f"rules[{rule_index}]")


# -- helpers ------------------------------------------------------------------


def _normalise_provision(provision: str) -> str:
    return provision.replace(",", "").lower()


def _mentions(provision_lower: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}", provision_lower) for word in words)


def _depth0_matches(text: str, pattern: re.Pattern[str]) -> list[re.Match[str]]:
    """Matches of ``pattern`` at parenthesis/bracket depth zero."""

    depth = 0
    depth_at: list[int] = []
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        depth_at.append(depth)
    return [m for m in pattern.finditer(text) if depth_at[m.start()] == 0]


def _first_identifier(text: str) -> Optional[str]:
    for match in _IDENT_RE.finditer(text):
        word = match.group(0)
        if word not in ("not", "and", "or", "if", "else", "in", "min", "max"):
            return word
    return None


def _operand_before(text: str, position: int) -> Optional[str]:
    left = text[:position]
    matches = list(_IDENT_RE.finditer(left))
    return matches[-1].group(0) if matches else None


def _perturb_number(token: str, forbidden: Callable[[str], bool]) -> Optional[str]:
    value = float(token)
    decimals = len(token.split(".")[1]) if "." in token else 0
    if value >= 1:
        candidates = [value * 1.25, value * 1.5, value * 0.8, value + 1]
    else:
        candidates = [min(value + 0.05, 0.99), min(value * 1.5, 0.99), value / 2]
    for candidate in candidates:
        if decimals:
            new = f"{candidate:.{decimals}f}"
        else:
            new = str(int(round(candidate)))
        if new != token and not forbidden(new):
            return new
    return None


# -- kind implementations -----------------------------------------------------


def _mutate_amount(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    provision_norm = _normalise_provision(provision)
    candidates: list[tuple[int, str, list[Any], str, re.Match[str]]] = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        text = str(raw)
        for match in _NUMBER_RE.finditer(text):
            token = match.group(1)
            if _YEAR_RE.match(token):
                continue
            if token not in provision_norm:
                continue
            candidates.append((rule_index, path, slot, text, match))
    rng.shuffle(candidates)
    for rule_index, path, slot, text, match in candidates:
        token = match.group(1)
        new_token = _perturb_number(token, lambda t: t in provision_norm)
        if new_token is None:
            continue
        container, key = slot
        new_text = text[: match.start(1)] + new_token + text[match.end(1) :]
        original = container[key]
        if isinstance(original, (int, float)) and not isinstance(original, bool):
            container[key] = (
                type(original)(new_token) if "." not in new_token else float(new_token)
            )
        else:
            container[key] = new_text
        index, name = _rule_meta(document, rule_index)
        return Locator(
            path=path,
            rule_index=index,
            rule_name=name,
            detail=f"amount {token} -> {new_token}",
            before=token,
            after=new_token,
            token=new_token,
        )
    return None


def _mutate_boundary(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    candidates = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        if not isinstance(raw, str):
            continue
        for match in _BOUNDARY_RE.finditer(raw):
            candidates.append((rule_index, path, slot, raw, match))
    if not candidates:
        return None
    rule_index, path, slot, text, match = rng.choice(candidates)
    operator = match.group(0)
    flipped = _BOUNDARY_FLIP[operator]
    container, key = slot
    container[key] = text[: match.start()] + flipped + text[match.end() :]
    index, name = _rule_meta(document, rule_index)
    return Locator(
        path=path,
        rule_index=index,
        rule_name=name,
        detail=f"boundary {operator} -> {flipped}",
        before=operator,
        after=flipped,
        token=_operand_before(text, match.start()),
    )


def _mutate_conjunct(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    candidates = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        if not isinstance(raw, str):
            continue
        flat = " ".join(raw.split())
        # ``if ...:`` / ``else:`` formulas interleave branches with conditions;
        # splitting them at depth zero could delete a branch head. Skip them.
        if ":" in flat:
            continue
        ands = _depth0_matches(flat, _AND_RE)
        if not ands:
            continue
        candidates.append((rule_index, path, slot, flat, ands))
    if not candidates:
        return None
    rule_index, path, slot, flat, ands = rng.choice(candidates)
    conjunct_count = len(ands) + 1
    drop = rng.randrange(conjunct_count)
    if drop == 0:
        removed = flat[: ands[0].start()]
        new_text = flat[ands[0].end() :]
    else:
        end = ands[drop].start() if drop < len(ands) else len(flat)
        removed = flat[ands[drop - 1].end() : end]
        new_text = flat[: ands[drop - 1].start()] + flat[end:]
    new_text = " ".join(new_text.split())
    if not new_text:
        return None
    container, key = slot
    container[key] = new_text
    index, name = _rule_meta(document, rule_index)
    dropped = " ".join(removed.split())
    return Locator(
        path=path,
        rule_index=index,
        rule_name=name,
        detail=f"dropped conjunct {drop + 1} of {conjunct_count}",
        before=dropped,
        after=None,
        token=_first_identifier(dropped),
    )


def _mutate_polarity(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    candidates = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        if not isinstance(raw, str):
            continue
        for pattern in (_AND_RE, _OR_RE):
            for match in pattern.finditer(raw):
                candidates.append((rule_index, path, slot, raw, match))
    if not candidates:
        return None
    rule_index, path, slot, text, match = rng.choice(candidates)
    word = match.group(0)
    swapped = "or" if word == "and" else "and"
    container, key = slot
    container[key] = text[: match.start()] + swapped + text[match.end() :]
    index, name = _rule_meta(document, rule_index)
    return Locator(
        path=path,
        rule_index=index,
        rule_name=name,
        detail=f"polarity {word} -> {swapped}",
        before=word,
        after=swapped,
        token=_operand_before(text, match.start()),
    )


def _shift_year(date_text: str) -> Optional[str]:
    match = _DATE_RE.match(date_text)
    if not match:
        return None
    year, month, day = (int(part) for part in match.groups())
    if year <= 1:
        return None
    new_year = year + 1
    if month == 2 and day == 29:
        day = 28
    return f"{new_year:04d}-{month:02d}-{day:02d}"


def _mutate_date_or_period(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    provision_lower = provision.lower()
    candidates: list[tuple[str, int, str, list[Any], str, str]] = []
    for rule_index, rule in enumerate(document["rules"]):
        if not isinstance(rule, dict):
            continue
        versions = rule.get("versions")
        if isinstance(versions, list):
            for position, version in enumerate(versions):
                if not isinstance(version, dict):
                    continue
                effective_raw = version.get("effective_from")
                if isinstance(effective_raw, datetime.date):
                    effective = effective_raw.isoformat()
                elif isinstance(effective_raw, str):
                    effective = effective_raw
                else:
                    continue
                shifted = _shift_year(effective)
                if shifted is None:
                    continue
                # Detectability: the window must state the original year and
                # must not also state the shifted year.
                if effective[:4] not in provision or shifted[:4] in provision:
                    continue
                candidates.append(
                    (
                        "effective_from",
                        rule_index,
                        f"rules[{rule_index}].versions[{position}].effective_from",
                        [version, "effective_from"],
                        effective,
                        shifted,
                    )
                )
        period = rule.get("period")
        if isinstance(period, str) and period in _PERIOD_SWAP:
            replacement = _PERIOD_SWAP[period]
            if _mentions(provision_lower, _PERIOD_WORDS[period]) and not _mentions(
                provision_lower, _PERIOD_WORDS[replacement]
            ):
                candidates.append(
                    (
                        "period",
                        rule_index,
                        f"rules[{rule_index}].period",
                        [rule, "period"],
                        period,
                        replacement,
                    )
                )
    if not candidates:
        return None
    field, rule_index, path, slot, before, after = rng.choice(candidates)
    container, key = slot
    if isinstance(container[key], datetime.date):
        # Keep the leaf's type so the only difference is the value itself.
        container[key] = datetime.date.fromisoformat(after)
    else:
        container[key] = after
    index, name = _rule_meta(document, rule_index)
    return Locator(
        path=path,
        rule_index=index,
        rule_name=name,
        detail=f"{field} {before} -> {after}",
        before=before,
        after=after,
        token=after,
    )


def _mutate_entity(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    provision_lower = provision.lower()
    present = {
        str(rule.get("entity"))
        for rule in document["rules"]
        if isinstance(rule, dict) and isinstance(rule.get("entity"), str)
    }
    candidates = []
    for rule_index, rule in enumerate(document["rules"]):
        if not isinstance(rule, dict):
            continue
        entity = rule.get("entity")
        if not isinstance(entity, str):
            continue
        words = _ENTITY_WORDS.get(entity)
        if not words or not _mentions(provision_lower, words):
            continue
        pool = [
            other
            for other in sorted(present) + list(_CANONICAL_ENTITIES)
            if other != entity
            and not _mentions(
                provision_lower, _ENTITY_WORDS.get(other, (other.lower(),))
            )
        ]
        # De-duplicate while keeping artifact-local entities first.
        seen: list[str] = []
        for other in pool:
            if other not in seen:
                seen.append(other)
        if not seen:
            continue
        candidates.append((rule_index, [rule, "entity"], entity, seen))
    if not candidates:
        return None
    rule_index, slot, before, pool = rng.choice(candidates)
    after = rng.choice(pool)
    container, key = slot
    container[key] = after
    index, name = _rule_meta(document, rule_index)
    return Locator(
        path=f"rules[{rule_index}].entity",
        rule_index=index,
        rule_name=name,
        detail=f"entity {before} -> {after}",
        before=before,
        after=after,
        token=after,
    )


_KIND_IMPLEMENTATIONS: dict[str, Callable[..., Optional[Locator]]] = {
    "amount_changed": _mutate_amount,
    "boundary_flipped": _mutate_boundary,
    "conjunct_dropped": _mutate_conjunct,
    "polarity_swapped": _mutate_polarity,
    "date_or_period_wrong": _mutate_date_or_period,
    "entity_wrong": _mutate_entity,
}

assert tuple(_KIND_IMPLEMENTATIONS) == DEFECT_KINDS


# -- public API ---------------------------------------------------------------


def mutate(
    artifact_text: str,
    provision_window: str,
    kind: str,
    *,
    rng: random.Random,
) -> Optional[Mutation]:
    """Plant one ``kind`` defect. ``None`` when the artifact offers no site.

    ``provision_window`` must be the text the judges will actually see (after
    truncation), since the detectability guards are checked against it.
    """

    if kind not in _KIND_IMPLEMENTATIONS:
        raise ValueError(f"unknown defect kind {kind!r}")
    control = parse_artifact(artifact_text)
    defective = copy.deepcopy(control)
    locator = _KIND_IMPLEMENTATIONS[kind](defective, provision_window, rng)
    if locator is None:
        return None
    if dump_yaml_document(control) == dump_yaml_document(defective):
        return None
    # Round-trip: the defective artifact must still parse to the edited tree.
    reparsed = load_yaml_document(dump_yaml_document(defective))
    if reparsed != defective:
        raise MutationError(f"canonical dump of the {kind} mutation did not round-trip")
    return Mutation(
        kind=kind,
        locator=locator,
        control_document=control,
        defective_document=defective,
    )


def leaf_differences(left: Any, right: Any, path: str = "") -> list[str]:
    """Paths of leaves that differ between two parsed documents (test aid)."""

    if isinstance(left, dict) and isinstance(right, dict):
        diffs: list[str] = []
        for key in sorted(set(left) | set(right), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in left or key not in right:
                diffs.append(child)
            else:
                diffs.extend(leaf_differences(left[key], right[key], child))
        return diffs
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [path]
        diffs = []
        for position, (a, b) in enumerate(zip(left, right)):
            diffs.extend(leaf_differences(a, b, f"{path}[{position}]"))
        return diffs
    return [] if left == right else [path]

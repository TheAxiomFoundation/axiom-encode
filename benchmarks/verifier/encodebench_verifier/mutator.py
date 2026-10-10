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
  ``rules[i].period``, the two fields that *are* the effective date and the
  period. It never touches proof excerpts, source hashes, citations,
  ``source_verification`` or any other metadata date.
* ``entity_wrong`` edits ``rules[i].entity``.

Detectability guards: a planted defect must be visible to a reader of the
provision window plus the artifact. An amount must equal a number the window
states (numeric equality on whole numbers, so ``60000`` matches ``$60,000.00``
but ``200`` does not match inside ``2008`` and ``11`` does not match inside
``3211(b)``), and its replacement must not; effective dates only move when
the window states the original year as a word of its own and not the shifted
one; periods and entities only change when the window mentions the original
and not the replacement. Conjuncts are only dropped from pure conjunctions:
a formula with a top-level ``or`` or an ``if``/``else`` is left alone.
Nothing inside a string literal is edited, and a formula with ``#`` outside
a literal is left alone by every formula kind.

Version history:

* 1.0.0 matched amounts and years by substring; two of its 180 pairs were
  undetectable for that reason. That build was superseded by a full 1.0.1
  rebuild, not filtered.
* 1.0.1 used numeric equality and word-bounded years and refused conjunct
  drops next to a top-level ``or``. Its period and entity guards matched word
  prefixes ("daylight" counted as "day"); four of the committed synthetic
  board's 30 date-or-period pairs were undetectable for that reason and are
  dropped from the board by a recorded filter. A dropped conjunct also
  reflowed its whole formula onto one line.
* 1.0.2 matches period and entity words whole (plurals allowed), refuses
  dotted codes (``7202.11.10``) as amounts and ``<<``/``>>`` as boundaries,
  ignores ``and``/``or`` inside string literals, refuses conjunct drops in
  colon-free conditionals (``x if c else y``, ``if c then x else y``), cuts a
  dropped conjunct from the original text so the formula keeps its layout,
  and refuses an edit that changes more than one leaf (YAML aliases).
* 1.0.3 edits nothing inside a string literal (1.0.2 still flipped a
  comparison or changed a number there), leaves alone any formula with a
  ``#`` outside a literal (a comment or a reference: this module does not
  parse which), does not read ``>>=``/``<<=`` as a comparison, knows the
  irregular plurals "families" and "people", and keeps a formula's leading
  and trailing whitespace when it drops the first or last conjunct.
* 1.0.4 treats an int or float leaf as one amount: a site must be the whole
  number, so a float rendered in exponent form (``1e+16``) offers no site
  (1.0.3 read its exponent as the amount and replaced the whole value).

Bump :data:`MUTATOR_VERSION` for any change to candidate selection, edit
arithmetic, guards or the canonical dump: boards refuse to fold runs whose
suites disagree on it.
"""

from __future__ import annotations

import copy
import datetime
import math
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Iterator, Optional

from . import DEFECT_KINDS
from .canonical import dump_yaml_document, load_yaml_document
from .cases import Locator

MUTATOR_VERSION = "1.0.4"

_YEAR_RE = re.compile(r"^(19|20)\d\d$")
# A number inside formula text. A dotted code such as ``"7202.11.10.00"`` is
# not an amount: the lookahead refuses a match followed by ``.<digit>``.
_NUMBER_RE = re.compile(r"(?<![\w.#:/-])(\d{2,}(?:\.\d+)?|0\.\d+)(?![\w/-]|\.\d)")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_AND_RE = re.compile(r"\band\b")
_OR_RE = re.compile(r"\bor\b")
_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
# A comparison operator; ``->``, ``=>``, ``<>``, ``<<``, ``>>``, ``>>=`` and
# ``<<=`` are not.
_BOUNDARY_RE = re.compile(
    r"(?<![<>])>=|(?<![<>])<=|(?<![-=<>])>(?![=>])|(?<![<>])<(?![=<>-])"
)
_CONDITIONAL_RE = re.compile(r"\b(?:if|else|then)\b")

_BOUNDARY_FLIP = {">=": ">", ">": ">=", "<=": "<", "<": "<="}
_PERIOD_SWAP = {"Year": "Month", "Month": "Year", "Week": "Month", "Day": "Month"}
_PERIOD_WORDS = {
    "Year": ("year", "annual", "annually", "yearly"),
    "Month": ("month", "monthly"),
    "Week": ("week", "weekly"),
    "Day": ("day", "daily"),
}
_ENTITY_WORDS = {
    "Person": (
        "individual",
        "person",
        "taxpayer",
        "employee",
        "people",
        "child",
        "children",
        "applicant",
    ),
    "Household": ("household",),
    "TaxUnit": ("taxpayer", "joint return", "tax unit", "spouse"),
    "Family": ("family", "families"),
    "Employer": ("employer",),
    "Business": ("business", "businesses", "trade"),
    "Asset": ("asset", "property", "properties", "vehicle", "resource"),
    "Payment": ("payment",),
    "TanfUnit": ("assistance unit", "tanf", "assistance group"),
    "AssistanceUnit": ("assistance unit", "assistance group"),
    "Corporation": ("corporation",),
}
_CANONICAL_ENTITIES = ("Person", "Household", "TaxUnit", "Family", "Employer")


class MutationError(ValueError):
    """The artifact could not be parsed as a RuleSpec document with rules."""


class RoundTripError(MutationError):
    """One kind's edit did not survive the canonical dump; other kinds may."""


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


_PROVISION_NUMBER_RE = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)*")


def provision_numbers(provision: str) -> set[Decimal]:
    """Every number the provision window states, as normalised decimals."""

    numbers: set[Decimal] = set()
    for match in _PROVISION_NUMBER_RE.finditer(provision):
        if match.group(0).count(".") > 1:
            continue  # a dotted code such as 7202.11.10, not a number
        raw = match.group(0).replace(",", "").rstrip(".")
        try:
            numbers.add(Decimal(raw).normalize())
        except InvalidOperation:
            continue
    return numbers


def _as_decimal(token: str) -> Optional[Decimal]:
    try:
        return Decimal(token).normalize()
    except InvalidOperation:
        return None


def _states_year(provision: str, year: str) -> bool:
    return (
        re.search(rf"(?<![\w.,]){re.escape(year)}(?![\w]|[.,]\d)", provision)
        is not None
    )


def _mentions(provision_lower: str, words: tuple[str, ...]) -> bool:
    """Whether the window uses one of ``words`` as a whole word (or its plural).

    Whole words only: "daylight" does not mention "day", "personal" does not
    mention "person", "trademark" does not mention "trade".
    """

    return any(
        re.search(rf"\b{re.escape(word)}s?\b", provision_lower) for word in words
    )


def _mask_quotes(text: str) -> str:
    """``text`` with every quoted string literal's contents blanked out.

    Positions are preserved, so a match found in the masked text slices the
    original. Nothing inside a literal (``"Bosnia and Herzegovina"``,
    ``"a > b"``, ``"11"``) is ever treated as an operator or an amount. An
    unterminated quote is left as it is.
    """

    out: list[str] = []
    position, length = 0, len(text)
    while position < length:
        char = text[position]
        if char in "\"'":
            end = position + 1
            while end < length and text[end] != char:
                end += 2 if text[end] == "\\" else 1
            if end < length:
                out.append(char + "_" * (end - position - 1) + char)
                position = end + 1
                continue
        out.append(char)
        position += 1
    return "".join(out)


def _editable(raw: str) -> Optional[str]:
    """The quote-masked text to look for edit sites in, or ``None``.

    A ``#`` outside a string literal may start a comment or sit inside a
    reference; this module does not parse which, and an edit after a comment
    marker would change nothing a reader can check. Such a formula is left
    alone by every formula kind.
    """

    masked = _mask_quotes(raw)
    return None if "#" in masked else masked


def _number_sites(text: str) -> list[re.Match[str]]:
    masked = _editable(text)
    return [] if masked is None else list(_NUMBER_RE.finditer(masked))


def _amount_sites(raw: Any) -> list[re.Match[str]]:
    """Number sites in a formula or value leaf.

    An int or float leaf is one amount, so its only site is the whole number:
    the exponent of ``1e+16`` is not an amount.
    """

    text = str(raw)
    sites = _number_sites(text)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        sites = [m for m in sites if m.start(1) == 0 and m.end(1) == len(text)]
    return sites


def _boundary_sites(raw: str) -> list[re.Match[str]]:
    masked = _editable(raw)
    return [] if masked is None else list(_BOUNDARY_RE.finditer(masked))


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
    stated = provision_numbers(provision)
    candidates: list[tuple[int, str, list[Any], str, re.Match[str]]] = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        text = str(raw)
        for match in _amount_sites(raw):
            token = match.group(1)
            if _YEAR_RE.match(token):
                continue
            if _as_decimal(token) not in stated:
                continue
            candidates.append((rule_index, path, slot, text, match))
    rng.shuffle(candidates)
    for rule_index, path, slot, text, match in candidates:
        token = match.group(1)
        new_token = _perturb_number(token, lambda t: _as_decimal(t) in stated)
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
        for match in _boundary_sites(raw):
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


def _conjunct_sites(raw: str) -> list[re.Match[str]]:
    """Depth-0 ``and`` operators of a pure conjunction (empty if not one).

    Matches are found in the quote-masked text, so their offsets slice
    ``raw`` and an ``and`` inside a string literal is never an operator.
    """

    masked = _editable(raw)
    if masked is None:
        return []
    # ``if ...:`` / ``else:`` formulas interleave branches with conditions;
    # splitting them at depth zero could delete a branch head. Skip them,
    # and the colon-free ``x if c else y`` / ``if c then x else y`` forms.
    if ":" in masked or _depth0_matches(masked, _CONDITIONAL_RE):
        return []
    # ``a and b or c`` parses as ``(a and b) or c``: removing the text
    # between two ``and`` tokens would delete the alternative as well.
    if _depth0_matches(masked, _OR_RE):
        return []
    return _depth0_matches(masked, _AND_RE)


def _drop_conjunct(raw: str, ands: list[re.Match[str]], drop: int) -> tuple[str, str]:
    """``(removed, remaining)`` after dropping conjunct ``drop`` (0-based).

    The original text is cut at the operator positions, so the rest of the
    formula keeps its layout and only the dropped conjunct changes. The
    formula's own leading and trailing whitespace stays (a lost trailing
    newline would also flip a YAML block scalar's chomping indicator).
    """

    lead = raw[: len(raw) - len(raw.lstrip())]
    trail = raw[len(raw.rstrip()) :]
    if drop == 0:
        return raw[: ands[0].start()], lead + raw[ands[0].end() :].lstrip()
    end = ands[drop].start() if drop < len(ands) else len(raw)
    removed = raw[ands[drop - 1].end() : end]
    kept = raw[: ands[drop - 1].start()]
    if drop < len(ands):
        return removed, kept + raw[end:]
    return removed, kept.rstrip() + trail


def _mutate_conjunct(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    candidates = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        if not isinstance(raw, str):
            continue
        ands = _conjunct_sites(raw)
        if ands:
            candidates.append((rule_index, path, slot, raw, ands))
    if not candidates:
        return None
    rule_index, path, slot, raw, ands = rng.choice(candidates)
    conjunct_count = len(ands) + 1
    drop = rng.randrange(conjunct_count)
    removed, new_text = _drop_conjunct(raw, ands, drop)
    if not new_text.strip():
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


def _polarity_sites(raw: str) -> list[re.Match[str]]:
    """Every ``and`` then every ``or``, never inside a string literal."""

    masked = _editable(raw)
    if masked is None:
        return []
    return [m for pattern in (_AND_RE, _OR_RE) for m in pattern.finditer(masked)]


def _mutate_polarity(
    document: dict[str, Any], provision: str, rng: random.Random
) -> Optional[Locator]:
    candidates = []
    for rule_index, path, slot, raw in iter_formula_targets(document):
        if not isinstance(raw, str):
            continue
        for match in _polarity_sites(raw):
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
                if not _states_year(provision, effective[:4]) or _states_year(
                    provision, shifted[:4]
                ):
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
    # Exactly one leaf may differ. YAML anchors and aliases make two paths share
    # one object, so a single assignment would change both: refuse those.
    if len(leaf_differences(control, defective)) != 1:
        return None
    # Round-trip: the defective artifact must still parse to the edited tree.
    reparsed = load_yaml_document(dump_yaml_document(defective))
    if reparsed != defective:
        raise RoundTripError(
            f"canonical dump of the {kind} mutation did not round-trip"
        )
    return Mutation(
        kind=kind,
        locator=locator,
        control_document=control,
        defective_document=defective,
    )


_PATH_TOKEN_RE = re.compile(r"([^.\[\]]+)|\[(\d+)\]")
# How the mutator writes a number: ASCII digits, no leading zeros.
_PLAIN_NUMBER_RE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_RULE_PERIOD_PATH = re.compile(r"rules\[\d+\]\.period")
_RULE_ENTITY_PATH = re.compile(r"rules\[\d+\]\.entity")
_VERSION_DATE_PATH = re.compile(r"rules\[\d+\]\.versions\[\d+\]\.effective_from")


def _leaf(document: Any, path: str) -> Any:
    node = document
    for key, position in _PATH_TOKEN_RE.findall(path):
        node = node[int(position)] if position else node[key]
    return node


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


_NO_AMOUNT = "no amount the window states changes to one it does not"


def _audit_numeric_amount(
    before: Any, after: Any, stated: set[Decimal]
) -> Optional[str]:
    """Audit an amount edit on an int or float ``formula``/``value`` leaf.

    The mutator keeps the leaf's type and writes the perturbed token with the
    token's decimal places; a float re-render may drop trailing zeros (0.30
    reads 0.3) or switch to exponent form (1.125e+16), so the written value is
    compared in plain positional form.
    """

    if isinstance(after, bool) or type(after) is not type(before):
        return "a numeric amount must keep its type"
    if isinstance(after, float) and not math.isfinite(after):
        return "a numeric amount must stay finite"
    written = format(Decimal(repr(after)), "f")
    for match in _amount_sites(before):
        token = match.group(1)
        if _YEAR_RE.match(token) or _as_decimal(token) not in stated:
            continue
        if not _PLAIN_NUMBER_RE.fullmatch(written):
            continue
        if _decimals(written) > _decimals(token):
            continue
        replacement = _as_decimal(written)
        if replacement != _as_decimal(token) and replacement not in stated:
            return None
    return _NO_AMOUNT


def _flat(text: str) -> str:
    return " ".join(text.split())


def audit_planted_edit(
    control_text: str, defective_text: str, provision_window: str, kind: str
) -> Optional[str]:
    """Why this version's guards would refuse a planted edit, or ``None``.

    Re-checks a pair built by an earlier mutator version: the defective
    artifact must differ from its control in exactly one leaf, and that edit
    must sit where this version plants ``kind`` (a formula or value leaf, a
    rule's ``period`` or ``entity``, a version's ``effective_from``) and pass
    this version's guards under the same window (the original stated, the
    replacement not, the site a real operator outside any string literal).
    A suite built under looser guards can then drop the pairs that fail by a
    recorded filter instead of being rebuilt and re-judged. A dropped
    conjunct that differs from this version's cut only in whitespace passes:
    1.0.1 reflowed the formula, which changes layout, not meaning.
    """

    if kind not in _KIND_IMPLEMENTATIONS:
        raise ValueError(f"unknown defect kind {kind!r}")
    control = parse_artifact(control_text)
    defective = parse_artifact(defective_text)
    diffs = leaf_differences(control, defective)
    if len(diffs) != 1:
        return f"changes {len(diffs)} leaves, not one"
    path = diffs[0]
    try:
        before, after = _leaf(control, path), _leaf(defective, path)
    except (KeyError, IndexError, TypeError):
        return f"edit at {path} adds or removes a key"
    lower = provision_window.lower()
    formula_paths = {target[1] for target in iter_formula_targets(control)}

    if kind == "amount_changed":
        if path not in formula_paths:
            return f"edits {path}, not a formula or value"
        stated = provision_numbers(provision_window)
        if isinstance(before, (int, float)) and not isinstance(before, bool):
            return _audit_numeric_amount(before, after, stated)
        if not isinstance(after, str):
            return "a formula amount must stay text"
        old, new = str(before), str(after)
        for match in _number_sites(old):
            token = match.group(1)
            if _YEAR_RE.match(token) or _as_decimal(token) not in stated:
                continue
            head, tail = old[: match.start(1)], old[match.end(1) :]
            if not (new.startswith(head) and new.endswith(tail)):
                continue
            written = new[len(head) : len(new) - len(tail)]
            # The mutator writes plain digits with the token's decimal places.
            if not _PLAIN_NUMBER_RE.fullmatch(written):
                continue
            if _decimals(written) != _decimals(token):
                continue
            replacement = _as_decimal(written)
            if replacement == _as_decimal(token):
                continue  # the same amount, reformatted
            if replacement not in stated:
                return None
        return _NO_AMOUNT

    if kind in ("boundary_flipped", "polarity_swapped", "conjunct_dropped"):
        if path not in formula_paths or not isinstance(before, str):
            return f"edits {path}, not formula text"
        if not isinstance(after, str):
            return f"edit at {path} changes the leaf type"
        if kind == "boundary_flipped":
            for match in _boundary_sites(before):
                flipped = _BOUNDARY_FLIP[match.group(0)]
                if before[: match.start()] + flipped + before[match.end() :] == after:
                    return None
            return "not a flip of a comparison operator"
        if kind == "polarity_swapped":
            for match in _polarity_sites(before):
                swapped = "or" if match.group(0) == "and" else "and"
                if before[: match.start()] + swapped + before[match.end() :] == after:
                    return None
            return "not a swap of an and/or operator outside string literals"
        ands = _conjunct_sites(before)
        if not ands:
            return "formula is not a pure conjunction this version may cut"
        cuts = [_drop_conjunct(before, ands, drop)[1] for drop in range(len(ands) + 1)]
        if after in cuts or _flat(after) in {_flat(cut) for cut in cuts}:
            return None
        return "not one dropped conjunct of the formula"

    if kind == "date_or_period_wrong":
        if _RULE_PERIOD_PATH.fullmatch(path):
            if not isinstance(before, str) or _PERIOD_SWAP.get(before) != after:
                return f"period {before!r} -> {after!r} is not this version's swap"
            if not _mentions(lower, _PERIOD_WORDS[before]):
                return f"window does not state the original period {before}"
            if _mentions(lower, _PERIOD_WORDS[after]):
                return f"window also states the replacement period {after}"
            return None
        if _VERSION_DATE_PATH.fullmatch(path):
            old = before.isoformat() if isinstance(before, datetime.date) else before
            new = after.isoformat() if isinstance(after, datetime.date) else after
            if type(before) is not type(after) or _shift_year(str(old)) != new:
                return f"effective_from {old!r} -> {new!r} is not a one-year shift"
            if not _states_year(provision_window, str(old)[:4]):
                return f"window does not state the original year {str(old)[:4]}"
            if _states_year(provision_window, str(new)[:4]):
                return f"window also states the shifted year {str(new)[:4]}"
            return None
        return f"edits {path}, not a period or effective date"

    # entity_wrong
    if not _RULE_ENTITY_PATH.fullmatch(path) or not isinstance(before, str):
        return f"edits {path}, not a rule's entity"
    words = _ENTITY_WORDS.get(before)
    if not words or not _mentions(lower, words):
        return f"window does not mention the original entity {before}"
    pool = set(_CANONICAL_ENTITIES) | {
        rule.get("entity")
        for rule in control["rules"]
        if isinstance(rule, dict) and isinstance(rule.get("entity"), str)
    }
    if not isinstance(after, str) or after == before or after not in pool:
        return f"entity {before!r} -> {after!r} is not a replacement"
    if _mentions(lower, _ENTITY_WORDS.get(after, (after.lower(),))):
        return f"window also mentions the replacement entity {after}"
    return None


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

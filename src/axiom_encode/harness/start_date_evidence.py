"""Start-date evidence for a date-silent source unit (#2212).

A source unit that prints no date gives no start date for its rules. This
module holds the one rule for such a unit, shared by the encoder prompt and by
the ``complete-source-unit:effective-date/start-unsupported`` check:

    An ``effective_from`` (including its ``from`` alias) is *supported* when
    it is printed in the unit text or in the start-date proof evidence of its
    own version, equals the corpus ``expression_date`` or ``source_as_of``,
    equals the evidence floor, or is the ``0001-01-01`` scaffold. A date
    copied from a neighbouring module is not support.

The *evidence floor* is the first day of the first full calendar month that
starts on or after ``source_as_of``. It is a proposed calendar convention, not
a claimed legal commencement date: the calculation only rounds a date. It
establishes no legal commencement, first corpus custody or text continuity,
and the other supported dates can lie before it.

The check is the narrow form of that rule, to keep false refusals low. It
looks only at a unit that is all of: a corpus ``manual`` document, date-silent
and without amendment context. There every start date must be one of the
supported dates above: the scaffold, the ``expression_date``, ``source_as_of``,
the evidence floor, or a date printed in the start-date proof evidence of
that version. Any other start is refused, also one that lies after the
snapshot.

*Start-date proof evidence* of a version is the evidence text of a proof atom
that belongs to the same rule and has ``kind: effective_period``. A date quoted
by any other atom (another kind, or an atom of another rule) is not support:
such an atom proves something else, for example an amount of a sibling
provision. The atom's ``path`` is not examined: public candidates anchor their
``effective_period`` atoms at ``versions[i].formula`` (replay control
``41648809989-control``), and a path rule would refuse them. The printed date
itself ties the atom to the version that starts on it. The atoms and evidence
texts are read with the proof validator's own reader, so every spelling that
proof validation authenticates (``metadata.proof`` or ``proof``;
``source.excerpt`` or ``source.quote``) is read here.

Imports can only add supported dates; they never take a module out of the
check. A start that equals a version start of an export the module verifiably
imports passes (whether such a date is support is an open decision of #2212).
An import that cannot be verified contributes nothing.

``effective_to``, ``values`` keys and companion test periods are not examined.
Rule-level starts are examined when the engine uses an implicit version: the
``versions`` list is absent or empty and the rule has a formula or start.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import yaml

from axiom_encode.checks import IssueSubject, ValidationIssue, declare_check
from axiom_encode.harness.proof_validator import (
    proof_source_evidence_texts,
    rule_proof_atoms,
)

SCAFFOLD_START = date(1, 1, 1)
MANUAL_DOCUMENT_CLASS = "manual"

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class StartDateEvidence:
    """The unit's corpus metadata dates, ``source_as_of`` and ``expression_date``.

    ``problems`` names every date field that is missing, malformed or
    contradictory. A field with a problem is ``None`` and supports nothing.
    ``amendment_context`` is true when the run binds amendment or commencement
    documents; their dates are evidence this module does not read, so the
    check leaves such a unit alone.
    """

    source_as_of: date | None
    expression_date: date | None
    problems: tuple[str, ...] = ()
    amendment_context: bool = False

    @property
    def evidence_floor(self) -> date | None:
        return None if self.source_as_of is None else evidence_floor(self.source_as_of)

    @property
    def supported_corpus_dates(self) -> frozenset[date]:
        """The scaffold and every date the corpus metadata itself supports."""

        dates = {SCAFFOLD_START, self.expression_date, self.source_as_of}
        dates.add(self.evidence_floor)
        return frozenset(item for item in dates if item is not None)

    @property
    def preferred_start(self) -> date:
        """The start to prefer when nothing prints a date.

        The evidence floor, or the ``expression_date`` when that is later: the
        corpus then says that the text is not in force before it. Without a
        snapshot, the ``expression_date``; without either, the scaffold.
        """

        floor = self.evidence_floor
        if floor is None:
            return self.expression_date or SCAFFOLD_START
        if self.expression_date is not None and self.expression_date > floor:
            return self.expression_date
        return floor


def evidence_floor(source_as_of: date) -> date:
    """First day of the first full calendar month on or after ``source_as_of``.

    A snapshot taken on the first day of a month covers that whole month, so
    the floor is that day. Any later day gives the first day of the next month.
    """

    if source_as_of.day == 1:
        return source_as_of
    return (source_as_of.replace(day=1) + timedelta(days=32)).replace(day=1)


def _strict_iso_date(value: object) -> date | None:
    """``YYYY-MM-DD`` and nothing else; a timestamp or a blank is not a date."""

    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def start_date_evidence_from_metadata(
    source_metadata: Mapping[str, Any] | None,
    *,
    amendment_context: bool = False,
) -> StartDateEvidence:
    """Read ``source_as_of`` and ``expression_date`` from a resolver attestation.

    The attestation states each field twice (top level and ``row``). Both
    copies must be the same strict ISO date; anything else is a problem and
    the field supports nothing.
    """

    attestation = (
        source_metadata.get("source_attestation")
        if isinstance(source_metadata, Mapping)
        else None
    )
    if not isinstance(attestation, Mapping):
        return StartDateEvidence(
            None,
            None,
            ("source_attestation is missing",),
            amendment_context,
        )
    row = attestation.get("row")
    values: dict[str, date | None] = {}
    problems: list[str] = []
    for name in ("source_as_of", "expression_date"):
        stated = [
            holder[name]
            for holder in (attestation, row)
            if isinstance(holder, Mapping) and holder.get(name) is not None
        ]
        parsed = {_strict_iso_date(item) for item in stated}
        if not stated:
            problems.append(f"source_attestation.{name} is missing")
            values[name] = None
        elif None in parsed:
            problems.append(f"source_attestation.{name} is not a YYYY-MM-DD date")
            values[name] = None
        elif len(parsed) != 1:
            problems.append(
                f"source_attestation.{name} and source_attestation.row.{name} differ"
            )
            values[name] = None
        else:
            values[name] = parsed.pop()
    return StartDateEvidence(
        values["source_as_of"],
        values["expression_date"],
        tuple(problems),
        amendment_context,
    )


# --------------------------------------------------------------------------
# Dates in text
# --------------------------------------------------------------------------

# "may" is left out of the case-insensitive month words: the modal verb is
# everywhere in a manual ("any 2 may be excluded"). The month May is matched
# with its capital letter only.
_MONTH_WORD = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\.?"
)
_ORDINAL = r"(?:st|nd|rd|th)?"


def _month_forms(month: str) -> tuple[str, ...]:
    """July 17, 2026; July 17; 17 July 2026; 11 Aug 25; July 2026."""

    return (
        rf"\b{month}\s+\d{{1,2}}{_ORDINAL}(?![\d)])\b",
        rf"(?<![\d/-])\d{{1,2}}{_ORDINAL}\s+{month},?\s+\d{{2,4}}\b",
        rf"\b{month},?\s+\d{{4}}\b",
    )


_ANY_DATE = re.compile(
    "|".join(
        (
            # 2026-07-17, 7/17/2026, 07-17-26, 07/2026
            r"(?<![\d/-])\d{4}-\d{1,2}-\d{1,2}(?![\d/-])",
            r"(?<![\d/-])\d{1,2}[/-]\d{1,2}[/-]\d{2,4}(?![\d/-])",
            r"(?<![\d/-])\d{1,2}/\d{4}(?![\d/-])",
            *_month_forms(_MONTH_WORD),
            # FY26, FY 2026
            r"\bFY\s?\d{2}(?:\d{2})?\b",
            # A bare year: 1900 to 2099, not money and not a percentage.
            r"(?<![$\d,.\-/])(?:19|20)\d{2}(?![,.\-/]\d)(?!\d)(?!\s*%)",
        )
    ),
    re.IGNORECASE,
)
_MAY_DATE = re.compile("|".join(_month_forms("May")))


def text_prints_any_date(text: str) -> bool:
    """Whether ``text`` prints anything that reads as a calendar date or year.

    Deliberately broad: a revision stamp, a fiscal year or a bare year all
    count. A unit with any of them is not date-silent, and the check leaves it
    alone.
    """

    return bool(_ANY_DATE.search(text) or _MAY_DATE.search(text))


def _month_alternatives(month: int) -> str:
    full = calendar.month_name[month]
    names = [full] if len(full) <= 3 else [full, rf"{full[:3]}\.?"]
    if month == 9:
        names.append(r"Sept\.?")
    return "(?:" + "|".join(names) + ")"


def text_prints_date(text: str, value: date) -> bool:
    """Whether ``text`` prints exactly the calendar date ``value``.

    Day-level forms: ``2026-03-01``, ``3/1/2026``, ``03-01-26``,
    ``March 1, 2026``, ``1 Mar 2026``. For the first day of a month, the
    month-level forms ``March 2026`` and ``03/2026`` also count, unless a day
    number stands in front of them (``15 March 2026``, ``10/03/2026``).
    """

    year, month, day = value.year, value.month, value.day
    name = _month_alternatives(month)
    ordinal = _ORDINAL
    year_forms = rf"(?:{year:04d}|{year % 100:02d})"
    patterns = [
        rf"(?<![\d/-]){year:04d}-0?{month}-0?{day}(?![\d/-])",
        rf"(?<![\d/-])0?{month}[/-]0?{day}[/-]{year_forms}(?![\d/-])",
        rf"\b{name}\s+0?{day}{ordinal},?\s+{year:04d}(?!\d)",
        rf"(?<![\d/-])0?{day}{ordinal}\s+{name},?\s+{year_forms}(?!\d)",
    ]
    if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
        return True
    if day != 1:
        return False
    if re.search(rf"(?<![\d/-])0?{month}/{year:04d}(?![\d/-])", text):
        return True
    for match in re.finditer(rf"\b{name},?\s+{year:04d}(?!\d)", text, re.IGNORECASE):
        if not re.search(rf"\d{{1,2}}{ordinal}\s+$", text[: match.start()]):
            return True
    return False


# --------------------------------------------------------------------------
# The check
# --------------------------------------------------------------------------

_SITE = "axiom_encode/harness/start_date_evidence.py:date_silent_start_issues"
_START_FIELD = "rules[].versions[].effective_from"

CHECK_START_UNSUPPORTED = declare_check(
    "complete-source-unit:effective-date/start-unsupported",
    title="Start date of a date-silent unit has no support in its own evidence",
    requires=(
        f"In a manual source unit that prints no date, every `{_START_FIELD}` "
        "(including its `from` alias and rule-level implicit versions) "
        "is the `0001-01-01` scaffold, the corpus `expression_date` or "
        "`source_as_of`, the evidence floor (the first day of the first full "
        "calendar month on or after `source_as_of`), or a date printed in the "
        "evidence text of an `effective_period` proof atom of the same rule."
    ),
    why=(
        "A start date that nothing in the unit supports was copied from a "
        "neighbouring module or guessed. The rule then claims to apply from a "
        "date for which the corpus holds no evidence about this unit, and "
        "compile, the other completeness checks and the companion tests all "
        "pass on it."
    ),
    fix=(
        f"Change `{_START_FIELD}` (and the companion test periods that depend "
        "on it) to a date the unit's own evidence supports: a date printed in "
        "`./source.txt`, a date printed in the evidence text of a proof atom "
        "of the same rule with `kind: effective_period`, the "
        "corpus `expression_date` or `source_as_of`, the evidence floor, or "
        "the `0001-01-01` scaffold. A date quoted by any other proof atom is "
        "not support. Do not copy the start date of a neighbouring "
        "module. The evidence floor is a proposed calendar convention, not a "
        "claimed legal commencement date. The same rule applies to a `from` "
        "alias and to the start on a rule with absent or empty `versions`."
    ),
    cause="encoder_output",
    gate="ci",
    site=_SITE,
)


def citation_document_class(corpus_citation_path: str | None) -> str | None:
    """The corpus document class: the second segment of a citation path."""

    parts = (corpus_citation_path or "").strip().split("/")
    return parts[1] if len(parts) > 2 and parts[0] and parts[1] else None


def is_manual_citation(corpus_citation_path: str | None) -> bool:
    """The scope of the start-date rule, for the prompt and for the check."""

    return citation_document_class(corpus_citation_path) == MANUAL_DOCUMENT_CLASS


def _mappings(node: object) -> Iterator[Mapping[str, Any]]:
    for item in node if isinstance(node, list) else ():
        if isinstance(item, Mapping):
            yield item


START_DATE_PROOF_KIND = "effective_period"


def start_date_proof_texts(rule: Mapping[str, Any]) -> tuple[str, ...]:
    """The start-date proof evidence of ``rule``.

    Each evidence text of a rule-level proof atom of this rule with ``kind:
    effective_period``. Atoms of another kind and atoms of another rule are
    not start-date evidence. The ``path`` is not examined (module docstring).

    The atoms and their texts come from the proof validator's reader. It reads
    ``rules[].metadata.proof.atoms``, or ``rules[].proof.atoms`` when the rule
    has no ``metadata.proof``, and both ``source.excerpt`` and
    ``source.quote``; proof validation authenticates exactly those texts
    against the corpus. An ``excerpt`` anywhere else is not proof evidence.
    """

    return tuple(
        text
        for atom in rule_proof_atoms(rule)
        if str(atom.get("kind") or "").strip() == START_DATE_PROOF_KIND
        for _field, text in proof_source_evidence_texts(atom.get("source"))
    )


_LENIENT_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def _version_start_field(version: Mapping[str, Any]) -> str:
    """Name the stated field, including a duplicate alias pair."""

    fields = (name for name in ("effective_from", "from") if name in version)
    return "/".join(fields) or "effective_from"


def _version_start(version: Mapping[str, Any]) -> date | str | None:
    """The start of one version: a date, an unreadable text, or ``None``.

    ``from`` is a serde alias of ``effective_from`` at the CI engine pin;
    supplying both is a duplicate field, not a precedence rule. ``None``
    means that the version states no start: the compiler refuses a missing
    start rather than supplying a default. The engine also compiles
    a start without zero padding (``2025-10-1``), and YAML reads an unquoted
    timestamp as a ``datetime``; both are read as their calendar day. Any
    other non-null value is returned as text: the caller cannot show that it
    is a supported date.
    """

    if "effective_from" in version and "from" in version:
        return "duplicate effective_from/from fields"
    raw = version.get("effective_from", version.get("from"))
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        match = _LENIENT_ISO_DATE.fullmatch(raw.strip())
        if match is not None:
            try:
                return date(*(int(part) for part in match.groups()))
            except ValueError:
                pass
    return repr(raw) if not isinstance(raw, str) else raw


@dataclass(frozen=True)
class _StatedStart:
    """One stated start of an effective version, and where it stands."""

    rule: Mapping[str, Any]
    rule_index: int
    field: str
    start: date | str

    @property
    def rule_name(self) -> str:
        return str(self.rule.get("name") or "").strip()

    @property
    def rule_label(self) -> str:
        """The rule name, or its position when the rule has no name."""

        return self.rule_name or f"rules[{self.rule_index}]"


def _stated_starts(payload: Mapping[str, Any]) -> Iterator[_StatedStart]:
    """Each stated start of the engine's effective versions.

    Mirror ``RuleDefinition.effective_versions`` at the CI engine pin:
    nonempty versions replace rule-level fields; absent or empty versions
    use the rule's formula and start as one implicit version. Malformed
    version containers remain the compiler's responsibility.
    """

    rules = payload.get("rules")
    for rule_index, rule in enumerate(rules if isinstance(rules, list) else ()):
        if not isinstance(rule, Mapping):
            continue
        versions = rule.get("versions", [])
        # serde_yaml reads a blank/comment-only scalar as an empty Vec. The
        # explicit null spellings also load as None here, but compile refuses
        # those independently rather than supplying an implicit start.
        if versions is None:
            versions = []
        if not isinstance(versions, list):
            continue
        selected: Iterator[tuple[Mapping[str, Any], str]]
        if versions:
            selected = (
                (version, f"versions[{index}].{_version_start_field(version)}")
                for index, version in enumerate(versions)
                if isinstance(version, Mapping)
            )
        elif (
            rule.get("formula") is not None
            or rule.get("effective_from") is not None
            or rule.get("from") is not None
        ):
            selected = iter(((rule, _version_start_field(rule)),))
        else:
            continue
        for version, field in selected:
            start = _version_start(version)
            if start is not None:
                yield _StatedStart(rule, rule_index, field, start)


def _version_starts(
    payload: Mapping[str, Any],
) -> Iterator[tuple[str, str, date | str]]:
    """``(rule name, field path, start)`` for the engine's effective versions."""

    for stated in _stated_starts(payload):
        yield stated.rule_name, stated.field, stated.start


_SYMBOL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _EngineStartLoader(yaml.SafeLoader):
    """Project raw keys without YAML merge inheritance, as serde structs do."""

    def flatten_mapping(self, node):
        # The engine ignores the unknown << field on RuleDefinition and
        # RuleVersion. Filter the raw nodes before SafeLoader can inherit
        # versions or an alias date that the engine never consumes.
        node.value = [
            (key, value)
            for key, value in node.value
            if key.tag != "tag:yaml.org,2002:merge"
        ]
        super().flatten_mapping(node)

    def construct_mapping(self, node, deep=False):
        payload = super().construct_mapping(node, deep=deep)
        for key, value in node.value:
            if (
                key.value == "versions"
                and value.tag == "tag:yaml.org,2002:null"
                and value.value
            ):
                # Unlike an empty scalar, explicit null/~ is not an engine
                # Vec. Keep it non-list so an invalid imported provider
                # cannot lend its unused rule-level start as evidence.
                payload["versions"] = value.value
        return payload


def _engine_start_payload(content: str) -> Mapping[str, Any]:
    payload = yaml.load(content, Loader=_EngineStartLoader)
    if not isinstance(payload, Mapping):
        raise ValueError("RuleSpec start-date input is not a mapping")
    return payload


def _import_symbols(imports: object) -> set[str]:
    """The ``#symbol`` names of an ``imports`` list."""

    return {
        item.rsplit("#", 1)[1].strip()
        for item in (imports if isinstance(imports, list) else ())
        if isinstance(item, str) and "#" in item
    }


def imported_export_starts(
    payload: Mapping[str, Any],
    imported_symbol_contents: Sequence[tuple[str, str]],
) -> frozenset[date]:
    """Version starts of the exports the module verifiably imports.

    An import can only add supported dates; it never takes a module out of
    the check. An ``imports`` item contributes the version starts of its
    export when it is a ``path#symbol`` string and a supplied provider of that
    symbol declares exactly one rule of that name. Every other item
    contributes nothing: a file-level import without ``#symbol``, a provider
    that was not supplied or cannot be read, a provider with no rule or with
    several rules of that name, and an item that is not a string.

    A module cannot support its own start through an import. An item whose
    symbol the module itself declares (a self-import) contributes nothing, and
    so does a provider that imports a symbol this module declares (a circular
    pair). The engine refuses both as an import cycle; this check does not
    rely on that.
    """

    imports = payload.get("imports")
    own_names = {
        rule.get("name")
        for rule in _mappings(payload.get("rules"))
        if isinstance(rule.get("name"), str)
    }
    providers: dict[str, list[str]] = {}
    for symbol, content in imported_symbol_contents:
        providers.setdefault(symbol, []).append(content)
    starts: set[date] = set()
    for item in imports if isinstance(imports, list) else ():
        if not isinstance(item, str) or "#" not in item:
            continue
        symbol = item.rsplit("#", 1)[1].strip()
        if not _SYMBOL.fullmatch(symbol) or symbol in own_names:
            continue
        for content in providers.get(symbol, ()):
            try:
                provider = _engine_start_payload(content)
            except (yaml.YAMLError, TypeError, ValueError):
                continue
            if not isinstance(provider, Mapping):
                continue
            if _import_symbols(provider.get("imports")) & own_names:
                continue
            exports = [
                rule
                for rule in _mappings(provider.get("rules"))
                if rule.get("name") == symbol
            ]
            if len(exports) != 1:
                continue
            starts.update(
                start
                for _name, _field, start in _version_starts({"rules": exports})
                if isinstance(start, date)
            )
    return frozenset(starts)


def _permitted_dates_text(evidence: StartDateEvidence) -> str:
    parts = [
        "a date printed in the unit text",
        "a date printed in the evidence text of a proof atom of the same rule "
        f"with `kind: {START_DATE_PROOF_KIND}` (a date quoted by any other "
        "proof atom is not support)",
    ]
    if evidence.expression_date is not None:
        parts.append(f"the corpus `expression_date` `{evidence.expression_date}`")
    if evidence.source_as_of is not None:
        parts.append(f"the corpus `source_as_of` `{evidence.source_as_of}`")
        parts.append(
            f"the evidence floor `{evidence.evidence_floor}` (first day of the "
            "first full calendar month on or after `source_as_of`; a "
            "proposed calendar convention, not a claimed legal commencement "
            "date)"
        )
    parts.append(f"the `{SCAFFOLD_START.isoformat()}` scaffold")
    return "; ".join(parts)


def date_silent_start_issues(
    payload: Mapping[str, Any],
    source_text: str,
    *,
    corpus_citation_path: str,
    evidence: StartDateEvidence,
    imported_symbol_contents: Sequence[tuple[str, str]] = (),
    raw_content: str | None = None,
) -> list[ValidationIssue]:
    """Refuse an unsupported start date of a date-silent manual unit.

    Out of scope (no issue): a unit that is not a corpus ``manual`` document,
    prints any date, or has amendment context. In scope, one issue per distinct
    refused date. When ``source_as_of`` is missing or unreadable there is no
    snapshot date and no evidence floor, so only the scaffold, a readable
    ``expression_date`` or a printed date pass: missing metadata never counts
    as support. A printed date is support for the rule that proves it only:
    the date must stand in the start-date proof evidence of that rule
    (:func:`start_date_proof_texts`).
    """

    if (
        not is_manual_citation(corpus_citation_path)
        or evidence.amendment_context
        or text_prints_any_date(source_text)
    ):
        return []
    # The analyzer's ordinary mapping may already contain PyYAML-expanded
    # merges. Select starts from raw fields; keep its authenticated proof
    # excerpts as the existing evidence channel.
    start_payload = (
        _engine_start_payload(raw_content) if raw_content is not None else payload
    )
    supported = evidence.supported_corpus_dates | imported_export_starts(
        start_payload, imported_symbol_contents
    )
    # Proof atoms come from the analyzer's mapping, which proof validation
    # authenticates; the rule list has the same order in both readings.
    proof_rules = payload.get("rules")
    proof_rules = proof_rules if isinstance(proof_rules, list) else []
    refused: dict[str, list[_StatedStart]] = {}
    for stated in _stated_starts(start_payload):
        start = stated.start
        if isinstance(start, date):
            if start in supported:
                continue
            proof_rule = (
                proof_rules[stated.rule_index]
                if stated.rule_index < len(proof_rules)
                else None
            )
            if isinstance(proof_rule, Mapping) and any(
                text_prints_date(text, start)
                for text in start_date_proof_texts(proof_rule)
            ):
                continue
            start = start.isoformat()
        # A start that is not a readable date cannot be shown to be supported.
        refused.setdefault(start, []).append(stated)
    issues: list[ValidationIssue] = []
    permitted = _permitted_dates_text(evidence)
    no_proof = f"no `{START_DATE_PROOF_KIND}` proof atom of the same rule prints it"
    for start, sites in sorted(refused.items()):
        first = sites[0]
        # A rule without a name is named by its position (`rules[2]`).
        named = ", ".join(f"`{site.rule_label}`" for site in sites[:5])
        more = f" and {len(sites) - 5} more" if len(sites) > 5 else ""
        if evidence.source_as_of is not None:
            reason = (
                "it is not the evidence floor "
                f"`{evidence.evidence_floor}`, the corpus `source_as_of` "
                f"`{evidence.source_as_of}` or the corpus `expression_date`, "
                f"and {no_proof}"
            )
        else:
            reason = (
                "the corpus `source_as_of` is unavailable ("
                + "; ".join(evidence.problems or ("source_as_of is missing",))
                + "), so there is no snapshot date and no evidence floor to "
                f"equal, it is not the corpus `expression_date`, and {no_proof}"
            )
        issues.append(
            ValidationIssue(
                "[complete-source-unit:effective-date] "
                f"`{_START_FIELD}` is `{start}` in {len(sites)} "
                f"version(s) (rules {named}{more}), but the source unit prints "
                f"no date and nothing in its own evidence supports that start: "
                f"{reason}. A date copied from a neighbouring module is not "
                f"support. Permitted start dates: {permitted}.",
                check=CHECK_START_UNSUPPORTED,
                subject=IssueSubject(
                    citation=corpus_citation_path or None,
                    rule=first.rule_name or None,
                    field=first.field,
                ),
                expected=permitted,
                found=start,
            )
        )
    return issues

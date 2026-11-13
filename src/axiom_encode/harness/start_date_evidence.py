"""Start-date evidence for a date-silent source unit (#2212).

A source unit that prints no date gives no start date for its rules. This
module holds the one rule for such a unit, shared by the encoder prompt and by
the ``complete-source-unit:effective-date/start-unsupported`` check:

    An ``effective_from`` is *supported* when it is printed in the unit text
    or in a proof excerpt, equals the corpus ``expression_date`` or
    ``source_as_of``, equals the evidence floor, or is the ``0001-01-01``
    scaffold. A date copied from a precedent module or an import is not
    support.

The *evidence floor* is the first day of the first full calendar month that
starts on or after ``source_as_of``. It is an evidence-coverage floor ("the
corpus holds this text from here; earlier coverage is unproven"), not a
claimed legal commencement date.

The check is the narrow form of that rule, to keep false refusals low. It
looks only at a unit that is all of: a corpus ``manual`` document, date-silent
and without amendment context. There every start date must be one of the
supported dates above: the scaffold, the ``expression_date``, ``source_as_of``,
the evidence floor, or a date printed in an excerpt of a proof atom
(``rules[].metadata.proof.atoms[].source.excerpt``). Any other start is
refused, also one that lies after the snapshot.

Imports: a start that equals a version start of an export the module imports
is not examined (whether such a date is support is an open decision of #2212).
A module with an import whose provider this module was not given is not
examined at all, because that comparison cannot be made.

``effective_to``, ``values`` keys and companion test periods are not examined.
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

SCAFFOLD_START = date(1, 1, 1)
MANUAL_DOCUMENT_CLASS = "manual"

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class StartDateEvidence:
    """What the corpus says about when it holds the unit's text.

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
        "is the `0001-01-01` scaffold, the corpus `expression_date` or "
        "`source_as_of`, the evidence floor (the first day of the first full "
        "calendar month on or after `source_as_of`), or a date printed in a "
        "proof-atom excerpt of the module."
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
        "`./source.txt` or in a proof-atom excerpt you cite, the corpus "
        "`expression_date` or `source_as_of`, the evidence floor, or the "
        "`0001-01-01` scaffold. Do not copy the start date of a neighbouring "
        "module. The evidence floor is an evidence-coverage floor, not a "
        "claimed legal commencement date."
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


def proof_excerpts(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Each ``rules[].metadata.proof.atoms[].source.excerpt`` string.

    An ``excerpt`` key anywhere else is not a proof excerpt: the proof
    validator authenticates only these against the corpus.
    """

    excerpts: list[str] = []
    for rule in _mappings(payload.get("rules")):
        metadata = rule.get("metadata")
        proof = metadata.get("proof") if isinstance(metadata, Mapping) else None
        atoms = proof.get("atoms") if isinstance(proof, Mapping) else None
        for atom in _mappings(atoms):
            source = atom.get("source")
            excerpt = source.get("excerpt") if isinstance(source, Mapping) else None
            if isinstance(excerpt, str):
                excerpts.append(excerpt)
    return tuple(excerpts)


_LENIENT_ISO_DATE = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def _version_start(version: Mapping[str, Any]) -> date | str | None:
    """The start of one version: a date, an unreadable text, or ``None``.

    ``None`` means that the version states no start. The engine also compiles
    a start without zero padding (``2025-10-1``), and YAML reads an unquoted
    timestamp as a ``datetime``; both are read as their calendar day. Any
    other non-null value is returned as text: the caller cannot show that it
    is a supported date.
    """

    raw = version.get("effective_from")
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


def _version_starts(
    payload: Mapping[str, Any],
) -> Iterator[tuple[str, int, date | str]]:
    """``(rule name, version index, start)`` for every stated version start."""

    for rule in _mappings(payload.get("rules")):
        for index, version in enumerate(_mappings(rule.get("versions"))):
            start = _version_start(version)
            if start is not None:
                yield str(rule.get("name") or "").strip(), index, start


def imported_export_starts(
    payload: Mapping[str, Any],
    imported_symbol_contents: Sequence[tuple[str, str]],
) -> frozenset[date] | None:
    """Version starts of the exports the module imports, or ``None``.

    ``None`` means that the comparison cannot be made: an ``imports`` entry is
    not a string, or the provider of an imported ``#symbol`` was not supplied
    or does not declare exactly one rule of that name. No imports, and
    file-level imports without a ``#symbol``, give an empty set.
    """

    imports = payload.get("imports")
    if imports is None or imports == []:
        return frozenset()
    if not isinstance(imports, list):
        return None
    providers: dict[str, list[str]] = {}
    for symbol, content in imported_symbol_contents:
        providers.setdefault(symbol, []).append(content)
    starts: set[date] = set()
    for item in imports:
        if not isinstance(item, str):
            return None
        if "#" not in item:
            # A file-level import names no export, so it brings no start date
            # and takes nothing out of the check.
            continue
        symbol = item.rsplit("#", 1)[1].strip()
        contents = providers.get(symbol)
        if not contents:
            return None
        for content in contents:
            try:
                provider = yaml.safe_load(content)
            except (yaml.YAMLError, TypeError, ValueError):
                return None
            rules = provider.get("rules") if isinstance(provider, Mapping) else None
            exports = [rule for rule in _mappings(rules) if rule.get("name") == symbol]
            if len(exports) != 1:
                return None
            starts.update(
                start
                for _name, _index, start in _version_starts({"rules": exports})
                if isinstance(start, date)
            )
    return frozenset(starts)


def _permitted_dates_text(evidence: StartDateEvidence) -> str:
    parts = [
        "a date printed in the unit text or in a proof-atom excerpt you cite",
    ]
    if evidence.expression_date is not None:
        parts.append(f"the corpus `expression_date` `{evidence.expression_date}`")
    if evidence.source_as_of is not None:
        parts.append(f"the corpus `source_as_of` `{evidence.source_as_of}`")
        parts.append(
            f"the evidence floor `{evidence.evidence_floor}` (first day of the "
            "first full calendar month on or after `source_as_of`; an "
            "evidence-coverage floor, not a claimed legal commencement date)"
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
) -> list[ValidationIssue]:
    """Refuse an unsupported start date of a date-silent manual unit.

    Out of scope (no issue): a unit that is not a corpus ``manual`` document,
    prints any date, has amendment context, or has an import whose provider
    is not in ``imported_symbol_contents``. In scope, one issue per distinct
    refused date. When ``source_as_of`` is missing or unreadable there is no
    snapshot date and no evidence floor, so only the scaffold, a readable
    ``expression_date`` or a printed date pass: missing metadata never counts
    as support.
    """

    if (
        not is_manual_citation(corpus_citation_path)
        or evidence.amendment_context
        or text_prints_any_date(source_text)
    ):
        return []
    imported_starts = imported_export_starts(payload, imported_symbol_contents)
    if imported_starts is None:
        return []
    supported = evidence.supported_corpus_dates | imported_starts
    excerpts = proof_excerpts(payload)
    refused: dict[str, list[tuple[str, int]]] = {}
    for rule_name, index, start in _version_starts(payload):
        if isinstance(start, date):
            if start in supported:
                continue
            if any(text_prints_date(excerpt, start) for excerpt in excerpts):
                continue
            start = start.isoformat()
        # A start that is not a readable date cannot be shown to be supported.
        refused.setdefault(start, []).append((rule_name, index))
    issues: list[ValidationIssue] = []
    permitted = _permitted_dates_text(evidence)
    for start, sites in sorted(refused.items()):
        first_rule, first_index = sites[0]
        named = ", ".join(f"`{name}`" for name, _index in sites[:5] if name)
        more = f" and {len(sites) - 5} more" if len(sites) > 5 else ""
        if evidence.source_as_of is not None:
            reason = (
                "it is not the evidence floor "
                f"`{evidence.evidence_floor}`, the corpus `source_as_of` "
                f"`{evidence.source_as_of}` or the corpus `expression_date`, "
                "and no proof-atom excerpt prints it"
            )
        else:
            reason = (
                "the corpus `source_as_of` is unavailable ("
                + "; ".join(evidence.problems or ("source_as_of is missing",))
                + "), so there is no snapshot date and no evidence floor to "
                "equal, it is not the corpus `expression_date`, and no "
                "proof-atom excerpt prints it"
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
                    rule=first_rule or None,
                    field=f"versions[{first_index}].effective_from",
                ),
                expected=permitted,
                found=start,
            )
        )
    return issues

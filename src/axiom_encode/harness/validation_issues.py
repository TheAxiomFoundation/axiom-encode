"""Structured validation issues persisted with every encode attempt.

Before this module, a failed attempt left one line behind (``Generated
RuleSpec failed CI validation``) plus a path to a ``*.repair.json`` manifest
that lived in a temporary directory. The validator's actual issue list was
lost with the directory. This module turns each validator issue string into a
:class:`ValidationIssue` record that is stored inline (``iterations_json``,
``session_events.metadata_json``, ``artifact_versions.metadata_json``) so a
failed generation is a labeled case on its own.

Issue shape (``axiom-encode/validation-issue/v1``)
--------------------------------------------------
``gate``
    Which validator produced the issue: ``compile``, ``ci``,
    ``numeric_occurrence``, ``generalist_review``, ``policyengine``,
    ``taxsim``, ``overlay`` (policy-repo overlay validation during
    ``--apply``), or ``attempt`` (the attempt's one-line error when no gate
    issue list was available).
``kind``
    The bounded message class from
    :func:`axiom_encode.harness.evals.classify_validation_issue` (for example
    ``ungrounded_literal``, ``fixture_execution``, ``proof_atoms``).
``message``
    The validator's full issue text, capped at
    :data:`MAX_ISSUE_MESSAGE_CHARS`.
``category``
    The bracketed validator category when the issue carries one, for example
    ``complete-source-unit:structure``.
``locator``
    Best-effort locus inside the generated artifact: ``rule:<name>``,
    ``test:<case>``, ``path:<file>``, ``line:<n>``, or the source branch a
    complete-source-unit issue names. ``None`` when the text names none.
``line``
    The line number when the message states one.
``value``
    The numeric literal or value pair the issue is about (``600000``,
    ``expected=12 actual=10``), when the text names one.
``clause``
    The statutory clause the issue cites (``26 USC 32(b)``, ``§ 1401(b)(1)``,
    ``(a)(2)(B)``), when the text names one.

Extraction is deliberately conservative: every field except ``gate``,
``kind`` and ``message`` is ``None`` unless a pattern matched. Nothing is
inferred that the validator did not say.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

ISSUE_SCHEMA_VERSION = "axiom-encode/validation-issue/v1"

#: Upper bounds so one pathological attempt cannot bloat a row.
MAX_ISSUES_PER_ATTEMPT = 500
MAX_ISSUE_MESSAGE_CHARS = 4000
MAX_ISSUES_JSON_BYTES = 256 * 1024

#: Metrics attributes in the order their gates run. ``ci_issues`` merges the
#: grounding, occurrence and PolicyEngine-hint issues, so it comes last: an
#: issue seen under a more specific gate keeps that label.
GATE_ISSUE_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("compile", "compile_issues"),
    ("numeric_occurrence", "numeric_occurrence_issues"),
    ("generalist_review", "generalist_review_issues"),
    ("policyengine", "policyengine_issues"),
    ("taxsim", "taxsim_issues"),
    ("ci", "ci_issues"),
)

_CATEGORY_RE = re.compile(r"(?:^|:\s)\[([A-Za-z0-9_-]+(?::[A-Za-z0-9_-]+)?)\]")
_TEST_CASE_RE = re.compile(r"[Tt]est case [`'\"]([^`'\"]+)[`'\"]")
_RULE_RE = re.compile(r"\b(?:rule|output|variable|parameter|definition) `([^`]+)`")
#: ``<Headline>: <name> line <n> ...`` (embedded scalar / decomposed date issues).
_NAME_LINE_RE = re.compile(r"(?:^|:\s+)([A-Za-z_][\w.]*) line (\d+)\b")
#: A backticked identifier-like token (must contain ``_`` or ``.`` so plain
#: words and values such as ```true``` are not mistaken for rule names).
_BACKTICK_IDENTIFIER_RE = re.compile(
    r"`([A-Za-z_][A-Za-z0-9]*(?:[._][A-Za-z0-9_.]*)+)`"
)
_PATH_RE = re.compile(r"`([^`\s]+\.(?:ya?ml|json|rulespec))`")
_LINE_RE = re.compile(r"\b(?:at |on )?line (\d+)\b", re.IGNORECASE)
_STRUCTURE_LOCATOR_RE = re.compile(
    r"\bat (.+?) is neither encoded nor precisely deferred"
)
_FORMULA_LOCATOR_RE = re.compile(r"\bin (.+?) has no principal derived/relation output")
_UNGROUNDED_RE = re.compile(
    r"Ungrounded generated numeric literal:\s*(.+?)\s+does not appear",
    re.IGNORECASE,
)
_SOURCE_OCCURRENCE_RE = re.compile(r"Source numeric value (\S+) appears", re.IGNORECASE)
#: A number that never ends in a separator, so a sentence comma stays out.
_NUMBER = r"-?\d(?:[\d,]*\d)?(?:\.\d+)?%?"
_EXPECTED_RE = re.compile(
    r"\bexpected(?:\s+value)?\s*[:=]?\s*[`'\"]?(" + _NUMBER + r"|true|false|null)",
    re.IGNORECASE,
)
_ACTUAL_RE = re.compile(
    r"\b(?:actual|got|returned|received|computed)(?:\s+value)?\s*[:=]?\s*[`'\"]?"
    r"(" + _NUMBER + r"|true|false|null)",
    re.IGNORECASE,
)
_PE_PAIR_RE = re.compile(
    r"\bPE=([^\s,]+),?\s.*?RuleSpec expects=([^\s,]+)", re.IGNORECASE
)
#: ``Embedded scalar literal: <name> line <n> embeds <literal> in ...``.
_EMBEDS_RE = re.compile(r"\bembeds (" + _NUMBER + r")\b")
#: ``Decomposed date scalar: <name> line <n> encodes calendar year <n> ...``.
_CALENDAR_RE = re.compile(r"\bencodes calendar (?:year|month|day) (\d+)\b")
#: ``Source verification RuleSpec mismatch: `x` declares A, but RuleSpec has B.``
_DECLARES_RE = re.compile(r"\bdeclares (.+?), but RuleSpec has (.+?)\.?$")
#: Citation patterns in priority order: a USC/CFR citation, a section sign,
#: a section word followed by something citation-shaped (never plain prose),
#: then a bare chain of parenthesised designators such as ``(a)(2)(B)``.
_CLAUSE_PATTERNS = (
    re.compile(
        r"\b\d+ (?:U\.?S\.?C\.?|C\.?F\.?R\.?) ?(?:§+ ?)?[\dA-Za-z][\w.\-]*(?:\([\w.\-]+\))*",
        re.IGNORECASE,
    ),
    re.compile(r"§+ ?[\dA-Za-z][\w.\-]*(?:\([\w.\-]+\))*"),
    re.compile(
        r"\b(?:section|subsection|paragraph|clause) (?=[\d(])[\d(][\w.\-]*(?:\([\w.\-]+\))*",
        re.IGNORECASE,
    ),
    re.compile(r"(?<![\w)])(?:\([\w]{1,4}\)){2,}"),
)


@dataclass(frozen=True)
class ValidationIssue:
    """One validator issue with its best-effort locator and value."""

    gate: str
    kind: str
    message: str
    category: str | None = None
    locator: str | None = None
    line: int | None = None
    value: str | None = None
    clause: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "gate": self.gate,
            "kind": self.kind,
            "message": self.message,
        }
        for name in ("category", "locator", "line", "value", "clause"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationIssue":
        line = data.get("line")
        return cls(
            gate=str(data.get("gate") or "attempt"),
            kind=str(data.get("kind") or "unclassified"),
            message=str(data.get("message") or ""),
            category=data.get("category"),
            locator=data.get("locator"),
            line=int(line)
            if isinstance(line, int) and not isinstance(line, bool)
            else None,
            value=data.get("value"),
            clause=data.get("clause"),
        )


def classify_issue_kind(message: str) -> str:
    """Bounded message class shared with the outcome telemetry."""
    from .evals import classify_validation_issue

    return classify_validation_issue(message)


def extract_locator(message: str) -> tuple[str | None, int | None]:
    """Return ``(locator, line)`` named by the issue text, if any."""
    line_match = _LINE_RE.search(message)
    line = int(line_match.group(1)) if line_match else None
    category = extract_category(message)
    if category == "complete-source-unit:structure":
        match = _STRUCTURE_LOCATOR_RE.search(message)
        if match:
            return " ".join(match.group(1).split()), line
    elif category == "complete-source-unit:formula-output":
        match = _FORMULA_LOCATOR_RE.search(message)
        if match:
            return " ".join(match.group(1).split()), line
    test_match = _TEST_CASE_RE.search(message)
    if test_match:
        return f"test:{test_match.group(1)}", line
    rule_match = _RULE_RE.search(message)
    if rule_match:
        return f"rule:{rule_match.group(1)}", line
    path_match = _PATH_RE.search(message)
    if path_match:
        return f"path:{path_match.group(1)}", line
    name_line = _NAME_LINE_RE.search(message)
    if name_line:
        return f"rule:{name_line.group(1)}", int(name_line.group(2))
    identifier = _BACKTICK_IDENTIFIER_RE.search(message)
    if identifier:
        return f"rule:{identifier.group(1)}", line
    if line is not None:
        return f"line:{line}", line
    return None, None


def extract_category(message: str) -> str | None:
    match = _CATEGORY_RE.search(message)
    return match.group(1).lower() if match else None


def extract_value(message: str) -> str | None:
    """Return the numeric literal or value pair the issue is about."""
    ungrounded = _UNGROUNDED_RE.search(message)
    if ungrounded:
        return ungrounded.group(1).strip()
    occurrence = _SOURCE_OCCURRENCE_RE.search(message)
    if occurrence:
        return occurrence.group(1)
    pe_pair = _PE_PAIR_RE.search(message)
    if pe_pair:
        return f"oracle={pe_pair.group(1)} expected={pe_pair.group(2)}"
    embeds = _EMBEDS_RE.search(message)
    if embeds:
        return embeds.group(1)
    calendar = _CALENDAR_RE.search(message)
    if calendar:
        return calendar.group(1)
    declares = _DECLARES_RE.search(message)
    if declares:
        return (
            f"expected={declares.group(1).strip()} actual={declares.group(2).strip()}"
        )
    expected = _EXPECTED_RE.search(message)
    actual = _ACTUAL_RE.search(message)
    if expected and actual:
        return f"expected={expected.group(1)} actual={actual.group(1)}"
    if expected:
        return f"expected={expected.group(1)}"
    if actual:
        return f"actual={actual.group(1)}"
    return None


def extract_clause(message: str) -> str | None:
    """The statutory clause the issue cites, preferring a real citation."""
    for pattern in _CLAUSE_PATTERNS:
        match = pattern.search(message)
        if match:
            return " ".join(match.group(0).split())
    return None


def structure_issue(gate: str, message: object) -> ValidationIssue:
    """Build one :class:`ValidationIssue` from a validator issue string."""
    text = str(message)
    if len(text) > MAX_ISSUE_MESSAGE_CHARS:
        text = text[:MAX_ISSUE_MESSAGE_CHARS]
    locator, line = extract_locator(text)
    return ValidationIssue(
        gate=str(gate),
        kind=classify_issue_kind(text),
        message=text,
        category=extract_category(text),
        locator=locator,
        line=line,
        value=extract_value(text),
        clause=extract_clause(text),
    )


def structure_labeled_issues(
    labeled_issues: Iterable[tuple[str, Sequence[object]]],
) -> list[ValidationIssue]:
    """Structure ``(gate, issues)`` groups, deduplicating repeated messages."""
    structured: list[ValidationIssue] = []
    seen: set[str] = set()
    for gate, issues in labeled_issues:
        if isinstance(issues, (str, bytes)) or not isinstance(issues, Sequence):
            continue
        for issue in issues:
            text = str(issue)
            if not text or text in seen:
                continue
            seen.add(text)
            structured.append(structure_issue(gate, text))
    return structured


def structure_attempt_issues(
    result: Any,
    *,
    extra_issues: Sequence[object] = (),
    extra_gate: str = "overlay",
    error: str | None = None,
) -> list[ValidationIssue]:
    """Structure every issue one generation attempt produced.

    ``result`` is the attempt's ``EvalResult`` (duck-typed). Gate issue lists
    come from ``result.metrics``; a gate whose pass flag is ``True`` is
    skipped because its issues are advisory. ``extra_issues`` are strings the
    apply path produced after standalone validation (overlay validation, YAML
    preflight); any not already covered by a gate are labeled ``extra_gate``.
    When nothing structured is available, ``error`` (the attempt's one-line
    failure) becomes a single ``attempt`` issue so a failed attempt always
    carries at least one record.
    """
    labeled: list[tuple[str, Sequence[object]]] = []
    metrics = getattr(result, "metrics", None)
    if metrics is not None:
        for gate, attribute in GATE_ISSUE_ATTRIBUTES:
            issues = getattr(metrics, attribute, None)
            if not issues:
                continue
            if getattr(metrics, f"{gate}_pass", None) is True:
                continue
            labeled.append((gate, issues))
    structured = structure_labeled_issues(labeled)
    seen = {issue.message for issue in structured}
    if isinstance(extra_issues, (str, bytes)):
        extra_issues = ()
    for issue in extra_issues:
        text = str(issue)
        if not text or text in seen:
            continue
        seen.add(text)
        structured.append(structure_issue(extra_gate, text))
    if not structured and isinstance(error, str) and error:
        structured.append(structure_issue("attempt", error))
    return structured


def issues_to_dicts(
    issues: Sequence[ValidationIssue],
) -> tuple[list[dict[str, Any]], int]:
    """Serialize issues under the per-attempt count and byte bounds.

    Returns ``(dicts, truncated_count)``; ``truncated_count`` is how many
    issues were dropped to stay within :data:`MAX_ISSUES_PER_ATTEMPT` and
    :data:`MAX_ISSUES_JSON_BYTES`.
    """
    payload: list[dict[str, Any]] = []
    total_bytes = 2
    kept = 0
    for issue in issues[:MAX_ISSUES_PER_ATTEMPT]:
        item = issue.to_dict()
        item_bytes = len(json.dumps(item)) + 1
        if total_bytes + item_bytes > MAX_ISSUES_JSON_BYTES:
            break
        payload.append(item)
        total_bytes += item_bytes
        kept += 1
    return payload, len(issues) - kept


def issues_from_dicts(items: object) -> list[ValidationIssue]:
    if not isinstance(items, list):
        return []
    return [ValidationIssue.from_dict(item) for item in items if isinstance(item, dict)]


def issue_summary_counts(issues: Sequence[ValidationIssue]) -> dict[str, int]:
    """``gate:kind`` counts, the same key shape as the outcome telemetry."""
    counts: dict[str, int] = {}
    for issue in issues:
        key = f"{issue.gate}:{issue.kind}"
        counts[key] = counts.get(key, 0) + 1
    return counts

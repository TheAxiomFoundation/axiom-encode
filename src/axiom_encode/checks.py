"""Declared validator checks and the typed issues they emit.

Every validator issue should come from a declared :class:`Check`. A check
states, next to the code that runs it, what it requires, why that matters, how
to fix a failure (for the model and for a person), and where a failure usually
comes from. The text is reviewed in the PR that adds or changes the check, so
the explanation of a failure comes from the encoder, not from a hand-written
list in a dashboard.

A check emits a :class:`ValidationIssue`. The issue is a ``str`` subclass: its
string value is the validator message exactly as before, so every consumer of
``list[str]`` issue lists (retry feedback, telemetry classifiers, tests that
pin message bytes) keeps working unchanged. The typed fields ride along:

``check``
    The id of the declared check, for example
    ``complete-source-unit:tests/output-unasserted``.
``subject``
    Where the problem is: citation path, source branch or clause, module file
    and line, rule, test case, or artifact field.
``expected`` / ``found``
    What the check required and what the artifact had, when they are short.
``evidence``
    A short source or artifact excerpt.
``fingerprint``
    A stable hash of the check, subject and expected value, so the same
    problem at the same place has the same id across tries and runs.

Issue record shape (``axiom-encode/validation-issue/v2``)
---------------------------------------------------------
A superset of ``axiom-encode/validation-issue/v1`` (PR #1660): ``gate``,
``kind``, ``message``, ``category``, ``locator``, ``line``, ``value`` and
``clause`` keep their v1 meaning; ``kind`` is the check id for a typed issue
and the :func:`~axiom_encode.harness.evals.classify_validation_issue` class
for an untyped one. v2 adds ``check``, ``cause``, ``subject``, ``expected``,
``found``, ``evidence`` and ``fingerprint`` (all ``None`` for an untyped
issue).

Registry shape (``axiom-encode/check-registry/v1``)
---------------------------------------------------
``axiom-encode checks --json`` prints every declared check with the encoder
version and a digest of the check definitions; the same document is committed
as ``src/axiom_encode/data/check_registry.json`` and a test keeps it in sync.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

ISSUE_SCHEMA_VERSION = "axiom-encode/validation-issue/v2"
REGISTRY_SCHEMA_VERSION = "axiom-encode/check-registry/v1"

Cause = Literal["encoder_output", "validator", "source_corpus", "infra"]
CAUSES: tuple[str, ...] = ("encoder_output", "validator", "source_corpus", "infra")
Severity = Literal["critical", "important", "minor", "info"]

#: Modules that declare checks. ``load_checks`` imports them so the registry is
#: complete wherever it is read; a test fails when a module that calls
#: :func:`declare_check` is missing here.
CHECK_MODULES: tuple[str, ...] = (
    "axiom_encode.harness.source_completeness",
    "axiom_encode.harness.proof_validator",
    "axiom_encode.harness.validator_pipeline",
    "axiom_encode.infra_checks",
)

#: Bounds for persisted issue lists, per try.
MAX_RECORDED_ISSUES = 200
MAX_RECORDED_MESSAGE_CHARS = 2000
MAX_RECORDED_FIELD_CHARS = 400
MAX_RECORDED_ISSUES_JSON_BYTES = 128 * 1024

_CHECK_ID_RE = re.compile(r"^[a-z][a-z0-9-]*(?::[a-z0-9-]+)*(?:/[a-z0-9-]+)?$")
_CATEGORY_RE = re.compile(r"^\[([A-Za-z0-9_-]+(?::[A-Za-z0-9_-]+)?)\]")


@dataclass(frozen=True)
class Check:
    """One declared validator check.

    ``requires`` says what the check demands of the artifact, ``why`` why it
    matters, ``fix`` what to change (it is shown to the model in retry
    feedback and to a person on the ops pages), and ``cause`` where a failure
    usually comes from. ``site`` names the code that emits it
    (``<module path>:<function>``).
    """

    id: str
    title: str
    requires: str
    why: str
    fix: str
    cause: Cause
    gate: str
    site: str
    severity: Severity = "important"

    @property
    def category(self) -> str:
        """The bracketed message category the check's messages start with."""
        return self.id.partition("/")[0]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "category": self.category}


_REGISTRY: dict[str, Check] = {}


def declare_check(
    id: str,
    *,
    title: str,
    requires: str,
    why: str,
    fix: str,
    cause: Cause,
    gate: str,
    site: str,
    severity: Severity = "important",
) -> Check:
    """Declare a check once, at import time of the module that runs it."""
    check = Check(
        id=id,
        title=title,
        requires=requires,
        why=why,
        fix=fix,
        cause=cause,
        gate=gate,
        site=site,
        severity=severity,
    )
    problems = check_definition_problems(check)
    if problems:
        raise ValueError(f"Invalid check {id!r}: {'; '.join(problems)}")
    existing = _REGISTRY.get(id)
    if existing is not None and existing != check:
        raise ValueError(f"Check {id!r} is declared twice with different text")
    _REGISTRY[id] = check
    return check


def check_definition_problems(check: Check) -> list[str]:
    """Return what is missing or malformed in a check definition."""
    problems: list[str] = []
    if not _CHECK_ID_RE.match(check.id):
        problems.append("id must be lowercase `<category>[:<part>...][/<name>]`")
    for name in ("title", "requires", "why", "fix", "gate", "site"):
        if not str(getattr(check, name) or "").strip():
            problems.append(f"`{name}` is empty")
    if check.cause not in CAUSES:
        problems.append(f"`cause` must be one of {', '.join(CAUSES)}")
    if check.severity not in ("critical", "important", "minor", "info"):
        problems.append("`severity` is not a run-log severity")
    if ":" not in check.site:
        problems.append("`site` must be `<module path>:<function>`")
    return problems


def load_checks() -> dict[str, Check]:
    """Import every check-declaring module and return the full registry."""
    for module in CHECK_MODULES:
        loaded = importlib.import_module(module)
        register = getattr(loaded, "register_checks", None)
        if callable(register):
            register(declare_check)
    return dict(sorted(_REGISTRY.items()))


def get_check(check_id: str) -> Check | None:
    return _REGISTRY.get(check_id)


def registry_document(
    encoder_version: str | None = None, *, include_version: bool = True
) -> dict[str, Any]:
    """The published registry: every declared check, versioned and digested.

    The committed copy (``data/check_registry.json``) leaves the encoder version
    out, so a version bump alone does not change it.
    """
    checks = [check.to_dict() for check in load_checks().values()]
    document: dict[str, Any] = {"schema": REGISTRY_SCHEMA_VERSION}
    if include_version:
        if encoder_version is None:
            from axiom_encode import __version__ as encoder_version
        document["encoder_version"] = encoder_version
    document["registry_sha256"] = registry_digest(checks)
    document["checks"] = checks
    return document


COMMITTED_REGISTRY_PATH = "data/check_registry.json"


def committed_registry_text() -> str:
    """The exact text of the committed registry file."""
    return (
        json.dumps(
            registry_document(include_version=False),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def registry_digest(checks: Sequence[Mapping[str, Any]] | None = None) -> str:
    """SHA-256 over the check definitions (not the encoder version)."""
    if checks is None:
        checks = [check.to_dict() for check in load_checks().values()]
    payload = json.dumps(list(checks), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IssueSubject:
    """Where an issue applies. Every field is optional."""

    citation: str | None = None
    branch: str | None = None
    clause: str | None = None
    module: str | None = None
    line: int | None = None
    rule: str | None = None
    test_case: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    def locator(self) -> str | None:
        """The v1 ``locator`` string for this subject."""
        if self.rule:
            return f"rule:{self.rule}"
        if self.test_case:
            return f"test:{self.test_case}"
        if self.field:
            return f"field:{self.field}"
        if self.branch or self.clause:
            return self.clause or self.branch
        if self.module:
            return f"path:{self.module}"
        if self.line is not None:
            return f"line:{self.line}"
        return None


def _short(value: Any, limit: int = MAX_RECORDED_FIELD_CHARS) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
    text = " ".join(str(text).split())
    if not text:
        return None
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ValidationIssue(str):
    """A validator message that carries the typed fields of its check.

    The string value is the message; equality and hashing are the message's,
    so de-duplication of issue lists behaves exactly as for plain strings.
    """

    check: str
    subject: IssueSubject
    expected: str | None
    found: str | None
    evidence: str | None

    def __new__(
        cls,
        message: str,
        *,
        check: Check | str,
        subject: IssueSubject | None = None,
        expected: Any = None,
        found: Any = None,
        evidence: Any = None,
    ) -> "ValidationIssue":
        issue = super().__new__(cls, message)
        issue.check = check.id if isinstance(check, Check) else str(check)
        issue.subject = subject or IssueSubject()
        issue.expected = _short(expected)
        issue.found = _short(found)
        issue.evidence = _short(evidence)
        return issue

    def __reduce__(self):  # keep the typed fields across pickling
        return (
            _rebuild_issue,
            (
                str(self),
                self.check,
                self.subject,
                self.expected,
                self.found,
                self.evidence,
            ),
        )

    @property
    def fingerprint(self) -> str:
        return issue_fingerprint(self.check, self.subject, self.expected)

    def definition(self) -> Check | None:
        return get_check(self.check)

    def to_finding(self):
        """The issue as a ``run_log.v1`` :class:`~axiom_encode.run_log.Finding`."""
        from axiom_encode.run_log import Finding

        check = self.definition()
        return Finding(
            code=self.check,
            severity=check.severity if check else "important",
            message=str(self),
            locator=self.subject.locator(),
            evidence=self.evidence,
        )


def _rebuild_issue(message, check, subject, expected, found, evidence):
    return ValidationIssue(
        message,
        check=check,
        subject=subject,
        expected=expected,
        found=found,
        evidence=evidence,
    )


def retype(original: Any, text: str) -> str:
    """Give ``text`` the typed fields of ``original`` when it has them.

    Use it where a validator rewrites an issue string (path normalization,
    bounding, a file prefix) so the typed fields survive the rewrite.
    """
    if isinstance(original, ValidationIssue) and not isinstance(text, ValidationIssue):
        return ValidationIssue(
            text,
            check=original.check,
            subject=original.subject,
            expected=original.expected,
            found=original.found,
            evidence=original.evidence,
        )
    return text


def map_issue_text(issue: Any, transform) -> Any:
    """Apply a text transform to an issue and its typed text fields."""
    if not isinstance(issue, ValidationIssue):
        return transform(issue)

    def apply(value: str | None) -> str | None:
        return None if value is None else transform(value)

    return ValidationIssue(
        transform(str(issue)),
        check=issue.check,
        subject=issue.subject,
        expected=apply(issue.expected),
        found=apply(issue.found),
        evidence=apply(issue.evidence),
    )


def issue_fingerprint(
    check_id: str, subject: IssueSubject | None, expected: str | None
) -> str:
    payload = json.dumps(
        {
            "check": check_id,
            "subject": (subject or IssueSubject()).to_dict(),
            "expected": expected,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def message_digest(message: str) -> str:
    """SHA-256 of an issue's full message, the replay key of a truncated record."""
    return hashlib.sha256(message.encode("utf-8")).hexdigest()


def issue_record(
    issue: Any,
    *,
    gate: str | None,
    message_limit: int = MAX_RECORDED_MESSAGE_CHARS,
) -> dict[str, Any]:
    """One ``validation-issue/v2`` record for a typed or untyped issue.

    ``gate`` ``None`` means the caller does not know which gate produced the
    issue: a typed issue then takes its check's gate, an untyped one
    ``validation``.
    """
    if gate is None:
        check = issue.definition() if isinstance(issue, ValidationIssue) else None
        gate = check.gate if check is not None else "validation"
    message = str(issue)
    category_match = _CATEGORY_RE.match(message)
    category = category_match.group(1) if category_match else None
    record: dict[str, Any] = {
        "gate": gate,
        "kind": None,
        "message": message[:message_limit],
        "category": category,
        "locator": None,
        "line": None,
        "value": None,
        "clause": None,
        "check": None,
        "cause": None,
        "subject": None,
        "expected": None,
        "found": None,
        "evidence": None,
        "fingerprint": None,
    }
    if len(message) > message_limit:
        record["message_truncated"] = len(message) - message_limit
        # The stored message is a display prefix; replay matches the full
        # message by this digest, so a long issue keeps its check and fix.
        record["message_sha256"] = message_digest(message)
    if isinstance(issue, ValidationIssue):
        check = issue.definition()
        subject = issue.subject
        record.update(
            kind=issue.check,
            check=issue.check,
            cause=check.cause if check else None,
            locator=subject.locator(),
            line=subject.line,
            clause=subject.clause or subject.citation,
            value=issue.found or issue.expected,
            subject=subject.to_dict() or None,
            expected=issue.expected,
            found=issue.found,
            evidence=issue.evidence,
            fingerprint=issue.fingerprint,
        )
        if check is not None and record["category"] is None:
            record["category"] = check.category
    else:
        from axiom_encode.harness.evals import classify_validation_issue

        record["kind"] = classify_validation_issue(message)
    return record


@dataclass
class IssueRecords:
    """A bounded list of v2 records plus what the bounds dropped."""

    issues: list[dict[str, Any]] = field(default_factory=list)
    truncated: int = 0
    total: int = 0
    typed: int = 0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema": ISSUE_SCHEMA_VERSION,
            "total": self.total,
            "typed": self.typed,
            "issues": self.issues,
        }
        if self.truncated:
            data["truncated"] = self.truncated
        return data


def issue_records(
    labeled_issues: Iterable[tuple[str | None, Iterable[Any]]],
    *,
    max_issues: int = MAX_RECORDED_ISSUES,
    max_bytes: int = MAX_RECORDED_ISSUES_JSON_BYTES,
) -> IssueRecords:
    """Bounded, de-duplicated v2 records for ``(gate, issues)`` pairs."""
    result = IssueRecords()
    seen: set[tuple[str, str]] = set()
    size = 2
    for gate, issues in labeled_issues:
        for issue in issues or ():
            identity = (str(gate), str(issue))
            if identity in seen:
                continue
            seen.add(identity)
            result.total += 1
            if isinstance(issue, ValidationIssue):
                result.typed += 1
            if len(result.issues) >= max_issues:
                result.truncated += 1
                continue
            record = issue_record(issue, gate=None if gate is None else str(gate))
            encoded = len(json.dumps(record, ensure_ascii=False).encode("utf-8")) + 1
            if size + encoded > max_bytes:
                result.truncated += 1
                continue
            size += encoded
            result.issues.append(record)
    return result


def typed_issue_document(
    labeled_issues: Iterable[tuple[str | None, Iterable[Any]]],
    *,
    max_issues: int = MAX_RECORDED_ISSUES,
    max_bytes: int = MAX_RECORDED_ISSUES_JSON_BYTES,
) -> dict[str, Any] | None:
    """The bounded v2 issue list plus the definitions of the checks it names.

    ``None`` when no issue is typed, so records of untyped failures keep their
    old shape. The document is self-describing: a reader needs no registry
    file to show each check's title, requirement, reason and fix.
    """
    records = issue_records(labeled_issues, max_issues=max_issues, max_bytes=max_bytes)
    if not records.typed:
        return None
    document = _issue_document(records)
    # ``max_bytes`` bounds the whole serialized document, envelope and check
    # definitions included: drop records from the end until it fits.
    overshoot = _document_bytes(document) - max_bytes
    while overshoot > 0 and records.issues:
        while overshoot > 0 and records.issues:
            overshoot -= _document_bytes(records.issues.pop()) + 2
            records.truncated += 1
        document = _issue_document(records)
        overshoot = _document_bytes(document) - max_bytes
    return document


def _issue_document(records: IssueRecords) -> dict[str, Any]:
    document = records.to_dict()
    document["checks"] = referenced_checks(document["issues"])
    document["check_registry_sha256"] = registry_digest()
    return document


def _document_bytes(value: Any) -> int:
    """UTF-8 bytes of ``json.dumps(value, ensure_ascii=False)``, the bounded form."""
    return len(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def issues_from_document(
    messages: Sequence[str], document: Mapping[str, Any] | None
) -> list[str]:
    """Re-type plain issue strings from a stored v2 document, by message."""
    if not isinstance(document, Mapping):
        return list(messages)
    load_checks()
    by_message: dict[str, Mapping[str, Any]] = {}
    by_digest: dict[str, Mapping[str, Any]] = {}
    for record in document.get("issues") or ():
        if not (
            isinstance(record, Mapping)
            and isinstance(record.get("message"), str)
            and isinstance(record.get("check"), str)
        ):
            continue
        if not record.get("message_truncated"):
            by_message.setdefault(record["message"], record)
        elif isinstance(record.get("message_sha256"), str):
            # A truncated record matches only the full message it digests.
            by_digest.setdefault(record["message_sha256"], record)
    retyped: list[str] = []
    for message in messages:
        record = by_message.get(message)
        if record is None and by_digest:
            record = by_digest.get(message_digest(message))
        if record is None or get_check(str(record["check"])) is None:
            retyped.append(message)
            continue
        subject = record.get("subject")
        subject_fields = {
            key: value
            for key, value in (subject.items() if isinstance(subject, Mapping) else ())
            if key in IssueSubject.__dataclass_fields__
        }
        retyped.append(
            ValidationIssue(
                message,
                check=str(record["check"]),
                subject=IssueSubject(**subject_fields),
                expected=record.get("expected"),
                found=record.get("found"),
                evidence=record.get("evidence"),
            )
        )
    return retyped


def referenced_checks(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Definitions of the checks a record list names, for a self-contained bundle."""
    checks: dict[str, Any] = {}
    for record in records:
        check_id = record.get("check")
        if not check_id or check_id in checks:
            continue
        check = get_check(str(check_id))
        if check is not None:
            checks[check.id] = check.to_dict()
    return checks


def issue_fix_feedback(issues: Sequence[Any]) -> list[str]:
    """Retry feedback lines for typed issues, one block per check.

    Each block names the check, what it requires and its fix once, then one
    line per subject with expected versus found when the check recorded them.
    Untyped issues add nothing here; their message stays the only feedback.
    """
    by_check: dict[str, list[ValidationIssue]] = {}
    for issue in issues:
        if isinstance(issue, ValidationIssue) and get_check(issue.check):
            by_check.setdefault(issue.check, []).append(issue)
    lines: list[str] = []
    for check_id, grouped in by_check.items():
        check = get_check(check_id)
        assert check is not None
        lines.append(f"- [{check_id}] {check.title}. Requires: {check.requires}")
        lines.append(f"  Fix: {check.fix}")
        shown: set[str] = set()
        for issue in grouped:
            locator = issue.subject.locator()
            if not locator or (issue.expected is None and issue.found is None):
                continue
            if issue.fingerprint in shown:
                continue
            shown.add(issue.fingerprint)
            parts = [f"  - {locator}"]
            if issue.expected is not None:
                parts.append(f"expected {issue.expected}")
            if issue.found is not None:
                parts.append(f"found {issue.found}")
            lines.append("; ".join(parts))
    return lines

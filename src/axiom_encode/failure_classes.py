"""Failure classes for encode outcomes (docs/design/encode-failure-classes.md).

A failure class says who owns the fix for a failed encode. It is advisory: it
never waives a failure and never admits an artifact.

``classify_outcome`` uses only deterministic signals whose mapping to a class
is certain. Everything else is ``unknown``; a later triage step (not in this
module) may classify those with a lower confidence.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FAILURE_CLASS_SCHEMA = "axiom-encode/failure-class/v1"

ENCODER = "encoder"
VALIDATOR_FP = "validator_fp"
CORPUS = "corpus"
DEPENDENCY = "dependency"
SELECTION = "selection"
ENGINE = "engine"
MIGRATION = "migration"
PROCESS = "process"
INFRA = "infra"
UNKNOWN = "unknown"

FAILURE_CLASSES: Mapping[str, tuple[str, ...]] = {
    ENCODER: ("omission", "formula", "test", "harness"),
    VALIDATOR_FP: (),
    CORPUS: ("content", "metadata"),
    DEPENDENCY: (),
    SELECTION: (),
    ENGINE: (),
    MIGRATION: ("schema", "elided_excerpt"),
    PROCESS: (),
    INFRA: (),
    UNKNOWN: (),
}

SOURCES = ("rule", "triage")
CONFIDENCES = ("certain", "high", "low")

PRE_ENCODE_BLOCKED_STATUS = "pre_encode_blocked"

# Issue prefixes that mean the harness could not read or decode the file the
# model wrote; the candidate was never judged.
_HARNESS_ISSUE_PREFIXES = ("[generated-yaml:io]", "[generated-yaml:decode]")
_MAX_EVIDENCE = 8
_MAX_EVIDENCE_CHARS = 200

# Classes whose owner is not the encoder: a repair retry of the same target
# with the same checkout, corpus and dispatch cannot fix them.
NON_ENCODER_RETRY_STOP_CLASSES = frozenset({MIGRATION, PROCESS, INFRA, CORPUS})

# An overlay or validator issue may carry a ``<file>.yaml: <gate>: `` or a
# ``<file>.yaml: `` prefix; the classified text follows it.
_ISSUE_FILE_PREFIX = re.compile(r"^[^\s:`]+\.ya?ml: (?:[a-z][a-z0-9_-]*: )?")

# Per-issue rules. Each prefix names an issue whose inputs do not come from
# the generated candidate, so every retry of the same invocation reports it
# again. The list is deliberately short: an issue whose text could come from
# the candidate (for example a compile error in an imported legacy module,
# which a candidate can stop importing) stays unknown.
_ISSUE_RULES: tuple[tuple[str, str, str | None, str], ...] = (
    (
        # The rules engine refuses the RuleSpec checkout root itself (an old
        # root-level content directory), before it loads any module.
        "Axiom rules engine compile failed: invalid RuleSpec repository root "
        "configuration: root-level `",
        MIGRATION,
        "schema",
        "rulespec-root-layout",
    ),
    (
        # The resolver-owned corpus body is empty.
        "[complete-source-unit:authoritative-source] The authoritative corpus "
        "body is required",
        CORPUS,
        "content",
        "complete-source-unit:authoritative-source",
    ),
    (
        # The invocation's citation is not the one the signed dispatch
        # contract names.
        "[required-deferred-output-contract] generated citation does not "
        "match the signed dispatch review contract",
        PROCESS,
        None,
        "required-deferred-output-contract:citation",
    ),
    (
        # The harness-chosen target path is not the one the signed dispatch
        # contract names.
        "[required-deferred-output-contract] generated RuleSpec path does not "
        "match the signed dispatch review contract",
        PROCESS,
        None,
        "required-deferred-output-contract:path",
    ),
)


@dataclass(frozen=True)
class FailureClassRecord:
    """One failure-class verdict for one encode outcome."""

    failure_class: str
    subclass: str | None
    source: str
    confidence: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.failure_class not in FAILURE_CLASSES:
            raise ValueError(f"unknown failure class: {self.failure_class!r}")
        if self.subclass is not None and (
            self.subclass not in FAILURE_CLASSES[self.failure_class]
        ):
            raise ValueError(
                f"unknown subclass {self.subclass!r} for {self.failure_class!r}"
            )
        if self.source not in SOURCES:
            raise ValueError(f"unknown failure-class source: {self.source!r}")
        if self.confidence not in CONFIDENCES:
            raise ValueError(f"unknown failure-class confidence: {self.confidence!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": FAILURE_CLASS_SCHEMA,
            "class": self.failure_class,
            "subclass": self.subclass,
            "source": self.source,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
        }


def _evidence(*items: object) -> tuple[str, ...]:
    values = []
    for item in items:
        text = str(item).strip()
        if text:
            values.append(text[:_MAX_EVIDENCE_CHARS])
        if len(values) >= _MAX_EVIDENCE:
            break
    return tuple(values)


def _rule(
    failure_class: str, subclass: str | None, *evidence: object
) -> FailureClassRecord:
    return FailureClassRecord(
        failure_class=failure_class,
        subclass=subclass,
        source="rule",
        confidence="certain",
        evidence=_evidence(*evidence),
    )


def _unknown(*evidence: object) -> FailureClassRecord:
    return _rule(UNKNOWN, None, *evidence)


def _blocking_gate_finding(outcome: Mapping[str, Any]) -> Mapping[str, Any] | None:
    gate = outcome.get("pre_encode_gate")
    if not isinstance(gate, Mapping):
        return None
    findings = gate.get("findings")
    if not isinstance(findings, list):
        return None
    for finding in findings:
        if isinstance(finding, Mapping) and finding.get("enforce") is True:
            return finding
    return None


def classify_issue(text: object) -> FailureClassRecord:
    """Classify one validator issue string from certain signals only.

    An issue that no rule names is ``unknown``. No rule yields ``encoder``
    or ``validator_fp``: issue text alone never assigns that fault.
    """

    if not isinstance(text, str):
        return _unknown("issue:not-a-string")
    body = _ISSUE_FILE_PREFIX.sub("", text.lstrip(), count=1)
    for prefix, failure_class, subclass, tag in _ISSUE_RULES:
        if body.startswith(prefix):
            return _rule(failure_class, subclass, f"issue:{tag}")
    return _unknown()


RETRY_STOPPED_SCHEMA = "axiom-encode/retry-stopped/v1"


def retry_stop_record(issues: Sequence[object]) -> dict[str, Any] | None:
    """Return a ``retry_stopped`` record when no repair retry can help.

    Every residual blocking issue must classify with ``certain`` confidence
    into a class the encoder does not own. One unknown or encoder issue, or
    no issues at all, returns ``None`` and the caller retries as before.
    """

    if isinstance(issues, (str, bytes)) or not isinstance(issues, Sequence):
        return None
    if not issues:
        return None
    records = [classify_issue(issue) for issue in issues]
    if any(
        record.confidence != "certain"
        or record.failure_class not in NON_ENCODER_RETRY_STOP_CLASSES
        for record in records
    ):
        return None
    first = records[0]
    classes: list[str] = []
    tags: list[str] = []
    for record in records:
        label = (
            record.failure_class
            if record.subclass is None
            else f"{record.failure_class}/{record.subclass}"
        )
        if label not in classes:
            classes.append(label)
        for item in record.evidence:
            if item not in tags:
                tags.append(item)
    return {
        "schema": RETRY_STOPPED_SCHEMA,
        "class": first.failure_class,
        "subclass": first.subclass,
        "reason": (
            "every residual blocking issue is certain non-encoder class "
            + ", ".join(classes)
        ),
        "evidence": list(_evidence(*tags, *issues)),
        "issue_count": len(records),
    }


def _retry_stopped_record(outcome: Mapping[str, Any]) -> FailureClassRecord | None:
    stopped = outcome.get("retry_stopped")
    if not isinstance(stopped, Mapping):
        return None
    failure_class = stopped.get("class")
    subclass = stopped.get("subclass")
    if failure_class not in NON_ENCODER_RETRY_STOP_CLASSES or (
        subclass is not None
        and subclass not in FAILURE_CLASSES.get(str(failure_class), ())
    ):
        return None
    evidence = stopped.get("evidence")
    return _rule(
        str(failure_class),
        subclass if isinstance(subclass, str) else None,
        f"status:{outcome.get('status')}",
        "retry_stopped",
        *(evidence if isinstance(evidence, list) else ()),
    )


def classify_outcome(outcome: Mapping[str, Any]) -> FailureClassRecord | None:
    """Classify one recorded encode outcome from certain signals only.

    Returns ``None`` for a successful outcome. A failed outcome whose cause is
    not certain from its recorded signals is ``unknown``: issue text alone is
    never read as an encoder or validator fault.
    """

    if not isinstance(outcome, Mapping):
        return _unknown("outcome:not-a-mapping")
    if outcome.get("final_success") is True:
        return None
    status = outcome.get("status")
    if status == PRE_ENCODE_BLOCKED_STATUS:
        finding = _blocking_gate_finding(outcome)
        if finding is None:
            return _unknown(f"status:{status}", "pre_encode_gate:no-enforced-finding")
        failure_class = finding.get("class")
        subclass = finding.get("subclass")
        if failure_class not in FAILURE_CLASSES or (
            subclass is not None
            and subclass not in FAILURE_CLASSES.get(str(failure_class), ())
        ):
            return _unknown(f"status:{status}", f"check:{finding.get('check')}")
        return _rule(
            str(failure_class),
            subclass if isinstance(subclass, str) else None,
            f"status:{status}",
            f"pre_encode_gate:{finding.get('check')}",
        )
    stopped = _retry_stopped_record(outcome)
    if stopped is not None:
        return stopped
    if outcome.get("timed_out") is True or outcome.get("failure_kind") == "timeout":
        return _rule(INFRA, None, f"status:{status}", "timeout")
    for key in ("apply_error", "primary_error"):
        detail = outcome.get(key)
        if isinstance(detail, str) and detail.startswith(_HARNESS_ISSUE_PREFIXES):
            return _rule(ENCODER, "harness", f"status:{status}", detail)
    return _unknown(f"status:{status}")


def failure_class_payload(outcome: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the outcome's ``failure_class`` value in its recorded dict form."""

    record = classify_outcome(outcome)
    return None if record is None else record.to_dict()

"""Failure classes for encode outcomes (docs/design/encode-failure-classes.md).

A failure class says who owns the fix for a failed encode. It is advisory: it
never waives a failure and never admits an artifact.

``classify_outcome`` uses only deterministic signals whose mapping to a class
is certain. Everything else is ``unknown``; a later triage step (not in this
module) may classify those with a lower confidence.
"""

from __future__ import annotations

from collections.abc import Mapping
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

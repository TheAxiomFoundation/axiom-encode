"""Judge event model + adapter onto the canonical run log (``axiom_encode.run_log.v1``).

Part 2 of the maximum-traceability program (the LLM judge stages). The canonical
event-sourced run log landed in :mod:`axiom_encode.run_log` (PR #1086) and
declares a forward-compatible ``judge`` :class:`~axiom_encode.run_log.PipelineStage`
whose event shape was agreed on #1087. Judge stages therefore emit **into that
canonical schema** — there is no second schema.

Each stage builds a rich internal :class:`JudgeEvent` (verdict, confidence,
escalation, token spend, structured findings). :meth:`JudgeEvent.to_run_log_event`
maps it onto the canonical :class:`~axiom_encode.run_log.RunLogEvent`
(``stage="judge"``; verdict/model/tokens in ``attrs``; judge findings mapped onto
canonical :class:`~axiom_encode.run_log.Finding`), and :meth:`JudgeEvent.emit`
appends it via the canonical :class:`~axiom_encode.run_log.RunLogWriter` to the
same ``run_id.jsonl`` as the encode run — so the per-run DAG and funnel fold the
judge stage automatically.

Design invariants encoded here:

* Every emitted event carries the canonical ``schema`` and ``stage == "judge"``
  and validates against the canonical model.
* ``verdict == "error"`` is the fail-closed state: the judge could not reach a
  verdict (API failure, parse failure, cross-family guard). It maps to
  ``status == "error"`` and NEVER to a pass. :func:`error_event` is the only
  sanctioned way to build one, and it always sets ``judge_error``.
* ``verdict == "skip"`` is the pre-classifier's route-to-skip-with-reason path;
  it maps to ``status == "skipped"`` (also never a pass). A skip may still carry
  a visible ``judge_error`` (a failed arbiter that was safely skipped, not
  dropped).
* Advisory first: a ``flag`` verdict maps to ``status == "passed"`` (the judge
  stage completed; it does not gate), carrying the findings + ``needs-review``.
  Promotion to a hard gate flips ``advisory`` off, mapping ``flag -> failed``
  with ``reason_code == "judge_rejected"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from axiom_encode.run_log import (
    SCHEMA_VERSION,
    RunLogEvent,
    RunLogWriter,
    Severity,
    StageStatus,
    utc_now_iso,
)
from axiom_encode.run_log import Finding as RunLogFinding

# The canonical pipeline stage id (declared in axiom_encode.run_log.PIPELINE_SPEC)
# that every judge event is folded under. ``SCHEMA_VERSION`` is re-exported from
# the canonical module so judge callers keep a single import site.
JUDGE_STAGE = "judge"


class JudgeStage(str, Enum):
    """The judge stage that produced an event (carried in ``attrs.judge_stage``)."""

    STATUTORY_FIDELITY = "statutory_fidelity"
    GRID_ADEQUACY = "grid_adequacy"
    DISPOSITION = "disposition"
    WORKLIST_PRECLASSIFY = "worklist_preclassify"
    GOLDEN_DRIFT = "golden_drift"


class Verdict(str, Enum):
    """Structured judge verdict (carried in ``attrs.verdict``).

    ``ERROR`` is the fail-closed state (never a pass); ``SKIP`` is the
    pre-classifier's route-to-skip-with-reason path (also not a pass).
    """

    PASS = "pass"
    FLAG = "flag"
    ERROR = "error"
    SKIP = "skip"


# Extensible finding taxonomy. Each kind maps to a canonical ``Finding.code`` and
# a severity. Kept as a mapping so tests and the run-log consumer can validate a
# ``code`` without importing every stage module.
_SEVERITY_BY_KIND: dict[str, Severity] = {
    "judge_error": Severity.critical,
    "amount_mismatch": Severity.critical,
    "unrepresented_clause": Severity.important,
    "untraceable_branch": Severity.important,
    "boundary_direction": Severity.important,
    "claim_mismatch": Severity.important,
    "drift": Severity.important,
    "insufficient_records": Severity.minor,
    "untested_region": Severity.minor,
    "routing": Severity.info,
}

FINDING_KINDS = frozenset(_SEVERITY_BY_KIND)


def coerce_confidence(value: Any) -> Optional[float]:
    """Safely coerce a model-supplied confidence to a clamped [0,1] float.

    Returns ``None`` for a missing or non-numeric value rather than raising — a
    model that answers ``"high"`` must not crash a stage (and, in the
    pre-classifier's batch path, take every other entry down with it). Values are
    clamped to [0,1] so an emitted event always satisfies ``validate_event_dict``.
    """

    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 0.0:
        f = 0.0
    elif f > 1.0:
        f = 1.0
    return round(f, 4)


def _round_conf(value: Optional[float]) -> Optional[float]:
    return coerce_confidence(value)


@dataclass(frozen=True)
class Finding:
    """A single machine-checkable judge finding.

    ``clause_ref`` locates the provision; ``rule_path`` locates the generated
    artifact locus; ``kind`` is drawn from :data:`FINDING_KINDS`. Mapped onto a
    canonical :class:`~axiom_encode.run_log.Finding` at emission time.
    """

    clause_ref: str
    rule_path: str
    kind: str
    explanation: str

    def to_run_log_finding(self) -> RunLogFinding:
        message = self.explanation
        if self.clause_ref:
            message = (
                f"[{self.clause_ref}] {message}" if message else f"[{self.clause_ref}]"
            )
        return RunLogFinding(
            code=self.kind,
            severity=_SEVERITY_BY_KIND.get(self.kind, Severity.info),
            message=message,
            locator=self.rule_path or self.clause_ref or None,
            evidence=None,
        )


@dataclass(frozen=True)
class TokenCounts:
    """Judge token spend, logged into every event for cost control."""

    input: int = 0
    output: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"input": int(self.input), "output": int(self.output)}

    def __add__(self, other: "TokenCounts") -> "TokenCounts":
        return TokenCounts(self.input + other.input, self.output + other.output)


@dataclass(frozen=True)
class JudgeError:
    """Populated iff ``verdict == error`` (or a preclassify skip-with-error).

    Fail-open is banned: an error is always visible in ``attrs.judge_error``.
    """

    type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "message": self.message}


@dataclass
class JudgeEvent:
    """A single judge verdict, mapped onto ``axiom_encode.run_log.v1`` on emit."""

    stage: JudgeStage
    verdict: Verdict
    confidence: Optional[float] = None
    advisory: bool = True
    findings: list[Finding] = field(default_factory=list)
    model: Optional[str] = None
    generator_model: Optional[str] = None
    escalated: bool = False
    tokens: TokenCounts = field(default_factory=TokenCounts)
    judge_error: Optional[JudgeError] = None
    run_id: Optional[str] = None
    subject_ref: Optional[str] = None
    # Optional sha256 of the judge prompt, for reproducibility auditing.
    judge_prompt_sha256: Optional[str] = None
    # Stage-specific machine-readable payload the deterministic follow-up hooks
    # consume (grid gaps -> suggested cells, disposition arithmetic, preclassify
    # routing). Kept off ``findings`` so findings stay uniform across stages.
    extra: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_error(self) -> bool:
        return self.verdict == Verdict.ERROR

    @property
    def passed(self) -> bool:
        # ERROR and SKIP are explicitly NOT passes.
        return self.verdict == Verdict.PASS

    @property
    def needs_review(self) -> bool:
        """Advisory label signal: a flagged referee wants human review."""
        return self.verdict == Verdict.FLAG

    # -- canonical mapping ------------------------------------------------

    @property
    def status(self) -> StageStatus:
        """Coarse canonical pipeline status for this verdict."""
        if self.verdict == Verdict.ERROR:
            return StageStatus.error
        if self.verdict == Verdict.SKIP:
            return StageStatus.skipped
        if self.verdict == Verdict.FLAG:
            # Advisory-first: a flag does not gate the pipeline. A promoted
            # (non-advisory) referee maps a flag to a hard failure.
            return StageStatus.passed if self.advisory else StageStatus.failed
        return StageStatus.passed

    @property
    def reason_code(self) -> Optional[str]:
        """Machine-readable reason for a non-passed status (Pareto key)."""
        if self.verdict == Verdict.ERROR:
            return self.judge_error.type if self.judge_error else "judge_error"
        if self.verdict == Verdict.SKIP:
            return str(self.extra.get("classification") or "skipped")
        if self.verdict == Verdict.FLAG and not self.advisory:
            return "judge_rejected"
        return None

    @property
    def reason(self) -> Optional[str]:
        """Human-readable reason string."""
        if self.verdict == Verdict.ERROR:
            return self.judge_error.message if self.judge_error else "judge error"
        if self.verdict == Verdict.SKIP:
            return str(self.extra.get("reason") or "") or (
                self.findings[0].explanation if self.findings else None
            )
        if self.verdict == Verdict.FLAG:
            if self.findings:
                return (
                    self.findings[0].explanation or f"{len(self.findings)} finding(s)"
                )
            return "advisory flag"
        return None

    def run_log_attrs(self) -> dict[str, Any]:
        """The judge-specific ``attrs`` payload on the canonical event."""
        attrs: dict[str, Any] = {
            "judge_stage": self.stage.value,
            "verdict": self.verdict.value,
            "advisory": bool(self.advisory),
            "escalated": bool(self.escalated),
            # Token budget is logged on every judge event (cost control).
            "tokens": self.tokens.to_dict(),
        }
        if self.model:
            attrs["judge_model"] = self.model
        if self.generator_model:
            attrs["generator_model"] = self.generator_model
        if self.confidence is not None:
            attrs["confidence"] = _round_conf(self.confidence)
        if self.judge_prompt_sha256:
            attrs["judge_prompt_sha256"] = self.judge_prompt_sha256
        if self.judge_error is not None:
            attrs["judge_error"] = self.judge_error.to_dict()
        # Merge stage-specific extra without clobbering the structured fields
        # above (notably a preclassify skip carries judge_error in ``extra``).
        for key, value in self.extra.items():
            attrs.setdefault(key, value)
        return attrs

    def run_log_findings(self) -> list[RunLogFinding]:
        return [f.to_run_log_finding() for f in self.findings]

    def to_run_log_event(
        self,
        *,
        run_id: Optional[str] = None,
        seq: int = 0,
        ts: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> RunLogEvent:
        """Build the canonical ``axiom_encode.run_log.v1`` event for this verdict."""
        return RunLogEvent(
            run_id=run_id or self.run_id or "",
            seq=seq,
            ts=ts or self.created_at or utc_now_iso(),
            stage=JUDGE_STAGE,
            status=self.status,
            reason_code=self.reason_code,
            reason=self.reason,
            duration_ms=duration_ms,
            attrs=self.run_log_attrs(),
            findings=self.run_log_findings(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the canonical run-log wire shape (``schema`` key = v1)."""
        return self.to_run_log_event().to_dict()

    def emit(
        self, writer: RunLogWriter, *, duration_ms: Optional[int] = None
    ) -> Optional[RunLogEvent]:
        """Append this judge event to a run log via the canonical writer.

        Returns the written :class:`~axiom_encode.run_log.RunLogEvent`, or ``None``
        if the writer is disabled or captured an IO error (logging never raises).
        """
        return writer.emit(
            JUDGE_STAGE,
            self.status,
            reason_code=self.reason_code,
            reason=self.reason,
            duration_ms=duration_ms,
            attrs=self.run_log_attrs(),
            findings=self.run_log_findings(),
        )


def error_event(
    stage: JudgeStage,
    message: str,
    *,
    error_type: str = "judge_error",
    model: Optional[str] = None,
    generator_model: Optional[str] = None,
    tokens: Optional[TokenCounts] = None,
    escalated: bool = False,
    run_id: Optional[str] = None,
    subject_ref: Optional[str] = None,
) -> JudgeEvent:
    """Build a fail-closed ``judge_error`` event.

    This is the ONLY sanctioned construction of an error verdict. A judge that
    cannot reach a verdict must return this — never a pass. The error is also
    surfaced as a ``judge_error`` finding so a consumer scanning ``findings``
    still sees it, and lands in ``attrs.judge_error`` so a ``status == "error"``
    event always carries its cause.
    """

    return JudgeEvent(
        stage=stage,
        verdict=Verdict.ERROR,
        confidence=None,
        advisory=True,
        findings=[
            Finding(
                clause_ref=subject_ref or "",
                rule_path=subject_ref or "",
                kind="judge_error",
                explanation=message,
            )
        ],
        model=model,
        generator_model=generator_model,
        escalated=escalated,
        tokens=tokens or TokenCounts(),
        judge_error=JudgeError(type=error_type, message=message),
        run_id=run_id,
        subject_ref=subject_ref,
    )


def validate_event_dict(payload: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty == valid) for a judge event dict.

    Used by tests and by any run-log consumer that wants to reject malformed
    judge events. Enforces the load-bearing fail-closed invariant: an error must
    carry its ``judge_error`` and must NEVER read as a pass (fail-open cannot
    masquerade as handled).
    """

    problems: list[str] = []
    if payload.get("schema") != SCHEMA_VERSION:
        problems.append(f"schema must be {SCHEMA_VERSION!r}")
    if payload.get("stage") != JUDGE_STAGE:
        problems.append(f"stage must be {JUDGE_STAGE!r}")

    status = payload.get("status")
    valid_status = {s.value for s in StageStatus}
    if status not in valid_status:
        problems.append(f"unknown status {status!r}")

    attrs = payload.get("attrs") or {}
    verdict = attrs.get("verdict")
    if verdict not in {v.value for v in Verdict}:
        problems.append(f"unknown verdict {verdict!r} in attrs")

    judge_error = attrs.get("judge_error")
    if status == StageStatus.error.value and not judge_error:
        problems.append("error status must carry attrs.judge_error (fail-open banned)")
    if verdict == Verdict.ERROR.value and status != StageStatus.error.value:
        problems.append("error verdict must map to status 'error'")
    if judge_error and status == StageStatus.passed.value:
        problems.append("judge_error present on a passed status (fail-open banned)")
    if judge_error and verdict == Verdict.PASS.value:
        problems.append("judge_error present on a pass verdict (fail-open banned)")

    conf = attrs.get("confidence")
    if conf is not None:
        try:
            if not (0.0 <= float(conf) <= 1.0):
                problems.append("confidence out of [0,1]")
        except (TypeError, ValueError):
            problems.append("confidence is not numeric")

    for i, finding in enumerate(payload.get("findings") or []):
        for key in ("code", "severity", "message"):
            if key not in finding:
                problems.append(f"finding[{i}] missing {key}")
        if finding.get("code") not in FINDING_KINDS:
            problems.append(f"finding[{i}] unknown code {finding.get('code')!r}")

    # Finally, confirm it constructs as a canonical run-log event (schema, stage,
    # finding-shape validation) so the judge worker cannot emit anything the
    # renderer / audit cannot fold.
    try:
        RunLogEvent.from_dict(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced as a validation problem
        problems.append(f"not a valid run-log event: {exc}")

    return problems

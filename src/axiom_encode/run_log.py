"""axiom_encode.run_log.v1 - event-sourced encoding run log.

The encoding pipeline emits one JSON event per stage transition (JSONL, one
event per line). Stages are declared in a machine-readable pipeline spec
(:data:`PIPELINE_SPEC`) so a renderer can draw the step DAG and fold the
aggregate funnel without bespoke, stage-name-hardcoded parsing.

Design goals
------------
* **Versioned + forward compatible.** Every event carries ``schema`` and every
  spec carries ``pipeline_spec_version``. A ``judge`` stage is declared up front
  with a structured ``findings`` payload so the parallel judge worker can emit
  into this schema without a second migration (see the coordination issue linked
  from the PR).
* **Absent != fabricated.** Missing fields are explicitly ``null``. The backfill
  exporter never invents a verdict it cannot source.
* **Non-fatal.** :class:`RunLogWriter` swallows its own IO/serialisation errors so
  logging can never break an encoding run.

The published run log is the single source of truth for the dashboard: the
per-run step DAG, the ``generated -> gates-passed -> judged -> applied -> pr ->
merged`` funnel, and the failure Pareto are all folded from these events.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "axiom_encode.run_log.v1"
PIPELINE_SPEC_VERSION = "axiom_encode.pipeline_spec.v1"


class StageStatus(str, Enum):
    """Verdict of a single stage transition event."""

    started = "started"
    passed = "passed"
    failed = "failed"
    skipped = "skipped"
    error = "error"
    superseded = "superseded"


#: Statuses that represent a completed (non-in-flight) stage outcome.
TERMINAL_STATUSES = frozenset(
    {
        StageStatus.passed,
        StageStatus.failed,
        StageStatus.skipped,
        StageStatus.error,
        StageStatus.superseded,
    }
)


class StageCategory(str, Enum):
    """Coarse grouping used by the renderer to lay out the DAG columns."""

    generate = "generate"
    gate = "gate"
    judge = "judge"
    apply = "apply"
    downstream = "downstream"


@dataclass(frozen=True)
class PipelineStage:
    """One node in the pipeline DAG.

    ``depends_on`` are the ids of stages that must (nominally) precede this one;
    a renderer draws an edge for each. ``optional`` marks stages that may be
    absent from a given run's log without that run being considered broken
    (e.g. ``judge`` while the judge worker is still being built, or the
    downstream GitHub stages for a run that has not been applied yet).
    """

    id: str
    label: str
    category: StageCategory
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    description: str = ""


#: The pipeline DAG. Order is generate -> gates -> judge -> apply -> downstream.
#: Keep this list and ``STAGE_IDS`` in sync; adding a stage here is the ONLY
#: place a new stage id becomes valid (see :meth:`RunLogEvent.validate_stage`).
PIPELINE_SPEC: tuple[PipelineStage, ...] = (
    PipelineStage(
        id="generate",
        label="Generate",
        category=StageCategory.generate,
        depends_on=(),
        description="Model produces a candidate RuleSpec artifact from source text.",
    ),
    PipelineStage(
        id="gate.compile",
        label="Compile",
        category=StageCategory.gate,
        depends_on=("generate",),
        description="RuleSpec compiles under the engine.",
    ),
    PipelineStage(
        id="gate.grounding",
        label="Numeric grounding",
        category=StageCategory.gate,
        depends_on=("gate.compile",),
        description="Generated numeric literals are grounded in source text.",
    ),
    PipelineStage(
        id="gate.proof",
        label="Proof atoms",
        category=StageCategory.gate,
        depends_on=("gate.compile",),
        optional=True,
        description="Proof/source atoms validate against the cited provisions.",
    ),
    PipelineStage(
        id="gate.ci",
        label="CI tests",
        category=StageCategory.gate,
        depends_on=("gate.compile",),
        description="Companion RuleSpec tests pass under the engine.",
    ),
    PipelineStage(
        id="gate.oracle",
        label="Oracle",
        category=StageCategory.gate,
        depends_on=("gate.ci",),
        optional=True,
        description="PolicyEngine oracle agreement.",
    ),
    PipelineStage(
        id="gate.review",
        label="Reviewers",
        category=StageCategory.gate,
        depends_on=("gate.ci",),
        optional=True,
        description="Checklist reviewers (rulespec/formula/parameter/integration).",
    ),
    PipelineStage(
        id="judge",
        label="LLM judge",
        category=StageCategory.judge,
        depends_on=("gate.review",),
        optional=True,
        description="LLM judge verdict with structured findings (parallel worker).",
    ),
    PipelineStage(
        id="apply",
        label="Apply + manifest",
        category=StageCategory.apply,
        depends_on=("gate.ci",),
        description="Files installed into a rulespec repo; signed manifest written.",
    ),
    PipelineStage(
        id="pr",
        label="PR opened",
        category=StageCategory.downstream,
        depends_on=("apply",),
        optional=True,
        description="Pull request opened in the rulespec repo.",
    ),
    PipelineStage(
        id="ci_run",
        label="CI run",
        category=StageCategory.downstream,
        depends_on=("pr",),
        optional=True,
        description="Repo CI run(s) for the PR.",
    ),
    PipelineStage(
        id="merge",
        label="Merged",
        category=StageCategory.downstream,
        depends_on=("ci_run",),
        optional=True,
        description="PR merged to the rulespec repo default branch.",
    ),
    PipelineStage(
        id="oracle_at_merge",
        label="Oracle at merge",
        category=StageCategory.downstream,
        depends_on=("merge",),
        optional=True,
        description="Oracle-suite / conformance-row result captured at merge.",
    ),
)

#: The set of stage ids that are valid for an event. An event whose ``stage`` is
#: not in this set fails validation (see the unknown-stage negative test).
STAGE_IDS: frozenset[str] = frozenset(stage.id for stage in PIPELINE_SPEC)

#: Ordered funnel steps for the aggregate dashboard. Each entry maps a funnel
#: bucket to the stage whose final folded status must be exactly ``passed`` to
#: admit a run into that bucket (a ``skipped`` apply does not count as applied).
FUNNEL_STEPS: tuple[tuple[str, str], ...] = (
    ("generated", "generate"),
    ("gates_passed", "gate.ci"),
    ("judged", "judge"),
    ("applied", "apply"),
    ("pr", "pr"),
    ("merged", "merge"),
)


class Severity(str, Enum):
    """Severity of a judge/reviewer finding."""

    critical = "critical"
    important = "important"
    minor = "minor"
    info = "info"


class Finding(BaseModel):
    """A single structured finding.

    This is the shape the parallel judge worker emits into (one entry per
    ``findings`` item on a ``judge`` stage event). It intentionally mirrors the
    existing reviewer issue vocabulary (critical/important/minor) plus an ``info``
    level and a machine-readable ``code`` so failures aggregate into a Pareto.
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(
        description="Machine-readable finding code, e.g. 'ungrounded_numeric'."
    )
    severity: Severity = Severity.info
    message: str = ""
    locator: Optional[str] = Field(
        default=None,
        description="Where the finding applies, e.g. 'file.yaml#symbol' or a line ref.",
    )
    evidence: Optional[str] = Field(
        default=None, description="Optional supporting quote/excerpt."
    )


class RunLogEvent(BaseModel):
    """One pipeline stage-transition event.

    Validated on construction: ``stage`` must be declared in :data:`PIPELINE_SPEC`
    and ``schema`` must match :data:`SCHEMA_VERSION`. Round-trips losslessly via
    :meth:`to_json_line` / :meth:`from_json_line`.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=True)

    schema_: str = Field(default=SCHEMA_VERSION, alias="schema")
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    run_id: str
    seq: int = Field(ge=0, description="Monotonic per-run sequence number.")
    ts: str = Field(description="ISO-8601 UTC timestamp of the transition.")
    stage: str
    status: StageStatus
    reason_code: Optional[str] = Field(
        default=None,
        description="Machine-readable reason for a non-passed status (Pareto key).",
    )
    reason: Optional[str] = Field(default=None, description="Human-readable reason.")
    duration_ms: Optional[int] = Field(default=None, ge=0)
    attrs: dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific structured payload."
    )
    findings: list[Finding] = Field(default_factory=list)

    @field_validator("schema_")
    @classmethod
    def _validate_schema(cls, value: str) -> str:
        if value != SCHEMA_VERSION:
            raise ValueError(
                f"unknown run-log schema {value!r}; expected {SCHEMA_VERSION!r}"
            )
        return value

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        if value not in STAGE_IDS:
            raise ValueError(
                f"unknown pipeline stage {value!r}; declared stages are "
                f"{sorted(STAGE_IDS)}"
            )
        return value

    def to_dict(self) -> dict[str, Any]:
        """Serialise with the wire key ``schema`` (not the ``schema_`` alias)."""
        return self.model_dump(by_alias=True, mode="json")

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunLogEvent":
        return cls.model_validate(data)

    @classmethod
    def from_json_line(cls, line: str) -> "RunLogEvent":
        return cls.model_validate(json.loads(line))


def pipeline_spec_dict() -> dict[str, Any]:
    """Serialise the pipeline spec for the renderer.

    The dashboard fetches this once and draws the DAG from ``stages`` + edges;
    it never hardcodes stage names.
    """
    return {
        "pipeline_spec_version": PIPELINE_SPEC_VERSION,
        "run_log_schema": SCHEMA_VERSION,
        "statuses": [s.value for s in StageStatus],
        "funnel_steps": [
            {"bucket": bucket, "stage": stage} for bucket, stage in FUNNEL_STEPS
        ],
        "stages": [
            {
                "id": stage.id,
                "label": stage.label,
                "category": stage.category.value,
                "depends_on": list(stage.depends_on),
                "optional": stage.optional,
                "description": stage.description,
            }
            for stage in PIPELINE_SPEC
        ],
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_log_dir(explicit: Optional[Path] = None) -> Path:
    """Resolve the directory run-log JSONL files are written to.

    Precedence: explicit arg > ``AXIOM_ENCODE_RUN_LOG_DIR`` env > ``.axiom/run-logs``
    under the current working directory. Works for local, headless, and cloud
    paths (the cloud runner sets the env var).
    """
    if explicit is not None:
        return Path(explicit)
    env = os.environ.get("AXIOM_ENCODE_RUN_LOG_DIR")
    if env:
        return Path(env)
    return Path.cwd() / ".axiom" / "run-logs"


class RunLogWriter:
    """Append run-log events for a single ``run_id`` to a JSONL file.

    Non-fatal by construction: any IO or serialisation failure is captured on
    ``self.last_error`` and swallowed, so a logging problem can never abort an
    encoding run. Sequence numbers are assigned monotonically from the count of
    events already on disk (so re-opening an existing run's log continues it).
    """

    def __init__(
        self,
        run_id: str,
        *,
        log_dir: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self.run_id = run_id
        self.enabled = enabled and bool(run_id)
        self.dir = run_log_dir(log_dir)
        self.path = self.dir / f"{run_id}.jsonl"
        self.last_error: Optional[str] = None
        self._seq = self._existing_event_count()

    def _existing_event_count(self) -> int:
        try:
            if not self.path.exists():
                return 0
            with self.path.open("r", encoding="utf-8") as handle:
                return sum(1 for line in handle if line.strip())
        except OSError as exc:  # pragma: no cover - defensive
            self.last_error = str(exc)
            return 0

    def emit(
        self,
        stage: str,
        status: StageStatus | str,
        *,
        reason_code: Optional[str] = None,
        reason: Optional[str] = None,
        duration_ms: Optional[int] = None,
        attrs: Optional[dict[str, Any]] = None,
        findings: Optional[Iterable[Finding | dict[str, Any]]] = None,
        ts: Optional[str] = None,
    ) -> Optional[RunLogEvent]:
        """Validate and append one event. Returns the event, or ``None`` on error."""
        if not self.enabled:
            return None
        try:
            event = RunLogEvent(
                run_id=self.run_id,
                seq=self._seq,
                ts=ts or utc_now_iso(),
                stage=stage,
                status=StageStatus(status),
                reason_code=reason_code,
                reason=reason,
                duration_ms=duration_ms,
                attrs=attrs or {},
                findings=[
                    f if isinstance(f, Finding) else Finding.model_validate(f)
                    for f in (findings or [])
                ],
            )
            self.dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json_line() + "\n")
            self._seq += 1
            return event
        except Exception as exc:  # noqa: BLE001 - logging must never raise
            self.last_error = str(exc)
            return None


def iter_events(path: Path) -> Iterator[RunLogEvent]:
    """Yield validated events from a run-log JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield RunLogEvent.from_json_line(line)


class RunRecord(BaseModel):
    """A folded per-run summary derived from a run's events.

    Purely derived - never authored - so it introduces no new ground truth. The
    dashboard funnel and Pareto read the published :class:`RunRecord` index for
    speed; the per-run DAG view reads the raw events.
    """

    run_id: str
    schema_: str = Field(default=SCHEMA_VERSION, alias="schema")
    citation: Optional[str] = None
    legal_id: Optional[str] = None
    backend: Optional[str] = None
    model: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    event_count: int = 0
    #: stage id -> best-known terminal status value (or 'started' if only in-flight).
    stage_status: dict[str, str] = Field(default_factory=dict)
    #: stage id -> total duration_ms observed for that stage.
    stage_duration_ms: dict[str, int] = Field(default_factory=dict)
    #: funnel bucket -> bool reached.
    funnel: dict[str, bool] = Field(default_factory=dict)
    #: first failing (gate/judge) stage id, if the run failed anywhere.
    first_failure_stage: Optional[str] = None
    first_failure_reason_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


def fold_run(events: Iterable[RunLogEvent]) -> RunRecord:
    """Fold an event stream into a :class:`RunRecord` summary."""
    ordered = sorted(events, key=lambda e: (e.seq, e.ts))
    record = RunRecord(run_id=ordered[0].run_id if ordered else "")
    record.event_count = len(ordered)
    stage_reason: dict[str, Optional[str]] = {}

    for event in ordered:
        if record.started_at is None:
            record.started_at = event.ts
        record.ended_at = event.ts

        # Pull identity attrs from wherever they first appear.
        for key in ("citation", "legal_id", "backend", "model"):
            value = event.attrs.get(key)
            if value and getattr(record, key) is None:
                setattr(record, key, value)

        status = StageStatus(event.status)
        prior = record.stage_status.get(event.stage)
        # Terminal statuses win over 'started'; later terminal statuses win over
        # earlier ones (e.g. a retried gate that passes after failing).
        if prior is None or (
            status in TERMINAL_STATUSES or prior == StageStatus.started.value
        ):
            record.stage_status[event.stage] = status.value

        if event.duration_ms:
            record.stage_duration_ms[event.stage] = (
                record.stage_duration_ms.get(event.stage, 0) + event.duration_ms
            )

        if status in (StageStatus.failed, StageStatus.error):
            stage_reason[event.stage] = event.reason_code or (
                event.findings[0].code if event.findings else None
            )

    # Derive the first failure from the FINAL folded status in pipeline order, so
    # a gate that failed then passed on retry is not counted as both a pass
    # (funnel) and a first failure (Pareto) - keeping the two views consistent.
    for stage in PIPELINE_SPEC:
        if record.stage_status.get(stage.id) in (
            StageStatus.failed.value,
            StageStatus.error.value,
        ):
            record.first_failure_stage = stage.id
            record.first_failure_reason_code = stage_reason.get(stage.id)
            break

    for bucket, stage_id in FUNNEL_STEPS:
        record.funnel[bucket] = record.stage_status.get(stage_id) == (
            StageStatus.passed.value
        )
    return record


def fold_run_from_file(path: Path) -> RunRecord:
    return fold_run(iter_events(path))

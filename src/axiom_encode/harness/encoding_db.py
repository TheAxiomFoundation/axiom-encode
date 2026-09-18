"""
Encoding Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.

Now also tracks full session transcripts for replay and analysis.

Schema
======
The local SQLite database (``encodings.db``) is created and migrated by
:meth:`EncodingDB._init_db`. Every statement there is idempotent and additive
(``CREATE TABLE IF NOT EXISTS``, ``ALTER TABLE ... ADD COLUMN`` guarded by the
duplicate-column error), so opening an old database upgrades it in place and
never drops rows. The one in-place update is the long-standing cost
normalisation on ``sessions`` (a $0 estimate with no recorded usage becomes
NULL), which is idempotent.

``encoding_runs``
    One row per ``axiom-encode encode`` invocation (``id`` is the run id).
    ``iterations_json`` holds one entry per generation attempt:
    ``{attempt, duration_ms, success, errors, model?, <token counters>?}``.
    Each ``errors`` entry is ``{error_type, message, variable, fix_applied,
    issues?, issues_truncated?}`` where ``issues`` is the validator's
    structured issue list for that attempt (see
    :mod:`axiom_encode.harness.validation_issues`). ``parent_run_id`` links a
    regeneration of the same citation to the run it followed and
    ``iteration`` is the parent's iteration plus one. ``outcome_json`` is the
    final encode/apply outcome. The token and cost columns are the run's
    aggregate ledger.
``sessions`` / ``session_events``
    The per-run session transcript (``sessions.id`` is ``encode-<run id>``
    for encode runs). ``encode_result`` events carry ``metadata_json.attempts``
    (one entry per attempt with its structured issues and artifact version
    ids); ``encode_issue`` events carry ``metadata_json.issues`` (the final
    failure's structured issues) next to the legacy ``repair_manifest`` path.
``artifact_versions``
    Every generated artifact version the encoder validated, one row per
    attempt and file role: ``artifact_type`` is ``rulespec`` or
    ``rulespec_tests``; ``content`` is the full text; ``content_hash`` is its
    SHA-256; ``version_label`` is ``attempt-<n>``; ``metadata_json`` records
    the validator outcome for that version (gate pass flags, issue count,
    error, model). ``effective_from`` is the attempt timestamp.
``run_artifacts``
    Links a run to its artifact versions with ``attempt`` and ``role``, so
    ``(run, attempt)`` resolves to the exact RuleSpec and test text that the
    attempt's issues describe.
``judge_events``
    Durable mirror of ``axiom_encode.run_log.v1`` ``judge`` stage events,
    keyed by the canonical ``event_id``: verdict, confidence, model, token
    spend, the canonical findings (``findings_json``), the judge's own
    findings with ``clause_ref`` and ``rule_path`` kept separate
    (``judge_findings_json``; entries carry ``derived: true`` when unfolded
    from a canonical event rather than supplied by the emitting judge),
    ``subject_ref`` (the citation the judge ruled on, supplied at emission),
    and the full event JSON, plus ``source`` (``live`` for events mirrored at
    emission, ``backfill:<path>`` for events ingested from a run-log file).
``calibration_snapshots``
    Per-metric calibration history (see :mod:`axiom_encode.harness.metrics`).

:mod:`axiom_encode.attempt_evidence` is the read-only view over these tables
that yields ``(run_id, attempt, artifact, issues, parent)`` records.
"""

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from .validation_issues import (
    ValidationIssue,
    issues_from_dicts,
    issues_to_dicts,
)

# Per-run token/cost ledger columns, shared with the Supabase sync so the
# local schema, the payload, and the fallback ladder cannot drift apart.
RUN_COST_COLUMNS = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "estimated_cost_usd",
    "actual_cost_usd",
    "generation_attempt_count",
)

RUN_COLUMNS = (
    "id",
    "timestamp",
    "citation",
    "file_path",
    "source_text",
    "complexity_json",
    "iterations_json",
    "total_duration_ms",
    "agent_type",
    "agent_model",
    "rulespec_content",
    "session_id",
    "iteration",
    "parent_run_id",
    "review_results_json",
    "lessons",
    "axiom_encode_version",
    "outcome_json",
) + RUN_COST_COLUMNS

SESSION_COLUMNS = (
    "id",
    "run_id",
    "started_at",
    "ended_at",
    "model",
    "cwd",
    "event_count",
    "total_tokens",
    "axiom_encode_version",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "estimated_cost_usd",
)


@dataclass
class TokenUsage:
    """Token usage for a session or run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def has_recorded_usage(self) -> bool:
        """True when any token counter is non-zero (usage was measured)."""
        return any(getattr(self, name) for name in TOKEN_USAGE_FIELDS)


# TokenUsage counters by name, so the attempt aggregation, the recorded-usage
# check, and the sync payload cannot silently disagree on the field list.
TOKEN_USAGE_FIELDS = tuple(f.name for f in fields(TokenUsage))

# Session token columns added by migration 007 (nullable remotely).
SESSION_LEDGER_COLUMNS = ("cache_creation_tokens", "reasoning_output_tokens")


# Session event types
EventType = Literal[
    "session_start",
    "session_end",
    "user_prompt",
    "assistant_response",
    "tool_call",
    "tool_result",
    "subagent_start",
    "subagent_end",
    "agent_start",
    "agent_assistant",
    "agent_end",
    "provenance_plan",
    "provenance_decision",
    "provenance_reasoning",
    "provenance_artifact",
    "provenance_sidecar",
    "provenance_validation",
    "provenance_review",
    # Validation events (3-tier pipeline)
    "validation_ci_start",
    "validation_ci_end",
    "validation_oracle_start",
    "validation_oracle_end",
    "validation_llm_start",
    "validation_llm_end",
    # Eval-backed encode events
    "encode_request",
    "encode_result",
    "encode_outcome",
    "encode_issue",
]


@dataclass
class SessionEvent:
    """A single event in a session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    sequence: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # EventType
    tool_name: Optional[str] = None
    content: str = ""  # Main content (prompt, response, tool input/output)
    metadata: dict = field(default_factory=dict)  # Extra data (tokens, duration, etc.)


@dataclass
class Session:
    """A full Claude Code session transcript."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None  # FK to EncodingRun if this is an encoding session
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    model: str = ""
    cwd: str = ""
    axiom_encode_version: str = ""
    event_count: int = 0
    total_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


@dataclass
class ComplexityFactors:
    """Upfront analysis of statute complexity."""

    cross_references: list[str] = field(default_factory=list)  # ["1402(a)", "164(f)"]
    has_nested_structure: bool = False
    has_numeric_thresholds: bool = False
    has_phase_in_out: bool = False
    estimated_variables: int = 1
    estimated_parameters: int = 0


@dataclass
class IterationError:
    """An error encountered during encoding.

    ``message`` is the attempt's one-line verdict; ``issues`` is the
    validator's structured issue list for the artifact version that attempt
    produced (empty for runs recorded before issues were persisted).
    ``issues_truncated`` counts issues dropped to stay within the per-attempt
    storage bound.
    """

    error_type: str  # "parse", "test", "import", "style", "other"
    message: str
    variable: Optional[str] = None  # Which variable failed, if applicable
    fix_applied: Optional[str] = None  # What fix was attempted
    issues: list[ValidationIssue] = field(default_factory=list)
    issues_truncated: int = 0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error_type": self.error_type,
            "message": self.message,
            "variable": self.variable,
            "fix_applied": self.fix_applied,
        }
        if self.issues:
            issue_dicts, dropped = issues_to_dicts(self.issues)
            payload["issues"] = issue_dicts
            truncated = self.issues_truncated + dropped
            if truncated:
                payload["issues_truncated"] = truncated
        elif self.issues_truncated:
            payload["issues_truncated"] = self.issues_truncated
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IterationError":
        truncated = data.get("issues_truncated")
        return cls(
            error_type=data["error_type"],
            message=data["message"],
            variable=data.get("variable"),
            fix_applied=data.get("fix_applied"),
            issues=issues_from_dicts(data.get("issues")),
            issues_truncated=int(truncated)
            if isinstance(truncated, int) and not isinstance(truncated, bool)
            else 0,
        )


ARTIFACT_TYPE_RULESPEC = "rulespec"
ARTIFACT_TYPE_RULESPEC_TESTS = "rulespec_tests"


def content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


@dataclass
class ArtifactVersion:
    """One generated artifact version the encoder validated."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    artifact_type: str = ARTIFACT_TYPE_RULESPEC
    content_hash: str = ""
    version_label: Optional[str] = None
    content: Optional[str] = None
    effective_from: str = field(default_factory=lambda: datetime.now().isoformat())
    effective_to: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RunArtifact:
    """A run's link to one artifact version, labeled by attempt and role."""

    run_id: str
    artifact_version_id: str
    attempt: Optional[int] = None
    role: Optional[str] = None


@dataclass(frozen=True)
class ParentRunRef:
    """The prior run a regeneration links to (identity only, no payloads)."""

    id: str
    citation: str = ""
    iteration: int = 1
    agent_type: str = ""
    agent_model: str = ""
    timestamp: Optional[str] = None
    ended_at: Optional[str] = None


ARTIFACT_VERSION_COLUMNS = (
    "id",
    "artifact_type",
    "content_hash",
    "version_label",
    "content",
    "effective_from",
    "effective_to",
    "metadata_json",
)

JUDGE_EVENT_COLUMNS = (
    "id",
    "run_id",
    "seq",
    "ts",
    "judge_stage",
    "verdict",
    "status",
    "reason_code",
    "reason",
    "confidence",
    "advisory",
    "escalated",
    "judge_model",
    "generator_model",
    "input_tokens",
    "output_tokens",
    "judge_error_json",
    "judge_prompt_sha256",
    "subject_ref",
    "duration_ms",
    "findings_json",
    "attrs_json",
    "event_json",
    "source",
    "ingested_at",
    "judge_findings_json",
)


@dataclass
class JudgeEventRow:
    """One persisted judge verdict (a mirror of a run-log ``judge`` event)."""

    id: str
    run_id: str
    seq: Optional[int] = None
    ts: Optional[str] = None
    judge_stage: Optional[str] = None
    verdict: Optional[str] = None
    status: Optional[str] = None
    reason_code: Optional[str] = None
    reason: Optional[str] = None
    confidence: Optional[float] = None
    advisory: Optional[bool] = None
    escalated: Optional[bool] = None
    judge_model: Optional[str] = None
    generator_model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    judge_error: Optional[dict] = None
    judge_prompt_sha256: Optional[str] = None
    subject_ref: Optional[str] = None
    duration_ms: Optional[int] = None
    findings: list[dict] = field(default_factory=list)
    attrs: dict = field(default_factory=dict)
    event: dict = field(default_factory=dict)
    source: str = "live"
    ingested_at: Optional[str] = None
    #: The judge's findings with ``clause_ref`` and ``rule_path`` separate.
    judge_findings: list[dict] = field(default_factory=list)


def unfold_canonical_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover ``clause_ref``/``rule_path`` from canonical judge findings.

    :meth:`axiom_encode.judges.run_log.Finding.to_run_log_finding` folds the
    clause into a ``[clause] `` message prefix and puts the rule path (or,
    failing that, the clause) into ``locator``. This reverses that fold for
    events read back from a run log; each entry is marked ``derived``.
    """
    unfolded: list[dict[str, Any]] = []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        message = str(finding.get("message") or "")
        clause_ref = ""
        explanation = message
        if message.startswith("["):
            close = message.find("]")
            if close > 0:
                clause_ref = message[1:close]
                explanation = message[close + 1 :].lstrip()
        locator = finding.get("locator")
        rule_path = ""
        if isinstance(locator, str) and locator and locator != clause_ref:
            rule_path = locator
        unfolded.append(
            {
                "clause_ref": clause_ref,
                "rule_path": rule_path,
                "kind": str(finding.get("code") or ""),
                "explanation": explanation,
                "derived": True,
            }
        )
    return unfolded


def judge_event_row_from_event(
    event: dict[str, Any],
    *,
    source: str = "live",
    subject_ref: Optional[str] = None,
    judge_findings: Optional[list[dict[str, Any]]] = None,
) -> JudgeEventRow:
    """Project a canonical run-log ``judge`` event dict onto a table row.

    ``subject_ref`` and ``judge_findings`` are supplied by the emitting judge
    (the canonical event does not carry them separately); when absent the
    findings are unfolded from the canonical shape.
    """
    attrs = event.get("attrs") or {}
    if not isinstance(attrs, dict):
        attrs = {}
    tokens = attrs.get("tokens") or {}
    if not isinstance(tokens, dict):
        tokens = {}
    judge_error = attrs.get("judge_error")
    findings = event.get("findings") or []
    event_id = str(event.get("event_id") or "")
    run_id = str(event.get("run_id") or "")
    if not event_id:
        digest_source = json.dumps(event, sort_keys=True, default=str)
        event_id = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:16]
    confidence = attrs.get("confidence")

    def _opt_int(value: object) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value

    def _opt_bool(value: object) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    resolved_subject = subject_ref if subject_ref else attrs.get("subject_ref")
    canonical_findings = [item for item in findings if isinstance(item, dict)]
    resolved_judge_findings = (
        [dict(item) for item in judge_findings if isinstance(item, dict)]
        if judge_findings is not None
        else unfold_canonical_findings(canonical_findings)
    )
    return JudgeEventRow(
        id=event_id,
        run_id=run_id,
        seq=_opt_int(event.get("seq")),
        ts=str(event["ts"]) if event.get("ts") is not None else None,
        judge_stage=attrs.get("judge_stage"),
        verdict=attrs.get("verdict"),
        status=event.get("status"),
        reason_code=event.get("reason_code"),
        reason=event.get("reason"),
        confidence=float(confidence)
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
        else None,
        advisory=_opt_bool(attrs.get("advisory")),
        escalated=_opt_bool(attrs.get("escalated")),
        judge_model=attrs.get("judge_model"),
        generator_model=attrs.get("generator_model"),
        input_tokens=_opt_int(tokens.get("input")),
        output_tokens=_opt_int(tokens.get("output")),
        judge_error=judge_error if isinstance(judge_error, dict) else None,
        judge_prompt_sha256=attrs.get("judge_prompt_sha256"),
        subject_ref=resolved_subject if isinstance(resolved_subject, str) else None,
        duration_ms=_opt_int(event.get("duration_ms")),
        findings=canonical_findings,
        attrs=attrs,
        event=event,
        source=source,
        judge_findings=resolved_judge_findings,
    )


@dataclass
class Iteration:
    """A single encoding attempt.

    ``model`` and the token counters record what this attempt actually spent, so a
    run whose attempts escalated across models (Terra, then Sol) can be re-priced
    from its own record. ``None`` means the attempt did not report usage.
    """

    attempt: int
    duration_ms: int
    errors: list[IterationError] = field(default_factory=list)
    success: bool = False
    model: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_creation_tokens: Optional[int] = None
    reasoning_output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None


ITERATION_USAGE_FIELDS = (
    "model",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_output_tokens",
    "estimated_cost_usd",
)


@dataclass
class OracleResult:
    """Detailed result from an oracle validator."""

    name: str
    score: Optional[float] = None  # Match rate 0-1
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    duration_ms: int = 0
    test_cases_run: int = 0
    test_cases_passed: int = 0


@dataclass
class ReviewResult:
    """Result from a single reviewer (checklist-based)."""

    reviewer: str = ""  # "rulespec", "formula", "parameter", "integration"
    passed: bool = False
    items_checked: int = 0
    items_passed: int = 0
    critical_issues: list[str] = field(default_factory=list)
    important_issues: list[str] = field(default_factory=list)
    minor_issues: list[str] = field(default_factory=list)
    lessons: str = ""


@dataclass
class ReviewResults:
    """Aggregated results from all reviewers."""

    reviews: list[ReviewResult] = field(default_factory=list)
    policyengine_match: Optional[float] = None
    oracle_context: dict = field(default_factory=dict)
    lessons: str = ""

    @property
    def passed(self) -> bool:
        """All reviews passed."""
        return all(r.passed for r in self.reviews) if self.reviews else False

    @property
    def total_critical_issues(self) -> int:
        """Total critical issues across all reviews."""
        return sum(len(r.critical_issues) for r in self.reviews)


@dataclass
class EncodingRun:
    """A complete encoding run from start to finish."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # What we're encoding
    citation: str = ""
    file_path: str = ""
    source_text: Optional[str] = None

    # Upfront analysis
    complexity: ComplexityFactors = field(default_factory=ComplexityFactors)

    # Review results
    review_results: Optional[ReviewResults] = None
    lessons: str = ""

    # Iteration tracking
    iteration: int = 1
    parent_run_id: Optional[str] = None

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    rulespec_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = ""
    axiom_encode_version: str = ""
    outcome: dict = field(default_factory=dict)

    # Model usage across all generation attempts (cache-aware)
    tokens: TokenUsage = field(default_factory=TokenUsage)
    # None means "unknown" (some attempt lacked usage/pricing), never 0.
    estimated_cost_usd: Optional[float] = None
    actual_cost_usd: Optional[float] = None
    generation_attempt_count: int = 0

    # Session linkage
    session_id: Optional[str] = None

    @property
    def iterations_needed(self) -> int:
        return len(self.iterations)

    @property
    def success(self) -> bool:
        if isinstance(self.outcome, dict) and "final_success" in self.outcome:
            return bool(self.outcome["final_success"])
        return self.iterations and self.iterations[-1].success

    @property
    def all_errors(self) -> list[IterationError]:
        errors = []
        for it in self.iterations:
            errors.extend(it.errors)
        return errors


def create_run(
    file_path: str,
    citation: str,
    agent_type: str,
    agent_model: str,
    rulespec_content: str,
    source_text: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    review_results: Optional[ReviewResults] = None,
    lessons: str = "",
) -> EncodingRun:
    """Factory function to create an EncodingRun with defaults."""
    from axiom_encode import __version__

    return EncodingRun(
        file_path=file_path,
        citation=citation,
        agent_type=agent_type,
        agent_model=agent_model,
        rulespec_content=rulespec_content,
        source_text=source_text,
        iteration=2 if parent_run_id else 1,
        parent_run_id=parent_run_id,
        review_results=review_results,
        lessons=lessons,
        axiom_encode_version=__version__,
    )


class EncodingDB:
    """SQLite-based encoding database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Encoding runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encoding_runs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                citation TEXT,
                file_path TEXT,
                source_text TEXT,
                complexity_json TEXT,
                iterations_json TEXT,
                total_duration_ms INTEGER,
                agent_type TEXT,
                agent_model TEXT,
                rulespec_content TEXT,
                session_id TEXT,
                iteration INTEGER DEFAULT 1,
                parent_run_id TEXT,
                review_results_json TEXT,
                lessons TEXT DEFAULT '',
                axiom_encode_version TEXT DEFAULT '',
                outcome_json TEXT DEFAULT '{}',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_creation_tokens INTEGER DEFAULT 0,
                reasoning_output_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                generation_attempt_count INTEGER DEFAULT 0
            )
        """)

        # Keep active developer databases usable when the schema grows.
        for col, col_type, default in [
            ("source_text", "TEXT", None),
            ("rulespec_content", "TEXT", None),
            ("session_id", "TEXT", None),
            ("iteration", "INTEGER", "1"),
            ("parent_run_id", "TEXT", None),
            ("review_results_json", "TEXT", None),
            ("lessons", "TEXT", "''"),
            ("axiom_encode_version", "TEXT", "''"),
            ("outcome_json", "TEXT", "'{}'"),
            ("input_tokens", "INTEGER", "0"),
            ("output_tokens", "INTEGER", "0"),
            ("cache_read_tokens", "INTEGER", "0"),
            ("cache_creation_tokens", "INTEGER", "0"),
            ("reasoning_output_tokens", "INTEGER", "0"),
            ("estimated_cost_usd", "REAL", None),
            ("actual_cost_usd", "REAL", None),
            ("generation_attempt_count", "INTEGER", "0"),
        ]:
            try:
                stmt = f"ALTER TABLE encoding_runs ADD COLUMN {col} {col_type}"
                if default is not None:
                    stmt += f" DEFAULT {default}"
                cursor.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON encoding_runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON encoding_runs(timestamp)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_parent
            ON encoding_runs(parent_run_id)
        """)

        # =====================================================================
        # Attempt evidence: artifact versions per attempt, run links, judge
        # verdicts. Additive; databases that predate these tables (or carry
        # the 2025 SCD2 shape of artifact_versions/run_artifacts) upgrade in
        # place without touching existing rows.
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifact_versions (
                id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                version_label TEXT,
                content TEXT,
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                metadata_json TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_type
            ON artifact_versions(artifact_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_hash
            ON artifact_versions(content_hash)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_artifacts (
                run_id TEXT NOT NULL,
                artifact_version_id TEXT NOT NULL,
                attempt INTEGER,
                role TEXT,
                PRIMARY KEY (run_id, artifact_version_id),
                FOREIGN KEY (run_id) REFERENCES encoding_runs(id),
                FOREIGN KEY (artifact_version_id) REFERENCES artifact_versions(id)
            )
        """)
        for col, col_type in (("attempt", "INTEGER"), ("role", "TEXT")):
            try:
                cursor.execute(f"ALTER TABLE run_artifacts ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_run_artifacts_run
            ON run_artifacts(run_id, attempt)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS judge_events (
                id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                seq INTEGER,
                ts TEXT,
                judge_stage TEXT,
                verdict TEXT,
                status TEXT,
                reason_code TEXT,
                reason TEXT,
                confidence REAL,
                advisory INTEGER,
                escalated INTEGER,
                judge_model TEXT,
                generator_model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                judge_error_json TEXT,
                judge_prompt_sha256 TEXT,
                subject_ref TEXT,
                duration_ms INTEGER,
                findings_json TEXT,
                attrs_json TEXT,
                event_json TEXT,
                source TEXT,
                ingested_at TEXT,
                judge_findings_json TEXT
            )
        """)
        try:
            cursor.execute(
                "ALTER TABLE judge_events ADD COLUMN judge_findings_json TEXT"
            )
        except sqlite3.OperationalError:
            pass  # Column already exists
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_judge_events_run
            ON judge_events(run_id, seq)
        """)

        # Calibration snapshots table (per-metric rows for trend analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                predicted_mean REAL,
                actual_mean REAL,
                mse REAL,
                n_samples INTEGER
            )
        """)

        # Sessions table - full Claude Code session transcripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES encoding_runs(id)
            )
        """)

        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN run_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_run ON sessions(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_started ON sessions(started_at)
        """)

        # Session events table - individual events within a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_session ON session_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON session_events(event_type)
        """)

        # =====================================================================
        # Add token usage columns to sessions (migration)
        # =====================================================================
        for col in [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "reasoning_output_tokens",
            "estimated_cost_usd",
            "axiom_encode_version",
        ]:
            try:
                if col == "axiom_encode_version":
                    col_type = "TEXT DEFAULT ''"
                elif col == "estimated_cost_usd":
                    # No default: NULL means "cost unknown", which must stay
                    # distinguishable from a measured $0.
                    col_type = "REAL"
                else:
                    col_type = "INTEGER DEFAULT 0"
                cursor.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        # Databases migrated before the ledger created that column as
        # "REAL DEFAULT 0", and SQLite cannot drop a column default. A session
        # with no recorded usage and a $0 cost was never measured, so restore
        # NULL (unknown) for those rows. The update is idempotent.
        cursor.execute(
            """
            UPDATE sessions SET estimated_cost_usd = NULL
            WHERE estimated_cost_usd = 0
              AND COALESCE(input_tokens, 0) = 0
              AND COALESCE(output_tokens, 0) = 0
              AND COALESCE(cache_read_tokens, 0) = 0
              AND COALESCE(cache_creation_tokens, 0) = 0
              AND COALESCE(reasoning_output_tokens, 0) = 0
            """
        )

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun) -> str:
        """Log a completed encoding run. Returns the run ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert dataclasses to JSON
        complexity_json = json.dumps(
            {
                "cross_references": run.complexity.cross_references,
                "has_nested_structure": run.complexity.has_nested_structure,
                "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
                "has_phase_in_out": run.complexity.has_phase_in_out,
                "estimated_variables": run.complexity.estimated_variables,
                "estimated_parameters": run.complexity.estimated_parameters,
            }
        )

        iterations_json = json.dumps(
            [
                {
                    "attempt": it.attempt,
                    "duration_ms": it.duration_ms,
                    "success": it.success,
                    "errors": [e.to_dict() for e in it.errors],
                    **{
                        name: getattr(it, name)
                        for name in ITERATION_USAGE_FIELDS
                        if getattr(it, name) is not None
                    },
                }
                for it in run.iterations
            ]
        )

        # Serialize review_results (new checklist-based format)
        review_results_json = None
        if run.review_results:
            review_results_json = json.dumps(
                {
                    "reviews": [
                        {
                            "reviewer": r.reviewer,
                            "passed": r.passed,
                            "items_checked": r.items_checked,
                            "items_passed": r.items_passed,
                            "critical_issues": r.critical_issues,
                            "important_issues": r.important_issues,
                            "minor_issues": r.minor_issues,
                            "lessons": r.lessons,
                        }
                        for r in run.review_results.reviews
                    ],
                    "policyengine_match": run.review_results.policyengine_match,
                    "oracle_context": run.review_results.oracle_context,
                    "lessons": run.review_results.lessons,
                }
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO encoding_runs
            (id, timestamp, citation, file_path, source_text, complexity_json,
             iterations_json, total_duration_ms, agent_type, agent_model,
             rulespec_content, session_id, iteration, parent_run_id,
             review_results_json, lessons, axiom_encode_version, outcome_json,
             input_tokens, output_tokens, cache_read_tokens,
             cache_creation_tokens, reasoning_output_tokens,
             estimated_cost_usd, actual_cost_usd, generation_attempt_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run.id,
                run.timestamp.isoformat(),
                run.citation,
                run.file_path,
                run.source_text,
                complexity_json,
                iterations_json,
                run.total_duration_ms,
                run.agent_type,
                run.agent_model,
                run.rulespec_content,
                run.session_id,
                run.iteration,
                run.parent_run_id,
                review_results_json,
                run.lessons,
                run.axiom_encode_version,
                json.dumps(run.outcome or {}),
                run.tokens.input_tokens,
                run.tokens.output_tokens,
                run.tokens.cache_read_tokens,
                run.tokens.cache_creation_tokens,
                run.tokens.reasoning_output_tokens,
                run.estimated_cost_usd,
                run.actual_cost_usd,
                run.generation_attempt_count,
            ),
        )

        conn.commit()
        conn.close()

        return run.id

    def update_run_outcome(self, run_id: str, outcome: dict) -> None:
        """Update final encode/apply outcome metadata for a run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE encoding_runs SET outcome_json = ? WHERE id = ?",
            (json.dumps(outcome or {}, sort_keys=True), run_id),
        )

        conn.commit()
        conn.close()

    # =========================================================================
    # Parent links (regenerations of the same citation)
    # =========================================================================

    def find_parent_run(
        self,
        *,
        citation: str,
        started_at: datetime,
        agent_type: Optional[str] = None,
        agent_model: Optional[str] = None,
        exclude_run_id: Optional[str] = None,
    ) -> Optional["ParentRunRef"]:
        """Return the run this citation's new run regenerates, if any.

        A prior run of the same citation qualifies only when it had finished
        (its session ended, or failing that its row was written) before the
        new run's generation started, so concurrent sibling runs from a
        fan-out never link to each other. Among qualifying runs the most
        recent one on the same backend and model wins, then the most recent
        one on any model.
        """
        if not citation:
            return None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT r.id, r.citation, r.iteration, r.agent_type, r.agent_model,
                   r.timestamp, s.ended_at
            FROM encoding_runs r
            LEFT JOIN sessions s ON s.id = r.session_id
            WHERE r.citation = ?
              AND r.id != ?
              AND COALESCE(s.ended_at, r.timestamp) <= ?
            ORDER BY
              (COALESCE(r.agent_type, '') = ? AND COALESCE(r.agent_model, '') = ?)
                DESC,
              r.timestamp DESC
            LIMIT 1
            """,
            (
                citation,
                exclude_run_id or "",
                started_at.isoformat(),
                agent_type or "",
                agent_model or "",
            ),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return ParentRunRef(
            id=row[0],
            citation=row[1] or "",
            iteration=int(row[2] or 1),
            agent_type=row[3] or "",
            agent_model=row[4] or "",
            timestamp=row[5],
            ended_at=row[6],
        )

    def update_run_parent(
        self, run_id: str, parent_run_id: Optional[str], iteration: int
    ) -> None:
        """Set the parent link and iteration number of an existing run."""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE encoding_runs SET parent_run_id = ?, iteration = ? WHERE id = ?",
            (parent_run_id, iteration, run_id),
        )
        conn.commit()
        conn.close()

    # =========================================================================
    # Artifact versions
    # =========================================================================

    def record_run_artifact(
        self,
        run_id: str,
        *,
        attempt: int,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
        version_label: Optional[str] = None,
        effective_from: Optional[str] = None,
    ) -> ArtifactVersion:
        """Store one attempt's artifact text and link it to the run.

        Idempotent per ``(run_id, attempt, role, content)``: recording the same
        text again returns the existing version. Different text for the same
        attempt and role (a post-validation auto-repair, for example) becomes
        a new version linked to the same attempt.
        """
        digest = content_sha256(content)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {", ".join("v." + column for column in ARTIFACT_VERSION_COLUMNS)}
            FROM run_artifacts ra
            JOIN artifact_versions v ON v.id = ra.artifact_version_id
            WHERE ra.run_id = ? AND ra.attempt = ? AND ra.role = ?
              AND v.content_hash = ?
            LIMIT 1
            """,
            (run_id, attempt, role, digest),
        )
        existing = cursor.fetchone()
        if existing:
            conn.close()
            return _artifact_version_from_row(existing)
        version = ArtifactVersion(
            artifact_type=role,
            content_hash=digest,
            version_label=version_label or f"attempt-{attempt}",
            content=content,
            effective_from=effective_from or datetime.now().isoformat(),
            metadata=dict(metadata or {}),
        )
        cursor.execute(
            f"""
            INSERT INTO artifact_versions ({", ".join(ARTIFACT_VERSION_COLUMNS)})
            VALUES ({", ".join("?" for _ in ARTIFACT_VERSION_COLUMNS)})
            """,
            (
                version.id,
                version.artifact_type,
                version.content_hash,
                version.version_label,
                version.content,
                version.effective_from,
                version.effective_to,
                json.dumps(version.metadata, sort_keys=True, default=str),
            ),
        )
        cursor.execute(
            """
            INSERT OR REPLACE INTO run_artifacts
            (run_id, artifact_version_id, attempt, role)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, version.id, attempt, role),
        )
        conn.commit()
        conn.close()
        return version

    def update_run_artifact_metadata(
        self, run_id: str, *, attempt: int, patch: dict[str, Any]
    ) -> int:
        """Merge ``patch`` into the metadata of every version of one attempt.

        Used when the final verdict arrives after the row was written (the
        apply validator runs after the durability boundary). Returns the
        number of versions updated.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT v.id, v.metadata_json
            FROM run_artifacts ra
            JOIN artifact_versions v ON v.id = ra.artifact_version_id
            WHERE ra.run_id = ? AND ra.attempt = ?
            """,
            (run_id, attempt),
        )
        rows = cursor.fetchall()
        for version_id, metadata_json in rows:
            try:
                metadata = json.loads(metadata_json) if metadata_json else {}
            except (TypeError, ValueError):
                metadata = {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.update(patch)
            cursor.execute(
                "UPDATE artifact_versions SET metadata_json = ? WHERE id = ?",
                (json.dumps(metadata, sort_keys=True, default=str), version_id),
            )
        conn.commit()
        conn.close()
        return len(rows)

    def get_artifact_version(self, artifact_id: str) -> Optional[ArtifactVersion]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {', '.join(ARTIFACT_VERSION_COLUMNS)} FROM artifact_versions "
            "WHERE id = ?",
            (artifact_id,),
        )
        row = cursor.fetchone()
        conn.close()
        return _artifact_version_from_row(row) if row else None

    def get_run_artifacts(
        self, run_id: str
    ) -> list[tuple[RunArtifact, ArtifactVersion]]:
        """Every artifact version linked to a run, ordered by attempt and role."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT ra.run_id, ra.artifact_version_id, ra.attempt, ra.role,
                   {", ".join("v." + column for column in ARTIFACT_VERSION_COLUMNS)}
            FROM run_artifacts ra
            JOIN artifact_versions v ON v.id = ra.artifact_version_id
            WHERE ra.run_id = ?
            ORDER BY ra.attempt, ra.role, v.effective_from, ra.rowid
            """,
            (run_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            (
                RunArtifact(
                    run_id=row[0],
                    artifact_version_id=row[1],
                    attempt=row[2],
                    role=row[3],
                ),
                _artifact_version_from_row(row[4:]),
            )
            for row in rows
        ]

    # =========================================================================
    # Judge events
    # =========================================================================

    def log_judge_event(
        self,
        event: dict[str, Any],
        *,
        source: str = "live",
        subject_ref: Optional[str] = None,
        judge_findings: Optional[list[dict[str, Any]]] = None,
    ) -> Optional[JudgeEventRow]:
        """Mirror one canonical run-log ``judge`` event into ``judge_events``.

        Keyed by the event's ``event_id``, so re-ingesting a run log is a
        no-op. Returns the stored row, or ``None`` when the event already
        existed or is not a judge-stage event. ``subject_ref`` and
        ``judge_findings`` come from the emitting judge (see
        :func:`judge_event_row_from_event`).
        """
        if not isinstance(event, dict) or event.get("stage") != "judge":
            return None
        row = judge_event_row_from_event(
            event,
            source=source,
            subject_ref=subject_ref,
            judge_findings=judge_findings,
        )
        if not row.run_id:
            return None
        row.ingested_at = datetime.now().isoformat()
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"""
            INSERT OR IGNORE INTO judge_events ({", ".join(JUDGE_EVENT_COLUMNS)})
            VALUES ({", ".join("?" for _ in JUDGE_EVENT_COLUMNS)})
            """,
            (
                row.id,
                row.run_id,
                row.seq,
                row.ts,
                row.judge_stage,
                row.verdict,
                row.status,
                row.reason_code,
                row.reason,
                row.confidence,
                None if row.advisory is None else int(row.advisory),
                None if row.escalated is None else int(row.escalated),
                row.judge_model,
                row.generator_model,
                row.input_tokens,
                row.output_tokens,
                json.dumps(row.judge_error, sort_keys=True)
                if row.judge_error is not None
                else None,
                row.judge_prompt_sha256,
                row.subject_ref,
                row.duration_ms,
                json.dumps(row.findings, sort_keys=True, default=str),
                json.dumps(row.attrs, sort_keys=True, default=str),
                json.dumps(row.event, sort_keys=True, default=str),
                row.source,
                row.ingested_at,
                json.dumps(row.judge_findings, sort_keys=True, default=str),
            ),
        )
        inserted = cursor.rowcount == 1
        conn.commit()
        conn.close()
        return row if inserted else None

    def get_judge_events(self, run_id: str) -> list[JudgeEventRow]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {', '.join(JUDGE_EVENT_COLUMNS)} FROM judge_events "
            "WHERE run_id = ? ORDER BY seq, ts",
            (run_id,),
        )
        rows = cursor.fetchall()
        conn.close()
        return [_judge_event_row_from_db(row) for row in rows]

    def get_run(self, run_id: str) -> Optional[EncodingRun]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs WHERE id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_recent_runs(self, limit: int = 20) -> list[EncodingRun]:
        """Get recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs "
            "ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_runs_for_citation(self, citation: str) -> list[EncodingRun]:
        """Get all runs for a specific citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs "
            "WHERE citation = ? ORDER BY timestamp DESC",
            (citation,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def update_review_results(self, run_id: str, review_results: ReviewResults) -> None:
        """Update a run with review results after validation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        review_results_json = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": r.reviewer,
                        "passed": r.passed,
                        "items_checked": r.items_checked,
                        "items_passed": r.items_passed,
                        "critical_issues": r.critical_issues,
                        "important_issues": r.important_issues,
                        "minor_issues": r.minor_issues,
                        "lessons": r.lessons,
                    }
                    for r in review_results.reviews
                ],
                "policyengine_match": review_results.policyengine_match,
                "oracle_context": review_results.oracle_context,
                "lessons": review_results.lessons,
            }
        )

        cursor.execute(
            "UPDATE encoding_runs SET review_results_json = ? WHERE id = ?",
            (review_results_json, run_id),
        )

        conn.commit()
        conn.close()

    def get_error_stats(self) -> dict:
        """Get error type distribution."""
        runs = self.get_recent_runs(limit=100)

        error_counts = {}
        for run in runs:
            for err in run.all_errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

        total = sum(error_counts.values())
        return {
            "counts": error_counts,
            "percentages": {
                k: v / total * 100 if total > 0 else 0 for k, v in error_counts.items()
            },
            "total_runs": len(runs),
            "total_errors": total,
        }

    def get_iteration_stats(self) -> dict:
        """Get iteration distribution."""
        runs = self.get_recent_runs(limit=100)

        iteration_counts = {}
        for run in runs:
            n = run.iterations_needed
            iteration_counts[n] = iteration_counts.get(n, 0) + 1

        total = len(runs)
        avg = (
            sum(n * c for n, c in iteration_counts.items()) / total if total > 0 else 0
        )

        return {
            "distribution": iteration_counts,
            "average": avg,
            "first_try_rate": iteration_counts.get(1, 0) / total * 100
            if total > 0
            else 0,
            "total_runs": total,
        }

    def _row_to_run(self, row) -> EncodingRun:
        """Convert a current-schema database row to EncodingRun."""
        return run_from_row(row)

    # =========================================================================
    # Session Logging Methods
    # =========================================================================

    def start_session(
        self,
        model: str = "",
        cwd: str = "",
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        axiom_encode_version: str = "",
        started_at: Optional[datetime] = None,
    ) -> Session:
        """Start a new session and return it.

        ``started_at`` records when the work actually began (an encode
        invocation passes the moment before its first generation attempt);
        it defaults to now for sessions created at their start.
        """
        session = Session(
            run_id=run_id,
            model=model,
            cwd=cwd or os.getcwd(),
            axiom_encode_version=axiom_encode_version,
        )
        if started_at is not None:
            session.started_at = started_at
        # Allow custom session_id for SDK orchestrator
        if session_id:
            session.id = session_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (
                id, run_id, started_at, model, cwd, event_count, total_tokens,
                estimated_cost_usd, axiom_encode_version
            )
            VALUES (?, ?, ?, ?, ?, 0, 0, NULL, ?)
        """,
            (
                session.id,
                session.run_id,
                session.started_at.isoformat(),
                session.model,
                session.cwd,
                session.axiom_encode_version,
            ),
        )

        conn.commit()
        conn.close()

        return session

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET ended_at = ? WHERE id = ?
        """,
            (datetime.now().isoformat(), session_id),
        )

        conn.commit()
        conn.close()

    def log_event(
        self,
        session_id: str,
        event_type: str,
        content: str = "",
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionEvent:
        """Log an event to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next sequence number
        cursor.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM session_events WHERE session_id = ?",
            (session_id,),
        )
        sequence = cursor.fetchone()[0]

        event = SessionEvent(
            session_id=session_id,
            sequence=sequence,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata=metadata or {},
        )

        cursor.execute(
            """
            INSERT INTO session_events (id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.id,
                event.session_id,
                event.sequence,
                event.timestamp.isoformat(),
                event.event_type,
                event.tool_name,
                event.content,
                json.dumps(event.metadata),
            ),
        )

        # Update event count
        cursor.execute(
            """
            UPDATE sessions SET event_count = event_count + 1 WHERE id = ?
        """,
            (session_id,),
        )

        conn.commit()
        conn.close()

        return event

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(SESSION_COLUMNS)}
            FROM sessions
            WHERE id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_session(row)

    @staticmethod
    def _row_to_session(row) -> Session:
        """Convert a SESSION_COLUMNS-ordered row to a Session."""
        if len(row) != len(SESSION_COLUMNS):
            raise ValueError(
                f"Expected {len(SESSION_COLUMNS)} session columns, got {len(row)}"
            )
        values = dict(zip(SESSION_COLUMNS, row))
        return Session(
            id=values["id"],
            run_id=values["run_id"],
            started_at=datetime.fromisoformat(values["started_at"])
            if values["started_at"]
            else datetime.now(),
            ended_at=datetime.fromisoformat(values["ended_at"])
            if values["ended_at"]
            else None,
            model=values["model"] or "",
            cwd=values["cwd"] or "",
            axiom_encode_version=values["axiom_encode_version"] or "",
            event_count=values["event_count"] or 0,
            total_tokens=values["total_tokens"] or 0,
            input_tokens=values["input_tokens"] or 0,
            output_tokens=values["output_tokens"] or 0,
            cache_read_tokens=values["cache_read_tokens"] or 0,
            cache_creation_tokens=values["cache_creation_tokens"] or 0,
            reasoning_output_tokens=values["reasoning_output_tokens"] or 0,
            estimated_cost_usd=(
                float(values["estimated_cost_usd"])
                if values["estimated_cost_usd"] is not None
                else None
            ),
        )

    def get_session_events(self, session_id: str) -> list[SessionEvent]:
        """Get all events for a session, ordered by sequence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json
            FROM session_events
            WHERE session_id = ?
            ORDER BY sequence
        """,
            (session_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(
                SessionEvent(
                    id=row[0],
                    session_id=row[1],
                    sequence=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                    if row[3]
                    else datetime.now(),
                    event_type=row[4] or "",
                    tool_name=row[5],
                    content=row[6] or "",
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            )

        return events

    def get_recent_sessions(self, limit: int = 20) -> list[Session]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT {", ".join(SESSION_COLUMNS)}
            FROM sessions
            ORDER BY started_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_session(row) for row in rows]

    def get_session_stats(self) -> dict:
        """Get session statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total = cursor.fetchone()[0]

        # Event type distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM session_events
            GROUP BY event_type
            ORDER BY count DESC
        """)
        event_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Tool usage
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM session_events
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 20
        """)
        tool_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Average events per session
        cursor.execute("SELECT AVG(event_count) FROM sessions")
        avg_events = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_sessions": total,
            "event_type_counts": event_counts,
            "tool_usage": tool_counts,
            "avg_events_per_session": round(avg_events, 1),
        }

    def update_session_tokens(
        self,
        session_id: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        reasoning_output_tokens: int = 0,
        estimated_cost_usd: float | None = None,
    ) -> None:
        """Accumulate token usage and its provider/model-aware cost estimate.

        Token counts add onto whatever the session has already recorded, so
        each pipeline stage (generation, repair, review) can report its own
        usage without clobbering earlier writes. The cost estimate follows
        the same rule, except that a token-spending increment with an unknown
        cost (``None``) poisons the session total to ``None``: a partial sum
        would read as a real, too-low figure. A zero-token increment without
        a cost leaves the total untouched, and once poisoned while tokens are
        recorded the total stays ``None``. The whole write is one UPDATE so
        concurrent stages cannot lose each other's cost contribution.
        """
        increment_token_sum = (
            input_tokens
            + output_tokens
            + cache_read_tokens
            + cache_creation_tokens
            + reasoning_output_tokens
        )
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions
            SET estimated_cost_usd = CASE
                    WHEN :has_cost = 0 AND :increment_tokens > 0 THEN NULL
                    WHEN :has_cost = 0 THEN estimated_cost_usd
                    WHEN estimated_cost_usd IS NULL
                         AND COALESCE(input_tokens, 0)
                             + COALESCE(output_tokens, 0)
                             + COALESCE(cache_read_tokens, 0)
                             + COALESCE(cache_creation_tokens, 0)
                             + COALESCE(reasoning_output_tokens, 0) > 0
                        THEN NULL
                    ELSE COALESCE(estimated_cost_usd, 0.0) + :cost
                END,
                input_tokens = COALESCE(input_tokens, 0) + :input,
                output_tokens = COALESCE(output_tokens, 0) + :output,
                cache_read_tokens = COALESCE(cache_read_tokens, 0) + :cache_read,
                cache_creation_tokens =
                    COALESCE(cache_creation_tokens, 0) + :cache_creation,
                reasoning_output_tokens =
                    COALESCE(reasoning_output_tokens, 0) + :reasoning,
                total_tokens = COALESCE(total_tokens, 0) + :input + :output
            WHERE id = :session_id
        """,
            {
                "session_id": session_id,
                "input": input_tokens,
                "output": output_tokens,
                "cache_read": cache_read_tokens,
                "cache_creation": cache_creation_tokens,
                "reasoning": reasoning_output_tokens,
                "increment_tokens": increment_token_sum,
                "has_cost": 0 if estimated_cost_usd is None else 1,
                "cost": float(estimated_cost_usd or 0.0),
            },
        )

        conn.commit()
        conn.close()


def _artifact_version_from_row(row) -> ArtifactVersion:
    values = dict(zip(ARTIFACT_VERSION_COLUMNS, row))
    metadata_json = values["metadata_json"]
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except (TypeError, ValueError):
        metadata = {}
    return ArtifactVersion(
        id=values["id"],
        artifact_type=values["artifact_type"] or "",
        content_hash=values["content_hash"] or "",
        version_label=values["version_label"],
        content=values["content"],
        effective_from=values["effective_from"] or "",
        effective_to=values["effective_to"],
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _judge_event_row_from_db(row) -> JudgeEventRow:
    values = dict(zip(JUDGE_EVENT_COLUMNS, row))

    def _load(text: object, default):
        if not text:
            return default
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return default

    return JudgeEventRow(
        id=values["id"],
        run_id=values["run_id"],
        seq=values["seq"],
        ts=values["ts"],
        judge_stage=values["judge_stage"],
        verdict=values["verdict"],
        status=values["status"],
        reason_code=values["reason_code"],
        reason=values["reason"],
        confidence=values["confidence"],
        advisory=None if values["advisory"] is None else bool(values["advisory"]),
        escalated=None if values["escalated"] is None else bool(values["escalated"]),
        judge_model=values["judge_model"],
        generator_model=values["generator_model"],
        input_tokens=values["input_tokens"],
        output_tokens=values["output_tokens"],
        judge_error=_load(values["judge_error_json"], None),
        judge_prompt_sha256=values["judge_prompt_sha256"],
        subject_ref=values["subject_ref"],
        duration_ms=values["duration_ms"],
        findings=_load(values["findings_json"], []),
        attrs=_load(values["attrs_json"], {}),
        event=_load(values["event_json"], {}),
        source=values["source"] or "live",
        ingested_at=values["ingested_at"],
        judge_findings=_load(values["judge_findings_json"], []),
    )


def run_from_row(row, *, strict: bool = True) -> EncodingRun:
    """Convert a ``RUN_COLUMNS``-ordered database row to an EncodingRun.

    With ``strict`` (the default, used by the live encoder) a
    ``review_results_json`` payload in an unsupported schema raises. Readers
    that only need the run's attempts and outcome (the attempt-evidence view,
    the backfill) pass ``strict=False`` and get ``review_results=None`` for
    such legacy rows instead of losing the run.
    """
    if len(row) != len(RUN_COLUMNS):
        raise ValueError(
            f"Expected {len(RUN_COLUMNS)} encoding run columns, got {len(row)}"
        )

    values = dict(zip(RUN_COLUMNS, row))
    id = values["id"]
    timestamp = values["timestamp"]
    citation = values["citation"]
    file_path = values["file_path"]
    source_text = values["source_text"]
    complexity_json = values["complexity_json"]
    iterations_json = values["iterations_json"]
    total_duration_ms = values["total_duration_ms"]
    agent_type = values["agent_type"]
    agent_model = values["agent_model"]
    rulespec_content = values["rulespec_content"]
    session_id = values["session_id"]
    iteration = values["iteration"]
    parent_run_id = values["parent_run_id"]
    review_results_json = values["review_results_json"]
    lessons = values["lessons"]
    axiom_encode_version = values["axiom_encode_version"]
    outcome_json = values["outcome_json"]
    tokens = TokenUsage(
        input_tokens=int(values["input_tokens"] or 0),
        output_tokens=int(values["output_tokens"] or 0),
        cache_read_tokens=int(values["cache_read_tokens"] or 0),
        cache_creation_tokens=int(values["cache_creation_tokens"] or 0),
        reasoning_output_tokens=int(values["reasoning_output_tokens"] or 0),
    )
    estimated_cost_usd = (
        float(values["estimated_cost_usd"])
        if values["estimated_cost_usd"] is not None
        else None
    )
    actual_cost_usd = (
        float(values["actual_cost_usd"])
        if values["actual_cost_usd"] is not None
        else None
    )
    generation_attempt_count = int(values["generation_attempt_count"] or 0)

    # Parse complexity
    c = json.loads(complexity_json) if complexity_json else {}
    complexity = ComplexityFactors(
        cross_references=c.get("cross_references", []),
        has_nested_structure=c.get("has_nested_structure", False),
        has_numeric_thresholds=c.get("has_numeric_thresholds", False),
        has_phase_in_out=c.get("has_phase_in_out", False),
        estimated_variables=c.get("estimated_variables", 1),
        estimated_parameters=c.get("estimated_parameters", 0),
    )

    # Parse iterations. Legacy rows may hold a non-list payload or non-object
    # entries; those carry no attempt record and are skipped rather than
    # making the whole run unreadable.
    iterations = []
    parsed_iterations = json.loads(iterations_json) if iterations_json else []
    if not isinstance(parsed_iterations, list):
        parsed_iterations = []
    for it_data in parsed_iterations:
        if not isinstance(it_data, dict):
            continue
        errors = [
            IterationError.from_dict(e)
            for e in it_data.get("errors") or []
            if isinstance(e, dict)
        ]
        iterations.append(
            Iteration(
                attempt=it_data["attempt"],
                duration_ms=it_data["duration_ms"],
                errors=errors,
                success=it_data.get("success", False),
                model=it_data.get("model"),
                input_tokens=it_data.get("input_tokens"),
                output_tokens=it_data.get("output_tokens"),
                cache_read_tokens=it_data.get("cache_read_tokens"),
                cache_creation_tokens=it_data.get("cache_creation_tokens"),
                reasoning_output_tokens=it_data.get("reasoning_output_tokens"),
                estimated_cost_usd=it_data.get("estimated_cost_usd"),
            )
        )

    # Parse review_results.
    review_results = None
    rr = None
    if review_results_json:
        rr = json.loads(review_results_json)
        allowed_fields = {
            "reviews",
            "policyengine_match",
            "oracle_context",
            "lessons",
        }
        if not isinstance(rr, dict) or set(rr) - allowed_fields:
            if strict:
                raise ValueError(
                    "encoding run review_results_json uses an unsupported schema"
                )
            rr = None
    if rr is not None:
        review_results = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer=r.get("reviewer", ""),
                    passed=r.get("passed", False),
                    items_checked=r.get("items_checked", 0),
                    items_passed=r.get("items_passed", 0),
                    critical_issues=r.get("critical_issues", []),
                    important_issues=r.get("important_issues", []),
                    minor_issues=r.get("minor_issues", []),
                    lessons=r.get("lessons", ""),
                )
                for r in rr.get("reviews", [])
            ],
            policyengine_match=rr.get("policyengine_match"),
            oracle_context=rr.get("oracle_context", {}),
            lessons=rr.get("lessons", ""),
        )

    return EncodingRun(
        id=id,
        timestamp=datetime.fromisoformat(timestamp),
        citation=citation,
        file_path=file_path,
        source_text=source_text,
        complexity=complexity,
        review_results=review_results,
        lessons=lessons or "",
        iteration=iteration or 1,
        parent_run_id=parent_run_id,
        iterations=iterations,
        total_duration_ms=total_duration_ms or 0,
        agent_type=agent_type or "encoder",
        agent_model=agent_model or "",
        axiom_encode_version=axiom_encode_version or "",
        outcome=json.loads(outcome_json) if outcome_json else {},
        rulespec_content=rulespec_content or "",
        tokens=tokens,
        estimated_cost_usd=estimated_cost_usd,
        actual_cost_usd=actual_cost_usd,
        generation_attempt_count=generation_attempt_count,
        session_id=session_id,
    )

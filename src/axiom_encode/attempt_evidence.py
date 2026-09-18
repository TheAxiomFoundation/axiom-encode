"""Read-only view over per-attempt encode evidence in ``encodings.db``.

The encoder persists, for every generation attempt, the artifact version it
validated (``artifact_versions`` + ``run_artifacts``), the validator's
structured issue list (``iterations_json`` errors), and the parent link between
regenerations of the same citation (``encoding_runs.parent_run_id``). This
module joins those into :class:`AttemptEvidence` records:

``(run_id, attempt, artifact, issues, parent)``

* ``artifact`` is the RuleSpec text the attempt produced (with its companion
  tests when captured), or ``None`` when the attempt's content was not
  retained (runs recorded before artifact persistence, or a retry whose
  candidate capture failed closed).
* ``issues`` is the attempt's structured validator issue list
  (:class:`~axiom_encode.harness.validation_issues.ValidationIssue`).
* ``parent`` is the attempt this one followed: the previous attempt of the
  same run, or for a run's first attempt the last attempt of its
  ``parent_run_id`` run. ``None`` for a first attempt with no parent run.

:func:`iter_repair_triples` derives the labeled cases a verifier consumes: a
failed version, the issues the validator raised against it, and the next
version that passed.

Every query opens the database read-only (``mode=ro``) and never runs the
schema migration, so it is safe against a database another process is
writing to and against a snapshot that must not change.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Sequence
from urllib.parse import quote

from .harness.encoding_db import (
    ARTIFACT_TYPE_RULESPEC,
    ARTIFACT_TYPE_RULESPEC_TESTS,
    ARTIFACT_VERSION_COLUMNS,
    RUN_COLUMNS,
    ArtifactVersion,
    EncodingRun,
    _artifact_version_from_row,
    run_from_row,
)
from .harness.validation_issues import ValidationIssue, issues_from_dicts

__all__ = [
    "AttemptArtifact",
    "AttemptEvidence",
    "AttemptRef",
    "RepairTriple",
    "iter_attempt_evidence",
    "iter_repair_triples",
    "open_readonly",
]


@dataclass(frozen=True)
class AttemptArtifact:
    """The artifact text one attempt produced."""

    rulespec: ArtifactVersion
    tests: Optional[ArtifactVersion] = None

    @property
    def content(self) -> str:
        return self.rulespec.content or ""

    @property
    def content_hash(self) -> str:
        return self.rulespec.content_hash

    @property
    def metadata(self) -> dict[str, Any]:
        return self.rulespec.metadata


@dataclass(frozen=True)
class AttemptRef:
    """Identifies one attempt and its artifact (used for ``parent``)."""

    run_id: str
    attempt: int
    artifact: Optional[AttemptArtifact]
    success: bool
    same_run: bool


@dataclass(frozen=True)
class AttemptEvidence:
    """One generation attempt with everything the validator recorded about it."""

    run_id: str
    attempt: int
    artifact: Optional[AttemptArtifact]
    issues: tuple[ValidationIssue, ...]
    parent: Optional[AttemptRef]
    success: bool
    citation: str
    model: Optional[str] = None
    error: Optional[str] = None
    parent_run_id: Optional[str] = None
    outcome: dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    issues_truncated: int = 0

    def as_tuple(
        self,
    ) -> tuple[
        str,
        int,
        Optional[AttemptArtifact],
        tuple[ValidationIssue, ...],
        Optional[AttemptRef],
    ]:
        """The ``(run_id, attempt, artifact, issues, parent)`` view."""
        return (self.run_id, self.attempt, self.artifact, self.issues, self.parent)


@dataclass(frozen=True)
class RepairTriple:
    """A failed version, the issues raised against it, and the version that passed."""

    bad: AttemptEvidence
    issues: tuple[ValidationIssue, ...]
    good: AttemptEvidence

    @property
    def citation(self) -> str:
        return self.bad.citation

    @property
    def cross_run(self) -> bool:
        return self.bad.run_id != self.good.run_id


def open_readonly(db_path: Path | str) -> sqlite3.Connection:
    """Open ``db_path`` read-only; raises ``sqlite3.OperationalError`` if absent."""
    path = Path(db_path)
    if not path.exists():
        raise sqlite3.OperationalError(f"encodings database not found: {path}")
    # SQLite URI filenames are percent-decoded and split on ``?``/``#``, so
    # the filesystem path must be quoted or a ``%41`` directory would open a
    # different file and a ``?`` in the path would fail to open at all.
    uri = "file:" + quote(path.resolve().as_posix(), safe="/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({name})")}


def _select_runs(
    conn: sqlite3.Connection,
    *,
    run_id: Optional[str],
    citation: Optional[str],
    limit: Optional[int],
    newest_first: bool,
) -> list[EncodingRun]:
    available = _table_columns(conn, "encoding_runs")
    columns = ", ".join(
        column if column in available else f"NULL AS {column}" for column in RUN_COLUMNS
    )
    clauses: list[str] = []
    params: list[Any] = []
    if run_id:
        clauses.append("id = ?")
        params.append(run_id)
    if citation:
        clauses.append("citation = ?")
        params.append(citation)
    sql = f"SELECT {columns} FROM encoding_runs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY timestamp " + ("DESC" if newest_first else "ASC") + ", id"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    runs: list[EncodingRun] = []
    for row in rows:
        try:
            runs.append(run_from_row(row, strict=False))
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    return runs


def _run_artifacts_by_attempt(
    conn: sqlite3.Connection, run_id: str
) -> dict[int, AttemptArtifact]:
    if not _table_exists(conn, "run_artifacts") or not _table_exists(
        conn, "artifact_versions"
    ):
        return {}
    link_columns = _table_columns(conn, "run_artifacts")
    if "attempt" not in link_columns or "role" not in link_columns:
        return {}
    rows = conn.execute(
        f"""
        SELECT ra.attempt, ra.role,
               {", ".join("v." + column for column in ARTIFACT_VERSION_COLUMNS)}
        FROM run_artifacts ra
        JOIN artifact_versions v ON v.id = ra.artifact_version_id
        WHERE ra.run_id = ?
        ORDER BY ra.attempt, v.effective_from, ra.rowid
        """,
        (run_id,),
    ).fetchall()
    rulespecs: dict[int, ArtifactVersion] = {}
    tests: dict[int, ArtifactVersion] = {}
    for row in rows:
        attempt, role = row[0], row[1]
        if not isinstance(attempt, int):
            continue
        version = _artifact_version_from_row(row[2:])
        # Later versions of the same attempt/role (a post-validation repair)
        # supersede earlier ones: every version of a run shares
        # ``effective_from``, so insertion order (``ra.rowid``) decides.
        if role == ARTIFACT_TYPE_RULESPEC:
            rulespecs[attempt] = version
        elif role == ARTIFACT_TYPE_RULESPEC_TESTS:
            tests[attempt] = version
    return {
        attempt: AttemptArtifact(rulespec=version, tests=tests.get(attempt))
        for attempt, version in rulespecs.items()
    }


def _final_artifact_from_run(run: EncodingRun) -> Optional[AttemptArtifact]:
    """Fallback for runs recorded before artifact persistence."""
    if not run.rulespec_content:
        return None
    from .harness.encoding_db import content_sha256

    version = ArtifactVersion(
        id=f"legacy:{run.id}",
        artifact_type=ARTIFACT_TYPE_RULESPEC,
        content_hash=content_sha256(run.rulespec_content),
        version_label=f"attempt-{max(len(run.iterations), 1)}",
        content=run.rulespec_content,
        effective_from=run.timestamp.isoformat(),
        metadata={"run_id": run.id, "legacy": True},
    )
    return AttemptArtifact(rulespec=version)


def _attempts_for_run(
    conn: sqlite3.Connection,
    run: EncodingRun,
    *,
    parent_tail: Optional[AttemptRef],
) -> list[AttemptEvidence]:
    artifacts = _run_artifacts_by_attempt(conn, run.id)
    iterations = list(run.iterations)
    if not iterations:
        iterations = []
    attempt_numbers = sorted({it.attempt for it in iterations} | set(artifacts))
    if not attempt_numbers:
        attempt_numbers = [1]
    final_attempt = attempt_numbers[-1]
    iteration_by_attempt = {it.attempt: it for it in iterations}
    records: list[AttemptEvidence] = []
    previous: Optional[AttemptRef] = parent_tail
    for attempt in attempt_numbers:
        iteration = iteration_by_attempt.get(attempt)
        artifact = artifacts.get(attempt)
        if artifact is None and attempt == final_attempt:
            artifact = _final_artifact_from_run(run)
        issues: list[ValidationIssue] = []
        truncated = 0
        error: Optional[str] = None
        if iteration is not None:
            for err in iteration.errors:
                issues.extend(err.issues)
                truncated += err.issues_truncated
                if error is None and err.message:
                    error = err.message
            # The writer already reconciled the workflow outcome into the
            # final iteration's verdict (an overlay failure marks it failed; a
            # blocked manifest after a clean validation leaves it passed), so
            # the per-attempt validator verdict is authoritative here. The
            # run-level ``final_success`` stays visible on ``outcome``.
            success = bool(iteration.success)
            model = iteration.model
        else:
            success = bool(run.success) if attempt == final_attempt else False
            model = None
        record = AttemptEvidence(
            run_id=run.id,
            attempt=attempt,
            artifact=artifact,
            issues=tuple(issues),
            parent=previous,
            success=success,
            citation=run.citation,
            model=model or run.agent_model or None,
            error=error,
            parent_run_id=run.parent_run_id,
            outcome=dict(run.outcome or {}) if attempt == final_attempt else {},
            timestamp=run.timestamp.isoformat(),
            issues_truncated=truncated,
        )
        records.append(record)
        previous = AttemptRef(
            run_id=run.id,
            attempt=attempt,
            artifact=artifact,
            success=success,
            same_run=True,
        )
    return records


def _parent_tail(
    conn: sqlite3.Connection, parent_run_id: Optional[str]
) -> Optional[AttemptRef]:
    if not parent_run_id:
        return None
    parents = _select_runs(
        conn, run_id=parent_run_id, citation=None, limit=1, newest_first=True
    )
    if not parents:
        return None
    parent = parents[0]
    attempts = _attempts_for_run(conn, parent, parent_tail=None)
    if not attempts:
        return None
    last = attempts[-1]
    return AttemptRef(
        run_id=last.run_id,
        attempt=last.attempt,
        artifact=last.artifact,
        success=last.success,
        same_run=False,
    )


def iter_attempt_evidence(
    db_path: Path | str,
    *,
    run_id: Optional[str] = None,
    citation: Optional[str] = None,
    limit: Optional[int] = None,
    failed_only: bool = False,
    newest_first: bool = True,
) -> Iterator[AttemptEvidence]:
    """Yield one :class:`AttemptEvidence` per generation attempt.

    ``limit`` bounds the number of runs scanned, not attempts. Attempts of one
    run are yielded in attempt order; runs are yielded newest first unless
    ``newest_first`` is ``False``. ``failed_only`` skips attempts whose
    validator verdict was a pass.
    """
    conn = open_readonly(db_path)
    try:
        runs = _select_runs(
            conn,
            run_id=run_id,
            citation=citation,
            limit=limit,
            newest_first=newest_first,
        )
        for run in runs:
            tail = _parent_tail(conn, run.parent_run_id)
            for record in _attempts_for_run(conn, run, parent_tail=tail):
                if failed_only and record.success:
                    continue
                yield record
    finally:
        conn.close()


def iter_repair_triples(
    db_path: Path | str,
    *,
    citation: Optional[str] = None,
    limit: Optional[int] = None,
    require_artifacts: bool = True,
    require_issues: bool = False,
) -> Iterator[RepairTriple]:
    """Yield ``(bad version, issues, good version)`` labeled cases.

    A triple is a failed attempt directly followed (within the run, or by the
    first attempt of a child run linked through ``parent_run_id``) by an
    attempt that passed. With ``require_artifacts`` both versions must have
    retained content, so the triple is a complete before/after pair; with
    ``require_issues`` the failed version must carry structured issues (runs
    recorded before issue persistence only have the one-line error).

    ``limit`` bounds the runs scanned, newest first (the same window as
    :func:`iter_attempt_evidence`); a parent run outside that window is
    fetched on demand so a cross-run repair whose child is in the window is
    still reported. Triples are yielded oldest child first.
    """
    conn = open_readonly(db_path)
    try:
        runs = _select_runs(
            conn, run_id=None, citation=citation, limit=limit, newest_first=True
        )
        runs.reverse()
        by_id: dict[str, list[AttemptEvidence]] = {}
        for run in runs:
            tail = _parent_tail(conn, run.parent_run_id)
            by_id[run.id] = _attempts_for_run(conn, run, parent_tail=tail)
        evidence_index: dict[tuple[str, int], AttemptEvidence] = {
            (record.run_id, record.attempt): record
            for records in by_id.values()
            for record in records
        }
        for records in by_id.values():
            for record in records:
                parent = record.parent
                if parent is None or not record.success or parent.success:
                    continue
                bad = evidence_index.get((parent.run_id, parent.attempt))
                if bad is None:
                    bad_runs = _select_runs(
                        conn,
                        run_id=parent.run_id,
                        citation=None,
                        limit=1,
                        newest_first=True,
                    )
                    if bad_runs:
                        bad_tail = _parent_tail(conn, bad_runs[0].parent_run_id)
                        for candidate in _attempts_for_run(
                            conn, bad_runs[0], parent_tail=bad_tail
                        ):
                            evidence_index[(candidate.run_id, candidate.attempt)] = (
                                candidate
                            )
                        bad = evidence_index.get((parent.run_id, parent.attempt))
                if bad is None:
                    continue
                if require_artifacts and (
                    bad.artifact is None or record.artifact is None
                ):
                    continue
                if require_issues and not bad.issues:
                    continue
                yield RepairTriple(bad=bad, issues=bad.issues, good=record)
    finally:
        conn.close()


def evidence_to_dict(record: AttemptEvidence) -> dict[str, Any]:
    """JSON-ready projection (artifact text included) of one record."""

    def _artifact(artifact: Optional[AttemptArtifact]) -> Optional[dict[str, Any]]:
        if artifact is None:
            return None
        payload: dict[str, Any] = {
            "artifact_version_id": artifact.rulespec.id,
            "content_hash": artifact.content_hash,
            "version_label": artifact.rulespec.version_label,
            "content": artifact.content,
            "metadata": artifact.metadata,
        }
        if artifact.tests is not None:
            payload["tests"] = {
                "artifact_version_id": artifact.tests.id,
                "content_hash": artifact.tests.content_hash,
                "content": artifact.tests.content,
            }
        return payload

    parent = record.parent
    return {
        "run_id": record.run_id,
        "attempt": record.attempt,
        "citation": record.citation,
        "success": record.success,
        "model": record.model,
        "error": record.error,
        "timestamp": record.timestamp,
        "parent_run_id": record.parent_run_id,
        "artifact": _artifact(record.artifact),
        "issues": [issue.to_dict() for issue in record.issues],
        "issues_truncated": record.issues_truncated,
        "parent": None
        if parent is None
        else {
            "run_id": parent.run_id,
            "attempt": parent.attempt,
            "success": parent.success,
            "same_run": parent.same_run,
            "artifact": _artifact(parent.artifact),
        },
        "outcome": record.outcome,
    }


def dump_attempt_evidence(
    db_path: Path | str,
    *,
    run_id: Optional[str] = None,
    citation: Optional[str] = None,
    limit: Optional[int] = None,
    failed_only: bool = False,
) -> str:
    """JSON lines, one :func:`evidence_to_dict` record per attempt."""
    lines = [
        json.dumps(evidence_to_dict(record), sort_keys=True, default=str)
        for record in iter_attempt_evidence(
            db_path,
            run_id=run_id,
            citation=citation,
            limit=limit,
            failed_only=failed_only,
        )
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def issues_for(records: Sequence[AttemptEvidence]) -> list[ValidationIssue]:
    """Flatten the issues across records (convenience for verifier sweeps)."""
    issues: list[ValidationIssue] = []
    for record in records:
        issues.extend(record.issues)
    return issues


def _parse_json(text: object, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def load_issues_from_iterations_json(text: object) -> list[ValidationIssue]:
    """Structured issues from a raw ``iterations_json`` value (all attempts)."""
    issues: list[ValidationIssue] = []
    for iteration in _parse_json(text, []) or []:
        if not isinstance(iteration, dict):
            continue
        for error in iteration.get("errors") or []:
            if isinstance(error, dict):
                issues.extend(issues_from_dicts(error.get("issues")))
    return issues

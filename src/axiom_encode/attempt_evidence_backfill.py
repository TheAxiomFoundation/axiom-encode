"""Backfill and live mirroring of attempt evidence into ``encodings.db``.

Three writers, all idempotent:

* :func:`mirror_judge_event` copies one canonical ``judge`` run-log event into
  ``judge_events`` at emission time (the call site is
  :meth:`axiom_encode.judges.run_log.JudgeEvent.emit`). It never raises and
  never creates a database as a side effect: it writes only to an explicit
  path, to ``AXIOM_ENCODE_DB``, or to the default database when that file
  already exists. Under pytest it stays off unless ``AXIOM_ENCODE_DB`` is set.
* :func:`backfill_judge_events` ingests every ``judge`` event from the run-log
  JSONL files it can find (``--log-dir`` arguments, ``AXIOM_ENCODE_RUN_LOG_DIR``,
  ``./.axiom/run-logs``), keyed by ``event_id`` so re-runs insert nothing.
* :func:`backfill_attempt_evidence` runs the schema migration, links
  ``parent_run_id`` for regenerations recorded before links were written,
  records each historical run's final RuleSpec text as an ``attempt-<n>``
  artifact version, and ingests judge events. It reports row counts before
  and after so the effect on a database is auditable.

What a backfill cannot recover is stated in the report: the structured issue
lists of historical attempts (their repair manifests lived in temporary
directories) and the text of retried attempts that were overwritten before
artifact persistence existed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence
from urllib.parse import quote

from .harness.encoding_db import (
    ARTIFACT_TYPE_RULESPEC,
    EncodingDB,
    EncodingRun,
    run_from_row,
)
from .harness.encoding_db import RUN_COLUMNS as _RUN_COLUMNS

DB_ENV_VAR = "AXIOM_ENCODE_DB"
RUN_LOG_DIR_ENV_VAR = "AXIOM_ENCODE_RUN_LOG_DIR"

#: Same default as ``axiom_encode.cli.DEFAULT_DB``; duplicated here so the
#: judge hook does not import the CLI module.
DEFAULT_ENCODINGS_DB = (
    Path.home() / "TheAxiomFoundation" / "axiom-encode" / "encodings.db"
)

EVIDENCE_TABLES = (
    "encoding_runs",
    "sessions",
    "session_events",
    "artifact_versions",
    "run_artifacts",
    "judge_events",
)


# ---------------------------------------------------------------------------
# Live judge mirror
# ---------------------------------------------------------------------------


def _running_under_tests() -> bool:
    import sys

    return "pytest" in sys.modules or bool(os.environ.get("PYTEST_CURRENT_TEST"))


def resolve_judge_mirror_db(explicit: Optional[Path | str] = None) -> Optional[Path]:
    """The database a live judge event is mirrored into, or ``None`` to skip."""
    if explicit:
        return Path(explicit)
    env = os.environ.get(DB_ENV_VAR, "").strip()
    if env:
        return Path(env)
    if _running_under_tests():
        return None
    if DEFAULT_ENCODINGS_DB.exists():
        return DEFAULT_ENCODINGS_DB
    return None


def mirror_judge_event(
    event: Any,
    *,
    db_path: Optional[Path | str] = None,
    source: str = "live",
    subject_ref: Optional[str] = None,
    judge_findings: Optional[Sequence[dict[str, Any]]] = None,
) -> bool:
    """Persist one judge event; returns ``True`` when a row was inserted.

    Accepts a :class:`~axiom_encode.run_log.RunLogEvent` or its dict form.
    ``subject_ref`` and ``judge_findings`` (entries with ``clause_ref``,
    ``rule_path``, ``kind``, ``explanation``) come from the emitting judge,
    which the canonical event does not carry separately. Logging must never
    break a judge run, so every failure is swallowed.
    """
    try:
        target = resolve_judge_mirror_db(db_path)
        if target is None:
            return False
        payload = event.to_dict() if hasattr(event, "to_dict") else event
        if not isinstance(payload, dict):
            return False
        return (
            EncodingDB(target).log_judge_event(
                payload,
                source=source,
                subject_ref=subject_ref,
                judge_findings=list(judge_findings)
                if judge_findings is not None
                else None,
            )
            is not None
        )
    except Exception:  # noqa: BLE001 - mirroring is best-effort by contract
        return False


# ---------------------------------------------------------------------------
# Run-log discovery and judge event ingestion
# ---------------------------------------------------------------------------


def discover_run_log_dirs(explicit: Sequence[Path | str] = ()) -> list[Path]:
    """Run-log directories to scan: explicit ones, else the known defaults.

    No whole-tree search: only ``AXIOM_ENCODE_RUN_LOG_DIR``, ``./.axiom/run-logs``
    and the axiom-encode checkout's own ``.axiom/run-logs`` are consulted.
    """
    if explicit:
        return [Path(item) for item in explicit]
    candidates: list[Path] = []
    env = os.environ.get(RUN_LOG_DIR_ENV_VAR, "").strip()
    if env:
        candidates.append(Path(env))
    candidates.append(Path.cwd() / ".axiom" / "run-logs")
    candidates.append(Path(__file__).resolve().parents[2] / ".axiom" / "run-logs")
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(candidate)
    return unique


def iter_run_log_files(log_dirs: Iterable[Path]) -> Iterable[Path]:
    for log_dir in log_dirs:
        directory = Path(log_dir)
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            if path.is_file():
                yield path


def backfill_judge_events(
    db: EncodingDB,
    log_dirs: Sequence[Path | str] = (),
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ingest ``judge`` events from run-log JSONL files into ``judge_events``."""
    dirs = discover_run_log_dirs(log_dirs)
    report: dict[str, Any] = {
        "log_dirs": [str(item) for item in dirs],
        "files_scanned": 0,
        "events_seen": 0,
        "judge_events_seen": 0,
        "inserted": 0,
        "invalid_lines": 0,
        "files": [],
    }
    for path in iter_run_log_files(dirs):
        report["files_scanned"] += 1
        inserted_here = 0
        judge_here = 0
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        report["invalid_lines"] += 1
                        continue
                    if not isinstance(event, dict):
                        report["invalid_lines"] += 1
                        continue
                    report["events_seen"] += 1
                    if event.get("stage") != "judge":
                        continue
                    judge_here += 1
                    if dry_run:
                        continue
                    if db.log_judge_event(event, source=f"backfill:{path}"):
                        inserted_here += 1
        except OSError:
            report["invalid_lines"] += 1
            continue
        report["judge_events_seen"] += judge_here
        report["inserted"] += inserted_here
        if judge_here:
            report["files"].append(
                {
                    "path": str(path),
                    "judge_events": judge_here,
                    "inserted": inserted_here,
                }
            )
    return report


# ---------------------------------------------------------------------------
# Artifact versions and parent links for historical runs
# ---------------------------------------------------------------------------


def _iter_runs(
    conn: sqlite3.Connection, where: str, params: Sequence[Any] = ()
) -> Iterable[EncodingRun]:
    cursor = conn.execute(
        f"SELECT {', '.join(_RUN_COLUMNS)} FROM encoding_runs r WHERE {where} "
        "ORDER BY r.timestamp ASC, r.id",
        tuple(params),
    )
    for row in cursor:
        try:
            yield run_from_row(row, strict=False)
        except (ValueError, KeyError, TypeError, AttributeError):
            continue


def _run_generation_started_at(run: EncodingRun) -> Optional[datetime]:
    """When generation began: the row time minus the recorded generation time.

    ``None`` when no duration was recorded (legacy rows reconstructed from
    manifests, batch-logged eval results): without a start time a run cannot
    be sequenced against earlier runs, so it is never auto-linked.
    """
    duration_ms = int(run.total_duration_ms or 0)
    if duration_ms <= 0:
        duration_ms = sum(int(it.duration_ms or 0) for it in run.iterations)
    if duration_ms <= 0:
        return None
    return run.timestamp - timedelta(milliseconds=duration_ms)


def backfill_parent_links(
    db: EncodingDB, *, dry_run: bool = False, limit: Optional[int] = None
) -> dict[str, Any]:
    """Fill ``parent_run_id`` for runs that predate parent linking.

    Only rows whose ``parent_run_id`` is NULL are touched, and only when a
    prior run of the same citation had finished before this run's generation
    started (see :meth:`EncodingDB.find_parent_run`). Runs without a recorded
    generation duration are skipped (their start is unknown). Runs are
    processed oldest first so ``iteration`` counts chain correctly.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "linked": 0,
        "skipped_unknown_start": 0,
        "dry_run": dry_run,
    }
    conn = sqlite3.connect(db.db_path)
    try:
        runs = list(_iter_runs(conn, "r.parent_run_id IS NULL"))
    finally:
        conn.close()
    if limit is not None:
        runs = runs[: int(limit)]
    for run in runs:
        report["candidates"] += 1
        started_at = _run_generation_started_at(run)
        if started_at is None:
            report["skipped_unknown_start"] += 1
            continue
        parent = db.find_parent_run(
            citation=run.citation,
            started_at=started_at,
            agent_type=run.agent_type,
            agent_model=run.agent_model,
            exclude_run_id=run.id,
        )
        if parent is None:
            continue
        report["linked"] += 1
        if not dry_run:
            db.update_run_parent(run.id, parent.id, int(parent.iteration or 1) + 1)
    return report


def _final_attempt_number(run: EncodingRun) -> int:
    if run.iterations:
        return max(int(it.attempt) for it in run.iterations)
    return max(int(run.generation_attempt_count or 0), 1)


def _final_attempt_metadata(run: EncodingRun) -> dict[str, Any]:
    outcome = run.outcome if isinstance(run.outcome, dict) else {}
    last = run.iterations[-1] if run.iterations else None
    error = None
    issue_count = 0
    if last is not None:
        for err in last.errors:
            if error is None and err.message:
                error = err.message
            issue_count += len(err.issues)
    return {
        "run_id": run.id,
        "citation": run.citation,
        "attempt": _final_attempt_number(run),
        "success": bool(run.success),
        "final_success": outcome.get("final_success"),
        "status": outcome.get("status"),
        "model": (last.model if last is not None and last.model else None)
        or run.agent_model
        or None,
        "backend": run.agent_type,
        "output_file": run.file_path,
        "error": error,
        "issue_count": issue_count,
        "axiom_encode_version": run.axiom_encode_version,
        "backfill": True,
    }


def backfill_artifact_versions(
    db: EncodingDB, *, dry_run: bool = False, limit: Optional[int] = None
) -> dict[str, Any]:
    """Record each historical run's final RuleSpec text as an artifact version.

    Only the final attempt's text survives in ``encoding_runs.rulespec_content``;
    retried attempts before artifact persistence are unrecoverable and are
    counted under ``attempts_unrecoverable``.
    """
    report: dict[str, Any] = {
        "candidates": 0,
        "recorded": 0,
        "attempts_unrecoverable": 0,
        "dry_run": dry_run,
    }
    conn = sqlite3.connect(db.db_path)
    try:
        runs = list(
            _iter_runs(
                conn,
                "COALESCE(r.rulespec_content, '') != '' AND NOT EXISTS ("
                "SELECT 1 FROM run_artifacts ra WHERE ra.run_id = r.id "
                "AND ra.role = 'rulespec')",
            )
        )
    finally:
        conn.close()
    if limit is not None:
        runs = runs[: int(limit)]
    for run in runs:
        report["candidates"] += 1
        attempt = _final_attempt_number(run)
        report["attempts_unrecoverable"] += max(attempt - 1, 0)
        if dry_run:
            continue
        db.record_run_artifact(
            run.id,
            attempt=attempt,
            role=ARTIFACT_TYPE_RULESPEC,
            content=run.rulespec_content,
            metadata=_final_attempt_metadata(run),
            effective_from=run.timestamp.isoformat(),
        )
        report["recorded"] += 1
    return report


# ---------------------------------------------------------------------------
# Whole backfill with before/after counts
# ---------------------------------------------------------------------------


def evidence_counts(db_path: Path | str) -> dict[str, Any]:
    """Row counts of every evidence table plus the parent-link count."""
    path = Path(db_path)
    counts: dict[str, Any] = {table: 0 for table in EVIDENCE_TABLES}
    counts.update(
        {
            "runs_with_parent": 0,
            "runs_with_structured_issues": 0,
            "encode_issue_events_with_issues": 0,
        }
    )
    if not path.exists():
        return counts
    conn = sqlite3.connect(
        "file:" + quote(path.resolve().as_posix(), safe="/") + "?mode=ro", uri=True
    )
    try:
        existing = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for table in EVIDENCE_TABLES:
            counts[table] = (
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if table in existing
                else 0
            )
        if "encoding_runs" in existing:
            columns = {
                row[1] for row in conn.execute("PRAGMA table_info(encoding_runs)")
            }
            counts["runs_with_parent"] = (
                conn.execute(
                    "SELECT COUNT(*) FROM encoding_runs WHERE parent_run_id IS NOT NULL"
                ).fetchone()[0]
                if "parent_run_id" in columns
                else 0
            )
            counts["runs_with_structured_issues"] = (
                conn.execute(
                    "SELECT COUNT(*) FROM encoding_runs "
                    "WHERE iterations_json LIKE '%\"issues\": [{%'"
                ).fetchone()[0]
                if "iterations_json" in columns
                else 0
            )
        if "session_events" in existing:
            counts["encode_issue_events_with_issues"] = conn.execute(
                "SELECT COUNT(*) FROM session_events WHERE event_type = 'encode_issue' "
                "AND metadata_json LIKE '%\"issues\": [{%'"
            ).fetchone()[0]
    finally:
        conn.close()
    return counts


def backfill_attempt_evidence(
    db_path: Path | str,
    *,
    log_dirs: Sequence[Path | str] = (),
    link_parents: bool = True,
    artifacts: bool = True,
    judge_events: bool = True,
    dry_run: bool = False,
    limit: Optional[int] = None,
) -> dict[str, Any]:
    """Migrate ``db_path`` and backfill what the historical rows still allow."""
    path = Path(db_path)
    before = evidence_counts(path)
    db = EncodingDB(path)  # runs the idempotent schema migration
    report: dict[str, Any] = {
        "db": str(path),
        "dry_run": dry_run,
        "before": before,
    }
    if link_parents:
        report["parent_links"] = backfill_parent_links(db, dry_run=dry_run, limit=limit)
    if artifacts:
        report["artifact_versions"] = backfill_artifact_versions(
            db, dry_run=dry_run, limit=limit
        )
    if judge_events:
        report["judge_events"] = backfill_judge_events(db, log_dirs, dry_run=dry_run)
    report["after"] = evidence_counts(path)
    report["unrecoverable"] = {
        "historical_issue_lists": (
            "encode_issue events recorded before this schema carry only the "
            "one-line error; their repair manifests were written to temporary "
            "directories and are gone"
        ),
        "retried_attempt_text": (
            "only the final attempt's RuleSpec text was stored before artifact "
            "persistence; earlier attempts of those runs cannot be reconstructed"
        ),
    }
    return report

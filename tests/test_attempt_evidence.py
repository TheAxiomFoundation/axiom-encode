"""Tests for per-attempt encode evidence.

Covers the read-only view (:mod:`axiom_encode.attempt_evidence`), the backfill
and live judge mirror (:mod:`axiom_encode.attempt_evidence_backfill`), the
``JudgeEvent.emit`` hook, and the two ``attempt-evidence`` CLI subcommands.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime
from types import SimpleNamespace

import pytest

from axiom_encode import (
    AttemptEvidence,
    EncodingDB,
    EncodingRun,
    Iteration,
    IterationError,
    RepairTriple,
    ValidationIssue,
    iter_attempt_evidence,
    iter_repair_triples,
)
from axiom_encode.attempt_evidence import (
    AttemptArtifact,
    AttemptRef,
    issues_for,
    load_issues_from_iterations_json,
    open_readonly,
)
from axiom_encode.attempt_evidence_backfill import (
    backfill_attempt_evidence,
    backfill_judge_events,
    evidence_counts,
    mirror_judge_event,
    resolve_judge_mirror_db,
)
from axiom_encode.harness.encoding_db import (
    ARTIFACT_TYPE_RULESPEC,
    ARTIFACT_TYPE_RULESPEC_TESTS,
    RUN_COLUMNS,
    run_from_row,
)
from axiom_encode.judges.run_log import JudgeEvent, JudgeStage, Verdict
from axiom_encode.run_log import RunLogWriter

CITATION = "26 USC 32"
OTHER_CITATION = "26 USC 36B"

BAD_RULESPEC = "# attempt 1\nearned_income_credit:\n  amount: 600000\n"
GOOD_RULESPEC = "# attempt 2\nearned_income_credit:\n  amount: 6000\n"
TESTS_CONTENT = "cases:\n  - name: baseline\n    output: 6000\n"

BASE_TIME = datetime(2026, 1, 1, 10, 0, 0)

ISSUE_UNGROUNDED = ValidationIssue(
    gate="ci",
    kind="ungrounded_literal",
    message=("Ungrounded generated numeric literal: 600000 does not appear in source"),
    value="600000",
)
ISSUE_FIXTURE = ValidationIssue(
    gate="ci",
    kind="fixture_execution",
    message="Test case `baseline` failed",
    locator="test:baseline",
)
ATTEMPT_ISSUES = (ISSUE_UNGROUNDED, ISSUE_FIXTURE)

FAILURE_MESSAGE = "Generated RuleSpec failed CI validation"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _failed_iteration(attempt, issues=ATTEMPT_ISSUES, *, message=FAILURE_MESSAGE):
    return Iteration(
        attempt=attempt,
        duration_ms=1000,
        errors=[
            IterationError(
                error_type="validation",
                message=message,
                issues=list(issues),
            )
        ],
        success=False,
        model="test-model",
    )


def _passed_iteration(attempt):
    return Iteration(
        attempt=attempt,
        duration_ms=1200,
        errors=[],
        success=True,
        model="test-model",
    )


def _log_run(
    db,
    *,
    run_id,
    citation=CITATION,
    iterations=(),
    rulespec_content="",
    timestamp=None,
    total_duration_ms=2000,
    parent_run_id=None,
    outcome=None,
    session_id=None,
    generation_attempt_count=0,
    iteration=1,
    agent_model="test-model",
    agent_type="encoder",
):
    """Log a run whose shape the attempt-evidence view has to read back."""
    run = EncodingRun(
        id=run_id,
        timestamp=timestamp or BASE_TIME,
        citation=citation,
        file_path=f"rules/{run_id}.yaml",
        iterations=list(iterations),
        total_duration_ms=total_duration_ms,
        rulespec_content=rulespec_content,
        agent_type=agent_type,
        agent_model=agent_model,
        outcome=dict(outcome or {}),
        parent_run_id=parent_run_id,
        iteration=iteration,
        session_id=session_id,
        generation_attempt_count=generation_attempt_count,
    )
    db.log_run(run)
    return run


@pytest.fixture
def two_attempt_db(tmp_path):
    """One run: attempt 1 failed with issues and an artifact, attempt 2 passed."""
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-two",
        iterations=[_failed_iteration(1), _passed_iteration(2)],
        rulespec_content=GOOD_RULESPEC,
        outcome={"final_success": True},
    )
    db.record_run_artifact(
        "run-two",
        attempt=1,
        role=ARTIFACT_TYPE_RULESPEC,
        content=BAD_RULESPEC,
        metadata={"attempt": 1, "issue_count": 2},
    )
    db.record_run_artifact(
        "run-two",
        attempt=2,
        role=ARTIFACT_TYPE_RULESPEC,
        content=GOOD_RULESPEC,
        metadata={"attempt": 2, "issue_count": 0},
    )
    db.record_run_artifact(
        "run-two",
        attempt=2,
        role=ARTIFACT_TYPE_RULESPEC_TESTS,
        content=TESTS_CONTENT,
    )
    return path


# ---------------------------------------------------------------------------
# iter_attempt_evidence
# ---------------------------------------------------------------------------


def test_two_attempt_run_yields_records_in_attempt_order(two_attempt_db):
    records = list(iter_attempt_evidence(two_attempt_db))

    assert [record.attempt for record in records] == [1, 2]
    assert all(isinstance(record, AttemptEvidence) for record in records)
    assert [record.success for record in records] == [False, True]
    assert [record.citation for record in records] == [CITATION, CITATION]


def test_as_tuple_is_run_attempt_artifact_issues_parent(two_attempt_db):
    first, second = list(iter_attempt_evidence(two_attempt_db))

    assert first.as_tuple() == (
        first.run_id,
        first.attempt,
        first.artifact,
        first.issues,
        first.parent,
    )
    run_id, attempt, artifact, issues, parent = first.as_tuple()
    assert run_id == "run-two"
    assert attempt == 1
    assert isinstance(artifact, AttemptArtifact)
    assert artifact.content == BAD_RULESPEC
    assert artifact.metadata["issue_count"] == 2
    assert issues == ATTEMPT_ISSUES
    assert parent is None

    assert second.as_tuple()[3] == ()
    assert second.artifact.content == GOOD_RULESPEC
    assert second.artifact.tests is not None
    assert second.artifact.tests.content == TESTS_CONTENT


def test_second_attempt_parent_is_first_attempt_with_bad_artifact(two_attempt_db):
    _, second = list(iter_attempt_evidence(two_attempt_db))

    parent = second.parent
    assert isinstance(parent, AttemptRef)
    assert parent.run_id == "run-two"
    assert parent.attempt == 1
    assert parent.same_run is True
    assert parent.success is False
    assert parent.artifact.content == BAD_RULESPEC


def test_issue_fields_round_trip_through_iterations_json(two_attempt_db):
    first, _ = list(iter_attempt_evidence(two_attempt_db))

    assert first.error == FAILURE_MESSAGE
    assert first.issues[0].value == "600000"
    assert first.issues[1].locator == "test:baseline"
    assert first.issues_truncated == 0


def test_child_run_first_attempt_parents_to_last_attempt_of_parent_run(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-parent",
        iterations=[_failed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    db.record_run_artifact(
        "run-parent",
        attempt=1,
        role=ARTIFACT_TYPE_RULESPEC,
        content=BAD_RULESPEC,
    )
    _log_run(
        db,
        run_id="run-child",
        iterations=[_passed_iteration(1)],
        rulespec_content=GOOD_RULESPEC,
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
        parent_run_id="run-parent",
        iteration=2,
        outcome={"final_success": True},
    )
    db.record_run_artifact(
        "run-child",
        attempt=1,
        role=ARTIFACT_TYPE_RULESPEC,
        content=GOOD_RULESPEC,
    )

    child = next(iter_attempt_evidence(path, run_id="run-child"))

    assert child.parent_run_id == "run-parent"
    assert child.parent.run_id == "run-parent"
    assert child.parent.attempt == 1
    assert child.parent.same_run is False
    assert child.parent.success is False
    assert child.parent.artifact.content == BAD_RULESPEC


def test_failed_only_filters_passing_attempts(two_attempt_db):
    records = list(iter_attempt_evidence(two_attempt_db, failed_only=True))

    assert [record.attempt for record in records] == [1]
    assert records[0].success is False


def test_citation_and_run_id_filters(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-eitc",
        citation=CITATION,
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    _log_run(
        db,
        run_id="run-ptc",
        citation=OTHER_CITATION,
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
    )

    by_citation = list(iter_attempt_evidence(path, citation=OTHER_CITATION))
    assert [record.run_id for record in by_citation] == ["run-ptc"]

    by_run = list(iter_attempt_evidence(path, run_id="run-eitc"))
    assert [record.run_id for record in by_run] == ["run-eitc"]


def test_limit_bounds_runs_not_attempts(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    for index in range(3):
        run_id = f"run-{index}"
        _log_run(
            db,
            run_id=run_id,
            iterations=[_failed_iteration(1), _passed_iteration(2)],
            timestamp=datetime(2026, 1, 1, 10 + index, 0, 0),
            outcome={"final_success": True},
        )

    records = list(iter_attempt_evidence(path, limit=1))

    # One run, both of its attempts; newest run first.
    assert {record.run_id for record in records} == {"run-2"}
    assert [record.attempt for record in records] == [1, 2]

    assert len(list(iter_attempt_evidence(path, limit=2))) == 4
    assert len(list(iter_attempt_evidence(path))) == 6


def test_newest_first_false_yields_oldest_run_first(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    for index in range(2):
        _log_run(
            db,
            run_id=f"run-{index}",
            iterations=[_passed_iteration(1)],
            timestamp=datetime(2026, 1, 1, 10 + index, 0, 0),
        )

    oldest_first = list(iter_attempt_evidence(path, newest_first=False))

    assert [record.run_id for record in oldest_first] == ["run-0", "run-1"]


def test_run_without_artifact_rows_falls_back_to_legacy_artifact(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-legacy-content",
        iterations=[_failed_iteration(1), _passed_iteration(2)],
        rulespec_content=GOOD_RULESPEC,
        outcome={"final_success": True},
    )

    first, second = list(iter_attempt_evidence(path))

    assert first.artifact is None
    assert second.artifact is not None
    assert second.artifact.rulespec.id == "legacy:run-legacy-content"
    assert second.artifact.content == GOOD_RULESPEC
    assert second.artifact.metadata["legacy"] is True
    assert second.artifact.rulespec.version_label == "attempt-2"


def test_later_version_of_an_attempt_supersedes_the_earlier_one(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(db, run_id="run-repaired", iterations=[_passed_iteration(1)])
    db.record_run_artifact(
        "run-repaired",
        attempt=1,
        role=ARTIFACT_TYPE_RULESPEC,
        content=BAD_RULESPEC,
        effective_from="2026-01-01T10:00:00",
    )
    repaired = db.record_run_artifact(
        "run-repaired",
        attempt=1,
        role=ARTIFACT_TYPE_RULESPEC,
        content=GOOD_RULESPEC,
        effective_from="2026-01-01T10:05:00",
    )

    record = next(iter_attempt_evidence(path))

    assert record.artifact.rulespec.id == repaired.id
    assert record.artifact.content == GOOD_RULESPEC
    assert len(db.get_run_artifacts("run-repaired")) == 2


def test_recording_the_same_attempt_text_twice_is_idempotent(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(db, run_id="run-idem", iterations=[_passed_iteration(1)])

    first = db.record_run_artifact(
        "run-idem", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=GOOD_RULESPEC
    )
    again = db.record_run_artifact(
        "run-idem", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=GOOD_RULESPEC
    )

    assert again.id == first.id
    assert len(db.get_run_artifacts("run-idem")) == 1
    assert db.get_artifact_version(first.id).content == GOOD_RULESPEC


def test_issue_helpers_read_raw_iteration_payloads(two_attempt_db):
    records = list(iter_attempt_evidence(two_attempt_db))
    assert tuple(issues_for(records)) == ATTEMPT_ISSUES

    conn = sqlite3.connect(two_attempt_db)
    try:
        raw = conn.execute("SELECT iterations_json FROM encoding_runs").fetchone()[0]
    finally:
        conn.close()

    assert load_issues_from_iterations_json(raw) == list(ATTEMPT_ISSUES)
    assert load_issues_from_iterations_json(None) == []
    assert load_issues_from_iterations_json("not json at all") == []


def test_reading_never_modifies_the_database_file(two_attempt_db):
    before = hashlib.sha256(two_attempt_db.read_bytes()).hexdigest()

    records = list(iter_attempt_evidence(two_attempt_db))
    triples = list(iter_repair_triples(two_attempt_db))

    assert records and triples
    assert hashlib.sha256(two_attempt_db.read_bytes()).hexdigest() == before


def test_missing_database_raises_operational_error(tmp_path):
    missing = tmp_path / "absent.db"

    with pytest.raises(sqlite3.OperationalError):
        list(iter_attempt_evidence(missing))

    with pytest.raises(sqlite3.OperationalError):
        list(iter_repair_triples(missing))

    with pytest.raises(sqlite3.OperationalError):
        open_readonly(missing)


# ---------------------------------------------------------------------------
# Legacy-shaped databases
# ---------------------------------------------------------------------------

_INT_RUN_COLUMNS = frozenset(
    {
        "total_duration_ms",
        "iteration",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_output_tokens",
        "generation_attempt_count",
    }
)
_REAL_RUN_COLUMNS = frozenset({"estimated_cost_usd", "actual_cost_usd"})


def _legacy_encoding_runs_ddl():
    columns = []
    for name in RUN_COLUMNS:
        if name == "id":
            columns.append("id TEXT PRIMARY KEY")
        elif name in _INT_RUN_COLUMNS:
            columns.append(f"{name} INTEGER")
        elif name in _REAL_RUN_COLUMNS:
            columns.append(f"{name} REAL")
        else:
            columns.append(f"{name} TEXT")
    return f"CREATE TABLE encoding_runs ({', '.join(columns)})"


def _iterations_json(*entries):
    return json.dumps(
        [
            {
                "attempt": attempt,
                "duration_ms": 1000,
                "success": success,
                "errors": []
                if success
                else [
                    {
                        "error_type": "validation",
                        "message": FAILURE_MESSAGE,
                        "variable": None,
                        "fix_applied": None,
                        "issues": [issue.to_dict() for issue in issues],
                    }
                ],
            }
            for attempt, success, issues in entries
        ]
    )


def _make_legacy_db(path, *, review_results_json=None):
    """A database that predates ``run_artifacts.attempt``/``role`` and judge events."""
    conn = sqlite3.connect(path)
    conn.execute(_legacy_encoding_runs_ddl())
    conn.execute(
        "CREATE TABLE artifact_versions ("
        "id TEXT PRIMARY KEY, artifact_type TEXT NOT NULL, "
        "content_hash TEXT NOT NULL, version_label TEXT, content TEXT, "
        "effective_from TEXT NOT NULL, effective_to TEXT, metadata_json TEXT)"
    )
    # The 2025 SCD2 link table: no attempt and no role.
    conn.execute(
        "CREATE TABLE run_artifacts ("
        "run_id TEXT NOT NULL, artifact_version_id TEXT NOT NULL, "
        "PRIMARY KEY (run_id, artifact_version_id))"
    )
    values = dict.fromkeys(RUN_COLUMNS)
    values.update(
        id="legacy-run",
        timestamp=BASE_TIME.isoformat(),
        citation=CITATION,
        file_path="rules/legacy.yaml",
        iterations_json=_iterations_json(
            (1, False, ATTEMPT_ISSUES),
            (2, True, ()),
        ),
        total_duration_ms=2000,
        agent_type="encoder",
        agent_model="test-model",
        rulespec_content=GOOD_RULESPEC,
        iteration=1,
        review_results_json=review_results_json,
        outcome_json=json.dumps({"final_success": True}),
    )
    conn.execute(
        f"INSERT INTO encoding_runs ({', '.join(RUN_COLUMNS)}) "
        f"VALUES ({', '.join('?' for _ in RUN_COLUMNS)})",
        tuple(values[name] for name in RUN_COLUMNS),
    )
    conn.execute(
        "INSERT INTO artifact_versions (id, artifact_type, content_hash, "
        "version_label, content, effective_from, effective_to, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "scd2-version",
            ARTIFACT_TYPE_RULESPEC,
            "0" * 64,
            "v1",
            BAD_RULESPEC,
            BASE_TIME.isoformat(),
            None,
            "{}",
        ),
    )
    conn.execute(
        "INSERT INTO run_artifacts (run_id, artifact_version_id) VALUES (?, ?)",
        ("legacy-run", "scd2-version"),
    )
    conn.commit()
    conn.close()
    return path


def test_reads_legacy_db_without_attempt_role_or_judge_events(tmp_path):
    path = _make_legacy_db(tmp_path / "legacy.db")

    records = list(iter_attempt_evidence(path))

    assert [record.attempt for record in records] == [1, 2]
    assert records[0].issues == ATTEMPT_ISSUES
    # The unlabeled SCD2 link carries no attempt, so it is not an attempt artifact.
    assert records[0].artifact is None
    assert records[1].artifact.rulespec.id == "legacy:legacy-run"
    assert records[1].artifact.content == GOOD_RULESPEC
    assert records[1].success is True

    # The read never migrated the schema.
    conn = sqlite3.connect(path)
    try:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        link_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(run_artifacts)")
        }
    finally:
        conn.close()
    assert "judge_events" not in tables
    assert link_columns == {"run_id", "artifact_version_id"}


def test_legacy_unsupported_review_schema_still_yields_records(tmp_path):
    path = _make_legacy_db(
        tmp_path / "legacy-review.db",
        review_results_json=json.dumps({"reviewers": [], "score": 4}),
    )

    records = list(iter_attempt_evidence(path))

    assert [record.attempt for record in records] == [1, 2]
    assert records[0].issues == ATTEMPT_ISSUES

    conn = sqlite3.connect(path)
    try:
        row = conn.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs"
        ).fetchone()
    finally:
        conn.close()
    with pytest.raises(ValueError):
        run_from_row(row, strict=True)
    # The lenient reader drops an unreadable review payload rather than
    # presenting it as a review by zero reviewers.
    assert run_from_row(row, strict=False).review_results is None


# ---------------------------------------------------------------------------
# iter_repair_triples
# ---------------------------------------------------------------------------


def test_within_run_repair_triple(two_attempt_db):
    triples = list(iter_repair_triples(two_attempt_db))

    assert len(triples) == 1
    triple = triples[0]
    assert isinstance(triple, RepairTriple)
    assert triple.cross_run is False
    assert triple.citation == CITATION
    assert triple.bad.attempt == 1
    assert triple.bad.artifact.content == BAD_RULESPEC
    assert triple.good.attempt == 2
    assert triple.good.artifact.content == GOOD_RULESPEC
    assert triple.issues == ATTEMPT_ISSUES
    assert triple.issues == triple.bad.issues


def test_cross_run_repair_triple_through_parent_run_id(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-bad",
        iterations=[_failed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    db.record_run_artifact(
        "run-bad", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=BAD_RULESPEC
    )
    _log_run(
        db,
        run_id="run-good",
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
        parent_run_id="run-bad",
        iteration=2,
        outcome={"final_success": True},
    )
    db.record_run_artifact(
        "run-good", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=GOOD_RULESPEC
    )

    triples = list(iter_repair_triples(path))

    assert len(triples) == 1
    triple = triples[0]
    assert triple.cross_run is True
    assert triple.bad.run_id == "run-bad"
    assert triple.good.run_id == "run-good"
    assert triple.issues == ATTEMPT_ISSUES


def test_no_triple_when_both_attempts_fail(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-both-failed",
        iterations=[_failed_iteration(1), _failed_iteration(2)],
        outcome={"final_success": False},
    )
    for attempt, content in ((1, BAD_RULESPEC), (2, BAD_RULESPEC + "# retry\n")):
        db.record_run_artifact(
            "run-both-failed",
            attempt=attempt,
            role=ARTIFACT_TYPE_RULESPEC,
            content=content,
        )

    assert list(iter_repair_triples(path)) == []


def test_require_artifacts_drops_triples_whose_bad_version_is_gone(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-no-bad-artifact",
        iterations=[_failed_iteration(1), _passed_iteration(2)],
        rulespec_content=GOOD_RULESPEC,
        outcome={"final_success": True},
    )
    db.record_run_artifact(
        "run-no-bad-artifact",
        attempt=2,
        role=ARTIFACT_TYPE_RULESPEC,
        content=GOOD_RULESPEC,
    )

    assert list(iter_repair_triples(path)) == []

    relaxed = list(iter_repair_triples(path, require_artifacts=False))
    assert len(relaxed) == 1
    assert relaxed[0].bad.artifact is None
    assert relaxed[0].good.artifact.content == GOOD_RULESPEC
    assert relaxed[0].issues == ATTEMPT_ISSUES


def test_repair_triples_citation_filter(two_attempt_db):
    assert list(iter_repair_triples(two_attempt_db, citation=OTHER_CITATION)) == []
    assert len(list(iter_repair_triples(two_attempt_db, citation=CITATION))) == 1


def test_triple_resolves_a_parent_run_outside_the_selected_set(tmp_path):
    """The parent run is fetched on demand when a filter excluded it."""
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    # The parent was recorded under a differently spelled citation, so the
    # citation filter selects only the child run.
    _log_run(
        db,
        run_id="run-bad",
        citation="26 U.S.C. 32",
        iterations=[_failed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
    )
    db.record_run_artifact(
        "run-bad", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=BAD_RULESPEC
    )
    _log_run(
        db,
        run_id="run-good",
        citation=CITATION,
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
        parent_run_id="run-bad",
        iteration=2,
        outcome={"final_success": True},
    )
    db.record_run_artifact(
        "run-good", attempt=1, role=ARTIFACT_TYPE_RULESPEC, content=GOOD_RULESPEC
    )

    triples = list(iter_repair_triples(path, citation=CITATION))

    assert len(triples) == 1
    assert triples[0].cross_run is True
    assert triples[0].bad.run_id == "run-bad"
    assert triples[0].bad.artifact.content == BAD_RULESPEC
    assert triples[0].issues == ATTEMPT_ISSUES


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------


def test_attempt_evidence_is_exported_from_the_package():
    import axiom_encode

    for name in (
        "iter_attempt_evidence",
        "iter_repair_triples",
        "AttemptEvidence",
        "AttemptRef",
        "AttemptArtifact",
        "RepairTriple",
        "ValidationIssue",
    ):
        assert name in axiom_encode.__all__
        assert getattr(axiom_encode, name) is not None

    assert axiom_encode.iter_attempt_evidence is iter_attempt_evidence
    assert axiom_encode.iter_repair_triples is iter_repair_triples
    assert axiom_encode.AttemptEvidence is AttemptEvidence
    assert axiom_encode.RepairTriple is RepairTriple
    assert axiom_encode.ValidationIssue is ValidationIssue


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def _judge_event_dict(run_id, *, seq, event_id, verdict):
    return {
        "schema": "axiom_encode.run_log.v1",
        "event_id": event_id,
        "run_id": run_id,
        "seq": seq,
        "ts": "2026-01-01T10:00:00+00:00",
        "stage": "judge",
        "status": "passed",
        "reason_code": None,
        "reason": None,
        "duration_ms": 12,
        "attrs": {
            "judge_stage": "statutory_fidelity",
            "verdict": verdict,
            "advisory": True,
            "escalated": False,
            "tokens": {"input": 10, "output": 5},
            "judge_model": "test-model",
        },
        "findings": [],
    }


def _write_run_log(log_dir, run_id="run-a"):
    """A JSONL with one generate event, two judge events and one corrupt line."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{run_id}.jsonl"
    lines = [
        json.dumps(
            {
                "schema": "axiom_encode.run_log.v1",
                "event_id": "evt-generate",
                "run_id": run_id,
                "seq": 0,
                "ts": "2026-01-01T09:59:00+00:00",
                "stage": "generate",
                "status": "passed",
                "attrs": {"model": "test-model"},
                "findings": [],
            }
        ),
        json.dumps(
            _judge_event_dict(run_id, seq=1, event_id="evt-judge-1", verdict="pass")
        ),
        "{not json at all",
        json.dumps(
            _judge_event_dict(run_id, seq=2, event_id="evt-judge-2", verdict="flag")
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_backfill_db(tmp_path, name="encodings.db"):
    """Two sequential runs of one citation, a concurrent sibling, and two edge rows."""
    path = tmp_path / name
    db = EncodingDB(path)
    # Sequential pair: run-a finished (session ended 10:00:30) before run-b started.
    _log_run(
        db,
        run_id="run-a",
        iterations=[_failed_iteration(1), _passed_iteration(2)],
        rulespec_content=GOOD_RULESPEC,
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
        total_duration_ms=600_000,
        session_id="session-a",
        outcome={"final_success": True},
    )
    _log_run(
        db,
        run_id="run-b",
        iterations=[_passed_iteration(1)],
        rulespec_content=GOOD_RULESPEC + "# b\n",
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
        total_duration_ms=600_000,
    )
    # Concurrent fan-out sibling: generation started 09:50, before run-a finished.
    _log_run(
        db,
        run_id="run-c",
        iterations=[_passed_iteration(1)],
        rulespec_content=GOOD_RULESPEC + "# c\n",
        timestamp=datetime(2026, 1, 1, 11, 5, 0),
        total_duration_ms=4_500_000,
    )
    # Unknown start: no recorded duration and no iterations to sum.
    _log_run(
        db,
        run_id="run-d",
        iterations=[],
        rulespec_content=GOOD_RULESPEC + "# d\n",
        timestamp=datetime(2026, 1, 1, 12, 0, 0),
        total_duration_ms=0,
    )
    # Already linked and no content: must not be touched at all.
    _log_run(
        db,
        run_id="run-e",
        citation=OTHER_CITATION,
        iterations=[_passed_iteration(1)],
        rulespec_content="",
        timestamp=datetime(2026, 1, 1, 13, 0, 0),
        total_duration_ms=5_000,
        parent_run_id="preset-parent",
        iteration=7,
    )
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "INSERT INTO sessions (id, run_id, started_at, ended_at) "
            "VALUES (?, ?, ?, ?)",
            ("session-a", "run-a", "2026-01-01T09:50:00", "2026-01-01T10:00:30"),
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _run_rows(path):
    conn = sqlite3.connect(path)
    try:
        return {
            row[0]: (row[1], row[2])
            for row in conn.execute(
                "SELECT id, parent_run_id, iteration FROM encoding_runs"
            )
        }
    finally:
        conn.close()


def test_backfill_links_only_the_sequential_regeneration(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    report = backfill_attempt_evidence(path, log_dirs=[log_dir])

    assert report["parent_links"] == {
        "candidates": 4,
        "linked": 1,
        "skipped_unknown_start": 1,
        "dry_run": False,
    }
    rows = _run_rows(path)
    assert rows["run-b"][0] == "run-a"
    assert rows["run-b"][1] == 2
    assert rows["run-a"][0] is None
    assert rows["run-c"][0] is None
    assert rows["run-d"][0] is None


def test_concurrent_sibling_is_unlinked_by_its_start_time_not_by_missing_candidates(
    tmp_path,
):
    path = _build_backfill_db(tmp_path)
    db = EncodingDB(path)

    # run-c's generation started at 09:50, before run-a had finished.
    assert (
        db.find_parent_run(
            citation=CITATION,
            started_at=datetime(2026, 1, 1, 9, 50, 0),
            agent_type="encoder",
            agent_model="test-model",
            exclude_run_id="run-c",
        )
        is None
    )
    # Had it started after run-b finished, run-b would have been its parent.
    later = db.find_parent_run(
        citation=CITATION,
        started_at=datetime(2026, 1, 1, 11, 4, 0),
        agent_type="encoder",
        agent_model="test-model",
        exclude_run_id="run-c",
    )
    assert later is not None
    assert later.id == "run-b"


def test_backfill_creates_a_missing_database_and_reports_zeros(tmp_path):
    path = tmp_path / "fresh.db"

    report = backfill_attempt_evidence(path, log_dirs=[tmp_path / "empty-logs"])

    assert path.exists()
    assert report["before"]["encoding_runs"] == 0
    assert report["before"]["runs_with_parent"] == 0
    assert report["after"]["encoding_runs"] == 0
    assert report["parent_links"]["candidates"] == 0
    assert report["artifact_versions"]["candidates"] == 0
    assert report["judge_events"]["files_scanned"] == 0
    assert list(iter_attempt_evidence(path)) == []


def test_backfill_leaves_existing_parent_links_untouched(tmp_path):
    path = _build_backfill_db(tmp_path)

    backfill_attempt_evidence(path, log_dirs=[tmp_path / "empty-logs"])

    assert _run_rows(path)["run-e"] == ("preset-parent", 7)


def test_backfill_records_one_artifact_version_per_run_with_content(tmp_path):
    path = _build_backfill_db(tmp_path)

    report = backfill_attempt_evidence(path, log_dirs=[tmp_path / "empty-logs"])

    assert report["artifact_versions"]["candidates"] == 4
    assert report["artifact_versions"]["recorded"] == 4
    # run-a retried once; only its final attempt's text survived.
    assert report["artifact_versions"]["attempts_unrecoverable"] == 1

    db = EncodingDB(path)
    linked = db.get_run_artifacts("run-a")
    assert len(linked) == 1
    link, version = linked[0]
    assert link.attempt == 2
    assert link.role == ARTIFACT_TYPE_RULESPEC
    assert version.content == GOOD_RULESPEC
    assert version.version_label == "attempt-2"
    assert version.metadata["backfill"] is True
    assert version.metadata["run_id"] == "run-a"
    assert version.metadata["citation"] == CITATION

    # run-d has no iterations, so its attempt number falls back to 1.
    (d_link, _) = db.get_run_artifacts("run-d")[0]
    assert d_link.attempt == 1
    # run-e carried no RuleSpec text, so nothing was invented for it.
    assert db.get_run_artifacts("run-e") == []


def test_backfill_ingests_judge_events_and_counts_invalid_lines(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    report = backfill_attempt_evidence(path, log_dirs=[log_dir])

    judge = report["judge_events"]
    assert judge["files_scanned"] == 1
    assert judge["events_seen"] == 3
    assert judge["judge_events_seen"] == 2
    assert judge["inserted"] == 2
    assert judge["invalid_lines"] == 1

    rows = EncodingDB(path).get_judge_events("run-a")
    assert [row.id for row in rows] == ["evt-judge-1", "evt-judge-2"]
    assert [row.verdict for row in rows] == ["pass", "flag"]
    assert rows[0].judge_stage == "statutory_fidelity"
    assert rows[0].source.startswith("backfill:")


def test_backfill_report_has_before_and_after_counts(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    report = backfill_attempt_evidence(path, log_dirs=[log_dir])

    assert report["db"] == str(path)
    assert report["dry_run"] is False
    assert report["before"]["encoding_runs"] == 5
    assert report["before"]["artifact_versions"] == 0
    assert report["before"]["run_artifacts"] == 0
    assert report["before"]["judge_events"] == 0
    assert report["before"]["runs_with_parent"] == 1
    assert report["after"]["artifact_versions"] == 4
    assert report["after"]["run_artifacts"] == 4
    assert report["after"]["judge_events"] == 2
    assert report["after"]["runs_with_parent"] == 2
    assert report["after"] == evidence_counts(path)
    assert set(report["unrecoverable"]) == {
        "historical_issue_lists",
        "retried_attempt_text",
    }


def test_backfill_is_idempotent(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    backfill_attempt_evidence(path, log_dirs=[log_dir])
    after_first = evidence_counts(path)
    rows_first = _run_rows(path)

    second = backfill_attempt_evidence(path, log_dirs=[log_dir])

    assert second["parent_links"]["linked"] == 0
    assert second["parent_links"]["candidates"] == 3
    assert second["artifact_versions"]["candidates"] == 0
    assert second["artifact_versions"]["recorded"] == 0
    assert second["judge_events"]["inserted"] == 0
    assert second["judge_events"]["judge_events_seen"] == 2
    assert second["after"] == after_first
    assert _run_rows(path) == rows_first


def test_backfill_falls_back_to_iteration_durations_for_the_start_time(tmp_path):
    path = tmp_path / "encodings.db"
    db = EncodingDB(path)
    _log_run(
        db,
        run_id="run-first",
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 10, 0, 0),
        total_duration_ms=600_000,
    )
    # No run-level duration, but the attempt recorded its own.
    _log_run(
        db,
        run_id="run-second",
        iterations=[_passed_iteration(1)],
        timestamp=datetime(2026, 1, 1, 11, 0, 0),
        total_duration_ms=0,
    )

    report = backfill_attempt_evidence(
        path, log_dirs=[tmp_path / "empty-logs"], artifacts=False
    )

    assert report["parent_links"]["skipped_unknown_start"] == 0
    assert report["parent_links"]["linked"] == 1
    assert _run_rows(path)["run-second"] == ("run-first", 2)


def test_backfill_dry_run_writes_no_rows(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    report = backfill_attempt_evidence(path, log_dirs=[log_dir], dry_run=True)

    assert report["dry_run"] is True
    assert report["parent_links"]["linked"] == 1
    assert report["parent_links"]["dry_run"] is True
    assert report["artifact_versions"]["candidates"] == 4
    assert report["artifact_versions"]["recorded"] == 0
    assert report["judge_events"]["judge_events_seen"] == 2
    assert report["judge_events"]["inserted"] == 0

    counts = evidence_counts(path)
    assert counts["artifact_versions"] == 0
    assert counts["run_artifacts"] == 0
    assert counts["judge_events"] == 0
    assert counts["runs_with_parent"] == 1
    assert _run_rows(path)["run-b"][0] is None


def test_backfill_switches_off_individual_stages(tmp_path):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    report = backfill_attempt_evidence(
        path,
        log_dirs=[log_dir],
        link_parents=False,
        artifacts=False,
        judge_events=False,
    )

    assert "parent_links" not in report
    assert "artifact_versions" not in report
    assert "judge_events" not in report
    assert report["after"] == report["before"]


def test_backfilled_artifacts_feed_the_attempt_evidence_view(tmp_path):
    path = _build_backfill_db(tmp_path)

    backfill_attempt_evidence(path, log_dirs=[tmp_path / "empty-logs"])

    records = list(iter_attempt_evidence(path, run_id="run-a"))
    assert [record.attempt for record in records] == [1, 2]
    assert records[0].artifact is None
    assert records[1].artifact.content == GOOD_RULESPEC
    assert records[1].artifact.metadata["backfill"] is True

    child = next(iter_attempt_evidence(path, run_id="run-b"))
    assert child.parent is not None
    assert child.parent.run_id == "run-a"
    assert child.parent.same_run is False


# ---------------------------------------------------------------------------
# Live judge mirror
# ---------------------------------------------------------------------------


def _judge_event(run_id="run-mirror"):
    return JudgeEvent(
        stage=JudgeStage.STATUTORY_FIDELITY,
        verdict=Verdict.FLAG,
        confidence=0.75,
        advisory=True,
        model="test-model",
        generator_model="test-model",
        run_id=run_id,
        subject_ref="rules/eitc.yaml",
    )


def test_emit_mirrors_judge_event_into_the_database(tmp_path, monkeypatch):
    db_path = tmp_path / "encodings.db"
    db = EncodingDB(db_path)
    monkeypatch.setenv("AXIOM_ENCODE_DB", str(db_path))

    writer = RunLogWriter("run-mirror", log_dir=tmp_path)
    written = _judge_event().emit(writer, duration_ms=42)

    assert written is not None
    rows = db.get_judge_events("run-mirror")
    assert len(rows) == 1
    assert rows[0].id == written.event_id
    assert rows[0].verdict == "flag"
    assert rows[0].judge_stage == "statutory_fidelity"
    assert rows[0].duration_ms == 42
    assert rows[0].source == "live"

    line = json.loads(writer.path.read_text(encoding="utf-8").strip())
    assert line["event_id"] == written.event_id
    assert line["stage"] == "judge"


def test_emit_writes_nothing_under_pytest_without_the_env_var(tmp_path, monkeypatch):
    db_path = tmp_path / "encodings.db"
    db = EncodingDB(db_path)
    monkeypatch.delenv("AXIOM_ENCODE_DB", raising=False)

    assert resolve_judge_mirror_db() is None

    writer = RunLogWriter("run-mirror", log_dir=tmp_path)
    written = _judge_event().emit(writer)

    assert written is not None
    assert db.get_judge_events("run-mirror") == []
    assert writer.path.exists()


def test_emit_honours_an_explicit_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_DB", raising=False)
    db_path = tmp_path / "explicit.db"

    writer = RunLogWriter("run-mirror", log_dir=tmp_path)
    written = _judge_event().emit(writer, db_path=db_path)

    assert written is not None
    rows = EncodingDB(db_path).get_judge_events("run-mirror")
    assert [row.id for row in rows] == [written.event_id]
    assert resolve_judge_mirror_db(db_path) == db_path


def test_mirror_never_raises_on_a_broken_db_path(tmp_path, monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_DB", raising=False)
    broken = tmp_path / "not-a-file"
    broken.mkdir()

    writer = RunLogWriter("run-mirror", log_dir=tmp_path)
    written = _judge_event().emit(writer, db_path=broken)

    assert written is not None
    assert mirror_judge_event(written, db_path=broken) is False
    assert broken.is_dir()


def test_mirror_accepts_dicts_and_rejects_non_events(tmp_path, monkeypatch):
    db_path = tmp_path / "encodings.db"
    db = EncodingDB(db_path)
    monkeypatch.setenv("AXIOM_ENCODE_DB", str(db_path))

    payload = _judge_event_dict("run-dict", seq=0, event_id="evt-dict", verdict="pass")
    assert mirror_judge_event(payload) is True
    assert mirror_judge_event(payload) is False  # same event_id
    assert mirror_judge_event("not an event") is False
    assert mirror_judge_event({"stage": "generate", "run_id": "run-dict"}) is False
    assert mirror_judge_event({"stage": "judge", "event_id": "no-run"}) is False

    assert [row.id for row in db.get_judge_events("run-dict")] == ["evt-dict"]


def test_backfilling_a_mirrored_run_log_inserts_nothing(tmp_path, monkeypatch):
    db_path = tmp_path / "encodings.db"
    db = EncodingDB(db_path)
    monkeypatch.setenv("AXIOM_ENCODE_DB", str(db_path))

    writer = RunLogWriter("run-mirror", log_dir=tmp_path)
    written = _judge_event().emit(writer)
    assert written is not None

    report = backfill_judge_events(db, [tmp_path])

    assert report["judge_events_seen"] == 1
    assert report["inserted"] == 0
    assert len(db.get_judge_events("run-mirror")) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _run_cli(monkeypatch, *argv):
    from axiom_encode.cli import main

    monkeypatch.setattr(sys, "argv", ["axiom-encode", *argv])
    try:
        main()
    except SystemExit as exit_error:
        return int(exit_error.code or 0)
    return 0


def test_cli_parses_attempt_evidence_backfill_flags(tmp_path, capsys, monkeypatch):
    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    exit_code = _run_cli(
        monkeypatch,
        "attempt-evidence-backfill",
        "--db",
        str(path),
        "--log-dir",
        str(log_dir),
        "--no-link-parents",
        "--dry-run",
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert "parent_links" not in report
    assert report["dry_run"] is True
    assert report["artifact_versions"]["candidates"] == 4
    assert report["artifact_versions"]["recorded"] == 0
    assert report["judge_events"]["judge_events_seen"] == 2
    assert report["judge_events"]["inserted"] == 0
    assert evidence_counts(path)["judge_events"] == 0
    assert _run_rows(path)["run-b"][0] is None


def test_cli_parses_attempt_evidence_flags(two_attempt_db, capsys, monkeypatch):
    exit_code = _run_cli(
        monkeypatch,
        "attempt-evidence",
        "--db",
        str(two_attempt_db),
        "--citation",
        CITATION,
        "--failed-only",
    )

    assert exit_code == 0
    records = [
        json.loads(line) for line in capsys.readouterr().out.strip().splitlines()
    ]
    assert [record["attempt"] for record in records] == [1]
    assert records[0]["artifact"]["content"] == BAD_RULESPEC


def test_cli_attempt_evidence_backfill_prints_the_report(tmp_path, capsys):
    from axiom_encode.cli import cmd_attempt_evidence_backfill

    path = _build_backfill_db(tmp_path)
    log_dir = tmp_path / "run-logs"
    _write_run_log(log_dir)

    cmd_attempt_evidence_backfill(
        SimpleNamespace(
            db=path,
            log_dir=[log_dir],
            link_parents=True,
            artifacts=True,
            judge_events=True,
            dry_run=False,
            limit=None,
        )
    )

    report = json.loads(capsys.readouterr().out)
    assert report["db"] == str(path)
    assert report["parent_links"]["linked"] == 1
    assert report["artifact_versions"]["recorded"] == 4
    assert report["judge_events"]["inserted"] == 2
    assert report["after"]["judge_events"] == 2


def test_cli_attempt_evidence_backfill_dry_run_reports_without_writing(
    tmp_path, capsys
):
    from axiom_encode.cli import cmd_attempt_evidence_backfill

    path = _build_backfill_db(tmp_path)

    cmd_attempt_evidence_backfill(
        SimpleNamespace(
            db=path,
            log_dir=[],
            link_parents=True,
            artifacts=True,
            judge_events=False,
            dry_run=True,
            limit=None,
        )
    )

    report = json.loads(capsys.readouterr().out)
    assert report["dry_run"] is True
    assert report["after"]["artifact_versions"] == 0
    assert _run_rows(path)["run-b"][0] is None


def test_cli_attempt_evidence_prints_json_lines(two_attempt_db, capsys):
    from axiom_encode.cli import cmd_attempt_evidence

    cmd_attempt_evidence(
        SimpleNamespace(
            db=two_attempt_db,
            run_id="run-two",
            citation=None,
            limit=None,
            failed_only=False,
        )
    )

    out = capsys.readouterr().out
    assert out.endswith("\n")
    records = [json.loads(line) for line in out.strip().splitlines()]
    assert [record["attempt"] for record in records] == [1, 2]
    assert records[0]["run_id"] == "run-two"
    assert records[0]["citation"] == CITATION
    assert records[0]["success"] is False
    assert records[0]["artifact"]["content"] == BAD_RULESPEC
    assert [issue["gate"] for issue in records[0]["issues"]] == ["ci", "ci"]
    assert records[0]["issues"][0]["value"] == "600000"
    assert records[0]["parent"] is None
    assert records[1]["parent"]["same_run"] is True
    assert records[1]["parent"]["attempt"] == 1
    assert records[1]["artifact"]["tests"]["content"] == TESTS_CONTENT


def test_cli_attempt_evidence_failed_only_and_citation(two_attempt_db, capsys):
    from axiom_encode.cli import cmd_attempt_evidence

    cmd_attempt_evidence(
        SimpleNamespace(
            db=two_attempt_db,
            run_id=None,
            citation=CITATION,
            limit=1,
            failed_only=True,
        )
    )

    records = [
        json.loads(line) for line in capsys.readouterr().out.strip().splitlines()
    ]
    assert [record["attempt"] for record in records] == [1]
    assert records[0]["success"] is False


def test_cli_attempt_evidence_prints_nothing_for_an_unknown_run(two_attempt_db, capsys):
    from axiom_encode.cli import cmd_attempt_evidence

    cmd_attempt_evidence(
        SimpleNamespace(
            db=two_attempt_db,
            run_id="no-such-run",
            citation=None,
            limit=None,
            failed_only=False,
        )
    )

    assert capsys.readouterr().out == ""

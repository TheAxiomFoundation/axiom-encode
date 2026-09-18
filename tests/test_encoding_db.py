"""
Tests for the experiment database.
"""

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from axiom_encode import (
    EncodingDB,
    Iteration,
    IterationError,
    ReviewResult,
    ReviewResults,
    ValidationIssue,
    create_run,
)
from axiom_encode.harness.encoding_db import (
    ARTIFACT_TYPE_RULESPEC,
    ARTIFACT_TYPE_RULESPEC_TESTS,
    RUN_COLUMNS,
    run_from_row,
)
from axiom_encode.judges.run_log import (
    Finding,
    JudgeEvent,
    JudgeStage,
    TokenCounts,
    Verdict,
    error_event,
)


class TestCreateRun:
    """Tests for the create_run factory function."""

    def test_create_run_generates_id(self):
        """Test that create_run generates a unique ID."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        assert run.id is not None
        assert len(run.id) == 8  # UUID[:8]

    def test_create_run_sets_timestamp(self):
        """Test that create_run sets current timestamp."""
        before = datetime.now()
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        after = datetime.now()
        assert before <= run.timestamp <= after

    def test_create_run_sets_iteration_1_for_new_run(self):
        """Test that new runs have iteration=1."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        assert run.iteration == 1
        assert run.parent_run_id is None

    def test_create_run_sets_iteration_2_for_revision(self):
        """Test that revisions have iteration=2."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            parent_run_id="abc12345",
        )
        assert run.iteration == 2
        assert run.parent_run_id == "abc12345"


class TestEncodingDBInit:
    """Tests for EncodingDB initialization."""

    def test_creates_database_file(self, temp_db_path):
        """Test that database file is created."""
        EncodingDB(temp_db_path)
        assert temp_db_path.exists()

    def test_creates_tables(self, experiment_db, temp_db_path):
        """Test that required tables are created."""
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check encoding_runs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
        )
        assert cursor.fetchone() is not None

        # Check calibration_snapshots table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_snapshots'"
        )
        assert cursor.fetchone() is not None

        conn.close()


class TestLogAndRetrieveRuns:
    """Tests for logging and retrieving encoding runs."""

    def test_log_run_and_retrieve(self, experiment_db, sample_encoding_run):
        """Test logging a run and retrieving it."""
        run_id = experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(run_id)

        assert retrieved is not None
        assert retrieved.id == sample_encoding_run.id
        assert retrieved.file_path == sample_encoding_run.file_path
        assert retrieved.citation == sample_encoding_run.citation
        assert retrieved.agent_type == sample_encoding_run.agent_type
        assert retrieved.agent_model == sample_encoding_run.agent_model
        assert retrieved.rulespec_content == sample_encoding_run.rulespec_content

    def test_log_run_with_review_results(self, experiment_db, sample_review_results):
        """Test logging a run with review results."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            review_results=sample_review_results,
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.review_results is not None
        assert len(retrieved.review_results.reviews) == 4
        assert retrieved.review_results.reviews[0].reviewer == "rulespec_reviewer"
        assert retrieved.review_results.reviews[0].passed is True
        assert retrieved.review_results.policyengine_match == 0.90

    def test_log_run_with_lessons(self, experiment_db):
        """Test logging a run with lessons."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            lessons="Learned that bracket syntax needs special handling.",
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert (
            retrieved.lessons == "Learned that bracket syntax needs special handling."
        )

    def test_log_run_with_final_outcome(self, experiment_db, sample_encoding_run):
        """Test final encode/apply outcomes are persisted and drive run success."""
        sample_encoding_run.iterations = [
            Iteration(attempt=1, duration_ms=1000, success=False)
        ]
        sample_encoding_run.outcome = {
            "standalone_validation_success": False,
            "apply_requested": True,
            "overlay_validation_success": True,
            "apply_success": True,
            "final_success": True,
            "status": "apply_applied",
        }

        experiment_db.log_run(sample_encoding_run)
        retrieved = experiment_db.get_run(sample_encoding_run.id)

        assert retrieved.outcome["status"] == "apply_applied"
        assert retrieved.iterations[0].success is False
        assert retrieved.success is True

    def test_iterations_keep_per_attempt_model_usage_and_cost(
        self, experiment_db, sample_encoding_run
    ):
        """An escalated run records each attempt's own model and spend, so the run can be re-priced."""
        sample_encoding_run.iterations = [
            Iteration(
                attempt=1,
                duration_ms=900,
                success=False,
                model="gpt-5.6-terra",
                input_tokens=40_000,
                output_tokens=5_000,
                cache_read_tokens=10_000,
                cache_creation_tokens=10_000,
                reasoning_output_tokens=800,
                estimated_cost_usd=0.11,
            ),
            Iteration(
                attempt=2,
                duration_ms=1200,
                success=True,
                model="gpt-5.6-sol",
                input_tokens=60_000,
                output_tokens=7_000,
                cache_read_tokens=20_000,
                cache_creation_tokens=20_000,
                reasoning_output_tokens=1_000,
                estimated_cost_usd=0.42,
            ),
            Iteration(
                attempt=3, duration_ms=100, success=False
            ),  # no usage reported: stays None, not 0
        ]

        experiment_db.log_run(sample_encoding_run)
        retrieved = experiment_db.get_run(sample_encoding_run.id)

        first, second, third = retrieved.iterations
        assert (first.model, first.input_tokens, first.estimated_cost_usd) == (
            "gpt-5.6-terra",
            40_000,
            0.11,
        )
        assert (
            second.model,
            second.cache_creation_tokens,
            second.estimated_cost_usd,
        ) == ("gpt-5.6-sol", 20_000, 0.42)
        assert (
            third.model is None
            and third.input_tokens is None
            and third.estimated_cost_usd is None
        )

    def test_log_run_with_review_issues(self, experiment_db):
        """Test logging a run with review issues at different severity levels."""
        review_results = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rulespec_reviewer",
                    passed=False,
                    items_checked=5,
                    items_passed=3,
                    critical_issues=["Missing entity declaration"],
                    important_issues=["Citation format incorrect"],
                    minor_issues=["Style: prefer lowercase"],
                ),
            ],
        )

        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
            review_results=review_results,
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        review = retrieved.review_results.reviews[0]
        assert review.passed is False
        assert review.critical_issues == ["Missing entity declaration"]
        assert review.important_issues == ["Citation format incorrect"]
        assert review.minor_issues == ["Style: prefer lowercase"]

    def test_get_nonexistent_run_returns_none(self, experiment_db):
        """Test that getting a nonexistent run returns None."""
        result = experiment_db.get_run("nonexistent-id")
        assert result is None


class TestUpdateReviewResults:
    """Tests for updating review results after validation."""

    def test_update_review_results(self, experiment_db, sample_review_results):
        """Test updating a run with review results."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        experiment_db.log_run(run)

        # Initially no review results
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.review_results is None

        # Update with review results
        experiment_db.update_review_results(run.id, sample_review_results)

        # Now has review results
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.review_results is not None
        assert len(retrieved.review_results.reviews) == 4
        assert retrieved.review_results.reviews[0].passed is True
        assert retrieved.review_results.policyengine_match == 0.90


class TestListRunsWithFilters:
    """Tests for listing runs with various filters."""

    def test_get_runs_for_citation(self, experiment_db):
        """Test getting all runs for a specific citation."""
        # Create runs for different citations
        for i in range(3):
            run = create_run(
                file_path=f"/path/to/file{i}.yaml",
                citation="26 USC 32",
                agent_type="axiom_encode:encoder",
                agent_model="claude-opus-4-6",
                rulespec_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        run_other = create_run(
            file_path="/path/to/other.yaml",
            citation="26 USC 24",  # Different citation
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# other content",
        )
        experiment_db.log_run(run_other)

        # Get runs for 26 USC 32
        runs = experiment_db.get_runs_for_citation("26 USC 32")
        assert len(runs) == 3
        assert all(r.citation == "26 USC 32" for r in runs)

        # Get runs for 26 USC 24
        runs = experiment_db.get_runs_for_citation("26 USC 24")
        assert len(runs) == 1
        assert runs[0].citation == "26 USC 24"

    def test_get_recent_runs(self, experiment_db):
        """Test getting most recent runs with limit."""
        # Create 5 runs
        for i in range(5):
            run = create_run(
                file_path=f"/path/to/file{i}.yaml",
                citation=f"26 USC {i}",
                agent_type="axiom_encode:encoder",
                agent_model="claude-opus-4-6",
                rulespec_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        # Get last 3
        runs = experiment_db.get_recent_runs(limit=3)
        assert len(runs) == 3

        # Most recent should be first (DESC order)
        # Since timestamps are very close, just verify we get 3 runs

    def test_get_recent_runs_default_limit(self, experiment_db):
        """Test getting recent runs with default limit."""
        run = create_run(
            file_path="/path/to/file.yaml",
            citation="26 USC 32",
            agent_type="axiom_encode:encoder",
            agent_model="claude-opus-4-6",
            rulespec_content="# content",
        )
        experiment_db.log_run(run)

        runs = experiment_db.get_recent_runs()
        assert len(runs) == 1


class TestReviewResultsProperties:
    """Tests for ReviewResults properties."""

    def test_passed_all_pass(self):
        """Test passed property when all reviews pass."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rulespec_reviewer", passed=True),
                ReviewResult(reviewer="formula_reviewer", passed=True),
            ]
        )
        assert rr.passed is True

    def test_passed_one_fails(self):
        """Test passed property when one review fails."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rulespec_reviewer", passed=True),
                ReviewResult(reviewer="formula_reviewer", passed=False),
            ]
        )
        assert rr.passed is False

    def test_passed_empty_reviews(self):
        """Test passed property with no reviews."""
        rr = ReviewResults(reviews=[])
        assert rr.passed is False

    def test_total_critical_issues(self):
        """Test total_critical_issues counts across all reviews."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rulespec",
                    critical_issues=["issue1", "issue2"],
                ),
                ReviewResult(
                    reviewer="formula",
                    critical_issues=["issue3"],
                ),
            ]
        )
        assert rr.total_critical_issues == 3


class TestSessionLogging:
    """Tests for session logging (used by SDK orchestrator)."""

    def test_start_session_generates_id(self, experiment_db):
        """Test that start_session generates a unique ID."""
        session = experiment_db.start_session(
            model="test-model", cwd="/tmp", axiom_encode_version="0.2.1"
        )
        assert session.id is not None
        assert len(session.id) == 8
        assert session.axiom_encode_version == "0.2.1"

    def test_start_session_with_custom_id(self, experiment_db):
        """Test that start_session accepts custom session_id."""
        session = experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="custom-123"
        )
        assert session.id == "custom-123"

    def test_start_session_links_run_id(self, experiment_db, sample_encoding_run):
        """Test that start_session can link SDK telemetry to an encoding run."""
        experiment_db.log_run(sample_encoding_run)
        session = experiment_db.start_session(
            model="test-model",
            cwd="/tmp",
            session_id="linked-session",
            run_id=sample_encoding_run.id,
        )

        assert session.run_id == sample_encoding_run.id
        retrieved = experiment_db.get_session("linked-session")
        assert retrieved is not None
        assert retrieved.run_id == sample_encoding_run.id

    def test_get_session_retrieves_by_id(self, experiment_db):
        """Test that get_session retrieves session by ID."""
        experiment_db.start_session(
            model="opus-4.5", cwd="/workspace", session_id="retrieve-test"
        )

        retrieved = experiment_db.get_session("retrieve-test")
        assert retrieved is not None
        assert retrieved.id == "retrieve-test"
        assert retrieved.model == "opus-4.5"
        assert retrieved.cwd == "/workspace"
        assert retrieved.axiom_encode_version == ""

    def test_get_session_returns_none_for_unknown(self, experiment_db):
        """Test that get_session returns None for unknown ID."""
        retrieved = experiment_db.get_session("nonexistent-id")
        assert retrieved is None

    def test_log_event_to_session(self, experiment_db):
        """Test logging events to a session."""
        experiment_db.start_session(session_id="event-test")

        event = experiment_db.log_event(
            session_id="event-test",
            event_type="agent_start",
            content="Test prompt",
            metadata={"agent_type": "encoder"},
        )

        assert event.sequence == 1
        assert event.event_type == "agent_start"

    def test_get_session_events(self, experiment_db):
        """Test retrieving all events for a session."""
        experiment_db.start_session(session_id="events-test")

        experiment_db.log_event(
            session_id="events-test", event_type="agent_start", content="Starting"
        )
        experiment_db.log_event(
            session_id="events-test", event_type="agent_end", content="Done"
        )

        events = experiment_db.get_session_events("events-test")
        assert len(events) == 2
        assert events[0].event_type == "agent_start"
        assert events[1].event_type == "agent_end"

    def test_session_event_count_updates(self, experiment_db):
        """Test that session event_count is tracked."""
        experiment_db.start_session(session_id="count-test")

        for i in range(3):
            experiment_db.log_event(session_id="count-test", event_type=f"event_{i}")

        session = experiment_db.get_session("count-test")
        assert session.event_count == 3


class TestRowToRun:
    """Tests for _row_to_run with the current schema."""

    def test_row_to_run(self, experiment_db):
        """Test _row_to_run parses current review result rows."""
        import json

        review_results = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": "rulespec_reviewer",
                        "passed": True,
                        "items_checked": 10,
                        "items_passed": 8,
                        "critical_issues": [],
                        "important_issues": [],
                        "minor_issues": [],
                        "lessons": "",
                    }
                ],
                "policyengine_match": 0.95,
                "oracle_context": {},
                "lessons": "Some lessons",
            }
        )
        row = (
            "test-id",
            "2024-01-01T00:00:00",
            "26 USC 32",
            "/path/file.yaml",
            "source text",
            "{}",
            "[]",
            3000,
            "encoder",
            "opus",
            "content",
            "sess-456",
            1,
            None,
            review_results,
            "Some lessons",
            "0.2.0",
            "{}",
            100,
            50,
            10,
            5,
            2,
            0.0123,
            0.0111,
            1,
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id"
        assert run.review_results is not None
        assert len(run.review_results.reviews) == 1
        assert run.review_results.reviews[0].passed is True
        assert run.review_results.policyengine_match == 0.95
        assert run.lessons == "Some lessons"
        assert run.rulespec_content == "content"
        assert run.source_text == "source text"
        assert run.axiom_encode_version == "0.2.0"
        assert run.outcome == {}

    def test_row_to_run_rejects_removed_taxsim_field(self, experiment_db):
        import json

        review_results = json.dumps(
            {
                "reviews": [],
                "policyengine_match": None,
                "taxsim_match": None,
                "oracle_context": {},
                "lessons": "",
            }
        )
        row = (
            "test-id",
            "2024-01-01T00:00:00",
            "26 USC 32",
            "/path/file.yaml",
            "source text",
            "{}",
            "[]",
            3000,
            "encoder",
            "opus",
            "content",
            None,
            1,
            None,
            review_results,
            "",
            "0.2.0",
            "{}",
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            0,
        )

        with pytest.raises(ValueError, match="unsupported schema"):
            experiment_db._row_to_run(row)


class TestAxiomEncodeVersion:
    """Tests for axiom_encode version tracking on encoding runs."""

    def test_create_run_sets_version(self):
        """Test that create_run auto-populates axiom_encode_version."""
        from axiom_encode import __version__
        from axiom_encode.harness.encoding_db import create_run

        run = create_run(
            file_path="/tmp/test.yaml",
            citation="26 USC 21",
            agent_type="encoder",
            agent_model="opus",
            rulespec_content="test",
        )
        assert run.axiom_encode_version == __version__

    def test_version_persisted_in_db(self, experiment_db, sample_encoding_run):
        """Test that axiom_encode_version is persisted and retrieved from DB."""
        sample_encoding_run.axiom_encode_version = "0.2.0"
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.axiom_encode_version == "0.2.0"

    def test_version_defaults_empty_for_old_runs(
        self, experiment_db, sample_encoding_run
    ):
        """Test that runs without axiom_encode_version default to empty string."""
        sample_encoding_run.axiom_encode_version = ""
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.axiom_encode_version == ""


class TestUpdateSessionTokens:
    """Tests for updating session tokens."""

    def test_update_session_tokens(self, experiment_db):
        """Test updating token usage for a session."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="token-test"
        )

        experiment_db.update_session_tokens(
            session_id="token-test",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            estimated_cost_usd=0.0123,
        )

        session = experiment_db.get_session("token-test")
        # Session.total_tokens = input_tokens + output_tokens = 1500
        assert session.total_tokens == 1500
        with sqlite3.connect(experiment_db.db_path) as conn:
            cost = conn.execute(
                "SELECT estimated_cost_usd FROM sessions WHERE id = 'token-test'"
            ).fetchone()[0]
        assert cost == 0.0123

    def test_accumulates_across_writes(self, experiment_db):
        """Each stage's write adds onto the session totals."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="accumulate-test"
        )

        experiment_db.update_session_tokens(
            session_id="accumulate-test",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
            cache_creation_tokens=100,
            reasoning_output_tokens=50,
            estimated_cost_usd=0.01,
        )
        experiment_db.update_session_tokens(
            session_id="accumulate-test",
            input_tokens=2000,
            output_tokens=1000,
            cache_read_tokens=400,
            cache_creation_tokens=200,
            reasoning_output_tokens=100,
            estimated_cost_usd=0.02,
        )

        session = experiment_db.get_session("accumulate-test")
        assert session.input_tokens == 3000
        assert session.output_tokens == 1500
        assert session.cache_read_tokens == 600
        assert session.cache_creation_tokens == 300
        assert session.reasoning_output_tokens == 150
        assert session.total_tokens == 4500
        assert session.estimated_cost_usd == pytest.approx(0.03)

    def test_unknown_cost_increment_poisons_total(self, experiment_db):
        """A token-spending write without a cost keeps the total unknown."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="poison-test"
        )

        experiment_db.update_session_tokens(
            session_id="poison-test",
            input_tokens=1000,
            output_tokens=500,
            estimated_cost_usd=None,
        )
        experiment_db.update_session_tokens(
            session_id="poison-test",
            input_tokens=2000,
            output_tokens=1000,
            estimated_cost_usd=0.02,
        )

        session = experiment_db.get_session("poison-test")
        assert session.input_tokens == 3000
        assert session.estimated_cost_usd is None

    def test_zero_token_costless_write_preserves_known_cost(self, experiment_db):
        """A bookkeeping write that spends nothing must not wipe a known cost."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="preserve-test"
        )

        experiment_db.update_session_tokens(
            session_id="preserve-test",
            input_tokens=1000,
            output_tokens=500,
            estimated_cost_usd=0.05,
        )
        experiment_db.update_session_tokens(session_id="preserve-test")

        session = experiment_db.get_session("preserve-test")
        assert session.input_tokens == 1000
        assert session.estimated_cost_usd == pytest.approx(0.05)

    def test_new_session_cost_starts_unknown(self, experiment_db):
        """A session that never records usage reports unknown cost, not $0."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="fresh-test"
        )

        session = experiment_db.get_session("fresh-test")
        assert session.estimated_cost_usd is None

    def test_missing_session_is_a_no_op(self, experiment_db):
        """Writing to an unknown session id must not raise."""
        experiment_db.update_session_tokens(
            session_id="does-not-exist", input_tokens=100
        )


class TestRunCostLedger:
    """Tests for per-run token usage and cost persistence."""

    def test_run_token_ledger_round_trips(self, experiment_db, sample_encoding_run):
        from axiom_encode.harness.encoding_db import TokenUsage

        sample_encoding_run.tokens = TokenUsage(
            input_tokens=76_000,
            output_tokens=3_700,
            cache_read_tokens=1_200,
            cache_creation_tokens=800,
            reasoning_output_tokens=400,
        )
        sample_encoding_run.estimated_cost_usd = 0.3322
        sample_encoding_run.actual_cost_usd = 0.31
        sample_encoding_run.generation_attempt_count = 2
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.tokens.input_tokens == 76_000
        assert retrieved.tokens.output_tokens == 3_700
        assert retrieved.tokens.cache_read_tokens == 1_200
        assert retrieved.tokens.cache_creation_tokens == 800
        assert retrieved.tokens.reasoning_output_tokens == 400
        assert retrieved.estimated_cost_usd == pytest.approx(0.3322)
        assert retrieved.actual_cost_usd == pytest.approx(0.31)
        assert retrieved.generation_attempt_count == 2

    def test_run_cost_defaults_to_unknown(self, experiment_db, sample_encoding_run):
        """Runs without usage keep cost as None (unknown), never zero."""
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.tokens.input_tokens == 0
        assert retrieved.estimated_cost_usd is None
        assert retrieved.actual_cost_usd is None
        assert retrieved.generation_attempt_count == 0


class TestLegacySessionCostBackfill:
    """Pre-ledger databases defaulted session cost to 0; unmeasured rows heal."""

    def _make_legacy_db(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_creation_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0,
                axiom_encode_version TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            "INSERT INTO sessions (id, started_at) VALUES ('legacy-unmeasured', '2024-01-01T00:00:00')"
        )
        conn.execute(
            "INSERT INTO sessions (id, started_at, input_tokens, estimated_cost_usd) "
            "VALUES ('legacy-free', '2024-01-01T00:00:00', 12, 0)"
        )
        conn.commit()
        conn.close()

    def test_unmeasured_legacy_session_cost_becomes_unknown(self, tmp_path):
        from axiom_encode.harness.encoding_db import EncodingDB

        db_path = tmp_path / "encodings.db"
        self._make_legacy_db(db_path)

        db = EncodingDB(db_path)
        assert db.get_session("legacy-unmeasured").estimated_cost_usd is None
        # A measured $0 (tokens recorded) is a real value and survives.
        assert db.get_session("legacy-free").estimated_cost_usd == 0.0
        # Re-opening is idempotent.
        assert EncodingDB(db_path).get_session("legacy-free").estimated_cost_usd == 0.0


# =============================================================================
# Attempt evidence: legacy migration, per-attempt issues, artifact versions,
# judge verdicts and parent links.
# =============================================================================

#: The shipped schema as it stood before attempt evidence landed: no
#: ``attempt``/``role`` on ``run_artifacts``, no ``judge_events`` table, no
#: ``idx_run_parent`` index, and none of the token/cost ledger columns.
LEGACY_SCHEMA_SQL = (
    """
    CREATE TABLE encoding_runs (
        id TEXT PRIMARY KEY,
        timestamp TEXT,
        citation TEXT,
        file_path TEXT,
        complexity_json TEXT,
        iterations_json TEXT,
        total_duration_ms INTEGER,
        final_scores_json TEXT,
        agent_type TEXT,
        agent_model TEXT,
        rac_content TEXT,
        predicted_scores_json TEXT,
        session_id TEXT,
        iteration INTEGER DEFAULT 1,
        parent_run_id TEXT,
        actual_scores_json TEXT,
        suggestions_json TEXT,
        review_results_json TEXT,
        lessons TEXT DEFAULT '',
        autorac_version TEXT DEFAULT '',
        source_text TEXT,
        rulespec_content TEXT,
        axiom_encode_version TEXT DEFAULT '',
        outcome_json TEXT DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE artifact_versions (
        id TEXT PRIMARY KEY,
        artifact_type TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        version_label TEXT,
        content TEXT,
        effective_from TEXT NOT NULL,
        effective_to TEXT,
        metadata_json TEXT
    )
    """,
    """
    CREATE TABLE run_artifacts (
        run_id TEXT NOT NULL,
        artifact_version_id TEXT NOT NULL,
        PRIMARY KEY (run_id, artifact_version_id)
    )
    """,
    """
    CREATE TABLE sessions (
        id TEXT PRIMARY KEY,
        run_id TEXT,
        started_at TEXT,
        ended_at TEXT,
        model TEXT,
        cwd TEXT,
        event_count INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE session_events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        sequence INTEGER,
        timestamp TEXT,
        event_type TEXT,
        tool_name TEXT,
        content TEXT,
        metadata_json TEXT
    )
    """,
)

LEGACY_ITERATIONS_JSON = json.dumps(
    [
        {
            "attempt": 1,
            "duration_ms": 1200,
            "success": False,
            "errors": [
                {
                    "error_type": "ci",
                    "message": "Generated RuleSpec failed CI validation",
                    "variable": None,
                    "fix_applied": None,
                }
            ],
        },
        {"attempt": 2, "duration_ms": 900, "success": True, "errors": []},
    ]
)

#: A review payload written by a schema this parser no longer supports.
UNSUPPORTED_REVIEW_JSON = json.dumps(
    {
        "reviews": [],
        "policyengine_match": None,
        "taxsim_match": None,
        "oracle_context": {},
        "lessons": "",
    }
)


def _create_legacy_db(db_path):
    """Build a pre-attempt-evidence database by hand."""
    conn = sqlite3.connect(str(db_path))
    for statement in LEGACY_SCHEMA_SQL:
        conn.execute(statement)
    conn.commit()
    conn.close()


def _insert_raw_run(
    db_path,
    *,
    run_id,
    citation="26 USC 32",
    timestamp="2026-03-01T10:00:00",
    iterations_json="[]",
    review_results_json=None,
    agent_type="encoder",
    agent_model="test-model",
    session_id=None,
    rulespec_content="# rulespec",
):
    """Insert an encoding_runs row directly, bypassing the writer."""
    if hasattr(timestamp, "isoformat"):
        timestamp = timestamp.isoformat()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO encoding_runs
            (id, timestamp, citation, file_path, complexity_json,
             iterations_json, total_duration_ms, agent_type, agent_model,
             rulespec_content, session_id, iteration, review_results_json,
             lessons, axiom_encode_version, outcome_json, source_text)
        VALUES (?, ?, ?, '/rules/eitc.yaml', '{}', ?, 1200, ?, ?, ?, ?, 1, ?,
                '', '', '{}', 'statute text')
        """,
        (
            run_id,
            timestamp,
            citation,
            iterations_json,
            agent_type,
            agent_model,
            rulespec_content,
            session_id,
            review_results_json,
        ),
    )
    conn.commit()
    conn.close()


def _fetch_run_row(db_path, run_id):
    """Fetch one run row in RUN_COLUMNS order, ready for run_from_row."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        f"SELECT {', '.join(RUN_COLUMNS)} FROM encoding_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    conn.close()
    return row


def _schema_snapshot(db_path):
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    conn.close()
    return rows


def _table_columns(db_path, table):
    conn = sqlite3.connect(str(db_path))
    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
    conn.close()
    return columns


def _count(db_path, table):
    conn = sqlite3.connect(str(db_path))
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()
    return total


class TestAttemptEvidenceMigration:
    """Opening a pre-attempt-evidence database upgrades it in place."""

    def _legacy_db_with_rows(self, db_path):
        _create_legacy_db(db_path)
        _insert_raw_run(
            db_path,
            run_id="legacy-run-1",
            iterations_json=LEGACY_ITERATIONS_JSON,
            session_id="legacy-sess",
        )
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            INSERT INTO sessions
                (id, run_id, started_at, ended_at, model, cwd, event_count,
                 total_tokens)
            VALUES ('legacy-sess', 'legacy-run-1', '2026-03-01T09:58:00',
                    '2026-03-01T10:00:00', 'test-model', '/work', 1, 4200)
            """
        )
        conn.execute(
            """
            INSERT INTO session_events
                (id, session_id, sequence, timestamp, event_type, tool_name,
                 content, metadata_json)
            VALUES ('legacy-event-1', 'legacy-sess', 1, '2026-03-01T09:58:30',
                    'encode_request', NULL, 'encode 26 USC 32', '{}')
            """
        )
        conn.commit()
        conn.close()
        return db_path

    def test_opening_legacy_db_migrates_without_touching_rows(self, tmp_path):
        db_path = self._legacy_db_with_rows(tmp_path / "encodings.db")

        db = EncodingDB(db_path)  # must not raise

        # The pre-existing rows survive untouched.
        conn = sqlite3.connect(str(db_path))
        run_row = conn.execute(
            "SELECT citation, agent_model, session_id, iterations_json "
            "FROM encoding_runs WHERE id = 'legacy-run-1'"
        ).fetchone()
        session_row = conn.execute(
            "SELECT run_id, ended_at, event_count, total_tokens "
            "FROM sessions WHERE id = 'legacy-sess'"
        ).fetchone()
        event_row = conn.execute(
            "SELECT session_id, sequence, event_type, content "
            "FROM session_events WHERE id = 'legacy-event-1'"
        ).fetchone()
        conn.close()
        assert run_row == (
            "26 USC 32",
            "test-model",
            "legacy-sess",
            LEGACY_ITERATIONS_JSON,
        )
        assert session_row == ("legacy-run-1", "2026-03-01T10:00:00", 1, 4200)
        assert event_row == (
            "legacy-sess",
            1,
            "encode_request",
            "encode 26 USC 32",
        )

        # run_artifacts gained the attempt-evidence columns.
        run_artifact_columns = _table_columns(db_path, "run_artifacts")
        assert "attempt" in run_artifact_columns
        assert "role" in run_artifact_columns

        # judge_events and the parent index now exist.
        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
        conn.close()
        assert "judge_events" in tables
        assert "idx_run_parent" in indexes

        # The migrated database is writable through the new surface.
        assert db.get_judge_events("legacy-run-1") == []

    def test_second_open_is_a_no_op(self, tmp_path):
        db_path = self._legacy_db_with_rows(tmp_path / "encodings.db")

        EncodingDB(db_path)
        schema_after_first = _schema_snapshot(db_path)
        counts_after_first = {
            table: _count(db_path, table)
            for table in ("encoding_runs", "sessions", "session_events")
        }

        EncodingDB(db_path)

        assert _schema_snapshot(db_path) == schema_after_first
        assert {
            table: _count(db_path, table)
            for table in ("encoding_runs", "sessions", "session_events")
        } == counts_after_first

    def test_get_run_reads_the_legacy_row(self, tmp_path):
        db_path = self._legacy_db_with_rows(tmp_path / "encodings.db")
        db = EncodingDB(db_path)

        run = db.get_run("legacy-run-1")

        assert run is not None
        assert run.citation == "26 USC 32"
        assert run.agent_model == "test-model"
        assert run.session_id == "legacy-sess"
        assert len(run.iterations) == 2
        assert run.iterations[0].errors[0].message == (
            "Generated RuleSpec failed CI validation"
        )
        # Rows written before issues were persisted read back as no issues.
        assert run.iterations[0].errors[0].issues == []
        assert run.iterations[0].errors[0].issues_truncated == 0
        # The ledger columns the migration added default to unknown/zero.
        assert run.tokens.input_tokens == 0
        assert run.estimated_cost_usd is None

    def test_object_iterations_json_parses_as_no_attempts(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        _create_legacy_db(db_path)
        _insert_raw_run(
            db_path,
            run_id="legacy-obj",
            iterations_json=json.dumps({"attempt": 1, "duration_ms": 10}),
        )
        EncodingDB(db_path)
        row = _fetch_run_row(db_path, "legacy-obj")

        assert run_from_row(row, strict=False).iterations == []
        # A non-list payload is not a schema violation, only an empty journey,
        # so the strict reader tolerates it too.
        assert run_from_row(row).iterations == []

    def test_unsupported_review_schema_raises_only_when_strict(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        _create_legacy_db(db_path)
        _insert_raw_run(
            db_path,
            run_id="legacy-review",
            review_results_json=UNSUPPORTED_REVIEW_JSON,
        )
        EncodingDB(db_path)
        row = _fetch_run_row(db_path, "legacy-review")

        with pytest.raises(ValueError, match="unsupported schema"):
            run_from_row(row)

        run = run_from_row(row, strict=False)
        assert run.id == "legacy-review"
        assert run.citation == "26 USC 32"

    def test_lenient_read_drops_unsupported_review_results(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        _create_legacy_db(db_path)
        _insert_raw_run(
            db_path,
            run_id="legacy-review-none",
            review_results_json=UNSUPPORTED_REVIEW_JSON,
        )
        EncodingDB(db_path)
        row = _fetch_run_row(db_path, "legacy-review-none")

        assert run_from_row(row, strict=False).review_results is None


class TestIterationIssues:
    """Structured validator issues round-trip with each attempt."""

    ISSUE = ValidationIssue(
        gate="ci",
        kind="ungrounded_literal",
        message=(
            "[grounding] Ungrounded generated numeric literal: 600000 does "
            "not appear in the source text for 26 USC 32(b) at line 42"
        ),
        category="grounding",
        locator="rule:eitc_phase_in",
        line=42,
        value="600000",
        clause="26 USC 32(b)",
    )

    @staticmethod
    def _raw_iterations_json(db, run_id):
        conn = sqlite3.connect(str(db.db_path))
        raw = conn.execute(
            "SELECT iterations_json FROM encoding_runs WHERE id = ?",
            (run_id,),
        ).fetchone()[0]
        conn.close()
        return json.loads(raw)

    def test_issues_round_trip_through_log_and_get(
        self, experiment_db, sample_encoding_run
    ):
        sample_encoding_run.iterations = [
            Iteration(
                attempt=1,
                duration_ms=1200,
                success=False,
                errors=[
                    IterationError(
                        error_type="ci",
                        message="Generated RuleSpec failed CI validation",
                        issues=[self.ISSUE],
                    )
                ],
            ),
            Iteration(attempt=2, duration_ms=800, success=True),
        ]
        experiment_db.log_run(sample_encoding_run)

        run = experiment_db.get_run(sample_encoding_run.id)
        issues = run.iterations[0].errors[0].issues
        assert len(issues) == 1
        restored = issues[0]
        assert restored == self.ISSUE
        assert restored.gate == "ci"
        assert restored.kind == "ungrounded_literal"
        assert restored.message == self.ISSUE.message
        assert restored.locator == "rule:eitc_phase_in"
        assert restored.line == 42
        assert restored.value == "600000"
        assert restored.clause == "26 USC 32(b)"
        assert run.iterations[1].errors == []

    def test_error_without_issues_keeps_the_old_byte_shape(
        self, experiment_db, sample_encoding_run
    ):
        sample_encoding_run.iterations = [
            Iteration(
                attempt=1,
                duration_ms=1200,
                success=False,
                errors=[
                    IterationError(
                        error_type="parse",
                        message="YAML parse failure",
                        variable="eitc",
                        fix_applied="reindent",
                    )
                ],
            )
        ]
        experiment_db.log_run(sample_encoding_run)

        payload = self._raw_iterations_json(experiment_db, sample_encoding_run.id)
        error = payload[0]["errors"][0]
        assert set(error) == {"error_type", "message", "variable", "fix_applied"}
        assert "issues" not in error
        assert "issues_truncated" not in error

    def test_old_format_row_parses_with_empty_issues(
        self, experiment_db, sample_encoding_run
    ):
        experiment_db.log_run(sample_encoding_run)
        legacy_payload = json.dumps(
            [
                {
                    "attempt": 1,
                    "duration_ms": 1500,
                    "success": False,
                    "errors": [
                        {
                            "error_type": "test",
                            "message": "fixture execution failed",
                            "variable": None,
                            "fix_applied": None,
                        }
                    ],
                }
            ]
        )
        conn = sqlite3.connect(str(experiment_db.db_path))
        conn.execute(
            "UPDATE encoding_runs SET iterations_json = ? WHERE id = ?",
            (legacy_payload, sample_encoding_run.id),
        )
        conn.commit()
        conn.close()

        run = experiment_db.get_run(sample_encoding_run.id)
        error = run.iterations[0].errors[0]
        assert error.error_type == "test"
        assert error.issues == []
        assert error.issues_truncated == 0

    def test_issues_truncated_persists(self, experiment_db, sample_encoding_run):
        sample_encoding_run.iterations = [
            Iteration(
                attempt=1,
                duration_ms=1200,
                success=False,
                errors=[
                    IterationError(
                        error_type="ci",
                        message="Generated RuleSpec failed CI validation",
                        issues=[self.ISSUE],
                        issues_truncated=7,
                    )
                ],
            )
        ]
        experiment_db.log_run(sample_encoding_run)

        payload = self._raw_iterations_json(experiment_db, sample_encoding_run.id)
        assert payload[0]["errors"][0]["issues_truncated"] == 7

        run = experiment_db.get_run(sample_encoding_run.id)
        error = run.iterations[0].errors[0]
        assert error.issues_truncated == 7
        assert len(error.issues) == 1


class TestArtifactVersions:
    """Per-attempt artifact text is stored once and linked to the run."""

    RULESPEC = "outputs:\n  eitc:\n    dtype: Money\n"
    TESTS = "- name: single filer\n  output: {eitc: 600}\n"

    def test_record_run_artifact_stores_content_and_hash(self, experiment_db):
        version = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
            metadata={"ci_pass": False, "issue_count": 3, "model": "test-model"},
        )

        assert version.artifact_type == ARTIFACT_TYPE_RULESPEC
        assert version.content == self.RULESPEC
        assert (
            version.content_hash
            == hashlib.sha256(self.RULESPEC.encode("utf-8")).hexdigest()
        )
        assert version.version_label == "attempt-1"
        assert version.metadata == {
            "ci_pass": False,
            "issue_count": 3,
            "model": "test-model",
        }

        stored = experiment_db.get_artifact_version(version.id)
        assert stored is not None
        assert stored.content == self.RULESPEC
        assert stored.metadata["issue_count"] == 3

    def test_recording_the_same_content_twice_is_idempotent(self, experiment_db):
        first = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
            metadata={"ci_pass": False},
        )
        second = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
            metadata={"ci_pass": False},
        )

        assert second.id == first.id
        assert _count(experiment_db.db_path, "artifact_versions") == 1
        assert _count(experiment_db.db_path, "run_artifacts") == 1

    def test_different_content_for_one_attempt_adds_a_version(self, experiment_db):
        first = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
        )
        repaired = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC + "# repaired\n",
        )

        assert repaired.id != first.id
        assert _count(experiment_db.db_path, "artifact_versions") == 2
        linked = experiment_db.get_run_artifacts("run-1")
        assert len(linked) == 2
        assert {version.id for _, version in linked} == {first.id, repaired.id}

    def test_get_run_artifacts_orders_by_attempt_then_role(self, experiment_db):
        experiment_db.record_run_artifact(
            "run-1",
            attempt=2,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC + "# second attempt\n",
        )
        experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC_TESTS,
            content=self.TESTS,
        )
        experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
        )

        linked = experiment_db.get_run_artifacts("run-1")
        assert [(link.attempt, link.role) for link, _ in linked] == [
            (1, ARTIFACT_TYPE_RULESPEC),
            (1, ARTIFACT_TYPE_RULESPEC_TESTS),
            (2, ARTIFACT_TYPE_RULESPEC),
        ]
        assert [version.version_label for _, version in linked] == [
            "attempt-1",
            "attempt-1",
            "attempt-2",
        ]

    def test_both_roles_coexist_for_one_attempt(self, experiment_db):
        rulespec = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC,
            content=self.RULESPEC,
        )
        tests = experiment_db.record_run_artifact(
            "run-1",
            attempt=1,
            role=ARTIFACT_TYPE_RULESPEC_TESTS,
            content=self.TESTS,
        )

        assert rulespec.id != tests.id
        by_role = {
            link.role: version
            for link, version in experiment_db.get_run_artifacts("run-1")
        }
        assert by_role[ARTIFACT_TYPE_RULESPEC].content == self.RULESPEC
        assert by_role[ARTIFACT_TYPE_RULESPEC_TESTS].content == self.TESTS

    def test_unknown_artifact_version_is_none(self, experiment_db):
        assert experiment_db.get_artifact_version("does-not-exist") is None


class TestJudgeEvents:
    """Canonical run-log judge events mirror into judge_events."""

    @staticmethod
    def _flag_event(run_id="r1", seq=0):
        event = JudgeEvent(
            stage=JudgeStage.STATUTORY_FIDELITY,
            verdict=Verdict.FLAG,
            confidence=0.82,
            findings=[
                Finding(
                    clause_ref="26 USC 32(b)",
                    rule_path="rules/eitc/phase_in_rate.yaml",
                    kind="amount_mismatch",
                    explanation="Phase-in rate does not match the statute.",
                )
            ],
            model="test-judge-model",
            generator_model="gpt-5.6-terra",
            tokens=TokenCounts(input=1234, output=56),
            judge_prompt_sha256="a" * 64,
        )
        return event.to_run_log_event(run_id=run_id, seq=seq).to_dict()

    def test_flag_verdict_is_stored_in_full(self, experiment_db):
        payload = self._flag_event()

        row = experiment_db.log_judge_event(payload)

        assert row is not None
        assert row.run_id == "r1"
        assert row.judge_stage == "statutory_fidelity"
        assert row.verdict == "flag"
        assert row.status == "passed"
        assert row.confidence == pytest.approx(0.82)
        assert row.judge_model == "test-judge-model"
        assert row.generator_model == "gpt-5.6-terra"
        assert row.input_tokens == 1234
        assert row.output_tokens == 56
        assert row.judge_prompt_sha256 == "a" * 64
        assert row.judge_error is None
        assert row.source == "live"
        assert len(row.findings) == 1
        assert row.findings[0]["code"] == "amount_mismatch"

        stored = experiment_db.get_judge_events("r1")
        assert len(stored) == 1
        assert stored[0].id == payload["event_id"]
        assert stored[0].verdict == "flag"
        assert stored[0].findings[0]["code"] == "amount_mismatch"
        assert stored[0].event["stage"] == "judge"

    def test_error_event_keeps_its_cause(self, experiment_db):
        event = error_event(
            JudgeStage.DISPOSITION,
            "judge backend timed out",
            error_type="judge_api_error",
            model="test-judge-model",
            generator_model="gpt-5.6-terra",
            tokens=TokenCounts(input=10, output=0),
            run_id="r1",
            subject_ref="26 USC 32",
        )
        payload = event.to_run_log_event(seq=1).to_dict()

        experiment_db.log_judge_event(payload, source="backfill:/logs/r1.jsonl")

        stored = experiment_db.get_judge_events("r1")[0]
        assert stored.verdict == "error"
        assert stored.status == "error"
        assert stored.reason_code == "judge_api_error"
        assert stored.judge_error == {
            "type": "judge_api_error",
            "message": "judge backend timed out",
        }
        assert stored.confidence is None
        assert stored.source == "backfill:/logs/r1.jsonl"

    def test_relogging_the_same_event_is_a_no_op(self, experiment_db):
        payload = self._flag_event()

        assert experiment_db.log_judge_event(payload) is not None
        assert experiment_db.log_judge_event(payload) is None
        assert _count(experiment_db.db_path, "judge_events") == 1

    def test_non_judge_stage_event_is_ignored(self, experiment_db):
        payload = self._flag_event()
        payload["stage"] = "generate"

        assert experiment_db.log_judge_event(payload) is None
        assert _count(experiment_db.db_path, "judge_events") == 0

    def test_event_without_run_id_is_ignored(self, experiment_db):
        event = JudgeEvent(
            stage=JudgeStage.GRID_ADEQUACY,
            verdict=Verdict.PASS,
            model="test-judge-model",
        )
        payload = event.to_run_log_event(seq=0).to_dict()
        assert payload["run_id"] == ""

        assert experiment_db.log_judge_event(payload) is None
        assert _count(experiment_db.db_path, "judge_events") == 0

    def test_get_judge_events_orders_by_seq(self, experiment_db):
        for seq in (2, 0, 1):
            experiment_db.log_judge_event(self._flag_event(seq=seq))

        assert [row.seq for row in experiment_db.get_judge_events("r1")] == [0, 1, 2]

    def test_advisory_and_escalated_round_trip_as_bools(self, experiment_db):
        advisory_event = JudgeEvent(
            stage=JudgeStage.GOLDEN_DRIFT,
            verdict=Verdict.FLAG,
            advisory=True,
            escalated=False,
            model="test-judge-model",
        )
        promoted_event = JudgeEvent(
            stage=JudgeStage.GOLDEN_DRIFT,
            verdict=Verdict.FLAG,
            advisory=False,
            escalated=True,
            model="test-judge-model",
        )
        experiment_db.log_judge_event(
            advisory_event.to_run_log_event(run_id="r1", seq=0).to_dict()
        )
        experiment_db.log_judge_event(
            promoted_event.to_run_log_event(run_id="r1", seq=1).to_dict()
        )

        advisory, promoted = experiment_db.get_judge_events("r1")
        assert advisory.advisory is True
        assert advisory.escalated is False
        assert advisory.status == "passed"
        assert promoted.advisory is False
        assert promoted.escalated is True
        assert promoted.status == "failed"
        assert promoted.reason_code == "judge_rejected"

    def test_subject_ref_from_stage_extra_is_persisted(self, experiment_db):
        event = JudgeEvent(
            stage=JudgeStage.WORKLIST_PRECLASSIFY,
            verdict=Verdict.SKIP,
            model="test-judge-model",
            extra={"subject_ref": "26 USC 32", "classification": "out_of_scope"},
        )
        experiment_db.log_judge_event(
            event.to_run_log_event(run_id="r1", seq=0).to_dict()
        )

        stored = experiment_db.get_judge_events("r1")[0]
        assert stored.subject_ref == "26 USC 32"
        assert stored.status == "skipped"
        assert stored.reason_code == "out_of_scope"

    def test_subject_ref_and_judge_findings_persist_through_emit(
        self, experiment_db, tmp_path
    ):
        # The canonical event folds clause_ref into the message and never
        # carries subject_ref, so both travel through the emit hook instead.
        from axiom_encode.run_log import RunLogWriter

        event = error_event(
            JudgeStage.DISPOSITION,
            "judge backend timed out",
            run_id="r1",
            subject_ref="26 USC 32",
        )
        written = event.emit(
            RunLogWriter("r1", log_dir=tmp_path), db_path=experiment_db.db_path
        )
        assert written is not None

        stored = experiment_db.get_judge_events("r1")[0]
        assert stored.id == written.event_id
        assert stored.subject_ref == "26 USC 32"
        assert stored.judge_findings == [
            {
                "clause_ref": "26 USC 32",
                "rule_path": "26 USC 32",
                "kind": "judge_error",
                "explanation": "judge backend timed out",
            }
        ]

    def test_backfilled_event_unfolds_clause_ref_and_rule_path(self, experiment_db):
        event = JudgeEvent(
            stage=JudgeStage.STATUTORY_FIDELITY,
            verdict=Verdict.FLAG,
            findings=[
                Finding(
                    clause_ref="26 USC 32(b)",
                    rule_path="eitc.yaml#rate",
                    kind="amount_mismatch",
                    explanation="600000 not in source",
                )
            ],
        ).to_run_log_event(run_id="r2", seq=0)

        experiment_db.log_judge_event(event.to_dict(), source="backfill:x.jsonl")

        stored = experiment_db.get_judge_events("r2")[0]
        assert stored.subject_ref is None
        assert stored.judge_findings == [
            {
                "clause_ref": "26 USC 32(b)",
                "rule_path": "eitc.yaml#rate",
                "kind": "amount_mismatch",
                "explanation": "600000 not in source",
                "derived": True,
            }
        ]


class TestFindParentRun:
    """A regeneration links to the run it followed, never to a sibling."""

    CITATION = "26 USC 32"
    T0 = datetime(2026, 3, 1, 9, 0, 0)

    def _log_run(
        self,
        db,
        *,
        run_id,
        timestamp,
        citation=None,
        agent_type="encoder",
        agent_model="test-model",
        session_id=None,
    ):
        run = create_run(
            file_path="/rules/eitc.yaml",
            citation=citation or self.CITATION,
            agent_type=agent_type,
            agent_model=agent_model,
            rulespec_content="# rulespec",
        )
        run.id = run_id
        run.timestamp = timestamp
        run.session_id = session_id
        run.total_duration_ms = 1200
        db.log_run(run)
        return run

    @staticmethod
    def _end_session_at(db, session_id, ended_at):
        conn = sqlite3.connect(str(db.db_path))
        conn.execute(
            "UPDATE sessions SET ended_at = ? WHERE id = ?",
            (ended_at.isoformat(), session_id),
        )
        conn.commit()
        conn.close()

    def test_sequential_regeneration_links_to_the_finished_run(self, experiment_db):
        prior = self._log_run(
            experiment_db,
            run_id="prior-1",
            timestamp=self.T0,
            session_id="sess-prior",
        )
        experiment_db.start_session(session_id="sess-prior", run_id=prior.id)
        self._end_session_at(
            experiment_db, "sess-prior", self.T0 + timedelta(minutes=5)
        )

        parent = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(hours=1),
            agent_type="encoder",
            agent_model="test-model",
        )

        assert parent is not None
        assert parent.id == "prior-1"
        assert parent.iteration == 1
        assert parent.citation == self.CITATION
        assert parent.agent_model == "test-model"
        assert parent.ended_at == (self.T0 + timedelta(minutes=5)).isoformat()

    def test_concurrent_sibling_does_not_link(self, experiment_db):
        sibling = self._log_run(
            experiment_db,
            run_id="sibling-1",
            timestamp=self.T0,
            session_id="sess-sibling",
        )
        experiment_db.start_session(session_id="sess-sibling", run_id=sibling.id)
        started_at = self.T0 + timedelta(minutes=1)
        # The sibling was still running when the new run started.
        self._end_session_at(
            experiment_db, "sess-sibling", started_at + timedelta(minutes=10)
        )

        assert (
            experiment_db.find_parent_run(
                citation=self.CITATION,
                started_at=started_at,
                agent_type="encoder",
                agent_model="test-model",
            )
            is None
        )

    def test_run_timestamp_is_used_when_no_session_row_exists(self, experiment_db):
        self._log_run(experiment_db, run_id="no-session", timestamp=self.T0)

        linked = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(minutes=30),
            agent_type="encoder",
            agent_model="test-model",
        )
        assert linked is not None
        assert linked.id == "no-session"
        assert linked.ended_at is None

        # The same row written after the new run started is not a parent.
        assert (
            experiment_db.find_parent_run(
                citation=self.CITATION,
                started_at=self.T0 - timedelta(minutes=30),
                agent_type="encoder",
                agent_model="test-model",
            )
            is None
        )

    def test_same_backend_and_model_beats_a_more_recent_run(self, experiment_db):
        self._log_run(experiment_db, run_id="same-model", timestamp=self.T0)
        self._log_run(
            experiment_db,
            run_id="other-model",
            timestamp=self.T0 + timedelta(minutes=30),
            agent_model="other-test-model",
        )

        parent = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(hours=1),
            agent_type="encoder",
            agent_model="test-model",
        )

        assert parent is not None
        assert parent.id == "same-model"

    def test_a_different_citation_never_links(self, experiment_db):
        self._log_run(
            experiment_db,
            run_id="other-citation",
            timestamp=self.T0,
            citation="26 USC 24",
        )

        assert (
            experiment_db.find_parent_run(
                citation=self.CITATION,
                started_at=self.T0 + timedelta(hours=1),
                agent_type="encoder",
                agent_model="test-model",
            )
            is None
        )

    def test_exclude_run_id_is_honored(self, experiment_db):
        self._log_run(experiment_db, run_id="older", timestamp=self.T0)
        self._log_run(
            experiment_db,
            run_id="newer",
            timestamp=self.T0 + timedelta(minutes=10),
        )

        without_exclusion = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(hours=1),
            agent_type="encoder",
            agent_model="test-model",
        )
        assert without_exclusion.id == "newer"

        excluded = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(hours=1),
            agent_type="encoder",
            agent_model="test-model",
            exclude_run_id="newer",
        )
        assert excluded is not None
        assert excluded.id == "older"

    def test_update_run_parent_persists_the_link(self, experiment_db):
        self._log_run(experiment_db, run_id="parent-run", timestamp=self.T0)
        child = self._log_run(
            experiment_db,
            run_id="child-run",
            timestamp=self.T0 + timedelta(hours=1),
        )
        assert child.parent_run_id is None

        experiment_db.update_run_parent("child-run", "parent-run", 2)

        stored = experiment_db.get_run("child-run")
        assert stored.parent_run_id == "parent-run"
        assert stored.iteration == 2

    def test_links_to_a_row_with_an_unsupported_review_schema(self, experiment_db):
        _insert_raw_run(
            experiment_db.db_path,
            run_id="legacy-parent",
            citation=self.CITATION,
            timestamp=self.T0,
            review_results_json=UNSUPPORTED_REVIEW_JSON,
        )

        parent = experiment_db.find_parent_run(
            citation=self.CITATION,
            started_at=self.T0 + timedelta(hours=1),
            agent_type="encoder",
            agent_model="test-model",
        )

        assert parent is not None
        assert parent.id == "legacy-parent"
        assert parent.iteration == 1
        # The reference is identity only: reading the row itself still raises.
        with pytest.raises(ValueError, match="unsupported schema"):
            experiment_db.get_run("legacy-parent")

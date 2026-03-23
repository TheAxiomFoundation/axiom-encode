"""
Tests for the experiment database.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac import (
    EncodingDB,
    ReviewResult,
    ReviewResults,
    create_run,
)


class TestCreateRun:
    """Tests for the create_run factory function."""

    def test_create_run_generates_id(self):
        """Test that create_run generates a unique ID."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
        )
        assert run.id is not None
        assert len(run.id) == 8  # UUID[:8]

    def test_create_run_sets_timestamp(self):
        """Test that create_run sets current timestamp."""
        before = datetime.now()
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
        )
        after = datetime.now()
        assert before <= run.timestamp <= after

    def test_create_run_sets_iteration_1_for_new_run(self):
        """Test that new runs have iteration=1."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
        )
        assert run.iteration == 1
        assert run.parent_run_id is None

    def test_create_run_sets_iteration_2_for_revision(self):
        """Test that revisions have iteration=2."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
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
        assert retrieved.rac_content == sample_encoding_run.rac_content

    def test_log_run_with_review_results(self, experiment_db, sample_review_results):
        """Test logging a run with review results."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
            review_results=sample_review_results,
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.review_results is not None
        assert len(retrieved.review_results.reviews) == 4
        assert retrieved.review_results.reviews[0].reviewer == "rac_reviewer"
        assert retrieved.review_results.reviews[0].passed is True
        assert retrieved.review_results.policyengine_match == 0.90

    def test_log_run_with_lessons(self, experiment_db):
        """Test logging a run with lessons."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
            lessons="Learned that bracket syntax needs special handling.",
        )

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert (
            retrieved.lessons == "Learned that bracket syntax needs special handling."
        )

    def test_log_run_with_review_issues(self, experiment_db):
        """Test logging a run with review issues at different severity levels."""
        review_results = ReviewResults(
            reviews=[
                ReviewResult(
                    reviewer="rac_reviewer",
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
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
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
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
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
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-6",
                rac_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        run_other = create_run(
            file_path="/path/to/other.rac",
            citation="26 USC 24",  # Different citation
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# other content",
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
                file_path=f"/path/to/file{i}.rac",
                citation=f"26 USC {i}",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-6",
                rac_content=f"# content {i}",
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
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
            rac_content="# content",
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
                ReviewResult(reviewer="rac_reviewer", passed=True),
                ReviewResult(reviewer="formula_reviewer", passed=True),
            ]
        )
        assert rr.passed is True

    def test_passed_one_fails(self):
        """Test passed property when one review fails."""
        rr = ReviewResults(
            reviews=[
                ReviewResult(reviewer="rac_reviewer", passed=True),
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
                    reviewer="rac",
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
            model="test-model", cwd="/tmp", autorac_version="0.2.1"
        )
        assert session.id is not None
        assert len(session.id) == 8
        assert session.autorac_version == "0.2.1"

    def test_start_session_with_custom_id(self, experiment_db):
        """Test that start_session accepts custom session_id."""
        session = experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="custom-123"
        )
        assert session.id == "custom-123"

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
        assert retrieved.autorac_version == ""

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


class TestRowToRunSchemaVersions:
    """Tests for _row_to_run with different schema versions."""

    def test_row_with_11_columns(self, experiment_db):
        """Test _row_to_run with legacy 11-column schema."""

        row = (
            "test-id",  # id
            "2024-01-01T00:00:00",  # timestamp
            "26 USC 32",  # citation
            "/path/file.rac",  # file_path
            "{}",  # complexity_json
            "[]",  # iterations_json
            1000,  # total_duration_ms
            None,  # final_scores_json
            "encoder",  # agent_type
            "opus",  # agent_model
            "content",  # rac_content
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id"
        assert run.citation == "26 USC 32"
        assert run.session_id is None
        assert run.iteration == 1

    def test_row_with_13_columns(self, experiment_db):
        """Test _row_to_run with 13-column schema (added predicted_scores, session_id)."""
        import json

        predicted = json.dumps(
            {
                "rac_reviewer": 8.0,
                "formula_reviewer": 7.0,
                "parameter_reviewer": 7.5,
                "integration_reviewer": 7.0,
                "ci_pass": True,
                "confidence": 0.6,
            }
        )
        row = (
            "test-id-13",
            "2024-01-01T00:00:00",
            "26 USC 24",
            "/path/file.rac",
            "{}",
            "[]",
            2000,
            None,
            "encoder",
            "opus",
            "content",
            predicted,  # predicted_scores_json
            "sess-123",  # session_id
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id-13"
        assert run.session_id == "sess-123"
        # Old predicted_scores_json is converted to review_results
        assert run.review_results is not None
        assert len(run.review_results.reviews) == 4
        assert run.iteration == 1

    def test_row_with_19_columns(self, experiment_db):
        """Test _row_to_run with 19-column schema (new review_results_json, lessons)."""
        import json

        review_results = json.dumps(
            {
                "reviews": [
                    {
                        "reviewer": "rac_reviewer",
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
                "taxsim_match": None,
                "oracle_context": {},
                "lessons": "Some lessons",
            }
        )
        row = (
            "test-id-19",
            "2024-01-01T00:00:00",
            "26 USC 32",
            "/path/file.rac",
            "{}",
            "[]",
            3000,
            None,  # final_scores_json (legacy)
            "encoder",
            "opus",
            "content",
            None,  # predicted_scores_json (legacy)
            "sess-456",
            1,
            None,
            None,  # actual_scores_json (legacy)
            None,  # suggestions_json (legacy)
            review_results,
            "Some lessons",
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id-19"
        assert run.review_results is not None
        assert len(run.review_results.reviews) == 1
        assert run.review_results.reviews[0].passed is True
        assert run.review_results.policyengine_match == 0.95
        assert run.lessons == "Some lessons"


class TestBackwardCompatProperties:
    """Test backward-compat properties on EncodingRun."""

    def test_predicted_setter_sets_review_results(self):
        """Test EncodingRun.predicted setter sets review_results."""
        from autorac import EncodingRun, ReviewResults

        run = EncodingRun(
            file_path="/path.rac",
            citation="26 USC 1",
            agent_type="encoder",
            agent_model="opus",
            rac_content="content",
        )
        rr = ReviewResults()
        run.predicted = rr
        assert run.review_results is rr
        assert run.predicted is rr

    def test_final_scores_setter_sets_review_results(self):
        """Test EncodingRun.final_scores setter sets review_results."""
        from autorac import EncodingRun, ReviewResults

        run = EncodingRun(
            file_path="/path.rac",
            citation="26 USC 1",
            agent_type="encoder",
            agent_model="opus",
            rac_content="content",
        )
        rr = ReviewResults()
        run.final_scores = rr
        assert run.review_results is rr
        assert run.final_scores is rr

    def test_suggestions_property_returns_empty_list(self):
        """Test EncodingRun.suggestions returns empty list."""
        from autorac import EncodingRun

        run = EncodingRun()
        assert run.suggestions == []
        # Setting should not raise
        run.suggestions = ["something"]
        assert run.suggestions == []


class TestMigration:
    """Tests for database migration (old 'runs' table to 'encoding_runs')."""

    def test_migrate_runs_table(self, temp_db_path):
        """Test that old 'runs' table is renamed to 'encoding_runs'."""
        import sqlite3

        # Create old-style DB with 'runs' table
        conn = sqlite3.connect(temp_db_path)
        conn.execute("""
            CREATE TABLE runs (
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
                rac_content TEXT
            )
        """)
        conn.execute("""
            INSERT INTO runs VALUES (
                'old-run', '2024-01-01T00:00:00', '26 USC 1',
                '/path.rac', '{}', '[]', 1000, NULL, 'encoder', 'opus', 'content'
            )
        """)
        conn.commit()
        conn.close()

        # Now init EncodingDB — should migrate
        EncodingDB(temp_db_path)

        # Old table should be gone
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cursor.fetchone() is None

        # encoding_runs should have the data
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
        )
        assert cursor.fetchone() is not None

        cursor.execute("SELECT id FROM encoding_runs")
        assert cursor.fetchone()[0] == "old-run"
        conn.close()


class TestAutoracVersion:
    """Tests for autorac version tracking on encoding runs."""

    def test_create_run_sets_version(self):
        """Test that create_run auto-populates autorac_version."""
        from autorac import __version__
        from autorac.harness.encoding_db import create_run

        run = create_run(
            file_path="/tmp/test.rac",
            citation="26 USC 21",
            agent_type="encoder",
            agent_model="opus",
            rac_content="test",
        )
        assert run.autorac_version == __version__

    def test_version_persisted_in_db(self, experiment_db, sample_encoding_run):
        """Test that autorac_version is persisted and retrieved from DB."""
        sample_encoding_run.autorac_version = "0.2.0"
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.autorac_version == "0.2.0"

    def test_version_defaults_empty_for_old_runs(
        self, experiment_db, sample_encoding_run
    ):
        """Test that runs without autorac_version default to empty string."""
        sample_encoding_run.autorac_version = ""
        experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(sample_encoding_run.id)
        assert retrieved.autorac_version == ""


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
        )

        session = experiment_db.get_session("token-test")
        # Session.total_tokens = input_tokens + output_tokens = 1500
        assert session.total_tokens == 1500

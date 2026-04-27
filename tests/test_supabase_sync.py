"""
Tests for supabase_sync module.

All external dependencies (supabase, sqlite DBs, env vars) are mocked.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode.supabase_sync import (
    fetch_runs_from_supabase,
    get_local_transcript_stats,
    get_supabase_client,
    sync_agent_sessions_to_supabase,
    sync_all_runs,
    sync_run_to_supabase,
    sync_transcripts_to_supabase,
)

# =========================================================================
# get_supabase_client
# =========================================================================


class TestGetSupabaseClient:
    def test_missing_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()

    def test_missing_key(self):
        with patch.dict(
            os.environ,
            {"AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()

    def test_with_secret_key(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_SECRET_KEY": "secret-key",
            },
            clear=True,
        ):
            with patch("axiom_encode.supabase_sync.create_client") as mock_create:
                mock_create.return_value = MagicMock()
                get_supabase_client()
                mock_create.assert_called_once_with(
                    "https://example.supabase.co", "secret-key"
                )

    def test_with_anon_key_fallback_for_reads(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_ANON_KEY": "anon-key",
            },
            clear=True,
        ):
            with patch("axiom_encode.supabase_sync.create_client") as mock_create:
                mock_create.return_value = MagicMock()
                get_supabase_client(require_write=False)
                mock_create.assert_called_once_with(
                    "https://example.supabase.co", "anon-key"
                )

    def test_write_client_requires_secret_key(self):
        with patch.dict(
            os.environ,
            {
                "AXIOM_ENCODE_SUPABASE_URL": "https://example.supabase.co",
                "AXIOM_ENCODE_SUPABASE_ANON_KEY": "anon-key",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Missing Supabase write credentials"):
                get_supabase_client()


# =========================================================================
# sync_run_to_supabase
# =========================================================================


class TestSyncRunToSupabase:
    def test_invalid_data_source(self):
        mock_run = MagicMock()
        with pytest.raises(ValueError, match="data_source must be one of"):
            sync_run_to_supabase(mock_run, "invalid_source")

    def test_valid_data_sources(self):
        for source in ["reviewer_agent", "ci_only", "mock", "manual_estimate"]:
            mock_run = MagicMock()
            mock_run.id = "test-123"
            mock_run.timestamp = datetime.now()
            mock_run.citation = "26 USC 1"
            mock_run.file_path = "test.yaml"
            mock_run.rulespec_content = ""
            mock_run.review_results = None

            mock_client = MagicMock()
            mock_client.table.return_value.upsert.return_value.execute.return_value = (
                MagicMock(data=[{"id": "test-123"}])
            )

            result = sync_run_to_supabase(mock_run, source, client=mock_client)
            assert result is True

    def test_with_review_results(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.rulespec_content = "format: rulespec/v1"
        mock_run.review_results = MagicMock(
            reviews=[
                MagicMock(
                    reviewer="rulespec_reviewer",
                    items_checked=10,
                    items_passed=8,
                    passed=True,
                ),
                MagicMock(
                    reviewer="formula_reviewer",
                    items_checked=10,
                    items_passed=7,
                    passed=True,
                ),
            ],
            policyengine_match=0.95,
            taxsim_match=0.90,
        )

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock(data=[{"id": "test-123"}])
        )

        result = sync_run_to_supabase(mock_run, "reviewer_agent", client=mock_client)
        assert result is True
        upsert_payload = mock_client.table.return_value.upsert.call_args.args[0]
        assert upsert_payload["rulespec_content"] == "format: rulespec/v1"
        assert upsert_payload["review_scores"]["rulespec_reviewer"] == 8.0
        assert upsert_payload["review_scores"]["formula_reviewer"] == 7.0

    def test_upsert_failure(self, capsys):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.side_effect = (
            Exception("Connection error")
        )

        result = sync_run_to_supabase(mock_run, "ci_only", client=mock_client)
        assert result is False

    def test_upsert_empty_result(self, capsys):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock(data=[])
        )

        result = sync_run_to_supabase(mock_run, "ci_only", client=mock_client)
        assert result is False

    def test_creates_client_if_not_provided(self):
        mock_run = MagicMock()
        mock_run.id = "test-123"
        mock_run.timestamp = datetime.now()
        mock_run.citation = "26 USC 1"
        mock_run.file_path = "test.yaml"
        mock_run.rulespec_content = ""
        mock_run.review_results = None

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock(data=[{"id": "test-123"}])
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ) as mock_get_client:
            result = sync_run_to_supabase(mock_run, "ci_only")
            assert result is True
            mock_get_client.assert_called_once_with()


# =========================================================================
# sync_all_runs
# =========================================================================


class TestSyncAllRuns:
    def test_sync_all_runs(self, tmp_path):
        from axiom_encode.harness.encoding_db import (
            EncodingDB,
            EncodingRun,
            Iteration,
        )

        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.yaml",
            iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
        )
        db.log_run(run)

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock(data=[{"id": run.id}])
        )

        stats = sync_all_runs(db_path, "ci_only", client=mock_client)
        assert stats["total"] >= 1
        assert stats["synced"] >= 1

    def test_sync_all_runs_with_failure(self, tmp_path):
        from axiom_encode.harness.encoding_db import (
            EncodingDB,
            EncodingRun,
            Iteration,
        )

        db_path = tmp_path / "test.db"
        db = EncodingDB(db_path)
        run = EncodingRun(
            citation="26 USC 1",
            file_path="test.yaml",
            iterations=[Iteration(attempt=1, duration_ms=1000, success=True)],
        )
        db.log_run(run)

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.side_effect = (
            Exception("Error")
        )

        stats = sync_all_runs(db_path, "ci_only", client=mock_client)
        assert stats["failed"] >= 1

    def test_creates_client_if_not_provided(self, tmp_path):
        from axiom_encode.harness.encoding_db import EncodingDB

        db_path = tmp_path / "test.db"
        EncodingDB(db_path)

        mock_client = MagicMock()
        mock_client.table.return_value.upsert.return_value.execute.return_value = (
            MagicMock(data=[])
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ):
            stats = sync_all_runs(db_path, "ci_only")
            assert stats["total"] == 0


# =========================================================================
# fetch_runs_from_supabase
# =========================================================================


class TestFetchRunsFromSupabase:
    def test_fetch_without_citation(self):
        mock_client = MagicMock()
        mock_result = MagicMock(data=[{"id": "1"}, {"id": "2"}])
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        results = fetch_runs_from_supabase(limit=20, client=mock_client)
        assert len(results) == 2

    def test_fetch_with_citation(self):
        mock_client = MagicMock()
        mock_result = MagicMock(data=[{"id": "1", "citation": "26 USC 1"}])
        mock_client.table.return_value.select.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value = mock_result

        results = fetch_runs_from_supabase(citation="26 USC 1", client=mock_client)
        assert len(results) == 1

    def test_creates_client_if_not_provided(self):
        mock_client = MagicMock()
        mock_client.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch(
            "axiom_encode.supabase_sync.get_supabase_client", return_value=mock_client
        ) as mock_get_client:
            results = fetch_runs_from_supabase()
            assert results == []
            mock_get_client.assert_called_once_with(require_write=False)


# =========================================================================
# sync_transcripts_to_supabase
# =========================================================================


class TestSyncTranscriptsToSupabase:
    def test_no_transcript_db(self):
        with patch.object(Path, "exists", return_value=False):
            result = sync_transcripts_to_supabase()
            assert result["total"] == 0
            assert "error" in result

    def test_sync_all_unsynced(self, tmp_path):
        # Create a temporary transcript DB
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["synced"] == 1
            assert result["total"] == 1

    def test_sync_with_session_filter(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', NULL, 5, '2024-01-01', NULL)"
        )
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (2, 'sess-2', 'tu-2', 'encoder', 'prompt', 'desc', 'summary', NULL, 3, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": 1}]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(
                session_id="sess-1", client=mock_client
            )
            assert result["synced"] == 1
            assert result["total"] == 1

    def test_sync_failure_during_upsert(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Connection error"
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["failed"] == 1

    def test_sync_empty_result(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'prompt', 'desc', 'summary', '[]', 5, '2024-01-01', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[]
        )

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = sync_transcripts_to_supabase(client=mock_client)
            assert result["failed"] == 1

    def test_creates_client_if_not_provided(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client",
                return_value=mock_client,
            ):
                result = sync_transcripts_to_supabase()
                assert result["total"] == 0


# =========================================================================
# sync_agent_sessions_to_supabase
# =========================================================================


class TestSyncAgentSessionsToSupabase:
    def test_no_experiments_db(self):
        with patch(
            "axiom_encode.supabase_sync.ENCODINGS_DB", Path("/nonexistent/path")
        ):
            result = sync_agent_sessions_to_supabase()
            assert result["total"] == 0

    def test_sync_session(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 2, 100, 50, 10, 0.01)"
        )
        conn.execute(
            "INSERT INTO session_events VALUES ('ev-1', 'agent-test-1', 1, '2024-01-01', 'tool_call', 'Read', 'content', '{}')"
        )
        conn.execute(
            "INSERT INTO session_events VALUES ('ev-2', 'agent-test-1', 2, '2024-01-01', 'tool_result', NULL, 'result', NULL)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(client=mock_client)
            assert result["synced"] == 1
            assert result["total"] == 1

    def test_sync_with_session_filter(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 0, 0, 0, 0, 0)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.return_value = MagicMock(
            data=[{"id": "test"}]
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(
                session_id="agent-test-1", client=mock_client
            )
            assert result["synced"] == 1

    def test_sync_failure(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.execute(
            "INSERT INTO sessions VALUES ('agent-test-1', '2024-01-01', '2024-01-01', 'opus', '/tmp', 0, 0, 0, 0, 0)"
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.schema.return_value.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "Error"
        )

        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            result = sync_agent_sessions_to_supabase(client=mock_client)
            assert result["failed"] == 1

    def test_creates_client_if_not_provided(self, tmp_path):
        db_path = tmp_path / "encodings.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                estimated_cost_usd REAL DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT
            )
        """)
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        with patch("axiom_encode.supabase_sync.ENCODINGS_DB", db_path):
            with patch(
                "axiom_encode.supabase_sync.get_supabase_client",
                return_value=mock_client,
            ):
                result = sync_agent_sessions_to_supabase()
                assert result["total"] == 0


# =========================================================================
# get_local_transcript_stats
# =========================================================================


class TestGetLocalTranscriptStats:
    def test_no_db(self):
        with patch(
            "axiom_encode.supabase_sync.TRANSCRIPT_DB", Path("/nonexistent/path")
        ):
            result = get_local_transcript_stats()
            assert result["exists"] is False

    def test_with_data(self, tmp_path):
        db_path = tmp_path / "transcripts.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE agent_transcripts (
                id INTEGER PRIMARY KEY,
                session_id TEXT,
                tool_use_id TEXT,
                subagent_type TEXT,
                prompt TEXT,
                description TEXT,
                response_summary TEXT,
                transcript TEXT,
                message_count INTEGER,
                created_at TEXT,
                uploaded_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (1, 'sess-1', 'tu-1', 'encoder', 'p', 'd', 's', '[]', 5, '2024-01-01', NULL)"
        )
        conn.execute(
            "INSERT INTO agent_transcripts VALUES (2, 'sess-1', 'tu-2', 'reviewer', 'p', 'd', 's', '[]', 3, '2024-01-01', '2024-01-02')"
        )
        conn.commit()
        conn.close()

        with patch("axiom_encode.supabase_sync.TRANSCRIPT_DB", db_path):
            result = get_local_transcript_stats()
            assert result["exists"] is True
            assert result["total"] == 2
            assert result["unsynced"] == 1
            assert result["synced"] == 1
            assert "encoder" in result["by_type"]
            assert "reviewer" in result["by_type"]

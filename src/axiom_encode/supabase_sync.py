"""
Supabase sync for axiom_encode encoding runs.

Syncs local SQLite encoding DB to Supabase for the public dashboard.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from supabase import Client, create_client

ENCODINGS_SCHEMA = "encodings"
TELEMETRY_SCHEMA = "telemetry"


def _review_results_to_scores(review_results) -> dict:
    """Convert checklist review results to the dashboard score shape."""
    scores = {
        "policyengine_match": review_results.policyengine_match,
        "taxsim_match": review_results.taxsim_match,
    }
    reviewer_keys = {
        "rulespec": "rulespec",
        "rulespec_reviewer": "rulespec",
        "formula": "formula",
        "formula_reviewer": "formula",
        "parameter": "parameter",
        "parameter_reviewer": "parameter",
        "integration": "integration",
        "integration_reviewer": "integration",
    }
    for review in review_results.reviews:
        key = reviewer_keys.get(review.reviewer)
        if not key:
            continue
        if review.items_checked:
            scores[key] = review.items_passed / review.items_checked * 10
        else:
            scores[key] = 10 if review.passed else 0
    return scores


def get_supabase_client(*, require_write: bool = True) -> Client:
    """Get Supabase client using environment variables."""
    url = os.environ.get("AXIOM_ENCODE_SUPABASE_URL")
    if require_write:
        key = os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY")
    else:
        key = os.environ.get("AXIOM_ENCODE_SUPABASE_SECRET_KEY") or os.environ.get(
            "AXIOM_ENCODE_SUPABASE_ANON_KEY"
        )

    if not url or not key:
        if require_write:
            raise ValueError(
                "Missing Supabase write credentials. Set AXIOM_ENCODE_SUPABASE_URL "
                "and AXIOM_ENCODE_SUPABASE_SECRET_KEY."
            )
        raise ValueError(
            "Missing Supabase credentials. Set AXIOM_ENCODE_SUPABASE_URL and "
            "AXIOM_ENCODE_SUPABASE_SECRET_KEY "
            "(or AXIOM_ENCODE_SUPABASE_ANON_KEY for read-only)."
        )

    return create_client(url, key)


def sync_run_to_supabase(
    run: "EncodingRun",
    data_source: str,  # REQUIRED: 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
    client: Optional[Client] = None,
) -> bool:
    """
    Sync a single encoding run to Supabase.

    Args:
        run: The EncodingRun to sync
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client (creates one if not provided)

    Returns:
        True if sync succeeded
    """

    # Validate data_source - this is a hard requirement to prevent fake data
    valid_sources = {"reviewer_agent", "ci_only", "mock", "manual_estimate"}
    if data_source not in valid_sources:
        raise ValueError(
            f"data_source must be one of {valid_sources}, got: {data_source}"
        )

    if client is None:
        client = get_supabase_client()

    # Convert to Supabase format
    data = {
        "id": run.id,
        "timestamp": run.timestamp.isoformat(),
        "citation": run.citation,
        "file_path": run.file_path,
        "complexity": {
            "cross_references": run.complexity.cross_references,
            "has_nested_structure": run.complexity.has_nested_structure,
            "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
            "has_phase_in_out": run.complexity.has_phase_in_out,
            "estimated_variables": run.complexity.estimated_variables,
            "estimated_parameters": run.complexity.estimated_parameters,
        },
        "iterations": [
            {
                "attempt": it.attempt,
                "duration_ms": it.duration_ms,
                "success": it.success,
                "errors": [
                    {
                        "error_type": e.error_type,
                        "message": e.message,
                        "variable": e.variable,
                        "fix_applied": e.fix_applied,
                    }
                    for e in it.errors
                ],
            }
            for it in run.iterations
        ],
        "total_duration_ms": run.total_duration_ms,
        "agent_type": run.agent_type,
        "agent_model": run.agent_model,
        "rulespec_content": run.rulespec_content,
        "session_id": run.session_id,
        "synced_at": datetime.now().isoformat(),
        "data_source": data_source,
    }

    if run.review_results:
        data["scores"] = _review_results_to_scores(run.review_results)

    try:
        # Upsert to handle both new and updated runs
        result = (
            client.schema(ENCODINGS_SCHEMA)
            .table("encoding_runs")
            .upsert(data)
            .execute()
        )
        return len(result.data) > 0
    except Exception as e:
        print(f"Error syncing run {run.id}: {e}")
        return False


def sync_all_runs(
    db_path: Path, data_source: str, client: Optional[Client] = None
) -> dict:
    """
    Sync all runs from local SQLite to Supabase.

    Args:
        db_path: Path to local encodings.db
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    from .harness.encoding_db import EncodingDB

    if client is None:
        client = get_supabase_client()

    db = EncodingDB(db_path)
    runs = db.get_recent_runs(limit=1000)  # Get all runs

    synced = 0
    failed = 0

    for run in runs:
        if sync_run_to_supabase(run, data_source, client):
            synced += 1
        else:
            failed += 1

    return {
        "total": len(runs),
        "synced": synced,
        "failed": failed,
    }


def fetch_runs_from_supabase(
    limit: int = 20,
    citation: Optional[str] = None,
    client: Optional[Client] = None,
) -> list[dict]:
    """
    Fetch encoding runs from Supabase.

    Args:
        limit: Maximum runs to fetch
        citation: Optional filter by citation
        client: Optional Supabase client

    Returns:
        List of run records
    """
    if client is None:
        client = get_supabase_client(require_write=False)

    query = client.schema(ENCODINGS_SCHEMA).table("encoding_runs").select("*")

    if citation:
        query = query.eq("citation", citation)

    query = query.order("timestamp", desc=True).limit(limit)

    result = query.execute()
    return result.data


# ============================================================================
# Transcript Sync (from PostToolUse hook local DB to Supabase)
# ============================================================================

TRANSCRIPT_DB = Path.home() / "TheAxiomFoundation" / "axiom-encode" / "transcripts.db"


def sync_transcripts_to_supabase(
    session_id: Optional[str] = None,
    client: Optional[Client] = None,
) -> dict:
    """
    Sync agent transcripts from local SQLite to Supabase.

    Args:
        session_id: Optional filter by session (syncs all unsynced if None)
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    import sqlite3

    if not TRANSCRIPT_DB.exists():
        return {"total": 0, "synced": 0, "failed": 0, "error": "No local transcript DB"}

    if client is None:
        client = get_supabase_client()

    conn = sqlite3.connect(str(TRANSCRIPT_DB))
    conn.row_factory = sqlite3.Row

    # Get unsynced transcripts
    query = "SELECT * FROM agent_transcripts WHERE uploaded_at IS NULL"
    params = []
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)

    rows = conn.execute(query, params).fetchall()

    synced = 0
    failed = 0

    for row in rows:
        try:
            data = {
                "session_id": row["session_id"],
                "tool_use_id": row["tool_use_id"],
                "subagent_type": row["subagent_type"],
                "prompt": row["prompt"],
                "description": row["description"],
                "response_summary": row["response_summary"],
                "transcript": json.loads(row["transcript"])
                if row["transcript"]
                else None,
                "message_count": row["message_count"],
                "created_at": row["created_at"],
            }

            result = (
                client.schema(TELEMETRY_SCHEMA)
                .table("agent_transcripts")
                .upsert(data)
                .execute()
            )

            if result.data:
                # Mark as uploaded
                conn.execute(
                    "UPDATE agent_transcripts SET uploaded_at = ? WHERE id = ?",
                    (datetime.now().isoformat(), row["id"]),
                )
                conn.commit()
                synced += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error syncing transcript {row['tool_use_id']}: {e}")
            failed += 1

    conn.close()

    return {
        "total": len(rows),
        "synced": synced,
        "failed": failed,
    }


ENCODINGS_DB = Path.home() / "TheAxiomFoundation" / "axiom-encode" / "encodings.db"


def sync_agent_sessions_to_supabase(
    session_id: Optional[str] = None,
    client: Optional[Client] = None,
) -> dict:
    """
    Sync agent sessions from encodings.db to Supabase.

    Args:
        session_id: Optional filter by session ID
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    import sqlite3

    if not ENCODINGS_DB.exists():
        return {"total": 0, "synced": 0, "failed": 0, "error": "No encodings.db"}

    if client is None:
        client = get_supabase_client()

    conn = sqlite3.connect(str(ENCODINGS_DB))
    conn.row_factory = sqlite3.Row

    # Get sessions created by agent-backed runs.
    query = "SELECT * FROM sessions WHERE id LIKE 'agent-%'"
    params = []
    if session_id:
        query = "SELECT * FROM sessions WHERE id = ?"
        params.append(session_id)

    sessions = conn.execute(query, params).fetchall()

    synced = 0
    failed = 0

    for session in sessions:
        try:
            # Get events for this session
            events = conn.execute(
                "SELECT * FROM session_events WHERE session_id = ? ORDER BY sequence",
                (session["id"],),
            ).fetchall()

            # Build session data
            session_data = {
                "id": session["id"],
                "started_at": session["started_at"],
                "ended_at": session["ended_at"],
                "model": session["model"],
                "cwd": session["cwd"],
                "event_count": session["event_count"],
                "input_tokens": session["input_tokens"],
                "output_tokens": session["output_tokens"],
                "cache_read_tokens": session["cache_read_tokens"],
                "estimated_cost_usd": float(session["estimated_cost_usd"] or 0),
            }

            # Build events data
            events_data = [
                {
                    "id": e["id"],
                    "session_id": e["session_id"],
                    "sequence": e["sequence"],
                    "timestamp": e["timestamp"],
                    "event_type": e["event_type"],
                    "tool_name": e["tool_name"],
                    "content": e["content"][:10000]
                    if e["content"]
                    else None,  # Truncate large content
                    "metadata": json.loads(e["metadata_json"])
                    if e["metadata_json"]
                    else None,
                }
                for e in events
            ]

            # Upsert session
            client.schema(TELEMETRY_SCHEMA).table("sdk_sessions").upsert(
                session_data
            ).execute()

            # Upsert events (in batches of 100)
            for i in range(0, len(events_data), 100):
                batch = events_data[i : i + 100]
                client.schema(TELEMETRY_SCHEMA).table("sdk_session_events").upsert(
                    batch
                ).execute()

            synced += 1
            print(f"  Synced session {session['id']} ({len(events)} events)")

        except Exception as e:
            print(f"  Error syncing session {session['id']}: {e}")
            failed += 1

    conn.close()

    return {
        "total": len(sessions),
        "synced": synced,
        "failed": failed,
    }


def get_local_transcript_stats() -> dict:
    """Get stats about local transcript database."""
    import sqlite3

    if not TRANSCRIPT_DB.exists():
        return {"exists": False}

    conn = sqlite3.connect(str(TRANSCRIPT_DB))

    total = conn.execute("SELECT COUNT(*) FROM agent_transcripts").fetchone()[0]
    unsynced = conn.execute(
        "SELECT COUNT(*) FROM agent_transcripts WHERE uploaded_at IS NULL"
    ).fetchone()[0]
    by_type = conn.execute(
        "SELECT subagent_type, COUNT(*) FROM agent_transcripts GROUP BY subagent_type"
    ).fetchall()

    conn.close()

    return {
        "exists": True,
        "total": total,
        "unsynced": unsynced,
        "synced": total - unsynced,
        "by_type": dict(by_type),
    }


if __name__ == "__main__":  # pragma: no cover
    # CLI usage: python -m axiom_encode.supabase_sync <db_path> <data_source>
    # data_source is REQUIRED to prevent syncing fake data
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m axiom_encode.supabase_sync runs <db_path> <data_source>")
        print("  python -m axiom_encode.supabase_sync transcripts [session_id]")
        print("  python -m axiom_encode.supabase_sync stats")
        print("")
        print("data_source for runs must be one of:")
        print("  reviewer_agent  - Scores from actual reviewer agent runs")
        print("  ci_only         - Only CI tests ran, no reviewer scores")
        print("  mock            - Fake/placeholder data for testing")
        print("  manual_estimate - Human-estimated scores (NOT from agents)")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "runs":
        if len(sys.argv) < 4:
            print(
                "Usage: python -m axiom_encode.supabase_sync runs <db_path> <data_source>"
            )
            sys.exit(1)
        db_path = Path(sys.argv[2])
        data_source = sys.argv[3]
        print(f"Syncing runs from {db_path} with data_source={data_source}...")
        stats = sync_all_runs(db_path, data_source)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )

    elif cmd == "transcripts":
        session_id = sys.argv[2] if len(sys.argv) > 2 else None
        print(
            f"Syncing transcripts{f' for session {session_id}' if session_id else ''}..."
        )
        stats = sync_transcripts_to_supabase(session_id)
        print(
            f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total"
        )

    elif cmd == "stats":
        stats = get_local_transcript_stats()
        if not stats.get("exists"):
            print("No local transcript database found")
        else:
            print(
                f"Local transcripts: {stats['total']} total, {stats['unsynced']} unsynced"
            )
            print(f"By agent type: {stats['by_type']}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)

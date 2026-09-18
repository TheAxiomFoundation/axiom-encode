"""Known-good artifacts from the local ``encodings.db`` run log (read-only).

This is the fallback source for the synthetic track: the encoder track's own
passing outputs on the pinned UK release are the primary source
(:mod:`.eval_suite`), but until those exist locally the calibration harness's
``apply_applied`` generations are the largest pool of artifacts that passed
compile, CI and apply. The database is opened with ``mode=ro`` and never
written.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from . import KnownGoodArtifact

GOOD_STATUS = "apply_applied"


def load_known_good(
    db_path: Path,
    *,
    generator_model: str = "gpt-5.5",
    citation_prefix: Optional[str] = None,
    max_artifact_chars: int = 12_000,
    status: str = GOOD_STATUS,
) -> tuple[list[KnownGoodArtifact], dict[str, Any]]:
    """Return known-good artifacts plus the source identity to record.

    Rows are returned in a stable order (timestamp, id) so a seeded builder
    is reproducible against the same database snapshot.
    """

    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"encodings database not found: {db_path}")
    uri = f"file:{db_path.resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        params: list[Any] = [generator_model, status, max_artifact_chars]
        where = [
            "source_text IS NOT NULL AND source_text != ''",
            "rulespec_content IS NOT NULL AND rulespec_content != ''",
            "agent_model = ?",
            "json_extract(outcome_json, '$.status') = ?",
            "length(rulespec_content) <= ?",
        ]
        if citation_prefix:
            where.append("citation LIKE ?")
            params.append(citation_prefix.replace("%", "") + "%")
        rows = conn.execute(
            "SELECT id, citation, timestamp, source_text, rulespec_content, "
            "agent_model, axiom_encode_version FROM encoding_runs WHERE "
            + " AND ".join(where)
            + " ORDER BY timestamp, id",
            params,
        ).fetchall()
    finally:
        conn.close()

    artifacts = [
        KnownGoodArtifact(
            key=str(row["id"]),
            citation=str(row["citation"] or ""),
            provision_text=str(row["source_text"]),
            artifact_text=str(row["rulespec_content"]),
            origin={
                "source": "encodings_db",
                "run_id": str(row["id"]),
                "timestamp": row["timestamp"],
                "generator_model": row["agent_model"],
                "axiom_encode_version": row["axiom_encode_version"] or None,
                "outcome_status": status,
            },
        )
        for row in rows
    ]
    identity = {
        "database": db_path.name,
        "generator_model": generator_model,
        "outcome_status": status,
        "citation_prefix": citation_prefix,
        "max_artifact_chars": max_artifact_chars,
        "row_count": len(artifacts),
    }
    return artifacts, identity

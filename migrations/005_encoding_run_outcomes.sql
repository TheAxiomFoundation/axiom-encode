-- Persist final encode/apply workflow status for dashboard and log review.

ALTER TABLE encodings.encoding_runs
    ADD COLUMN IF NOT EXISTS outcome JSONB NOT NULL DEFAULT '{}'::jsonb;

DROP FUNCTION IF EXISTS encodings.get_encoding_runs(INTEGER, INTEGER);

CREATE OR REPLACE FUNCTION encodings.get_encoding_runs(
    limit_count INTEGER DEFAULT 100,
    offset_count INTEGER DEFAULT 0
)
RETURNS TABLE (
    id TEXT,
    "timestamp" TIMESTAMPTZ,
    citation TEXT,
    iterations JSONB,
    outcome JSONB,
    scores JSONB,
    has_issues BOOLEAN,
    note TEXT,
    total_duration_ms INTEGER,
    agent_type TEXT,
    agent_model TEXT,
    data_source TEXT,
    session_id TEXT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = encodings
AS $$
    SELECT
        encoding_runs.id,
        encoding_runs.timestamp,
        encoding_runs.citation,
        encoding_runs.iterations,
        encoding_runs.outcome,
        COALESCE(encoding_runs.scores, encoding_runs.final_scores, '{}'::jsonb) AS scores,
        encoding_runs.has_issues,
        encoding_runs.note,
        encoding_runs.total_duration_ms,
        encoding_runs.agent_type,
        encoding_runs.agent_model,
        encoding_runs.data_source,
        encoding_runs.session_id
    FROM encodings.encoding_runs
    ORDER BY encoding_runs.timestamp DESC
    LIMIT GREATEST(1, LEAST(limit_count, 500))
    OFFSET GREATEST(0, offset_count);
$$;

GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO anon;
GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO authenticated;

-- RuleSpec encoding run summaries for the Axiom app.

CREATE SCHEMA IF NOT EXISTS encodings;
GRANT USAGE ON SCHEMA encodings TO postgres, service_role, anon, authenticated;

CREATE TABLE IF NOT EXISTS encodings.encoding_runs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    citation TEXT NOT NULL,
    file_path TEXT,
    complexity JSONB NOT NULL DEFAULT '{}'::jsonb,
    iterations JSONB NOT NULL DEFAULT '[]'::jsonb,
    outcome JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_duration_ms INTEGER,
    scores JSONB NOT NULL DEFAULT '{}'::jsonb,
    final_scores JSONB,
    has_issues BOOLEAN,
    note TEXT,
    agent_type TEXT,
    agent_model TEXT,
    rulespec_content TEXT,
    session_id TEXT,
    synced_at TIMESTAMPTZ,
    encoder_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_source TEXT NOT NULL DEFAULT 'unknown'
        CHECK (data_source IN ('reviewer_agent', 'ci_only', 'mock', 'manual_estimate', 'unknown'))
);

CREATE INDEX IF NOT EXISTS idx_encoding_runs_timestamp
    ON encodings.encoding_runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_encoding_runs_citation
    ON encodings.encoding_runs(citation);
CREATE INDEX IF NOT EXISTS idx_encoding_runs_file_path
    ON encodings.encoding_runs(file_path)
    WHERE file_path IS NOT NULL;

ALTER TABLE encodings.encoding_runs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS anon_read ON encodings.encoding_runs;
CREATE POLICY anon_read ON encodings.encoding_runs
    FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS authenticated_read ON encodings.encoding_runs;
CREATE POLICY authenticated_read ON encodings.encoding_runs
    FOR SELECT TO authenticated USING (true);

GRANT SELECT ON encodings.encoding_runs TO anon, authenticated;
GRANT ALL ON encodings.encoding_runs TO postgres, service_role;

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

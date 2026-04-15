-- Create encoding_runs table for the AutoRAC dashboard
-- Run this in Supabase SQL Editor: https://supabase.com/dashboard/project/nsupqhfchdtqclomlrgs/sql

CREATE TABLE IF NOT EXISTS encoding_runs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    citation TEXT NOT NULL,
    file_path TEXT,
    complexity JSONB DEFAULT '{}'::jsonb,
    iterations JSONB NOT NULL DEFAULT '[]'::jsonb,
    total_duration_ms INTEGER,
    predicted_scores JSONB,
    final_scores JSONB,
    agent_type TEXT,
    agent_model TEXT,
    rac_content TEXT,
    session_id TEXT,
    synced_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- REQUIRED: Prevents syncing fake/made-up data without explicit declaration
    -- Valid values: 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
    data_source TEXT NOT NULL DEFAULT 'unknown'
        CHECK (data_source IN ('reviewer_agent', 'ci_only', 'mock', 'manual_estimate', 'unknown'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_encoding_runs_timestamp ON encoding_runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_encoding_runs_citation ON encoding_runs(citation);

-- RLS
ALTER TABLE encoding_runs ENABLE ROW LEVEL SECURITY;

-- Base table remains private; public dashboard access should go through the
-- get_encoding_runs RPC below.
CREATE POLICY "Allow service write" ON encoding_runs FOR ALL TO service_role USING (true) WITH CHECK (true);

-- RPC function for frontend (matches what getEncodingRuns expects)
-- NOTE: data_source is REQUIRED to display appropriate warnings for non-reviewer_agent data
CREATE OR REPLACE FUNCTION get_encoding_runs(limit_count INTEGER DEFAULT 100, offset_count INTEGER DEFAULT 0)
RETURNS TABLE (
    "id" TEXT,
    "timestamp" TIMESTAMPTZ,
    "citation" TEXT,
    "iterations" JSONB,
    "scores" JSONB,
    "has_issues" BOOLEAN,
    "note" TEXT,
    "total_duration_ms" INTEGER,
    "agent_type" TEXT,
    "agent_model" TEXT,
    "data_source" TEXT
) LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    RETURN QUERY
    SELECT
        er.id,
        er.timestamp,
        er.citation,
        er.iterations,
        -- Map final_scores to scores format expected by frontend
        COALESCE(
            jsonb_build_object(
                'rac', (er.final_scores->>'rac_reviewer')::numeric,
                'formula', (er.final_scores->>'formula_reviewer')::numeric,
                'parameter', (er.final_scores->>'parameter_reviewer')::numeric,
                'integration', (er.final_scores->>'integration_reviewer')::numeric
            ),
            '{}'::jsonb
        ) as scores,
        -- Derive has_issues from iterations
        EXISTS(
            SELECT 1 FROM jsonb_array_elements(er.iterations) it
            WHERE jsonb_array_length(it->'errors') > 0
        ) as has_issues,
        NULL::TEXT as note,
        er.total_duration_ms,
        er.agent_type,
        er.agent_model,
        er.data_source
    FROM encoding_runs er
    ORDER BY er.timestamp DESC
    LIMIT limit_count
    OFFSET offset_count;
END;
$$;

GRANT EXECUTE ON FUNCTION get_encoding_runs(INTEGER, INTEGER) TO anon;
GRANT EXECUTE ON FUNCTION get_encoding_runs(INTEGER, INTEGER) TO authenticated;

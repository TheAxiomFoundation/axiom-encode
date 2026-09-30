-- GitHub Actions run identity on encoding runs: the workflow run that
-- produced each row, so the ops dashboard can join an encoding run to its
-- Actions run (and PR) exactly instead of by citation plus a time window.
--
-- Every column is nullable with no default: NULL means the run did not
-- execute inside GitHub Actions (local runs, apply-manifest reconstructions,
-- rows synced before this migration). The encoder ships these columns only
-- when it detects an Actions run, and retries without them when Supabase
-- reports an unknown column, so this migration can be applied by hand at any
-- time after the encoder change ships.
--
-- Live runs need no schema change: the same fields (plus github_workflow)
-- ride in encodings.live_encoding_runs.runner.

ALTER TABLE encodings.encoding_runs
    ADD COLUMN IF NOT EXISTS github_run_id TEXT,
    ADD COLUMN IF NOT EXISTS github_run_attempt INTEGER,
    ADD COLUMN IF NOT EXISTS github_run_url TEXT;

CREATE INDEX IF NOT EXISTS idx_encoding_runs_github_run_id
    ON encodings.encoding_runs(github_run_id)
    WHERE github_run_id IS NOT NULL;

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
    session_id TEXT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    cache_read_tokens BIGINT,
    cache_creation_tokens BIGINT,
    reasoning_output_tokens BIGINT,
    estimated_cost_usd NUMERIC,
    actual_cost_usd NUMERIC,
    generation_attempt_count INTEGER,
    github_run_id TEXT,
    github_run_attempt INTEGER,
    github_run_url TEXT
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
        encoding_runs.session_id,
        encoding_runs.input_tokens,
        encoding_runs.output_tokens,
        encoding_runs.cache_read_tokens,
        encoding_runs.cache_creation_tokens,
        encoding_runs.reasoning_output_tokens,
        encoding_runs.estimated_cost_usd,
        encoding_runs.actual_cost_usd,
        encoding_runs.generation_attempt_count,
        encoding_runs.github_run_id,
        encoding_runs.github_run_attempt,
        encoding_runs.github_run_url
    FROM encodings.encoding_runs
    ORDER BY encoding_runs.timestamp DESC
    LIMIT GREATEST(1, LEAST(limit_count, 500))
    OFFSET GREATEST(0, offset_count);
$$;

GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO anon;
GRANT EXECUTE ON FUNCTION encodings.get_encoding_runs(INTEGER, INTEGER) TO authenticated;

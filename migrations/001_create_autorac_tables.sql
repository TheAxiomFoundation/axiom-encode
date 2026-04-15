-- AutoRAC Supabase Schema
-- Run this in Supabase SQL editor to create the tables

-- Create autorac schema
CREATE SCHEMA IF NOT EXISTS autorac;

-- =============================================================================
-- Encoding Runs table - tracks each encoding attempt
-- =============================================================================
CREATE TABLE IF NOT EXISTS autorac.runs (
    id TEXT PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    citation TEXT NOT NULL,
    file_path TEXT,
    statute_text TEXT,

    -- Complexity factors (JSONB for flexibility)
    complexity JSONB DEFAULT '{}'::JSONB,

    -- Predicted scores (upfront estimates)
    predicted_scores JSONB,

    -- The encoding journey
    iterations JSONB DEFAULT '[]'::JSONB,
    total_duration_ms INTEGER DEFAULT 0,

    -- Final validation scores
    final_scores JSONB,

    -- Agent info
    agent_type TEXT DEFAULT 'encoder',
    agent_model TEXT DEFAULT 'claude-opus-4-5-20251101',

    -- RAC content (could be large)
    rac_content TEXT,

    -- Session linkage
    session_id TEXT,

    -- Sync metadata
    synced_at TIMESTAMPTZ DEFAULT NOW(),
    source_db TEXT DEFAULT 'local'
);

CREATE INDEX IF NOT EXISTS idx_runs_citation ON autorac.runs(citation);
CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON autorac.runs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_runs_agent ON autorac.runs(agent_type, agent_model);

-- =============================================================================
-- Sessions table - Claude Code session transcripts
-- =============================================================================
CREATE TABLE IF NOT EXISTS autorac.sessions (
    id TEXT PRIMARY KEY,
    run_id TEXT REFERENCES autorac.runs(id),
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    model TEXT,
    cwd TEXT,
    event_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_creation_tokens INTEGER DEFAULT 0,
    estimated_cost_usd NUMERIC(10, 4) DEFAULT 0,
    synced_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_run ON autorac.sessions(run_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON autorac.sessions(started_at DESC);

-- =============================================================================
-- Session Events table - individual events within sessions
-- =============================================================================
CREATE TABLE IF NOT EXISTS autorac.session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES autorac.sessions(id),
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL,
    tool_name TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}'::JSONB,
    synced_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_session ON autorac.session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON autorac.session_events(event_type);

-- =============================================================================
-- Artifact Versions table - SCD2 versioning of plugins, specs, etc.
-- =============================================================================
CREATE TABLE IF NOT EXISTS autorac.artifact_versions (
    id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version_label TEXT,
    content TEXT,
    effective_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    effective_to TIMESTAMPTZ,
    metadata JSONB DEFAULT '{}'::JSONB,
    synced_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_artifact_type ON autorac.artifact_versions(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifact_current ON autorac.artifact_versions(artifact_type, effective_to);
CREATE INDEX IF NOT EXISTS idx_artifact_hash ON autorac.artifact_versions(content_hash);

-- =============================================================================
-- Run-Artifact junction table
-- =============================================================================
CREATE TABLE IF NOT EXISTS autorac.run_artifacts (
    run_id TEXT NOT NULL REFERENCES autorac.runs(id),
    artifact_version_id TEXT NOT NULL REFERENCES autorac.artifact_versions(id),
    PRIMARY KEY (run_id, artifact_version_id)
);

-- =============================================================================
-- Views for easy querying
-- =============================================================================

-- Latest runs with summary
CREATE OR REPLACE VIEW autorac.runs_summary AS
SELECT
    r.id,
    r.timestamp,
    r.citation,
    r.file_path,
    r.agent_type,
    r.agent_model,
    r.total_duration_ms,
    jsonb_array_length(r.iterations) as iteration_count,
    (r.iterations->-1->>'success')::boolean as final_success,
    r.final_scores->>'rac_reviewer' as rac_score,
    r.final_scores->>'formula_reviewer' as formula_score,
    r.final_scores->>'parameter_reviewer' as parameter_score,
    r.final_scores->>'integration_reviewer' as integration_score
FROM autorac.runs r
ORDER BY r.timestamp DESC;

-- Calibration data (predicted vs actual)
CREATE OR REPLACE VIEW autorac.calibration_data AS
SELECT
    r.id,
    r.timestamp,
    r.citation,
    -- Predicted
    r.predicted_scores->>'rac' as predicted_rac,
    r.predicted_scores->>'formula' as predicted_formula,
    r.predicted_scores->>'param' as predicted_param,
    r.predicted_scores->>'integration' as predicted_integration,
    r.predicted_scores->>'iterations' as predicted_iterations,
    r.predicted_scores->>'confidence' as predicted_confidence,
    -- Actual
    r.final_scores->>'rac_reviewer' as actual_rac,
    r.final_scores->>'formula_reviewer' as actual_formula,
    r.final_scores->>'parameter_reviewer' as actual_param,
    r.final_scores->>'integration_reviewer' as actual_integration,
    jsonb_array_length(r.iterations) as actual_iterations
FROM autorac.runs r
WHERE r.predicted_scores IS NOT NULL
  AND r.final_scores IS NOT NULL
ORDER BY r.timestamp DESC;

-- =============================================================================
-- Row Level Security (RLS)
-- =============================================================================
-- Keep base tables private. Public access should go through narrow RPCs only.

ALTER TABLE autorac.runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE autorac.sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE autorac.session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE autorac.artifact_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE autorac.run_artifacts ENABLE ROW LEVEL SECURITY;

-- Write access requires service_role. Public dashboard access should use
-- security-definer views or RPCs that expose only the intended fields.
CREATE POLICY "Allow service write access to runs" ON autorac.runs
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow service write access to sessions" ON autorac.sessions
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow service write access to events" ON autorac.session_events
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow service write access to artifacts" ON autorac.artifact_versions
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow service write access to run_artifacts" ON autorac.run_artifacts
    FOR ALL TO service_role USING (true) WITH CHECK (true);

COMMENT ON SCHEMA autorac IS 'AutoRAC experiment tracking - encoding runs, sessions, calibration';
COMMENT ON TABLE autorac.runs IS 'Individual encoding runs with predictions, iterations, and final scores';
COMMENT ON TABLE autorac.sessions IS 'Claude Code session transcripts';
COMMENT ON TABLE autorac.session_events IS 'Individual events within sessions';
COMMENT ON TABLE autorac.artifact_versions IS 'SCD2 versioned artifacts (plugin, RAC spec, prompts)';

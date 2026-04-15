-- Create SDK orchestrator session tables for encoding pipeline logging
-- Run this in Supabase SQL Editor

-- SDK Sessions (one per encoding run)
CREATE TABLE IF NOT EXISTS rac.sdk_sessions (
    id TEXT PRIMARY KEY,  -- e.g., 'sdk-20260104-140232'
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    model TEXT,  -- e.g., 'claude-opus-4-5-20251101'
    cwd TEXT,    -- Working directory
    event_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    estimated_cost_usd NUMERIC(10,4) DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- SDK Session Events (multiple per session)
CREATE TABLE IF NOT EXISTS rac.sdk_session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES rac.sdk_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,  -- 'user', 'assistant', 'tool_use', 'tool_result'
    tool_name TEXT,            -- For tool events
    content TEXT,              -- Truncated content (max 10k chars)
    metadata JSONB,            -- Additional event metadata
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sdk_sessions_started ON rac.sdk_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sdk_session_events_session ON rac.sdk_session_events(session_id);
CREATE INDEX IF NOT EXISTS idx_sdk_session_events_sequence ON rac.sdk_session_events(session_id, sequence);

-- Enable RLS
ALTER TABLE rac.sdk_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rac.sdk_session_events ENABLE ROW LEVEL SECURITY;

-- Base tables remain private; public dashboard access should use the
-- rac.get_sdk_sessions RPC below.
CREATE POLICY "Allow service write sdk_sessions" ON rac.sdk_sessions FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY "Allow service write sdk_session_events" ON rac.sdk_session_events FOR ALL TO service_role USING (true) WITH CHECK (true);

-- RPC function to get sessions with summary
CREATE OR REPLACE FUNCTION rac.get_sdk_sessions(limit_count INTEGER DEFAULT 20)
RETURNS TABLE (
    id TEXT,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    model TEXT,
    event_count INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd NUMERIC
) LANGUAGE plpgsql SECURITY DEFINER AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.id,
        s.started_at,
        s.ended_at,
        s.model,
        s.event_count,
        s.input_tokens,
        s.output_tokens,
        s.estimated_cost_usd
    FROM rac.sdk_sessions s
    ORDER BY s.started_at DESC
    LIMIT limit_count;
END;
$$;

GRANT EXECUTE ON FUNCTION rac.get_sdk_sessions(INTEGER) TO anon;
GRANT EXECUTE ON FUNCTION rac.get_sdk_sessions(INTEGER) TO authenticated;

COMMENT ON TABLE rac.sdk_sessions IS 'SDK orchestrator encoding sessions';
COMMENT ON TABLE rac.sdk_session_events IS 'Events from SDK orchestrator sessions (tool calls, messages)';

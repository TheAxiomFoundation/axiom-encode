-- Encoder SDK sessions and event logs for the Axiom telemetry schema.

CREATE SCHEMA IF NOT EXISTS telemetry;
GRANT USAGE ON SCHEMA telemetry TO postgres, service_role, anon, authenticated;

CREATE TABLE IF NOT EXISTS telemetry.sdk_sessions (
    id TEXT PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    model TEXT,
    cwd TEXT,
    event_count INTEGER NOT NULL DEFAULT 0,
    input_tokens BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    cache_read_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC,
    encoder_version TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS telemetry.sdk_session_events (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES telemetry.sdk_sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    event_type TEXT NOT NULL,
    tool_name TEXT,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (session_id, sequence)
);

CREATE INDEX IF NOT EXISTS idx_sdk_sessions_started_at
    ON telemetry.sdk_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sdk_session_events_session_sequence
    ON telemetry.sdk_session_events(session_id, sequence);

ALTER TABLE telemetry.sdk_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE telemetry.sdk_session_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS anon_read ON telemetry.sdk_sessions;
CREATE POLICY anon_read ON telemetry.sdk_sessions
    FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS authenticated_read ON telemetry.sdk_sessions;
CREATE POLICY authenticated_read ON telemetry.sdk_sessions
    FOR SELECT TO authenticated USING (true);

DROP POLICY IF EXISTS anon_read ON telemetry.sdk_session_events;
CREATE POLICY anon_read ON telemetry.sdk_session_events
    FOR SELECT TO anon USING (true);

DROP POLICY IF EXISTS authenticated_read ON telemetry.sdk_session_events;
CREATE POLICY authenticated_read ON telemetry.sdk_session_events
    FOR SELECT TO authenticated USING (true);

GRANT SELECT ON telemetry.sdk_sessions TO anon, authenticated;
GRANT SELECT ON telemetry.sdk_session_events TO anon, authenticated;
GRANT ALL ON telemetry.sdk_sessions TO postgres, service_role;
GRANT ALL ON telemetry.sdk_session_events TO postgres, service_role;

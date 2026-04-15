-- Create agent_transcripts table for automatic subagent logging
-- This is populated by the PostToolUse hook when Task tool completes

CREATE TABLE IF NOT EXISTS autorac.agent_transcripts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL,
    tool_use_id TEXT NOT NULL,
    subagent_type TEXT NOT NULL,
    prompt TEXT,
    description TEXT,
    response_summary TEXT,
    transcript JSONB,  -- Full JSONL transcript as JSON array
    message_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Indexes for common queries
    CONSTRAINT unique_tool_use UNIQUE (tool_use_id)
);

-- Index for querying by session
CREATE INDEX IF NOT EXISTS idx_agent_transcripts_session
ON autorac.agent_transcripts(session_id);

-- Index for querying by agent type
CREATE INDEX IF NOT EXISTS idx_agent_transcripts_type
ON autorac.agent_transcripts(subagent_type);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_agent_transcripts_created
ON autorac.agent_transcripts(created_at DESC);

-- Enable RLS
ALTER TABLE autorac.agent_transcripts ENABLE ROW LEVEL SECURITY;

-- Subagent transcripts may contain prompts and tool output, so restrict the
-- base table to service_role writes only.
CREATE POLICY "Allow service access to agent_transcripts"
ON autorac.agent_transcripts
FOR ALL
TO service_role
USING (true)
WITH CHECK (true);

COMMENT ON TABLE autorac.agent_transcripts IS 'Automatic logging of subagent transcripts from Task tool calls';

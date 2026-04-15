-- Tighten previously permissive RLS policies on AutoRAC telemetry tables.
-- Apply this to existing Supabase deployments before exposing any public dashboard.

DO $$
BEGIN
    IF to_regclass('public.encoding_runs') IS NOT NULL THEN
        ALTER TABLE public.encoding_runs ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read" ON public.encoding_runs;
        DROP POLICY IF EXISTS "Allow service write" ON public.encoding_runs;
        DROP POLICY IF EXISTS "encoding_runs_read_all" ON public.encoding_runs;
        DROP POLICY IF EXISTS "encoding_runs_service_write" ON public.encoding_runs;
        CREATE POLICY "Allow service write" ON public.encoding_runs
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF to_regclass('autorac.agent_transcripts') IS NOT NULL THEN
        ALTER TABLE autorac.agent_transcripts ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow anonymous access to agent_transcripts" ON autorac.agent_transcripts;
        DROP POLICY IF EXISTS "Allow service access to agent_transcripts" ON autorac.agent_transcripts;
        CREATE POLICY "Allow service access to agent_transcripts" ON autorac.agent_transcripts
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('rac.agent_transcripts') IS NOT NULL THEN
        ALTER TABLE rac.agent_transcripts ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow anonymous access to agent_transcripts" ON rac.agent_transcripts;
        DROP POLICY IF EXISTS "Allow service access to agent_transcripts" ON rac.agent_transcripts;
        CREATE POLICY "Allow service access to agent_transcripts" ON rac.agent_transcripts
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF to_regclass('autorac.runs') IS NOT NULL THEN
        ALTER TABLE autorac.runs ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read access to runs" ON autorac.runs;
        DROP POLICY IF EXISTS "Allow service write access to runs" ON autorac.runs;
        CREATE POLICY "Allow service write access to runs" ON autorac.runs
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('autorac.sessions') IS NOT NULL THEN
        ALTER TABLE autorac.sessions ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read access to sessions" ON autorac.sessions;
        DROP POLICY IF EXISTS "Allow service write access to sessions" ON autorac.sessions;
        CREATE POLICY "Allow service write access to sessions" ON autorac.sessions
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('autorac.session_events') IS NOT NULL THEN
        ALTER TABLE autorac.session_events ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read access to events" ON autorac.session_events;
        DROP POLICY IF EXISTS "Allow service write access to events" ON autorac.session_events;
        CREATE POLICY "Allow service write access to events" ON autorac.session_events
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('autorac.artifact_versions') IS NOT NULL THEN
        ALTER TABLE autorac.artifact_versions ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read access to artifacts" ON autorac.artifact_versions;
        DROP POLICY IF EXISTS "Allow service write access to artifacts" ON autorac.artifact_versions;
        CREATE POLICY "Allow service write access to artifacts" ON autorac.artifact_versions
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('autorac.run_artifacts') IS NOT NULL THEN
        ALTER TABLE autorac.run_artifacts ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read access to run_artifacts" ON autorac.run_artifacts;
        DROP POLICY IF EXISTS "Allow service write access to run_artifacts" ON autorac.run_artifacts;
        CREATE POLICY "Allow service write access to run_artifacts" ON autorac.run_artifacts
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

DO $$
BEGIN
    IF to_regclass('rac.sdk_sessions') IS NOT NULL THEN
        ALTER TABLE rac.sdk_sessions ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read sdk_sessions" ON rac.sdk_sessions;
        DROP POLICY IF EXISTS "Allow service write sdk_sessions" ON rac.sdk_sessions;
        CREATE POLICY "Allow service write sdk_sessions" ON rac.sdk_sessions
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;

    IF to_regclass('rac.sdk_session_events') IS NOT NULL THEN
        ALTER TABLE rac.sdk_session_events ENABLE ROW LEVEL SECURITY;
        DROP POLICY IF EXISTS "Allow public read sdk_session_events" ON rac.sdk_session_events;
        DROP POLICY IF EXISTS "Allow service write sdk_session_events" ON rac.sdk_session_events;
        CREATE POLICY "Allow service write sdk_session_events" ON rac.sdk_session_events
            FOR ALL TO service_role
            USING (true)
            WITH CHECK (true);
    END IF;
END
$$;

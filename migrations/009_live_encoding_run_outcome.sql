-- Terminal outcome on live encode runs: why an `axiom-encode encode`
-- invocation ended, recorded on its encodings.live_encoding_runs row so a
-- run that stops before any encodings.encoding_runs row exists (an exception
-- or exit before the first model call) still has a recorded reason.
--
-- outcome_status: short machine token. The encode outcome's status on the
--   normal path (standalone_validated, apply_applied,
--   apply_blocked_validation, ...); `aborted` (an exception escaped),
--   `exited` (SystemExit) or `interrupted` (KeyboardInterrupt) otherwise.
-- error_rule: typed, bounded identifier of the cause, never a message or a
--   path: `exception:<ExceptionClassName>`, `exit:<int code>` (or
--   `exit:message`), or the check id of the final attempt's first typed
--   validation issue.
--
-- Both columns are nullable with no default: NULL means the run is still
-- running, finished before this migration, was written by an encoder that
-- predates these fields, or (error_rule) succeeded or had no typed cause.
-- The encoder sends the fields only in the finish update and retries without
-- them when Supabase reports an unknown column, so this migration can be
-- applied by hand at any time after the encoder change ships.
--
-- No function or view returns live_encoding_runs columns explicitly, so
-- nothing else needs re-creating (unlike 008 and get_encoding_runs).

ALTER TABLE encodings.live_encoding_runs
    ADD COLUMN IF NOT EXISTS outcome_status TEXT,
    ADD COLUMN IF NOT EXISTS error_rule TEXT;

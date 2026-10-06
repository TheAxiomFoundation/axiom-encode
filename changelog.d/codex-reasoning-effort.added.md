`encode --codex-reasoning-effort` selects Codex reasoning effort for generation
and retries, retaining `low` as the default. The Codex harness now uses the
`model_reasoning_effort` config key honored by Codex CLI 0.159.0 and records
the effective effort in its generation trace.

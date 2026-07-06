Default `axiom-encode encode` to the Codex backend with `gpt-5.5` and add a
pre-run auth check: when the resolved backend is `codex` and neither
`~/.codex/auth.json` (honoring `CODEX_HOME`) nor `OPENAI_API_KEY` is present,
the command now stops with an actionable error instead of failing deep inside
the Codex subprocess. Other backends stay available explicitly via
`--backend claude` or `--backend openai`. Reserves Claude/Fable capacity for
orchestration, gating, and review (encode#1054).

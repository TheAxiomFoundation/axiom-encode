Enforce the Codex backend default for `axiom-encode encode` with a pre-run
auth check. `encode` already defaulted to `--backend codex` with `gpt-5.5`;
now, when the resolved backend is `codex` and neither `~/.codex/auth.json`
(honoring `CODEX_HOME`) nor `OPENAI_API_KEY` is present, the command stops with
an actionable error instead of failing deep inside the Codex subprocess. Other
backends stay available explicitly via `--backend claude` or `--backend openai`.
Documents the default and the auth path, reserving Claude/Fable capacity for
orchestration, gating, and review (encode#1054).

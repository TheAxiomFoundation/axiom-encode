`encode --codex-reasoning-effort` selects Codex reasoning effort for generation
and retries, retaining `low` as the default. The Codex harness now passes the
effort as the `model_reasoning_effort` config key honored by Codex CLI 0.159.0
instead of `reasoning_effort`. This also changes existing Codex eval runs,
which passed the old key: the requested effort (default `low`) now reaches
Codex under the key it reads. The generation trace records the requested
effort; it does not measure the effort the model applied.

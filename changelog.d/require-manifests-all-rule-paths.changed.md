Make the encoder-first policy a structural property of the guard rather than a
convention. `programs/` (composed pilot pipelines) now joins `statutes/`,
`regulations/`, and `policies/` as a guarded rule-bearing root, so every
rule-bearing file needs an encoder apply-manifest or a manual attestation.
`sign-applied-files` gains `--all` (backfill an entire corpus that has no
manifests), `--roots`, and `--manual-exception`. Apply manifests now carry a
`supersedes` chain that embeds a compact, tamper-evident-but-self-attested
record (content hash, prior backend, run id, generation-prompt hash, signature)
of the manifest they overwrite, so an encoder-generated rule that is later
hand-repaired stays distinguishable from one hand-written from scratch. The
guard is fail-closed: a NEW rule file (a plain add OR the destination of a
rename/copy) is exempt only when a covering manifest records a recognized
machine-generator backend (codex/openai/claude, or a deterministic repair);
any other backend — `manual`, empty, absent, unknown, or a mis-cased /
space-padded name — requires `manual_exception: composition | repair |
fixtures | <issue-ref>`. A new `manifest-census` command reports per-repo
encoder-generated vs manual vs unmanifested coverage and can emit a Shields
badge JSON (encode#1053).

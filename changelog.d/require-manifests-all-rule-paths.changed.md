Make the encoder-first policy a structural property of the guard rather than a
convention. `programs/` (composed pilot pipelines) now joins `statutes/`,
`regulations/`, and `policies/` as a guarded rule-bearing root, so every
rule-bearing file needs an encoder apply-manifest or a manual attestation.
`sign-applied-files` gains `--all` (backfill an entire corpus that has no
manifests), `--roots`, and `--manual-exception`. Apply manifests now carry a
`supersedes` chain that embeds a compact record (content hash, prior backend,
run id, generation-prompt hash, signature) of the manifest they overwrite, so
an encoder-generated rule that is later hand-repaired stays distinguishable
from one hand-written from scratch. The guard rejects a `backend: manual`
manifest that introduces a NEW rule file unless it declares
`manual_exception: composition | repair | fixtures | <issue-ref>`. A new
`manifest-census` command reports per-repo encoder-generated vs manual vs
unmanifested coverage and can emit a Shields badge JSON (encode#1053).

# Issue 1557 progress

## State

- Branch: `feat/legacy-cleanup-receipts`
- Resumed on 2026-08-31 under the accepted one-transition `B -> H` contract; no
  checkout, reset, clean, or destructive recovery has been used.
- Starting implementation checkpoint: `9323860365d790c923ecc5fee8684be72fff56c8`.
- Exact locally cached protected base / `origin/main`: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21` (tree `436af4b6e279d1d4dcd9f64d83442ca1e4de7751`; live fetch is blocked by workspace DNS).
- Salvage ref `refs/codex-salvage/feat-legacy-cleanup-receipts-20260830-212800-65267`
  resolves to `f3303be312bec9dec2b6a0bf9a66d1941192a09b`; all five surviving
  worktree files are byte-identical to that commit.
- The rejected tracked CLI and targeted-reencode workflow edits were removed by
  reverse-applying their exact worktree diffs; the salvage ref remains available
  for archaeology and no reset or clean was used.
- The first implementation checkpoint is an isolated pure-contract layer:
  `legacy_cleanup.py`, `legacy_cleanup_git.py`, and focused tests. Signing and
  executed-validation modules are still being integrated and are not part of this
  checkpoint.

## Done

- Inspected branch, status, remotes, local/cached main identity, commit graph, staged diff, unstaged diff, and every untracked implementation file before editing.
- Confirmed the branch began with only the committed progress checkpoint beyond the protected base and no staged changes.
- Read `AGENTS.md`, `CLAUDE.md`, the committed progress checkpoint, all four surviving worktree changes, the WIP salvage commit, and the complete independent contract review at `20260830-193604-review-1557-contract/out.md`.
- Retried `git fetch origin main --prune`; it failed without changing refs because `github.com` could not be resolved.
- Retried a non-pruning live fetch on resume; DNS failed again. The browser-visible
  public repository page was reachable but did not expose an authoritative current
  main SHA, so `origin/main` remains a cached rather than live-verified base.
- Verified the dirty payload exactly matches the salvage tree before making any
  edits, and confirmed there are no staged changes or other untracked files.
- Classified the surviving topology against every Critical/High finding and Fast reject condition before implementation.
- Removed every surviving candidate-commit, receipt-child, applied-manifest-signing,
  and targeted-reencode cleanup change from the active worktree.
- Implemented the v1 cleanup-only receipt contract, semantic identity, strict and
  bounded canonical JSON, 1--64 exact primary/companion groups, cleanup signature
  payload domain, immutable `B` commit/tree/blob/mode/SHA-256 proof, protected
  toolchain/waiver pins, projected deletion-only tree, provenance inventory, and
  surviving-reference proof.
- Ran `UV_CACHE_DIR=/private/tmp/axiom-encode-uv-cache uv run pytest -q
  tests/test_legacy_cleanup.py`: 47 passed.

## Next

- Integrate cleanup-specific inner-domain signing and genuinely executed projected
  validation evidence, then wire the CLI to one receipt-first durable transaction
  that deletes every exact primary/companion preimage.
- Add cryptographic fail-closed verification for every immutable-base provenance
  record, separate guard authorization, exact deletion transport/staging, dedicated
  workflow, exclusions, docs, changelog/version updates, and the complete
  adversarial matrix.
- Run focused and broader validation, conduct the required independent review-fix cycle, and update this file after each coherent committed step.
- Retry fetch/push and open a draft PR linked to issue 1557 when GitHub connectivity is available; do not merge.

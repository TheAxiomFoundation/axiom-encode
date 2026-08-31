# Issue 1557 progress

## State

- Branch: `feat/legacy-cleanup-receipts`
- Starting implementation checkpoint: `9323860365d790c923ecc5fee8684be72fff56c8`.
- Exact locally cached protected base / `origin/main`: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21` (tree `436af4b6e279d1d4dcd9f64d83442ca1e4de7751`; live fetch is blocked by workspace DNS).
- Surviving uncommitted work is preserved in `.github/workflows/targeted-signed-reencode.yml`, `src/axiom_encode/cli.py`, `src/axiom_encode/legacy_cleanup.py`, and `tests/test_legacy_cleanup.py`.
- That work implements the rejected deletion-candidate plus receipt-only-child topology. Useful base-blob, canonical-path, strict-JSON, bounded-group, and transaction test material may be salvaged, but candidate-commit/receipt-child bindings, unchanged applied-manifest signing, and targeted-reencode workflow edits are not accepted contract.

## Done

- Inspected branch, status, remotes, local/cached main identity, commit graph, staged diff, unstaged diff, and every untracked implementation file before editing.
- Confirmed the branch began with only the committed progress checkpoint beyond the protected base and no staged changes.
- Read `AGENTS.md`, `CLAUDE.md`, the committed progress checkpoint, all four surviving worktree changes, the WIP salvage commit, and the complete independent contract review at `20260830-193604-review-1557-contract/out.md`.
- Retried `git fetch origin main --prune`; it failed without changing refs because `github.com` could not be resolved.
- Classified the surviving topology against every Critical/High finding and Fast reject condition before implementation.

## Next

- Map the existing apply transaction, signature broker, provenance inventories, generated-change guard, staging/publication, CI-parity, and immutable-reference call sites from the exact protected base.
- Replace the rejected topology with one base-to-head atomic contraction: receipt plus exact primary/companion deletions in one durable transaction, with semantic receipt identity and no containing-head/final-tree binding.
- Add fail-closed immutable-base ownership proof, typed cleanup signing/verification, separate guard authorization, exact deletion transport/staging, dedicated workflow, exclusions, docs, changelog/version updates, and the complete adversarial matrix.
- Run focused and broader validation, conduct the required independent review-fix cycle, and update this file after each coherent committed step.
- Retry fetch/push and open a draft PR linked to issue 1557 when GitHub connectivity is available; do not merge.

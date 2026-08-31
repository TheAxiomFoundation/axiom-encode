# Issue 1558 progress

## State

- Branch: `fix/1558-waiver-toolchain-transition`.
- Live-fetched `origin/main` and merge base: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21`.
- Resumed implementation: `f03d4f9b4a9fcfa1bbe7195c80fab6e15e5322c0`; resumed tracking head: `00de31256addce7c0b8f1d82a3b97b7779925cc5`; neither is approved.
- Salvage ref `refs/codex-salvage/fix-1558-waiver-toolchain-transition-20260830-212600-98091` resolves to `3ee0586841748ca3a57c92088b226d7f8f1799cc`.
- Three modified test files contain the salvage ref's exact 673-line adversarial patch; none were reset or cleaned.
- Pending creation is byte-bound, but pure pending-to-active consumption still accepts semantic evidence without requiring the protected-base/head waiver and toolchain snapshots.
- `git fetch --prune origin` succeeds, but GitHub API access and the GitNexus command registry are unavailable in this sandbox; retry issue/PR inspection, push, and draft-PR creation after local checks are green.

## Done

- Inspected repository instructions, branch/upstream state, remotes, commit metadata, complete modified-file list, full current patch, and untracked changelog fragment.
- Confirmed the surviving patch covers the toolchain digest-rebind helper, protected-base audit wiring, initial tests, README guidance, CI-parity dependency note, and changelog.
- Passed 435 focused tests across validation-waiver semantics, toolchain binding, stable evidence reads, and audit CLI integration.
- Read both independent review reports and reproduced their core blocker in the implementation: the consumption branch bypasses exact byte/toolchain binding.
- Re-read the assignment, repository `AGENTS.md`, generated repository context, commit history, live-fetched upstream, salvage commit, and exact worktree-to-salvage comparison on 2026-08-31.
- Read the complete surviving uncommitted test diff and retained the useful semantic-no-op, expiry, toolchain-formatting, evidence-mutation, and transition cases for reconciliation.
- Confirmed the creation contract must remain exactly one new pending field in the exact waiver/toolchain pair, with active state and corpus pins unchanged and both raw snapshots bound.

## Next

- Define and test the consumption protocol as the exact waiver/toolchain pair: changing waiver bytes necessarily requires the one digest substitution so the head toolchain binds the head waiver bytes.
- Make all protected-base evidence mandatory, validate creation and consumption from one stable snapshot, recheck every input path, and reject mixed/multiple/stale/raced transitions.
- Update the reusable workflow dependency/fixtures and documentation to the same contract, add real Git worktree integration tests, and preserve decrement behavior.
- Run focused and broad checks, complete independent review-fix cycles, verify commit messages and PR body, then refresh/push/open a draft PR linked to #1558 without merging.

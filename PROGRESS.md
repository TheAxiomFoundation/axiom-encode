# Issue 1558 progress

## State

- Branch: `fix/1558-waiver-toolchain-transition`.
- Starting commit and locally cached `origin/main`: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21`.
- Resumed at `f03d4f9b4a9fcfa1bbe7195c80fab6e15e5322c0`; the current commit is not approved.
- Three modified test files contain 673 preserved lines of additional adversarial coverage; none were reset or cleaned.
- Pending creation is byte-bound, but pure pending-to-active consumption still accepts semantic evidence without requiring the protected-base/head waiver and toolchain snapshots.
- GitHub network access and GitNexus's home-directory registry are unavailable in the current sandbox; retry remote comparison, push, and draft-PR creation after local checks are green.

## Done

- Inspected repository instructions, branch/upstream state, remotes, commit metadata, complete modified-file list, full current patch, and untracked changelog fragment.
- Confirmed the surviving patch covers the toolchain digest-rebind helper, protected-base audit wiring, initial tests, README guidance, CI-parity dependency note, and changelog.
- Passed 435 focused tests across validation-waiver semantics, toolchain binding, stable evidence reads, and audit CLI integration.
- Read both independent review reports and reproduced their core blocker in the implementation: the consumption branch bypasses exact byte/toolchain binding.
- Read the complete surviving uncommitted test diff and retained the useful semantic-no-op, expiry, toolchain-formatting, evidence-mutation, and transition cases for reconciliation.
- Confirmed the creation contract must remain exactly one new pending field in the exact waiver/toolchain pair, with active state and corpus pins unchanged and both raw snapshots bound.

## Next

- Define and test the consumption protocol as the exact waiver/toolchain pair: changing waiver bytes necessarily requires the one digest substitution so the head toolchain binds the head waiver bytes.
- Make all protected-base evidence mandatory, validate creation and consumption from one stable snapshot, recheck every input path, and reject mixed/multiple/stale/raced transitions.
- Update the reusable workflow dependency/fixtures and documentation to the same contract, add real Git worktree integration tests, and preserve decrement behavior.
- Run focused and broad checks, complete independent review-fix cycles, verify commit messages and PR body, then refresh/push/open a draft PR linked to #1558 without merging.

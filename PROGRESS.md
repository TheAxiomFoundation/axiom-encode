# Issue 1558 progress

## State

- Branch: `fix/1558-waiver-toolchain-transition`.
- Live-fetched `origin/main` and merge base: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21`.
- Resumed implementation: `f03d4f9b4a9fcfa1bbe7195c80fab6e15e5322c0`; resumed tracking head: `00de31256addce7c0b8f1d82a3b97b7779925cc5`; neither is approved.
- Salvage ref `refs/codex-salvage/fix-1558-waiver-toolchain-transition-20260830-212600-98091` resolves to `3ee0586841748ca3a57c92088b226d7f8f1799cc`.
- The salvage ref's exact 673-line adversarial test patch has been reconciled intact and committed as its own recovery checkpoint.
- Core transition classification and the audit CLI now apply one shared exact-byte/toolchain proof to both creation and consumption; guard integration and full generated-path closure remain in progress.
- Stable transition snapshots now carry exact filesystem identity as well as bytes, so a later same-byte path replacement cannot satisfy the recheck.
- Contract decision: consumption cannot be a waiver/toolchain-only pull request; it must be induced by the exact consumed module and rebind the exact waiver/toolchain pair. The reusable workflow's current `{waiver, toolchain}`-only activation draft is incompatible and must be corrected before pinning.
- `git fetch --prune origin` succeeds, but GitHub API access and the GitNexus command registry are unavailable in this sandbox; retry issue/PR inspection, push, and draft-PR creation after local checks are green.

## Done

- Inspected repository instructions, branch/upstream state, remotes, commit metadata, complete modified-file list, full current patch, and untracked changelog fragment.
- Confirmed the surviving patch covers the toolchain digest-rebind helper, protected-base audit wiring, initial tests, README guidance, CI-parity dependency note, and changelog.
- Passed 435 focused tests across validation-waiver semantics, toolchain binding, stable evidence reads, and audit CLI integration.
- Read both independent review reports and reproduced their core blocker in the implementation: the consumption branch bypasses exact byte/toolchain binding.
- Re-read the assignment, repository `AGENTS.md`, generated repository context, commit history, live-fetched upstream, salvage commit, and exact worktree-to-salvage comparison on 2026-08-31.
- Read the complete surviving uncommitted test diff and retained the useful semantic-no-op, expiry, toolchain-formatting, evidence-mutation, and transition cases for reconciliation.
- Passed 483 recovered focused tests across validation-waiver semantics, toolchain binding, stable evidence reads, and audit CLI integration on 2026-08-31.
- Added failing contract tests proving consumption must use exact base/head waiver and toolchain bytes, the consumed module plus waiver/toolchain path set, and exactly one unchanged-surroundings waiver entry (`3 failed, 42 passed` against the incomplete core).
- Implemented the shared consumption proof and restored the focused transition suite to `45 passed`.
- Added a reusable identity-bearing stable-file snapshot and a same-byte atomic-replacement regression test.
- Made the audit interface require protected-base toolchain evidence and explicit `nul-v1` changed paths, parse semantics from the five captured buffers, and recheck both bytes and filesystem identities.
- Passed `134` focused audit CLI, parallel-audit, and public command-plane tests after the mandatory evidence update.
- Derived the audit's corpus release identity from the captured head waiver/toolchain pair and threaded it through serial, parallel, and isolated execution, eliminating those downstream live evidence rereads (`182` focused tests passed).
- Generalized pending consumption to an exact caller-authenticated changed-path closure while requiring that closure to contain the waiver, toolchain, and consumed module; direct callers retain the strict three-path default (`94` focused core/toolchain tests passed).
- Confirmed the creation contract must remain exactly one new pending field in the exact waiver/toolchain pair, with active state and corpus pins unchanged and both raw snapshots bound.

## Next

- Extend the module-induced consumption scope from the core three-path proof to the exact manifest-authenticated generated-file closure used by `guard-generated`.
- Add real Git cross-worktree/race coverage for the five-file evidence boundary.
- Update the reusable workflow dependency/fixtures and documentation to the same contract, add real Git worktree integration tests, and preserve decrement behavior.
- Run focused and broad checks, complete independent review-fix cycles, verify commit messages and PR body, then refresh/push/open a draft PR linked to #1558 without merging.

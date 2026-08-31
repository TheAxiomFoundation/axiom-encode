# Issue 1558 progress

## State

- Branch: `fix/1558-waiver-toolchain-transition`.
- The branch is stacked directly on optional-inventory PR #1566 at exact head `410c81383826e9620ab969057631a9550d95e64b`, whose base is the live-fetched `origin/main` at `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21`; the core review diff therefore excludes the prerequisite's commits.
- Live-fetched `origin/main` and merge base: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21`.
- Resumed implementation: `f03d4f9b4a9fcfa1bbe7195c80fab6e15e5322c0`; resumed tracking head: `00de31256addce7c0b8f1d82a3b97b7779925cc5`; neither is approved.
- Salvage ref `refs/codex-salvage/fix-1558-waiver-toolchain-transition-20260830-212600-98091` resolves to `3ee0586841748ca3a57c92088b226d7f8f1799cc`.
- The salvage ref's exact 673-line adversarial test patch has been reconciled intact and committed as its own recovery checkpoint.
- Core transition classification, the audit CLI, and `guard-generated` now apply one shared exact-byte/toolchain proof to both creation and consumption. Pending consumption additionally requires the exact authenticated generated-file closure; an observed changed-path set is never accepted as authority.
- Stable transition snapshots now carry exact filesystem identity as well as bytes, so a later same-byte path replacement cannot satisfy the recheck.
- Contract decision: consumption cannot be a waiver/toolchain-only pull request; it must be induced by the exact consumed module and rebind the exact waiver/toolchain pair. The reusable workflow's current `{waiver, toolchain}`-only activation draft is incompatible and must be corrected before pinning.
- Both lawful consumption forms remain supported: a pending-only base may initialize active for a newly encoded module, and an active-plus-pending base may replace active. Both require the same authenticated module and signed-manifest generated closure.
- The inspected reusable-workflow sibling head `3e7976cc2aaab4e3e712285814e335493187a950` is not compatible: it loses NUL framing, omits protected-base toolchain evidence, accepts the wrong consumption shape, and lacks exact Git object-mode proof. No workflow pin or historical fixture is being changed until that dependency implements this contract.
- Both supported historical workflow pins explicitly declare immutable transition evidence unsupported. Local CI now runs the current library's stricter immutable-evidence compatibility check without claiming that the hosted historical workflows implement it.
- Live GitHub fetches on 2026-08-31 verified both the main and prerequisite heads. The core reserves encoder version `0.2.1752`, after #1566's `0.2.1751` and before the dependent cleanup's planned `0.2.1753`.

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
- Generalized pending consumption to an exact caller-authenticated changed-path closure while requiring that closure to contain the waiver, toolchain, and consumed module. Callers that omit authenticated closure evidence now fail closed.
- Confirmed the creation contract must remain exactly one new pending field in the exact waiver/toolchain pair, with active state and corpus pins unchanged and both raw snapshots bound.
- Froze Git refs in `guard-generated`, loaded protected-base waiver/toolchain evidence only from exact `100644 blob` objects, captured stable head evidence, derived the exact signed-manifest generated closure, and rechecked evidence identity and bytes before success.
- Made audit consumption require exactly one surviving changed signed v5 model manifest, verify its signature and base-waiver binding from captured bytes, require the consumed module and every manifest-listed applied file, reject deletions and unrelated paths, and retain that manifest in the final stability recheck.
- Hardened semantic-no-op rejection so an envelope that claims waiver and toolchain changes cannot pass merely because unrelated paths also changed.
- Added real Git repository and linked-worktree integration coverage for creation and consumption, including signed manifests, invalid signatures, stale or omitted modules, NUL-delimited adversarial paths, immutable base blobs, materialized evidence mutations, and same-byte atomic replacements across every protected evidence file.
- Passed `181` focused core/audit/Git integration tests and all `91` `guard-generated` tests, plus targeted Ruff, on 2026-08-31.
- Hardened local CI evidence materialization by freezing base and head commits, neutralizing ambient Git routing/configuration, requiring bounded exact `100644` base waiver/toolchain blobs, preserving raw changed paths as capped NUL-v1 bytes, and passing frozen commits to selection and guard execution.
- Kept both historical workflow fixtures unchanged and marked their immutable-transition capability false; the strict local invocation is labeled a compatibility check rather than hosted parity.
- Passed all `36` CI-parity tests, including real Git byte preservation, ref movement, adversarial newline paths, size caps, fail-closed materialization, non-0644/symlink/missing evidence, and ambient `GIT_DIR`/`GIT_WORK_TREE` redirection, plus targeted Ruff, on 2026-08-31.
- Updated the README, CI-parity guide, shared toolchain helper documentation, and issue changelog to one exact creation/consumption contract, including the signed v5 generated closure, byte/digest proof, NUL-v1 transport, stability rechecks, Git blob modes, and the blocked workflow dependency.
- Passed `84` focused toolchain and CI-parity tests, targeted Ruff, focused compileall, and `git diff --check` after the documentation alignment.
- The first broad-suite pass reached the intentional encoder-version provenance gate (`94 passed, 1 failed`) and required a synchronized version bump because this branch changes encoder-affecting files after `0.2.1750`.
- Bumped `pyproject.toml`, `src/axiom_encode/__init__.py`, and the root `uv.lock` package entry together to `0.2.1751`.
- Synchronized all nine exact version assertions used by packaged oracle/RuleSpec registry tests to `0.2.1751`; no oracle pin, mapping, or registry material changed.
- Passed `28` focused version-provenance and packaged-registry synchronization tests after the bump.
- Repository-wide Ruff, compileall, and `git diff --check` passed; both historical workflow fixture SHA-256 values remain unchanged.
- Reinspected the complete prior assignment, repository instructions, clean branch state, 17-commit history, last fetched base, salvage ref, progress ledger, and both independent review locations at resumed head `b4139370`.
- Confirmed the salvage ref `3ee05868` is a divergent recovery snapshot, not an ancestor of the current head; its useful 673-line test patch was reconciled separately in `c32b6f4a`, followed by the implementation and validation checkpoints.
- Fresh independent review reproduced the prior direct-audit blocker: an empty ledger or empty matrix partition can skip both protected-base and head waiver/toolchain binding when no module executes.
- Added four failing pre-partition regressions covering bad base/head bindings across empty-ledger and empty-partition audits (`4 failed` against the incomplete audit gate).
- Made exact protected-base and head waiver/toolchain binding unconditional before audit partitioning or execution, closing the empty-ledger and empty-partition bypass.
- Preserved both pending-only and active-plus-pending consumption, and extended the real linked-worktree, signed-manifest adversarial, materialized-evidence mutation, and same-byte replacement-race matrix across both forms.
- Passed `217` focused waiver/toolchain/audit/Git integration tests and all `91` `guard-generated` tests, plus targeted Ruff, after the final binding and state-machine correction.
- Rebased the complete core branch directly onto exact prerequisite head `410c81383826e9620ab969057631a9550d95e64b`, preserving a pre-stack salvage ref and an isolated core-only diff.
- Fixed the broad-suite non-Git excluded-program regression without weakening Git-backed validation: real repositories freeze HEAD before classification, while a genuine non-Git helper invocation requires two identical excluded-only observations after binding the corpus to waiver/toolchain bytes. Added a protected-path mutation regression; the guard matrices pass (`96 passed`).
- Synchronized `0.2.1752` across package metadata, lock metadata, and every exact packaged oracle/RuleSpec version assertion. The focused core (`217 passed`) and toolchain/CI-parity (`84 passed`) matrices remain green on the stacked tree.

## Next

- Complete the running packaged-version/RuleSpec matrix, reconcile the reusable workflow to the exact authenticated-closure contract, then complete independent exact-head review-fix cycles.
- Verify commit messages and PR body, refresh live upstream, push, and open a draft PR linked to #1558 only if every required check remains green; do not merge.

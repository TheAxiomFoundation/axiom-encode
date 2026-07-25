# Benchmark-validity defensive audit progress

## State

Audit started on 2026-07-25 in the requested writable worktree on
`fix/benchmark-validity-timeout-tmpdir`, based on `origin/main` at `f60fd29d`
(`axiom-encode` 0.2.1380). The starting worktree was clean. No production or
test changes have been made yet.

The audit has two bounded fixes:

1. Normalize ephemeral validator staging roots at issue-capture time without
   removing `compile_issues` or `ci_issues` from persisted-row comparison.
2. Replace the Claude encoder's hardcoded 600-second timeout with the existing
   environment-driven configuration convention while leaving Codex behavior
   unchanged.

## Done

- Read and adopted the repository instructions and required PR review-fix
  discipline.
- Verified the requested worktree, branch, base commit, and clean starting
  state.
- Read the GitNexus debugging workflow. The current worktree is not indexed;
  per that workflow, the next diagnostic step is a local analysis before
  querying the two failing execution paths.
- Confirmed that no final-report output path is configured in the environment.
  The final report will be written to
  `/private/tmp/benchmark-validity-final-report.md`.
- Established the test-first, implementation, version-bump, validation,
  independent-review, and PR plan.

## Next

- Analyze/query the repository graph and confirm both root causes in source.
- Add and run a failing regression test for defect 1, implement production
  normalization, rerun the focused tests, and commit the coherent fix.
- Repeat the red/green sequence and commit separately for defect 2.
- Commit version 0.2.1381 and changelog fragments, run the explicit version
  gate and required full check battery, complete an independent review-fix
  cycle, then push and open one unmerged PR against `main` referencing #1189.

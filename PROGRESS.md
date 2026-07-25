# Benchmark-validity defensive audit progress

## State

Audit started on 2026-07-25 in the requested writable worktree on
`fix/benchmark-validity-timeout-tmpdir`, based on `origin/main` at `f60fd29d`
(`axiom-encode` 0.2.1380). The starting worktree was clean. Defect 1 is
implemented and focused checks pass; defect 2 and the version batch remain.

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
  built a local graph and queried it directly after registry registration was
  blocked by the sandbox. The two relevant source files exceed GitNexus's
  indexing size limit, so direct source and test inspection was authoritative.
- Confirmed that no final-report output path is configured in the environment.
  The final report will be written to
  `/private/tmp/benchmark-validity-final-report.md`.
- Established the test-first, implementation, version-bump, validation,
  independent-review, and PR plan.
- Traced defect 1 from `_rulespec_validation_target`'s fresh overlay through
  validator compiler stderr capture, `EvalArtifactMetrics`, and exact
  reviewer-independent persisted-row revalidation. Confirmed production
  normalization is correct and no stop condition applies.
- Added the defect 1 regression first and observed the expected failure: the
  compile issue retained a fresh `/private/var/.../tmp...` root instead of the
  stable placeholder.
- Implemented production normalization in `ValidatorPipeline`. Generated eval
  validation supplies only the exact resolved overlay parent; compile/CI issue,
  error, and raw-output text replace that prefix with
  `<rulespec-validation-root>` while retaining the checkout, jurisdiction,
  source-relative path, and all other evidence.
- Added explicit tamper checks proving revalidation still rejects drift in
  either `compile_issues` or `ci_issues`.
- Reran the focused defect 1 tests: 3 passed. Reran adjacent overlay and
  persisted-revalidation tests: 16 passed. Focused Ruff check passed; Ruff
  formatting was applied and will be rechecked before commit.

## Next

- Recheck defect 1 formatting/lint/tests and commit the coherent first fix.
- Add and run the failing Claude timeout/trace regressions, implement the
  configurable structured timeout behavior, rerun focused checks, and commit
  defect 2 separately.
- Commit version 0.2.1381 and changelog fragments, run the explicit version
  gate and required full check battery, complete an independent review-fix
  cycle, then push and open one unmerged PR against `main` referencing #1189.

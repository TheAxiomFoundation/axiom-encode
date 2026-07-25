# Benchmark-validity defensive audit progress

## State

Audit started on 2026-07-25 in the requested writable worktree on
`fix/benchmark-validity-timeout-tmpdir`, based on `origin/main` at `f60fd29d`
(`axiom-encode` 0.2.1380). The starting worktree was clean. Defect 1 is
committed. Defect 2 is committed. The 0.2.1381 version/changelog batch is
committed, and its explicit provenance gate passes. The required checks and
independent review-fix cycle are complete with no remaining actionable
findings; PR publication remains.

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
  formatting and compilation checks passed.
- Committed defect 1 separately as `75496c29`.
- Traced defect 2 from the Claude backend's hardcoded `timeout=600` through the
  uncaught `TimeoutExpired` path. Confirmed that it escaped before an
  `EvalPromptResponse`, so no timeout trace or artifact evidence could be
  recorded.
- Added the Claude timeout regressions first and observed all six expected
  failures: overrides were ignored, unset/invalid values stayed at 600
  seconds, and `TimeoutExpired` escaped.
- Replaced the Claude-only cutoff with
  `AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS`, using the existing positive-integer
  environment parser. The default is 1,800 seconds: about 8.5 times the
  measured 211-second median and equal to the existing Codex long-source wall
  ceiling, retaining a bounded hung-process guard without systematically
  censoring the observed long tail.
- Claude traces now record `timed_out`, `timeout_reason`, and the effective
  `timeout_seconds` on success and error. Wall expiration returns a structured
  `Claude eval timed out` response so existing retry/persistence behavior can
  retain infrastructure evidence.
- Added an isolation test proving the Claude encoder env var does not change
  Codex's existing timeout selection. The adjacent Claude/Codex timeout slice
  passes: 13 tests. Focused Ruff check, format check, and compilation pass.
- Committed defect 2 separately as `0e2fce58`.
- Updated `pyproject.toml`, `src/axiom_encode/__init__.py`, and the
  `axiom-encode` package entry in `uv.lock` to 0.2.1381.
- Added separate `#1189` fixed fragments for location-independent validator
  evidence and the configurable Claude encoder timeout.
- Committed the 0.2.1381 version/changelog batch; the explicit nonstandard
  provenance gate passes after the final ledger amendment (1 passed, 1,018
  deselected).
- Passed the required Ruff lint check, Ruff format check, and Python compileall
  check.
- Ran the exact required test battery. It reached 2,797 passed and 1 skipped;
  its sole failure was the pre-existing macOS path-alias test attempting to
  create a temporary directory directly under sandbox-denied `/var/tmp`.
  Rerunning the same battery with only that environment-incompatible test
  deselected passed: 2,797 passed, 1 skipped, 1 deselected.
- Completed independent reviews of both implementation commits. Reviewers
  found no actionable correctness, coverage, cleanup, trace-persistence, or
  Codex-isolation issues. The release review found this progress ledger was
  stale after the version commit; this update resolves that process finding.
- Completed the release re-review after correcting the ledger. The reviewer
  independently reran the version gate, confirmed the finding is resolved, and
  approved current HEAD with no remaining actionable findings.

## Next

- Push and open one unmerged PR against `main` referencing #1189, inspect CI,
  and write the final report to the recorded output path.

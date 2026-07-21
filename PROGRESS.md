# Receipt.sign adoption progress

## State

Implementation and validation are complete within the binding allowed-file
scope. Handoff is blocked on one repository provenance requirement: the changed
encoder paths must be behind a version bump, but a valid bump must also change
`src/axiom_encode/__init__.py`, which this task explicitly forbids touching.
Independent reviewers reproduced the conflict and found no other correctness
issue. No configured final-report output path exists, so the report will be
written outside the repository under `/private/tmp` at handoff.

## Done

- Confirmed the worktree and branch match the requested starting point.
- Read the repository instructions and the GitNexus exploration/impact workflows.
- Established the implementation, test, review, and reporting plan.
- Traced all three production callers and every existing issue-string branch.
- Read `signing_broker.py`, the relevant unchanged tests and fixtures, and the
  full `receipt.sign` source at the clean local `v0.2.0` tag.
- Ran the nine-test focused behavior baseline: 9 passed.
- Replaced the final direct Ed25519 verification with receipt's 1-of-1 keyring;
  signing and every preceding envelope/trust-root check remain untouched.
- Refreshed `uv.lock` with receipt 0.2.0 using uv's offline cached resolver;
  the wheel and sdist SHA-256 hashes exactly match the task's known-good values.
- Installed the resolved wheel without dependencies in an isolated temporary
  environment and read its complete 459-line `receipt.sign` source. Confirmed
  the installed version is 0.2.0 and its `verify_threshold` API matches the
  implementation (with no 0.3.0-only `allow_legacy` parameter).
- Completed read-only implementation, packaging, and test-design audits; no
  correctness defect was found in the interrupted shim.
- Added seven dedicated receipt-adoption tests covering the real signing path,
  four cryptographic refusal cases, exact receipt delegation inputs, and every
  unchanged envelope/trust-root issue class and relevant check ordering.
- Added the `ops#3` changelog fragment without claiming a trust-model change.
- Ran the new test file against the installed receipt 0.2.0 wheel: 7 passed.
- Ran the unchanged focused behavior oracle together with the new tests:
  16 passed.
- Ran repository lint, formatting, and compilation checks: Ruff check passed,
  Ruff format check passed after applying its one-line domain-expression fix,
  and `compileall` passed.
- Ran the broadest locally runnable suite. After correcting an isolated test
  environment that initially imported the parent checkout and rerunning two
  temp-root-sensitive tests in the normal system temp directory, the runnable
  tests total 4,427 passed and 31 skipped. Eleven host/provenance nodes cannot
  pass locally: one task-caused version guard, one set-id semantics check, eight
  root-owned Homebrew Git/provisioning checks, and one sandboxed `/var/tmp`
  write check.
- Reproduced the task-caused guard independently:
  `test_current_encoder_affecting_changes_are_behind_version_bump` reports that
  `pyproject.toml`, `src/axiom_encode/cli.py`, and `uv.lock` changed after
  `fc04012d30d0`. The guard itself requires matching versions in
  `pyproject.toml`, `src/axiom_encode/__init__.py`, and `uv.lock`.
- Cross-checked the locked receipt wheel and sdist hashes again against the
  supplied known-good values; both match exactly.
- Completed independent code, packaging, and test reviews. All reviewers found
  the receipt mechanics, domain bytes, issue behavior, tests, hashes, scope,
  and changelog correct. The packaging review's only actionable formatting
  finding was fixed and rechecked. The remaining version conflict was confirmed
  by two reviewers as a blocker requiring a scope decision.
- Reran the changed-path focused set after the review fix: 10 passed, with Ruff
  check, Ruff format check, `compileall`, hash assertions, and `git diff --check`
  all passing.

## Next

- Obtain explicit maintainer/user permission to add the required coordinated
  version bump in `pyproject.toml`, `src/axiom_encode/__init__.py`, and `uv.lock`,
  then rerun the provenance guard and the full suite. Without that scope
  exception, retain this documented blocker and do not mark the branch ready or
  merge it.

# Receipt.sign adoption progress

## State

In progress. The verify-side shim, exact dependency pin, dedicated equivalence
tests, and changelog fragment are implemented. Focused and full validation plus
the required independent review-fix cycle remain. No configured final-report
output path exists, so the report will be written outside the repository under
`/private/tmp` at handoff.

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

## Next

- Rerun the unchanged focused behavior oracle and the repository's full local
  checks, then complete the independent review-fix cycle.

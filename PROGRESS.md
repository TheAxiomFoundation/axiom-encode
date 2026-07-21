# Receipt.sign adoption progress

## State

In progress. The verify-side shim and exact dependency pin are implemented, and
the real lock refresh records both known-good PyPI artifact hashes. Dedicated
mechanical-equivalence tests and the changelog fragment are next. No configured
final-report output path exists, so the report will be written outside the
repository under `/private/tmp` at handoff.

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

## Next

- Add the dedicated receipt-adoption equivalence tests and changelog fragment.
- Rerun focused and broad checks, then complete the independent review-fix cycle.

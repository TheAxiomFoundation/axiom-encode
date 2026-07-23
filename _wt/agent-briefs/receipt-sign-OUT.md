Implemented and committed the receipt verification shim. Worktree is clean at `a63f869c`; nothing was pushed and no PR was opened.

The detailed [final report](/private/tmp/receipt-sign-adoption-final-report.md) and committed [PROGRESS.md](/Users/maxghenis/TheAxiomFoundation/axiom-encode/_wt/receipt-sign-adoption/PROGRESS.md) contain the full audit.

### Validation

- Ruff lint and format: passed
- `compileall`: passed
- Receipt-focused tests: 7 passed
- Focused unchanged/new oracle set: 16 passed
- Independent follow-up set: 17 passed
- Locked wheel and sdist hashes: exact matches
- Three-party review-fix-repeat cycle: no actionable implementation findings

Full runnable-suite tail:

```text
4427 passed, 31 skipped, 11 deselected, 12 warnings in 79.00s
```

The 31 skips were optional pandas/numpy, unavailable sibling/runtime resources, sandbox socket support, signing-CI fixture, and filesystem set-id support. The 11 deselections were the version guard plus local root-owned Git, provisioning, set-id, and `/var/tmp` host constraints; exact nodes and reasons are in the report.

### Self-audit

- Every envelope, algorithm, trust-root, key-ID, value-type, base64, length, and ordering check remains unchanged and is pinned by exact-string tests.
- Receipt receives the same unsigned bytes and domain as `canonical_signing_message("apply_ed25519", payload)`.
- The broker-delivered root remains authoritative; the receipt fingerprint is derived from it at call time.
- Flipped signature, payload, wrong key, and wrong domain all produce exactly `has an invalid encoder apply manifest signature`.
- `SignError` is contained as that existing issue; other programming faults remain fail-hard.
- Missing receipt is intentionally an eager `ImportError`.
- Successful verification still returns `None`; callers are unchanged.

### Remaining blocker

The repository provenance test fails because `pyproject.toml`, `cli.py`, and `uv.lock` changed after `fc04012d`. A valid version bump also requires `src/axiom_encode/__init__.py`, which the brief forbids touching.

Clearing this requires explicit permission for that coordinated three-file version bump. Until then, the branch should not be marked ready or merged.
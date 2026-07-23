# APPROVE

No actionable findings.

- Merge resolution preserves `origin/main`; only intended version/dependency/lock deltas remain. Receipt hashes match exactly.
- Delegation, domain, issue ordering, `SignError` mapping, and `allow_legacy=False` are correct. Protected scopes and proof-excerpt work are untouched.
- Exact `uv run` commands were blocked by the read-only uv cache. Direct fallbacks had zero assertion failures: 7/7 adoption tests, 3 runnable provenance tests, and 690 runnable focused tests; all remaining cases required unavailable temporary-file writes.

Approved for squash once GitHub CI is green.
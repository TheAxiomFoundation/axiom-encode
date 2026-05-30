# Axiom Encode Agent Instructions

## PR Discipline

- Every PR to `TheAxiomFoundation/axiom-encode` must go through `/cycle` or an equivalent review-fix loop before it is marked ready for review, and again before merge if substantive changes were made after review. Draft PRs may be opened before the cycle when needed to get CI or review context.
- A valid cycle means: request an independent review, fix actionable findings, rerun the relevant focused tests/checks, and repeat until there are no actionable findings or a documented blocker.
- A blocker permits stopping the cycle, but it does not permit merging with unresolved actionable findings unless the PR records an explicit user or maintainer override.
- Apply this even to prompt, oracle, harness, and validation changes. For docs-only changes, run a lightweight cycle and record the cycle outcome and checks in the PR description or final handoff comment.
- Do not merge encoder PRs while CI is red, stale, or still pending unless the user explicitly overrides that for an emergency.

## Local Checks

```bash
uv run ruff check pyproject.toml src/axiom_encode scripts tests
python -m compileall -q src/axiom_encode scripts
uv run pytest
```

Use focused tests for small changes, then broaden checks when the change touches shared encoder behavior.

# Axiom Encode Agent Instructions

## Workflow Scope

- Use this repository's docs, CLI commands, RuleSpec specs, oracle registry, eval suites, and validation pipeline as the operating procedure for Axiom work.
- Name generated concepts per the applicable section of the naming contract, `axiom-rules-engine/docs/concept-naming.md` (executable rules, relations, tables/selectors, acronyms). Nothing in `encode --apply` updates the display acronym registries: when a change introduces a new acronym token, the PR author or reviewer flags it per the contract's Acronyms section and closes the flag by updating both registries or linking an owned follow-up.
- Do not use PolicyEngine workflow skills or PolicyEngine implementation skills for Axiom encoding, RuleSpec, corpus, or oracle-parity tasks. PolicyEngine is an oracle/comparison dependency here, not the workflow owner.
- If a reusable workflow is missing, add or propose an Axiom-specific Codex skill or project instruction instead of borrowing a PolicyEngine skill.

## PR Discipline

- Every PR to `TheAxiomFoundation/axiom-encode` must go through `/cycle` or an equivalent review-fix loop before it is marked ready for review, and again before merge if substantive changes were made after review. Draft PRs may be opened before the cycle when needed to get CI or review context.
- A valid cycle means: request an independent review, fix actionable findings, rerun the relevant focused tests/checks, and repeat until there are no actionable findings or a documented blocker.
- A blocker permits stopping the cycle, but it does not permit merging with unresolved actionable findings unless the PR records an explicit user or maintainer override.
- Apply this even to prompt, oracle, harness, and validation changes. For docs-only changes, run a lightweight cycle and record the cycle outcome and checks in the PR description or final handoff comment.
- Do not merge encoder PRs while CI is red, stale, or still pending unless the user explicitly overrides that for an emergency.

## PR Linkage

Every PR body to `TheAxiomFoundation/axiom-encode` carries two lines, which the `PR linkage` check (`src/axiom_encode/pr_linkage.py`) enforces:

```text
Encoding-Target: us/regulation/7/273/7
Fixes-Run: 42409703021
```

- When you open a PR that fixes a failing encode, fill both lines from the run you are fixing. `Encoding-Target` is the run's corpus citation path (the `citation` input; the last word of the run name), not a `.yaml` path. `Fixes-Run` is the `targeted-signed-reencode.yml` run id (or its `actions/runs` URL); for a local encode, write `local:<encoding_runs id>` (the `run_id` the encoder logs).
- List every encoding and run the PR fixes, comma-separated. CI checks that each signed run exists, is a `targeted-signed-reencode.yml` `workflow_dispatch` run on `main`, and that its citation is listed in `Encoding-Target`. That is the run's primary citation (the last word of the run name); list it also when the fix is for a dependent the run repaired, and list the dependent too.
- For generic work that fixes no particular encoding or run, write `Encoding-Target: none` and `Fixes-Run: none`. Do not omit the lines and do not guess a target.
- PRs opened by the encoder version-bump App are exempt.

## Local Checks

```bash
uv run ruff check pyproject.toml src/axiom_encode scripts tests
python -m compileall -q src/axiom_encode scripts
uv run pytest
```

Use focused tests for small changes, then broaden checks when the change touches shared encoder behavior.

Tests that execute the real axiom-rules-engine take the `real_engine` fixture (`tests/real_engine.py`) and skip without an engine. CI's `real-engine` job builds the commit pinned in `.github/workflows/ci.yml` (`AXIOM_RULES_ENGINE_REF`) with `axiom-encode engine-bind`, which receipts the binary's bytes, and runs them with `AXIOM_REQUIRE_REAL_ENGINE=1`, which fails rather than skips them. To run them locally against that pin, from a checkout of it:

```bash
uv run axiom-encode engine-bind --axiom-rules-engine-path <engine checkout> --pin <pin>
AXIOM_RULES_ENGINE_ROOT=<engine checkout> AXIOM_RULES_ENGINE_REF=<pin> AXIOM_REQUIRE_REAL_ENGINE=1 uv run pytest -m real_engine
```

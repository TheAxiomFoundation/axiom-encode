# Axiom Encode

AI-assisted encoding infrastructure for Axiom RuleSpec YAML.

## Quick Start

```bash
uv run axiom-encode encode "26 USC 21"
uv run axiom-encode validate path/to/rules.yaml
uv run axiom-encode eval-suite benchmarks/us_snap_eligibility_refresh.yaml
```

## Active Surface

- Package: `axiom-encode`
- Python module: `axiom_encode`
- CLI binary: `axiom-encode`
- Rule runtime repo: `axiom-rules-engine`
- Jurisdiction repos: canonical policy repositories such as `rulespec-us`, `rulespec-us-tn`, and `rulespec-us-nc`

The encoder emits RuleSpec YAML artifacts. Source documents are expected to live outside Git, with jurisdiction repositories keeping source registries and manifests that point to durable storage.

## PR Discipline

- Every PR to `TheAxiomFoundation/axiom-encode` must go through `/cycle` or an equivalent review-fix loop before it is marked ready for review, and again before merge if substantive changes were made after review. Draft PRs may be opened before the cycle when needed to get CI or review context.
- A valid cycle means: request an independent review, fix actionable findings, rerun the relevant focused tests/checks, and repeat until there are no actionable findings or a documented blocker.
- A blocker permits stopping the cycle, but it does not permit merging with unresolved actionable findings unless the PR records an explicit user or maintainer override.
- Apply this even to prompt, oracle, harness, and validation changes. For docs-only changes, run a lightweight cycle and record the cycle outcome and checks in the PR description or final handoff comment.
- Do not merge encoder PRs while CI is red, stale, or still pending unless the user explicitly overrides that for an emergency.

## Main Components

- `src/axiom_encode/cli.py` - public command surface
- `src/axiom_encode/harness/evals.py` - model eval, prompt, artifact, and benchmark-suite orchestration
- `src/axiom_encode/harness/validator_pipeline.py` - RuleSpec compile, CI, review, and oracle validation
- `src/axiom_encode/harness/backends.py` - Codex/OpenAI/Claude runner adapters
- `src/axiom_encode/harness/encoding_db.py` - local run/session logging
- `src/axiom_encode/supabase_sync.py` - telemetry sync

## Checks

```bash
uv run ruff check pyproject.toml src/axiom_encode scripts tests
python -m compileall -q src/axiom_encode scripts
uv run pytest -q tests/test_cli.py tests/test_rulespec_validation.py tests/test_evals.py -k "rulespec or EncoderPrompt"
```

Run the full test suite with `uv run pytest` before publishing broad migration changes.

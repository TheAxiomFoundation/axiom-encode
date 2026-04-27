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
- Rule runtime repo: `axiom-rules`
- Jurisdiction repos: canonical policy repositories such as `rules-us`, `rules-us-tn`, and `rules-us-nc`

The encoder emits RuleSpec YAML artifacts. Source documents are expected to live outside Git, with jurisdiction repositories keeping source registries and manifests that point to durable storage.

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

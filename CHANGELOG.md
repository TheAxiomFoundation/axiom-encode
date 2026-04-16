# Changelog

All notable changes to autorac will be documented here.

## Unreleased

### Changed
- Extracted model pricing rates into `src/autorac/harness/pricing_rates.toml`
  with a `version` and `effective_date` so rate changes can be tracked
  independently of Python code. Public pricing API is unchanged.
- Added `__version__` constants and a versioning policy for embedded
  encoder and reviewer prompts. See `docs/prompt-versioning.md`.
- Defensive handling for reviewer output JSON parse failures in
  `ValidatorPipeline._run_reviewer`: malformed output now returns a
  `reviewer_parse_failed` result and logs a truncated warning instead of
  relying on a generic exception catch.
- Added a deprecation banner to `26_USC_1_VALIDATION_REPORT.md` noting that
  the report predates the live oracle pipeline and is pending refresh.

### Open TODOs needing design input
These are tracked so they don't get lost in code:

- `src/autorac/harness/encoder_harness.py` fallback formula — emitted when
  encoding fails. The placeholder `return 0` is intentional; authoring a
  real formula requires human review. Tagged inline as
  `TODO(#issue-needed)`.

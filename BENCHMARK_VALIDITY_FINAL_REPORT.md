# Benchmark-validity defensive correctness and completeness audit

Date: 2026-07-25  
Branch: `fix/benchmark-validity-timeout-tmpdir`  
Base: `f60fd29d74661641cf2fc9d1cef2180a110145fa` (`axiom-encode` 0.2.1380)  
Prepared version: 0.2.1381

## Outcome

Both measured benchmark-validity defects are fixed, covered test-first, split
into separate commits, versioned, and independently reviewed as a defensive
correctness and completeness change. No actionable review findings remain.

Publication is blocked by the execution environment. Direct Git transport
fails before authentication with `Could not resolve host: github.com`, and the
connected GitHub app canceled each requested write confirmation. Therefore:

- PR number: not created
- CI state: not started
- Merge state: no merge was attempted

The prepared branch can be published from a network-enabled shell with:

```bash
git push -u origin fix/benchmark-validity-timeout-tmpdir
```

After that, open one PR against `main`, reference issue `#1189`, leave it
unmerged, and inspect CI.

## Defect 1: location-independent validation evidence

Production normalization replaces only the exact generated validation staging
root when it is followed by `/`, `\`, or the end of the string. The replacement
is the stable placeholder:

```text
<rulespec-validation-root>
```

The checkout, jurisdiction, source-relative path, and the rest of the compiler
or CI evidence remain intact. Normalization covers `issues`, `error`, and
`raw_output` at compile/CI result capture. Real checkout paths are not
normalized. `compile_issues` and `ci_issues` remain in the reviewer-independent
metric projection, so persisted-row tamper detection is not weakened.

Example:

```text
/private/var/.../T/tmpfk4vkfqh/rulespec-uk/uk/statute/example.yaml
```

becomes:

```text
<rulespec-validation-root>/rulespec-uk/uk/statute/example.yaml
```

## Defect 2: fair, configurable Claude encoder timeout

The Claude encoder now reads:

```text
AXIOM_ENCODE_ENCODER_TIMEOUT_SECONDS
```

through the repository's existing positive-integer environment parser. The
default is 1,800 seconds. This is about 8.5 times the measured 211-second
claude-fable-5 median, compared with the defective 600-second cutoff's roughly
2.8 times median, and matches the existing Codex long-source wall ceiling. It
keeps a bounded hung-process guard without systematically truncating the
observed slow-model tail.

Claude traces record the effective `timeout_seconds`, `timed_out`, and
`timeout_reason`. A wall timeout returns a structured
`Claude eval timed out` response so existing retry and persistence paths retain
infrastructure evidence. Codex timeout selection and behavior are unchanged.

## Test-first coverage

Defect 1:

- `TestEvaluateArtifact.test_normalizes_ephemeral_validation_root_in_compile_and_ci_issues`
- `test_persisted_revalidation_still_rejects_issue_text_drift`
  (`compile_issues` and `ci_issues` parameters)

The first regression was observed failing because the fresh
`/private/var/.../tmp...` root remained in metrics. After the fix, the focused
slice passed 3 tests and the adjacent overlay/revalidation slice passed 16.

Defect 2:

- `TestClaudePromptEval.test_prompt_eval_uses_configurable_encoder_timeout_and_records_it`
- `TestClaudePromptEval.test_prompt_eval_returns_structured_timeout_failure`
- `TestClaudePromptEval.test_prompt_eval_uses_fair_default_for_invalid_encoder_timeout`
  (unset, invalid, zero, and negative parameters)
- `test_claude_encoder_timeout_does_not_change_codex_timeouts`

The red phase produced all 6 expected failures: overrides were ignored,
unset/invalid values remained at 600 seconds, and `TimeoutExpired` escaped.
After the fix, all 6 passed; the adjacent Claude/Codex timeout slice passed 13.

## Required checks

- `uv run ruff check pyproject.toml src/axiom_encode scripts tests`: passed
- `uv run ruff format --check src/axiom_encode tests`: passed, 125 files
  already formatted
- `uv run python -m compileall -q src/axiom_encode scripts`: passed
- Explicit version gate:
  `uv run pytest tests/test_cli.py -k test_current_encoder_affecting_changes_are_behind_version_bump`:
  passed, 1 passed and 1,018 deselected
- Exact requested pytest battery:
  `uv run pytest tests/test_evals.py tests/test_rulespec_validation.py tests/test_cli.py`:
  2,797 passed, 1 skipped, and 1 failed. The sole failure was the pre-existing
  `test_rulespec_target_resolution_accepts_macos_system_path_alias`, which
  directly creates a temporary directory under sandbox-denied `/var/tmp`.
- Same battery with only that environment-incompatible test deselected:
  2,797 passed, 1 skipped, 1 deselected.

Git diff whitespace validation passed. The final worktree is clean.

## Independent review-fix cycle

Independent reviewers separately audited staging-root derivation and
normalization coverage, persisted tamper comparison, Claude subprocess cleanup
and retry behavior, trace persistence, Codex isolation, release/version
consistency, tests, and commit separation.

The only actionable finding was a stale `PROGRESS.md` status after the version
commit. It was corrected and independently re-reviewed. The final review found
no remaining actionable issues and recommended approval. The `/var/tmp` test
failure was classified as a local sandbox limitation to be confirmed by CI,
not a branch regression.

## Commits

- `fc1498b6` — `chore: initialize benchmark validity audit progress`
- `75496c29` — `fix: normalize eval validation staging paths`
- `0e2fce58` — `fix: make Claude encoder timeout configurable`
- `458eb326` — `chore: bump version to 0.2.1381`
- `a90e0249` — `chore: record benchmark audit publication blocker`

The version batch updates `pyproject.toml`,
`src/axiom_encode/__init__.py`, the `axiom-encode` version entry in `uv.lock`,
and two `changelog.d` fragments for issue `#1189`.

# Validator checks and typed issues

Every validator issue should come from a declared check. A check states, in the
code that runs it, what it requires, why that matters, how to fix a failure and
where a failure usually comes from. The explanation of a failure then comes
from the encoder, in the PR that adds or changes the check, not from a list in
a dashboard.

Module: [`src/axiom_encode/checks.py`](../src/axiom_encode/checks.py).

## Declare a check

```python
from axiom_encode.checks import IssueSubject, ValidationIssue, declare_check

CHECK_TESTS_OUTPUT_UNASSERTED = declare_check(
    "complete-source-unit:tests/output-unasserted",
    title="Principal output is never asserted",
    requires="Every principal output is asserted by at least one companion case.",
    why="Companion tests are the executed evidence that ...",
    fix="Add a companion case whose `output` asserts the named principal output.",
    cause="encoder_output",  # encoder_output | validator | source_corpus | infra
    gate="ci",
    site="axiom_encode/harness/source_completeness.py:_companion_test_issues",
)
```

- `id` is `<category>/<name>`. The category is the bracketed tag the message
  already starts with (`[complete-source-unit:tests]`), so old classifiers keep
  working; the name splits one category into its distinct failure modes.
- `fix` is shown to the model in retry feedback and to people on the ops pages.
  Write it as an instruction.
- Declare the check next to the function that emits it. Add a new declaring
  module to `checks.CHECK_MODULES` (a test fails otherwise).

## Emit a typed issue

```python
issues.append(
    ValidationIssue(
        "[complete-source-unit:tests] Principal output "
        f"`{name}` is never asserted by a companion test.",
        check=CHECK_TESTS_OUTPUT_UNASSERTED,
        subject=IssueSubject(citation=corpus_citation_path, rule=name),
        expected="at least one asserting companion case",
        found="no case asserts it",
    )
)
```

`ValidationIssue` is a `str` subclass whose value is the message, so every
`list[str]` consumer is unchanged and message bytes stay identical. Code that
rewrites an issue string (path normalization, bounding, a file prefix) calls
`checks.retype(original, new_text)` or `checks.map_issue_text(issue, fn)` so the
typed fields survive.

`fingerprint` hashes the check, the subject and the expected value (not the
found value), so the same problem at the same place has the same id across
tries and runs.

## Where typed issues go

| Surface | Field | Shape |
| --- | --- | --- |
| Retry feedback to the model | `=== BEGIN CHECK FIXES ===` block after the validator messages | one block per failed check: title, requirement, fix, then `locator; expected ...; found ...` per subject |
| Outcome telemetry (`encoding_runs.outcome_json`) | `validation_failure_counts` | typed issues count under `<gate>:<check id>`; untyped ones keep `<gate>:<message class>` |
| | `validation_failures[]` | typed entries add `check` and `fingerprint` |
| | `validation_issues` | `validation-issue/v2` document (below) |
| Per try (`encoding_runs.iterations[].errors[]`, Supabase `iterations`) | `issues` | `validation-issue/v2` document of that try |
| Failure bundle (`issues.json`) | `typed_issues` (optional) | `validation-issue/v2` document |
| Run log v1 (`gate.compile`, `gate.ci` findings) | `Finding.code` | the check id; `locator` and `evidence` from the issue |
| Workflow refusals (`scripts/infra_issue.py fail infra:<step>`) | `$RUNNER_TEMP/axiom-infra-issues.jsonl`, step summary, error annotation | one `validation-issue/v2` record per refusal, with `fix` |

Each field is absent when no issue is typed, so untyped records keep their old
bytes.

### `axiom-encode/validation-issue/v2` document

```json
{
  "schema": "axiom-encode/validation-issue/v2",
  "total": 3, "typed": 2, "truncated": 0,
  "issues": [
    {
      "gate": "ci", "kind": "complete-source-unit:tests/output-unasserted",
      "message": "...", "category": "complete-source-unit:tests",
      "locator": "rule:x", "line": null, "value": "no case asserts it",
      "clause": "us/statutes/7/2014/e",
      "check": "complete-source-unit:tests/output-unasserted",
      "cause": "encoder_output",
      "subject": {"citation": "us/statutes/7/2014/e", "rule": "x"},
      "expected": "at least one asserting companion case",
      "found": "no case asserts it", "evidence": null,
      "fingerprint": "3f0c..."
    }
  ],
  "checks": {"complete-source-unit:tests/output-unasserted": {"title": "...", "fix": "..."}},
  "check_registry_sha256": "9bc0..."
}
```

The issue fields are a superset of `axiom-encode/validation-issue/v1`
(PR #1660): `gate`, `kind`, `message`, `category`, `locator`, `line`, `value`
and `clause` keep their v1 meaning. Bounds: 200 issues, 128 KiB and 2,000
message characters per document. `checks` holds the definitions of the checks
the document names, so a reader needs no other file.

## Publish the registry

- `axiom-encode checks` lists the checks; `axiom-encode checks --json` prints
  `axiom-encode/check-registry/v1`: `encoder_version`, `registry_sha256` and
  every check.
- [`src/axiom_encode/data/check_registry.json`](../src/axiom_encode/data/check_registry.json)
  is the same document without the encoder version, committed and kept in sync
  by `tests/test_checks.py`. Regenerate it after changing a check:

  ```bash
  uv run python -c 'from axiom_encode.checks import committed_registry_text; print(committed_registry_text(), end="")' > src/axiom_encode/data/check_registry.json
  ```

## Ratchet

`tests/test_checks.py` counts untyped `*issues.append(...)` sites under
`src/axiom_encode` and bare `echo ... >&2` + `exit 1` refusals in
`targeted-signed-reencode.yml`. Each count must equal its baseline: a new
untyped site fails the test, and a converted site asks you to lower the
baseline. A check without `fix` or `why`, or whose `site` names no function or
workflow step, also fails.

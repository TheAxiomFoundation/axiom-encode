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
tries and runs. Two subjects need to differ wherever two problems differ:

- A source-branch subject carries `span`, the `start:end` character offsets of
  the branch in the authoritative source text. Two computations (or rounding or
  exception clauses) under one paragraph share `branch` and `clause`; the span
  keeps them apart, and it does not change when the other one is repaired.
- A compile subject names the module by its path relative to the policy repo.
  When that root is a canonical jurisdiction content root (`rulespec-us/us-ky`),
  the jurisdiction is the first path part (`us-ky/statutes/1/1.yaml`), as it is
  relative to the checkout root, so the same relative path in two
  jurisdictions gets two fingerprints.

## Where typed issues go

| Surface | Field | Shape |
| --- | --- | --- |
| Retry feedback to the model | `=== BEGIN CHECK FIXES ===` block after the validator messages | one block per failed check: title, requirement, fix, then `locator; expected ...; found ...` per subject |
| Outcome telemetry (`encoding_runs.outcome_json`) | `validation_failure_counts` | typed issues count under `<gate>:<check id>`; untyped ones keep `<gate>:<message class>` |
| | `validation_failures[]` | typed entries add `check` and `fingerprint` |
| | `validation_issues` | `validation-issue/v2` document (below) |
| Per try (`encoding_runs.iterations[].errors[]`, Supabase `iterations`) | `issue_document` | `validation-issue/v2` document of that try |
| Failure bundle (`issues.json`) | `typed_issues` (optional) | `validation-issue/v2` document |
| Run log v1 (`gate.compile`, `gate.ci` findings) | `Finding.code` | the check id; `locator` and `evidence` from the issue |
| Workflow refusals (`scripts/infra_issue.py record infra:<step>`) | `$RUNNER_TEMP/axiom-infra-issues.jsonl`, step summary, error annotation | one `validation-issue/v2` record per refusal, with `fix`; the annotation text starts with `[infra:<step>]` |
| Any other failed pre-model or publish step of `targeted-signed-reencode.yml` | the failed step's name (jobs API, or `failed_steps` in the failure bundle) | the registry check whose `site` is `.github/workflows/targeted-signed-reencode.yml:<step name>` |
| Every dispatch of `targeted-signed-reencode.yml` (`scripts/dispatch_record.py`) | artifact `dispatch-record-<run id>-<attempt>` (`dispatch-record.json`), and the name of the step `Upload dispatch record: ...` in job `Record dispatch` | `axiom-encode/dispatch-record/v1` (below) |

Each field is absent when no issue is typed, so untyped records keep their old
bytes.

### Step checks and the dispatch record

A step of the targeted re-encode workflow that fails without a typed issue of
its own still has a check: `infra_checks.STEP_CHECK_DEFINITIONS` declares one
`infra:<step>` check for each step of the `attempt_budget` and `encode` jobs
that runs before the model or publishes its result. Those steps do not print
the id. Map the failed step's name to the check through `site` in
`check_registry.json`. A typed id in the error message (`[<check id>] ...`) is
more specific and wins over the step check.

The `dispatch_record` job writes one `axiom-encode/dispatch-record/v1` record
per dispatch, for every trigger and before the signing approval:

| Field | Meaning |
| --- | --- |
| `trigger_kind` | `promotion`, `repair`, `queue` or `adhoc`: the most specific of the three flags below, in that order |
| `is_queue`, `is_repair`, `is_promotion` | `queue_id` is set; `repair_run_id` is set; the source bundle is a reviewed-PR promotion or carries `reviewed_candidate_promotion: true` (`null` when the bundle is unreadable) |
| `live_reviewed_pr`, `reviewed_candidate_promotion` | the two promotion forms behind `is_promotion`: a `reviewed-pr-promotion/v1` bundle; a transaction that sets `reviewed_candidate_promotion: true` |
| `open_pr`, `pr_base_branch`, `repair_run_id`, `queue_id`, `queue_item_id`, `queue_manifest_sha256`, `queue_item_generation_sha256`, `queue_dispatcher_run_id`, `citation`, `country`, `rulespec_ref`, `corpus_ref`, `rules_engine_ref` | the dispatch input, or `null` when it is empty or does not match its allowlist pattern |
| `inputs_present` | `true`/`false` for each free-text input; the value is never recorded |
| `invalid_inputs` | names of inputs that were set but did not match their pattern |
| `repository`, `run_id`, `run_attempt`, `event_name`, `ref`, `head_sha`, `actor`, `triggering_actor`, `recorded_at` | run identity |

The upload step's name repeats the main fields as
`kind=<kind> open_pr=<bool> pr_base_branch=<branch> repair_run_id=<id>`, so the
jobs API returns them without an artifact download. The job has no secret,
continues on every error, and nothing depends on it. The record says what the
dispatch requested, not what the workflow admitted: a later step can still
refuse the promotion or the repair. The record, the tag and the artifact never
hold a free-text value; the job log shows the raw inputs, as the logs of the
budget and preflight steps already do.

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

Each issue record has every field name of `axiom-encode/validation-issue/v1`
(PR #1660): `gate`, `kind`, `message`, `category`, `locator`, `line`, `value`
and `clause`. The names match, but for a typed issue some values do not match
what the v1 code writes for the same issue:

| Field | v1 (#1660) | v2, typed issue |
| --- | --- | --- |
| `kind` | message class, such as `complete_source_unit_tests_principal_output` | the check id, such as `complete-source-unit:tests/output-unasserted` |
| `value` | parsed from the message text (`expected=...`/`actual=...`), else `null` | `found`, else `expected` |
| `clause` | a citation parsed from the message text, else `null` | `subject.clause`, else `subject.citation` |
| `locator`, `line` | parsed from the message text | from the subject; a source branch adds ` chars <start:end>` to the locator |

`gate`, `message` and `category` keep their v1 meaning. An untyped issue keeps
the v1 `kind`; its other fields stay `null`. The container also differs: v1 stores a list of
records under `iterations[].errors[].issues`, and this document is stored under
its own key, `iterations[].errors[].issue_document`, so a v1 reader never
reads a v2 document as an empty list. Bounds: 200 issues, 128 KiB and 2,000
message characters per document. The byte bound covers the whole document as
`json.dumps(document, ensure_ascii=False)` writes it, the envelope and `checks`
included. `checks` holds the definitions of the checks the document names, so a
reader needs no other file.

A record whose message is longer than 2,000 characters stores the first 2,000,
`message_truncated` (the characters cut) and `message_sha256` (the SHA-256 of
the full message). `checks.issues_from_document` matches a full message to such
a record only by that digest, so a long issue keeps its check and fix when a
`--repair-candidate-root` bundle is replayed. The workflow's repair extractor
(`scripts/extract_repair_candidate.py`) checks `typed_issues` but writes only
the candidate pair, so a workflow repair replay has no prior issues to re-type.

## Bounded physical allowance alternatives

The source-explicit-condition factual-gate check has a closed reader for a
same-period expense allowance list: a shared minimum distinct-object count,
followed by physical `any of the following applies` bullets. Each supported
bullet coordinates an action with billing, and either states a bounded causal
alternative or repeats the shared period. The reader consumes the whole source
unit; it is not a general English parser. Other source shapes retain existing
analysis. Recognized lists with missing or ambiguous proof ownership fail.

This reader requires an explicit shared-set identifier contract. For a source
object such as `warehouses`, a common prefix containing that object token, such
as `stored_warehouses`, identifies the caller-classified set. Other prefix words
are unrestricted; the object token is required:

- `<prefix>_distinct_expense_warehouse_count` is a `Count`, not an invoice count.
- `<prefix>_billed_for_those_warehouses` and
  `<prefix>_<action>_those_warehouses` are separate Boolean facts for that same
  set. The literal source action `moves` instead requires
  `<prefix>_moves_between_those_warehouses`, meaning the actor moves between
  the objects, not physically moves the objects themselves. Other actions
  cannot use `between` to replace a direct-object assertion.
- `<prefix>_because_of_<source_cause>` supplies a separate causal fact when the
  source requires one. The supported coordinated causal wording is bounded,
  such as `research or training away from office`.

The object and actions come from the source, not a jurisdiction or concept
allowlist. Unsupported identifiers and nonempty free-form input descriptions
cannot corroborate this closed contract; the reader does not guess whether
prose redefines the count, billing set, or causal relationship. Facts and output
must share the declared entity and source period. The count comparison must
match the source minimum, including its supported integer complement.

Every signed granting formula path must supply the shared prerequisites and
one entire factual alternative. A false prefix in an `or` expression cannot
provide positive evidence. Local derived dependencies retain temporal
partitions; gaps, overlaps, cycles, unknown references, unsupported expressions,
and path caps fail closed. Exact authoritative formula proofs must cover the
shared clause and every physical alternative.

This verifies neither raw object identities nor raw causal evidence. Those
remain caller-classification obligations. Passing this check alone establishes
neither coverage of every legal alternative nor complete semantic equivalence,
successful protected validation, or oracle parity. No historical failed run is
reclassified, and generated candidates are not manually rewritten to fit it.

## Start date of a date-silent unit

`complete-source-unit:effective-date/start-unsupported`
([`harness/start_date_evidence.py`](../src/axiom_encode/harness/start_date_evidence.py),
issue #2212) checks the start dates of a corpus `manual` unit that prints no
date, in a run without amendment context. For the same units, the encoder gets
the rule in its prompt, with dates that code computes from the corpus
metadata; encodes of other document classes get the earlier prompt text. The
prompt asks the encoder to prefer a date that the unit text prints, then the
evidence floor, or the `expression_date` when that date falls after the floor.

The maintainer has not decided the six design choices that issue #2212 lists:

- the evidence floor or the scaffold as the default start;
- whether an imported start counts as support;
- how to treat existing modules;
- the manual-only scope;
- whether any year that the detector recognizes takes the unit out of scope; and
- how to treat unreadable metadata.

The check leaves a unit alone when:

- the citation's document class (second path segment) differs from `manual`;
- the run binds amendment or commencement documents;
- the pipeline has no corpus date channel; or
- the unit text prints a date that the check's detector recognizes: a full
  date, a month and year, a fiscal year such as `FY26`, or a bare year from
  1900 to 2099 outside a dollar amount or a percentage.

The detector matches a fixed set of spellings, so a unit that prints a date
only in another spelling stays in scope.

In a unit that it examines, the check accepts a start that equals one of these
dates:

- the `0001-01-01` scaffold;
- the corpus `expression_date` or `source_as_of`;
- the evidence floor, the first day of the first full calendar month that
  starts on or after `source_as_of` (2026-07-17 gives 2026-08-01; 2026-07-01
  gives 2026-07-01);
- a date that a `rules[].metadata.proof.atoms[].source.excerpt` anywhere in
  the module prints; or
- a version start of an export that the module verifiably imports.

The check uses the floor as a proposed calendar convention. Its calculation
establishes no legal commencement, first corpus custody or text continuity,
and the other channels can support earlier starts. Copying another module's
start adds no support on its own.

The check refuses every other start, including one after the snapshot, with
one issue for each distinct value it refuses. It does not examine
`effective_to`, `values` keys or companion test periods.

Each issue names up to five rules that use the value. Its message and its
`expected` field list the scaffold and, when the check can read them, the
corpus `expression_date`, `source_as_of` and the evidence floor. They mention
dates in the unit text or a proof excerpt only as a kind ("a date printed in
the unit text or in a proof-atom excerpt you cite") and name no imported
start. The check's `fix` text names no import either. The message gives the
field as `rules[].versions[].effective_from` at every site; the issue's
`subject.field` names the first actual site, such as `versions[0].from` or a
rule-level `effective_from`.

### Which start the check reads

For start selection, the check follows the explicit and implicit version
forms at the CI engine pin:

- A nonempty `versions` list replaces the rule-level fields, so the check
  reads each version's start and ignores a rule-level start.
- When a rule sets a formula or a start, the engine can build an implicit
  version from the rule-level fields. It does so when the rule has no
  `versions` key, or its `versions` holds `[]`, a blank value or only a
  comment. The check then reads the rule-level start.
- The check skips a rule whose `versions` holds some other non-list value,
  such as a mapping, and leaves that error to the compiler.
- `from` works as an alias of `effective_from`, in a version and at rule
  level. A mapping that sets both carries a duplicate field, not a precedence
  order. The check refuses it: the issue shows
  `duplicate effective_from/from fields` as the value, with the explanation it
  gives an unsupported date, and `subject.field` ends in
  `effective_from/from`.
- The check supplies no default for a missing start; the compiler reports the
  missing start.
- The check reads a start without zero padding (`2025-10-1`) and a YAML
  timestamp as their calendar day, without deciding whether the compiler
  accepts that spelling. It refuses any other non-null value that it cannot
  read as a date.

The analyzer passes the raw candidate text to the check as `raw_content`. The
check parses that text without YAML merge inheritance, because the engine
ignores a `<<` key on rules and versions. When a `<<` mapping would supply a
version list or a start under PyYAML, the check ignores it, as the engine
does. An explicit `versions: null` or `versions: ~` differs from a blank
value: the engine refuses it, and the check reads no start from that rule.
The check parses imported providers the same way, so a provider's `<<`
mapping or explicit null `versions` cannot lend a start that the engine would
not compile. The check still takes proof excerpts from the analyzer's parsed
mapping. A direct caller that passes only a parsed mapping hands the check
PyYAML's reading, in which PyYAML has already expanded merges, and the check
then treats an explicit null like a blank value.

### Imported starts

An import can only add dates that the check accepts; it never takes a module
out of the check. A `path#symbol` item adds the version starts of its export
when the validator received a provider for that symbol and the check can
parse it. The check requires that provider to declare exactly one rule of
that name and import no symbol that this module declares. The check matches a
provider by symbol name, and the module does not have to use the export.
These items add nothing:

- a file-level import without `#symbol`;
- a symbol outside identifier syntax, or one that the module declares itself;
- a provider that the validator did not receive or that the check cannot
  parse;
- a provider with no rule or several rules of that name; and
- a non-string item.

The check refuses a start that only such an import would support.

### Corpus dates

Missing metadata never supports a start. The check reads `source_as_of` and
`expression_date` from the resolver attestation, which carries each field at
its top level and under `row`. The check reads non-null copies as strict
`YYYY-MM-DD` dates and accepts one valid copy when the other supplies no
value. A malformed copy or two conflicting copies remove support from that
field. Without a readable `source_as_of`, the check has no snapshot date and
no evidence floor. The scaffold, a readable `expression_date`, a date that a
proof excerpt prints and an imported start that the check accepts still pass.
When it reports an unsupported start without a snapshot date, the check names
the metadata problems in its message.

The corpus dates reach the check this way:

- An encode passes the resolver attestation as `source_metadata`.
- `axiom-encode validate --require-complete-source-unit` passes no
  `source_metadata`, so the pipeline reads the unit's row from its corpus
  release. When that release does not resolve the citation, the check treats
  the metadata as unreadable.
- A pipeline with neither `source_metadata` nor a corpus release has no corpus
  date channel, and the check does not run.

The check runs inside the complete-source analysis, which the pipeline applies
only in `--require-complete-source-unit` mode. The targeted re-encode workflow
adds that flag when its `require_complete_source_unit` setting holds `true`,
and `axiom-encode validate` without the flag skips the check. The replay
benchmark passes each fixture's attestation dates and import providers, and
no amendment context.

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

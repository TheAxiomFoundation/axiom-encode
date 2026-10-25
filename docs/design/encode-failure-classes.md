# Encode failure classes and the pre-encode gate

Status: design, 2026-10-23. Owner: Pavel. Code refs are origin/main at 0.2.2250. Stage 1 is implemented in 0.2.2254; the Stage 1 section below describes what was built.

## Problem

About 80-90% of local encodes fail. A hand-adjudicated sample (30 hard citations, 313 failed runs; taxonomy data in the local A/B archive, `2026-10-23-failure-taxonomy/`) shows that only 47% of failures are encoder or validator problems. 37% come from the target itself: it needs a concept that is not encoded yet (20%), or it is the wrong target (17%: TOC and admin pages, repealed text, overlap with a sibling). The pipeline spends up to 4 model tries on each of these, and the repair loop cannot fix any of them.

Today nothing records why a run failed. Issues are plain strings (`ValidationResult.issues: list[str]`, `validator_pipeline.py:34951`); the only typed field is `EvalFailureKind = timeout | validation | error` (`evals.py:186`). No table has a failure-class column.

## Failure classes (v1)

| Code | Class | Owner | Subclasses |
|---|---|---|---|
| `encoder` | Encoder error | encoder | `omission`, `formula`, `test`, `harness` |
| `validator_fp` | Validator false positive | validator | — |
| `corpus` | Corpus gap | corpus | `content`, `metadata` |
| `dependency` | Missing upstream dependency | queue/planning | — |
| `selection` | Target selection or scope | queue/selection | — |
| `engine` | Engine or RuleSpec language limitation | engine | — |
| `migration` | Migration debt | engine/toolchain | `schema`, `elided_excerpt` |
| `process` | Policy or process block | policy-review | — |
| `infra` | Infra or transient | infra | — |
| `unknown` | Not classified | — | — |

Rules:
- Every record has `class`, optional `subclass`, `source` (`rule` or `triage`), `confidence` (`certain` for rules), and `evidence` (short strings: issue tags, paths). A successful outcome has `failure_class: null`.
- A class is advisory. It never waives a failure and never admits an artifact.
- Add a subclass only if it changes the fix inside one owner and has about 10 or more cases. Review the list each quarter.

## Stages

### Stage 0: measure

- Finish draft #1891 (offline replay benchmark, 32 real US failure fixtures). Add `expected_class` / `expected_subclass` to each `fixture.json` (`load_fixture` tolerates extra keys) from the taxonomy adjudication.
- Add a deterministic `classify_outcome()` and report its agreement with `expected_class` in `replay` output.
- Live encode A/B stays the local process (`docs/local-encoder-ab.md`); no change here.

### Stage 1: failure-class module and pre-encode gate

New module `src/axiom_encode/failure_classes.py`: the class and subclass codes, the frozen record type `FailureClassRecord`, and `classify_outcome(outcome) -> FailureClassRecord | None` using only deterministic signals. In v1 the certain rules are:

- `final_success: true` → `None` (no failure).
- `status: pre_encode_blocked` → the class of the enforced gate finding (`selection` for `repealed_target` and a heading-only body; `corpus/content` for an empty or punctuation-only body).
- a timeout (`timed_out` or `failure_kind: timeout` on the final try) → `infra`.
- an `apply_error` or `primary_error` that starts with `[generated-yaml:io]` or `[generated-yaml:decode]` → `encoder/harness`.
- anything else → `unknown`. No other bracket tag maps to one class with certainty today, so issue text alone never yields `encoder` or `validator_fp`, and report-mode gate findings never classify a later failure.

New module `src/axiom_encode/harness/pre_encode_gate.py`:

```
evaluate_pre_encode_gate(source_unit, *, policy_repo_path, replacement_target) -> GateResult
GateResult: findings: list[GateFinding]; blocked: bool
GateFinding: check, class, subclass, enforce (bool), reason, evidence
```

v1 checks, all deterministic, all on data the encoder already has at `cli.py:34668-34700`:

| Check | Class | Mode | Signal |
|---|---|---|---|
| `toc_page` | selection | report (demoted from enforce, see below) | at least 10 dot-leader runs (the queue builder's `_is_table_of_contents`, now `axiom_encode.source_text_signals.is_table_of_contents`; same thresholds) **and** at most 15% of the body's non-space text outside contents entries (an entry is a title, a leader run and a page reference, not an amount with thousands separators) **and** the word "contents" in the body **and** nothing outside the entries that could state a rule: no prose sentence (8 or more words, mostly lower case), no modal, predicate or negation word (`shall`, `must`, `is`, `not`, ...), and no `$` or `%`; and every word left outside the entries is explained: contents-page furniture ("Table of Contents (cont'd) Page 3", "Chapter 11"), a roman page number, or a word that also occurs in an entry or in the row's own labels (a running head, a chapter title listed below). Every number left outside the entries must be a locator: after a unit word (page, chapter, section, part, title, article, appendix), the row's own page number (from its labels) at the start of a line, or a revision date. Capitalization proves nothing. The last two conditions were added after measurement: whole manual chapters that open with their own contents list, fill-in forms, tax-form lines and appropriation tables all pass the leader count. |
| `no_operative_text` | corpus/content (empty or punctuation-only body: the row lost its text in extraction); selection (heading-only body) | enforce | body is empty or punctuation only; or one of the row's own labels (heading, citation label, title) occurs in the body and, after it is removed, nothing is left but at most one citation locator that names the target itself (a locator that names another provision, such as `Section 401(a)(9)(A).`, is a cross-reference and keeps the body) |
| `repealed_target` | selection | enforce | the resolver's repealed-row refusal (`corpus_resolver.py:704`), now raised as `RepealedCorpusSourceError`, a subclass of `InvalidActiveCorpusSourceError`; what the resolver accepts is unchanged, and a partly repealed container still resolves |
| `existing_module` | selection | report | canonical RuleSpec path already exists and neither `--replace-rulespec-path` nor `--replace-legacy-rulespec-path` is given |
| `unencoded_reference` | dependency | report | the body cites a USC or CFR section (explicit `N U.S.C. M` / `N CFR M.N` forms only; at most 8 lookups) that has an exact active corpus row and no RuleSpec module at or under its canonical path |

Rules:
- `toc_page` was designed to enforce and is report-only in Stage 1 (decision 2026-10-23). It is a judgement heuristic: seven review rounds each built synthetic operative text next to a real contents list that defeated the current rules, and when in doubt the default must be to encode. It accounted for 63 of 502 measured blocks. It moves to enforce only with measured evidence on real rows, not synthetic ones.
- `enforce` checks stop the encode before any model call. Only checks with zero false blocks on the benchmark and on the queue inventory may enforce.
- `report` checks only record findings. One of them moves to `enforce` only after measurement shows no false blocks.
- `--pre-encode-gate {report,enforce}`; default `enforce` (which still runs report checks as report). `report` turns every check advisory.
- Insertion: once per invocation in `_run_encode_attempts_with_retries`, before `while True` (`cli.py:34413`). The gate runs the first try's own preflights in the same order (dependency roots, compose path, Codex auth, corpus and rules-engine paths, checkout, pending apply transaction), then resolves the target with the same functions and hands the resolved release and source unit to the first try, so the target is resolved once. If that resolution fails for any reason other than a repealed row, the gate is skipped and the first try reports the same error as before. On a block: outcome `status = "pre_encode_blocked"`, `pre_encode_gate` findings and `failure_class` are recorded through `_initial_encode_outcome` and `_record_encode_outcome` (`cli.py:65688`) on a run with no iterations (`generation_attempt_count = 0`), the run is synced like any other (with a measured `generation_attempt_count` of 0), its run log records `generate` and `apply` as skipped with reason `pre_encode_blocked`, and the command exits non-zero without a model call. In `report` mode a repealed target still raises the resolver error in the first try, as before.
- A run that is not blocked records `pre_encode_gate` in its outcome only when there are findings.
- When the encoder also gets amendment documents for the target (a principal provision with an authenticated amendment, or an amending provision), the target's own body is not the whole source: `toc_page` and `no_operative_text` then only report.
- Every outcome gets `failure_class` (from `classify_outcome`). It flows into `encoding_runs.outcome_json` and Supabase `encoding_runs.outcome` with no migration.

Measurement before merge: run the gate offline in `report` mode over every item of the all-states SNAP queue inventory and the 32 benchmark fixtures. Count blocks per check and hand-check every `enforce` block (or a sample of 30 if more). Put the table in the PR. (Done in the Stage 1 PR: the first run found whole manual chapters that `toc_page` would block, and the first review found a fill-in form and a cross-reference body; a scan of every body row in the local corpus then found tax-form lines and an appropriations table. Those led to the extra `toc_page` conditions and the self-locator rule above.)

### Stage 2: stop repairs that cannot work

- `_encode_attempt_was_validator_rejected` (`cli.py:34208`) is the single retry predicate. Return False when every residual blocking issue classifies as a non-encoder class with `confidence = certain` (today: `migration`, `process`, `infra`, `selection`). Record `retry_stopped: {class, reason}` with the existing `generation_retry_blocked` pattern (`cli.py:34440-34470`).
- Do not escalate to the larger model for those classes.
- Build on Max's draft #1660 (`validation-issue/v1`). Do not duplicate its persistence. If #1660 has not merged, stage 2 adds only the class to the outcome and the retry predicate.

### Stage 3: triage agent (later, separate design)

- Runs locally on the subscription (no API sends), after a final failure, with evidence: source, candidate, validator output, gate findings, corpus probe.
- Independent of the generator (different prompt; different model where possible).
- Writes `source = triage` records with a confidence; low confidence stays `unknown`.
- Weekly human check of 20 random cases; agreement per class is reported.

## Out of scope

- Changing what validation accepts.
- A database migration.
- The validator/generator package split (separate design).

## Risks

- A gate check that is too broad blocks valid targets. Mitigation: enforce only measured zero-false-block checks; `--pre-encode-gate report` as an escape hatch.
- Version bump conflicts: each `src` PR bumps the encoder version. Keep PRs small and land them one at a time.
- Overlap: #1564 (scope check in the workflow) stays separate; this gate is in the encoder and runs for local and workflow encodes.

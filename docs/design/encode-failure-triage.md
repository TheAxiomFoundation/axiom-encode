# Stage 3: failure triage agent

Status: design approved 2026-10-25 (decisions below). Owner: Pavel. Follows `docs/design/encode-failure-classes.md` (Stages 0–2, PRs #1891, #1997, #1999). Build step 1 (the evidence bundle and the five probes, no model) is implemented in 0.2.2256; see "PR 1: evidence bundle and probes" below.

## Problem

Stages 1 and 2 classify only what deterministic rules can prove. On recorded data that is a small share:

- The Stage 1 gate stops 439 of 17,721 queue items (2.5%), all empty corpus bodies.
- The Stage 2 retry stop fires on 0 of 1,359 recorded attempts.
- In the 32 labelled benchmark fixtures, the classes that matter most are judgement classes: encoder 10, dependency 5, selection 5, validator false positive 5, unknown 4.
- In 38 of 40 sampled "deferral not precise" rejections, the error message pointed to the wrong class.

So most failures stay `unknown`. Owners never see them, and the same work is retried and re-dispatched. Stage 3 adds an evidence-based classifier for these failures. It labels; it does not decide what lands.

## Goals

1. Give every final failure a class, a confidence and evidence.
2. Measure classifier accuracy per class, and report only classes that meet a threshold.
3. Route each confident failure to its owner as a draft work item.

## Non-goals

- Triage never waives a failure, admits an artifact, or changes validation.
- Triage does not change the queue order in v1 (see Phase B).
- No API sends: triage runs only on the local subscription login, like the local A/B process.

## Where it runs

| Mode | Input | When |
|---|---|---|
| Local encode | the run's own outcome and workspace | after a final failure, when `--triage` is set (default off in v1) |
| Batch | workflow failure bundles (`failed-reencode-diagnostics/v1`) and local pilot runs | on demand: `axiom-encode triage-failures <dir>` |

CI and workflow encodes do not call a model for triage (no subscription in CI). Their bundles are triaged in batch, locally.

## Independence

The generator runs on OpenAI models (`gpt-6-luna`, escalated to `gpt-6-sol`). Triage runs on Claude through the `claude` CLI on the subscription. The triage prompt never sees the generator's prompt or reasoning, only the evidence below. A second opinion with codex `gpt-6-astra` is used only for calibration disagreements.

## Evidence bundle (input)

Built deterministically before the model call. Bounded to a fixed size; truncation is recorded.

1. Target: citation, document class, corpus row identity, the source text, amendment context.
2. Final candidate: RuleSpec YAML and companion tests (the best attempt).
3. Validator output: all issues of the final attempt, with gate and bracket tag (`validation-issue/v1` from #1660 when it lands; text otherwise).
4. Attempt history: per-attempt issue counts and which issues persisted across all attempts.
5. Deterministic probes, run before the model:
   - **Gate findings** from Stage 1, including report-only checks (`existing_module`, `unencoded_reference`, `toc_page`).
   - **Registry probe:** for every concept the candidate declares as an input or deferral provider, whether a RuleSpec module exports it at the run's `rulespec_ref`.
   - **Corpus probe:** for every citation the source references, whether the corpus row exists and its status metadata.
   - **Import probe:** whether each compile error is in the target's own file or in an imported module that the attempt did not author.
   - **Persistence probe:** issues that every attempt had (likely not encoder-fixable) versus issues that changed.

The probes carry most of the signal. The model reads them with the source and decides the root cause.

## Output schema

```
axiom-encode/failure-triage/v1
{
  run_id, citation, triage_version, model, prompt_sha256, evidence_sha256,
  class, subclass,            # the 10 class codes and subclasses from Stage 1
  confidence: high | medium | low,
  rationale,                  # at most 3 sentences
  evidence: [ {kind, ref, quote?} ],   # each must point into the bundle
  owner,                      # derived from class, not chosen by the model
  next_action                 # one of a fixed list per class (below)
}
```

Rules:
- An answer whose `evidence` refs do not resolve inside the bundle is rejected and recorded as `unknown`.
- `low` confidence is stored but reported as `unknown`.
- The record is stored beside the deterministic `failure_class`, never over it. `failure_class.source` stays `rule`; triage writes `failure_triage`. When both exist, the rule record wins for reporting.

## Next actions (fixed list)

| Class | next_action | Draft target |
|---|---|---|
| encoder | `retry_with_feedback` (omission/formula/test) or `fix_harness` | encoder backlog |
| validator_fp | `validator_issue` | axiom-encode issue draft with the issue text and the source quote |
| corpus | `corpus_issue` | axiom-corpus issue draft |
| dependency | `encode_dependency_first` | the missing concept and its citation |
| selection | `drop_or_rescope_target` | queue item note |
| engine | `engine_issue` | axiom-rules-engine issue draft |
| migration | `migrate_module` | module path list |
| process | `unblock_process` | note only |
| infra | `retry_later` | note only |

v1 writes drafts to a local file and to telemetry only. Filing issues or editing queues stays a person's action.

## Persistence

- `outcome.failure_triage` in `encoding_runs.outcome_json`; flows to Supabase `encoding_runs.outcome` unchanged (no migration), like Stage 1.
- Batch mode writes `triage.jsonl` next to the input bundles.

## Calibration and trust

Labelled sets:
- the 32 #1891 fixtures (`expected_class`, hand labels);
- the 30 hand-adjudicated local citations in the 2026-10-23 taxonomy;
- a growing weekly sample.

Gate before any class is reported as trusted:
1. Run triage on both labelled sets.
2. Per class, compute precision and recall against hand labels.
3. A class is **trusted** when precision ≥ 0.85 on at least 8 labelled cases. Untrusted classes are reported as `unknown` (with the raw triage stored).
4. Report the confusion matrix in the PR that turns triage on.

Weekly check:
- 20 random triaged failures, stratified by class, labelled by a person blind to the triage answer.
- Agreement per class is tracked. A trusted class that drops below 0.85 on the rolling last 40 checks loses trust automatically.
- Disagreements go into the labelled set, and a fixed prompt version is re-calibrated on the whole set before any prompt change merges.

## Phases

**Phase A (this design):** labels and drafts only. Measures accuracy. Changes no behaviour.

**Phase B (separate decision, after Phase A data):** use trusted classes to act.
- `dependency`: reorder the queue so the missing concept is encoded first.
- `selection`: set the queue item to `blocked` with evidence.
- `validator_fp`: group by issue tag; the top groups become validator fixes.
- Only trusted classes act. Each action is reversible and logged.

## Cost

- One triage call per final failure, not per attempt. Batch over the 451 recorded bundles is a one-time run.
- Local subscription only. With a 20-minute encode, triage adds about 1–2 minutes per failed run.

## Build plan

1. `src/axiom_encode/triage/evidence.py`: the bundle builder and the 5 probes (deterministic, unit-tested, no model).
2. `src/axiom_encode/triage/classify.py`: prompt, CLI runner (`claude` CLI, subscription only, refuses to run when an API key is set), output validation.
3. `axiom-encode triage-failures` batch command and `--triage` on `encode`.
4. Calibration script over the labelled sets; confusion matrix in the PR.
5. Turn triage on for local encodes only after step 4 shows at least 3 trusted classes.

Each step is its own small PR with the review cycle. Steps 1 and 3 are useful without a model: the probes alone already improve the failure bundle.

## Risks

- **Confident wrong labels.** Mitigated by evidence refs that must resolve, per-class trust thresholds, and the weekly check.
- **Label drift with prompt changes.** Prompt and model are versioned in every record; any change re-runs calibration.
- **Generator/triage collusion.** Different vendor and no shared prompt.
- **Cost creep.** One call per final failure; batch is opt-in.

## Decisions (approved 2026-10-25)

1. Triage model: Claude via the `claude` CLI on the subscription.
2. Weekly human check size: 20.
3. Trust threshold: precision ≥ 0.85 on ≥ 8 cases.
4. Phase B (queue actions): decided after Phase A data, not now.

## PR 1: evidence bundle and probes (implemented, 0.2.2256)

Package `src/axiom_encode/triage/` (no model call anywhere):

- `evidence.py`: loaders for a local encode (outcome, failed attempts and its `--output` workspace), a local pilot run directory (`run.json`, `encodings.db`, `failed_attempts.jsonl`, `generated/`, `final-rejected-candidate/`) and a workflow failure bundle (`failed-reencode-diagnostics/v1`, `metadata.json` or `<dir>/x/metadata.json`); the bundle builder; `resolve_ref`.
- `probes.py`: the five probes. `views.py`: a RuleSpec state read from local git objects at the run's `rulespec_ref` (never fetched: no lazy fetch, no transport, no replace objects), or a checkout on disk; an unverified index of a local corpus checkout. `issues.py`: issue file prefix, gate, bracket tag and a normalized signature.
- CLI: `axiom-encode triage-evidence <bundle-or-run-dir> [--out FILE]`, and `--batch <dir>` writes `evidence.jsonl` (one canonical JSON line per run) and prints a summary. `--rulespec-repo`, `--rulespec-ref`, `--rulespec-checkout` and `--corpus-root` name the local views; without them the probes that need them record `not_checked` or `skipped`.

Bundle `axiom-encode/failure-triage-evidence/v1`:

- `run`, `target` (citation, attestation without the runner's absolute root, source text, amendment context, the dispatch's review findings), `candidate` (RuleSpec and tests, runner, sibling modules), `final_issues` (the issues of the selected final candidate, whose attempt number is not recorded: `text`, `file`, `gate`, `tag`), `attempts` (local runs: per-attempt `issue_count` and normalized `issue_signatures`; workflow bundles record none), `views` (identities of the RuleSpec state, corpus checkout and concept registry), `probes`, `truncation`.
- Bounded: per-field limits (source and candidate 48,000 characters, 120 final issues of 1,200 characters, 8 attempts of 60 signatures, 64 findings per probe, free-form metadata 8,000 characters) and 512 KiB for the written JSON line; a bundle over the total is rebuilt with tighter limits (three tiers), and one that still does not fit is refused. A fixed allowance covers `input`, so where a run is stored never changes the limits. Every cut is recorded in `truncation` with the original size and SHA-256. Output is canonical JSON (sorted keys, no spaces).
- Refs: `target`, `source`, `source:L<n>`, `candidate:L<n>`, `tests:L<n>`, `issue:final:<i>`, `attempt:<n>`, `attempt:<n>:issue:<i>`, `probe:<name>:<i>` (each finding's `id`). A probe ref that does not resolve in the bounded bundle (including the cut last line of a truncated text) is dropped and counted. Candidate refs come from the YAML node of the item, not a text search.
- `evidence_sha256`: SHA-256 of the canonical JSON without `input` (where the files were read from) and the digest itself.

Probes and what they read:

| Probe | Reads | Finding statuses |
|---|---|---|
| gate | `evaluate_pre_encode_gate` in report mode on the recorded source text, with the row's labels (attested version only) from the local corpus; `existing_module` runs on a sparse copy of the RuleSpec state. `unencoded_reference` is not run: it needs the run's release-bound active corpus rows, which a local checkout cannot establish (the corpus probe reports both of its facts per reference) | one finding per gate finding, with `enforceable`; `checks_not_run` with reasons |
| registry | every `imports` item, every `deferred_outputs[].blocked_by` entry and every free formula name the packaged concept registry knows (its producer anchors), resolved with `_rulespec_import_file_candidates_static` / `_resolve_rulespec_import_file_static` on a sparse copy of the RuleSpec state, exports by `_rulespec_rule_names_from_content` | `exported`, `symbol_missing`, `module_missing`, `module_present`, `module_unreadable` (listed at the ref but not readable, a tracked symlink, or not UTF-8; never reported as missing), `unresolvable_target`, `no_producer_registered`, `not_checked`; plus `in_workspace_context` and `authored_by_run` |
| corpus | the target, explicit USC/CFR references in the source (the gate's `explicit_federal_references`) and the candidate's `corpus_citation_path` values, looked up in the local corpus checkout (not a verified release) | `present`, `repealed`, `absent`, `absent_at_attested_version` (rows exist, but not at the version the run read; another version is never substituted), `present_multiple_versions`, `unavailable` (part of the citation's bucket could not be read or parsed, so absence is unknown); for each source reference, `rulespec_module` (`present`, `absent`, `unreadable`, `not_checked`) at the run's ref; and whether the attested provision version exists locally |
| import | each final compile error's module path | `own_module`, `authored_sibling`, `workspace_context_module`, `imported_not_authored`, `unlocated` |
| persistence | per-attempt issues (local runs only: workflow bundles record the selected candidate's issues alone), compared by whole normalized signature | `persistent` (only with the complete history: every attempt recorded with its whole issue list), else `in_all_recorded`; `changed`; `history_complete`, `persistent_tags`, `tags_in_all_recorded` |

Relation to other work: #1660 persists per-attempt evidence on the write side (`validation-issue/v1`, artifact versions); this package only reads what a run already left behind and will read #1660's structured issues once they exist. #1987 types validator issues at the source; the text-level `tag` here is the stopgap until then.

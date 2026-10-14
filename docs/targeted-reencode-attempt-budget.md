# Targeted re-encode attempt limits

The ad hoc re-encode guard blocks after three consecutive failed model-bearing
runs by default. Triage the actual failure before granting another attempt;
regeneration alone does not fix a validator or source defect. Queue dispatches,
authenticated repair replays and existing repository-wide overrides retain their
established behavior for this numeric limit. The repeated-failure rule below
also covers repair replays. This is a cost limit, not a signing or validation
waiver.

For a reviewed retry that must leave concurrent citations at their existing
limits, set the repository Actions variable `ATTEMPT_BUDGET_BY_CITATION_JSON`
to an object keyed by exact, case-sensitive corpus citation paths. For example:

```json
{
  "de/statute/estg/78": {
    "budget": 4,
    "expires_at": "2026-09-10T00:00:00Z",
    "reason": "Reviewed dependency-link fix in axiom-encode#1588"
  }
}
```

The budget is a positive JSON integer, the expiration is an ISO datetime with a
timezone, and the triage reason is a nonempty string. Set an appropriate future
expiration when applying the example. A limit of four allows a fourth attempt
following three consecutive failures; a fourth failure blocks again. It does
not put the selected citation into report-only mode. Unlisted citations,
including descendants or similar spellings, keep the repository-wide limit.

At the expiration instant the selected entry stops applying. Malformed selected
entries or malformed JSON retain the repository-wide limit with a warning;
unrelated entries do not affect a citation. Remove temporary entries after the
reviewed retry. Preserve other operators' entries when changing the variable.

`ATTEMPT_BUDGET` still sets the default numeric limit. The existing
`ATTEMPT_BUDGET_OVERRIDE=true` remains a repository-wide report-only switch;
it takes precedence over numeric limits and should not be used for an isolated
retry while another citation is being worked on. All protected environment,
source, behavior, completeness and signature gates remain in force.

## Repeated failures with the same rules

The same unprotected job also refuses an ad hoc dispatch, repair replays
included, when the two most recent consecutive model-bearing failures for the
citation ended with the same final rule-code set. The guard reads each run's
failure artifact (`targeted-reencode-failure-<run>-<attempt>`) and collects the
bracketed validator rule codes, such as `[complete-source-unit:tests]`, from
every lane's `final-rejected-candidate/issues.json`. A run without that
artifact, or with any final issue that names no rule code (a YAML parse,
compile or expected-value error), has no known set and never triggers the
rule. A success resets the streak, and runs that never reached the encode step
stay neutral, as for the numeric limit. API or artifact trouble fails open.

Another attempt rarely clears the same validator rules. Fix the generator side
(prompt, repair guidance or deterministic repair), or a validator rule that is
wrong, or give a review finding that addresses those rules. Then, to allow the
retry, add an expiring entry for the exact citation to the repository variable
`REPEATED_FAILURE_OVERRIDE_JSON`:

```json
{
  "us/regulation/7/273/4": {
    "expires_at": "2026-10-20T00:00:00Z",
    "reason": "Reviewed paired-test repair guidance before one more retry"
  }
}
```

Malformed JSON or a malformed selected entry keeps the rule with a warning.
`ATTEMPT_BUDGET_OVERRIDE=true` also lifts it. Queue dispatches report only.
Over the US runs dispatched since 2026-09-01, the rule would have refused 33 of
292 runs: 32 of them failed, and 1 opened a draft PR that has not merged.

## Retaining a primary while repairing dependents

A failed single-target run can supply the retained primary for a later request
with one or two direct dependents. Keep the primary citation, replacement path,
corpus and engine identities bound to the prior artifact. Leave source/test
bundles, refreshes, legacy moves and queue inputs empty. A second dependent
requires a first dependent. The workflow first applies the existing strict
target-artifact checks. If they fail with two dependents requested, the request
stops. With one dependent, it may try the existing strict dependent-artifact
checks. The extractor's transaction identity checks are unchanged.

After a candidate authenticates, failures in candidate validation or RuleSpec
base-advance verification terminate the request; they never select another
lane. The current direct-dependency inventory is still checked before model
execution. Retained bytes go only to the selected primary; each newly requested
dependent receives its own source-complete native generation and validation.
Diagnostics record the prior run, selected candidate lane and hashes separately
from the requested dependent citations. This does not certify the new dependents
as part of the old artifact or waive any signing or publication check.

## Encoder pin admission

A signed RuleSpec pull request lands only when the target repository's CI
admits the encoder that signed it. That CI runs the encoder pinned in
`.axiom/workflow-toolchain.toml` (`axiom_encode_ref`)
(`docs/signed-manifest-encoder-admission.md`):

| Pin | RuleSpec CI admits |
| --- | --- |
| Older than axiom-encode #1859 | The exact pin only |
| Contains #1859 but not #1883 | Nothing: the guard cannot run its Git checks under the verification runtime, so every changed signed manifest fails, the exact pin included |
| Contains #1883 | The exact pin, or a producer that descends from the pin on the first-parent history of encoder `main` |

The unprotected `Check RuleSpec encoder pin admission` job makes the same
decision before the `production-signing` environment asks for approval and
before any model call, for queue and ad hoc runs. It reads the pin from the
target repository at `rulespec_ref` and the dispatch commit's history from a
full axiom-encode checkout (`scripts/check_encoder_pin_admission.py`). A run
that opens a pull request (`open_pr: true`) and whose encoder the pin cannot
admit fails there, and the encode job is skipped. An artifact-only run is
reported and not blocked. A refused run does not count against the
failed-attempt budget, and the SNAP dispatch queue neither counts it as an
attempt nor dispatches again against the same RuleSpec tip for a day. The
protected signed queue (`scripts/prepare_signed_queue.py`) does not yet tell a
refusal apart: its item becomes `failed-workflow-run:failure` until a person
bumps the attempt, so move the pin before resuming such a queue.

To unblock, move the target repository's `axiom_encode_ref` to an admissible
encoder first: today, a commit that contains #1883 (`b5e85ab4`, 0.2.2187) or a
later `main` commit. To generate anyway, for example while a pin bump is in
review, set the repository variable `ENCODER_PIN_ADMISSION_OVERRIDE=true`; the
check then reports without blocking. Remove it afterwards. A target repository
that declares no `axiom_encode_ref` is reported and not blocked. Missing
history, an unknown pin or an unreadable toolchain blocks the run.

The job reads the target repository with the read-scoped workflow token, not
the protected repository token, so it needs the RuleSpec repository to be
public, as every `rulespec-*` repository is today. A private one fails the
check and blocks its runs.

## Superseded encoder PRs

Each run opens one draft RuleSpec PR on `axiom/signed-backfill-<country>-<run>-<attempt>`
titled `Add signed encoding manifest for <citation>`. Retries leave the older
drafts open. `scripts/close_superseded_encoder_prs.py` lists them and plans to
close a draft when a newer open encoder PR names the same citation on the same
base branch, or when the default branch holds a signed manifest for the
citation that was generated after the draft's own. It keeps a PR that is ready
for review, that has a commit not both authored and committed by
`axiom-encode-bot`, whose branch is the base of an open PR that stays open (a
stacked repair), or whose RuleSpec CI passes while the newer PR's does not.
Each line of the plan shows the CI state of both PRs.

The script is a dry run by default. Read the plan first:

```bash
uv run python scripts/close_superseded_encoder_prs.py --country us
```

Then run it again with `--apply --only <numbers>`, naming the PRs you reviewed,
to close each one and comment why. A number that the new plan no longer closes
is skipped. It never deletes branches, so a closed PR can be reopened. The
encode workflow does not run it: at publish time RuleSpec CI has not judged the
new PR yet, so the choice between two drafts belongs to a person.

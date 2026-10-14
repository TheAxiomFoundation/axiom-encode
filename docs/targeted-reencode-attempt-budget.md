# Targeted re-encode attempt limits

The ad hoc re-encode guard blocks after three consecutive failed model-bearing
runs by default. Triage the actual failure before granting another attempt;
regeneration alone does not fix a validator or source defect. Queue dispatches,
authenticated repair replays and existing repository-wide overrides retain their
established behavior. This is a cost limit, not a signing or validation waiver.

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
`.axiom/workflow-toolchain.toml` (`axiom_encode_ref`). A pin admits its exact
commit. A pin that contains axiom-encode #1859 also admits a producer that
descends from the pin on the first-parent history of encoder `main`
(`docs/signed-manifest-encoder-admission.md`).

The unprotected `Check RuleSpec encoder pin admission` job makes the same
decision before the `production-signing` environment asks for approval and
before any model call, for queue and ad hoc runs. It reads the pin from the
target repository at `rulespec_ref` and the dispatch commit's history from a
full axiom-encode checkout (`scripts/check_encoder_pin_admission.py`). A run
whose encoder the pin cannot admit fails there, and the encode job is skipped.
A refused run does not count against the failed-attempt budget.

To unblock, move the target repository's `axiom_encode_ref` to an admissible
encoder first. To generate anyway, for example while a pin bump is in review,
set the repository variable `ENCODER_PIN_ADMISSION_OVERRIDE=true`; the check
then reports without blocking. Remove it afterwards. A target repository that
declares no `axiom_encode_ref` is reported and not blocked. Missing history, an
unknown pin or an unreadable toolchain blocks the run.

Known gap: the check mirrors the admission rule, not every CI defect. Draft
axiom-encode #1879 reports that pins from #1859 up to that fix cannot run the
ancestry proof under the supervised Git wrapper, so their CI rejects every
changed signed manifest. The check does not model that window.

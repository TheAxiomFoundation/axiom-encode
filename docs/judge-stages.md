# Judge stages

The judge stages are the model-backed checks that run beside the deterministic
gates. They live in `src/axiom_encode/judges/`, share one event shape
(`JudgeEvent` in `judges/run_log.py`, mapped onto the canonical
`axiom_encode.run_log.v1` run log as a `judge` stage event) and one contract: a
judge that cannot reach a verdict emits a `judge_error` event and never a silent
pass (`error_event` is the only way to build one, and `validate_event_dict`
rejects an error without its cause). This page documents the statutory-fidelity
screen in depth and the other stages briefly. Every mechanism claim below points
at code or a test.

## Stages at a glance

| Stage | `judge_stage` | Runs on | Emits |
|---|---|---|---|
| Statutory-fidelity screen | `statutory_fidelity_screen` | TypeSafe System One (family `typesafe`) | Advisory verdict, per-kind probabilities, a cascade decision |
| Statutory-fidelity referee | `statutory_fidelity` | Claude (family `anthropic`) | Advisory verdict with located findings; the `needs-review` label |
| Grid adequacy | `grid_adequacy` | Claude | Untested boundaries as gaps with follow-up cells |
| Disposition | `disposition` | Claude plus arithmetic | Whether a causal claim reproduces on sampled records |
| Worklist pre-classifier | `worklist_preclassify` | Heuristics plus a Claude arbiter | Route to generate or skip-with-reason |
| Golden drift | `golden_drift` | Packaged regenerator | Drift between merged and regenerated modules |

The cross-family rule (`model_family` and `cross_family_problem` in
`judges/client.py`) refuses a judge whose model family matches the generator's
and refuses any model whose family it cannot classify. Families are
`anthropic`, `openai`, `google` and, for the screen, `typesafe` (model ids
starting with `jev`). Every successful event records the judge model, the
generator model and token spend. An error event always records its cause and
records the model only when one responded.

## Statutory-fidelity screen

The screen (`judges/statutory_fidelity_screen.py`) runs before the
statutory-fidelity referee on the same inputs: the provision window, truncated
with `truncate_provision` exactly as the referee truncates it, and the generated
RuleSpec artifact. It calls TypeSafe System One through
`judges/system_one.py`, a typed-question client that sits beside the referee's
`JudgeClient`. TypeSafe describes its training as reinforcement learning for
calibrated decisions; reward and data undisclosed.

### What it asks

One `Choice` and five `Noul` questions, with the state keys and question text
verbatim from the 2026-09-17 pilot (`build_state` and `build_questions`):

- `verdict`: a pass or flag choice, with a probability per label and a
  reported confidence.
- One `Noul` per fidelity finding kind: `unrepresented_clause`,
  `untraceable_branch`, `boundary_direction`, `amount_mismatch`. Each returns a
  probability in [0, 1].
- `faithful`: the pilot's overall faithfulness probability, recorded only.

Changing the state shape or the question text invalidates the pilot evidence
behind the placeholder threshold, so the event records
`screen.questions_version` and `judge_prompt_sha256` (a digest of the state
and questions actually sent).

### What it records

The event carries `judge_stage = statutory_fidelity_screen`, `advisory =
true` (hard-wired; the screen has no promoted form), the responding model id
(`jev-1.13.0` in the pilot), `judge_model_family = typesafe`, token usage, and
under `attrs.screen`: the policy mode and thresholds, every probability,
latency in milliseconds, and the cascade decision. On an error the event
carries the policy, the questions version, latency and the cascade decision,
but no probabilities and, unless a model responded, no model id.

Findings carry a kind and a probability and nothing else. System One returns
no clause reference, no rule path and no explanation, so both locators are
left empty rather than fabricated, the explanation says so, and the probability
lands in the canonical finding's `evidence` field as `probability=0.9700`. A
finding is emitted only for a kind whose probability reaches its configured
threshold. The verdict is `flag` when the choice is flag or any finding was
emitted, `pass` otherwise, and `error` when the client failed for any reason.

The screen never produces the `needs-review` label; `needs_review_label`
in `judges/statutory_fidelity.py` answers only for the referee's stage.

### Cascade policy

`ScreenPolicy` holds a mode and per-kind thresholds. `cascade_decision`
turns a screen event into a `CascadeDecision` that says whether the referee
should run and why:

| Situation | Referee | Reason |
|---|---|---|
| Screen errored | requested | `screen_error` |
| A configured kind has no probability | requested | `screen_incomplete` |
| Mode `advisory` (default) | requested | `advisory_mode` |
| Mode `cascade`, some kind at or above its threshold | requested | `threshold_exceeded` |
| Mode `cascade`, every kind below its threshold | skipped | `below_threshold` |

By default only the two kinds the pilot validated carry a threshold
(`amount_mismatch`, `boundary_direction`). The other two are recorded but never
reported as findings or used by the cascade unless a per-kind threshold is
configured. Cascade mode refuses a policy with no thresholds at all
(`ScreenPolicy.__post_init__`), since that would skip the referee on every
artifact.

### Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TYPESAFE_API_KEY` | unset | Required. Missing key is a fail-closed `missing_api_key` error. Never printed or logged. |
| `AXIOM_JUDGE_SCREEN_MODE` | `advisory` | `advisory` or `cascade`. |
| `AXIOM_JUDGE_SCREEN_THRESHOLD` | `0.25` (placeholder) | Threshold for the cascade kinds. |
| `AXIOM_JUDGE_SCREEN_THRESHOLD_<KIND>` | unset | Per-kind override; also enables a record-only kind (for example `AXIOM_JUDGE_SCREEN_THRESHOLD_UNREPRESENTED_CLAUSE`). |
| `AXIOM_JUDGE_SCREEN_MODEL` | unset (SDK default `jev-latest`) | Pin a System One model id. The responding id is recorded regardless. |
| `AXIOM_JUDGE_SCREEN_TIMEOUT_SECONDS` | `30` | Per-request timeout. |
| `AXIOM_JUDGE_SCREEN_MAX_RETRIES` | `2` | SDK-side retries on transient failures. |
| `AXIOM_JUDGE_PROVISION_CHARS` | `24000` | Provision window, shared with the referee. |
| `AXIOM_GENERATOR_MODEL` | `gpt-5.6-terra` | Generator whose family the guard compares against. |

The `typesafe` extra pins the SDK: `pip install axiom-encode[typesafe]`
installs `typesafe-sdk==0.6.0`. The SDK is imported lazily, so the package and
its tests work without it; a missing SDK is a fail-closed `sdk_missing` error.

### Running it

```bash
axiom-encode judge-fidelity-screen --root rulespec-us --corpus-path axiom-corpus \
  --corpus-citation-path us/statute/26/32 --rule-file rulespec-us/us/statutes/26/32.yaml --json
```

```bash
AXIOM_JUDGE_SCREEN_MODE=cascade axiom-encode judge-fidelity --screen \
  --root rulespec-us --corpus-path axiom-corpus \
  --corpus-citation-path us/statute/26/32 --rule-file rulespec-us/us/statutes/26/32.yaml \
  --run-id <encode-run-id> --json
```

With `--screen`, `judge-fidelity` prints one JSON object with `screen`,
`cascade` and `referee` (null when the cascade skipped it). Both events append
to the run log when `--run-id` is given; the referee event also carries the
cascade decision under `attrs.screen_cascade`. `--screen-mode` overrides the
environment for one run.

### Placeholder threshold and the evidence behind it

The default threshold of 0.25 is a placeholder, marked as such in
`constants.py`. It comes from the 2026-09-17 pilot (scripts, raw results and a
README under `_axiom-runs/jev-judge-pilot-2026-09-17/` in the foundation
mirror), which planted one defect in each of 30 known-good artifacts per kind
and scored the original and mutated artifact against the same provision:

| Planted defect | Jev's kind probability rose | Mean rise | AUC mutated vs original |
|---|---|---|---|
| Amount changed (number also in the provision) | 30 of 30 | +0.75 | 1.000 |
| Boundary flipped (`>=` to `>`, `<=` to `<`) | 30 of 30 | +0.41 | 0.953 |
| One `and` conjunct dropped | 21 of 30 | +0.09 | 0.603 |

Operating points read off the same 180 calls. The cascade rule sends an
artifact to the referee when its amount or boundary probability reaches the
threshold; the 90 originals are clean artifacts. The last column applies the
boundary probability alone, to show what the boundary question by itself sees.

| Threshold | Clean sent to referee | Planted amounts caught by the cascade | Planted boundaries caught by the cascade | Boundary probability alone |
|---|---|---|---|---|
| 0.20 | 43 of 90 | 30 of 30 | 30 of 30 | 29 of 30 |
| 0.25 | 32 of 90 | 30 of 30 | 30 of 30 | 29 of 30 |
| 0.30 | 25 of 90 | 30 of 30 | 28 of 30 | 28 of 30 |
| 0.50 | 10 of 90 | 30 of 30 | 22 of 30 | 22 of 30 |

0.25 is the lowest value at which the cascade still skips most clean artifacts
while sending every planted amount and boundary defect to the referee. The
boundary probability alone misses one flip (scored 0.16); the cascade still
sends that artifact on its amount probability (0.29). The value is
deliberately low because a skipped referee on a real defect costs far more
than a referee call.

The calibration harness in `judges/calibration.py` cannot set this value: its
labels are apply and blocked outcomes of the compile and CI gates, which a
reader of provision plus artifact cannot see, and both the screen and the
referee were near chance on them (Jev AUC 0.546). The verifier track
(TheAxiomFoundation/axiom-encode#1657, landing under `benchmarks/verifier/`)
is the authority for the real threshold.

### Live check on real generations

On 2026-09-17 the shipped stage ran live on three `gpt-5.6-terra` generations
with outcome `apply_applied` from `encodings.db` (script, results and run logs
under `_axiom-runs/jev-prescreen-live-2026-09-17/` in the foundation mirror).
The responding model was `jev-1.13.0`. The three calls used 15,196 input
tokens, about 0.06 cents at the published price, with latencies of 394 to 467
milliseconds. Every event validated, each was appended to a run log, and
neither the key nor an authorization header appeared in anything written.

| Citation | Choice verdict (confidence) | Amount | Boundary | Unrepresented clause | Cascade at 0.25 |
|---|---|---|---|---|---|
| `us-mn/statute/290.0661` | flag (0.41) | 0.12 | 0.14 | 0.96 | referee skipped |
| `us-co/regulation/9-ccr-2503-5/3.544` | flag (0.94) | 0.44 | 0.32 | 0.93 | referee requested |
| `us-il/statute/35/5/201` | pass (0.27) | 0.12 | 0.10 | 0.96 | referee skipped |

All three merged generations scored 0.93 or higher on `unrepresented_clause`.
That matches the pilot, where the kind did not separate clean from defective
artifacts, and it is why the kind stays record-only. Three cases are a wiring
check, not evidence for a threshold.

### Limits

- The screen cannot see a dropped conjunct (AUC 0.603), so a cascade that
  skips the referee is blind to that kind. This is why cascade mode is off by
  default and why the kind carries no threshold.
- The pilot's defects are synthetic single edits, 30 per kind, with amounts
  chosen so the original number appears verbatim in the provision. Real
  defects may score differently.
- The binary choice verdict flagged about half of clean originals in the
  pilot and is not used by the cascade; only the per-kind probabilities are.
- The screen returns no location, so it cannot replace the referee's located
  findings; it can only decide whether to request them.

### Cost and latency

In the pilot, 180 calls used 535,243 input tokens with a median latency of
0.19 seconds. At the price published at the time (0.042 dollars per million
input tokens, output free) that is about two cents for the whole pilot, against
a median of 4.5 seconds per call for the Haiku referee.

### What the verifier track must decide

- Whether the cascade may skip the referee at all, given the screen's
  blindness to dropped conjuncts and any other kinds the real-defect corpus
  surfaces.
- The threshold per kind on real defects, replacing the placeholder, and
  whether `unrepresented_clause` or `untraceable_branch` ever earn one.
- Whether the choice verdict or the `faithful` probability should join the
  cascade trigger.
- Whether the referee should run only on the triggered kinds, which would need
  a narrower referee prompt.

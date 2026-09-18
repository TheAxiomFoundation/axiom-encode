# EncodeBench verifier track

How to measure fidelity *judges* against each other on a fixed set of
(provision, artifact) cases with known defects, and how to read the result.
The encoder track (`docs/encodebench.md`) scores the models that write
RuleSpec; this track scores the models that check it. Code lives under
`benchmarks/verifier/`; nothing here changes the encoder or the judges
package.

## Why a judge benchmark

The statutory-fidelity referee (`src/axiom_encode/judges/statutory_fidelity.py`)
is wired advisory: a `flag` adds a `needs-review` label and never gates. Its
calibration harness (`judges/calibration.py`) labels historical generations
good or bad by the recorded apply outcome (`apply_applied` versus
`apply_blocked_validation`). Those labels are compile and CI outcomes that a
reader of provision plus artifact cannot see, so they cannot validate a
fidelity judge. The 2026-09-17 pilot (`_axiom-runs/jev-judge-pilot-2026-09-17`)
showed both the Haiku referee and TypeSafe's Jev near chance on those labels
(Haiku flagged 31 of 40 known-good and passed 16 of 40 known-bad; Jev 25 and
19), while on planted single-edit defects Jev's kind-specific probability rose
in 30 of 30 changed amounts and 30 of 30 flipped boundaries, and 21 of 30
dropped conjuncts. Ground truth by construction is the only label a fidelity
judge can be measured against, and that is what this track supplies.

## What it is

- **Case sources**, both first class:
  - `synthetic`: a versioned, seeded mutator (`mutator.py`, `MUTATOR_VERSION`)
    plants one defect in a known-good artifact. Every defective case ships
    with its unmodified original as the control. Known-good artifacts come
    from the encoder track's own gate passes on the pinned UK release
    (`sources/eval_suite.py`, reading canonical `eval-suite` output through
    the board's strict loader) or, until those exist locally, from
    `encodings.db` `apply_applied` generations (`sources/encodings_db.py`,
    opened read-only).
  - `real`: recorded repair rounds under `benchmarks/verifier/real_defects_v0/`
    (produced by another session; this code only reads it). The pre-fix
    artifact is the defective case, the post-fix artifact is the control, and
    every real control is marked `unverified` because a repair round makes an
    artifact better, not proven clean.
- **Defect kinds**: `amount_changed` (a number that also appears verbatim in
  the provision window), `boundary_flipped` (`>=` and `>`, `<=` and `<`, both
  directions), `conjunct_dropped` (one `and` conjunct deleted),
  `polarity_swapped` (`and` for `or` or the reverse), `date_or_period_wrong`
  (a version's effective date moved a year, or a rule's period changed),
  `entity_wrong` (a rule's entity replaced).
- **Judges** (`judges/`), one interface: the incumbent referee on one Claude
  model (`referee:<model>`, the statutory-fidelity prompt and schema imported
  unchanged from the judges package and sent through the repo's own
  `JudgeClient`, escalation pinned off); Jev through `typesafe-sdk`
  (`jev:jev-1.13.0`, one `Choice` pass/flag verdict plus one `Noul` per
  defect kind); and a `replay:<file>` runner for tests and re-folds.
- **Board** (`board.py`): folds one `results.json` per runner into a
  leaderboard and refuses inputs whose suite digest, case identities or
  runner names do not line up, mirroring `harness/eval_board.py`.

## How the mutator works, and what it never touches

Both members of a pair are produced by parsing the artifact, editing the
parsed tree (only for the defective member) and re-serialising both through
one canonical dumper. Consequences:

- the two artifacts differ in exactly one leaf (a test asserts this for every
  kind, and the full-pool dry run over 1,421 distinct citations found zero
  multi-leaf diffs);
- the defective artifact is always well-formed YAML (a line-splice mutator,
  which the pilot used, can cut a quoted multi-line formula in half);
- formatting is normalised for both members, so the judged text is not
  byte-identical to what the encoder emitted. The control is the encoder's
  content, not its bytes.

Edits by kind: the four formula kinds edit only `formula` and `value` strings
under `rules`. `date_or_period_wrong` edits `rules[i].versions[j].effective_from`
or `rules[i].period`, the two fields that are the effective date and the
period; it never touches proof excerpts, source hashes, citations,
`source_verification`, or any other metadata date. `entity_wrong` edits
`rules[i].entity`. Detectability guards make each planted defect visible to a
reader of the window plus the artifact: an amount must equal a number the
window states (numeric equality on whole numbers, so `60000` matches
`$60,000.00` but `200` does not match inside `2008`) and its replacement must
not; an effective date only moves when the window states the original year as
a word of its own and not the shifted one; a period or entity only changes
when the window mentions the original and not the replacement. Year-like
numbers are never treated as amounts. Conjuncts are only dropped from pure
conjunctions: a formula with a top-level `or` or an `if`/`else` is left
alone, because deleting the text between two `and` tokens there would remove
more than one condition. Each source artifact is used for at most one pair,
and kinds are filled by deficit so the rarer sites get first pick.

Mutator 1.0.0 matched amounts and years by substring. Two of the 180 pairs it
built were undetectable for that reason (`200` found only inside `2008`, `11`
only inside `3211(b)`); 1.0.1 closes the gap, and the first board drops those
two pairs by a recorded filter rather than re-spending on a rebuild.

The provision window is the referee's own truncation
(`truncate_provision`, 24,000 characters, head and tail kept). Guards are
checked against the window, and both judge families receive the identical
window.

## Metrics

Every judge output is normalised to a verdict (`pass`, `flag`, `error`), a
verdict score (probability the artifact is defective) and one score per
defect kind:

- referee: the verdict is the production verdict, derived exactly as
  `statutory_fidelity.run` derives it (a raw pass that still lists findings
  is a flag, because the incumbent's contract says a faithful artifact gets
  an empty findings list; a test asserts the two agree payload for payload).
  The verdict score is the self-reported confidence when the verdict is a
  flag and one minus it when it is a pass. A coerced verdict's confidence is
  ambiguous, so the row records the raw verdict and the board counts
  coercions per judge (the `coerced` column) instead of guessing. The kind
  score is 1 when a finding names the mapped referee kind (`amount_changed`
  to `amount_mismatch`, `boundary_flipped` to `boundary_direction`,
  `conjunct_dropped` to `unrepresented_clause`, `polarity_swapped` to
  `untraceable_branch` or `unrepresented_clause`) and 0 otherwise; a finding
  kind outside the referee's four is recorded and never credited. The referee
  has no question about dates, periods or entities, so for
  `date_or_period_wrong` and `entity_wrong` the kind score falls back to the
  verdict score and the board marks those cells, and the mean, with ‡.
- cross-family guard: production refuses to judge an artifact whose
  generator shares the judge's family. That is a pipeline policy, not a
  measurement rule. The referee runner builds its client with the repo's
  declared generator so the guard is satisfied, and every row records the
  case's true generator and whether it shares the judge's family; the board
  counts same-family rows per judge. Cases whose generator is unrecorded
  carry `null` there.
- Jev: verdict score is the `Choice` probability of `flag`; kind scores are
  the six `Noul` probabilities. Jev returns no clause reference, rule path
  or explanation, so its findings list is empty and its localization column
  is blank by construction.

Per judge and per kind the board reports: detection AUC (pooled
Mann-Whitney, defective versus control within the kind, ties count one half),
paired rise rate and mean paired delta on complete pairs, detection at the
false-alarm ceiling (share of defective cases scoring above the control score
that admits at most the ceiling's share of controls), native detection rate
(flag rate on defective cases), native false-alarm rate (flag rate on clean
controls), localization rate (a finding names the mutated rule by name or
index, or mentions the edited token), errors, median latency, mean tokens in
and out, and cost per case where a published price is recorded in
`benchmarks/verifier/pricing.json`. Every price entry names its source;
models without an entry render cost as blank. Anthropic prices come from the
claude-api skill's model table, never from memory; Sonnet 4.5 has no row
there and so no cost.

**Headline**: per-kind detection AUC on the judge's kind channel, subject to
a false-alarm ceiling on the judge's native verdict. The default ceiling is
25 percent (`--false-alarm-ceiling`), stated on every board. A judge whose
native flag rate on clean controls exceeds it is shown but not ranked (†);
a judge with no scored controls, or with a kind that has no AUC at all, is
unrankable (§) and shown last. The mean AUC that ranks judges is only
computed when every kind in the suite has an AUC, so a judge is never ranked
on a five-kind mean against six-kind means; a native-only mean is reported
beside it in the JSON output. The ceiling is there because a judge that
flags everything has perfect recall and no use; the pilot found both
incumbents flagging most clean originals at the verdict level, which is
exactly the fact the headline must not hide. The ceiling is applied only
when the suite's controls are gate-verified; on a real-defects suite, whose
controls are post-fix artifacts not proven clean, the board says so and
unranks no one for flagging them. Defect kinds outside the synthetic
taxonomy (`other:<kind>` from a real corpus) get their own columns, scored on
every judge's verdict channel. Tokens, latency and cost cover every call,
errors included.

An `error` verdict (API failure, parse failure, cross-family guard) is never
a pass and never a score: it is counted, excluded from AUC and pairs, and
makes the run incomplete. The board refuses incomplete runs unless
`--allow-partial` records the fact.

## Running it

Prerequisites: the repo environment plus `anthropic` (for the referee) and
`typesafe-sdk==0.6.0` (for Jev). Keys are read from the environment by the
SDKs; never print or store them.

```bash
export ANTHROPIC_API_KEY="$(agent-secret get ANTHROPIC_API_KEY)"
export TYPESAFE_API_KEY="$(agent-secret get TYPESAFE_API_KEY)"
```

Build a synthetic suite. From the encoder track's outputs on the pinned UK
release (the primary source once those outputs exist locally):

```bash
uv run python benchmarks/verifier/verifier.py build-synthetic \
  --from-eval-suite results/capability-v1/terra results/capability-v1/fable \
  --per-kind 30 --seed 7 --name "EncodeBench verifier synthetic UK v1" \
  --out _axiom-runs/encodebench-verifier/synthetic_uk_v1
```

Or from the local run log (the fallback used for the first board):

```bash
uv run python benchmarks/verifier/verifier.py build-synthetic \
  --from-encodings-db encodings.db --generator-model gpt-5.5 \
  --per-kind 30 --seed 7 --name "EncodeBench verifier synthetic US v1" \
  --out _axiom-runs/encodebench-verifier/synthetic_us_v1
```

`build-synthetic` writes `suite.json` (full texts) and `suite.manifest.json`
(identities and digests only, small enough to commit). Load the real corpus
with `build-real --dir benchmarks/verifier/real_defects_v0 --out ...`.

`encodings.db` holds generations for several jurisdictions. To restrict a
built suite, derive a child suite by citation prefix and, where a case is
later found unfair, by pair id with a stated reason; pairs are kept or
dropped whole, and the child records its parent's digest, the filter, the
reason and every dropped pair:

```bash
uv run python benchmarks/verifier/verifier.py filter-suite \
  --suite _axiom-runs/encodebench-verifier/synthetic_us_v1 \
  --drop-citation-prefix uk/ be/ \
  --drop-pair amount_changed-cb832034 amount_changed-c1803dd0 \
  --reason "US set only; two amount pairs fail the 1.0.1 numeric-equality guard" \
  --name "EncodeBench verifier synthetic US v1 (US only, guard-checked)" \
  --out _axiom-runs/encodebench-verifier/synthetic_us_v1_final
```

A filter must never depend on judge outputs. Rows already judged against the
parent fold into the child without re-judging: point `run` at the child suite
and the same `--out` directory, and it re-assembles `results.json` from
`cases.jsonl`, re-stamping row positions. Parent-suite and child-suite runs
carry different digests and never fold together.

Run each judge into its own output directory (resumable; rows land in
`cases.jsonl` as they finish and error rows are retried on resume):

```bash
uv run python benchmarks/verifier/verifier.py run \
  --suite _axiom-runs/encodebench-verifier/synthetic_us_v1 \
  --judge referee:claude-haiku-4-5-20251001 --name haiku \
  --out _axiom-runs/encodebench-verifier/runs/haiku --workers 4
```

```bash
uv run python benchmarks/verifier/verifier.py run \
  --suite _axiom-runs/encodebench-verifier/synthetic_us_v1 \
  --judge jev:jev-1.13.0 --name jev \
  --out _axiom-runs/encodebench-verifier/runs/jev --workers 6
```

Fold the board:

```bash
uv run python benchmarks/verifier/verifier.py board \
  _axiom-runs/encodebench-verifier/runs/haiku _axiom-runs/encodebench-verifier/runs/jev \
  --false-alarm-ceiling 0.25 --markdown-out board.md --json-out board.json --csv-out board.csv
```

The board refuses two inputs whose suite sha256 differs (the digest binds
the suite name, source kind, corpus release, mutator version, provision
window size, derivation and the ordered case identities with their content
digests and generator), two inputs naming the same runner, two inputs whose
runner identity (family, model, prompt or question-set digest, window) is the
same under different names, and any incomplete run without `--allow-partial`.

### What a run directory guarantees

- Every row in `cases.jsonl` is bound to the case's provision and artifact
  digests and to a digest of the runner's identity. Resume reuses a row only
  when all three match the current suite and runner; rows judged by another
  judge, under another prompt, or against a rebuilt artifact are ignored and
  the case is judged again. Filtered child suites keep their parent's content
  digests, so they re-assemble without re-judging.
- `results.json` carries a payload digest over every section except its
  timestamp, recomputes the suite digest from its own case identities, and
  derives its coverage counters from its rows. A hand edit to the runner
  name, the price source, the case list or a row is refused at load.
- Errors never become passes. A runner that raises, an SDK that is missing
  or unkeyed, a response without a verdict or without every kind's score, a
  served model other than the pinned one, and an unparseable confidence are
  all recorded as error rows with their cause, retried on the next run, and
  keep the run incomplete until they clear.
- An interrupt (Ctrl-C) cancels the queued cases, writes what finished, and
  exits 130; re-running the same command resumes. `--fresh` rotates the old
  `cases.jsonl` and `results.json` to `.bak` files rather than deleting them.
  `--limit` judges only the first N cases but never downgrades a finished run.
- Cost is recomputed at assembly from each row's reported tokens and the one
  price the payload names; a row whose usage the provider did not report is
  unpriced and counted as such, never charged zero.

## Self-agreement (test and retest)

A judge that changes its verdict on the same text between two calls is noisy
in a way no detection AUC shows. Two runs of one judge are compared on cases
whose provision and artifact digests are identical (joined on content, never
on case ids, so runs over different suites still compare where they share
text):

```bash
uv run python benchmarks/verifier/verifier.py agreement \
  _axiom-runs/encodebench-verifier/runs/haiku _axiom-runs/encodebench-verifier/runs/haiku-retest
```

It reports the number of identical texts judged in both runs, verdict
agreement, identical finding-kind sets (for judges that return findings), and
the median and maximum absolute change in verdict score and in each kind
score. It refuses to compare two different judges, and says so when the two
runs' identities differ (a different prompt or window is a comparison of two
configurations, not a retest).

## Adding a judge

Implement the `JudgeRunner` protocol in `judges/` (a `name`, `family`,
`model`, `supports_localization`, an `identity()` dict of score-affecting
configuration, and `judge(case) -> JudgeResponse`), register its spec prefix
in `judges/__init__.py`, add a price row with a source to `pricing.json` if
one is published, and run it against the same suite. Kind scores must be
probabilities of defect in [0, 1]; a judge without a kind-specific answer
must fall back to its verdict score and say so through
`kind_score_channels`, never invent one.

## Changing the suite

Any change to the mutator's candidate selection, arithmetic, guards or the
canonical dumper must bump `MUTATOR_VERSION`; the suite digest changes and
old results stop folding, by design. Rebuilding with a different seed,
quota, source or generator model is a new suite. Keep the first board's
manifest (`benchmarks/verifier/boards/synthetic_us_v1/suite.manifest.json`)
as the record of which cases it scored.

## Limits, stated plainly

- Synthetic defects are single edits and a lower bound on difficulty. A real
  encoder error is rarely one token.
- Controls are gate-passing, not human-verified. "Known-good" means the
  artifact passed compile, CI and apply (or the encoder track's gate
  battery); it does not mean a lawyer confirmed it faithful. A judge that
  flags a control may be right about a defect the gates cannot see, so the
  native false-alarm rate is an upper bound on true false alarms. The paired
  and pooled within-kind metrics are less exposed, because a pre-existing
  defect sits in both members of a pair.
- A binary kind channel has one operating point. When a judge names a kind
  on more controls than the ceiling allows, its detection at the ceiling is
  zero for that kind: the channel cannot be run at that false-alarm budget,
  whatever its AUC.
- A judge that has seen the artifact shape before may recognise the canonical
  formatting rather than read the law; both members of a pair share it, so
  paired metrics are unaffected, but native flag rates can be.
- Real cases may have post-fix artifacts that are not clean. They are
  labelled `unverified` and never mixed into a synthetic board (different
  suite, different digest).
- Provision windows over 24,000 characters are truncated head and tail the
  way the referee truncates them; a defect whose evidence sits in the cut
  middle is undetectable by design, and the amount guard only accepts
  numbers inside the window.
- The referee's kind channel is binary (named the kind or not), so its
  per-kind AUC equals balanced accuracy and carries many ties; Jev's kind
  channel is continuous. The verdict-channel AUC is reported beside it for
  a like-for-like comparison. A binary channel also collapses at the
  ceiling: once the referee names a kind on more than the ceiling's share of
  clean controls, the admissible threshold is the top score itself and
  nothing scores strictly above it, so `det@ceil` reads 0 percent. That is
  the correct reading of the operating point, not a rendering fault; the
  paired rise rate and AUC beside it carry the graded picture.
- The `entity_wrong` guard keys on entity vocabulary (person, household,
  taxpayer, employer, ...) and is the weakest of the six; treat that column
  as indicative.
- Cost is quoted only where a price is on file; blank is not zero.
- The first board is drawn from `encodings.db` generations by `gpt-5.5`,
  which is fine for the artifacts being judged but is not the pinned UK
  release; the UK synthetic set follows the encoder track's outputs.

## First board

_Filled in below once the runs complete._

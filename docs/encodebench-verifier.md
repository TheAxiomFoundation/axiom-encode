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
    (axiom-encode PR #1659, 520 cases mined from rulespec-us and rulespec-uk
    fix history; this code only reads it). The pre-fix artifact is the
    defective case, the post-fix artifact is the control, and every real
    control is marked `unverified` because a repair round makes an artifact
    better, not proven clean. Five of the corpus's eight defect kinds map onto
    a synthetic kind with a kind-specific judge channel; `unrepresented_clause`,
    `untraceable_branch` and `other` keep their own `other:` columns on the
    verdict channel. By default the loader keeps family representatives only
    and `triage_status: fidelity` only, skipping metadata-only records.
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
claude-api skill's model table or the official pricing page (with the fetch
date recorded), never from memory.

**Headline**: per-kind detection AUC on the judge's kind channel, subject to
a false-alarm ceiling on the judge's native verdict. The default ceiling is
10 percent (`--false-alarm-ceiling`), stated on every board. Ten because a
referee flag adds a human-review label: a reviewer who finds a clean
artifact one time in ten keeps reading, one time in four stops. At 10
percent the ranking of the first board is unchanged from 25 (every judge is
over either line) and only the detection-at-ceiling column moves. A judge whose
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
with `build-real --dir benchmarks/verifier/real_defects_v0 --out ...`; the
defaults keep family representatives with `triage_status: fidelity`
(`--all-family-members`, `--triage-status`, `--min-confidence` and
`--jurisdiction` change the selection, and the selection is recorded in the
suite identity).

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

`--max-case-chars N` drops pairs whose larger member (provision window plus
artifact) exceeds N characters, for a like-for-like board when one judge has
an input cap. A filter must never depend on judge outputs beyond such a
stated, size-based rule. Rows already judged against the
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
- The referee's production output budget truncates on large artifacts.
  `JudgeClient` defaults to 2,048 output tokens; on the real corpus the
  referee's findings list for modules of 30,000 input tokens and more ran
  past it, the JSON was cut, and the case became a `parse_error` row (never
  a pass). That is a finding about the production default worth carrying to
  the judges package. The real-corpus referee runs use `--max-tokens 8192`,
  recorded in the runner identity, so the board measures reading rather than
  a token cap; the synthetic board ran at the production 2,048 and hit it
  once in 1,080 referee calls.
- Jev has an input cap. On the real corpus every case up to 100,670
  characters of provision window plus artifact was answered and every case
  from 101,617 characters up was refused with a 400 `max_tokens_exceeded`
  (about 25 to 30 thousand tokens). Those cases are error rows for Jev,
  never passes, and the full real board folds them with `--allow-partial`
  so the limitation shows. A like-for-like board over the cases every judge
  could read is derived with `filter-suite --max-case-chars 100000`; both
  boards are reported.
- The first board is drawn from `encodings.db` generations by `gpt-5.5`,
  which is fine for the artifacts being judged but is not the pinned UK
  release; the UK synthetic set follows the encoder track's outputs.

## First board (2026-09-18)

Suite `EncodeBench verifier synthetic US v1`, digest `5b228af3d17c`: 180
pairs (30 per kind) planted by mutator 1.0.1 in `gpt-5.5` `apply_applied`
generations from `encodings.db` with US citations, provision window 24,000
characters. Every judge scored all 360 cases with no errors. The board, its
JSON and CSV, and the suite manifest are committed under
`benchmarks/verifier/boards/synthetic_us_v1/`; full suite texts and per-run
rows live in `_axiom-runs/encodebench-verifier-2026-09-17/`.

| judge | model | native FAR | native det | mean kind AUC | verdict AUC | localize | median s | cost/case | total |
|---|---|---|---|---|---|---|---|---|---|
| jev | jev-1.13.0 | 72% | 95% | 0.908 | 0.779 | blank by construction | 0.19 | $0.00013 | $0.048 |
| sonnet | claude-sonnet-4-5 | 38% | 76% | 0.738 ‡ | 0.726 | 68% | 5.24 | $0.01472 | $5.30 |
| opus | claude-opus-4-6 | 63% | 86% | 0.705 ‡ | 0.752 | 80% | 11.11 | $0.02800 | $10.08 |
| haiku | claude-haiku-4-5-20251001 | 72% | 89% | 0.680 ‡ | 0.639 | 69% | 3.88 | $0.00514 | $1.85 |

Per-kind kind-channel AUC (Jev / Sonnet / Opus / Haiku): amount 0.998 /
0.983 / 0.950 / 0.983; boundary 0.924 / 0.800 / 0.783 / 0.700; conjunct
0.753 / 0.650 / 0.583 / 0.467; polarity 0.979 / 0.800 / 0.717 / 0.683;
date or period 0.853 / 0.622 ‡ / 0.642 ‡ / 0.659 ‡; entity 0.942 / 0.570 ‡ /
0.557 ‡ / 0.588 ‡.

How to read it:

- **Nobody ranks.** All four judges flag far more than 10 percent of the
  clean controls at their native verdict (Sonnet the fewest at 38 percent),
  so the headline gate excludes them all. That is the pilot's finding reproduced on
  180 pairs and four judges: as a pass/flag gate, none of these is usable
  yet. The judges also agree with each other on the controls (in the
  superseded 1.0.0 run, Haiku and Jev both flagged 114 of 180), and sampled
  Haiku findings on controls are plausible fidelity complaints, so part of
  that rate is real defects the compile and CI gates cannot see; native FAR
  is an upper bound.
- **On the kind channel Jev separates defective from control far better
  than any referee configuration**, at roughly one two-hundredth of Opus's
  cost, one hundredth of Sonnet's and one fortieth of Haiku's, and twenty to
  sixty times faster. Its weakest kinds
  are dropped conjuncts (0.753) and wrong dates or periods (0.853), the same
  shape as the pilot.
- **The referee's kind channel is binary**, so its per-kind AUC is a balanced
  accuracy and its detection at the ceiling is often zero: when it names a
  kind on more than a quarter of the controls, the channel has no operating
  point under the ceiling. Its two ‡ kinds fall back to the verdict score.
- **Sonnet, not Opus, is the best referee on the kind channel** (0.738
  against 0.705), and it has the lowest native false-alarm rate; Opus
  localizes best (80 percent of its defective-case findings name the
  mutated rule or token) and has the best verdict-channel AUC among the
  referees (0.752).
- **No coerced verdicts.** With the structured-output schema, no referee
  answer was a raw pass with findings, so the coercion question the review
  raised did not arise in this run.
- **Self-agreement.** Twenty-nine texts were judged twice: once in a
  superseded 1.0.0 build and once here, identical provision and artifact
  bytes. Jev repeated its verdict on 28 of 29 (median change in P(flag)
  0.03, maximum 0.07); Haiku repeated its verdict on 23 of 29 and produced
  the same set of finding kinds on only 8 of 29. That join was computed from
  the superseded run's suite texts because its rows predate the content
  digests; `verifier.py agreement` does the same for any two current runs.
- **Sonnet's cost** was blank on the first fold because the claude-api
  skill's model table carries no Sonnet 4.5 row; the official pricing page
  (fetched 2026-09-19) lists $3 in and $15 out per million, and the board was
  re-costed from the same rows under that source.

What this board does not say: nothing about the UK release (the suite is US
generations from the run log), nothing about real repair rounds (the real
corpus folds into its own board), and nothing about multi-edit defects.

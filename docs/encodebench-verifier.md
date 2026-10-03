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
- **Defect kinds**: `amount_changed` (a number equal to one the provision
  window states), `boundary_flipped` (`>=` and `>`, `<=` and `<`, both
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

- the two artifacts differ in exactly one leaf: `mutate` refuses any edit
  that changes more than one (a YAML alias can make one assignment change
  two paths), and tests assert it for every kind and for 150 generated
  artifacts;
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
not; a dotted code such as `7202.11.10` is never an amount; an effective date
only moves when the window states the original year as a word of its own and
not the shifted one; a period or entity only changes when the window uses a
word for the original (whole words, regular plurals plus "families",
"people" and "children": "daylight" is not "day", "personal" is not "person")
and none for the replacement. Year-like numbers are never treated as amounts,
and `<<`, `>>`, `->`, `=>`, `>>=` and `<<=` are never boundaries. Nothing
inside a string literal is edited: not `and`/`or` (`"Bosnia and
Herzegovina"`), not a comparison (`"a > b"`), not a number (`"11"`). A formula
with a `#` outside a literal is left alone by every formula kind, because the
mutator does not parse whether it starts a comment. Conjuncts are only
dropped from pure conjunctions: a formula with a top-level `or`, a colon, or
a top-level conditional keyword (`x if c else y`, `if c then x else y`) is
left alone, because deleting the text between two `and` tokens there would
remove more than one condition. A dropped conjunct is cut from the original
text, so the rest of the formula keeps its layout and its leading and
trailing whitespace. Each source artifact is used for at most one pair, and
kinds are filled by deficit so the rarer sites get first pick.

Version history. Mutator 1.0.0 matched amounts and years by substring; two of
its 180 pairs were undetectable for that reason (`200` found only inside
`2008`, `11` only inside `3211(b)`), and that suite was superseded by a full
1.0.1 rebuild. The committed board's suite was built by 1.0.1, whose period
and entity guards matched word prefixes: four Day-to-Month pairs on tariff
headings passed only because "eastern daylight time" contains "day". 1.0.1
also reflowed a formula onto one line when it dropped a conjunct, a layout
change with no change in meaning. 1.0.2 fixed both. 1.0.3 closed what an
executed review of 1.0.2 found: comparisons and numbers inside string
literals, `#` comments, `>>=`, the irregular plurals and the end-conjunct
whitespace. Rather than rebuild and re-judge, `audit-suite` re-checks every
committed pair against the current guards. Under 1.0.2 exactly those four
pairs failed, and the board drops them by a recorded filter (see the boards
section); the remaining 176 pass the 1.0.3 audit unchanged.

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
every judge's verdict channel. Tokens, latency and cost cover every row in
the results, errors included. A retried error row is replaced by its retry,
so the failed call's own spend is not in the results; `cases.jsonl` keeps
every row, and the spend totals below are deduplicated by row.

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
reason and every dropped pair. When the mutator's guards tighten,
`audit-suite` names the pairs the current version would not plant (exit 1 if
any), and `--ids` prints them for the filter. This is how the committed
board's child suite was made:

```bash
uv run python benchmarks/verifier/verifier.py audit-suite \
  _axiom-runs/encodebench-verifier/synthetic_us_v1
```

```bash
uv run python benchmarks/verifier/verifier.py filter-suite \
  --suite _axiom-runs/encodebench-verifier/synthetic_us_v1 \
  --drop-pair date_or_period_wrong-58d498f1 date_or_period_wrong-66ceaa04 \
    date_or_period_wrong-73274712 date_or_period_wrong-aa3de659 \
  --reason "fail the mutator 1.0.2 period guard (audit-suite): the window states no daily period; 1.0.1 matched 'day' inside 'daylight'" \
  --name "EncodeBench verifier synthetic US v1 (1.0.2 audit)" \
  --out _axiom-runs/encodebench-verifier/synthetic_us_v1_audited
```

`--max-case-chars N` drops pairs whose larger member (provision window plus
artifact) exceeds N characters, for a like-for-like board when one judge has
an input cap. A filter must never depend on judge outputs beyond such a
stated, size-based rule. Rows already judged against the
parent fold into the child without re-judging. Use `reassemble`, which builds
no judge and needs no keys:

```bash
uv run python benchmarks/verifier/verifier.py reassemble \
  --suite _axiom-runs/encodebench-verifier/synthetic_us_v1_audited \
  --from _axiom-runs/encodebench-verifier/runs/haiku \
  --out _axiom-runs/encodebench-verifier/runs_audited/haiku
```

It checks that the child was derived from the run's own suite, keeps the
run's rows whose content digests the child carries, and writes `results.json`
and `cases.jsonl` under the run's recorded runner identity and price. Pointing
`run` at the child suite and the parent's `--out` also re-assembles, but it
rebuilds the judge first, and any drift in the judge's identity (an SDK
upgrade, a changed default output budget) makes every row a stranger, so the
whole suite is judged again and paid for. Parent-suite and child-suite runs
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
  --markdown-out board.md --json-out board.json --csv-out board.csv
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
  or unkeyed, a response without a verdict or without every kind's score, an
  unparseable confidence, and (for Jev) a served model other than the pinned
  one are all recorded as error rows with their cause, retried on the next
  run, and keep the run incomplete until they clear. The referee cannot check
  its served model: `JudgeClient` reports the model it requested, not the one
  the API answered with, so a referee's "served model" on the board is its
  requested model id.
- A crash can leave a half-written last line in `cases.jsonl`. The next read
  drops it with a warning, and the next run trims it before appending, so the
  run directory stays readable however many times it resumes.
- An interrupt (Ctrl-C) cancels the queued cases, writes what finished, and
  exits 130; re-running the same command resumes. `--fresh` rotates the old
  `cases.jsonl` and `results.json` to `.bak` files rather than deleting them.
  `--limit` judges only the first N cases but never downgrades a finished
  run, and keeps error rows outside the first N (with their cost) rather
  than dropping them unjudged.
- Cost and localization are recomputed at assembly from what the judge
  returned (tokens, findings) under the one price the payload names and the
  current matcher. A row whose usage the provider did not report is unpriced
  and counted as such, never charged zero; for the referee, whose client
  reports missing usage as zero tokens, a 0/0 reply is recorded as unknown.

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

## Breakdown (on which cases does a judge see the defect?)

A board answers whether a judge separates defective from control; the
breakdown answers where. It reports paired rise (the defective case scored
strictly above its own control on the kind channel) and pooled AUC per
bucket of module size, the share of the module's lines the fix changed
(`diff`, from a line diff: a same-length rewrite of every line is 100
percent, a one-line fix in a thousand-line module 0.1 percent), triage
confidence, fix stage, kind or jurisdiction. A case without the metadata a
property needs (a synthetic case has no triage confidence) lands in an
`unknown` bucket:

```bash
uv run python benchmarks/verifier/verifier.py breakdown \
  --suite _axiom-runs/encodebench-verifier/real_v0 \
  --run _axiom-runs/encodebench-verifier/runs_real/jev --by size diff fix_stage
```

It refuses a run judged against a different suite. It is the tool for the
question a flat headline hides: whether a judge near chance overall still
separates some kinds of case (large corrections, say) and fails on others.

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
canonical dumper must bump `MUTATOR_VERSION`; a suite built by the new
version has a new digest and old results stop folding with it, by design. An
existing suite keeps the version that built it; `audit-suite` says which of
its pairs the new version would refuse. Rebuilding with a different seed,
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
- The referee's output budget bounds what it can say. `JudgeClient`
  defaulted to 2,048 output tokens when these runs were made; on real modules
  of 30,000 input tokens and more the findings JSON ran past it, the reply
  was cut, and the case became a `parse_error` row (never a pass). The
  judges package now defaults to 16,000 and names a cut-off reply as a
  `max_tokens` error (axiom-encode #1759). The output budget is recorded in
  every referee's identity. On the committed synthetic board the three 4.x
  referees ran at 2,048, and one call across their 1,080 cases (Haiku, a
  4,323-token input) was cut at exactly 2,048 output tokens and retried. Opus 5 and
  Sonnet 5 ran at 8,192, and Sonnet 5 was still cut there four times, each
  retried.
- Jev has an input cap. On the real corpus every case up to 100,670
  characters of provision window plus artifact was answered and every case
  from 101,617 characters up was refused with a 400 `max_tokens_exceeded`
  (about 25 to 30 thousand tokens). Those cases are error rows for Jev,
  never passes, and a board folds them only with `--allow-partial`, so the
  limitation shows. A like-for-like board over the cases every judge could
  read is derived with `filter-suite --max-case-chars 100000`.
- The first board is drawn from `encodings.db` generations by `gpt-5.5`,
  which is fine for the artifacts being judged but is not the pinned UK
  release; the UK synthetic set follows the encoder track's outputs.

## Boards (2026-10-03)

One board is committed, under `benchmarks/verifier/boards/synthetic_us_v1/`,
with its markdown, JSON, CSV and the suite manifest that identifies exactly
which cases it scored. Full suite texts and per-run rows live in
`_axiom-runs/encodebench-verifier-2026-09-17/`. The roster: TypeSafe Jev
1.13.0, and the incumbent referee on Haiku 4.5 (the judges package's default
judge model), Sonnet 4.5, Sonnet 5, Opus 4.6 and Opus 5. Each judge's configuration, output
budget included, is in its results payload.

The judging ran on 2026-09-18 and 2026-09-19. On 2026-10-03 the board was
re-derived from those recorded rows without judging anything again:
`audit-suite` found four pairs that the 1.0.2 guards refuse (the other 176
also pass 1.0.3), `filter-suite`
dropped them with that reason recorded, `reassemble` folded each judge's rows
onto the child suite, and localization was recomputed under the whole-word
matcher. The block below is generated from the committed `board.json` by
`verifier.py report`, and a test fails if it drifts from the board.

A real-defects board was also folded on 2026-09-19, over the 172
family-representative fidelity pairs of the PR #1659 corpus. It is held out
of this PR. Review found that about 60 percent of the sampled fidelity cases
ship a provision window without the defect's evidence, because only the first
citation is resolved. That board would measure the corpus as much as the
judges, so it comes back only once the corpus checks for the evidence. The
board and its reproduction test are on branch
`encodebench-verifier-real-board`.

<!-- begin generated boards: verifier.py report; edit boards, not this -->

### EncodeBench verifier synthetic US v1 (1.0.2 audit)

176 pairs (352 cases), suite `bc57111f0129`, source `encodings_db`, built with mutator 1.0.1, provision window 24,000 characters.
Filtered from suite `5b228af3d17c` (EncodeBench verifier synthetic US v1): 4 pair(s) dropped, because they fail the mutator 1.0.2 period guard (audit-suite): the window states no daily period; 1.0.1 matched 'day' inside 'daylight'.

| judge | model | native FAR | native det | mean kind AUC | verdict AUC | localize | median s | cost/case | total |
|---|---|---|---|---|---|---|---|---|---|
| jev† | jev-1.13.0 | 71% | 95% | 0.911 | 0.783 | blank by construction | 0.19 | $0.00013 | $0.05 |
| opus-5† | claude-opus-5 | 52% | 91% | 0.790 ‡ | 0.814 | 86% | 14.41 | $0.05305 | $18.67 |
| sonnet† | claude-sonnet-4-5 | 39% | 78% | 0.741 ‡ | 0.730 | 69% | 5.55 | $0.01480 | $5.21 |
| sonnet-5† | claude-sonnet-5 | 66% | 88% | 0.709 ‡ | 0.722 | 75% | 20.31 | $0.03267 | $11.50 |
| opus† | claude-opus-4-6 | 65% | 88% | 0.708 ‡ | 0.755 | 82% | 11.26 | $0.02822 | $9.93 |
| haiku† | claude-haiku-4-5-20251001 | 72% | 89% | 0.684 ‡ | 0.642 | 69% | 3.88 | $0.00513 | $1.81 |

Per-kind kind-channel AUC (jev / opus-5 / sonnet / sonnet-5 / opus / haiku): amount (n=30) 0.998 / 0.983 / 0.983 / 1.000 / 0.950 / 0.983; boundary (n=30) 0.924 / 0.983 / 0.800 / 0.950 / 0.783 / 0.700; conjunct (n=30) 0.753 / 0.667 / 0.650 / 0.567 / 0.583 / 0.467; polarity (n=30) 0.979 / 0.767 / 0.800 / 0.617 / 0.717 / 0.683; date or period (n=26) 0.868 / 0.724 ‡ / 0.642 ‡ / 0.598 ‡ / 0.660 ‡ / 0.681 ‡; entity (n=30) 0.942 / 0.614 ‡ / 0.570 ‡ / 0.524 ‡ / 0.557 ‡ / 0.588 ‡.

Computed from the board:

- No judge ranks: every judge flags more than 10% of the clean controls at its native verdict. The lowest rate is sonnet's, at 39%.
- Highest mean kind-channel AUC: jev (0.911), at $0.00013 and 0.19 s a case. The best referee configuration is opus-5 (0.790), at $0.05305 and 14.41 s.
- jev's weakest kinds: conjunct (0.753) and date or period (0.868).
- Best localization: opus-5, 86% of defective cases with a finding naming the mutated rule or token.
- Slowest median call: sonnet-5, 20.31 s.
- Coerced verdicts (a raw pass that carried findings): 0 in total.
- Spend recorded in this board's results: $47.17.

<!-- end generated boards -->

How to read it:

- **No judge is usable as a pass/flag gate yet.** Every judge flags far more
  than a tenth of the gate-passing controls. Part of that rate may be real
  defects the compile and CI gates cannot see. The judges agree with each
  other on which controls to flag more often than chance (mean pairwise
  Cohen's kappa 0.34 over the 176 controls, from 0.16 to 0.49), so the native
  false-alarm rate is an upper bound on true false alarms, not a measurement
  of them.
- **The referee's kind channel is binary.** The kind score is 1 when a
  finding names the mapped kind and 0 otherwise, so the referee's per-kind AUC
  is a balanced accuracy. Its detection at the ceiling is often zero: once it
  names a kind on more than a tenth of the controls, the channel has no
  operating point under the ceiling. Its two ‡ kinds (date or period, entity)
  fall back to the verdict score, because the referee asks no question about
  them. Jev's kind channel is continuous.
- **Thinking models spend their output budget.** Sonnet 5 and Opus 5 write
  replies several times longer than the 4.x referees (mean output tokens are
  in `board.md`), which is where their latency and cost go.
- **Self-agreement.** 29 texts were judged twice by Haiku and by Jev: once in
  the superseded 1.0.0 build and once here, with identical provision and
  artifact bytes (joined on the texts' sha256). Jev repeated its verdict on 28
  of 29, with a median change in P(flag) of 0.03 and a maximum of 0.07. Haiku
  repeated its verdict on 23 of 29 and gave the same set of finding kinds on
  only 8. The 1.0.0 runs predate the results digest, so `agreement` refuses
  them; these figures were computed from the two runs' rows directly.

Real API spend for the whole build was $190.99. That figure was recomputed on
2026-10-03 from every recorded row in the run directory, `.bak` rotations
included, deduplicated by judging event and priced from `pricing.json`. It
covers the superseded 1.0.0 run, the runs stopped for the output-budget fix
and the real-corpus runs, so it is larger than the spend the board above
records.

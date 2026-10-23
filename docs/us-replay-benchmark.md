# US replay benchmark

The US replay benchmark gives every validator or prompt PR an offline
before/after number. It needs no model call, network, rules engine or corpus
checkout. It runs in about 15 seconds.

## What it measures

`scripts/replay_us_failure_bundles.py` replays `analyze_complete_source_unit`
(the `[complete-source-unit:*]` rules of the CI validator) over 32 committed
fixtures in `tests/fixtures/us_replay_benchmark/`. Each fixture is a real
rejected US candidate from a `targeted-reencode-failure` artifact:

- `candidate.yaml` and `candidate.test.yaml`: the final rejected candidate and
  its companion tests. Their hashes match the bundle inventory and the
  recorded `issues.json`.
- `source.txt` and `source-metadata.json`: the exact workspace source. Its hash
  matches the resolver attestation.
- `issues.json`: the recorded pipeline issues for that run.
- `context/`: only the workspace context modules that the candidate imports or
  names as a deferral provider (`blocked_by`).
- `providers/`: a module that the candidate needs but the workspace context
  lacks, copied from the RuleSpec repository at the run's `rulespec_ref`. Only
  one fixture has one.
- `fixture.json`: the run id and URL, citation, lane, runner, encoder commit and
  version, corpus and RuleSpec refs, recorded failure classes, the origin of
  each context module and the SHA-256 of every file.

The replay passes the same inputs as `ValidatorPipeline._complete_source_unit_issues`:
the citation's numeric profile, the companion test cases, imported symbol
contents, resolved deferral providers and same-act aliases from the source
metadata. `--no-context` leaves out the last three.

The output is a JSON report with per-rule totals, the number of modules that
have each rule, the full issue list of each fixture, and any import or
deferral provider that the context cannot resolve.

The set has 32 distinct citations: federal statutes, CFR regulations, IRS and
FNS guidance, and state statutes, manuals and guidance. It includes every
completeness rule and the recorded failure classes `compile`, `oracle`,
`proof:excerpt-not-found`, `tests:wrong-expected-value`, `tests:missing-input`
and `grounding:literal`. Five fixtures replay clean at 0.2.2250. They failed for other
reasons (compile, oracle, proof excerpts, test inputs), or later validator fixes
removed their completeness issues. The snapshot keeps them clean. The fixtures
total about 1.4 MB.

## Failure-class labels

Each `fixture.json` also carries a hand label of the root cause of the
failure, from the failure-class design (v1):

- `expected_class`: one of `encoder`, `validator_fp`, `corpus`, `dependency`,
  `selection`, `engine`, `migration`, `process`, `infra` or `unknown`.
- `expected_subclass` (optional): `encoder` takes `omission`, `formula`,
  `test` or `harness`; `corpus` takes `content` or `metadata`; `migration`
  takes `schema` or `elided_excerpt`.
- `class_evidence`: one sentence that names the deciding evidence.
- `class_source`: `hand` for these labels.

A label names the root cause, not the symptom. A fixture whose blocker no
candidate can fix (an unencoded upstream concept, a non-operative target, a
removed schema field in a context module) takes that class, even when the
candidate also has encoder errors. `unknown` means that the evidence does not
decide between two classes. `load_fixture` refuses an unknown class, a
subclass of another class, or a label without evidence. `fixture.json` is not
in its own hash list, so a label change does not change any file hash.

`replay` prints the number of fixtures per class and puts `expected_class`
and `expected_subclass` on each fixture row of the JSON report. The labels do
not enter `snapshot.json`; the ratchet stays per rule.

## Fidelity

Replayed with the encoder source of their own run, 11 of 12 sampled fixtures
reproduce the recorded completeness rule counts exactly. Without context, 10
of 12 do. On all 197 replayable US bundles since 2026-09-01, context removes 5
numeric-recall and 10 tests issues and adds 9 tests issues (5,711 to 5,705). An
imported module can also add demands, for example a rounding rule. Only one
bundle in that set has deferral providers that its workspace context lacks.

## Limits

- Only the completeness analyzer runs. Compilation, companion-test execution,
  oracles, proof-excerpt checks (`source_context` is not passed), reviewers and
  signing are not measured. A fixture that replays clean can still fail CI.
- Context comes from the workspace snapshot, not from the live checkout. In one
  sampled fixture, production did not bind a value that the context provides.
  The test requires that every import and deferral provider of every fixture
  resolves; `extract --rulespec-root` fills a gap from a RuleSpec checkout at
  the run's `rulespec_ref`.
- The candidates are fixed. The benchmark measures validator changes. It does
  not show whether a prompt change makes the model write better candidates. Use
  `axiom-encode eval-suite` for that.
- The fixtures are rejected candidates. A lower count can mean a removed false
  positive or a weakened rule. Check every removed issue against the
  fixture's `source.txt`.

## How PR authors report numbers

1. On `main`, run
   `uv run python scripts/replay_us_failure_bundles.py replay --out before.json`.
2. On your branch, run the same command with `--out after.json`.
3. Run `uv run python scripts/replay_us_failure_bundles.py compare before.json after.json`.
4. In the PR body, report the issue total and clean modules before and after,
   and removed and added issues by rule. `compare` leaves out a fixture that
   errors on either side and lists it under `ERRORED`; report those too.
5. For removed issues, list examples that you checked against `source.txt`.
   Report any removed issue that was a true positive.

`tests/test_us_replay_benchmark.py` pins the per-fixture, per-rule counts in
`snapshot.json`. The pin is a ratchet:

- A count that rises fails the test as a regression. A new replay error or a
  removed fixture also fails.
- A count that falls also fails until the same PR records it. Run
  `uv run python scripts/replay_us_failure_bundles.py snapshot` and commit the
  new `snapshot.json`.
- To record a rise, a new error or a removed fixture, add `--allow-increase`
  and justify each one in the PR, for example a stricter rule that rejects real
  omissions.

## Adding fixtures

Download a failure artifact and extract the tar. The artifact name ends with
the run attempt:

```bash
gh run download <run_id> --repo TheAxiomFoundation/axiom-encode \
  -n targeted-reencode-failure-<run_id>-<attempt> -D bundles/<run_id>
mkdir bundles/<run_id>/x
tar -xf bundles/<run_id>/targeted-reencode-failure.tar -C bundles/<run_id>/x
```

1. Run `python -I scripts/replay_us_failure_bundles.py extract bundles/<run_id>/x`.
2. If `extract` warns about an unresolved import or deferral provider, delete
   the new fixture directory. Then rerun `extract` with
   `--rulespec-root <RuleSpec checkout at the run's rulespec_ref>`. The fixture
   records the git blob id of each copied module, so a reviewer can compare it
   with the RuleSpec repository at that ref.
3. Run `snapshot`. It records new fixtures without `--allow-increase`.

Keep one or two fixtures for each citation. Prefer fixtures that add a rule
class or a citation. Failure artifacts expire after 90 days.

To replay a larger, uncommitted set, run
`python -I scripts/replay_us_failure_bundles.py replay --bundles bundles/ [--latest-per-citation]`.
Bundles are untrusted data: the script only parses them, refuses symlinks and
keeps every path inside its bundle.

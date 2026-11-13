# US replay benchmark

The US replay benchmark gives every validator PR an offline before/after
measure on real candidates. It needs no model call, network, rules engine or
corpus checkout. The `replay benchmark` CI job runs it as a required check.

## What it measures

`scripts/replay_us_failure_bundles.py` replays `analyze_complete_source_unit`
(the `[complete-source-unit:*]` rules of the CI validator) over 47 committed
fixtures in `tests/fixtures/us_replay_benchmark/`. Hosted fixtures are real candidates from `targeted-reencode-failure` artifacts
of signed runs:

- `candidate.yaml` and `candidate.test.yaml`: the final candidate and its
  companion tests. For a rejected candidate, their hashes match the bundle
  inventory and the recorded `issues.json`.
- `source.txt` and `source-metadata.json`: the exact workspace source. Its hash
  matches the resolver attestation.
- `issues.json`: the recorded pipeline issues for that run.
- `context/`: only the workspace context modules that the candidate imports or
  names as a deferral provider (`blocked_by`).
- `providers/`: a module that the candidate needs but the workspace context
  lacks, copied from the RuleSpec repository at the run's `rulespec_ref`.
- `fixture.json`: the run id and URL, citation, lane, runner, encoder commit and
  version, corpus and RuleSpec refs, recorded failure classes, the origin of
  each context module and the SHA-256 of every file. Optional keys:
  `recorded_verdict` (default `rejected`; any other value marks a control, a
  candidate that production validation accepted) and `split` (`visible`, the
  default, or `held_out`; only a private fixture set uses `held_out`).

Local diagnostic failures may also be committed with explicit
`origin: local-diagnostic` provenance. They retain exact public source,
candidate, companion and actual rejected replay findings, a producer commit
and tree, the reproducible replay command, and byte-identical repository input
origins. They cannot claim a hosted run URL, signed source attestation or native
bundle inventory. The same file hashes, public-only restrictions and snapshot
gates apply. Local fixtures are rejected examples, never production controls.
The Schedule 2 floor fixture uses this route; its offline findings remain
unresolved because replay has no live execution evidence. Its separate
real-engine test verifies the targeted floor witnesses. The NL479 fixture retains
an earlier primary-document diagnostic; it is separate from the full parent-form
engine replay and does not inherit that replay’s execution evidence.

The replay passes the same inputs as `ValidatorPipeline._complete_source_unit_issues`:
the citation's numeric profile, the companion test cases, imported symbol
contents, resolved deferral providers and same-act aliases from the source
metadata. `--no-context` leaves out the last three.

The set has 39 distinct citations: federal statutes, CFR regulations, IRS and
FNS guidance, state statutes, manuals and guidance, one Canadian CRA guide
(three runs), two Canadian CRA forms (local diagnostics), and one Israeli statute. It has one control (an Arizona candidate
that passed validation and then failed a provenance guard). At 0.2.2349, 12
fixtures replay clean and 33 have 3,313 findings; four 7 CFR 273.7 candidates
hold 2,671 of them. Read clean counts and per-fixture changes, not only the
total. The fixtures total about 3.3 MB and `snapshot.json` about 1.3 MB.

## Public fixtures and the private held-out set

Every committed fixture is public and visible. The repository holds no
held-out case and no held-out label: no held-out case id, expected outcome or
count. It also never holds:

- hand failure-class labels (`expected_class` and related keys);
- sources from a private-only corpus (12 CFR 1005 and 1006);
- raw failure bundles.

The held-out set is built only from cases that were never committed. It stays
in the maintainer's private benchmark, outside the repository, and a
maintainer replays it locally (`replay --fixtures <private fixture directory>`
and `compare --split <private map>`). `compare --split` prints held-out cases
as totals only (`--show-held-out` lists them, for the maintainer). The CI job
replays only the committed fixtures and expects no held-out data. A case that
is committed, now or in the past, is public and never goes back into the
held-out set.

## The required CI check

The `replay benchmark` job in `.github/workflows/ci.yml` runs
`scripts/replay_us_failure_bundles.py ci`. On a pull request it first checks
the paths that the pull request changes. When none of the trigger paths
changes (`REPLAY_TRIGGER_PATHS`: `src/axiom_encode/harness/`,
`bare_structure.py`, `corpus_resolver.py`, the replay script, the fixtures,
`ci.yml`, `pyproject.toml`, `uv.lock`), and no other `axiom_encode` module
that the replay loads changes, the job passes without a replay. Renames count
both paths. The merge queue and pushes to `main` always replay. A manual
dispatch on a branch has no base snapshot: it checks errors, drift and
controls only. The replay takes 20 to 55 s with 4 workers; the whole job took
67 s on a GitHub runner.

The job fails when:

1. **The replay errors.** A fixture does not load or authenticate, the
   analyzer raises, or an import or deferral provider does not resolve.
2. **A clean fixture or a control becomes refused.** A fixture that the base
   branch's `snapshot.json` pins with no finding, or a control, now has a
   finding or an error, or was removed. The only way through is a reason in
   the PR's own `snapshot.json` (`allowed_regressions`, written by
   `snapshot --allow-regression ID=REASON`), which a reviewer sees in the diff.
3. **Any finding differs from the committed `snapshot.json`.** The snapshot
   holds every fixture's exact findings. A removed, added or reworded finding,
   a new fixture or a removed fixture fails until the same PR rewrites the
   snapshot. The committed file must also be exactly what the `snapshot`
   command writes, so a hand edit to counts or totals fails, and every
   `allowed_regressions` reason must excuse a real regression against the
   base.

The job's summary lists every changed fixture against the base branch.

### Updating the snapshot

When a change to the findings is intended:

```bash
uv run python scripts/replay_us_failure_bundles.py snapshot --base-ref origin/main --jobs auto
```

Commit `tests/fixtures/us_replay_benchmark/snapshot.json`. The rewrite is a
reviewable diff: one line per finding. The command refuses to pin replay
errors. It also refuses to pin a finding on a fixture that the base pins clean
(or on a control) unless you pass `--allow-regression ID=REASON` for it. A
rewrite keeps only the reasons that it is given again. The gate counts only
the reasons that a PR adds: a reason that the base already carries neither
excuses a new regression nor fails other PRs.

`snapshot --check` reports drift without writing.

Two PRs that both change findings conflict in `snapshot.json` (at least in
`totals`). Merge `main` into the second one and rerun `snapshot`.

## Rule for false-positive fix PRs

Every PR that fixes a validator false positive:

1. Adds the failing candidate as a fixture (`extract`, below), from public
   sources only. A candidate from a private-only corpus or a case of the
   private held-out set stays out of the repository; do not name a held-out
   run in the PR body.
2. Rewrites the snapshot. The new fixture's false findings must be gone.
3. Explains, in the PR body, every change to other fixtures: each removed
   finding checked against that fixture's `source.txt` (a removed true
   positive is a regression), and each added finding.

## Fidelity

Each fixture was replayed with the encoder source of its own run. Almost all
reproduce the recorded completeness findings exactly or within two findings,
with no replay errors. Three 7 CFR 273.7 runs stopped before the completeness
check in production; the replay finds 653 to 695 findings each.

## Limits

- Only the completeness analyzer runs. Compilation, companion-test execution,
  oracles, proof-excerpt checks (`source_context` is not passed), literal
  grounding, guards, reviewers and signing are not measured. A fixture that
  replays clean can still fail CI, and many rejected candidates also had a
  finding of such a kind.
- Context comes from the workspace snapshot, not from the live checkout.
  `extract --rulespec-root` fills a gap from a RuleSpec checkout at the run's
  `rulespec_ref`.
- The candidates are fixed. The benchmark measures validator changes. It does
  not show whether a prompt change makes the model write better candidates. Use
  `axiom-encode eval-suite` for that.
- A lower count can mean a removed false positive or a weakened rule. Check
  every removed finding against the fixture's `source.txt`.

## Comparing two encoder commits

1. On `main`, run
   `uv run python scripts/replay_us_failure_bundles.py replay --jobs auto --out before.json`.
2. On your branch, run the same command with `--out after.json`.
3. Run `uv run python scripts/replay_us_failure_bundles.py compare before.json after.json`
   (add `--split <map>` for a private held-out set).

`compare` gives totals per split, removed and added findings by rule, newly
clean and newly failing fixtures, and every changed fixture. It leaves out a
fixture that errors on either side and lists it under `ERRORED`.

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
3. For a control, add `"recorded_verdict": "<verdict>"` to `fixture.json`.
4. Run `snapshot --base-ref origin/main`.

Failure artifacts expire after 90 days.

To replay a larger, uncommitted set, run
`python -I scripts/replay_us_failure_bundles.py replay --bundles bundles/ [--latest-per-citation]`
or `replay --fixtures <private fixture directory>`. Bundles are untrusted
data: the script only parses them, refuses symlinks and keeps every path
inside its bundle.

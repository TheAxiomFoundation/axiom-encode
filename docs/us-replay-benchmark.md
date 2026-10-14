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
  names as a deferral provider.
- `fixture.json`: the run id and URL, citation, lane, runner, encoder commit and
  version, corpus ref, recorded failure classes and the SHA-256 of every file.

The replay passes the same inputs as `ValidatorPipeline._complete_source_unit_issues`:
the citation's numeric profile, the companion test cases, imported symbol
contents, resolved deferral providers and same-act aliases from the source
metadata. Imports resolve from the bundle's workspace context.

The output is a JSON report with per-rule totals, the number of modules that
have each rule, and the full issue list of each fixture.

The set has 32 distinct citations: federal statutes, CFR regulations, IRS and
FNS guidance, and state statutes, manuals and guidance. It includes every
completeness rule and the recorded failure classes `compile`, `oracle`,
`proof:excerpt-not-found`, `tests:wrong-expected-value`, `tests:missing-input`
and `grounding:literal`. Seven fixtures replay clean. They failed for other
reasons, and the snapshot keeps them clean. The fixtures total about 1.4 MB.

## Fidelity

Replayed with the encoder source of their own run, 10 of 11 sampled fixtures
reproduce the recorded completeness rule counts exactly. Without import
resolution, 9 of 11 do. On all 197 replayable US bundles since 2026-09-01,
import resolution removes 5 numeric-recall and 10 tests issues, and it adds 9
tests issues. An imported module can also add demands, for example a rounding
rule.

## Limits

- Only the completeness analyzer runs. Compilation, companion-test execution,
  oracles, proof-excerpt checks (`source_context` is not passed), reviewers and
  signing are not measured. A fixture that replays clean can still fail CI.
- Imports resolve from the workspace context snapshot, not from the live
  checkout. In one sampled fixture, production did not bind a value that the
  context provides.
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
   and removed and added issues by rule.
5. For removed issues, list examples that you checked against `source.txt`.
   Report any removed issue that was a true positive.

`tests/test_us_replay_benchmark.py` pins the per-fixture, per-rule counts in
`snapshot.json`. The pin is a ratchet:

- A count that rises fails the test as a regression.
- A count that falls also fails until the same PR records it. Run
  `uv run python scripts/replay_us_failure_bundles.py snapshot` and commit the
  new `snapshot.json`.
- To record a rise, add `--allow-increase` and justify each increase in the PR,
  for example a stricter rule that rejects real omissions.

## Adding fixtures

Download a failure artifact and extract the tar:

```bash
gh run download <run_id> --repo TheAxiomFoundation/axiom-encode \
  -n targeted-reencode-failure-<run_id>-1 -D bundles/<run_id>
mkdir bundles/<run_id>/x
tar -xf bundles/<run_id>/targeted-reencode-failure.tar -C bundles/<run_id>/x
```

Then run `python -I scripts/replay_us_failure_bundles.py extract bundles/<run_id>/x`
and `snapshot`. Keep one or two fixtures for each citation. Prefer fixtures
that add a rule class or a citation. Failure artifacts expire after 90 days.

To replay a larger, uncommitted set, run
`replay --bundles bundles/ [--latest-per-citation]`. Bundles are untrusted data:
the script only parses them, refuses symlinks and keeps every path inside its
bundle.

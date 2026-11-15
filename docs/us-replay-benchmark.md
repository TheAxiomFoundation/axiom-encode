# US replay benchmark

The US replay benchmark compares validator versions offline using fixed public
candidate fixtures. It needs no model call, network, rules engine or corpus
checkout. The `replay benchmark` CI job checks the replay against the committed
snapshot and guards clean fixtures against regressions.

## What it measures

`scripts/replay_us_failure_bundles.py` runs `analyze_complete_source_unit`
(the `[complete-source-unit:*]` checks in the CI validator) over 57 fixtures in
`tests/fixtures/us_replay_benchmark/`. It supplies the citation's numeric
profile, companion test cases, imported symbol contents, resolved deferral
providers and same-act aliases from source metadata, as
`ValidatorPipeline._complete_source_unit_issues` does. `--no-context` leaves
out imported symbols, deferral providers and aliases.

Hosted fixtures retain candidates from `targeted-reencode-failure` artifacts
of signed runs:

- `candidate.yaml` and `candidate.test.yaml` retain the final candidate and
  companion tests. For a rejected candidate, their hashes match the bundle
  inventory and the run's `issues.json`.
- `source.txt` and `source-metadata.json` retain the workspace source and its
  metadata. The source hash matches the resolver attestation.
- `issues.json` retains the run's pipeline findings.
- `context/` retains workspace modules that the candidate imports or names
  as deferral providers (`blocked_by`).
- `providers/` supplies modules that the candidate needs but the workspace
  lacks, using the RuleSpec repository at the run's `rulespec_ref`.
- `fixture.json` records the run id and URL, citation, lane, runner, encoder
  commit and version, corpus and RuleSpec refs, recorded failure classes,
  context-module origins and each file's SHA-256. Its optional
  `recorded_verdict` defaults to `rejected`; another value marks a control
  whose candidate passed production validation. Its optional `split`
  defaults to `visible`; private fixture sets may use `held_out`.

Local fixtures declare `origin: local-diagnostic`. Their metadata records the
public source, candidate and companion origins, producer commit and tree,
replay command and file hashes. They retain the rejected offline findings and
byte-identical repository inputs. They carry no hosted run URL, signed source
attestation or native bundle inventory. The same hash checks and public-data
restrictions apply, and local fixtures never count as production controls.

The Schedule 2 floor fixture retains offline findings that require live
execution evidence; its separate real-engine test checks the floor witnesses.
The NL479 fixture retains a primary-document diagnostic, separate from the
full parent-form engine replay. The T936 printed-row fixture retains an
unsigned source-only pair. None of these offline fixtures receives private
execution evidence from a separate engine test.

The New Brunswick Schedule 12 and Schedule M fixtures retain complete public
diagnostic pairs. The Schedule M companion exports the `/cases` value from
its public JSON record, with origin hashes. It does not reconstruct
historical engine companion bytes. Offline replay retains six findings for
each fixture without captured execution evidence; separate Schedule M
real-engine tests check the supported and unsupported engine profiles.

`local-schedule-n-20261113` and `local-t1170-source-20261113` retain the
historical public diagnostic inputs. Their metadata records producer
`6df4522a6ff20c78d8d70991c55f0353d49338b5`, tree
`d7746713dd0823cdf7983ce2edff73d2ec422a5f`, and each input's Git blob.
Their offline replays retain 48 and 35 findings respectively, without guide
context or private execution evidence. Historical call records supply test
data and grant no fresh execution or source authority.

### The derived T1170 printed-table fixture

`local-t1170-printed-tables-20261115` retains a derived 64-rule candidate and
its full 124-case companion: 91 positive cases and 33 expected refusals.
Its `fixture.json` identifies the historical input hashes and Git blobs,
the derivation recipe and the old analyzer that recorded its rejection.
The recipe removes two unsupported cross-table rules from the 66-rule input:
`federal_2025_t1170_total_gain_eligible_zero_inclusion` and
`federal_2025_t1170_total_gain_subject_to_50_percent_inclusion`. It removes
exactly eight expected output entries from the 82-case historical companion,
then appends the positive and refusal cases from the named public records.
It retains each printed table's totals and destinations. All 14 historical
T1170 fixture files keep their original bytes.

The benchmark candidate and companion match the raw input pair in
`tests/fixtures/t1170-printed-table-diagnostic/` and the inputs that
`tests/test_capital_gift_real_engine.py` executes on the supported
`89571cc2a938707fd60a5489b951135345697296` engine profile. The derived files
carry no historical encoder-output claim, runtime observations, signed
production artifact or source authority. The separate synthetic projection
of historical compiler and process records supplies unit-test inputs; it
does not supply fresh runtime evidence.

The replay at `d4292fb5e7aa78af75912ff97fcd34c0ff535c0a` and the current
snapshot each retain 35 findings on this fixture. Offline replay supplies
no private execution evidence or authenticated release. The fixture therefore
remains a rejected example, not a clean control. `public-source-context/p113.txt` retains public
reproduction context; replay does not pass it to the analyzer or grant it
source authority.

The separate maintained Schedule N test,
`tests/test_investment_adjustment_real_engine.py`, executes 34 input cases
on the supported 89571 profile: 33 positive cases and one expected refusal.
It checks live source certification and rejects absent or copied execution
evidence. On engine `363c6a6c56f7ad130996b63f7a09cfd4c3d9b021`, the test
executes the 33 positive cases and checks that they grant no 89571 capture
authority. These input-case counts do not count distinct process requests.
Signed test releases authenticate local test fixtures; they do not
authenticate production sources or activate a protected runtime.

### Other public diagnostics

The section 219(g) fixture retains a RuleSpec candidate and six companion
cases from public `rulespec-us` commit `a7c6d805`, plus the decoded public
corpus-row body at `bcf433c5`.
`tests/fixtures/proof-parenthetical-219g/provenance.json` records the Git
origins and body-export method. The separate
`tests/test_public_retained_proof_fixture.py` checks the joint-return proof
excerpt while retaining an unrelated ellipsis refusal. Completeness replay
does not run that proof check.

The T2036 country-case-selection fixture retains an unsigned local repair
candidate and its complete 45-case companion. Its country-condition cases
exercise evidence selection alongside unrelated table cases. Offline replay
retains three finding groups without the authenticated secondary-source
context that ordinary validation uses. The fixture carries no native receipt
or publication result.

The NL479, New Brunswick seniors' home renovation, student-loan interest,
child-credit and retirement-income-transfer fixtures copy existing public
candidate and source bytes. The student-loan companion exports the public
JSON `cases` value without changing scalar types; the retirement-transfer
companion preserves the public JSON
`companion_raw` string as UTF-8 bytes. Their metadata records each export's
origin, JSON pointer and hashes. Offline replay retains findings that require
live execution evidence.

### Snapshot and fixture size

The snapshot covers 47 distinct citations, including federal statutes, CFR
regulations, IRS and FNS guidance, state statutes, manuals, Canadian guidance
and forms, and one Israeli statute. Its 57 fixtures include one Arizona
control whose candidate passed validation before a provenance guard refused
the run. The snapshot pins 12 clean fixtures, 45 refused fixtures and 3,463
findings, with no replay errors or regression allowances.

The snapshot preserves all 56 rows from
`d4292fb5e7aa78af75912ff97fcd34c0ff535c0a` and all 54 rows from integration
base `5218e0fbe47a63dac406c7d891cacdb9b72a7289` without changing any row
field. The derived printed-table fixture adds its 35 findings. Compare clean counts and individual findings as well as totals.

`tests/test_us_replay_benchmark.py` caps the derived printed-table fixture at
786,432 bytes, or twelve 64 KiB blocks. That fixture occupies 762,124 bytes.
All other fixture directories together occupy 5,237,565 bytes and retain
their combined 5 MiB limit of 5,242,880 bytes. The full set occupies
5,999,689 bytes against a 6,029,312-byte ceiling. The test enforces the named
fixture allowance and the other-fixture subtotal separately; another fixture
cannot consume the printed-table allowance.

## Public fixtures and the private held-out set

The repository contains public, visible fixtures. Keep held-out cases and
labels outside it, including case ids, expected outcomes and counts. Also
keep these out of the repository:

- hand failure-class labels (`expected_class` and related keys);
- sources from a private-only corpus, including 12 CFR 1005 and 1006;
- raw failure bundles.

Maintainers build the private held-out set from cases that have never entered
the repository and replay it locally with
`replay --fixtures <private fixture directory>` and
`compare --split <private map>`. `compare --split` prints held-out totals;
`--show-held-out` lists cases for the maintainer. CI replays the public
fixtures and expects no held-out data. Once a case enters the repository,
it stays public and never enters the held-out set.

## The required CI check

The `replay benchmark` job in `.github/workflows/ci.yml` runs
`scripts/replay_us_failure_bundles.py ci`. On a pull request, it checks which
paths changed. It skips replay when the PR changes neither a trigger path
nor another `axiom_encode` module that replay loads. `REPLAY_TRIGGER_PATHS`
covers `src/axiom_encode/harness/`, `bare_structure.py`,
`corpus_resolver.py`, the replay script, fixture directory, `ci.yml`,
`pyproject.toml` and `uv.lock`. Renames count both paths. The merge queue and
pushes to `main` always replay. A manual branch dispatch checks errors,
snapshot drift and controls without a base snapshot.

The job fails when:

1. **Replay errors.** A fixture fails to load or authenticate, the analyzer
   raises an exception, or an import or deferral provider fails to resolve.
2. **A clean fixture or a new control regresses.** The guard protects
   fixtures that the base snapshot pins without findings or errors, and
   controls that the base does not contain. It refuses findings, errors or
   removal unless the PR records a reason in `allowed_regressions`, using
   `snapshot --allow-regression ID=REASON`. A reason cannot excuse a replay
   error; replay errors fail independently.
3. **Findings differ from `snapshot.json`.** Added, removed or reworded
   findings, or added or removed fixtures, require a snapshot rewrite in the
   same PR. The job also checks that the snapshot command would produce the
   committed values and that each new regression allowance excuses a real
   regression against the base.

The job summary lists each fixture's changes against the base snapshot.

### Updating the snapshot

To record an intended change to findings, run `snapshot` against the base:

```bash
uv run python scripts/replay_us_failure_bundles.py snapshot --base-ref origin/main --jobs auto
```

Commit `tests/fixtures/us_replay_benchmark/snapshot.json` and review its diff,
which gives each finding its own line. The command refuses replay errors and
requires `--allow-regression ID=REASON` when a base-clean fixture or new
control regresses. Each rewrite keeps the reasons you supply again. The CI
gate counts reasons that the PR adds; an unchanged reason from the base
neither excuses a new regression nor fails another PR.

`snapshot --check` reports finding drift without writing.

When two PRs change findings, their snapshots can conflict, including in
`totals`. Merge `main` into the second PR and rerun `snapshot`.

## Rule for false-positive fix PRs

Every PR that fixes a validator false positive must:

1. Add the failing candidate as a fixture with `extract`, using public
   sources. Keep private-only sources and held-out cases outside the
   repository, and keep held-out run ids out of the PR body.
2. Rewrite the snapshot and remove the new fixture's false findings.
3. Explain every change to other fixtures in the PR body. Check each removed
   finding against that fixture's `source.txt`; removing a true positive
   weakens the validator. Explain added findings too.

## Limits

- Replay runs the completeness analyzer. It does not compile candidates,
  execute companion tests, call oracles, check proof excerpts, run literal
  grounding, apply guards, invoke reviewers or sign results. It does not
  pass `source_context`. A clean replay can still fail other validation.
- Replay reads context from the retained workspace snapshot.
  `extract --rulespec-root` fills missing modules from a RuleSpec checkout
  at the run's `rulespec_ref`.
- Fixed candidates measure validator changes. They do not measure whether
  a prompt helps a model write better candidates. Use
  `axiom-encode eval-suite` for that.
- Fewer findings can reflect either a false-positive fix or a weaker check.
  Check each removed finding against the fixture's `source.txt`.

## Comparing two encoder commits

1. On `main`, run
   `uv run python scripts/replay_us_failure_bundles.py replay --jobs auto --out before.json`.
2. On your branch, run the same command with `--out after.json`.
3. Run `uv run python scripts/replay_us_failure_bundles.py compare before.json after.json`
   (add `--split <map>` for a private held-out set).

`compare` reports totals per split, added and removed findings by rule,
newly clean and newly failing fixtures, and each changed fixture. It leaves
out fixtures that error on either side and lists them under `ERRORED`.

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
   the new fixture directory. Rerun `extract` with
   `--rulespec-root <RuleSpec checkout at the run's rulespec_ref>`. The fixture
   records each copied module's Git blob id for comparison with that ref.
3. For a control, add `"recorded_verdict": "<verdict>"` to `fixture.json`.
4. Run `snapshot --base-ref origin/main`.

The failure-artifact upload in `.github/workflows/targeted-signed-reencode.yml`
configures 90 days of retention. Check a run's artifact availability before
trying to download it.

For a larger uncommitted set, run
`python -I scripts/replay_us_failure_bundles.py replay --bundles bundles/ [--latest-per-citation]`
or `replay --fixtures <private fixture directory>`. The script treats bundles
as untrusted data: it parses their files, refuses symlinks and keeps file
paths inside the bundle.

# US replay benchmark

The US replay benchmark gives every validator PR an offline before/after
measure on committed candidates. It needs no model call, network, rules engine
or corpus checkout. The `replay benchmark` CI job runs it as a required check.

## What it measures

`scripts/replay_us_failure_bundles.py` replays `analyze_complete_source_unit`
(the `[complete-source-unit:*]` rules of the CI validator) over 56 committed
fixtures in `tests/fixtures/us_replay_benchmark/`. For each hosted fixture,
`extract` copies a real candidate from the `targeted-reencode-failure` artifact
of a `targeted-signed-reencode.yml` run:

- `candidate.yaml` and `candidate.test.yaml`: the final candidate and its
  companion tests. For a rejected candidate, their hashes match the bundle
  inventory and `issues.json`.
- `source.txt` and `source-metadata.json`: the exact workspace source. Its hash
  matches the resolver attestation.
- `issues.json`: the issues that the pipeline recorded for that run.
- `context/`: only the workspace context modules that the candidate imports or
  names as a deferral provider (`blocked_by`).
- `providers/`: a module that `extract` copies from the RuleSpec repository
  at the run's `rulespec_ref` when the candidate needs it and the workspace
  context lacks it.
- `fixture.json`: the run id and URL, citation, lane, runner, encoder commit and
  version, corpus and RuleSpec refs, the failure classes that the run recorded,
  the origin of each context module and the SHA-256 of every file. It may also
  set `recorded_verdict` (default `rejected`; any other value marks a control, a
  candidate that production validation accepted) and `split` (`visible`, the
  default, or `held_out`; only a private fixture set uses `held_out`).

The repository also holds local diagnostic failures with explicit
`origin: local-diagnostic` provenance. Each keeps exact public source,
candidate and companion bytes, the replay findings that rejected it, a producer
commit and tree, the reproducible replay command, and byte-identical repository
input origins. None can claim a hosted run URL, signed source attestation or
native bundle inventory. The same file hashes, public-only restrictions and
snapshot gates apply. A local fixture records a rejected example, never a
production control.
The Schedule 2 floor fixture uses this route, and offline replay keeps its
findings because replay has no live execution evidence. Its separate
real-engine test verifies the targeted floor witnesses. The NL479 fixture
retains an earlier primary-document diagnostic, separate from the full
parent-form engine replay; it does not inherit that replay’s execution evidence.
The T936 printed-row fixture preserves an unsigned source-only diagnostic pair; its offline replay
does not receive private execution evidence or authenticated resolver context.
The NB Schedule 12 and Schedule M fixtures retain complete public diagnostic pairs.
The Schedule M companion exports the verbatim `/cases` value from its public
JSON record, with explicit origin hashes; it does not reconstruct historical
engine companion bytes. Offline replay retains six findings for each fixture
because it supplies no captured execution evidence. Separate Schedule M real-engine tests
check the supported and unsupported engine profiles.
The section 219(g) fixture retains a public RuleSpec candidate and its six
companion cases from `rulespec-us` at `a7c6d805`, plus the decoded body of the
public corpus row at `bcf433c5`.
`tests/fixtures/proof-parenthetical-219g/provenance.json` records the public Git
origins and body-export method. The separate
`tests/test_public_retained_proof_fixture.py` checks the joint-return proof
excerpt while retaining an unrelated ellipsis refusal. Completeness replay
does not measure that proof check.

The `local-date-silent-merge-41088317595` fixture holds a syntax mutation of
the public fixture `41088317595`, an Arizona manual unit that prints no date.
No encoder run produced the mutation. Its candidate moves the `0001-01-01`
start and formula `3` of `minimum_allotment_actual_benefit_participant_cutoff`
from an explicit version onto the rule itself. It also adds an unused `<<`
merge mapping holding a version that starts on `2025-10-01`, a date the source
never prints. The rules engine ignores `<<`: at engine commit `363c6a6c`, the
parent and the mutation compile to identical bytes, and all 14 companion cases
pass for each. The mutation leaves every formula and companion expectation
alone. Its source, companion, source metadata and imported context match the
parent byte for byte, so it keeps the parent's `source_as_of` of 2026-07-17.
`fixture.json` labels the case a mutation rather than a historical encoder
output and names the parent's input origins, and
`tests/fixtures/date_silent_merge_start/candidate.yaml` holds the same
candidate bytes. Its `issues.json` keeps the one start-date finding that the
analyzer at encoder commit `36deb0c9` reported on the mutation, where it
reported none on the parent. The current analyzer reports no finding on either,
so the snapshot pins the fixture clean. The fixture tests validator and engine
behavior on retained public inputs and does not track current Arizona law.

The `local-date-proof-alias-41648809989` fixture mutates the proof fields of
public `41648809989-control`. It moves `metadata.proof` to `proof` on
`expanded_categorical_eligibility_fpl_limit_rate` and changes that rule's
`effective_period` atom from `source.excerpt` to `source.quote`. Reversing
these two aliases recovers every parsed value and type in the parent
candidate. The source, source metadata and companion retain the parent's
exact bytes; neither fixture supplies imported context. `fixture.json`
identifies a local diagnostic mutation and records its raw recipe and input
origins. `tests/fixtures/date_silent_proof_alias/candidate.yaml` holds the same
candidate bytes.

Its `issues.json` records one unsupported `2026-03-01` start from the official
replay CLI at encoder commit `e85c629eb`. The analyzer at `b89c050a` reports no
finding on either the mutation or its parent, so the snapshot pins the
mutation clean. Separate controls at engine commit `363c6a6c` compile both
candidates to identical 19,982-byte outputs, and both unchanged 15-case
companions pass. Separate proof-validator controls also give both candidates
the same result. The proof validator accepts all 20 atoms structurally and
reports four unresolved-source findings with the retained source-text map for
`us-az/manual/des/archived-policy/2026-03-23-whats-changed`. The fixture lacks
that change-notice body, so these results do not establish full release-bound
proof authentication.

The T2036 country-case-selection fixture retains an unsigned local repair
candidate and its complete 45-case companion. Unrelated table cases must not
disable evidence that the country-condition cases supply. Offline replay still
retains three finding groups because it does not receive the authenticated
secondary-source context that ordinary validation uses; the fixture provides
neither a native receipt nor a publication result.

Five Canadian form fixtures cover NL479, the New Brunswick seniors' home
renovation schedule, student-loan interest, child credits and retirement-income
transfers. They copy existing public candidate and source bytes. The
student-loan companion exports the existing JSON `cases` value without
changing scalar types; the retirement-transfer companion preserves the
existing JSON `companion_raw` string as UTF-8 bytes. Their metadata records
each export's origin, JSON pointer and hashes. Offline replay retains findings
that require live execution evidence.

The replay passes the same inputs as `ValidatorPipeline._complete_source_unit_issues`:
the citation's numeric profile, the companion test cases, imported symbol
contents, resolved deferral providers and same-act aliases from the source
metadata. `--no-context` leaves out the last three. It also passes the
attestation's `source_as_of` and `expression_date` to the date-silent
start-date check (`docs/validator-checks.md`), always without amendment
context, because a fixture records no amendment documents.

The set has 45 distinct citations: federal statutes, CFR regulations, IRS and
FNS guidance, state statutes, manuals and guidance, eight Canadian citations
and one Israeli statute. Its 56 fixtures include one Arizona control that passed
validation and then failed a provenance guard. The current snapshot pins 14
clean fixtures and 42 fixtures with 3,345 findings between them, and it records
no replay error and no `allowed_regressions` entry. The proof-alias fixture
adds one clean row; all 55 earlier rows keep their exact findings. The
child-credit and retirement-transfer diagnostics retain six and 19 findings;
the section 219(g)
diagnostic retains 33. Read clean counts and per-fixture changes, not only the
total.

`tests/test_us_replay_benchmark.py` caps every other fixture together at
5 MiB and each of these named cases separately at 64 KiB:

- `local-date-silent-merge-41088317595`: 61,524 bytes;
- `local-date-proof-alias-41648809989`: 38,278 bytes.

The budget counts files inside each fixture directory.

## Public fixtures and the private held-out set

The repository commits only public, visible fixtures. It holds no held-out
case and no held-out label: no held-out case id, expected outcome or count. It
also never holds:

- hand failure-class labels (`expected_class` and related keys);
- sources from a private-only corpus (12 CFR 1005 and 1006);
- raw failure bundles.

A maintainer builds the held-out set only from cases that nobody has committed.
It stays in the maintainer's private benchmark, outside the repository, and a
maintainer replays it locally (`replay --fixtures <private fixture directory>`
and `compare --split <private map>`). `compare --split` prints held-out cases
as totals only (`--show-held-out` lists them, for the maintainer). The CI job
replays only the committed fixtures and expects no held-out data. Once anyone
commits a case, it stays public and never goes back into the held-out set.

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
controls only. The benchmark's author earlier reported that the replay took
20 to 55 s with 4 workers and the whole job 67 s on a GitHub runner.

The job fails when:

1. **The replay errors.** A fixture does not load or authenticate, the
   analyzer raises, or an import or deferral provider does not resolve.
2. **A clean fixture or a control regresses.** A fixture that the base
   branch's `snapshot.json` pins with no finding, or a control, now has a
   finding or an error, or the PR removes it. Only a reason in the PR's own
   `snapshot.json` lets it through: `snapshot --allow-regression ID=REASON`
   writes the reason under `allowed_regressions`, and a reviewer sees it in
   the diff.
3. **Any finding differs from the committed `snapshot.json`.** The snapshot
   holds every fixture's exact findings. If a PR removes, adds or rewords a
   finding, or adds or removes a fixture, the job fails until the same PR
   rewrites the snapshot. The committed file must also equal what the
   `snapshot` command writes, so a hand edit to counts or totals fails, and
   every `allowed_regressions` reason must excuse a real regression against
   the base.

The job's summary lists every fixture that differs from the base branch.

### Updating the snapshot

When you intend a change to the findings:

```bash
uv run python scripts/replay_us_failure_bundles.py snapshot --base-ref origin/main --jobs auto
```

Commit `tests/fixtures/us_replay_benchmark/snapshot.json`. The rewrite gives a
reviewable diff, one line per finding. The command refuses to pin replay
errors. It also refuses to pin a finding on a fixture that the base pins clean
(or on a control) unless you pass `--allow-regression ID=REASON` for it. A
rewrite keeps only the reasons that you pass again. The gate counts only the
reasons that a PR adds: a reason that the base already carries neither excuses
a new regression nor fails other PRs.

`snapshot --check` reports drift without writing.

Two PRs that both change findings can conflict in `snapshot.json`. Merge
`main` into the second one and rerun `snapshot`.

## Rule for false-positive fix PRs

Every PR that fixes a validator false positive:

1. Adds the failing candidate as a fixture (`extract`, below), from public
   sources only. A candidate from a private-only corpus or a case of the
   private held-out set stays out of the repository; do not name a held-out
   run in the PR body.
2. Rewrites the snapshot, and the new fixture replays without its false
   findings.
3. Explains, in the PR body, every change to other fixtures. It checks each
   finding that the PR removes against that fixture's `source.txt`, since
   removing a true positive counts as a regression, and explains each finding
   that the PR adds.

## Fidelity

The benchmark's author earlier reported replaying the hosted fixtures with
each run's encoder source. In that report, almost all hosted replays reproduced
the completeness findings that each run recorded, exactly or within two
findings, with no replay errors. In three 7 CFR 273.7 runs from that report,
production stopped before the completeness check, and the offline replays
found 653 to 695 findings each. Local diagnostics record fresh offline findings
and producer metadata separately.

## Limits

- Only the completeness analyzer runs. The benchmark does not measure
  compilation, companion-test execution, oracles, proof-excerpt checks (the
  replay passes no `source_context`), literal grounding, guards, reviewers or
  signing. A fixture that replays clean can still fail CI, and many rejected
  candidates also had a finding of one of those kinds.
- Context comes from the workspace snapshot, not from the live checkout.
  `extract --rulespec-root` fills a gap from a RuleSpec checkout at the run's
  `rulespec_ref`.
- The benchmark replays fixed candidates and measures validator changes. It
  does not show whether a prompt change makes the model write better
  candidates. Use `axiom-encode eval-suite` for that.
- A lower count can come from removing a false positive or from weakening a
  rule. Check every finding that a change removes against the fixture's
  `source.txt`.

## Comparing two encoder commits

1. On `main`, run
   `uv run python scripts/replay_us_failure_bundles.py replay --jobs auto --out before.json`.
2. On your branch, run the same command with `--out after.json`.
3. Run `uv run python scripts/replay_us_failure_bundles.py compare before.json after.json`
   (add `--split <map>` for a private held-out set).

`compare` reports totals per split, finding removals and additions by rule,
fixtures that turn clean or start failing, and every fixture that changes. It
leaves out a fixture that errors on either side and lists it under `ERRORED`.

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
   records the git blob id of each module that `extract` copies, so a reviewer
   can compare it with the RuleSpec repository at that ref.
3. For a control, add `"recorded_verdict": "<verdict>"` to `fixture.json`.
4. Run `snapshot --base-ref origin/main`.

The signed re-encode workflow (`.github/workflows/targeted-signed-reencode.yml`)
uploads each failure artifact with `retention-days: 90`.

To replay a larger, uncommitted set, run
`python -I scripts/replay_us_failure_bundles.py replay --bundles bundles/ [--latest-per-citation]`
or `replay --fixtures <private fixture directory>`. Treat bundles as untrusted
data: the script only parses them, refuses symlinks and keeps every path
inside its bundle.

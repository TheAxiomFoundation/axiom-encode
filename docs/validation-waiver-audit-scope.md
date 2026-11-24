# Validation waiver audit scope

`axiom-encode validation-waivers audit` re-executes waived modules and compares
each result with the waiver ledger (`known-validation-gaps.yaml`). It has two
scopes.

- `--scope full` (the default) audits every waiver of the ledger.
- `--scope incremental` audits only the waivers whose result the changed paths
  can alter, and falls back to the full audit whenever it cannot prove that.

The ledger transition checks against the protected base run in both scopes,
before any selection.

## What the incremental scope proves

The incremental audit proves one statement: every waiver it did not audit has
the same result on the head tree as on the base tree. It does not re-prove that
the base tree's ledger is correct. That needs a full audit of the base commit
with the same verifier, corpus release and engine. So the full audit must keep
running where the base is established: on push to the default branch, on the
scheduled run, and in the merge queue.

Two things can change a waiver's result with no changed path in the
repository: a different verifier (encoder) and a different toolchain. The
toolchain is pinned by files under `.axiom/`, and any change of those files
forces the full audit. A verifier that is not pinned by a repository file
(verifier `main` mode) can change between two runs with no changed path. The
incremental audit cannot see that. The full audit on push, on schedule and in
the merge queue is the control for it.

## How a changed path is classified

Every changed path must be positively classified. One unclassified path forces
the full audit.

| Kind | Paths | Effect |
|---|---|---|
| module | `<jurisdiction>/{statutes,regulations,policies,legislation}/**/<name>.yaml`, a `rulespec/v1` file | selects by the rules below |
| companion | `<module>.test.yaml` next to an existing `rulespec/v1` module | selects the waivers that read it |
| manifest | `.axiom/encoding-manifests/<module path>.json`, where `<module path>.yaml` is a module path and that module exists as a `rulespec/v1` file | selects exactly what a change of that module selects |
| inert | `*.md` at the repository root or under `docs/` | selects nothing |
| anything else | see below | full audit |

A signed encoder pull request adds a module, its companion and its applied
manifest. All three are classified, so its scope comes from the module.

The manifest rule is exact. These paths are **not** manifests of a module and
force the full audit:

- every other path under `.axiom/`: `toolchain.toml`, `workflow-toolchain.toml`,
  `policyengine-runtime.toml`, `pending-validation-fingerprints.json`,
  `target-validation-passes.json`, `reviewed-migrations.json`,
  `repository-structure.yaml`, `validation-waiver-audit-costs.json`,
  `legacy-replacements/**`, any index;
- a file under `.axiom/encoding-manifests/` that does not end in `.json`, or
  whose path is not a module path: no jurisdiction (`statutes/26/1.json`), a
  ProgramSpec (`programs/...`), a companion (`<name>.test.json`), a path with
  `.axiom`, `.github` or `programs` inside it;
- a manifest directory that is not the repository's own
  (`<jurisdiction>/.axiom/encoding-manifests/...`);
- a manifest whose module does not exist or is not a `rulespec/v1` file;
- a manifest of another country's jurisdiction;
- a deleted manifest, a symlink, a hard-linked file, a directory at the
  manifest path, and a manifest inside a configured execution input.

The rules that held before still hold: the waiver ledger, `.github/**`,
`programs/**`, a deleted path of any kind, a symlinked or hard-linked changed
path, a module of another country, a path inside the configured corpus, engine
or composer, a changed Git configuration input, a checkout with a symlink, and
a file that is not valid `rulespec/v1` at a module path all force the full
audit.

## Which waivers a classified change selects

A waiver is selected when one of these holds for one changed path. A manifest
counts as its module.

1. The changed path is the waived module, or its companion.
2. Validation reads the changed path: it is in the transitive closure of the
   waived module and its companion over imports and canonical targets. The
   closure keeps every candidate path of every target, present or absent, so a
   target that is added or deleted is covered.
3. Validation searches the changed module: the changed module is in the
   waiver's own jurisdiction or in an ancestor jurisdiction. The validator's
   placement check indexes every module of those jurisdictions.
4. The waiver is a composition (`module.kind: composition`) and any module
   changed. Composing a program loads and indexes every module of the country
   checkout, so a module in any jurisdiction is an input.
5. The waiver is a composition, only companions changed, and a module anywhere
   in the country names the changed companion as a target.
6. The waiver's own read set cannot be classified: a file of its closure does
   not parse, has an import that cannot be classified, or has a directory at a
   dependency path. That waiver is audited on every run, whatever changed.

Rule 6 is per waiver. Each waiver is decided from its own read set and the
changed paths, never from another waiver. So a waiver with an established read
set is still decided normally when another waiver's closure cannot be read. A
filesystem error is not a property of one waiver: it audits every waiver the
selector was given.

For a new module in a state jurisdiction this means: the waivers of that state,
plus every composition waiver of the country, plus the waivers of rule 6. A new
federal module selects every waiver, because every jurisdiction searches its
federal ancestor.

## Cost

The selector runs inside each audit job on that job's own partition of the
ledger. A partition's selection is its share of the whole ledger's selection.

Every repository-wide input is computed once per run: the checkout admission,
the candidate content roots of a jurisdiction, the placement-search roots of a
jurisdiction, and the parse of each file. A waiver's decision is then a walk
over a shared dependency graph, and the walk's cost follows the waiver's own
closure.

Measured on `rulespec-us` main at `25ddba37` (1,930 waivers, 4,888 atomic
modules), one process, on a loaded workstation:

| Selector run | Before | After |
|---|---|---|
| One Colorado waiver, one changed Arizona module | not finished after 240 s | part of the rows below |
| Whole ledger, signed Arizona change (module, companion, manifest) | not finished after 1,500 s | 77 to 88 s |
| One of 52 partitions (37 or 38 waivers), same change | not measured | 3.0 to 4.9 s (12 partitions measured) |
| Whole ledger, companion-only change (needs the country-wide walk of rule 5) | not measured | 120 s |

Before, each target resolution re-audited the whole checkout for symlinks
(about 3,100 directories, 25 ms) and re-validated the routing caches. One
Colorado composition names several thousand targets, each resolved four times,
and no parse or resolution was shared between waivers. In a 240 s profile of
one waiver, 188 s were 7,631 checkout symlink audits (25 million `lstat`
calls), and only 333 files had been parsed.

## Evidence that unselected waivers do not change

Three kinds of evidence support the statement above.

**Real-engine invariant tests** (`tests/test_validation_waiver_incremental_invariant.py`,
CI job `real-engine`). They build small repositories, run the full audit
before and after an edit, and require every waiver whose verdict changed to be
selected. The edits include an applied manifest alone, and a new module with
its companion and manifest in a state, in a sibling state and in the federal
jurisdiction.

**A resolver comparison** (`tests/test_validation_waiver_scope_signed_pr.py`).
The shared graph's closure is compared with a closure built from the
validator's own resolver functions, and its import candidates with the
validator's candidates for prefixed, unprefixed, absent and refused spellings.

**A fingerprint replay on `rulespec-us`.** The replay computes fingerprints
without the audit's ledger comparison, with the module-level entry point the
audit's workers use, and compares two trees.

The head tree is `rulespec-us` main at `25ddba37`. A base tree is that tree
with the module, companion and manifest files of past pull requests put back to
their state before each pull request: a file the pull request added is removed,
a file it changed is restored. 80 merged pull requests with one to five changed
module files are used, in 44 states, each with files that main still holds
unchanged. They are split by state into two groups. One base tree reverts
group A, one reverts group B and one reverts both.

For each base tree the selector runs on the head tree with the reverted paths
as the changed paths. Every replayed waiver outside that selection must have
the same fingerprint in both trees. The waivers of a group's own states are
inside its selection, so they are replayed against the other group's tree.

Two more base trees test one class of change each: main with no applied
manifest at all, and main with only the companions of the 80 pull requests
reverted.

The fingerprints use encoder main, the engine and corpus release that
`rulespec-us` pins, and no composer.

## Limit: composition waivers

Rule 4 is necessary with the composer as it is. It reads every YAML file under
the country checkout and builds a name index over all modules before it
composes one program. So the encoder cannot prove that a module outside a
program's scope leaves the composed program unchanged.

On `rulespec-us` main at `25ddba37`, 109 of the 1,930 waivers are
compositions: 100 generated tariff chapters, 6 state SNAP benefit
calculations and 3 federal compositions. By the per-module audit times of one
hosted run they are 2,164 of 5,524 worker-minutes (39%). A signed Arizona
change therefore selects 141 waivers: 30 Arizona waivers (108 worker-minutes),
108 compositions outside Arizona (2,145 worker-minutes) and 3 waivers of
rule 6. That is 41% of the full audit's worker time, not 2%.

Removing that cost needs a change outside this selector. Two options:

- The composer reports the files a composition used, and the validator
  records them, so the selector can use a real read set for a composition.
- The audit keeps a digest of each composed program. A composition whose
  program composes to the same bytes, with the same owner, needs no
  re-validation.

## Adoption in a RuleSpec repository

This document describes the encoder only. The reusable validation workflow
does not pass `--scope incremental` yet. These later changes turn it on.

1. **Verifier.** The repository's validation must run an encoder that contains
   this selector. Either move the repository's encoder pin to such a version,
   or run the verifier in `main` mode.
2. **Reusable workflow** (`validate-rulespec.yml`, step "Enforce validation
   waiver ratchet"). Add an input, for example
   `validation-waiver-audit-scope` with the values `full` (default) and
   `incremental`. Pass `--scope incremental` to `validation-waivers audit` only
   when all of these hold: the input is `incremental`, the event is
   `pull_request`, the verifier's `validation-waivers audit --help` lists
   `--scope` and both partition flags, and the run is not a verifier restate
   (`--verifier-restate` requires the full scope). Keep the present command,
   which is the full audit, for `push`, `schedule`, `workflow_dispatch` and
   `merge_group`, and for a verifier without the flag. The audit already
   receives the changed paths and the protected base ledger; nothing else
   changes.
3. **Caller** (the RuleSpec repository's `repository-checks.yml`). Move the
   reusable-workflow pin to the commit of step 2 and pass the input.
4. **Leg selection.** Import-aware leg selection must stay off until the
   incremental audit is on. With a scoped matrix and the full audit, one leg
   owns the whole ledger.

The full audit stays on push to the default branch, on the scheduled run and
in the merge queue. A pull request that changes a toolchain file, the ledger,
a workflow, a ProgramSpec or any unclassified path still gets the full audit
from the selector itself.

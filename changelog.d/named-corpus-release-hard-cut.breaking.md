Require each RuleSpec checkout to pin a verified release-object/v2 through
`[toolchain].axiom_corpus_release` and
`[toolchain].axiom_corpus_release_content_sha256`. Remove the mutable `current`
release, selector-only identity, release-validation opt-outs, legacy corpus
layouts, ambient checkout discovery, and the remote Supabase corpus resolver.
Corpus resolution now has one content-addressed local provenance path.

Bind the canonical validation-waiver set into the RuleSpec toolchain and every
new applied-encoding manifest. Remove unattested v1 manifests, hard-coded UK
corpus aliases, the raw source-text pinning helper, and the legacy
`--corpus-root` CLI spelling; callers must migrate to canonical release-owned
paths and `--corpus-path`.

Persist the named release identity and canonical RuleSpec content root in eval
suite state and ledgers, reject mixed-root eval invocations, and fail resume or
revalidation when that identity changes. Retire mutable corpus claim artifacts:
RuleSpec proofs must cite release-bound corpus provisions or explicit hashed
RuleSpec imports.

Remove the eval-suite `source_id` field and the `encode --source-id` and
`eval-source --source-id` overrides. Every encoding now has one machine
identity: its canonical corpus citation, resolved into a release-bound source
unit whose attestation is carried through results and validation. Judge
regeneration rejects modules whose checked-in path differs from the canonical
path derived from that citation; migrate or re-encode them instead of replaying
an old logical path. `name` remains a display label only. The loader rejects the
removed field rather than treating it as an alias. Existing suite outputs use
the old identity contract and must be rerun into a fresh output directory; they
cannot be resumed, revalidated, reported, or archived under the new contract.
Eval-suite result/evidence schema v5 and verdict schema v5 bind every generated
RuleSpec, model trace, and context manifest to its exact SHA-256 digest. Resume,
revalidation, report, and archive admission reject missing, changed,
cross-runner, or non-canonical
result artifacts. Dedicated Ed25519 evidence authenticates the immutable run,
manifest, case, runner set, corpus release, encoder, engine, RuleSpec, toolchain,
waiver, generation, cost, access, source, and validation identities outside the
mutable output tree. Admission rejects old schemas and replay across runs,
releases, cases, or toolchains before it reruns compile, CI, reviewers, and
configured oracles. Eval-suite manifests are exact schemas with manifest-only
runners and mandatory finite readiness gates. Maximum-cost gates fail closed on
missing or invalid per-result accounting.

Require `guard-generated` to receive the exact local `--corpus-path`. The guard
now re-resolves every model source attestation through the RuleSpec-pinned,
signed release object and rejects validly signed apply manifests whose claimed
artifact digest, path, row identity, selector, or source bytes differ from that
release.

Require each RuleSpec module to declare exactly one resolver-attested
`module.source_verification.corpus_citation_path`; the plural
`corpus_citation_paths` field is rejected even when it contains one entry.
Encode additional legal sources as separately attested modules and import them.
Remove the CMS eligibility-level and CHIP-composition deterministic generators
and the one-shot SSA POMS optional-supplement migration repair instead of
grandfathering their multi-source output contracts. Remove the multi-source
tax-status component repair, the manual sign-only apply surface, and its legacy
manifest category. `manifest-census` now reports only model-generated and
unmanifested files; legacy or unknown backends are unmanifested rather than
receiving a compatibility classification.

Bump applied-encoding manifests to `axiom-encode/applied-rulespec/v5` and
reject earlier schemas outright. Model manifests carry mandatory
validation-execution provenance bound to the exact clean pinned encoder commit
and version, and must use the exact `axiom-encode encode --apply` command.
Legacy deterministic manifests are rejected and must be migrated by
re-encoding.

Protect the canonical atomic `legislation/` RuleSpec root alongside statutes,
regulations, and policies. Whole-repository guard runs now include
protected paths deleted between their exact base and head revisions and fail
closed on an empty RuleSpec corpus, so deleting content and its manifest cannot
bypass provenance enforcement.

Changed-file discovery now compares the exact base and head revisions with
rename detection disabled and consumes Git's NUL-delimited path stream, so
renames out of protected roots and unusual filenames cannot hide deletions.
Centralize the RuleSpec layout on five filesystem roots and split it from the
four atomic module roots used across validation, evals, waivers, hashing,
judges, imports, and apply manifests. `programs/` contains declarative
`axiom-compose` ProgramSpecs and never enters an atomic encoder surface. Reject
`.yml` files rather than maintaining a second spelling. Remove the
flat-repository-only California SNAP
shelter-surface repair command and its guidance migration instead of carrying a
one-shot layout compatibility surface into launch.
Remove every policy-, jurisdiction-, and section-specific deterministic repair
command, including the remaining generic repair commands, and remove the
deterministic applied-manifest backend. The sole v5 writer is the atomic,
fully validated model `encode --apply` transaction; signed `retire` may only
transform an already verified model manifest. Reusable deterministic
transformations remain private implementation details inside the model overlay
where still needed, and old deterministic manifests must be re-encoded rather
than grandfathered.

Serialize apply and retire under a private checkout lock and a bounded,
canonical transaction-journal v2. The journal records exact regular-file
preimages, modes, and intended digests through no-follow file access; recovery
verifies the entire journal before changing any path and restores or completes
the transaction after interruption, including process death between file and
manifest replacement. Apply and retire recover before checking new command
requirements, while every other command fails closed when a pending journal is
present. Overlapping or inconsistently grouped retire manifests are rejected.

Replace shared-secret apply-manifest HMACs with scope-domain-bound Ed25519
signatures. Verification uses pairwise-distinct apply, eval, and corpus-release
roots from one protected config; environment values cannot define trust.
Signing requires the externally attached broker capability. The model-capable
Python CLI rejects all legacy/current private-key environment inputs, and there
is no shared-secret, in-process private-key, or old-schema fallback.

Add `oracle-coverage --fail-on-pending` so release workflows can reject
`pending_classification` declarations as strictly as unmapped outputs. A
pending debt file is no longer a release-admission substitute for an exact
oracle mapping or explicit not-comparable disposition.

Run validation pipelines from the exact jurisdiction content root selected by
each RuleSpec module. Country checkouts remain the toolchain boundary, but are
never passed to the compiler as though they were a jurisdiction root.

Make reporting, migration inventory, pending coverage, SNAP readiness, run-log
export, and source-attestation commands take explicit canonical country or
jurisdiction roots. Remove sibling/workspace/current-directory discovery and
pending-file mutation, fail migration inventory on an empty RuleSpec corpus,
and persist the exact requested and resolved canonical citation identities.

Bind `preclassify`, `judge-fidelity`, and `judge-grid` to an explicit canonical
RuleSpec checkout, local corpus checkout, and release-owned citation worklist.
Remove raw source-text, source-file, provision-file, and public calibration
inputs. Every judge event now carries the full resolver source attestation.

Make concept audit scan exactly the four canonical atomic RuleSpec roots. Symlinks,
legacy `.yml`, malformed `.yaml`, and parse failures are errors instead of
being silently skipped.

Rename SNAP readiness output from repository-shaped `repo`, `repo_path`, and
`total_repos` fields to canonical `module`, `module_path`, and `total_modules`
fields. The command emits no compatibility aliases.

Remove `proof-validate --money-atom-root` and its current-directory fallback.
Money-atom ratchet paths now derive only from each module's exact canonical
`rulespec-<country>` checkout, including multi-checkout invocations.

# Atomic unmanifested-legacy RuleSpec cleanup

`cleanup-unmanifested-legacy` removes obsolete RuleSpec that predates trusted
encoder ownership. It is a contraction with signed negative provenance, not an
encoding, retirement, migration, replacement, or import.

## Atomic topology

The only accepted publication graph is:

```text
protected base B
  -> validate and sign B-bound preimages
  -> one durable transaction: add receipt + delete exact groups
  -> stage that exact set
  -> one PR-head commit H
```

`H` must have exactly one parent, `B`, and `B..H` must consist of exactly one
new receipt plus the receipt-authorized primary and companion deletions. A
deletion-candidate commit followed by a receipt-only child is rejected. It
cannot survive squash/rebase faithfully and must not appear in code, tests,
documentation, artifacts, or publication instructions.

The receipt cannot bind `H`, its own path-dependent full digest, or the final
tree containing itself. Instead it binds:

- the canonical `github.com/TheAxiomFoundation/rulespec-xx` repository and Git
  object format;
- `B`, the exact tree at `B`, and the projected post-deletion tree before the
  receipt is inserted;
- each primary and mechanically derived companion as a regular `100644` base
  blob, including its object ID and SHA-256;
- the complete deletion set, with between one and 64 sorted, unique groups;
- immutable encoder, rules-engine, RuleSpec-dependency, corpus-release,
  toolchain-file, and validation-waiver pins;
- fail-closed ownership and surviving-reference inventories computed from
  immutable base blobs; and
- the output hashes and target lists of the validation checks actually
  executed against the projected deletion state.

Receipt JSON is strict and bounded. Its filename is the semantic-identity
SHA-256 as a direct child of
`.axiom/legacy-rulespec-deletion-receipts/`. A cleanup-specific inner signature
domain is wrapped by the protected `apply_ed25519` broker capability, preventing
replay as an applied manifest, migration receipt, evaluation artifact, or an
untyped apply payload. Historical receipts are append-only.

## Admission and validation

The caller supplies canonical primary paths only. The command derives every
`.test.yaml` companion and rejects missing companions, primary-only groups,
symlinks, executable files, non-blobs, ambiguous index stages, overlaps,
duplicates, noncanonical paths, and a dirty checkout. The full base commit must
equal clean `HEAD`.

"Unmanifested" is proved from the immutable base, not inferred from an empty
live manifest lookup. Admission rejects a claim by any canonical or
supplemental applied manifest, retirement, path migration, legacy replacement,
manual/deterministic ownership record, prior cleanup receipt, or malformed,
oversized, duplicate-key, ambiguous, or unreadable provenance record. Deleting
or corrupting ownership evidence in the proposed change therefore cannot make
a generated file eligible for cleanup.

The projected checkout executes the fixed repository test, repository-layout,
waiver, remaining RuleSpec, companion-test, proof, money-atom, oracle-coverage,
and metadata/reference-closure checks. The receipt records exact commands,
target-list hashes, exit codes, and output hashes. No check is marked passed
unless it ran successfully, and `--fast` is not part of this contract.

Receipt creation is ordered before every primary and companion deletion inside
the existing journaled transaction. Under its lock the implementation rechecks
the base/tree, exact blobs and modes, ownership, pins, signer identity, receipt
collision, and planned payload. A receipt-specific postcheck proves the exact
filesystem state while the journal still exists. Recovery restores either the
complete preimage or the complete postimage; it never accepts a receipt-only or
deletion-only state.

## Guard, staging, and publication

`guard-generated` discovers cleanup receipt changes before its generated-file
early exits. Its cleanup authorization is separate from applied-manifest
coverage. It verifies historical receipt integrity and rejects modified,
deleted, renamed, replayed, orphaned, overlapping, stale-base, partial-group,
extra-deletion, and mixed cleanup/apply changes. It then proves `B..H` is the
exact authorized contraction.

`stage-signed-backfill --legacy-cleanup-base-ref B` independently verifies the
same receipt, base proof, signature, toolchain, exact worktree change set, and
post-stage absence before staging explicit paths. It does not use broad
`git add -A` or deletion-blind file overlay.

The dedicated `.github/workflows/atomic-legacy-cleanup.yml` workflow transports
the exact receipt bytes and deletion inventory. Its signer job has no push
token. Its publisher has no signing or model secrets, reconstructs from the
exact protected base, reapplies only listed deletions, reruns the guard, stages
the exact set, and creates one draft PR-head commit. The targeted signed
re-encode workflow does not sign, transport, stage, or publish cleanup evidence.

## Deliberate non-interference

Cleanup receipts are intentionally outside every positive-provenance surface:

- They never enter `.axiom/encoding-manifests/`, applied-manifest coverage, or
  generated ownership. A census simply stops counting the two deleted live
  YAML files per group; its encoder count does not increase.
- They do not weaken `retire`. Retirement still requires one uniquely owning,
  signature-valid current-v5 model manifest and embeds that prior manifest.
- Neither a receipt nor a deleted cleanup target is eligible for
  `signed-import-inventory`, which continues to require a live historical
  signed-v5 model module at the immutable base.
- They create no encoder run ID, run-log event, dashboard funnel credit,
  reconstructed run, or Supabase `data_source=apply_manifest` row. Run-log and
  Supabase scans remain rooted only at `.axiom/encoding-manifests/`.
- They do not participate in model generation, corpus-source ownership,
  validation-waiver ownership, legacy replacement, targeted signed-reencode
  queue/import evidence, or publication credit.
- Path migration and legacy replacement treat path-migration,
  legacy-replacement, and cleanup receipts as immutable persisted provenance.
  If a later requested rewrite would alter any historical signed receipt byte,
  that operation fails instead of rewriting it.

These boundaries make the receipt evidence only for deletion authorization:
it proves why exact base blobs may be absent, but it never claims that they
were generated, validated as an encoding, imported, retired, or published as
an encoding run.

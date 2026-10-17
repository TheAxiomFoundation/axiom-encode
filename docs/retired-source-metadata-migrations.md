# Retired source metadata migrations

`migrate-retired-source-metadata` performs a deterministic, base-bound metadata
migration through the existing protected apply signer. It does not generate
rules or establish source fidelity. Normal repository validation, independent
review, and publication checks still apply.

The exact plan contains `schema_version`, an immutable `base_commit`, and an
explicit `modules` array. The hosted workflow accepts this plan through
`source_bundle_json`; generation and dependent-repair inputs cannot be mixed
with it. The dispatched RuleSpec reference must equal the plan's base commit.

## Versioned transformations

- Plan/receipt v1 retain their original replay behavior: remove retired
  `module.source_verification.values` and collapse plural citation paths only
  when a unique equal or ancestor citation is provable.
- Plan/receipt v2 additionally remove
  `module.source_verification.source_urls`. This requires an existing valid
  singular corpus citation and a nonempty canonical block list of nonempty
  strings. Ordered list values, duplicates, and the exact removed YAML block
  are preserved in the receipt. No URL is fetched, normalized, or deduplicated.

The v2 schema is `axiom-encode/retired-source-metadata-migration-plan/v2`;
its receipt uses `axiom-encode/retired-source-metadata-migration-receipt/v2`.
V1 receipts are replayed under v1 semantics and remain byte-identical.

Other metadata, executable rules, source proofs, dates, and companion files
remain unchanged. The existing complete proof-pin cascade updates affected
import hashes and records each change. Replay reconstructs the entire change
set from authenticated base files; changed history, schema, or postimages fail
verification. Unknown sibling fields are preserved, not silently repaired.

`--dry-run` writes postimages and an unsigned receipt into a scratch checkout;
it is not a read-only operation and its output is not publication evidence.
Protected publication still requires the established signing environment and
matching producer/verifier identities. A successful neutral migration does not
clear unrelated source, ownership, runtime, waiver, or admission failures.

Repository CI must pass `guard-generated --all --base-ref "$PROTECTED_BASE_REF"`
with the base selected by its trusted review or dispatch context. When any
retired source metadata migration receipt is present in the tree, `--all`
requires protected-base admission before accepting its replacement manifests
as generated. Without an explicit protected base, the guard refuses the
receipt and names its path. A receipt's self-declared `base_commit` is replay
evidence; it does not select the trusted review base.

Admission verifies the complete replayed transaction against that base:
receipt, rewritten primary and cascade importers, their replacement manifests,
and every live postimage must agree. The authoritative Git change set must
match the transaction exactly. Diff mode and the `changed_files` API enforce
the same admission even when the caller supplies a selected path list; omitting
a receipt or cascade path cannot make a partial transaction acceptable.

The selected base must be the original protected pre-migration commit. A base
that already contains the receipt cannot admit it as a new transaction. Trees
containing sequential receipts from different bases cannot satisfy this exact
transaction gate with one base.

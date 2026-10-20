# Registered reviewed repairs

`axiom-encode apply-reviewed-repair` validates and applies an exact reviewed patch
without model generation. The production registry is empty. This capability alone
admits no package and does not make any candidate ready for publication.

A package must be registered by a separately reviewed encoder change. Registration
binds an immutable official encoder commit and the exact Git blob, SHA-256, and
size of each of three fixed members: `patch.json`, `rules.patch.yaml`, and
`tests.patch.yaml` under `reviewed-repair-patches/<patch_id>/`. No caller-supplied
path or digest can authorize a package. The manifest binds the exact RuleSpec base,
predecessor path/blob/bytes, authenticated failed-run seed, corpus release and source
body, engine commit, complete v6 transaction, declared object changes, and raw and
parsed postimages. The existing strict repair overlay performs reconstruction.

The protected main-only targeted workflow accepts the closed
`axiom-encode/reviewed-repair-transaction/v1` wrapper in `source_bundle_json`:

```json
{"schema":"axiom-encode/reviewed-repair-transaction/v1","patch_id":"registered-id","transaction":{"schema":"axiom-encode/atomic-source-transaction/v6","repair_mode":"full_artifact","transaction":{}}}
```

The inner transaction above is only a schema illustration: a real request must
contain the complete validated source contract and required test cases. The lane
requires one existing target, full-source validation, an authenticated distinct
prior repair run, and `open_pr=false`. Queues, dependent lanes, existing imports,
legacy replacement/retained-successor bundles, and source/canonical-refresh bundles
are excluded. The normal attempt budget is enforced even when a repair run or
budget override is supplied. This is not a report-only replay exemption.

The command retains the ordinary replacement ownership, source attestation,
compiler, companion execution, complete-source, exact required-case, collision,
waiver, transaction snapshot, manifest non-shrink, and post-install closure checks.
Any unregistered postimage or supplemental mutation refuses application. Signing
uses the existing protected signer and token boundaries. There is no model fallback.

The signed tool is `axiom-encode apply-reviewed-repair`, with a bounded versioned
reconstruction receipt, not model/backend/prompt provenance. The receipt embeds the
seed and three package members so an updated consumer can reconstruct the exact
postimage against its shipped registry without fetching arbitrary content. It also
checks the registered trusted-base lineage and ordinary signed manifest execution
and source bindings. A consumer without this tool/registry support cannot admit it.

A historical v1 predecessor may be bound as exact tracked lineage where the ordinary
replacement route permits it. This neither authenticates its historical HMAC nor
removes or reclassifies it. Full-repository admission can still refuse that old
manifest. Successful changed-file validation does not establish all-files or
publication readiness.

Tests cover fixture-only registrations, strict parsing, immutable reconstruction,
request exclusions, budget enforcement, ordinary signed changed-file verification,
controlled overlay flow, and transaction refusal/rollback. Controlled pipeline and
signer fixtures are not a protected end-to-end workflow run. Real policy diagnostics
are separate evidence and do not authorize a registry entry or dispatch.

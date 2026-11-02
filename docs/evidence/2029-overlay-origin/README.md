# PR2029 overlay execution evidence

These records share engine `89571cc2a938707fd60a5489b951135345697296` and policy base `94789006ce372a1967c258ef47c12b15a1fe511e`, but have three distinct encoder execution contexts. The ordinary-checkout reference ran unchanged producer `bcd7ad40af654fd0d8ae8db2ab57d8894bfc0099` (v2270). Eval ran the frozen, uncommitted semantic files that were subsequently committed in `7d4487c4bf14ab27afce29dff3b882ab4378f7e5`; it was not a clean-commit provenance run. Only the final apply run executed the clean `7d4487c4bf14ab27afce29dff3b882ab4378f7e5` commit (v2271) and emitted the portable provenance shown here. They are historical local deterministic diagnostics, not native re-encoding, signing, publication clearance, or results from a later integration commit. Caller unit tests use stubs and prove forwarding only; they do not substantiate these runtime results.

`manifest.json` binds every included record and each decompressed capture. Gzip files use deterministic compression and contain the original bytes. The eval/apply captures match the independently executed ordinary-checkout reference. The compiled artifact hash is recorded, not the binary artifact itself. The engine receipt binds the existing macOS binary; it does not claim equivalence to a hosted Linux binary.

`apply-result.json` records successful validation and portable provenance. `clean-apply-review.json` records 21 cases/143 typed CAD assertions and preservation of 532 baseline files. Before/after inventories show the generated diagnostic candidate's three ordinary quote normalizations; those are parsed-equivalent, not byte-identical. Source findings are empty. Harness snapshots show the observation wrappers delegating actual compilation/runtime/source checks; absolute paths identify historical local inputs and are not portable execution instructions.

The logs and reports are author-produced historical evidence, not independently signed attestations. The earlier agent review (posted from Max's account) reproduced the unit-test checks, not these local real-engine runs. Earlier dirty-checkout and local-origin setup failures remain described in `comparison.json`; successful apply used the separately fetched official engine checkout. No guard was relaxed.

To inspect compressed records, use `gzip -dc <file.json.gz>`. Verify file digests against `manifest.json` before interpreting the results. Original 532 policy-file bytes and the compiled binary are not redistributed; inventory preservation is demonstrated by their hashes, not a fresh independent source audit.

The compressed `step-3.yaml.gz` and `step-3.test.yaml.gz` are the exact diagnostic apply candidate and its companion, included to inspect expected assertions and proof context. They are unsigned diagnostic evidence, not publishable policies. Apply normalizes three quoted numeric scalars; the companion remains byte-identical to the authenticated failed run.

## Test-command provenance

The original commands for `shared-tests.log`, `caller-tests.log`, and `version-tests.log` were not captured and could not be recovered. Their 190/228/27 counts are unverified historical author reports, not evidence of a particular test selection. Each file now labels this limitation and the manifest retains the original progress-output hash. Current review-follow-up regression evidence records its exact invocation separately; it does not retroactively authenticate these historical counts.

The optional optimization of per-case Git freshness checks is deliberately deferred. This change preserves the existing fail-closed checks rather than changing when provenance is checked during a test/documentation follow-up. Transient Git failures and their diagnostics remain a limitation; no performance improvement is claimed.

`current-review-regressions.log` records the exact command for the follow-up run: 303 tests passed in 13.43 seconds. It includes the updated replacement-scope tests, same-case checks, and existing caller regressions. The manifest binds the tested file hash. Four separate copied-source gate-deletion controls failed at the new origin assertions (both eval/apply callers and canonical/legacy scopes); production files remained unchanged.

## Historical review limitations

The packaging-review digest in `committed-review.json` and the clearance summary in `replacement-exclusion-final-review.json` do not include the underlying review narratives or finding dispositions. Their claims of independent clearance, including resolution of the two initial implementation findings, are unverified historical author reports. File hashes and test results establish their narrower stated checks; they do not establish that a review occurred. The separately posted GitHub agent reviews describe their own inspections and executed checks.

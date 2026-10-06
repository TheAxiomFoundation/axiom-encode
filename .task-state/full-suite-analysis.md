# Full-suite failure analysis

Run: parallel full-suite pytest attempt; see the parent validation report for the exact command and environment.
Log: `.task-state/pytest-full-parallel.log`.

The run was interrupted after more than 10 minutes without progress on the final two tests and exited 130. Pytest printed failure details, then interruption during teardown prevented its final statistics line. Parent-observed progress counts: **15,377 passed reports, 83 skipped, 247 failed, 51 setup errors, 2 unreported of 15,760 collected**. These are partial progress counts, not a completed suite result.

All 247 printed failure blocks are classified below. These classifications describe observed messages; they do not establish that every failure is environmental or predates this change.

| Observed failure class | Count |
| --- | ---: |
| Runtime-bound assertion | 11 |
| File mode mismatch: required 0644, actual 0600 | 202 |
| Immutable HEAD versus runtime version mismatch | 1 |
| Remaining assertions/exceptions | 8 |
| Isolated Python subprocess timeout | 4 |
| Canonical RuleSpec fixture root rejection | 6 |
| Trusted Git executable ownership check | 10 |
| Filesystem permission denial at /var/tmp | 1 |
| required_mode API mismatch | 4 |

All **51 setup errors** occur in `tests/test_signing_supervisor.py` and report a nonzero `go build` exit. The fixture captures stderr and does not expose it in this traceback, so the underlying Go-build cause is unconfirmed. No socket permission-denial message appears in the log.

No `FAILED` or `ERROR` summary entry names `tests/test_companion_relations.py` or `tests/test_companion_relations_engine.py`.

Direct evidence:

- File modes: `changed manifest must have mode 0644, found 0600`; also applies to legacy replacement inputs, import manifests, and canonical refresh files.
- Ownership: `trusted git path is not root-owned: /opt/homebrew/Cellar/git/2.55.0/bin/git`.
- Provenance: `version metadata is inconsistent (pyproject=0.2.1201, package=0.2.1201, lock=0.2.1201, runtime=0.2.2053)`; independently reproduced in `.task-state/provenance-test.log`.
- Filesystem restriction: `PermissionError: [Errno 1] Operation not permitted: /var/tmp/tmpzieu_oqf` in `test_rulespec_target_resolution_accepts_macos_system_path_alias`.
- Timing examples: `9.695673250011168 < 3.0` fails in Hebrew compound scanning; `6.7119096660171635 < 1.5` fails in CFR outline recognition. Resource contention is plausible but unproven without isolated reruns.
- Canonical-root checks reject temporary `rulespec-us` fixture roots. Root names in the message alone do not establish why validation rejected them.

## required_mode API mismatch (4 tests)

Each subprocess traceback ends with `TypeError: read_bounded_regular_file() got an unexpected keyword argument 'required_mode'`. This is not established as a sandbox failure.

- `tests/test_signing_supervisor.py::test_targeted_signed_reencode_orders_target_and_dependents[1--target-existing]`
- `tests/test_signing_supervisor.py::test_targeted_signed_reencode_orders_target_and_dependents[1-proof-import-subset-target-existing]`
- `tests/test_signing_supervisor.py::test_targeted_signed_reencode_orders_target_and_dependents[2--target-existing]`
- `tests/test_signing_supervisor.py::test_targeted_signed_reencode_reuses_verified_existing_import`

## Remaining assertions/exceptions (8 tests)

These require separate investigation. They occur in legacy replacement, signed backfill, policy runtime, and apply transaction code; none of their displayed traces point into the new companion request builder.

- `tests/test_cli.py::test_legacy_pending_detection_requires_signed_untampered_evidence`
  - AssertionError: assert ['invalid:.ax...aaaaaaa.json'] == ['invalid:.ax...aaaaaaa.json']
  - At index 0 diff: 'invalid:.axiom/legacy-replacements/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json' != 'invalid:.axiom/encoding-manifests/us/statutes/47/32.json'
  - Right contains one more item: 'invalid:.axiom/legacy-replacements/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json'
- `tests/test_prepare_signed_backfill.py::test_persisted_verifier_reconstructs_nonempty_v4_retained_identity`
  - assert False
  - +  where False = any(<generator object test_persisted_verifier_reconstructs_nonempty_v4_retained_identity.<locals>.<genexpr> at 0x1135fb100>)
- `tests/test_cli.py::TestResolverOwnedManifestWriter::test_legacy_verifier_rejects_protected_yaml_metadata_rewrite`
  - AssertionError: .axiom/encoding-manifests/us/statutes/47/32.json.replacement_manifest does not match the exact model v5 schema (missing axiom_encode_git, axiom_encode_version, citation, context_manifest_file, context_manifest_sha256, generated_at, generated_output_file, generated_output_root, generated_output_sha256, generation_prompt_sha256, model, run_id, runner, signature, source_attestation, trace_file, trace_sha256, validation_execution, validation_waiver_set_sha256)
  - .axiom/encoding-manifests/us/statutes/47/32.json.replacement_manifest generated_at is not an RFC3339 timestamp
  - .axiom/encoding-manifests/us/statutes/47/32.json.replacement_manifest axiom_encode_version is invalid
  - .axiom/encoding-manifests/us/statutes/47/32.json.replacement_manifest axiom_encode_git is malformed
  - .axiom/encoding-manifests/us/statutes/47/32.json.replacement_manifest is missing an encoder apply manifest signature
- `tests/test_cli.py::TestCmdEncode::test_encode_apply_resolves_authenticated_legacy_pending_dependent[False]`
  - AssertionError: assert ['invalid:.ax...aaaaaaa.json'] == ['us/policies...pendent.yaml']
  - At index 0 diff: 'invalid:.axiom/legacy-replacements/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json' != 'us/policies/income_tax/dependent.yaml'
- `tests/test_cli.py::TestCmdEncode::test_encode_apply_resolves_authenticated_legacy_pending_dependent[True]`
  - AssertionError: assert ['invalid:.ax...aaaaaaa.json'] == ['us/policies...nt.test.yaml']
  - At index 0 diff: 'invalid:.axiom/legacy-replacements/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.json' != 'us/policies/income_tax/dependent.test.yaml'
- `tests/test_policyengine_runtime.py::test_python_executable_must_not_be_setid`
  - Failed: DID NOT RAISE <class 'axiom_encode.harness.policyengine_runtime.PolicyEngineRuntimeError'>
- `tests/test_prepare_signed_backfill.py::test_reconcile_retired_manifest_inventory_resists_parent_symlink_race`
  - FileNotFoundError: [Errno 2] No such file or directory: '/private/var/folders/9l/_wztzgbx7mgc7l1r0416cy7m0000gn/T/pytest-of-maxghenis/pytest-27195/popen-gw7/test_reconcile_retired_manifes13/rulespec-us/tests-pinned/test_encoding_manifests.py'
- `tests/test_cli.py::TestCmdEncode::test_apply_transaction_recovery_enforces_cumulative_byte_limit`
  - RuntimeError: Cannot safely recover externally changed apply target: /private/var/folders/9l/_wztzgbx7mgc7l1r0416cy7m0000gn/T/pytest-of-maxghenis/pytest-27195/popen-gw2/test_apply_transaction_recover1/rulespec-us/us/statutes/26/1.yaml
  - AssertionError: Regex pattern did not match.
  - Expected regex: 'cumulative byte limit'
  - Actual message: 'Cannot safely recover externally changed apply target: /private/var/folders/9l/_wztzgbx7mgc7l1r0416cy7m0000gn/T/pytest-of-maxghenis/pytest-27195/popen-gw2/test_apply_transaction_recover1/rulespec-us/us/statutes/26/1.yaml'

The two unfinished test identifiers cannot be recovered from this quiet progress log; no final active-node names were emitted.

No broad test rerun or source changes were made during this analysis.

# Deterministic producer extension

[Max's 22 September decision](https://github.com/TheAxiomFoundation/axiom-encode/pull/1662#issuecomment-5784793928)
adds a second runtime kind alongside personal Codex. This document specifies
the extension implemented in #1662; it does not rewrite the approved v33
design or pretend that model metadata applies to deterministic generators.

The NZ pilot remains personal Codex with truthful sampling metadata (#1664).
The tariff-table generator remains deterministic. Implementation PRs may
merge before deployment; the deployment, custody and pilot gates still apply
to production activation.

## Protected enrollment

Existing `axiom/notary-producer-enrollments/v1` files and their Codex entries
retain their exact schema and runtime identity. Version 2 uses the same top
level `schema`, `lane`, and `runtimes`, with a required `runtime_kind` tag on
every entry. It accepts exactly `codex` or `deterministic`.

Both kinds retain the producer/actor fingerprints, numeric `github_user_ids`,
the exact encoder identity, and `custody_evidence_sha256`. A v2 Codex entry
retains `codex_cli`. A deterministic entry instead requires:

| Field | Binding |
| --- | --- |
| `generator` | `name`, `version`, Python `entrypoint`, and sorted unique `files` rows containing `path` and `sha256`; entrypoint must occur in the measured bundle |
| `runtime` | Relative `python_path` and `tree_sha256` covering the materialized Python runtime tree, including interpreter, stdlib, installed packages and encoder |
| `inputs` | Sorted unique `path`/`sha256` rows for all staged input files |
| `parameters` | Exact flat string-to-string map; the reviewed adapter validates its semantic vocabulary |
| `outputs` | Sorted unique repository-relative RuleSpec/companion YAML paths |

Paths are relative, traversal-free and bounded. Inputs and parameters are
fixed in the protected-base enrollment, not supplied by a PR or generate
request. Changing a recipe requires a reviewed protected trust-file update.
The runtime identity is the SHA-256 of the canonical enrollment entry.
Changing from v1 to tagged v2 changes that identity and requires new emissions;
old records are not relabeled. Existing immutable lineage remains retained.

Each producer key has exactly one enrollment and one runtime kind. Actor
keys remain separate and unique. Current repository write permission, numeric
operator eligibility, same-repository PR and current encoder pin are checked
by the existing admission, publication and finalization paths.

## Host and adapter interface

Use `axiom/supervised-deterministic-producer-host/v1` with the common host fields
from `producer_host.parse_config`: lane/content branch, epoch/notary root,
producer/actor key files, socket and state directory, operator UID bindings,
encoder identity, dependency inventory, Python path, worker UID/GID and timeout.
Replace the Codex/model/corpus/reference fields with absolute custodian-owned
`generator_root`, `input_root` and `runtime_root` paths. No personal Codex
credential is accepted or needed by this host kind.

The runtime tree must be root-owned, non-writable by the worker, and contain
regular single-link files and directories only. Materialize interpreter and
library symlinks when building the deployment image. `measure_runtime` hashes
the sorted `[relative_path, permission_mode, sha256]` file rows using JCS.
The selected Python executable must be the enrollment's `python_path` under
that tree. A controller-owned bootstrap checks the interpreter's prefix,
base prefix, executable and every import path before importing the worker.
A copied virtual environment that still imports an external standard library
is refused. Provision a standalone runtime with the encoder's Git-free runtime
attestation; the normal running-identity check still applies. OS libraries and
host-image configuration remain part of the
custody audit; a producer signature is not hardware remote attestation.

The controller performs these steps for `generate`:

1. Authenticate the Unix/SSH operator and load enrollment from the finalized,
   activated base. Recheck the exact encoder and runtime kind.
2. Measure the installed runtime and capture each generator/input file after
   checking ownership, path safety and the enrolled digest. Copy only the
   measured generator and input files into immutable job directories.
3. Start the dedicated non-login worker under the existing Linux/systemd
   isolation, additionally disabling network access. No signer key, Codex
   credential, caller command or arbitrary environment enters this job.
4. Invoke the enrolled Python adapter in its own fresh working directory and
   worker cgroup. The adapter receives one JSON descriptor with schema
   `axiom/deterministic-adapter-request/v1`, `input_root`, `output_root`,
   `parameters` and `outputs`. The entrypoint and those values derive from
   protected enrollment. Exit failure or timeout refuses the emission.
5. Stop that worker cgroup, reclaim its directory for the controller, check
   that the UID is idle, and remeasure the runtime, generator and inputs.
   Capture the exact output bytes in controller memory through no-follow,
   regular-file, single-link checks. Repeat in a separate directory and cgroup.
   Reject extra/missing files, changed measurements or unequal bytes between
   the two executions. The second worker cannot change the first observation.
6. Sign a deterministic generation record and return the ordinary public
   producer export. Reusing the run ID retrieves the result without rerunning.

The adapter is reviewed executable code. Its contract limits semantic inputs
to the staged files and parameters; review must reject time/host-dependent
behavior or undeclared ambient inputs. Two matching executions check observed
repeatability, not a mathematical proof of determinism.

The existing client accepts:

```sh
python -m axiom_encode.notary.producer_client generate \
  --socket /run/axiom-producer/service.sock \
  --run-id <new-32-hex-run-id> \
  --export /private/work/generation.json
```

The deterministic response contains no refreshed credential. SSH transport,
status/retry ownership and verified export application use the existing client.

## Evidence and signatures

`axiom/lineage-deterministic-generation/v1` has exactly:
`schema`, `lane`, `epoch_sha256`, `runtime_identity`, `generator`, `runtime`,
`inputs`, `parameters`, `emitted_at`, and `transitions`. The first four bind
context and enrollment; generator/runtime/input/parameter values must equal
the protected recipe. Transitions contain the controller-observed before/after
hashes and modes. This runtime emits only the enrolled output set and cannot
claim deletions. Unchanged outputs produce no transitions.

There are no model, CLI, prompt, temperature, seed or draw-set placeholders.
Bodies retain the content-addressed `.axiom/lineage/<sha>.json` layout and
`.producer.sig` sidecar. The signature domain is
`axiom/lineage-deterministic-generation/v1`, using the existing registered
producer key role and existing signing frame. The verifier chooses this
domain from the parsed body schema, never from caller/sidecar metadata.
Codex and deterministic signatures cannot be substituted for one another.

The verifier retains ordinary exact-byte coverage, protected-path rules,
correction review, validation gates and receipt approval. CI/notary performs
no generator or model calls. Older verifiers refuse this schema; deploy the
reviewed producer and verifier versions together during activation.

## B1.6 adapter and temporary correction authorization

`deploy/notary/adapters/b16_note_tables.py` supplies a concrete adapter for
Pavel's review. It calls the measured `generate_incidence_tables.py` function
with exactly `brazil-50` and `reciprocal-52`, using the captured notes artifact
staged as `notes.jsonl`. Both Python files belong in the measured bundle. The
existing generator independently checks the notes artifact digest. Its code
and source artifact must be reviewed/pinned during enrollment; no production
fingerprint or active enrollment is asserted by this package.

Per Max's decision, the temporary correction authorization for
[rulespec-us#1383](https://github.com/TheAxiomFoundation/rulespec-us/pull/1383)
covers only these four paths under
`us/policies/usitc/us-tariff-incidence/generated/`:

- `note50-brazil-exemptions.yaml`
- `note50-brazil-exemptions.test.yaml`
- `note52-reciprocal-exemptions.yaml`
- `note52-reciprocal-exemptions.test.yaml`

That authorization is conditional on the notary being active on rulespec-us,
requires actor signatures and hardware correction review, and ends when the
deterministic producer exists. It does not cover #1383's generator code,
index, receipt or toolchain changes, waive their independent requirements,
turn off a guard, or automatically sign/merge the PR. The current US path
is not changed here. Pavel should use deterministic generation once that
runtime is available rather than extending the temporary authorization.

## Coordination and acceptance

Max retains the administrative key. Pavel owns enrollment preparation,
generator-adapter review, deployment and operations and takes one review or
approval custodian role. These are the owners Max named, not a claim that
their existing CI/signing work is unfinished or that new keys are needed.
Reuse their verified deployment work and identify only actual remaining
steps. Do not announce ceremony readiness until Pavel confirms preparation;
then give Max the exact remaining administrative action and reviewed inputs.

The implementation tests exercise unchanged Codex authorization, both producer
types, mixed lineage with reviewed corrections, exact export application and
notary signing without generation. Refusal tests cover unregistered operators,
forks, revoked permission, changed measurements/parameters/outputs, wrong
signature domains, missing approval and nondeterministic output. Real
deployment and NZ pilot evidence are still required for activation.

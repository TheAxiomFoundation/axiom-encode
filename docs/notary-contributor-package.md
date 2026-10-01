# Enrolled contributor implementation and activation contract

This package implements the enrolled producer, verification, signing and publication path for
[Max's enrolled-producer decision](https://github.com/TheAxiomFoundation/axiom-encode/pull/1629#issuecomment-5732652158),
using the approved [v33 design](notary-admission-design.md). It is **not an
enabled US admission path**. A merge cannot enroll a host, install private
keys, deploy services, or establish GitHub App permissions.

[Max's 22 September follow-up](https://github.com/TheAxiomFoundation/axiom-encode/pull/1662#issuecomment-5784793928)
permits implementation PRs to merge before activation and adds a separately
enrolled deterministic runtime. The [extension contract](notary-deterministic-producer.md)
specifies its measured recipe, signature domain and B1.6 adapter. The NZ pilot
remains personal Codex; custody, deployment and pilot gates remain activation
requirements. Pavel owns enrollment preparation, generator-adapter review,
deployment and operations; Max retains the administrative key.

The contributor outcome is unchanged: an enrolled team contributor generates
with personal Codex; CI verifies the exact submitted bytes without model
calls. Current repository write access is necessary. An outsider, fork, stale
PR head, unsupported encoder identity, unknown runtime, missing evidence, or
changed output must refuse. Existing unsigned drafts are not retrospectively
authenticated by attaching a new signature.

## Review map

| Boundary | Implementation | Evidence |
| --- | --- | --- |
| Raw Git and canonical JSON | `notary/manifest.py`, `canonical.py` | Real Git structural fixtures, JCS and signature-envelope tests; reused from Max's #1511 |
| Registered lineage | `registry.py`, `signatures.py`, `lineage.py` | All seven roles, frozen legacy roots, append-only history, closed generation/correction bodies |
| Exact changes | `coverage.py` | Unique atomic record assignment, predecessor constraints, compatible topological replay, protected mode wall |
| Trusted recomputation | `protocol.py`, `verification.py` | Pass/refusal schemas, truthful established prefixes, trust-surface refusal order, strict toolchain and gate declarations |
| Contributor authorization | `identity.py`, `producers.py` | GitHub-signed OIDC plus live job metadata; current PR author, numeric operator identity, write permission, same-repository head, exact encoder/runtime binding |
| Administrative state | `administration.py`, `consumer.py`, `chain.py` | Frozen v5 inventory, genesis partition, five-path activation, predecessor-authorized rotation, complete bundles, terminal markers |
| Receipt service | `provenance.py`, `signer.py` | Run-scoped artifact provenance; independent reconciliation; digest-bound approver signatures; typed receipt/genesis/transition signing |
| Production reads | `remote.py`, `github_inputs.py`, `protection.py` | Fresh bare repositories, immutable GitHub artifacts, environment and ruleset audits, live bootstrap lock and sole-actor check |
| Publication state machine | `publication.py` | Complete pending bundles, idempotent retry, exact merged-tree binding, permanent voids, sibling supersession, notary rotation |
| Supervised personal-Codex runtime | `producer_host.py`, `producer_runtime.py`, `producer_worker.py`, `producer_client.py` | Root controller, isolated non-login worker, measured binaries, exact-byte export, private refreshed auth, durable idempotent status, authenticated apply |
| Supervised deterministic runtime | `deterministic_contract.py`, `deterministic_runtime.py`, `producers.py` | Tagged enrollment, measured generator/runtime/input files, fixed parameters, offline repeat execution, exact observed output and a separate signature domain |
| Hardware approvals | `approval.py`, `approval_inbox.py` | Direct USB device identity, non-exportable role-separated keys, explicit digest confirmation, bounded public sidecar deposit |
| External services | `service.py`, `deployment.py`, `readplane.py` | Typed OIDC operations, separate custody, read-only proxy, no caller-supplied signing bytes or keys |
| Live publication | `apps.py`, `leases.py`, `github_publication.py`, `publisher.py`, `finalizer.py` | Two App scope audits, durable revocable leases, first-push/fast-forward CAS, App checks, merge and revocation handling |
| CI and shared guard | `runner.py`, `workflows.py`, `assets.py`, `merge_guard.py` | Four fresh jobs, immutable inputs, real process gate outcomes, base-controlled migration, request-only finalizer |

The signer never executes candidate code. It holds the notary key and only
read-side dependencies. Its public operations accept a candidate digest and
approval sidecar, not caller-selected roots, repositories, manifests, or a
generic signing scope. Production callers must install the concrete
`GitHubSignerInputs` adapter; test adapters do not establish remote authority.

`PublicationPlan` is enforced by `GitPublication` atomic ref updates. The
publisher runner receives only the two operation-specific short-lived App
tokens; broker-side finalization never returns a write credential. The
source-App-bound check remains independent of the generated-file guard.
The [installation and contributor runbook](../deploy/notary/README.md) maps
the implemented entry points to the custody and activation evidence.

## Enrollment binding proposed for review

V33's normative public-key registry remains `.axiom/notary/keys.json` and
retains its exact closed schema. Generation and correction bodies also keep
their v33 schemas. This implementation proposes supplementary enrollment
metadata at `.axiom/notary/producers.json`, protected as an administrative
trust surface. It is not an alternate key registry or signature authority.
Max should explicitly review this enrollment representation before rollout.

The closed `axiom/notary-producer-enrollments/v1` object contains `lane` and
`runtimes`, sorted and unique by producer SPKI. Each runtime entry binds:

- producer and actor SPKI fingerprints already present in the normative
  registry;
- sorted GitHub numeric operator IDs;
- the official encoder repository, exact Git commit, version, and measured
  package-tree SHA-256;
- measured Codex CLI version and executable SHA-256;
- the digest of the custody evidence Max reviewed.

The proposed policy requires the runtime entries to cover every registered
producer key exactly once, and prohibits sharing one actor fingerprint across
runtime entries. These are enrollment policy choices for Max's review, not
additional fields in v33's normative registry.

`runtime_identity` is `axiom-runtime:sha256:` followed by the JCS digest of
that enrollment entry. The notary authenticates the producer signature and
compares this identity with the approved enrollment record. The runtime host
must actually measure and enforce the enrolled configuration; a declared
package digest is not remote attestation. The existing verification-only
supervisor does not supply the protected producer emission operation.

Intake intersects the operators permitted for all consumed generation or
correction signers, authenticates the current PR author through GitHub, and
rechecks that author's current write/admin permission. It compares the
enrolled encoder commit and version with the **base's** workflow toolchain.
Candidate-provided enrollment files cannot authorize their own changes.
Permission loss refuses fresh intake; lifecycle handling must also invalidate
pending checks and carry out any necessary administrative key revocation.

## Concrete implementation choices for Max's review

The implementation now supplies the runtime, service adapters, hardware tooling,
CI renderer and guard integration previously missing from this package. These
choices need explicit review against v33; they are not represented as decisions
Max has already made:

| Proposal | Why it is present |
| --- | --- |
| Linux/systemd 250+ producer with root controller, non-login worker and peer-UID/SSH operator authentication | Keeps host-held producer/actor keys outside the model process and observes output before signing |
| Supplemental protected `producers.json` enrollment | Binds registered keys to current team eligibility and measured encoder/runtime identity |
| Direct USB YubiHSM2 approval adapter | Provides a concrete non-exportable hardware-key interface without selecting production devices or provisioning keys |
| Separate authenticated read-plane service | Supplies read-only Actions/org/protection data without broadening either publication App's root permissions |
| Null temperature amendment in a companion PR | Represents sampling metadata the Codex backend does not expose; strict v33 currently requires a numeric value |

Production configuration is custodian-owned. The supplied systemd units are
installation templates, not evidence of installed services. The signer rebuilds
its job policy from the current protected base for every request. Public
runner configuration uses a zero epoch template and derives the real epoch
from frozen genesis inputs or the authenticated consumer, preserving v33's
five-path activation.

The base commits exact public asset pins and commands for the approved gate
profile. The runner executes those commands and records real exit outcomes.
No-op gates, classifier scripts mislabeled as oracle execution, fabricated
production IDs and fixture keys are not valid deployment inputs.

## Review and rollout dependencies

1. Review this implementation and the supplemental enrollment/custody choices.
   Review the separate sampling amendment; no production run should invent a
   temperature to fit the current schema.
2. Review the dedicated protected-workflow migration and Linux isolation CI
   changes. Keep toolchain/workflow repins separate from encoding feature PRs
   under organization issue #39.
3. Supply actual custodian-controlled hosts, hardware, endpoints, measured
   identities, App installations, rulesets and environment reviewer IDs using
   the runbook. Reconcile this with Pavel's existing preparation and audit the
   v33 §9 prerequisites before activation. Per Max's 22 September decision,
   implementation PRs can merge while that preparation continues.
4. Commit the actual NZ deployment/profile configuration and rendered immutable
   workflows in its dedicated gated PR. Execute the locked genesis ceremony,
   exact five-path activation and real personal-Codex NZ pilot with proof of
   unchanged admitted bytes and no CI model calls.
5. After the NZ audit, review the US rollout using the resulting real pins and
   service identities. Keep the existing US guard until the App-bound notary
   admission path is installed. Verify both acceptance and refusal cases.

The complete engineering handoff includes service and host code, tests,
deployment instructions, workflow generation and migration, and explicit
acceptance evidence requirements. It does not claim that unperformed hardware
custody, GitHub settings, NZ oracle execution or a US signed encoding already
succeeded. Final lane activation PRs depend on those actual public values;
committing invented values would not make the package deployable.

Two chain rulesets are required: an App-only creation/update rule with the
App's bypass, and a separate no-bypass integrity rule enforcing no deletion,
no force update and linear history. Combining them would give the App a bypass
of the integrity rules too. `update_allows_fetch_and_merge` must be explicitly
false. A new genesis refuses if the dedicated chain ref already exists;
atomic branch creation must still close the concurrent-first-push race.

## What the tests establish

Automated tests use ephemeral fixture keys and controlled GitHub responses.
They exercise real signatures and raw Git parsing, report reconciliation,
enrollment refusal, host ownership and correction behavior, hardware-interface
refusals, service routes, token/lease recovery, publication compare-and-swap,
administrative transitions, receipt bundles and chain state changes. An opt-in
root Linux test exercises real systemd namespace isolation on an ephemeral
runner; it is skipped on macOS and is not production-host attestation. They do not establish deployed host isolation, production
GitHub App permissions, live hardware-key operation, a completed NZ pilot or
a green US contributor PR. Those require the integration and rollout above.

The v33 residual remains explicit: gate outcomes are authenticated declarations
from the proposing run, checked against the profile. Signatures do not prove
model authorship, statistical independence, trusted timestamps, or independent
re-execution of candidate gates.

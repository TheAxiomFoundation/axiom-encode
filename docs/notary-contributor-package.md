# Enrolled contributor implementation and activation contract

This package implements the verification and service core for
[Max's enrolled-producer decision](https://github.com/TheAxiomFoundation/axiom-encode/pull/1629#issuecomment-5732652158),
using the approved [v33 design](notary-admission-design.md). It is **not an
enabled US admission path**. A merge cannot enroll a host, install private
keys, deploy services, or establish GitHub App permissions.

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

The signer never executes candidate code. It holds the notary key and only
read-side dependencies. Its public operations accept a candidate digest and
approval sidecar, not caller-selected roots, repositories, manifests, or a
generic signing scope. Production callers must install the concrete
`GitHubSignerInputs` adapter; test adapters do not establish remote authority.

`PublicationPlan` is deliberately a plan, not proof of a remote write. Its
`expected_chain_commit` must be enforced by the separately deployed publisher
broker's atomic Git ref update. A runner must never receive a finalization
credential. No production write adapter or required-check integration is
installed by this package.

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

## Deployment information needed to finish integration

These are public configuration and custody decisions, not requests to put
private keys into this repository or a chat. The provided Slack decision
assigns custody roles but does not name a production deployment platform or
service interface. The implementation must be connected to those actual
boundaries before it can be called a complete contributor path.

| Required decision or input | Why it is needed |
| --- | --- |
| Supervised producer host, OS/process isolation boundary, custodian and operator authentication | Implement and test the real emission operation; prevent a client or model process from asking to sign arbitrary uploaded files |
| Protected producer/actor key interface on that host | Emit lineage while keeping both private keys on the host |
| Notary signer and separate publisher-broker deployment targets and service interfaces | Install typed operations without combining signing and repository-write authority |
| Two GitHub App IDs/installations and audited root permissions | Implement token vending, pending publication, App-bound required checks and broker-side finalization |
| Hardware receipt-approver and correction-review device interfaces; administrative ceremony record | Provide usable digest-bound approval tooling without exporting hardware keys |
| Dedicated notary repositories, ruleset IDs, environment reviewers, bootstrap actor/team and locked tip | Configure the concrete reader and verify v33's live protection requirements |
| Approved NZ pilot verifier profile and dependency inventory | Execute the actual deterministic gates and record their declared outcomes; an oracle label alone is not an executed oracle comparison |

Production config must be custodian-owned. `Deployment` and
`BootstrapCeremony` objects cannot be deserialized from a caller's request.
`GitHubSignerInputs.job_policy()` derives `workflow_sha` from the current
protected base commit. The deployment wrapper must call that factory and
construct `NotarySigner` for each request: the signer captures its supplied
policy at construction, so reusing yesterday's policy would incorrectly
reject later ordinary merges. The full finalized base manifest, not a
workflow's self-description, binds the code and policies used by that run.

## Remaining integration and rollout, in order

1. Connect the supervised encoder to a host-held producer emission operation.
   Export only exact generated files and v33 evidence; keep personal Codex
   credentials and refreshed auth out of the evidence archive and CI. Supply
   actor tooling separately from hardware correction-review approval.
2. Install the external signer and separate publisher-token broker. The latter
   must verify the two distinct Apps' **root** installation scopes, vend only
   the operation-specific publisher tokens, and execute finalization itself.
   Wire `PublicationPlan` to atomic branch updates and the source-App-bound
   required check. Revalidate the current base and contributor at publication.
3. Add the four-job, non-reusable `verify → recompute → approve → publish`
   workflow on fresh runners, plus the request-only finalizer. Perform cheap
   evidence checks before the costly validation audit. Job 1 has contents-read
   only, no environment or secrets; no job invokes a model for admission.
4. Complete and audit every v33 §9 prerequisite before admission-capable merge
   or activation. Bootstrap uses a locked lane, its sole audited administrative
   bypass, first-push CAS, and exact five-path activation; remove that bypass
   only after activation finalization.
5. Run the NZ pilot with real service boundaries. Then submit the dedicated US
   consumer/workflow/pin rollout PR required by the organization rules. Keep
   the existing US guard until the App-bound notary admission path is installed.
6. Enroll the contributor's runtime and run one small personal-Codex encoding.
   Record the PR, before/after digests, generation record, approval, receipt,
   pending check and finalization. Require unchanged submitted bytes and no
   CI generation. Repeat the refusal checks for an outsider, an unregistered
   producer, a wrong encoder, a changed file and a revoked contributor.

Two chain rulesets are required: an App-only creation/update rule with the
App's bypass, and a separate no-bypass integrity rule enforcing no deletion,
no force update and linear history. Combining them would give the App a bypass
of the integrity rules too. `update_allows_fetch_and_merge` must be explicitly
false. A new genesis refuses if the dedicated chain ref already exists;
atomic branch creation must still close the concurrent-first-push race.

## What the tests establish

Automated tests use ephemeral fixture keys and controlled GitHub responses.
They exercise real signatures and raw Git parsing, report reconciliation,
enrollment refusal, administrative transitions, receipt bundles and chain
state changes. They do not establish deployed host isolation, production
GitHub App permissions, live hardware-key operation, a completed NZ pilot or
a green US contributor PR. Those require the integration and rollout above.

The v33 residual remains explicit: gate outcomes are authenticated declarations
from the proposing run, checked against the profile. Signatures do not prove
model authorship, statistical independence, trusted timestamps, or independent
re-execution of candidate gates.

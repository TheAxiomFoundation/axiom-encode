# Custodian installation and contributor runbook

These are reviewable reference deployments, not a declaration that a host,
key or GitHub App is enrolled. Max's v33 §9 audit is a prerequisite to an
admission-capable merge or activation. Do not generate production identities
from the test fixtures. The Linux/systemd host, direct-USB YubiHSM2 adapter,
supplementary enrollment file and read-plane service are concrete proposals
for Max's review; his decision did not select those implementations.

## Components and private custody

Use separate dedicated hosts for the supervised producer, typed signer,
publisher broker and read plane. Install the reviewed encoder commit into a
clean canonical `TheAxiomFoundation/axiom-encode` checkout, with the locked
Python environment and root-owned non-writable ancestors. Resolve the Python
executable; do not grant a model process write access to it, its libraries,
encoder checkout, Codex executable, corpus or engine. Measure both Git identity
and package tree using `require_running_identity`; the producer also measures
the actual Codex executable and version. Never put private configuration or
personal auth into a repository, artifact or Actions secret.

The service entry point is:

```
/opt/axiom/notary/.venv/bin/python -I -m axiom_encode.notary.deployment \
  --config /etc/axiom-notary/config.json --socket /run/axiom-notary/service.sock
```

`axiom-notary.service` runs exactly one configured role. Create its unprivileged
service account and `/etc/axiom-notary` as custodian-owned directories. Private
key/token files must be regular, single-link, mode 0600; every ancestor must
be owned by root or the service account and not writable by others. The signer
holds only the notary key and a read credential. The publisher holds only the
two distinct App private keys. The read plane holds a read credential and the
chain App's **public** key, not a third App key. Its purpose is read-only Actions,
artifact, protection and organization inspection that the two narrowly scoped
publication Apps cannot perform. It authenticates broker requests using a
separate issuer/audience with the existing chain App key.

Terminate HTTPS on port 443 separately for signer, publisher and read plane.
Only the private Unix socket is exposed by the Python process. The reference
unit/socket are private by default: the deployment must grant its dedicated
proxy account execute access to the runtime directory and read/write access
to the socket using narrowly scoped ACLs after socket creation, without
changing key-file access or granting directory write permission. Reapply those
ACLs on restart. Disable proxy request/header/body logging, redirecting upstreams
and debug logging. Service access logs are disabled. Do not run the proxy as
the key-owning account. Fixed endpoints refuse redirects and environment proxies.

`deployment.build_app` validates the complete closed configuration. Shared
fields are `schema: axiom/notary-service-config/v1`, `role`, `deployment`,
`ceremony`, `encoder_identity` and `credentials`. `Deployment` in
`notary/github_inputs.py` defines every public field. Obtain real numeric repo,
owner, reviewer, ruleset, App and installation IDs from the deployment; do not
substitute handles or fixture numbers. Credentials differ by role:

| Role | Exact credential fields |
| --- | --- |
| signer | `github_read_token_file`, `notary_key_file`, `approval_directory` |
| publisher-broker | `chain_app`, `lane_app`, `read_plane_endpoint`, `state_database` |
| read-plane | `github_read_token_file`, `broker_app_public_key_file`, `broker_app_id`, `audience` |

Each publisher App value contains `scope` (the exact `AppScope` fields) and
`key_file`. Set the read-plane audience to the broker's exact configured read
endpoint. Store its SQLite lease DB in `/var/lib/axiom-notary`; preserve the DB
and lock files across restarts. The broker audits root permissions and all
installations before vending short-lived operation-specific tokens. Finalization
uses broker-owned credentials and never returns those credentials to a runner.

## Supervised producer

Install `axiom-producer.service` on a dedicated Linux host with systemd 250+.
Create `axiom-producer-operators` and a separate non-login worker account with
UID/GID at least 1000 and no supplementary groups. The worker account must be
idle and inaccessible by SSH. The controller needs root to create confined
systemd workers, inspect their lifecycle and lock output before signing.
Operators do not get root, worker identity, arbitrary sudo, or model-command
selection. Map authenticated local operator UIDs to their GitHub numeric IDs
in the custodian config. For remote use, configure SSH with a **fixed** forced
command, no forwarding, PTY, user environment, shell or agent forwarding:

```
/opt/axiom/producer/bin/python -I -m axiom_encode.notary.producer_client \
  relay --socket /run/axiom-producer/service.sock
```

The closed `axiom/supervised-producer-host/v1` configuration is validated by
`producer_host.parse_config`. It includes lane/branch/epoch/notary pin;
producer and actor key paths; measured encoder identity and dependency inventory;
Codex/Python/engine/corpus/dependency paths; model and sampling metadata;
operator UID/ID mappings; worker UID/GID; state/socket paths and socket group;
references and timeout. Set `socket_gid` to the numeric GID of `axiom-producer-operators`, and
`socket_path` to `/run/axiom-producer/service.sock` (a new socket filename). Create the state directory on a
local disk owned by root, mode 0711; records/results are private 0700. The
controller validates all three permissions. Protect backups the same way.

Personal Codex auth is submitted only to this host. The worker receives a
private per-run Codex home, canonical read-only inputs, its own writable job
area and no producer/actor signing keys. The host terminates the whole worker
cgroup, observes exact output bytes and records those bytes under the enrolled
producer key. The public export excludes auth. The credential response is
returned separately, mode 0600, to the same UID **and** numeric GitHub identity.
Never use historical local drafts as authenticated emissions.

The current v33 sampling schema requires a numeric temperature even though the
Codex adapter does not set one and the documented Codex configuration does not
expose one. Activation therefore also requires Max's decision on the companion
sampling-metadata amendment. Do not invent a temperature to satisfy the schema.

The host retains private jobs/results for idempotent status recovery, including
credential responses. Custodians must document retention, encrypted storage and
backup handling, then purge completed private jobs/results after the operator
acknowledges receipt. Do not purge an in-flight request or reuse its run ID;
a lost result must not silently trigger paid generation again.

## Hardware approval

The supplied adapter proposes direct USB YubiHSM2. Receipt approver,
correction-review and administrative-approver are distinct registered keys
with custody evidence. Provision device-generated, non-exportable Ed25519 keys
with SIGN_EDDSA only. The tool verifies serial, key origin, capabilities, SPKI,
lane and epoch, and requires a well-formed custody-evidence digest; it offers no software fallback or key import.
Custodians must verify the referenced custody evidence, genuine hardware and
separation; the tool does not retrieve or authenticate that evidence, and
fixtures cannot prove it.

On the hardware workstation, install the optional USB adapter into the locked
reviewed checkout first: `uv sync --locked --extra notary-hardware`. Use that
environment's Python for the following commands.

Download the immutable candidate from the recompute job and review its exact
contents and displayed digest. Then run on the enrolled hardware workstation:

```
python -I -m axiom_encode.notary.approval \
  --device /etc/axiom-approval/device.json --candidate candidate.json \
  --expected-sha256 REVIEWED_SHA256 --output approval.sig
```

The CLI requires explicit digest confirmation and device authentication. Deposit
only the **public** sidecar into the signer's private approval inbox using a
custodian-operated, fixed restricted channel:

```
python -I -m axiom_encode.notary.approval_inbox \
  --directory /var/lib/axiom-notary/approvals < approval.sig
```

Do not expose deposit as an unauthenticated HTTP operation or grant the hardware
operator a shell on the signer. The inbox is not authority: the typed signer
independently authenticates the GitHub job and validates the signature. The
approve job waits up to an hour for the exact candidate approval. A changed
candidate requires new approval. Correction-review sidecars belong with the
actor correction evidence; they are not receipt-approval inbox entries.

## Lane configuration and CI

Before the locked genesis ceremony, prepare these public base files in a
**dedicated gated** lane PR: `.axiom/notary/runner.json`, `producers.json`,
`assets.json`, `gate-commands.json` and the rendered workflows. The generator
consumes the exact closed runner schema, with the measured encoder identity,
fixed service endpoints, real deployment IDs and ceremony inputs:

```
python -I -m axiom_encode.notary.workflows \
  --config .axiom/notary/runner.json --output .
```

The public runner `deployment.epoch_sha256` is always 64 zeroes: the runner
derives the real epoch from frozen genesis inputs before activation and the
authenticated consumer afterwards. This avoids a circular genesis digest and
keeps activation to v33's exact five files. Service and producer configs use
the actual ceremony epoch. All workflows pin the final reviewed encoder commit
and full Action SHAs recorded in the dependency inventory. Render after that
commit exists; do not commit a floating branch pin or a fictional digest.

The admission workflow is four fresh, non-reusable jobs: verify (contents-read),
recompute (contents/actions-read), approve (OIDC only, signing environment),
publish (OIDC only, publishing environment). Only verify executes candidate
gate commands; no job generates with a model. The push finalizer only requests
broker-owned finalization. The generated-file guard selects the new verifier
only from the trusted base's valid activated consumer. No consumer means the
mandatory legacy guard; malformed consumer means failure, never fallback.
Its success is separate from the mandatory App-bound admission check.

`assets.json` pins canonical corpus/engine/optional cross-country repositories
by full Git OID, plus the immutable corpus-release download base URL. Records
are sorted by name (`corpus`, `engine`, `rulespec_XX` with lowercase country).
The toolchain's exact signed release name/hash selects the download, which is
verified using authenticated registry roots before source use. Asset checkout
failure or signature failure never produces a pass declaration.

`gate-commands.json` is a sorted exact match for the profile's required gate IDs;
each entry contains `gate_id`, an argv array and `timeout_seconds` (1–3600).
Placeholders include `{python}`, `{checkout}`, `{corpus_public_key}`,
`{corpus_root}`, `{engine_root}` and optional `{rulespec_nz_root}` equivalents.
Use actual deterministic compile/proof/companion-test and oracle comparison
commands for the selected NZ provision. An oracle reconciliation/classification
script is not a substitute for executing an oracle comparison. Failed commands,
timeouts and missing tools are failures. No empty profile, `true`, no-op or
invented oracle result is suitable for the production pilot.

## Activation evidence and sequence

1. Review the implementation choices above and the sampling amendment. Record
   operator eligibility, host custody, actual measurements, role/device keys,
   frozen legacy roots, model billing and the selected real NZ gate profile.
2. Deploy and test the separate services and hardware. Audit both Apps' exact
   root installation permissions; environment reviewers and disabled admin
   bypass; strict lane protection requiring the source-App-bound check; disabled
   rebase merges; two chain rulesets (App writer rule plus no-bypass integrity
   rule). Record numeric IDs and read-only API evidence.
3. Lock the NZ lane at the ceremony base with the sole audited bootstrap actor.
   Create genesis using first-push CAS and its separate genesis-finalization
   marker as the second chain commit. Then publish the pending activation
   transition and the exact five-file lane activation. Finalize exact merged
   bytes, then remove the temporary administrative bypass. Preserve all hashes.
4. Exercise personal-Codex generation, hardware approval, receipt publication,
   ordinary merge and broker finalization on NZ. Record the real PR/run/artifact
   links. Repeat refusal probes for outsider, fork, unknown runtime, wrong
   encoder, changed bytes and revoked operator. Test retries and token expiry.
5. Only after the NZ pilot audit, prepare the US dedicated consumer/workflow/pin
   rollout with actual finalized configuration. Keep the US legacy guard until
   its required App-bound check and new consumer are installed. Repeat the
   small contributor acceptance test before announcing general availability.

The fixture suite validates code behavior; it cannot fill any of these custody,
hardware, repository-settings or live-pilot evidence fields.

## Contributor commands after activation

Use a new 32-hex run ID, the enrolled host, a verified host-key file and a
personal SSH identity. Keep output outside Git until verified:

```
python -I -m axiom_encode.notary.producer_client encode \
  --ssh-host operator@enrolled-host --known-hosts /private/axiom-known-hosts \
  --identity-file /private/axiom-ssh-key --run-id RUN_ID_32_HEX \
  --citation 'APPROVED PILOT PROVISION' --draw-set-id DRAW_SET \
  --auth-file /private/codex/auth.json --export /private/output.json \
  --refreshed-auth /private/refreshed-auth.json
python -I -m axiom_encode.notary.producer_client apply \
  --export /private/output.json --checkout /work/rulespec-nz
```

The checkout must be clean at the recorded finalized base. Apply authenticates
the chain and signatures before writing. Push an org branch and open its PR.
Dispatch Notary admission with operation `receipt` and that PR number. Request
the designated hardware approver's review; the production environments still
require Max/Pavel approval under v33. On a timeout, use `status` with the same
run ID and output arguments; do not generate again under a new ID blindly.

For corrections, submit the explicit `edits`, `reason` and `predecessor_run_id`
through `correction --correction-request FILE`. The host records the actual
actor transition. Obtain a separate hardware correction-review signature over
the exact correction body and include that sidecar before admission. Write
permission alone does not authorize signing arbitrary local edits.

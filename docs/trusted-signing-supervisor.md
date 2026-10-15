# Trusted signing supervisor

Signed apply, retire, and evaluation operations cross a compiled trust boundary.
Private Ed25519 keys remain inside two distinct external signers. Neither the Go
supervisor/broker nor Python receives raw private-key material.

## Deployment boundary

Build the production binary separately from the Python wheel:

```bash
CGO_ENABLED=0 go build -trimpath -buildvcs=false -ldflags='-buildid=' \
  -o build/axiom-encode-signing-supervisor \
  ./cmd/axiom-encode-signing-supervisor
test "$(build/axiom-encode-signing-supervisor --build-kind)" = production
```

The `signing_supervisor_test_fixture` build tag produces a binary that reports
`test-fixture-nonpublishable`. It exists only for tests and must never be
published or installed.

Install the production binary, one self-contained Python runtime, the complete
`axiom_encode` package, its explicit import roots, and the trust-root config as
root-owned paths that the service account cannot change through modes or ACLs.
The runtime must contain only regular files/directories and no symlinks, special
files, `.pth`, `.egg-link`, `sitecustomize.py`, `usercustomize.py`,
`pyvenv.cfg`, or `__editable__*` injection. Set-ID/sticky bits,
group/other-write access, Linux `security.capability` xattrs, and caller-writable
paths are rejected. The caller must be unprivileged with matching identities;
Linux permitted, effective, and ambient capabilities must all be zero.

Python is executed directly as:

```text
/opt/axiom/python/bin/python3.13 -I -S \
  /opt/axiom/python/lib/python3.13/site-packages/axiom_encode/_trusted_signing_bootstrap.py ...
```

The protected bootstrap runs before attaching the broker descriptor. It checks
the exact runtime/package/import roots, isolated/no-site flags, every initial
`sys.path` root, compiled and virtual-environment prefixes, `sysconfig` paths,
the interpreter path, and the exact `axiom_encode.__file__` and package path.
Only then does it authenticate and attach the anonymous broker socket.

## Protected trust roots

Every invocation requires one protected JSON file containing three pairwise
distinct public roots, even when the command uses no signing capability:

```json
{
  "schema": "axiom-encode/signing-trust-roots/v2",
  "apply_ed25519_public_key": "BASE64_RAW_32_BYTES",
  "eval_ed25519_public_key": "DISTINCT_BASE64_RAW_32_BYTES",
  "corpus_release_ed25519_public_key": "THIRD_DISTINCT_BASE64_RAW_32_BYTES"
}
```

The v2 form remains valid and supplies a one-key corpus-release verification
ring. During corpus-release key rotation, use the v3 form. The first key is the
current signing identity; later keys are retired and verification-only:

```json
{
  "schema": "axiom-encode/signing-trust-roots/v3",
  "apply_ed25519_public_key": "BASE64_RAW_32_BYTES",
  "eval_ed25519_public_key": "DISTINCT_BASE64_RAW_32_BYTES",
  "corpus_release_ed25519_public_keys": [
    "CURRENT_BASE64_RAW_32_BYTES",
    "RETIRED_BASE64_RAW_32_BYTES"
  ]
}
```

A v3 file may retain the singular field, use the plural field, or include both
when the singular key equals the first plural key. The supervisor rejects an
empty ring, invalid base64, non-32-byte entries, duplicate or aliased roots,
unknown schemas, and conflicting singular/plural current keys. The broker
retains its singular current-key status field and also passes the ordered ring
to Python release verification. Release signing is unchanged and uses only the
current private signing identity.

PKIX public-key PEM is also accepted for singular key fields; plural keyring
entries are raw 32-byte keys in base64. Environment public-key values never
define trust and are rejected, including `AXIOM_CORPUS_RELEASE_PUBLIC_KEY`.
These three legacy/current private-key environment names are fatal when
present, including with an empty value:

```text
AXIOM_ENCODE_APPLY_SIGNING_KEY
AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY
AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY
```

They are documented here solely as forbidden inputs; do not configure them.

## Invocation

The process manager pre-opens one connected local Unix stream socket per
operation scope. Descriptor 3 below is the apply signer:

```bash
build/axiom-encode-signing-supervisor \
  --apply-signer-fd 3 \
  --trusted-signing-roots /etc/axiom/signing-trust-roots.json \
  --trusted-python-runtime-root /opt/axiom/python \
  --trusted-python-import-root \
    /opt/axiom/python/lib/python3.13/site-packages \
  --trusted-python-package-root \
    /opt/axiom/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom/python/bin/axiom-encode encode ... --apply
```

Use `--eval-signer-fd` for eval evidence and both flags only when both
capabilities are needed. Verification-only invocations omit both signer flags;
the broker still exposes all three protected public roots but no signing
capability.
The launcher is validated but never executed; the
supervisor executes the validated interpreter and protected bootstrap directly.

The Python environment is constructed from empty. It contains fixed locale,
bytecode/isolation controls, a fixed path containing only the validated runtime
tool directory, and the anonymous
broker markers only. Ambient Python/loader paths, Git configuration, proxies,
cloud/GitHub credentials are not forwarded. The exact Axiom Supabase and
OpenTelemetry variables documented by the CLI are parent-only allowlist values;
the subprocess scrub removes them before model/reviewer execution.
Every required subprocess tool must therefore be installed at a protected
absolute location in that runtime; ambient system or user tool lookup is not a
deployment input.
Purpose-specific model/reviewer subprocess environments are separately rebuilt
without the broker capability or parent-only service configuration.

The Python client marks the broker descriptor non-inheritable immediately and
closes it in fork children. It authenticates the actual peer using kernel socket
credentials (a Linux `SO_PASSCRED`/`SCM_CREDENTIALS` challenge response from the
post-fork broker; Darwin `LOCAL_PEERPID`, plus peer euid/egid when exposed),
rather than trusting the environment PID marker or socket-creation credentials.

## Protocols and signature domains

External signer protocol v2 and internal broker protocol v4 use unsigned
32-bit big-endian length-prefixed JSON with exact schemas. They enforce bounded frames,
exact fields, no duplicate/unknown keys, positive increasing IDs, fixed scopes,
and ten-second external-signer deadlines.

The supervisor completes the broker's private initialization frame before
Python starts. On Linux, the post-fork broker then answers a one-byte pre-frame
challenge while `SO_PASSCRED` is enabled; Python admits v4 traffic only when the
attached `SCM_CREDENTIALS` PID/euid/egid matches the expected broker. This avoids
the stale creator credentials returned by `SO_PEERCRED` for a pre-fork
socketpair.

The v2 challenge signs:

```text
"axiom-encode/external-signer-challenge/v2\0" + scope + "\0" + nonce
```

For every sign request the signer signs, and the broker verifies, these exact
persisted-signature bytes:

```text
"axiom-encode/external-signer-sign/v2\0" + scope + "\0" + canonical_payload
```

The only scopes are `apply_ed25519` and `eval_ed25519`. Apply signatures cannot
verify as eval signatures or vice versa, even if the same test key is used.
Production also rejects any equal pair among the three roots before connecting
either signer.

This breaking cut emits only apply manifest
`axiom-encode/applied-rulespec/v5`, eval verdict
`axiom-encode/eval-result-verdict/v5`, and homogeneous eval suite v5 schemas,
with signature algorithm `ed25519-domain-v1`. Older schemas/protocols are
rejected; there is no translator or compatibility path.

## Byte-identical manifest refresh

`refresh-applied-manifest` is the supported path for rebinding an existing
model apply manifest to the current validation toolchain without regenerating
the RuleSpec. It must run with the protected apply signer. The command requires
a completely clean canonical RuleSpec checkout, verifies the historical
signature, requires that the manifest exclusively covers the exact tracked
primary and companion bytes at `HEAD`, revalidates those bytes against the
current engine and corpus release, and aborts if validation changes any
RuleSpec byte. A successful run changes only the signed manifest.

The targeted workflow selects this mode with an exact
`axiom-encode/atomic-source-transaction/v4` envelope whose
`manifest_only_refresh` member is `true`. The source, canonical-refresh, and
required-test bundles must be empty, and the mode cannot be combined with
repair, import, dependency, migration, or queue operations.

## Reviewed candidate promotion

`promote-reviewed-candidate` is the non-regenerating path for a reviewed
RuleSpec head that has no historical signed manifest. The country and exact
commit must appear in the encoder's reviewed-head allowlist, the checkout must
be clean and byte-identical to that commit, the canonical primary and companion
must pass current complete-source overlay validation, and the protected signer
may add only a new manifest. The manifest identifies the reviewed RuleSpec
commit and current validation execution; it does not claim that the reviewed
bytes were freshly generated by a model.

The targeted workflow selects this mode with an exact
`axiom-encode/atomic-source-transaction/v5` envelope whose
`reviewed_candidate_promotion` member is `true`. All source, refresh, import,
repair, dependency, migration, and queue inputs must be empty. Pull-request
publication remains limited to an allowlisted reviewed-head/base-branch pair.

## Platform hardening

Linux and macOS set the core limit to zero. Linux sets
`PR_SET_DUMPABLE=0` and `PR_SET_NO_NEW_PRIVS=1`; macOS calls
`PT_DENY_ATTACH`. The broker revalidates its own protected binary and
unprivileged identity after re-exec. Only the broker receives the external
signer descriptors, and it exits when the anonymous Python capability closes.

## Production external apply-signer (`cmd/axiom-encode-apply-signer`)

The supervisor and broker never hold a private key: they connect out to an
*external signer* that speaks protocol v2 and holds the key. `cmd/axiom-encode-signing-supervisor`
is the broker (protocol v2 client); `cmd/axiom-encode-apply-signer` is the
production signer (protocol v2 server) plus a launcher that wires it to the
supervisor inside GitHub Actions. It is what makes an autonomous, signed
`encode --apply` CI leg possible without ever placing a raw key in the encoder's
environment.

The binary has two subcommands and a single production build (its `--build-kind`
is always `production`; there is no fixture policy because it validates no
filesystem trust chain — it is an unprivileged leaf that holds one key).

### `serve` — the leaf signer

Serves protocol v2 over one pre-connected Unix stream socket (`--socket-fd`) with
one operation-scoped key (`--scope apply_ed25519`). Key ingestion and capability:

- The private key is read from a **pipe or socket** descriptor (`--key-fd`),
  never from the environment, never from argv, never from a regular file, and
  never written to disk. A regular-file, tty, or device descriptor is refused so
  the key is never read from a persistent, re-readable path. The accepted formats
  are a PKCS8 PEM or the base64 of the raw 32-byte Ed25519 seed (the format used
  by the eval and corpus-release keys). A legacy HMAC secret is not a valid
  Ed25519 key and is refused — the signer fails closed rather than producing an
  unverifiable signature. The key material buffer is zeroized after parsing.
- It signs only its provisioned scope: a request for any other scope (including a
  valid-but-different one such as `eval_ed25519`) is refused. It answers only the
  two domain-bound message shapes protocol v2 defines — the challenge and the
  persisted signature — and there is no generic blob-signing endpoint. The
  payload is opaque to the signer; the `axiom-encode/external-signer-sign/v2\0apply_ed25519\0`
  domain is applied by the signer itself, so a `sign` request cannot be coerced
  into producing a signature that verifies in any other context.
- The request envelope is strict: protocol version 2 only, positive strictly
  increasing IDs, exact field sets, bounded frames, and duplicate-key/trailing/
  unknown-field rejection, mirroring the broker's own frame discipline.

Context binding (the trust that the run is genuine CI, enforced from explicit
flags rather than the ambient environment alone):

- `--expected-github-repository`, one or more `--allowed-workflow-ref`, and one
  or more `--allowed-event-name` are required. The signer refuses unless
  `GITHUB_ACTIONS=true` and the ambient `GITHUB_REPOSITORY` / `GITHUB_WORKFLOW_REF`
  / `GITHUB_EVENT_NAME` match those explicit allowlists.
- `pull_request` and `pull_request_target` are refused unconditionally, beneath
  the allowlist, because those events run with fork-controlled refs.
- `--allow-local-dev` is the only way to run outside Actions, and it is refused
  *inside* Actions. In local-dev mode the signer **self-generates a throwaway
  keypair and never reads `--key-fd`**, printing the throwaway public key so a
  developer can build a matching trust root. It is therefore structurally
  impossible to sign with a production key outside GitHub Actions.

Each process is hardened (core dumps denied; Linux `PR_SET_DUMPABLE=0` /
`PR_SET_NO_NEW_PRIVS=1`; macOS `PT_DENY_ATTACH`) as its first action, before the
key is ingested. This ordering matters: `execve` resets `PR_SET_DUMPABLE` to 1,
so the signer re-hardens immediately on start and only then reads the key — no
key material is ever resident while the process is dumpable. Zeroization of the
key buffer is best-effort: Go's `ed25519.PrivateKey` retains an internal copy the
language does not let us wipe, so the actual control against key recovery is the
process hardening (no core dump, no `ptrace`/`/proc/<pid>/mem` access from a
sibling) plus the short-lived, single-purpose process — not the `zero()` call.
Structured audit lines go to stdout — bind context, per-request content and
message SHA-256, a best-effort sanitized citation, and a shutdown count — and
never contain key material.

### `run` — the launcher / process manager

The launcher is the process manager the deployment model calls for. It:

1. hardens itself (as its first instruction in `main`, before flags are parsed —
   the key is already in its inherited environment), then reads the base64/PEM
   key from a named environment variable (`--key-env`) and **clears that variable
   from its own environment before spawning any child**, so no descendant — least
   of all the supervised encoder — can inherit it;
2. spawns the signer by its **running inode** (`/proc/self/exe` on Linux), never
   an on-disk path an operator flag or a same-UID writer could redirect, over a
   pre-opened connected socket (`socketpair`), a key pipe, and a readiness pipe,
   with a minimal `GITHUB_*`-only environment;
3. **delivers the key only after the signer signals — over the readiness pipe —
   that it has hardened** (denied core dumps, ptrace, and `/proc/<pid>/fd`
   access). The key therefore never sits on the pipe while the freshly-`exec`'d
   signer is still dumpable; if the signer exits without signaling (e.g. a
   context-binding refusal), the launcher observes EOF and aborts without writing
   the key. The key buffer is zeroized immediately after the write;
4. executes the compiled supervisor with the signer attached on
   `--apply-signer-fd 3`, the trusted-python flags, and the `-- axiom-encode
   encode … --apply` command, passing through the environment the encoder needs
   (model API keys, corpus/engine paths) but not the key (already cleared); and
5. propagates the supervisor's exit code and tears the signer down.

The supervisor performs its own forbidden-private-key-environment rejection and
per-child scrub, so the launcher is a wiring layer, not a second trust boundary.

### ChatGPT subscription generation

The tier-1 provisioner can additionally install the repository-pinned Codex CLI:

```bash
sudo .venv/bin/python scripts/provision_verification_supervisor.py \
  --destination /opt/axiom-verification \
  --supervisor build/axiom-encode-signing-supervisor \
  --site-packages .venv/lib/python3.13/site-packages \
  --apply-root "$APPLY_ROOT" --eval-root "$EVAL_ROOT" \
  --corpus-release-root "$CORPUS_RELEASE_ROOT" \
  --retired-corpus-release-root "$RETIRED_CORPUS_RELEASE_ROOT" \
  --git /usr/bin/git \
  --encoder-origin-repository github.com/TheAxiomFoundation/axiom-encode \
  --encoder-commit "$(git rev-parse HEAD)" --encoder-git-root "$PWD" \
  --install-pinned-codex-cli
```

Omit `--retired-corpus-release-root` to retain the v2 one-key file. Repeat it
for each retired verification-only key; the provisioner writes a v3 ring with
the current `--corpus-release-root` first. Repository encode workflows require
the `AXIOM_CORPUS_RELEASE_RETIRED_PUBLIC_KEY` organization variable during the
2026-07 rotation so they cannot silently provision only the new key.

Install custody remains with the operator: never automate this sudo invocation.
For a subscription run, explicitly name the source credential and its refreshed
outbox (both are credential-bearing files):

```bash
CODEX_AUTH_SOURCE="$CODEX_HOME/auth.json"
CODEX_AUTH_OUTBOX="$CODEX_HOME/auth.json.refreshed"
unset CODEX_HOME
/opt/axiom-verification/axiom-encode-signing-supervisor \
  --trusted-signing-roots /opt/axiom-verification/signing-trust-roots.json \
  --trusted-codex-cli-config /opt/axiom-verification/codex-cli.json \
  --codex-subscription-auth "$CODEX_AUTH_SOURCE" \
  --codex-auth-outbox "$CODEX_AUTH_OUTBOX" \
  --trusted-python-runtime-root /opt/axiom-verification/python \
  --trusted-python-import-root /opt/axiom-verification/python/lib/python3.13/site-packages \
  --trusted-python-package-root /opt/axiom-verification/python/lib/python3.13/site-packages/axiom_encode \
  -- /opt/axiom-verification/axiom-encode encode ... --backend codex \
     --model gpt-6.1-sol --escalation-model gpt-6.1-sol
```

The explicit flags select GPT-6.1 Sol for both generation and escalation;
omit them to use the encoder defaults (`gpt-6-luna`, escalating to `gpt-6-sol`).
The provisioner pins Codex CLI 0.159.0 and verifies both the archive digest and
the extracted executable digest. Model availability depends on the signed-in
account and CLI; availability hints are emitted only for actual account rejections.

The supervisor creates an euid-owned per-run `0700` home in the operating
system's temporary directory, copies auth into it, sets `CODEX_HOME` only to
that directory, disables update checks, binds execution and provenance to the
exact hash-verified Codex executable, exports any refresh atomically, and
removes the scratch tree. The outbox must be an absolute path in an
operator-owned directory with no group/other write access; symlinked directories
and symlink/special-file destinations are rejected.

This is environment and lookup isolation, not same-uid filesystem isolation.
The child cannot discover the operator home through `HOME`, `CODEX_HOME`, XDG
variables, or the curated `PATH`, and Codex receives only the scratch copy.
Because the supervisor and child intentionally run as the same non-root
operator, a malicious child that already knows an absolute path to a readable
operator-home file can still open it. Enforcing otherwise requires a separate
OS sandbox or identity boundary; changing `HOME` alone cannot provide that
property. Without the subscription flags, the original environment and API-key
execution path are unchanged.

### Workflow deployment and secrets discipline

Production signed re-encodes run centrally through
[`targeted-signed-reencode.yml`](../.github/workflows/targeted-signed-reencode.yml)
in `TheAxiomFoundation/axiom-encode`. Its signing job requires approval through
the `production-signing` environment. The apply private key belongs only in that
protected environment; do not configure ungated repository or organization
copies or pass signing keys from jurisdiction repositories.

The ungated apply-signing and caller App keys were removed during the October
2026 key rotation. The one-time `migrate-apply-signing-key.yml` export lane and its
`signing-key-migration` environment are retired, as is
`signed-apply-reusable.yml`, which previously accepted caller-provided keys.
Route signed re-encodes through the central targeted workflow.

The central signed-apply leg runs the same provisioning as the verification
supervisor (`scripts/provision_verification_supervisor.py`, a root-owned tree), then invokes
`axiom-encode-apply-signer run …` in place of a bare supervisor call. The private
key reaches the signing step only as a protected environment secret bound to
that step's `env:`; the launcher consumes and clears it.

**The signing leg runs only on `workflow_dispatch` from the main-branch
workflow definition, never on `pull_request`.** The workflow requires
`refs/heads/main` before entering the protected environment. The signer's own
event-name binding (refusing `pull_request*` beneath the allowlist) and
workflow-ref allowlist (pinned to `@refs/heads/main`) enforce this a second time,
independently of the YAML.

### Threat model

- **Who can trigger it.** Only an actor who can dispatch the main-branch
  workflow — i.e. a repository collaborator — can request signing, which still
  requires the protected environment approval. A fork contributor
  cannot: their PR neither receives the secret nor matches the pinned workflow
  ref / permitted event.
- **What the context binding is (and is not).** The `GITHUB_*` values and the
  allowlist flags both originate from the workflow definition, so the binding is
  **defense-in-depth against misconfiguration** (an accidental `pull_request`
  trigger, or invoking the signer from the wrong workflow), not a cryptographic
  authentication of the runner. The primary boundary is external to this code:
  the private key is held in the approval-gated `production-signing` environment,
  and the workflow is restricted to `workflow_dispatch` on main. An
  actor who could actually forge `GITHUB_*` would have to be able to edit the
  main-branch workflow — a trusted collaborator who can already dispatch it and
  request signing approval. The binding therefore never widens who can
  sign; it only narrows it further.
- **A compromised or malicious PR** can change the module content a future
  generation session encodes, but it cannot exfiltrate the key: the signing leg
  does not run on its event, the secret is absent from PR-triggered runs, and the
  supervised encoder never has the key in its environment or argv. The worst a PR
  can do is propose bad content, which the supervised `validate` gate and human
  review still catch.
- **Signer misuse for arbitrary payloads.** The signer signs only domain-bound
  `apply_ed25519` messages presented over the socket. There is no endpoint that
  signs an attacker-chosen pre-image, and an apply signature cannot be replayed as
  an eval or corpus-release signature (distinct domains and scopes).
- **Log/artifact exfiltration.** Audit lines carry content hashes, not key bytes;
  the sanitized citation is bounded and stripped of control characters to prevent
  audit-line forgery. GitHub also masks the registered secret value in logs.
- **Local-dev flag abuse.** `--allow-local-dev` cannot be used to strip context
  binding inside Actions (it is refused there) and cannot sign with a real key (it
  self-generates and ignores `--key-fd`).

### Key rotation

The apply signing identity remains singular. A signer must answer its startup
challenge with the active `apply_ed25519_public_key`; retired roots cannot answer
that challenge or authorize a new apply, refresh, migration, or retirement
signature. Changing the active public root and private signer independently
fails closed at startup.

Before rotating the root, the owner freezes protected main for each affected
repository, records its exact commit SHA, and builds and reviews an inventory of
the historical attestations that remain in use. Provision that inventory with
the retired **public** roots into the protected runner trust file:

```bash
scripts/provision_verification_supervisor.py \
  --retired-apply-root "$RETIRED_APPLY_PUBLIC_ROOT" \
  --retired-apply-inventory "$OWNER_FROZEN_APPLY_INVENTORY" \
  ...
```

Repeat `--retired-apply-root` for additional retired public roots. Neither
argument is a verification-command input. Provisioning copies the reviewed
inventory inline into the root-owned trust configuration, and the compiled
supervisor passes it to Python over the authenticated broker. Repository files,
environment variables, moving branches, manifest timestamps, and CLI callers
cannot supply historical authority. Existing v2/v3 configurations continue to
read as before; without retired apply trust, provisioning keeps their existing
format. The provisioner's inventory input must itself be a root-owned,
non-writable regular file under protected parents, with no symlink components.
The v4 trust file is bounded to 16 MiB; existing v2/v3 files retain their 64 KiB
bound. Retired apply trust uses v4 with the active apply and evaluation roots,
the corpus release ring, and these additional fields:

```json
{
  "schema": "axiom-encode/signing-trust-roots/v4",
  "apply_ed25519_public_key": "ACTIVE_BASE64_RAW_32_BYTES",
  "eval_ed25519_public_key": "DISTINCT_BASE64_RAW_32_BYTES",
  "corpus_release_ed25519_public_keys": ["THIRD_BASE64_RAW_32_BYTES"],
  "retired_apply_ed25519_public_keys": ["RETIRED_BASE64_RAW_32_BYTES"],
  "retired_apply_inventory": {
    "schema": "axiom-encode/retired-apply-inventory/v1",
    "repositories": [{
      "repository": "TheAxiomFoundation/rulespec-us",
      "cutoff_commit": "0123456789abcdef0123456789abcdef01234567",
      "artifacts": [{
        "path": ".axiom/encoding-manifests/us/statutes/26/1.json",
        "sha256": "EXACT_ARTIFACT_BYTES_SHA256_HEX",
        "covered_files": [{
          "path": "us/statutes/26/1.yaml",
          "sha256": "EXACT_GENERATED_BYTES_SHA256_HEX"
        }]
      }]
    }]
  }
}
```

All paths are canonical checkout-relative paths. Digests are 64 lowercase
hexadecimal characters and bind the exact serialized manifest or receipt bytes,
including whitespace, and the exact covered generated-file bytes. The repository
identity must match the canonical GitHub origin, and the immutable cutoff must
be available locally with complete Git history and no graft or replacement
metadata. Shallow checkouts fail closed. Inventory entries grant no authority to a relocated,
newly introduced, changed, or replayed attestation. A committed post-cutoff
replacement or removal prevents direct admission on descendant heads, even if
the original bytes are later restored. Explicit retrospective verification can
still admit frozen evidence at pre-cutoff heads.

Inventory every signed container that the historical readers consume, including
standalone migration and legacy replacement receipts, and include the file hashes
from all recursively embedded `applied_files` evidence. A covered file normally
binds its live bytes and the cutoff blob. If a retirement deleted it, or migration
changed its path or contents before the cutoff, its `covered_files` entry may also
contain `"source_commit": "FULL_LOWERCASE_SHA"`. That immutable source commit must
be an ancestor of the cutoff and provide the exact original file bytes. It grants
only historical evidence verification; it does not authorize restoring those
bytes in the live checkout. Distinct historical versions of the same path need
separate entries with their respective source commits and byte digests.

Persisted artifact readers explicitly request historical verification:
`guard-generated --all`, golden regeneration admission, signed import inventory,
manifest census, and the input admission for `refresh-applied-manifest`. They
still enforce their existing source, toolchain, path, byte, and structural
checks. After successful retired-root signature and frozen-byte admission,
producer checks bind the frozen signed encoder identity; deploying a newer
verifier does not turn that historical producer into a fresh one. Its internal
provenance and execution evidence must still agree. Active-root manifests retain
the existing producer-pin checks. A successful refresh signs its replacement
with the active root. Ordinary
`guard-generated` diffs require the active root. A scheduled or manual audit of
an already frozen commit must explicitly use `guard-generated --retrospective`;
its actual checkout head must equal or precede the trusted cutoff, and its
`--head-ref` must be `HEAD` or that exact SHA.

Nested retirement, migration, legacy replacement, and retained successor
evidence uses the same protected inventory. Each retired signature must identify
the exact embedded historical payload preserved by a frozen inventoried
container at its original repository and path. Wrapping a new old-key payload in
an active-root outer attestation cannot admit it. Active-root signatures continue
to verify without historical inventory.

After deploying the reviewed verifier and trust configuration, the owner changes
the active public root and the central private signer while entrypoints remain
frozen. Prove both a new-root artifact and the frozen historical readers before
resuming. The shared verification workflow must separately provision the retired
roots and owner inventory and select restricted retrospective mode for historical
scheduled guards; callers must deploy the corresponding reviewed verifier pins.
The shared-workflow follow-up must source the inventory from owner-reviewed
protected configuration at an immutable revision, provision it into protected
runner paths, and keep pull-request inputs and repository artifacts out of that
trust source. Deploy the reviewed shared-workflow and encoder pins before the
owner changes either active signing identity.

The active public root is published as the `AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY`
Actions variable, which the workflows provision into the supervisors' root-owned
trust file. The active private seed is the `production-signing` environment's
signing secret.

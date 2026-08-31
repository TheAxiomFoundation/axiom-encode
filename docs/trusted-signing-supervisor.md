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

## Platform hardening

Linux and macOS set the core limit to zero. Linux sets
`PR_SET_DUMPABLE=0` and `PR_SET_NO_NEW_PRIVS=1`; macOS calls
`PT_DENY_ATTACH`. The broker revalidates its own protected binary and
unprivileged identity after re-exec. Only the broker receives the external
signer descriptors, and it exits when the anonymous Python capability closes.

The supervisor remains resident while the encoder runs. On Linux it becomes a
child subreaper before either the broker or encoder starts, puts the encoder in
a new process group, observes the leader exit without reaping it, kills that
pinned process group, and repeatedly kills/reaps adopted descendants before a
signature or successful status can be published. This includes descendants
that escape the original group through `setsid` and double-forking. Cleanup
failure overrides the encoder result; otherwise the original exit code or
signal is preserved. The production runner is Linux. macOS retains the resident
supervisor and process-group separation but has no equivalent child-subreaper
facility, so local macOS runs do not make the stronger descendant-drain claim.

The isolated runtime's Git broker accepts only its existing read/write command
allowlist. An optional `-C` repository must be an absolute, normalized,
realpath-identical existing directory. The broker then supplies an exact
command-scoped `safe.directory` value to the root-owned Git binary. This lets a
dedicated verifier UID inspect root-owned pinned encoder/corpus snapshots without
consulting global/system Git configuration or admitting caller-provided `-c`
options, symlink aliases, or wildcard safe directories.

## Production external apply-signer (`cmd/axiom-encode-apply-signer`)

The supervisor and broker never hold a private key: they connect out to an
*external signer* that speaks protocol v2 and holds the key. `cmd/axiom-encode-signing-supervisor`
is the broker (protocol v2 client); `cmd/axiom-encode-apply-signer` is the
production signer (protocol v2 server) plus a launcher that wires it to the
supervisor inside GitHub Actions. It is what makes an autonomous, signed
`encode --apply` CI leg possible without ever placing a raw key in the encoder's
environment.

The binary has two subcommands and a single production build (its `--build-kind`
is always `production`). The `serve` subprocess is an unprivileged leaf that
holds one key. Production `run` is deliberately different: it retains a root
identity long enough to create two children under distinct, explicit non-root
signer and verifier identities and to revoke both capabilities if either
supervised boundary is compromised.

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
5. observes the verifier supervisor with `waitid(WEXITED|WSTOPPED|WNOWAIT)`;
   a stop is a compromise event, not a pause. It kills the distinct-UID signer
   first, kills the verifier supervisor's process group, reaps both children,
   and only then propagates success or the supervisor's exit status.

Production `run` requires real, effective, and saved root UID/GID; effective
`CAP_KILL`, `CAP_SETUID`, and `CAP_SETGID`; empty inheritable and ambient
capability sets; and four explicit, nonzero signer/verifier UID/GID values with
the two identities distinct. Both children receive empty supplementary groups
and a parent-death signal. The supervisor executable must be a symlink-free,
root-owned, non-writable static ELF named
`axiom-encode-signing-supervisor`. The supervisor still performs its own
forbidden-private-key-environment rejection and per-child scrub; these checks are
independent layers rather than substitutes for the launcher's identity boundary.

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
  -- /opt/axiom-verification/axiom-encode encode ... --backend codex
```

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

The signed-apply leg runs the same provisioning as the verification supervisor
(`scripts/provision_verification_supervisor.py`, a root-owned tree), installs
static, root-owned guardian, launcher, and supervisor binaries, and creates
separate non-login `axiom-signer` and `axiom-verifier` accounts. It verifies that
their UID/GID values differ from each other and from the ordinary runner before
invocation. A root-identity outer process guardian then runs the root-identity
launcher, which drops only the leaf signer and verifier supervisor to their
dedicated identities. Verification-only commands use the guardian's ordinary
`run --uid/--gid` mode under `axiom-verifier` and receive no signing descriptor.

The outer guardian is a root child subreaper. It accepts only a protected native
`axiom-encode-apply-signer run` target in root-controller mode, refuses
`--allow-local-dev`, and drains the entire launcher tree before returning. This
closes the same-UID escape in which an untrusted verifier could stop its launcher,
kill an inner supervisor, and retain the broker descriptor in a detached
descendant. The verifier cannot signal either the root launcher or the distinct-
UID signer; a stop or exit of the verifier supervisor causes immediate signer
revocation and verifier-group teardown, while the outer guardian kills and reaps
escaped descendants before publication resumes.

The private key reaches the job only as a secret bound to that one step's
`env:`. The runner shell passes it through the non-dumpable root guardian to the
root launcher; those already-trusted ancestor processes retain their inherited
environment until they exit. The launcher consumes the value and removes it
from its own environment before spawning either child, so no signer/verifier
descendant inherits the environment variable. No model process receives the key
or runs with the signer, launcher, runner, or root identity.

**The signing leg must run only on `workflow_dispatch` or `schedule` from the
main-branch workflow definition, never on `pull_request`.** GitHub does not expose
repository/organization secrets to workflows triggered by a fork PR, and a PR that
edits the workflow text runs with the PR's own definition — so fork- or
PR-modified workflow text can never see the secret. The signer's own event-name
binding (refusing `pull_request*` beneath the allowlist) and workflow-ref
allowlist (pinned to `@refs/heads/main`) enforce this a second time, independently
of the YAML.

#### Explicit policy targets

The targeted signed-reencode workflow's `replace_rulespec_path` dispatch input
names one explicit target, while the exact atomic-source/v2 envelope
authenticates `target_operation: create|replace`. Existence at the pinned base
is only a consistency check: it never infers the operation. The workflow and
CLI pass the corresponding explicit create/replace flag, bind the operation in
the per-lane review-contract/v3 context, and record it in the signed v5 apply
manifest. The CLI
admits creation only with `encode --apply --mode repo-augmented`, under the
requested jurisdiction's `policies/` root, and only when the primary, companion,
canonical manifest, pre-monorepo manifest placement, and every base-manifest
claim are absent. Required signed imports and structured review/test contracts
remain bound by the same externally signed apply lane. Every creation case must
assert an output from the exact new module; bounded nested values let that
signed witness include entity/relation rows without weakening exact type or
shape matching.

This is deliberately a one-target primitive. Creation requires every source
module and signed import to be present at the reviewed PR base. A nonempty
citation-derived source bundle combined with `target_operation: create` is
rejected before encoding or signing: committing source imports first would move
the encoder's clean HEAD away from the reviewed base and make the creation
receipt describe the wrong absence boundary. Supporting both in one run requires
a future transaction-wide design that reproves base-A absence across signed
intermediate commits. Ordered dependent policy changes likewise require a
bundle schema that binds every explicit citation/path/import/review lane and its
final validation boundary.

Ordinary nonlegacy replacement is also limited to one direct target lane. It may
reuse already signed imports and may run before the existing dependent lanes,
but a nonempty source bundle or canonical-refresh dispatch bundle is rejected
both during normalized dispatch validation and during final trusted artifact
reparse. Those intermediate commits do not yet carry signed
`replacement_target` base evidence or expected-parent lineage, so accepting them
would leave the target's original ownership/digest boundary unauthenticated
across lanes. Bundled replacement remains gated until both contracts exist;
source and refresh lanes must land in separate signed transactions meanwhile.

An in-place legacy replacement may also consume the exact generated-v1 manifest
shape stranded at the old manifest root by the documented federal consolidation.
Admission is limited to `us/{policies,statutes}/...` and its deterministic old-root
counterpart,
the historical field/null-artifact shape, clean-HEAD file hashes, and no
coexisting canonical owner. The old HMAC is never trusted: the completed fresh
replacement receives the normal v5 Ed25519 apply manifest and signed migration
receipt.

#### Protected publication snapshot

The targeted workflow reconstructs the captured RuleSpec Git tree under a
root-owned snapshot parent and gives only that one checkout to a dedicated
`axiom-verifier` system UID. The encoder, corpus checkout, runtime, and artifact
directory remain root-owned; the one materialized corpus release object is
made root-owned `0444` so the verifier can read but not replace it. Protected
stage/guard runs consume the pinned encoder through the trusted Git broker
described above.

Every untrusted encoder/model invocation also runs as `axiom-verifier`, never as
the GitHub runner UID. Before the first invocation, the workflow verifies that
the runner temporary directory, `_runner_file_commands` directory, and the four
active `GITHUB_ENV`, `GITHUB_PATH`, `GITHUB_OUTPUT`, and `GITHUB_STEP_SUMMARY`
files are canonical, runner-owned, distinct where required, and not writable by
group or other identities. Among workflow mutation surfaces, only the scoped
RuleSpec checkout, generation database, and lane output are writable by
`axiom-verifier`; pinned encoder, corpus, rules-engine, and repair inputs are
root-owned and read-only to it. The supervisor may also create its documented
verifier-owned per-run scratch home, and the unprivileged UID retains ordinary
OS temporary-directory access; neither is accepted as a publication input.
Consequently a model process cannot persist `BASH_ENV`, `PATH`, or another
command-file mutation into a later runner step. Ownership of generated outputs
never returns to the runner. After the resident Linux supervisor has drained
the model's complete descendant set, a bounded root process copies only regular,
single-link worktree files into a fresh protected clone without reading the
model-owned `.git`; protected staging must reproduce the checkpoint tree before
publication can continue.

Audit output first crosses a bounded flat-file copier, is revalidated against
the exact signed transaction, and is then sealed root-owned and read-only with
a canonical content manifest. The reviewed Git tree and artifact-manifest
digest are bound into a root-written publication witness and commit trailer.
Publication pushes that exact commit SHA through a root-owned credential
configuration and uses an environment-cleared GitHub API client; the returned
draft PR must bind the reviewed base, head, and repository before a separate
root-written receipt is accepted. Artifact upload reads only the sealed copy.

### Threat model

- **Who can trigger it.** Only an actor who can dispatch (or schedule) the
  main-branch workflow — i.e. a repository collaborator. A fork contributor
  cannot: their PR neither receives the secret nor matches the pinned workflow
  ref / permitted event.
- **What the context binding is (and is not).** The `GITHUB_*` values and the
  allowlist flags both originate from the workflow definition, so the binding is
  **defense-in-depth against misconfiguration** (an accidental `pull_request`
  trigger, or invoking the signer from the wrong workflow), not a cryptographic
  authentication of the runner. The primary boundary is external to this code:
  GitHub does not expose repository/organization secrets to fork-PR-triggered
  runs, and the workflow is restricted to `workflow_dispatch`/`schedule`. An
  actor who could actually forge `GITHUB_*` would have to be able to edit the
  main-branch workflow — a trusted collaborator who can already dispatch it and
  obtain signatures legitimately. The binding therefore never widens who can
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

Rotating the apply key means generating a new Ed25519 keypair, publishing the new
public half to the `AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY` Actions variable (which
the verification and signing supervisors load as a trust root), and placing the
new private seed into the signing secret. Because the trust root and the signer's
key are checked against each other at the broker challenge, a mismatched pair
fails closed at startup rather than producing an unverifiable manifest. Old
manifests remain verifiable only while the corresponding public root is still
published, so rotate the variable and re-verify historical manifests together.

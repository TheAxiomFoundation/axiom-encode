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

PKIX public-key PEM is also accepted. Environment public-key values never
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

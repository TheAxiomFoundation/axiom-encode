# Local validate-rulespec parity

`axiom-encode ci --repo /path/to/rulespec-<country>` runs the validation gates
declared by that lane's reusable `validate-rulespec.yml` caller. It reads the
four pinned dependency commits and caller inputs, verifies the strict
`.axiom/toolchain.toml` and waiver-set binding, authenticates each local checkout
against `origin/main`, selects changes relative to `--base-ref origin/main`, and
runs every applicable gate in workflow order.

Parity is fail-closed and pin-relative: this release implements workflow pins
`615c1df9b9ace7deea84da65efd137f46f8bad2b` and
`34bcfab235c585c47292c95f51be1a4f4f91d29e`, including their differing treatment
of `programs/` in the money-atom file scan. A caller pinned to another workflow
revision is rejected until the fixture, registry, and implementation are updated
together. The changed-file classifier likewise requires the installed
`axiom-oracles` VCS commit to match the dependency declared by the pinned encoder.

Use explicit checkout overrides when the dependencies are not siblings:

```console
axiom-encode ci --repo ../rulespec-dk \
  --corpus-path ../axiom-corpus \
  --engine-path ../axiom-rules-engine \
  --rulespec-us-path ../rulespec-us \
  --encode-path . \
  --corpus-release-public-key "$PUBLIC_KEY" \
  --offline
```

The public key must be supplied only through
`--corpus-release-public-key`. Retrieve the organization variable for a local
run with:

```console
gh api /orgs/TheAxiomFoundation/actions/variables/AXIOM_CORPUS_RELEASE_PUBLIC_KEY --jq .value
```

The command never reads `AXIOM_CORPUS_RELEASE_PUBLIC_KEY` from the environment.
It constructs a verification-only release binding in the library, acquires no
signing capability and never runs `--apply`. It writes temporary report inputs;
the sole non-report write is caching a missing immutable public release object
at the workflow-defined corpus release path. The command downloads that object
from the caller's `corpus-release-base-url` unless `--offline` is set.

A dependency checkout whose `HEAD` differs from its caller pin fails resolution
and names both SHAs. `--allow-ref-mismatch` permits all gates to run, but a
successful run receives the qualified `PASS-WITH-MISMATCHED-DEPS` verdict and
exit code 3. Its mandatory final banner enumerates every dependency mismatch.
Plain `PASS` and exit code 0 require all four dependency checkouts to match.
Gate failures and resolution failures remain `FAIL` with exit code 1. JSON
reports expose the same value in `verdict` and list structured
`dependency_mismatches`.

The imported (ambient) `axiom_encode` implementation is also fail-closed against
the caller's encoder pin: source-checkout `HEAD` is compared when available,
otherwise package versions are compared. Exact parity means running this tool
from the pinned encoder checkout. `--allow-encoder-mismatch` is the honest mode
for development loops on newer encoders and produces the same qualified verdict
and mismatch banner. Executing every gate in a subprocess sourced from the
pinned checkout is the longer-term gold path and is intentionally out of scope
for this command version.

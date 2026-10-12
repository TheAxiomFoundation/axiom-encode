# Signed manifest encoder admission

The generated-change guard verifies each changed manifest's production signature
against the protected apply trust root before considering its producer identity.
It admits the existing exact encoder pin identity, or a producer commit `P` for
which the pinned encoder commit `B` is an ancestor and `P` belongs to the
first-parent history of encoder `main`. The proof uses
`git merge-base --is-ancestor B P` and membership in
`git rev-list --first-parent <main-ref>`.

Version numbers do not establish order. Different main commits can carry the
same version. At the exact pin, the existing repository, commit and version
binding remains required. At a newer admitted commit, all manifest identity
blocks must still agree internally on commit, version and identity source.
Final-byte hashes, source citations, release and toolchain bindings, file
ownership and every other manifest check remain required. The pinned validator
continues to validate the output bytes.

## History source and failures

`--expected-encoder-checkout` identifies the clean, provisioned encoder checkout
at the consumer pin. The guard derives the pin from that checkout and carries
its Git history separately from the portable identity record; manifests cannot
supply a checkout path. Without this argument, a clean running encoder Git
checkout supplies the context. A Git-free runtime can verify the exact pin, but
cannot prove admission of a different producer without the explicit checkout.

The shared `validate-rulespec.yml` authenticates provisioned dependencies by
fetching complete history with `--unshallow` when necessary. For axiom-encode's
current main branch, that fetch creates **`refs/remotes/origin/main`** in the
provisioned encoder checkout. The guard prefers this explicitly fetched ref.
If it is absent, a full development checkout can use **`refs/heads/main`**.
Both names are hardcoded: neither `origin/HEAD` nor the default-branch symref is
used to authorize producers. An existing remote main ref takes precedence over
a potentially stale local main.

The guard performs no fetch. Missing history, an unknown commit, a shallow
checkout, a missing main ref, missing history needed for the proof, a timeout or a Git
error rejects a different producer with an ancestry diagnostic. Failed fetches
must fail the workflow's existing dependency-authentication step. The history
proof disables replacement objects and lazy object fetches. Feature commits and
merged second-parent commits remain inadmissible even when the pin is their
ancestor. Producers older than the pin remain inadmissible even with the same
version. Historical manifests nested in retirement or migration receipts retain
their existing exact identity and signature verification.

## Scope and pin protection

This admission rule resolves the targeted signed-output producer/pin mismatch
in #1844, including reviewed-candidate promotion and deterministic wrappers.
It does not change production signing, protected trust roots, deployment policy
or the shared validation workflow. Protecting consumer pins against pull-request
changes, enforcing protected dependency inputs and enforcing authenticated,
fresh main-history fetches remain the separate shared-workflow work.

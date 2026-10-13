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

Callers must authenticate the encoder checkout and provision complete main
history before invoking the guard. The proof prefers `FETCH_HEAD` only when its
record unambiguously identifies a fetch of the official encoder repository's
`main` branch. If `FETCH_HEAD` is absent, it requires a direct, non-symbolic
`refs/heads/main`; a present invalid or ambiguous fetch record rejects
ancestry admission. `refs/remotes/origin/main` does not authorize producers.
Callers must ensure the selected main history is authentic and fresh; this
repository does not establish that an external workflow fulfills these
requirements.

The guard performs no fetch. Missing history, an unknown commit, a shallow
checkout, a missing main ref, missing history needed for the proof, a timeout or
a Git error rejects a different producer with an ancestry diagnostic. Callers
must stop on failed history fetches. Git commands used to verify the checkout
and prove ancestry discard ambient `GIT_*` overrides, disable graft files,
replacement objects and lazy object fetches. Git-backed checkout resolution and
ancestry proofs refuse repositories containing `info/grafts` or `refs/replace/*`.
Symbolic selected local main refs and unusable fetched-main records reject
ancestry admission. Feature commits and merged second-parent commits remain
inadmissible even when the pin is their ancestor. Producers older than the pin
remain inadmissible even with the same version. Historical manifests nested in
retirement or migration receipts retain their existing exact identity and
signature verification.

## Scope and pin protection

This admission rule applies to signed model output, reviewed-candidate promotion
and deterministic wrappers. It does not change production signing, protected
trust roots, deployment policy or external validation workflows. Callers remain
responsible for protecting consumer pins and dependency inputs against
pull-request changes and for authenticating fresh main history.

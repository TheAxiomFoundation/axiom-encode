# Pinning the engine for encode

`axiom-encode encode` accepts `--axiom-rules-engine-ref` with a full lowercase
40-character Git commit SHA, alongside `--axiom-rules-engine-path`. The engine
checkout must be clean and its HEAD must equal that ref. A receipt for another
commit inside that checkout does not satisfy this encode contract.

Prepare the executable through `axiom-encode engine-bind --axiom-rules-engine-path
<checkout> --pin <commit>`. Encode can also build a missing pinned executable using
the existing locked release-build path. Binding receipts are local attestations
against accidental drift; they do not grant signing authority or replace protected
runtime and key custody requirements.

The explicit ref follows each generation job, retained-candidate preflight,
deterministic repair, retry, companion-test validation, and apply overlay,
including dependent baseline checks. Pipeline instances capture the pin before
worker execution. No ambient process setting selects the override. Omitting the
option preserves the existing engine-selection behavior.

Each job checks the source and bound executable before generation and again
before emitting a successful result. Apply validation captures the executable
path, ref and SHA-256 before its overlay checks and refuses changes while
recording success, before apply, and during the installation closure check. These
rechecks cannot rebuild a changed executable. The captured Git identity must also
agree with the explicit ref. The additional binding stays in the in-memory
validation snapshot; the signed portable engine identity remains exactly
`repository` and `commit`.

This option neither updates a RuleSpec toolchain file nor relaxes the supervised
encoder, generated-module guard, corpus, signing, or apply requirements. Existing
protected installations need an operator-reviewed upgrade before they can use a
new CLI option; preparing a local engine receipt alone does not update them.

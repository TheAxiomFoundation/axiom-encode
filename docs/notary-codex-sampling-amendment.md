# Proposed v33 amendment: unexposed sampling metadata

Status: **proposal for Max's explicit approval**, not an already-approved v33
rule. The original `notary-admission-design.md` is unchanged in this PR. This
change must be accepted before enabling the personal-Codex producer; merging
code alone does not waive the v33 §9 audit or enroll a runtime.

## Problem and evidence

V33 §2.1/§2.2 require `sampling.temperature` to be a canonical-decimal string;
`seed` may already be null when the runtime does not expose it. The existing
`CodexCLIBackend._run_codex_exec` in `harness/backends.py` does not pass a
temperature or seed to Codex. The official [Codex configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference)
checked on 20 September 2026 documents neither setting. We cannot truthfully
record a numeric value from that interface. A configured guess such as `0` or
`0.5` would misdescribe the emission.

## Exact proposed change

In the existing closed generation-event v1 `sampling` object, permit
`temperature: null` when the runtime does not expose the value, parallel to
`seed: null`. Numeric values remain canonical-decimal strings, never JSON
numbers. Missing/extra keys and malformed numeric strings still refuse.
Null means **not exposed/unknown**, never zero or a claim of determinism.

For this supervised Codex adapter specifically, require exactly
`{"temperature":null,"seed":null}` in the custodian configuration. Validate
this before accepting any generation request. The host has no temperature/seed
setting and must not copy a user-declared numeric value into an emission.
A future backend exposing those values needs its own measured adapter change.

This is a deliberate amendment to the existing schema acceptance, proposed
because the fields are explicitly informational and non-authorizing in v33.
It introduces no new key, role, signing authority or generation path. Max may
instead choose a versioned successor; that decision should precede activation.
Older strict-v33 verifiers will refuse null temperature, so update all deployed
producer/verifier pins together through the dedicated rollout.

## Acceptance

- Existing canonical decimal-string records still parse.
- Null temperature and null seed survive the actual host export and signature
  verification path.
- JSON numbers, missing/extra fields and noncanonical decimal strings refuse.
- The Codex host refuses configured numeric sampling values before generation.
- Enrollment, live writer permission, exact encoder/runtime identity, signature
  checks, byte coverage, hardware approval and publication are unchanged.

This PR is stacked on the complete implementation #1662. It contains the code
and regression tests needed if Max approves the proposal; it does not silently
rewrite the already-approved design document.

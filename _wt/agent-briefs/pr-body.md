Adopts the shared `receipt` package for the final Ed25519 verification step of encoder apply manifests — the fourth parallel implementation of this primitive consolidates onto the extraction (ops#3), per the adoption plan bound in receipt#7 (Role 1 / sequencing step 2) and consistent with the notary charter #1192: this does **not** reinterpret `apply_ed25519` — same domain bytes, same key, same claim, same issue strings; the v5 generator class keeps its meaning forever under the dual-era rule, and a hardened shared verifier is how that class stays verifiable indefinitely. The notary's own key/domain (`notary_ed25519`, `axiom/notary-acceptance/v1`) will consume `receipt.sign` from birth with role separation, per the phase-1 design round.

## What changes

`_applied_encoding_manifest_signature_issue` only: every envelope check (missing/malformed signature object, unsupported algorithm, trust-root load failure, unknown signing key, value type/base64/length) keeps its exact issue string and ordering; the final cryptographic step becomes a 1-of-1 `receipt.sign.verify_threshold` — `raw-sha256` keyring whose fingerprint is derived from the broker-delivered root at call time, domain composed from `signing_broker`'s own constants (`_SIGNATURE_DOMAIN_PREFIX` + scope), `allow_legacy=False` stated explicitly (the apply keyring declares no legacy generations; receipt 0.3.0 requires the statement at every call site). `SignError` maps to the existing `has an invalid encoder apply manifest signature`. The signing side and the broker are untouched.

**Trust model unchanged, stated plainly:** the root still arrives from the protected broker/org-var configuration; this PR adds no code-committed pin. What the encoder gains is the shared, differential-hardened verification mechanics (receipt's `.sign` is proven byte-equivalent to the ledger verifier by a 57-test differential re-run on every package change) and the N-of-M-ready shape for the three scoped org roots.

## Pinning

`receipt==0.3.0`, hash-pinned in `uv.lock` and cross-checked against the verified PyPI artifacts (whl `89414893…`, sdist `31085b3c…`). Upgrade rule in the import-site comment: any receipt bump reruns this repo's signature tests at the new pin first.

## Verification

- `tests/test_receipt_sign_adoption.py` (7): sign→verify round trip through the real signer/verifier pair; flipped signature, flipped payload, wrong key, wrong domain scope each yield exactly the existing issue string; envelope issue strings and ordering pinned; a delegation-observation test pins the exact receipt call including `allow_legacy=False`.
- Version-provenance suite 16/16 (coordinated pyproject/`__init__`/uv.lock bump to 0.2.1302).
- Focused check set (CLAUDE.md): 1054 passed. Ruff + compileall clean.
- Full suite: **4432 passed, 29 skipped**; the 8 failures are `test_provision_supervisor`/`test_policyengine_runtime`/`test_provision_verification_supervisor` host-constraint classes (root-owned git, official PE checkout, base-runtime replacement) — reproduced identically on pristine base `fc04012d`, pre-existing on this host, untouched by the branch.

## Cycle record (CLAUDE.md PR discipline)

Build ran a three-party review-fix cycle with no actionable implementation findings; one declared blocker (the version-provenance bump requires `src/axiom_encode/__init__.py`, fenced off in the build brief) was adjudicated and executed as a coordinated three-file bump. Fable adjudication pass on top: receipt 0.2.0→0.3.0 pin, `allow_legacy=False`, spy-test expectation. Draft pending encode-lane coordination per the .github#39 regime; no workflows, toolchain files, waivers, or pinned refs touched.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

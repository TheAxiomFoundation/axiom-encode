# Task: axiom-encode adopts receipt.sign for apply-manifest verification (mechanical-equivalence shim)

You are working in `/Users/maxghenis/TheAxiomFoundation/axiom-encode/_wt/receipt-sign-adoption` (a git worktree on branch `receipt-sign-adoption`, based on origin/main = fc04012d). This replaces the inner Ed25519 verification of encoder apply manifests with the shared `receipt` package (PyPI receipt==0.2.0, repo TheAxiomFoundation/receipt) while keeping the encoder's observable behavior byte-identical. The signing BROKER and everything on the signing side stay untouched — this adoption is verify-side only.

Read first:
1. `src/axiom_encode/cli.py` — the functions `_applied_encoding_manifest_signature_issue`, `_unsigned_applied_encoding_manifest_bytes`, `_applied_encoding_manifest_key_id`, `_coerce_applied_encoding_manifest_verifier`, `_raw_ed25519_public_key` (near line 18240), and every caller of `_applied_encoding_manifest_signature_issue`.
2. `src/axiom_encode/signing_broker.py` — `canonical_signing_message`, `_SIGNATURE_DOMAIN_PREFIX`, the scopes. Do not modify this file.
3. `tests/test_cli.py` around line 9570 — the existing issue-string assertions are the behavior oracle; they must stay green UNCHANGED.
4. The receipt API: `receipt.sign` provides `KeySpec`, `KeyringSpec`, `verify_threshold(payload, signatures, public_keys, keyring, *, domain, label)`, `raw_public_key_sha256`, `SignError`. Read the installed module source once you add the dependency.

## The change

In `_applied_encoding_manifest_signature_issue` ONLY:

- Keep every existing envelope check and its exact issue string and ordering: missing/malformed signature object, unsupported algorithm, trust-root load failure, unknown signing key (key_id comparison), non-string / non-base64 / wrong-length value checks.
- Replace the final cryptographic step — currently `public_key.verify(raw_signature, canonical_signing_message("apply_ed25519", unsigned_bytes))` guarded by `except InvalidSignature` — with the receipt equivalent:
  - Build a 1-of-1 `KeyringSpec`: single `KeySpec` with `key_id` of your choosing (e.g. `"apply-root"`), `fingerprint=receipt.sign.raw_public_key_sha256(raw_public_key_bytes)`, `scheme="raw-sha256"`, where `raw_public_key_bytes` comes from the already-coerced verifier public key.
  - Call `verify_threshold(unsigned_bytes, {"apply-root": raw_signature}, {"apply-root": raw_public_key_bytes}, keyring, domain=<the exact bytes canonical_signing_message prepends for scope "apply_ed25519">, label=<a stable label>)`. Compose the domain from the imported `_SIGNATURE_DOMAIN_PREFIX` and scope constants — do not re-hardcode the prefix bytes.
  - `except SignError: return "has an invalid encoder apply manifest signature"` — the same issue string as today, so callers and tests observe nothing new.
- The trust model is unchanged and the PR must not claim otherwise: the root still arrives from the protected broker/org-var configuration; the keyring fingerprint is computed from that delivered root at call time. What the encoder gains is the shared, differential-hardened verification mechanics and the N-of-M-ready shape (the three scoped org roots can later become a real multi-key ring).
- Import `receipt.sign` at whatever placement matches the file's import discipline. A missing receipt package must be a hard ImportError at that point, never a skipped check.

## Packaging

- Add `receipt==0.2.0` to `pyproject.toml` dependencies and refresh `uv.lock` with `uv add 'receipt==0.2.0'` (or the repo's equivalent flow) so the lock records real hashes. Cross-check the recorded hashes against the known-good PyPI artifacts:
  - whl sha256 `365b680ebec7e27de108cf990c807f8f45a8e9ccb40801a570c451afdd4aaf7a`
  - sdist sha256 `41ef973355cf0cc18fdb1f5f78e116a3b95b5b7b27f67f48cb4fcbdd1561448d`
  A mismatch is stop-and-report.
- Header comment near the import or the function: receipt is pinned by exact version and hash; any receipt upgrade re-runs this repo's signature tests at the new pin before the bump lands.

## Tests

- The existing suite is the oracle: every current test touching apply-manifest signatures must pass UNCHANGED. Run the relevant subset first (grep for the issue strings and the signature fixtures), then the fullest suite you can run locally; if the full suite has unrelated pre-existing failures or needs unavailable services, report exactly what you ran and what you skipped — do not paper over.
- Add `tests/test_receipt_sign_adoption.py`: with the repo's existing signing fixtures (see `tests/signing_broker_fixtures.py` / `tests/production_signing_fixture.py`), prove sign→verify round trip through the real `_sign_applied_encoding_manifest` + `_applied_encoding_manifest_signature_issue` path; a flipped signature byte, a flipped payload byte, a wrong key, and a wrong domain scope each yield exactly "has an invalid encoder apply manifest signature"; the envelope-level issue strings are unchanged for their malformed classes; and the delegation genuinely goes through receipt (e.g. monkeypatch `receipt.sign.verify_threshold` to observe the call — one test only, the rest exercise the real path).

## Constraints (binding — this repo runs under the agent-PR provenance regime, .github#39)

- Touch ONLY: `src/axiom_encode/cli.py` (the one function + import), `pyproject.toml`, `uv.lock`, the new test file, and a `changelog.d/` entry following the repo's changelog format (look at existing entries; reference the shared-package adoption and ops#3).
- Do NOT touch: `.axiom/` anything, any `.github/workflows/`, CODEOWNERS, waivers, corpus files, RuleSpec content, `signing_broker.py`, the eval-evidence signing path (that is a stated follow-up, not this PR).
- Do NOT change any pinned refs anywhere.
- Match the file's style; ruff clean if the repo uses it.
- Commit in logical units on `receipt-sign-adoption`. Do NOT push. Do not open a PR.
- Final answer: what you ran, the test output tail, and a self-audit — every place the new path could emit a different issue string, message, or exception type than before, and how a test pins it; anything skipped and why.

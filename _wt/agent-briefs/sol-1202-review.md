# Review task: merge-gate verdict on axiom-encode PR #1202 (receipt.sign adoption)

You are reviewing, not building. Worktree: `/Users/maxghenis/TheAxiomFoundation/axiom-encode/_wt/receipt-sign-adoption`, branch `receipt-sign-adoption` (head 964fe80a = the build commits + a just-resolved merge of origin/main). This is the final cross-family gate before merge: Claude (Fable) has reviewed and adjudicated; your verdict completes the two-party agreement rule.

Scope: `git diff origin/main...receipt-sign-adoption` (the PR's effective diff). The build itself went through a three-party review-fix cycle earlier; focus on what has NOT already been adjudicated:

1. **The merge resolution (commit 964fe80a).** origin/main advanced ~18 commits (proof-excerpt repair work, version to 0.2.1320) while the branch carried 0.2.1302. Resolution took main's version base, re-bumped to 0.2.1321, restored `receipt==0.3.0` to dependencies, and regenerated uv.lock. Confirm: no upstream change was clobbered (the three conflicted files were pyproject.toml, `src/axiom_encode/__init__.py`, uv.lock — verify each matches main plus exactly the intended deltas); the lock's receipt hashes are `31085b3c…` (sdist) and `89414893…` (whl); nothing else in the diff touches upstream's new proof-excerpt code.
2. **Adoption correctness on the merged tree**: `_applied_encoding_manifest_signature_issue` delegates its final crypto step to a 1-of-1 `receipt.sign.verify_threshold` keyring (`raw-sha256`, domain composed from `signing_broker`'s `_SIGNATURE_DOMAIN_PREFIX` + scope, `allow_legacy=False`); every envelope issue string and its ordering unchanged; `SignError` maps to the existing invalid-signature issue string; broker and signing side untouched; no workflow/toolchain/waiver/pinned-ref changes anywhere in the diff (the .github#39 constraint set).
3. **Tests**: `tests/test_receipt_sign_adoption.py` still pins the delegation call including `allow_legacy=False`; run it plus the version/provenance subset and the repo's focused check set yourself:
   - `uv run python -m pytest tests/test_receipt_sign_adoption.py --no-cov -q`
   - `uv run python -m pytest tests/test_cli.py -k "provenance or version_" --no-cov -q`
   - `uv run python -m pytest -q --no-cov tests/test_cli.py tests/test_rulespec_validation.py tests/test_evals.py -k "rulespec or EncoderPrompt"`
   Claude's runs on this head: 7/7, 16/16, 1067 passed. Reproduce or explain any divergence.

Verdict: APPROVE or REQUEST-CHANGES with numbered findings (file:line, severity, speculative labeled). Your final answer is the verdict; on APPROVE the PR merges (squash) once GitHub CI is green.

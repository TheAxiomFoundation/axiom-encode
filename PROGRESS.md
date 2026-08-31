# Issue 1557 progress

## State

- Branch: `feat/legacy-cleanup-receipts`
- Resumed on 2026-08-31 under the accepted one-transition `B -> H` contract; no
  checkout, reset, clean, or destructive recovery has been used.
- Second recovery inspection on 2026-08-31 found a clean worktree at
  `7ee389f6cf07ced35d4893203918b98b9f9ee76a`, nine linear commits above the
  cached protected base. The implementation is preserved unchanged while an
  independent contract/test/workflow audit runs.
- Starting implementation checkpoint: `9323860365d790c923ecc5fee8684be72fff56c8`.
- Exact locally cached protected base / `origin/main`: `f1bfe0a47ee7a9123d56e00a5c41edb6f272ea21` (tree `436af4b6e279d1d4dcd9f64d83442ca1e4de7751`; live fetch is blocked by workspace DNS).
- Salvage ref `refs/codex-salvage/feat-legacy-cleanup-receipts-20260830-212800-65267`
  resolves to `f3303be312bec9dec2b6a0bf9a66d1941192a09b`; all five surviving
  worktree files are byte-identical to that commit.
- The rejected tracked CLI and targeted-reencode workflow edits were removed by
  reverse-applying their exact worktree diffs; the salvage ref remains available
  for archaeology and no reset or clean was used.
- The pure contract, signing, exact executed validation, atomic durable transaction,
  cleanup-aware guard authorization, exact staging, dedicated workflow/transport,
  and documentation/non-interference layers are committed. They are not yet frozen
  for review: the independent review-fix cycle and complete local check matrix remain.

## Done

- Inspected branch, status, remotes, local/cached main identity, commit graph, staged diff, unstaged diff, and every untracked implementation file before editing.
- Confirmed the branch began with only the committed progress checkpoint beyond the protected base and no staged changes.
- Read `AGENTS.md`, `CLAUDE.md`, the committed progress checkpoint, all four surviving worktree changes, the WIP salvage commit, and the complete independent contract review at `20260830-193604-review-1557-contract/out.md`.
- Retried `git fetch origin main --prune`; it failed without changing refs because `github.com` could not be resolved.
- Retried a non-pruning live fetch on resume; DNS failed again. The browser-visible
  public repository page was reachable but did not expose an authoritative current
  main SHA, so `origin/main` remains a cached rather than live-verified base.
- On the second recovery inspection, attempted `git fetch --prune origin` before
  touching the worktree; DNS resolution failed again without moving any ref. Read
  all repository instructions (`AGENTS.md` and `CLAUDE.md`), all 216 lines of the
  independent contract review, the exact nine-commit branch history, and the full
  salvage/current file inventory.
- Confirmed `HEAD`, index, tracked worktree, and untracked inventory were clean;
  `HEAD^{tree}` is `49fbfba28a43f65e002958745334760b402633b0` and the salvage
  ref remains intact at `f3303be312bec9dec2b6a0bf9a66d1941192a09b`.
- Verified the dirty payload exactly matches the salvage tree before making any
  edits, and confirmed there are no staged changes or other untracked files.
- Classified the surviving topology against every Critical/High finding and Fast reject condition before implementation.
- Removed every surviving candidate-commit, receipt-child, applied-manifest-signing,
  and targeted-reencode cleanup change from the active worktree.
- Implemented the v1 cleanup-only receipt contract, semantic identity, strict and
  bounded canonical JSON, 1--64 exact primary/companion groups, cleanup signature
  payload domain, immutable `B` commit/tree/blob/mode/SHA-256 proof, protected
  toolchain/waiver pins, projected deletion-only tree, provenance inventory, and
  surviving-reference proof.
- Ran `UV_CACHE_DIR=/private/tmp/axiom-encode-uv-cache uv run pytest -q
  tests/test_legacy_cleanup.py`: 47 passed.
- Added a cleanup-only signing adapter that wraps the cleanup v1 inner domain in
  the existing protected `apply_ed25519` broker scope without reusing applied
  manifest serialization or verification. It prevalidates the exact unsigned
  schema, verifies broker output before persistence, and rejects cross-domain,
  cross-scope, wrong-key, malformed, and replayed signatures.
- Ran focused cleanup plus signing tests: 65 passed; focused Ruff and compileall
  also passed.
- Implemented verification-only corpus-key loading and nine genuinely executed
  projected-tree checks in an immutable private clone: repository tests/layout,
  waiver audit, remaining RuleSpec/companion/proof validation, money-atom proof,
  oracle coverage, and metadata-reference closure. Evidence is deterministic,
  bounded, pinned to the exact engine revision, and fails closed on stale,
  incomplete, empty, timed-out, or background-process results.
- Implemented the `cleanup-unmanifested-legacy` command as a receipt-first journaled
  transaction. It re-proves immutable `B`, toolchain pins, provenance, validation,
  clean checkout state, and the signed receipt under the transaction lock before
  installing exactly one receipt and deleting the exact primary/companion set.
  Recovery uses recorded preimages and rejects collisions, mode drift, reappearance,
  index flags, and concurrent mutation.
- Added cleanup-specific committed and worktree guards. The committed guard requires
  `H` to have sole parent `B` and proves the complete `B..H` change is exactly one
  canonical signed receipt addition plus its authorized deletions. Historical
  receipt append-only integrity, replay, overlap, orphan, rename, modified-target,
  mixed-change, and projected-tree failures are rejected.
- Added cleanup-aware exact staging without giving cleanup receipts applied-manifest
  ownership or credit. The staging path proves the live receipt, stages only exact
  authorized bytes/deletions, and verifies index modes, blobs, absences, and the
  resulting staged set.
- Hardened every cleanup Git read/write environment against repository-local hooks,
  filters, autocrlf, fsmonitor, untracked-cache, sparse-checkout, and alternate
  index/object/worktree injection. Reference scans use token-aware matching and
  immutable base blobs rather than raw substring or mutable-worktree evidence.
- Preserved the targeted signed-reencode workflow as a separate pipeline; its only
  change is passing the already-required engine checkout to the shared guard.
- Ran the focused cleanup contract/CLI/guard/validation suite: 148 passed in 64.81s.
- Ran related staging, toolchain, CI-parity, validation-waiver, and shared CLI
  regressions: 172 passed, 1,448 deselected in 26.60s.
- Documented the single atomic `B -> H` cleanup contract, immutable-base
  admission/evidence, signer/publisher separation, exact staging/publication,
  recovery, and deliberate non-interference boundaries in the README, dedicated
  contract guide, trusted-signing guide, and changelog.
- Added the JSON-only future-jurisdiction layout rule for the cleanup receipt
  namespace and production-API adversarial coverage proving that receipts create
  no applied-manifest ownership, retire/import authority, census encoder credit,
  run-log or Supabase credit, targeted-reencode behavior, or permission to rewrite
  historical provenance.
- Ran the documentation/layout/non-interference and related migration/run-log/
  Supabase regressions: 128 passed; focused Ruff, compileall, and diff checks passed.
- Advanced the package, exported `__version__`, and lockfile together from
  `0.2.1750` to `0.2.1751` for the cleanup feature.
- Added a dedicated manual-main cleanup workflow with protected signer and
  credentialed publisher in separate environments. Its only handoff is one
  canonical signed receipt plus one strict deletion inventory; the publisher
  reconstructs fresh exact `B`, materializes/stages only the authorized set,
  creates one sole-parent `H`, runs the committed guard, rechecks main, performs a
  non-force same-repository push, and opens only a topology-verified draft linked
  to issue 1557. Disabling draft publication skips the publisher job entirely.
- Added adversarial exact-transport coverage for mutated/lost/extra/symlinked
  artifacts, dirty or stale bases, mixed signer state, postcheck rollback, public
  trust-root parsing, and hostile sparse-checkout configuration, plus static
  workflow checks for permissions, credentials, action pins, shell syntax, exact
  staging, one-commit topology, non-force publication, and draft-only linkage.
- Ran actionlint on both changed workflows with no findings. Cleaned two inherited
  ShellCheck style findings in the targeted workflow without changing its
  behavior. Workflow/transport/guard tests: 68 passed; targeted workflow syntax,
  contract, and repair-preflight regressions: 5 passed.

## Next

- Finish the active independent review-fix cycle, then run the complete local check
  matrix until every required and relevant broad check is green.
- Re-verify exact commit messages and draft PR body, update this file after each
  coherent commit, and produce the frozen-review report with exact base/head,
  checks, and residual risks.
- Retry fetch/push and open a draft PR linked to issue 1557 when GitHub connectivity is available; do not merge.

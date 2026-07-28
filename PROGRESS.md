# PR 1322 round-2 confirmation progress

## State

The requested head passes the merge, version, CLI, and routing confirmation
checks. The full CLI file collected all 1,108 expected tests: 1,107 passed and
the sandbox explicitly skipped the set-user-ID mode-bit test because it blocks
that filesystem operation. Diff-scope audit and final independent review are
next; no PR branch, remote, or GitHub writes will be made.

## Done

- Inspected the primary checkout before editing. It is detached with unrelated
  untracked worktree directories, so it remains untouched.
- Located and read the round-1 ledger and final report under
  `.git/review-worktrees/pr-1322-blind`.
- Confirmed the requested head exists locally and the implementation worktree
  points to that exact commit.
- Attempted to refresh `origin/main` read-only; network name resolution is
  unavailable. The existing local `origin/main` is `45dd4a3e`, and that exact
  commit is the second parent of merge commit `e5a305af` beneath the requested
  head.
- Created this disposable worktree under `.git/review-worktrees/` from the
  exact requested head on a separate review-only branch.
- Committed the initial review ledger before running substantive checks.
- Confirmed `e5a305af` is a genuine merge commit, not a rebase: its first
  parent is the round-1 head `7b46fb1a`, and its second parent is the exact
  local `origin/main` tip `45dd4a3e`. The merge base is also `45dd4a3e`, and
  `origin/main...e9af588e` has counts `0 13`.
- Compared `pyproject.toml` and `uv.lock` directly with `origin/main`. The only
  differences are the local `axiom-encode` version lines; the
  `axiom-oracles` pin `678dd840...b399` and all other lock content are
  byte-for-byte inherited from main.
- Confirmed the terminal commit `e9af588e` modifies exactly
  `pyproject.toml`, `src/axiom_encode/__init__.py`, and `uv.lock`, moving the
  merged metadata to `0.2.1415` in the same commit.
- Independently parsed and imported the checkout: pyproject, package
  `__version__`, lock, runtime import, and installed distribution metadata all
  report `0.2.1415`.
- Ran
  `tests/test_cli.py::test_current_encoder_affecting_changes_are_behind_version_bump`;
  it passed.
- Ran full `tests/test_cli.py` twice. Both runs collected the expected 1,108
  tests and finished with 1,107 passed, one explicit environment skip, and no
  failures. The skipped node is
  `TestCmdEncode::test_apply_transaction_rejects_special_target_mode_bits`;
  this sandbox rejects `chmod 4755` with `Operation not permitted`, and the
  test is designed to skip when the filesystem cannot preserve that bit.
- Reran the exact round-1 six-case regression selection: four checkout-root
  source types plus ProgramSpec writer and checkout-root manifest placement.
  All six passed.
- Spot-checked jurisdiction-prefixed UK placement in the passing selection:
  both `statutes/26/36B.yaml` and already-prefixed
  `uk/statutes/26/36B.yaml` map to manifest path
  `uk/statutes/26/36B.yaml` at the checkout root.
- Spot-checked the passing ProgramSpec signing case: it writes
  `.axiom/encoding-manifests/programs/us-sc/snap/fy-2026.json`, cites
  `programs/us-sc/snap/fy-2026`, records applied path
  `programs/us-sc/snap/fy-2026.yaml`, and contains a nonempty signature.

## Next

- Audit the exact two-dot diff, complete an independent review cycle, and write
  and commit `WORKER-REPORT.md`.

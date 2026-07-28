VERDICT: APPROVE

# PR 1322 round-2 confirmation

Reviewed target:

- PR: `TheAxiomFoundation/axiom-encode#1322`
- requested head: `e9af588e7bb3f45482752d338b1005ad202d4a55`
- merge commit: `e5a305af5a88f4970798508a425b6ea5e3f305f9`
- reviewed `origin/main`: `45dd4a3e502099158953e8240247ac01ab76e8a6`
- disposable branch: `review/pr-1322-round-2-confirm`

No PR branch, remote, or GitHub writes were made.

## Confirmation

### Merge and pin preservation

`e5a305af` is a real two-parent merge, not a rebase. Its first parent is the
round-1 head `7b46fb1a`; its second parent is exactly `origin/main` at
`45dd4a3e`. That main ref was fetched locally at 10:14:23 EDT, immediately
before the merge, and is the merge base of the requested head. The comparison
counts are `0 13`, so main has no commits absent from the requested head.

The `origin/main..e9af588e` diffs in `pyproject.toml` and `uv.lock` change only
the local `axiom-encode` version. Main's current `axiom-oracles` pin,
`678dd840b4c64e54c63805b0183f05c0f769b399`, is identical in the target's
project dependency, lock requirement, and resolved lock source. This preserves
PR 1321's oracle work and all later main-side pin changes.

The shell could not refresh GitHub again during review because DNS was
unavailable; public web and in-app-browser fallbacks were also unavailable.
That prevented a second live freshness query but does not change the exact
local-ref/merge-parent identity above.

### Version repair and CLI gate

At the requested head, all five observed version surfaces agree:

- `pyproject.toml`: `0.2.1415`
- package `__version__`: `0.2.1415`
- `uv.lock`: `0.2.1415`
- direct runtime import: `0.2.1415`
- installed distribution metadata: `0.2.1415`

Terminal commit `e9af588e` changes exactly `pyproject.toml`,
`src/axiom_encode/__init__.py`, and `uv.lock`; the version bump and merged lock
therefore ride in the same commit.

The version-bump gate passed:

`tests/test_cli.py::test_current_encoder_affecting_changes_are_behind_version_bump`

Full `tests/test_cli.py` was run twice. Both runs collected the expected 1,108
tests and had no failures: **1,107 passed and 1 skipped**. The sole skip is
`TestCmdEncode::test_apply_transaction_rejects_special_target_mode_bits`; this
sandbox rejects `chmod 4755` with `Operation not permitted`, and the test
explicitly skips when the filesystem cannot preserve that bit. This is an
environment capability limitation, not a branch regression.

### Routing confirmation

The exact round-1 six-case regression selection passed **6/6**:

- all four checkout-root source types: `policies`, `programs`, `regulations`,
  and `statutes`
- checkout-root ProgramSpec manifest writer
- checkout-root manifest placement

The explicit jurisdiction-routing selection also passed **2/2**. The US
spot-check keeps
`us-sc/policies/dss/snap-policy-manual/page-159.yaml` at that checkout-relative
manifest path with citation
`us-sc:policies/dss/snap-policy-manual/page-159`.

The ProgramSpec signing spot-check writes
`.axiom/encoding-manifests/programs/us-sc/snap/fy-2026.json`, cites
`programs/us-sc/snap/fy-2026`, records
`programs/us-sc/snap/fy-2026.yaml`, and contains a nonempty signature.

### Diff scope

`git diff --check origin/main..e9af588e` passed. The exact two-dot diff contains
only these nine intended files:

- `PROGRESS.md`
- `changelog.d/1312-program-spec-manifest-root.fixed.md`
- `pyproject.toml`
- `src/axiom_encode/__init__.py`
- `src/axiom_encode/cli.py`
- `src/axiom_encode/repo_routing.py`
- `tests/test_cli.py`
- `tests/test_repo_routing.py`
- `uv.lock`

No manifest JSON or `.axiom` path is changed. `PROGRESS.md` is the expected
tracked repository ledger, not a review-session artifact.

## Independent review

An independent read-only scope audit found no actionable issue and returned
`APPROVE`. It independently confirmed the two-parent merge, pin equality,
version trio/runtime, terminal bump scope, nine-file two-dot diff, and clean
`git diff --check`.

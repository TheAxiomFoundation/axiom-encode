VERDICT: REQUEST-CHANGES

# PR 1322 blind adversarial review

Target verified:

- PR branch: `fix/1312-program-spec-manifest-root`
- requested head: `7b46fb1a80ff6600267eb2358f942b7ebec262bf`
- literal head parent: `abb37e209c89a3c6bdd23ebb01bdb7a43771969a`
- current local `origin/main`: `589e2b10091b8573164ff75d8bbd01960ecfc8f0`
- merge base: `f80bdcc58d591621aefac00803c43857c3c22db7`

No PR-branch, remote, or GitHub writes were made.

## Blocking findings

### 1. The branch adds eight deterministic exact-version parity failures

The exact-head full suite collected 6,091 tests and finished with **18 failed,
6,042 passed, and 31 skipped**. Clean current `origin/main`, run with its exact
locked `axiom-oracles` source, collected 6,086 tests and finished with **10
failed, 6,045 passed, and 31 skipped**.

The ten shared failures match by node ID and cause and are environmental:
non-root-owned Homebrew Git/system-provisioning expectations, one set-id
expectation, and the sandbox's denial of `/var/tmp` creation. The head adds
these eight repository-local failures:

- `tests/test_il_2026_oracle_registry.py::test_packaged_il_2026_runtime_pin_version_and_precedence_are_exact`
- `tests/test_in_2026_oracle_registry.py::test_packaged_in_2026_runtime_pin_version_and_precedence_are_exact`
- `tests/test_mn_2026_oracle_registry.py::test_packaged_mn_2026_runtime_pin_version_and_precedence_are_exact`
- `tests/test_pa_2026_oracle_registry.py::test_packaged_pa_2026_runtime_pin_version_and_precedence_are_exact`
- `tests/test_sc_2026_oracle_registry.py::test_packaged_sc_2026_runtime_pin_version_and_precedence_are_exact`
- `tests/test_rulespec_validation.py::test_packaged_dc_2026_registry_text_hash_runtime_and_precedence_are_exact`
- `tests/test_rulespec_validation.py::test_packaged_ca_2026_bhst_text_hash_runtime_and_precedence_are_exact`
- `tests/test_rulespec_validation.py::test_packaged_ny_2026_text_hash_runtime_pin_and_precedence_are_exact`

Each observes the PR's `0.2.1407` and compares it with a hardcoded
`0.2.1405`. A direct eight-test rerun produced **8 failed**. The stale values
are in six test files. This is not an environment failure.

The version trio itself is internally consistent at `0.2.1407`, its terminal
bump follows the repository's three-file convention, and seven generic
version-provenance tests pass. The omission is that the exact-version
runtime-pin consumers were not updated. Recent merged PR 1321 bumped
`0.2.1405` to `0.2.1406` and updated these consumers in the same change.

Required action: after updating from main and resolving the terminal version,
update every runtime-pin assertion to that version and rerun the full suite.

### 2. The requested head is stale and not merge-clean with current main

`git diff --name-status origin/main..7b46fb1a` contains 18 files, including
apparent deletion/reversion of PR 1321's Oklahoma changelog, registry test, and
mapping/test updates. The intended merge-base diff is nine files:
`PROGRESS.md`, the issue-1312 changelog, the version trio, `cli.py`,
`repo_routing.py`, and the two routing test files.

A three-way `git merge-tree` emits conflict markers in:

- `pyproject.toml`
- `src/axiom_encode/__init__.py`
- `uv.lock`

This also means the current two-dot hygiene requirement is not met. The
branch must be rebased or merged with current main, retaining main's oracle
pin/Oklahoma work and choosing the next terminal encoder version before the
gates can be trusted.

## Routing correctness

The implementation is principled at the routing layer; I found no routing
defect.

- Independent cases created real checkout-root `policies/`, `programs/`,
  `regulations/`, and `statutes/` outputs. All four selected the checkout as
  both content root and manifest root and preserved the checkout-relative
  output path.
- A mixed `rulespec-us` checkout containing root source directories still
  routed `us-sc/policies/dss/snap-policy-manual/page-159.yaml` through
  `rulespec-us/us-sc`, with manifest-relative path unchanged and citation
  `us-sc:policies/dss/snap-policy-manual/page-159`.
- Both UK issue-1078 forms were preserved:
  `rulespec-uk/uk` plus `statutes/...`, and `rulespec-uk` plus
  `uk/statutes/...`, both resolve to the checkout-root manifest mirror at
  `uk/statutes/...`.
- Checkout aliases, source-root symlinks, and wrong-country jurisdiction
  prefixes remained rejected.
- The new source-checkout admission flag defaults to false, is included in the
  routing-cache key, and does not alter the existing policy/composition
  predicates.

Direct call-site analysis found 17 production callers of
`_rulespec_apply_content_root`, three of `_rulespec_checkout_root`, two of
manifest placement, and five of anchor construction. Public atomic
`encode --apply` enters these helpers with an exact jurisdiction content root,
so established jurisdiction routing stays on the prior path.

## Manifest shape parity

The reference on rulespec-us main at commit `187d8d8e` is:

`.axiom/encoding-manifests/programs/us-az/snap/fy-2026.json`

It uses citation `programs/us-az/snap/fy-2026` and applied-file path
`programs/us-az/snap/fy-2026.yaml`. The head writer produced the SC analogue:

`.axiom/encoding-manifests/programs/us-sc/snap/fy-2026.json`

with citation `programs/us-sc/snap/fy-2026` and applied-file path
`programs/us-sc/snap/fy-2026.yaml`. There is no path or citation divergence.
The head unit writer correctly emits the current v5 contract; the independent
pinned `sign-applied-files` replay emitted v1 with exactly the Arizona
reference's field set.

## Fail-first evidence

The six-case regression selection comprises the four checkout-root source
parameters, the ProgramSpec manifest writer, and checkout-root manifest
placement.

- At `e8e59b6d`, the regression-bearing parent of fix commit `c9c5fd45`:
  **6 failed**.
- At requested head `7b46fb1a`: **6 passed**.
- At literal requested-head parent `abb37e20`: **6 passed**.

The literal parent cannot be fail-first: `abb37e20..7b46fb1a` changes only
`PROGRESS.md`; all implementation fixes are already in `abb37e20`. This is an
ancestry clarification, not a routing failure.

## SC end-to-end claim

The claim is independently verified.

- Located the dropped rulespec-us change at reflog commit `f93f556c`: add page
  159/remove page 369 from the SC SNAP ProgramSpec and move three
  `SNAP-SC-UTIL` worklist rows from `pending-local` to `merged`.
- Replayed the exact blobs on disposable rulespec-us base `187d8d8e`.
- The pinned bridge at `3869d66d` failed first at
  `path.relative_to(manifest_root)`, trying to rebase checkout-root
  `programs/...` against `<scratch>/us`.
- The target routing/citation seam selected the checkout root and bare
  ProgramSpec citation. Backported into the disposable pinned signer, it wrote
  one manifest and reported that the guard passed.
- A separate external
  `guard-generated --base-ref 187d8d8e --head-ref HEAD --roots programs`
  exited 0: `All changed RuleSpec files have encoder apply manifests.`
- The final scratch rulespec diff contains only the ProgramSpec, three worklist
  status changes, and the new manifest.
- `/Users/maxghenis/TheAxiomFoundation/wt-snap-sc` remained at `8da79dd4` with
  only its two pre-existing untracked reports.

## Existing-manifest blast radius

- rulespec-us main has 886 checkout-root manifests.
- All 429 active jurisdiction-prefixed protected manifests resolved at the
  head with **0 routing errors and 0 manifest-path changes**.
- Only 13 referenced source files still exist at a checkout-root source path;
  all are ProgramSpecs. All 13 retain their manifest and applied-file paths.
- Eleven of those 13 already use the new bare citation convention. Two
  historical outliers would normalize on re-sign:
  `us-ca:programs/snap/fy-2026` and
  `us:programs/us-nh/income-tax/fy-2026`. Because the old root ProgramSpec
  route crashed, these are canonicalization migrations, not silent changes to
  a previously successful current route.
- The other 426 root-path legacy manifests refer to atomic files since moved
  beneath `us/`; that predates this PR.
- rulespec-uk main currently has zero encoding manifests.

I found no existing successful US/UK manifest route that silently changes path
or citation because of this PR.

## Other gates and hygiene

- Focused routing/writer/repository-routing/provenance selection: **64 passed**.
- Full Ruff: passed.
- `python -m compileall -q src/axiom_encode scripts`: passed.
- Merge-base `git diff --check`: passed.
- The intended merge-base diff contains only the nine expected files, including
  the load-bearing `PROGRESS.md`.

## Environment and sandbox disclosures

- `uv run` could not initialize the default cache at `~/.cache/uv` (`EPERM`).
  Tests and lint used the populated offline issue-1312 environment with
  explicit target `PYTHONPATH`.
- The prompt's expected approximately 35 environmental failures did not
  reproduce. The exact clean-main baseline available here is 10.
- GitNexus local analysis completed, but publication to
  `~/.gitnexus/registry.json` was denied (`EPERM`). Direct local-backend
  context/impact queries worked; oversized `cli.py` was omitted by the index,
  so its callers were also audited with AST and direct search.
- `apply_patch` was denied for `/private/tmp`; routing probes ran inline, and
  the SC scratch work moved to the repository's ignored `_wt/` area.
- Directly importing the target CLI in the isolated SC scratch lacked the
  `receipt` dependency. The exact target function ASTs and target routing
  module were executed, then the equivalent seam was applied to the pinned
  signer that still exposes `sign-applied-files`.
- No network, remote, PR-branch, or GitHub writes occurred.

## Review-cycle outcome

Independent routing/blast-radius, gate-comparison, SC end-to-end, and final
challenge reviews all agree on the evidence above. The routing fix itself has
no actionable finding, but the merge conflicts and eight branch-only failures
remain actionable. They cannot be repaired in this review because the
assignment forbids PR-branch writes. The PR should not be marked ready until
they are fixed and the required gates/cycle are rerun.

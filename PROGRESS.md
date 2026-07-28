# PR 1322 blind adversarial review progress

## State

Review in progress on throwaway branch `review/pr-1322-blind`. Routing and the
SC end-to-end claim are passing so far, but the requested head is not currently
merge-clean and has branch-only version-provenance failures. No PR branch,
remote, or GitHub writes are permitted.

## Done

- Read the repository instructions and the GitNexus PR-review workflow.
- Verified the requested branch and exact target/parent commit identities.
- Created this disposable worktree under `.git/review-worktrees/` from the
  exact requested head, on a separate throwaway branch.
- Established the correctness, parity, fail-first, end-to-end, blast-radius,
  hygiene, provenance, and gate plan.
- Confirmed current `origin/main` is `589e2b10`, while the PR merge base is
  `f80bdcc5`; the literal two-dot diff has 18 files because main advanced, while
  the intended merge-base diff has nine files.
- Confirmed the requested head's literal parent is `abb37e20` and differs only
  in `PROGRESS.md`; the actual first implementation commit is `c9c5fd45`, whose
  regression-bearing parent is `e8e59b6d`.
- Ran the six-case fail-first selection at `e8e59b6d`: 6 failed. Ran the same
  selection at the requested head: 6 passed.
- Inspected the head-produced ProgramSpec manifest. It is placed at
  `.axiom/encoding-manifests/programs/us-sc/snap/fy-2026.json`, cites
  `programs/us-sc/snap/fy-2026`, and lists
  `programs/us-sc/snap/fy-2026.yaml`.
- Compared those shape fields with rulespec-us main's Arizona reference at
  `187d8d8e`; citation and applied-file path conventions match, while the head
  correctly emits the current v5 schema rather than the reference's legacy v1.
- Independently reconstructed the exact dropped SC ProgramSpec/worklist change
  from `f93f556c` on a disposable rulespec-us base, reproduced the pinned-bridge
  crash, applied the target-equivalent routing/citation seam, signed one
  checkout-root ProgramSpec manifest, and passed external
  `guard-generated --roots programs`.
- Verified `/Users/maxghenis/TheAxiomFoundation/wt-snap-sc` stayed at
  `8da79dd4` with only its two pre-existing untracked report files.
- Found that merging current `origin/main` into the target conflicts in
  `pyproject.toml`, `src/axiom_encode/__init__.py`, and `uv.lock`.
- Found branch-only version-provenance assertions still pinned to `0.2.1405`
  despite the head's `0.2.1407` bump.
- Attempted the required GitNexus index. Local indexing completed, but registry
  publication to `~/.gitnexus/registry.json` was sandbox-blocked with `EPERM`;
  direct local-backend queries worked. The graph indexed 752 files, 6,460
  symbols, and 300 processes, but omitted oversized `cli.py`, so its critical
  helper impacts were established with AST and direct call-site analysis.
- Ran independent synthetic routing cases using real directories for all four
  requested checkout roots: `policies/`, `programs/`, `regulations/`, and
  `statutes/`. Every case selected the checkout as content and manifest root and
  preserved the relative path. Separate mixed-layout probes preserved
  `us-sc/policies/...` and `uk/statutes/...` jurisdiction routing, and both
  issue-1078 UK caller forms placed `uk/statutes/...` under the checkout
  manifest tree.
- Independently confirmed checkout aliases, source-root symlinks, and
  wrong-country jurisdiction prefixes remain rejected.
- Audited production call sites: 17 calls to `_rulespec_apply_content_root`,
  three to `_rulespec_checkout_root`, two to manifest placement, and five to
  anchor construction. Existing public atomic apply enters through an exact
  jurisdiction content root, preserving prior routing.
- Scanned rulespec-us main's 886 checkout-root manifests. All 429 active
  jurisdiction-prefixed manifests produced zero route errors and zero path
  changes. All 13 extant root-level sources are ProgramSpecs and retain their
  manifest/applied-file paths; 11 already have bare citations, while two
  historical noncanonical citations would normalize when re-signed.
- Confirmed rulespec-uk main currently has zero encoding manifests.
- Ran full Ruff, `compileall`, and merge-base `git diff --check`: all passed.
- Ran the exact head full suite in an isolated worktree: 18 failed, 6,042
  passed, 31 skipped (6,091 collected). Clean current-main with its exact
  oracle lock produced 10 failed, 6,045 passed, 31 skipped (6,086 collected).
- Matched all 10 shared failures to system/sandbox causes (non-root-owned
  Homebrew Git, set-id expectation, and `/var/tmp` denial). The head's other
  eight failures are unique, deterministic `0.2.1407 != 0.2.1405`
  provenance assertions across IL, IN, MN, PA, SC, DC, CA, and NY tests.
- The prompt's expected approximately 35 environment failures did not
  reproduce in the available offline environment; the exact observed clean
  baseline is 10.
- Confirmed the terminal three-file bump itself follows convention, but recent
  merged PR 1321 also updated every hardcoded runtime-pin test; this PR did not.

## Next

- Run an independent review-fix cycle over the evidence and findings.
- Write and commit `WORKER-REPORT.md` with the final verdict and evidence.

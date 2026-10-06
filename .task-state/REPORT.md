# axiom-encode #1703 handoff

Workspace: `/Users/maxghenis/.subfleet/worktrees/20260927-153130-producer-axiom-encode`

Patch: `/Users/maxghenis/.subfleet/worktrees/20260927-153130-producer-axiom-encode/.task-state/axiom-encode-1703.patch`

Baseline: freshly fetched `origin/main`, `aa731bf832e0f36d6ba0c43e5c136a8415f41092`.
The assigned checkout started clean at `427edd81aec1`. Its Git metadata is outside
the writable sandbox, so normal fetch was rejected. An isolated bare repository
at `.task-state/upstream.git` fetched main, and tracked workspace files were updated
to that baseline. The original checkout's HEAD and refs were not changed. Apply
the supplied patch to the fetched baseline, rather than using a diff from the
original checkout's stale HEAD or origin/main. No commits, pushes, or PRs.

## Changes and invariants

- `src/axiom_encode/companion_relations.py:22`: resolves binary relation placement
  from actual compiled `expr`/`where` executable nodes, including versions,
  aggregates, memberships, and relation derivations. Executable usage wins over
  contradictory declarations; only unused typed relations use declared positions.
- `src/axiom_encode/cli.py:5394`: passes the compiled program into request
  construction, labels typed owners and related rows with their real kinds,
  resolves exact canonical names, and gives different relation lists distinct IDs.
- Every emitted typed tuple places each known ID in the slot expecting its labelled
  kind. Ambiguous/conflicting executable orientations produce an explicit error.
- Entire requests for old all-untyped artifacts retain legacy related-first
  tuples, IDs, generic labels, and aliases. No lenient binding opt-out is sent.
- Hypothesis is a dev dependency; `uv.lock` and a changelog fragment are included.

## Validation

- `.venv/bin/python -m pytest tests/test_companion_relations.py --no-cov -q`:
  **25 passed**. Includes 150 generated household/tax-unit cases with 0–40 members,
  both declared/executable orders, and exact legacy request JSON equality.
- `.venv/bin/pytest -q --no-cov tests/test_companion_relations_engine.py`:
  **4 passed**. Both executable orders, with matching and contradictory serialized
  declarations. All compilation and evaluation use the supplied real engine.
- Before/after evidence: legacy requests fail strict binding; the supplied main
  binary accepts the owner-first mismatch and returns 0. Correct requests return
  **1500**, match an explicit reference request, and report strict binding with no
  request/CLI opt-out. This is a self-contained engineering fixture, not a CDCC
  legal encoding.
- `uv run ruff check pyproject.toml src/axiom_encode scripts tests`: passed.
- `.venv/bin/ruff check pyproject.toml src/axiom_encode scripts tests`: passed on
  the final source.
- `.venv/bin/ruff format --check src/ tests/`: passed (190 files).
- `.venv/bin/python -m compileall -q src/axiom_encode scripts`: passed.
- `.venv/bin/python -m towncrier build --draft --version 0.0.0`: passed.
- Full-suite command:
  `COVERAGE_CORE=sysmon PYTHONUNBUFFERED=1 .venv/bin/pytest tests/ -n 8 --dist worksteal --tb=short`.
  Result: **incomplete, exit 130** after hanging on the final two of 15,760
  collected tests. The progress log reported **15,377 passed, 83 skipped,
  247 failed, 51 setup errors**; these are partial progress counts, not a final
  pytest success summary. Ctrl-C released the failure report; a further interrupt
  ended stalled teardown while importing Hypothesis. Coverage did not finalize.
  Log: `.task-state/pytest-full-parallel.log`.
  Neither new companion test module appears in the failure/error summary.
  Failure traceback groups: 202 mode-0644/0600 mismatches, 10 system-Git ownership
  failures, 11 timing-bound assertions, 4 subprocess timeouts, 6 canonical-root
  fixture errors, 1 HEAD/runtime version mismatch, 1 sandbox /var/tmp permission
  denial, 4 unsupported `required_mode` arguments, and 8 other failures. All
  51 setup errors are signing-supervisor Go-build failures whose captured stderr
  was not included in pytest's report. These results do not establish a clean
  full suite, and not all failures have been attributed to the environment.
  Further details: `.task-state/full-suite-analysis.md`.
  Native coverage monitoring and parallel workers retained the full CI test set
  and coverage configuration while reducing runtime on a host with load averages
  over 100. `pytest-xdist` was installed only in the local test environment; it is
  not a project dependency. Earlier single-worker `uv run pytest` and
  `.venv/bin/pytest tests/` attempts were interrupted during slow collection or
  execution, including one before the final review fixes were loaded.
- Isolated Git provenance check:
  `.venv/bin/pytest --no-cov --tb=short -q tests/test_cli.py -k test_current_encoder_affecting_changes_are_behind_version_bump`:
  **1 failed, 1388 deselected**. It reads version 0.2.1201 from the immutable
  original Git HEAD and compares it to the fresh-main runtime version 0.2.2053.
  This checkout-metadata limitation is unrelated to the companion changes; the
  task explicitly forbids changing published version numbers. Evidence:
  `.task-state/provenance-test.log`.
- Independent review/fix/review cycle: no remaining actionable findings.

The UV commands used `UV_CACHE_DIR="$PWD/.task-state/uv-cache"` to keep cache writes
in this workspace. Dependencies were installed with
`uv sync --extra dev --python 3.13`; Python is 3.13.9.

## Corpus expectations and limits

The inspected CDCC/CTC companion expectations do not depend on legacy tuple order
and need no rewriting. In the local corpus, `us/statutes/26/21.test.yaml:109` expects CDCC 1500;
the provider exception expects 0 at line 146. CTC's composed after-advance credit
is 2050 at `us/statutes/26/24.test.yaml:89`; subsection h's maximum is 3200 at
`us/statutes/26/24/h.test.yaml:92`.

rulespec-us receives this fix only when its `axiom_encode_ref` commit pin is moved
in `.axiom/workflow-toolchain.toml`. The locally available corpus is older and has
the pin in `.axiom/toolchain.toml` instead.

Optional actual-corpus replay command:
`PYTHONPATH=src .venv/bin/python .task-state/corpus-replay/replay.py`.
All three local files (15 cases total) were blocked before compilation by fresh
main's existing checkout admission (`UnsafeRulespecContextPath`, rejection code
`atomic-root-at-checkout`). No corpus files or published outputs were changed.
Details: `.task-state/corpus-replay/report.json`.

## Patch generation

New files were added with intent-to-add only in the isolated metadata. The patch
was generated with that metadata and the workspace as its work tree, using
`git diff origin/main` restricted to the seven changed task files. It was checked
with `git apply --check --reverse .task-state/axiom-encode-1703.patch` against the
resulting workspace. This leaves the assigned checkout changes uncommitted.

Final `git diff --check origin/main` and reverse-application patch check both passed
using the isolated Git metadata.

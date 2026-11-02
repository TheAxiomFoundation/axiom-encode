# Reproduce the final-byte checks

`final-checks-summary.json` binds the final production modules, both generated
regression files and changelog. `final-check-output.txt` publishes the observed
logs with cache, basetemp and checkout prefixes replaced by portable values.
Trailing terminal padding is removed from the published logs. The executed
and portable runner hashes are recorded separately. The portable
runner substitutes those filesystem values and also checks the changelog hash;
it preserves every pytest argument, Hypothesis setting and explicit example.

The current commands passed 812 tests: 295 original generated nodes, 14 new
safety nodes, 384 earlier regressions, 10 selected complete-source checks,
44 selected numeric checks, 64 upstream controls and one enabled locked-set
check. Four disjoint generated shards preserve all 295 nodes exactly once.
The corrected 14-node safety file failed at its operative-value assertions
against the exact reviewed head. Both profiles retain currency table cells;
size-cell parsing is covered only for the profile supported by current main.
The inherited legacy size-cell limitation is outside this change.

Use a locked GIL CPython 3.13 environment. Export `UV_CACHE_DIR` to the fleet's
shared cache, `PYTEST_BASETEMP_PREFIX` to a unique outside-workspace temporary
prefix, `UV_EXECUTABLE` to the real uv executable and `UV_CACHE_LAUNCHER_DIR`
to a scratch directory for a transparent launcher. The launcher adds the
shared cache argument because the locked-set test strips UV environment
variables before invoking uv. The runner removes each pytest basetemp.

Restore these inert snapshots, from the repository root:

```sh
mkdir -p .hub-scratch/r6 .hub-scratch/r6-resumed/tests
mkdir -p .hub-scratch/resume-final/tests/final .hub-scratch/r6/main-integration
```

| Snapshot | Restore as |
| --- | --- |
| `final-run-focused-checks.py.txt` | `.hub-scratch/resume-final/tests/final/run_checks.py` |
| `upstream-manual-controls.py.txt` | `.hub-scratch/r6/main-integration/test_upstream_manual_controls.py` |
| `old-complete-plugin.py.txt` | `.hub-scratch/r6-resumed/tests/old_complete_plugin.py` |

The upstream test snapshot comes from
`c2ee166fd1813392022fe5cf4c1e08ae1f09aba0:tests/test_source_completeness_manual_controls.py`.
Only its fixture-root expression is adapted to the repository root. The new
summary binds the published snapshot rather than assuming scratch files exist.

Create the transparent launcher after setting the four environment variables:

```sh
.venv/bin/python - <<'PY_LAUNCHER'
import os
from pathlib import Path
import shlex
launcher = Path(os.environ["UV_CACHE_LAUNCHER_DIR"]) / "uv"
launcher.parent.mkdir(parents=True, exist_ok=True)
launcher.write_text(
    "#!/bin/sh\nexec " + shlex.quote(os.environ["UV_EXECUTABLE"])
    + " --cache-dir " + shlex.quote(os.environ["UV_CACHE_DIR"]) + ' "$@"\n'
)
launcher.chmod(0o755)
PY_LAUNCHER
```

Restore the exact reviewed modules without modifying the checkout:

```sh
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/source_completeness.py > .hub-scratch/r6/source_completeness_old.py
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/manual_locator_metadata.py > .hub-scratch/r6/manual_locator_metadata_old.py
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/validator_pipeline.py > .hub-scratch/r6/pre_collision_validator.py
```

Verify their hashes against `review_head_runtime_sha256`. Collect the unchanged
295-node suite, remove only that collection basetemp, then run the negative
and current modes:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
HYPOTHESIS_STORAGE_DIRECTORY=.hub-scratch/resume-final/tests/final/hypothesis-current .venv/bin/pytest -q -o addopts='' --no-cov -p no:cacheprovider --basetemp="${PYTEST_BASETEMP_PREFIX}collection" --collect-only tests/test_source_completeness_manual_mask_fail_safe.py > .hub-scratch/r6-resumed/tests/collection.log
.venv/bin/python -c 'import os, shutil; shutil.rmtree(os.environ["PYTEST_BASETEMP_PREFIX"] + "collection")'
.venv/bin/python .hub-scratch/resume-final/tests/final/run_checks.py old-safety
.venv/bin/python .hub-scratch/resume-final/tests/final/run_checks.py current
```

The negative pytest child is expected to exit 1 with 14 failures; its wrapper
accepts that expected outcome. Every current pytest child must exit 0. The
runner uses at most four children, with all five numerical thread caps set to
one in each child. No assertions, explicit examples, generation settings or
numeric profiles are reduced. The exact executed records remain in scratch;
these published records normalize filesystem values and trailing terminal
padding only.

The previous `checks-summary.json`, `check-output.txt` and reproduction files
are historical evidence for earlier bytes. Their counts are excluded from the
812 final-byte passes. The initial unsupported legacy size-cell expectations
are excluded from both the final passes and the corrected negative control.
No full local matrix, Rust build, remote CI result, signing event or production
outcome is claimed. The hub must bind PR links to the commit containing these
validated bytes, and green exact-head CI remains the landing gate.

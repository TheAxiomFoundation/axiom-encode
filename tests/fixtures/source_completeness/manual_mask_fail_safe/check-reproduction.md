# Reproduce the completed checks

`checks-summary.json` records the exact pytest node arguments, observed exits,
summaries and SHA256s. `check-output.txt` contains the observed summaries.
The final source, metadata, extractor, generated tests and changelog hashes
must match before these results apply. The four generated shards cover all
295 collected nodes exactly once, with unchanged Hypothesis settings and
explicit examples. Focused checks passed 384 earlier regressions, 10 selected
complete-source tests, 44 selected numeric tests and 64 upstream controls.
The enabled locked-environment check passed separately.

Use the repository's locked GIL CPython 3.13 environment and dev extra. Set
`UV_CACHE_DIR` to the shared cache and `PYTEST_BASETEMP_PREFIX` to a unique
outside-workspace temporary prefix. Numeric test subprocesses use one thread
each; the four generated shards therefore use at most four such workers.
Set `OMP_WAIT_POLICY=PASSIVE`. Remove each generated basetemp afterward.
The portable runner makes only cache/basetemp environment substitutions;
its executed and published hashes are distinguished in the summary.

Restore these inert script snapshots to the corresponding scratch paths:

| Published snapshot | Restore as |
| --- | --- |
| `run-focused-checks.py.txt` | `.hub-scratch/r6-resumed/tests/run_checks.py` |
| `upstream-manual-controls.py.txt` | `.hub-scratch/r6/main-integration/test_upstream_manual_controls.py` |
| `old-complete-plugin.py.txt` | `.hub-scratch/r6-resumed/tests/old_complete_plugin.py` |
| `old-mask-plugin.py.txt` | `.hub-scratch/r6/old_mask_plugin.py` |

The upstream controls originate from
`c2ee166fd1813392022fe5cf4c1e08ae1f09aba0:tests/test_source_completeness_manual_controls.py`.
Only its fixture-root assignment was adapted to the current repository root;
assertions, parametrizations and imports are unchanged. Its executed snapshot
is published rather than assuming an untracked scratch test will exist.

For the current tests, collect the nodes and run both modes from the repository
root after restoring the runner and upstream controls:

```sh
: "${PYTEST_BASETEMP_PREFIX:?Set a unique outside-workspace temporary prefix}"
HYPOTHESIS_STORAGE_DIRECTORY=.hub-scratch/r6-resumed/tests/hypothesis-current .venv/bin/pytest -q -o addopts='' --no-cov -p no:cacheprovider --basetemp="${PYTEST_BASETEMP_PREFIX}collection" --collect-only tests/test_source_completeness_manual_mask_fail_safe.py > .hub-scratch/r6-resumed/tests/collection.log
.venv/bin/python .hub-scratch/r6-resumed/tests/run_checks.py generated
.venv/bin/python .hub-scratch/r6-resumed/tests/run_checks.py dependent
```

To reproduce the negative controls, first restore the three exact reviewed
modules. These commands read existing Git objects without changing the checkout:

```sh
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/source_completeness.py > .hub-scratch/r6/source_completeness_old.py
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/manual_locator_metadata.py > .hub-scratch/r6/manual_locator_metadata_old.py
git show bb068ca8444ca9445f93b4292788285e770c1e75:src/axiom_encode/harness/validator_pipeline.py > .hub-scratch/r6/pre_collision_validator.py
```

Verify all three against `exact_old_head_reproduction.runtime_review_head_sha256`.
Set `PYTEST_BASETEMP` to a unique outside-workspace temporary directory for
this negative run, independently of the runner's `PYTEST_BASETEMP_PREFIX`.
With numeric thread caps set to four, restore the exact plugin and execute:

```sh
: "${PYTEST_BASETEMP:?Set a unique outside-workspace temporary directory}"
HYPOTHESIS_STORAGE_DIRECTORY=.hub-scratch/r6-resumed/tests/hypothesis-old-exact PYTHONPATH=.hub-scratch/r6-resumed/tests:src .venv/bin/pytest -q -o addopts='' --no-cov -p no:cacheprovider --basetemp="$PYTEST_BASETEMP" -p old_complete_plugin tests/test_source_completeness_manual_mask_fail_safe.py::test_round_five_operative_values_are_demanded 'tests/test_source_completeness_manual_mask_fail_safe.py::test_generated_monetary_caption_and_column_qualifiers_keep_every_value[parentheses-Fees in dollars]' 'tests/test_source_completeness_manual_mask_fail_safe.py::test_generated_referenced_stamp_date_survives_every_source_shape[preceding-The month shown below determines eligibility.]'
```

This command is expected to exit 1 with 14 operative-value assertion failures:
12 original examples plus both generated properties. The separate isolated-mask
run used the same old source/metadata with the current extractor; it also failed
all 12 original examples, isolating the source-mask regressions.
Remove only the negative run's temporary directory when the command finishes.

Earlier interrupted or skipped attempts are excluded from the successful
counts. These reports establish the recorded tests on the pinned file bytes;
they do not claim a complete local matrix, remote CI result, signing event or
production outcome. After the hub commits the validated bytes, PR evidence
links must identify that resulting head. Upstream history merging and its
version-literal conflicts remain a separate hub landing step.

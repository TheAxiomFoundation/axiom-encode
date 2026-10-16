# Reproduce the final raw-manual differential

Use the source, metadata and extractor bytes named in `corpus-summary.json`,
the locked GIL CPython 3.13 environment and development dependencies. The
recorded baseline is `4f3547461300bfa70569f57c83728b2ad25d9d0b` and corpus
commit is `8f7d60aaced28ee4252b9237f9d6e02360dc34bc`. The full raw inputs total
238,184,678 bytes; no source prose is duplicated in these evidence artifacts.

Main advanced during execution to
`52c76d75849a628bbfd62b31795fd06e6bf9fdd1`. The complete Python package tree,
all 99 Git blobs, `pyproject.toml` and `uv.lock` are byte-identical to the
recorded baseline. `corpus-main-forward-equivalence.json` records both package
tree IDs, every blob identity and lock hashes; its `.py.txt` snapshot reproduces
that check. Numeric results transfer through exact bytes. The emitted execution
summary retains its actual baseline rather than being relabeled.

The captured runner checks that `origin/main` identifies the recorded baseline.
For future replay after upstream advances, use a disposable checkout containing
the published candidate bytes and pin its local remote-tracking reference with
`git update-ref refs/remotes/origin/main 4f3547461300bfa70569f57c83728b2ad25d9d0b`.
This is a replay checkout operation and is not a remote push.

Restore these inert snapshots with exactly the indicated repository-relative
layout; the runner derives paths from that layout.

```sh
mkdir -p .hub-scratch/resume-final .hub-scratch/r6/corpus
```

| Published snapshot | Restore as |
| --- | --- |
| `corpus-runner.py.txt` | `.hub-scratch/resume-final/final_differential.py` |
| `corpus-ast-graph.py.txt` | `.hub-scratch/resume-final/baseline_ast_proof.py` |
| `corpus-guard-utility.py.txt` | `.hub-scratch/r6/corpus/guard_observer.py` |
| `corpus-probe-generator.py.txt` | `.hub-scratch/r6/corpus/differential_guard_observer_final.py` |
| `corpus-differential-utility.py.txt` | `.hub-scratch/r6/corpus/differential.py` |
| `corpus-verifier.py.txt` | `.hub-scratch/resume-final/verify_final_differential.py` |
| `corpus-independent-proof-check.py.txt` | `.hub-scratch/resume-final/review-corpus-proof-check.py` |
| `corpus-independent-observer-smoke.py.txt` | `.hub-scratch/resume-final/review-observer-smoke.py` |
| `corpus-independent-completion-check.py.txt` | `.hub-scratch/resume-final/review-corpus-completion-check.py` |
| `corpus-main-forward-equivalence.py.txt` | `.hub-scratch/resume-final/corpus_main_forward_equivalence.py` |

The probe-generator and differential-utility snapshots contain historical runners
and constants. The final runner calls only `numeric_proof` from the former and
its declaration helpers from the latter; it does not execute either historical
runner. Preserve their bytes. The final initializer restores the authenticated
baseline package before executable imports and asserts both module bindings.

Create `.hub-scratch/resume-final/current-main-package`, then reconstruct the
exact archived baseline package:

```sh
mkdir -p .hub-scratch/resume-final/current-main-package
git archive 4f3547461300bfa70569f57c83728b2ad25d9d0b src/axiom_encode | tar -x -C .hub-scratch/resume-final/current-main-package
```

`corpus-provenance.json` records the archive SHA256 and every Python file hash.
`corpus-independent-proof.json` separately records all 99 Git blob identities.
The replay candidate package is copied from the checkout by the final runner.
Verify the checkout's three candidate hashes before running.

`corpus-input-manifest.json` contains a `files` array with immutable URLs, byte
counts, SHA256s, Git blob SHA1s and row counts. Authenticate each downloaded
file and put it under `.hub-scratch/r6/corpus/input/` followed by its recorded
`file` path. Reconstruct the runner's expected plain-list manifest in original
order; each object contains only `file`, `sha256`, `rows`, in that order:

```python
import json
from pathlib import Path

published_manifest = Path("tests/fixtures/source_completeness/manual_mask_fail_safe/corpus-input-manifest.json")
scratch_manifest = Path(".hub-scratch/r6/corpus/corpus-manifest.json")
scratch_manifest.parent.mkdir(parents=True, exist_ok=True)
manifest = json.loads(published_manifest.read_text())["files"]
entries = [{key: row[key] for key in ("file", "sha256", "rows")} for row in manifest]
scratch_manifest.write_text(json.dumps(entries, indent=2) + "\n")
```

The reconstructed manifest's SHA256 must equal `input_manifest_sha256` in the
summary. The run authenticates every input file again before processing it.
There must be 231 input files and exactly 68,848 raw rows.

From the repository root, with the output directory initially absent:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
PYTHONPATH=.hub-scratch/resume-final/current-main-package/src .venv/bin/python -u .hub-scratch/resume-final/final_differential.py --output .hub-scratch/resume-final/final-current-main-verified --workers 4
.venv/bin/python .hub-scratch/resume-final/verify_final_differential.py --output .hub-scratch/resume-final/final-current-main-verified
```

The first command runs an eleven-case observer smoke, then an all-row fresh
comparison with bounded within-run memoization. Pure recall, helper, probe,
observer and extraction calls use LRU caches of at most 256 input keys, in
addition to the numeric tokenizer's existing cache. A cache hit resolves the
same deterministic call within this execution; it does not execute that function
again. No previous-run or incomplete-run rows or inventories are reused. The
captured runner docstring's per-row execution wording must be read with this
memoization qualification; it is not a count of unwrapped function executions.
The run verifies baseline package and
candidate source/metadata/extractor hashes at completion. Equal recalled bytes
use the documented extraction equivalence proofs; different recalled bytes run
both actual extractors in both profiles. The verifier independently reconciles
all input-row hashes, complete unique row keys, deltas, proof modes and totals.

For the supplemental review, run the independent observer snapshot first and
then the independent proof checker. The first creates a private runtime-binding
JSON under scratch; the checker authenticates those paths before writing its
portable report. The published portable JSON contains case hashes and outcomes,
without the private paths. These two numeric commands use the same six thread
caps as the full run.

```sh
.venv/bin/python .hub-scratch/resume-final/review-observer-smoke.py
.venv/bin/python .hub-scratch/resume-final/review-corpus-proof-check.py
.venv/bin/python .hub-scratch/resume-final/review-corpus-completion-check.py
```

The independent completion checker reconciles the full ledgers separately from
the main verifier. `corpus-independent-completion.json` records that final result
and binds the same emitted summary hash. The construction report records its
earlier construction-only review; the completion report supplies the later
full-output outcome.

Repeated executions can reorder streamed records and change timing hashes.
Compare authenticated row keys, numeric deltas, proof counts, pinned source/input
hashes and completion guards. The published gzip decompresses to the exact
recorded raw-row ledger hash; `corpus-artifacts.json` authenticates every
published artifact.

This differential covers raw manual bodies. Generated public-gate tests provide
separate raw/composed coverage. It does not establish a complete composed-book
differential, remote CI result, signing event or production outcome.

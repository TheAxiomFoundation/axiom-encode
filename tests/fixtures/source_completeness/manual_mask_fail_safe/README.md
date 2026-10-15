# Manual-mask validation evidence

`final-checks-summary.json` binds the final source, locator registry, numeric
extractor, both generated regression files, and changelog to their SHA256s.
The final focused checks passed **812 tests**, including all 295 original
generated cases, 14 additional safety cases, and the earlier regression and
replay files. `final-check-output.txt` preserves normalized observed logs;
`final-check-reproduction.md` explains the exact selections, original
Hypothesis settings, transparent shared-cache launcher, and expected negative
control. All 14 corrected safety cases fail on the exact reviewed original
source, registry, and extractor. The earlier probe files remain unchanged.

`checks-summary.json`, `check-output.txt`, and their reproduction snapshots
remain historical evidence for earlier implementation hashes. Their results
are excluded from the 812 final-byte passes. Interrupted attempts and the two
initial unsupported expectations in the new test file are also excluded.
No pre-existing assertion was weakened. Currency cells are covered in both
numeric profiles; size-only cells use the interpretation current main supports.

`corpus-summary.json` records the completed comparison of **68,848 raw rows
from 231 files** against main `4f3547461300bfa70569f57c83728b2ad25d9d0b`.
The subsequently observed main `52c76d75849a628bbfd62b31795fd06e6bf9fdd1` has an identical encoder source-package tree and lockfiles; `corpus-main-forward-equivalence.json` records that equality without relabeling the executed baseline.
The pinned corpus is `8f7d60aaced28ee4252b9237f9d6e02360dc34bc`.
68,848 raw rows in 231 files; en-US: 244 changed inventories, 422 added row/value demands and 525 added occurrences; legacy: 661 changed inventories, 1,008 added row/value demands and 1,680 added occurrences; zero removed demands, occurrences or typed entries in both profiles.
Both profiles remove **zero demanded values, numeric occurrences, or typed
entries**. Additions are conservative retention and can include locators,
publication dates, and ambiguous numeric readings; they are not all recovered
operative quantities. Demand additions count distinct row/value pairs;
occurrence additions also include values already demanded in the same row.
Typed comparisons retain value/kind/unit metadata and ignore raw spelling
and character offsets.

`corpus-rows.jsonl.gz` contains every authenticated input-row key/hash,
recall/helper hashes, proof mode, and helper length/idempotence assertion.
It is 7.7 MB, so it is not committed; it is retained with the review evidence,
and its SHA256 (`424c1a35108287db3ab6a97893e729cfa6ac3adac54939a99e8177c4036b66ac`)
is authenticated in `corpus-artifacts.json`.
`corpus-deltas.jsonl` lists every changed row and its exact numeric deltas.
`corpus-input-manifest.json` supplies immutable addresses, SHA256s, Git blob
SHA1s, byte sizes, and row counts for all inputs. `corpus-artifacts.json`
authenticates the compact published bundle and the decompressed ledger hash.

All rows are checked through the actual main/candidate recall functions with
bounded within-run memoization; no previous-run or interrupted-run records
are reused. Differing recall text runs both actual extractors and profiles.
Equal text uses identical profiled extraction dependencies plus an exact
legacy candidate probe and guard-decision observer. A possible divergent
legacy decision runs the actual main extractor. The scripts and generated
probe/observer are published as inert `.py.txt` snapshots.

`corpus-pipeline-proof.json`, the independent construction review, and the
completion reports document the proof's limits and checks. The supplemental
review authenticates all 99 baseline package Git blobs and the identical
top-level ordinal dictionary population omitted by the declaration graph.
The completion verifier and a separate independent audit reconcile all 68,848 unique row keys and hashes,
all deltas, both profiles' totals, and all output/current-source hashes.
`corpus-reproduction.md` gives the precise restoration layout and commands.

The corpus covers raw bodies. Generated public-gate properties separately
exercise qualified monetary captions, referenced stamps, sole body chunks,
other composition layouts, unknown contents titles, and genuine locators.
This does not establish a complete composed-book differential, full artifact
validation, historical signing event, or production outcome.

Results identify file bytes, not an uncreated commit. The hub must commit the
validated bytes, bind PR artifact links to that resulting head, resolve the
upstream merge, and require green resulting-head CI before landing.

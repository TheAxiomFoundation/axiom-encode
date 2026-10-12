# Manual-mask corpus evidence

`corpus-summary.json` binds the fresh raw-manual differential to the exact
encoder source, locator metadata, numeric extractor, baseline commits, and all
231 immutable corpus input files. `corpus-deltas.jsonl` records every changed
raw row, its authenticated input-row hash, and its exact numeric deltas. The
summary gives the ledger hash and the full execution-evidence hashes.
`corpus-rows.jsonl.gz` makes every raw row's proof mode, hashes, mask counts, and
length/idempotence checks available without source prose or timings.
`sole-page-rows.jsonl.gz` similarly records every US helper qualification row
and its exact numeric deltas when fresh inventory comparisons run. The summary
records each compressed ledger's filename, SHA256, row count, and byte size.

The completed initial raw scan processes all 68,848 manual rows at the pinned
corpus commit, with candidate source SHA256
`03952d547c40cd42378428c6bc470193633678cbc7346591e1deec20edddd618`.
Every row runs the actual cleaner prefixes from the main baseline, reviewed
baseline, and that candidate snapshot. Equal prefix bytes plus identical suffix and
extractor dependencies establish equal inventories. Unequal prefixes run each
actual complete recall function and freshly extract both numeric profiles.
Inventories are compared with occurrence counts, excluding only raw text and
character offsets from semantic identity. No historical inventories or
interrupted results are reused.

When `execution_scope` is `fresh_final_byte_bridge`, final evidence adds a fresh
helper and actual-prefix comparison for every row against the completed raw
snapshot. Exact prefix equality and unchanged suffix/extractor dependencies
permit inherited inventories; changed prefixes rerun actual complete recalls
and both profiles. The summary and complete ledger record this bridge
explicitly. It is not another full extraction run, and the earlier completed
snapshot is not relabeled as the final source.
`previous-complete-corpus-summary.json` and
`previous-complete-corpus-rows.jsonl.gz` preserve that original completed scan.
The final `corpus-summary.json` and `corpus-rows.jsonl.gz` carry the current
source hashes and fresh bridge proofs.
`previous-complete-source.patch` reconstructs the completed candidate source
from the final committed source. Apply it only to a copied source in an isolated
scratch tree, then verify the resulting SHA256 against the recorded candidate
hash. The locator metadata is identical between these two snapshots.
Recorded runner and verifier snapshots are data files ending in `.py.txt`;
their exact hashes, pipeline proofs, and independent completion reports are
listed in the summary. For inspection or reproduction, restore these files as
scripts under `.hub-scratch/r6/corpus/`, with authenticated input files under
its `input/` directory, and restore `corpus-input-manifest.json` as
`corpus-manifest.json`. Immutable input addresses and the preserved-input audit
are also included. The original three-snapshot runner must receive the
reconstructed candidate source as its source snapshot, while the explicit
bridge and qualification use the final source. Do not substitute final bytes
into a run recorded against the candidate hash.

The compact ledger classifies additions as conservative source-number
retention. Such additions can include genuine section locators and publication
dates; they must not all be described as recovered operative quantities.
Context flags describe the stored delta excerpts, while their hashes bind the
full excerpts retained in execution evidence. Both main-to-final and
reviewed-to-final comparisons remove zero demanded values, numeric occurrences,
or typed inventory entries in both profiles.

The separate sole-page qualification wraps every US body with `Page 1` and
compares final locator-helper body views, masked digit positions, Python string
length, and idempotence. Changed body views additionally run the actual final
complete recall function and fresh typed extraction in both profiles. This is
a helper qualification, not a complete composed-book differential. Generated
tests in `test_source_completeness_manual_mask_fail_safe.py` exercise the public
complete-source gate across caption qualifiers and referenced stamps in other
composition shapes.
The qualification completed in two explicit segments at identical final source,
metadata, extractor, and authenticated input bytes: 47,180 records from the
original process, followed by fresh execution of its 58 outstanding rows. The
original process stalled and was interrupted; no successful original exit is
claimed. `segmented_execution` binds both segment hashes, and the independent
verifier checks their disjoint, complete 47,238-row input coverage and totals.

To reproduce the input scope, download each immutable URL listed in the summary
and verify its byte count, SHA256, Git blob SHA1, and row count. Compare the
complete manual provision-file set against the pinned corpus tree. Load the
recorded baseline source and metadata bytes with `git show`, load the final
bytes whose hashes are recorded, and call `authoritative_numeric_recall_text`
and `extract_typed_numeric_inventory_occurrences_from_text` with each recorded
profile. Check source and input hashes again when the run completes. Use GIL
CPython 3.13.9, at most four workers, the recorded thread caps, and streamed
row output.

These artifacts document the current execution. They do not authenticate a
historical signing event, production encoding, or full artifact validation.

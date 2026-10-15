# Seed entry review

Historical seed counts and earlier review chronology below are reported results: the four seed artifacts are named by hash but not included in this repository. The current list does not depend on them, because every candidate was re-joined to authenticated corpus bytes.

The seed is the 366 `removed_physical_from_baseline` occurrences in the round-6
coverage differential. They occupy **180 physical rows**. The historical
“194 labels / 160 rows” count describes unique removed labels and rows with a
label removal; it does not count rows that lose one occurrence while retaining
another occurrence of the same label.

The first review accepts 345 physical occurrences on 159 rows and rejects 21.
There are 343 `nested_under_numbered_parent` entries and two complete
`repealed_tombstone` entries. Every encoding target is the exact section citation
in its corpus row, with `requested_scope: null`. The review identity remains
`gpt-6.1-sol (hub build); independently verified 2026-10-15: every entry joined to signed corpus bytes, sample hand-audited`, recording the independent verification of every entry.

[entry-review.jsonl](entry-review.jsonl) records each accepted occurrence's exact
source identity, physical window, owning marker offset and context, full literal
owner-to-candidate span and its SHA256, scope judgment, and review rationale.
`owner_evidence.citation_path` names the full owning provision, including
outer lettered and numbered divisions. Its `physical_start` identifies the
owning numbered marker within the exact corpus row, including repeated numbers
under different outer divisions. The sheet records its complete literal path.
The two Louisiana chains `A.(1)(a)` and `B.(1)(a)`
were read directly, with owner offsets corrected to their actual `(1)` markers.

[rejected-candidates.json](rejected-candidates.json) records each rejected identity
and the source prefix through the candidate. Twenty Indiana `(b)` candidates are
real top-level demands: the preceding numbers are children of `(a)`, and `(b)`
starts the next subsection. They are not owned by the last number. One Wisconsin
`(f)` candidate remains demanded because the review does not authenticate the
intervening mixed-case `(eL)` and unusual multi-letter marker sequence.

All 11 artifacts containing candidates were retrieved from corpus commit
`e79781bddeaba4d3697c5375c7371b2a038f0f74`. Their full artifact SHA256, byte
lengths, and row counts match the release snapshot. Every one of the 366 row
body SHA256 values, row citations, and physical label offsets was checked before
review. The round-4 extraction supplied an additional independently hash-checked
body join for 130 candidates, including 109 literal parent proofs. Historical
proofs were review context, not automatic approval.

The source release is `us-rulespec-2026-08-08-obbb-alien-snap`, content SHA256
`0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc`.
Source evidence SHA256 values:

| Evidence | SHA256 |
| --- | --- |
| `round6-corpus.changes.jsonl` | `bb4f8bbbfe2aaac5b1f707f11cd138d3434ce4c2597e4ae425535d4d4731eff6` |
| `round4-corpus.physical-source-review.json` | `51304fb7efb849c0a3199552514633e76d571cd5da30a228e3e784cbefacbb5f` |
| `round4-corpus.physical-source-extract.jsonl` | `d32c8e7ebfcc8a504253e939f6bdad4ee8cf0f6615ed55ddee130f0c6a559bfe` |
| `round4-corpus.snapshot.json` | `ec93ccea12213ee35289113c3944f8e581d8953216022db150c245feaa5278ce` |

The packaged registry parses successfully and its 345 keys join one-to-one with
the committed accepted-entry review sheet. Corpus staleness and the runtime
removal differential are reported separately in this directory.

# Coverage exemption validation

All checks use release `us-rulespec-2026-08-08-obbb-alien-snap`, content SHA256
`0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc`, corpus
commit `e79781bddeaba4d3697c5375c7371b2a038f0f74`. Release signatures and full
artifact hashes, byte lengths and row counts were verified before scanning.

| Replay | Artifacts | Rows | Main physical demands | Physical removals | Row-label removals | Additions | Stale entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete signed release | 272 | 94,261 | 8,095 | 345 | 173 | 0 | 0 |
| Historical review slice | 186 | 75,934 | 5,438 | 345 | 173 | 0 | 0 |
| CI artifact set | 10 | 9,331 | 2,204 | 345 | 173 | 0 | 0 |

Each replay independently computes literal list matches and compares them with
the actual gate's removals. All 345 entries matched demands emitted by main at
their exact physical offsets. There are zero untested entries. The ledgers are
[corpus-differential.json](corpus-differential.json),
[historical-slice-differential.json](historical-slice-differential.json) and
[ci-artifacts-differential.json](ci-artifacts-differential.json).
The historical slice's pinned artifact inventory is
[corpus-artifacts.json](corpus-artifacts.json).

The final count is smaller than the historical 194 labels / 366 occurrences:
20 Indiana root demands and one Wisconsin unknown-marker candidate remain
mandatory. See [review-sheet.md](review-sheet.md) and
[rejected-candidates.json](rejected-candidates.json). No ownership inference
removes a demand at runtime.

## Reproduction

With the signed release and its artifacts materialized at `CORPUS_ROOT`:

```sh
uv run --locked --no-sync python scripts/check_coverage_exemptions.py \
  --release us-rulespec-2026-08-08-obbb-alien-snap \
  --release-sha256 0d69a0cdbe024fc2276f3c261c00402bb0a47488ac5f482cc20bd3404980adbc \
  --corpus-path CORPUS_ROOT --output ledger.json
```

Add `--artifact-manifest docs/review-evidence/coverage-exemptions/corpus-artifacts.json`
for the historical slice, or `--listed-artifacts-only` for the CI set. Artifact
ranges allow bounded shards. All three recorded replays completed in single
bounded invocations.

The exact fetch code in `.github/workflows/coverage-exemptions.yml` was replayed
locally against the public endpoints, then its downloaded bytes passed the
listed-artifact checker. The first replay using Python's HTTP client received
HTTP 403; bounded `curl` downloads succeeded. Hosted CI has not run in this
workspace. The workflow runs the stale-entry check on future PRs.

## Properties and mutations

`tests/test_coverage_exemptions.py` runs 300 arbitrary-Unicode outline examples
with random exact and near-match registries, 100 generated ASCII texts with
every source byte flipped independently, and 100 Unicode source changes beyond
the literal window. The oracle computes expected removals independently of the
runtime match helper. Byte flips retain all original demands supplied to the
gate; recomputing main's demands on the changed text also leaves them untouched.
ASCII flips remain valid UTF-8; malformed byte sequences never reach the string
scanner. Separate Unicode cases cover changed UTF-8 bytes outside the window.

Actual matcher mutations were applied in isolated test processes. Each mutation
caused a targeted test failure; no source mutation was left in the working tree.

| Mutation | Targeted regression | Result |
| --- | --- | --- |
| Citation equality becomes prefix matching | `test_citation_and_scope_match_exactly_without_prefixes_or_casefold` | Killed: 1 failed, 7 passed |
| Citation equality becomes casefolded equality | Same exact-match regressions | Killed: 1 failed, 7 passed |
| Both literal window and window digest checks removed | `test_literal_window_is_required_even_with_matching_window_digest` | Killed: 1 failed |

Round 1 reran 21 runtime and 17 tooling mutations and added eight scope/identity
mutations. All **46 mutations were killed**, with no test errors, skips or
timeouts. The round-1 runner compiled mutated modules in memory, verified the
working files were unchanged, and removed each pytest temporary directory.
The package-version mutation was excluded under this round's instruction to
leave version files unchanged.

Schema tests check reason enum, reviewer, literal keys, window digest and required
provenance. Proposal/checker tests cover signature and byte changes, physical
JSONL rows containing Unicode line separators, exact scope, stale offsets and
rows, and unapproved candidate output that cannot load as approved entries.

## Final focused checks

The combined coverage, properties, proposal, baseline differential, repair
overlay and corpus resolver run passed **796 tests**, with no failures, errors
or skips. The dependent RuleSpec checklist, eval and CLI repair selection passed
**45 tests**, with 5,378 deselected and no failures, errors or skips. These are
841 distinct passing tests. The full repository suite and real-engine suite
were not run.

The workflow pins file passed **252 tests**, with its opt-in environment check
skipped. Enabling that check separately passed **1 test**, confirming the
environment matches the lock. The focused lock-policy test also passed after
reproducing its failure against the original workflow in memory.

```sh
uv run --locked --no-sync pytest -q -o addopts= \
  tests/test_coverage_exemptions.py tests/test_propose_coverage_exemptions.py \
  tests/test_nested_branch_coverage.py tests/test_nested_branch_coverage_differential.py \
  tests/test_nested_branch_coverage_fail_safe.py tests/test_nested_branch_coverage_review.py \
  tests/test_nested_branch_coverage_structural_proof.py tests/test_editorial_source_boundaries.py \
  tests/test_evals_deferred_repair.py tests/test_repair_overlay_failure_is_retryable.py \
  tests/test_corpus_resolver.py
uv run --locked --no-sync pytest -q -o addopts= \
  tests/test_rulespec_validation.py tests/test_evals.py tests/test_cli.py \
  -k 'source_subparagraph_coverage or subparagraph_coverage_checklist or repair_candidate_overlay or mixed_missing_deferred_output or missing_deferred_output_repair or nonoperative_source_coverage_repair'
```

Local runs used named disposable pytest temporary directories, removed after
verification. The generated wheel contains byte-identical registry data. Exact
source comparison confirms both main scanner functions are unchanged, there is
one runtime filter call, and the existing authoritative waiver conditions remain
the left-hand conjuncts of the narrowed waiver. Only the encoder version and its
lock entry change; dependency and runtime pins remain unchanged.

Round 1 adds 54 regression cases. Signed test-release fixtures cover uppercase
and lowercase requested rows, absent or active requested rows alongside a
case-distinct tombstone, human and RuleSpec target inputs, request precedence,
and rules-file fallback from section and subsection parents. All existing
notice conditions remain required. Registry gate, reviewed-waiver and proposal
tests also require literal requested-scope case. The new cases reproduced the
folding and duplicated-ancestor failures before their fixes.

## Independent review cycle

Two independent reviews of the final list ran on 2026-10-15. Both joined every entry and rejected candidate to signed corpus bytes; the first also hand-audited 55 accepted entries and all 21 rejections. The earlier review and fix chronology below is reported, not reproducible from this repository.

An agent independent of the seed author opened every accepted physical window
and owner span, including outer ancestry where needed; it also checked every
rejection. Other agents independently reviewed the runtime gate, waiver, corpus
status reader, proposal/checker and CI. Findings about full owner citations,
relevant waiver labels, physical JSONL line splitting, exact status, and proposal
scope were fixed and reviewed again. The final CI transport change also received
a separate review. No actionable findings remain. This is review by separate GPT-6.1 Sol runs, not a
claim of human approval. After the reviews, each entry's `reviewer` field was
changed from the pending-review placeholder to a record of that verification;
no other byte changed. [independent-review.json](independent-review.json) binds the
reviewed registry bytes (`1832cbc8…`); the shipped file (`3dfa0032…`) differs
only in `reviewer`, and records the review roles and limits.

The round-1 independent source review caught two related callers during the
fix cycle: the offline proposer still folded scope keys, and a rules-file
fallback from a subsection parent duplicated an ancestor in the metadata
lookup. Both were fixed, covered by regressions, and reviewed again with no
actionable findings remaining. Corpus authentication now uses a separately
normalized literal requested citation; only the outline scanner folds scope
labels. Both workflow commands explicitly retain the lock.

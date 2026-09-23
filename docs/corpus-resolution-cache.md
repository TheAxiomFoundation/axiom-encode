# Corpus resolution cache

`corpus_resolver.resolve_local_corpus_source` used to re-read, sha256-hash and
`json.loads` every provisions file of the requested jurisdiction/document-class
bucket on every call. With the wave-4-lineage US unions that cost grows with
the bucket (us/statute is 32.9 MB on the current rulespec-us pin, 60.5 MB on
wave4-r2 and 100.6 MB in the projected HTS full-schedule union), and
validation calls the resolver many times per module. This note records the
design that makes resolution cheap, the invariants it keeps, and before/after
measurements on real corpus data.

## What is cached

Each `LocalCorpusRelease` owns a private `_LocalCorpusResolutionCache`. It is
not a dataclass field: equality, hashing, `repr`, `asdict` and `replace`
ignore it, and copies, unpickled instances and `with_fresh_reads()` start
empty. Nothing is shared between release objects.

- **Verified buckets.** `_verified_corpus_bucket` reads, hashes, byte-counts
  and parses a bucket's provisions artifacts exactly as the uncached resolver
  did, charging the caller's `_LocalCorpusReadBudget`. Any failure propagates
  and nothing is cached. Later calls reuse the parsed rows, which are bound to
  the immutable release object's sha256 and byte counts, and replay the same
  limit checks in the same order (per-artifact size, file count, per-artifact
  rows, aggregate bytes and rows), so a per-call limit still fails closed with
  the message an uncached read would give.
- **Citation index.** On first resolution each artifact gets a
  `_CorpusCitationIndex`: exact `citation_path` lookup and a sorted key list
  for prefix (descendant) scans with `bisect`. It replaces full scans of every
  row per lookup group, per descendant scan and per inactive-row check.
- **Resolution memo.** Outcomes are memoized per
  `("source", identifier, exact_only)`. The key is the caller's exact `str`
  identifier because the result echoes it (`requested`); `str` subclasses are
  not memoized.
- **Composition memo.** Bodyless parents composed from descendants are
  memoized per `("descendants", citation_path, scope)`, so every child that
  falls back to the same parent shares one composition. On releases without a
  full HTS schedule, thousands of tariff-line citations fall back to the
  bodyless `us/statute/hts` parent.
- **Failures.** `CorpusResolutionError` outcomes of resolution and composition
  are memoized as traceback-free chains (`_detached_error_chain`) and copied
  again for every raise: no caller frame is retained, no raise extends
  another's traceback, mutating a raised error cannot change a later one, and
  `__context__` is kept only where it is the explicit cause. Inside an active
  exception handler a memoized failure is recomputed instead, because Python
  chains every raise there to the handled exception and only a real raise
  builds the chain an uncached call would. A chain through any other
  exception type is never memoized. Invalid identifiers and read or
  verification failures are never memoized.
- **Limits.** Outcomes depend on the resolver's limits (`MAX_*` and
  `_MAX_*`, except the cache bounds). The cache records the values in force
  and clears every memo when one changes; cached rows stay, and the read
  limits are replayed on every call.

Two LRU budgets bound what one release object retains:

- **Rows.** Buckets are weighed at `_RESIDENT_BYTES_PER_ARTIFACT_BYTE` (4)
  times their artifact bytes; measured parsed rows plus indexes are 2.7 to
  3.0 times (292 MiB for the 100.6 MB HTS us/statute bucket, 232 MiB for the
  91.6 MB us/manual bucket). `MAX_CACHED_CORPUS_BYTES` (1 GiB) holds the HTS
  union's us/statute, us/manual and us/regulation buckets together, and
  `MAX_CACHED_CORPUS_BUCKETS` is 128. The most recently used bucket always
  stays; evicting a bucket drops its memo.
- **Memo.** `MAX_CACHED_CORPUS_MEMO_BYTES` (64 MiB) across all buckets'
  memos. An entry is weighed by what only it keeps alive (`sys.getsizeof` of
  its identifier, sliced or composed text, error messages, and 512 bytes per
  owned row); text and rows shared with the bucket or with a composition entry
  in the same memo are not charged again. When an entry does not fit, the
  least recently used buckets' memos are cleared first, then the bucket's
  own memo, and an entry that fits only into a restarted memo is not stored.

`validation-waivers audit` runs one worker per CI shard; locally it defaults
to up to eight worker processes, each with its own release object, so the
worst-case steady state there is eight times the per-release bound. A
`threading.Lock` guards the cache; bucket loads happen under it.

## Invariants

1. **Outcome equivalence.** Every value the cache retains is a pure function
   of release-verified bytes and its key, so a cached outcome equals what an
   uncached resolver would return or raise: same value, same error type,
   message, attributes and cause.
2. **Fail-closed reads.** Bytes are verified against the signed release on
   first read; a failed load is never cached. Cached rows are never re-read,
   so checks that must observe the current checkout re-hash it themselves:
   `resolve_local_corpus_dependency_artifacts` re-reads and re-hashes each
   artifact on every call (once per distinct artifact within a call), and
   `source_hash` staleness checks start from `release.with_fresh_reads()`.
3. **Isolation.** Returned objects are immutable or copies: resolutions are
   frozen dataclasses, iterated row metadata is deep-copied (iteratively, so
   any nesting `json.loads` accepts), and errors are fresh copies.
4. **Partition invariance.** `validation-waivers audit` validates many
   modules against one shared release per worker. Because of (1), a module's
   waiver fingerprint is identical whether or not its neighbors warmed the
   cache. `_fingerprint_validation_waiver_modules` still opens a fresh
   `_rulespec_resolution_cache_scope` per module for filesystem admissions,
   which are not pure.

## Tests

- `tests/test_corpus_resolution_cache.py`: cached outcomes equal fresh-release
  outcomes for every outcome class; a randomized differential test (20 seeds)
  against a linear-scan reference index; `_exact_only` and identifier spelling
  as memo keys; one computation per outcome and per parent composition; read
  once per bucket; isolation between equal release objects, copies and
  pickles; first-read failures not cached; tampering after caching still
  caught by fresh releases and dependency checks; every read limit replayed on
  warm calls with the cold message, alone and in combination; LRU eviction by
  bytes and count, including while iterating; the memo budget's clearing order
  and entry weights (identifiers charged, shared composition rows not);
  accounting guards under eviction and re-entry during a computation; limit
  changes invalidating memos, including for a computation already in flight;
  concurrent resolution from eight threads; detached failure chains (no
  retained caller frames, no stale context, no shared state, uncached chaining
  inside handlers); 5,000-deep metadata.
- `tests/test_validator_corpus_cache.py`: waiver fingerprints of seven modules
  (resolved, missing, ambiguous, sliced, composed, mixed, invalid citations)
  computed standalone with fresh releases equal those computed in a batch
  sharing one warm release, in both orders; each distinct cited path is
  fetched once with the same mapping, key order and overwrite semantics.
- `tests/test_corpus_release.py`: the grouped membership check matches the
  per-scope reference scan on 3,000 random cases covering every error;
  canonical bytes exclude the signature without mutating the payload.
- `tests/test_source_hash.py`: staleness checks observe on-disk changes made
  after another operation warmed the same release.

A mutation matrix over 45 mutations of the new code (budget
replay, memo keys, error replay, eviction and accounting guards, memo weights,
limit invalidation, index, composition memo, metadata copy, staleness,
dependency re-hashing, validator dedup, membership grouping, canonical bytes)
fails at least one of these tests for each.

## Other changes

- `ValidatorPipeline._cited_source_texts_for_rulespec_content` fetches each
  distinct cited path once per call (note20-china-301 has 10,026 proof atoms
  citing 42 provisions).
- `corpus_release._validate_scope_artifact_membership` groups source
  artifacts by scope directory in one pass instead of scanning every artifact
  per scope. Scope components match `_SCOPE_COMPONENT_RE` and contain no `/`,
  so a path starts with `data/corpus/sources/<j>/<d>/<v>/` exactly when its
  fourth to sixth segments are `(<j>, <d>, <v>)` and a seventh follows.
- `canonical_release_object_bytes` drops the top-level signature from a
  shallow copy; `verify_release_object` already works on a deep copy.
- `resolve_local_corpus_dependency_artifacts` hashes each distinct artifact
  once per call instead of once per component row.
- Row iteration and descendant reads compute an artifact's
  repository-relative path once instead of once per row.

## Benchmarks

All numbers are wall-clock seconds from one process on an Apple M5 Max that
other work shared throughout (load averages 70 to 90), with Python 3.14.4.
"Before" is origin/main at 5d80d753; "after" is this change. Treat absolute
times as noisy; the ratios are the point.

Data:

- `us-rulespec-2026-08-08-obbb-alien-snap` (signed; the current rulespec-us
  pin; axiom-corpus 8f7d60aa; us/statute 32.9 MB).
- `us-rulespec-2026-09-14-wave4-r2-union` (signed; axiom-corpus 9b0641af;
  us/statute 60.5 MB, us/manual 91.6 MB).
- `us-rulespec-2026-09-22-hts-full-schedule-union`, **projected**: wave4-r2
  plus the axiom-corpus#729 scopes at 4b6a4c15, signed with a throwaway key
  because the release is not published (us/statute 100.6 MB, 48,616 rows).

Release objects came from Supabase `corpus.release_objects`; provisions came
from sparse detached axiom-corpus worktrees. `MAX_RELEASE_OBJECT_BYTES` was
raised to 64 MiB in-process, as in axiom-encode#1675. Module content is
rulespec-us main at f43dec52.

### Resolver calls

| | obbb before | after | wave4-r2 before | after | HTS before | after |
|---|---:|---:|---:|---:|---:|---:|
| `LocalCorpusRelease` construction | 0.42 | 0.49 | 2.74 | 1.15 | 3.26 | 1.06 |
| of which the membership check | 0.18 | 0.005 | 2.12 | 0.09 | 2.38 | 0.07 |
| first resolve (loads us/statute) | 0.41 | 0.63 | 0.68 | 0.70 | 0.94 | 1.36 |
| repeated resolve | 0.29 | 0.00001 | 0.39 | 0.00001 | 1.05 | 0.00001 |
| note20's 42 distinct citations | 10.1 | 0.39 | 21.6 | 0.93 | 34.5 | 1.62 |
| note20 cited-source pass, 10,026 atoms x 2 | 7,100* | 15.2 | 13,294* | 12.2 | 22,576* | 14.7 |

\* Projected from the first 100 atoms (one citation, 200 resolver calls);
the full module was not run before the change. The after figure is mostly
PyYAML parsing the 2.7 MB module twice. Peak memory during a first call is
unchanged (89, 189 and 342 MiB); memory retained afterwards rises from 0.1 MiB
to 85, 166 and 292 MiB, the cached bucket.

Iterating the active HTS us/statute rows: 5.3 s cold and 5.4 s again before;
2.6 s cold and 1.2 s again after.

### Every federal citation in rulespec-us

rulespec-us main cites 14,379 distinct federal (`us/...`) corpus paths. Each
was resolved with `_exact_only` false and true (28,758 lookups) plus a full
us/statute iteration, giving 28,759 outcome signatures per release: the
attestation, body and proof evidence of a success, or the error type,
message, rows, reason and cause of a failure.

- **Equivalence.** The new code's signatures equal origin/main's for all
  28,759 on each release: obbb (3,089 successes, 12,846 not found, 12,824
  descendant-structure errors), wave4-r2, and the HTS union (28,753
  successes). That holds for one warm release with the lookups shuffled and a
  third repeated, and with the cache bound cut to 50 MB and 200 MB to force
  eviction churn, across the change's successive revisions.
- **Speed.** Before, with only file reads memoized by the harness (so the
  resolver's scans are all that remain), the 28,758 lookups took 1,658 s
  (obbb), 2,304 s (wave4-r2) and 572 s (HTS). After, the same lookups plus a
  repeated third (38,344 in all) took 3.0 s, 4.0 s and 6.2 s, bucket loads
  included.

### `axiom-encode validate`

In-process `validate --skip-reviewers` with axiom-rules-engine at the pinned
af6e4ea2, instrumented around `resolve_local_corpus_source`:

| Module | Release | Before: resolver s (calls) | After: resolver s (calls) | After: wall s |
|---|---|---:|---:|---:|
| note20-china-301 | obbb | 1,739 in the first 7,597 of about 20,059 calls; stopped at 30 min | 0.25 (84) | 507 |
| note20-china-301 | wave4-r2 | not run | 0.80 (84) | 331 |
| note20-china-301 | HTS | not run | 0.42 (84) | 364 |
| lines/generated/ch84 | HTS | not run | 0.08 (1,648) | 120 |
| statutes/26/3121/b/13 | obbb / wave4-r2 / HTS | 3.4 / 6.6 / 14.5 (14 each) | 0.001 (8 each) | 5 or less |
| statutes/7/2014/c | obbb / wave4-r2 / HTS | 4.1 / 9.4 / 23.9 (16 each) | 0.002 (8 each) | under 10 |

At the measured 0.229 s per call, note20's baseline resolver time on obbb
extrapolates to about 4,600 s, in line with the 107 minutes its waiver-audit
partition took in CI. Validation results (pass/fail and every issue) were
identical before and after for every module both runs covered. After the
change, note20's remaining wall time is almost all PyYAML: a profile puts 98%
of it in about 34 re-parses of the module, which is separate follow-up work.

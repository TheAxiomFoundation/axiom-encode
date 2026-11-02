# Corpus resolution cache

`corpus_resolver.resolve_local_corpus_source` used to re-read, sha256-hash and
`json.loads` every provisions file of the requested jurisdiction/document-class
bucket on every call. With the wave-4-lineage US unions that cost grows with
the bucket (us/statute is 32.9 MB on the current rulespec-us pin, 60.5 MB on
wave4-r2 and 100.6 MB in the projected HTS full-schedule union), and
validation calls the resolver many times per module. This note records the
design that makes resolution cheap, the invariants it keeps, and how much
work it saves, counted in reads and parses.

## What is cached

Each `LocalCorpusRelease` owns a private `_LocalCorpusResolutionCache`. It is
not a dataclass field: equality, hashing, `repr`, `dataclasses.fields` and
`replace` ignore it, and copies, unpickled instances and `with_fresh_reads()` start
empty. Nothing is shared between release objects.

- **Verified buckets.** `_verified_corpus_bucket` reads, hashes, byte-counts
  and parses a bucket's provisions artifacts exactly as the uncached resolver
  did, charging the caller's `_LocalCorpusReadBudget`. Any failure propagates
  and nothing is cached. Later calls reuse the parsed rows, which are bound to
  the immutable release object's sha256 and byte counts, and replay the same
  limit checks in the same order (per-artifact size, file count, per-artifact
  rows, aggregate bytes and rows), so a per-call limit still fails closed with
  the message an uncached read would give.
- **Disk changes.** Each cached bucket records the identity (device, inode,
  mode, size, modification and change times, by `lstat`) of every provisions
  file it read and, in a lock-file checkout, of the lock under
  `.axiom/corpus-locks/` that pins each one, taken before the read. Every
  later call first selects the bucket's files as an uncached read does, so
  every path component is checked again and a parent directory swapped for a
  symlink fails with the uncached error; it then compares the identities. If any differ, or a file is gone, the bucket
  and its memo are dropped and the call reads the checkout again, exactly as
  an uncached call does. Replacing a file by rename (as corpus placement and
  git checkouts do) always changes its identity. An in-place rewrite that
  keeps the inode, size and both timestamps, which a coarse-timestamp
  filesystem allows within one tick, is not detected; the dependency check
  and staleness checks below re-hash for that reason.
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
  memos. An entry is weighed by what only it keeps alive: a 512-byte base
  (1 KiB for a resolved source, which also holds its own row identity and
  digests), `sys.getsizeof` of its identifier, sliced or composed text,
  source history and error messages, and 512 bytes per owned row. Text and
  rows shared with the bucket or with a composition entry in the same memo
  are not charged again. A tracemalloc test holds the memory a memo of
  resolved sources keeps alive within 20% of its charge; failures and
  compositions are over-charged. When an entry does not fit, the
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
   first read; a failed load is never cached. A cached bucket is served only
   while its files and locks keep the identities they had when it was read.
   Checks that must observe the current bytes re-hash them regardless:
   `resolve_local_corpus_dependency_artifacts` re-reads and re-hashes each
   artifact on every call (once per distinct artifact within a call), and
   `source_hash` staleness checks start from `release.with_fresh_reads()`.
3. **Isolation.** Returned objects are immutable or copies: resolutions are
   frozen dataclasses, the selected row's metadata in a resolution is
   deep-copied from the parsed row and again for each caller when it holds
   nested lists or objects, iterated row metadata is deep-copied (iteratively, so
   any nesting `json.loads` accepts), and errors are fresh copies.
4. **Release isolation.** Each release object owns its cache, so a release
   never serves another release's entries, and binding a release after the
   release object or the lock changed starts empty.
5. **Partition invariance.** `validation-waivers audit` validates many
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
  pickles; first-read failures not cached; a provisions file replaced after
  caching is read again and fails like a fresh read, and the dependency check
  fails too; every read limit replayed on
  warm calls with the cold message, alone and in combination; LRU eviction by
  bytes and count, including while iterating; the memo budget's clearing order
  and entry weights (identifiers charged, shared composition rows not);
  accounting guards under eviction and re-entry during a computation; limit
  changes invalidating memos, including for a computation already in flight;
  concurrent resolution from eight threads; detached failure chains (no
  retained caller frames, no stale context, no shared state, uncached chaining
  inside handlers); 5,000-deep metadata; two-version buckets (row provenance,
  inactive rows in a later artifact, and limit replay across artifacts,
  including a limit equal to an artifact's size); an exact weight for every
  charged term; memo clearing order across three buckets; bounds that admit
  an exact fit; private `_MAX_*` limits in the invalidation snapshot.
- `tests/test_corpus_resolution_cache_properties.py` (Hypothesis): for any
  sequence of resolutions interleaved with replacing the provisions file,
  restoring it and replacing its lock, on plain and lock-file checkouts, the
  warm release returns exactly what a freshly bound release returns, and the
  first call after each change reads the checkout again; interleaved calls on
  two releases over the same citation paths each match their own release,
  uncached, and attest their own release; a replaced or deleted lock drops the
  bucket; an in-place rewrite with a new timestamp is read again; a parent
  directory swapped for a symlink is refused on a warm call; and the
  operation counts quoted below.
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

Before main was merged in and disk-change invalidation added, a mutation
matrix over 62 mutations of the new code (budget
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

## Work saved

`test_operation_count_of_repeated_resolution` counts the work of 100
resolutions of distinct citations in one 200-row bucket:

| | provisions reads (each also hashed) | rows parsed | `lstat` calls for the fingerprint |
|---|---:|---:|---:|
| uncached (a fresh release per call, as before this change) | 100 | 20,000 | none |
| cached (one release) | 1 | 200 | 100 |

In general, N resolutions in a bucket of R rows over A artifacts go from
N × A reads and N × R parses to A reads and R parses, plus N × A `lstat`
calls (N × 2A in a lock-file checkout).

Wall-clock figures for real US releases need a benchmark run against the
signed release objects and corpus checkouts, which live outside this
repository; until one is rerun on this revision, this note quotes none.

# Corpus resolution cache

`corpus_resolver.resolve_local_corpus_source` used to re-read, sha256-hash and
`json.loads` every provisions file of the requested jurisdiction/document-class
bucket on every call. That cost grows with the bucket, which for US statutes
runs to tens of megabytes, and validation calls the resolver many times per
module. This note records the
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

- **Rows.** Buckets are charged `_RESIDENT_BYTES_PER_ARTIFACT_BYTE` (4)
  times their artifact bytes against `MAX_CACHED_CORPUS_BYTES` (1 GiB), and
  `MAX_CACHED_CORPUS_BUCKETS` is 128. Evicting a bucket drops its memo. The
  charge is an estimate, so 1 GiB is not a hard bound on resident memory:
  - What a bucket really keeps depends on row length.
    `scripts/measure_corpus_cache_memory.py` binds synthetic buckets and
    measures them with `tracemalloc`. On 20,000 rows it reports, retained
    after one cold resolution as a multiple of the file's bytes:

    | Body chars | File MiB | Retained / file | Peak / file | Charged (4×) MiB |
    |---:|---:|---:|---:|---:|
    | 40 | 7.6 | 5.40× | 8.33× | 30.3 |
    | 200 | 10.6 | 4.14× | 7.09× | 42.6 |
    | 1,200 | 29.7 | 2.12× | 5.11× | 118.8 |

    Short rows keep more than they are charged, so a cache of short-row
    buckets can pass 1 GiB; long rows keep less. The load itself peaks at 5
    to 8 times the file's bytes.
  - The most recently used bucket always stays, however large: one bucket
    can exceed the budget on its own, up to what a single call's read limits
    allow.
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

Nothing bounds the total across release objects. `validation-waivers audit`
runs one worker per CI shard; locally it defaults to up to eight worker
processes, and each task binds its own release object, so up to eight caches
are live at once. A `threading.Lock` guards each cache, and bucket loads
happen under it, so threads sharing one release wait for each other's first
loads.

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
5. **Partition invariance.** With one worker, `validation-waivers audit`
   fingerprints every module in one serial call that binds one release and
   shares it across the modules. With two or more, each task covers one
   module, and `_fingerprint_waiver_chunk` binds a new `LocalCorpusRelease`
   for it, so nothing is shared. Because of (1), a module's waiver
   fingerprint is the same either way, whether or not its neighbors warmed
   the cache. `_fingerprint_validation_waiver_modules` still opens a fresh
   `_rulespec_resolution_cache_scope` per module for filesystem admissions,
   which are not pure.
6. **Forks.** A child process forked while another thread holds a cache's
   lock would inherit a held lock that no thread of its own can release. A
   handler registered with `os.register_at_fork` gives every cache in the
   child a fresh lock and empties it, so a forked child starts as a spawned
   one does. The parent's caches are untouched.

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
  uncached, and attest their own release. Example tests cover a replaced or
  deleted lock; an in-place rewrite with a new modification time; an in-place
  same-size rewrite with restored times, which only the change time reveals;
  a file swapped after the cold read; a parent directory swapped for a
  symlink, before or during a warm call; caller mutations of nested metadata,
  memoized or not; a child forked while the cache lock is held; and the
  operation counts quoted below.

  The freshly bound release these properties compare with runs the same
  bucket, index and resolution code cold, so they show the cache is
  consistent with its own cold path, not that it agrees with the linear scan
  the resolver used before. `test_indexed_cached_resolution_matches_linear_uncached_reference`
  in `tests/test_corpus_resolution_cache.py` compares with a linear-scan
  reference index, and `tests/test_corpus_resolver.py`, which this change
  leaves as it was, carries the resolver's own expectations.
- `tests/test_validator_corpus_cache.py`: waiver fingerprints of seven modules
  (resolved, missing, ambiguous, sliced, composed, mixed, invalid citations)
  computed standalone with fresh releases equal those computed in a batch
  sharing one warm release, in both orders; each distinct cited path is
  fetched once with the same mapping, key order and overwrite semantics.
- `tests/test_corpus_release.py`: the grouped membership check matches the
  per-scope reference scan on 3,000 random cases covering every error;
  canonical bytes exclude the signature without mutating the payload.
- `tests/test_source_hash.py`: staleness checks observe on-disk changes made
  after another operation warmed the same release, including when the disk
  fingerprint is blind to the change.

Each of these mutations of the cache fails at least one of these tests:
dropping the change time from the file identity; staleness checks without
`with_fresh_reads()`; taking the cold-path fingerprint after the read; a
shallow metadata copy where rows are read; rebuilding the citation index on
every call; trusting the fingerprint without comparing it; leaving locks out
of it; one cache shared by every release; no per-caller metadata copy; the
path checks before the fingerprint; and no fork handler.

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

| | provisions reads (each also hashed) | rows parsed | fingerprint `lstat` calls | citation-index builds |
|---|---:|---:|---:|---:|
| uncached (a fresh release per call) | 100 | 20,000 | 100 | 100 |
| cached (one release) | 1 | 200 | 100 | 1 |

The uncached row runs this change's cold path, which takes the fingerprint
before it reads; the resolver before this change made the same reads and
parses and took no fingerprint. In general, N resolutions in a bucket of R
rows over A artifacts go from N × A reads and N × R parses to A reads and R
parses. Every call, cached or not, takes A fingerprint `lstat` calls (2A in a
lock-file checkout) and the path checks an uncached read makes.

Wall-clock figures for real US releases need a benchmark run against the
signed release objects and corpus checkouts, which live outside this
repository; until one is rerun on this revision, this note quotes none.

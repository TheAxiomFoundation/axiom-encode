"""Properties of the per-release corpus resolution cache.

For any sequence of resolutions, interleaved with changes to the provisions
files and their locks, a warm ``LocalCorpusRelease`` returns exactly what a
freshly bound, uncached release returns for the same call. A change to a
provisions file or to the lock that pins it drops the cached bucket, so the
next call reads the checkout again. Two releases never serve each other's
entries.

Files are changed by writing a new file and renaming it into place, as
corpus placement and git checkouts do. That gives the file a new identity
on every filesystem; an in-place rewrite that keeps the inode, size and both
timestamps (possible within one timestamp tick) is not detected, which
``docs/corpus-resolution-cache.md`` records.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_resolver
from axiom_encode.corpus_materialize import NO_FETCH_ENV, lock_path_for
from axiom_encode.corpus_resolver import resolve_local_corpus_source
from tests.release_object_fixtures import bind_test_corpus_release
from tests.test_corpus_materialize import _write_locks
from tests.test_corpus_resolution_cache import (
    STATUTE,
    _assert_accounting,
    _cache,
    _fresh,
    _outcome,
    _provisions_path,
    _random_corpus,
    _write_rows,
)

RELEASE = "cache-property-release"
OTHER_RELEASE = "cache-property-other-release"
OTHER_STATUTE = "2026-02-01-statute"

PROPERTY_SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

_resolve_op = st.tuples(
    st.just("resolve"), st.integers(min_value=0, max_value=10**6), st.booleans()
)
_operations = st.lists(
    st.one_of(
        _resolve_op,
        _resolve_op,
        st.tuples(st.just("tamper")),
        st.tuples(st.just("restore")),
        st.tuples(st.just("relock")),
    ),
    min_size=1,
    max_size=30,
)


def _replace(path: Path, data: bytes) -> None:
    """Write ``data`` to a new file and rename it over ``path``."""

    temporary = path.with_name(f".{path.name}.replacement")
    temporary.write_bytes(data)
    os.replace(temporary, path)


@contextmanager
def _provision_reads() -> Iterator[list[Path]]:
    original = corpus_resolver.read_bounded_regular_file
    reads: list[Path] = []

    def tracked(root, candidate, *, label, max_bytes, **kwargs):
        if label == "corpus provision file":
            reads.append(Path(candidate))
        return original(root, candidate, label=label, max_bytes=max_bytes, **kwargs)

    corpus_resolver.read_bounded_regular_file = tracked
    try:
        yield reads
    finally:
        corpus_resolver.read_bounded_regular_file = original


@contextmanager
def _no_fetch() -> Iterator[None]:
    previous = os.environ.get(NO_FETCH_ENV)
    os.environ[NO_FETCH_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(NO_FETCH_ENV, None)
        else:
            os.environ[NO_FETCH_ENV] = previous


def _lock_path(root: Path, provisions: Path) -> Path:
    lock = lock_path_for(provisions.relative_to(root).as_posix())
    assert lock is not None
    return root / lock


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10**6),
    operations=_operations,
    locked=st.booleans(),
)
# Each change kind between two warm calls, so every run covers them.
@example(
    seed=1,
    operations=[("resolve", 0, False), ("relock",), ("resolve", 0, False)],
    locked=True,
)
@example(
    seed=2,
    operations=[
        ("resolve", 3, True),
        ("tamper",),
        ("resolve", 3, True),
        ("restore",),
        ("resolve", 3, True),
        ("resolve", 4, False),
    ],
    locked=False,
)
def test_property_warm_release_matches_an_uncached_release_for_any_sequence(
    seed: int, operations: list[tuple[object, ...]], locked: bool
):
    with tempfile.TemporaryDirectory() as directory, _no_fetch():
        root = Path(directory)
        rng = random.Random(seed)
        identifiers = _random_corpus(root, rng)
        provisions = _provisions_path(root, "statute", STATUTE)
        original = provisions.read_bytes()
        tampered = original.replace(b"Text of", b"Txet of")
        if tampered == original:
            tampered = original + b"\n"
        if locked:
            _write_locks(root, {provisions.relative_to(root).as_posix(): original})
        release = bind_test_corpus_release(root, RELEASE, [("us", "statute", STATUTE)])
        assert release.uses_corpus_locks is locked
        lock = _lock_path(root, provisions)

        changed = True  # nothing is cached yet
        for operation in operations:
            kind = operation[0]
            if kind == "tamper":
                _replace(provisions, tampered)
                changed = True
                continue
            if kind == "restore":
                _replace(provisions, original)
                changed = True
                continue
            if kind == "relock":
                if locked:
                    _replace(lock, lock.read_bytes())
                    changed = True
                continue
            _, pick, exact = operation
            identifier = identifiers[int(pick) % len(identifiers)]
            with _provision_reads() as reads:
                warm = _outcome(
                    lambda: resolve_local_corpus_source(
                        identifier, release, _exact_only=bool(exact)
                    )
                )
            cold = _outcome(
                lambda: resolve_local_corpus_source(
                    identifier, _fresh(release), _exact_only=bool(exact)
                )
            )
            assert warm == cold, (identifier, exact, operation)
            if changed:
                # The first call after a change reads the checkout again.
                assert reads == [provisions.resolve()]
            else:
                assert reads == []
            # A failed read caches nothing, so the next call reads again.
            changed = ("us", "statute") not in _cache(release).buckets
        _assert_accounting(release)


@PROPERTY_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=10**6),
    picks=st.lists(
        st.tuples(
            st.booleans(), st.integers(min_value=0, max_value=10**6), st.booleans()
        ),
        min_size=1,
        max_size=40,
    ),
)
def test_property_two_releases_never_serve_each_others_entries(
    seed: int, picks: list[tuple[bool, int, bool]]
):
    """Interleaved calls on two releases each match that release, uncached."""

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        identifiers = _random_corpus(root, random.Random(seed))
        # The other release has the same citation paths with other text.
        statute = _provisions_path(root, "statute", STATUTE).read_bytes()
        other_rows = statute.replace(b"Text of", b"Other text of").replace(
            STATUTE.encode(), OTHER_STATUTE.encode()
        )
        other_path = _provisions_path(root, "statute", OTHER_STATUTE)
        other_path.write_bytes(other_rows)
        first = bind_test_corpus_release(root, RELEASE, [("us", "statute", STATUTE)])
        second = bind_test_corpus_release(
            root, OTHER_RELEASE, [("us", "statute", OTHER_STATUTE)]
        )
        assert first.content_sha256 != second.content_sha256
        releases = {False: first, True: second}

        for use_second, pick, exact in picks:
            release = releases[use_second]
            identifier = identifiers[pick % len(identifiers)]
            warm = _outcome(
                lambda: resolve_local_corpus_source(
                    identifier, release, _exact_only=exact
                )
            )
            cold = _outcome(
                lambda: resolve_local_corpus_source(
                    identifier, _fresh(release), _exact_only=exact
                )
            )
            assert warm == cold, (use_second, identifier, exact)
            if warm[0] == "ok":
                resolved = warm[1]
                assert resolved.release_name == release.name
                assert resolved.release_content_sha256 == release.content_sha256
        assert _cache(first) is not _cache(second)
        _assert_accounting(first)
        _assert_accounting(second)


def test_a_changed_lock_drops_the_cached_bucket(tmp_path: Path, monkeypatch):
    monkeypatch.setenv(NO_FETCH_ENV, "1")
    _write_rows(tmp_path, STATUTE, [{"citation_path": "us/statute/1", "body": "One."}])
    provisions = _provisions_path(tmp_path, "statute", STATUTE)
    _write_locks(
        tmp_path, {provisions.relative_to(tmp_path).as_posix(): provisions.read_bytes()}
    )
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])
    lock = _lock_path(tmp_path, provisions)

    with _provision_reads() as reads:
        first = resolve_local_corpus_source("us/statute/1", release)
        assert resolve_local_corpus_source("us/statute/1", release) == first
        assert len(reads) == 1

        _replace(lock, lock.read_bytes())
        assert resolve_local_corpus_source("us/statute/1", release) == first
        assert len(reads) == 2

        lock.unlink()
        assert resolve_local_corpus_source("us/statute/1", release) == first
        assert len(reads) == 3
        assert resolve_local_corpus_source("us/statute/1", release) == first
        assert len(reads) == 3


def test_an_in_place_rewrite_with_a_new_timestamp_is_read_again(tmp_path: Path):
    _write_rows(tmp_path, STATUTE, [{"citation_path": "us/statute/1", "body": "One."}])
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])
    assert resolve_local_corpus_source("us/statute/1", release).body == "One."
    provisions = _provisions_path(tmp_path, "statute", STATUTE)
    stat = provisions.stat()
    provisions.write_bytes(provisions.read_bytes().replace(b"One.", b"Two."))
    os.utime(provisions, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    assert _outcome(lambda: resolve_local_corpus_source("us/statute/1", release)) == (
        _outcome(lambda: resolve_local_corpus_source("us/statute/1", _fresh(release)))
    )
    assert ("us", "statute") not in _cache(release).buckets


def test_operation_count_of_repeated_resolution(tmp_path: Path, monkeypatch):
    """The work repeated resolution does, cached and uncached, as counts.

    The pull request and ``docs/corpus-resolution-cache.md`` quote these
    counts instead of wall-clock times.
    """

    rows, calls = 200, 100
    _write_rows(
        tmp_path,
        STATUTE,
        [
            {"citation_path": f"us/statute/1/{index}", "body": f"Body {index}."}
            for index in range(rows)
        ],
    )
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])
    counts = {"parsed": 0, "stat": 0}
    real_loads = corpus_resolver.json.loads
    real_identity = corpus_resolver._file_identity

    def loads(*args, **kwargs):
        counts["parsed"] += 1
        return real_loads(*args, **kwargs)

    def identity(path):
        counts["stat"] += 1
        return real_identity(path)

    monkeypatch.setattr(corpus_resolver.json, "loads", loads)
    monkeypatch.setattr(corpus_resolver, "_file_identity", identity)

    def run(pick) -> tuple[int, int, int]:
        counts.update(parsed=0, stat=0)
        with _provision_reads() as reads:
            for index in range(calls):
                resolve_local_corpus_source(f"us/statute/1/{index}", pick())
        return len(reads), counts["parsed"], counts["stat"]

    fresh_releases = [_fresh(release) for _ in range(calls)]
    uncached = run(iter(fresh_releases).__next__)
    cached = run(lambda: release)
    # Uncached: every call reads, hashes and parses the whole bucket.
    assert uncached[:2] == (calls, calls * rows)
    # Cached: one read and one parse of each row; each call takes one
    # lstat of the provisions file for the disk fingerprint.
    assert cached == (1, rows, calls)


def test_mutating_selected_row_metadata_cannot_change_a_later_result(
    tmp_path: Path,
):
    _write_rows(
        tmp_path,
        STATUTE,
        [
            {
                "citation_path": "us/statute/1",
                "body": "One.",
                "metadata": {"notes": ["first"], "detail": {"kind": "rule"}},
            }
        ],
    )
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])
    expected = resolve_local_corpus_source("us/statute/1", _fresh(release))

    for _ in range(3):
        resolved = resolve_local_corpus_source("us/statute/1", release)
        assert resolved == expected
        record = resolved.selected_record
        assert record is not None
        record.metadata["notes"].append("mutated")
        record.metadata["detail"]["kind"] = "mutated"
    assert resolve_local_corpus_source("us/statute/1", release) == expected

"""Correctness and isolation of the per-release corpus resolution cache.

A ``LocalCorpusRelease`` memoizes parsed, release-verified provisions per
jurisdiction/document-class bucket and resolution outcomes per
``(identifier, _exact_only)``. These tests pin that the cache never changes an
outcome, stays fail-closed, is bounded, and is private to one release object.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pickle
import random
import sys
import threading
from pathlib import Path

import pytest

from axiom_encode import corpus_resolver
from axiom_encode.corpus_resolver import (
    AmbiguousCorpusSourceError,
    CorpusDescendantStructureError,
    CorpusResolutionError,
    CorpusRowStructureError,
    CorpusSourceNotFoundError,
    InactiveCorpusSourceError,
    InvalidActiveCorpusSourceError,
    InvalidCorpusCitationError,
    LocalCorpusRelease,
    iter_active_local_corpus_rows,
    resolve_local_corpus_dependency_artifacts,
    resolve_local_corpus_source,
    resolve_scoped_local_corpus_source,
)
from tests.release_object_fixtures import bind_test_corpus_release

RELEASE = "cache-test-release"
STATUTE = "2026-01-01-statute"
REGULATION = "2026-01-01-regulation"


def _provisions_path(root: Path, document_class: str, version: str) -> Path:
    return (
        root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / document_class
        / f"{version}.jsonl"
    )


def _write_rows(
    root: Path,
    version: str,
    rows: list[dict[str, object]],
    *,
    document_class: str = "statute",
) -> Path:
    path = _provisions_path(root, document_class, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, row in enumerate(rows):
        lines.append(
            json.dumps(
                {
                    "id": f"row-{version}-{index}",
                    "jurisdiction": "us",
                    "document_class": document_class,
                    "version": version,
                    "source_path": f"sources/us/{document_class}/{version}/x.xml",
                    "source_as_of": "2026-01-02",
                    "expression_date": "2026-01-01",
                    **row,
                },
                sort_keys=True,
            )
        )
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


STATUTE_ROWS: list[dict[str, object]] = [
    {
        "citation_path": "us/statute/7/2014",
        "body": "(a) First rule.\n(e) The standard deduction is $198.\n(f) Sibling.",
        "metadata": {"source_history": ["Amended 2025."]},
    },
    {"citation_path": "us/statute/26/1", "body": "Tax imposed."},
    {"citation_path": "us/statute/42/9", "body": None, "heading": "Parent"},
    {"citation_path": "us/statute/42/9/b", "body": "second", "ordinal": 2},
    {"citation_path": "us/statute/42/9/a", "body": "first", "ordinal": 1},
    {"citation_path": "us/statute/5/5", "body": "one"},
    {"citation_path": "us/statute/5/5", "body": "two"},
    {
        "citation_path": "us/statute/8/8",
        "body": None,
        "metadata": {"status": "repealed"},
    },
    {"citation_path": "us/statute/8/8/1", "body": "stale"},
    {"citation_path": "us/statute/9/9", "body": "retired", "version": "retired"},
    {"citation_path": "us/statute/10/1", "body": "no id", "id": ""},
    {"citation_path": "us/statute/11/1", "body": None},
    {"citation_path": "us/statute/11/1/Bad Segment!", "body": "bad"},
]
REGULATION_ROWS: list[dict[str, object]] = [
    {"citation_path": "us/regulation/7/273/9", "body": "Regulation body."},
]

# Every outcome class the resolver can produce for a loaded bucket.
IDENTIFIERS = (
    "us/statute/7/2014",
    "us/statute/7/2014/e",
    " us/statute/7/2014/e ",
    "us:statutes/7/2014/e",
    "us/statute/26/1",
    "us/statute/42/9",
    "us/statute/5/5",
    "us/statute/8/8",
    "us/statute/9/9",
    "us/statute/10/1",
    "us/statute/11/1",
    "us/statute/404",
    "us/regulation/7/273/9",
    "us/regulation/7/273/10",
    "us/form/1040",
)


def _corpus(root: Path) -> LocalCorpusRelease:
    _write_rows(root, STATUTE, STATUTE_ROWS)
    _write_rows(root, REGULATION, REGULATION_ROWS, document_class="regulation")
    return bind_test_corpus_release(
        root,
        RELEASE,
        [("us", "statute", STATUTE), ("us", "regulation", REGULATION)],
    )


def _fresh(release: LocalCorpusRelease) -> LocalCorpusRelease:
    return LocalCorpusRelease(
        release.root, release.name, release.content_sha256, release.public_key
    )


def _outcome(call) -> tuple[object, ...]:
    """A value that is equal iff two calls produced the same observable outcome."""

    try:
        resolved = call()
    except CorpusResolutionError as exc:
        return (
            "error",
            type(exc),
            str(exc),
            tuple(sorted((key, repr(value)) for key, value in vars(exc).items())),
            type(exc.__cause__),
            str(exc.__cause__) if exc.__cause__ is not None else None,
        )
    if isinstance(resolved, corpus_resolver.ResolvedCorpusSource):
        return ("ok", resolved, resolved.to_attestation(), resolved.proof_evidence_text)
    return ("ok", resolved)


def _cache(release: LocalCorpusRelease):
    return release._resolution_cache


def _assert_accounting(release: LocalCorpusRelease) -> None:
    cache = _cache(release)
    buckets = list(cache.buckets.values())
    assert cache.row_bytes == sum(bucket.row_bytes for bucket in buckets)
    assert cache.memo_bytes == sum(bucket.memo_bytes for bucket in buckets)
    assert cache.memo_bytes <= corpus_resolver.MAX_CACHED_CORPUS_MEMO_BYTES
    assert all((bucket.memo_bytes == 0) == (not bucket.memo) for bucket in buckets)


def _row_weight(path: Path) -> int:
    return corpus_resolver._RESIDENT_BYTES_PER_ARTIFACT_BYTE * path.stat().st_size


def _entry_weight(identifier: str) -> int:
    """Memo weight of an exact-row outcome, whose body the bucket already holds."""

    return 512 + sys.getsizeof(identifier)


@pytest.fixture
def provision_reads(monkeypatch):
    original = corpus_resolver.read_bounded_regular_file
    reads: list[Path] = []

    def tracked(root, candidate, *, label, max_bytes, **kwargs):
        if label == "corpus provision file":
            reads.append(Path(candidate))
        return original(root, candidate, label=label, max_bytes=max_bytes, **kwargs)

    monkeypatch.setattr(corpus_resolver, "read_bounded_regular_file", tracked)
    return reads


def test_cached_outcomes_equal_fresh_release_outcomes(tmp_path: Path):
    release = _corpus(tmp_path)
    expected = {
        identifier: _outcome(
            lambda identifier=identifier: resolve_local_corpus_source(
                identifier, _fresh(release)
            )
        )
        for identifier in IDENTIFIERS
    }
    kinds = {value[1] if value[0] == "error" else "ok" for value in expected.values()}
    assert kinds == {
        "ok",
        AmbiguousCorpusSourceError,
        CorpusDescendantStructureError,
        CorpusRowStructureError,
        CorpusSourceNotFoundError,
        InactiveCorpusSourceError,
        InvalidActiveCorpusSourceError,
        InvalidCorpusCitationError,
    }

    order = list(IDENTIFIERS) * 3
    random.Random(7).shuffle(order)
    for identifier in order:
        assert (
            _outcome(lambda: resolve_local_corpus_source(identifier, release))
            == expected[identifier]
        ), identifier
    _assert_accounting(release)


def test_exact_only_and_scoped_resolution_are_memoized_separately(tmp_path: Path):
    release = _corpus(tmp_path)
    child = "us/statute/7/2014/e"

    with pytest.raises(CorpusSourceNotFoundError):
        resolve_local_corpus_source(child, release, _exact_only=True)
    sliced = resolve_local_corpus_source(child, release)
    with pytest.raises(CorpusSourceNotFoundError):
        resolve_local_corpus_source(child, release, _exact_only=True)

    assert sliced.body == "(e) The standard deduction is $198."
    assert sliced.slice_required is True
    parent = resolve_local_corpus_source("us/statute/7/2014", release)
    scoped = resolve_scoped_local_corpus_source(parent, child, release)
    assert scoped.body == sliced.body
    assert scoped.slice_required is True


def test_identifier_spelling_is_part_of_the_memo_key(tmp_path: Path):
    release = _corpus(tmp_path)

    padded = resolve_local_corpus_source(" us/statute/26/1 ", release)
    canonical = resolve_local_corpus_source("us/statute/26/1", release)

    assert padded.requested == " us/statute/26/1 "
    assert canonical.requested == "us/statute/26/1"
    assert dataclasses.replace(padded, requested=canonical.requested) == canonical


def test_non_str_identifiers_resolve_without_memoization(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)

    class Citation(str):
        pass

    calls = []
    original = corpus_resolver._resolve_in_verified_bucket

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(corpus_resolver, "_resolve_in_verified_bucket", counted)

    first = resolve_local_corpus_source(Citation("us/statute/26/1"), release)
    second = resolve_local_corpus_source(Citation("us/statute/26/1"), release)
    resolve_local_corpus_source("us/statute/26/1", release)
    resolve_local_corpus_source("us/statute/26/1", release)

    assert first == second
    assert len(calls) == 3
    with pytest.raises(InvalidCorpusCitationError):
        resolve_local_corpus_source(["us", "statute"], release)  # type: ignore[arg-type]


def test_each_outcome_is_computed_once_per_release(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    calls: list[str] = []
    original = corpus_resolver._resolve_in_verified_bucket

    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(corpus_resolver, "_resolve_in_verified_bucket", counted)

    for _ in range(3):
        resolve_local_corpus_source("us/statute/26/1", release)
        with pytest.raises(AmbiguousCorpusSourceError):
            resolve_local_corpus_source("us/statute/5/5", release)
        with pytest.raises(CorpusSourceNotFoundError):
            resolve_local_corpus_source("us/statute/404", release)

    assert sorted(calls) == ["us/statute/26/1", "us/statute/404", "us/statute/5/5"]
    resolve_local_corpus_source("us/statute/26/1", _fresh(release))
    assert len(calls) == 4


def test_invalid_identifiers_are_rejected_on_every_call(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    for _ in range(2):
        with pytest.raises(InvalidCorpusCitationError):
            resolve_local_corpus_source("us/nonsense/1", release)
    assert all(not bucket.memo for bucket in _cache(release).buckets.values())
    with pytest.raises(TypeError, match="validated LocalCorpusRelease"):
        resolve_local_corpus_source("us/statute/26/1", object())  # type: ignore[arg-type]


def test_memoized_failures_are_raised_as_independent_equal_exceptions(
    tmp_path: Path,
):
    release = _corpus(tmp_path)
    raised: list[CorpusResolutionError] = []
    for _ in range(3):
        with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
            resolve_local_corpus_source("us/statute/5/5", release)
        raised.append(exc_info.value)

    first, *rest = raised
    for error in rest:
        assert error is not first
        assert str(error) == str(first)
        assert error.citation_path == first.citation_path
        assert error.rows == first.rows
        # Each raise carries its own short traceback; nothing accumulates.
        depth = 0
        frame = error.__traceback__
        while frame is not None:
            depth += 1
            frame = frame.tb_next
        assert depth <= 3

    with pytest.raises(CorpusRowStructureError) as first_structure:
        resolve_local_corpus_source("us/statute/10/1", release)
    with pytest.raises(CorpusRowStructureError) as second_structure:
        resolve_local_corpus_source("us/statute/10/1", release)
    assert second_structure.value.reason == first_structure.value.reason
    assert type(second_structure.value.__cause__) is type(
        first_structure.value.__cause__
    )
    assert second_structure.value.__cause__ is not None


def test_bucket_artifacts_are_read_once_per_release(tmp_path: Path, provision_reads):
    release = _corpus(tmp_path)
    statute = _provisions_path(tmp_path, "statute", STATUTE).resolve()
    regulation = _provisions_path(tmp_path, "regulation", REGULATION).resolve()

    for identifier in ("us/statute/26/1", "us/statute/42/9", "us/statute/7/2014/e"):
        resolve_local_corpus_source(identifier, release)
    # The fixture's malformed rows make whole-bucket iteration fail closed
    # after the read; the read itself is still shared with resolution.
    _outcome(
        lambda: list(iter_active_local_corpus_rows(release, document_class="statute"))
    )
    assert provision_reads == [statute]

    resolve_local_corpus_source("us/regulation/7/273/9", release)
    _outcome(lambda: list(iter_active_local_corpus_rows(release)))
    assert provision_reads == [statute, regulation]

    resolve_local_corpus_source("us/statute/26/1", _fresh(release))
    assert provision_reads == [statute, regulation, statute]


def test_equal_release_objects_do_not_share_a_cache(tmp_path: Path, provision_reads):
    release = _corpus(tmp_path)
    twin = _fresh(release)

    assert twin == release and hash(twin) == hash(release)
    assert _cache(twin) is not _cache(release)
    resolve_local_corpus_source("us/statute/26/1", release)
    assert not _cache(twin).buckets
    resolve_local_corpus_source("us/statute/26/1", twin)
    assert len(provision_reads) == 2


def test_copies_and_unpickled_releases_start_with_an_empty_cache(tmp_path: Path):
    release = _corpus(tmp_path)
    expected = resolve_local_corpus_source("us/statute/42/9", release)

    for clone in (
        copy.copy(release),
        copy.deepcopy(release),
        pickle.loads(pickle.dumps(release)),
        dataclasses.replace(release),
    ):
        assert clone == release
        assert _cache(clone) is not _cache(release)
        assert not _cache(clone).buckets
        assert resolve_local_corpus_source("us/statute/42/9", clone) == expected
    assert "_resolution_cache" not in repr(release)
    assert "_resolution_cache" not in dataclasses.asdict(release)


def test_first_read_failure_is_not_cached(tmp_path: Path):
    release = _corpus(tmp_path)
    statute = _provisions_path(tmp_path, "statute", STATUTE)
    original = statute.read_bytes()
    statute.write_bytes(original.replace(b"Tax imposed.", b"Tax removed."))

    for _ in range(2):
        with pytest.raises(CorpusResolutionError, match="do not match the verified"):
            resolve_local_corpus_source("us/statute/26/1", release)
    assert not _cache(release).buckets

    statute.write_bytes(original)
    assert resolve_local_corpus_source("us/statute/26/1", release).body == (
        "Tax imposed."
    )


def test_cached_rows_are_the_verified_bytes_and_disk_checks_still_rehash(
    tmp_path: Path,
):
    release = _corpus(tmp_path)
    before = resolve_local_corpus_source("us/statute/26/1", release)
    composed = resolve_local_corpus_source("us/statute/42/9", release)
    assert resolve_local_corpus_dependency_artifacts("us/statute/42/9", release)

    statute = _provisions_path(tmp_path, "statute", STATUTE)
    statute.write_bytes(statute.read_bytes().replace(b"Tax imposed.", b"Tax removed."))

    # The release object still attests the original bytes, so cached rows are
    # exactly what an uncached read verified; nothing reads the tampered file.
    assert resolve_local_corpus_source("us/statute/26/1", release) == before
    assert resolve_local_corpus_source("us/statute/42/9", release) == composed
    # A fresh read and the on-disk dependency check both fail closed.
    with pytest.raises(CorpusResolutionError, match="do not match the verified"):
        resolve_local_corpus_source("us/statute/26/1", _fresh(release))
    for identifier in ("us/statute/26/1", "us/statute/42/9"):
        with pytest.raises(CorpusResolutionError, match="changed after resolution"):
            resolve_local_corpus_dependency_artifacts(identifier, release)

    statute.unlink()
    assert resolve_local_corpus_source("us/statute/26/1", release) == before
    with pytest.raises(CorpusResolutionError, match="changed after resolution"):
        resolve_local_corpus_dependency_artifacts("us/statute/26/1", release)


@pytest.mark.parametrize(
    ("limits", "message"),
    [
        ({"MAX_LOCAL_CORPUS_FILES": 0}, "file safety limit"),
        ({"MAX_LOCAL_CORPUS_AGGREGATE_BYTES": 1}, "byte aggregate safety limit"),
        ({"MAX_LOCAL_CORPUS_ROWS": 1}, "1-row safety limit"),
        ({"MAX_LOCAL_CORPUS_ROWS": len(STATUTE_ROWS)}, "row aggregate safety limit"),
        ({"MAX_CORPUS_PROVISION_BYTES": 10}, "10-byte safety limit"),
        # Several limits at once: the first check an uncached read applies wins.
        (
            {"MAX_CORPUS_PROVISION_BYTES": 10, "MAX_LOCAL_CORPUS_FILES": 0},
            "10-byte safety limit",
        ),
        (
            {"MAX_LOCAL_CORPUS_FILES": 0, "MAX_LOCAL_CORPUS_ROWS": 1},
            "file safety limit",
        ),
        # The statute bucket fails its row check, the one-row regulation
        # bucket its byte charge; each call still matches its cold outcome.
        (
            {"MAX_LOCAL_CORPUS_ROWS": 1, "MAX_LOCAL_CORPUS_AGGREGATE_BYTES": 1},
            "safety limit",
        ),
    ],
)
def test_read_limits_apply_to_every_call_on_a_warm_cache(
    tmp_path: Path, monkeypatch, limits: dict[str, int], message: str
):
    release = _corpus(tmp_path)
    resolve_local_corpus_source("us/statute/26/1", release)
    with pytest.raises(CorpusSourceNotFoundError):
        resolve_local_corpus_source("us/statute/404", release)
    resolve_local_corpus_source("us/regulation/7/273/9", release)
    assert _outcome(lambda: list(iter_active_local_corpus_rows(release)))[0] == "error"

    for limit, value in limits.items():
        monkeypatch.setattr(corpus_resolver, limit, value)

    calls = (
        lambda target: resolve_local_corpus_source("us/statute/26/1", target),
        lambda target: resolve_local_corpus_source("us/statute/404", target),
        lambda target: resolve_local_corpus_source("us/regulation/7/273/9", target),
        lambda target: list(iter_active_local_corpus_rows(target)),
    )
    failures = []
    for call in calls:
        warm = _outcome(lambda call=call: call(release))
        # Identical to what an uncached read of the same bytes reports.
        assert warm == _outcome(lambda call=call: call(_fresh(release)))
        if warm[0] == "error" and "safety limit" in warm[2]:
            failures.append(warm[2])
    assert failures and all(message in failure for failure in failures)


def test_iteration_budget_spans_cached_buckets(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    statute_rows = len(STATUTE_ROWS)
    monkeypatch.setattr(corpus_resolver, "MAX_LOCAL_CORPUS_ROWS", statute_rows)
    resolve_local_corpus_source("us/statute/26/1", release)
    resolve_local_corpus_source("us/regulation/7/273/9", release)

    def iterate(target: LocalCorpusRelease):
        return _outcome(lambda: tuple(iter_active_local_corpus_rows(target)))

    cold = iterate(_fresh(release))
    assert cold[0] == "error" and "row aggregate safety limit" in cold[2]
    assert iterate(release) == cold


def test_iteration_over_cached_buckets_matches_a_fresh_release(tmp_path: Path):
    _write_rows(tmp_path, STATUTE, STATUTE_ROWS[:5] + STATUTE_ROWS[9:10])
    _write_rows(tmp_path, REGULATION, REGULATION_ROWS, document_class="regulation")
    release = bind_test_corpus_release(
        tmp_path,
        RELEASE,
        [("us", "statute", STATUTE), ("us", "regulation", REGULATION)],
    )
    for identifier in ("us/statute/26/1", "us/regulation/7/273/9"):
        resolve_local_corpus_source(identifier, release)

    for filters in ({}, {"document_class": "statute"}, {"jurisdiction": "us"}):
        assert tuple(iter_active_local_corpus_rows(release, **filters)) == tuple(
            iter_active_local_corpus_rows(_fresh(release), **filters)
        )


def test_cache_evicts_least_recently_used_buckets(
    tmp_path: Path, monkeypatch, provision_reads
):
    release = _corpus(tmp_path)
    statute = _provisions_path(tmp_path, "statute", STATUTE)
    regulation = _provisions_path(tmp_path, "regulation", REGULATION)
    monkeypatch.setattr(
        corpus_resolver,
        "MAX_CACHED_CORPUS_BYTES",
        max(_row_weight(statute), _row_weight(regulation)),
    )

    resolve_local_corpus_source("us/statute/26/1", release)
    resolve_local_corpus_source("us/regulation/7/273/9", release)
    assert list(_cache(release).buckets) == [("us", "regulation")]
    _assert_accounting(release)

    expected = resolve_local_corpus_source("us/statute/26/1", _fresh(release))
    assert resolve_local_corpus_source("us/statute/26/1", release) == expected
    assert list(_cache(release).buckets) == [("us", "statute")]
    assert len(provision_reads) == 4
    _assert_accounting(release)


def test_cache_bounds_bucket_count(tmp_path: Path, monkeypatch, provision_reads):
    release = _corpus(tmp_path)
    monkeypatch.setattr(corpus_resolver, "MAX_CACHED_CORPUS_BUCKETS", 2)

    for identifier in ("us/statute/26/1", "us/regulation/7/273/9", "us/form/1040"):
        _outcome(
            lambda identifier=identifier: resolve_local_corpus_source(
                identifier, release
            )
        )
    assert list(_cache(release).buckets) == [("us", "regulation"), ("us", "form")]
    resolve_local_corpus_source("us/regulation/7/273/9", release)
    assert list(_cache(release).buckets) == [("us", "form"), ("us", "regulation")]
    _assert_accounting(release)


def test_iteration_evicts_as_it_loads(tmp_path: Path, monkeypatch):
    release = _clean_corpus(tmp_path)
    monkeypatch.setattr(corpus_resolver, "MAX_CACHED_CORPUS_BUCKETS", 1)

    assert tuple(iter_active_local_corpus_rows(release)) == tuple(
        iter_active_local_corpus_rows(_fresh(release))
    )
    assert list(_cache(release).buckets) == [("us", "statute")]
    _assert_accounting(release)


def test_an_oversized_bucket_is_kept_alone_with_its_memo(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    statute = _provisions_path(tmp_path, "statute", STATUTE)
    monkeypatch.setattr(
        corpus_resolver, "MAX_CACHED_CORPUS_BYTES", _row_weight(statute) // 2
    )
    expected = {
        identifier: _outcome(
            lambda identifier=identifier: resolve_local_corpus_source(
                identifier, _fresh(release)
            )
        )
        for identifier in IDENTIFIERS
    }

    for identifier, outcome in expected.items():
        assert (
            _outcome(
                lambda identifier=identifier: resolve_local_corpus_source(
                    identifier, release
                )
            )
            == outcome
        )
        cache = _cache(release)
        bound = corpus_resolver.MAX_CACHED_CORPUS_BYTES
        assert cache.row_bytes <= bound or len(cache.buckets) == 1
        if identifier.lstrip().startswith("us/statute"):
            assert list(cache.buckets) == [("us", "statute")]
        _assert_accounting(release)
    assert _cache(release).memo_bytes > 0


def test_memo_bound_clears_other_memos_before_its_own(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    statute_entry = _entry_weight("us/statute/26/1")
    monkeypatch.setattr(
        corpus_resolver, "MAX_CACHED_CORPUS_MEMO_BYTES", statute_entry + 10
    )

    resolve_local_corpus_source("us/regulation/7/273/9", release)
    resolve_local_corpus_source("us/statute/26/1", release)

    cache = _cache(release)
    assert list(cache.buckets) == [("us", "regulation"), ("us", "statute")]
    assert cache.buckets[("us", "regulation")].memo == {}
    assert cache.buckets[("us", "statute")].memo_bytes == statute_entry
    _assert_accounting(release)

    # An outcome larger than the whole bound is never stored.
    with pytest.raises(AmbiguousCorpusSourceError):
        resolve_local_corpus_source("us/statute/5/5", release)
    assert cache.buckets[("us", "statute")].memo_bytes == statute_entry
    # One that fits only after clearing this bucket's own memo restarts it and
    # is not stored into the restarted memo.
    resolve_local_corpus_source("us/statute/26/1 ", release)
    assert cache.buckets[("us", "statute")].memo == {}
    _assert_accounting(release)


def test_memo_stays_within_its_bound_and_outcomes_hold(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    monkeypatch.setattr(corpus_resolver, "MAX_CACHED_CORPUS_MEMO_BYTES", 3_000)
    expected = {
        identifier: _outcome(
            lambda identifier=identifier: resolve_local_corpus_source(
                identifier, _fresh(release)
            )
        )
        for identifier in IDENTIFIERS
    }
    order = list(IDENTIFIERS) * 3
    random.Random(5).shuffle(order)
    for identifier in order:
        assert (
            _outcome(lambda: resolve_local_corpus_source(identifier, release))
            == expected[identifier]
        )
        _assert_accounting(release)


def test_memo_weights_charge_identifiers_but_not_shared_rows(tmp_path: Path):
    release = _corpus(tmp_path)
    padded = " " * 4_000 + "us/statute/26/1"
    resolve_local_corpus_source(padded, release)
    composed = resolve_local_corpus_source("us/statute/42/9", release)

    statute = _cache(release).buckets[("us", "statute")]
    scope = ("us", "statute", STATUTE)
    composition = statute.memo[
        ("descendants", "us/statute/42/9", corpus_resolver.ReleaseScope(*scope))
    ]
    assert composition[1] is composed.component_rows
    # The padded identifier is charged; the composed parent's outcome is not
    # charged again for the body and rows its composition entry holds.
    assert statute.memo_bytes == (
        _entry_weight(padded)
        + _entry_weight("us/statute/42/9")
        + (_entry_weight("us/statute/42/9") + sys.getsizeof(composition[0]))
        + 512 * len(composition[1])
    )
    assert sys.getsizeof(padded) > 4_000
    _assert_accounting(release)


def test_concurrent_resolution_reads_once_and_agrees(tmp_path: Path, provision_reads):
    release = _corpus(tmp_path)
    expected = {
        identifier: _outcome(
            lambda identifier=identifier: resolve_local_corpus_source(
                identifier, _fresh(release)
            )
        )
        for identifier in IDENTIFIERS
    }
    provision_reads.clear()
    barrier = threading.Barrier(8)
    failures: list[str] = []

    def worker(seed: int) -> None:
        order = list(IDENTIFIERS) * 4
        random.Random(seed).shuffle(order)
        barrier.wait()
        for identifier in order:
            actual = _outcome(
                lambda identifier=identifier: resolve_local_corpus_source(
                    identifier, release
                )
            )
            if actual != expected[identifier]:
                failures.append(identifier)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert sorted(path.name for path in provision_reads) == sorted(
        [f"{STATUTE}.jsonl", f"{REGULATION}.jsonl"]
    )
    _assert_accounting(release)


class _LinearCitationIndex:
    """Reference lookups: the uncached resolver's full scan of every row."""

    def __init__(self, artifact) -> None:
        self._rows = artifact.rows

    @property
    def rows_by_citation(self):
        rows = self._rows

        class _Contains:
            def __contains__(self, citation_path: object) -> bool:
                return any(
                    record.get("citation_path") == citation_path for _, record in rows
                )

        return _Contains()

    def rows_for(self, citation_path: str):
        return [
            row for row in self._rows if row[1].get("citation_path") == citation_path
        ]

    def rows_under(self, prefix: str):
        return [
            row
            for row in self._rows
            if isinstance(row[1].get("citation_path"), str)
            and row[1]["citation_path"].startswith(prefix)
        ]


_SEGMENTS = ("1", "2", "10", "a", "ab", "b", "i", "ii", "x–y", "A")


def _random_corpus(root: Path, rng: random.Random) -> list[str]:
    paths = sorted(
        {
            "/".join(("us", "statute", *rng.choices(_SEGMENTS, k=rng.randint(1, 4))))
            for _ in range(40)
        }
    )
    rows: list[dict[str, object]] = []
    for path in paths:
        for _ in range(rng.choice((1, 1, 1, 2))):
            row: dict[str, object] = {"citation_path": path}
            roll = rng.random()
            if roll < 0.55:
                marker = path.rsplit("/", 1)[-1]
                row["body"] = f"({marker}) Text of {path}.\n(z) Tail."
            elif roll < 0.8:
                row["body"] = None
            elif roll < 0.88:
                row["body"] = None
                row["metadata"] = {"status": "repealed"}
            else:
                row["body"] = f"Retired {path}."
                row["version"] = "retired"
            if rng.random() < 0.3:
                row["ordinal"] = rng.randint(0, 5)
            rows.append(row)
    rows.append({"citation_path": 17, "body": "numeric path"})
    rows.append({"citation_path": None, "body": "null path"})
    rows.append({"citation_path": "us/statute/1/Bad Segment!", "body": "bad"})
    rng.shuffle(rows)
    _write_rows(root, STATUTE, rows)
    requested = set(paths)
    for path in paths:
        requested.add(f"{path}/{rng.choice(_SEGMENTS)}")
        requested.add(path.rsplit("/", 1)[0])
    return sorted(requested)


@pytest.mark.parametrize("seed", range(20))
def test_indexed_cached_resolution_matches_linear_uncached_reference(
    tmp_path: Path, monkeypatch, seed: int
):
    rng = random.Random(seed)
    identifiers = _random_corpus(tmp_path, rng)
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])

    with monkeypatch.context() as patch:
        patch.setattr(
            corpus_resolver._CorpusCitationIndex, "build", _LinearCitationIndex
        )
        expected = {
            (identifier, exact): _outcome(
                lambda identifier=identifier, exact=exact: resolve_local_corpus_source(
                    identifier, _fresh(release), _exact_only=exact
                )
            )
            for identifier in identifiers
            for exact in (False, True)
        }

    keys = list(expected) * 2
    rng.shuffle(keys)
    for identifier, exact in keys:
        actual = _outcome(
            lambda: resolve_local_corpus_source(identifier, release, _exact_only=exact)
        )
        assert actual == expected[(identifier, exact)], (identifier, exact)
    assert {value[0] for value in expected.values()} == {"ok", "error"}
    _assert_accounting(release)


def test_parent_composition_is_shared_by_every_falling_back_child(
    tmp_path: Path, monkeypatch
):
    rows: list[dict[str, object]] = [
        {"citation_path": "us/statute/hts", "body": None, "heading": "Schedule"},
        {"citation_path": "us/statute/hts/1/a", "body": "(a) Line a.", "ordinal": 1},
        {"citation_path": "us/statute/hts/1/b", "body": "(b) Line b.", "ordinal": 2},
        {"citation_path": "us/statute/crossed", "body": None},
        {"citation_path": "us/statute/crossed/a", "body": "kept"},
    ]
    _write_rows(tmp_path, STATUTE, rows)
    _write_rows(
        tmp_path,
        "2026-02-01-other",
        [{"citation_path": "us/statute/crossed/b", "body": "other scope"}],
    )
    release = bind_test_corpus_release(
        tmp_path,
        RELEASE,
        [("us", "statute", STATUTE), ("us", "statute", "2026-02-01-other")],
    )
    children = [f"us/statute/hts/{line}" for line in ("1/a", "1/b", "2", "9/z")] + [
        f"us/statute/crossed/{line}" for line in ("x", "y", "z/1")
    ]
    expected = {
        child: _outcome(
            lambda child=child: resolve_local_corpus_source(child, _fresh(release))
        )
        for child in children
    }
    assert "cross active release scopes" in expected["us/statute/crossed/x"][2]
    compositions: list[str] = []
    original = corpus_resolver._compose_descendant_text

    def counted(citation_path, records, **kwargs):
        compositions.append(citation_path)
        return original(citation_path, records, **kwargs)

    monkeypatch.setattr(corpus_resolver, "_compose_descendant_text", counted)

    for child in children * 2:
        assert (
            _outcome(lambda child=child: resolve_local_corpus_source(child, release))
            == expected[child]
        ), child
    assert sorted(compositions) == ["us/statute/crossed", "us/statute/hts"]
    _assert_accounting(release)


def test_dependency_check_hashes_each_artifact_once_per_call(
    tmp_path: Path, monkeypatch
):
    rows: list[dict[str, object]] = [
        {"citation_path": "us/statute/hts", "body": None},
        *(
            {"citation_path": f"us/statute/hts/{index:02d}", "body": f"Line {index}."}
            for index in range(30)
        ),
    ]
    statute = _write_rows(tmp_path, STATUTE, rows)
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])
    assert (
        len(resolve_local_corpus_source("us/statute/hts", release).component_rows) == 30
    )
    original = corpus_resolver.read_bounded_regular_file
    reads: list[str] = []

    def tracked(root, candidate, *, label, max_bytes, **kwargs):
        reads.append(label)
        return original(root, candidate, label=label, max_bytes=max_bytes, **kwargs)

    monkeypatch.setattr(corpus_resolver, "read_bounded_regular_file", tracked)

    for _ in range(2):
        assert resolve_local_corpus_dependency_artifacts("us/statute/hts", release) == (
            statute.resolve(),
        )
    assert reads == ["resolved corpus provision"] * 2

    statute.write_bytes(statute.read_bytes().replace(b"Line 29.", b"Line 99."))
    with pytest.raises(CorpusResolutionError, match="changed after resolution"):
        resolve_local_corpus_dependency_artifacts("us/statute/hts", release)


def _clean_corpus(root: Path) -> LocalCorpusRelease:
    """The fixture without the rows that make whole-bucket iteration fail."""

    _write_rows(root, STATUTE, STATUTE_ROWS[:5])
    _write_rows(root, REGULATION, REGULATION_ROWS, document_class="regulation")
    return bind_test_corpus_release(
        root,
        RELEASE,
        [("us", "statute", STATUTE), ("us", "regulation", REGULATION)],
    )


def test_iterated_metadata_cannot_alter_cached_rows(tmp_path: Path):
    release = _clean_corpus(tmp_path)
    expected = resolve_local_corpus_source(" us/statute/7/2014 ", _fresh(release))

    for row in iter_active_local_corpus_rows(release):
        if row.row.citation_path == "us/statute/7/2014":
            row.metadata["source_history"].append("Forged history.")
            row.metadata["source_history"][0] = "Forged amendment."

    assert resolve_local_corpus_source(" us/statute/7/2014 ", release) == expected
    assert "Forged" not in expected.proof_evidence_text
    assert tuple(iter_active_local_corpus_rows(release)) == tuple(
        iter_active_local_corpus_rows(_fresh(release))
    )


def test_memoized_failures_retain_no_caller_frames(tmp_path: Path):
    import gc
    import weakref

    release = _corpus(tmp_path)

    class Payload:
        pass

    def resolve_while_handling_another_error():
        payload = Payload()
        try:
            raise RuntimeError("the caller's own failure")
        except RuntimeError:
            for identifier in ("us/statute/10/1", "us/statute/404", "us/statute/5/5"):
                with pytest.raises(CorpusResolutionError):
                    resolve_local_corpus_source(identifier, release)
        return weakref.ref(payload)

    payload = resolve_while_handling_another_error()
    gc.collect()
    assert payload() is None
    assert len(_cache(release).buckets[("us", "statute")].memo) == 3


def test_replayed_failures_carry_no_stale_context_or_shared_state(tmp_path: Path):
    release = _corpus(tmp_path)
    try:
        raise RuntimeError("unrelated in-flight error")
    except RuntimeError:
        with pytest.raises(CorpusSourceNotFoundError):
            resolve_local_corpus_source("us/statute/404", release)
        with pytest.raises(CorpusRowStructureError) as first:
            resolve_local_corpus_source("us/statute/10/1", release)
    assert isinstance(first.value.__context__, CorpusResolutionError)
    first.value.__cause__.args = ("mutated cause",)
    first.value.add_note("caller note")

    with pytest.raises(CorpusSourceNotFoundError) as not_found:
        resolve_local_corpus_source("us/statute/404", release)
    with pytest.raises(CorpusSourceNotFoundError) as fresh_not_found:
        resolve_local_corpus_source("us/statute/404", _fresh(release))
    assert not_found.value.__context__ is None
    assert fresh_not_found.value.__context__ is None

    with pytest.raises(CorpusRowStructureError) as replayed:
        resolve_local_corpus_source("us/statute/10/1", release)
    cause = replayed.value.__cause__
    assert isinstance(cause, CorpusResolutionError)
    assert cause.__traceback__ is None
    assert str(cause) != "mutated cause"
    assert replayed.value.__context__ is cause
    assert replayed.value.__suppress_context__ is True
    assert not hasattr(replayed.value, "__notes__")
    with pytest.raises(CorpusRowStructureError) as uncached:
        resolve_local_corpus_source("us/statute/10/1", _fresh(release))
    assert str(cause) == str(uncached.value.__cause__)


def test_failures_outside_the_replayable_chain_are_recomputed(
    tmp_path: Path, monkeypatch
):
    release = _corpus(tmp_path)
    calls = []
    original = corpus_resolver._resolve_in_verified_bucket

    def failing(*args, **kwargs):
        calls.append(args[0])
        try:
            raise OSError("not a resolver failure")
        except OSError as exc:
            raise CorpusSourceNotFoundError("chained to an OS error") from exc

    monkeypatch.setattr(corpus_resolver, "_resolve_in_verified_bucket", failing)
    for _ in range(2):
        with pytest.raises(CorpusSourceNotFoundError) as info:
            resolve_local_corpus_source("us/statute/26/1", release)
        assert isinstance(info.value.__cause__, OSError)
    assert calls == ["us/statute/26/1"] * 2
    monkeypatch.setattr(corpus_resolver, "_resolve_in_verified_bucket", original)
    assert resolve_local_corpus_source("us/statute/26/1", release).body == (
        "Tax imposed."
    )


def _load_regulation_during(monkeypatch, release: LocalCorpusRelease) -> None:
    """Make the next statute composition touch the regulation bucket first."""

    original = corpus_resolver._compose_descendant_text

    def compose_after_other_bucket(citation_path, records, **kwargs):
        resolve_local_corpus_source("us/regulation/7/273/9", release)
        return original(citation_path, records, **kwargs)

    monkeypatch.setattr(
        corpus_resolver, "_compose_descendant_text", compose_after_other_bucket
    )


def test_outcomes_computed_in_an_evicted_bucket_are_not_accounted(
    tmp_path: Path, monkeypatch
):
    release = _corpus(tmp_path)
    expected = resolve_local_corpus_source("us/statute/42/9", _fresh(release))
    monkeypatch.setattr(corpus_resolver, "MAX_CACHED_CORPUS_BUCKETS", 1)
    _load_regulation_during(monkeypatch, release)

    assert resolve_local_corpus_source("us/statute/42/9", release) == expected
    assert list(_cache(release).buckets) == [("us", "regulation")]
    _assert_accounting(release)


def test_remembering_an_outcome_marks_its_bucket_recently_used(
    tmp_path: Path, monkeypatch
):
    release = _corpus(tmp_path)
    _load_regulation_during(monkeypatch, release)

    resolve_local_corpus_source("us/statute/42/9", release)
    # The regulation bucket was used during the composition, but remembering
    # the statute outcomes afterwards makes the statute bucket most recent.
    assert list(_cache(release).buckets) == [("us", "regulation"), ("us", "statute")]
    _assert_accounting(release)


def test_a_reentrant_duplicate_outcome_is_accounted_once(tmp_path: Path, monkeypatch):
    release = _corpus(tmp_path)
    original = corpus_resolver._resolve_in_verified_bucket
    reentered: list[str] = []

    def reentrant(identifier, *args, **kwargs):
        if not reentered:
            reentered.append(identifier)
            resolve_local_corpus_source(identifier, release)
        return original(identifier, *args, **kwargs)

    monkeypatch.setattr(corpus_resolver, "_resolve_in_verified_bucket", reentrant)
    resolve_local_corpus_source("us/statute/26/1", release)

    statute = _cache(release).buckets[("us", "statute")]
    assert statute.memo_bytes == _entry_weight("us/statute/26/1")
    _assert_accounting(release)


def test_with_fresh_reads_returns_an_equal_release_with_an_empty_cache(
    tmp_path: Path, provision_reads
):
    release = _corpus(tmp_path)
    resolve_local_corpus_source("us/statute/26/1", release)

    fresh = release.with_fresh_reads()

    assert fresh == release and fresh is not release
    assert not _cache(fresh).buckets and _cache(release).buckets
    assert resolve_local_corpus_source("us/statute/26/1", fresh) == (
        resolve_local_corpus_source("us/statute/26/1", release)
    )
    assert len(provision_reads) == 2


@pytest.mark.parametrize(
    ("limit", "value"),
    [
        ("MAX_COMPOSED_CORPUS_BYTES", 1),
        ("MAX_CORPUS_DESCENDANT_ROWS", 1),
        ("MAX_COMPOSITION_NODES", 1),
        ("MAX_LEGAL_MARKER_TOKEN_LENGTH", 0),
    ],
)
def test_limit_changes_invalidate_memoized_outcomes(
    tmp_path: Path, monkeypatch, limit: str, value: int
):
    release = _corpus(tmp_path)
    identifiers = ("us/statute/42/9", "us/statute/7/2014/e", "us/statute/26/1")
    for identifier in identifiers:
        resolve_local_corpus_source(identifier, release)

    monkeypatch.setattr(corpus_resolver, limit, value)

    outcomes = []
    for identifier in identifiers:
        warm = _outcome(lambda: resolve_local_corpus_source(identifier, release))
        assert warm == _outcome(
            lambda: resolve_local_corpus_source(identifier, _fresh(release))
        ), identifier
        outcomes.append(warm[0])
    assert "error" in outcomes


def test_failure_replays_inside_handlers_chain_like_uncached_raises(
    tmp_path: Path,
):
    release = _corpus(tmp_path)
    with pytest.raises(CorpusRowStructureError):
        resolve_local_corpus_source("us/statute/10/1", release)

    def raise_inside_handler(target: LocalCorpusRelease):
        try:
            raise RuntimeError("in flight")
        except RuntimeError as handled:
            with pytest.raises(CorpusRowStructureError) as info:
                resolve_local_corpus_source("us/statute/10/1", target)
            return handled, info.value

    def chain(handled: BaseException, error: BaseException) -> tuple[object, ...]:
        return (
            error.__context__ is error.__cause__,
            error.__cause__.__context__ is handled,
            str(error),
            str(error.__cause__),
        )

    assert chain(*raise_inside_handler(release)) == chain(
        *raise_inside_handler(_fresh(release))
    )

    replays = []
    for _ in range(2):
        with pytest.raises(CorpusRowStructureError) as info:
            resolve_local_corpus_source("us/statute/10/1", release)
        replays.append(info.value)
        info.value.add_note("caller note")
        info.value.__cause__.args = ("mutated cause",)
    first, second = replays
    assert second is not first and second.__cause__ is not first.__cause__
    assert second.__notes__ == ["caller note"]
    assert str(second.__cause__) == "mutated cause"
    with pytest.raises(CorpusRowStructureError) as third:
        resolve_local_corpus_source("us/statute/10/1", release)
    assert not hasattr(third.value, "__notes__")
    assert str(third.value.__cause__) != "mutated cause"


def test_deeply_nested_metadata_is_copied_without_recursion(tmp_path: Path):
    depth = 5_000
    nested = "[" * depth + '"leaf"' + "]" * depth
    row = (
        json.dumps(
            {
                "id": "row-deep",
                "jurisdiction": "us",
                "document_class": "statute",
                "version": STATUTE,
                "source_path": f"sources/us/statute/{STATUTE}/x.xml",
                "source_as_of": "2026-01-02",
                "expression_date": "2026-01-01",
                "citation_path": "us/statute/1/1",
                "body": "Deep metadata.",
            },
            sort_keys=True,
        )[:-1]
        + f', "metadata": {{"deep": {nested}}}}}\n'
    )
    path = _provisions_path(tmp_path, "statute", STATUTE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(row, encoding="utf-8")
    release = bind_test_corpus_release(tmp_path, RELEASE, [("us", "statute", STATUTE)])

    for _ in range(2):
        (active,) = tuple(iter_active_local_corpus_rows(release))
        value, levels = active.metadata["deep"], 0
        while isinstance(value, list):
            assert len(value) == 1
            value, levels = value[0], levels + 1
        assert (levels, value) == (depth, "leaf")
        active.metadata["deep"].append("mutated")


@pytest.mark.parametrize("direction", ["success-then-lowered", "failure-then-raised"])
def test_outcomes_computed_before_a_limit_change_are_not_stored(
    tmp_path: Path, monkeypatch, direction: str
):
    release = _corpus(tmp_path)
    resolve_local_corpus_source("us/statute/26/1", release)
    before, after = (
        (1_000_000, 1) if direction == "success-then-lowered" else (1, 1_000_000)
    )
    monkeypatch.setattr(corpus_resolver, "MAX_COMPOSED_CORPUS_BYTES", before)
    original = corpus_resolver._compose_descendant_text

    def compose_then_change_limits(citation_path, records, **kwargs):
        # The composition finishes under the old limit; another call observes
        # the new limit (clearing the memos) before this one can store it.
        try:
            return original(citation_path, records, **kwargs)
        finally:
            monkeypatch.setattr(corpus_resolver, "MAX_COMPOSED_CORPUS_BYTES", after)
            resolve_local_corpus_source("us/statute/26/1", release)

    monkeypatch.setattr(
        corpus_resolver, "_compose_descendant_text", compose_then_change_limits
    )
    _outcome(lambda: resolve_local_corpus_source("us/statute/42/9", release))
    monkeypatch.setattr(corpus_resolver, "_compose_descendant_text", original)

    assert _outcome(
        lambda: resolve_local_corpus_source("us/statute/42/9", release)
    ) == (
        _outcome(
            lambda: resolve_local_corpus_source("us/statute/42/9", _fresh(release))
        )
    )
    _assert_accounting(release)


def test_replays_never_share_list_state_of_a_memoized_failure(
    tmp_path: Path, monkeypatch
):
    release = _corpus(tmp_path)

    def failing_with_note(*args, **kwargs):
        error = CorpusSourceNotFoundError("annotated failure")
        error.add_note("recorded when the failure was computed")
        raise error

    monkeypatch.setattr(
        corpus_resolver, "_resolve_in_verified_bucket", failing_with_note
    )
    replays = []
    for _ in range(3):
        with pytest.raises(CorpusSourceNotFoundError) as info:
            resolve_local_corpus_source("us/statute/26/1", release)
        replays.append(info.value)
        info.value.add_note("added by one caller")

    assert [error.__notes__ for error in replays] == [
        ["recorded when the failure was computed", "added by one caller"]
    ] * 3

"""Focused tests for release-aware corpus source resolution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_resolver
from axiom_encode.corpus_resolver import (
    AmbiguousCorpusSourceError,
    CorpusDescendantStructureError,
    CorpusLayoutError,
    CorpusResolutionError,
    CorpusSourceNotFoundError,
    CorpusSourceSliceError,
    InvalidActiveCorpusSourceError,
    InvalidCorpusCitationError,
    InvalidCorpusReleaseError,
    LocalCorpusRelease,
    UnsafeCorpusPathError,
    iter_active_local_corpus_rows,
    normalize_corpus_identifier,
    require_canonical_corpus_citation_path,
    resolve_local_corpus_source,
    resolve_scoped_local_corpus_source,
    scope_resolved_corpus_source,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
    write_test_release_object,
)

CITATION = "us/statute/7/2014/e"
TEST_RELEASE = "test-release"


def _write_selector(
    root: Path,
    scopes: list[dict[str, str]],
    *,
    name: str = TEST_RELEASE,
) -> Path:
    path = root / "manifests" / "releases" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "description": "test active corpus scopes",
                "name": name,
                "scopes": scopes,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _release(root: Path, *, name: str = TEST_RELEASE) -> LocalCorpusRelease:
    selector_path = root / "manifests" / "releases" / f"{name}.json"
    if not selector_path.exists():
        return LocalCorpusRelease(root, name, "0" * 64, TEST_RELEASE_PUBLIC_KEY)
    payload = json.loads(selector_path.read_text(encoding="utf-8"))
    scopes = [
        (
            str(scope["jurisdiction"]),
            str(scope["document_class"]),
            str(scope["version"]),
        )
        for scope in payload["scopes"]
    ]
    return bind_test_corpus_release(root, name, scopes)


def _scope(version: str) -> dict[str, str]:
    return {
        "jurisdiction": "us",
        "document_class": "statute",
        "version": version,
    }


def test_existing_checkout_without_provisions_is_structural_error(tmp_path):
    with pytest.raises(CorpusLayoutError, match="Canonical data/corpus/provisions"):
        _release(tmp_path)


def test_local_resolver_rejects_unbound_root(tmp_path: Path):
    with pytest.raises(TypeError, match="validated LocalCorpusRelease"):
        resolve_local_corpus_source(CITATION, tmp_path)  # type: ignore[arg-type]


def test_machine_identity_requires_exact_canonical_corpus_path():
    assert require_canonical_corpus_citation_path(CITATION) == CITATION

    for alias in (
        " us/statute/7/2014/e ",
        "us//statute/7/2014/e",
        "us:statutes/7/2014/e",
        "7 USC 2014(e)",
        "us/statutes/7/2014/e",
    ):
        with pytest.raises(InvalidCorpusCitationError, match="exact canonical"):
            require_canonical_corpus_citation_path(alias)


@pytest.mark.parametrize(
    "citation_path",
    (
        "nz/district-plan",
        "nz/district-plan/wellington-city/2024/muz/r13",
    ),
)
def test_machine_identity_accepts_district_plan_paths(citation_path: str):
    assert require_canonical_corpus_citation_path(citation_path) == citation_path
    assert normalize_corpus_identifier(citation_path) == citation_path


def test_machine_identity_still_rejects_unknown_document_classes():
    with pytest.raises(InvalidCorpusCitationError, match="document-class"):
        require_canonical_corpus_citation_path("nz/zoning-code/wellington-city/2024")


def test_human_citation_normalization_remains_available():
    assert normalize_corpus_identifier("7 USC 2014(e)") == CITATION


def _write_rows(
    root: Path,
    version: str,
    rows: list[dict[str, object]],
    *,
    filename: str | None = None,
    document_class: str = "statute",
) -> Path:
    path = (
        root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / document_class
        / (filename or f"{version}.jsonl")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for row in rows:
        normalized.append(
            {
                "id": f"row-{version}",
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
                "source_path": f"sources/us/{document_class}/{version}/source.xml",
                "source_as_of": "2026-01-02",
                "expression_date": "2026-01-01",
                **row,
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in normalized),
        encoding="utf-8",
    )
    return path


def _minimal_release(root: Path) -> LocalCorpusRelease:
    version = "test-version"
    _write_selector(root, [_scope(version)])
    _write_rows(
        root,
        version,
        [{"citation_path": CITATION, "body": "test body"}],
    )
    return _release(root)


def test_resolves_one_active_row_with_complete_identity_and_hashes(tmp_path: Path):
    version = "2026-01-02-title-7"
    _write_selector(tmp_path, [_scope(version)])
    body = "The standard deduction is $198."
    provision_file = _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": body}],
    )

    release = _release(tmp_path)
    resolved = resolve_local_corpus_source(CITATION, release)

    body_sha = hashlib.sha256(body.encode()).hexdigest()
    file_sha = hashlib.sha256(provision_file.read_bytes()).hexdigest()
    assert resolved.requested == CITATION
    assert resolved.citation_path == CITATION
    assert resolved.body == body
    assert resolved.stored_body_sha256 == body_sha
    assert resolved.resolved_text_sha256 == body_sha
    assert resolved.provision_file == (
        f"data/corpus/provisions/us/statute/{version}.jsonl"
    )
    assert resolved.provision_file_sha256 == file_sha
    assert resolved.release_name == TEST_RELEASE
    assert resolved.release_content_sha256 == release.content_sha256
    assert resolved.release_selector_sha256 == release.selector_sha256
    assert resolved.row.line_number == 1
    assert resolved.row.record_id == f"row-{version}"
    assert resolved.row.source_as_of == "2026-01-02"
    assert resolved.row.expression_date == "2026-01-01"
    assert resolved.row.body_sha256 == body_sha
    assert resolved.component_rows == ()
    attestation = resolved.to_attestation()
    assert attestation["source_sha256"] == body_sha
    assert attestation["resolved_text_sha256"] == body_sha
    assert attestation["row"]["version"] == version
    assert attestation["corpus_release_content_sha256"] == release.content_sha256
    assert attestation["corpus_release_selector_sha256"] == release.selector_sha256


def test_resolver_exposes_source_history_only_as_proof_evidence(tmp_path: Path):
    version = "2026-01-02-title-7"
    _write_selector(tmp_path, [_scope(version)])
    body = "The credit is available."
    history = "Acts 2007, No. 278, section 1, effective January 1, 2008."
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": CITATION,
                "body": body,
                "metadata": {"source_history": [history]},
            }
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    body_sha = hashlib.sha256(body.encode()).hexdigest()
    assert resolved.body == body
    assert resolved.stored_body_sha256 == body_sha
    assert resolved.resolved_text_sha256 == body_sha
    assert resolved.source_history == (history,)
    assert resolved.proof_evidence_segments == (body, history)
    assert "source_history" not in resolved.to_attestation()


def test_scoped_parent_slice_drops_parent_source_history(tmp_path: Path):
    version = "2026-01-02-title-7"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/2014"
    body = "(a) First rule.\n(e) Second rule."
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": body,
                "metadata": {"source_history": ["History for the parent provision."]},
            }
        ],
    )
    release = _release(tmp_path)
    source = resolve_local_corpus_source(parent, release)

    scoped = scope_resolved_corpus_source(source, CITATION)

    assert scoped.body == "(e) Second rule."
    assert scoped.source_history == ()
    assert scoped.proof_evidence_text == scoped.body
    assert scoped.selected_record.body == scoped.body
    assert (
        scoped.selected_record
        == resolve_local_corpus_source(CITATION, release).selected_record
    )


def test_active_scope_beats_newer_inactive_duplicate(tmp_path: Path):
    active = "2025-01-01-active"
    inactive = "2026-01-01-inactive"
    _write_selector(tmp_path, [_scope(active)])
    _write_rows(
        tmp_path,
        active,
        [{"citation_path": CITATION, "body": "active body"}],
    )
    _write_rows(
        tmp_path,
        inactive,
        [{"citation_path": CITATION, "body": "newer inactive body"}],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "active body"
    assert resolved.row.version == active


def test_release_rejects_noncanonical_provisions_artifact_filename(tmp_path: Path):
    version = "2026-04-29"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "active split artifact"}],
        filename="2026-05-10-snap-sections.jsonl",
    )

    with pytest.raises(AssertionError, match="canonical provisions artifact"):
        _release(tmp_path)


def test_inactive_only_citation_is_not_found(tmp_path: Path):
    active = "2025-01-01-active"
    inactive = "2026-01-01-inactive"
    _write_selector(tmp_path, [_scope(active)])
    _write_rows(
        tmp_path,
        active,
        [{"citation_path": "us/statute/7/other", "body": "other"}],
    )
    _write_rows(
        tmp_path,
        inactive,
        [{"citation_path": CITATION, "body": "inactive body"}],
    )

    with pytest.raises(CorpusSourceNotFoundError, match="No active corpus source"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_active_row_iterator_filters_inactive_scopes(tmp_path: Path):
    active = "2025-01-01-active"
    inactive = "2026-01-01-inactive"
    _write_selector(tmp_path, [_scope(active)])
    _write_rows(
        tmp_path,
        active,
        [{"citation_path": CITATION, "body": "active body"}],
    )
    _write_rows(
        tmp_path,
        inactive,
        [
            {
                "citation_path": "us/statute/7/inactive",
                "body": "inactive body",
            }
        ],
    )

    release = _release(tmp_path)
    rows = list(iter_active_local_corpus_rows(release, jurisdiction="us"))

    assert [(row.row.citation_path, row.body) for row in rows] == [
        (CITATION, "active body")
    ]
    assert rows[0].release_name == TEST_RELEASE
    assert rows[0].release_content_sha256 == release.content_sha256
    assert rows[0].release_selector_sha256 == release.selector_sha256


def test_active_row_iterator_rejects_duplicate_active_citations(tmp_path: Path):
    first = "2025-01-01-first"
    second = "2026-01-01-second"
    _write_selector(tmp_path, [_scope(first), _scope(second)])
    _write_rows(
        tmp_path,
        first,
        [{"citation_path": CITATION, "body": "first"}],
    )
    _write_rows(
        tmp_path,
        second,
        [{"citation_path": CITATION, "body": "second"}],
    )

    with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
        iter_active_local_corpus_rows(_release(tmp_path))

    assert {row.version for row in exc_info.value.rows} == {first, second}


@pytest.mark.parametrize("bodies", [("first", "second"), ("same", "same")])
def test_rejects_multiple_active_rows_even_when_bodies_match(
    tmp_path: Path, bodies: tuple[str, str]
):
    first = "2025-01-01-first"
    second = "2026-01-01-second"
    _write_selector(tmp_path, [_scope(first), _scope(second)])
    _write_rows(
        tmp_path,
        first,
        [{"citation_path": CITATION, "body": bodies[0]}],
    )
    _write_rows(
        tmp_path,
        second,
        [{"citation_path": CITATION, "body": bodies[1]}],
    )

    with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
        resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert len(exc_info.value.rows) == 2
    assert {row.version for row in exc_info.value.rows} == {first, second}


def test_rejects_duplicate_rows_inside_one_active_file(tmp_path: Path):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": "same"},
            {"citation_path": CITATION, "body": "same"},
        ],
    )

    with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
        resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert [row.line_number for row in exc_info.value.rows] == [1, 2]


def test_rejects_active_release_row_missing_required_metadata(tmp_path: Path):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    path = _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    )
    row = json.loads(path.read_text())
    del row["source_as_of"]
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(CorpusResolutionError, match="missing required metadata"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_legacy_doc_type_alias(tmp_path: Path):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    path = _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    )
    row = json.loads(path.read_text())
    row["doc_type"] = row["document_class"]
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(CorpusResolutionError, match="Legacy doc_type"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_legacy_text_body_alias(tmp_path: Path):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    path = _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    )
    row = json.loads(path.read_text())
    row["text"] = row.pop("body")
    path.write_text(json.dumps(row) + "\n")

    with pytest.raises(CorpusResolutionError, match="Legacy text body"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_symlinked_provision_file(tmp_path: Path):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    real = tmp_path / "outside.jsonl"
    real.write_text("{}\n")
    target = (
        tmp_path
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{version}.jsonl"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(real)

    with pytest.raises(UnsafeCorpusPathError, match="symlink"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_unsafe_local_release_name_before_selector_lookup(tmp_path: Path):
    _write_rows(
        tmp_path,
        "would-have-loaded",
        [{"citation_path": CITATION, "body": "body"}],
    )
    escaped_selector = tmp_path / "manifests/evil.json"
    escaped_selector.parent.mkdir(parents=True)
    escaped_selector.write_text(
        json.dumps(
            {
                "name": "../evil",
                "scopes": [_scope("would-have-loaded")],
            }
        )
    )

    with pytest.raises(InvalidCorpusReleaseError, match="Unsafe corpus release"):
        _release(tmp_path, name="../evil")


@pytest.mark.parametrize(
    "name",
    ["nz.rulespec", "nz_rulespec", "nz--rulespec", "n" * 129],
)
def test_rejects_noncanonical_local_release_name(tmp_path: Path, name: str):
    with pytest.raises(InvalidCorpusReleaseError, match="Corpus release names"):
        _release(tmp_path, name=name)


def test_rejects_provision_file_over_size_limit(tmp_path: Path, monkeypatch):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    )
    monkeypatch.setattr(corpus_resolver, "MAX_CORPUS_PROVISION_BYTES", 1)

    with pytest.raises(UnsafeCorpusPathError, match="safety limit"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_unattested_extra_provision_files_are_not_scanned(tmp_path: Path):
    version = "2026-01-01-file-count"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "first"}],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": f"{CITATION}/other", "body": "second"}],
        filename="second.jsonl",
    )
    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "first"


def test_rejects_local_aggregate_byte_limit(tmp_path: Path, monkeypatch):
    version = "2026-01-01-aggregate-bytes"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    )
    monkeypatch.setattr(corpus_resolver, "MAX_LOCAL_CORPUS_AGGREGATE_BYTES", 1)

    with pytest.raises(CorpusResolutionError, match="aggregate safety limit"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_local_aggregate_row_limit(tmp_path: Path, monkeypatch):
    version = "2026-01-01-aggregate-rows"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": "body"},
            {"citation_path": f"{CITATION}/other", "body": "other"},
        ],
    )
    monkeypatch.setattr(corpus_resolver, "MAX_LOCAL_CORPUS_ROWS", 1)

    with pytest.raises(CorpusResolutionError, match="1-row safety limit"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_named_local_release_requires_release_object(tmp_path: Path):
    _write_rows(
        tmp_path,
        "legacy",
        [{"citation_path": CITATION, "body": "legacy body"}],
    )

    with pytest.raises(InvalidCorpusReleaseError, match="release object not found"):
        _release(tmp_path)


@pytest.mark.parametrize(
    ("padding_scopes", "minimum_mib"),
    [
        # Just past the former 16 MiB cap, which rejected every wave-4 union.
        (400, 16),
        # A union of about 30 MiB pretty-printed must still load.
        (650, 30),
    ],
)
def test_release_object_larger_than_former_cap_loads_and_resolves(
    tmp_path: Path, padding_scopes: int, minimum_mib: int
):
    version = "2026-01-01-active"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "inside"}],
    )
    content_sha256 = write_test_release_object(
        tmp_path,
        TEST_RELEASE,
        [("us", "statute", version)],
        unmaterialized_scopes=[
            ("us-zz", "manual", f"2026-01-01-padding-{index:04d}")
            for index in range(padding_scopes)
        ],
        unmaterialized_sources_per_scope=100,
    )
    release_object = tmp_path / "releases" / TEST_RELEASE / f"{content_sha256}.json"
    size = release_object.stat().st_size
    assert minimum_mib * 1024 * 1024 < size <= corpus_resolver.MAX_RELEASE_OBJECT_BYTES

    release = LocalCorpusRelease(
        tmp_path,
        TEST_RELEASE,
        content_sha256,
        TEST_RELEASE_PUBLIC_KEY,
    )

    assert len(release.scopes) == padding_scopes + 1
    assert resolve_local_corpus_source(CITATION, release).body == "inside"


def test_release_object_over_size_limit_is_rejected_before_read(
    tmp_path: Path, monkeypatch
):
    _write_rows(
        tmp_path,
        "2026-01-01-active",
        [{"citation_path": CITATION, "body": "inside"}],
    )
    content_sha256 = "0" * 64
    release_object = tmp_path / "releases" / TEST_RELEASE / f"{content_sha256}.json"
    release_object.parent.mkdir(parents=True)
    with release_object.open("wb") as stream:
        stream.truncate(corpus_resolver.MAX_RELEASE_OBJECT_BYTES + 1)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("an oversized release object must not be read")

    monkeypatch.setattr(corpus_resolver, "read_bounded_regular_file", unexpected_read)

    with pytest.raises(
        UnsafeCorpusPathError,
        match=f"exceeds the {corpus_resolver.MAX_RELEASE_OBJECT_BYTES}-byte safety",
    ):
        LocalCorpusRelease(
            tmp_path,
            TEST_RELEASE,
            content_sha256,
            TEST_RELEASE_PUBLIC_KEY,
        )


def test_release_object_growing_past_size_limit_after_check_fails_closed(
    tmp_path: Path, monkeypatch
):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "inside"}],
    )
    release = _release(tmp_path)
    monkeypatch.setattr(
        corpus_resolver,
        "MAX_RELEASE_OBJECT_BYTES",
        release.release_object_path.stat().st_size,
    )
    original = corpus_resolver._safe_file

    def grow_after_check(root, candidate, *, label, max_bytes):
        resolved = original(root, candidate, label=label, max_bytes=max_bytes)
        if label == "corpus release object" and resolved is not None:
            with resolved.open("ab") as stream:
                stream.write(b" ")
        return resolved

    monkeypatch.setattr(corpus_resolver, "_safe_file", grow_after_check)

    with pytest.raises(UnsafeCorpusPathError, match="byte safety limit"):
        LocalCorpusRelease(
            tmp_path,
            TEST_RELEASE,
            release.content_sha256,
            TEST_RELEASE_PUBLIC_KEY,
        )


def test_bounded_read_rejects_oversized_file_before_reading(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "release-object.json"
    target.write_bytes(b"x" * 17)

    def unexpected_read(descriptor, size):
        raise AssertionError("an oversized file must not be read")

    monkeypatch.setattr(os, "read", unexpected_read)

    with pytest.raises(UnsafeCorpusPathError, match="16-byte safety limit"):
        corpus_resolver.read_bounded_regular_file(
            tmp_path,
            target,
            label="corpus release object",
            max_bytes=16,
        )


def test_bounded_read_stops_at_limit_when_file_grows_after_fstat(
    tmp_path: Path, monkeypatch
):
    target = tmp_path / "release-object.json"
    target.write_bytes(b"x" * 16)
    real_fstat = os.fstat
    real_read = os.read
    returned: list[int] = []

    def fstat_then_grow(descriptor):
        result = real_fstat(descriptor)
        if os.path.samestat(result, os.stat(target)):
            with target.open("ab") as stream:
                stream.write(b"y" * (4 * 1024 * 1024))
        return result

    def counting_read(descriptor, size):
        chunk = real_read(descriptor, size)
        returned.append(len(chunk))
        return chunk

    monkeypatch.setattr(os, "fstat", fstat_then_grow)
    monkeypatch.setattr(os, "read", counting_read)

    with pytest.raises(UnsafeCorpusPathError, match="16-byte safety limit"):
        corpus_resolver.read_bounded_regular_file(
            tmp_path,
            target,
            label="corpus release object",
            max_bytes=16,
        )
    assert sum(returned) == 17


def test_does_not_invent_compact_uk_alias_for_corpus_path(tmp_path: Path):
    version = "2026-06-03-uk"
    expanded = "uk/statute/legislation.gov.uk/ukpga/2007/3/section/11d"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "uk",
                "document_class": "statute",
                "version": version,
            }
        ],
    )
    path = tmp_path / f"data/corpus/provisions/uk/statute/{version}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "uk-row",
                "jurisdiction": "uk",
                "document_class": "statute",
                "version": version,
                "citation_path": expanded,
                "body": "Savings income is charged.",
                "source_path": "sources/uk/statute/source.xml",
                "source_as_of": "2026-06-03",
                "expression_date": "2026-04-06",
            }
        )
        + "\n"
    )

    with pytest.raises(CorpusSourceNotFoundError, match="No active corpus source"):
        resolve_local_corpus_source("uk/statute/ukpga/2007/3/11D", _release(tmp_path))


def test_parent_fallback_slices_child_and_hashes_full_parent(tmp_path: Path):
    version = "2026-01-01-parent"
    _write_selector(tmp_path, [_scope(version)])
    parent_body = "(a) First.\n(e) Target $198.\n(f) Sibling $999."
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": "us/statute/7/2014",
                "body": parent_body,
                "metadata": {"source_history": ["History for the parent provision."]},
            }
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))
    assert resolved.citation_path == "us/statute/7/2014"
    assert resolved.body == "(e) Target $198."
    assert resolved.source_history == ()
    assert resolved.proof_evidence_text == resolved.body
    assert (
        resolved.stored_body_sha256 == hashlib.sha256(parent_body.encode()).hexdigest()
    )
    assert (
        resolved.resolved_text_sha256
        == hashlib.sha256(resolved.body.encode()).hexdigest()
    )


def test_resolver_owned_generation_scope_updates_exact_input_digest(tmp_path: Path):
    version = "2026-01-01-scope"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": "us/statute/7/2014",
                "body": "(a) First.\n(e) Target $198.\n(f) Sibling $999.",
            }
        ],
    )
    resolved = resolve_local_corpus_source("us/statute/7/2014", _release(tmp_path))

    scoped = scope_resolved_corpus_source(resolved, "us:statutes/7/2014/e")

    assert scoped.body == "(e) Target $198."
    assert scoped.requested == "us/statute/7/2014/e"
    assert scoped.citation_path == "us/statute/7/2014"
    assert scoped.slice_required is True
    assert (
        scoped.resolved_text_sha256 == hashlib.sha256(scoped.body.encode()).hexdigest()
    )


def test_scoped_resolver_rejects_raw_rulespec_identifier(tmp_path: Path):
    version = "2026-01-01-scope"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": "us/statute/7/2014", "body": "(e) Target."}],
    )
    parent = resolve_local_corpus_source("us/statute/7/2014", _release(tmp_path))

    with pytest.raises(InvalidCorpusCitationError, match="normalized"):
        resolve_scoped_local_corpus_source(
            parent, "us:statutes/7/2014/e", _release(tmp_path)
        )


def test_scoped_resolver_fails_when_exact_child_diverges_from_parent(tmp_path: Path):
    version = "2026-01-01-scope"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {
                "id": "parent",
                "citation_path": "us/statute/7/2014",
                "body": "(e) Parent.",
            },
            {"id": "child", "citation_path": CITATION, "body": "Exact child."},
        ],
    )
    parent = resolve_local_corpus_source("us/statute/7/2014", _release(tmp_path))

    with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
        resolve_scoped_local_corpus_source(parent, CITATION, _release(tmp_path))

    assert {row.record_id for row in exc_info.value.rows} == {"parent", "child"}


def test_scoped_resolver_uses_parent_slice_only_when_exact_child_absent(tmp_path: Path):
    version = "2026-01-01-scope"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": "us/statute/7/2014", "body": "(a) Other.\n(e) Target."}],
    )
    parent = resolve_local_corpus_source("us/statute/7/2014", _release(tmp_path))

    scoped = resolve_scoped_local_corpus_source(parent, CITATION, _release(tmp_path))

    assert scoped.body == "(e) Target."


def test_scoped_resolver_rejects_mismatched_bound_release(tmp_path: Path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    parent_path = "us/statute/7/2014"
    _write_selector(first_root, [_scope("first-version")])
    _write_rows(
        first_root,
        "first-version",
        [{"citation_path": parent_path, "body": "(e) First."}],
    )
    _write_selector(second_root, [_scope("second-version")])
    _write_rows(
        second_root,
        "second-version",
        [{"citation_path": parent_path, "body": "(e) Second."}],
    )
    parent = resolve_local_corpus_source(parent_path, _release(first_root))

    with pytest.raises(InvalidCorpusReleaseError, match="identities do not match"):
        resolve_scoped_local_corpus_source(parent, CITATION, _release(second_root))


def test_guidance_parent_is_not_a_generation_fallback(tmp_path: Path):
    version = "2026-01-01-guidance"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": "guidance",
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": "us/guidance/irs/example",
                "body": "Page 1. Parent guidance.",
            }
        ],
        document_class="guidance",
    )

    with pytest.raises(CorpusSourceNotFoundError):
        resolve_local_corpus_source(
            "us/guidance/irs/example/page-1", _release(tmp_path)
        )


def test_generic_parent_slice_bounds_cumulative_work_across_depth(
    tmp_path: Path,
    monkeypatch,
):
    version = "2026-01-01-generic-work-budget"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "uk",
                "document_class": "statute",
                "version": version,
            }
        ],
    )
    parent = "uk/statute/example/section"
    depth = 10
    body = "(1)" * depth + " " + "x" * 256
    path = tmp_path / f"data/corpus/provisions/uk/statute/{version}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "generic-parent",
                "jurisdiction": "uk",
                "document_class": "statute",
                "version": version,
                "citation_path": parent,
                "body": body,
                "source_path": "sources/uk/statute/example.xml",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        corpus_resolver,
        "MAX_GENERIC_PARENT_SLICE_CHARACTER_WORK",
        2_000,
    )
    requested = "/".join((parent, *("1" for _unused in range(depth))))

    with pytest.raises(CorpusSourceSliceError, match="character-work safety limit"):
        resolve_local_corpus_source(requested, _release(tmp_path))


def test_parent_slice_ignores_cfr_through_references(tmp_path: Path):
    version = "2026-01-01-cfr"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
            }
        ],
    )
    citation = "us/regulation/42/435/601"
    parent_body = (
        "(d) Use of less restrictive methodologies.\n\n"
        "(1) At State option, and subject to the conditions of "
        "paragraphs (d)(2) through (5) of this section, the agency "
        "may apply less restrictive methodologies.\n\n"
        "(2) The methodologies may be less restrictive but no more "
        "restrictive than SSI methodologies.\n\n"
        "(3) A methodology is no more restrictive if additional "
        "individuals may be eligible and none are made ineligible.\n\n"
        "(4) The methodology must be comparable within each category.\n\n"
        "(5) The methodology must be consistent with subpart K FFP "
        "limitations."
    )
    path = tmp_path / f"data/corpus/provisions/us/regulation/{version}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "cfr-row",
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
                "citation_path": citation,
                "body": parent_body,
                "source_path": "sources/us/regulation/cfr.xml",
                "source_as_of": "2026-01-02",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    paragraph_one = resolve_local_corpus_source(f"{citation}/d/1", _release(tmp_path))
    paragraph_five = resolve_local_corpus_source(f"{citation}/d/5", _release(tmp_path))

    assert paragraph_one.body.startswith("(1) At State option")
    assert "paragraphs (d)(2) through (5)" in paragraph_one.body
    assert "(2) The methodologies may be" not in paragraph_one.body
    assert paragraph_five.body == (
        "(5) The methodology must be consistent with subpart K FFP limitations."
    )


def test_parent_slice_preserves_inline_cfr_hierarchy(tmp_path: Path):
    version = "2026-01-01-cfr"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
            }
        ],
    )
    citation = "us/regulation/42/435/602"
    parent_body = (
        "(a)(1) First paragraph.\n\n"
        "(2) Second paragraph.\n\n"
        "(i) First clause.\n\n"
        "(ii) Second clause.\n\n"
        "(b) Next subsection."
    )
    path = tmp_path / f"data/corpus/provisions/us/regulation/{version}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "cfr-inline-row",
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
                "citation_path": citation,
                "body": parent_body,
                "source_path": "sources/us/regulation/cfr.xml",
                "source_as_of": "2026-01-02",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subsection = resolve_local_corpus_source(f"{citation}/a", _release(tmp_path))
    paragraph = resolve_local_corpus_source(f"{citation}/a/1", _release(tmp_path))
    clause = resolve_local_corpus_source(f"{citation}/a/2/i", _release(tmp_path))

    assert subsection.body.startswith("(a)(1) First paragraph.")
    assert "(b) Next subsection." not in subsection.body
    assert paragraph.body == "(1) First paragraph."
    assert clause.body == "(i) First clause."


@pytest.mark.parametrize(
    "reference_text",
    [
        "Eligibility is subject to conditions under (1) of this section.",
        "Eligibility is described in (1) of this section.",
        "Eligibility is subject to (1) of this section.",
        "Eligibility applies pursuant to (1) of this section.",
        "Eligibility applies in accordance with (1) of this section.",
        "Eligibility is specified at (1) of this section.",
        "Eligibility is provided by (1) of this section.",
        "Eligibility is defined in (1) of this section.",
        "Eligibility is subject to conditions in (1) of this section.",
        "See (1) for details.",
        "Eligibility is determined according to (1).",
        "Compare (1) and (2).",
        "See benefits (1) before proceeding.",
        "Compare rates (1) and (2).",
        "The rule provides benefits (1) in this case.",
        "For benefits (1) through (3), special rules apply.",
        "Consider benefits (1) before proceeding.",
        "The agency determines eligibility (1) in this case.",
        "Rules concerning income (1) do not apply.",
        "With respect to benefits (1) through (3), special rules apply.",
        "For purposes of eligibility (1) through (3), special rules apply.",
        "Notwithstanding income (1) through (3), special rules apply.",
        "In calculating benefits (1) through (3), special rules apply.",
        "Paragraph 42 (1) shall apply.",
        "Clause 42 (1) shall apply.",
        "Definitions § 42 (1) shall apply.",
        "In general, (1) shall apply.",
        "Paragraph (1) shall apply.",
        "Clause (1) shall apply.",
        "Item (1) shall apply.",
        "Definitions (1) and (2) apply.",
        "General rule (1) shall apply.",
        "In general (1) shall apply.",
        "Allowable financial resources (1) shall apply.",
        "First month benefits prorated (1) shall apply.",
        "Eligibility is determined elsewhere. (1) of section 5 shall apply.",
        "See the following rule. (1) shall apply.",
        "Compare the prior provision. (1) and (2) apply.",
        "See the following rule." + (" " * 60) + "(1) shall apply.",
    ],
)
def test_parent_slice_rejects_inline_unlabelled_reference(
    tmp_path: Path,
    reference_text: str,
):
    version = "2026-01-01-inline-reference"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/2014"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": f"(a) {reference_text}\n(b) A separate rule applies.",
            }
        ],
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_parent_fallback_fails_closed_when_child_marker_is_missing(tmp_path: Path):
    version = "2026-01-01-parent"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": "us/statute/7/2014", "body": "Undivided text."}],
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


@pytest.mark.parametrize(
    ("document_class", "parent", "child", "body", "expected"),
    [
        (
            "statute",
            "us/statute/7/9999/a",
            "1",
            "(1) First paragraph.\n\n(2) Second paragraph.",
            "(1) First paragraph.",
        ),
        (
            "statute",
            "us/statute/7/9999/a/1",
            "A",
            "(A) First subparagraph.\n\n(B) Second subparagraph.",
            "(A) First subparagraph.",
        ),
        (
            "regulation",
            "us/regulation/7/999/1/a",
            "1",
            "(1) First paragraph.\n\n(2) Second paragraph.",
            "(1) First paragraph.",
        ),
    ],
)
def test_canonical_hierarchy_slices_from_intermediate_parent_rows(
    tmp_path: Path,
    document_class: str,
    parent: str,
    child: str,
    body: str,
    expected: str,
):
    version = "2026-01-01-intermediate-parent"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
        document_class=document_class,
    )

    resolved = resolve_local_corpus_source(f"{parent}/{child}", _release(tmp_path))

    assert resolved.body == expected


def test_us_statute_hierarchy_accepts_inline_heading_child_markers(tmp_path: Path):
    version = "2026-01-01-statute-hierarchy"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/2014"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(e) Deductions from income (1) Standard deduction "
                    "(A) In general (i) Deduction Text. (B) Guam Text. "
                    "(C) Requirement Text. (2) Earned income deduction "
                    "(A) Second paragraph.\n\n"
                    "(f) Other deductions."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/e/1", _release(tmp_path))

    assert resolved.body.startswith("(1) Standard deduction")
    assert "(C) Requirement Text." in resolved.body
    assert "(2) Earned income deduction" not in resolved.body


def test_us_statute_hierarchy_accepts_punctuation_child_with_vetted_heading(
    tmp_path: Path,
):
    version = "2026-01-01-punctuation-sibling"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(a) Introductory text. (1) Total amount.— First paragraph. "
                    "(2) Included assets.— Second paragraph.\n\n"
                    "(b) Later subsection."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert resolved.body == "(1) Total amount.— First paragraph."


def test_us_statute_hierarchy_rejects_punctuation_advancing_reference(
    tmp_path: Path,
):
    version = "2026-01-01-punctuation-advancing-reference"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Target paragraph. Compare the prior provision. "
        "(2) of section 5 applies.\n\n(b) End."
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    paragraph_one = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert "(2) of section 5 applies." in paragraph_one.body
    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    "reference",
    [
        "exception under section 5 applies",
        "Exception under section 5 applies",
        "Exception Under section 5 applies",
        "Exception “Under section 5” applies",
        "Exception 5 controls",
        "Exception [Under section 5] applies",
        "Exception (3) applies",
        "Exception",
        "Exception The deduction described in subparagraph (b) of section 5 applies",
        "deduction under paragraph (4) shall apply",
        "Deduction Described in paragraph (4) applies",
        "Deduction The Secretary shall allow a standard deduction under paragraph (4)",
        "minimum amount described in section 5 applies",
        "Minimum amount Under section 5 controls",
        "Minimum amount Notwithstanding clause (i) of section 5 controls",
        "Additional tax In addition to the tax imposed by section 5 applies",
        "As used in this subsection, the term under section 5 applies",
        "As used in this subsection, the terminology under section 5 applies",
        "As used in this subsection, the termite rule applies",
        "standard deduction under paragraph (3) applies",
        "Standard deduction Under paragraph (3) applies",
        "utility standard in paragraph (4) controls",
        "Utility standard In paragraph (4) controls",
        "Included assets.- not a structural heading",
        "Shelter costs- not a structural heading",
    ],
)
def test_us_statute_hierarchy_rejects_heading_prefix_sentence_reference(
    tmp_path: Path,
    reference: str,
):
    version = "2026-01-01-heading-prefix-reference"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"(a)\n\n(1) Target paragraph. Compare prior. (2) {reference}.\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    paragraph_one = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert f"(2) {reference}." in paragraph_one.body
    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_statute_hierarchy_rejects_bare_allowlisted_phrase_at_end(
    tmp_path: Path,
):
    version = "2026-01-01-bare-heading-prefix"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = "(a)\n\n(1) Target. Compare prior. (2) Exception"
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    ("cue", "candidate"),
    [
        (
            "Compare the prior provision because " + "x" * 260 + ".",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "Consult the previous provision.",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "The preceding paragraph controls.",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "Review the foregoing section.",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "As described earlier.",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "Cross-reference the earlier rule.",
            "Exception The deduction described in subparagraph (b) applies.",
        ),
        (
            "“Compare the prior provision.”",
            "As used in this subsection, the term under section 5 applies.",
        ),
        (
            "See the previous paragraph.",
            "Standard deduction— applies.",
        ),
        (
            "Refer to the prior section.",
            "Shelter costs-- applies.",
        ),
        ("See the following rule.", "Standard deduction— applies."),
        ("Consult the next paragraph.", "Standard deduction— applies."),
        ("The section below controls.", "Standard deduction— applies."),
        ("Review the subsequent provision.", "Standard deduction— applies."),
        ("Look at the later clause.", "Standard deduction— applies."),
        ("See section 5.", "Standard deduction— applies."),
        ("Compare paragraph (b).", "Standard deduction— applies."),
        ("Refer to subsection (c).", "Standard deduction— applies."),
        ("Cross-reference rule 7.", "Standard deduction— applies."),
        ("Consult clause (i).", "Standard deduction— applies."),
    ],
)
def test_us_statute_hierarchy_rejects_weak_heading_after_reference_context(
    tmp_path: Path,
    cue: str,
    candidate: str,
):
    version = "2026-01-01-reference-context"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"(a)\n\n(1) Target. {cue} (2) {candidate}\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_statute_hierarchy_rejects_truncated_heading_evidence_window(
    tmp_path: Path,
):
    version = "2026-01-01-truncated-heading-window"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Target. (2)"
        + " " * 163
        + "As used in this subsection, the terminology applies.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    "cue",
    [
        "Compare the prior provision under U.S.C.",
        "Compare the prior provision, e.g.",
        "The paragraph above controls:",
        "The section previously mentioned controls:",
        "The rule set out above controls:",
        "Under the paragraph above:",
    ],
)
def test_us_hierarchy_vetoes_reference_context_before_strong_line_marker(
    tmp_path: Path,
    cue: str,
):
    version = "2026-01-01-strong-reference-context"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"(a)\n\n(1) Target. {cue}\n(2) Candidate.\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    "preceding",
    [
        "Target sentence.",
        "Target sentence. Compare paragraph (2).",
    ],
)
def test_us_hierarchy_requires_corroboration_for_replayable_content_marker(
    tmp_path: Path,
    preceding: str,
):
    version = "2026-01-01-content-corroboration"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        f"(a)\n\n(1) {preceding} (2) As used in this subsection, the term "
        "under section 5 applies.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    ("opening", "candidate", "closing"),
    [
        ('The quoted phrase is "', "(2) Standard deduction— applies.", '"'),
        ("The document says “", "(2) Standard deduction— applies.", "”"),
        ("Text: [", "(2) As used in this subsection, the term applies.", "]"),
        ("Example—{", "(2) Standard deduction— applies.", "}"),
        ("Example—(", "(2) Standard deduction— applies.", ")"),
        ("The code says `", "(2) Standard deduction— applies.", "`"),
        ("The quotation says «", "(2) Standard deduction— applies.", "»"),
        ("The quotation says ‹", "(2) Standard deduction— applies.", "›"),
        ("The phrase is '", "(2) Standard deduction— applies.", "'"),
    ],
)
def test_us_hierarchy_rejects_quoted_or_bracketed_marker_replay(
    tmp_path: Path,
    opening: str,
    candidate: str,
    closing: str,
):
    version = "2026-01-01-quoted-marker"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"(a)\n\n(1) {opening}{candidate}{closing}\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


@pytest.mark.parametrize(
    ("opening", "filler", "closing"),
    [
        (
            "The long quotation says “",
            "x" * (corpus_resolver._MAX_PRECEDING_REFERENCE_CONTEXT_CHARS + 1),
            "”",
        ),
        ("Example (outer (inner)", "x", ")"),
        ("Example [outer {inner}", "x", "]"),
        ("Nested quote “outer “inner” still outer ", "x", "”"),
        ("Unclosed outer “outer “inner” still outer ", "x", ""),
        (
            "Shared close “outer “inner” still outer ",
            "x",
            "\n“Later quoted paragraph.”",
        ),
        ("Nested quote 'outer 'inner' still outer ", "x", "'"),
        ("Curly quote ‘employee’s rule remains quoted ", "x", "’"),
        ("Curly quote ‘the workers’ rights remain quoted ", "x", "’"),
        (
            "Upper possessive ‘James’ Rights Act remains quoted ",
            "x",
            "\n‘Later quoted paragraph.’",
        ),
        ("Curly quote ‘the workers’, according to the quote, ", "x", "’"),
        ("Curly quote ‘the ’90 rule remains quoted ", "x", "’"),
        ("Straight quote 'the workers' rights remain quoted ", "x", "'"),
        (
            "Upper possessive 'James' Rights Act remains quoted ",
            "x",
            "\n'Later quoted paragraph.'",
        ),
        ("Straight quote 'the workers', according to the quote, ", "x", "'"),
        ('Measurement "a 5" clearance remains quoted ', "x", '"'),
        (
            'Upper measurement "a 5" Clearance remains quoted ',
            "x",
            '\n"Later quoted paragraph."',
        ),
        ('Measurement "a 5", according to the quote, ', "x", '"'),
        ("Crossed delimiters “outer [inner” still bracket ", "x", "]"),
        ('Crossed quotes “outer "inner” still inner ', "x", '"'),
        ('Even escapes \\\\"outer ', "x", '"'),
        ("Code span ``outer ", "x", "``"),
    ],
)
def test_us_hierarchy_replays_full_delimiter_state_for_distant_or_nested_openers(
    tmp_path: Path,
    opening: str,
    filler: str,
    closing: str,
):
    version = "2026-01-01-delimiter-replay"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    quoted = (
        f"{opening}{filler}\n"
        "(2) Standard deduction— applies.\n"
        f"(3) Shelter costs— applies.{closing}"
    )
    body = f"(a)\n\n(1) {quoted}\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_hierarchy_accepts_repeated_opening_quote_for_multiline_quotation(
    tmp_path: Path,
):
    version = "2026-01-01-multiline-quotation"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a) The notice states “first quoted paragraph.\n"
        "“Second quoted paragraph.”\n\n"
        "(b) Target subsection.\n\n(1) Target child.\n\n(c) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    resolved = resolve_local_corpus_source(f"{parent}/b/1", _release(tmp_path))

    assert resolved.body == "(1) Target child."


def test_us_hierarchy_accepts_balanced_mixed_nested_delimiters(tmp_path: Path):
    version = "2026-01-01-balanced-mixed-delimiters"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a) Intro ‹outer [‹inner›] outer›\n\n"
        "(b) Target subsection.\n\n(1) Target child.\n\n(c) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    resolved = resolve_local_corpus_source(f"{parent}/b/1", _release(tmp_path))

    assert resolved.body == "(1) Target child."


@pytest.mark.parametrize(
    "quoted_term",
    [
        "The term ‘worker’ means employee.",
        "The term 'worker' means employee.",
        "The term 'A' means alpha.",
        'The label "5" means five.',
        "The term 'workers' means employees.",
        "The term ‘workers’ does not apply.",
        "The term ‘workers’ may include contractors.",
        "The term ‘workers’ must be construed broadly.",
        "The term ‘workers’ under subsection (a) applies.",
        "The term 'workers' does not apply.",
        'The label "a 5" appears in the schedule.',
        "The phrase, the ‘workers’ does not apply.",
        'The expression: a "a 5" appears in the schedule.',
    ],
)
def test_us_hierarchy_accepts_ordinary_balanced_quoted_terms(
    tmp_path: Path,
    quoted_term: str,
):
    version = "2026-01-01-balanced-quoted-term"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        f"(a) Intro. {quoted_term}\n\n"
        "(b) Target subsection.\n\n(1) Target child.\n\n(c) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    resolved = resolve_local_corpus_source(f"{parent}/b/1", _release(tmp_path))

    assert resolved.body == "(1) Target child."


def test_us_hierarchy_ignores_escaped_directional_quote_in_future_balance(
    tmp_path: Path,
):
    version = "2026-01-01-escaped-directional-quote"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Intro ‘workers’ rights remain quoted.\n\n"
        "(2) Standard deduction— applies.\n\n"
        "\\‘ escaped opener text.\n\n"
        "(3) Shelter costs— applies.’\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_hierarchy_bounds_delimiter_nesting(tmp_path: Path, monkeypatch):
    version = "2026-01-01-delimiter-depth"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = "(a) Intro ((2) Candidate.)\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])
    monkeypatch.setattr(corpus_resolver, "_MAX_DELIMITER_NESTING_DEPTH", 1)

    with pytest.raises(CorpusSourceSliceError, match="delimiter nesting"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_measurement_evidence_replay_uses_a_bounded_suffix(monkeypatch):
    evidence_lengths: list[int] = []
    original_search = corpus_resolver.re.search

    def recording_search(pattern, string, *args, **kwargs):
        if "(?:a|an)" in str(pattern) and r"\d+" in str(pattern):
            evidence_lengths.append(len(string))
        return original_search(pattern, string, *args, **kwargs)

    monkeypatch.setattr(corpus_resolver.re, "search", recording_search)
    text = '"' + "x" * 10_000 + ' a 5" remains quoted.\n(2) Replay.'
    marker_start = text.index("(2)")

    enclosed = corpus_resolver._delimiter_enclosed_marker_starts(
        text,
        (marker_start,),
    )

    assert marker_start in enclosed
    assert evidence_lengths
    assert max(evidence_lengths) <= corpus_resolver._MAX_INTERNAL_MARK_EVIDENCE_CHARS


def test_delimiter_replay_rejects_a_negative_first_marker_offset():
    with pytest.raises(CorpusSourceSliceError, match="outside corpus source text"):
        corpus_resolver._delimiter_enclosed_marker_starts("xx", (-1,))


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("", id="empty"),
        pytest.param("Plain text.", id="plain"),
        pytest.param("(annotation)", id="final-parenthesis"),
        pytest.param("[annotation]", id="final-bracket"),
        pytest.param("{annotation}", id="final-brace"),
        pytest.param("“quoted”", id="directional-double"),
        pytest.param("‘quoted’", id="directional-single"),
        pytest.param("«quoted»", id="guillemets"),
        pytest.param("‹quoted›", id="single-guillemets"),
        pytest.param('"quoted"', id="symmetric-double"),
        pytest.param("'quoted'", id="symmetric-single"),
        pytest.param("`code`", id="code-span"),
        pytest.param("``code``", id="double-code-span"),
        pytest.param("(outer [“inner”] outer)", id="mixed-nesting"),
        pytest.param("“outer “inner” outer”", id="nested-quotes"),
        pytest.param("The workers' rights apply.", id="word-apostrophe"),
        pytest.param(r"An escaped \( opener.", id="escaped-parenthesis"),
        pytest.param(r"An escaped \“ opener.", id="escaped-quote"),
        pytest.param(
            "The notice states “first quoted paragraph.\n“Second quoted paragraph.”",
            id="repeated-line-quotes",
        ),
        pytest.param(
            "Notice: “First sentence. “Second sentence.”; and\n\n",
            id="colon-sentence-terminal-quotes",
        ),
    ],
)
def test_delimiter_replay_eof_accepts_balanced_source(text: str):
    assert len(text) not in corpus_resolver._delimiter_enclosed_marker_starts(
        text, (len(text),), allow_eof=True
    )


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("Unclosed (annotation", id="parenthesis"),
        pytest.param("Unclosed [annotation", id="bracket"),
        pytest.param("Unclosed {annotation", id="brace"),
        pytest.param("Unclosed “quotation", id="directional-double"),
        pytest.param("Unclosed ‘quotation", id="directional-single"),
        pytest.param('Unclosed "quotation', id="symmetric-double"),
        pytest.param("Unclosed 'quotation", id="symmetric-single"),
        pytest.param("Unclosed `code", id="code-span"),
        pytest.param("([)])", id="crossed-brackets-then-all-closed"),
        pytest.param("“outer [inner” rest]”", id="crossed-quote-then-all-closed"),
        pytest.param('("inner) still")', id="crossed-symmetric-then-all-closed"),
        pytest.param(
            "Notice: “First sentence. “Second sentence.”",
            id="colon-sentence-without-terminal-evidence",
        ),
    ],
)
def test_delimiter_replay_eof_marks_unclosed_or_crossed_source(text: str):
    assert len(text) in corpus_resolver._delimiter_enclosed_marker_starts(
        text, (len(text),), allow_eof=True
    )


@pytest.mark.parametrize("text", ["", "Plain text.", "(annotation)"])
def test_delimiter_replay_rejects_eof_marker_without_explicit_opt_in(text: str):
    with pytest.raises(CorpusSourceSliceError, match="outside corpus source text"):
        corpus_resolver._delimiter_enclosed_marker_starts(text, (len(text),))


def test_delimiter_replay_eof_opt_in_keeps_later_offsets_out_of_bounds():
    text = "(annotation)"
    with pytest.raises(CorpusSourceSliceError, match="outside corpus source text"):
        corpus_resolver._delimiter_enclosed_marker_starts(
            text, (len(text) + 1,), allow_eof=True
        )


def test_delimiter_replay_indexes_deep_quote_stack_without_scanning_it():
    text = "“" * 255 + "” “" * 10_000 + "(2) Replay."
    marker_start = text.index("(2)")

    enclosed = corpus_resolver._delimiter_enclosed_marker_starts(
        text,
        (marker_start,),
    )

    assert marker_start in enclosed


def test_delimiter_replay_bounds_total_delimiter_tokens(monkeypatch):
    monkeypatch.setattr(corpus_resolver, "_MAX_DELIMITER_REPLAY_TOKENS", 4)
    text = "“quoted” []\n(2) Replay."
    marker_start = text.index("(2)")

    with pytest.raises(CorpusSourceSliceError, match="bounded token limit"):
        corpus_resolver._delimiter_enclosed_marker_starts(text, (marker_start,))


@pytest.mark.parametrize(
    ("prefix", "suffix"),
    [
        ("Intro. See the previous subsection:\n", ""),
        ("Compare section 5:\n", ""),
        ("Example:\n", ""),
        ("The quotation begins “\n", "”"),
    ],
)
def test_us_hierarchy_preserves_context_before_first_requested_marker(
    tmp_path: Path,
    prefix: str,
    suffix: str,
):
    version = "2026-01-01-first-marker-context"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"{prefix}(a) Standard deduction— applies.{suffix}\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a", _release(tmp_path))


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_us_hierarchy_rejects_quoted_sibling_corroboration(
    tmp_path: Path,
    separator: str,
):
    version = "2026-01-01-quoted-sibling"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    quoted = (
        "“(2) As used in this subsection, the term fake means nothing."
        f"{separator}(3) Standard deduction— also fake.”"
    )
    body = f"(a)\n\n(1) The text says {quoted}\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_hierarchy_accepts_candidate_marker_referenced_by_preceding_scope(
    tmp_path: Path,
):
    version = "2026-01-01-candidate-reference"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    expected = "(B) Deduction Except as provided in subparagraph (C), body text."
    body = (
        "(a)\n\n(1) Paragraph.\n\n(A) Start.\n\n"
        f"{expected} (C) Exception The deduction described in subparagraph (B) "
        "does not apply.\n\n(2) Next paragraph.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    resolved = resolve_local_corpus_source(f"{parent}/a/1/B", _release(tmp_path))

    assert resolved.body == expected


def test_us_hierarchy_fails_closed_when_reference_context_bound_is_exceeded(
    tmp_path: Path,
    monkeypatch,
):
    version = "2026-01-01-reference-bound"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = f"(a)\n\n(1) {'x' * 80}\n(2) Candidate.\n\n(b) End."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])
    monkeypatch.setattr(
        corpus_resolver,
        "_MAX_PRECEDING_REFERENCE_CONTEXT_CHARS",
        16,
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/2", _release(tmp_path))


def test_us_statute_hierarchy_rejects_weak_reference_with_real_later_sibling(
    tmp_path: Path,
):
    version = "2026-01-01-weak-reference-real-sibling"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Paragraph.\n\n(A) Actual A. Compare the prior provision. "
        "(B) of section 5 applies.\n\n(C) Actual C.\n\n(2) End.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    actual_a = resolve_local_corpus_source(f"{parent}/a/1/A", _release(tmp_path))

    assert "(B) of section 5 applies." in actual_a.body
    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/1/B", _release(tmp_path))


@pytest.mark.parametrize(
    ("target", "body"),
    [
        (
            "a/6",
            "(a)\n\n(5) Actual fifth paragraph. Compare the prior provision. "
            "(6) of section 5 applies.\n\n(A) Actual subparagraph.\n\n"
            "(i) Actual clause.\n\n(I) Actual subclause.\n\n"
            "(7) Deep numeric item.\n\n(b) End.",
        ),
        (
            "a/1/H",
            "(a)\n\n(1) Paragraph.\n\n(G) Actual subparagraph. Compare prior. "
            "(H) of section 5 applies.\n\n(i) Clause.\n\n(I) Subclause.\n\n"
            "(J) Later subparagraph.\n\n(2) End.\n\n(b) End.",
        ),
        (
            "a/1/A/i",
            "(a)\n\n(1) Paragraph.\n\n(A) Actual subparagraph. Compare prior. "
            "(i) of section 5 applies.\n\n(v) Actual later subsection.\n\n(w) End.",
        ),
    ],
)
def test_us_statute_hierarchy_rejects_provisional_cross_level_reference(
    tmp_path: Path,
    target: str,
    body: str,
):
    version = "2026-01-01-provisional-cross-level"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))


def test_us_statute_hierarchy_does_not_cross_rejected_parent_boundary(
    tmp_path: Path,
):
    version = "2026-01-01-rejected-parent-boundary"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Actual first paragraph. Compare prior. "
        "(A) of section 5 applies. (2) Actual second paragraph.\n\n"
        "(B) Real subparagraph under paragraph two.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/1/A", _release(tmp_path))


def test_us_statute_hierarchy_rejects_advancing_marker_below_provisional_scope(
    tmp_path: Path,
):
    version = "2026-01-01-provisional-advance"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Target. (A) Unvetted. (i) Unvetted. (I) Unvetted. "
        "(1) Unvetted.\n\n(2) Actual second paragraph.\n\n"
        "(3) Actual third.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="provisional scope"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_us_statute_hierarchy_rejects_advancing_below_provisional_ancestor(
    tmp_path: Path,
):
    version = "2026-01-01-provisional-ancestor-advance"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Target. (A) Unvetted parent.\n\n(i) Strong.\n\n"
        "(I) Strong.\n\n(1) Strong deep numeric.\n\n"
        "(2) Actual second paragraph.\n\n(3) Actual third.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="provisional scope"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_us_statute_hierarchy_retains_rejected_preceding_heading_scope(
    tmp_path: Path,
):
    version = "2026-01-01-preceding-heading-scope"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(a)\n\n(1) Standard deduction (A) Actual subparagraph.\n\n"
        "(i) Actual nested clause.\n\n"
        "(2) Actual second paragraph.\n\n(b) End."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    paragraph = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert "(i) Actual nested clause." in paragraph.body
    assert "(2) Actual second paragraph." not in paragraph.body
    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/1/A", _release(tmp_path))


@pytest.mark.parametrize(
    ("document_class", "parent", "body", "target"),
    [
        (
            "statute",
            "us/statute/7/9999/a",
            "(a)(1) First.\n\n(2) Second.",
            "1",
        ),
        (
            "regulation",
            "us/regulation/7/999/1/a",
            "(a)(1) First.\n\n(2) Second.",
            "1",
        ),
        (
            "statute",
            "us/statute/7/9999/a",
            "(a) Deductions from income (1) Standard deduction (A) First. "
            "(2) Earned income deduction (A) Second.",
            "1",
        ),
    ],
)
def test_us_hierarchy_accepts_intermediate_parent_body_retaining_marker(
    tmp_path: Path,
    document_class: str,
    parent: str,
    body: str,
    target: str,
):
    version = "2026-01-01-retained-parent-marker"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
        document_class=document_class,
    )

    resolved = resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))

    assert resolved.body.startswith("(1) First") or resolved.body.startswith(
        "(1) Standard deduction (A) First"
    )


@pytest.mark.parametrize(
    ("document_class", "parent", "body", "child"),
    [
        (
            "statute",
            "us/statute/7/9999/a/1/A/i/I",
            "(1) Deep child.\n\n(2) Sibling.",
            "1",
        ),
        (
            "statute",
            "us/statute/7/9999/a/1/A/i/I/1",
            "(A) Deep child.\n\n(B) Sibling.",
            "A",
        ),
        (
            "regulation",
            "us/regulation/7/999/1/a/1/i/A",
            "(1) Deep child.\n\n(2) Sibling.",
            "1",
        ),
        (
            "statute",
            "us/statute/7/9999/a/1/A/i/I",
            "(I)\n\n(1) Deep child.\n\n(2) Sibling.",
            "1",
        ),
        (
            "statute",
            "us/statute/7/9999/a/1",
            "(a)(1)(A) Deep child.\n\n(B) Sibling.",
            "A",
        ),
        (
            "statute",
            "us/statute/7/9999/a/1/A",
            "(a)(1)(A)(i) Deep child.\n\n(ii) Sibling.",
            "i",
        ),
        (
            "regulation",
            "us/regulation/7/999/1/a/1/i",
            "(a)(1)(i)(A) Deep child.\n\n(B) Sibling.",
            "A",
        ),
    ],
)
def test_us_hierarchy_handles_cycles_in_intermediate_parent_representation(
    tmp_path: Path,
    document_class: str,
    parent: str,
    body: str,
    child: str,
):
    version = "2026-01-01-parent-cycle"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
        document_class=document_class,
    )

    resolved = resolve_local_corpus_source(f"{parent}/{child}", _release(tmp_path))

    assert resolved.body == f"({child}) Deep child."


@pytest.mark.parametrize(
    ("parent", "body", "child"),
    [
        (
            "us/statute/7/9999/a",
            "(1) Paragraph.\n\n(A) Subparagraph.\n\n(i) Clause.\n\n"
            "(I) Subclause.\n\n(a) Item.\n\n(2) Deep numeric.\n\n"
            "(b) Item sibling.",
            "2",
        ),
        (
            "us/statute/7/9999",
            "(a) Subsection.\n\n(1) Paragraph.\n\n(A) Subparagraph.\n\n"
            "(i) Clause.\n\n(I) Subclause.\n\n(1) Item.\n\n"
            "(a) Subitem.\n\n(b) Nested subitem.",
            "b",
        ),
    ],
)
def test_us_hierarchy_does_not_promote_repeated_deep_marker_to_parent_child(
    tmp_path: Path,
    parent: str,
    body: str,
    child: str,
):
    version = "2026-01-01-cycle-promotion"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="unassigned legal marker"):
        resolve_local_corpus_source(f"{parent}/{child}", _release(tmp_path))


@pytest.mark.parametrize(
    ("document_class", "parent", "body", "requested", "expected"),
    [
        (
            "statute",
            "us/statute/7/9999",
            "(f) Prior subsection. (i) In general .— Nested clause.\n\n"
            "(g) Target subsection.\n\n(1) Target paragraph.\n\n(h) End.",
            "g/1",
            "(1) Target paragraph.",
        ),
        (
            "regulation",
            "us/regulation/7/999/1",
            "(b) Prior paragraph.\n\n(v) Nested clause.\n\n"
            "(e) Interviews.\n\n(f) Verification.\n\n"
            "(1) Target paragraph.\n\n(g) End.",
            "f/1",
            "(1) Target paragraph.",
        ),
    ],
)
def test_us_hierarchy_reestablishes_unambiguous_top_level_boundary(
    tmp_path: Path,
    document_class: str,
    parent: str,
    body: str,
    requested: str,
    expected: str,
):
    version = "2026-01-01-top-level-recovery"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
        document_class=document_class,
    )

    resolved = resolve_local_corpus_source(f"{parent}/{requested}", _release(tmp_path))

    assert resolved.body == expected


@pytest.mark.parametrize(
    ("parent", "body", "child"),
    [
        (
            "us/statute/7/9999/a/1",
            "Parent heading.\n\n(1) Retained parent with no requested child.\n\n"
            "(2) Sibling.\n\n(A) Later unrelated marker.\n\n(B) End.",
            "A",
        ),
        (
            "us/statute/7/9999/a",
            "Parent heading.\n\n(a) Retained parent with no requested child.\n\n"
            "(b) Sibling.\n\n(1) Later unrelated marker.\n\n(2) End.",
            "1",
        ),
    ],
)
def test_us_hierarchy_detects_retained_parent_marker_after_heading_preamble(
    tmp_path: Path,
    parent: str,
    body: str,
    child: str,
):
    version = "2026-01-01-retained-heading-preamble"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError):
        resolve_local_corpus_source(f"{parent}/{child}", _release(tmp_path))


@pytest.mark.parametrize("separator", ["", " ", "\n", "\n\n"])
def test_us_hierarchy_rejects_ambiguous_middle_retained_ancestor_chain(
    tmp_path: Path,
    separator: str,
):
    version = "2026-01-01-middle-retained-chain"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999/a/1/A/i/I"
    body = (
        f"(1){separator}(A){separator}(i){separator}(I) "
        "Retained ancestor chain with no deep child.\n\n(II) Sibling."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="middle retained-ancestor"):
        resolve_local_corpus_source(f"{parent}/1", _release(tmp_path))


def test_us_hierarchy_rejects_middle_retained_chain_with_inline_headings(
    tmp_path: Path,
):
    version = "2026-01-01-middle-retained-headings"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999/a/1/A/i/I"
    body = (
        "(1) Paragraph heading (A) Subparagraph heading (i) Clause heading "
        "(I) Retained parent, no child.\n\n(II) Sibling."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="middle retained-ancestor"):
        resolve_local_corpus_source(f"{parent}/1", _release(tmp_path))


@pytest.mark.parametrize(
    "preamble",
    [
        "Parent heading.\n\n",
        "Heading\n\n",
        "§ 9999. Heading\n\n",
        "\ufeffParent heading.\n\n",
        "(2026) Edition\n\n",
        "(Note) Parent heading\n\n",
        "(Pub) L. heading\n\n",
    ],
)
def test_us_hierarchy_rejects_middle_retained_chain_after_preamble(
    tmp_path: Path,
    preamble: str,
):
    version = "2026-01-01-middle-retained-preamble"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999/a/1/A/i/I"
    body = f"{preamble}(1)\n(A)\n(i)\n(I) Retained parent, no child.\n\n(II) Sibling."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="middle retained-ancestor"):
        resolve_local_corpus_source(f"{parent}/1", _release(tmp_path))


def test_us_hierarchy_fails_closed_when_retained_preamble_scan_exhausts(
    tmp_path: Path,
    monkeypatch,
):
    version = "2026-01-01-retained-preamble-bound"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999/a/1/A/i/I"
    body = "(Note) Heading\n\n(1)\n(A)\n(i)\n(I) Retained parent."
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])
    monkeypatch.setattr(corpus_resolver, "_MAX_US_LEGAL_HIERARCHY_MARKERS", 1)

    with pytest.raises(CorpusSourceSliceError, match="bounded marker limit"):
        resolve_local_corpus_source(f"{parent}/1", _release(tmp_path))


@pytest.mark.parametrize("token", ["01", "001", "iiii", "iviv", "iix"])
def test_us_hierarchy_rejects_noncanonical_marker_aliases(
    tmp_path: Path,
    token: str,
):
    version = "2026-01-01-noncanonical-marker"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    if token[0].isdigit():
        target = f"a/{token}"
        body = f"(a) Parent.\n\n({token}) Invalid numeric marker.\n\n(b) End."
    else:
        target = f"a/1/A/{token}"
        body = (
            "(a) Parent.\n\n(1) Paragraph.\n\n(A) Subparagraph.\n\n"
            f"({token}) Invalid Roman marker.\n\n(v) Next.\n\n(b) End."
        )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="not legal hierarchy markers"):
        resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))


@pytest.mark.parametrize(
    ("requested", "resolved", "body"),
    [
        (
            "us/statute/7/9999/see",
            "us/statute/7/9999",
            "(a) Actual.\n\n(see) Prose note.\n\n(b) End.",
        ),
        (
            "us/statute/7/2014",
            "us/statute/7",
            "(2014) Parenthetical year.\n\n(2015) Later year.",
        ),
        (
            "us/regulation/7/273/2",
            "us/regulation/7/273",
            "(2) Parenthetical paragraph.\n\n(3) Later paragraph.",
        ),
        (
            "us/regulation/7/273",
            "us/regulation/7",
            "(273) Parenthetical part.\n\n(274) Later part.",
        ),
    ],
)
def test_canonical_us_parent_slice_rejects_nonhierarchy_fallback(
    requested: str,
    resolved: str,
    body: str,
):
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body,
            requested_path=requested,
            resolved_path=resolved,
        )


@pytest.mark.parametrize(
    ("subsection", "heading"),
    [
        ("g", "Allowable financial resources"),
        ("c", "First month benefits prorated"),
    ],
)
def test_us_statute_hierarchy_accepts_live_benchmark_headings(
    tmp_path: Path,
    subsection: str,
    heading: str,
):
    version = "2026-01-01-live-heading"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    f"({subsection}) {heading} (1) First paragraph. "
                    "(2) Included assets.— Second paragraph.\n\n"
                    "(z) Later subsection."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(
        f"{parent}/{subsection}/1", _release(tmp_path)
    )

    assert resolved.body == "(1) First paragraph."


@pytest.mark.parametrize(
    ("subsection", "heading", "target_body", "sibling", "expected_sha256"),
    [
        (
            "g",
            "Allowable financial resources",
            (
                "(1) Total amount.— (A) In general .— The Secretary shall prescribe "
                "the types and allowable amounts of financial resources (liquid and "
                "nonliquid assets) an eligible household may own, and shall, in so "
                "doing, assure that a household otherwise eligible to participate in "
                "the supplemental nutrition assistance program will not be eligible "
                "to participate if its resources exceed $2,000 (as adjusted in "
                "accordance with subparagraph (B)), or, in the case of a household "
                "which consists of or includes an elderly or disabled member, if its "
                "resources exceed $3,000 (as adjusted in accordance with subparagraph "
                "(B)). (B) Adjustment for inflation.— (i) In general .— Beginning on "
                "October 1, 2008 , and each October 1 thereafter, the amounts specified "
                "in subparagraph (A) shall be adjusted and rounded down to the nearest "
                "$250 increment to reflect changes for the 12-month period ending the "
                "preceding June in the Consumer Price Index for All Urban Consumers "
                "published by the Bureau of Labor Statistics of the Department of "
                "Labor. (ii) Requirement .— Each adjustment under clause (i) shall be "
                "based on the unrounded amount for the prior 12-month period."
            ),
            "(2) Included assets.— Sibling.",
            "b5cdbb164b6eab841e0d537e666341975d6b784228a705b796b2df3c4f1a0d99",
        ),
        (
            "c",
            "First month benefits prorated",
            (
                "(1) The value of the allotment issued to any eligible household for "
                "the initial month or other initial period for which an allotment is "
                "issued shall have a value which bears the same ratio to the value of "
                "the allotment for a full month or other initial period for which the "
                "allotment is issued as the number of days (from the date of "
                "application) remaining in the month or other initial period for which "
                "the allotment is issued bears to the total number of days in the month "
                "or other initial period for which the allotment is issued, except that "
                "no allotment may be issued to a household for the initial month or "
                "period if the value of the allotment which such household would "
                "otherwise be eligible to receive under this subsection is less than "
                "$10. Households shall receive full months’ allotments for all months "
                "within a certification period, except as provided in the first "
                "sentence of this paragraph with respect to an initial month."
            ),
            "(2) As used in this subsection, the term “initial month” means a period.",
            "c9d1c6cb17e55cb9b6543c1febdf8eb87e7cf2251fb6d87a7431ecbc5e0f094f",
        ),
    ],
)
def test_us_statute_hierarchy_preserves_live_benchmark_slice_hashes(
    tmp_path: Path,
    subsection: str,
    heading: str,
    target_body: str,
    sibling: str,
    expected_sha256: str,
):
    version = f"2026-01-01-live-hash-{subsection}"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    f"({subsection}) {heading} {target_body} {sibling} "
                    "(3) Optional combined allotment for expedited "
                    "households .— Later paragraph.\n\n(z) End."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(
        f"{parent}/{subsection}/1", _release(tmp_path)
    )

    assert resolved.body == target_body
    assert resolved.resolved_text_sha256 == expected_sha256


def test_us_statute_top_level_slice_ignores_nested_roman_alpha_ambiguity(
    tmp_path: Path,
):
    version = "2026-01-01-live-subsection-e-shape"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    expected = (
        "(e) Deductions.\n\n(6) Shelter.\n\n(B) Schedule. "
        "(i) first amount; (ii) second amount; (iii) third amount; "
        "(iv) fourth amount; (v) fifth amount; and (vi) sixth amount."
    )
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": f"(a) Earlier.\n\n{expected}\n\n(f) Later.",
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/e", _release(tmp_path))

    assert resolved.body == expected


@pytest.mark.parametrize(
    ("target", "body"),
    [
        (
            "a/1",
            "(a) In general (1) shall apply.\n\n"
            "(b) Other subsection.\n\n(2) Paragraph under subsection b.",
        ),
        (
            "a/1/A",
            "(a)(1) Standard deduction (A) shall apply.\n\n"
            "(2) Other paragraph.\n\n(B) Subparagraph under paragraph 2.",
        ),
        (
            "a/1/A/i",
            "(a)(1)(A) In general (i) shall apply.\n\n"
            "(B) Other subparagraph.\n\n(ii) Clause under subparagraph B.",
        ),
        (
            "a/1",
            "(a) In general (1) shall apply.\n\n"
            "(a) Replacement subsection.\n\n(2) Paragraph there.",
        ),
        (
            "b/1",
            "(b) In general (1) shall apply.\n\n"
            "(a) Backward subsection.\n\n(2) Paragraph there.",
        ),
        (
            "a/1/A",
            "(a)(1) Standard deduction (A) shall apply.\n\n"
            "(1) Replacement paragraph.\n\n(B) Subparagraph there.",
        ),
    ],
)
def test_inline_heading_sibling_evidence_stays_within_parent_scope(
    tmp_path: Path,
    target: str,
    body: str,
):
    version = "2026-01-01-sibling-scope"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    with pytest.raises(
        CorpusSourceSliceError,
        match="Could not isolate|Malformed legal marker sequence",
    ):
        resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))


def test_inline_heading_accepts_skipped_structural_sibling(tmp_path: Path):
    version = "2026-01-01-skipped-inline-sibling"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(a) Work requirement (1) First paragraph.\n\n"
                    "(3) Third paragraph.\n\n(b) Later subsection."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert resolved.body == "(1) First paragraph."


@pytest.mark.parametrize(
    "body",
    [
        "(a) Work requirement (3) shall apply.\n\n"
        "(2) Out-of-order paragraph.\n\n(4) Later paragraph.\n\n(b) End.",
        "(a) Work requirement (1) shall apply.\n\n"
        "(1) Duplicate paragraph.\n\n(2) Later paragraph.\n\n(b) End.",
        "(a) Work requirement (1) shall apply.\n\n"
        "(0) Invalid paragraph.\n\n(2) Later paragraph.\n\n(b) End.",
    ],
)
def test_inline_heading_rejects_malformed_sibling_sequences(
    tmp_path: Path,
    body: str,
):
    version = "2026-01-01-malformed-siblings"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )
    target = "3" if "(3)" in body else "1"

    with pytest.raises(CorpusSourceSliceError):
        resolve_local_corpus_source(f"{parent}/a/{target}", _release(tmp_path))


@pytest.mark.parametrize(
    ("document_class", "parent", "target", "body"),
    [
        (
            "statute",
            "us/statute/7/9999",
            "a/1/A/i/I/1",
            "(a) Top.\n\n(1) Paragraph.\n\n(A) Subparagraph.\n\n"
            "(i) Clause.\n\n(I) In general (1) shall apply.\n\n"
            "(2) Actual paragraph under subsection a.\n\n(b) End.",
        ),
        (
            "regulation",
            "us/regulation/7/999/1",
            "a/1/i/A/1",
            "(a) Top.\n\n(1) Paragraph.\n\n(i) Clause.\n\n"
            "(A) In general (1) shall apply.\n\n"
            "(2) Actual paragraph under subsection a.\n\n(b) End.",
        ),
    ],
)
def test_inline_heading_rejects_repeated_hierarchy_kind_ambiguity(
    tmp_path: Path,
    document_class: str,
    parent: str,
    target: str,
    body: str,
):
    version = "2026-01-01-repeated-kind"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": document_class,
                "version": version,
            }
        ],
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
        document_class=document_class,
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))


def test_inline_heading_evidence_has_bounded_recursion_and_work(tmp_path: Path):
    version = "2026-01-01-bounded-evidence"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    markers = ("1", "A", "i", "I") * 250
    body = "(a) In general " + " In general ".join(f"({marker})" for marker in markers)
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    with pytest.raises(CorpusSourceSliceError):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_inline_heading_evidence_bounds_branching_work(tmp_path: Path):
    version = "2026-01-01-bounded-branching"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    cycle = (
        ("Work requirement", "1", "2"),
        ("Standard deduction", "A", "B"),
        ("In general", "i", "ii"),
        ("Guam", "I", "II"),
    )
    levels = tuple(cycle[index % len(cycle)] for index in range(24))
    body = "(a) " + "".join(
        f"{heading} ({marker}) " for heading, marker, _sibling in levels
    )
    body += "Leaf."
    body += "".join(
        f"\n\n({sibling}) Sibling." for _heading, _marker, sibling in reversed(levels)
    )
    body += "\n\n(b) End."
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    resolved = resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))

    assert resolved.body.startswith("(1) Standard deduction (A)")
    assert "(b) End." not in resolved.body


def test_us_legal_hierarchy_bounds_provisional_scope_depth(tmp_path: Path):
    version = "2026-01-01-provisional-depth"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    cycle = ("A", "i", "I", "1")
    provisional = " ".join(
        f"({cycle[index % len(cycle)]}) Unvetted heading."
        for index in range(corpus_resolver._MAX_US_LEGAL_HIERARCHY_DEPTH + 1)
    )
    body = f"(a)\n\n(1) Target. {provisional}"
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="bounded depth limit"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_us_legal_hierarchy_rejects_unbounded_marker_count(tmp_path: Path):
    version = "2026-01-01-marker-limit"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = "(a) " + "\n".join(f"({index}) item" for index in range(4_097))
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    with pytest.raises(CorpusSourceSliceError, match="bounded marker limit"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


@pytest.mark.parametrize(
    ("target", "body"),
    [
        ("a", "(a) First block.\n\n(a) Duplicate block.\n\n(b) End."),
        (
            "a/1",
            "(a) Top.\n\n(1) First block.\n\n(1) Duplicate block.\n\n(2) End.",
        ),
        ("b", "(b) First block.\n\n(a) Backward block.\n\n(c) End."),
    ],
)
def test_us_legal_hierarchy_rejects_duplicate_or_backward_direct_markers(
    tmp_path: Path,
    target: str,
    body: str,
):
    version = "2026-01-01-direct-marker-order"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    with pytest.raises(CorpusSourceSliceError, match="duplicate or backward"):
        resolve_local_corpus_source(f"{parent}/{target}", _release(tmp_path))


def test_rejects_oversized_citation_segment_before_numeric_parsing(tmp_path: Path):
    oversized = "1" * 513
    release = _minimal_release(tmp_path)

    with pytest.raises(InvalidCorpusCitationError, match="Unsafe citation path"):
        resolve_local_corpus_source(f"us/statute/7/{oversized}", release)


@pytest.mark.parametrize(
    ("citation", "message"),
    [
        (
            "/".join(["us", "statute", *(["x"] * 63)]),
            "maximum segment count",
        ),
        (
            "/".join(["us", "statute", *(["x" * 500] * 9)]),
            "maximum length",
        ),
    ],
)
def test_rejects_unbounded_citation_shape_before_parent_expansion(
    tmp_path: Path,
    citation: str,
    message: str,
):
    release = _minimal_release(tmp_path)

    with pytest.raises(InvalidCorpusCitationError, match=message):
        resolve_local_corpus_source(citation, release)


def test_oversized_body_marker_fails_with_controlled_resolution_error(tmp_path: Path):
    version = "2026-01-01-oversized-marker"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": f"(a) ({'1' * 5_000}) text."}],
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/a/1", _release(tmp_path))


def test_single_us_subsection_stops_at_skipped_sibling_and_keeps_nested_roman(
    tmp_path: Path,
):
    version = "2026-01-01-skipped-sibling"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(e) Target subsection.\n\n"
                    "(1) Paragraph text.\n\n"
                    "(A) Subparagraph text.\n\n"
                    "(i) Nested Roman clause.\n\n"
                    "(g) Later nonconsecutive subsection."
                ),
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/e", _release(tmp_path))

    assert resolved.body.startswith("(e) Target subsection.")
    assert "(i) Nested Roman clause." in resolved.body
    assert "(g) Later nonconsecutive subsection." not in resolved.body


def test_single_us_subsection_preserves_bracketed_repealed_marker(tmp_path: Path):
    version = "2026-01-01-bracketed-repeal"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/26/3306"
    body = (
        "(k) Agricultural labor.\n\n"
        "[(l) Repealed. Sept. 1, 1954.]\n\n"
        "(m) American vessel and aircraft."
    )
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": parent, "body": body}],
    )

    preceding = resolve_local_corpus_source(f"{parent}/k", _release(tmp_path))
    repealed = resolve_local_corpus_source(f"{parent}/l", _release(tmp_path))

    assert preceding.body == "(k) Agricultural labor."
    assert repealed.body == "[(l) Repealed. Sept. 1, 1954.]"


@pytest.mark.parametrize("marker_separator", [" ", "\n\n"])
def test_us_statute_hierarchy_uses_target_to_disambiguate_nested_roman_i(
    tmp_path: Path,
    marker_separator: str,
):
    version = "2026-01-01-statute-roman"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(h) Work requirement (1) Standard deduction (A) In general"
                    f"{marker_separator}(i) Deduction First clause.\n\n"
                    "(ii) Minimum amount Second clause.\n\n"
                    "(B) Next subparagraph.\n\n(2) Next paragraph.\n\n"
                    "(i) Next subsection."
                ),
            }
        ],
    )

    ancestor = resolve_local_corpus_source(f"{parent}/h", _release(tmp_path))
    resolved = resolve_local_corpus_source(f"{parent}/h/1/A/i", _release(tmp_path))

    assert "(i) Deduction First clause." in ancestor.body
    assert "(ii) Minimum amount Second clause." in ancestor.body
    assert "(i) Next subsection." not in ancestor.body
    assert resolved.body == "(i) Deduction First clause."


def test_us_statute_hierarchy_does_not_cross_into_true_subsection_i(
    tmp_path: Path,
):
    version = "2026-01-01-statute-no-roman-child"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": parent,
                "body": (
                    "(h) Work requirement (1) Standard deduction "
                    "(A) In general\n\n(B) Next subparagraph. "
                    "(2) Next paragraph.\n\n(i) Next subsection."
                ),
            }
        ],
    )

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/h/1/A/i", _release(tmp_path))


def test_us_statute_top_level_roman_alpha_does_not_resolve_nested_clause(
    tmp_path: Path,
):
    version = "2026-01-01-top-level-roman-alpha"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    body = (
        "(h) Work requirement.\n\n(1) Paragraph.\n\n(A) Subparagraph.\n\n"
        "(i) Nested clause.\n\n(ii) Nested sibling.\n\n"
        "(j) Actual top-level subsection."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        resolve_local_corpus_source(f"{parent}/i", _release(tmp_path))


def test_us_statute_top_level_roman_alpha_resolves_actual_subsection(
    tmp_path: Path,
):
    version = "2026-01-01-actual-top-level-roman-alpha"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999"
    expected = "(i) Actual top-level subsection."
    body = (
        "(h) Work requirement.\n\n(1) Paragraph.\n\n(A) Subparagraph.\n\n"
        "(i) Nested clause.\n\n(ii) Nested sibling.\n\n"
        f"{expected}\n\n(j) Later subsection."
    )
    _write_rows(tmp_path, version, [{"citation_path": parent, "body": body}])

    resolved = resolve_local_corpus_source(f"{parent}/i", _release(tmp_path))

    assert resolved.body == expected


@pytest.mark.parametrize("marker_separator", [" ", "\n\n"])
def test_us_cfr_hierarchy_uses_target_to_disambiguate_nested_roman_i(
    tmp_path: Path,
    marker_separator: str,
):
    version = "2026-01-01-cfr-roman"
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
            }
        ],
    )
    parent = "us/regulation/7/999/1"
    path = tmp_path / f"data/corpus/provisions/us/regulation/{version}.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "id": "cfr-roman-row",
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
                "citation_path": parent,
                "body": (
                    "(h) Work requirement (1) In general"
                    f"{marker_separator}(i) Deduction First clause.\n\n"
                    "(ii) Minimum amount Second clause.\n\n"
                    "(2) Next paragraph.\n\n"
                    "(i) Next subsection."
                ),
                "source_path": "sources/us/regulation/cfr.xml",
                "source_as_of": "2026-01-02",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    ancestor = resolve_local_corpus_source(f"{parent}/h", _release(tmp_path))
    resolved = resolve_local_corpus_source(f"{parent}/h/1/i", _release(tmp_path))

    assert "(i) Deduction First clause." in ancestor.body
    assert "(ii) Minimum amount Second clause." in ancestor.body
    assert "(i) Next subsection." not in ancestor.body
    assert resolved.body == "(i) Deduction First clause."


def test_us_cfr_hierarchy_ignores_reused_nested_numeric_markers(tmp_path: Path):
    version = "2026-01-01-cfr-hierarchy"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/regulation/7/273/2"
    path = tmp_path / f"data/corpus/provisions/us/regulation/{version}.jsonl"
    path.parent.mkdir(parents=True)
    row = {
        "id": "cfr-row",
        "jurisdiction": "us",
        "document_class": "regulation",
        "version": version,
        "citation_path": parent,
        "body": (
            "(f) Verification.\n\n"
            "(1) Mandatory verification.\n\n"
            "(ii) Alien eligibility.\n\n"
            "(B) Pending verification.\n\n"
            "(1) First nested condition.\n\n"
            "(2) Second nested condition.\n\n"
            "(iii) Utility expenses.\n\n"
            "(2) Verification of questionable information.\n\n"
            "(g) Processing."
        ),
        "source_path": "sources/us/regulation/7-cfr-273.2.xml",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _write_selector(
        tmp_path,
        [
            {
                "jurisdiction": "us",
                "document_class": "regulation",
                "version": version,
            }
        ],
    )

    resolved = resolve_local_corpus_source(f"{parent}/f/1", _release(tmp_path))

    assert resolved.body.startswith("(1) Mandatory verification")
    assert "(2) Second nested condition." in resolved.body
    assert "(iii) Utility expenses." in resolved.body
    assert "(2) Verification of questionable information." not in resolved.body


def test_us_cfr_273_11_c_slice_uses_walker_after_source_text_changes():
    parent = "us/regulation/7/273/11"
    body = (
        "(a) First subsection.\n\n"
        "(b) Second subsection.\n\n"
        "(2) Nested condition.\n\n"
        "(i) First Roman item.\n\n"
        "(ii) Second Roman item.\n\n"
        "(c) Treatment of income and resources of certain nonhousehold members.\n\n"
        "(1) Target condition.\n\n"
        "(2) Second target condition.\n\n"
        "(d) Treatment of income and resources of other nonhousehold members."
    )
    assert corpus_resolver._slice_parent_body(
        body, requested_path=f"{parent}/c", resolved_path=parent
    ) == (
        "(c) Treatment of income and resources of certain nonhousehold members.\n\n"
        "(1) Target condition.\n\n"
        "(2) Second target condition."
    )
    # The former hash fallback made this source-text edit refuse, recording the
    # walker's Roman/alpha defect rather than a supported hierarchy distinction.
    assert corpus_resolver._slice_parent_body(
        body.replace("Second target condition", "Changed target condition"),
        requested_path=f"{parent}/c",
        resolved_path=parent,
    ).endswith("(2) Changed target condition.")


def test_us_cfr_hierarchy_keeps_contiguous_roman_c_nested():
    parent = "us/regulation/7/273/11"
    body = (
        "(a) First subsection.\n\n"
        "(b) Second subsection.\n\n"
        "(2) Nested condition.\n\n"
        "(xcix) Ninety-nine.\n\n"
        "(c) One hundred.\n\n"
        "(ci) One hundred one.\n\n"
        "(d) Next subsection."
    )

    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/b/2/c", resolved_path=parent
        )
        == "(c) One hundred."
    )
    with pytest.raises(CorpusSourceSliceError, match="missing structural marker"):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c", resolved_path=parent
        )


def test_us_cfr_hierarchy_keeps_skipped_roman_v_nested():
    parent = "us/regulation/7/273/11"
    body = (
        "(u) Subsection.\n\n"
        "(1) Condition.\n\n"
        "(i) First.\n\n"
        "(ii) Second.\n\n"
        "(v) Fifth.\n\n"
        "(vi) Sixth.\n\n"
        "(w) Next subsection."
    )

    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/u/1/v", resolved_path=parent
        )
        == "(v) Fifth."
    )
    assert "(v) Fifth." in corpus_resolver._slice_parent_body(
        body, requested_path=f"{parent}/u", resolved_path=parent
    )
    with pytest.raises(CorpusSourceSliceError, match="missing structural marker"):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/v", resolved_path=parent
        )


def test_us_cfr_hierarchy_does_not_promote_roman_v_with_inline_numeric_reference():
    parent = "us/regulation/7/273/11"
    body = (
        "(u) Subsection.\n\n"
        "(1) Condition.\n\n"
        "(i) First.\n\n"
        "(ii) Second.\n\n"
        "(v) Fifth. (1) A reference, not a child.\n\n"
        "(vi) Sixth.\n\n"
        "(w) Next subsection."
    )

    assert "(v) Fifth." in corpus_resolver._slice_parent_body(
        body, requested_path=f"{parent}/u", resolved_path=parent
    )
    with pytest.raises(CorpusSourceSliceError, match="missing structural marker"):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/v", resolved_path=parent
        )


@pytest.mark.parametrize(
    ("document_class", "previous_path", "target", "following", "target_children"),
    [
        pytest.param("regulation", "c/2/iii", "d", "e", ("1",), id="c-2-iii-to-d"),
        pytest.param("regulation", "b/2/ii", "c", "d", ("1",), id="b-2-ii-to-c"),
        pytest.param("regulation", "k/1/iii", "l", "m", (), id="k-1-iii-to-l"),
        pytest.param("regulation", "l/1/ii", "m", "n", (), id="l-1-ii-to-m"),
        pytest.param("regulation", "u/1/iii", "v", "w", (), id="u-1-iii-to-v"),
        pytest.param("regulation", "w/1/iii", "x", "y", (), id="w-1-iii-to-x"),
        pytest.param("regulation", "c/1/i/A", "d", "e", ("1",), id="deep-cfr-to-d"),
        pytest.param("statute", "c/1/A/i", "d", "e", ("1",), id="statute-to-d"),
        pytest.param("statute", "a/1/C/i/I", "a/1/D", "a/1/E", ("i",), id="upper-D"),
    ],
)
def test_us_hierarchy_prefers_exact_next_alpha_over_skipped_roman(
    document_class: str,
    previous_path: str,
    target: str,
    following: str,
    target_children: tuple[str, ...],
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    previous = "\n\n".join(
        f"({token}) Previous depth {depth}."
        for depth, token in enumerate(previous_path.split("/"))
    )
    expected = "\n\n".join(
        [f"({target.split('/')[-1]}) Target paragraph."]
        + [f"({token}) Target child." for token in target_children]
    )
    body = f"{previous}\n\n{expected}\n\n({following.split('/')[-1]}) Following."

    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{target}", resolved_path=parent
        )
        == expected
    )
    previous_alpha = "/".join(previous_path.split("/")[: len(target.split("/"))])
    previous_span = "\n\n".join(previous.split("\n\n")[len(target.split("/")) - 1 :])
    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{previous_alpha}", resolved_path=parent
        )
        == previous_span
    )


@pytest.mark.parametrize(
    ("previous_alpha", "previous_roman", "token", "following", "inline"),
    [
        pytest.param("b", "xcix", "c", "ci", "", id="contiguous-c-ci"),
        pytest.param("u", "ii", "v", "vi", "", id="skipped-v-vi"),
        pytest.param(
            "u",
            "ii",
            "v",
            "vi",
            " It mentions (1) only as an inline reference.",
            id="inline-reference",
        ),
        pytest.param("a", "ii", "c", "b", "", id="non-successor-c"),
        pytest.param("w", "ix", "x", "y", "", id="contiguous-x-known-limit"),
    ],
)
def test_us_hierarchy_preserves_corroborated_or_non_successor_roman(
    previous_alpha: str,
    previous_roman: str,
    token: str,
    following: str,
    inline: str,
):
    parent = "us/regulation/7/999/1"
    expected = f"({token}) Roman clause.{inline}"
    body = (
        f"({previous_alpha}) Previous paragraph.\n\n(1) Numeric child.\n\n"
        f"({previous_roman}) Previous Roman clause.\n\n{expected}\n\n"
        f"({following}) Following paragraph."
    )
    assert (
        corpus_resolver._slice_parent_body(
            body,
            requested_path=f"{parent}/{previous_alpha}/1/{token}",
            resolved_path=parent,
        )
        == expected
    )
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{token}", resolved_path=parent
        )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_keeps_roman_continuation_after_closed_prose_parenthesis(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) A.\n" if document_class == "statute" else ""
    expected = (
        "(u) U.\n(1) N.\n"
        f"{statute_subparagraph}(ii) II.\n"
        "(v) Roman (closed)\n(vi) Continuation."
    )
    body = f"{expected}\n(w) W."
    # Line-start (vi) still corroborates Roman (v) when a closing prose
    # parenthesis triggers the walker's existing lexical structural veto.
    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/u", resolved_path=parent
        )
        == expected
    )
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/v", resolved_path=parent
        )


def test_us_hierarchy_prefers_exact_next_alpha_without_following_marker():
    parent = "us/regulation/7/999/1"
    body = "(u) Previous.\n\n(1) Child.\n\n(ii) Roman.\n\n(v) Terminal paragraph."

    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/v", resolved_path=parent
        )
        == "(v) Terminal paragraph."
    )


def test_us_hierarchy_keeps_existing_slice_when_eof_probe_exceeds_depth_limit():
    parent = "us/regulation/7/999/1"
    body = (
        "(b) Existing paragraph.\n"
        + "(" * (corpus_resolver._MAX_DELIMITER_NESTING_DEPTH + 1)
        + "Tail."
    )
    # Ordinary marker replay stops at (b), so the base walker never scans the
    # deep tail. Existing targets do not need the extra recovered-path EOF check.
    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/b", resolved_path=parent
        )
        == body
    )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_rejects_alpha_recovery_with_unclosed_source_delimiter(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) Subparagraph.\n" if document_class == "statute" else ""
    body = (
        "(b) Previous.\n(1) Child.\n"
        f"{statute_subparagraph}(ii) Roman.\n"
        "(c) Target (unclosed prose.\n(d) Following."
    )
    # Recovering (c) would turn the base refusal into a successful slice that
    # swallows (d), whose marker the unmatched prose parenthesis encloses.
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c", resolved_path=parent
        )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_rejects_unbounded_later_target_after_alpha_recovery(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) A.\n" if document_class == "statute" else ""
    body = (
        "(b) B.\n(1) N.\n"
        f"{statute_subparagraph}(ii) R.\n(c) C.\n(1) N.\n"
        "(e) E (open\n(f) F."
    )
    # Finding (e) now succeeds after the earlier recovery repairs the stack.
    # Its malformed ending remains unsafe even though (c) is no longer an
    # ancestor of the requested target.
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/e", resolved_path=parent
        )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
@pytest.mark.parametrize("following", ["", "\n(e) Later accepted boundary."])
def test_us_hierarchy_rejects_recovered_slice_after_vetoed_ending_marker(
    document_class: str,
    following: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) Subparagraph.\n" if document_class == "statute" else ""
    body = (
        "(b) Previous.\n(1) Child.\n"
        f"{statute_subparagraph}(ii) Roman.\n"
        f"(c) Target (closed)\n(d) Plausible ending.{following}"
    )
    # The existing preceding-parenthesis veto rejects line-start (d). A new
    # (c) reading must refuse rather than swallow it, including when accepted
    # (e) would otherwise give the slice a later, bounded ending.
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c", resolved_path=parent
        )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_rejects_unbounded_descendant_of_recovered_alpha(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) Subparagraph.\n" if document_class == "statute" else ""
    body = (
        "(b) Previous.\n(1) Child.\n"
        f"{statute_subparagraph}(ii) Roman.\n"
        "(c) Recovered paragraph.\n"
        "(1) Target child (unclosed prose.\n(d) Following."
    )
    # The numeric marker is ordinary, but finding its ancestor (c) depends on
    # recovery. That does not establish an end boundary hidden by open prose.
    with pytest.raises(CorpusSourceSliceError):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c/1", resolved_path=parent
        )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_recovers_bounded_alpha_before_unclosed_later_paragraph(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) Subparagraph.\n" if document_class == "statute" else ""
    body = (
        "(b) Previous.\n(1) Child.\n"
        f"{statute_subparagraph}(ii) Roman.\n"
        "(c) Target (closed prose).\n(d) Following (unclosed later prose."
    )
    # A proved (d) boundary ends this slice before the malformed later prose.
    # Requiring the whole section to balance would reject this bounded target.
    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c", resolved_path=parent
        )
        == "(c) Target (closed prose)."
    )


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_recovers_alpha_after_balanced_source_delimiters(
    document_class: str,
):
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    statute_subparagraph = "(A) Subparagraph.\n" if document_class == "statute" else ""
    body = (
        "(b) Previous.\n(1) Child.\n"
        f"{statute_subparagraph}(ii) Roman.\n"
        "(c) Target (closed prose).\n(d) Following."
    )
    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c", resolved_path=parent
        )
        == "(c) Target (closed prose)."
    )


def test_us_hierarchy_sibling_evidence_stops_at_promoted_alpha_boundary():
    parent = "us/regulation/7/999/1"
    body = (
        "(c) Deductions from income (1) Numeric child.\n\n"
        "(iii) Roman child.\n\n(d) Next top level.\n\n"
        "(2) Child of d.\n\n(e) End."
    )
    # The inline (1) needs a later sibling to prove its depth. Paragraph (d)'s
    # (2) cannot supply that evidence across a top-level boundary.
    with pytest.raises(CorpusSourceSliceError, match="missing structural marker"):
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/c/1", resolved_path=parent
        )


def test_us_cfr_hierarchy_slices_verbatim_claims_excerpt():
    # Verbatim 273.18(b) through the first line of (f), from the pinned
    # us-rulespec-2026-10-29-ecfr-section-text section row, without repairs.
    body = (
        Path(__file__).parent / "fixtures" / "cfr_273_18_claims_excerpt.txt"
    ).read_text(encoding="utf-8")
    parent = "us/regulation/7/273/18"
    slices = {
        letter: corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{letter}", resolved_path=parent
        )
        for letter in "bcde"
    }
    for letter, following in zip("bcde", "cdef", strict=True):
        start = 0 if letter == "b" else body.index(f"\n({letter})") + 1
        end = body.index(f"\n({following})") + 1
        assert slices[letter] == body[start:end].strip()
        assert not body[start + len(slices[letter]) : end].strip()
    assert slices["c"].endswith("the basis for the trafficking determination.")
    assert "Claim referral management" not in slices["c"]
    assert slices["d"].startswith("(d) Claim referral management.")
    assert slices["e"].startswith(
        "(e) Initiating collection action and managing claims"
    )
    table_row = (
        "(B) do not apply the earned income deduction to that part of any earned "
        "income that the household failed to report in a timely manner when this "
        "act is the basis for the claim | the claim is an AE claim | apply the "
        "earned income deduction."
    )
    assert table_row in slices["c"]


_OUTLINE_ROMANS = (
    "i",
    "ii",
    "iii",
    "iv",
    "v",
    "vi",
    "vii",
    "viii",
    "ix",
    "x",
    "xi",
    "xii",
)


@st.composite
def _generated_us_legal_outlines(draw):
    document_class = draw(st.sampled_from(("regulation", "statute")))
    letters = draw(st.sampled_from(("abcd", "klmn", "uvw", "wxy", "ghij")))
    top_count = draw(st.integers(min_value=1, max_value=len(letters)))
    marker_levels = (
        (tuple("12"), _OUTLINE_ROMANS[:4], tuple("ABC"))
        if document_class == "regulation"
        else (tuple("12"), tuple("ABC"), _OUTLINE_ROMANS[:4], ("I", "II"))
    )
    expected: dict[tuple[str, ...], str] = {}

    def render(path: tuple[str, ...]) -> str:
        parts = [f"({path[-1]}) Paragraph {'/'.join(path)}."]
        if len(path) <= len(marker_levels):
            markers = marker_levels[len(path) - 1]
            child_count = draw(st.integers(min_value=0, max_value=len(markers)))
            if path[0] == "h" and markers[0] == "i" and child_count:
                # A lone (i) after (h) is already ambiguous in the base walker.
                # Require immediate (i),(ii) evidence instead of assigning an
                # unsupported intended span to that outline (issue #1808).
                child_count = max(2, child_count)
            if (
                path[0] == "u"
                and "v" in letters[:top_count]
                and markers[0] == "i"
                and child_count == 4
            ):
                # Existing contiguous-collision limit: (u)(1)(iv), then (v),
                # is Roman iv/v even when this generator intends top-level v.
                # Keep that ambiguity out of soundness inputs; the separate
                # contiguous-list property exercises iv/v as nested clauses.
                child_count = 3
            if path[0] == "h" and path[-1] == "i":
                child_count = 0
            parts.extend(render((*path, token)) for token in markers[:child_count])
        text = "\n\n".join(parts)
        expected[path] = text
        return text

    body = "\n\n".join(render((letter,)) for letter in letters[:top_count])
    return document_class, body, expected


@settings(max_examples=50, deadline=None, database=None)
@given(_generated_us_legal_outlines())
def test_us_hierarchy_generated_outline_soundness(outline):
    """Every successful generated path has exactly its independently rendered span."""
    document_class, body, expected = outline
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    for path, span in expected.items():
        try:
            actual = corpus_resolver._slice_parent_body(
                body, requested_path=f"{parent}/{'/'.join(path)}", resolved_path=parent
            )
        except CorpusSourceSliceError:
            continue
        assert actual == span, path


@settings(max_examples=50, deadline=None, database=None)
@given(
    document_class=st.sampled_from(("regulation", "statute")),
    target=st.sampled_from(tuple("cdlmvx")),
    numeric_count=st.integers(min_value=1, max_value=3),
    roman_count=st.integers(min_value=1, max_value=3),
    separator=st.sampled_from(("\n", "\n\n", "\n \n")),
)
def test_us_hierarchy_collision_liveness_and_partition(
    document_class: str,
    target: str,
    numeric_count: int,
    roman_count: int,
    separator: str,
):
    """Uncorroborated exact-next collision letters resolve and top slices abut."""
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    previous = chr(ord(target) - 1)
    following = chr(ord(target) + 1)
    spans = {}
    for letter in (previous, target, following):
        parts = [f"({letter}) Paragraph {letter}."]
        for number in range(1, numeric_count + 1):
            parts.append(f"({number}) Numeric child {number}.")
            if document_class == "statute":
                parts.append("(A) Subparagraph.")
            parts.extend(
                f"({roman}) Clause {roman}." for roman in _OUTLINE_ROMANS[:roman_count]
            )
        spans[letter] = separator.join(parts)
    body = separator.join(spans.values())
    end = 0
    for letter, span in spans.items():
        actual = corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{letter}", resolved_path=parent
        )
        assert actual == span
        start = body.index(actual, end)
        assert not body[end:start].strip()
        end = start + len(actual)
    assert not body[end:].strip()


@pytest.mark.parametrize("length", range(1, 13))
@settings(max_examples=8, deadline=None, database=None)
@given(
    document_class=st.sampled_from(("regulation", "statute")),
    letter=st.sampled_from(tuple("bkuw")),
)
def test_us_hierarchy_contiguous_roman_lists_stay_nested(
    length: int, document_class: str, letter: str
):
    """Every length through twelve keeps each Roman clause inside its parent."""
    parent = (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )
    prefix = f"({letter}) Paragraph.\n\n(1) Numeric child."
    path = f"{letter}/1"
    if document_class == "statute":
        prefix += "\n\n(A) Subparagraph."
        path += "/A"
    clauses = [f"({roman}) Clause {roman}." for roman in _OUTLINE_ROMANS[:length]]
    expected = "\n\n".join([prefix, *clauses])
    # Finish at an unambiguous later top-level letter. Roman continuation is
    # generated inside this paragraph, independently of the boundary under test.
    following = "e" if letter == "b" else "y"
    body = f"{expected}\n\n({following}) Following paragraph."

    assert (
        corpus_resolver._slice_parent_body(
            body, requested_path=f"{parent}/{letter}", resolved_path=parent
        )
        == expected
    )
    for roman, clause in zip(_OUTLINE_ROMANS[:length], clauses, strict=True):
        assert (
            corpus_resolver._slice_parent_body(
                body, requested_path=f"{parent}/{path}/{roman}", resolved_path=parent
            )
            == clause
        )


def _synthetic_us_parent(document_class: str) -> str:
    return (
        "us/regulation/7/999/1"
        if document_class == "regulation"
        else "us/statute/7/9999"
    )


def _slice_synthetic(body: str, parent: str, path: str) -> str:
    return corpus_resolver._slice_parent_body(
        body, requested_path=f"{parent}/{path}", resolved_path=parent
    )


# The walk reads the "(c)" that follows this prefix as paragraph (c): as a
# Roman numeral it would be clause 100 right after clause 2.
_RECOVERED_C_PREFIX = "(b) B.\n\n(1) B one.\n\n(ii) B one ii.\n\n"


@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("(b) B.\n\n", id="plain"),
        pytest.param(_RECOVERED_C_PREFIX, id="after-recovered-letter"),
    ],
)
def test_us_cfr_hierarchy_prefers_exact_next_paragraph_over_italic_gap(prefix: str):
    parent = "us/regulation/7/999/1"
    c3 = "(3) C three.\n\n(i) C three i.\n\n(A) C three i A.\n\n(1) Italic one."
    body = (
        f"{prefix}(c) C.\n\n(1) C one.\n\n(2) C two.\n\n{c3}\n\n(4) C four.\n\n(d) D."
    )
    # "(4)" is the next paragraph after (c)(3), not italic (4) after a gap
    # under (c)(3)(i)(A), just as 12 CFR 1005.18(b)(7) follows (b)(6). Main
    # read it as the italic numeral; after a recovered (c) that made (c)(3)
    # swallow (c)(4).
    assert _slice_synthetic(body, parent, "c/3") == c3
    assert (
        _slice_synthetic(body, parent, "c/3/i/A")
        == "(A) C three i A.\n\n(1) Italic one."
    )
    assert _slice_synthetic(body, parent, "c/4") == "(4) C four."


@pytest.mark.parametrize(
    "following", ["", "\n\n(iii) C one iii."], ids=["single", "roman-run"]
)
@pytest.mark.parametrize(
    "prefix",
    [
        pytest.param("(b) B.\n\n", id="plain"),
        pytest.param(_RECOVERED_C_PREFIX, id="after-recovered-letter"),
    ],
)
def test_us_cfr_hierarchy_returns_roman_clause_to_its_list_after_italic_numeral(
    prefix: str,
    following: str,
):
    parent = "us/regulation/7/999/1"
    c1i = "(i) C one i.\n\n(A) C one i A.\n\n(1) Italic one."
    c1 = f"(1) C one.\n\n{c1i}\n\n(ii) C one ii.{following}"
    body = f"{prefix}(c) C.\n\n{c1}\n\n(2) C two.\n\n(d) D."
    # "(ii)" follows clause (i) rather than opening an italic level under
    # (1), as in 7 CFR 273.7(c)(18)(ii), and a following "(iii)" doesn't
    # change that.
    assert _slice_synthetic(body, parent, "c/1/i") == c1i
    assert _slice_synthetic(body, parent, "c/1/ii") == "(ii) C one ii."
    assert _slice_synthetic(body, parent, "c/1") == c1
    assert _slice_synthetic(body, parent, "c/2") == "(2) C two."


# The walk reads "(C)" after (B)(ii)(II) as subparagraph (C): as a Roman
# numeral it would be subclause 100 right after subclause 2.
_RECOVERED_SUBPARAGRAPH_PREFIX = (
    "(B) Definitions.— (i) First term.— Text. (ii) Second term.— "
    "(I) In general.— Text. (II) Inclusion.— Text.\n\n"
    "(C) Middle subparagraph.\n\n"
)


@pytest.mark.parametrize(
    "subparagraph",
    [
        pytest.param(
            "(D) (i) Each agency may exempt members. (ii) Each agency may exempt more.",
            id="period",
        ),
        pytest.param(
            "(D) (i) the lease shall run one year; (ii) the owner shall not evict.",
            id="semicolon",
        ),
        pytest.param(
            "(D) (i) Each agency may exempt members. (iii) Each agency may report.",
            id="past-repealed-sibling",
        ),
    ],
)
def test_us_statute_hierarchy_refuses_recovered_clause_with_inline_sibling(
    subparagraph: str,
):
    parent = "us/statute/7/9999/d/4"
    body = f"{_RECOVERED_SUBPARAGRAPH_PREFIX}{subparagraph}\n\n(E) Last subparagraph."
    # The recovered (C) lets the walk reach (D)(i), but the walker never ends
    # a slice at the inline "(ii)", so (D)(i) would run on through it, as
    # 7 USC 2015(d)(4)(D)(i) did. It refuses; the subparagraphs still slice.
    with pytest.raises(CorpusSourceSliceError, match="Inline sibling marker"):
        _slice_synthetic(body, parent, "D/i")
    assert _slice_synthetic(body, parent, "C") == "(C) Middle subparagraph."
    assert _slice_synthetic(body, parent, "D") == subparagraph
    assert _slice_synthetic(body, parent, "E") == "(E) Last subparagraph."


def test_us_statute_hierarchy_inline_sibling_rule_skips_roman_readable_letters():
    parent = "us/statute/7/9999"
    recovered = "(b) B.\n\n(1) N.\n\n(A) A.\n\n(ii) R.\n\n(c) C.\n\n(d) D.\n\n"
    roman_target = "(h) Target lists records, notices, etc. (i) In general.— A clause."
    roman_body = (
        f"{recovered}(e) E.\n\n(f) F.\n\n(g) G.\n\n{roman_target}\n\n(j) Following."
    )
    plain_body = (
        f"{recovered}(e) Target paragraph. (f) Another paragraph runs inline."
        "\n\n(g) Following."
    )
    # An inline "(i)" after "etc." more likely opens a clause than paragraph
    # (i), as in 26 USC 42(h), so (h) keeps its slice. A plain inline "(f)"
    # after a sentence can only be paragraph (f), so (e) refuses.
    assert _slice_synthetic(roman_body, parent, "h") == roman_target
    with pytest.raises(CorpusSourceSliceError, match="Inline sibling marker"):
        _slice_synthetic(plain_body, parent, "e")


def test_us_statute_hierarchy_refuses_skipped_roman_readable_boundary_after_recovery():
    parent = "us/statute/7/9999/d/1"
    duration = (
        "(C) Duration.— (i) First .— Ineligible until— (I) one; (II) two; or "
        "(III) three. (ii) Second .— Ineligible until— (I) one; or (II) two. "
        "(iii) Third .— Ineligible until— (I) one; (II) two; (III) three; or "
        "(IV) four."
    )
    body = (
        f"{duration}\n\n"
        "(D) Administration.— (i) Good cause .— Text. (ii) Voluntary quit .— "
        "Text. (iii) Determination.— (I) In general .— Subject to subclause "
        "(II), an agency shall decide. (II) Not less restrictive .— Text. "
        "(iv) Strike .— Text. (v) Head of household.— (I) In general .— For "
        "purposes of this paragraph, a household may select its head. "
        "(II) Time for designation .— Text."
    )
    # As in 7 USC 2015(d)(1), the walk recovers (D) and then reads the second
    # "(I) In general" as subparagraph (I), skipping (E) through (H). It also
    # reads as subclause (I) of (D)(v), so (D) refuses rather than stop there
    # and (I) refuses rather than start there.
    assert _slice_synthetic(body, parent, "C") == duration
    with pytest.raises(CorpusSourceSliceError, match="skipped ending marker"):
        _slice_synthetic(body, parent, "D")
    with pytest.raises(CorpusSourceSliceError, match="skipped target marker"):
        _slice_synthetic(body, parent, "I")


@pytest.mark.parametrize(
    ("document_class", "target", "tail"),
    [
        pytest.param(
            "regulation",
            "(c) Target (see paragraph (b)(1))",
            "\n\n(d) Following.\n\n(e) E.",
            id="reference-cue",
        ),
        pytest.param(
            "statute",
            "(c) Target (see paragraph (b)(1))",
            "\n\n(d) Following.\n\n(e) E.",
            id="reference-cue-statute",
        ),
        pytest.param(
            "regulation",
            "(c) Target (see paragraph (b)(1))",
            "\n\n(d) Following.",
            id="reference-cue-end-of-source",
        ),
        pytest.param(
            "regulation",
            "(c) Target (unclosed prose.",
            "\n\n(d) Following.\n\n(e) Closed here).\n\n(f) After.",
            id="enclosed-parenthesis",
        ),
        pytest.param(
            "regulation",
            "(c) Target [unclosed prose.",
            "\n\n(d) Following.\n\n(e) Closed here].\n\n(f) After.",
            id="enclosed-bracket",
        ),
        pytest.param(
            "statute",
            "(c) Target (unclosed prose.",
            "\n\n(d) Following.\n\n(e) Closed here).\n\n(f) After.",
            id="enclosed-statute",
        ),
    ],
)
def test_us_hierarchy_rejects_recovered_slice_with_hidden_ending_marker(
    document_class: str,
    target: str,
    tail: str,
):
    parent = _synthetic_us_parent(document_class)
    statute_subparagraph = "(A) A.\n\n" if document_class == "statute" else ""
    previous = f"(b) B.\n\n(1) N.\n\n{statute_subparagraph}(ii) R."
    body = f"{previous}\n\n{target}{tail}"
    # The structural checks veto line-start "(d)" after the line that ends
    # in ")", or find it inside a delimiter that closes later. Reference
    # wording before it and a later closing delimiter are no proof that "(d)"
    # doesn't end the recovered (c), so (c) refuses rather than swallow it.
    with pytest.raises(CorpusSourceSliceError, match="Ambiguous ending marker"):
        _slice_synthetic(body, parent, "c")
    assert _slice_synthetic(body, parent, "b") == previous


def test_us_cfr_hierarchy_refuses_target_after_vetoed_match_of_its_path():
    parent = "us/regulation/7/999/1"
    body = (
        f"{_RECOVERED_C_PREFIX}(c) Paragraph c.\n\n(d) Paragraph d.\n\n"
        "(1) Paragraph d/1. (closed)\n\n(3) Paragraph d/3. (closed)\n\n"
        "(e) Paragraph e.\n\n(3) Paragraph e/3."
    )
    # The structural checks veto "(3)" under (d) and the "(e)" after it, each
    # after a line that ends in ")". The stack still ends at (d)(1) when
    # "(3) Paragraph e/3." comes, so that marker takes the path d/3. The
    # vetoed marker that matched the path first leaves two readings, so d/3
    # refuses rather than return paragraph (e)(3).
    with pytest.raises(CorpusSourceSliceError, match="Ambiguous starting marker"):
        _slice_synthetic(body, parent, "d/3")
    assert _slice_synthetic(body, parent, "c") == "(c) Paragraph c."


def test_us_cfr_hierarchy_refuses_skipped_roman_readable_marker_after_vetoed_numeral():
    parent = "us/regulation/7/999/1"
    body = (
        f"{_RECOVERED_C_PREFIX}(c) Paragraph c.\n\n(1) Paragraph c/1.\n\n"
        "(i) Paragraph c/1/i. (closed)\n\n(2) Paragraph c/2.\n\n(i) Paragraph c/2/i."
    )
    # The structural checks veto "(2)" after the line that ends in ")", so the
    # stack still ends at (c)(1)(i) when the second "(i)" comes. That marker
    # then fits only as paragraph (i), which skips (d) through (h) and also
    # reads as a Roman item, so (c) refuses to end there and (i) refuses to
    # start there.
    with pytest.raises(CorpusSourceSliceError, match="skipped ending marker"):
        _slice_synthetic(body, parent, "c")
    with pytest.raises(CorpusSourceSliceError, match="skipped target marker"):
        _slice_synthetic(body, parent, "i")


@pytest.mark.parametrize("document_class", ["regulation", "statute"])
def test_us_hierarchy_ignores_recovered_reading_of_vetoed_marker(document_class: str):
    parent = _synthetic_us_parent(document_class)
    statute_subparagraph = "(A) A.\n\n" if document_class == "statute" else ""
    first = (
        f"(1) N.\n\n{statute_subparagraph}"
        "(ii) R (as amended)\n(c) Applies to this paragraph."
    )
    body = f"(b) B.\n\n{first}\n\n(2) Two.\n\n(3) Three."
    # "(c)" reads as the letter after (b), but the structural checks veto it
    # after the line that ends in ")". It stays off the stack, so the walk
    # runs as main's does, and these slices stay as main returns them. Only
    # an accepted recovered letter puts a walk under the recovered-hierarchy
    # checks, which would refuse them.
    assert _slice_synthetic(body, parent, "b/1") == first
    assert _slice_synthetic(body, parent, "b/2") == "(2) Two."


@pytest.mark.parametrize(
    "citation",
    [
        pytest.param("paragraph\n(c) of this section.", id="label-before"),
        pytest.param("in\n(c)(1) of this section.", id="tail-after"),
    ],
)
def test_us_cfr_hierarchy_keeps_wrapped_citation_out_of_recovered_letters(
    citation: str,
):
    parent = "us/regulation/7/999/1"
    body = (
        f"(b) B.\n\n(1) N.\n\n(ii) R, as described in {citation}\n\n(2) Two.\n\n(c) C."
    )
    # The source wraps a citation onto a new line, so a line starts with a
    # "(c)" that names paragraph (c) rather than starting it. Read as a
    # recovered letter, it would cut (b) short and hand (c) the citation.
    assert _slice_synthetic(body, parent, "b") == body[: body.index("\n\n(c) C.")]
    try:
        paragraph_c = _slice_synthetic(body, parent, "c")
    except CorpusSourceSliceError:
        return
    assert paragraph_c == "(c) C."


def test_us_cfr_hierarchy_keeps_recovered_slice_with_vetoed_deeper_marker():
    parent = "us/regulation/7/999/1"
    c = "(c) C.\n\n(1) C one (closed)\n\n(2) C two."
    body = f"{_RECOVERED_C_PREFIX}{c}\n\n(d) D."
    # The structural checks veto "(2)" after the line that ends in ")". It
    # sits below (c), so it can't end (c), and its path isn't d, so it can't
    # start (d) either: both slices stand. It could end (c)(1), which refuses.
    assert _slice_synthetic(body, parent, "c") == c
    assert _slice_synthetic(body, parent, "d") == "(d) D."
    with pytest.raises(CorpusSourceSliceError, match="Ambiguous ending marker"):
        _slice_synthetic(body, parent, "c/1")


def test_us_cfr_hierarchy_keeps_skipped_roman_items_after_recovered_letter():
    parent = "us/regulation/7/999/1"
    body = (
        f"{_RECOVERED_C_PREFIX}(c) C.\n\n(1) C one.\n\n(i) I.\n\n(iii) III.\n\n(d) D."
    )
    # Clause (ii) is missing, as when a regulation removes an item. A Roman
    # item that skips another has no letter reading to doubt, so the
    # skipped-marker checks leave (i) and (iii) alone after a recovered (c).
    assert _slice_synthetic(body, parent, "c/1/i") == "(i) I."
    assert _slice_synthetic(body, parent, "c/1/iii") == "(iii) III."


def test_us_cfr_hierarchy_recovers_letter_despite_unrelated_earlier_predecessor():
    parent = "us/regulation/7/999/1"
    body = (
        "(t) T.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III.\n(iv) IV.\n"
        "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(v) V.\n(1) V one.\n(w) W."
    )
    # "(v)" after (u)(1)(ii) is a plausible gap, but the "(1)" after it opens
    # a paragraph, not a clause, so "(v)" is paragraph (v). The "(iv)" under
    # (t) belongs to another list and is no Roman predecessor for it.
    assert _slice_synthetic(body, parent, "u") == "(u) U.\n(1) N.\n(i) I.\n(ii) II."
    assert _slice_synthetic(body, parent, "v") == "(v) V.\n(1) V one."


def test_us_cfr_hierarchy_recovers_letter_before_next_letter_past_enclosed_marker():
    parent = "us/regulation/7/999/1"
    v = "(v) V (see the note\n(vi) Six)."
    body = f"(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III.\n{v}\n(w) W."
    # "(v)" after (u)(1)(iii) is a plausible gap, and the next anchored
    # marker, "(w)", follows it as a letter. The line-start "(vi)" sits inside
    # the open parenthesis, so it can't continue the clauses.
    expected_u = "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III."
    assert _slice_synthetic(body, parent, "u") == expected_u
    assert _slice_synthetic(body, parent, "v") == v


def test_us_cfr_hierarchy_recovers_letter_whose_first_child_runs_inline():
    parent = "us/regulation/7/999/1"
    x = "(x) Non-payment. (1) First child runs inline.\n(2) Second child."
    body = f"(w) W.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III.\n(iv) IV.\n{x}\n(y) Y."
    # Read as a clause, "(x)" would be item 10 after item 4, so it is
    # paragraph (x), as "(d) Non-payment. (1) ..." is in 47 CFR 54.1810. Its
    # first child runs inline, so the next anchored marker, "(2)", is its
    # second child, not a second item under (w).
    assert _slice_synthetic(body, parent, "w") == body[: body.index("\n(x)")]
    assert _slice_synthetic(body, parent, "x") == x


def test_us_cfr_hierarchy_recovers_letter_at_end_of_source():
    parent = "us/regulation/7/999/1"
    body = "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III.\n(v) V."
    # Nothing anchored follows "(v)", so no later marker can continue the
    # clauses, and the end of the source counts as evidence for the letter.
    expected_u = "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III."
    assert _slice_synthetic(body, parent, "u") == expected_u
    assert _slice_synthetic(body, parent, "v") == "(v) V."


def test_us_cfr_hierarchy_refuses_recovered_clause_with_inline_sibling():
    """Refuse a clause that swallows an inline sibling in flattened CFR definitions."""
    parent = "us/regulation/7/999/1"
    c1 = "(1) (i) Term one means one. (ii) Term two means two."
    body = f"{_RECOVERED_C_PREFIX}(c) C.\n\n{c1}\n\n(2) C two.\n\n(d) D."
    # The line-leading "(1) (i)" starts clause (i), but its inline sibling
    # cannot end it. After the recovered (c), the clause must refuse while
    # the enclosing paragraph still slices.
    with pytest.raises(CorpusSourceSliceError, match="Inline sibling marker"):
        _slice_synthetic(body, parent, "c/1/i")
    assert _slice_synthetic(body, parent, "c/1") == c1


def test_us_cfr_hierarchy_keeps_recovered_paragraph_with_line_start_italic_sibling():
    """Keep CFR paragraph text containing a separate line for each italic item."""
    parent = "us/regulation/7/999/1"
    c1 = (
        "(1) C one.\n\n(i) C one i.\n\n(A) C one i A.\n\n(1) Italic one."
        "\n\n(2) Italic two."
    )
    body = f"{_RECOVERED_C_PREFIX}(c) C.\n\n{c1}\n\n(2) C two.\n\n(d) D."
    assert _slice_synthetic(body, parent, "c/1") == c1
    assert _slice_synthetic(body, parent, "c/2") == "(2) C two."


def test_us_cfr_hierarchy_keeps_recovered_paragraph_with_inline_italic_sibling():
    """Keep CFR household conditions numbered inline inside subparagraph (A)."""
    parent = "us/regulation/7/999/1"
    c1 = "(1) C one.\n\n(i) C one i.\n\n(A) The household: (1) is X; (2) is Y."
    body = f"{_RECOVERED_C_PREFIX}(c) C.\n\n{c1}\n\n(2) C two.\n\n(d) D."
    assert _slice_synthetic(body, parent, "c/1") == c1
    assert _slice_synthetic(body, parent, "c/2") == "(2) C two."


@pytest.mark.parametrize(
    "last_condition",
    [
        "(A) Last condition ends. (2) C two.",
        "(A) The household: (1) meets the definition. (2) C two.",
    ],
)
def test_us_cfr_hierarchy_refuses_inline_paragraph_after_subparagraph(last_condition):
    """Refuse a CFR paragraph swallowing the next paragraph after its last child."""
    parent = "us/regulation/7/999/1"
    c1 = f"(1) C one.\n\n(i) C one i.\n\n{last_condition}"
    body = f"{_RECOVERED_C_PREFIX}(c) C.\n\n{c1}\n\n(3) C three.\n\n(d) D."
    with pytest.raises(CorpusSourceSliceError, match="Inline sibling marker"):
        _slice_synthetic(body, parent, "c/1")


def test_us_cfr_inline_sibling_evidence_keeps_continued_roman_run():
    """A skipped Roman run cannot hide a later paragraph sibling.

    This models a CFR work requirement whose inline first paragraph contains
    clauses (iii), (v), and (vi) before the second paragraph.
    """
    parent = "us/regulation/7/999/1"
    first_paragraph = (
        "(1) First paragraph.\n\n(iii) Third clause.\n\n"
        "(v) Fifth clause.\n\n(vi) Sixth clause."
    )
    body = (
        f"(u) Work requirement {first_paragraph}\n\n"
        "(2) Second paragraph.\n\n(w) Next subsection."
    )
    assert _slice_synthetic(body, parent, "u/1") == first_paragraph


def test_us_cfr_inline_sibling_evidence_recovers_next_letter():
    """A recovered next letter can prove an inline subsection boundary.

    This models a flattened CFR definition subsection followed by numbered
    content and a new subsection (c), rather than Roman clause one hundred.
    """
    parent = "us/regulation/7/999/1"
    subsection = (
        "(b) As used in this subsection, the term means this.\n\n"
        "(1) First paragraph.\n\n(ii) Second clause."
    )
    body = f"(a) Prior. {subsection}\n\n(c) Next subsection."
    assert _slice_synthetic(body, parent, "b") == subsection


def test_us_cfr_hierarchy_recovers_letter_after_label_that_ends_a_paragraph():
    parent = "us/regulation/7/999/1"
    previous = "(b) B.\n\n(1) N.\n\n(ii) Duties under this section"
    body = f"{previous}\n\n(c) C.\n\n(1) C one."
    # A blank line after "section" ends clause (ii), so "(c)" starts paragraph
    # (c) rather than finishing a citation.
    assert _slice_synthetic(body, parent, "b") == previous
    assert _slice_synthetic(body, parent, "c") == "(c) C.\n\n(1) C one."


@pytest.mark.parametrize(
    "first_clause",
    [
        pytest.param(
            "(i) Except as provided in clause (ii), each agency may act.",
            id="reference",
        ),
        pytest.param(
            "(i) Section 5 is amended by striking “text; (ii) old text” and "
            "inserting “new”.",
            id="quoted",
        ),
    ],
)
def test_us_statute_hierarchy_keeps_recovered_clause_with_inline_reference(
    first_clause: str,
):
    parent = "us/statute/7/9999/d/4"
    subparagraph = f"(D) {first_clause}\n\n(ii) Second clause.— Text."
    body = f"{_RECOVERED_SUBPARAGRAPH_PREFIX}{subparagraph}\n\n(E) Last subparagraph."
    # An inline "(ii)" that a reference introduces, or that sits inside a
    # quotation, names clause (ii) rather than starting it, so the inline
    # sibling check leaves (D)(i) to end at the line-start "(ii)".
    assert _slice_synthetic(body, parent, "D/i") == first_clause
    assert _slice_synthetic(body, parent, "D/ii") == "(ii) Second clause.— Text."
    assert _slice_synthetic(body, parent, "D") == subparagraph


@pytest.mark.parametrize(
    ("document_class", "body", "list_path", "list_end", "clause_path", "letter"),
    [
        pytest.param(
            "regulation",
            "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III (closed)\n(iv) Four.\n"
            "(v) Five.\n(w) W.",
            "u",
            "\n(w)",
            "u/1/v",
            "v",
            id="vetoed-predecessor",
        ),
        pytest.param(
            "regulation",
            "(u) U.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III. (iv) Four.\n"
            "(v) Five.\n(w) W.",
            "u",
            "\n(w)",
            "u/1/v",
            "v",
            id="inline-predecessor",
        ),
        pytest.param(
            "regulation",
            "(w) W.\n(1) N.\n(i) I.\n(ii) II.\n(iii) III.\n(iv) IV.\n(v) V.\n"
            "(vi) VI.\n(vii) VII.\n(viii) VIII (closed)\n(ix) Nine.\n(x) Five.\n"
            "(y) Y.",
            "w",
            "\n(y)",
            "w/1/x",
            "x",
            id="vetoed-ix-before-x",
        ),
        pytest.param(
            "statute",
            "(u) U.\n(1) N.\n(A) A.\n(i) I.\n(ii) II.\n(iii) III (closed)\n"
            "(iv) Four.\n(v) Five.\n(w) W.",
            "u",
            "\n(w)",
            "u/1/A/v",
            "v",
            id="statute-lower",
        ),
        pytest.param(
            "statute",
            "(a) A.\n(1) N.\n(U) U.\n(i) I.\n(I) UI.\n(II) UII.\n"
            "(III) UIII (closed)\n(IV) UIV.\n(V) Five.\n(W) W.",
            "a/1/U",
            "\n(W)",
            "a/1/U/i/V",
            "a/1/V",
            id="statute-upper",
        ),
    ],
)
def test_us_hierarchy_keeps_roman_item_after_raw_predecessor(
    document_class: str,
    body: str,
    list_path: str,
    list_end: str,
    clause_path: str,
    letter: str,
):
    parent = _synthetic_us_parent(document_class)
    list_marker = f"({list_path.split('/')[-1]})"
    expected_list = body[body.index(list_marker) : body.index(list_end)]
    clause_marker = clause_path.split("/")[-1]
    # The walker vetoes the predecessor after the line that ends in ")" or
    # sees it inline, so its stack skips one Roman item. The predecessor in
    # the source still makes the next marker a Roman item, not a letter.
    assert _slice_synthetic(body, parent, list_path) == expected_list
    assert _slice_synthetic(body, parent, clause_path) == f"({clause_marker}) Five."
    with pytest.raises(CorpusSourceSliceError):
        _slice_synthetic(body, parent, letter)


@pytest.mark.parametrize(
    "five",
    ["(v) Five.", "(v) Five.\n(A) Five A."],
    ids=["leaf", "with-child"],
)
@pytest.mark.parametrize("romans", [("i", "ii", "iii"), ("i", "ii")])
def test_us_cfr_hierarchy_keeps_skipped_roman_item_when_parent_list_continues(
    five: str,
    romans: tuple[str, ...],
):
    parent = "us/regulation/7/999/1"
    first = "\n".join(["(1) N.", *(f"({roman}) R{roman}." for roman in romans), five])
    body = f"(u) U.\n{first}\n(2) Two.\n(w) W."
    # "(v)" after (u)(1)(iii) or (u)(1)(ii) drops no more items than the list
    # keeps, a plausible gap, and the next marker continues (u)(1) rather than
    # following a letter (v). Nothing supports the letter, so "(v)" stays a
    # clause. Two items is the last list length where that holds.
    assert _slice_synthetic(body, parent, "u") == body[: body.index("\n(w)")]
    assert _slice_synthetic(body, parent, "u/1") == first
    assert _slice_synthetic(body, parent, "u/1/v") == five
    assert _slice_synthetic(body, parent, "u/2") == "(2) Two."
    with pytest.raises(CorpusSourceSliceError):
        _slice_synthetic(body, parent, "v")


def test_us_statute_hierarchy_slices_later_subsection_after_repeated_recovered_letter():
    parent = "us/statute/19/9999"
    clauses = "(c) C.\n\n(1) N.\n\n(A) UA.\n\n(i) R one.\n\n(ii) R two."
    tail = (
        "\n\n(d) First D.\n\n(d) Second D.\n\n(1) Report.\n\n(2) Omitted.\n\n"
        "(5) Renumbered.\n\n(f) F.\n\n(1) N.\n\n(2) Two."
    )
    body = f"(a) A.\n\n(b) B.\n\n{clauses}{tail}"
    # 19 USC 1862 designates two subsections (d). Under the Roman reading the
    # first (d) is clause 500 and the second is the only (d); under the letter
    # reading the second repeats the first. (f) starts and ends at the same
    # markers either way, so it slices. The repeated letter and its children,
    # however they are numbered, differ between the readings, so they refuse.
    assert _slice_synthetic(body, parent, "f") == "(f) F.\n\n(1) N.\n\n(2) Two."
    assert _slice_synthetic(body, parent, "f/1") == "(1) N."
    assert _slice_synthetic(body, parent, "c") == clauses
    for path in ("d", "d/1", "d/2", "d/5"):
        with pytest.raises(CorpusSourceSliceError):
            _slice_synthetic(body, parent, path)
    # A repeat of a letter that the walk did not recover stays ambiguous for
    # every later target, as on main.
    plain = f"(a) A.\n\n(b) B.\n\n(c) C.\n\n(1) N.{tail}"
    with pytest.raises(CorpusSourceSliceError, match="unassigned legal marker"):
        _slice_synthetic(plain, parent, "f")


def _assert_verbatim_fixture_slices(
    body: str,
    parent: str,
    expected: dict[str, tuple[str, str]],
) -> None:
    for path, (start_text, next_text) in expected.items():
        assert body.count(start_text) == 1, start_text
        start = body.index(start_text)
        actual = _slice_synthetic(body, parent, path)
        assert actual == body[start : start + len(actual)], path
        assert body[start + len(actual) :].startswith(next_text), path


def test_us_cfr_hierarchy_slices_verbatim_prepaid_disclosure_excerpt():
    # Verbatim 12 CFR 1005.18(b) through the first line of (c), from the
    # pinned 2026-10-19-title-12-part-1005 section row, without repairs.
    body = (
        Path(__file__).parent
        / "fixtures"
        / "cfr_1005_18_b_prepaid_disclosures_excerpt.txt"
    ).read_text(encoding="utf-8")
    next_c = "\n\n(c) Access to prepaid account information"
    next_7 = "\n\n(7) Specific formatting requirements"
    _assert_verbatim_fixture_slices(
        body,
        "us/regulation/12/1005/18",
        {
            "b": ("(b) Pre-acquisition disclosure requirements", next_c),
            "b/2/ix": (
                "(ix) Disclosure of additional fee types",
                "\n\n(x) Statement regarding overdraft credit features",
            ),
            "b/2/x": (
                "(x) Statement regarding overdraft credit features",
                "\n\n(xi) Statement regarding registration",
            ),
            "b/2/xiv": (
                "(xiv) Additional content for payroll card accounts",
                "\n\n(3) Short form disclosure of variable fees",
            ),
            "b/6": ("(6) Form of pre-acquisition disclosures", next_7),
            "b/6/iii": ("(iii) Tabular format", next_7),
            "b/6/iii/B": ("(B) Multiple service plans—(1) Short form", next_7),
            "b/7": (
                "(7) Specific formatting requirements",
                "\n\n(8) Terminology of pre-acquisition disclosures",
            ),
            "b/8": (
                "(8) Terminology of pre-acquisition disclosures",
                "\n\n(9) Prepaid accounts acquired in foreign languages",
            ),
            "b/9": ("(9) Prepaid accounts acquired in foreign languages", next_c),
        },
    )


def test_us_cfr_hierarchy_slices_verbatim_snap_employment_training_excerpt():
    # Verbatim 7 CFR 273.7(c) through the first line of (d), from the pinned
    # 2026-05-10-snap-7-cfr-273-r2026-07-15-self-contained section row,
    # without repairs.
    body = (
        Path(__file__).parent
        / "fixtures"
        / "cfr_273_7_c_state_agency_responsibilities_excerpt.txt"
    ).read_text(encoding="utf-8")
    next_d = "\n\n(d) Federal financial participation"
    next_ii = "\n\n(ii) From the time an E&T provider determines"
    _assert_verbatim_fixture_slices(
        body,
        "us/regulation/7/273/7",
        {
            "c": ("(c) State agency responsibilities.", next_d),
            "c/18": ("(18)(i) The State agency must ensure E&T providers", next_d),
            "c/18/i": ("(i) The State agency must ensure E&T providers", next_ii),
            "c/18/i/B": ("(B) Take the most suitable action", next_ii),
            "c/18/i/B/4": (
                "(4) Coordinate, to the maximum extent practicable,",
                next_ii,
            ),
            "c/18/ii": ("(ii) From the time an E&T provider determines", next_d),
        },
    )


def test_us_statute_hierarchy_slices_verbatim_snap_work_requirement_excerpt():
    # Verbatim 7 USC 2015(d)(4)(B) through the first line of (F), from the
    # pinned 2026-07-22-rulespec-title-7-consolidated paragraph row, without
    # repairs.
    body = (
        Path(__file__).parent / "fixtures" / "usc_7_2015_d_4_b_through_f_excerpt.txt"
    ).read_text(encoding="utf-8")
    parent = "us/statute/7/2015/d/4"
    _assert_verbatim_fixture_slices(
        body,
        parent,
        {
            "C": (
                "(C) The State agency may provide",
                "\n\n(D) (i) Each State agency may exempt",
            ),
            "D": (
                "(D) (i) Each State agency may exempt",
                "\n\n(E) Requirements for participation",
            ),
            "E": (
                "(E) Requirements for participation",
                "\n\n(F) (i) The total hours of work",
            ),
        },
    )
    # Each first clause runs on through the inline "(ii)" that follows a
    # sentence, so it refuses rather than return its sibling with it.
    for path in ("D/i", "E/i"):
        with pytest.raises(CorpusSourceSliceError, match="Inline sibling marker"):
            _slice_synthetic(body, parent, path)


_LEGAL_KINDS = ("lower_alpha", "numeric", "lower_roman", "upper_alpha", "upper_roman")


@settings(max_examples=200, deadline=None, database=None)
@given(
    document_class=st.sampled_from(("regulation", "statute")),
    ordinals=st.lists(st.integers(min_value=1, max_value=30), max_size=9),
    kind=st.sampled_from(_LEGAL_KINDS),
    ordinal=st.integers(min_value=1, max_value=31),
)
def test_new_level_or_exact_sibling_takes_deepest_exact_predecessor(
    document_class: str,
    ordinals: list[int],
    kind: str,
    ordinal: int,
):
    """A later item never opens a level while its predecessor waits above.

    Invariant, for every stack and marker reading: the level is the deepest
    one whose item is the marker's exact predecessor of the same kind, and a
    new level only when the marker is a first item or no such level exists.
    """
    stack = [
        (
            corpus_resolver._legal_marker_kind_for_level(
                level, document_class=document_class
            ),
            "x",
            value,
            level,
        )
        for level, value in enumerate(ordinals)
    ]
    exact = [
        level
        for level, item in enumerate(stack)
        if item[0] == kind and item[2] + 1 == ordinal
    ]
    level = corpus_resolver._new_level_or_exact_sibling(stack, kind, ordinal)
    if ordinal == 1 or not exact:
        assert level == len(stack)
    else:
        assert level == max(exact)


# Line endings that make the walker's structural checks veto the next
# line-start marker, with and without the reference wording that the old
# guard exempted, and an opener that never closes.
_OUTLINE_NOISE = ("", "", "", " (closed)", " (see paragraph (b)(1))", " (open")


@st.composite
def _generated_recovered_outlines(draw):
    """Outlines whose every walk crosses a recovered paragraph (c), with noise.

    Each outline starts with (b)(1)(ii), so the walk reads the next "(c)" as
    paragraph (c), not clause 100. It then draws numbering gaps in numeric
    lists, top-level gaps to letters that read only as letters, vetoed
    line-start markers, reference wording before a veto, an opener that
    never closes, clauses flattened onto one line after "." or ";",
    and a (u)(1) Roman list that runs through (iv) to (v) before (w).
    """
    document_class = draw(st.sampled_from(("regulation", "statute")))
    letters = draw(
        st.sampled_from(
            (("c", "d", "e", "f"), ("c", "d", "f", "g"), ("c", "e", "g"))
            + (("c", "d", "u", "w"), ("c", "u", "w"))
        )
    )
    letters = letters[: draw(st.integers(min_value=1, max_value=len(letters)))]
    marker_levels = (
        ("numeric", "lower_roman", "upper_alpha")
        if document_class == "regulation"
        else ("numeric", "upper_alpha", "lower_roman", "upper_roman")
    )
    tokens = {
        "numeric": tuple(str(number) for number in range(1, 6)),
        "lower_roman": _OUTLINE_ROMANS[:5],
        "upper_alpha": tuple("ABC"),
        "upper_roman": ("I", "II"),
    }
    expected: dict[tuple[str, ...], str] = {}

    def children_of(path: tuple[str, ...]) -> list[str]:
        if len(path) > len(marker_levels):
            return []
        kind = marker_levels[len(path) - 1]
        available = tokens[kind]
        if kind == "lower_roman":
            # Only (u)(1) may reach (v), and only with (w) as the next
            # paragraph: there "(v)" continues the clauses (iv) began.
            reaches_v = path[:2] == ("u", "1") and "v" not in letters
            available = available[: 5 if reaches_v else 4]
        count = draw(st.integers(min_value=0, max_value=min(3, len(available))))
        if kind == "numeric" and count:
            # Numeric lists may skip numbers; no other list here repeats the
            # numeric kind, so a gap can't read as a deeper item.
            picks = draw(
                st.lists(
                    st.sampled_from(available),
                    min_size=count,
                    max_size=count,
                    unique=True,
                )
            )
            return sorted(picks, key=int)
        if kind == "lower_roman" and path[:2] == ("u", "1") and len(available) == 5:
            return list(available[: draw(st.sampled_from((0, 4, 5)))])
        return list(available[:count])

    def render(path: tuple[str, ...], *, flattened: str | None = None) -> str:
        child_tokens = children_of(path)
        if flattened is not None:
            # A flattened clause ends with the separator its sibling follows.
            parts = [f"({path[-1]}) Clause {'/'.join(path)}{flattened}"]
            parts.extend(
                render((*path, token), flattened=flattened) for token in child_tokens
            )
            text = " ".join(parts)
            expected[path] = text
            return text
        flatten = len(path) >= 2 and child_tokens and draw(st.booleans())
        if flatten:
            separator = draw(st.sampled_from((".", ";")))
            # The first clause follows its parent's marker directly, as in
            # 7 USC 2015(d)(4)(D): "(D) (i) Each State agency ... (ii) ...".
            text = f"({path[-1]}) " + " ".join(
                render((*path, token), flattened=separator) for token in child_tokens
            )
            expected[path] = text
            return text
        noise = draw(st.sampled_from(_OUTLINE_NOISE))
        parts = [f"({path[-1]}) Paragraph {'/'.join(path)}.{noise}"]
        parts.extend(render((*path, token)) for token in child_tokens)
        text = "\n\n".join(parts)
        expected[path] = text
        return text

    statute_subparagraph = "(A) A.\n\n" if document_class == "statute" else ""
    prefix = f"(b) B.\n\n(1) N.\n\n{statute_subparagraph}(ii) R."
    expected[("b",)] = prefix
    body = "\n\n".join([prefix, *(render((letter,)) for letter in letters)])
    return document_class, body, expected


@settings(max_examples=150, deadline=None, database=None)
@given(_generated_recovered_outlines())
def test_us_hierarchy_recovered_outline_soundness_under_noise(outline):
    """A walk through a recovered letter returns the intended span or refuses.

    Invariant, for every generated outline: each requested path either
    refuses or returns exactly its independently rendered span, whatever the
    gaps, vetoes, reference wording, unclosed openers or flattened clauses.
    """
    document_class, body, expected = outline
    parent = _synthetic_us_parent(document_class)
    for path, span in expected.items():
        try:
            actual = _slice_synthetic(body, parent, "/".join(path))
        except CorpusSourceSliceError:
            continue
        assert actual == span, path


def test_metadata_parent_composes_descendants_deterministically(tmp_path: Path):
    version = "2026-01-01-children"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": CITATION,
                "body": None,
                "heading": "Parent",
                "metadata": {"amends": "principal document"},
            },
            {
                "citation_path": f"{CITATION}/2",
                "body": "second",
                "ordinal": 2,
            },
            {
                "citation_path": f"{CITATION}/1",
                "body": "first",
                "heading": "First heading",
                "ordinal": 1,
            },
        ],
    )

    release = _release(tmp_path)
    resolved = resolve_local_corpus_source(CITATION, release)

    assert resolved.body == "First heading\n\nfirst\n\nsecond"
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{CITATION}/1",
        f"{CITATION}/2",
    ]
    assert all(row.provision_file_sha256 for row in resolved.component_rows)
    assert resolved.selected_record.row == resolved.row
    assert resolved.selected_record.body == resolved.body
    assert resolved.selected_record.heading == "Parent"
    assert dict(resolved.selected_record.metadata) == {"amends": "principal document"}
    records = list(iter_active_local_corpus_rows(release, include_bodyless=True))
    selected = next(record for record in records if record.row == resolved.row)
    assert selected.body == ""
    assert selected.row.body_sha256 is None
    assert selected.metadata == resolved.selected_record.metadata


def test_metadata_parent_composes_descendants_in_hierarchy_preorder(tmp_path: Path):
    version = "2026-01-01-children-preorder"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None, "ordinal": 0},
            {"citation_path": f"{CITATION}/2", "body": "second", "ordinal": 2},
            {"citation_path": f"{CITATION}/1", "body": None, "ordinal": 1},
            {
                "citation_path": f"{CITATION}/1/B",
                "body": "nested B",
                "ordinal": 2,
            },
            {
                "citation_path": f"{CITATION}/1/A",
                "body": "nested A",
                "ordinal": 1,
            },
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "nested A\n\nnested B\n\nsecond"
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{CITATION}/1/A",
        f"{CITATION}/1/B",
        f"{CITATION}/2",
    ]


def test_metadata_parent_uses_shallowest_body_bearing_descendant_cover(
    tmp_path: Path,
):
    version = "2026-01-01-children-minimal-cover"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None, "ordinal": 0},
            {
                "citation_path": f"{CITATION}/1/A",
                "body": "duplicated nested text",
                "ordinal": 1,
            },
            {
                "citation_path": f"{CITATION}/1",
                "body": "first with nested text",
                "ordinal": 1,
            },
            {"citation_path": f"{CITATION}/2", "body": "second", "ordinal": 2},
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "first with nested text\n\nsecond"
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{CITATION}/1",
        f"{CITATION}/2",
    ]


def test_metadata_parent_rejects_uncovered_bodyless_branch(tmp_path: Path):
    version = "2026-01-01-uncovered-branch"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "covered"},
            {"citation_path": f"{CITATION}/2", "body": None},
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_omits_explicit_repealed_bodyless_leaf(tmp_path: Path):
    version = "2026-01-01-repealed-leaf"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "operative text"},
            {
                "citation_path": f"{CITATION}/2",
                "body": None,
                "heading": "Repealed provision. (Repealed)",
                "metadata": {"status": "repealed"},
            },
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "operative text"
    assert [row.citation_path for row in resolved.component_rows] == [f"{CITATION}/1"]


def test_metadata_parent_rejects_top_level_repealed_status_field(tmp_path: Path):
    version = "2026-01-01-top-level-repealed"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "operative text"},
            {
                "citation_path": f"{CITATION}/2",
                "body": None,
                "status": "repealed",
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_rejects_repealed_bodyless_nonleaf(tmp_path: Path):
    version = "2026-01-01-repealed-nonleaf"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "operative text"},
            {
                "citation_path": f"{CITATION}/2",
                "body": None,
                "metadata": {"status": "repealed"},
            },
            {"citation_path": f"{CITATION}/2/A", "body": "stale repealed text"},
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="malformed repealed"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_rejects_directly_selected_repealed_bodyless_parent(tmp_path: Path):
    version = "2026-01-01-repealed-parent"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": CITATION,
                "body": None,
                "metadata": {"status": "repealed"},
            },
            {"citation_path": f"{CITATION}/1", "body": "stale repealed text"},
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="is repealed"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_rejects_repealed_body_bearing_leaf(tmp_path: Path):
    version = "2026-01-01-repealed-body"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "operative text"},
            {
                "citation_path": f"{CITATION}/2",
                "body": "stale repealed text",
                "metadata": {"status": "repealed"},
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="malformed repealed"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_repealed_leaf_does_not_disable_operative_sibling_ordinals(tmp_path: Path):
    version = "2026-01-01-repealed-ordering"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/10", "body": "first", "ordinal": 1},
            {"citation_path": f"{CITATION}/2", "body": "second", "ordinal": 2},
            {
                "citation_path": f"{CITATION}/3",
                "body": None,
                "metadata": {"status": "repealed"},
            },
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "first\n\nsecond"


def test_rejects_malformed_repealed_row_below_body_bearing_ancestor(
    tmp_path: Path,
):
    version = "2026-01-01-covered-repealed"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "complete subtree"},
            {
                "citation_path": f"{CITATION}/1/A",
                "body": "stale repealed text",
                "metadata": {"status": "repealed"},
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="malformed repealed"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_active_bodyless_row_without_descendants_is_invalid(tmp_path: Path):
    version = "2026-01-01-no-descendants"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(tmp_path, version, [{"citation_path": CITATION, "body": None}])
    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_active_bodyless_row_with_only_bodyless_descendants_is_invalid(
    tmp_path: Path,
):
    version = "2026-01-01-only-bodyless"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": None},
        ],
    )
    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_rejects_bodyless_reserved_heading_leaf(tmp_path: Path):
    version = "2026-01-01-reserved-leaf"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "covered", "ordinal": 1},
            {
                "citation_path": f"{CITATION}/2",
                "body": None,
                "heading": "(Reserved)",
                "ordinal": 2,
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_explicit_reserved_regulation_leaf_does_not_block_parent_source(
    tmp_path: Path,
):
    version = "2026-01-01-reserved-regulation"
    parent = "us/regulation/106-cmr/364"
    _write_selector(
        tmp_path,
        [{"jurisdiction": "us", "document_class": "regulation", "version": version}],
    )
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": parent, "body": None},
            {"citation_path": f"{parent}/710", "body": "Operative text.", "ordinal": 1},
            {
                "citation_path": f"{parent}/800",
                "body": "",
                "heading": "800 Reserved",
                "metadata": {"status": "reserved", "section_label": "800"},
                "ordinal": 2,
            },
        ],
        document_class="regulation",
    )

    resolved = resolve_local_corpus_source(parent, _release(tmp_path))

    assert resolved.body == "Operative text."
    assert [row.citation_path for row in resolved.component_rows] == [f"{parent}/710"]
    with pytest.raises(InvalidActiveCorpusSourceError, match="no body-bearing"):
        resolve_local_corpus_source(f"{parent}/800", _release(tmp_path))


@pytest.mark.parametrize(
    ("heading", "metadata"),
    [
        ("800 Reserved", {"section_label": "800"}),
        ("800 General", {"status": "reserved", "section_label": "800"}),
        ("800 Reserved", {"status": "reserved", "section_label": "801"}),
        ("All rights reserved", {"status": "reserved", "section_label": "800"}),
        ("800 Reserved", {"status": [], "section_label": "800"}),
    ],
)
def test_reserved_regulation_leaf_requires_explicit_consistent_status(
    tmp_path: Path, heading: str, metadata: dict[str, object]
):
    version = "2026-01-01-bad-reserved-regulation"
    parent = "us/regulation/106-cmr/364"
    _write_selector(
        tmp_path,
        [{"jurisdiction": "us", "document_class": "regulation", "version": version}],
    )
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": parent, "body": None},
            {"citation_path": f"{parent}/710", "body": "Operative text."},
            {
                "citation_path": f"{parent}/800",
                "body": "",
                "heading": heading,
                "metadata": metadata,
            },
        ],
        document_class="regulation",
    )

    with pytest.raises(InvalidActiveCorpusSourceError):
        resolve_local_corpus_source(parent, _release(tmp_path))


@pytest.mark.parametrize("malformation", ["body", "list_body", "object_body", "child"])
def test_reserved_regulation_leaf_cannot_hide_operative_text(
    tmp_path: Path, malformation: str
):
    version = "2026-01-01-reserved-with-text"
    parent = "us/regulation/106-cmr/364"
    reserved = f"{parent}/800"
    _write_selector(
        tmp_path,
        [{"jurisdiction": "us", "document_class": "regulation", "version": version}],
    )
    rows = [
        {"citation_path": parent, "body": None},
        {"citation_path": f"{parent}/710", "body": "Operative text."},
        {
            "citation_path": reserved,
            "body": {
                "body": "Unexpected body.",
                "list_body": ["Unexpected body."],
                "object_body": {"text": "Unexpected body."},
            }.get(malformation),
            "heading": "800 Reserved",
            "metadata": {"status": "reserved", "section_label": "800"},
        },
    ]
    if malformation == "child":
        rows.append({"citation_path": f"{reserved}/1", "body": "Hidden child."})
    _write_rows(tmp_path, version, rows, document_class="regulation")

    with pytest.raises(InvalidActiveCorpusSourceError, match="malformed reserved"):
        resolve_local_corpus_source(parent, _release(tmp_path))


def test_metadata_parent_does_not_treat_generic_reserved_suffix_as_text(
    tmp_path: Path,
):
    version = "2026-01-01-generic-reserved"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "covered"},
            {
                "citation_path": f"{CITATION}/2",
                "body": None,
                "heading": "All rights reserved",
            },
        ],
    )

    with pytest.raises(InvalidActiveCorpusSourceError, match="uncovered bodyless"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_rejects_cross_version_descendant_composition(
    tmp_path: Path,
):
    first = "2026-01-01-parent"
    second = "2026-01-02-foreign-child"
    _write_selector(tmp_path, [_scope(first), _scope(second)])
    _write_rows(
        tmp_path,
        first,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "from v1"},
        ],
    )
    _write_rows(
        tmp_path,
        second,
        [{"citation_path": f"{CITATION}/2", "body": "from v2"}],
    )

    with pytest.raises(CorpusResolutionError, match="cross active release scopes"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_enforces_local_descendant_row_bound(
    tmp_path: Path,
    monkeypatch,
):
    version = "2026-01-01-descendant-bound"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "one"},
            {"citation_path": f"{CITATION}/2", "body": "two"},
        ],
    )
    monkeypatch.setattr(corpus_resolver, "MAX_CORPUS_DESCENDANT_ROWS", 1)

    with pytest.raises(CorpusDescendantStructureError, match="1-row safety limit"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_enforces_composed_byte_bound(tmp_path: Path, monkeypatch):
    version = "2026-01-01-composed-bound"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "too long"},
        ],
    )
    monkeypatch.setattr(corpus_resolver, "MAX_COMPOSED_CORPUS_BYTES", 3)

    with pytest.raises(CorpusResolutionError, match="Composed corpus source"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "match"),
    [
        ("MAX_COMPOSITION_NODES", 1, "1-node safety limit"),
        ("MAX_COMPOSITION_PREFIX_BYTES", 1, "1-byte prefix safety limit"),
    ],
)
def test_metadata_parent_bounds_composition_prefix_trie(
    tmp_path: Path,
    monkeypatch,
    limit_name: str,
    limit_value: int,
    match: str,
):
    version = "2026-01-01-composition-trie-bound"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1/A/i", "body": "deep body"},
        ],
    )
    monkeypatch.setattr(corpus_resolver, limit_name, limit_value)

    with pytest.raises(CorpusDescendantStructureError, match=match):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


@pytest.mark.parametrize(
    ("children", "expected_paths"),
    [
        (
            [
                {"citation_path": f"{CITATION}/10", "body": "ten"},
                {"citation_path": f"{CITATION}/2", "body": "two"},
            ],
            [f"{CITATION}/2", f"{CITATION}/10"],
        ),
        (
            [
                {"citation_path": f"{CITATION}/10", "body": "ten"},
                {
                    "citation_path": f"{CITATION}/2",
                    "body": "two",
                    "ordinal": 1,
                },
            ],
            [f"{CITATION}/2", f"{CITATION}/10"],
        ),
        (
            [
                {
                    "citation_path": f"{CITATION}/10",
                    "body": "ten",
                    "ordinal": 1,
                },
                {
                    "citation_path": f"{CITATION}/2",
                    "body": "two",
                    "ordinal": 2,
                },
            ],
            [f"{CITATION}/10", f"{CITATION}/2"],
        ),
        (
            [
                {
                    "citation_path": f"{CITATION}/10",
                    "body": "ten",
                    "ordinal": 1,
                },
                {
                    "citation_path": f"{CITATION}/2",
                    "body": "two",
                    "ordinal": 1,
                },
            ],
            [f"{CITATION}/2", f"{CITATION}/10"],
        ),
        (
            [
                {
                    "citation_path": f"{CITATION}/10",
                    "body": "ten",
                    "ordinal": 1,
                },
                {
                    "citation_path": f"{CITATION}/2/A",
                    "body": "two A",
                    "ordinal": 1,
                },
            ],
            [f"{CITATION}/2/A", f"{CITATION}/10"],
        ),
    ],
)
def test_metadata_parent_orders_sibling_subtrees_with_partial_metadata(
    tmp_path: Path,
    children: list[dict[str, object]],
    expected_paths: list[str],
):
    version = "2026-01-01-ordering-metadata"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": None}, *children],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert [row.citation_path for row in resolved.component_rows] == expected_paths


def test_metadata_parent_rejects_negative_ordering_metadata(tmp_path: Path):
    version = "2026-01-01-negative-ordering"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {
                "citation_path": f"{CITATION}/1",
                "body": "child",
                "ordinal": -1,
            },
        ],
    )

    with pytest.raises(CorpusResolutionError, match="non-negative integer"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_orders_virtual_canonical_roman_siblings(tmp_path: Path):
    version = "2026-01-01-virtual-roman-order"
    _write_selector(tmp_path, [_scope(version)])
    parent = "us/statute/7/9999/a/1/A"
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": parent, "body": None},
            {"citation_path": f"{parent}/ix/A", "body": "ninth"},
            {"citation_path": f"{parent}/v/A", "body": "fifth"},
        ],
    )

    resolved = resolve_local_corpus_source(parent, _release(tmp_path))

    assert resolved.body == "fifth\n\nninth"
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{parent}/v/A",
        f"{parent}/ix/A",
    ]


def test_metadata_parent_rejects_duplicate_bodyless_descendant(tmp_path: Path):
    version = "2026-01-01-children"
    _write_selector(tmp_path, [_scope(version)])
    child = f"{CITATION}/1"
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": child, "body": None},
            {"citation_path": child, "body": "duplicate"},
        ],
    )

    with pytest.raises(AmbiguousCorpusSourceError) as exc_info:
        resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert exc_info.value.citation_path == child
    assert len(exc_info.value.rows) == 2


def test_metadata_parent_validates_descendant_citation_before_body_filter(
    tmp_path: Path,
):
    version = "2026-01-01-children"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/..", "body": None},
            {"citation_path": f"{CITATION}/1", "body": "valid"},
        ],
    )

    with pytest.raises(CorpusDescendantStructureError, match="Invalid descendant"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_types_malformed_descendant_ordering(tmp_path: Path):
    version = "2026-01-01-ordering"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "child", "ordinal": "first"},
        ],
    )

    with pytest.raises(CorpusDescendantStructureError, match="ordering metadata"):
        resolve_local_corpus_source(CITATION, _release(tmp_path))


def test_metadata_parent_ignores_malformed_inactive_descendant(tmp_path: Path):
    active = "2026-01-01-active"
    inactive = "2025-01-01-inactive"
    _write_selector(tmp_path, [_scope(active)])
    _write_rows(
        tmp_path,
        active,
        [
            {"citation_path": CITATION, "body": None},
            {"citation_path": f"{CITATION}/1", "body": "active child"},
        ],
    )
    _write_rows(
        tmp_path,
        inactive,
        [
            {
                "citation_path": f"{CITATION}/stale",
                "jurisdiction": "ca",
                "body": "malformed inactive child",
            }
        ],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "active child"
    assert [row.citation_path for row in resolved.component_rows] == [f"{CITATION}/1"]


def test_local_artifact_is_read_once_for_hash_and_resolution(
    tmp_path: Path, monkeypatch
):
    version = "2026-01-01-active"
    _write_selector(tmp_path, [_scope(version)])
    provision_file = _write_rows(
        tmp_path,
        version,
        [{"citation_path": CITATION, "body": "body"}],
    ).resolve()
    original = corpus_resolver.read_bounded_regular_file
    reads: list[Path] = []

    def tracked_read(root, candidate, *, label, max_bytes):
        if Path(candidate) == provision_file:
            reads.append(Path(candidate))
        return original(root, candidate, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(corpus_resolver, "read_bounded_regular_file", tracked_read)

    resolve_local_corpus_source(CITATION, _release(tmp_path))
    assert len(reads) == 1
    reads.clear()
    list(iter_active_local_corpus_rows(_release(tmp_path)))
    assert len(reads) == 1


def test_local_artifact_swap_to_outside_symlink_fails_closed(
    tmp_path: Path, monkeypatch
):
    corpus_root = tmp_path / "corpus"
    version = "2026-01-01-active"
    _write_selector(corpus_root, [_scope(version)])
    provision_file = _write_rows(
        corpus_root,
        version,
        [{"citation_path": CITATION, "body": "inside"}],
    )
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        provision_file.read_text(encoding="utf-8").replace("inside", "outside"),
        encoding="utf-8",
    )
    original = corpus_resolver._safe_file
    swapped = False

    def swap_after_check(root, candidate, *, label, max_bytes):
        nonlocal swapped
        resolved = original(root, candidate, label=label, max_bytes=max_bytes)
        if label == "corpus provision file" and resolved is not None and not swapped:
            swapped = True
            resolved.unlink()
            resolved.symlink_to(outside)
        return resolved

    monkeypatch.setattr(corpus_resolver, "_safe_file", swap_after_check)

    with pytest.raises(UnsafeCorpusPathError, match="safely open"):
        resolve_local_corpus_source(CITATION, _release(corpus_root))


def test_local_artifact_swap_to_fifo_fails_closed_without_blocking(
    tmp_path: Path, monkeypatch
):
    corpus_root = tmp_path / "corpus"
    version = "2026-01-01-active"
    _write_selector(corpus_root, [_scope(version)])
    _write_rows(
        corpus_root,
        version,
        [{"citation_path": CITATION, "body": "inside"}],
    )
    original = corpus_resolver._safe_file
    swapped = False

    def swap_after_check(root, candidate, *, label, max_bytes):
        nonlocal swapped
        resolved = original(root, candidate, label=label, max_bytes=max_bytes)
        if label == "corpus provision file" and resolved is not None and not swapped:
            swapped = True
            resolved.unlink()
            os.mkfifo(resolved)
        return resolved

    monkeypatch.setattr(corpus_resolver, "_safe_file", swap_after_check)

    with pytest.raises(UnsafeCorpusPathError, match="not a regular file"):
        resolve_local_corpus_source(CITATION, _release(corpus_root))


def test_release_object_swap_to_outside_symlink_fails_closed(
    tmp_path: Path, monkeypatch
):
    corpus_root = tmp_path / "corpus"
    version = "2026-01-01-active"
    _write_selector(corpus_root, [_scope(version)])
    _write_rows(
        corpus_root,
        version,
        [{"citation_path": CITATION, "body": "inside"}],
    )
    release = _release(corpus_root)
    outside = tmp_path / "outside-release-object.json"
    outside.write_text(
        release.release_object_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    original = corpus_resolver._safe_file
    swapped = False

    def swap_after_check(root, candidate, *, label, max_bytes):
        nonlocal swapped
        resolved = original(root, candidate, label=label, max_bytes=max_bytes)
        if label == "corpus release object" and resolved is not None and not swapped:
            swapped = True
            resolved.unlink()
            resolved.symlink_to(outside)
        return resolved

    monkeypatch.setattr(corpus_resolver, "_safe_file", swap_after_check)

    with pytest.raises(UnsafeCorpusPathError, match="safely open"):
        LocalCorpusRelease(
            corpus_root,
            TEST_RELEASE,
            release.content_sha256,
            TEST_RELEASE_PUBLIC_KEY,
        )


@pytest.mark.parametrize(
    "status_fields,expected",
    [
        ({"metadata": {"status": "repealed"}}, True),
        ({"metadata": {"status": "active"}}, False),
        ({"metadata": {"status": "REPEALED"}}, False),
        ({"metadata": {"status": "repealed "}}, False),
        ({"status": "repealed"}, False),
        ({}, False),
    ],
)
def test_exact_release_row_status_can_authenticate_repeal(
    tmp_path, status_fields, expected
):
    version = "2026-10-15-reviewed-repeal"
    _write_selector(tmp_path, [_scope(version)])
    child = CITATION + "/2"
    _write_rows(
        tmp_path,
        version,
        [{"citation_path": child, "body": None, **status_fields}],
    )
    release = _release(tmp_path)
    assert corpus_resolver.local_corpus_row_is_repealed(child, release) is expected
    # An exact descendant tombstone does not authenticate its parent or sibling.
    assert not corpus_resolver.local_corpus_row_is_repealed(CITATION, release)
    assert not corpus_resolver.local_corpus_row_is_repealed(CITATION + "/3", release)


def test_repealed_row_status_refuses_aliases_and_ambiguous_rows(tmp_path):
    version = "2026-10-15-ambiguous-repeal"
    _write_selector(tmp_path, [_scope(version)])
    row = {
        "citation_path": CITATION,
        "body": None,
        "metadata": {"status": "repealed"},
    }
    _write_rows(tmp_path, version, [row, row])
    release = _release(tmp_path)
    with pytest.raises(InvalidCorpusCitationError):
        corpus_resolver.local_corpus_row_is_repealed("7 USC 2014(e)", release)
    with pytest.raises(AmbiguousCorpusSourceError):
        corpus_resolver.local_corpus_row_is_repealed(CITATION, release)


def test_coverage_repeal_status_uses_only_the_bound_exact_release_row(tmp_path):
    from axiom_encode.harness import validator_pipeline as vp

    version = "2026-10-15-bound-repeal"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {
                "citation_path": CITATION + "/2",
                "body": None,
                "metadata": {"status": "repealed"},
            }
        ],
    )
    token = vp._AUTHORITATIVE_CORPUS_RELEASE.set(_release(tmp_path))
    try:
        assert vp._requested_corpus_row_is_repealed(CITATION + "/2")
        assert not vp._requested_corpus_row_is_repealed(CITATION + "/3")
        assert not vp._requested_corpus_row_is_repealed(None)
    finally:
        vp._AUTHORITATIVE_CORPUS_RELEASE.reset(token)


@pytest.mark.parametrize("label", ["I", "i"])
@pytest.mark.parametrize("status", ["repealed", "active", None])
@pytest.mark.parametrize(
    "request_form",
    ["corpus", "human", "target", "rules_file", "rules_file_subsection", "precedence"],
)
def test_coverage_repeal_authenticates_case_sensitive_requested_row(
    tmp_path, label, status, request_form
):
    import yaml

    from axiom_encode.harness import validator_pipeline as vp

    section = "us/statute/42/1"
    citation = section + "/a" if request_form == "rules_file_subsection" else section
    requested = f"{section}/a/2/{label}"
    other = f"{section}/a/2/{label.swapcase()}"
    source = (
        '(a) "benefit" means payable.\n(2) A person is eligible.\n'
        f"({label}) Repealed.\n(b) Effective date.\n"
    )
    version = "case-sensitive-repeal"
    _write_selector(tmp_path, [_scope(version)])
    rows = [{"citation_path": citation, "body": source}]
    if status is not None:
        rows.append(
            {
                "citation_path": requested,
                "body": None if status == "repealed" else "An operative child.",
                "metadata": {"status": status},
            }
        )
    # A missing/active requested row must not borrow the other case's status.
    # Positive cases have only the literal requested tombstone.
    if status != "repealed":
        rows.append(
            {
                "citation_path": other,
                "body": None,
                "metadata": {"status": "repealed"},
            }
        )
    _write_rows(tmp_path, version, rows)
    release = _release(tmp_path)
    assert corpus_resolver.local_corpus_row_is_repealed(requested, release) is (
        status == "repealed"
    )
    requested_source = {
        "corpus": f" /{requested}/ ",
        "human": f"42 USC 1(a)(2)({label})",
        "target": f"us:statutes/42/1/a/2/{label}",
        "rules_file": None,
        "rules_file_subsection": None,
        "precedence": requested,
    }[request_form]
    file_label = label.swapcase() if request_form == "precedence" else label
    rules_file = tmp_path / f"statutes/42/1/a/2/{file_label}.yaml"

    def issues(
        *, summary=f"({label}) Repealed.", text=source, module_status="deferred"
    ):
        content = yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {
                    "status": module_status,
                    "summary": summary,
                    "source_verification": {"corpus_citation_path": citation},
                },
                "rules": [],
            }
        )
        return vp.find_source_subparagraph_coverage_issues(
            content,
            source_texts={citation: text},
            requested_source=requested_source,
            rules_file=rules_file,
        )

    token = vp._AUTHORITATIVE_CORPUS_RELEASE.set(release)
    try:
        assert len(issues()) == (0 if status == "repealed" else 1)
        # Signed metadata supplements all of the original notice conditions.
        assert len(issues(summary="(3) Repealed.")) == 1
        assert len(issues(summary=f"({label}) An operative provision.")) == 1
        assert len(issues(module_status="active")) == 1
        assert len(issues(text=source.replace("Repealed.", "An operative child."))) == 1
    finally:
        vp._AUTHORITATIVE_CORPUS_RELEASE.reset(token)

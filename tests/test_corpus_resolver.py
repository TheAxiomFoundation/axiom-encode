"""Focused tests for release-aware corpus source resolution."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

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
        [{"citation_path": "us/statute/7/2014", "body": parent_body}],
    )

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))
    assert resolved.citation_path == "us/statute/7/2014"
    assert resolved.body == "(e) Target $198."
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


def test_metadata_parent_composes_descendants_deterministically(tmp_path: Path):
    version = "2026-01-01-children"
    _write_selector(tmp_path, [_scope(version)])
    _write_rows(
        tmp_path,
        version,
        [
            {"citation_path": CITATION, "body": None, "heading": "Parent"},
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

    resolved = resolve_local_corpus_source(CITATION, _release(tmp_path))

    assert resolved.body == "First heading\n\nfirst\n\nsecond"
    assert [row.citation_path for row in resolved.component_rows] == [
        f"{CITATION}/1",
        f"{CITATION}/2",
    ]
    assert all(row.provision_file_sha256 for row in resolved.component_rows)


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

"""Focused tests for release-aware dependency corpus lookup."""

from __future__ import annotations

import json
from pathlib import Path

from axiom_encode.harness.dependency_stubs import (
    _corpus_file_contains_citation_path,
    find_corpus_provision_artifacts,
)

CITATION_PATH = "us/statute/7/2014/e/4"


def _write_release(corpus_root: Path, *, version: str) -> None:
    selector = corpus_root / "manifests" / "releases" / "current.json"
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(
        json.dumps(
            {
                "name": "current",
                "scopes": [
                    {
                        "jurisdiction": "us",
                        "document_class": "statute",
                        "version": version,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_provision(
    corpus_root: Path,
    *,
    filename: str,
    version: str,
    record_id: str,
    citation_path: str = CITATION_PATH,
    body: str | None = "Authoritative active source text.",
) -> Path:
    artifact = (
        corpus_root / "data" / "corpus" / "provisions" / "us" / "statute" / filename
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(
            {
                "id": record_id,
                "citation_path": citation_path,
                "jurisdiction": "us",
                "document_class": "statute",
                "version": version,
                "body": body,
                "source_path": "sources/us/statute/example.html",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact


def test_find_corpus_provision_artifacts_returns_only_active_release_file(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    inactive = _write_provision(
        corpus_root,
        filename="inactive.jsonl",
        version="old-release",
        record_id="inactive-row",
    )
    active = _write_provision(
        corpus_root,
        filename="active.jsonl",
        version="active-release",
        record_id="active-row",
    )
    _write_release(corpus_root, version="active-release")

    assert find_corpus_provision_artifacts(
        CITATION_PATH,
        rules_root,
        corpus_root=corpus_root,
    ) == [active.resolve()]
    assert _corpus_file_contains_citation_path(active, CITATION_PATH)
    assert not _corpus_file_contains_citation_path(inactive, CITATION_PATH)


def test_find_corpus_provision_artifacts_returns_component_provenance(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    parent = _write_provision(
        corpus_root,
        filename="parent.jsonl",
        version="active-release",
        record_id="parent-row",
        body=None,
    )
    component = _write_provision(
        corpus_root,
        filename="component.jsonl",
        version="active-release",
        record_id="component-row",
        citation_path=f"{CITATION_PATH}/1",
    )
    _write_release(corpus_root, version="active-release")

    assert find_corpus_provision_artifacts(
        CITATION_PATH,
        rules_root,
        corpus_root=corpus_root,
    ) == [parent.resolve(), component.resolve()]
    assert _corpus_file_contains_citation_path(parent, CITATION_PATH)
    assert _corpus_file_contains_citation_path(component, CITATION_PATH)


def test_find_corpus_provision_artifacts_requires_release_selector(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    artifact = _write_provision(
        corpus_root,
        filename="unselected.jsonl",
        version="unselected-release",
        record_id="unselected-row",
    )

    assert (
        find_corpus_provision_artifacts(
            CITATION_PATH,
            rules_root,
            corpus_root=corpus_root,
        )
        == []
    )
    assert not _corpus_file_contains_citation_path(artifact, CITATION_PATH)


def test_find_corpus_provision_artifacts_fails_closed_on_active_duplicate(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    first = _write_provision(
        corpus_root,
        filename="first.jsonl",
        version="active-release",
        record_id="first-row",
    )
    second = _write_provision(
        corpus_root,
        filename="second.jsonl",
        version="active-release",
        record_id="second-row",
    )
    _write_release(corpus_root, version="active-release")

    assert (
        find_corpus_provision_artifacts(
            CITATION_PATH,
            rules_root,
            corpus_root=corpus_root,
        )
        == []
    )
    assert not _corpus_file_contains_citation_path(first, CITATION_PATH)
    assert not _corpus_file_contains_citation_path(second, CITATION_PATH)

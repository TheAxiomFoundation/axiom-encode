"""Focused tests for release-aware dependency corpus lookup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_encode.corpus_resolver import (
    AmbiguousCorpusSourceError,
    CorpusSourceSliceError,
    InvalidCorpusReleaseError,
    LocalCorpusRelease,
)
from axiom_encode.harness.dependency_stubs import (
    UnsafeRulespecContextPath,
    find_corpus_provision_artifacts,
    import_target_to_relative_rulespec_path,
    resolve_canonical_concepts_from_text,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
)

CITATION_PATH = "us/statute/7/2014/e/4"
TEST_SELECTOR = "dependency-test-release"


def _write_release(corpus_root: Path, *, version: str) -> LocalCorpusRelease:
    selector = corpus_root / "manifests" / "releases" / f"{TEST_SELECTOR}.json"
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_text(
        json.dumps(
            {
                "name": TEST_SELECTOR,
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
    return bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [("us", "statute", version)],
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
    del filename
    artifact = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / "us"
        / "statute"
        / f"{version}.jsonl"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    existing = artifact.read_text(encoding="utf-8") if artifact.exists() else ""
    artifact.write_text(
        existing
        + json.dumps(
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


def _write_concept_module(content_root: Path, source_root: str) -> Path:
    path = content_root / source_root / "example.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        '  summary: The "canonical benefit concept" means the source-defined condition.\n'
        "rules:\n"
        "  - name: canonical_benefit_concept\n"
        "    kind: derived\n"
        "    entity: Person\n"
        "    period: Year\n"
        "    dtype: Boolean\n"
        "    versions:\n"
        "      - effective_from: 2026-01-01\n"
        "        formula: true\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "source_root",
    ["legislation", "policies", "regulations", "statutes"],
)
def test_canonical_concept_discovery_accepts_each_atomic_root(tmp_path, source_root):
    content_root = tmp_path / "rulespec-us" / "us"
    module = _write_concept_module(content_root, source_root)

    concepts = resolve_canonical_concepts_from_text(
        "A canonical benefit concept applies.",
        content_root,
    )

    assert len(concepts) == 1
    assert concepts[0].rulespec_file == module.resolve()
    assert concepts[0].import_target == (
        f"us:{source_root}/example#canonical_benefit_concept"
    )


def test_canonical_concept_discovery_does_not_read_composition_specs(tmp_path):
    content_root = tmp_path / "rulespec-us" / "us"
    module = _write_concept_module(content_root, "statutes")
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: do-not-read\n", encoding="utf-8")
    program_alias = content_root / "programs" / "benefit" / "fy-2026.yaml"
    program_alias.parent.mkdir(parents=True)
    program_alias.symlink_to(outside)

    concepts = resolve_canonical_concepts_from_text(
        "A canonical benefit concept applies.",
        content_root,
    )

    assert [concept.rulespec_file for concept in concepts] == [module.resolve()]


@pytest.mark.parametrize(
    "import_target",
    [
        "programs/snap/fy-2026#snap_eligible",
        "us:programs/snap/fy-2026#snap_eligible",
    ],
)
def test_dependency_stub_path_rejects_composition_spec_imports(import_target):
    with pytest.raises(UnsafeRulespecContextPath, match="four atomic"):
        import_target_to_relative_rulespec_path(import_target)


def test_canonical_concept_discovery_rejects_singular_regulation_root(tmp_path):
    content_root = tmp_path / "rulespec-us" / "us"
    _write_concept_module(content_root, "regulation")

    assert (
        resolve_canonical_concepts_from_text(
            "A canonical benefit concept applies.",
            content_root,
        )
        == []
    )


def test_find_corpus_provision_artifacts_returns_only_active_release_file(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    _write_provision(
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
    release = _write_release(corpus_root, version="active-release")

    assert find_corpus_provision_artifacts(
        CITATION_PATH,
        rules_root,
        corpus_release=release,
    ) == [active.resolve()]


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
    release = _write_release(corpus_root, version="active-release")

    assert find_corpus_provision_artifacts(
        CITATION_PATH,
        rules_root,
        corpus_release=release,
    ) == [parent.resolve()]
    assert parent.resolve() == component.resolve()


def test_find_corpus_provision_artifacts_requires_release_selector(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    _write_provision(
        corpus_root,
        filename="unselected.jsonl",
        version="unselected-release",
        record_id="unselected-row",
    )

    with pytest.raises(InvalidCorpusReleaseError):
        LocalCorpusRelease(
            corpus_root,
            TEST_SELECTOR,
            "0" * 64,
            TEST_RELEASE_PUBLIC_KEY,
        )


def test_find_corpus_provision_artifacts_fails_closed_on_active_duplicate(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-us"
    rules_root.mkdir()
    _write_provision(
        corpus_root,
        filename="first.jsonl",
        version="active-release",
        record_id="first-row",
    )
    _write_provision(
        corpus_root,
        filename="second.jsonl",
        version="active-release",
        record_id="second-row",
    )
    release = _write_release(corpus_root, version="active-release")

    with pytest.raises(AmbiguousCorpusSourceError):
        find_corpus_provision_artifacts(
            CITATION_PATH,
            rules_root,
            corpus_release=release,
        )


def test_uk_statute_dependency_uses_shared_resolver_parent_fallback(tmp_path):
    corpus_root = tmp_path / "axiom-corpus"
    rules_root = tmp_path / "rulespec-uk"
    rules_root.mkdir()
    version = "2026-uk-statute"
    artifact = corpus_root / f"data/corpus/provisions/uk/statute/{version}.jsonl"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "id": "ukpga-2007-3",
                "citation_path": "uk/statute/ukpga/2007/3",
                "jurisdiction": "uk",
                "document_class": "statute",
                "version": version,
                "body": "(35) Personal allowance applies.\n\n(36) A sibling rule.",
                "source_path": "sources/uk/statute/ukpga-2007-3.xml",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n"
    )
    release = bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [("uk", "statute", version)],
    )

    assert find_corpus_provision_artifacts(
        "uk:statute/ukpga/2007/3/35",
        rules_root,
        corpus_release=release,
    ) == [artifact.resolve()]

    # Statutes participate in parent fallback, but still fail closed when
    # the requested child marker cannot be isolated from the parent body.
    with pytest.raises(CorpusSourceSliceError, match="Could not isolate"):
        find_corpus_provision_artifacts(
            "uk:statute/ukpga/2007/3/999",
            rules_root,
            corpus_release=release,
        )

"""Offline proposal and corpus-check provenance must fail closed."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from axiom_encode.corpus_release import CorpusReleaseObjectError
from axiom_encode.harness import coverage_exemptions as registry
from scripts import check_coverage_exemptions as checker
from scripts import propose_coverage_exemptions as proposal
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    write_test_release_object,
)


@pytest.fixture
def corpus(tmp_path):
    citation = "us-co/statute/39/39-1-1"
    body = '(1) General definitions.\n(a) "benefit" means an amount payable.\n'
    artifact = "data/corpus/provisions/us-co/statute/2026-fixture.jsonl"
    path = tmp_path / artifact
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"citation_path": citation, "body": body}) + "\n")
    digest = write_test_release_object(
        tmp_path, "coverage-fixture", [("us-co", "statute", "2026-fixture")]
    )
    return proposal.load_release(
        tmp_path, "coverage-fixture", digest, public_keys=(TEST_RELEASE_PUBLIC_KEY,)
    )


def _entry(row):
    demand = proposal.scan_main_demands(row.text)[0]
    window = registry.source_window(row.text, demand.physical_start)
    return registry.CoverageExemption.from_mapping(
        {
            "citation_path": row.citation_path,
            "requested_scope": None,
            "source_text_sha256": proposal.sha256_text(row.text),
            "label": demand.label,
            "physical_start": demand.physical_start,
            "window": window,
            "window_sha256": proposal.sha256_text(window),
            "reason": "nested_under_numbered_parent",
            "owner_evidence": {
                "citation_path": row.citation_path + "/1",
                "physical_start": 0,
            },
            "corpus_artifact": row.corpus_artifact,
            "artifact_sha256": row.artifact_sha256,
            "row": row.row,
            "reviewer": "fixture reviewer",
            "reviewed_at": "2026-10-15",
        }
    )


def test_candidates_need_review_and_cannot_load_as_approved_entries(corpus):
    row = next(proposal.iter_release_rows(corpus))
    candidates = list(proposal.propose_row(row, None))
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["candidate"] is True
    assert candidate["reviewer"] is None
    assert candidate["reviewed_at"] is None
    assert candidate["reason"] is None
    assert candidate["owner_evidence"] is None
    assert candidate["proposal_context"]["source_text"] == row.text
    assert candidate["source_text_sha256"] == proposal.sha256_text(row.text)
    with pytest.raises(ValueError, match="missing or unknown fields"):
        registry.CoverageExemption.from_mapping(candidate)


def test_proposal_cli_only_prints_candidates(corpus, monkeypatch, capsys):
    monkeypatch.setattr(proposal, "load_release", lambda *args: corpus)
    before = {path.relative_to(corpus.root) for path in corpus.root.rglob("*")}
    assert (
        proposal.main(
            ["--release", "coverage-fixture", "--corpus-path", str(corpus.root)]
        )
        == 0
    )
    output = capsys.readouterr()
    assert json.loads(output.out)["candidate"] is True
    assert "1 unapproved candidates" in output.err
    assert {path.relative_to(corpus.root) for path in corpus.root.rglob("*")} == before


def test_proposals_keep_the_main_scanner_requested_scope(corpus):
    row = next(proposal.iter_release_rows(corpus))
    row = replace(row, citation_path=row.citation_path + "/b")
    assert list(proposal.propose_row(row, ["b"])) == []


def test_exact_row_suffix_has_no_parent_fallback_scope(corpus):
    row = next(proposal.iter_release_rows(corpus))
    row = replace(row, citation_path=row.citation_path + "/f")
    candidates = list(proposal.propose_row(row))
    assert len(candidates) == 1
    assert candidates[0]["requested_scope"] is None


def test_requested_source_is_exact_target_or_descendant(corpus):
    row = next(proposal.iter_release_rows(corpus))
    assert (
        proposal.requested_scope_for_source(row.citation_path, row.citation_path)
        is None
    )
    assert proposal.requested_scope_for_source(
        row.citation_path, row.citation_path + "/a/2"
    ) == ["a", "2"]
    with pytest.raises(ValueError, match="exact row or its descendant"):
        proposal.requested_scope_for_source(row.citation_path, "us/statute/26/63")


def test_explicit_requested_source_requires_one_row(corpus):
    with pytest.raises(SystemExit) as error:
        proposal.main(
            [
                "--release",
                "coverage-fixture",
                "--corpus-path",
                str(corpus.root),
                "--requested-source",
                "us/statute/26/63",
            ]
        )
    assert error.value.code == 2


def test_proposal_context_records_literal_requested_source(corpus):
    row = next(proposal.iter_release_rows(corpus))
    source = row.citation_path + "/a"
    candidate = next(proposal.propose_row(row, ["a"], requested_source=source))
    assert candidate["proposal_context"]["requested_source"] == source


@pytest.mark.parametrize("label", ["I", "i"])
def test_proposal_cli_keeps_literal_requested_scope(corpus, monkeypatch, capsys, label):
    row = next(proposal.iter_release_rows(corpus))
    requested = f"{row.citation_path}/a/2/{label}"
    monkeypatch.setattr(proposal, "load_release", lambda *args: corpus)
    assert (
        proposal.main(
            [
                "--release",
                "coverage-fixture",
                "--corpus-path",
                str(corpus.root),
                "--citation",
                row.citation_path,
                "--requested-source",
                requested,
            ]
        )
        == 0
    )
    candidate = json.loads(capsys.readouterr().out)
    assert candidate["requested_scope"] == ["a", "2", label]
    assert candidate["proposal_context"]["requested_source"] == requested


def _resign_rows(corpus, records):
    path = corpus.root / corpus.artifacts[0].path
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    )
    digest = write_test_release_object(
        corpus.root, "coverage-fixture", [("us-co", "statute", "2026-fixture")]
    )
    return proposal.load_release(
        corpus.root,
        "coverage-fixture",
        digest,
        public_keys=(TEST_RELEASE_PUBLIC_KEY,),
    )


def test_unicode_line_separators_inside_json_preserve_physical_rows(corpus):
    body = '(a) "benefit" means Unicode\u2028and\u0085markup.\n'
    corpus = _resign_rows(
        corpus, [{"citation_path": "us-co/statute/39/39-1-1", "body": body}]
    )
    rows = list(proposal.iter_release_rows(corpus))
    assert len(rows) == 1
    assert rows[0].row == 1
    assert rows[0].text == body


@pytest.mark.parametrize(
    ("top_level_status", "metadata", "expected"),
    [
        ("active", {"status": "repealed"}, "repealed"),
        ("repealed", {}, None),
        ("repealed", {"status": "REPEALED"}, None),
    ],
)
def test_proposal_status_uses_exact_resolver_metadata(
    corpus, top_level_status, metadata, expected
):
    row = next(proposal.iter_release_rows(corpus))
    corpus = _resign_rows(
        corpus,
        [
            {
                "citation_path": row.citation_path,
                "body": row.text,
                "status": top_level_status,
                "metadata": metadata,
            }
        ],
    )
    assert next(proposal.iter_release_rows(corpus)).status == expected


def test_corpus_check_reproduces_exact_list_and_rejects_stale_row(corpus):
    row = next(proposal.iter_release_rows(corpus))
    entry = _entry(row)
    valid = checker.check_rows(corpus, (entry,))
    assert valid["passed"] is True
    assert valid["entries_validated"] == 1
    assert valid["counts"]["physical_removals"] == 1
    assert valid["counts"]["physical_additions"] == 0
    stale = checker.check_rows(corpus, (replace(entry, row=2),))
    assert stale["passed"] is False
    assert stale["errors"][0]["kind"] == "unseen_entry"


@pytest.mark.parametrize(
    "field", ["source_text_sha256", "artifact_sha256", "physical_start"]
)
def test_corpus_check_rejects_invented_provenance(corpus, field):
    entry = _entry(next(proposal.iter_release_rows(corpus)))
    value = 999 if field == "physical_start" else "0" * 64
    ledger = checker.check_rows(corpus, (replace(entry, **{field: value}),))
    assert ledger["passed"] is False
    assert ledger["entries_validated"] == 0


def test_corpus_check_detects_widened_runtime_filter(corpus, monkeypatch):
    entry = _entry(next(proposal.iter_release_rows(corpus)))
    monkeypatch.setattr(registry, "filter_coverage_demands", lambda *args, **kwargs: [])
    ledger = checker.check_rows(corpus, ())
    assert ledger["passed"] is False
    assert ledger["errors"][0]["kind"] == "differential_mismatch"
    assert checker.check_rows(corpus, (entry,))["passed"] is True


def test_corpus_check_rejects_an_entry_outside_its_requested_scope(corpus):
    entry = _entry(next(proposal.iter_release_rows(corpus)))
    ledger = checker.check_rows(corpus, (replace(entry, requested_scope=("b",)),))
    assert ledger["passed"] is False
    assert ledger["entries_validated"] == 0
    assert ledger["counts"]["physical_removals"] == 0
    assert ledger["errors"][0]["kind"] == "stale_entry"


def test_artifact_byte_change_is_rejected(corpus):
    path = corpus.root / corpus.artifacts[0].path
    path.write_bytes(path.read_bytes().replace(b"payable", b"payablf"))
    with pytest.raises(ValueError, match="artifact bytes do not match"):
        list(proposal.iter_release_rows(corpus))


def test_release_signature_and_explicit_pin_are_verified(corpus):
    object_path = (
        corpus.root
        / "releases"
        / corpus.release.name
        / f"{corpus.release.content_sha256}.json"
    )
    with pytest.raises(ValueError, match="requested release pin"):
        proposal.load_release(
            corpus.root,
            corpus.release.name,
            "0" * 64,
            object_path,
            public_keys=(TEST_RELEASE_PUBLIC_KEY,),
        )
    payload = json.loads(object_path.read_text())
    payload["signature"]["value"] = "A" * 88
    object_path.write_text(json.dumps(payload))
    with pytest.raises(CorpusReleaseObjectError, match="signature"):
        proposal.load_release(
            corpus.root,
            corpus.release.name,
            release_object=object_path,
            public_keys=(TEST_RELEASE_PUBLIC_KEY,),
        )


def test_explicit_artifact_slice_is_bound_to_signed_inventory(corpus, tmp_path):
    artifact = corpus.artifacts[0]
    manifest = {
        "release": corpus.release.name,
        "release_sha256": corpus.release.content_sha256,
        "artifacts": [
            {
                **asdict(artifact),
                "bytes": artifact.byte_count,
                "rows": artifact.row_count,
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    assert proposal.select_artifacts(corpus, path).artifacts == corpus.artifacts
    manifest["artifacts"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="signed inventory"):
        proposal.select_artifacts(corpus, path)

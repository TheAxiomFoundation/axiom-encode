"""Tests for the dry-run corpus attestation carry check."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from axiom_encode import entrypoint
from axiom_encode.corpus_attestation_carry import (
    RELEASE_IDENTITY_FIELDS,
    CarryStatus,
    ResolvedUnitEvidence,
    _corpus_resolver,
    classify_attestation_carry,
    iter_manifest_attestations,
    run_corpus_attestation_carry,
)
from axiom_encode.corpus_release import verify_release_object
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    bind_test_corpus_release,
    write_test_release_object,
)

UNIT = ("us", "statute", "2026-01-01-unit")
SIBLING = ("us", "statute", "2026-01-02-sibling")
OTHER_PAIR = ("us", "regulation", "2026-01-03-other")
ADDED = ("us", "statute", "2026-01-04-added")
UNIT_CITATION = "us/statute/7/2014/e"


def _provisions_path(root: Path, scope: tuple[str, str, str]) -> Path:
    jurisdiction, document_class, version = scope
    return (
        root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / document_class
        / f"{version}.jsonl"
    )


def _write_scope(
    root: Path,
    scope: tuple[str, str, str],
    citation_path: str,
    body: str,
) -> None:
    jurisdiction, document_class, version = scope
    path = _provisions_path(root, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "id": f"row-{version}",
        "jurisdiction": jurisdiction,
        "document_class": document_class,
        "version": version,
        "citation_path": citation_path,
        "body": body,
        "source_path": f"sources/{jurisdiction}/{document_class}/{version}/s.xml",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def _signed(root: Path, name: str, scopes: list[tuple[str, str, str]]):
    content_sha256 = write_test_release_object(root, name, scopes)
    payload = json.loads(
        (root / "releases" / name / f"{content_sha256}.json").read_text("utf-8")
    )
    return verify_release_object(payload, public_key=TEST_RELEASE_PUBLIC_KEY)


def _rel(scope: tuple[str, str, str]) -> str:
    return "data/corpus/provisions/{}/{}/{}.jsonl".format(*scope)


def _attestation(root: Path, release, scope=UNIT, component_scope=None) -> dict:
    path = _provisions_path(root, scope)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    row = {"provision_file": _rel(scope), "provision_file_sha256": digest}
    components = []
    if component_scope is not None:
        component_digest = hashlib.sha256(
            _provisions_path(root, component_scope).read_bytes()
        ).hexdigest()
        components.append(
            {
                "provision_file": _rel(component_scope),
                "provision_file_sha256": component_digest,
            }
        )
    return {
        "corpus_release": release.name,
        "corpus_release_content_sha256": release.content_sha256,
        "corpus_release_selector_sha256": release.selector_sha256,
        "requested_corpus_citation_path": UNIT_CITATION,
        "provision_file": _rel(scope),
        "provision_file_sha256": digest,
        "row": row,
        "component_rows": components,
        "source_sha256": "a" * 64,
        "generation_input_sha256": "b" * 64,
    }


@pytest.fixture()
def base(tmp_path: Path):
    _write_scope(tmp_path, UNIT, UNIT_CITATION, "unit words")
    _write_scope(tmp_path, SIBLING, "us/statute/7/2015", "sibling words")
    _write_scope(tmp_path, OTHER_PAIR, "us/regulation/7/273/1", "other words")
    old = _signed(tmp_path, "carry-old", [UNIT, SIBLING, OTHER_PAIR])
    return tmp_path, old


def test_current_release_needs_no_carry(base):
    root, old = base
    verdict = classify_attestation_carry(
        _attestation(root, old), target=old, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CURRENT


def test_additive_successor_carries(base):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CARRIED
    assert verdict.resolution_checked is False


def test_changed_unit_bytes_do_not_carry(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, UNIT, UNIT_CITATION, "unit words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.UNIT_CHANGED


def test_dropped_unit_scope_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.UNIT_MISSING


def test_changed_same_pair_scope_does_not_carry(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, SIBLING, "us/statute/7/2015", "sibling words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any(_rel(SIBLING) in reason for reason in verdict.reasons)


def test_dropped_same_pair_scope_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, OTHER_PAIR])
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


def test_change_in_another_pair_still_carries(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, OTHER_PAIR, "us/regulation/7/273/1", "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CARRIED


def test_changed_component_row_file_does_not_carry(base):
    root, old = base
    attestation = _attestation(root, old, component_scope=OTHER_PAIR)
    _write_scope(root, OTHER_PAIR, "us/regulation/7/273/1", "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any("component file" in reason for reason in verdict.reasons)


def test_unsupplied_attested_release_is_not_in_lineage(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[new]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


def test_attested_selector_must_match_the_attested_release(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["corpus_release_selector_sha256"] = new.selector_sha256
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


def test_attested_file_digest_must_be_in_the_attested_release(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["provision_file_sha256"] = "c" * 64
    attestation["row"]["provision_file_sha256"] = "c" * 64
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a["row"].update(provision_file_sha256="d" * 64),
        lambda a: a["row"].update(provision_file="data/corpus/provisions/x.jsonl"),
        lambda a: a.update(
            provision_file="data/corpus/provisions/us/statute.jsonl",
            row={
                "provision_file": "data/corpus/provisions/us/statute.jsonl",
                "provision_file_sha256": a["provision_file_sha256"],
            },
        ),
        lambda a: a.update(component_rows=[{"provision_file": 1}]),
        lambda a: a.update(component_rows="none"),
        lambda a: a.pop("row"),
    ],
)
def test_inconsistent_attestations_are_malformed(base, mutate):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    mutate(attestation)
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.MALFORMED


def _evidence(attestation: dict, **overrides) -> ResolvedUnitEvidence:
    resolved = {
        key: value
        for key, value in attestation.items()
        if key != "generation_input_sha256"
    }
    for field in RELEASE_IDENTITY_FIELDS:
        resolved[field] = "the-new-release-identity"
    resolved.update(overrides.pop("fields", {}))
    return ResolvedUnitEvidence(
        attestation=resolved,
        generation_input_sha256=overrides.pop(
            "generation_input_sha256", attestation["generation_input_sha256"]
        ),
        has_amendment_context=overrides.pop("has_amendment_context", False),
    )


@pytest.mark.parametrize(
    ("overrides", "status"),
    [
        ({}, CarryStatus.CARRIED),
        ({"fields": {"source_sha256": "e" * 64}}, CarryStatus.UNIT_CHANGED),
        ({"fields": {"resolved_text_sha256": "f" * 64}}, CarryStatus.UNIT_CHANGED),
        ({"generation_input_sha256": "0" * 64}, CarryStatus.UNIT_CHANGED),
        ({"has_amendment_context": True}, CarryStatus.CONTEXT_CHANGED),
    ],
)
def test_resolution_evidence_decides_after_artifacts(base, overrides, status):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    calls: list[str] = []

    def resolve(requested: str) -> ResolvedUnitEvidence:
        calls.append(requested)
        return _evidence(attestation, **dict(overrides))

    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old], resolve=resolve
    )
    assert calls == [UNIT_CITATION]
    assert verdict.status is status
    assert verdict.resolution_checked is True


def test_unresolvable_citation_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])

    def resolve(requested: str) -> ResolvedUnitEvidence:
        raise ValueError("not found")

    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[old], resolve=resolve
    )
    assert verdict.status is CarryStatus.UNIT_MISSING
    assert verdict.resolution_checked is True


def test_missing_requested_citation_is_malformed_when_resolving(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation.pop("requested_corpus_citation_path")
    verdict = classify_attestation_carry(
        attestation,
        target=new,
        attested_releases=[old],
        resolve=lambda requested: pytest.fail("must not resolve"),
    )
    assert verdict.status is CarryStatus.MALFORMED


def test_artifact_refusal_happens_before_resolution(base):
    root, old = base
    new = _signed(root, "carry-new", [SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        _attestation(root, old),
        target=new,
        attested_releases=[old],
        resolve=lambda requested: pytest.fail("must not resolve"),
    )
    assert verdict.status is CarryStatus.UNIT_MISSING
    assert verdict.resolution_checked is False


def test_real_resolver_carries_an_attestation_to_an_additive_successor(tmp_path):
    _write_scope(tmp_path, UNIT, UNIT_CITATION, "unit words")
    _write_scope(tmp_path, SIBLING, "us/statute/7/2015", "sibling words")
    old_release = bind_test_corpus_release(tmp_path, "carry-old", [UNIT, SIBLING])
    unit = resolve_corpus_source_unit(UNIT_CITATION, old_release)
    attestation = dict(unit.source_attestation)
    attestation["generation_input_sha256"] = hashlib.sha256(
        unit.body.encode("utf-8")
    ).hexdigest()
    old = _signed(tmp_path, "carry-old", [UNIT, SIBLING])
    _write_scope(tmp_path, ADDED, "us/statute/7/2016", "added words")
    new = _signed(tmp_path, "carry-new", [UNIT, SIBLING, ADDED])
    resolve = _corpus_resolver(tmp_path, new, TEST_RELEASE_PUBLIC_KEY)

    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old], resolve=resolve
    )
    assert verdict.status is CarryStatus.CARRIED
    assert verdict.resolution_checked is True

    tampered = dict(attestation, source_sha256="9" * 64)
    verdict = classify_attestation_carry(
        tampered, target=new, attested_releases=[old], resolve=resolve
    )
    assert verdict.status is CarryStatus.UNIT_CHANGED


def _manifest_repo(tmp_path: Path, attestations: dict[str, dict]) -> Path:
    repo = tmp_path / "rulespec-xx"
    for relative, payload in attestations.items():
        path = repo / ".axiom" / "encoding-manifests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return repo


def test_manifest_walk_reads_nested_attestations(tmp_path):
    repo = _manifest_repo(
        tmp_path,
        {
            "a.json": {"source_attestation": {"corpus_release": "x"}},
            "b/c.json": {
                "retired_manifest": {"source_attestation": {"corpus_release": "y"}},
                "replacement_manifest": {"source_attestation": {"corpus_release": "z"}},
            },
            "legacy.json": {"applied_files": []},
        },
    )
    (repo / ".axiom" / "encoding-manifests" / "broken.json").write_text("{")
    found = dict(iter_manifest_attestations(repo / ".axiom" / "encoding-manifests"))
    assert found == {
        "a.json": {"corpus_release": "x"},
        "b/c.json#retired_manifest": {"corpus_release": "y"},
        "b/c.json#replacement_manifest": {"corpus_release": "z"},
        "broken.json": {},
    }


def _release_file(root: Path, release) -> Path:
    return root / "releases" / release.name / f"{release.content_sha256}.json"


def test_cli_lists_verdicts_and_exits_by_result(base, capsys):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    repo = _manifest_repo(
        root,
        {
            "carried.json": {"source_attestation": _attestation(root, old)},
            "current.json": {"source_attestation": _attestation(root, new)},
        },
    )
    argv = [
        "--rulespec-repo",
        str(repo),
        "--to-release",
        str(_release_file(root, new)),
        "--from-release",
        str(_release_file(root, old)),
        "--public-key",
        TEST_RELEASE_PUBLIC_KEY,
        "--json",
    ]
    assert run_corpus_attestation_carry(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {"carried": 1, "current": 1}
    assert {row["manifest"]: row["status"] for row in report["manifests"]} == {
        "carried.json": "carried",
        "current.json": "current",
    }

    assert run_corpus_attestation_carry(argv[:4] + argv[6:-1]) == 1
    text = capsys.readouterr().out
    assert "not-in-lineage\tcarried.json" in text
    assert "carry-new: 2 attestations: current 1, not-in-lineage 1" in text


def test_cli_refuses_an_unverifiable_release_object(base, capsys, tmp_path):
    root, old = base
    bad = tmp_path / "bad.json"
    payload = json.loads(_release_file(root, old).read_text("utf-8"))
    payload["content"]["created_at"] = "2026-07-11T00:00:00Z"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    code = run_corpus_attestation_carry(
        [
            "--rulespec-repo",
            str(root),
            "--to-release",
            str(bad),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
        ]
    )
    assert code == 2
    assert "content sha256 does not match" in capsys.readouterr().err


def test_cli_requires_a_public_key(base, monkeypatch):
    root, old = base
    monkeypatch.delenv("AXIOM_CORPUS_RELEASE_PUBLIC_KEY", raising=False)
    with pytest.raises(SystemExit):
        run_corpus_attestation_carry(
            [
                "--rulespec-repo",
                str(root),
                "--to-release",
                str(_release_file(root, old)),
            ]
        )


def test_console_entrypoint_dispatches_the_dry_run(base, monkeypatch, capsys):
    root, old = base
    repo = _manifest_repo(
        root, {"current.json": {"source_attestation": _attestation(root, old)}}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "corpus-attestation-carry",
            "--rulespec-repo",
            str(repo),
            "--to-release",
            str(_release_file(root, old)),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
        ],
    )
    assert entrypoint.main() == 0
    assert "current\tcurrent.json" in capsys.readouterr().out


def test_attested_release_must_match_name_and_content_digest(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["corpus_release_content_sha256"] = "7" * 64
    verdict = classify_attestation_carry(
        attestation, target=new, attested_releases=[old]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


@pytest.mark.parametrize(
    ("documents", "request_path", "expected"),
    [
        ((), None, False),
        (("amending act",), None, True),
        ((), "us/statute/7/2014/e", True),
    ],
)
def test_corpus_resolver_reports_amendment_context(
    base, monkeypatch, documents, request_path, expected
):
    root, old = base
    from axiom_encode.harness import evals

    class _Unit:
        body = "unit\r\nwords"
        source_attestation = {"source_sha256": "a" * 64}
        amendment_documents = documents
        amendment_request = request_path

    monkeypatch.setattr(
        evals, "resolve_corpus_source_unit", lambda requested, release: _Unit()
    )
    evidence = _corpus_resolver(root, old, TEST_RELEASE_PUBLIC_KEY)(UNIT_CITATION)
    assert evidence.has_amendment_context is expected
    assert (
        evidence.generation_input_sha256 == hashlib.sha256(b"unit\nwords").hexdigest()
    )

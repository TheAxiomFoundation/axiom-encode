"""Tests for the dry-run corpus attestation carry check."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

import pytest

from axiom_encode import entrypoint
from axiom_encode.corpus_attestation_carry import (
    RELEASE_IDENTITY_FIELDS,
    CarryStatus,
    ResolvedCitationEvidence,
    ResolvedUnitEvidence,
    _corpus_resolvers,
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
ADDED_OTHER_PAIR = ("us", "guidance", "2026-01-05-added")
UNIT_CITATION = "us/statute/7/2014/e"
OTHER_CITATION = "us/regulation/7/273/1"


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


def _write_rows(root: Path, scope: tuple[str, str, str], rows: list[dict]) -> None:
    jurisdiction, document_class, version = scope
    path = _provisions_path(root, scope)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index, row in enumerate(rows):
        record = {
            "id": f"{version}-{index}",
            "jurisdiction": jurisdiction,
            "document_class": document_class,
            "version": version,
            "source_path": f"sources/{jurisdiction}/{document_class}/{version}/s.xml",
            "source_as_of": "2026-01-02",
            "expression_date": "2026-01-01",
        }
        record.update(row)
        lines.append(json.dumps(record))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scope(root: Path, scope, citation_path: str, body: str) -> None:
    _write_rows(root, scope, [{"citation_path": citation_path, "body": body}])


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
    _write_scope(tmp_path, OTHER_PAIR, OTHER_CITATION, "other words")
    old = _signed(tmp_path, "carry-old", [UNIT, SIBLING, OTHER_PAIR])
    return tmp_path, old


def _carry(attestation, new, old, **kwargs):
    return classify_attestation_carry(
        attestation, target=new, attested_releases=[old], **kwargs
    )


def test_current_release_needs_no_carry(base):
    root, old = base
    verdict = _carry(_attestation(root, old), old, old)
    assert verdict.status is CarryStatus.CURRENT


@pytest.mark.parametrize(
    "field", ["corpus_release_content_sha256", "corpus_release_selector_sha256"]
)
def test_current_needs_every_identity_field(base, field):
    root, old = base
    attestation = _attestation(root, old)
    attestation[field] = "5" * 64
    verdict = _carry(attestation, old, old)
    assert verdict.status is not CarryStatus.CURRENT


def test_unchanged_pairs_carry_without_resolution(base):
    root, old = base
    _write_scope(root, ADDED_OTHER_PAIR, "us/guidance/x/1", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED_OTHER_PAIR])
    verdict = _carry(_attestation(root, old), new, old)
    assert verdict.status is CarryStatus.CARRIED
    assert verdict.resolution_checked is False
    assert verdict.pair_gained_scopes == ()


def test_scope_added_to_the_unit_pair_needs_resolution(base):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    verdict = _carry(_attestation(root, old), new, old)
    assert verdict.status is CarryStatus.NOT_VERIFIABLE
    assert verdict.pair_gained_scopes == ("us/statute/2026-01-04-added",)


def test_changed_unit_bytes_do_not_carry(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, UNIT, UNIT_CITATION, "unit words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    assert _carry(attestation, new, old).status is CarryStatus.UNIT_CHANGED


def test_dropped_unit_scope_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [SIBLING, OTHER_PAIR])
    verdict = _carry(_attestation(root, old), new, old)
    assert verdict.status is CarryStatus.UNIT_MISSING


def test_changed_same_pair_scope_does_not_carry(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, SIBLING, "us/statute/7/2015", "sibling words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(attestation, new, old)
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any(_rel(SIBLING) in reason for reason in verdict.reasons)


def test_dropped_same_pair_scope_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, OTHER_PAIR])
    verdict = _carry(_attestation(root, old), new, old)
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


def test_change_in_an_uncited_pair_still_carries(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, OTHER_PAIR, OTHER_CITATION, "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    assert _carry(attestation, new, old).status is CarryStatus.CARRIED


def test_changed_component_row_file_does_not_carry(base):
    root, old = base
    attestation = _attestation(root, old, component_scope=OTHER_PAIR)
    _write_scope(root, OTHER_PAIR, OTHER_CITATION, "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(attestation, new, old)
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any("component file" in reason for reason in verdict.reasons)


def test_cited_pair_is_closed_like_the_unit_pair(base):
    """H1: validation reads cited corpus text outside the unit's pair."""

    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, OTHER_PAIR, OTHER_CITATION, "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(attestation, new, old, cited_paths=[OTHER_CITATION])
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any(_rel(OTHER_PAIR) in reason for reason in verdict.reasons)


def test_scope_added_to_a_cited_pair_needs_resolution(base):
    root, old = base
    added = ("us", "regulation", "2026-01-06-added")
    _write_scope(root, added, "us/regulation/7/273/2", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, added])
    verdict = _carry(_attestation(root, old), new, old, cited_paths=[OTHER_CITATION])
    assert verdict.status is CarryStatus.NOT_VERIFIABLE
    assert verdict.pair_gained_scopes == ("us/regulation/2026-01-06-added",)


def test_non_canonical_cited_path_is_malformed(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(_attestation(root, old), new, old, cited_paths=["us/statutes/7"])
    assert verdict.status is CarryStatus.MALFORMED


def test_unsupplied_attested_release_is_not_in_lineage(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[new]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


def test_attested_release_must_match_name_and_content_digest(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["corpus_release_content_sha256"] = "7" * 64
    assert _carry(attestation, new, old).status is CarryStatus.NOT_IN_LINEAGE
    renamed = dataclasses.replace(old, name="carry-other")
    verdict = classify_attestation_carry(
        _attestation(root, old), target=new, attested_releases=[renamed]
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


def test_attested_selector_must_match_the_attested_release(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["corpus_release_selector_sha256"] = new.selector_sha256
    assert _carry(attestation, new, old).status is CarryStatus.NOT_IN_LINEAGE


def test_attested_file_digest_must_be_in_the_attested_release(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation["provision_file_sha256"] = "c" * 64
    attestation["row"]["provision_file_sha256"] = "c" * 64
    assert _carry(attestation, new, old).status is CarryStatus.NOT_IN_LINEAGE


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
        lambda a: a.update(
            provision_file="data/corpus/provisions/us/statute/x/y.jsonl",
            row={
                "provision_file": "data/corpus/provisions/us/statute/x/y.jsonl",
                "provision_file_sha256": a["provision_file_sha256"],
            },
        ),
        lambda a: a.update(
            provision_file="data/corpus/provisions/us/statutes/x.jsonl",
            row={
                "provision_file": "data/corpus/provisions/us/statutes/x.jsonl",
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
    assert _carry(attestation, new, old).status is CarryStatus.MALFORMED


def test_component_files_are_bound_to_the_attested_release(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old, component_scope=OTHER_PAIR)
    attestation["component_rows"][0]["provision_file_sha256"] = "8" * 64
    assert _carry(attestation, new, old).status is CarryStatus.NOT_IN_LINEAGE


def _evidence(attestation: dict, **overrides) -> ResolvedUnitEvidence:
    resolved = {
        key: value
        for key, value in attestation.items()
        if key != "generation_input_sha256"
    }
    for identity_field in RELEASE_IDENTITY_FIELDS:
        resolved[identity_field] = "the-new-release-identity"
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
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    attestation = _attestation(root, old)
    calls: list[str] = []

    def resolve(requested: str) -> ResolvedUnitEvidence:
        calls.append(requested)
        return _evidence(attestation, **dict(overrides))

    verdict = _carry(attestation, new, old, resolve=resolve)
    assert calls == [UNIT_CITATION]
    assert verdict.status is status
    assert verdict.resolution_checked is True
    assert verdict.pair_gained_scopes == ("us/statute/2026-01-04-added",)


def _cited_resolver(files):
    def resolve_citation(path: str) -> ResolvedCitationEvidence:
        if files is None:
            raise ValueError("not found")
        return ResolvedCitationEvidence(tuple(files))

    return resolve_citation


def test_cited_paths_must_resolve_to_identical_artifacts(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    other = (
        _rel(OTHER_PAIR),
        _attestation(root, old, OTHER_PAIR)["provision_file_sha256"],
    )

    def resolve(requested):
        return _evidence(attestation)

    good = _carry(
        attestation,
        new,
        old,
        cited_paths=[OTHER_CITATION],
        resolve=resolve,
        resolve_citation=_cited_resolver([other]),
    )
    assert good.status is CarryStatus.CARRIED
    assert good.resolution_checked is True
    for files in (
        None,
        [],
        [(other[0], "4" * 64)],
        [("data/corpus/provisions/us/regulation/new.jsonl", other[1])],
    ):
        verdict = _carry(
            attestation,
            new,
            old,
            cited_paths=[OTHER_CITATION],
            resolve=resolve,
            resolve_citation=_cited_resolver(files),
        )
        assert verdict.status is CarryStatus.CONTEXT_CHANGED, files
        assert verdict.resolution_checked is True


def test_unit_resolution_without_a_citation_resolver_is_not_enough(base):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    attestation = _attestation(root, old)
    verdict = _carry(
        attestation,
        new,
        old,
        cited_paths=[UNIT_CITATION],
        resolve=lambda requested: _evidence(attestation),
    )
    assert verdict.status is CarryStatus.NOT_VERIFIABLE


def test_unresolvable_citation_does_not_carry(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])

    def resolve(requested: str) -> ResolvedUnitEvidence:
        raise ValueError("not found")

    verdict = _carry(_attestation(root, old), new, old, resolve=resolve)
    assert verdict.status is CarryStatus.UNIT_MISSING
    assert verdict.resolution_checked is True


def test_missing_requested_citation_is_malformed_when_resolving(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    attestation = _attestation(root, old)
    attestation.pop("requested_corpus_citation_path")
    verdict = _carry(
        attestation, new, old, resolve=lambda requested: pytest.fail("must not")
    )
    assert verdict.status is CarryStatus.MALFORMED


def test_artifact_refusal_happens_before_resolution(base):
    root, old = base
    new = _signed(root, "carry-new", [SIBLING, OTHER_PAIR])
    verdict = _carry(
        _attestation(root, old),
        new,
        old,
        resolve=lambda requested: pytest.fail("must not resolve"),
    )
    assert verdict.status is CarryStatus.UNIT_MISSING
    assert verdict.resolution_checked is False


# Attacks from the independent review of #2348: each adds a scope to the unit's
# pair in B without changing any artifact of A. Rules 1-3 alone must not carry
# them, and the real resolver must refuse them.

STATUTE_BASE = ("us", "statute", "2026-01-01-base")
STATUTE_ADDED = ("us", "statute", "2026-02-01-recovery")
SECTION = "(a) In general.--Base text.\n(b) Limit.--The limit is $100.\n(c) Other."


def _real_attestation(root: Path, scopes, citation: str) -> dict:
    release = bind_test_corpus_release(root, "rel-a", scopes)
    unit = resolve_corpus_source_unit(citation, release)
    attestation = dict(unit.source_attestation)
    attestation["generation_input_sha256"] = hashlib.sha256(
        unit.body.encode("utf-8")
    ).hexdigest()
    return attestation


def _attack(root: Path, attestation: dict, new_scopes, cited=()):
    old = _signed(root, "rel-a", [STATUTE_BASE])
    new = _signed(root, "rel-b", new_scopes)
    plain = classify_attestation_carry(
        attestation, target=new, attested_releases=[old], cited_paths=cited
    )
    resolve, resolve_citation = _corpus_resolvers(root, new, TEST_RELEASE_PUBLIC_KEY)
    checked = classify_attestation_carry(
        attestation,
        target=new,
        attested_releases=[old],
        cited_paths=cited,
        resolve=resolve,
        resolve_citation=resolve_citation,
    )
    return plain, checked


def _assert_refused(plain, checked):
    assert plain.status is CarryStatus.NOT_VERIFIABLE
    assert plain.pair_gained_scopes
    assert checked.status is not CarryStatus.CARRIED
    assert checked.resolution_checked is True


def test_attack_added_exact_child_row_shadows_a_parent_slice(tmp_path):
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [{"citation_path": "us/statute/7/2014", "body": SECTION}],
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], "us/statute/7/2014/b")
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [{"citation_path": "us/statute/7/2014/b", "body": "(b) Limit.--$900."}],
    )
    _assert_refused(*_attack(tmp_path, attestation, [STATUTE_BASE, STATUTE_ADDED]))


def test_attack_added_scope_duplicates_the_exact_citation(tmp_path):
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [{"citation_path": "us/statute/7/2015", "body": "Original."}],
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], "us/statute/7/2015")
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [{"citation_path": "us/statute/7/2015", "body": "Re-ingested, different."}],
    )
    _assert_refused(*_attack(tmp_path, attestation, [STATUTE_BASE, STATUTE_ADDED]))


def test_attack_added_scope_amends_the_unit(tmp_path):
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [{"citation_path": "us/statute/7/2016", "body": "The amount is $100."}],
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], "us/statute/7/2016")
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [
            {
                "citation_path": "us/statute/pl-119-99/sec-5",
                "body": "Section 2016 is amended by striking $100 and inserting $500.",
                "source_path": "sources/us/statute/pl/s.xml",
                "metadata": {
                    "document_type": "amendment",
                    "amendment_targets": ["us/statute/7/2016"],
                },
            }
        ],
    )
    _assert_refused(*_attack(tmp_path, attestation, [STATUTE_BASE, STATUTE_ADDED]))


def test_attack_dormant_row_activated_by_an_added_scope(tmp_path):
    dormant = ("us", "statute", "2026-03-01-dormant")
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [
            {"citation_path": "us/statute/7/2017", "body": "Original."},
            {
                "citation_path": "us/statute/7/2017",
                "body": "Shadow.",
                "version": dormant[2],
            },
        ],
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], "us/statute/7/2017")
    _write_rows(
        tmp_path, dormant, [{"citation_path": "us/statute/7/9999", "body": "filler"}]
    )
    _assert_refused(*_attack(tmp_path, attestation, [STATUTE_BASE, dormant]))


def test_attack_composed_container_gains_a_descendant(tmp_path):
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [
            {
                "citation_path": "us/statute/7/2018",
                "body": None,
                "heading": "Sec. 2018",
            },
            {
                "citation_path": "us/statute/7/2018/a",
                "body": "(a) First.",
                "heading": "a",
            },
        ],
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], "us/statute/7/2018")
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [
            {
                "citation_path": "us/statute/7/2018/b",
                "body": "(b) New rule.",
                "heading": "b",
            }
        ],
    )
    _assert_refused(*_attack(tmp_path, attestation, [STATUTE_BASE, STATUTE_ADDED]))


def test_real_resolver_carries_an_unaffected_unit(tmp_path):
    _write_rows(
        tmp_path, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "unit words"}]
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], UNIT_CITATION)
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [{"citation_path": "us/statute/7/2016", "body": "added"}],
    )
    plain, checked = _attack(
        tmp_path, attestation, [STATUTE_BASE, STATUTE_ADDED], cited=[UNIT_CITATION]
    )
    assert plain.status is CarryStatus.NOT_VERIFIABLE
    assert checked.status is CarryStatus.CARRIED
    assert checked.resolution_checked is True
    assert checked.pair_gained_scopes == ("us/statute/2026-02-01-recovery",)

    tampered = dict(attestation, source_sha256="9" * 64)
    _plain, checked = _attack(tmp_path, tampered, [STATUTE_BASE, STATUTE_ADDED])
    assert checked.status is CarryStatus.UNIT_CHANGED


def test_real_resolver_refuses_a_cited_path_that_does_not_resolve(tmp_path):
    _write_rows(
        tmp_path, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "unit words"}]
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], UNIT_CITATION)
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [{"citation_path": "us/statute/7/2016", "body": "added"}],
    )
    _plain, checked = _attack(
        tmp_path,
        attestation,
        [STATUTE_BASE, STATUTE_ADDED],
        cited=["us/statute/7/4040"],
    )
    assert checked.status is CarryStatus.CONTEXT_CHANGED
    assert any("does not resolve" in reason for reason in checked.reasons)


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
    resolve, _resolve_citation = _corpus_resolvers(root, old, TEST_RELEASE_PUBLIC_KEY)
    evidence = resolve(UNIT_CITATION)
    assert evidence.has_amendment_context is expected
    assert (
        evidence.generation_input_sha256 == hashlib.sha256(b"unit\nwords").hexdigest()
    )


def _rulespec(citations: list[str]) -> bytes:
    lines = ["format: rulespec/v1", "module:", "  source_verification:"]
    lines.append(f"    corpus_citation_path: {citations[0]}")
    lines.append("  proof_validation:")
    lines.append("    atoms:")
    for citation in citations[1:]:
        lines.append("    - source:")
        lines.append(f"        corpus_citation_path: {citation}")
    lines.append("  interpretation:")
    lines.append("    corpus_citation_paths: [us/policy/x/1]")
    return ("\n".join(lines) + "\n").encode()


def _manifest_repo(tmp_path: Path, manifests: dict[str, object]) -> Path:
    repo = tmp_path / "rulespec-xx"
    for relative, payload in manifests.items():
        path = repo / ".axiom" / "encoding-manifests" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    return repo


def _bind(repo: Path, relative: str, raw: bytes) -> dict:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}


def test_manifest_walk_reads_nested_attestations_and_citations(tmp_path):
    repo = _manifest_repo(tmp_path, {})
    module = _bind(repo, "us/a.yaml", _rulespec([UNIT_CITATION, "us/guidance/g/1"]))
    ledger = {"path": "known-validation-gaps.yaml", "sha256": "0" * 64}
    (repo / "known-validation-gaps.yaml").write_text("entries: []\n")
    repo = _manifest_repo(
        tmp_path,
        {
            "a.json": {
                "source_attestation": {"corpus_release": "x"},
                "applied_files": [
                    module,
                    ledger,
                    {"path": "gone.yaml", "deleted": True},
                ],
            },
            "b/c.json": {
                "applied_files": [],
                "retired_manifest": {
                    "source_attestation": {"corpus_release": "y"},
                    "applied_files": [],
                },
                "replacement_manifest": {
                    "source_attestation": {"corpus_release": "z"},
                    "applied_files": [],
                },
            },
            "legacy.json": {"applied_files": []},
            "list.json": [1, 2],
        },
    )
    manifests = repo / ".axiom" / "encoding-manifests"
    (manifests / "broken.json").write_text("{")
    (manifests / "linked.json").symlink_to(manifests / "a.json")
    found = {entry.label: entry for entry in iter_manifest_attestations(manifests)}
    assert found["a.json"].attestation == {"corpus_release": "x"}
    assert found["a.json"].cited_paths == (
        "us/guidance/g/1",
        "us/policy/x/1",
        UNIT_CITATION,
    )
    assert found["a.json"].problems == ()
    assert found["b/c.json#retired_manifest"].attestation == {"corpus_release": "y"}
    assert found["b/c.json#replacement_manifest"].attestation == {"corpus_release": "z"}
    assert "b/c.json" not in found
    assert found["legacy.json"].attestation is None
    assert found["legacy.json"].problems == ()
    assert found["list.json"].problems == ("manifest is not a JSON object",)
    assert found["broken.json"].problems == ("manifest is not valid JSON",)
    assert found["linked.json"].problems == ("manifest is a symlink",)


@pytest.mark.parametrize(
    ("entry", "problem"),
    [
        ({"path": "us/a.yaml", "sha256": "1" * 64}, "differs from the manifest"),
        ({"path": "../a.yaml", "sha256": "1" * 64}, "not repository-relative"),
        ({"path": "us/missing.yaml", "sha256": "1" * 64}, "not a regular file"),
    ],
)
def test_bound_module_problems_are_reported(tmp_path, entry, problem):
    repo = _manifest_repo(tmp_path, {})
    _bind(repo, "us/a.yaml", _rulespec([UNIT_CITATION]))
    repo = _manifest_repo(
        tmp_path,
        {"a.json": {"source_attestation": {}, "applied_files": [entry]}},
    )
    (entry_found,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
    assert len(entry_found.problems) == 1
    assert problem in entry_found.problems[0]


def _release_file(root: Path, release) -> Path:
    return root / "releases" / release.name / f"{release.content_sha256}.json"


def test_cli_lists_verdicts_and_counts_every_manifest(base, capsys):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    repo = _manifest_repo(
        root,
        {
            "gained.json": {
                "source_attestation": _attestation(root, old),
                "applied_files": [],
            },
            "current.json": {
                "source_attestation": _attestation(root, new),
                "applied_files": [],
            },
            "legacy.json": {"applied_files": []},
            "list.json": [],
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
    assert run_corpus_attestation_carry(argv) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["counts"] == {
        "current": 1,
        "malformed": 1,
        "no-attestation": 1,
        "not-verifiable": 1,
    }
    gained = next(r for r in report["manifests"] if r["manifest"] == "gained.json")
    assert gained["resolution_checked"] is False
    assert gained["pair_gained_scopes"] == ["us/statute/2026-01-04-added"]

    assert run_corpus_attestation_carry(argv[:-1]) == 1
    text = capsys.readouterr().out
    assert (
        "not-verifiable\tgained.json\tresolution_checked: false\t"
        "pair_gained_scopes: 1" in text
    )
    assert "legacy.json" not in text
    assert "carry-new: 4 manifest entries: current 1, malformed 1" in text


def test_cli_exit_zero_when_everything_carries(base, capsys):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    repo = _manifest_repo(
        root,
        {
            "carried.json": {
                "source_attestation": _attestation(root, old),
                "applied_files": [],
            },
            "legacy.json": {"applied_files": []},
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
    ]
    assert run_corpus_attestation_carry(argv) == 0
    assert "carried\tcarried.json\tresolution_checked: false" in capsys.readouterr().out
    assert run_corpus_attestation_carry(argv[:4] + argv[6:]) == 1


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


def test_cli_resolves_with_a_corpus_root(tmp_path, capsys):
    _write_rows(
        tmp_path, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "unit words"}]
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], UNIT_CITATION)
    old = _signed(tmp_path, "rel-a", [STATUTE_BASE])
    _write_rows(
        tmp_path,
        STATUTE_ADDED,
        [{"citation_path": "us/statute/7/2016", "body": "added"}],
    )
    new = _signed(tmp_path, "rel-b", [STATUTE_BASE, STATUTE_ADDED])
    repo = _manifest_repo(tmp_path, {})
    module = _bind(repo, "us/a.yaml", _rulespec([UNIT_CITATION]))
    repo = _manifest_repo(
        tmp_path,
        {"a.json": {"source_attestation": attestation, "applied_files": [module]}},
    )
    code = run_corpus_attestation_carry(
        [
            "--rulespec-repo",
            str(repo),
            "--to-release",
            str(_release_file(tmp_path, new)),
            "--from-release",
            str(_release_file(tmp_path, old)),
            "--corpus-root",
            str(tmp_path),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    # The module cites us/policy/x/1, which no release holds.
    assert code == 1
    (row,) = report["manifests"]
    assert row["status"] == "context-changed"
    assert row["resolution_checked"] is True
    assert row["cited_paths"] == 2


def test_console_entrypoint_dispatches_the_dry_run(base, monkeypatch, capsys):
    root, old = base
    repo = _manifest_repo(
        root,
        {
            "current.json": {
                "source_attestation": _attestation(root, old),
                "applied_files": [],
            }
        },
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


def test_cited_artifact_must_be_identical_in_both_releases(tmp_path):
    only_old = ("us", "policy", "2026-01-07-old-only")
    _write_scope(tmp_path, UNIT, UNIT_CITATION, "unit words")
    _write_scope(tmp_path, OTHER_PAIR, OTHER_CITATION, "other words")
    _write_scope(tmp_path, only_old, "us/policy/p/1", "old policy")
    old = _signed(tmp_path, "carry-old", [UNIT, OTHER_PAIR, only_old])
    attestation = _attestation(tmp_path, old)
    _write_scope(tmp_path, ADDED_OTHER_PAIR, "us/guidance/x/1", "added words")
    new = _signed(tmp_path, "carry-new", [UNIT, OTHER_PAIR, ADDED_OTHER_PAIR])

    def digest(scope):
        return hashlib.sha256(
            _provisions_path(tmp_path, scope).read_bytes()
        ).hexdigest()

    for scope in (only_old, ADDED_OTHER_PAIR):
        verdict = _carry(
            attestation,
            new,
            old,
            cited_paths=[OTHER_CITATION],
            resolve=lambda requested: _evidence(attestation),
            resolve_citation=_cited_resolver([(_rel(scope), digest(scope))]),
        )
        assert verdict.status is CarryStatus.CONTEXT_CHANGED, scope

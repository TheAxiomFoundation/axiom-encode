"""Tests for the dry-run corpus attestation carry check."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from axiom_encode import entrypoint
from axiom_encode.corpus_attestation_carry import (
    RELEASE_IDENTITY_FIELDS,
    CarryStatus,
    CitationNotFound,
    CorpusView,
    ResolvedCitationEvidence,
    ResolvedUnitEvidence,
    classify_attestation_carry,
    corpus_view,
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
        "rulespec_root": "rulespec-us/us",
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
    resolved.update(overrides.pop("fields", {}))
    return ResolvedUnitEvidence(
        attestation=resolved,
        generation_input_sha256=overrides.pop(
            "generation_input_sha256", attestation["generation_input_sha256"]
        ),
        has_amendment_context=overrides.pop("has_amendment_context", False),
    )


def _view(unit=None, citations=None, row_scopes=None) -> CorpusView:
    def resolve_unit(requested):
        if isinstance(unit, Exception):
            raise unit
        return unit

    def resolve_citation(path):
        value = (citations or {}).get(path, CitationNotFound(path))
        if isinstance(value, Exception):
            raise value
        return value

    return CorpusView(
        unit=resolve_unit,
        citation=resolve_citation,
        row_scopes=lambda path: frozenset((row_scopes or {}).get(path, ())),
    )


@pytest.fixture()
def gained(base):
    root, old = base
    _write_scope(root, ADDED, "us/statute/7/2016", "added words")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR, ADDED])
    return root, old, new, _attestation(root, old)


@pytest.mark.parametrize(
    ("b_overrides", "status"),
    [
        ({}, CarryStatus.CARRIED),
        ({"fields": {"source_sha256": "e" * 64}}, CarryStatus.UNIT_CHANGED),
        ({"fields": {"resolved_text_sha256": "f" * 64}}, CarryStatus.UNIT_CHANGED),
        ({"fields": {"extra_field": "x"}}, CarryStatus.UNIT_CHANGED),
        ({"generation_input_sha256": "0" * 64}, CarryStatus.UNIT_CHANGED),
        ({"has_amendment_context": True}, CarryStatus.CONTEXT_CHANGED),
    ],
)
def test_unit_resolution_in_a_and_b_decides(gained, b_overrides, status):
    _root, old, new, attestation = gained
    view_a = _view(_evidence(attestation))
    view_b = _view(_evidence(attestation, **dict(b_overrides)))
    verdict = _carry(attestation, new, old, view_a=view_a, view_b=view_b)
    assert verdict.status is status
    assert verdict.resolution_checked is True
    assert verdict.pair_gained_scopes == ("us/statute/2026-01-04-added",)


def test_identity_fields_do_not_count_as_a_unit_change(gained):
    _root, old, new, attestation = gained
    renamed = {field: "the-new-identity" for field in RELEASE_IDENTITY_FIELDS}
    verdict = _carry(
        attestation,
        new,
        old,
        view_a=_view(_evidence(attestation)),
        view_b=_view(_evidence(attestation, fields=renamed)),
    )
    assert verdict.status is CarryStatus.CARRIED


@pytest.mark.parametrize(
    "a_overrides",
    [
        {"fields": {"source_sha256": "e" * 64}},
        {"generation_input_sha256": "0" * 64},
    ],
)
def test_attestation_must_match_its_resolution_in_a(gained, a_overrides):
    _root, old, new, attestation = gained
    verdict = _carry(
        attestation,
        new,
        old,
        view_a=_view(_evidence(attestation, **dict(a_overrides))),
        view_b=_view(_evidence(attestation, **dict(a_overrides))),
    )
    assert verdict.status is CarryStatus.NOT_IN_LINEAGE


def test_amendment_context_in_a_alone_refuses(gained):
    _root, old, new, attestation = gained
    verdict = _carry(
        attestation,
        new,
        old,
        view_a=_view(_evidence(attestation, has_amendment_context=True)),
        view_b=_view(_evidence(attestation)),
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


@pytest.mark.parametrize(
    ("view_a_unit", "view_b_unit", "status"),
    [
        (ValueError("gone"), None, CarryStatus.NOT_IN_LINEAGE),
        (None, ValueError("gone"), CarryStatus.UNIT_MISSING),
    ],
)
def test_unit_that_does_not_resolve(gained, view_a_unit, view_b_unit, status):
    _root, old, new, attestation = gained
    good = _evidence(attestation)
    verdict = _carry(
        attestation,
        new,
        old,
        view_a=_view(view_a_unit if view_a_unit is not None else good),
        view_b=_view(view_b_unit if view_b_unit is not None else good),
    )
    assert verdict.status is status
    assert verdict.resolution_checked is True


def test_one_view_is_not_enough(gained):
    _root, old, new, attestation = gained
    good = _view(_evidence(attestation))
    for views in ({"view_b": good}, {"view_a": good}):
        verdict = _carry(attestation, new, old, **views)
        assert verdict.status is CarryStatus.NOT_VERIFIABLE


def test_missing_requested_citation_is_malformed_when_resolving(gained):
    _root, old, new, attestation = gained
    attestation.pop("requested_corpus_citation_path")
    verdict = _carry(attestation, new, old, view_a=_view(None), view_b=_view(None))
    assert verdict.status is CarryStatus.MALFORMED


def test_artifact_refusal_happens_before_resolution(base):
    root, old = base
    new = _signed(root, "carry-new", [SIBLING, OTHER_PAIR])
    failing = _view(ValueError("must not resolve"))
    verdict = _carry(_attestation(root, old), new, old, view_a=failing, view_b=failing)
    assert verdict.status is CarryStatus.UNIT_MISSING
    assert verdict.resolution_checked is False


def _other_digest(root):
    return hashlib.sha256(_provisions_path(root, OTHER_PAIR).read_bytes()).hexdigest()


def _cited(files, route=("r",), text="t"):
    return ResolvedCitationEvidence(tuple(files), route, text)


def test_cited_paths_resolve_identically_in_both_releases(gained):
    root, old, new, attestation = gained
    other = (_rel(OTHER_PAIR), _other_digest(root))
    unit = _evidence(attestation)
    same = {OTHER_CITATION: _cited([other])}
    good = _carry(
        attestation,
        new,
        old,
        cited_paths=[OTHER_CITATION],
        view_a=_view(unit, same),
        view_b=_view(unit, same),
    )
    assert good.status is CarryStatus.CARRIED
    variants = [
        {},
        {OTHER_CITATION: ValueError("ambiguous")},
        {OTHER_CITATION: _cited([other], route=("slice",))},
        {OTHER_CITATION: _cited([other], text="other text")},
        {OTHER_CITATION: _cited([(other[0], "4" * 64)])},
        {OTHER_CITATION: _cited([])},
    ]
    for b_citations in variants:
        verdict = _carry(
            attestation,
            new,
            old,
            cited_paths=[OTHER_CITATION],
            view_a=_view(unit, same),
            view_b=_view(unit, b_citations),
        )
        assert verdict.status is CarryStatus.CONTEXT_CHANGED, b_citations
    for files in ([(other[0], "4" * 64)], []):
        both = {OTHER_CITATION: _cited(files)}
        verdict = _carry(
            attestation,
            new,
            old,
            cited_paths=[OTHER_CITATION],
            view_a=_view(unit, both),
            view_b=_view(unit, both),
        )
        assert verdict.status is CarryStatus.CONTEXT_CHANGED, files


def test_import_candidates_may_be_absent_in_both_releases(gained):
    _root, old, new, attestation = gained
    unit = _evidence(attestation)
    candidate = "us/statute/7/9000"
    verdict = _carry(
        attestation,
        new,
        old,
        dependency_paths=[candidate],
        view_a=_view(unit),
        view_b=_view(unit),
    )
    assert verdict.status is CarryStatus.NOT_VERIFIABLE
    assert verdict.resolution_checked is True
    both = {candidate: _cited([])}
    verdict = _carry(
        attestation,
        new,
        old,
        dependency_paths=[candidate],
        view_a=_view(unit, both),
        view_b=_view(unit, both),
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    verdict = _carry(attestation, new, old, dependency_paths=[candidate])
    assert verdict.status is CarryStatus.NOT_VERIFIABLE
    appears = {candidate: _cited([])}
    verdict = _carry(
        attestation,
        new,
        old,
        dependency_paths=[candidate],
        view_a=_view(unit),
        view_b=_view(unit, appears),
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    verdict = _carry(
        attestation,
        new,
        old,
        cited_paths=[candidate],
        view_a=_view(unit),
        view_b=_view(unit),
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


def test_import_candidate_pairs_are_closed(base):
    root, old = base
    _write_scope(root, OTHER_PAIR, OTHER_CITATION, "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(
        _attestation(root, old), new, old, dependency_paths=["us/regulation/9/1"]
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


def test_dormant_row_naming_a_gained_scope_refuses(gained):
    _root, old, new, attestation = gained
    unit = _evidence(attestation)
    dormant = {_rel(SIBLING): {"us/statute/2026-01-04-added"}}
    verdict = _carry(
        attestation,
        new,
        old,
        view_a=_view(unit),
        view_b=_view(unit, row_scopes=dormant),
    )
    assert verdict.status is CarryStatus.CONTEXT_CHANGED
    assert any("rows of a scope" in reason for reason in verdict.reasons)


# Attacks from the independent reviews of #2348, with the real resolver.

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
    attestation["rulespec_root"] = "rulespec-us/us"
    return attestation


def _attack(root: Path, attestation: dict, a_scopes, new_scopes, cited=()):
    old = _signed(root, "rel-a", a_scopes)
    new = _signed(root, "rel-b", new_scopes)
    plain = classify_attestation_carry(
        attestation, target=new, attested_releases=[old], cited_paths=cited
    )
    checked = classify_attestation_carry(
        attestation,
        target=new,
        attested_releases=[old],
        cited_paths=cited,
        view_a=corpus_view(root, old, [TEST_RELEASE_PUBLIC_KEY]),
        view_b=corpus_view(root, new, [TEST_RELEASE_PUBLIC_KEY]),
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
    _assert_refused(
        *_attack(tmp_path, attestation, [STATUTE_BASE], [STATUTE_BASE, STATUTE_ADDED])
    )


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
    _assert_refused(
        *_attack(tmp_path, attestation, [STATUTE_BASE], [STATUTE_BASE, STATUTE_ADDED])
    )


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
    _assert_refused(
        *_attack(tmp_path, attestation, [STATUTE_BASE], [STATUTE_BASE, STATUTE_ADDED])
    )


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
    _assert_refused(
        *_attack(tmp_path, attestation, [STATUTE_BASE], [STATUTE_BASE, dormant])
    )


def test_attack_composed_container_gains_a_descendant(tmp_path):
    _write_rows(
        tmp_path,
        STATUTE_BASE,
        [
            {"citation_path": "us/statute/7/2018", "body": None, "heading": "Sec."},
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
        [{"citation_path": "us/statute/7/2018/b", "body": "(b) New.", "heading": "b"}],
    )
    _assert_refused(
        *_attack(tmp_path, attestation, [STATUTE_BASE], [STATUTE_BASE, STATUTE_ADDED])
    )


REGULATION = ("us", "regulation", "2026-01-06-reg")
DORMANT_REGULATION = ("us", "regulation", "2026-02-01-dormant")
REGULATION_CITATION = "us/regulation/7/273/1"


def _dormant_cited_setup(root: Path):
    _write_rows(root, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "Unit."}])
    _write_rows(
        root,
        REGULATION,
        [
            {
                "citation_path": REGULATION_CITATION,
                "body": "(a) Alpha amount is $100.\n(b) Beta amount.",
            },
            {
                "citation_path": REGULATION_CITATION + "/a",
                "body": "(a) Alpha amount is $999.",
                "version": DORMANT_REGULATION[2],
            },
        ],
    )
    _write_rows(
        root,
        DORMANT_REGULATION,
        [{"citation_path": "us/regulation/7/9999", "body": "filler"}],
    )


def test_attack_cited_slice_becomes_a_dormant_exact_row(tmp_path):
    """Round 2, A1: a cited child sliced from its parent in A resolves to a
    dormant exact row in B, from the same unchanged file."""

    _dormant_cited_setup(tmp_path)
    attestation = _real_attestation(tmp_path, [STATUTE_BASE, REGULATION], UNIT_CITATION)
    plain, checked = _attack(
        tmp_path,
        attestation,
        [STATUTE_BASE, REGULATION],
        [STATUTE_BASE, REGULATION, DORMANT_REGULATION],
        cited=[REGULATION_CITATION + "/a"],
    )
    _assert_refused(plain, checked)
    assert any("differently" in r or "rows of a scope" in r for r in checked.reasons)


def test_attack_unit_slice_becomes_a_dormant_exact_row(tmp_path):
    """Round 2, A1u: the same trick on the unit itself."""

    _dormant_cited_setup(tmp_path)
    attestation = _real_attestation(
        tmp_path, [STATUTE_BASE, REGULATION], REGULATION_CITATION + "/a"
    )
    _assert_refused(
        *_attack(
            tmp_path,
            attestation,
            [STATUTE_BASE, REGULATION],
            [STATUTE_BASE, REGULATION, DORMANT_REGULATION],
        )
    )


def test_real_resolver_carries_an_unaffected_unit(tmp_path):
    _write_rows(
        tmp_path, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "unit words"}]
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], UNIT_CITATION)
    _write_rows(
        tmp_path, STATUTE_ADDED, [{"citation_path": "us/statute/7/2016", "body": "x"}]
    )
    plain, checked = _attack(
        tmp_path,
        attestation,
        [STATUTE_BASE],
        [STATUTE_BASE, STATUTE_ADDED],
        cited=[UNIT_CITATION],
    )
    assert plain.status is CarryStatus.NOT_VERIFIABLE
    assert checked.status is CarryStatus.CARRIED
    assert checked.resolution_checked is True
    assert checked.pair_gained_scopes == ("us/statute/2026-02-01-recovery",)

    tampered = dict(attestation, source_sha256="9" * 64)
    _plain, checked = _attack(
        tmp_path, tampered, [STATUTE_BASE], [STATUTE_BASE, STATUTE_ADDED]
    )
    assert checked.status is CarryStatus.NOT_IN_LINEAGE


def test_real_resolver_refuses_a_cited_path_that_does_not_resolve(tmp_path):
    _write_rows(
        tmp_path, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "unit words"}]
    )
    attestation = _real_attestation(tmp_path, [STATUTE_BASE], UNIT_CITATION)
    _write_rows(
        tmp_path, STATUTE_ADDED, [{"citation_path": "us/statute/7/2016", "body": "x"}]
    )
    _plain, checked = _attack(
        tmp_path,
        attestation,
        [STATUTE_BASE],
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
def test_corpus_view_reports_amendment_context(
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
    evidence = corpus_view(root, old, [TEST_RELEASE_PUBLIC_KEY]).unit(UNIT_CITATION)
    assert evidence.has_amendment_context is expected
    assert evidence.generation_input_sha256 == (
        hashlib.sha256(b"unit\nwords").hexdigest()
    )


def test_corpus_view_reads_row_scopes(base):
    root, old = base
    _write_rows(
        root,
        SIBLING,
        [
            {"citation_path": "us/statute/7/2015", "body": "a"},
            {"citation_path": "us/statute/7/2015/x", "body": "b", "version": "v9"},
        ],
    )
    view = corpus_view(root, old, [TEST_RELEASE_PUBLIC_KEY])
    assert view.row_scopes(_rel(SIBLING)) == {
        "us/statute/2026-01-02-sibling",
        "us/statute/v9",
    }
    with pytest.raises(ValueError):
        view.row_scopes("data/corpus/provisions/us/statute/missing.jsonl")


def test_cited_artifact_must_be_identical_in_both_releases(tmp_path):
    only_old = ("us", "policy", "2026-01-07-old-only")
    _write_scope(tmp_path, UNIT, UNIT_CITATION, "unit words")
    _write_scope(tmp_path, OTHER_PAIR, OTHER_CITATION, "other words")
    _write_scope(tmp_path, only_old, "us/policy/p/1", "old policy")
    old = _signed(tmp_path, "carry-old", [UNIT, OTHER_PAIR, only_old])
    attestation = _attestation(tmp_path, old)
    _write_scope(tmp_path, ADDED_OTHER_PAIR, "us/guidance/x/1", "added words")
    new = _signed(tmp_path, "carry-new", [UNIT, OTHER_PAIR, ADDED_OTHER_PAIR])
    unit = _evidence(attestation)
    for scope in (only_old, ADDED_OTHER_PAIR):
        digest = hashlib.sha256(_provisions_path(tmp_path, scope).read_bytes())
        cited = {OTHER_CITATION: _cited([(_rel(scope), digest.hexdigest())])}
        verdict = _carry(
            attestation,
            new,
            old,
            cited_paths=[OTHER_CITATION],
            view_a=_view(unit, cited),
            view_b=_view(unit, cited),
        )
        assert verdict.status is CarryStatus.CONTEXT_CHANGED, scope


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
    repo = tmp_path / "rulespec-us"
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
    ledger = _bind(repo, "known-validation-gaps.yaml", b"entries: []\n")
    tests_file = _bind(repo, "us/a.test.yaml", b"- name: case\n  output: {}\n")
    repo = _manifest_repo(
        tmp_path,
        {
            "a.json": {
                "source_attestation": {
                    "corpus_release": "x",
                    "rulespec_root": "rulespec-us/us",
                },
                "applied_files": [
                    module,
                    ledger,
                    tests_file,
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
    assert found["a.json"].attestation == {
        "corpus_release": "x",
        "rulespec_root": "rulespec-us/us",
    }
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


def _single_entry(
    tmp_path: Path,
    raw: bytes,
    *,
    name="us/a.yaml",
    tamper=None,
    attestation=None,
    applied=None,
):
    repo = _manifest_repo(tmp_path, {})
    module = _bind(repo, name, raw)
    if tamper is not None:
        tamper(repo)
    if attestation is None:
        attestation = {"rulespec_root": "rulespec-us/us"}
    repo = _manifest_repo(
        tmp_path,
        {
            "a.json": {
                "source_attestation": attestation,
                "applied_files": [module] if applied is None else applied,
            }
        },
    )
    (entry,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
    return entry


@pytest.mark.parametrize(
    "citation",
    [
        " us/guidance/x/1",
        "us/guidance/x/1 ",
        "US/guidance/x/1",
        "/us/guidance/x/1",
        "us/guidance/x/1/",
        "us/guidances/x/1",
    ],
)
def test_non_canonical_citations_are_refused(base, tmp_path, citation):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    entry = _single_entry(tmp_path, _rulespec([json.dumps(citation)]))
    assert entry.problems == ()
    assert citation in entry.cited_paths
    verdict = _carry(_attestation(root, old), new, old, cited_paths=entry.cited_paths)
    assert verdict.status is CarryStatus.MALFORMED


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        (b"format: [unclosed\n", "not strict YAML"),
        (b"format: rulespec/v2\nmodule: {}\n", "unsupported format"),
        (
            b"format: rulespec/v1\nmodule:\n  source_verification:\n"
            b"    corpus_citation_path: us/guidance/x/1\n"
            b"    corpus_citation_path: us/statute/7/2015\n",
            "duplicate key",
        ),
        (b"- just\n- a list\n", "not a YAML mapping"),
        (
            b"format: rulespec/v1\nmodule:\n  corpus_citation_path: [us/x/1]\n",
            "not a string",
        ),
        (
            b"format: rulespec/v1\nmodule:\n  corpus_citation_paths: us/x/1\n",
            "not a list of strings",
        ),
    ],
)
def test_bound_files_that_do_not_parse_strictly_are_malformed(tmp_path, raw, problem):
    entry = _single_entry(tmp_path, raw)
    assert any(problem in item for item in entry.problems), entry.problems


def test_sha256_is_checked_before_the_format(tmp_path):
    def replace(repo):
        (repo / "us" / "a.yaml").write_text("format: rulespec/v2\n")

    entry = _single_entry(tmp_path, _rulespec([UNIT_CITATION]), tamper=replace)
    assert entry.problems == ("applied file us/a.yaml differs from the manifest",)


def test_yml_files_and_imports_are_read(tmp_path):
    raw = _rulespec([UNIT_CITATION]) + b"imports:\n- us-az:policies/des/x/y#z\n"
    entry = _single_entry(tmp_path, raw, name="us/a.yml")
    assert entry.problems == ()
    assert UNIT_CITATION in entry.cited_paths
    # The import names no existing module, so validation looks nothing up.
    assert entry.dependency_paths == ()


def test_bound_file_under_a_symlinked_directory_is_refused(tmp_path):
    def link(repo):
        (repo / "us").rename(repo / "real")
        (repo / "us").symlink_to(repo / "real")

    entry = _single_entry(tmp_path, _rulespec([UNIT_CITATION]), tamper=link)
    assert any("not a regular file" in item for item in entry.problems)


def test_duplicate_json_keys_are_malformed(tmp_path):
    manifests = tmp_path / "repo" / ".axiom" / "encoding-manifests"
    manifests.mkdir(parents=True)
    (manifests / "a.json").write_text(
        '{"source_attestation": {"corpus_release": "evil"},'
        ' "source_attestation": {}, "applied_files": []}'
    )
    (entry,) = iter_manifest_attestations(manifests)
    assert entry.attestation is None
    assert "duplicate key" in entry.problems[0]


def test_retired_key_is_accepted_for_the_attested_release_only(base, capsys):
    from base64 import b64encode

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    other = b64encode(
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode()
    repo = _manifest_repo(
        root,
        {
            "a.json": {
                "source_attestation": _attestation(root, old),
                "applied_files": [],
            }
        },
    )
    common = [
        "--rulespec-repo",
        str(repo),
        "--to-release",
        str(_release_file(root, new)),
    ]
    from_old = ["--from-release", str(_release_file(root, old))]
    # The fixture key signs both releases: as a retired key it verifies A only.
    assert (
        run_corpus_attestation_carry(
            [
                *common,
                *from_old,
                "--public-key",
                other,
                "--retired-public-key",
                TEST_RELEASE_PUBLIC_KEY,
            ]
        )
        == 2
    )
    assert "signature is invalid" in capsys.readouterr().err
    assert (
        run_corpus_attestation_carry(
            [
                "--rulespec-repo",
                str(repo),
                "--to-release",
                str(_release_file(root, old)),
                "--from-release",
                str(_release_file(root, old)),
                "--public-key",
                other,
                "--retired-public-key",
                TEST_RELEASE_PUBLIC_KEY,
            ]
        )
        == 2
    )
    capsys.readouterr()
    assert (
        run_corpus_attestation_carry(
            [
                *common,
                *from_old,
                "--public-key",
                other,
                "--public-key",
                TEST_RELEASE_PUBLIC_KEY,
            ]
        )
        == 0
    )


def _tree_bytes(*roots: Path) -> dict[str, bytes]:
    return {
        str(path): path.read_bytes()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_a_plain_corpus_and_the_rulespec_repo_are_not_written(tmp_path, capsys):
    corpus = tmp_path / "corpus"
    _write_rows(corpus, STATUTE_BASE, [{"citation_path": UNIT_CITATION, "body": "u"}])
    attestation = _real_attestation(corpus, [STATUTE_BASE], UNIT_CITATION)
    old = _signed(corpus, "rel-a", [STATUTE_BASE])
    _write_rows(
        corpus, STATUTE_ADDED, [{"citation_path": "us/statute/7/2016", "body": "x"}]
    )
    new = _signed(corpus, "rel-b", [STATUTE_BASE, STATUTE_ADDED])
    repo = _manifest_repo(tmp_path, {})
    module = _bind(
        repo,
        "us/a.yaml",
        f"format: rulespec/v1\nmodule:\n  corpus_citation_path: {UNIT_CITATION}\n".encode(),
    )
    repo = _manifest_repo(
        tmp_path,
        {"a.json": {"source_attestation": attestation, "applied_files": [module]}},
    )
    before = _tree_bytes(corpus, repo)
    code = run_corpus_attestation_carry(
        [
            "--rulespec-repo",
            str(repo),
            "--to-release",
            str(_release_file(corpus, new)),
            "--from-release",
            str(_release_file(corpus, old)),
            "--corpus-root",
            str(corpus),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
            "--json",
        ]
    )
    assert _tree_bytes(corpus, repo) == before
    report = json.loads(capsys.readouterr().out)
    (row,) = report["manifests"]
    assert (code, row["status"], row["resolution_checked"]) == (0, "carried", True)


def test_attested_release_may_be_signed_with_the_retired_key(base, tmp_path, capsys):
    from base64 import b64encode

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from axiom_encode.corpus_release import canonical_release_object_bytes

    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    current = Ed25519PrivateKey.generate()
    current_public = b64encode(
        current.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
    ).decode()
    payload = json.loads(_release_file(root, new).read_text("utf-8"))
    payload["signature"]["value"] = b64encode(
        current.sign(canonical_release_object_bytes(payload))
    ).decode()
    resigned = tmp_path / "b.json"
    resigned.write_text(json.dumps(payload), encoding="utf-8")
    repo = _manifest_repo(
        root,
        {
            "a.json": {
                "source_attestation": _attestation(root, old),
                "applied_files": [],
            }
        },
    )
    argv = [
        "--rulespec-repo",
        str(repo),
        "--to-release",
        str(resigned),
        "--from-release",
        str(_release_file(root, old)),
        "--public-key",
        current_public,
    ]
    assert run_corpus_attestation_carry(argv) == 2
    capsys.readouterr()
    assert (
        run_corpus_attestation_carry(
            [*argv, "--retired-public-key", TEST_RELEASE_PUBLIC_KEY]
        )
        == 0
    )


def test_changed_regulation_under_an_import_does_not_carry(base):
    root, old = base
    attestation = _attestation(root, old)
    _write_scope(root, OTHER_PAIR, OTHER_CITATION, "other words changed")
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(attestation, new, old, dependency_paths=["us/regulation/7/273/1"])
    assert verdict.status is CarryStatus.CONTEXT_CHANGED


@pytest.mark.parametrize(
    ("raw", "problem"),
    [
        (b"module: {}\nimports:\n- regulations/7/1#x\n", "unsupported format None"),
        (b"rules: []\n", "unsupported format None"),
    ],
)
def test_bound_module_without_a_format_is_malformed(tmp_path, raw, problem):
    entry = _single_entry(tmp_path, raw)
    assert any(problem in item for item in entry.problems), entry.problems
    assert entry.dependency_paths == ()


def test_waiver_ledger_is_bound_without_a_format(tmp_path):
    entry = _single_entry(tmp_path, b"entries: []\n", name="known-validation-gaps.yaml")
    assert entry.problems == ()


@pytest.mark.parametrize(
    ("applied", "problem"),
    [
        (["us/a.yaml"], "is not an object"),
        ([{"sha256": "0" * 64}], "no string path"),
        ([{"path": "us/a.YAML", "sha256": "0" * 64}], "unexpected suffix case"),
        ([{"path": "us/a.Yml", "deleted": True}], "unexpected suffix case"),
    ],
)
def test_applied_files_entries_of_unexpected_shape_are_malformed(
    tmp_path, applied, problem
):
    entry = _single_entry(tmp_path, _rulespec([UNIT_CITATION]), applied=applied)
    assert any(problem in item for item in entry.problems), entry.problems


def test_null_source_attestation_is_malformed(tmp_path):
    manifests = tmp_path / "repo" / ".axiom" / "encoding-manifests"
    manifests.mkdir(parents=True)
    (manifests / "a.json").write_text(
        '{"source_attestation": null, "applied_files": []}'
    )
    (entry,) = iter_manifest_attestations(manifests)
    assert entry.problems == ("source_attestation is not an object",)


def test_cli_refuses_a_missing_manifest_directory(base, tmp_path, capsys):
    root, old = base
    code = run_corpus_attestation_carry(
        [
            "--rulespec-repo",
            str(tmp_path / "nowhere"),
            "--to-release",
            str(_release_file(root, old)),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
        ]
    )
    assert code == 2
    assert "no manifest directory" in capsys.readouterr().err


def test_cli_refuses_a_corpus_without_the_attested_release(base, tmp_path, capsys):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    other = tmp_path / "other-corpus"
    _write_scope(other, UNIT, UNIT_CITATION, "unit words")
    _signed(other, "carry-new-copy", [UNIT])
    repo = _manifest_repo(tmp_path, {})
    (repo / ".axiom" / "encoding-manifests").mkdir(parents=True)
    code = run_corpus_attestation_carry(
        [
            "--rulespec-repo",
            str(repo),
            "--to-release",
            str(_release_file(root, new)),
            "--from-release",
            str(_release_file(root, old)),
            "--corpus-root",
            str(root),
            "--from-corpus-root",
            str(other),
            "--public-key",
            TEST_RELEASE_PUBLIC_KEY,
        ]
    )
    assert code == 2
    assert "cannot load a corpus" in capsys.readouterr().err


def test_import_candidates_without_corpora_are_not_verifiable(base):
    root, old = base
    new = _signed(root, "carry-new", [UNIT, SIBLING, OTHER_PAIR])
    verdict = _carry(
        _attestation(root, old), new, old, dependency_paths=["us/regulation/9/1"]
    )
    assert verdict.status is CarryStatus.NOT_VERIFIABLE
    assert verdict.pair_gained_scopes == ()


def test_only_the_root_waiver_ledger_may_lack_a_format(tmp_path):
    entry = _single_entry(
        tmp_path, b"entries: []\n", name="us/known-validation-gaps.yaml"
    )
    assert any("unsupported format" in item for item in entry.problems)


# Round 5: import lookups come from the encoder's own functions.

STUB = b"format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n"
REAL = b"format: rulespec/v1\nmodule:\n  summary: real\nrules: []\n"


def _import_repo(tmp_path: Path, root: str, module: bytes, extra: dict[str, bytes]):
    content = root.split("/", 1)[1]
    repo = tmp_path / "rulespec-us"
    for relative, raw in extra.items():
        _bind(repo, f"{content}/{relative}", raw)
    bound = _bind(repo, f"{content}/policies/a/m.yaml", module)
    repo = _manifest_repo(
        tmp_path,
        {
            "m.json": {
                "source_attestation": {"rulespec_root": root},
                "applied_files": [bound],
            }
        },
    )
    return repo, repo / content


def _encoder_lookups(content_root: Path, module: Path) -> set[str]:
    """What ValidatorPipeline itself looks up in the corpus for one module."""

    from axiom_encode.harness import validator_pipeline
    from axiom_encode.harness.dependency_stubs import (
        _candidate_corpus_paths_for_import_target,
    )

    calls: list[tuple[str, Path]] = []

    def record(import_target, rules_repo_root, *, corpus_release):
        calls.append((import_target, rules_repo_root))
        return False

    pipeline = object.__new__(validator_pipeline.ValidatorPipeline)
    pipeline.policy_repo_path = content_root
    pipeline.rulespec_dependency_roots = ()
    pipeline.local_corpus_release = None
    original = validator_pipeline.has_corpus_provision_for_import_target
    validator_pipeline.has_corpus_provision_for_import_target = record
    try:
        pipeline._check_imported_stub_dependencies(module)
        pipeline._check_promoted_stub_file(module)
    finally:
        validator_pipeline.has_corpus_provision_for_import_target = original
    found: set[str] = set()
    for target, root in calls:
        found.update(_candidate_corpus_paths_for_import_target(target, root))
    return found


@pytest.mark.parametrize("root", ["rulespec-us/us", "rulespec-us/us-az"])
@pytest.mark.parametrize(
    "imports",
    [
        "imports:\n- policies/b/stub#x\n",
        "module:\n  imports:\n  - policies/b/stub#x\n",
        "imports:\n- policies/b/real#x\n",
        "imports:\n- policies/b/missing#x\n",
        "imports:\n- regulations/7/273/1#x\n",
    ],
)
def test_import_lookups_equal_the_encoders(tmp_path, monkeypatch, root, imports):
    module = b"format: rulespec/v1\n" + imports.encode() + b"rules: []\n"
    repo, content_root = _import_repo(
        tmp_path,
        root,
        module,
        {"policies/b/stub.yaml": STUB, "policies/b/real.yaml": REAL},
    )
    expected = _encoder_lookups(content_root, content_root / "policies/a/m.yaml")
    results = []
    for cwd in (tmp_path, repo, content_root / "policies"):
        monkeypatch.chdir(cwd)
        (entry,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
        assert entry.problems == ()
        results.append(set(entry.dependency_paths))
    assert results == [expected] * 3
    if "stub#" in imports:
        assert expected, "a stub import is a corpus lookup"
    else:
        assert not expected, "a non-stub or missing module import is not"


def test_a_stub_module_is_looked_up_by_its_own_path(tmp_path):
    repo, content_root = _import_repo(tmp_path, "rulespec-us/us-az", STUB, {})
    (entry,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
    expected = _encoder_lookups(content_root, content_root / "policies/a/m.yaml")
    assert expected and set(entry.dependency_paths) == expected


@pytest.mark.parametrize(
    ("root", "problem"),
    [
        ("rulespec-us/us-wt", "is not a directory"),
        ("rulespec-uk/uk", "attestation names 'rulespec-uk'"),
        ("rulespec-us", "no canonical rulespec_root"),
    ],
)
def test_attested_root_must_match_the_checkout(tmp_path, root, problem):
    repo = tmp_path / "rulespec-us"
    bound = _bind(repo, "us/policies/a/m.yaml", REAL)
    repo = _manifest_repo(
        tmp_path,
        {
            "m.json": {
                "source_attestation": {"rulespec_root": root},
                "applied_files": [bound],
            }
        },
    )
    (entry,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
    assert any(problem in item for item in entry.problems), entry.problems


def test_module_outside_the_attested_content_root_is_malformed(tmp_path):
    repo = tmp_path / "rulespec-us"
    (repo / "us-az").mkdir(parents=True)
    bound = _bind(repo, "us/policies/a/m.yaml", REAL)
    repo = _manifest_repo(
        tmp_path,
        {
            "m.json": {
                "source_attestation": {"rulespec_root": "rulespec-us/us-az"},
                "applied_files": [bound],
            }
        },
    )
    (entry,) = iter_manifest_attestations(repo / ".axiom" / "encoding-manifests")
    assert any("outside the attested content root" in p for p in entry.problems)


def test_checkout_with_another_name_is_refused(tmp_path):
    repo = tmp_path / "rulespec-us-wt"
    bound = _bind(repo, "us/policies/a/m.yaml", REAL)
    manifests = repo / ".axiom" / "encoding-manifests"
    manifests.mkdir(parents=True)
    (manifests / "m.json").write_text(
        json.dumps(
            {
                "source_attestation": {"rulespec_root": "rulespec-us/us"},
                "applied_files": [bound],
            }
        )
    )
    (entry,) = iter_manifest_attestations(manifests)
    assert any("'rulespec-us-wt'" in item for item in entry.problems)


@pytest.mark.skipif(
    not os.environ.get("AXIOM_CARRY_DIFFERENTIAL_RULESPEC"),
    reason="set AXIOM_CARRY_DIFFERENTIAL_RULESPEC to a rulespec-us checkout",
)
def test_import_lookups_equal_the_encoders_on_a_real_checkout():
    repo = Path(os.environ["AXIOM_CARRY_DIFFERENTIAL_RULESPEC"]).resolve()
    from axiom_encode.corpus_attestation_carry import encoder_import_corpus_lookups

    checked = 0
    for content_root in sorted(
        p
        for p in repo.iterdir()
        if p.is_dir() and p.name[:2].isalpha() and not p.name.startswith(".")
    ):
        for module in sorted(content_root.rglob("*.yaml")):
            if module.name.endswith(".test.yaml") or module.is_symlink():
                continue
            text = module.read_text(encoding="utf-8", errors="replace")
            if "imports:" not in text:
                continue
            try:
                tool = set(encoder_import_corpus_lookups(module, content_root, text))
            except ValueError:
                continue
            assert tool == _encoder_lookups(content_root, module), module
            checked += 1
    assert checked

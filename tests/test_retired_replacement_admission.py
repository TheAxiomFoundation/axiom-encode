"""Retired input admission must not weaken generated-output source gates."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.cli import _resolve_encode_replacement_target
from axiom_encode.corpus_resolver import CorpusRowIdentity, ResolvedCorpusSource
from axiom_encode.harness.proof_validator import find_plural_corpus_citation_path_issues

ROOT = "us/statute/42/402/a"
CHILD = ROOT + "/1"


@pytest.fixture
def world(tmp_path, monkeypatch):
    checkout = tmp_path / "rulespec-us"
    target = checkout / "us/statutes/42/402/a.yaml"
    target.parent.mkdir(parents=True)
    companion = target.with_suffix(".test.yaml")
    companion.write_text("cases: []\n")
    row = CorpusRowIdentity(
        provision_file="us.jsonl",
        provision_file_sha256="a" * 64,
        line_number=1,
        record_id="402",
        citation_path="us/statute/42/402",
        jurisdiction="us",
        document_class="statute",
        version="2026",
        source_path=None,
        source_as_of=None,
        expression_date=None,
        body_sha256="b" * 64,
    )
    resolved = ResolvedCorpusSource(
        requested=ROOT,
        citation_path=row.citation_path,
        body="(a) Eligibility. (1) Fully insured. (2) Age 62.",
        stored_body_sha256="b" * 64,
        resolved_text_sha256="c" * 64,
        source="local",
        provision_file=row.provision_file,
        provision_file_sha256=row.provision_file_sha256,
        row=row,
        component_rows=(),
        release_name="release",
        release_content_sha256="d" * 64,
        release_selector_sha256="e" * 64,
    )

    def unit(source):
        return SimpleNamespace(
            requested=source.requested,
            citation_path=source.citation_path,
            body=source.body,
            resolved_source=source,
        )

    sources = {
        ROOT: unit(resolved),
        CHILD: unit(replace(resolved, requested=CHILD, body="(1) Fully insured.")),
    }
    release = object()
    calls = []

    def resolve(citation, passed_release):
        assert passed_release is release
        calls.append(citation)
        if citation not in sources:
            raise ValueError("missing or ambiguous source")
        return sources[citation]

    monkeypatch.setattr("axiom_encode.cli.resolve_corpus_source_unit", resolve)
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {
                "corpus_citation_path": ROOT,
                "corpus_citation_paths": [ROOT, CHILD],
            }
        },
        "rules": [],
    }

    def run():
        return _resolve_encode_replacement_target(
            SimpleNamespace(
                replace_rulespec_path=Path("us/statutes/42/402/a.yaml"),
                apply=True,
                mode="repo-augmented",
            ),
            policy_checkout_path=checkout,
            policy_repo_path=checkout / "us",
            source_unit=sources[ROOT],
            corpus_release=release,
        )

    def write():
        target.write_text(yaml.safe_dump(payload))

    return SimpleNamespace(**locals())


def test_admits_contained_descendants_without_mutating_context(world):
    world.write()
    before = world.target.read_bytes()
    result = world.run()
    assert result.context_paths == (world.target, world.companion)
    assert world.target.read_bytes() == before
    assert CHILD in world.calls
    # The same historical bytes are still invalid as newly generated output.
    assert find_plural_corpus_citation_path_issues(world.payload)


@pytest.mark.parametrize(
    "path", ["us/statute/42/402", ROOT + "b", "ca/statute/example"]
)
def test_rejects_ancestors_siblings_and_other_sources(world, path):
    world.payload["module"]["source_verification"]["corpus_citation_paths"] = [path]
    world.write()
    with pytest.raises(ValueError, match="descendants"):
        world.run()


def test_same_row_does_not_admit_text_outside_requested_slice(world):
    world.sources[CHILD] = world.unit(
        replace(world.resolved, requested=CHILD, body="(b) Other benefits.")
    )
    world.write()
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_content_sha256", "f" * 64),
        ("provision_file_sha256", "f" * 64),
        ("row", None),
    ],
)
def test_matching_text_does_not_admit_different_provenance(world, field, value):
    child = world.sources[CHILD].resolved_source
    world.sources[CHILD] = world.unit(replace(child, **{field: value}))
    world.write()
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


def test_evidence_cannot_match_across_segment_boundaries(world):
    root = replace(world.resolved, body="first", source_history=("second",))
    world.sources[ROOT] = world.unit(root)
    world.sources[CHILD] = world.unit(
        replace(root, requested=CHILD, body="firstsecond", source_history=())
    )
    world.write()
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


@pytest.mark.parametrize(
    "plural", [[], "not-a-list", [1], [CHILD, CHILD], ["us:statutes/42/402/a/1"]]
)
def test_rejects_malformed_or_noncanonical_lists(world, plural):
    world.payload["module"]["source_verification"]["corpus_citation_paths"] = plural
    world.write()
    with pytest.raises(ValueError):
        world.run()


def test_missing_source_stops_admission(world):
    del world.sources[CHILD]
    world.write()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        world.run()


def test_plural_fields_elsewhere_still_fail(world):
    world.payload["rules"] = [{"corpus_citation_paths": [ROOT]}]
    world.write()
    with pytest.raises(ValueError, match="invalid source verification"):
        world.run()


def test_duplicate_yaml_keys_do_not_discard_historical_citations(world):
    world.target.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {ROOT}\n"
        "    corpus_citation_paths: [us/statute/42/402]\n"
        f"    corpus_citation_paths: [{CHILD}]\nrules: []\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        world.run()


def test_duplicate_mapping_cannot_hide_retired_source_metadata(world):
    world.target.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {ROOT}\n"
        "    corpus_citation_paths: [us/statute/42/402]\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {ROOT}\nrules: []\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        world.run()


def test_aliased_plural_field_outside_module_is_not_removed(world):
    world.target.write_text(
        "format: rulespec/v1\nmodule:\n  source_verification: &source\n"
        f"    corpus_citation_path: {ROOT}\n"
        f"    corpus_citation_paths: [{ROOT}, {CHILD}]\n"
        "rules:\n  - source_verification: *source\n"
    )
    before = world.target.read_bytes()
    with pytest.raises(ValueError, match="invalid source verification"):
        world.run()
    assert world.target.read_bytes() == before

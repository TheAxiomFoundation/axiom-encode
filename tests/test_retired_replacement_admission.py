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
    with pytest.raises(ValueError, match="missing or ambiguous|jurisdiction"):
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


@pytest.fixture
def exact_world(world):
    parent = replace(
        world.resolved, citation_path=ROOT, row=replace(world.row, citation_path=ROOT)
    )
    child = replace(
        parent,
        requested=CHILD,
        citation_path=CHILD,
        body="(1) Fully insured.",
        stored_body_sha256="f" * 64,
        row=replace(
            parent.row,
            citation_path=CHILD,
            record_id="child",
            line_number=2,
            body_sha256="f" * 64,
        ),
    )
    world.sources[ROOT] = world.unit(parent)
    world.sources[CHILD] = world.unit(child)
    world.write()
    return world


def test_exact_descendant_records_separate_attestations_and_containment(exact_world):
    import hashlib

    world = exact_world
    original = world.target.read_bytes()
    result = world.run()
    evidence = result.retired_source_admission
    assert evidence["legacy_rulespec_sha256"] == hashlib.sha256(original).hexdigest()
    assert evidence["requested_attestation"]["row"]["citation_path"] == ROOT
    child = evidence["sources"][1]
    assert child["attestation"]["row"]["citation_path"] == CHILD
    assert child["attestation"]["source_sha256"] == "f" * 64
    span = child["containment"][0]
    parent = world.sources[ROOT].resolved_source.proof_evidence_segments[
        span["parent_segment"]
    ]
    text = parent[span["start"] : span["end"]]
    assert text == world.sources[CHILD].body
    assert hashlib.sha256(text.encode()).hexdigest() == span["segment_sha256"]
    assert world.target.read_bytes() == original
    assert find_plural_corpus_citation_path_issues(world.payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("release_name", "other"),
        ("release_selector_sha256", "a" * 64),
        ("provision_file", "other.jsonl"),
        ("provision_file_sha256", "e" * 64),
        ("requested", ROOT),
        ("citation_path", ROOT),
        ("component_rows", ("composed",)),
        ("slice_required", True),
        ("source_history", ("Additional noncontained authority",)),
        ("body", ""),
    ],
)
def test_exact_descendant_rejects_incompatible_resolution(exact_world, field, value):
    world = exact_world
    world.sources[CHILD] = world.unit(
        replace(world.sources[CHILD].resolved_source, **{field: value})
    )
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


@pytest.mark.parametrize(
    "field,value",
    [
        ("jurisdiction", "ca"),
        ("document_class", "policy"),
        ("version", "other"),
        ("source_path", "different.pdf"),
        ("source_as_of", "2025-01-01"),
        ("expression_date", "2025-01-01"),
        ("citation_path", ROOT),
    ],
)
def test_exact_descendant_rejects_incompatible_row(exact_world, field, value):
    world = exact_world
    source = world.sources[CHILD].resolved_source
    world.sources[CHILD] = world.unit(
        replace(source, row=replace(source.row, **{field: value}))
    )
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


def test_actual_signed_release_resolves_distinct_contained_rows(tmp_path):
    from axiom_encode.cli import _admit_retired_replacement_source_verification
    from axiom_encode.harness.evals import resolve_corpus_source_unit
    from tests.test_corpus_resolver import (
        _release,
        _scope,
        _write_rows,
        _write_selector,
    )

    version = "test-contained"
    root = tmp_path / "corpus"
    _write_selector(root, [_scope(version)])
    _write_rows(
        root,
        version,
        [
            {
                "id": "parent",
                "citation_path": ROOT,
                "body": "Every individual (1) Fully insured. (2) Age 62.",
            },
            {"id": "child", "citation_path": CHILD, "body": "Fully insured."},
        ],
    )
    release = _release(root)
    source = resolve_corpus_source_unit(ROOT, release)
    content = yaml.safe_dump(
        {
            "module": {
                "source_verification": {
                    "corpus_citation_path": ROOT,
                    "corpus_citation_paths": [ROOT, CHILD],
                }
            }
        }
    ).encode()
    evidence = _admit_retired_replacement_source_verification(
        content, source_unit=source, corpus_release=release
    )
    parent, child = [item["attestation"] for item in evidence["sources"]]
    assert parent["row"]["record_id"] == "parent"
    assert child["row"]["record_id"] == "child"
    assert parent["source_sha256"] != child["source_sha256"]
    assert parent["provision_file_sha256"] == child["provision_file_sha256"]


EXTERNAL = "us/guidance/example/conversion"


def external_world(world):
    external = replace(
        world.resolved,
        requested=EXTERNAL,
        citation_path=EXTERNAL,
        body="The annual amount is exactly 12 times the monthly amount.",
        row=replace(
            world.resolved.row,
            citation_path=EXTERNAL,
            record_id="conversion",
            document_class="guidance",
        ),
    )
    world.sources[EXTERNAL] = world.unit(external)
    world.payload["module"]["source_verification"]["corpus_citation_paths"].append(
        EXTERNAL
    )
    world.payload["rules"] = [
        {
            "name": "months_per_year",
            "kind": "parameter",
            "dtype": "Integer",
            "unit": "months",
            "versions": [{"effective_from": "1983-04-20", "formula": "12\n"}],
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "amount",
                            "source": {
                                "corpus_citation_path": EXTERNAL,
                                "excerpt": "exactly 12 times the monthly",
                            },
                        }
                    ]
                }
            },
        }
    ]
    return world


def test_external_scalar_retains_attestation_and_entire_legacy_rule(world):
    external_world(world).write()
    before = world.target.read_bytes()
    admission = world.run().retired_source_admission
    assert admission["contract"] == "retired-source-external-parameters/v1"
    assert admission["required_unchanged_rules"] == world.payload["rules"]
    assert (
        admission["sources"][-1]["attestation"]["requested_corpus_citation_path"]
        == EXTERNAL
    )
    assert world.target.read_bytes() == before
    assert find_plural_corpus_citation_path_issues(world.payload)


@pytest.mark.parametrize(
    "change",
    [
        "release",
        "alias",
        "slice",
        "unproved",
        "derived",
        "formula",
        "table",
        "wrong_excerpt",
        "wrong_anchor",
        "duplicate",
    ],
)
def test_external_scalar_rejects_unbound_or_nonscalar_evidence(world, change):
    external_world(world)
    rule = world.payload["rules"][0]
    source = world.sources[EXTERNAL].resolved_source
    if change == "release":
        world.sources[EXTERNAL] = world.unit(
            replace(source, release_content_sha256="f" * 64)
        )
    elif change == "alias":
        world.sources[EXTERNAL] = world.unit(
            replace(source, citation_path=EXTERNAL + "/other")
        )
    elif change == "slice":
        world.sources[EXTERNAL] = world.unit(replace(source, slice_required=True))
    elif change == "unproved":
        rule["metadata"]["proof"]["atoms"] = []
    elif change == "derived":
        rule["kind"] = "derived"
    elif change == "formula":
        rule["versions"][0]["formula"] = "6 * 2"
    elif change == "table":
        rule["indexed_by"] = "category"
    elif change == "wrong_excerpt":
        rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "twelve monthly periods"
        )
    elif change == "wrong_anchor":
        rule["metadata"]["proof"]["atoms"][0]["path"] = "source"
    elif change == "duplicate":
        world.payload["rules"].append(rule)
    world.write()
    with pytest.raises(ValueError):
        world.run()


@pytest.mark.parametrize(
    "change", ["remove", "value", "date", "unit", "proof", "duplicate"]
)
def test_external_obligation_rejects_generated_evidence_loss(world, change):
    import copy

    from axiom_encode.legacy_external_parameters import (
        external_parameter_preservation_issues,
    )

    external_world(world).write()
    admission = world.run().retired_source_admission
    generated = copy.deepcopy(world.payload)
    del generated["module"]["source_verification"]["corpus_citation_paths"]
    assert not external_parameter_preservation_issues(generated, admission)
    rule = generated["rules"][0]
    if change == "remove":
        generated["rules"] = []
    elif change == "value":
        rule["versions"][0]["formula"] = "13"
    elif change == "date":
        rule["versions"][0]["effective_from"] = "2026-01-01"
    elif change == "unit":
        rule["unit"] = "years"
    elif change == "proof":
        rule["metadata"]["proof"]["atoms"] = []
    elif change == "duplicate":
        generated["rules"].append(copy.deepcopy(rule))
    assert external_parameter_preservation_issues(generated, admission)


def test_external_scalar_does_not_hide_top_level_derived_proof(world):
    import copy

    external_world(world)
    rule = copy.deepcopy(world.payload["rules"][0])
    rule["name"] = "derived_with_external_proof"
    rule["kind"] = "derived"
    rule["proof"] = rule.pop("metadata")["proof"]
    world.payload["rules"].append(rule)
    world.write()
    with pytest.raises(ValueError, match="outside admitted scalar"):
        world.run()


def test_whitespace_containment_retains_replayable_offsets(world):
    world.sources[CHILD] = world.unit(
        replace(world.sources[CHILD].resolved_source, body="(1)\n\tFully   insured.")
    )
    world.write()
    admission = world.run().retired_source_admission
    match = admission["sources"][1]["containment"][0]
    assert admission["contract"] == "retired-source-containment/v2"
    assert match["normalization"] == "unicode-whitespace/v1"
    raw = world.resolved.proof_evidence_segments[0][match["start"] : match["end"]]
    assert " ".join(raw.split()) == "(1) Fully insured."


def test_whitespace_containment_never_normalizes_changed_words(world):
    world.sources[CHILD] = world.unit(
        replace(world.sources[CHILD].resolved_source, body="(1)\nNot fully insured.")
    )
    world.write()
    with pytest.raises(ValueError, match="not fully covered"):
        world.run()


@pytest.mark.parametrize("location", ["formula", "metadata"])
def test_external_preservation_distinguishes_booleans_from_integers(location):
    import copy

    from axiom_encode.legacy_external_parameters import (
        external_parameter_preservation_issues,
    )

    rule = {
        "name": "scalar",
        "versions": [{"formula": 1}],
        "metadata": {"numeric_marker": 1},
    }
    changed = copy.deepcopy(rule)
    if location == "formula":
        changed["versions"][0]["formula"] = True
    else:
        changed["metadata"]["numeric_marker"] = True
    assert external_parameter_preservation_issues(
        {"rules": [changed]}, {"required_unchanged_rules": [rule]}
    )

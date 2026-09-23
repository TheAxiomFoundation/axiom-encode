"""Validation-level reuse of corpus resolution.

``validation-waivers audit`` validates every waived module against one shared
``LocalCorpusRelease`` per worker, so a module's corpus lookups may be served
by resolutions its neighbors already performed. Waiver fingerprints must not
depend on that: these tests compare every module's fingerprint against its
standalone value computed with a fresh release.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode import cli, corpus_resolver, validation_waivers
from axiom_encode.corpus_resolver import LocalCorpusRelease
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from tests.release_object_fixtures import bind_test_corpus_release

_BASE_ROW = {
    "jurisdiction": "nz",
    "source_path": "sources/nz/fixture.xml",
    "source_as_of": "2026-01-02",
    "expression_date": "2026-01-01",
}
_STATUTE_ROWS = [
    {"citation_path": "nz/statute/example/levy", "body": "The levy rate is 0.05"},
    {"citation_path": "nz/statute/example/dup", "body": "The levy rate is 0.05"},
    {"citation_path": "nz/statute/example/dup", "body": "The levy rate is 0.05"},
    {
        "citation_path": "nz/statute/example/act",
        "body": "(1) The levy rate is 0.05.\n(2) The surcharge rate is 0.07.",
    },
    {"citation_path": "nz/statute/example/parts", "body": None},
    {"citation_path": "nz/statute/example/parts/a", "body": "The levy rate is 0.05"},
    {"citation_path": "nz/statute/example/parts/b", "body": "Other text"},
]
_GUIDANCE_ROWS = [
    {"citation_path": "nz/guidance/example/rates", "body": "Rates: 0.05 and 0.07"},
]
_MODULE_ATOMS = {
    "a_resolved": ["nz/statute/example/levy"] * 3,
    "b_missing": ["nz/guidance/example/levy-rates", "nz/statute/example/nope"],
    "c_ambiguous": ["nz/statute/example/dup"],
    "d_sliced": ["nz/statute/example/act/1", "nz/statute/example/act/2"],
    "e_composed": ["nz/statute/example/parts", "nz/guidance/example/rates"],
    "f_mixed": [
        "nz/statute/example/levy",
        "nz/guidance/example/rates",
        "nz/statute/example/nope",
        "nz/statute/example/levy",
        "nz/statute/example/act/1",
    ],
    "g_invalid": ["nz/statute/example/Bad Segment!", "nz/statute/example/levy"],
}
_ATOM = """\
          - kind: parameter
            path: versions[0].formula
            source:
              corpus_citation_path: {path}
              excerpt: "The levy rate is 0.05"
"""
_MODULE = """\
format: rulespec/v1
module:
  title: Example levy {name}
  jurisdiction: nz
  citation_path: nz/statutes/example/{name}
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: {source}
rules:
  - name: {name}_rate
    output: nz:statutes/example/{name}#{name}_rate
    metadata:
      proof:
        atoms:
{atoms}    versions:
      - effective_from: '2026-04-01'
        formula: |-
          0.05
"""
_COMPANION = {"present": True, "passed": True, "path": None, "cases": 1}


def _write_provisions(corpus_root: Path, document_class: str, rows) -> None:
    path = corpus_root / "data/corpus/provisions/nz" / document_class / "v1.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(
                {
                    **_BASE_ROW,
                    "id": f"{document_class}-{index}",
                    "document_class": document_class,
                    "version": "v1",
                    **row,
                },
                sort_keys=True,
            )
            + "\n"
            for index, row in enumerate(rows)
        ),
        encoding="utf-8",
    )


@pytest.fixture
def fixture_checkout(tmp_path: Path):
    corpus_root = tmp_path / "axiom-corpus"
    _write_provisions(corpus_root, "statute", _STATUTE_ROWS)
    _write_provisions(corpus_root, "guidance", _GUIDANCE_ROWS)
    release = bind_test_corpus_release(
        corpus_root,
        "test-release",
        [("nz", "statute", "v1"), ("nz", "guidance", "v1")],
    )
    content_root = tmp_path / "rulespec-nz" / "nz"
    module_dir = content_root / "statutes" / "example"
    module_dir.mkdir(parents=True)
    modules = []
    for name, paths in _MODULE_ATOMS.items():
        module = module_dir / f"{name}.yaml"
        module.write_text(
            _MODULE.format(
                name=name,
                source=paths[0],
                atoms="".join(_ATOM.format(path=path) for path in paths),
            ),
            encoding="utf-8",
        )
        modules.append(module)
    return SimpleNamespace(
        release=release,
        content_root=content_root,
        engine=tmp_path / "axiom-rules-engine",
        modules=modules,
    )


def _pipeline(checkout, release: LocalCorpusRelease) -> ValidatorPipeline:
    pipeline = ValidatorPipeline(
        policy_repo_path=checkout.content_root,
        axiom_rules_path=checkout.engine,
        enable_oracles=False,
        local_corpus_release=release,
    )

    def fake_compile(rules_file, output_path):
        return (
            subprocess.CompletedProcess(["axiom-rules-engine"], 0, "", ""),
            {"program": {"parameters": [], "derived": [], "relations": []}},
        )

    pipeline._compile_rulespec_to_artifact = fake_compile  # type: ignore[method-assign]
    return pipeline


def _fresh(release: LocalCorpusRelease) -> LocalCorpusRelease:
    return LocalCorpusRelease(
        release.root, release.name, release.content_sha256, release.public_key
    )


def _fingerprint(pipeline: ValidatorPipeline, module: Path) -> tuple[str, dict]:
    outcome = cli._validation_waiver_validate_outcome(
        pipeline.validate(module, skip_reviewers=True)
    )
    return validation_waivers.fingerprint_outcome(outcome, _COMPANION), outcome


def test_waiver_fingerprints_do_not_depend_on_a_shared_warm_release(
    fixture_checkout, monkeypatch
):
    checkout = fixture_checkout
    standalone = {
        module.name: _fingerprint(_pipeline(checkout, _fresh(checkout.release)), module)
        for module in checkout.modules
    }
    issues = "\n".join(
        issue
        for _fingerprint_value, outcome in standalone.values()
        for validator in outcome["validators"].values()
        for issue in validator["issues"]
    )
    # The fully resolved module's only finding is its missing companion tests.
    assert standalone["a_resolved.yaml"][1]["validators"]["ci"]["issues"] == [
        "No tests found."
    ]
    for expected in (
        "Proof source unresolved",
        "Ambiguous active corpus citation",
        "Proof source evidence not found",
        "Invalid corpus citation path segment",
    ):
        assert expected in issues
    assert "Sibling rule name collision" not in issues
    assert len({value for value, _ in standalone.values()}) == len(standalone)

    reads: list[Path] = []
    original = corpus_resolver.read_bounded_regular_file

    def tracked(root, candidate, *, label, max_bytes, **kwargs):
        if label == "corpus provision file":
            reads.append(Path(candidate))
        return original(root, candidate, label=label, max_bytes=max_bytes, **kwargs)

    monkeypatch.setattr(corpus_resolver, "read_bounded_regular_file", tracked)
    for order in (checkout.modules, list(reversed(checkout.modules))):
        reads.clear()
        shared = _fresh(checkout.release)
        pipeline = _pipeline(checkout, shared)
        for _ in range(2):
            for module in order:
                assert _fingerprint(pipeline, module) == standalone[module.name], (
                    module.name
                )
        # Both buckets were read once for the whole batch.
        assert sorted(path.parent.name for path in reads) == ["guidance", "statute"]


def test_cited_source_texts_fetch_each_distinct_path_once(monkeypatch):
    fetched: list[tuple[str, str]] = []

    def fake_fetch(kind: str):
        def fetch(citation_path: str):
            fetched.append((kind, citation_path))
            if citation_path == "nz/statute/bad":
                raise corpus_resolver.InvalidCorpusCitationError("bad citation")
            return f"{kind}:{citation_path}"

        return fetch

    monkeypatch.setattr(
        validator_pipeline, "_fetch_corpus_source_text", fake_fetch("body")
    )
    monkeypatch.setattr(
        validator_pipeline,
        "_fetch_corpus_proof_evidence_text",
        fake_fetch("proof"),
    )
    paths = [
        "nz/statute/b",
        "nz/statute/a",
        "nz/statute/b",
        " nz/statute/a ",
        "nz/statute/bad",
        "nz/statute/bad",
        "nz/statute/c",
    ]
    content = _MODULE.format(
        name="dedup",
        source="nz/statute/a",
        atoms="".join(_ATOM.format(path=f"'{path}'") for path in paths),
    )
    source_texts = {"nz/statute/a": "trusted body", "unrelated": "kept"}

    for kind, proof_evidence in (("proof", True), ("body", False)):
        fetched.clear()
        resolved = ValidatorPipeline._cited_source_texts_for_rulespec_content(
            SimpleNamespace(),  # type: ignore[arg-type]
            content,
            source_texts=source_texts,
            proof_evidence=proof_evidence,
        )
        assert fetched == [
            (kind, "nz/statute/b"),
            (kind, "nz/statute/a"),
            (kind, "nz/statute/bad"),
            (kind, "nz/statute/c"),
        ]
        # Same mapping, key order, and overwrite of caller-supplied text as a
        # fetch per atom would produce.
        assert list(resolved.items()) == [
            ("nz/statute/a", f"{kind}:nz/statute/a"),
            ("unrelated", "kept"),
            ("nz/statute/b", f"{kind}:nz/statute/b"),
            ("nz/statute/bad", None),
            ("nz/statute/c", f"{kind}:nz/statute/c"),
        ]
    assert source_texts == {"nz/statute/a": "trusted body", "unrelated": "kept"}

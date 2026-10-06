from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

import pytest

from axiom_encode.corpus_resolver import (
    AmbiguousCorpusSourceError,
    InvalidCorpusReleaseError,
)
from axiom_encode.oracles.policyengine.snap_readiness import (
    SnapReadinessConfigurationError,
    build_snap_readiness_report,
)
from axiom_encode.toolchain import RuleSpecToolchainError
from tests.release_object_fixtures import bind_test_corpus_release

TEST_SELECTOR = "snap-readiness-test-release"


def _write_toolchain(repo: Path, *, content_sha256: str = "0" * 64) -> None:
    waiver = repo / "known-validation-gaps.yaml"
    waiver.parent.mkdir(parents=True, exist_ok=True)
    waiver.write_text("validate_failures: {}\n", encoding="utf-8")
    waiver_sha256 = hashlib.sha256(waiver.read_bytes()).hexdigest()
    toolchain = repo / ".axiom" / "toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        f'axiom_corpus_release = "{TEST_SELECTOR}"\n'
        f'axiom_corpus_release_content_sha256 = "{content_sha256}"\n'
        f'validation_waiver_set_sha256 = "{waiver_sha256}"\n',
        encoding="utf-8",
    )


def _write_empty_release(
    corpus_root: Path,
    *,
    jurisdiction: str,
    bind_toolchain: bool = True,
) -> None:
    version = "empty-release"
    provisions = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / "regulation"
        / f"{version}.jsonl"
    )
    provisions.parent.mkdir(parents=True, exist_ok=True)
    provisions.write_text(
        json.dumps(
            {
                "id": f"{jurisdiction}-other-benefit",
                "citation_path": f"{jurisdiction}/regulation/demo/other-benefit",
                "jurisdiction": jurisdiction,
                "document_class": "regulation",
                "version": version,
                "body": "Unrelated program text.",
                "metadata": {"program": "OTHER"},
                "source_path": f"sources/{jurisdiction}/regulation/{version}/source",
                "source_as_of": "2026-01-01",
                "expression_date": "2026-01-01",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    release = bind_test_corpus_release(
        corpus_root,
        TEST_SELECTOR,
        [(jurisdiction, "regulation", version)],
    )
    if bind_toolchain:
        _write_toolchain(
            corpus_root.parent / "rulespec-us",
            content_sha256=release.content_sha256,
        )


def _bound_release_scopes(corpus_root: Path) -> list[tuple[str, str, str]]:
    toolchain = corpus_root.parent / "rulespec-us" / ".axiom" / "toolchain.toml"
    if not toolchain.is_file():
        return []
    payload = tomllib.loads(toolchain.read_text(encoding="utf-8"))
    content_sha256 = payload.get("toolchain", {}).get(
        "axiom_corpus_release_content_sha256"
    )
    if not isinstance(content_sha256, str):
        return []
    release_object = corpus_root / "releases" / TEST_SELECTOR / f"{content_sha256}.json"
    if not release_object.is_file():
        return []
    release_payload = json.loads(release_object.read_text(encoding="utf-8"))
    return [
        (
            str(scope["jurisdiction"]),
            str(scope["document_class"]),
            str(scope["version"]),
        )
        for scope in release_payload["content"]["scopes"]
    ]


def _write_rulespec(
    module: Path,
    relative: str,
    *,
    rule_name: str = "snap_eligible",
    kind: str = "derived",
):
    _write_toolchain(module.parent)
    path = module / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "derived_relation":
        body = (
            f"  - name: {rule_name}\n"
            "    kind: derived_relation\n"
            "    derived_relation:\n"
            "      arity: 2\n"
            "      source_relation: member_of_household\n"
            "      entity: SnapUnit\n"
            "      member_relation: members\n"
            "      slot_entities: [Person, Household]\n"
            "    versions:\n"
            "      - effective_from: '2025-10-01'\n"
            "        formula: snap_member_eligible\n"
        )
    else:
        body = (
            f"  - name: {rule_name}\n"
            f"    kind: {kind}\n"
            "    entity: Household\n"
            "    dtype: Judgment\n"
            "    period: Month\n"
            "    versions:\n"
            "      - effective_from: '2025-10-01'\n"
            "        formula: true\n"
        )
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  kind: law\n"
        "  source_verification:\n"
        "    corpus_citation_path: us-tn/regulation/demo/snap\n"
        "rules:\n"
        f"{body}"
    )
    return path


def _write_corpus_provision(
    corpus_root: Path,
    citation_path: str,
    *,
    body: str = "Supplemental Nutrition Assistance Program text.",
    program: str = "SNAP",
    version: str = "demo-active",
    active: bool = True,
):
    jurisdiction, document_class, *_ = citation_path.split("/")
    path = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / document_class
        / f"{version}.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": f"{jurisdiction}-{document_class}-{version}",
        "citation_path": citation_path,
        "jurisdiction": jurisdiction,
        "document_class": document_class,
        "version": version,
        "heading": f"Demo {program} provision",
        "body": body,
        "metadata": {"program": program},
        "source_path": f"sources/{jurisdiction}/{document_class}/{version}/source.html",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
    }
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    if active:
        scope = (jurisdiction, document_class, version)
        scopes = _bound_release_scopes(corpus_root)
        if scope not in scopes:
            scopes.append(scope)
        release = bind_test_corpus_release(
            corpus_root,
            TEST_SELECTOR,
            scopes,
        )
        _write_toolchain(
            corpus_root.parent / "rulespec-us",
            content_sha256=release.content_sha256,
        )
    return path


def test_snap_readiness_reports_ready_to_encode_when_corpus_exists_without_rulespec(
    tmp_path,
):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-tn"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    _write_corpus_provision(corpus_root, "us-tn/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-tn"]["status"] == "ready_to_encode"
    assert by_jurisdiction["us-tn"]["corpus_snap_provisions"] == 1
    assert by_jurisdiction["us-tn"]["rulespec_files"] == 0
    assert by_jurisdiction["us-tn"]["module"] == "rulespec-us/us-tn"
    assert report["rulespec_repo"] == str(root)


def test_snap_readiness_ignores_archived_standalone_state_repos(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-co"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    _write_empty_release(corpus_root, jurisdiction="us-co")
    (tmp_path / "rulespec-us-tn").mkdir(parents=True)

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    assert report["total_modules"] == 1
    assert [item["jurisdiction"] for item in report["items"]] == ["us-co"]


def test_snap_readiness_rejects_canonical_monorepo_without_state_modules(tmp_path):
    root = tmp_path / "rulespec-us"
    root.mkdir(parents=True)

    with pytest.raises(
        SnapReadinessConfigurationError,
        match="contains no us-xx state modules",
    ):
        build_snap_readiness_report(root, corpus_root=tmp_path / "axiom-corpus")


def test_snap_readiness_rejects_workspace_root(tmp_path):
    checkout = tmp_path / "rulespec-us"
    (checkout / "us-tn").mkdir(parents=True)

    with pytest.raises(
        SnapReadinessConfigurationError,
        match="exact canonical rulespec-us checkout",
    ):
        build_snap_readiness_report(tmp_path, corpus_root=tmp_path / "axiom-corpus")


@pytest.mark.parametrize(
    "relative, expected",
    [
        ("policies/demo.yml", "canonical .yaml extension"),
        ("legacy/demo.yaml", "canonical module root"),
    ],
)
def test_snap_readiness_rejects_noncanonical_module_layout(
    tmp_path,
    relative,
    expected,
):
    root = tmp_path / "rulespec-us"
    module = root / "us-tn"
    module.mkdir(parents=True)
    _write_empty_release(tmp_path / "axiom-corpus", jurisdiction="us-tn")
    path = module / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("format: rulespec/v1\nrules: []\n")

    with pytest.raises(SnapReadinessConfigurationError, match=expected):
        build_snap_readiness_report(
            root,
            corpus_root=tmp_path / "axiom-corpus",
        )


def test_snap_readiness_propagates_missing_root_toolchain(tmp_path):
    root = tmp_path / "rulespec-us"
    (root / "us-tn").mkdir(parents=True)
    _write_empty_release(
        tmp_path / "axiom-corpus",
        jurisdiction="us-tn",
        bind_toolchain=False,
    )

    with pytest.raises(RuleSpecToolchainError, match="toolchain.toml"):
        build_snap_readiness_report(root, corpus_root=tmp_path / "axiom-corpus")


def test_snap_readiness_propagates_missing_named_release(tmp_path):
    root = tmp_path / "rulespec-us"
    module = root / "us-tn"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    (tmp_path / "axiom-corpus" / "data" / "corpus" / "provisions").mkdir(parents=True)

    with pytest.raises(InvalidCorpusReleaseError, match="release object not found"):
        build_snap_readiness_report(root, corpus_root=tmp_path / "axiom-corpus")


def test_snap_readiness_reports_populace_ready_for_configured_program_module(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-co"
    _write_rulespec(module, "policies/cdhs/snap/fy-2026-benefit-calculation.yaml")
    _write_corpus_provision(corpus_root, "us-co/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    colorado = by_jurisdiction["us-co"]
    assert colorado["status"] == "populace_ready"
    assert colorado["policyengine_populace_configured"] is True
    assert colorado["program_module_exists"] is True
    assert colorado["executable_outputs"] == 1


def test_snap_readiness_counts_derived_relation_outputs(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-co"
    _write_rulespec(
        module,
        "policies/cdhs/snap/fy-2026-benefit-calculation.yaml",
        rule_name="snap_unit",
        kind="derived_relation",
    )
    _write_corpus_provision(corpus_root, "us-co/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-co"]["executable_outputs"] == 1


def test_snap_readiness_ignores_composition_specs(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-co"
    _write_rulespec(module, "policies/cdhs/snap/fy-2026-benefit-calculation.yaml")
    program_spec = module / "programs" / "snap" / "fy-2026.yml"
    program_spec.parent.mkdir(parents=True)
    program_spec.write_text("not: [valid\n", encoding="utf-8")
    _write_corpus_provision(corpus_root, "us-co/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    colorado = {item["jurisdiction"]: item for item in report["items"]}["us-co"]
    assert colorado["rulespec_files"] == 1
    assert colorado["executable_outputs"] == 1


def test_snap_readiness_flags_rules_without_policyengine_config(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-tn"
    _write_rulespec(module, "policies/tdhs/snap/fy-2026-benefit-calculation.yaml")
    _write_corpus_provision(corpus_root, "us-tn/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-tn"]["status"] == "needs_populace_config"
    assert (
        "missing PolicyEngine Populace jurisdiction config"
        in by_jurisdiction["us-tn"]["blockers"]
    )


def test_snap_readiness_distinguishes_empty_repo_without_corpus(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-al"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    _write_empty_release(corpus_root, jurisdiction="us-al")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-al"]["status"] == "needs_corpus"
    assert by_jurisdiction["us-al"]["blockers"] == ["no SNAP corpus provisions found"]


def test_snap_readiness_ignores_inactive_snap_rows(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-tn"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    _write_corpus_provision(
        corpus_root,
        "us-tn/regulation/demo/other",
        body="Unrelated assistance text.",
        program="Other",
        version="active-v1",
    )
    _write_corpus_provision(
        corpus_root,
        "us-tn/regulation/demo/snap",
        version="inactive-v2",
        active=False,
    )

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-tn"]["corpus_snap_provisions"] == 0
    assert by_jurisdiction["us-tn"]["status"] == "needs_corpus"


def test_snap_readiness_rejects_ambiguous_active_corpus_rows(tmp_path):
    root = tmp_path / "rulespec-us"
    corpus_root = tmp_path / "axiom-corpus"
    module = root / "us-tn"
    module.mkdir(parents=True)
    _write_toolchain(module.parent)
    citation_path = "us-tn/regulation/demo/snap"
    _write_corpus_provision(corpus_root, citation_path, version="active-v1")
    _write_corpus_provision(corpus_root, citation_path, version="active-v2")

    with pytest.raises(AmbiguousCorpusSourceError, match="Ambiguous active"):
        build_snap_readiness_report(root, corpus_root=corpus_root)

from __future__ import annotations

import json
from pathlib import Path

from axiom_encode.oracles.policyengine.snap_readiness import (
    build_snap_readiness_report,
)


def _write_rulespec(
    repo: Path,
    relative: str,
    *,
    rule_name: str = "snap_eligible",
    kind: str = "derived",
):
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  kind: law\n"
        "  source_verification:\n"
        "    corpus_citation_path: us-tn/regulation/demo/snap\n"
        "rules:\n"
        f"  - name: {rule_name}\n"
        f"    kind: {kind}\n"
        "    entity: Household\n"
        "    dtype: Judgment\n"
        "    period: Month\n"
        "    versions:\n"
        "      - effective_from: '2025-10-01'\n"
        "        formula: true\n"
    )
    return path


def _write_corpus_provision(
    corpus_root: Path,
    citation_path: str,
    *,
    body: str = "Supplemental Nutrition Assistance Program text.",
    program: str = "SNAP",
):
    jurisdiction, document_class, *_ = citation_path.split("/")
    path = (
        corpus_root
        / "data"
        / "corpus"
        / "provisions"
        / jurisdiction
        / document_class
        / "demo.jsonl"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "citation_path": citation_path,
        "jurisdiction": jurisdiction,
        "document_class": document_class,
        "heading": "Demo SNAP provision",
        "body": body,
        "metadata": {"program": program},
    }
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")
    return path


def test_snap_readiness_reports_ready_to_encode_when_corpus_exists_without_rulespec(
    tmp_path,
):
    root = tmp_path / "workspace"
    corpus_root = root / "axiom-corpus"
    (root / "rulespec-us-tn").mkdir(parents=True)
    _write_corpus_provision(corpus_root, "us-tn/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-tn"]["status"] == "ready_to_encode"
    assert by_jurisdiction["us-tn"]["corpus_snap_provisions"] == 1
    assert by_jurisdiction["us-tn"]["rulespec_files"] == 0


def test_snap_readiness_reports_ecps_ready_for_configured_program_module(tmp_path):
    root = tmp_path / "workspace"
    corpus_root = root / "axiom-corpus"
    repo = root / "rulespec-us-co"
    _write_rulespec(repo, "policies/cdhs/snap/fy-2026-benefit-calculation.yaml")
    _write_corpus_provision(corpus_root, "us-co/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    colorado = by_jurisdiction["us-co"]
    assert colorado["status"] == "ecps_ready"
    assert colorado["policyengine_ecps_configured"] is True
    assert colorado["program_module_exists"] is True
    assert colorado["executable_outputs"] == 1


def test_snap_readiness_counts_derived_relation_outputs(tmp_path):
    root = tmp_path / "workspace"
    corpus_root = root / "axiom-corpus"
    repo = root / "rulespec-us-co"
    _write_rulespec(
        repo,
        "policies/cdhs/snap/fy-2026-benefit-calculation.yaml",
        rule_name="snap_unit",
        kind="derived_relation",
    )
    _write_corpus_provision(corpus_root, "us-co/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-co"]["executable_outputs"] == 1


def test_snap_readiness_flags_rules_without_policyengine_config(tmp_path):
    root = tmp_path / "workspace"
    corpus_root = root / "axiom-corpus"
    repo = root / "rulespec-us-tn"
    _write_rulespec(repo, "policies/tdhs/snap/fy-2026-benefit-calculation.yaml")
    _write_corpus_provision(corpus_root, "us-tn/regulation/demo/snap")

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-tn"]["status"] == "needs_ecps_config"
    assert (
        "missing PolicyEngine ECPS jurisdiction config"
        in by_jurisdiction["us-tn"]["blockers"]
    )


def test_snap_readiness_distinguishes_empty_repo_without_corpus(tmp_path):
    root = tmp_path / "workspace"
    corpus_root = root / "axiom-corpus"
    (root / "rulespec-us-al").mkdir(parents=True)

    report = build_snap_readiness_report(root, corpus_root=corpus_root)

    by_jurisdiction = {item["jurisdiction"]: item for item in report["items"]}
    assert by_jurisdiction["us-al"]["status"] == "needs_corpus"
    assert by_jurisdiction["us-al"]["blockers"] == ["no SNAP corpus provisions found"]

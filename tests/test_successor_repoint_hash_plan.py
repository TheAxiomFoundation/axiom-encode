"""The Git-object repoint plan closes over proof import hashes deterministically."""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.cli import _plan_successor_repoint
from axiom_encode.successor_repoint import (
    SuccessorRepointError,
    load_repoint_request_payload,
)
from tests.successor_repoint_fixtures import (
    DEPENDENT,
    ENVELOPE,
    TRANSITIVE,
    _v1_manifest,
    build_repoint_fixture,
)

CHAIN = "us/statutes/26/25.yaml"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _own(repo, path):
    manifest = repo / ".axiom/encoding-manifests" / Path(path).with_suffix(".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        _v1_manifest(
            [{"path": path, "sha256": _sha((repo / path).read_bytes())}],
            tool="axiom-encode repair-proof-import-hashes",
            backend="deterministic",
            runner="deterministic-repair",
            citation=path,
        )
    )


def _pin_text(target, pin):
    return (
        "format: rulespec/v1\n"
        f"imports: [{target}]\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/statute/26/25\n"
        "rules:\n"
        "  - name: downstream\n"
        "    kind: derived\n"
        "    proof:\n"
        "      atoms:\n"
        "        - import:\n"
        f"            target: {target}#upstream\n"
        f"            hash: sha256:{pin}\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: upstream\n"
    )


def _plan(fixture, envelope=None):
    return _plan_successor_repoint(
        fixture.repo,
        commit=fixture.base,
        request=load_repoint_request_payload(envelope or ENVELOPE),
        verify_live=False,
    )


def _refresh_records(plan, primary):
    return [
        replacement
        for dependent in plan.dependent_records
        if dependent["primary"] == primary
        for rewrite in dependent["rewrites"]
        for replacement in rewrite["replacements"]
        if replacement.get("operation") == "refresh_proof_import_hash"
    ]


def test_plan_refuses_undeclared_hash_pinning_importer(tmp_path, monkeypatch):
    fixture = build_repoint_fixture(tmp_path, monkeypatch)
    with pytest.raises(SuccessorRepointError, match=rf"{TRANSITIVE} must be declared"):
        _plan(fixture, {**ENVELOPE, "dependents": [DEPENDENT]})


def test_plan_accepts_declared_hash_only_importer(tmp_path, monkeypatch):
    fixture = build_repoint_fixture(tmp_path, monkeypatch)
    plan = _plan(fixture)
    before = _sha(fixture.preimages[DEPENDENT])
    after = _sha(plan.postimages[Path(DEPENDENT)])
    assert plan.postimages[Path(TRANSITIVE)] == fixture.preimages[TRANSITIVE].replace(
        f"sha256:{before}".encode(), f"sha256:{after}".encode()
    )
    (record,) = _refresh_records(plan, TRANSITIVE)
    assert record["from"] == f"sha256:{before}"
    assert record["to"] == f"sha256:{after}"
    assert record["count"] == 1


def test_plan_refuses_already_stale_pin(tmp_path, monkeypatch):
    def stale(repo):
        path = repo / TRANSITIVE
        old = _sha((repo / DEPENDENT).read_bytes())
        path.write_text(path.read_text().replace(old, "0" * 64))
        _own(repo, TRANSITIVE)

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=stale)
    with pytest.raises(SuccessorRepointError, match="already-stale proof import hash"):
        _plan(fixture)


def test_plan_refreshes_two_level_chain_in_topological_order(tmp_path, monkeypatch):
    def add(repo):
        (repo / CHAIN).write_text(
            _pin_text("us:statutes/26/24/d", _sha((repo / TRANSITIVE).read_bytes()))
        )
        _own(repo, CHAIN)
        index = repo / ".axiom/index/provisions_to_rules.json"
        payload = json.loads(index.read_text())
        payload["provisions"]["us/statute/26/25"] = [
            {"module": CHAIN, "via": ["module"]}
        ]
        index.write_text(json.dumps(payload))

    envelope = {**ENVELOPE, "dependents": [CHAIN, TRANSITIVE, DEPENDENT]}
    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=add)
    plan = _plan(fixture, envelope)
    assert [item["primary"] for item in plan.dependent_records] == [
        DEPENDENT,
        TRANSITIVE,
        CHAIN,
    ]
    post = yaml.safe_load(plan.postimages[Path(CHAIN)])
    assert post["rules"][0]["proof"]["atoms"][0]["import"]["hash"] == (
        "sha256:" + _sha(plan.postimages[Path(TRANSITIVE)])
    )
    assert len(_refresh_records(plan, CHAIN)) == 1
    with pytest.raises(SuccessorRepointError, match=rf"{CHAIN} must be declared"):
        _plan(fixture)


def test_plan_refuses_pin_cycle(tmp_path, monkeypatch):
    def cycle(repo):
        path = repo / DEPENDENT
        path.write_text(
            path.read_text()
            + "  - name: cyclic_pin\n"
            + "    proof:\n"
            + "      atoms:\n"
            + "        - import:\n"
            + "            target: us:statutes/26/24/d#child_credit\n"
            + f"            hash: sha256:{'0' * 64}\n"
        )
        _own(repo, DEPENDENT)
        # The relative manifest's companion entry is stale in this fixture.
        # Keep the canonical companion's partial ownership while rebinding primary.
        from tests.successor_repoint_fixtures import (
            DEPENDENT_COMPANION,
            DEPENDENT_MANIFEST,
        )

        owner = repo / DEPENDENT_MANIFEST
        payload = json.loads(owner.read_text())
        payload["applied_files"].append(
            {
                "path": DEPENDENT_COMPANION,
                "sha256": _sha((repo / DEPENDENT_COMPANION).read_bytes()),
            }
        )
        owner.write_text(json.dumps(payload))

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=cycle)
    with pytest.raises(SuccessorRepointError, match="hash dependency cycle"):
        _plan(fixture)


def test_plan_finds_escaped_pinning_target(tmp_path, monkeypatch):
    def escape(repo):
        path = repo / TRANSITIVE
        path.write_text(
            path.read_text()
            .replace("target: us:statutes/26/32#", 'target: "us:statutes/26/\\u00332#')
            .replace("#eitc_earned_income_amount\n", '#eitc_earned_income_amount"\n')
        )
        _own(repo, TRANSITIVE)

    fixture = build_repoint_fixture(tmp_path, monkeypatch, before_commit=escape)
    with pytest.raises(SuccessorRepointError, match=rf"{TRANSITIVE} must be declared"):
        _plan(fixture, {**ENVELOPE, "dependents": [DEPENDENT]})

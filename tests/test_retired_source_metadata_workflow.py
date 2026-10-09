"""Pin the protected model-free lane and receipt-bound publication boundaries."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import yaml

from axiom_encode.retired_source_metadata import (
    MIGRATION_TOOL,
    PLAN_SCHEMA,
    build_migration,
    load_plan_bytes,
)
from axiom_encode.retired_source_metadata_workflow import (
    EXCLUSIVE_ARRAY_INPUTS,
    EXCLUSIVE_INPUTS,
    migration_changes,
    resolve_request,
    stage_inventory,
    verify_committed_inventory,
    verify_inventory,
)
from scripts.enforce_attempt_budget import is_retired_source_metadata_plan

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"
PRIMARY = "us/statutes/26/1.yaml"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def _plan(base: str = "a" * 40) -> dict[str, object]:
    return {"schema_version": PLAN_SCHEMA, "base_commit": base, "modules": [PRIMARY]}


@lru_cache(maxsize=1)
def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _step(name: str) -> dict:
    return next(
        step
        for step in _workflow()["jobs"]["encode"]["steps"]
        if step.get("name") == name
    )


def test_dispatch_uses_existing_source_input_and_same_protected_authority() -> None:
    workflow = _workflow()
    trigger = workflow.get("on", workflow.get(True))
    assert len(trigger["workflow_dispatch"]["inputs"]) == 25
    assert set(trigger) == {"workflow_dispatch"}
    assert workflow["jobs"]["encode"]["environment"] == "production-signing"
    request = _step("Resolve retired source metadata migration request")
    assert request["env"]["ATOMIC_SOURCE_JSON"] == "${{ inputs.source_bundle_json }}"
    assert request["env"]["RULESPEC_REF"] == "${{ inputs.rulespec_ref }}"
    assert set(EXCLUSIVE_INPUTS) | set(EXCLUSIVE_ARRAY_INPUTS) <= set(request["env"])
    migration = _step("Migrate retired source metadata")
    command = migration["run"]
    assert "axiom-encode-apply-signer run" in command
    assert "--scope apply_ed25519" in command
    assert "--key-env AXIOM_ENCODE_APPLY_SIGNING_KEY" in command
    assert "--expected-github-repository TheAxiomFoundation/axiom-encode" in command
    assert (
        "--allowed-workflow-ref TheAxiomFoundation/axiom-encode/.github/workflows/"
        "targeted-signed-reencode.yml@refs/heads/main"
    ) in command
    assert "--allowed-event-name workflow_dispatch" in command
    assert (
        "-- /opt/axiom-verification/axiom-encode migrate-retired-source-metadata"
        in command
    )
    assert '--plan "$RUNNER_TEMP/retired-source-metadata-plan.json"' in command
    assert set(migration["env"]) == {
        "PYTHONUNBUFFERED",
        "AXIOM_ENCODE_APPLY_SIGNING_KEY",
        "RULESPEC_CHECKOUT",
    }
    assert "--codex-auth" not in command
    assert "CODEX_HOME" not in command
    assert "OPENAI_API_KEY" not in command
    assert "SUPABASE" not in command
    assert "--dry-run" not in command


@pytest.mark.parametrize(
    "name",
    [
        "Validate atomic source inputs",
        "Verify existing signed imports",
        "Encode, review, validate, and apply",
        "Verify existing signed import integrity",
        "Package exact generated changes",
        "Commit reviewed lane changes locally",
        "Push lane branch and open draft pull request",
    ],
)
def test_migration_cannot_reach_model_or_model_publication_steps(name: str) -> None:
    assert (
        "steps.retired_source_metadata_request.outputs.enabled != 'true'"
        in _step(name)["if"]
    )


def test_guard_runs_before_receipt_packaging_and_commit_and_publish_recheck() -> None:
    names = [step.get("name") for step in _workflow()["jobs"]["encode"]["steps"]]
    assert names.index("Migrate retired source metadata") < names.index(
        "Verify generated provenance"
    )
    assert names.index("Verify generated provenance") < names.index(
        "Package retired source metadata migration changes"
    )
    assert "if" not in _step("Verify generated provenance")
    assert "--expected-encoder-checkout" in _step("Verify generated provenance")["run"]
    package = _step("Package retired source metadata migration changes")["run"]
    assert "retired-source-metadata-changes.json" in package
    assert "retired-source-metadata-receipt.json" in package
    assert "retired-source-metadata-migration-artifact/v1" in package
    commit = _step("Commit retired source metadata migration locally")["run"]
    assert '"$helper" verify' in commit
    assert '"$helper" stage' in commit
    assert '"$helper" verify-committed' in commit
    publish = _step(
        "Push retired source metadata migration branch and open draft pull request"
    )["run"]
    assert "verify-committed" in publish
    assert "draft: true" in publish
    assert "--input" in publish
    assert ".base.ref == $branch and .base.sha == $sha" in publish
    assert "do not assert source fidelity" in publish


@pytest.mark.parametrize(
    "name",
    [
        "Package retired source metadata migration changes",
        "Commit retired source metadata migration locally",
        "Push retired source metadata migration branch and open draft pull request",
    ],
)
def test_publication_refuses_without_trusted_interpreter(name: str) -> None:
    command = _step(name)["run"]
    assert "workflow_python=(/opt/axiom-verification/python/bin/python -I)" in command
    assert 'test -x "${workflow_python[0]}"' in command
    assert "AXIOM_TEST_PYTHON" not in command


def test_resolver_admits_only_exact_base_bound_plan() -> None:
    plan = _plan()
    assert resolve_request(json.dumps(plan), base_ref="a" * 40, environment={}) == plan
    assert resolve_request("[]", base_ref="a" * 40, environment={}) is None
    with pytest.raises(ValueError, match="base_commit"):
        resolve_request(json.dumps(plan), base_ref="b" * 40, environment={})
    with pytest.raises(ValueError):
        resolve_request(
            json.dumps(dict(plan, extra="overbroad")), base_ref="a" * 40, environment={}
        )
    with pytest.raises(ValueError, match="duplicate"):
        resolve_request(
            '{"schema_version":"' + PLAN_SCHEMA + '","schema_version":"other"}',
            base_ref="a" * 40,
            environment={},
        )


@pytest.mark.parametrize("name", EXCLUSIVE_INPUTS + EXCLUSIVE_ARRAY_INPUTS)
def test_resolver_refuses_mixed_transaction_modes(name: str) -> None:
    with pytest.raises(ValueError, match=name):
        resolve_request(
            json.dumps(_plan()),
            base_ref="a" * 40,
            environment={
                name: '["mixed"]' if name in EXCLUSIVE_ARRAY_INPUTS else "mixed"
            },
        )


def test_model_attempt_budget_cannot_block_deterministic_lane() -> None:
    assert is_retired_source_metadata_plan(json.dumps(_plan()))
    assert not is_retired_source_metadata_plan("[]")
    assert not is_retired_source_metadata_plan('{"schema_version":"other"}')
    assert not is_retired_source_metadata_plan("{")
    workflow = _workflow()
    attempt = next(
        step
        for step in workflow["jobs"]["attempt_budget"]["steps"]
        if step.get("name") == "Enforce failed-attempt budget"
    )
    assert attempt["env"]["SOURCE_BUNDLE_JSON"] == "${{ inputs.source_bundle_json }}"


@pytest.fixture
def migrated(tmp_path: Path):
    repo = tmp_path / "rulespec-us"
    primary = repo / PRIMARY
    primary.parent.mkdir(parents=True)
    before = (
        b"module:\n  source_verification:\n"
        b"    corpus_citation_path: us/statute/26/1\n"
        b"    values:\n      old: retained in receipt\n"
        b"rules: []\n"
    )
    primary.write_bytes(before)
    primary.chmod(0o644)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "base",
    )
    base = _git(repo, "rev-parse", "HEAD").strip()
    tree = _git(repo, "rev-parse", "HEAD^{tree}").strip()
    payload = _plan(base)
    request = tmp_path / "plan.json"
    request.write_text(json.dumps(payload))
    migration = build_migration(
        load_plan_bytes(request.read_bytes()),
        base_tree=tree,
        base_files={PRIMARY: before},
    )
    for item in migration.files:
        (repo / item.path).write_bytes(item.after)
    receipt = repo / migration.receipt_relative
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes(migration.receipt_bytes)
    receipt.chmod(0o644)
    manifest = repo / ".axiom/encoding-manifests" / Path(PRIMARY).with_suffix(".json")
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "tool": MIGRATION_TOOL,
                "applied_files": [
                    {
                        "path": PRIMARY,
                        "sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
                    }
                ],
                "retired_source_metadata": {
                    "receipt_path": migration.receipt_relative.as_posix(),
                    "receipt_sha256": migration.receipt_sha256,
                },
            }
        )
    )
    manifest.chmod(0o644)
    return repo, base, request, migration


def test_inventory_and_commit_bind_exact_postimages(migrated, tmp_path: Path) -> None:
    repo, base, request, migration = migrated
    changes = migration_changes(repo, base, request=request)
    assert changes["receipt_sha256"] == migration.receipt_sha256
    assert {item["path"] for item in changes["changes"]} == {
        PRIMARY,
        migration.receipt_relative.as_posix(),
        ".axiom/encoding-manifests/us/statutes/26/1.json",
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(changes))
    stage_inventory(repo, request=request, inventory=inventory)
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "migration",
    )
    verify_committed_inventory(repo, request=request, inventory=inventory)


def test_packaging_refuses_extra_untracked_file(migrated) -> None:
    repo, base, request, _migration = migrated
    (repo / "stray.txt").write_text("unrelated")
    with pytest.raises(ValueError, match="unexpected=.*stray.txt"):
        migration_changes(repo, base, request=request)


def test_packaging_refuses_postimage_tampering(migrated) -> None:
    repo, base, request, _migration = migrated
    with (repo / PRIMARY).open("ab") as stream:
        stream.write(b"# drift\n")
    with pytest.raises(ValueError, match="postimage"):
        migration_changes(repo, base, request=request)


def test_packaging_refuses_another_plan(migrated) -> None:
    repo, base, request, _migration = migrated
    request.write_text(json.dumps(dict(_plan(base), modules=["us/statutes/26/2.yaml"])))
    with pytest.raises(ValueError, match="dispatched plan"):
        migration_changes(repo, base, request=request)


def test_stage_refuses_changes_after_packaging(migrated, tmp_path: Path) -> None:
    repo, base, request, _migration = migrated
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(migration_changes(repo, base, request=request)))
    (repo / "stray.txt").write_text("drift")
    with pytest.raises(ValueError, match="unexpected"):
        stage_inventory(repo, request=request, inventory=inventory)
    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_stage_refuses_packaged_manifest_drift(migrated, tmp_path: Path) -> None:
    repo, base, request, _migration = migrated
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(migration_changes(repo, base, request=request)))
    manifest = repo / ".axiom/encoding-manifests/us/statutes/26/1.json"
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after packaging"):
        verify_inventory(repo, request=request, inventory=inventory)


def test_workflow_resolver_executes_without_model_credentials(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/prepare_retired_source_metadata.py"),
            "resolve-request",
            json.dumps(_plan()),
            "--base-ref",
            "a" * 40,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": str(ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == _plan()

"""Retired apply roots admit frozen artifacts through each persisted reader."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode import cli
from axiom_encode.harness.evals import resolve_corpus_source_unit
from axiom_encode.judges import regeneration
from tests.environment_assertions import Names
from tests.signing_broker_fixtures import SigningBrokerFixture
from tests.test_cli import (
    TEST_APPLY_SIGNING_BROKER,
    TEST_PINNED_ENCODER_IDENTITY,
    _bind_test_corpus_release,
    _signed_manifest_payload,
)
from tests.test_cli import TestGuardGenerated as _GuardFixtures

FROZEN_PRODUCER_IDENTITY = {
    "repository": cli.APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    "commit": "c" * 40,
    "version": "0.2.1000",
}


def _set_producer_identity(
    payload: dict[str, object], identity: dict[str, str]
) -> None:
    """Keep both signed producer declarations consistent for model fixtures."""

    payload["axiom_encode_version"] = identity["version"]
    payload["axiom_encode_git"].update(
        commit=identity["commit"], version=identity["version"]
    )
    execution = payload.get("validation_execution")
    if isinstance(execution, dict):
        execution["axiom_encode"].update(identity)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
        env=Names(
            {
                **os.environ,
                "GIT_AUTHOR_NAME": "Synthetic fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.com",
                "GIT_COMMITTER_NAME": "Synthetic fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.com",
            }
        ),
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


class _RotatedBroker(SigningBrokerFixture):
    def __init__(self, inventory: dict[str, object]):
        active = Ed25519PrivateKey.generate()
        private_raw = active.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        public_raw = active.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        super().__init__(
            apply_private_key=base64.b64encode(private_raw).decode("ascii"),
            apply_public_key=base64.b64encode(public_raw).decode("ascii"),
            retired_apply_public_keys=(
                base64.b64encode(TEST_APPLY_SIGNING_BROKER.apply_public_key_raw).decode(
                    "ascii"
                ),
            ),
            retired_apply_inventory=inventory,
        )


@dataclass
class _FrozenRepo:
    repo: Path
    corpus: Path
    release: object
    rule: Path
    manifest: Path
    payload: dict[str, object]
    manifest_bytes: bytes
    base: str
    old_head: str
    cutoff: str
    broker: _RotatedBroker

    @property
    def rule_relative(self) -> str:
        return self.rule.relative_to(self.repo).as_posix()

    @property
    def manifest_relative(self) -> str:
        return self.manifest.relative_to(self.repo).as_posix()


@pytest.fixture
def frozen_repo(tmp_path: Path, monkeypatch) -> _FrozenRepo:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-us.git",
    )
    release = _bind_test_corpus_release(
        repo, tmp_path / "axiom-corpus", body="frozen authoritative test source\n"
    )
    base = _commit(repo, "pre-generation trusted toolchain")
    attestation = resolve_corpus_source_unit(
        "us/statute/26/1", release
    ).source_attestation
    attestation["generation_input_sha256"] = attestation["resolved_text_sha256"]
    attestation["rulespec_root"] = "rulespec-us/us"
    rule = repo / "us/statutes/26/1.yaml"
    rule.parent.mkdir(parents=True)
    rule.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        "    corpus_citation_path: us/statute/26/1\n"
        f"    source_sha256: {attestation['source_sha256']}\n"
        "rules: []\n"
    )
    rule.chmod(0o644)
    manifest = repo / cli._applied_encoding_manifest_path(rule.relative_to(repo))
    manifest.parent.mkdir(parents=True)
    payload = _signed_manifest_payload(
        {
            "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "backend": "codex",
            "citation": "us/statute/26/1",
            "source_attestation": attestation,
            "applied_files": [
                {
                    "path": rule.relative_to(repo).as_posix(),
                    "sha256": hashlib.sha256(rule.read_bytes()).hexdigest(),
                }
            ],
        }
    )
    _set_producer_identity(payload, FROZEN_PRODUCER_IDENTITY)
    _signed_manifest_payload(payload)
    manifest.write_text(json.dumps(payload) + "\n")
    manifest.chmod(0o644)
    old_head = _commit(repo, "frozen generated artifact")
    (repo / "README.md").write_text("Frozen main cutoff.\n")
    cutoff = _commit(repo, "owner frozen cutoff")
    manifest_bytes = manifest.read_bytes()
    inventory = {
        "schema": "axiom-encode/retired-apply-inventory/v1",
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": cutoff,
                "artifacts": [
                    {
                        "path": manifest.relative_to(repo).as_posix(),
                        "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                        "covered_files": copy.deepcopy(payload["applied_files"]),
                    }
                ],
            }
        ],
    }
    broker = _RotatedBroker(inventory)
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: broker)
    monkeypatch.setattr(cli, "get_signing_broker", lambda **_kwargs: broker)
    monkeypatch.setattr(
        cli,
        "_read_only_guard_encoder_execution_identity",
        lambda *_args: dict(TEST_PINNED_ENCODER_IDENTITY),
    )
    monkeypatch.setattr(
        cli,
        "_current_guard_encoder_execution_identity",
        lambda: dict(TEST_PINNED_ENCODER_IDENTITY),
    )
    return _FrozenRepo(
        repo,
        release.root,
        release,
        rule,
        manifest,
        payload,
        manifest_bytes,
        base,
        old_head,
        cutoff,
        broker,
    )


READERS = ("guard-all", "golden", "imports", "refresh", "census")


def _reader_accepts(fixture: _FrozenRepo, reader: str) -> bool:
    if reader == "guard-all":
        return not cli.guard_generated_change_issues(
            fixture.repo, corpus_path=fixture.corpus, all_files=True
        )
    if reader == "golden":
        try:
            regeneration.load_replay_manifest(
                fixture.repo, PurePosixPath(fixture.rule_relative)
            )
        except regeneration.RegenerationInputError:
            return False
        return True
    if reader == "imports":
        try:
            cli.signed_import_inventory(
                fixture.repo,
                corpus_path=fixture.corpus,
                base_ref=_git(fixture.repo, "rev-parse", "HEAD"),
                rulespec_paths=(fixture.rule_relative,),
            )
        except ValueError:
            return False
        return True
    if reader == "refresh":
        try:
            cli._historical_manifest_refresh_inputs(
                fixture.repo, fixture.rule_relative, signing_broker=fixture.broker
            )
        except ValueError:
            return False
        return True
    if reader == "census":
        census = cli._manifest_census(
            fixture.repo, roots=("statutes",), local_corpus_release=fixture.release
        )
        return census["encoder"] == 1 and census["unmanifested"] == 0
    raise AssertionError(reader)


@pytest.mark.parametrize("reader", READERS)
def test_frozen_old_artifact_passes_each_persisted_reader(frozen_repo, reader):
    assert (
        frozen_repo.payload["axiom_encode_version"]
        != TEST_PINNED_ENCODER_IDENTITY["version"]
    )
    assert (
        frozen_repo.payload["axiom_encode_git"]["commit"]
        != TEST_PINNED_ENCODER_IDENTITY["commit"]
    )
    assert _reader_accepts(frozen_repo, reader)


@pytest.mark.parametrize("reader", READERS)
def test_active_root_artifact_passes_each_reader(frozen_repo, reader):
    _set_producer_identity(frozen_repo.payload, dict(TEST_PINNED_ENCODER_IDENTITY))
    cli._sign_applied_encoding_manifest(frozen_repo.payload, frozen_repo.broker)
    frozen_repo.manifest.write_text(json.dumps(frozen_repo.payload) + "\n")
    _commit(frozen_repo.repo, "active root refresh")
    assert _reader_accepts(frozen_repo, reader)


@pytest.mark.parametrize("reader", ("guard-all", "golden", "census"))
def test_active_root_does_not_admit_older_producer_from_retired_inventory(
    frozen_repo, reader
):
    cli._sign_applied_encoding_manifest(frozen_repo.payload, frozen_repo.broker)
    frozen_repo.manifest.write_text(json.dumps(frozen_repo.payload) + "\n")
    _commit(frozen_repo.repo, "active root must retain current producer admission")
    assert not _reader_accepts(frozen_repo, reader)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize(
    "mutation",
    (
        "manifest-bytes",
        "generated-bytes",
        "resigned",
        "relocated",
        "new",
        "superseded",
        "repository",
    ),
)
def test_each_reader_rejects_old_artifact_outside_frozen_inventory(
    frozen_repo, reader, mutation
):
    fixture = frozen_repo
    if mutation == "manifest-bytes":
        fixture.manifest.write_bytes(fixture.manifest_bytes + b"\n")
    elif mutation == "generated-bytes":
        fixture.rule.write_bytes(fixture.rule.read_bytes() + b"# changed\n")
    elif mutation == "resigned":
        fixture.payload["generated_at"] = "2026-10-14T01:00:00+00:00"
        cli._sign_applied_encoding_manifest(fixture.payload, TEST_APPLY_SIGNING_BROKER)
        fixture.manifest.write_text(json.dumps(fixture.payload) + "\n")
    elif mutation in {"relocated", "new"}:
        old_rule, old_manifest = fixture.rule, fixture.manifest
        fixture.rule = old_rule.with_name("2.yaml")
        fixture.manifest = old_manifest.with_name("2.json")
        fixture.rule.write_bytes(old_rule.read_bytes())
        fixture.rule.chmod(0o644)
        fixture.payload["applied_files"][0]["path"] = fixture.rule_relative
        cli._sign_applied_encoding_manifest(fixture.payload, TEST_APPLY_SIGNING_BROKER)
        fixture.manifest.write_text(json.dumps(fixture.payload) + "\n")
        fixture.manifest.chmod(0o644)
        if mutation == "relocated":
            old_rule.unlink()
            old_manifest.unlink()
    elif mutation == "superseded":
        active = copy.deepcopy(fixture.payload)
        _set_producer_identity(active, dict(TEST_PINNED_ENCODER_IDENTITY))
        cli._sign_applied_encoding_manifest(active, fixture.broker)
        fixture.manifest.write_text(json.dumps(active) + "\n")
        _commit(fixture.repo, "active root supersedes old manifest")
        fixture.manifest.write_bytes(fixture.manifest_bytes)
    elif mutation == "repository":
        _git(
            fixture.repo,
            "remote",
            "set-url",
            "origin",
            "https://github.com/Untrusted/rulespec-us.git",
        )
    if mutation != "repository":
        _commit(fixture.repo, "attempt historical artifact replay")
    assert not _reader_accepts(fixture, reader)


def test_ordinary_diff_requires_active_root_even_for_cutoff_commit(frozen_repo):
    fixture = frozen_repo
    _git(fixture.repo, "checkout", "--detach", fixture.old_head)
    issues = cli.guard_generated_change_issues(
        fixture.repo,
        corpus_path=fixture.corpus,
        base_ref=fixture.base,
        head_ref=fixture.old_head,
    )
    assert any(
        "unknown encoder apply manifest signing key" in issue for issue in issues
    )


@pytest.mark.parametrize("head", ("cutoff", "ancestor"))
def test_explicit_retrospective_diff_accepts_only_frozen_history(frozen_repo, head):
    fixture = frozen_repo
    selected = fixture.cutoff if head == "cutoff" else fixture.old_head
    _git(fixture.repo, "checkout", "--detach", selected)
    issues = cli.guard_generated_change_issues(
        fixture.repo,
        corpus_path=fixture.corpus,
        base_ref=fixture.base,
        head_ref=selected,
        retrospective=True,
    )
    assert issues == []


def test_retrospective_diff_rejects_postcutoff_head(frozen_repo):
    fixture = frozen_repo
    (fixture.repo / "README.md").write_text("Post-rotation main.\n")
    head = _commit(fixture.repo, "post-cutoff main")
    assert cli.guard_generated_change_issues(
        fixture.repo,
        corpus_path=fixture.corpus,
        base_ref=fixture.base,
        head_ref=head,
        retrospective=True,
    )


def test_active_root_passes_ordinary_diff(frozen_repo):
    fixture = frozen_repo
    _set_producer_identity(fixture.payload, dict(TEST_PINNED_ENCODER_IDENTITY))
    cli._sign_applied_encoding_manifest(fixture.payload, fixture.broker)
    fixture.manifest.write_text(json.dumps(fixture.payload) + "\n")
    fixture.rule.write_bytes(
        fixture.rule.read_bytes() + b"# active-root regeneration\n"
    )
    fixture.payload["applied_files"][0]["sha256"] = hashlib.sha256(
        fixture.rule.read_bytes()
    ).hexdigest()
    cli._sign_applied_encoding_manifest(fixture.payload, fixture.broker)
    fixture.manifest.write_text(json.dumps(fixture.payload) + "\n")
    _commit(fixture.repo, "active-root generated update")
    assert (
        cli.guard_generated_change_issues(
            fixture.repo, corpus_path=fixture.corpus, base_ref=fixture.cutoff
        )
        == []
    )


def _retirement_payload(
    fixture: _FrozenRepo, prior: dict[str, object]
) -> dict[str, object]:
    return _signed_manifest_payload(
        {
            "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "source_attestation": copy.deepcopy(prior["source_attestation"]),
            "retired_manifest": prior,
            "applied_files": [{"path": fixture.rule_relative, "deleted": True}],
        }
    )


@pytest.mark.parametrize("modified_nested", (False, True))
def test_active_retirement_requires_exact_frozen_nested_model(
    frozen_repo, modified_nested
):
    fixture = frozen_repo
    prior = copy.deepcopy(fixture.payload)
    if modified_nested:
        prior["generated_at"] = "2026-10-14T01:00:00+00:00"
        cli._sign_applied_encoding_manifest(prior, TEST_APPLY_SIGNING_BROKER)
    outer = _retirement_payload(fixture, prior)
    cli._sign_applied_encoding_manifest(outer, fixture.broker)
    fixture.manifest.write_text(json.dumps(outer) + "\n")
    fixture.rule.unlink()
    _commit(fixture.repo, "active root retirement preserving historical model")
    _payload, _prefix, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            fixture.repo,
            fixture.manifest_relative,
            signing_broker=fixture.broker,
            historical_verification=True,
            local_corpus_release=fixture.release,
        )
    )
    assert bool(issues) is modified_nested


def test_frozen_retirement_binds_deleted_generated_bytes_to_ancestor(frozen_repo):
    fixture = frozen_repo
    outer = _retirement_payload(fixture, copy.deepcopy(fixture.payload))
    _set_producer_identity(outer, FROZEN_PRODUCER_IDENTITY)
    _signed_manifest_payload(outer)
    fixture.manifest.write_text(json.dumps(outer) + "\n")
    fixture.rule.unlink()
    cutoff = _commit(fixture.repo, "owner freezes historical retirement")
    inventory_repo = fixture.broker.retired_apply_inventory["repositories"][0]
    inventory_repo["cutoff_commit"] = cutoff
    artifact = inventory_repo["artifacts"][0]
    artifact["sha256"] = hashlib.sha256(fixture.manifest.read_bytes()).hexdigest()
    artifact["covered_files"][0]["source_commit"] = fixture.old_head
    _payload, _prefix, _digest, issues = (
        cli._load_verified_applied_encoding_manifest_payload(
            fixture.repo,
            fixture.manifest_relative,
            signing_broker=fixture.broker,
            historical_verification=True,
            local_corpus_release=fixture.release,
        )
    )
    assert issues == []
    artifact["covered_files"][0]["sha256"] = "0" * 64
    assert cli._load_verified_applied_encoding_manifest_payload(
        fixture.repo,
        fixture.manifest_relative,
        signing_broker=fixture.broker,
        historical_verification=True,
        local_corpus_release=fixture.release,
    )[3]


def _recursive_applied_files(payload: object):
    if isinstance(payload, dict):
        applied = payload.get("applied_files")
        if isinstance(applied, list):
            for item in applied:
                if isinstance(item, dict) and isinstance(item.get("sha256"), str):
                    yield item
        for child in payload.values():
            yield from _recursive_applied_files(child)
    elif isinstance(payload, list):
        for child in payload:
            yield from _recursive_applied_files(child)


def _freeze_artifacts(repo: Path, cutoff: str, prior: str) -> dict[str, object]:
    artifacts = []
    for manifest in sorted((repo / ".axiom").rglob("*.json")):
        payload = json.loads(manifest.read_bytes())
        if not isinstance(payload, dict) or "signature" not in payload:
            continue
        covered = {}
        for entry in _recursive_applied_files(payload):
            path, digest = entry["path"], entry["sha256"]
            current = repo / path
            source = (
                None
                if current.is_file()
                and hashlib.sha256(current.read_bytes()).hexdigest() == digest
                else prior
            )
            item = {"path": path, "sha256": digest}
            if source is not None:
                item["source_commit"] = source
            covered[(path, source)] = item
        artifacts.append(
            {
                "path": manifest.relative_to(repo).as_posix(),
                "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "covered_files": list(covered.values()),
            }
        )
    return {
        "schema": "axiom-encode/retired-apply-inventory/v1",
        "repositories": [
            {
                "repository": "TheAxiomFoundation/rulespec-us",
                "cutoff_commit": cutoff,
                "artifacts": artifacts,
            }
        ],
    }


def test_frozen_migration_receipt_and_nested_model_verify_after_rotation(
    frozen_repo, tmp_path, monkeypatch
):
    fixture = frozen_repo
    migrated_repo = tmp_path / "migrated" / "rulespec-us"
    migrated_repo.mkdir(parents=True)
    builders = _GuardFixtures()
    builders.corpus_release = fixture.release
    builders.corpus_path = fixture.corpus
    builders.source_attestation = copy.deepcopy(fixture.payload["source_attestation"])
    repo, plan, prior = builders._path_migration_repo(migrated_repo)
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.relative_to(repo).parts:
            path.chmod(0o644)
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-us.git",
    )
    with monkeypatch.context() as preparation:
        builders._enable_path_migration(preparation)
        preparation.setattr(
            cli,
            "_applied_encoding_manifest_verifier",
            lambda: TEST_APPLY_SIGNING_BROKER,
        )
        cli.cmd_migrate_rulespec_paths(
            SimpleNamespace(
                policy_repo_path=repo, corpus_path=fixture.corpus, plan=plan
            )
        )
    cutoff = _commit(repo, "freeze migrated artifacts")
    broker = _RotatedBroker(_freeze_artifacts(repo, cutoff, prior))
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: broker)
    assert (
        cli.guard_generated_change_issues(
            repo, corpus_path=fixture.corpus, all_files=True
        )
        == []
    )
    receipt = next(
        (repo / cli.APPLIED_ENCODING_PATH_MIGRATION_RECEIPT_DIR).glob("*.json")
    )
    receipt.write_bytes(receipt.read_bytes() + b"\n")
    _commit(repo, "modify historical migration receipt bytes")
    assert cli.guard_generated_change_issues(
        repo, corpus_path=fixture.corpus, all_files=True
    )


@pytest.mark.parametrize(
    "mutation", (None, "receipt-bytes", "receipt-path", "owner-bytes")
)
def test_retained_successor_receipt_admission_binds_frozen_envelopes(
    frozen_repo, monkeypatch, mutation
):
    """Exercise this reader's receipt/owner signatures before semantic replay.

    The full cascade semantics remain covered by the existing apply cascade
    tests; this fixture isolates the additional frozen signature boundary.
    """

    fixture = frozen_repo
    owner_path = Path(".axiom/encoding-manifests/us/statutes/26/3.json")
    receipt_path = (
        cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_DIR / "synthetic.json"
    )
    successor_hash = hashlib.sha256(fixture.manifest_bytes).hexdigest()
    legacy_path = ".axiom/encoding-manifests/us/statutes/26:1.json"
    legacy_hash = "a" * 64
    replacement = {
        "model_manifest_path": owner_path.as_posix(),
        "retained_successors": [
            {
                "source": "us/statutes/26:1.yaml",
                "destination": fixture.rule_relative,
                "successor_files": copy.deepcopy(fixture.payload["applied_files"]),
                "legacy_manifest": {"path": legacy_path, "sha256": legacy_hash},
                "successor_manifest": {
                    "sha256": successor_hash,
                    "payload": copy.deepcopy(fixture.payload),
                },
            }
        ],
    }
    receipt = dict.fromkeys(cli._LEGACY_REPLACEMENT_RECEIPT_FIELDS)
    receipt.update(
        {
            "schema_version": cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_RECEIPT_SCHEMA,
            "tool": cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL,
            "replacement": replacement,
        }
    )
    cli._sign_applied_encoding_manifest(receipt, TEST_APPLY_SIGNING_BROKER)
    receipt_file = fixture.repo / receipt_path
    receipt_file.parent.mkdir(parents=True)
    receipt_file.write_text(json.dumps(receipt) + "\n")
    receipt_file.chmod(0o644)
    owner = dict.fromkeys(cli._LEGACY_REPLACEMENT_APPLY_MANIFEST_FIELDS)
    owner.update(
        {
            "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": cli.APPLIED_ENCODING_LEGACY_REPLACEMENT_TOOL,
            "replacement": replacement,
            "applied_files": copy.deepcopy(fixture.payload["applied_files"]),
        }
    )
    cli._sign_applied_encoding_manifest(owner, TEST_APPLY_SIGNING_BROKER)
    owner_file = fixture.repo / owner_path
    owner_file.write_text(json.dumps(owner) + "\n")
    owner_file.chmod(0o644)
    cutoff = _commit(fixture.repo, "freeze retained successor receipt and owner")
    broker = _RotatedBroker(_freeze_artifacts(fixture.repo, cutoff, fixture.old_head))
    binding = {
        "receipt_path": receipt_path.as_posix(),
        "receipt_sha256": hashlib.sha256(receipt_file.read_bytes()).hexdigest(),
        "source": "us/statutes/26:1.yaml",
        "destination": fixture.rule_relative,
        "legacy_manifest_path": legacy_path,
        "legacy_manifest_sha256": legacy_hash,
        "successor_manifest_sha256": successor_hash,
    }
    retained = {
        "tool": cli.APPLIED_ENCODING_LEGACY_RETAINED_SUCCESSOR_TOOL,
        "legacy_migration": binding,
        "retained_successor_manifest": fixture.payload,
        "applied_files": fixture.payload["applied_files"],
    }
    if mutation == "receipt-bytes":
        receipt_file.write_bytes(receipt_file.read_bytes() + b"\n")
        binding["receipt_sha256"] = hashlib.sha256(
            receipt_file.read_bytes()
        ).hexdigest()
    elif mutation == "receipt-path":
        relocated = receipt_file.with_name("relocated.json")
        relocated.write_bytes(receipt_file.read_bytes())
        relocated.chmod(0o644)
        binding["receipt_path"] = relocated.relative_to(fixture.repo).as_posix()
    elif mutation == "owner-bytes":
        owner_file.write_bytes(owner_file.read_bytes() + b"\n")
    if mutation is not None:
        _commit(fixture.repo, "attempt retained successor envelope replay")
    semantic_calls = []
    monkeypatch.setattr(
        cli,
        "_legacy_replacement_manifest_issues",
        lambda *args, **kwargs: semantic_calls.append(args[0]) or [],
    )
    issues = cli._legacy_retained_successor_manifest_issues(
        retained,
        repo_path=fixture.repo,
        manifest_label=fixture.manifest_relative,
        signing_broker=broker,
        expected_waiver_set_sha256=fixture.payload["validation_waiver_set_sha256"],
        local_corpus_release=fixture.release,
    )
    assert bool(issues) is (mutation is not None)
    assert len(semantic_calls) == (1 if mutation is None else 0)

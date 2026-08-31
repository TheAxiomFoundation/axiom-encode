"""Focused transaction tests for the cleanup-only CLI writer."""

from __future__ import annotations

import hashlib
import subprocess
from argparse import Namespace
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import axiom_encode.cli as cli
from axiom_encode.legacy_cleanup import (
    LEGACY_CLEANUP_RECEIPT_DIR,
    LEGACY_CLEANUP_VALIDATION_CHECKS,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    parse_receipt_bytes,
)
from axiom_encode.legacy_cleanup_signing import (
    verify_legacy_cleanup_receipt_signature,
)
from axiom_encode.signing_broker import canonical_signing_message

PRIMARY = Path("be/statutes/legacy.yaml")
COMPANION = Path("be/statutes/legacy.test.yaml")
SURVIVOR = Path("be/statutes/survivor.yaml")
SURVIVOR_COMPANION = Path("be/statutes/survivor.test.yaml")
WAIVER = b"version: 1\nwaivers: []\n"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write(repo: Path, relative: Path, raw: bytes) -> None:
    target = repo / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)


def _base_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "cleanup@example.com")
    _git(repo, "config", "user.name", "Cleanup Test")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-be.git",
    )
    waiver_sha256 = hashlib.sha256(WAIVER).hexdigest()
    _write(
        repo,
        Path(".axiom/toolchain.toml"),
        (
            "[toolchain]\n"
            'axiom_corpus_release = "cleanup-test"\n'
            f'axiom_corpus_release_content_sha256 = "{"c" * 64}"\n'
            f'validation_waiver_set_sha256 = "{waiver_sha256}"\n'
        ).encode(),
    )
    _write(repo, Path("known-validation-gaps.yaml"), WAIVER)
    _write(repo, PRIMARY, b"module:\n  name: legacy\n")
    _write(repo, COMPANION, b"tests:\n  - name: legacy\n")
    _write(repo, SURVIVOR, b"module:\n  name: survivor\n")
    _write(repo, SURVIVOR_COMPANION, b"tests:\n  - name: survivor\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "protected base")
    return repo, _git(repo, "rev-parse", "HEAD")


class _Broker:
    def __init__(self, *, fail: bool = False) -> None:
        self.private_key = Ed25519PrivateKey.from_private_bytes(b"\x41" * 32)
        self.apply_public_key_raw = self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        self.corpus_release_public_keys_raw = (b"\x42" * 32,)
        self.fail = fail

    def apply_ed25519_sign(self, payload: bytes) -> bytes:
        if self.fail:
            raise RuntimeError("injected cleanup signer failure")
        return self.private_key.sign(
            canonical_signing_message("apply_ed25519", payload)
        )


def _validation(plan) -> dict[str, object]:
    return {
        "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
        "status": "passed",
        "engine_execution": True,
        "projected_post_deletion_tree": plan.projected_post_deletion_tree,
        "checks": [
            {
                "name": name,
                "command": ["executed", name],
                "target_count": 1,
                "target_list_sha256": hashlib.sha256(
                    f"targets:{name}".encode()
                ).hexdigest(),
                "exit_code": 0,
                "output_sha256": hashlib.sha256(
                    f"output:{name}".encode()
                ).hexdigest(),
            }
            for name in LEGACY_CLEANUP_VALIDATION_CHECKS
        ],
    }


def _args(repo: Path, base: str, tmp_path: Path) -> Namespace:
    return Namespace(
        policy_repo_path=repo,
        base_ref=base,
        paths=[PRIMARY.as_posix()],
        reason="Remove one exact unmanifested legacy aggregation",
        corpus_path=tmp_path / "axiom-corpus",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        expected_encoder_checkout=tmp_path / "axiom-encode",
        rulespec_dependency_root=[],
    )


def _install_test_contract(monkeypatch, repo: Path, broker: _Broker) -> None:
    release = SimpleNamespace(
        root="/portable/corpus",
        name="cleanup-test",
        content_sha256="c" * 64,
        selector_sha256="d" * 64,
    )

    def binding(*, plan, **_kwargs):
        return (
            {
                "axiom_encode": {
                    "repository": "github.com/TheAxiomFoundation/axiom-encode",
                    "object_format": "sha1",
                    "commit": "a" * 40,
                    "version": "0.2.test",
                },
                "axiom_rules_engine": {
                    "repository": (
                        "github.com/TheAxiomFoundation/axiom-rules-engine"
                    ),
                    "object_format": "sha1",
                    "commit": "b" * 40,
                },
                "rulespec_dependencies": [],
                "corpus_release": {
                    "name": release.name,
                    "content_sha256": release.content_sha256,
                    "selector_sha256": release.selector_sha256,
                },
                "validation_waiver_set_sha256": (
                    plan.toolchain_values["validation_waiver_set_sha256"]
                ),
                "base_files": plan.base_files,
            },
            release,
            (),
        )

    monkeypatch.setattr(
        cli,
        "_resolve_canonical_rulespec_checkout",
        lambda *_args, **_kwargs: repo,
    )
    monkeypatch.setattr(cli, "_legacy_cleanup_toolchain_binding", binding)
    monkeypatch.setattr(
        cli,
        "execute_projected_validation",
        lambda _repo, plan, **_kwargs: _validation(plan),
    )
    monkeypatch.setattr(
        cli,
        "_require_applied_encoding_manifest_signer",
        lambda: broker,
    )
    monkeypatch.setattr(cli, "get_signing_broker", lambda: broker)


def test_cleanup_cli_installs_receipt_first_and_exact_deletions(
    tmp_path,
    monkeypatch,
):
    repo, base = _base_repo(tmp_path)
    broker = _Broker()
    _install_test_contract(monkeypatch, repo, broker)
    original_install = cli._install_apply_transaction
    installed_order: list[str] = []

    def capture(files, **kwargs):
        installed_order.extend(
            target.relative_to(repo).as_posix() for target, _raw in files
        )
        return original_install(files, **kwargs)

    monkeypatch.setattr(cli, "_install_apply_transaction", capture)

    cli.cmd_cleanup_unmanifested_legacy(_args(repo, base, tmp_path))

    receipts = list((repo / LEGACY_CLEANUP_RECEIPT_DIR).glob("*.json"))
    assert len(receipts) == 1
    receipt_relative = receipts[0].relative_to(repo)
    assert installed_order == [
        receipt_relative.as_posix(),
        PRIMARY.as_posix(),
        COMPANION.as_posix(),
    ]
    payload = parse_receipt_bytes(
        receipts[0].read_bytes(),
        expected_path=receipt_relative,
    )
    verify_legacy_cleanup_receipt_signature(payload, broker)
    assert payload["repository"]["base_commit"] == base
    assert "head_commit" not in payload["repository"]
    assert "final_tree" not in payload["repository"]
    assert not (repo / PRIMARY).exists()
    assert not (repo / COMPANION).exists()
    assert (repo / SURVIVOR).is_file()
    assert _git(repo, "diff", "--name-status", "HEAD") == (
        f"D\t{COMPANION.as_posix()}\nD\t{PRIMARY.as_posix()}"
    )


def test_cleanup_signer_failure_leaves_complete_preimage(tmp_path, monkeypatch):
    repo, base = _base_repo(tmp_path)
    _install_test_contract(monkeypatch, repo, _Broker(fail=True))

    with pytest.raises(RuntimeError, match="signer failure"):
        cli.cmd_cleanup_unmanifested_legacy(_args(repo, base, tmp_path))

    assert (repo / PRIMARY).is_file()
    assert (repo / COMPANION).is_file()
    assert not (repo / LEGACY_CLEANUP_RECEIPT_DIR).exists()
    assert not _git(repo, "status", "--porcelain=v1")


def test_cleanup_postcheck_failure_rolls_back_receipt_and_both_files(
    tmp_path,
    monkeypatch,
):
    repo, base = _base_repo(tmp_path)
    _install_test_contract(monkeypatch, repo, _Broker())
    monkeypatch.setattr(
        cli,
        "_legacy_cleanup_post_install_issues",
        lambda *_args, **_kwargs: ["injected postcheck failure"],
    )

    with pytest.raises(RuntimeError, match="postcheck failure"):
        cli.cmd_cleanup_unmanifested_legacy(_args(repo, base, tmp_path))

    assert (repo / PRIMARY).read_bytes() == b"module:\n  name: legacy\n"
    assert (repo / COMPANION).read_bytes() == b"tests:\n  - name: legacy\n"
    assert not (repo / LEGACY_CLEANUP_RECEIPT_DIR).exists()
    assert not (repo / ".axiom/.apply-transaction").exists()
    assert not _git(repo, "status", "--porcelain=v1")


def test_cleanup_reproves_target_mutation_under_transaction_lock(
    tmp_path,
    monkeypatch,
):
    repo, base = _base_repo(tmp_path)
    _install_test_contract(monkeypatch, repo, _Broker())
    original_install = cli._install_apply_transaction

    def mutate_before_lock(files, **kwargs):
        (repo / PRIMARY).write_text("concurrent mutation\n", encoding="utf-8")
        return original_install(files, **kwargs)

    monkeypatch.setattr(cli, "_install_apply_transaction", mutate_before_lock)

    with pytest.raises(Exception, match="clean|changed|proof"):
        cli.cmd_cleanup_unmanifested_legacy(_args(repo, base, tmp_path))

    assert (repo / PRIMARY).read_text() == "concurrent mutation\n"
    assert (repo / COMPANION).is_file()
    assert not (repo / LEGACY_CLEANUP_RECEIPT_DIR).exists()
    assert not (repo / ".axiom/.apply-transaction").exists()


def test_cleanup_receipt_signature_is_not_apply_manifest_replay(
    tmp_path,
    monkeypatch,
):
    repo, base = _base_repo(tmp_path)
    broker = _Broker()
    _install_test_contract(monkeypatch, repo, broker)
    cli.cmd_cleanup_unmanifested_legacy(_args(repo, base, tmp_path))
    receipt = next((repo / LEGACY_CLEANUP_RECEIPT_DIR).glob("*.json"))
    payload = parse_receipt_bytes(receipt.read_bytes())
    signature = payload["signature"]
    assert isinstance(signature, dict)
    assert signature["domain"].endswith("legacy-rulespec-deletion-receipt/v1")
    assert signature["value"] != b64encode(bytes(64)).decode("ascii")

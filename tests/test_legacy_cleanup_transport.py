"""Focused adversarial tests for cleanup receipt/deletion transport."""

from __future__ import annotations

import json
import os
import subprocess
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization

import axiom_encode.legacy_cleanup_transport as transport
from axiom_encode.legacy_cleanup import canonical_receipt_bytes
from axiom_encode.legacy_cleanup_guard import (
    verify_worktree_legacy_cleanup_transition,
)
from axiom_encode.legacy_cleanup_transport import (
    LEGACY_CLEANUP_INVENTORY_NAME,
    LegacyCleanupTransportError,
    build_transport_artifact,
    canonical_inventory_bytes,
    load_apply_trust_root,
    materialize_transport_artifact,
    parse_inventory_bytes,
)
from tests.test_legacy_cleanup_guard import (
    COMPANION,
    PRIMARY,
    PUBLIC_KEY,
    _base_repo,
    _delete_group,
    _git,
    _plan,
    _sign,
    _unsigned_payload,
    _write_receipt,
)


def _trust_roots(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/signing-trust-roots/v2",
                "apply_ed25519_public_key": b64encode(
                    PUBLIC_KEY.public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ).decode("ascii"),
                "eval_ed25519_public_key": b64encode(b"\x55" * 32).decode(
                    "ascii"
                ),
                "corpus_release_ed25519_public_key": b64encode(
                    b"\x56" * 32
                ).decode("ascii"),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o644)
    return path


def _source_cleanup(tmp_path: Path):
    source_parent = tmp_path / "source"
    source_parent.mkdir()
    repo, base = _base_repo(source_parent)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    receipt = _write_receipt(repo, payload)
    _delete_group(repo)
    return repo, base, payload, receipt


def _fresh_base_checkout(tmp_path: Path, source: Path, base: str) -> Path:
    parent = tmp_path / "publisher"
    parent.mkdir()
    checkout = parent / "rulespec-be"
    subprocess.run(
        ["git", "clone", "--no-local", str(source), str(checkout)],
        check=True,
        capture_output=True,
    )
    _git(
        checkout,
        "remote",
        "set-url",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-be.git",
    )
    _git(checkout, "checkout", "--detach", base)
    return checkout


def _built_artifact(tmp_path: Path):
    source, base, payload, receipt = _source_cleanup(tmp_path)
    artifact = tmp_path / "artifact"
    inventory = build_transport_artifact(
        source,
        base_ref=base,
        artifact_root=artifact,
        verifier=PUBLIC_KEY,
    )
    return source, base, payload, receipt, artifact, inventory


def test_transport_round_trip_preserves_exact_receipt_and_deletions(
    tmp_path: Path,
) -> None:
    source, base, payload, receipt, artifact, inventory = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)

    materialized = materialize_transport_artifact(
        checkout,
        base_ref=base,
        artifact_root=artifact,
        verifier=PUBLIC_KEY,
    )

    assert materialized == inventory
    assert (checkout / receipt).read_bytes() == canonical_receipt_bytes(payload)
    assert not (checkout / PRIMARY).exists()
    assert not (checkout / COMPANION).exists()
    result = verify_worktree_legacy_cleanup_transition(
        checkout,
        base_ref=base,
        verifier=PUBLIC_KEY,
        expected_toolchain=payload["toolchain"],
    )
    assert result.authorized
    assert result.authorized_paths == (receipt, PRIMARY, COMPANION)


def test_artifact_is_exactly_one_canonical_receipt_and_inventory(
    tmp_path: Path,
) -> None:
    _, _, _, receipt, artifact, inventory = _built_artifact(tmp_path)
    files = sorted(
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file()
    )

    assert files == [receipt.as_posix(), LEGACY_CLEANUP_INVENTORY_NAME]
    raw = (artifact / LEGACY_CLEANUP_INVENTORY_NAME).read_bytes()
    assert raw == canonical_inventory_bytes(inventory)
    assert parse_inventory_bytes(raw) == inventory


def test_materialize_rejects_an_extra_artifact_file_before_mutation(
    tmp_path: Path,
) -> None:
    source, base, _, receipt, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    (artifact / "extra.txt").write_text("not authorized\n", encoding="utf-8")

    with pytest.raises(LegacyCleanupTransportError, match="exactly one"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()
    assert not (checkout / receipt).exists()


def test_materialize_rejects_receipt_mutation_before_deletion(tmp_path: Path) -> None:
    source, base, _, receipt, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    target = artifact / receipt
    target.write_bytes(target.read_bytes().replace(b"20:00:00Z", b"20:00:01Z"))

    with pytest.raises(LegacyCleanupTransportError, match="signature"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()


def test_materialize_rejects_lost_deletion_before_mutation(tmp_path: Path) -> None:
    source, base, _, receipt, artifact, inventory = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    inventory["deletions"] = inventory["deletions"][:-1]
    (artifact / LEGACY_CLEANUP_INVENTORY_NAME).write_bytes(
        canonical_inventory_bytes(inventory)
    )

    with pytest.raises(LegacyCleanupTransportError, match="deletion inventory"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()
    assert not (checkout / receipt).exists()


def test_materialize_rejects_symlink_artifact_before_mutation(tmp_path: Path) -> None:
    source, base, _, _, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    inventory = artifact / LEGACY_CLEANUP_INVENTORY_NAME
    inventory.unlink()
    inventory.symlink_to(artifact / "missing")

    with pytest.raises(LegacyCleanupTransportError, match="regular 0644"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()


def test_materialize_rejects_symlink_artifact_root_before_mutation(
    tmp_path: Path,
) -> None:
    source, base, _, _, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    symlink = tmp_path / "artifact-link"
    symlink.symlink_to(artifact, target_is_directory=True)

    with pytest.raises(LegacyCleanupTransportError, match="not a real directory"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=symlink,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()


def test_materialize_rejects_dirty_or_stale_base_before_mutation(
    tmp_path: Path,
) -> None:
    source, base, _, receipt, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    (checkout / PRIMARY).write_text("changed\n", encoding="utf-8")

    with pytest.raises(LegacyCleanupTransportError, match="immutable base proof"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).read_text(encoding="utf-8") == "changed\n"
    assert (checkout / COMPANION).is_file()
    assert not (checkout / receipt).exists()


def test_materialize_rolls_back_if_exact_postcheck_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, base, _, receipt, artifact, _ = _built_artifact(tmp_path)
    checkout = _fresh_base_checkout(tmp_path, source, base)
    monkeypatch.setattr(
        transport,
        "verify_worktree_legacy_cleanup_transition",
        lambda *_args, **_kwargs: SimpleNamespace(
            issues=("injected postcheck failure",),
            receipt_path=None,
        ),
    )

    with pytest.raises(LegacyCleanupTransportError, match="postcheck failure"):
        materialize_transport_artifact(
            checkout,
            base_ref=base,
            artifact_root=artifact,
            verifier=PUBLIC_KEY,
        )

    assert (checkout / PRIMARY).is_file()
    assert (checkout / COMPANION).is_file()
    assert not (checkout / receipt).exists()
    assert _git(checkout, "status", "--porcelain") == ""


def test_build_rejects_mixed_signer_worktree(tmp_path: Path) -> None:
    source, base, _, _, = _source_cleanup(tmp_path)
    (source / "extra.txt").write_text("mixed\n", encoding="utf-8")

    with pytest.raises(LegacyCleanupTransportError, match="extra or mixed"):
        build_transport_artifact(
            source,
            base_ref=base,
            artifact_root=tmp_path / "artifact",
            verifier=PUBLIC_KEY,
        )


def test_trust_root_loader_accepts_public_only_config_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    path = _trust_roots(tmp_path / "trust.json")
    assert load_apply_trust_root(path).public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ) == PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["private_key"] = "forbidden"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LegacyCleanupTransportError, match="wrong fields"):
        load_apply_trust_root(path)


def test_transport_git_inspection_explicitly_disables_sparse_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(arguments, **_kwargs):
        captured.extend(arguments)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(transport.subprocess, "run", fake_run)

    transport._git(Path("/tmp/rulespec-be"), "status", "--porcelain")

    index = captured.index("core.sparseCheckout=false")
    assert captured[index - 1] == "-c"

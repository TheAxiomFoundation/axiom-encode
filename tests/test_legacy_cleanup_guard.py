"""Adversarial committed and worktree guards for legacy cleanup receipts."""

from __future__ import annotations

import hashlib
import os
import subprocess
from copy import deepcopy
from pathlib import Path, PurePosixPath

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from axiom_encode.legacy_cleanup import (
    LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
    LEGACY_CLEANUP_PROVENANCE_ASSERTIONS,
    LEGACY_CLEANUP_PROVENANCE_CLASS,
    LEGACY_CLEANUP_RECEIPT_SCHEMA,
    LEGACY_CLEANUP_TOOL,
    LEGACY_CLEANUP_VALIDATION_CHECKS,
    LEGACY_CLEANUP_VALIDATION_SCHEMA,
    canonical_receipt_bytes,
    companion_path,
    receipt_identity_sha256,
    receipt_path,
)
from axiom_encode.legacy_cleanup_git import plan_legacy_cleanup_base
from axiom_encode.legacy_cleanup_guard import (
    verify_committed_legacy_cleanup_transition,
    verify_worktree_legacy_cleanup_transition,
)
from axiom_encode.legacy_cleanup_signing import sign_legacy_cleanup_receipt
from axiom_encode.prepare_signed_backfill import (
    authorized_changed_paths,
    stage_authorized_changes,
)
from axiom_encode.signing_broker import canonical_signing_message

PRIMARY = Path("be/statutes/legacy.yaml")
COMPANION = companion_path(PRIMARY)
WAIVER_BYTES = b"version: 1\nwaivers: []\n"
WAIVER_SHA256 = hashlib.sha256(WAIVER_BYTES).hexdigest()
PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(b"\x35" * 32)
PUBLIC_KEY = PRIVATE_KEY.public_key()


class _Broker:
    apply_public_key_raw = PUBLIC_KEY.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    def apply_ed25519_sign(self, payload: bytes) -> bytes:
        return PRIVATE_KEY.sign(canonical_signing_message("apply_ed25519", payload))


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, message: str, *, allow_empty: bool = False) -> str:
    _git(repo, "add", "-A")
    arguments = ["commit", "-m", message]
    if allow_empty:
        arguments.append("--allow-empty")
    _git(repo, *arguments)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "rulespec-be"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "legacy-cleanup@example.com")
    _git(repo, "config", "user.name", "Legacy Cleanup Test")
    _git(
        repo,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/rulespec-be.git",
    )
    return repo


def _write_base_files(repo: Path) -> None:
    toolchain = repo / ".axiom/toolchain.toml"
    toolchain.parent.mkdir(parents=True, exist_ok=True)
    toolchain.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "legacy-cleanup-test"\n'
        f'axiom_corpus_release_content_sha256 = "{"c" * 64}"\n'
        f'validation_waiver_set_sha256 = "{WAIVER_SHA256}"\n',
        encoding="utf-8",
    )
    (repo / "known-validation-gaps.yaml").write_bytes(WAIVER_BYTES)
    target = repo / PRIMARY
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("module:\n  name: legacy\n", encoding="utf-8")
    (repo / COMPANION).write_text("tests:\n  - name: legacy\n", encoding="utf-8")
    (repo / "README.md").write_text("fixture\n", encoding="utf-8")


def _base_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = _init_repo(tmp_path)
    _write_base_files(repo)
    return repo, _commit(repo, "protected base")


def _plan(repo: Path, base: str):
    return plan_legacy_cleanup_base(
        repo,
        base_ref=base,
        primary_paths=[PRIMARY],
        require_clean_checkout=False,
    )


def _unsigned_payload(plan) -> dict[str, object]:
    checks = [
        {
            "name": name,
            "command": ["axiom-encode", "validate", name],
            "target_count": 1,
            "target_list_sha256": hashlib.sha256(
                f"targets:{name}".encode()
            ).hexdigest(),
            "exit_code": 0,
            "output_sha256": hashlib.sha256(f"output:{name}".encode()).hexdigest(),
        }
        for name in LEGACY_CLEANUP_VALIDATION_CHECKS
    ]
    payload: dict[str, object] = {
        "schema_version": LEGACY_CLEANUP_RECEIPT_SCHEMA,
        "tool": LEGACY_CLEANUP_TOOL,
        "provenance_class": LEGACY_CLEANUP_PROVENANCE_CLASS,
        "generated_at": "2026-08-30T20:00:00Z",
        "reason": "Remove an exact unmanifested legacy aggregation",
        "repository": {
            "repository": plan.repository,
            "object_format": plan.object_format,
            "base_commit": plan.base_commit,
            "base_tree": plan.base_tree,
            "projected_post_deletion_tree": plan.projected_post_deletion_tree,
        },
        "toolchain": {
            "axiom_encode": {
                "repository": "github.com/TheAxiomFoundation/axiom-encode",
                "object_format": "sha1",
                "commit": "a" * 40,
                "version": "0.2.test",
            },
            "axiom_rules_engine": {
                "repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
                "object_format": "sha1",
                "commit": "b" * 40,
            },
            "rulespec_dependencies": [],
            "corpus_release": {
                "name": plan.toolchain_values["axiom_corpus_release"],
                "content_sha256": plan.toolchain_values[
                    "axiom_corpus_release_content_sha256"
                ],
                "selector_sha256": "d" * 64,
            },
            "validation_waiver_set_sha256": plan.toolchain_values[
                "validation_waiver_set_sha256"
            ],
            "base_files": deepcopy(plan.base_files),
        },
        "base_proof": {
            "schema": LEGACY_CLEANUP_BASE_PROOF_SCHEMA,
            "ownership_inventory_sha256": plan.ownership_inventory_sha256,
            "provenance_record_count": len(plan.provenance_records),
            "surviving_reference_inventory_sha256": (
                plan.surviving_reference_inventory_sha256
            ),
            "surviving_blob_count": plan.surviving_blob_count,
        },
        "validation_execution": {
            "schema": LEGACY_CLEANUP_VALIDATION_SCHEMA,
            "status": "passed",
            "engine_execution": True,
            "projected_post_deletion_tree": plan.projected_post_deletion_tree,
            "checks": checks,
        },
        "provenance_assertions": dict(LEGACY_CLEANUP_PROVENANCE_ASSERTIONS),
        "groups": deepcopy(list(plan.groups)),
    }
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    return payload


def _sign(payload: dict[str, object]) -> dict[str, object]:
    sign_legacy_cleanup_receipt(payload, _Broker())
    return payload


def _write_receipt(repo: Path, payload: dict[str, object]) -> Path:
    identity = payload["receipt_identity_sha256"]
    assert isinstance(identity, str)
    path = receipt_path(identity)
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_receipt_bytes(payload))
    os.chmod(target, 0o644)
    return path


def _delete_group(repo: Path, *, companion: bool = True) -> None:
    (repo / PRIMARY).unlink()
    if companion:
        (repo / COMPANION).unlink()


def _atomic_cleanup(
    repo: Path,
    base: str,
    *,
    delete: str = "both",
    extra: bool = False,
    receipt_mode: int = 0o644,
) -> tuple[str, dict[str, object], Path]:
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    path = _write_receipt(repo, payload)
    os.chmod(repo / path, receipt_mode)
    if delete == "both":
        _delete_group(repo)
    elif delete == "primary":
        _delete_group(repo, companion=False)
    elif delete != "none":
        raise AssertionError(delete)
    if extra:
        (repo / "README.md").write_text("mixed\n", encoding="utf-8")
    return _commit(repo, "atomic cleanup"), payload, path


def _assert_issue(result, fragment: str) -> None:
    assert not result.authorized
    assert result.authorized_paths == ()
    assert any(fragment in issue for issue in result.issues), result.issues


def test_committed_guard_accepts_exact_single_squashed_b_to_h_contraction(
    tmp_path: Path,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, path = _atomic_cleanup(repo, base)

    result = verify_committed_legacy_cleanup_transition(
        repo,
        base_ref=base,
        head_ref=head,
        verifier=PUBLIC_KEY,
        expected_toolchain=payload["toolchain"],
    )

    assert result.authorized
    assert result.receipt_path == path
    assert result.authorized_paths == (path, PRIMARY, COMPANION)
    assert result.receipt == payload


def test_committed_guard_requires_full_commit_oids(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    head, _, _ = _atomic_cleanup(repo, base)

    result = verify_committed_legacy_cleanup_transition(
        repo,
        base_ref=base[:12],
        head_ref=head,
        verifier=PUBLIC_KEY,
    )

    _assert_issue(result, "full lowercase Git commit OID")


def test_committed_guard_rejects_two_commit_topology_even_when_final_diff_matches(
    tmp_path: Path,
) -> None:
    repo, base = _base_repo(tmp_path)
    _commit(repo, "unrelated empty commit", allow_empty=True)
    head, _, _ = _atomic_cleanup(repo, base)

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "one atomic commit")


@pytest.mark.parametrize(
    ("delete", "extra", "fragment"),
    [
        ("none", False, "missing exact changes"),
        ("primary", False, "missing exact changes"),
        ("both", True, "extra or mixed changes"),
    ],
    ids=["orphan-receipt", "partial-deletion", "mixed-extra-change"],
)
def test_committed_guard_rejects_nonexact_change_sets(
    tmp_path: Path,
    delete: str,
    extra: bool,
    fragment: str,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, _, _ = _atomic_cleanup(repo, base, delete=delete, extra=extra)

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, fragment)


def test_committed_guard_rejects_stale_base_binding(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _unsigned_payload(_plan(repo, base))
    repository = payload["repository"]
    assert isinstance(repository, dict)
    repository["base_commit"] = "f" * 40
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    _sign(payload)
    _write_receipt(repo, payload)
    _delete_group(repo)
    head = _commit(repo, "stale cleanup")

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "binding is stale")


def test_committed_guard_rejects_wrong_receipt_mode(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    head, _, _ = _atomic_cleanup(repo, base, receipt_mode=0o755)

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "not a regular 100644 blob")


def _orphan_receipt_base(
    tmp_path: Path,
) -> tuple[Path, str, dict[str, object], Path]:
    repo, original = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, original)))
    path = _write_receipt(repo, payload)
    return repo, _commit(repo, "historical receipt fixture"), payload, path


def test_unchanged_historical_receipt_cannot_authorize_replay(tmp_path: Path) -> None:
    repo, base, _, _ = _orphan_receipt_base(tmp_path)
    _delete_group(repo)
    head = _commit(repo, "attempt receipt replay")

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "exactly one new cleanup receipt")


def test_new_receipt_cannot_overlap_historical_coverage(tmp_path: Path) -> None:
    repo, base, historical, _ = _orphan_receipt_base(tmp_path)
    payload = deepcopy(historical)
    payload.pop("signature")
    payload["reason"] = "Attempt overlapping legacy contraction"
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    _sign(payload)
    _write_receipt(repo, payload)
    _delete_group(repo)
    head = _commit(repo, "overlapping cleanup")

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "overlaps historical receipt coverage")


@pytest.mark.parametrize("mutation", ["modify", "delete", "rename"])
def test_receipt_history_is_append_only_even_without_yaml_changes(
    tmp_path: Path, mutation: str
) -> None:
    repo, base, payload, path = _orphan_receipt_base(tmp_path)
    target = repo / path
    if mutation == "modify":
        raw = target.read_bytes().replace(b"20:00:00", b"20:00:01")
        target.write_bytes(raw)
    elif mutation == "delete":
        target.unlink()
    else:
        renamed = path.with_name(f"{'f' * 64}.json")
        target.rename(repo / renamed)
    head = _commit(repo, f"{mutation} historical receipt")

    result = verify_committed_legacy_cleanup_transition(
        repo, base_ref=base, head_ref=head, verifier=PUBLIC_KEY
    )

    expected = "modified" if mutation == "modify" else "deleted or renamed"
    _assert_issue(result, expected)
    assert payload


def test_expected_introduction_toolchain_is_exact(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, _ = _atomic_cleanup(repo, base)
    expected = deepcopy(payload["toolchain"])
    assert isinstance(expected, dict)
    expected["axiom_encode"] = {"unexpected": True}

    result = verify_committed_legacy_cleanup_transition(
        repo,
        base_ref=base,
        head_ref=head,
        verifier=PUBLIC_KEY,
        expected_toolchain=expected,
    )

    _assert_issue(result, "does not match current pins")


def test_worktree_guard_authorizes_exact_staged_and_unstaged_paths(
    tmp_path: Path,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    path = _write_receipt(repo, payload)
    _delete_group(repo)
    _git(repo, "add", "--", path.as_posix(), PRIMARY.as_posix())

    result = verify_worktree_legacy_cleanup_transition(
        repo,
        base_ref=base,
        verifier=PUBLIC_KEY,
        expected_toolchain=payload["toolchain"],
    )

    assert result.authorized
    assert result.authorized_paths == (path, PRIMARY, COMPANION)


def test_worktree_guard_rejects_unrelated_untracked_path(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    _write_receipt(repo, payload)
    _delete_group(repo)
    (repo / "extra.tmp").write_text("mixed\n", encoding="utf-8")

    result = verify_worktree_legacy_cleanup_transition(
        repo, base_ref=base, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "extra or mixed paths")


def test_worktree_guard_rejects_ignored_path(tmp_path: Path) -> None:
    repo = _init_repo(tmp_path)
    _write_base_files(repo)
    (repo / ".gitignore").write_text("ignored.tmp\n", encoding="utf-8")
    base = _commit(repo, "protected base")
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    _write_receipt(repo, payload)
    _delete_group(repo)
    (repo / "ignored.tmp").write_text("ignored\n", encoding="utf-8")

    result = verify_worktree_legacy_cleanup_transition(
        repo, base_ref=base, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "contains ignored path")


def test_worktree_guard_rejects_receipt_symlink(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    path = _write_receipt(repo, payload)
    (repo / path).unlink()
    (repo / path).symlink_to(repo / "README.md")
    _delete_group(repo)

    result = verify_worktree_legacy_cleanup_transition(
        repo, base_ref=base, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "not a regular 0644 file")


def test_worktree_guard_rejects_renamed_target(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    _write_receipt(repo, payload)
    renamed = PRIMARY.with_name("renamed.yaml")
    _git(repo, "mv", PRIMARY.as_posix(), renamed.as_posix())
    (repo / COMPANION).unlink()

    result = verify_worktree_legacy_cleanup_transition(
        repo, base_ref=base, verifier=PUBLIC_KEY
    )

    _assert_issue(result, "rename or copy")


def _cleanup_staging_arguments(
    base: str, payload: dict[str, object]
) -> dict[str, object]:
    toolchain = payload["toolchain"]
    assert isinstance(toolchain, dict)
    return {
        "legacy_cleanup_base_ref": base,
        "legacy_cleanup_verifier": PUBLIC_KEY,
        "legacy_cleanup_expected_toolchain": toolchain,
    }


def _write_cleanup_worktree(
    repo: Path,
    base: str,
    *,
    delete: str = "both",
    extra: bool = False,
) -> tuple[dict[str, object], Path]:
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    path = _write_receipt(repo, payload)
    if delete == "both":
        _delete_group(repo)
    elif delete == "primary":
        _delete_group(repo, companion=False)
    elif delete != "none":
        raise AssertionError(delete)
    if extra:
        (repo / "mixed.tmp").write_text("mixed\n", encoding="utf-8")
    return payload, path


def test_prepare_staging_authorizes_and_stages_only_atomic_cleanup(
    tmp_path: Path,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload, path = _write_cleanup_worktree(repo, base)
    arguments = _cleanup_staging_arguments(base, payload)

    authorized = authorized_changed_paths(repo, **arguments)
    stage_authorized_changes(repo, **arguments)

    assert authorized == {
        PurePosixPath(path.as_posix()),
        PurePosixPath(PRIMARY.as_posix()),
        PurePosixPath(COMPANION.as_posix()),
    }
    assert _git(repo, "diff", "--cached", "--name-status").splitlines() == [
        f"A\t{path.as_posix()}",
        f"D\t{COMPANION.as_posix()}",
        f"D\t{PRIMARY.as_posix()}",
    ]
    assert _git(repo, "diff", "--name-only") == ""


@pytest.mark.parametrize(
    ("delete", "extra", "fragment"),
    [
        ("none", False, "missing exact paths"),
        ("primary", False, "missing exact paths"),
        ("both", True, "extra or mixed paths"),
    ],
    ids=["orphan", "partial", "mixed"],
)
def test_prepare_staging_rejects_nonatomic_cleanup_worktrees(
    tmp_path: Path,
    delete: str,
    extra: bool,
    fragment: str,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload, _ = _write_cleanup_worktree(
        repo,
        base,
        delete=delete,
        extra=extra,
    )

    with pytest.raises(ValueError, match=fragment):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, payload),
        )

    assert _git(repo, "diff", "--cached", "--name-only") == ""


def test_prepare_staging_rejects_cleanup_rename(tmp_path: Path) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    _write_receipt(repo, payload)
    renamed = PRIMARY.with_name("renamed.yaml")
    _git(repo, "mv", PRIMARY.as_posix(), renamed.as_posix())
    (repo / COMPANION).unlink()

    with pytest.raises(ValueError, match="renamed/copied"):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, payload),
        )


def test_prepare_staging_rejects_modified_historical_receipt(tmp_path: Path) -> None:
    repo, base, historical, path = _orphan_receipt_base(tmp_path)
    target = repo / path
    target.write_bytes(target.read_bytes().replace(b"20:00:00", b"20:00:01"))

    with pytest.raises(ValueError, match="historical cleanup receipt was modified"):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, historical),
        )


def test_prepare_staging_rejects_unchanged_receipt_replay(tmp_path: Path) -> None:
    repo, base, historical, _ = _orphan_receipt_base(tmp_path)
    _delete_group(repo)

    with pytest.raises(ValueError, match="cleanup receipt-root transition"):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, historical),
        )


def test_prepare_staging_rejects_overlapping_cleanup_receipt(tmp_path: Path) -> None:
    repo, base, historical, _ = _orphan_receipt_base(tmp_path)
    payload = deepcopy(historical)
    payload.pop("signature")
    payload["reason"] = "Attempt overlapping legacy contraction"
    payload["receipt_identity_sha256"] = receipt_identity_sha256(payload)
    _sign(payload)
    _write_receipt(repo, payload)
    _delete_group(repo)

    with pytest.raises(ValueError, match="overlaps historical receipt coverage"):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, payload),
        )


@pytest.mark.parametrize(
    "missing",
    ["base", "verifier", "toolchain"],
)
def test_prepare_staging_requires_complete_cleanup_transport(
    tmp_path: Path,
    missing: str,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload, _ = _write_cleanup_worktree(repo, base)
    arguments = _cleanup_staging_arguments(base, payload)
    argument_name = {
        "base": "legacy_cleanup_base_ref",
        "verifier": "legacy_cleanup_verifier",
        "toolchain": "legacy_cleanup_expected_toolchain",
    }[missing]
    arguments[argument_name] = None

    with pytest.raises(ValueError, match="together"):
        stage_authorized_changes(repo, **arguments)


def test_prepare_staging_rejects_cleanup_intent_without_receipt_transition(
    tmp_path: Path,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))

    with pytest.raises(ValueError, match="require a cleanup receipt-root transition"):
        stage_authorized_changes(
            repo,
            **_cleanup_staging_arguments(base, payload),
        )

"""Historical root receipts can only be retired with exact prior-byte evidence."""

import hashlib
import json

import pytest

from axiom_encode import cli


def owner(tmp_path):
    checkout = tmp_path / "rulespec-ca"
    (checkout / "ca/policies").mkdir(parents=True)
    target = checkout / ".axiom/encoding-manifests/policies/example.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"historical untrusted owner\n")
    target.chmod(0o644)
    return checkout, target, hashlib.sha256(target.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "defect", ["omit", "none", "malformed", "wrong", "write", "unlisted"]
)
def test_historical_retirement_requires_exact_old_hash_and_deletion(tmp_path, defect):
    root, target, digest = owner(tmp_path)
    before = target.read_bytes()
    expected = (
        {}
        if defect == "omit"
        else {
            target: None
            if defect == "none"
            else "bad"
            if defect == "malformed"
            else "0" * 64
            if defect == "wrong"
            else digest
        }
    )
    with pytest.raises(RuntimeError):
        cli._install_apply_transaction(
            [(target, b"new owner" if defect == "write" else None)],
            checkout_root=root,
            expected_originals=expected,
            historical_manifest_deletions=frozenset()
            if defect == "unlisted"
            else frozenset({target}),
        )
    assert target.read_bytes() == before


def test_verified_retirement_and_failed_install_rollback(tmp_path):
    root, target, digest = owner(tmp_path)
    before = target.read_bytes()

    def refuse():
        assert not target.exists()
        raise RuntimeError("post-install refusal")

    with pytest.raises(RuntimeError, match="post-install refusal"):
        cli._install_apply_transaction(
            [(target, None)],
            checkout_root=root,
            expected_originals={target: digest},
            historical_manifest_deletions=frozenset({target}),
            post_install_check=refuse,
        )
    assert target.read_bytes() == before
    cli._install_apply_transaction(
        [(target, None)],
        checkout_root=root,
        expected_originals={target: digest},
        historical_manifest_deletions=frozenset({target}),
    )
    assert not target.exists()


@pytest.mark.parametrize(
    "tail", ["misc/example.json", "policies/example.test.json", "policies/example.yaml"]
)
def test_other_historical_surfaces_remain_refused(tmp_path, tail):
    root, _, _ = owner(tmp_path)
    with pytest.raises(RuntimeError, match="outside the canonical"):
        cli._apply_transaction_target_relative_path(
            root,
            root / ".axiom/encoding-manifests" / tail,
            historical_manifest_deletion=True,
        )


@pytest.mark.parametrize(
    "defect",
    [
        None,
        "write",
        "absent",
        "newhash",
        "backup",
        "path",
        "live_changed",
        "live_missing",
    ],
)
@pytest.mark.parametrize("state", ["prepared", "applying"])
def test_recovery_preserves_strict_historical_deletion_state(tmp_path, defect, state):
    root, target, digest = owner(tmp_path)
    old = target.read_bytes()
    if state == "applying":
        target.unlink()
    directory = root / ".axiom/.apply-transaction"
    backups = directory / "backups"
    backups.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)
    backups.chmod(0o700)
    backup = backups / "000000.old"
    backup.write_bytes(b"tampered" if defect == "backup" else old)
    backup.chmod(0o600)
    entry = {
        "path": target.relative_to(root).as_posix(),
        "existed": True,
        "mode": 0o644,
        "old_sha256": digest,
        "backup": "000000.old",
        "delete": True,
        "new_sha256": None,
    }
    if defect == "write":
        entry.update(delete=False, new_sha256=digest)
    if defect == "absent":
        entry.update(existed=False, old_sha256=None, backup=None)
    if defect == "newhash":
        entry["new_sha256"] = digest
    if defect == "path":
        entry["path"] = ".github/workflows/example.json"
    journal = directory / "journal.json"
    journal.write_text(
        json.dumps(
            {
                "schema": cli._APPLY_TRANSACTION_SCHEMA,
                "state": state,
                "entries": [entry],
                "created_directories": [],
            }
        )
    )
    journal.chmod(0o600)
    if defect == "live_changed":
        target.write_bytes(b"externally changed")
    if defect == "live_missing":
        target.unlink(missing_ok=True)
    if (
        defect is None
        or (state == "prepared" and defect == "backup")
        or (state == "applying" and defect == "live_missing")
    ):
        # An unchanged live target is authoritative; an unused backup is never read.
        cli._recover_apply_transaction(root)
        assert target.read_bytes() == old
        assert not directory.exists()
    else:
        with pytest.raises(RuntimeError):
            cli._recover_apply_transaction(root)
        if defect == "live_changed":
            assert target.read_bytes() == b"externally changed"
        elif state == "prepared" and defect != "live_missing":
            assert target.read_bytes() == old
        else:
            assert not target.exists()


def test_unrelated_unsafe_target_retains_canonical_surface_refusal(tmp_path):
    root, _, _ = owner(tmp_path)
    target = root / ".github/workflows/unsafe.json"
    with pytest.raises(
        RuntimeError, match="outside the canonical RuleSpec/manifest mutation surface"
    ):
        cli._install_apply_transaction(
            [(target, None)],
            checkout_root=root,
            historical_manifest_deletions=frozenset({target}),
        )

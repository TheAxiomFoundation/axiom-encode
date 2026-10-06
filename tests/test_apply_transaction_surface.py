"""The apply-transaction mutation surface a successor repoint needs.

The allowlist admits the repoint receipt directory (one ``<sha256>.json``), the
two repoint-only metadata files only with an explicit repoint capability,
and only the ProgramSpec paths one transaction declares.  Declared ProgramSpecs are journaled (schema v3) so a
killed install recovers under the same predicate, and jurisdiction-less v1
manifests may be deleted but never written.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from axiom_encode.cli import (
    _APPLY_TRANSACTION_SCHEMA,
    _APPLY_TRANSACTION_SCHEMA_V3,
    _install_apply_transaction,
    _is_canonical_apply_transaction_target,
    _load_apply_transaction_journal,
)

PROGRAM_SPEC = "programs/us/fiit/fy-2026.yaml"


class TestInstallSurface:
    @pytest.mark.parametrize(
        "relative",
        [
            ".axiom/legacy-successor-repoints/" + "a" * 64 + ".json",
            ".axiom/upstream-source-check-baseline.txt",
            "known-missing-money-atoms.yaml",
            ".axiom/encoding-manifests/policies/irs/legacy-table.json",
        ],
    )
    def test_admits_the_repoint_surface(self, tmp_path, relative):
        assert _is_canonical_apply_transaction_target(
            tmp_path, Path(relative), successor_repoint=True
        )

    @pytest.mark.parametrize(
        "relative",
        [
            ".axiom/legacy-successor-repoints/not-a-digest.json",
            ".axiom/legacy-successor-repoints/nested/" + "a" * 64 + ".json",
            PROGRAM_SPEC,
            "us/programs/fiit/fy-2026.yaml",
            "README.md",
            "tests/test_other.py",
        ],
    )
    def test_refuses_everything_else(self, tmp_path, relative):
        assert not _is_canonical_apply_transaction_target(tmp_path, Path(relative))

    def test_admits_only_the_declared_program_specs(self, tmp_path):
        declared = frozenset({Path(PROGRAM_SPEC)})
        assert _is_canonical_apply_transaction_target(
            tmp_path, Path(PROGRAM_SPEC), declared_program_specs=declared
        )
        assert not _is_canonical_apply_transaction_target(
            tmp_path,
            Path("programs/us/other/fy-2026.yaml"),
            declared_program_specs=declared,
        )

    def _checkout(self, tmp_path) -> Path:
        repo = tmp_path / "rulespec-us"
        (repo / "programs/us/fiit").mkdir(parents=True)
        (repo / PROGRAM_SPEC).write_text("program: us/fiit\n")
        return repo

    def test_installs_a_declared_program_spec_and_journals_it(self, tmp_path):
        repo = self._checkout(tmp_path)
        seen: dict[str, object] = {}

        def capture() -> None:
            journal = _load_apply_transaction_journal(
                repo / ".axiom/.apply-transaction", checkout_root=repo.resolve()
            )
            seen.update(journal)

        _install_apply_transaction(
            [(repo / PROGRAM_SPEC, b"program: us/fiit\nscope: {}\n")],
            checkout_root=repo,
            declared_program_specs=(Path(PROGRAM_SPEC),),
            post_install_check=capture,
        )
        assert (repo / PROGRAM_SPEC).read_bytes() == b"program: us/fiit\nscope: {}\n"
        assert seen["schema"] == _APPLY_TRANSACTION_SCHEMA_V3
        assert seen["declared_program_specs"] == [PROGRAM_SPEC]

    def test_refuses_an_undeclared_program_spec(self, tmp_path):
        repo = self._checkout(tmp_path)
        with pytest.raises(RuntimeError, match="outside the canonical"):
            _install_apply_transaction(
                [(repo / PROGRAM_SPEC, b"x\n")], checkout_root=repo
            )

    def test_refuses_to_write_a_jurisdictionless_manifest(self, tmp_path):
        repo = self._checkout(tmp_path)
        target = repo / ".axiom/encoding-manifests/policies/irs/legacy-table.json"
        with pytest.raises(RuntimeError, match="may only delete"):
            _install_apply_transaction(
                [(target, b"{}\n")], checkout_root=repo, successor_repoint=True
            )

    def test_recovery_refuses_a_journal_that_widens_its_program_specs(self, tmp_path):
        repo = self._checkout(tmp_path)
        transaction = repo / ".axiom/.apply-transaction"
        transaction.parent.mkdir(parents=True, exist_ok=True)
        transaction.mkdir(mode=0o700)
        (transaction / "backups").mkdir(mode=0o700)
        journal = {
            "schema": _APPLY_TRANSACTION_SCHEMA_V3,
            "state": "prepared",
            "entries": [
                {
                    "path": "programs/us/other/fy-2026.yaml",
                    "existed": False,
                    "mode": 0o644,
                    "old_sha256": None,
                    "backup": None,
                    "delete": False,
                    "new_sha256": "0" * 64,
                }
            ],
            "created_directories": [],
            "declared_program_specs": [PROGRAM_SPEC],
        }
        path = transaction / "journal.json"
        path.write_text(json.dumps(journal))
        os.chmod(path, 0o600)
        with pytest.raises(RuntimeError, match="target is not canonical"):
            _load_apply_transaction_journal(transaction, checkout_root=repo.resolve())

    @pytest.mark.parametrize(
        "declared",
        [
            [],
            ["programs/../x.yaml"],
            [PROGRAM_SPEC, PROGRAM_SPEC],
            ["us/statutes/1.yaml"],
        ],
    )
    def test_recovery_refuses_a_malformed_program_spec_list(self, tmp_path, declared):
        repo = self._checkout(tmp_path)
        transaction = repo / ".axiom/.apply-transaction"
        transaction.parent.mkdir(parents=True, exist_ok=True)
        transaction.mkdir(mode=0o700)
        journal = {
            "schema": _APPLY_TRANSACTION_SCHEMA_V3,
            "state": "prepared",
            "entries": [],
            "created_directories": [],
            "declared_program_specs": declared,
        }
        path = transaction / "journal.json"
        path.write_text(json.dumps(journal))
        os.chmod(path, 0o600)
        with pytest.raises(RuntimeError, match="ProgramSpec"):
            _load_apply_transaction_journal(transaction, checkout_root=repo.resolve())

    def test_a_v2_journal_cannot_carry_program_specs(self, tmp_path):
        repo = self._checkout(tmp_path)
        transaction = repo / ".axiom/.apply-transaction"
        transaction.parent.mkdir(parents=True, exist_ok=True)
        transaction.mkdir(mode=0o700)
        path = transaction / "journal.json"
        path.write_text(
            json.dumps(
                {
                    "schema": _APPLY_TRANSACTION_SCHEMA,
                    "state": "prepared",
                    "entries": [],
                    "created_directories": [],
                    "declared_program_specs": [PROGRAM_SPEC],
                }
            )
        )
        os.chmod(path, 0o600)
        with pytest.raises(RuntimeError, match="unsupported shape"):
            _load_apply_transaction_journal(transaction, checkout_root=repo.resolve())


_REPOINT_ONLY_TARGETS = [
    (".axiom/upstream-source-check-baseline.txt", b"new\n"),
    ("known-missing-money-atoms.yaml", b"new\n"),
    (".axiom/encoding-manifests/policies/irs/legacy-table.json", None),
]


@pytest.mark.parametrize(("relative", "postimage"), _REPOINT_ONLY_TARGETS)
def test_non_repoint_transaction_refuses_repoint_only_targets(
    tmp_path, relative, postimage
):
    repo = tmp_path / "rulespec-us"
    target = repo / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old\n")
    with pytest.raises(RuntimeError, match="outside the canonical"):
        _install_apply_transaction([(target, postimage)], checkout_root=repo)
    assert target.read_bytes() == b"old\n"


@pytest.mark.parametrize(("relative", "postimage"), _REPOINT_ONLY_TARGETS)
def test_repoint_transaction_journals_capability_and_recovers_same_surface(
    tmp_path, relative, postimage
):
    repo = tmp_path / "rulespec-us"
    target = repo / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"old\n")
    seen = {}

    def check_recovery_surface():
        transaction = repo / ".axiom/.apply-transaction"
        seen.update(_load_apply_transaction_journal(transaction, checkout_root=repo))
        assert seen["successor_repoint"] is True
        assert seen["declared_program_specs"] == []

    _install_apply_transaction(
        [(target, postimage)],
        checkout_root=repo,
        successor_repoint=True,
        post_install_check=check_recovery_surface,
    )
    assert seen["schema"] == _APPLY_TRANSACTION_SCHEMA_V3
    if postimage is None:
        assert not target.exists()
    else:
        assert target.read_bytes() == postimage


@pytest.mark.parametrize(("relative", "postimage"), _REPOINT_ONLY_TARGETS)
@pytest.mark.parametrize("version", ["v2", "v3-absent", "v3-false"])
def test_non_repoint_recovery_refuses_repoint_only_targets(
    tmp_path, relative, postimage, version
):
    repo = tmp_path / "rulespec-us"
    transaction = repo / ".axiom/.apply-transaction"
    transaction.mkdir(parents=True, mode=0o700)
    journal = {
        "schema": _APPLY_TRANSACTION_SCHEMA,
        "state": "prepared",
        "entries": [
            {
                "path": relative,
                "existed": True,
                "mode": 0o644,
                "old_sha256": "0" * 64,
                "backup": "000000.old",
                "delete": postimage is None,
                "new_sha256": "1" * 64 if postimage is not None else None,
            }
        ],
        "created_directories": [],
    }
    if version != "v2":
        journal.update(
            schema=_APPLY_TRANSACTION_SCHEMA_V3,
            declared_program_specs=[PROGRAM_SPEC],
        )
    if version == "v3-false":
        journal["successor_repoint"] = False
    path = transaction / "journal.json"
    path.write_text(json.dumps(journal))
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="target is not canonical"):
        _load_apply_transaction_journal(transaction, checkout_root=repo)


@pytest.mark.parametrize("marker", [None, "true", 1, [], {}])
def test_recovery_refuses_non_boolean_repoint_capability(tmp_path, marker):
    repo = tmp_path / "rulespec-us"
    transaction = repo / ".axiom/.apply-transaction"
    transaction.mkdir(parents=True, mode=0o700)
    path = transaction / "journal.json"
    path.write_text(
        json.dumps(
            {
                "schema": _APPLY_TRANSACTION_SCHEMA_V3,
                "state": "prepared",
                "entries": [],
                "created_directories": [],
                "declared_program_specs": [],
                "successor_repoint": marker,
            }
        )
    )
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="repoint marker is invalid"):
        _load_apply_transaction_journal(transaction, checkout_root=repo)


def test_v2_journal_cannot_carry_repoint_capability(tmp_path):
    repo = tmp_path / "rulespec-us"
    transaction = repo / ".axiom/.apply-transaction"
    transaction.mkdir(parents=True, mode=0o700)
    path = transaction / "journal.json"
    path.write_text(
        json.dumps(
            {
                "schema": _APPLY_TRANSACTION_SCHEMA,
                "state": "prepared",
                "entries": [],
                "created_directories": [],
                "successor_repoint": True,
            }
        )
    )
    path.chmod(0o600)
    with pytest.raises(RuntimeError, match="unsupported shape"):
        _load_apply_transaction_journal(transaction, checkout_root=repo)

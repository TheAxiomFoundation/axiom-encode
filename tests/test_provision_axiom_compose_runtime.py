"""Unit coverage for the commit-bound axiom-compose provisioner."""

from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_BASE_SPEC = importlib.util.spec_from_file_location(
    "provision_verification_supervisor",
    _ROOT / "scripts/provision_verification_supervisor.py",
)
base_provisioner = importlib.util.module_from_spec(_BASE_SPEC)
sys.modules[_BASE_SPEC.name] = base_provisioner
_BASE_SPEC.loader.exec_module(base_provisioner)

_SPEC = importlib.util.spec_from_file_location(
    "provision_axiom_compose_runtime",
    _ROOT / "scripts/provision_axiom_compose_runtime.py",
)
provisioner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = provisioner
_SPEC.loader.exec_module(provisioner)


def _write_archive(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:") as bundle:
        for name, body in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))


def test_require_sha_rejects_noncanonical_ref() -> None:
    with pytest.raises(SystemExit, match="full lowercase commit SHA"):
        provisioner._require_sha("main")


def test_compose_git_metadata_is_copied_into_owned_snapshot(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    caller_git = checkout / ".git"
    caller_git.mkdir(parents=True)
    (caller_git / "HEAD").write_text("ref: refs/heads/main\n")
    staging = tmp_path / "staging"
    staging.mkdir()

    git_dir = provisioner._own_compose_git_directory(checkout, staging)

    assert git_dir == staging / "compose.git"
    assert git_dir.stat().st_uid == provisioner.os.geteuid()
    assert (git_dir / "HEAD").read_text() == "ref: refs/heads/main\n"


def test_compose_git_snapshot_rejects_gitfile_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / ".git").write_text("gitdir: elsewhere\n")

    with pytest.raises(SystemExit, match="is not a plain directory"):
        provisioner._own_compose_git_directory(checkout, tmp_path / "staging")


def test_safe_extract_accepts_only_compose_package(tmp_path: Path) -> None:
    archive = tmp_path / "compose.tar"
    _write_archive(
        archive,
        {
            "src/axiom_compose/__init__.py": b"",
            "src/axiom_compose/cli.py": b"def main(): return 0\n",
        },
    )

    package = provisioner._safe_extract_package(archive, tmp_path / "export")

    assert package == tmp_path / "export/src/axiom_compose"
    assert (package / "cli.py").is_file()


def test_safe_extract_rejects_files_outside_compose_package(tmp_path: Path) -> None:
    archive = tmp_path / "compose.tar"
    _write_archive(
        archive,
        {
            "src/axiom_compose/cli.py": b"def main(): return 0\n",
            "pyproject.toml": b"[project]\n",
        },
    )

    with pytest.raises(SystemExit, match="unsafe axiom-compose archive member"):
        provisioner._safe_extract_package(archive, tmp_path / "export")


def test_tree_hash_binds_relative_names_and_contents(tmp_path: Path) -> None:
    package = tmp_path / "axiom_compose"
    package.mkdir()
    module = package / "cli.py"
    module.write_text("first\n")
    first = provisioner._tree_sha256(package)

    module.write_text("second\n")

    assert provisioner._tree_sha256(package) != first

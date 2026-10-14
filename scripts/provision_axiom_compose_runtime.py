"""Provision a self-contained, commit-bound axiom-compose runtime.

The signed encoder may generate a composition module, so its deterministic
validator must be able to run the exact composer pinned by the RuleSpec
checkout.  This script is invoked with Python 3.14, snapshots the requested Git
commit (never mutable worktree bytes), overlays that package into a relocated
runtime, and emits one isolated launcher for the signing supervisor to call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from provision_verification_supervisor import (
    DEFAULT_MAX_RUNTIME_FILES,
    _assert_sane_source_prefix,
    _assert_self_contained,
    _hardened_git,
    _hardened_git_text,
    _relocate_elf_rpaths,
    _resolve_trusted_git,
    _stage_runtime_tree,
)

_SHA_PATTERN = "0123456789abcdef"


def _require_sha(value: str) -> str:
    if len(value) != 40 or any(character not in _SHA_PATTERN for character in value):
        raise SystemExit("--compose-ref must be a full lowercase commit SHA")
    return value


def _own_compose_git_directory(checkout: Path, staging: Path) -> Path:
    """Copy Git metadata into a snapshot owned by the provisioning identity."""

    caller_git = checkout / ".git"
    if caller_git.is_symlink() or not caller_git.is_dir():
        raise SystemExit(
            f"refusing to provision: {checkout}/.git is not a plain directory"
        )
    git_dir = staging / "compose.git"
    shutil.copytree(caller_git, git_dir, symlinks=True)
    if git_dir.stat().st_uid != os.geteuid():
        raise SystemExit(
            "refusing to provision: copied axiom-compose git dir is not owned "
            f"by the provisioning identity: {git_dir}"
        )
    return git_dir


def _safe_extract_package(archive: Path, destination: Path) -> Path:
    prefix = PurePosixPath("src/axiom_compose")
    with tarfile.open(archive, "r:") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = PurePosixPath(member.name)
            parent_directory = member.isdir() and path in {
                PurePosixPath("src"),
                prefix,
            }
            if (
                path.is_absolute()
                or ".." in path.parts
                or (not parent_directory and not path.is_relative_to(prefix))
                or member.issym()
                or member.islnk()
                or member.isdev()
            ):
                raise SystemExit(
                    f"refusing unsafe axiom-compose archive member: {member.name}"
                )
        bundle.extractall(destination, members=members, filter="data")
    package = destination / prefix
    if not package.is_dir() or not (package / "cli.py").is_file():
        raise SystemExit("axiom-compose commit archive lacks src/axiom_compose")
    return package


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        body = path.read_bytes()
        digest.update(len(body).to_bytes(8, "big"))
        digest.update(body)
    return digest.hexdigest()


def provision(
    *,
    destination: Path,
    site_packages: Path,
    compose_checkout: Path,
    compose_ref: str,
    require_prefix_under: Path | None,
    patchelf: str | None,
    max_runtime_files: int,
) -> None:
    compose_ref = _require_sha(compose_ref)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"destination already exists: {destination}")
    checkout = compose_checkout.resolve(strict=True)
    source_runtime = Path(sys.base_prefix).resolve(strict=True)
    source_interpreter = Path(sys.executable).resolve(strict=True)
    _assert_sane_source_prefix(source_runtime, require_prefix_under, max_runtime_files)

    snapshot = Path(tempfile.mkdtemp(prefix="axiom-compose-snapshot-"))
    try:
        trusted_git = _resolve_trusted_git(Path("/usr/bin/git"))
        git_home = snapshot / "home"
        git_home.mkdir()
        git_dir = _own_compose_git_directory(checkout, snapshot)
        if _hardened_git_text(
            trusted_git, git_dir, git_home, "rev-parse", "HEAD"
        ) != compose_ref:
            raise SystemExit("axiom-compose snapshot HEAD does not match --compose-ref")
        origin = _hardened_git_text(
            trusted_git, git_dir, git_home, "config", "--get", "remote.origin.url"
        )
        accepted_origins = {
            "https://github.com/TheAxiomFoundation/axiom-compose",
            "https://github.com/TheAxiomFoundation/axiom-compose.git",
            "git@github.com:TheAxiomFoundation/axiom-compose.git",
        }
        if origin not in accepted_origins:
            raise SystemExit(f"unexpected axiom-compose origin: {origin}")

        archive = snapshot / "compose.tar"
        archive.write_bytes(
            _hardened_git(
                trusted_git,
                git_dir,
                git_home,
                "archive",
                "--format=tar",
                compose_ref,
                "--",
                "src/axiom_compose",
            )
        )
        package = _safe_extract_package(archive, snapshot / "export")

        runtime = destination / "python"
        _stage_runtime_tree(source_runtime, runtime, site_packages.resolve(strict=True))
        runtime_site_packages = (
            runtime
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )
        installed_package = runtime_site_packages / "axiom_compose"
        if installed_package.exists() or installed_package.is_symlink():
            shutil.rmtree(installed_package)
        shutil.copytree(package, installed_package, symlinks=False)

        if sys.platform == "linux":
            resolved_patchelf = patchelf or shutil.which("patchelf")
            if resolved_patchelf is None:
                raise SystemExit("patchelf is required on linux")
            _relocate_elf_rpaths(runtime, resolved_patchelf)
        interpreter = runtime / source_interpreter.relative_to(source_runtime)
        _assert_self_contained(runtime, source_runtime, interpreter)

        launcher = destination / "axiom-compose"
        launcher.write_text(
            f"#!{interpreter} -I\n"
            "from axiom_compose.cli import main\n"
            "raise SystemExit(main())\n"
        )
        launcher.chmod(0o755)
        attestation = {
            "schema": "axiom-encode/trusted-axiom-compose/v1",
            "repository": "github.com/TheAxiomFoundation/axiom-compose",
            "commit": compose_ref,
            "package_tree_sha256": _tree_sha256(installed_package),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        (destination / "runtime-attestation.json").write_text(
            json.dumps(attestation, sort_keys=True) + "\n"
        )
    finally:
        shutil.rmtree(snapshot, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--compose-checkout", type=Path, required=True)
    parser.add_argument("--compose-ref", required=True)
    parser.add_argument("--require-prefix-under", type=Path)
    parser.add_argument("--patchelf")
    parser.add_argument(
        "--max-runtime-files", type=int, default=DEFAULT_MAX_RUNTIME_FILES
    )
    args = parser.parse_args()
    provision(
        destination=args.destination.resolve(),
        site_packages=args.site_packages.resolve(),
        compose_checkout=args.compose_checkout.resolve(),
        compose_ref=args.compose_ref,
        require_prefix_under=(
            args.require_prefix_under.resolve()
            if args.require_prefix_under is not None
            else None
        ),
        patchelf=args.patchelf,
        max_runtime_files=args.max_runtime_files,
    )


if __name__ == "__main__":
    main()

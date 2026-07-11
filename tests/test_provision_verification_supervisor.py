import json
import subprocess
import sys
from pathlib import Path


def test_provision_replaces_base_runtime_site_packages(tmp_path: Path) -> None:
    source_site_packages = (
        Path(sys.base_prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert (source_site_packages / "pip").is_dir()
    assert (source_site_packages / "README.txt").is_file()

    trusted_packages = tmp_path / "trusted-packages"
    intended_package = trusted_packages / "intended_package"
    intended_package.mkdir(parents=True)
    (intended_package / "__init__.py").write_text("TRUSTED = True\n")
    supervisor = tmp_path / "supervisor"
    supervisor.write_text("supervisor\n")
    destination = tmp_path / "provisioned"

    subprocess.run(
        [
            sys.executable,
            "scripts/provision_verification_supervisor.py",
            "--destination",
            str(destination),
            "--supervisor",
            str(supervisor),
            "--site-packages",
            str(trusted_packages),
            "--apply-root",
            "apply-root",
            "--eval-root",
            "eval-root",
            "--corpus-release-root",
            "corpus-release-root",
        ],
        check=True,
    )

    runtime = destination / "python"
    interpreter = runtime / Path(sys.executable).resolve().relative_to(
        Path(sys.base_prefix).resolve()
    )
    assert (destination / "axiom-encode").read_text().splitlines()[0] == (
        f"#!{interpreter} -I"
    )
    assert json.loads((destination / "signing-trust-roots.json").read_text()) == {
        "schema": "axiom-encode/signing-trust-roots/v2",
        "apply_ed25519_public_key": "apply-root",
        "eval_ed25519_public_key": "eval-root",
        "corpus_release_ed25519_public_key": "corpus-release-root",
    }

    provisioned_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert sorted(
        path.relative_to(provisioned_site_packages)
        for path in provisioned_site_packages.rglob("*")
    ) == [
        Path("intended_package"),
        Path("intended_package/__init__.py"),
    ]
    assert not any(path.is_symlink() for path in destination.rglob("*"))
    forbidden_patterns = (
        "*.pth",
        "*.egg-link",
        "sitecustomize.py",
        "usercustomize.py",
        "pyvenv.cfg",
        "__editable__*",
    )
    assert not any(
        path.is_file()
        for pattern in forbidden_patterns
        for path in destination.rglob(pattern)
    )

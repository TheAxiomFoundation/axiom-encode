"""Build a protected, self-contained verification-supervisor execution tree."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def provision(
    destination: Path,
    supervisor: Path,
    site_packages: Path,
    apply_root: str,
    eval_root: str,
    corpus_release_root: str,
) -> None:
    source_runtime = Path(sys.base_prefix).resolve(strict=True)
    source_interpreter = Path(sys.executable).resolve(strict=True)
    runtime = destination / "python"
    shutil.copytree(source_runtime, runtime, symlinks=False)
    for forbidden in (
        *runtime.rglob("*.pth"),
        *runtime.rglob("*.egg-link"),
        *runtime.rglob("sitecustomize.py"),
        *runtime.rglob("usercustomize.py"),
        *runtime.rglob("pyvenv.cfg"),
        *runtime.rglob("__editable__*"),
    ):
        if forbidden.is_file():
            forbidden.unlink()
    target_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    shutil.copytree(site_packages.resolve(strict=True), target_site_packages)
    interpreter = runtime / source_interpreter.relative_to(source_runtime)
    launcher = destination / "axiom-encode"
    launcher.write_text(f"#!{interpreter} -I\nraise SystemExit('launcher executed')\n")
    launcher.chmod(0o755)
    trust = destination / "signing-trust-roots.json"
    trust.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/signing-trust-roots/v2",
                "apply_ed25519_public_key": apply_root,
                "eval_ed25519_public_key": eval_root,
                "corpus_release_ed25519_public_key": corpus_release_root,
            },
            sort_keys=True,
        )
        + "\n"
    )
    trust.chmod(0o644)
    shutil.copy2(supervisor, destination / "axiom-encode-signing-supervisor")
    (destination / "axiom-encode-signing-supervisor").chmod(0o755)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--supervisor", type=Path, required=True)
    parser.add_argument("--site-packages", type=Path, required=True)
    parser.add_argument("--apply-root", required=True)
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--corpus-release-root", required=True)
    args = parser.parse_args()
    provision(
        args.destination.resolve(),
        args.supervisor.resolve(),
        args.site_packages.resolve(),
        args.apply_root,
        args.eval_root,
        args.corpus_release_root,
    )

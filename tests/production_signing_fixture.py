"""Prepare the root-owned signing-supervisor fixture used only by CI."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from base64 import b64encode
from pathlib import Path

import _cffi_backend
import cryptography
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).parents[1]


def _public(seed: bytes) -> str:
    return b64encode(
        Ed25519PrivateKey.from_private_bytes(seed)
        .public_key()
        .public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")


def prepare(destination: Path, supervisor: Path) -> None:
    source_runtime = Path(sys.base_prefix).resolve()
    source_interpreter = Path(sys.executable).resolve()
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
    interpreter = runtime / source_interpreter.relative_to(source_runtime)
    site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    site_packages.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(cryptography.__file__).resolve().parent,
        site_packages / "cryptography",
        dirs_exist_ok=True,
        symlinks=False,
    )
    shutil.copy2(Path(_cffi_backend.__file__).resolve(), site_packages)
    package = site_packages / "axiom_encode"
    package.mkdir()
    (package / "__init__.py").write_text('"""Production CI fixture."""\n')
    shutil.copy2(ROOT / "src/axiom_encode/signing_broker.py", package)
    shutil.copy2(ROOT / "src/axiom_encode/_trusted_signing_bootstrap.py", package)
    (package / "entrypoint.py").write_text(
        """from __future__ import annotations
import json
from base64 import b64encode

def main():
    from axiom_encode.signing_broker import get_signing_broker
    broker = get_signing_broker()
    print(json.dumps({
        "apply": b64encode(broker.apply_ed25519_sign(b"production-apply")).decode(),
        "eval": b64encode(broker.eval_ed25519_sign(b"production-eval")).decode(),
    }, sort_keys=True))
    broker.close()
    return 0
"""
    )
    launcher = destination / "axiom-encode"
    launcher.write_text(f"#!{interpreter} -I\nraise SystemExit('launcher executed')\n")
    launcher.chmod(0o755)
    trust = destination / "signing-trust-roots.json"
    trust.write_text(
        json.dumps(
            {
                "schema": "axiom-encode/signing-trust-roots/v2",
                "apply_ed25519_public_key": _public(b"\xab" * 32),
                "eval_ed25519_public_key": _public(b"\xcd" * 32),
                "corpus_release_ed25519_public_key": _public(b"\x17" * 32),
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
    arguments = parser.parse_args()
    prepare(arguments.destination.resolve(), arguments.supervisor.resolve())

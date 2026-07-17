"""Build a protected, self-contained verification-supervisor execution tree."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Prefixes that are never a self-contained CPython runtime. Copying one of
# these (e.g. /usr, whose sys.base_prefix the system interpreter reports)
# means the caller resolved the wrong interpreter; fail before the copy.
FORBIDDEN_PREFIXES = ("/", "/usr", "/usr/local", "/opt", "/etc", "/bin", "/sbin", "/lib")

# A real CPython prefix (toolcache or standalone build) is ~20k files. Anything
# past this is a wrong-prefix copy about to fill the disk.
MAX_RUNTIME_FILES = 100_000


def _assert_sane_source_prefix(source_runtime: Path, require_under: Path | None) -> None:
    if str(source_runtime) in FORBIDDEN_PREFIXES:
        raise SystemExit(
            f"refusing to provision from {source_runtime}: not a self-contained "
            "python prefix (the invoking interpreter is a system python)"
        )
    if require_under is not None:
        resolved_parent = require_under.resolve(strict=True)
        if not source_runtime.is_relative_to(resolved_parent):
            raise SystemExit(
                f"refusing to provision: source prefix {source_runtime} is not "
                f"under required parent {resolved_parent}"
            )
    count = 0
    for _root, _dirs, files in os.walk(source_runtime):  # followlinks=False: cycle-safe
        count += len(files)
        if count > MAX_RUNTIME_FILES:
            raise SystemExit(
                f"refusing to provision: {source_runtime} holds more than "
                f"{MAX_RUNTIME_FILES} files — not a python runtime prefix"
            )


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


def _relocate_elf_rpaths(runtime: Path, source_runtime: Path, patchelf: str) -> int:
    """Rewrite every copied ELF whose RPATH/RUNPATH escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. RPATH (--force-rpath) rather than
    RUNPATH so LD_LIBRARY_PATH cannot outrank the pinned path either.
    """
    target_rpath = str(runtime / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file() or not _is_elf(path):
            continue
        probe = subprocess.run(
            [patchelf, "--print-rpath", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            continue  # not a patchable dynamic object (e.g. ET_REL)
        current = probe.stdout.strip()
        if not current:
            continue
        needs_rewrite = str(source_runtime) in current or any(
            component.startswith("/") and not component.startswith(str(runtime))
            for component in current.split(":")
            if component
        )
        if not needs_rewrite:
            continue
        subprocess.run(
            [patchelf, "--force-rpath", "--set-rpath", target_rpath, str(path)],
            check=True,
        )
        rewritten += 1
    return rewritten


def _assert_self_contained(
    runtime: Path, source_runtime: Path, interpreter: Path
) -> None:
    """Prove the launcher's interpreter runs entirely from the provisioned tree.

    Runs with an explicit minimal environment (no LD_LIBRARY_PATH) and, on
    Linux, requires every file mapping to stay out of the source prefix and
    every libpython mapping to live under the runtime.
    """
    probe_code = (
        "import json, sys\n"
        "maps = []\n"
        "try:\n"
        "    with open('/proc/self/maps') as handle:\n"
        "        for line in handle:\n"
        "            parts = line.split(None, 5)\n"
        "            if len(parts) == 6 and parts[5].startswith('/'):\n"
        "                maps.append(parts[5].strip())\n"
        "except FileNotFoundError:\n"
        "    pass\n"
        "print(json.dumps({\n"
        "    'base_prefix': sys.base_prefix,\n"
        "    'version': list(sys.version_info[:2]),\n"
        "    'maps': sorted(set(maps)),\n"
        "}))\n"
    )
    result = subprocess.run(
        [str(interpreter), "-I", "-S", "-c", probe_code],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    report = json.loads(result.stdout)
    if Path(report["base_prefix"]).resolve() != runtime.resolve():
        raise SystemExit(
            f"provisioned interpreter reports base_prefix {report['base_prefix']}, "
            f"expected {runtime}"
        )
    if tuple(report["version"]) != sys.version_info[:2]:
        raise SystemExit(
            f"provisioned interpreter is python {report['version']}, the "
            f"provisioning interpreter is {list(sys.version_info[:2])}"
        )
    if sys.platform == "linux":
        escaped = [m for m in report["maps"] if m.startswith(str(source_runtime))]
        if escaped:
            raise SystemExit(
                "provisioned interpreter still maps code from the source "
                f"prefix: {escaped}"
            )
        libpython = [m for m in report["maps"] if "libpython" in Path(m).name]
        stray = [m for m in libpython if not m.startswith(str(runtime))]
        if stray:
            raise SystemExit(f"libpython mapped outside the runtime: {stray}")


def provision(
    destination: Path,
    supervisor: Path,
    site_packages: Path,
    apply_root: str,
    eval_root: str,
    corpus_release_root: str,
    require_prefix_under: Path | None = None,
    patchelf: str | None = None,
) -> None:
    source_runtime = Path(sys.base_prefix).resolve(strict=True)
    source_interpreter = Path(sys.executable).resolve(strict=True)
    _assert_sane_source_prefix(source_runtime, require_prefix_under)
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
    shutil.rmtree(target_site_packages)
    shutil.copytree(site_packages.resolve(strict=True), target_site_packages)
    if sys.platform == "linux":
        if not patchelf:
            patchelf = shutil.which("patchelf")
            if patchelf is None:
                raise SystemExit(
                    "patchelf is required on linux to relocate the runtime "
                    "(install it or pass --patchelf)"
                )
        _relocate_elf_rpaths(runtime, source_runtime, patchelf)
    interpreter = runtime / source_interpreter.relative_to(source_runtime)
    _assert_self_contained(runtime, source_runtime, interpreter)
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
    parser.add_argument(
        "--require-prefix-under",
        type=Path,
        default=None,
        help="Refuse unless the invoking interpreter's prefix is under this "
        "directory (CI passes the tool-cache root).",
    )
    parser.add_argument(
        "--patchelf",
        default=None,
        help="patchelf binary for the linux RPATH relocation pass.",
    )
    args = parser.parse_args()
    provision(
        args.destination.resolve(),
        args.supervisor.resolve(),
        args.site_packages.resolve(),
        args.apply_root,
        args.eval_root,
        args.corpus_release_root,
        args.require_prefix_under,
        args.patchelf,
    )

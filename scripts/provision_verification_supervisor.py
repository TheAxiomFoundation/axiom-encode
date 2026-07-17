"""Build a protected, self-contained verification-supervisor execution tree.

Containment guarantee: after provisioning, the launcher interpreter resolves
every link-time dependency (RPATH, DT_NEEDED, loader) from inside the
provisioned runtime or root-owned system loader paths, executes no startup
code from the staged site-packages (.pth/sitecustomize are purged after
staging), and demonstrably maps nothing from the source prefix. Absolute
paths passed to dlopen() at run time by third-party code are out of scope of
static auditing; the supervisor's ownership/writability policy and the
minimal child environment bound that residual.
"""

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
# far past that is a wrong-prefix copy about to fill the disk. Overridable for
# legitimately larger local prefixes (--max-runtime-files).
DEFAULT_MAX_RUNTIME_FILES = 100_000

_ET_EXEC = 2
_ET_DYN = 3
_PT_DYNAMIC = 2

# Startup-code carriers that must not exist anywhere in the provisioned
# runtime, INCLUDING the staged site-packages (the launcher runs -I, which
# still executes site-packages .pth lines at interpreter start).
_FORBIDDEN_STARTUP_GLOBS = (
    "*.pth",
    "*.egg-link",
    "sitecustomize.py",
    "usercustomize.py",
    "pyvenv.cfg",
    "__editable__*",
)


def _count_files_copy_equivalent(source: Path, limit: int) -> None:
    """Bound the copy the way copytree(symlinks=False) will actually perform it.

    copytree with symlinks=False FOLLOWS symlinks, so the count must too, and
    a link escaping the source prefix (`x -> /usr`) must refuse rather than
    count (or copy) the outside world. Diamonds (`lib64 -> lib`) are counted
    twice exactly as copytree copies them twice; only a directory whose real
    path appears among its own traversal ancestors — copytree's infinite
    recursion — refuses.
    """
    source_resolved = source.resolve(strict=True)

    def walk(directory: Path, ancestors: frozenset[Path], count: int) -> int:
        real_directory = directory.resolve(strict=True)
        if real_directory in ancestors:
            raise SystemExit(
                f"refusing to provision: symlink cycle at {directory} in {source}"
            )
        ancestors = ancestors | {real_directory}
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                if entry.is_symlink():
                    target = entry_path.resolve(strict=False)
                    if not target.is_relative_to(source_resolved):
                        raise SystemExit(
                            "refusing to provision: symlink escapes the source "
                            f"prefix: {entry_path} -> {target}"
                        )
                if entry.is_dir(follow_symlinks=True):
                    count = walk(entry_path, ancestors, count)
                else:
                    count += 1
                    if count > limit:
                        raise SystemExit(
                            f"refusing to provision: {source} holds more than "
                            f"{limit} files — not a python runtime prefix"
                        )
        return count

    walk(source, frozenset(), 0)


def _assert_sane_source_prefix(
    source_runtime: Path, require_under: Path | None, max_files: int
) -> None:
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
    _count_files_copy_equivalent(source_runtime, max_files)


def _stage_runtime_tree(source_runtime: Path, runtime: Path, site_packages: Path) -> None:
    """Copy the interpreter prefix, swap in staged site-packages, then purge.

    The purge MUST run after the site-packages swap: the staged tree is
    pip-produced content and could itself carry .pth/sitecustomize startup
    hooks that the launcher's -I mode would execute.
    """
    shutil.copytree(source_runtime, runtime, symlinks=False)
    target_site_packages = (
        runtime
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    shutil.rmtree(target_site_packages)
    shutil.copytree(site_packages.resolve(strict=True), target_site_packages)
    for pattern in _FORBIDDEN_STARTUP_GLOBS:
        for forbidden in runtime.rglob(pattern):
            if forbidden.is_file():
                forbidden.unlink()


def _elf_header(path: Path) -> bytes | None:
    try:
        with path.open("rb") as handle:
            header = handle.read(64)
    except OSError:
        return None
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    return header


def _elf_is_dynamic(path: Path) -> bool:
    """True when the ELF participates in dynamic linking (has PT_DYNAMIC).

    Static ET_EXEC objects have no dynamic section for patchelf to read, and
    nothing for the loader to resolve, so probe failures on them are benign;
    on anything dynamic a probe failure must be fatal, not skipped.
    """
    header = _elf_header(path)
    if header is None:
        return False
    is_64 = header[4] == 2
    byteorder = "little" if header[5] == 1 else "big"
    e_type = int.from_bytes(header[16:18], byteorder)
    if e_type not in (_ET_EXEC, _ET_DYN):
        return False
    if not is_64:
        return True  # no 32-bit objects are expected; treat as dynamic (fatal path)
    e_phoff = int.from_bytes(header[32:40], byteorder)
    e_phentsize = int.from_bytes(header[54:56], byteorder)
    e_phnum = int.from_bytes(header[56:58], byteorder)
    try:
        with path.open("rb") as handle:
            handle.seek(e_phoff)
            table = handle.read(e_phentsize * e_phnum)
    except OSError:
        return True
    for index in range(e_phnum):
        entry = table[index * e_phentsize : (index + 1) * e_phentsize]
        if len(entry) >= 4 and int.from_bytes(entry[0:4], byteorder) == _PT_DYNAMIC:
            return True
    return False


def _component_escapes(component: str, runtime_resolved: Path) -> bool:
    return component.startswith("/") and not Path(component).resolve(
        strict=False
    ).is_relative_to(runtime_resolved)


def _rpath_escapes(rpath: str, runtime_resolved: Path) -> bool:
    return any(
        _component_escapes(component, runtime_resolved)
        for component in rpath.split(":")
        if component
    )


def _rewrite_rpath(rpath: str, runtime_resolved: Path, target: str) -> str:
    """Replace escaping components with the runtime lib dir, keeping the rest.

    $ORIGIN-relative components from manylinux wheels stay untouched and in
    order; only absolute components resolving outside the runtime collapse to
    the pinned target.
    """
    parts: list[str] = []
    for component in rpath.split(":"):
        if not component:
            continue
        replacement = (
            target if _component_escapes(component, runtime_resolved) else component
        )
        if replacement not in parts:
            parts.append(replacement)
    return ":".join(parts)


def _relocate_elf_rpaths(runtime: Path, patchelf: str) -> int:
    """Repin every copied dynamic ELF whose linkage escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. RPATH (--force-rpath) rather than
    RUNPATH so LD_LIBRARY_PATH cannot outrank the pinned path either. Absolute
    DT_NEEDED entries outside the runtime are refused outright (none are
    expected; system libraries are referenced by bare soname).
    """
    runtime_resolved = runtime.resolve(strict=True)
    target_rpath = str(runtime_resolved / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file() or _elf_header(path) is None:
            continue
        probe = subprocess.run(
            [patchelf, "--print-rpath", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode != 0:
            if _elf_is_dynamic(path):
                raise SystemExit(
                    f"patchelf could not inspect dynamic object {path}: "
                    f"{probe.stderr.strip()}"
                )
            continue
        needed = subprocess.run(
            [patchelf, "--print-needed", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        for entry in needed.stdout.split():
            if "/" in entry and _component_escapes(entry, runtime_resolved):
                raise SystemExit(
                    f"{path} declares an absolute dependency outside the "
                    f"runtime: {entry}"
                )
        current = probe.stdout.strip()
        if not current or not _rpath_escapes(current, runtime_resolved):
            continue
        subprocess.run(
            [
                patchelf,
                "--force-rpath",
                "--set-rpath",
                _rewrite_rpath(current, runtime_resolved, target_rpath),
                str(path),
            ],
            check=True,
        )
        rewritten += 1
    return rewritten


def _assert_self_contained(
    runtime: Path, source_runtime: Path, interpreter: Path
) -> None:
    """Prove the launcher's interpreter runs entirely from the provisioned tree.

    Runs the interpreter exactly as the launcher will (-I, site enabled, so a
    surviving .pth would execute and show up) with an explicit minimal
    environment (no LD_LIBRARY_PATH) and, on Linux, requires every file
    mapping to stay out of the source prefix and every libpython mapping to
    live under the runtime.
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
        [str(interpreter), "-I", "-c", probe_code],
        capture_output=True,
        text=True,
        check=True,
        env={"PATH": "/usr/bin:/bin"},
    )
    report = json.loads(result.stdout)
    runtime_resolved = runtime.resolve(strict=True)
    if Path(report["base_prefix"]).resolve() != runtime_resolved:
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
        escaped = [
            m for m in report["maps"] if Path(m).is_relative_to(source_runtime)
        ]
        if escaped:
            raise SystemExit(
                "provisioned interpreter still maps code from the source "
                f"prefix: {escaped}"
            )
        libpython = [m for m in report["maps"] if "libpython" in Path(m).name]
        stray = [
            m for m in libpython if not Path(m).is_relative_to(runtime_resolved)
        ]
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
    max_runtime_files: int = DEFAULT_MAX_RUNTIME_FILES,
) -> None:
    source_runtime = Path(sys.base_prefix).resolve(strict=True)
    source_interpreter = Path(sys.executable).resolve(strict=True)
    _assert_sane_source_prefix(source_runtime, require_prefix_under, max_runtime_files)
    runtime = destination / "python"
    _stage_runtime_tree(source_runtime, runtime, site_packages)
    if sys.platform == "linux":
        if not patchelf:
            patchelf = shutil.which("patchelf")
            if patchelf is None:
                raise SystemExit(
                    "patchelf is required on linux to relocate the runtime "
                    "(install it or pass --patchelf)"
                )
        _relocate_elf_rpaths(runtime, patchelf)
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
    parser.add_argument(
        "--max-runtime-files",
        type=int,
        default=DEFAULT_MAX_RUNTIME_FILES,
        help="Refuse prefixes holding more files than this (copy-equivalent "
        "count, following symlinks the way the copy will).",
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
        args.max_runtime_files,
    )

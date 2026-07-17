"""Build a protected, self-contained verification-supervisor execution tree.

Exact Linux containment checks performed (nothing broader is claimed):

* Source prefix: a system/FHS prefix, a prefix outside ``--require-prefix-under``,
  one exceeding the file cap, or one whose tree escapes itself via a symlink is
  refused BEFORE any copy.
* Startup code: every ``.pth``/``._pth``/``.egg-link``/``pyvenv.cfg``/
  ``pybuilddir.txt``/``__editable__*`` path-configuration carrier and every
  importable ``sitecustomize``/``usercustomize`` form (module, package dir,
  ``.pyc``, extension module) is purged AFTER the staged site-packages swap, so
  the launcher's ``-I`` (site-enabled) start executes none of them.
* Dynamic ELFs: each object's ``DT_RPATH``/``DT_RUNPATH`` entries must, after
  per-object ``$ORIGIN`` expansion and lexical normalization, be absolute and
  strictly inside the runtime; every other form (empty, cwd-relative, or
  escaping) is rewritten to the pinned ``<runtime>/lib`` as a forced ``DT_RPATH``
  (so ``LD_LIBRARY_PATH`` cannot outrank it). Any path-bearing ``DT_NEEDED`` is
  refused, ``PT_INTERP`` must be absolute and outside the (user-writable) source
  prefix, and a patchelf probe that fails on a genuinely dynamic object is fatal.
* Empirical backstop: the launcher interpreter is run under a minimal env and
  ``/proc/self/maps`` is required to map ``libpython`` and all code from inside
  the runtime and nothing from the source prefix.

Out of scope (bounded by the supervisor's root-ownership/writability policy and
the minimal child environment, not by this script): absolute paths passed to
``dlopen()`` at run time by third-party code, and ``DT_AUDIT``/``DT_DEPAUDIT``
audit hooks.
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
FORBIDDEN_PREFIXES = (
    "/",
    "/usr",
    "/usr/local",
    "/opt",
    "/etc",
    "/bin",
    "/sbin",
    "/lib",
)

# A real CPython prefix (toolcache or standalone build) is ~20k files. Anything
# far past that is a wrong-prefix copy about to fill the disk. Overridable for
# legitimately larger local prefixes (--max-runtime-files).
DEFAULT_MAX_RUNTIME_FILES = 100_000

_ET_EXEC = 2
_ET_DYN = 3
_PT_DYNAMIC = 2

# Startup-code carriers that must not exist anywhere in the provisioned
# runtime, INCLUDING the staged site-packages (the launcher runs -I, which
# still processes site-packages .pth lines and imports sitecustomize at start).
# CPython imports sitecustomize/usercustomize by MODULE NAME, so every
# importable form must be covered — a bare `sitecustomize.py` glob leaves the
# package dir, the .pyc, and the extension-module forms executing.
_FORBIDDEN_STARTUP_GLOBS = (
    "*.pth",
    "*._pth",
    "*.egg-link",
    "pyvenv.cfg",
    "pybuilddir.txt",
    "__editable__*",
    "sitecustomize",
    "sitecustomize.*",
    "usercustomize",
    "usercustomize.*",
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


def _stage_runtime_tree(
    source_runtime: Path, runtime: Path, site_packages: Path
) -> None:
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
            if forbidden.is_symlink() or forbidden.is_file():
                forbidden.unlink()
            elif forbidden.is_dir():
                # `sitecustomize/` as a package directory is importable too.
                shutil.rmtree(forbidden)


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


def _lexically_inside(path_text: str, runtime_resolved: Path) -> bool:
    """True iff the NORMALIZED (symlink-preserving) absolute path is inside.

    Deliberately does NOT call .resolve(): a component like
    ``/outside/alias`` whose target is a symlink into ``<runtime>/lib`` would
    pass a resolve()-only check yet still be an out-of-runtime, retargetable
    path the loader would read literally. Lexical normalization alone
    (``os.path.normpath``) collapses ``..`` without following links.
    """
    normalized = Path(os.path.normpath(path_text))
    return normalized.is_relative_to(runtime_resolved)


def _origin_expand(component: str, object_dir: Path) -> str:
    for token in ("${ORIGIN}", "$ORIGIN"):
        component = component.replace(token, str(object_dir))
    return component


def _rpath_component_inside(
    component: str, object_dir: Path, runtime_resolved: Path
) -> bool:
    """A DT_RPATH/DT_RUNPATH entry is safe only if, after $ORIGIN expansion, it
    is absolute and strictly inside the runtime.

    ``object_dir`` is the ELF's directory — during provisioning that IS its
    final runtime location (we provision in place), so $ORIGIN evaluation is
    exact. Empty, cwd-relative (no $ORIGIN, not absolute), and runtime-escaping
    entries are all unsafe.
    """
    if not component:
        return False
    expanded = _origin_expand(component, object_dir)
    if not expanded.startswith("/"):
        return False
    return _lexically_inside(expanded, runtime_resolved)


def _rewrite_rpath_for_object(
    rpath: str, object_dir: Path, runtime_resolved: Path, target: str
) -> tuple[str, bool]:
    """Keep safe components verbatim; collapse every unsafe one to the pinned
    lib dir. Returns (new_rpath, changed)."""
    parts: list[str] = []
    changed = False
    for component in rpath.split(":"):
        if _rpath_component_inside(component, object_dir, runtime_resolved):
            keep = component
        else:
            keep = target
            changed = True
        if keep not in parts:
            parts.append(keep)
    return ":".join(parts), changed


def _needed_is_pathbearing(entry: str) -> bool:
    """A legitimate DT_NEEDED is a bare soname; any '/' (absolute OR relative)
    makes the loader treat it as a path and is refused."""
    return "/" in entry


def _interp_trusted(interpreter: str, source_runtime: Path) -> bool:
    """PT_INTERP (the ELF program loader) must be an absolute path that is NOT
    inside the user-writable source prefix. The system loader
    (/lib64/ld-linux-*) satisfies this; a loader under the toolcache does not.
    """
    if not interpreter.startswith("/"):
        return False
    return not _lexically_inside(interpreter, source_runtime)


def _patchelf_query(
    patchelf: str, flag: str, path: Path
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [patchelf, flag, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )


def _relocate_elf_rpaths(runtime: Path, source_runtime: Path, patchelf: str) -> int:
    """Repin every copied dynamic ELF whose linkage escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. Every rpath entry is re-evaluated
    per object with $ORIGIN expansion; anything not provably inside is forced to
    the pinned <runtime>/lib as DT_RPATH (LD_LIBRARY_PATH cannot outrank it).
    Path-bearing DT_NEEDED and an untrusted PT_INTERP are refused; a patchelf
    probe failure on a genuinely dynamic object is fatal (never skipped open).
    """
    runtime_resolved = runtime.resolve(strict=True)
    source_resolved = source_runtime.resolve(strict=True)
    target_rpath = str(runtime_resolved / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file() or _elf_header(path) is None:
            continue
        dynamic = _elf_is_dynamic(path)
        rpath_probe = _patchelf_query(patchelf, "--print-rpath", path)
        if rpath_probe.returncode != 0:
            if dynamic:
                raise SystemExit(
                    f"patchelf could not read rpath of dynamic object {path}: "
                    f"{rpath_probe.stderr.strip()}"
                )
            continue
        needed_probe = _patchelf_query(patchelf, "--print-needed", path)
        if needed_probe.returncode != 0:
            if dynamic:
                raise SystemExit(
                    f"patchelf could not read DT_NEEDED of dynamic object {path}: "
                    f"{needed_probe.stderr.strip()}"
                )
        else:
            for entry in needed_probe.stdout.split():
                if _needed_is_pathbearing(entry):
                    raise SystemExit(
                        f"{path} declares a path-bearing DT_NEEDED: {entry}"
                    )
        interp_probe = _patchelf_query(patchelf, "--print-interpreter", path)
        if interp_probe.returncode == 0:
            interp = interp_probe.stdout.strip()
            if interp and not _interp_trusted(interp, source_resolved):
                raise SystemExit(
                    f"{path} names an untrusted program interpreter: {interp}"
                )
        current = rpath_probe.stdout.strip()
        if not current:
            continue
        new_rpath, changed = _rewrite_rpath_for_object(
            current, path.parent, runtime_resolved, target_rpath
        )
        if not changed:
            continue
        subprocess.run(
            [patchelf, "--force-rpath", "--set-rpath", new_rpath, str(path)],
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
    source_resolved = source_runtime.resolve(strict=True)
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
        # /proc/self/maps paths are kernel-canonical; compare against resolved
        # bases so a symlinked prefix component cannot desync the containment.
        escaped = [m for m in report["maps"] if Path(m).is_relative_to(source_resolved)]
        if escaped:
            raise SystemExit(
                "provisioned interpreter still maps code from the source "
                f"prefix: {escaped}"
            )
        libpython = [m for m in report["maps"] if "libpython" in Path(m).name]
        stray = [m for m in libpython if not Path(m).is_relative_to(runtime_resolved)]
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

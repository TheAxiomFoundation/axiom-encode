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
* Dynamic ELFs: the loader-facing metadata is parsed DIRECTLY from the program
  headers (``PT_INTERP``, ``PT_DYNAMIC``) and dynamic array — not via patchelf's
  section view, which a crafted object could desync from its segments; a
  non-64-bit-LE-v1 ELF is fatal, not skipped. Each object's ``DT_RPATH``/
  ``DT_RUNPATH`` entries must, after per-object ``$ORIGIN`` expansion, be
  absolute and — both lexically and after symlink resolution — strictly inside
  the runtime; every other form (empty, cwd-relative, or escaping) is rewritten
  to the pinned ``<runtime>/lib`` as a forced ``DT_RPATH`` (so ``LD_LIBRARY_PATH``
  cannot outrank it) and the object is re-parsed to confirm the result is clean.
  A path-bearing ``DT_NEEDED``/``DT_AUDIT``/``DT_DEPAUDIT``/``DT_FILTER``/
  ``DT_AUXILIARY`` must resolve inside the runtime (``$ORIGIN``-anchored is fine —
  CPython's own ``libpython3.so`` links that way); ``PT_INTERP`` must be an
  absolute, root-owned, non-writable system-loader path outside the source prefix.
* Empirical backstop: the launcher interpreter is run under a minimal env and
  ``/proc/self/maps`` is required to map ``libpython`` and all code from inside
  the runtime and nothing from the source prefix.

Out of scope (bounded by the supervisor's root-ownership/writability policy and
the minimal child environment, not by this script): absolute paths a third-party
library passes to ``dlopen()`` at run time — these are not encoded in any header
this script can read.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass, field
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
    # CPython puts the canonical <prefix>/lib/pythonXY.zip on sys.path
    # unconditionally (even when absent), and -I still lets `site` import
    # modules — including sitecustomize — from it. Remove it if the copy carried
    # one; the full lib/ tree we copied makes it redundant.
    stdlib_zip = (
        runtime / "lib" / f"python{sys.version_info.major}{sys.version_info.minor}.zip"
    )
    if stdlib_zip.is_file():
        stdlib_zip.unlink()
    # Materialize every match BEFORE deleting: rglob is a lazy walker, and
    # rmtree-ing a directory it has yielded but not yet descended into would
    # make the walk fail. Collect first, then delete, guarding against a path
    # already removed by an earlier (ancestor) deletion.
    doomed = sorted(
        {
            match
            for pattern in _FORBIDDEN_STARTUP_GLOBS
            for match in runtime.rglob(pattern)
        }
    )
    for forbidden in doomed:
        if forbidden.is_symlink() or forbidden.is_file():
            forbidden.unlink()
        elif forbidden.is_dir():
            # `sitecustomize/` as a package directory is importable too.
            shutil.rmtree(forbidden)


# The audit reads the exact bytes the LOADER consumes — program headers
# (PT_INTERP, PT_DYNAMIC) and the dynamic array — rather than trusting
# patchelf's section-based view, which a crafted object could desync from its
# segments. patchelf is used ONLY to rewrite, after which the object is
# re-parsed to confirm the loader-facing result is clean.
_PT_LOAD = 1
_PT_DYNAMIC = 2
_PT_INTERP = 3
_DT_NULL = 0
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_RPATH = 15
_DT_RUNPATH = 29
_DT_AUDIT = 0x6FFFFEF8
_DT_DEPAUDIT = 0x6FFFFEF7
_DT_FILTER = 0x7FFFFFFF
_DT_AUXILIARY = 0x7FFFFFFD
# String-table-offset tags whose values are paths the loader may act on.
_DT_STR_PATH_TAGS = {
    _DT_NEEDED: "DT_NEEDED",
    _DT_RPATH: "DT_RPATH",
    _DT_RUNPATH: "DT_RUNPATH",
    _DT_AUDIT: "DT_AUDIT",
    _DT_DEPAUDIT: "DT_DEPAUDIT",
    _DT_FILTER: "DT_FILTER",
    _DT_AUXILIARY: "DT_AUXILIARY",
}
# gABI: $ORIGIN / ${ORIGIN}; a bare $ORIGIN token ends at a non-identifier
# char, so "$ORIGIN_evil" is NOT the token and stays unexpanded (→ unsafe).
_ORIGIN_TOKEN = re.compile(r"\$(?:\{ORIGIN\}|ORIGIN(?![0-9A-Z_a-z]))")
# Root-owned system directories a program interpreter may legitimately live in.
_SYSTEM_LOADER_DIRS = ("/lib", "/lib64", "/usr/lib", "/usr/lib64")


@dataclass
class _ElfInfo:
    is_dynamic: bool
    interp: str | None = None
    # tag name -> ordered raw string values as the loader would read them
    dyn_strings: dict[str, list[str]] = field(default_factory=dict)


def _parse_elf(path: Path) -> _ElfInfo | None:
    """Parse the loader-facing ELF metadata, or return None for a non-ELF file.

    Only 64-bit little-endian ELF v1 is supported; any other class/data/version
    is fatal rather than silently skipped (a flipped EI_DATA on a native object
    would otherwise make a naive parser miss segments the kernel still reads).
    """
    data = path.read_bytes()
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    ei_class, ei_data, ei_version = data[4], data[5], data[6]
    if ei_class != 2:
        raise SystemExit(f"{path}: unsupported ELF class {ei_class} (only 64-bit)")
    if ei_data != 1:
        raise SystemExit(f"{path}: unsupported ELF data encoding {ei_data} (only LE)")
    if ei_version != 1:
        raise SystemExit(f"{path}: unsupported ELF version {ei_version}")

    def at(offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > len(data):
            raise SystemExit(f"{path}: ELF truncated (need {size} bytes at {offset})")
        return data[offset : offset + size]

    e_phoff = struct.unpack_from("<Q", data, 32)[0]
    e_phentsize = struct.unpack_from("<H", data, 54)[0]
    e_phnum = struct.unpack_from("<H", data, 56)[0]
    # ET_REL objects (a stdlib `.o` such as config-*/python.o) carry no program
    # header table — no PT_INTERP, no PT_DYNAMIC — so the loader never acts on
    # them. No segments means nothing to audit.
    if e_phnum == 0:
        return _ElfInfo(is_dynamic=False)
    if e_phentsize < 56:
        raise SystemExit(f"{path}: implausible e_phentsize {e_phentsize}")
    loads: list[tuple[int, int, int]] = []  # (vaddr, offset, filesz)
    dyn_span: tuple[int, int] | None = None
    interp: str | None = None
    for index in range(e_phnum):
        phdr = at(e_phoff + index * e_phentsize, 56)
        p_type = struct.unpack_from("<I", phdr, 0)[0]
        p_offset, p_vaddr = struct.unpack_from("<QQ", phdr, 8)
        p_filesz = struct.unpack_from("<Q", phdr, 32)[0]
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == _PT_DYNAMIC:
            dyn_span = (p_offset, p_filesz)
        elif p_type == _PT_INTERP:
            interp = (
                at(p_offset, p_filesz)
                .split(b"\x00", 1)[0]
                .decode("utf-8", "surrogateescape")
            )
    if dyn_span is None:
        return _ElfInfo(is_dynamic=False, interp=interp)

    def vaddr_to_offset(vaddr: int) -> int:
        for seg_vaddr, seg_offset, seg_filesz in loads:
            if seg_vaddr <= vaddr < seg_vaddr + seg_filesz:
                return seg_offset + (vaddr - seg_vaddr)
        raise SystemExit(f"{path}: DT string vaddr {vaddr:#x} outside every PT_LOAD")

    dyn_offset, dyn_size = dyn_span
    entries: list[tuple[int, int]] = []
    strtab_vaddr: int | None = None
    cursor = dyn_offset
    while cursor + 16 <= dyn_offset + dyn_size:
        d_tag, d_val = struct.unpack_from("<qQ", data, cursor)
        cursor += 16
        if d_tag == _DT_NULL:
            break
        entries.append((d_tag, d_val))
        if d_tag == _DT_STRTAB:
            strtab_vaddr = d_val
    dyn_strings: dict[str, list[str]] = {}
    if strtab_vaddr is not None:
        strtab_offset = vaddr_to_offset(strtab_vaddr)

        def read_string(value_offset: int) -> str:
            start = strtab_offset + value_offset
            end = data.find(b"\x00", start)
            if start < 0 or start > len(data) or end < 0:
                raise SystemExit(f"{path}: unterminated DT string at {value_offset}")
            return data[start:end].decode("utf-8", "surrogateescape")

        for d_tag, d_val in entries:
            name = _DT_STR_PATH_TAGS.get(d_tag)
            if name is not None:
                dyn_strings.setdefault(name, []).append(read_string(d_val))
    return _ElfInfo(is_dynamic=True, interp=interp, dyn_strings=dyn_strings)


def _path_inside(path_text: str, runtime_resolved: Path) -> bool:
    """Inside iff BOTH the lexically normalized path AND the symlink-resolved
    path stay within the runtime.

    Lexical-only accepts ``/runtime/alias/../lib`` where ``alias`` is a symlink
    pointing outside (the loader follows the link before ``..``, landing out);
    resolve-only accepts ``/outside/alias`` symlinked into the runtime (the
    loader reads the literal, retargetable path). Requiring both closes each.
    """
    normalized = Path(os.path.normpath(path_text))
    if not normalized.is_relative_to(runtime_resolved):
        return False
    resolved = Path(path_text).resolve(strict=False)
    return resolved.is_relative_to(runtime_resolved)


def _origin_expand(component: str, object_dir: Path) -> str:
    return _ORIGIN_TOKEN.sub(str(object_dir), component)


def _rpath_component_inside(
    component: str, object_dir: Path, runtime_resolved: Path
) -> bool:
    """A DT_RPATH/DT_RUNPATH (or path-bearing DT_NEEDED) entry is safe only if,
    after per-object $ORIGIN expansion, it is absolute and — both lexically and
    after symlink resolution — strictly inside the runtime.

    ``object_dir`` is the ELF's directory; we provision in place at the final
    location, so $ORIGIN evaluation is exact. Empty, cwd-relative, and escaping
    entries are all unsafe.
    """
    if not component:
        return False
    expanded = _origin_expand(component, object_dir)
    if not expanded.startswith("/"):
        return False
    return _path_inside(expanded, runtime_resolved)


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
    """A bare soname (no '/') is resolved via the search path; a '/'-bearing
    DT_NEEDED is a literal path the loader opens directly and must be audited
    like an rpath component (CPython's own libpython3.so uses
    ``$ORIGIN/../lib/…`` legitimately, so path-bearing is not itself a fault)."""
    return "/" in entry


def _interp_trusted(interpreter: str, source_resolved: Path) -> bool:
    """PT_INTERP (the program loader) must be an absolute path whose lexical AND
    resolved forms sit under a root-owned system loader directory and outside
    the user-writable source prefix; if it exists it must be root-owned and not
    group/other-writable. The system loader (/lib64/ld-linux-*) qualifies; a
    loader under the toolcache, /tmp, or a symlink into the source does not.
    """
    if not interpreter.startswith("/"):
        return False
    lexical = Path(os.path.normpath(interpreter))
    resolved = Path(interpreter).resolve(strict=False)
    for candidate in (lexical, resolved):
        if not any(candidate.is_relative_to(d) for d in _SYSTEM_LOADER_DIRS):
            return False
        if candidate.is_relative_to(source_resolved):
            return False
    try:
        status = os.stat(resolved)
    except OSError:
        return True  # not present in this environment; the allowlist stands
    return status.st_uid == 0 and not (status.st_mode & 0o022)


def _audit_dynamic_elf(
    path: Path,
    info: _ElfInfo,
    object_dir: Path,
    runtime_resolved: Path,
    source_resolved: Path,
) -> None:
    """Refuse an object whose loader-facing metadata would pull code from
    outside the runtime, before any rewrite."""
    if info.interp is not None and not _interp_trusted(info.interp, source_resolved):
        raise SystemExit(
            f"{path} names an untrusted program interpreter: {info.interp}"
        )
    for tag in ("DT_NEEDED", "DT_AUDIT", "DT_DEPAUDIT", "DT_FILTER", "DT_AUXILIARY"):
        for entry in info.dyn_strings.get(tag, []):
            if _needed_is_pathbearing(entry) and not _rpath_component_inside(
                entry, object_dir, runtime_resolved
            ):
                raise SystemExit(f"{path} declares an escaping {tag}: {entry}")


def _combined_rpath(info: _ElfInfo) -> str:
    """The loader honours DT_RPATH then DT_RUNPATH; audit them together as one
    colon list (patchelf collapses both into a single forced DT_RPATH)."""
    return ":".join(
        info.dyn_strings.get("DT_RPATH", []) + info.dyn_strings.get("DT_RUNPATH", [])
    )


def _relocate_elf_rpaths(runtime: Path, source_runtime: Path, patchelf: str) -> int:
    """Repin every copied dynamic ELF whose linkage escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. Each object's loader-facing metadata
    is parsed directly and audited; escaping DT_RPATH/DT_RUNPATH is rewritten to
    the pinned <runtime>/lib as a forced DT_RPATH (LD_LIBRARY_PATH cannot
    outrank it), then the object is re-parsed to confirm the result is clean.
    """
    runtime_resolved = runtime.resolve(strict=True)
    source_resolved = source_runtime.resolve(strict=True)
    target_rpath = str(runtime_resolved / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        info = _parse_elf(path)
        if info is None or not info.is_dynamic:
            continue
        _audit_dynamic_elf(path, info, path.parent, runtime_resolved, source_resolved)
        combined = _combined_rpath(info)
        if not combined:
            continue
        new_rpath, changed = _rewrite_rpath_for_object(
            combined, path.parent, runtime_resolved, target_rpath
        )
        if not changed:
            continue
        subprocess.run(
            [patchelf, "--force-rpath", "--set-rpath", new_rpath, str(path)],
            check=True,
        )
        # Re-read from the loader's view: the rewrite must have taken and left
        # nothing escaping (a decoy section could make patchelf write a DT_RPATH
        # the segments don't reflect).
        after = _parse_elf(path)
        if after is None:
            raise SystemExit(f"{path}: no longer parses as ELF after relocation")
        _, still_escapes = _rewrite_rpath_for_object(
            _combined_rpath(after), path.parent, runtime_resolved, target_rpath
        )
        if still_escapes:
            raise SystemExit(
                f"{path}: rpath still escapes the runtime after relocation: "
                f"{_combined_rpath(after)!r}"
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

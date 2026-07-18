"""Build a protected verification-supervisor execution tree.

The load-bearing guarantee is EMPIRICAL: the provisioned launcher interpreter is
run under a minimal environment and ``/proc/self/maps`` is required to show it
maps its OWN binary from the root-owned runtime, maps NO file from the
(user-writable) source prefix, and pins any mapped ``libpython`` to the runtime —
even under a hostile ``LD_LIBRARY_PATH``. This bounds the launcher interpreter's
own load-time closure; it does NOT assert that every mapping is inside the runtime
(the root-owned system loader and libc legitimately map from ``/lib``, ``/usr/lib``).
The check observes exactly what the loader does (interpreter, every mapped library,
audit hooks, ``$ORIGIN`` expansion) with no dependence on hand-parsed ELF metadata.
The static passes below make that outcome reachable and durable:

* Source prefix: a system/FHS prefix, a prefix outside ``--require-prefix-under``,
  one exceeding the file cap, or one whose tree escapes itself via a symlink is
  refused BEFORE any copy.
* Startup code: every ``.pth``/``._pth``/``.egg-link``/``pyvenv.cfg``/
  ``pybuilddir.txt``/``__editable__*`` path-configuration carrier, the canonical
  stdlib ``pythonXY[t].zip``, and every importable ``sitecustomize``/
  ``usercustomize`` form (module, package dir, ``.pyc``, extension module) is
  purged AFTER the staged site-packages swap, so the launcher's ``-I``
  (site-enabled) start executes none of them.
* Run paths: every copied ELF whose ``DT_RPATH``/``DT_RUNPATH`` (read via
  ``patchelf --print-rpath``) escapes the runtime after per-object ``$ORIGIN``
  expansion — requiring both lexical and symlink-resolved containment — is
  repinned to ``<runtime>/lib`` as a forced ``DT_RPATH`` (which ``LD_LIBRARY_PATH``
  cannot outrank), then re-read to confirm nothing still escapes. This neutralizes
  the toolchain's absolute build-prefix RUNPATH so ``libpython`` loads from the
  runtime, and is the pass the maps check then verifies for the launcher.

Trust boundary and residual: the inputs are the GitHub-provided toolcache stdlib
and pinned pip dependencies, and the finished tree is chowned root and stripped
of group/other write by the caller (the supervisor re-verifies that ownership).
A pinned dependency carrying hostile ELF metadata (an absolute ``DT_NEEDED``, a
``DT_AUDIT`` hook, a decoy section) is NOT statically audited here — such a
dependency already executes its own code inside the signer, so metadata-level
auditing adds no protection beyond the dependency-provenance trust already
extended, and the run-path repin plus root-ownership bound the writable-path
vectors that do matter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
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
    # CPython puts the canonical stdlib zip on sys.path unconditionally (even
    # when absent), and -I still lets `site` import modules — including
    # sitecustomize — from it. Its name/dir vary with PLATLIBDIR (lib vs lib64)
    # and the free-threaded ABI ("t" suffix, e.g. python313t.zip), so match the
    # version-prefixed zip anywhere in the copied tree; the full lib/ tree we
    # copied makes it redundant.
    zip_glob = f"python{sys.version_info.major}{sys.version_info.minor}*.zip"
    for stdlib_zip in runtime.rglob(zip_glob):
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


# gABI: $ORIGIN / ${ORIGIN}; a bare $ORIGIN token ends at a non-identifier
# char, so "$ORIGIN_evil" is NOT the token and stays unexpanded (→ unsafe).
_ORIGIN_TOKEN = re.compile(r"\$(?:\{ORIGIN\}|ORIGIN(?![0-9A-Z_a-z]))")


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"\x7fELF"
    except OSError:
        return False


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


def _print_rpath(patchelf: str, path: Path) -> str | None:
    """Return the object's combined DT_RPATH/DT_RUNPATH, or None if patchelf
    cannot read it (a non-dynamic object such as a `.o`, or one with no run
    path). patchelf collapses RPATH and RUNPATH into a single reported value.
    """
    result = subprocess.run(
        [patchelf, "--print-rpath", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    # patchelf appends exactly one trailing newline; a leading space would be
    # part of a (malformed) run-path value, so strip only that newline.
    return result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout


def _relocate_elf_rpaths(runtime: Path, patchelf: str) -> int:
    """Repin every copied ELF whose DT_RPATH/DT_RUNPATH escapes the runtime.

    The hosted-toolchain CPython is built --enable-shared with an absolute
    RUNPATH into its build prefix, so a byte-copy keeps loading libpython from
    the ORIGINAL (user-writable) location. Each escaping run-path component
    (after per-object $ORIGIN expansion, requiring both lexical and resolved
    containment) is rewritten to the pinned <runtime>/lib as a forced DT_RPATH,
    which LD_LIBRARY_PATH cannot outrank; patchelf --print-rpath then re-reads
    the object to confirm nothing still escapes.

    This is a best-effort hardening pass over trusted, pinned inputs (the
    toolcache stdlib and pinned pip deps). The load-bearing guarantee for the
    launcher interpreter is the empirical /proc/self/maps check in
    _assert_self_contained, not this static rewrite.
    """
    runtime_resolved = runtime.resolve(strict=True)
    target_rpath = str(runtime_resolved / "lib")
    rewritten = 0
    for path in sorted(runtime.rglob("*")):
        if path.is_symlink() or not path.is_file() or not _is_elf(path):
            continue
        current = _print_rpath(patchelf, path)
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
        after = _print_rpath(patchelf, path)
        _, still_escapes = _rewrite_rpath_for_object(
            after or "", path.parent, runtime_resolved, target_rpath
        )
        if still_escapes:
            raise SystemExit(
                f"{path}: run-path still escapes the runtime after relocation: "
                f"{after!r}"
            )
        rewritten += 1
    return rewritten


def _assert_self_contained(
    runtime: Path, source_runtime: Path, interpreter: Path
) -> None:
    """Prove the launcher's interpreter loads nothing from the source prefix and
    pins libpython to the runtime.

    Runs the interpreter exactly as the launcher will (-I, site enabled, so a
    surviving .pth would execute and show up) with an explicit minimal
    environment (no LD_LIBRARY_PATH). On Linux it then requires, from a
    non-empty /proc/self/maps: the provisioned interpreter's own binary is
    mapped (non-vacuity — proves the runtime interpreter actually ran, for both
    shared and static-libpython builds), NO file mapping under the
    (user-writable) source prefix, and every libpython mapping (if any) under
    the runtime. It deliberately does NOT require every mapping to be inside the
    runtime — the root-owned system loader and libc legitimately map from /lib,
    /usr/lib, etc. This bounds the launcher interpreter's own load-time closure.
    C-extensions the encoder imports later are OUTSIDE this probe: the run-path
    repin neutralizes their escaping DT_RPATH/DT_RUNPATH, but an absolute
    DT_NEEDED or DT_AUDIT in one of them is not covered here and remains bounded
    by the pinned-dependency provenance boundary documented below.

    The launcher is a `#!<interpreter> -I` shebang, and Linux truncates the
    shebang interpreter at whitespace, so a destination path containing
    whitespace would make the launcher select a different interpreter than this
    execve-based probe validated. Refuse such a path up front.
    """
    if any(character.isspace() for character in str(interpreter)):
        raise SystemExit(
            f"interpreter path is not shebang-safe (contains whitespace): {interpreter}"
        )
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
        # Fail closed: an empty maps list means the probe observed nothing (a
        # kernel without /proc, a hardened readfile), so the containment claim
        # is unverified rather than satisfied.
        if not report["maps"]:
            raise SystemExit(
                "self-containment probe observed no /proc/self/maps entries; "
                "cannot verify the provisioned runtime"
            )
        # Non-vacuity, build-agnostic: the probe MUST have mapped this exact
        # interpreter binary from the runtime, proving it ran the provisioned
        # interpreter rather than passing on an empty/foreign map set. This holds
        # for both --enable-shared (libpython.so) and static-libpython builds
        # (e.g. python-build-standalone), where there is no shared libpython to
        # require.
        interpreter_resolved = str(interpreter.resolve(strict=True))
        if interpreter_resolved not in report["maps"]:
            raise SystemExit(
                "self-containment probe did not map the provisioned interpreter "
                f"{interpreter_resolved}; observed {sorted(report['maps'])[:5]}"
            )
        # /proc/self/maps paths are kernel-canonical; compare against resolved
        # bases so a symlinked prefix component cannot desync the containment.
        escaped = [m for m in report["maps"] if Path(m).is_relative_to(source_resolved)]
        if escaped:
            raise SystemExit(
                "provisioned interpreter still maps code from the source "
                f"prefix: {escaped}"
            )
        # Any libpython that IS mapped (shared builds) must come from the runtime;
        # a static build maps none, which is fine — libpython is baked into the
        # root-owned interpreter binary already required above.
        libpython = [m for m in report["maps"] if "libpython" in Path(m).name]
        stray = [m for m in libpython if not Path(m).is_relative_to(runtime_resolved)]
        if stray:
            raise SystemExit(f"libpython mapped outside the runtime: {stray}")


def _resolve_trusted_git(raw_git: Path) -> Path:
    """Admit one immutable, root-owned Git executable by exact path."""

    if not raw_git.is_absolute() or Path(os.path.normpath(raw_git)) != raw_git:
        raise SystemExit("trusted git path must be absolute and normalized")
    cursor = Path(raw_git.anchor)
    for part in raw_git.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise SystemExit(f"trusted git path contains a symlink: {raw_git}")
    try:
        resolved = raw_git.resolve(strict=True)
    except OSError as exc:
        raise SystemExit(f"trusted git path does not exist: {raw_git}") from exc
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode) or not os.access(resolved, os.X_OK):
        raise SystemExit(f"trusted git path is not an executable file: {resolved}")
    for component in (resolved, *resolved.parents):
        component_metadata = component.stat()
        if component_metadata.st_uid != 0:
            raise SystemExit(f"trusted git path is not root-owned: {component}")
        if component_metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise SystemExit(
                f"trusted git path is group- or other-writable: {component}"
            )
    if metadata.st_mode & (stat.S_ISUID | stat.S_ISGID):
        raise SystemExit("trusted git executable must not be setuid or setgid")
    return resolved


def _install_trusted_git_wrapper(
    destination: Path, interpreter: Path, trusted_git: Path
) -> Path:
    """Install a read-only Git command broker into the isolated PATH."""

    wrapper = destination / "git"
    wrapper.write_text(
        f"#!{interpreter} -I\n"
        "import os\n"
        "import re\n"
        "import subprocess\n"
        "import sys\n"
        f"git = {str(trusted_git)!r}\n"
        "args = sys.argv[1:]\n"
        "cursor = 0\n"
        "if args[:1] == ['-C']:\n"
        "    if len(args) < 3:\n"
        "        raise SystemExit('trusted git wrapper requires a command after -C')\n"
        "    cursor = 2\n"
        "allowed = {\n"
        "    'cat-file', 'diff', 'log', 'ls-files', 'ls-tree', 'merge-base',\n"
        "    'remote', 'rev-list', 'rev-parse', 'show', 'status',\n"
        "}\n"
        "if cursor >= len(args) or args[cursor] not in allowed:\n"
        "    selected = args[cursor] if cursor < len(args) else '<missing>'\n"
        "    raise SystemExit(f'trusted git wrapper refused command: {selected}')\n"
        "command = args[cursor]\n"
        "options = args[cursor + 1:]\n"
        "def safe_ref(value):\n"
        "    return re.fullmatch(\n"
        "        r'(?:HEAD|main|origin/main|[0-9a-f]{40})(?:\\^\\{commit\\})?',\n"
        "        value,\n"
        "    ) is not None\n"
        "def safe_range(value):\n"
        "    parts = value.split('..')\n"
        "    return len(parts) == 2 and all(safe_ref(part) for part in parts)\n"
        "def safe_path(value):\n"
        "    return (\n"
        "        bool(value)\n"
        "        and not value.startswith(('-', ':', '/'))\n"
        "        and all(part not in {'', '.', '..'} for part in value.split('/'))\n"
        "    )\n"
        "def allowed_options():\n"
        "    if command == 'cat-file':\n"
        "        return options == ['--batch'] or (\n"
        "            len(options) == 2\n"
        "            and options[0] == 'blob'\n"
        "            and re.fullmatch(r'[0-9a-f]{40}', options[1]) is not None\n"
        "        )\n"
        "    if command == 'diff':\n"
        "        if (\n"
        "            options[:3] == ['--binary', 'HEAD', '--']\n"
        "            and all(safe_path(value) for value in options[3:])\n"
        "        ):\n"
        "            return True\n"
        "        changed_prefix = [\n"
        "            '--name-only', '--no-renames', '--diff-filter=ACDMRT', '-z'\n"
        "        ]\n"
        "        if options[:4] == changed_prefix:\n"
        "            return len(options[4:]) in {1, 2} and all(\n"
        "                safe_ref(value) for value in options[4:]\n"
        "            )\n"
        "        return (\n"
        "            options[:2] == ['--name-only', '--diff-filter=ACMRT']\n"
        "            and len(options) == 3\n"
        "            and safe_range(options[2])\n"
        "        )\n"
        "    if command == 'log':\n"
        "        return options == [\n"
        "            '--format=%H', '--', 'pyproject.toml',\n"
        "            'src/axiom_encode/__init__.py', 'uv.lock',\n"
        "        ]\n"
        "    if command == 'ls-files':\n"
        "        if options == ['-z']:\n"
        "            return True\n"
        "        prefix = ['--others', '--exclude-standard', '-z']\n"
        "        if options == prefix:\n"
        "            return True\n"
        "        return (\n"
        "            options[:4] == [*prefix, '--']\n"
        "            and all(safe_path(value) for value in options[4:])\n"
        "        )\n"
        "    if command == 'ls-tree':\n"
        "        return (\n"
        "            len(options) == 4\n"
        "            and options[0] == '-z'\n"
        "            and safe_ref(options[1])\n"
        "            and options[2] == '--'\n"
        "            and safe_path(options[3])\n"
        "        ) or (\n"
        "            len(options) == 3\n"
        "            and options[:2] == ['-r', '-z']\n"
        "            and safe_ref(options[2])\n"
        "        )\n"
        "    if command == 'merge-base':\n"
        "        return (\n"
        "            len(options) == 3\n"
        "            and options[0] == '--is-ancestor'\n"
        "            and all(safe_ref(value) for value in options[1:])\n"
        "        )\n"
        "    if command == 'remote':\n"
        "        return options == ['get-url', 'origin']\n"
        "    if command == 'rev-list':\n"
        "        return (\n"
        "            len(options) == 4\n"
        "            and options[:3] == ['--parents', '-n', '1']\n"
        "            and safe_ref(options[-1])\n"
        "        )\n"
        "    if command == 'rev-parse':\n"
        "        if options in (\n"
        "            ['HEAD'], ['--is-inside-work-tree'], ['--show-prefix'],\n"
        "            ['--show-toplevel'],\n"
        "        ):\n"
        "            return True\n"
        "        return (\n"
        "            len(options) == 2\n"
        "            and options[0] == '--verify'\n"
        "            and safe_ref(options[1])\n"
        "        )\n"
        "    if command == 'show':\n"
        "        if len(options) != 1 or ':' not in options[0]:\n"
        "            return False\n"
        "        ref, path = options[0].split(':', 1)\n"
        "        return safe_ref(ref) and safe_path(path)\n"
        "    if command == 'status':\n"
        "        return options in (\n"
        "            ['--porcelain'], ['--porcelain', '--untracked-files=no'],\n"
        "        )\n"
        "    return False\n"
        "if not allowed_options():\n"
        "    raise SystemExit(f'trusted git wrapper refused arguments for {command}')\n"
        "for name in tuple(os.environ):\n"
        "    if name.startswith('GIT_') or name == 'SSH_ASKPASS':\n"
        "        os.environ.pop(name, None)\n"
        "os.environ['GIT_ATTR_NOSYSTEM'] = '1'\n"
        "os.environ['GIT_CONFIG_GLOBAL'] = '/dev/null'\n"
        "os.environ['GIT_CONFIG_NOSYSTEM'] = '1'\n"
        "os.environ['GIT_OPTIONAL_LOCKS'] = '0'\n"
        "os.environ['GIT_TERMINAL_PROMPT'] = '0'\n"
        "trusted = [\n"
        "    git, '--no-pager',\n"
        "    '-c', 'core.fsmonitor=false',\n"
        "    '-c', 'core.hooksPath=/dev/null',\n"
        "    '-c', 'credential.helper=',\n"
        "    '-c', 'core.attributesFile=/dev/null',\n"
        "    '-c', 'core.sshCommand=false',\n"
        "    '-c', 'protocol.ext.allow=never',\n"
        "    '-c', 'diff.external=/usr/bin/false',\n"
        "    '-c', 'submodule.recurse=false',\n"
        "]\n"
        "def worktree_sensitive():\n"
        "    if command == 'status':\n"
        "        return True\n"
        "    if command != 'diff':\n"
        "        return False\n"
        "    if options[:3] == ['--binary', 'HEAD', '--']:\n"
        "        return True\n"
        "    prefix = [\n"
        "        '--name-only', '--no-renames', '--diff-filter=ACDMRT', '-z'\n"
        "    ]\n"
        "    return options[:4] == prefix and len(options) == 5\n"
        "def assert_no_worktree_filters():\n"
        "    repository = args[:cursor]\n"
        "    try:\n"
        "        listed = subprocess.run(\n"
        "            [*trusted, *repository, 'ls-files', '-co',\n"
        "             '--exclude-standard', '-z'],\n"
        "            check=True, capture_output=True, env=os.environ,\n"
        "        ).stdout\n"
        "        if not listed:\n"
        "            return\n"
        "        checked = subprocess.run(\n"
        "            [*trusted, *repository, 'check-attr', '-z', '--stdin',\n"
        "             'filter'],\n"
        "            input=listed, check=True, capture_output=True, env=os.environ,\n"
        "        ).stdout\n"
        "    except (OSError, subprocess.CalledProcessError) as error:\n"
        "        raise SystemExit(\n"
        "            'trusted git wrapper could not audit worktree filters'\n"
        "        ) from error\n"
        "    records = checked.split(b'\\0')\n"
        "    if records and records[-1] == b'':\n"
        "        records.pop()\n"
        "    if len(records) % 3:\n"
        "        raise SystemExit('trusted git wrapper received invalid attributes')\n"
        "    for index in range(0, len(records), 3):\n"
        "        path, attribute, value = records[index:index + 3]\n"
        "        if attribute != b'filter' or value in {b'unspecified', b'unset'}:\n"
        "            continue\n"
        "        display = os.fsdecode(path)\n"
        "        raise SystemExit(\n"
        "            f'trusted git wrapper refused worktree filter for {display}'\n"
        "        )\n"
        "if worktree_sensitive():\n"
        "    assert_no_worktree_filters()\n"
        "if command == 'diff':\n"
        "    args[cursor + 1:cursor + 1] = ['--no-ext-diff', '--no-textconv']\n"
        "os.execv(git, [*trusted, *args])\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def provision(
    destination: Path,
    supervisor: Path,
    site_packages: Path,
    apply_root: str,
    eval_root: str,
    corpus_release_root: str,
    git: Path,
    require_prefix_under: Path | None = None,
    patchelf: str | None = None,
    max_runtime_files: int = DEFAULT_MAX_RUNTIME_FILES,
) -> None:
    trusted_git = _resolve_trusted_git(git)
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
    _install_trusted_git_wrapper(destination, interpreter, trusted_git)
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
        "--git",
        type=Path,
        required=True,
        help="Exact root-owned Git executable exposed inside the isolated PATH.",
    )
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
        args.git,
        args.require_prefix_under,
        args.patchelf,
        args.max_runtime_files,
    )

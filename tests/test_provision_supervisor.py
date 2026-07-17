"""Unit coverage for the provisioner's containment logic.

The root/ELF end-to-end paths run on the hosted image in
.github/workflows/provision-selftest.yml; these tests pin the pure logic —
prefix-boundary predicates, rpath rewriting, staging order, and the
copy-equivalent preflight — with no root or patchelf required.
"""

from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "provision_verification_supervisor",
    Path(__file__).resolve().parent.parent
    / "scripts"
    / "provision_verification_supervisor.py",
)
provisioner = importlib.util.module_from_spec(_SPEC)
# Register before exec so @dataclass can resolve the module via sys.modules.
sys.modules[_SPEC.name] = provisioner
_SPEC.loader.exec_module(provisioner)


class TestPathInside:
    def test_absolute_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._path_inside("/opt/hostedtoolcache/x/lib", runtime)

    def test_inside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        assert provisioner._path_inside(str(runtime / "lib"), runtime)

    def test_sibling_prefix_collision_outside(self, tmp_path):
        """<runtime>-evil must not pass as inside (the startswith trap)."""
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._path_inside(str(tmp_path / "runtime-evil"), runtime)

    def test_dotdot_traversal_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        sneaky = str(runtime / "lib" / ".." / ".." / "outside")
        assert not provisioner._path_inside(sneaky, runtime)

    def test_outside_alias_symlinked_into_runtime_is_outside(self, tmp_path):
        """/outside/alias -> <runtime>/lib resolves inside, but the loader reads
        the literal (retargetable) path: lexical check keeps it out."""
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        alias = tmp_path / "outside" / "alias"
        alias.parent.mkdir()
        alias.symlink_to(runtime / "lib")
        assert not provisioner._path_inside(str(alias), runtime)

    def test_inside_alias_symlinked_out_is_outside(self, tmp_path):
        """<runtime>/alias/../lib normalizes lexically to <runtime>/lib, but the
        loader follows `alias` (to outside) BEFORE `..`: resolve check keeps it
        out. This is the case lexical-only would wrongly accept."""
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (runtime / "alias").symlink_to(outside)
        assert not provisioner._path_inside(
            str(runtime / "alias" / ".." / "lib"), runtime
        )


class TestRpathComponentInside:
    def _runtime(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        return runtime, runtime / "lib"

    def test_origin_relative_staying_inside_is_safe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert provisioner._rpath_component_inside("$ORIGIN", object_dir, runtime)
        assert provisioner._rpath_component_inside(
            "$ORIGIN/../lib", object_dir, runtime
        )
        assert provisioner._rpath_component_inside("${ORIGIN}/foo", object_dir, runtime)

    def test_origin_climbing_out_is_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "$ORIGIN/../../../../usr/lib", object_dir, runtime
        )

    def test_bogus_origin_token_not_expanded(self, tmp_path):
        """$ORIGIN_evil is not the gABI token (identifier char follows), so it
        stays unexpanded, is not absolute, and is therefore unsafe — not
        silently accepted as a rewritten <dir>_evil path."""
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "$ORIGIN_evil", object_dir, runtime
        )

    def test_empty_and_cwd_relative_are_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside("", object_dir, runtime)
        assert not provisioner._rpath_component_inside("lib", object_dir, runtime)
        assert not provisioner._rpath_component_inside(".", object_dir, runtime)
        # a leading-space value must not be silently trimmed into an abs path
        assert not provisioner._rpath_component_inside(" /x", object_dir, runtime)

    def test_absolute_outside_is_unsafe(self, tmp_path):
        runtime, object_dir = self._runtime(tmp_path)
        assert not provisioner._rpath_component_inside(
            "/opt/hostedtoolcache/Python/3.14.0/x64/lib", object_dir, runtime
        )


class TestOriginExpand:
    def test_only_real_token_expands(self, tmp_path):
        object_dir = Path("/rt/lib")
        assert provisioner._origin_expand("$ORIGIN/x", object_dir) == "/rt/lib/x"
        assert provisioner._origin_expand("${ORIGIN}/x", object_dir) == "/rt/lib/x"
        # identifier char after ORIGIN → not the token, left verbatim
        assert provisioner._origin_expand("$ORIGIN_evil", object_dir) == "$ORIGIN_evil"
        assert provisioner._origin_expand("$ORIGINX", object_dir) == "$ORIGINX"


class TestRewriteRpathForObject:
    def test_mixed_list_rewrites_only_unsafe_preserving_origin(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        mixed = "$ORIGIN/../lib:/opt/hostedtoolcache/Python/3.14.0/x64/lib"
        new_rpath, changed = provisioner._rewrite_rpath_for_object(
            mixed, object_dir, runtime, target
        )
        assert changed
        assert new_rpath == f"$ORIGIN/../lib:{target}"

    def test_all_safe_list_is_unchanged(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        _, changed = provisioner._rewrite_rpath_for_object(
            "$ORIGIN:$ORIGIN/../lib", object_dir, runtime, target
        )
        assert not changed

    def test_empty_and_escaping_collapse_deduplicated(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        target = str(runtime / "lib")
        # empty component + two escapes + one origin-safe → target once, then origin
        new_rpath, changed = provisioner._rewrite_rpath_for_object(
            ":/one/outside:/two/outside:$ORIGIN", object_dir, runtime, target
        )
        assert changed
        assert new_rpath == f"{target}:$ORIGIN"


class TestNeededPathbearing:
    def test_bare_soname_is_not_pathbearing(self):
        assert not provisioner._needed_is_pathbearing("libc.so.6")
        assert not provisioner._needed_is_pathbearing("libpython3.14.so.1.0")

    def test_slash_bearing_is_pathbearing(self):
        assert provisioner._needed_is_pathbearing("/opt/evil/libx.so")
        assert provisioner._needed_is_pathbearing("./libx.so")
        assert provisioner._needed_is_pathbearing("sub/libx.so")
        # CPython's own libpython3.so ships this — path-bearing but $ORIGIN-safe;
        # _audit_dynamic_elf allows it via _rpath_component_inside, not here.
        assert provisioner._needed_is_pathbearing("$ORIGIN/../lib/libpython3.13.so.1.0")


class TestInterpTrusted:
    def test_system_loader_dirs_trusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        # non-existent paths under allowlisted dirs pass on the lexical allowlist
        assert provisioner._interp_trusted("/lib64/ld-linux-x86-64.so.2", source)
        assert provisioner._interp_trusted("/usr/lib/ld-2.99.so", source)

    def test_non_allowlisted_dir_untrusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        assert not provisioner._interp_trusted("/tmp/evil-ld.so", source)
        assert not provisioner._interp_trusted("/opt/evil/ld.so", source)

    def test_loader_under_source_prefix_untrusted(self, tmp_path):
        # even if it were under an allowlisted dir, being inside the source
        # (user-writable) prefix disqualifies it
        source = (tmp_path / "usr" / "lib" / "python").resolve()
        source.mkdir(parents=True)
        assert not provisioner._interp_trusted(str(source / "ld.so"), source)

    def test_relative_loader_untrusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        assert not provisioner._interp_trusted("ld.so", source)

    def test_existing_but_writable_loader_untrusted(self, tmp_path, monkeypatch):
        source = (tmp_path / "toolcache").resolve()
        source.mkdir()

        # An allowlisted, existing loader that is world-writable → untrusted.
        # stat() runs on the RESOLVED path, which on usrmerge systems differs
        # from the literal (/lib64 -> /usr/lib64), so intercept unconditionally.
        class FakeStat:
            st_uid = 0
            st_mode = 0o100777  # regular file, world-writable

        monkeypatch.setattr(provisioner.os, "stat", lambda *a, **k: FakeStat())
        assert not provisioner._interp_trusted("/lib64/ld-linux-x86-64.so.2", source)

    def test_existing_root_owned_loader_trusted(self, tmp_path, monkeypatch):
        source = (tmp_path / "toolcache").resolve()
        source.mkdir()

        class FakeStat:
            st_uid = 0
            st_mode = 0o100755  # regular file, not group/other writable

        monkeypatch.setattr(provisioner.os, "stat", lambda *a, **k: FakeStat())
        assert provisioner._interp_trusted("/lib64/ld-linux-x86-64.so.2", source)


class TestStageRuntimeTree:
    def _make_source(self, tmp_path: Path) -> Path:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        source = tmp_path / "source"
        site = source / "lib" / version / "site-packages"
        site.mkdir(parents=True)
        (source / "bin").mkdir()
        (source / "bin" / "python3").write_text("fake interpreter")
        (site / "toolcache-only.txt").write_text("replaced wholesale")
        return source

    def test_staged_startup_hooks_are_purged(self, tmp_path):
        """Every importable/path-config startup carrier in the STAGED tree must
        be gone: the launcher runs -I (site enabled), which executes them. A
        bare `sitecustomize.py` glob would leave the package dir, .pyc, and
        extension-module forms importable, and ._pth/pybuilddir.txt reconfigure
        sys.path."""
        source = self._make_source(tmp_path)
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "evil.pth").write_text("import os; os.system('pwned')")
        (staged / "python._pth").write_text("import site")
        (staged / "pybuilddir.txt").write_text("build")
        (staged / "sitecustomize.py").write_text("print('pwned')")
        (staged / "sitecustomize.pyc").write_bytes(b"\x00")
        (staged / "sitecustomize.cpython-314-x86_64-linux-gnu.so").write_bytes(b"\x00")
        (staged / "usercustomize.py").write_text("print('pwned')")
        pkg = staged / "sitecustomize"  # importable package form
        pkg.mkdir()
        (pkg / "__init__.py").write_text("import os; os.system('pwned')")
        # A carrier nested INSIDE a doomed dir: exercises parent-then-child
        # deletion (the guard must skip the already-removed child, not raise).
        (pkg / "nested.pth").write_text("import os")
        (staged / "yaml.py").write_text("legit = True")
        runtime = tmp_path / "dest" / "python"
        runtime.parent.mkdir()
        provisioner._stage_runtime_tree(source, runtime, staged)
        for orphan in (
            "*.pth",
            "*._pth",
            "pybuilddir.txt",
            "sitecustomize*",
            "usercustomize*",
        ):
            assert not list(runtime.rglob(orphan)), orphan
        assert not any(p.name == "sitecustomize" for p in runtime.rglob("*"))
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        site = runtime / "lib" / version / "site-packages"
        assert (site / "yaml.py").read_text() == "legit = True"
        assert not (site / "toolcache-only.txt").exists()


class TestSanePrefixPreflight:
    def test_forbidden_system_prefix_refused(self):
        with pytest.raises(SystemExit, match="not a self-contained"):
            provisioner._assert_sane_source_prefix(Path("/usr"), None, 100)

    def test_prefix_outside_required_parent_refused(self, tmp_path):
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        required = tmp_path / "elsewhere"
        required.mkdir()
        with pytest.raises(SystemExit, match="not under required parent"):
            provisioner._assert_sane_source_prefix(prefix, required, 100)

    def test_file_cap_refuses(self, tmp_path):
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        for index in range(5):
            (prefix / f"file{index}").write_text("x")
        with pytest.raises(SystemExit, match="more than 3 files"):
            provisioner._assert_sane_source_prefix(prefix, None, 3)

    def test_symlink_escaping_prefix_refused(self, tmp_path):
        """copytree(symlinks=False) would follow this into the outside world,
        so the copy-equivalent count must refuse it up front."""
        prefix = tmp_path / "prefix"
        prefix.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "huge").write_text("x")
        (prefix / "leak").symlink_to(outside)
        with pytest.raises(SystemExit, match="symlink escapes the source prefix"):
            provisioner._assert_sane_source_prefix(prefix, None, 100)

    def test_internal_diamond_counted_like_the_copy(self, tmp_path):
        """lib64 -> lib is copied twice by copytree, not a cycle: it must pass
        (and count double, exactly like the copy it bounds)."""
        prefix = tmp_path / "prefix"
        lib = prefix / "lib"
        lib.mkdir(parents=True)
        (lib / "libpython.so").write_text("x")
        (prefix / "lib64").symlink_to(lib)
        provisioner._assert_sane_source_prefix(prefix, None, 100)
        with pytest.raises(SystemExit, match="more than 1 files"):
            provisioner._assert_sane_source_prefix(prefix, None, 1)

    def test_ancestor_cycle_refused(self, tmp_path):
        prefix = tmp_path / "prefix"
        inner = prefix / "inner"
        inner.mkdir(parents=True)
        (inner / "loop").symlink_to(prefix)
        with pytest.raises(SystemExit, match="symlink cycle"):
            provisioner._assert_sane_source_prefix(prefix, None, 100)


def _build_elf(
    *,
    dyn=(),
    interp=None,
    include_dynamic=True,
    ei_class=2,
    ei_data=1,
    ei_version=1,
    e_type=3,
):
    """Assemble a minimal but structurally valid 64-bit LE ELF.

    `dyn` is a list of (tag, value); a str value is placed in the string table
    and referenced by offset, an int value is used raw. A single PT_LOAD maps
    the whole file identity (vaddr == offset) so DT_STRTAB resolves trivially.
    """
    n_ph = 1 + (1 if interp is not None else 0) + (1 if include_dynamic else 0)
    ph_off = 64
    body_off = ph_off + 56 * n_ph
    cursor = body_off
    interp_bytes = b""
    interp_off = interp_sz = 0
    if interp is not None:
        interp_bytes = interp.encode() + b"\x00"
        interp_off, interp_sz = cursor, len(interp_bytes)
        cursor += interp_sz

    strtab = bytearray(b"\x00")
    resolved = []
    for tag, value in dyn:
        if isinstance(value, str):
            offset = len(strtab)
            strtab += value.encode() + b"\x00"
            resolved.append((tag, offset))
        else:
            resolved.append((tag, value))

    dyn_bytes = b""
    dyn_off = dyn_size = strtab_off = 0
    if include_dynamic:
        dyn_off = cursor
        dyn_size = (len(resolved) + 2) * 16  # + DT_STRTAB + DT_NULL
        strtab_off = dyn_off + dyn_size
        for tag, val in resolved:
            dyn_bytes += struct.pack("<qQ", tag, val)
        dyn_bytes += struct.pack("<qQ", 5, strtab_off)  # DT_STRTAB (vaddr==offset)
        dyn_bytes += struct.pack("<qQ", 0, 0)  # DT_NULL
        file_len = strtab_off + len(strtab)
    else:
        file_len = cursor

    program_headers = struct.pack(
        "<IIQQQQQQ", 1, 5, 0, 0, 0, file_len, file_len, 0x1000
    )  # PT_LOAD, whole file
    if interp is not None:
        program_headers += struct.pack(
            "<IIQQQQQQ", 3, 4, interp_off, interp_off, 0, interp_sz, interp_sz, 1
        )  # PT_INTERP
    if include_dynamic:
        program_headers += struct.pack(
            "<IIQQQQQQ", 2, 6, dyn_off, dyn_off, 0, dyn_size, dyn_size, 8
        )  # PT_DYNAMIC

    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4], ehdr[5], ehdr[6] = ei_class, ei_data, ei_version
    struct.pack_into("<H", ehdr, 16, e_type)
    struct.pack_into("<Q", ehdr, 32, ph_off)
    struct.pack_into("<H", ehdr, 54, 56)
    struct.pack_into("<H", ehdr, 56, n_ph)

    out = bytearray(file_len)
    out[0:64] = ehdr
    out[ph_off : ph_off + len(program_headers)] = program_headers
    if interp is not None:
        out[interp_off : interp_off + len(interp_bytes)] = interp_bytes
    if include_dynamic:
        out[dyn_off : dyn_off + len(dyn_bytes)] = dyn_bytes
        out[strtab_off : strtab_off + len(strtab)] = strtab
    return bytes(out)


# DT tag numbers used by the tests (mirror the module's private constants).
_NEEDED, _RPATH, _RUNPATH, _AUDIT = 1, 15, 29, 0x6FFFFEF8


class TestParseElf:
    def test_non_elf_is_none(self, tmp_path):
        path = tmp_path / "script"
        path.write_bytes(b"#!/bin/sh\necho hi\n")
        assert provisioner._parse_elf(path) is None

    def test_static_object_is_not_dynamic(self, tmp_path):
        path = tmp_path / "static"
        path.write_bytes(_build_elf(include_dynamic=False))
        info = provisioner._parse_elf(path)
        assert info is not None and not info.is_dynamic

    def test_relocatable_object_without_program_headers(self, tmp_path):
        """A stdlib .o (ET_REL, config-*/python.o) has no program header table
        (e_phnum == e_phentsize == 0); the loader never acts on it, so it must
        parse as non-dynamic rather than tripping the phentsize sanity check."""
        elf = bytearray(_build_elf(include_dynamic=False))
        struct.pack_into("<H", elf, 16, 1)  # e_type = ET_REL
        struct.pack_into("<Q", elf, 32, 0)  # e_phoff = 0
        struct.pack_into("<H", elf, 54, 0)  # e_phentsize = 0
        struct.pack_into("<H", elf, 56, 0)  # e_phnum = 0
        path = tmp_path / "python.o"
        path.write_bytes(bytes(elf))
        info = provisioner._parse_elf(path)
        assert info is not None and not info.is_dynamic

    def test_reads_needed_rpath_and_interp(self, tmp_path):
        path = tmp_path / "lib.so"
        path.write_bytes(
            _build_elf(
                dyn=[
                    (_NEEDED, "libc.so.6"),
                    (_RUNPATH, "$ORIGIN/../lib:/opt/toolcache/lib"),
                ],
                interp="/lib64/ld-linux-x86-64.so.2",
            )
        )
        info = provisioner._parse_elf(path)
        assert info.is_dynamic
        assert info.interp == "/lib64/ld-linux-x86-64.so.2"
        assert info.dyn_strings["DT_NEEDED"] == ["libc.so.6"]
        assert info.dyn_strings["DT_RUNPATH"] == ["$ORIGIN/../lib:/opt/toolcache/lib"]

    def test_flipped_ei_data_is_fatal(self, tmp_path):
        """A native x86-64 object with EI_DATA flipped to big-endian: the kernel
        still reads the native headers, so a naive parser skipping it would miss
        the audit. Must be fatal, not skipped."""
        path = tmp_path / "flipped"
        path.write_bytes(_build_elf(dyn=[(_NEEDED, "libc.so.6")], ei_data=2))
        with pytest.raises(SystemExit, match="unsupported ELF data encoding"):
            provisioner._parse_elf(path)

    def test_unsupported_class_is_fatal(self, tmp_path):
        path = tmp_path / "elf32"
        path.write_bytes(_build_elf(ei_class=1))
        with pytest.raises(SystemExit, match="unsupported ELF class"):
            provisioner._parse_elf(path)


class TestAuditDynamicElf:
    def _runtime(self, tmp_path):
        runtime = (tmp_path / "python").resolve()
        (runtime / "lib").mkdir(parents=True)
        return runtime

    def test_origin_safe_pathbearing_needed_passes(self, tmp_path):
        runtime = self._runtime(tmp_path)
        obj = runtime / "lib" / "libpython3.so"
        # CPython's own stub: path-bearing NEEDED, but $ORIGIN-anchored inside.
        info = provisioner._ElfInfo(
            is_dynamic=True,
            dyn_strings={"DT_NEEDED": ["$ORIGIN/../lib/libpython3.13.so.1.0"]},
        )
        provisioner._audit_dynamic_elf(obj, info, obj.parent, runtime, runtime)

    def test_escaping_needed_refused(self, tmp_path):
        runtime = self._runtime(tmp_path)
        obj = runtime / "lib" / "x.so"
        info = provisioner._ElfInfo(
            is_dynamic=True, dyn_strings={"DT_NEEDED": ["/opt/evil/libx.so"]}
        )
        with pytest.raises(SystemExit, match="escaping DT_NEEDED"):
            provisioner._audit_dynamic_elf(obj, info, obj.parent, runtime, runtime)

    def test_escaping_audit_hook_refused(self, tmp_path):
        runtime = self._runtime(tmp_path)
        obj = runtime / "lib" / "x.so"
        info = provisioner._ElfInfo(
            is_dynamic=True, dyn_strings={"DT_AUDIT": ["/opt/evil/audit.so"]}
        )
        with pytest.raises(SystemExit, match="escaping DT_AUDIT"):
            provisioner._audit_dynamic_elf(obj, info, obj.parent, runtime, runtime)

    def test_untrusted_interp_refused(self, tmp_path):
        runtime = self._runtime(tmp_path)
        source = (tmp_path / "toolcache").resolve()
        source.mkdir()
        obj = runtime / "lib" / "python3"
        info = provisioner._ElfInfo(is_dynamic=True, interp="/tmp/evil-ld.so")
        with pytest.raises(SystemExit, match="untrusted program interpreter"):
            provisioner._audit_dynamic_elf(obj, info, obj.parent, runtime, source)


class TestRelocateElfRpaths:
    def _runtime_with_elf(self, tmp_path, rpath):
        runtime = (tmp_path / "python").resolve()
        (runtime / "lib").mkdir(parents=True)
        obj = runtime / "lib" / "libpython.so"
        obj.write_bytes(_build_elf(dyn=[(_NEEDED, "libc.so.6"), (_RPATH, rpath)]))
        return runtime, obj

    def test_noop_rewrite_is_caught_by_reparse(self, tmp_path, monkeypatch):
        """If patchelf were a no-op (or a decoy left the segment unchanged), the
        re-parse must catch that the escaping rpath is still present."""
        runtime, _ = self._runtime_with_elf(
            tmp_path, "/opt/hostedtoolcache/Python/3.14/x64/lib"
        )
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "", ""),
        )
        with pytest.raises(SystemExit, match="still escapes the runtime"):
            provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")

    def test_origin_only_rpath_needs_no_rewrite(self, tmp_path, monkeypatch):
        runtime, _ = self._runtime_with_elf(tmp_path, "$ORIGIN")

        def forbid(*a, **k):
            raise AssertionError("patchelf must not be called for a clean object")

        monkeypatch.setattr(provisioner.subprocess, "run", forbid)
        assert provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf") == 0

    # The real-patchelf rewrite round-trip is NOT unit-tested with a synthetic
    # ELF on purpose: patchelf operates on ELF *sections*, which the hand-built
    # program-header-only fixtures here deliberately lack (that section/segment
    # split is exactly what the direct parser defends against). The authoritative
    # coverage for the real rewrite is provision-selftest.yml, which relocates a
    # genuine escaping RUNPATH in the toolchain libpython on hosted Ubuntu and
    # confirms via /proc/self/maps that libpython loads from inside the runtime.

"""Unit coverage for the provisioner's containment logic.

The root/ELF end-to-end paths run on the hosted image in
.github/workflows/provision-selftest.yml; these tests pin the pure logic —
prefix-boundary predicates, rpath rewriting, staging order, and the
copy-equivalent preflight — with no root or patchelf required.
"""

from __future__ import annotations

import importlib.util
import struct
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
_SPEC.loader.exec_module(provisioner)


class TestComponentEscapes:
    def test_absolute_outside_escapes(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        assert provisioner._component_escapes("/opt/hostedtoolcache/x/lib", runtime)

    def test_inside_does_not_escape(self, tmp_path):
        runtime = tmp_path / "runtime"
        (runtime / "lib").mkdir(parents=True)
        assert not provisioner._component_escapes(str(runtime / "lib"), runtime)

    def test_sibling_prefix_collision_escapes(self, tmp_path):
        """<runtime>-evil must not pass as inside (the startswith trap)."""
        runtime = tmp_path / "runtime"
        evil = tmp_path / "runtime-evil"
        runtime.mkdir()
        evil.mkdir()
        assert provisioner._component_escapes(str(evil), runtime)

    def test_dotdot_traversal_escapes(self, tmp_path):
        runtime = tmp_path / "runtime"
        (runtime / "lib").mkdir(parents=True)
        (tmp_path / "outside").mkdir()
        sneaky = str(runtime / "lib" / ".." / ".." / "outside")
        assert provisioner._component_escapes(sneaky, runtime)

    def test_origin_relative_is_not_absolute(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        assert not provisioner._component_escapes("$ORIGIN/../lib", runtime)


class TestRpathRewrite:
    def test_pure_origin_list_never_flagged(self, tmp_path):
        runtime = tmp_path / "runtime"
        runtime.mkdir()
        assert not provisioner._rpath_escapes("$ORIGIN/../lib:$ORIGIN", runtime)

    def test_mixed_list_flagged_and_rewritten_preserving_origin(self, tmp_path):
        runtime = tmp_path / "runtime"
        (runtime / "lib").mkdir(parents=True)
        target = str(runtime / "lib")
        mixed = "$ORIGIN/../lib:/opt/hostedtoolcache/Python/3.14.0/x64/lib"
        assert provisioner._rpath_escapes(mixed, runtime)
        assert (
            provisioner._rewrite_rpath(mixed, runtime, target)
            == f"$ORIGIN/../lib:{target}"
        )

    def test_multiple_escapes_collapse_deduplicated(self, tmp_path):
        runtime = tmp_path / "runtime"
        (runtime / "lib").mkdir(parents=True)
        target = str(runtime / "lib")
        rpath = "/one/outside:/two/outside:$ORIGIN"
        assert provisioner._rewrite_rpath(rpath, runtime, target) == f"{target}:$ORIGIN"


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
        """A .pth or sitecustomize in the STAGED tree must not survive: the
        launcher runs -I (site enabled), which would execute them."""
        source = self._make_source(tmp_path)
        staged = tmp_path / "staged"
        staged.mkdir()
        (staged / "evil.pth").write_text("import os; os.system('pwned')")
        (staged / "sitecustomize.py").write_text("print('pwned')")
        (staged / "yaml.py").write_text("legit = True")
        runtime = tmp_path / "dest" / "python"
        runtime.parent.mkdir()
        provisioner._stage_runtime_tree(source, runtime, staged)
        assert not list(runtime.rglob("*.pth"))
        assert not list(runtime.rglob("sitecustomize.py"))
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


def _elf64(e_type: int, phdr_types: list[int]) -> bytes:
    ehdr = bytearray(64)
    ehdr[0:4] = b"\x7fELF"
    ehdr[4] = 2  # 64-bit
    ehdr[5] = 1  # little-endian
    struct.pack_into("<H", ehdr, 16, e_type)
    struct.pack_into("<Q", ehdr, 32, 64)  # e_phoff: right after the header
    struct.pack_into("<H", ehdr, 54, 56)  # e_phentsize
    struct.pack_into("<H", ehdr, 56, len(phdr_types))  # e_phnum
    body = b"".join(struct.pack("<I", p_type) + bytes(52) for p_type in phdr_types)
    return bytes(ehdr) + body


class TestElfIsDynamic:
    def test_dynamic_executable_detected(self, tmp_path):
        path = tmp_path / "dyn"
        path.write_bytes(_elf64(provisioner._ET_DYN, [1, provisioner._PT_DYNAMIC]))
        assert provisioner._elf_is_dynamic(path)

    def test_static_executable_not_dynamic(self, tmp_path):
        path = tmp_path / "static"
        path.write_bytes(_elf64(provisioner._ET_EXEC, [1, 1]))
        assert not provisioner._elf_is_dynamic(path)

    def test_non_elf_ignored(self, tmp_path):
        path = tmp_path / "script"
        path.write_bytes(b"#!/bin/sh\necho hi\n")
        assert not provisioner._elf_is_dynamic(path)

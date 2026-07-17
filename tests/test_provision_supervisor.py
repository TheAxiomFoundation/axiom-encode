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
_SPEC.loader.exec_module(provisioner)


class TestLexicallyInside:
    def test_absolute_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._lexically_inside("/opt/hostedtoolcache/x/lib", runtime)

    def test_inside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        assert provisioner._lexically_inside(str(runtime / "lib"), runtime)

    def test_sibling_prefix_collision_outside(self, tmp_path):
        """<runtime>-evil must not pass as inside (the startswith trap)."""
        runtime = (tmp_path / "runtime").resolve()
        assert not provisioner._lexically_inside(
            str(tmp_path / "runtime-evil"), runtime
        )

    def test_dotdot_traversal_outside(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        sneaky = str(runtime / "lib" / ".." / ".." / "outside")
        assert not provisioner._lexically_inside(sneaky, runtime)

    def test_symlink_alias_into_runtime_still_outside(self, tmp_path):
        """The blocking round-3 case: /outside/alias -> <runtime>/lib resolves
        inside, but the loader reads the literal (retargetable) path, so a
        resolve()-only check would wrongly accept it. Lexical stays outside."""
        runtime = (tmp_path / "runtime").resolve()
        (runtime / "lib").mkdir(parents=True)
        alias = tmp_path / "outside" / "alias"
        alias.parent.mkdir()
        alias.symlink_to(runtime / "lib")
        assert not provisioner._lexically_inside(str(alias), runtime)


class TestRpathComponentInside:
    def test_origin_relative_staying_inside_is_safe(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        assert provisioner._rpath_component_inside("$ORIGIN", object_dir, runtime)
        assert provisioner._rpath_component_inside(
            "$ORIGIN/../lib", object_dir, runtime
        )
        assert provisioner._rpath_component_inside("${ORIGIN}/foo", object_dir, runtime)

    def test_origin_climbing_out_is_unsafe(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        assert not provisioner._rpath_component_inside(
            "$ORIGIN/../../../../usr/lib", object_dir, runtime
        )

    def test_empty_and_cwd_relative_are_unsafe(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        assert not provisioner._rpath_component_inside("", object_dir, runtime)
        assert not provisioner._rpath_component_inside("lib", object_dir, runtime)
        assert not provisioner._rpath_component_inside(".", object_dir, runtime)

    def test_absolute_outside_is_unsafe(self, tmp_path):
        runtime = (tmp_path / "runtime").resolve()
        object_dir = runtime / "lib"
        object_dir.mkdir(parents=True)
        assert not provisioner._rpath_component_inside(
            "/opt/hostedtoolcache/Python/3.14.0/x64/lib", object_dir, runtime
        )


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


class TestNeededAndInterp:
    def test_bare_soname_allowed(self):
        assert not provisioner._needed_is_pathbearing("libc.so.6")
        assert not provisioner._needed_is_pathbearing("libpython3.14.so.1.0")

    def test_path_bearing_needed_refused(self):
        assert provisioner._needed_is_pathbearing("/opt/evil/libx.so")
        assert provisioner._needed_is_pathbearing("./libx.so")
        assert provisioner._needed_is_pathbearing("sub/libx.so")

    def test_system_loader_is_trusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        assert provisioner._interp_trusted("/lib64/ld-linux-x86-64.so.2", source)
        assert provisioner._interp_trusted("/usr/lib/ld.so", source)

    def test_loader_under_source_prefix_untrusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        assert not provisioner._interp_trusted(str(source / "lib" / "ld.so"), source)

    def test_relative_loader_untrusted(self, tmp_path):
        source = (tmp_path / "toolcache" / "python").resolve()
        source.mkdir(parents=True)
        assert not provisioner._interp_trusted("ld.so", source)


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


def _fake_patchelf(responses):
    """Build a subprocess.run stand-in keyed by patchelf flag.

    `responses[flag]` is (returncode, stdout). --set-rpath records the value.
    """
    recorded = {"set_rpath": []}

    def run(cmd, *args, **kwargs):
        if "--set-rpath" in cmd:
            recorded["set_rpath"].append(cmd[cmd.index("--set-rpath") + 1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        flag = cmd[1]
        rc, out = responses.get(flag, (0, ""))
        return subprocess.CompletedProcess(cmd, rc, out, "boom" if rc else "")

    run.recorded = recorded
    return run


class TestRelocateElfAudit:
    """Drive the real _relocate_elf_rpaths with a stubbed patchelf so the audit
    branches (fail-open guards, DT_NEEDED, PT_INTERP) are covered without a
    real toolchain."""

    def _runtime_with_dynamic_elf(self, tmp_path):
        runtime = (tmp_path / "python").resolve()
        (runtime / "lib").mkdir(parents=True)
        elf = runtime / "lib" / "libpython.so"
        elf.write_bytes(_elf64(provisioner._ET_DYN, [provisioner._PT_DYNAMIC]))
        return runtime

    def test_print_needed_failure_on_dynamic_object_is_fatal(
        self, tmp_path, monkeypatch
    ):
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf(
            {
                "--print-rpath": (0, ""),
                "--print-needed": (1, ""),
                "--print-interpreter": (1, ""),
            }
        )
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="could not read DT_NEEDED"):
            provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")

    def test_path_bearing_needed_refused(self, tmp_path, monkeypatch):
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf(
            {
                "--print-rpath": (0, ""),
                "--print-needed": (0, "libc.so.6\n/opt/evil/libx.so\n"),
                "--print-interpreter": (1, ""),
            }
        )
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="path-bearing DT_NEEDED"):
            provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")

    def test_untrusted_interpreter_refused(self, tmp_path, monkeypatch):
        source = (tmp_path / "toolcache").resolve()
        source.mkdir()
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf(
            {
                "--print-rpath": (0, ""),
                "--print-needed": (0, ""),
                "--print-interpreter": (0, str(source / "ld.so")),
            }
        )
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="untrusted program interpreter"):
            provisioner._relocate_elf_rpaths(runtime, source, "patchelf")

    def test_print_rpath_failure_on_dynamic_object_is_fatal(
        self, tmp_path, monkeypatch
    ):
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf({"--print-rpath": (1, "")})
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        with pytest.raises(SystemExit, match="could not read rpath"):
            provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")

    def test_escaping_rpath_is_rewritten(self, tmp_path, monkeypatch):
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf(
            {
                "--print-rpath": (0, "/opt/hostedtoolcache/Python/3.14.0/x64/lib"),
                "--print-needed": (0, "libc.so.6"),
                "--print-interpreter": (0, "/lib64/ld-linux-x86-64.so.2"),
            }
        )
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        count = provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")
        assert count == 1
        assert fake.recorded["set_rpath"] == [str(runtime / "lib")]

    def test_origin_only_rpath_is_left_alone(self, tmp_path, monkeypatch):
        runtime = self._runtime_with_dynamic_elf(tmp_path)
        fake = _fake_patchelf(
            {
                "--print-rpath": (0, "$ORIGIN"),
                "--print-needed": (0, "libc.so.6"),
                "--print-interpreter": (1, ""),
            }
        )
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        count = provisioner._relocate_elf_rpaths(runtime, runtime, "patchelf")
        assert count == 0
        assert fake.recorded["set_rpath"] == []

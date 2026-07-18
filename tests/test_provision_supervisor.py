"""Unit coverage for the provisioner's containment logic.

The root/ELF end-to-end paths run on the hosted image in
.github/workflows/provision-selftest.yml; these tests pin the pure logic —
prefix-boundary predicates, rpath rewriting, staging order, and the
copy-equivalent preflight — with no root or patchelf required.
"""

from __future__ import annotations

import importlib.util
import shutil
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
# Register in sys.modules before exec so the module resolves itself by name.
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

    def test_stdlib_zip_is_purged_across_platlibdir_and_abi(self, tmp_path):
        """The canonical stdlib zip is on sys.path even absent; purge it in lib,
        lib64 (PLATLIBDIR), and the free-threaded ``t`` name."""
        source = self._make_source(tmp_path)
        major, minor = sys.version_info.major, sys.version_info.minor
        for rel in (
            f"lib/python{major}{minor}.zip",
            f"lib64/python{major}{minor}.zip",
            f"lib/python{major}{minor}t.zip",
        ):
            zip_path = source / rel
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(b"PK\x03\x04")
        staged = tmp_path / "staged-empty"
        staged.mkdir()
        runtime = tmp_path / "dest" / "python"
        runtime.parent.mkdir()
        provisioner._stage_runtime_tree(source, runtime, staged)
        assert not list(runtime.rglob(f"python{major}{minor}*.zip"))

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


class TestTrustedGit:
    def test_accepts_root_owned_system_git(self):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        resolved = Path(git).resolve()
        assert provisioner._resolve_trusted_git(resolved) == resolved

    def test_rejects_relative_path(self):
        with pytest.raises(SystemExit, match="absolute and normalized"):
            provisioner._resolve_trusted_git(Path("git"))

    def test_rejects_symlinked_path(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        alias = tmp_path / "git"
        alias.symlink_to(Path(git).resolve())
        with pytest.raises(SystemExit, match="contains a symlink"):
            provisioner._resolve_trusted_git(alias)

    def test_installed_wrapper_blocks_local_executable_config(self, tmp_path):
        git = shutil.which("git")
        if git is None:
            pytest.skip("Git is required")
        destination = tmp_path / "destination"
        destination.mkdir()
        wrapper = provisioner._install_trusted_git_wrapper(
            destination,
            Path(sys.executable).resolve(),
            provisioner._resolve_trusted_git(Path(git).resolve()),
        )
        repository = tmp_path / "rulespec-us"
        subprocess.run([git, "init", "--quiet", str(repository)], check=True)
        (repository / "a.txt").write_text("original\n")
        (repository / ".gitattributes").write_text("*.txt diff=hostile\n")
        subprocess.run([git, "-C", str(repository), "add", "."], check=True)
        subprocess.run(
            [
                git,
                "-c",
                "user.name=Axiom test",
                "-c",
                "user.email=test@axiom.invalid",
                "-C",
                str(repository),
                "commit",
                "--quiet",
                "-m",
                "fixture",
            ],
            check=True,
        )
        marker = tmp_path / "helper-executed"
        helper = tmp_path / "hostile-helper"
        helper.write_text(
            f"#!{Path(sys.executable).resolve()}\n"
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).touch()\n"
        )
        helper.chmod(0o755)
        for key in ("core.fsmonitor", "diff.external", "diff.hostile.textconv"):
            subprocess.run(
                [git, "-C", str(repository), "config", key, str(helper)], check=True
            )
        (repository / "a.txt").write_text("changed\n")
        clean_environment = {
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "HOME": str(tmp_path),
            "PATH": str(destination),
        }
        subprocess.run(
            [str(wrapper), "-C", str(repository), "status", "--porcelain"],
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        subprocess.run(
            [str(wrapper), "-C", str(repository), "diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
            env=clean_environment,
        )
        assert not marker.exists()

        refused = subprocess.run(
            [str(wrapper), "-C", str(repository), "fetch"],
            check=False,
            capture_output=True,
            text=True,
            env=clean_environment,
        )
        assert refused.returncode != 0
        assert "refused command: fetch" in refused.stderr


class TestIsElf:
    def test_elf_magic(self, tmp_path):
        elf = tmp_path / "obj"
        elf.write_bytes(b"\x7fELF" + b"\x00" * 60)
        assert provisioner._is_elf(elf)

    def test_non_elf(self, tmp_path):
        script = tmp_path / "script"
        script.write_bytes(b"#!/bin/sh\n")
        assert not provisioner._is_elf(script)


class _FakePatchelf:
    """subprocess.run stand-in for a single object under test.

    Constructed with the object's current run path (a string, or None to
    simulate a --print-rpath returncode!=0 read failure). --set-rpath
    records and updates the object's current run path; a subsequent
    --print-rpath then returns that new value.
    """

    def __init__(self, rpath):
        # single object under test: sequence of run-path states over time
        self._current = rpath
        self.set_calls = []

    def __call__(self, cmd, *args, **kwargs):
        if "--print-rpath" in cmd:
            if self._current is None:
                return subprocess.CompletedProcess(cmd, 1, "", "no rpath")
            return subprocess.CompletedProcess(cmd, 0, self._current + "\n", "")
        if "--set-rpath" in cmd:
            value = cmd[cmd.index("--set-rpath") + 1]
            self.set_calls.append(value)
            self._current = value
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected patchelf call: {cmd}")


class TestRelocateElfRpaths:
    def _runtime_with_elf(self, tmp_path):
        runtime = (tmp_path / "python").resolve()
        (runtime / "lib").mkdir(parents=True)
        obj = runtime / "lib" / "libpython.so"
        obj.write_bytes(b"\x7fELF" + b"\x00" * 60)  # magic only; patchelf is stubbed
        return runtime, obj

    def test_escaping_runpath_is_repinned(self, tmp_path, monkeypatch):
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("/opt/hostedtoolcache/Python/3.14/x64/lib")
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 1
        assert fake.set_calls == [str(runtime / "lib")]

    def test_origin_only_runpath_untouched(self, tmp_path, monkeypatch):
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("$ORIGIN")
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 0
        assert fake.set_calls == []

    def test_unreadable_rpath_is_skipped(self, tmp_path, monkeypatch):
        # a .o-style object patchelf can't read a run path from → skipped, no set
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf(None)
        monkeypatch.setattr(provisioner.subprocess, "run", fake)
        assert provisioner._relocate_elf_rpaths(runtime, "patchelf") == 0
        assert fake.set_calls == []

    def test_rewrite_that_does_not_take_is_fatal(self, tmp_path, monkeypatch):
        """If --set-rpath silently failed to change the object, the confirming
        re-read must catch that an escaping component survives."""
        runtime, _ = self._runtime_with_elf(tmp_path)
        fake = _FakePatchelf("/opt/toolcache/lib")

        def stubborn(cmd, *a, **k):
            if "--set-rpath" in cmd:  # pretend the write was a no-op
                return subprocess.CompletedProcess(cmd, 0, "", "")
            return fake(cmd, *a, **k)

        monkeypatch.setattr(provisioner.subprocess, "run", stubborn)
        with pytest.raises(SystemExit, match="still escapes the runtime"):
            provisioner._relocate_elf_rpaths(runtime, "patchelf")

    # The real-patchelf rewrite round-trip is covered end-to-end on hosted Ubuntu
    # by provision-selftest.yml, which relocates a genuine escaping RUNPATH in the
    # toolchain libpython and confirms via /proc/self/maps that libpython loads
    # from inside the runtime. patchelf edits ELF sections that a byte-magic stub
    # can't model, so the rewrite itself is not unit-tested with a synthetic file.


class TestPrintRpath:
    def test_strips_only_trailing_newline(self, monkeypatch):
        # a leading space is part of the (malformed) value and must survive
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, " /x:$ORIGIN\n", ""),
        )
        assert provisioner._print_rpath("patchelf", Path("obj")) == " /x:$ORIGIN"

    def test_nonzero_return_is_none(self, monkeypatch):
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 1, "", "err"),
        )
        assert provisioner._print_rpath("patchelf", Path("obj")) is None


class TestAssertSelfContained:
    """The empirical probe's fail-closed and shebang-safety guards. The probe
    shells out to the interpreter, so inject the maps report by stubbing
    subprocess.run; sys.platform is forced to linux to reach the maps checks."""

    def _run_with(self, monkeypatch, tmp_path, maps):
        runtime = (tmp_path / "python").resolve()
        (runtime / "bin").mkdir(parents=True)
        interpreter = runtime / "bin" / "python3"
        interpreter.write_text("x")
        source = (tmp_path / "src").resolve()
        source.mkdir()
        monkeypatch.setattr(provisioner.sys, "platform", "linux")
        # Stub the probe subprocess and its parse so the base_prefix/version
        # checks pass and we reach the maps checks with the injected report.
        monkeypatch.setattr(
            provisioner.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(a[0], 0, "{}", ""),
        )
        monkeypatch.setattr(
            provisioner.json,
            "loads",
            lambda _s: {
                "base_prefix": str(runtime),
                "version": list(sys.version_info[:2]),
                "maps": maps,
            },
        )
        provisioner._assert_self_contained(runtime, source, interpreter)
        return runtime

    def _interp(self, tmp_path):
        return str((tmp_path / "python").resolve() / "bin" / "python3")

    def test_empty_maps_fails_closed(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="no /proc/self/maps entries"):
            self._run_with(monkeypatch, tmp_path, [])

    def test_interpreter_not_mapped_fails_closed(self, tmp_path, monkeypatch):
        # maps present but the provisioned interpreter binary is absent → the
        # probe did not actually run the runtime interpreter (vacuous pass).
        maps = ["/usr/lib/x86_64-linux-gnu/libc.so.6", "/lib64/ld-linux-x86-64.so.2"]
        with pytest.raises(SystemExit, match="did not map the provisioned interpreter"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_static_build_without_libpython_ok(self, tmp_path, monkeypatch):
        # A static-libpython interpreter (python-build-standalone) maps no
        # libpython.so; that is fine as long as its own binary is mapped.
        maps = [self._interp(tmp_path), "/usr/lib/x86_64-linux-gnu/libc.so.6"]
        self._run_with(monkeypatch, tmp_path, maps)  # no raise

    def test_system_libc_allowed_shared_libpython_pinned(self, tmp_path, monkeypatch):
        runtime = (tmp_path / "python").resolve()
        maps = [
            self._interp(tmp_path),
            str(runtime / "lib" / "libpython3.so"),
            "/usr/lib/x86_64-linux-gnu/libc.so.6",
            "/lib64/ld-linux-x86-64.so.2",
        ]
        self._run_with(monkeypatch, tmp_path, maps)  # no raise

    def test_source_prefix_mapping_rejected(self, tmp_path, monkeypatch):
        source = (tmp_path / "src").resolve()
        maps = [self._interp(tmp_path), str(source / "lib" / "libpython3.so")]
        with pytest.raises(SystemExit, match="maps code from the source"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_stray_libpython_rejected(self, tmp_path, monkeypatch):
        maps = [self._interp(tmp_path), "/opt/elsewhere/lib/libpython3.so"]
        with pytest.raises(SystemExit, match="libpython mapped outside the runtime"):
            self._run_with(monkeypatch, tmp_path, maps)

    def test_whitespace_destination_is_refused(self, tmp_path):
        runtime = tmp_path / "python"
        runtime.mkdir()
        bad_interp = tmp_path / "a b" / "python3"
        with pytest.raises(SystemExit, match="not shebang-safe"):
            provisioner._assert_self_contained(runtime, tmp_path, bad_interp)

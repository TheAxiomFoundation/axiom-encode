"""Cycle-15 regressions for committed engine snapshots and build provenance."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity
from axiom_encode.engine_binding import engine_binding_receipt_path
from tests.environment_assertions import expect

_REAL_RUN = subprocess.run
_REAL_WHICH = shutil.which
_ENGINE_NAME = "axiom-rules-engine"
_FILES = {
    ".gitignore": b"/target/\n/build.rs\n/rust-toolchain.toml\n/extra/ignored.txt\n",
    "Cargo.toml": (
        b'[package]\nname = "axiom-rules-engine"\nversion = "0.0.0"\nedition = "2021"\n'
    ),
    "Cargo.lock": b"# committed lock\nversion = 4\n",
    "src/main.rs": b'fn main() { println!("committed"); }\n',
}


def _git(repo: Path, *arguments: str, **kwargs: Any) -> subprocess.CompletedProcess:
    return _REAL_RUN(
        ["git", "-C", str(repo), *arguments],
        env={
            **{
                name: value
                for name, value in os.environ.items()
                if not name.startswith("GIT_")
            },
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "LC_ALL": "C",
            "TZ": "UTC",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        **kwargs,
    )


def _commit_tree(
    repo: Path,
    files: dict[str, bytes],
    parent: str | None = None,
    *,
    symlinks: dict[str, str] | None = None,
) -> str:
    # Seed disposable fixtures without git add/commit or changing shared refs.
    stream = bytearray(
        b"commit refs/heads/fixture\n"
        b"committer CI fixture <ci@example.invalid> 1700000000 +0000\n"
        b"data 7\nfixture\n"
    )
    if parent is not None:
        stream.extend(f"from {parent}\n".encode())
    stream.extend(b"deleteall\n")
    for name, payload in sorted(files.items()):
        stream.extend(f"M 100644 inline {name}\ndata {len(payload)}\n".encode())
        stream.extend(payload)
        stream.extend(b"\n")
    for name, target in sorted((symlinks or {}).items()):
        payload = target.encode()
        stream.extend(f"M 120000 inline {name}\ndata {len(payload)}\n".encode())
        stream.extend(payload)
        stream.extend(b"\n")
    stream.extend(b"\n")
    _git(repo, "fast-import", "--quiet", input=bytes(stream))
    _git(repo, "checkout", "--quiet", "--force", "fixture")
    return _git(repo, "rev-parse", "HEAD").stdout.decode().strip()


def _engine(
    root: Path,
    files: dict[str, bytes] | None = None,
    *,
    symlinks: dict[str, str] | None = None,
) -> tuple[Path, str]:
    if _REAL_WHICH("git") is None:
        pytest.skip("git unavailable")
    engine = root / "engine"
    engine.mkdir(parents=True)
    _git(engine, "init", "--quiet", "--template=")
    return engine, _commit_tree(
        engine, _FILES if files is None else files, symlinks=symlinks
    )


def _inventory(root: Path, *, exclude_git: bool = False) -> dict[str, bytes]:
    files = {}
    for directory, children, names in os.walk(root):
        links = [name for name in children if Path(directory, name).is_symlink()]
        for name in [*names, *links]:
            path = Path(directory, name)
            relative = path.relative_to(root)
            if exclude_git and ".git" in relative.parts:
                continue
            files[str(relative)] = (
                os.fsencode(os.readlink(path))
                if path.is_symlink()
                else path.read_bytes()
            )
    return files


def _tree_inventory(engine: Path, pin: str) -> dict[str, bytes]:
    names = _git(engine, "ls-tree", "-rz", "--name-only", pin).stdout.split(b"\0")
    return {
        name.decode(): _git(engine, "show", f"{pin}:{name.decode()}").stdout
        for name in names
        if name
    }


def _binary(path: Path, payload: bytes = b"fresh private build") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755)
    return path


def _artifact(path: Path, *, fresh: bool = False) -> str:
    return json.dumps(
        {
            "reason": "compiler-artifact",
            "target": {"name": _ENGINE_NAME, "kind": ["bin"]},
            "executable": str(path),
            "fresh": fresh,
        }
    )


class _Build:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        root: Path,
        *,
        cargo: Callable[[Path, Path], Path] | None = None,
        rustup: str | None = "stable-test (default)\n",
        rustup_code: int = 0,
    ) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []
        self.snapshot: Path | None = None
        self.snapshot_files: dict[str, bytes] | None = None
        self.cargo_calls = 0
        self.rustup_calls = 0
        home = root / "home"
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HOME", str(home.resolve()))

        def which(name: str, *args: Any, **kwargs: Any) -> str | None:
            if name == "rustup":
                return None if rustup is None else "/mock/rustup"
            return _REAL_WHICH(name, *args, **kwargs)

        def child(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
            name = Path(command[0]).name
            if name == "git":
                return _REAL_RUN(command, **kwargs)
            self.calls.append((command, kwargs))
            if name == "rustup":
                self.rustup_calls += 1
                expect(command[1:] == ["show", "active-toolchain"], command)
                return subprocess.CompletedProcess(command, rustup_code, rustup, "")
            if name == "rustc":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "rustc 1.94.1\nrelease: 1.94.1\nhost: c15-test-host\n",
                    "",
                )
            expect(name == "cargo", command)
            self.cargo_calls += 1
            self.snapshot = Path(kwargs["cwd"])
            self.snapshot_files = _inventory(self.snapshot)
            target = Path(kwargs["env"]["CARGO_TARGET_DIR"])
            executable = (
                _binary(target / "debug" / _ENGINE_NAME)
                if cargo is None
                else cargo(self.snapshot, target)
            )
            return subprocess.CompletedProcess(command, 0, _artifact(executable), "")

        monkeypatch.setattr(ci_parity.shutil, "which", which)
        monkeypatch.setattr(ci_parity.subprocess, "run", child)


def _build(engine: Path, pin: str | None, temp: Path, **kwargs: Any) -> Path:
    return ci_parity._build_workflow_engine(
        engine, temp, verified_pin=pin, offline=True, **kwargs
    )


def test_c15_ignored_build_inputs_are_absent_from_snapshot(tmp_path, monkeypatch):
    """6f11be26 L1068-1071/L1492: checkout and build committed inputs only."""
    engine, pin = _engine(tmp_path)
    extras = {
        "build.rs": b'fn main() { println!("cargo:rustc-cfg=local"); }\n',
        "rust-toolchain.toml": b'[toolchain]\nchannel = "nightly"\n',
        "target/debug/build/stale/output": b"unverified build-script output\n",
    }
    for name, payload in extras.items():
        path = engine / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    assert not _git(engine, "status", "--porcelain", "--untracked-files=all").stdout
    build = _Build(monkeypatch, tmp_path)
    _build(engine, pin, tmp_path / "run")
    assert build.snapshot_files == _tree_inventory(engine, pin)
    assert build.snapshot != engine
    assert all(
        (engine / name).read_bytes() == payload for name, payload in extras.items()
    )


def test_c15_relative_home_fails_before_cargo(tmp_path, monkeypatch):
    """Cargo and config inspection must use the same absolute HOME."""
    engine, pin = _engine(tmp_path)
    build = _Build(monkeypatch, tmp_path)
    relative = Path("target/userhome")
    config = engine / relative / ".cargo" / "config.toml"
    config.parent.mkdir(parents=True)
    config.write_text('[build]\nrustc-wrapper = "unverified-wrapper"\n')
    monkeypatch.setenv("HOME", str(relative))
    with pytest.raises(ValueError, match="absolute HOME"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


def test_c15_absolute_home_is_shared_by_inspection_rustup_and_cargo(
    tmp_path, monkeypatch
):
    engine, pin = _engine(tmp_path)
    build = _Build(monkeypatch, tmp_path)
    notes: list[str] = []
    _build(engine, pin, tmp_path / "run", notes=notes)
    home = str((tmp_path / "home").resolve())
    assert all(kwargs["env"]["HOME"] == home for _command, kwargs in build.calls)
    assert build.rustup_calls == 1
    assert any("stable-test" in note for note in notes)
    assert all(kwargs["cwd"] == build.snapshot for _command, kwargs in build.calls)


def test_c15_redirected_artifact_symlink_alias_is_rejected(tmp_path, monkeypatch):
    """A foreign-target path may resolve to debug while differing lexically."""
    engine, pin = _engine(tmp_path)

    def cargo(_snapshot, target):
        _binary(target / "debug" / _ENGINE_NAME)
        (target / "foreign-target").symlink_to(".", target_is_directory=True)
        return target / "foreign-target" / "debug" / _ENGINE_NAME

    _Build(monkeypatch, tmp_path, cargo=cargo)
    with pytest.raises(ValueError, match="artifact.*expected"):
        _build(engine, pin, tmp_path / "run")


@pytest.mark.parametrize("component", ["target", "debug", "binary"])
def test_c15_expected_artifact_path_rejects_each_symlink_component(
    tmp_path, monkeypatch, component
):
    """For every output path component, a symlink fails closed before copying."""
    engine, pin = _engine(tmp_path)

    def cargo(_snapshot, target):
        foreign = tmp_path / "foreign-output"
        expected = target / "debug" / _ENGINE_NAME
        if component == "target":
            if target.exists():
                target.rmdir()
            foreign.mkdir()
            target.symlink_to(foreign, target_is_directory=True)
            _binary(expected)
        elif component == "debug":
            target.mkdir(parents=True, exist_ok=True)
            foreign.mkdir()
            (target / "debug").symlink_to(foreign, target_is_directory=True)
            _binary(expected)
        else:
            expected.parent.mkdir(parents=True, exist_ok=True)
            _binary(foreign)
            expected.symlink_to(foreign)
        return expected

    _Build(monkeypatch, tmp_path, cargo=cargo)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        _build(engine, pin, tmp_path / "run")


def test_c15_stale_target_cannot_supply_receipted_bytes(tmp_path, monkeypatch):
    """6f11be26 L1492 builds with a fresh target, regardless of cargo freshness."""
    engine, pin = _engine(tmp_path)
    stale = _binary(engine / "target" / "debug" / _ENGINE_NAME, b"altered stale binary")

    def cargo(_snapshot, target):
        assert target != engine / "target"
        assert not (target / "debug" / _ENGINE_NAME).exists()
        return _binary(target / "debug" / _ENGINE_NAME, b"compiled committed bytes")

    _Build(monkeypatch, tmp_path, cargo=cargo)
    view = _build(engine, pin, tmp_path / "run")
    bound = view / "target" / "debug" / _ENGINE_NAME
    assert bound.read_bytes() == b"compiled committed bytes"
    assert stale.read_bytes() == b"altered stale binary"
    assert engine_binding_receipt_path(bound).exists()


def test_c15_midbuild_checkout_edits_do_not_change_snapshot(tmp_path, monkeypatch):
    engine, pin = _engine(tmp_path)
    original = (engine / "src/main.rs").read_bytes()

    def cargo(snapshot, target):
        assert (snapshot / "src/main.rs").read_bytes() == original
        (engine / "src/main.rs").write_bytes(b"edited while cargo runs\n")
        assert (snapshot / "src/main.rs").read_bytes() == original
        return _binary(target / "debug" / _ENGINE_NAME)

    _Build(monkeypatch, tmp_path, cargo=cargo)
    view = _build(engine, pin, tmp_path / "run")
    assert engine_binding_receipt_path(view / "target/debug" / _ENGINE_NAME).exists()
    assert (engine / "src/main.rs").read_bytes() == b"edited while cargo runs\n"


def test_c15_verified_pin_is_archived_even_if_checkout_head_moves(
    tmp_path, monkeypatch
):
    engine, pin = _engine(tmp_path)
    changed = {**_FILES, "src/main.rs": b"different HEAD source\n"}
    head = _commit_tree(engine, changed, parent=pin)
    assert head != pin
    build = _Build(monkeypatch, tmp_path)
    _build(engine, pin, tmp_path / "run")
    assert build.snapshot_files == _tree_inventory(engine, pin)
    assert (engine / "src/main.rs").read_bytes() == changed["src/main.rs"]


def test_c15_mismatched_checkout_archives_head_without_receipt(tmp_path, monkeypatch):
    engine, old_pin = _engine(tmp_path)
    changed = {**_FILES, "src/main.rs": b"explicitly mismatched HEAD source\n"}
    head = _commit_tree(engine, changed, parent=old_pin)
    build = _Build(monkeypatch, tmp_path)
    view = _build(engine, None, tmp_path / "run")
    assert build.snapshot_files == _tree_inventory(engine, head)
    assert not engine_binding_receipt_path(
        view / "target/debug" / _ENGINE_NAME
    ).exists()
    assert build.snapshot != engine


def test_c15_lock_rewrite_stays_in_snapshot_and_keeps_receipt(tmp_path, monkeypatch):
    """CI permits Cargo.lock rewrites: 6f11be26 L1492 / 0effa6a5 L1139."""
    engine, pin = _engine(tmp_path)
    original = (engine / "Cargo.lock").read_bytes()

    def cargo(snapshot, target):
        (snapshot / "Cargo.lock").write_bytes(b"cargo rewrote the private lock\n")
        return _binary(target / "debug" / _ENGINE_NAME)

    _Build(monkeypatch, tmp_path, cargo=cargo)
    notes: list[str] = []
    view = _build(engine, pin, tmp_path / "run", notes=notes)
    assert (engine / "Cargo.lock").read_bytes() == original
    assert engine_binding_receipt_path(view / "target/debug" / _ENGINE_NAME).exists()
    assert any("Cargo.lock rewritten" in note for note in notes)
    assert not any("receipt omitted" in note for note in notes)


@pytest.mark.parametrize(
    "output",
    [
        "nightly-test (directory override for '/private/engine')\n",
        "nightly-test (overridden by '/private/engine/rust-toolchain.toml')\n",
        "nightly-test (environment override by RUSTUP_TOOLCHAIN)\n",
        "stable-test\n",
        "stable-test (unknown reason)\n",
        "stable-test (default)\nnightly-test (default)\n",
    ],
)
def test_c15_nondefault_or_unidentified_rustup_source_fails_before_cargo(
    tmp_path, monkeypatch, output
):
    """6f11be26 L1117-1118 selects stable, without local Rustup overrides."""
    engine, pin = _engine(tmp_path)
    build = _Build(monkeypatch, tmp_path, rustup=output)
    with pytest.raises(ValueError, match="toolchain"):
        _build(engine, pin, tmp_path / "run")
    assert build.rustup_calls == 1
    assert build.cargo_calls == 0


def test_c15_failed_rustup_inspection_fails_before_cargo(tmp_path, monkeypatch):
    engine, pin = _engine(tmp_path)
    build = _Build(monkeypatch, tmp_path, rustup="no active toolchain\n", rustup_code=1)
    with pytest.raises(ValueError, match="toolchain"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


def test_c15_missing_rustup_records_unchecked_toolchain_source(tmp_path, monkeypatch):
    engine, pin = _engine(tmp_path)
    build = _Build(monkeypatch, tmp_path, rustup=None)
    notes: list[str] = []
    _build(engine, pin, tmp_path / "run", notes=notes)
    assert build.rustup_calls == 0
    assert any("rustup" in note and "could not be checked" in note for note in notes)


@pytest.mark.parametrize("kind", ["manifest", "toolchain", "config"])
def test_c15_unused_checkout_ancestors_do_not_affect_cargo(tmp_path, monkeypatch, kind):
    engine, pin = _engine(tmp_path / "developer-workspace")
    if kind == "manifest":
        (engine.parent / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
    elif kind == "toolchain":
        (engine.parent / "rust-toolchain.toml").write_text(
            '[toolchain]\nchannel="nightly"\n'
        )
    else:
        (engine.parent / ".cargo").mkdir()
        (engine.parent / ".cargo/config.toml").write_text(
            '[build]\nrustc-wrapper="local"\n'
        )
    build = _Build(monkeypatch, tmp_path)
    _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 1
    assert build.snapshot is not None and engine.parent not in build.snapshot.parents


@pytest.mark.parametrize("kind", ["manifest", "toolchain", "config"])
def test_c15_private_snapshot_ancestors_are_inspected(tmp_path, monkeypatch, kind):
    engine, pin = _engine(tmp_path / "developer-workspace")
    run = tmp_path / "run"
    run.mkdir()
    if kind == "manifest":
        (run / "Cargo.toml").write_text("[workspace]\nmembers = []\n")
        message = "ambient Cargo manifest"
    elif kind == "toolchain":
        (run / "rust-toolchain.toml").write_text('[toolchain]\nchannel="nightly"\n')
        message = "Rust toolchain override"
    else:
        (run / ".cargo").mkdir()
        (run / ".cargo/config.toml").write_text('[build]\nrustc-wrapper="local"\n')
        message = "unsupported Cargo config"
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match=message):
        _build(engine, pin, run)
    assert build.cargo_calls == 0


def test_c15_committed_config_is_inspected_in_snapshot(tmp_path, monkeypatch):
    engine, pin = _engine(
        tmp_path,
        {**_FILES, ".cargo/config.toml": b'[build]\nrustc-wrapper="committed"\n'},
    )
    (engine / ".cargo/config.toml").unlink()
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="committed Cargo config"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


_EXTRA_PATHS = st.one_of(
    st.builds(
        lambda directory, name: f"extra/generated/{directory}/{name}.bin",
        st.text("abcdef0123456789_-", min_size=1, max_size=10),
        st.text("abcdef0123456789_-", min_size=1, max_size=10),
    ),
    st.sampled_from(
        [
            "build.rs",
            "rust-toolchain.toml",
            "target/debug/axiom-rules-engine",
            "target/debug/.fingerprint/stale",
            "extra/ignored.txt",
            "extra/untracked.txt",
            "src/bin/automatic.rs",
        ]
    ),
)


def test_c15_snapshot_contains_exact_committed_tree_for_all_checkout_extras(tmp_path):
    """For all extra path/byte sets, snapshot paths and bytes equal git ls-tree."""

    @given(extras=st.dictionaries(_EXTRA_PATHS, st.binary(max_size=40), max_size=7))
    @settings(deadline=None, max_examples=25, database=None)
    def check(extras):
        # Each Hypothesis example has its own fixture and private target.
        root = tmp_path / str(check.calls)
        check.calls += 1
        engine, pin = _engine(root)
        for name, payload in extras.items():
            path = engine / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        with pytest.MonkeyPatch.context() as patch:
            build = _Build(patch, root)
            _build(engine, pin, root / "run")
        assert build.snapshot_files == _tree_inventory(engine, pin)

    check.calls = 0
    check()


def test_c15_snapshot_matches_fresh_real_git_clone_checkout(tmp_path, monkeypatch):
    """Differential: archive paths/bytes equal actions/checkout's fresh tree."""
    engine, pin = _engine(
        tmp_path,
        symlinks={
            "plain-manifest": "Cargo.toml",
            "normalized-manifest": "src/../Cargo.toml",
            "linked-source": "src",
        },
    )
    (engine / "build.rs").write_bytes(b"ignored local script\n")
    (engine / "src/main.rs").write_bytes(b"modified developer source\n")
    clone = tmp_path / "fresh-clone"
    _git(engine, "clone", "--quiet", "--no-local", str(engine), str(clone))
    _git(clone, "checkout", "--quiet", "--detach", pin)
    build = _Build(monkeypatch, tmp_path)
    _build(engine, pin, tmp_path / "run")
    assert build.snapshot_files == _inventory(clone, exclude_git=True)


@pytest.mark.parametrize("attribute", ["export-ignore", "export-subst"])
def test_c15_archive_export_attributes_fail_closed(tmp_path, monkeypatch, attribute):
    """An archive altered by attributes cannot represent actions/checkout."""
    files = {
        **_FILES,
        ".gitattributes": f"src/main.rs {attribute}\n".encode(),
        "src/main.rs": b"// $Format:%H$\nfn main() {}\n",
    }
    engine, pin = _engine(tmp_path, files)
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="archive differs from the committed tree"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


def test_c15_git_info_attributes_cannot_alter_archived_inputs(tmp_path, monkeypatch):
    engine, pin = _engine(tmp_path)
    (engine / ".git/info").mkdir(exist_ok=True)
    (engine / ".git/info/attributes").write_text("src/main.rs export-ignore\n")
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="archive differs from the committed tree"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


def test_c15_committed_target_state_fails_before_cargo(tmp_path, monkeypatch):
    engine, pin = _engine(
        tmp_path,
        {**_FILES, "target/debug/axiom-rules-engine": b"committed stale binary"},
    )
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="commit contains target state"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0


def test_c15_git_replacement_refs_cannot_change_verified_pin_snapshot(
    tmp_path, monkeypatch
):
    engine, pin = _engine(tmp_path)
    changed = {**_FILES, "src/main.rs": b"replacement object source\n"}
    replacement = _commit_tree(engine, changed, parent=pin)
    _git(engine, "checkout", "--quiet", "--detach", pin)
    _git(engine, "update-ref", f"refs/replace/{pin}", replacement)
    build = _Build(monkeypatch, tmp_path)
    _build(engine, pin, tmp_path / "run")
    assert build.snapshot_files == _FILES
    assert _tree_inventory(engine, pin) == _FILES


@pytest.mark.parametrize(
    "symlinks",
    [
        {"external-input": "../outside.txt"},
        {"a": "z/../outside", "z": "."},
    ],
    ids=["direct-external", "forward-symlink-chain"],
)
def test_c15_external_committed_symlink_fails_before_cargo(
    tmp_path, monkeypatch, symlinks
):
    engine, pin = _engine(tmp_path, symlinks=symlinks)
    (tmp_path / "outside.txt").write_bytes(b"uncommitted external input\n")
    build = _Build(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="snapshot|outside|escapes"):
        _build(engine, pin, tmp_path / "run")
    assert build.cargo_calls == 0

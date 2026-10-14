"""Offline tests for scripts/check_locked_builds.py, the CI build-pin gate.

The gate itself runs against PyPI and GitHub in the `locked-builds` CI job.
These tests cover its parts without the network: export parsing, the wheel-tag
classifier (against the real uv.lock and, as properties, against an independent
tag expansion), and every way the orchestration must fail closed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from packaging.tags import Tag, compatible_tags, cpython_tags

import scripts.check_locked_builds as gate

ROOT = Path(__file__).resolve().parents[1]
LOCK = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
LINUX_X86_64 = [f"manylinux_2_{minor}_x86_64" for minor in range(39, 4, -1)] + [
    "manylinux2014_x86_64",
    "manylinux2010_x86_64",
    "manylinux1_x86_64",
    "linux_x86_64",
]
MACOS_ARM64 = [f"macosx_{major}_0_arm64" for major in range(26, 10, -1)] + [
    "macosx_11_0_universal2"
]


def _tags(version: tuple[int, int], platforms: list[str]) -> frozenset[Tag]:
    return frozenset(
        [
            *cpython_tags(version, platforms=platforms),
            *compatible_tags(version, platforms=platforms),
        ]
    )


CP313_LINUX = _tags((3, 13), LINUX_X86_64)
CP314_LINUX = _tags((3, 14), LINUX_X86_64)
CP313_MACOS = _tags((3, 13), MACOS_ARM64)


def _locked(name: str, version: str) -> dict:
    [entry] = [
        p for p in LOCK["package"] if p["name"] == name and p.get("version") == version
    ]
    return entry


# -- export parsing ---------------------------------------------------------

LINUX_313 = {
    "python_version": "3.13",
    "python_full_version": "3.13.9",
    "sys_platform": "linux",
    "platform_system": "Linux",
    "platform_machine": "x86_64",
    "os_name": "posix",
    "implementation_name": "cpython",
    "platform_python_implementation": "CPython",
    "implementation_version": "3.13.9",
    "platform_release": "",
    "platform_version": "",
    "extra": "",
}


def test_parse_export_splits_pins_and_git_and_honours_markers():
    registry, git = gate.parse_export(
        "# header\n"
        "\n"
        "PyYAML==6.0.3\n"
        "pywin32==311 ; sys_platform == 'win32'\n"
        "axiom-oracles @ git+https://github.com/TheAxiomFoundation/axiom-oracles@"
        + "a" * 40
        + "\n",
        LINUX_313,
    )

    assert registry == [("pyyaml", "6.0.3")]
    assert git == [
        gate.GitSource(
            "axiom-oracles",
            "https://github.com/TheAxiomFoundation/axiom-oracles",
            "a" * 40,
        )
    ]


@pytest.mark.parametrize(
    "line",
    [
        "requests>=2.28",
        "requests",
        "requests==2.32.5,<3",
        "axiom-oracles @ git+https://github.com/TheAxiomFoundation/axiom-oracles@main",
        "axiom-oracles @ https://example.com/axiom_oracles-0.2.1.tar.gz",
    ],
)
def test_parse_export_rejects_anything_but_exact_pins(line):
    with pytest.raises(gate.CheckFailed):
        gate.parse_export(line + "\n", LINUX_313)


# -- the classifier against the real lock -------------------------------------


def test_pyiceberg_builds_from_its_sdist_everywhere():
    pyiceberg = _locked("pyiceberg", "0.10.0")

    for tags in (CP313_LINUX, CP314_LINUX, CP313_MACOS):
        assert gate.needs_build(pyiceberg, tags)
    sdist = gate.sdist_of(pyiceberg)
    assert sdist.filename == "pyiceberg-0.10.0.tar.gz"
    assert sdist.sha256 == pyiceberg["sdist"]["hash"].removeprefix("sha256:")


def test_pyroaring_builds_from_its_sdist_only_on_314():
    pyroaring = _locked("pyroaring", "1.0.3")

    assert not gate.needs_build(pyroaring, CP313_LINUX)
    assert not gate.needs_build(pyroaring, CP313_MACOS)
    assert gate.needs_build(pyroaring, CP314_LINUX)


def test_pure_python_wheels_never_build():
    assert not gate.needs_build(_locked("pyyaml", "6.0.3"), CP314_LINUX)
    assert not gate.needs_build(_locked("hatchling", "1.32.4"), CP313_LINUX)


@pytest.mark.parametrize("tags", [CP313_LINUX, CP314_LINUX, CP313_MACOS])
def test_every_locked_build_backend_installs_as_a_wheel(tags):
    # A build backend built from source would be built with unpinned backends.
    for package in LOCK["package"]:
        if package["name"] in {
            "cython",
            "editables",
            "hatchling",
            "packaging",
            "pathspec",
            "pluggy",
            "poetry-core",
            "setuptools",
            "tomlkit",
            "trove-classifiers",
            "wheel",
        }:
            assert not gate.needs_build(package, tags), package["name"]


# -- the classifier as properties ---------------------------------------------

INTERPRETERS = st.sampled_from(["cp313", "cp314", "py3", "py2.py3", "cp313.cp314"])
ABIS = st.sampled_from(["cp313", "cp314", "abi3", "none", "cp314t"])
PLATFORMS = st.sampled_from(
    [
        "any",
        "manylinux_2_17_x86_64.manylinux2014_x86_64",
        "manylinux_2_28_x86_64",
        "manylinux_2_17_aarch64",
        "musllinux_1_2_x86_64",
        "macosx_11_0_arm64",
        "win_amd64",
        "manylinux_2_5_i686.manylinux1_i686",
    ]
)
WHEELS = st.lists(st.tuples(INTERPRETERS, ABIS, PLATFORMS), max_size=6)
TARGETS = st.sampled_from([CP313_LINUX, CP314_LINUX, CP313_MACOS])


def _package(wheels) -> dict:
    return {
        "name": "pkg",
        "version": "1.0",
        "wheels": [
            {
                "url": f"https://files.example/pkg-1.0-{i}-{a}-{p}.whl",
                "hash": "sha256:" + "0" * 64,
            }
            for i, a, p in wheels
        ],
    }


def _expanded(interpreter: str, abi: str, platform: str) -> set[str]:
    # An independent reading of a compressed wheel tag set (PEP 425): every
    # combination of the dot-separated alternatives.
    return {
        f"{i}-{a}-{p}"
        for i in interpreter.split(".")
        for a in abi.split(".")
        for p in platform.split(".")
    }


@settings(max_examples=300, deadline=None)
@given(wheels=WHEELS, target=TARGETS)
def test_classifier_matches_an_independent_tag_expansion(wheels, target):
    supported = {str(tag) for tag in target}
    oracle = not any(_expanded(*wheel) & supported for wheel in wheels)

    assert gate.needs_build(_package(wheels), target) == oracle


@settings(max_examples=200, deadline=None)
@given(wheels=WHEELS, extra=st.tuples(INTERPRETERS, ABIS, PLATFORMS), target=TARGETS)
def test_adding_a_wheel_never_forces_a_build(wheels, extra, target):
    if not gate.needs_build(_package(wheels), target):
        assert not gate.needs_build(_package([*wheels, extra]), target)


@settings(max_examples=200, deadline=None)
@given(wheels=WHEELS, target=TARGETS, data=st.data())
def test_classifier_ignores_wheel_order(wheels, target, data):
    shuffled = data.draw(st.permutations(wheels))

    assert gate.needs_build(_package(wheels), target) == gate.needs_build(
        _package(shuffled), target
    )


# -- planning ---------------------------------------------------------------


def _lock(*packages: dict) -> dict:
    return {
        "package": [
            {**p, "source": {"registry": "https://pypi.org/simple"}} for p in packages
        ]
    }


SDIST_ONLY = {
    "name": "slow",
    "version": "1.0",
    "sdist": {
        "url": "https://files.example/slow-1.0.tar.gz",
        "hash": "sha256:" + "a" * 64,
    },
}
WHEEL_ONLY = {
    "name": "fast",
    "version": "2.0",
    "wheels": [
        {
            "url": "https://files.example/fast-2.0-py3-none-any.whl",
            "hash": "sha256:" + "b" * 64,
        }
    ],
}


def test_plan_lists_sdist_builds_and_git_sources():
    planned = gate.plan(
        _lock(SDIST_ONLY, WHEEL_ONLY),
        "slow==1.0\nfast==2.0\naxiom-oracles @ git+https://github.com/o/r@"
        + "c" * 40
        + "\n",
        "fast==2.0\n",
        CP313_LINUX,
        LINUX_313,
    )

    assert planned.sdists == (
        gate.Sdist("slow", "1.0", SDIST_ONLY["sdist"]["url"], "a" * 64),
    )
    assert planned.git == (
        gate.GitSource("axiom-oracles", "https://github.com/o/r", "c" * 40),
    )


def test_plan_refuses_a_build_backend_without_a_wheel():
    with pytest.raises(gate.CheckFailed, match="has no wheel"):
        gate.plan(
            _lock(SDIST_ONLY, WHEEL_ONLY),
            "fast==2.0\n",
            "slow==1.0\n",
            CP313_LINUX,
            LINUX_313,
        )


def test_plan_refuses_a_package_with_neither_wheel_nor_sdist():
    bare = {"name": "bare", "version": "1.0"}
    with pytest.raises(
        gate.CheckFailed, match="neither a compatible wheel nor an sdist"
    ):
        gate.plan(_lock(bare), "bare==1.0\n", "", CP313_LINUX, LINUX_313)


def test_plan_refuses_a_git_build_backend():
    with pytest.raises(gate.CheckFailed, match="git requirements"):
        gate.plan(
            _lock(WHEEL_ONLY),
            "fast==2.0\n",
            "hatchling @ git+https://github.com/o/h@" + "d" * 40 + "\n",
            CP313_LINUX,
            LINUX_313,
        )


def test_parse_built_reads_uv_build_lines():
    stderr = (
        "Resolved 2 packages in 3ms\n"
        "   Building pyiceberg==0.10.0\n"
        "      Built pyiceberg==0.10.0\n"
        "      Built Py_Roaring.x==1.0.3\n"
        "Installed 2 packages in 4ms\n"
        " + pyiceberg==0.10.0\n"
    )

    assert gate.parse_built(stderr) == {
        ("pyiceberg", "0.10.0"),
        ("py-roaring-x", "1.0.3"),
    }


# -- orchestration ----------------------------------------------------------

SDIST_BYTES = b"pretend sdist"
SDIST = {
    "name": "slow",
    "version": "1.0",
    "sdist": {
        "url": "https://files.example/slow-1.0.tar.gz",
        "hash": "sha256:" + hashlib.sha256(SDIST_BYTES).hexdigest(),
    },
}
COMMIT = "c" * 40
INSTALLED = (
    f"slow==1.0\nfast==2.0\naxiom-oracles @ git+https://github.com/o/r@{COMMIT}\n"
)
BUILD = "fast==2.0\n"


class FakeWorld:
    def __init__(self, tmp_path: Path, **overrides):
        self.root = tmp_path / "root"
        self.root.mkdir()
        (self.root / "uv.lock").write_text(_toml_lock(_lock(SDIST, WHEEL_ONLY)))
        self.work = tmp_path / "work"
        self.work.mkdir()
        self.calls: list[list[str]] = []
        self.envs: list[dict] = []
        self.built = overrides.get("built", "      Built slow==1.0\n")
        self.head = overrides.get("head", COMMIT)
        self.sdist_bytes = overrides.get("sdist_bytes", SDIST_BYTES)
        self.fail_build_of = overrides.get("fail_build_of")
        self.editable = overrides.get("editable", ["fast"])

    def run(self, command, **kwargs):
        self.calls.append(command)
        self.envs.append(kwargs.get("env", {}))
        out, err, code = "", "", 0
        if command[:2] == ["uv", "export"]:
            if "--output-file" in command:
                target = Path(command[command.index("--output-file") + 1])
                target.write_text("exported\n")
            elif "--only-group" in command:
                out = BUILD
            else:
                out = INSTALLED
        elif command[:3] == ["uv", "pip", "install"] and "--target" in command:
            err = self.built
        elif command[:2] == ["uv", "build"]:
            if self.fail_build_of and command[-1].endswith(self.fail_build_of):
                code, err = 2, "No solution found: cython must be pinned upfront"
        elif command[:1] == ["git"] and "rev-parse" in command:
            out = self.head + "\n"
        elif command[1:3] == ["-I", "-c"]:
            out = json.dumps(self.editable)
        return subprocess.CompletedProcess(command, code, out, err)

    def fetch(self, url, destination):
        destination.write_bytes(self.sdist_bytes)

    def check(self):
        return gate.check(
            root=self.root,
            work=self.work,
            uv="uv",
            python="/py",
            compatible=CP313_LINUX,
            runner=self.run,
            fetch=self.fetch,
            log=lambda _message: None,
        )


def _toml_lock(lock: dict) -> str:
    lines = []
    for package in lock["package"]:
        lines.append("[[package]]")
        lines.append(f'name = "{package["name"]}"')
        lines.append(f'version = "{package["version"]}"')
        lines.append('source = { registry = "https://pypi.org/simple" }')
        if "sdist" in package:
            s = package["sdist"]
            lines.append(f'sdist = {{ url = "{s["url"]}", hash = "{s["hash"]}" }}')
        if "wheels" in package:
            wheels = ", ".join(
                f'{{ url = "{w["url"]}", hash = "{w["hash"]}" }}'
                for w in package["wheels"]
            )
            lines.append(f"wheels = [{wheels}]")
        lines.append("")
    return "\n".join(lines)


def test_check_builds_every_source_strictly(tmp_path, monkeypatch):
    monkeypatch.setenv("UV_NO_CACHE", "0")
    monkeypatch.setenv("UV_BUILD_CONSTRAINT", "/elsewhere.txt")
    world = FakeWorld(tmp_path)

    report = world.check()

    constraints = str(world.work / "build-constraints.txt")
    builds = [call for call in world.calls if call[:2] == ["uv", "build"]]
    assert [call[-1] for call in builds] == [
        str(world.work / "sdists" / "slow-1.0.tar.gz"),
        str(world.work / "git" / "axiom-oracles"),
        str(world.root),
    ]
    for call in builds:
        assert call[call.index("--build-constraints") + 1] == constraints
        assert {"--require-hashes", "--no-cache", "--wheel"} <= set(call)
        assert call[call.index("--python") + 1] == "/py"
    [staged] = [
        c for c in world.calls if c[:3] == ["uv", "pip", "install"] and "--target" in c
    ]
    assert {"--no-cache", "--require-hashes", "--no-deps"} <= set(staged)
    assert staged[staged.index("--build-constraints") + 1] == constraints
    # The git dependency is built from its locked commit, not staged unhashed.
    [locked_export] = [
        c
        for c in world.calls
        if c[:2] == ["uv", "export"] and "--all-extras" in c and "--output-file" in c
    ]
    assert (
        locked_export[locked_export.index("--no-emit-package") + 1] == "axiom-oracles"
    )
    assert not any(key.startswith("UV_") for env in world.envs for key in env)
    # No user or parent uv.toml reaches any uv call, as in the staging steps.
    uv_calls = [call for call in world.calls if call[0] == "uv"]
    assert uv_calls and all(
        call[2] == "--no-config" or call[3] == "--no-config" for call in uv_calls
    )
    assert report["sdists"] == ["slow==1.0"]
    assert report["git"] == [f"axiom-oracles@{COMMIT}"]
    assert report["editable_requires"] == ["fast"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"built": ""}, "different set of sdists", id="uv-built-nothing"),
        pytest.param(
            {"built": "      Built slow==1.0\n      Built fast==2.0\n"},
            "different set of sdists",
            id="uv-built-more",
        ),
        pytest.param(
            {"sdist_bytes": b"tampered"},
            "does not match its uv.lock hash",
            id="sdist-hash",
        ),
        pytest.param({"head": "e" * 40}, "checked out", id="git-commit"),
        pytest.param(
            {"fail_build_of": "slow-1.0.tar.gz"}, "exited 2", id="sdist-build"
        ),
        pytest.param({"fail_build_of": "axiom-oracles"}, "exited 2", id="git-build"),
        pytest.param({"fail_build_of": "root"}, "exited 2", id="project-build"),
        pytest.param(
            {"editable": ["fast", "editables~=0.3"]},
            "does not pin",
            id="editable-unpinned",
        ),
        pytest.param(
            {"editable": ["fast>=3"]}, "but the build group pins", id="editable-version"
        ),
    ],
)
def test_check_fails_closed(tmp_path, overrides, message):
    world = FakeWorld(tmp_path, **overrides)

    with pytest.raises(gate.CheckFailed, match=message):
        world.check()


def test_a_tampered_sdist_is_never_built(tmp_path):
    world = FakeWorld(tmp_path, sdist_bytes=b"tampered")

    with pytest.raises(gate.CheckFailed):
        world.check()
    assert not any(call[:2] == ["uv", "build"] for call in world.calls)


def test_main_reports_a_failure(monkeypatch, capsys):
    def fail(**_kwargs):
        raise gate.CheckFailed("cython is not pinned")

    monkeypatch.setattr(gate, "check", fail)

    assert gate.main(["--uv", "uv"]) == 1
    assert "error: cython is not pinned" in capsys.readouterr().err

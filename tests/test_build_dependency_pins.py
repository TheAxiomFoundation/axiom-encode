"""uv.lock pins the build backends, and every place that applies them agrees.

Three records name the build backends:

- the ``build`` dependency group, which uv.lock resolves and hash-pins (the
  staging workflows pass its export to ``uv pip install --build-constraints``);
- ``[tool.uv] build-constraint-dependencies`` in pyproject.toml, which
  ``uv sync``, ``uv lock`` and ``uv pip install`` apply by version, since
  ``uv sync`` takes no constraints file;
- uv.lock's ``[manifest] build-constraints``, uv's record of the second, which
  makes ``uv sync --locked`` fail when it drifts from pyproject.toml.

These tests hold all three to the same name==version set. That set covers the
locked group's whole dependency closure, so ``uv sync`` pins hatchling's own
dependencies too, not just hatchling. Whether the group covers every build a
locked install runs is checked against real builds by
``scripts/check_locked_builds.py`` in CI.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
LOCK = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _project_entry() -> dict:
    [entry] = [p for p in LOCK["package"] if p["name"] == "axiom-encode"]
    return entry


def _registry_entries() -> dict[str, list[dict]]:
    entries: dict[str, list[dict]] = {}
    for package in LOCK["package"]:
        entries.setdefault(canonicalize_name(package["name"]), []).append(package)
    return entries


def _locked_group_closure(group: str) -> dict[str, dict]:
    """The lock entries a dependency group installs, by name."""
    entries = _registry_entries()
    roots = _project_entry()["dev-dependencies"][group]
    pending = [canonicalize_name(dependency["name"]) for dependency in roots]
    closure: dict[str, dict] = {}
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        # One version of each: a forked resolution would make the group's
        # pins depend on the platform, which one constraint list cannot say.
        [entry] = entries[name]
        closure[name] = entry
        pending.extend(
            canonicalize_name(dependency["name"])
            for dependency in entry.get("dependencies", [])
        )
    return closure


def _pins(requirements) -> dict[str, str]:
    pins: dict[str, str] = {}
    for text in requirements:
        requirement = Requirement(text)
        [specifier] = requirement.specifier
        assert specifier.operator == "==", text
        assert requirement.marker is None and not requirement.extras, text
        name = canonicalize_name(requirement.name)
        assert name not in pins, f"{name} is pinned twice"
        pins[name] = specifier.version
    return pins


def test_build_group_is_locked_with_hashes():
    closure = _locked_group_closure("build")

    assert {"hatchling", "editables", "poetry-core", "cython"} <= set(closure)
    for name, entry in closure.items():
        assert "registry" in entry["source"], name
        files = [
            *entry.get("wheels", []),
            *([entry["sdist"]] if "sdist" in entry else []),
        ]
        assert files, name
        for item in files:
            algorithm, _, digest = item["hash"].partition(":")
            assert algorithm == "sha256" and len(digest) == 64, (name, item)


def test_sync_pins_equal_the_locked_build_group():
    locked = {
        name: entry["version"] for name, entry in _locked_group_closure("build").items()
    }
    pinned = _pins(PYPROJECT["tool"]["uv"]["build-constraint-dependencies"])

    assert pinned == locked


def test_lock_records_the_sync_pins():
    # uv's own record of [tool.uv] build-constraint-dependencies; `uv sync
    # --locked` refuses a lock whose manifest disagrees with pyproject.toml.
    manifest = {
        canonicalize_name(item["name"]): item["specifier"]
        for item in LOCK["manifest"]["build-constraints"]
    }
    pinned = _pins(PYPROJECT["tool"]["uv"]["build-constraint-dependencies"])

    assert manifest == {name: f"=={version}" for name, version in pinned.items()}


def test_build_group_names_only_backends_and_states_no_versions():
    # The group says which backends; the lock says which versions. A pin here
    # would be a second place to bump.
    group = PYPROJECT["dependency-groups"]["build"]
    names = [canonicalize_name(Requirement(text).name) for text in group]

    assert all(Requirement(text).specifier == "" for text in group), group
    assert names == sorted(set(names)), group


def test_build_group_covers_this_projects_backend():
    requires = PYPROJECT["build-system"]["requires"]
    group = {
        canonicalize_name(Requirement(text).name)
        for text in PYPROJECT["dependency-groups"]["build"]
    }

    assert {canonicalize_name(Requirement(text).name) for text in requires} <= group
    assert PYPROJECT["build-system"]["build-backend"] == "hatchling.build"
    # hatchling's get_requires_for_build_editable asks for editables, which
    # `uv sync` needs to install this project; check_locked_builds.py asks the
    # locked hatchling itself in CI.
    assert "editables" in group


def test_no_dependency_group_is_installed_by_default():
    # `uv sync` installs default groups ("dev" unless configured); these two are
    # exported by workflows, never synced into an environment.
    assert "default-groups" not in PYPROJECT["tool"]["uv"]
    assert "dev" not in PYPROJECT.get("dependency-groups", {})


def test_provision_selftest_group_is_the_locked_pyyaml():
    closure = _locked_group_closure("provision-selftest")
    runtime = [p for p in LOCK["package"] if p["name"] == "pyyaml"]

    assert set(closure) == {"pyyaml"}
    assert closure["pyyaml"] is not None and [closure["pyyaml"]] == runtime

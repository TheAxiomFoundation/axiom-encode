"""Prove that uv.lock's ``build`` group pins every build a locked install runs.

The staging workflows install the protected verification tree with
``uv pip install --no-config --no-cache --build-constraints <export of the
build group>``; every uv call here takes the same ``--no-config``. uv 0.11.7
verifies the hash of every build dependency that file lists, even under
``--require-hashes``, but a build dependency the file omits resolves to
whatever the index serves, and one it lists without a hash keeps its version
pin but goes unverified. ``uv build --require-hashes`` is stricter: it fails
unless every build requirement is pinned with a hash. This script
runs that stricter build for every source a locked install builds on the
running interpreter, so a lock change that brings in a new build backend
fails CI instead of floating in production.

The sources a locked install builds are:

- each registry package the lock pins (all extras, and the ``build`` group
  itself) that has no wheel compatible with this interpreter, from its
  hash-checked sdist;
- each git dependency, at its locked commit;
- the project.

The registry sdists are found two ways, which must agree: from the lock's
wheel tags against this interpreter's tags, and from what uv reports it
built when it installs the locked set the way the staging workflows do.

Run it with the interpreter the workflows stage for (CI runs it once per
Python version), from a checkout whose ``.venv`` was made by
``uv sync --locked``::

    .venv/bin/python scripts/check_locked_builds.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from packaging.requirements import Requirement
from packaging.tags import Tag, sys_tags
from packaging.utils import canonicalize_name, parse_wheel_filename

ROOT = Path(__file__).resolve().parents[1]
BUILD_GROUP = "build"
# `uv pip install` reports each source build on stderr as `Built name==version`.
BUILT = re.compile(r"^\s*Built (?P<name>[A-Za-z0-9._-]+)==(?P<version>\S+)\s*$")
GIT_URL = re.compile(r"^git\+(?P<repository>https://[^@#?]+)@(?P<commit>[0-9a-f]{40})$")

Runner = Callable[..., subprocess.CompletedProcess]
Fetcher = Callable[[str, Path], None]


class CheckFailed(RuntimeError):
    """A locked build is not fully pinned, or the evidence disagrees."""


@dataclass(frozen=True)
class Sdist:
    name: str
    version: str
    url: str
    sha256: str

    @property
    def filename(self) -> str:
        return urllib.parse.unquote(self.url.rsplit("/", 1)[1])


@dataclass(frozen=True)
class GitSource:
    name: str
    repository: str
    commit: str


@dataclass(frozen=True)
class Plan:
    sdists: tuple[Sdist, ...]
    git: tuple[GitSource, ...]


def parse_export(text: str, environment: dict[str, str] | None = None):
    """Split `uv export --no-hashes` output into registry pins and git pins.

    Requirements whose marker is false for ``environment`` (this interpreter
    by default) are dropped, as uv drops them when it installs the export.
    Anything other than a ``name==version`` pin or a commit-pinned git URL is
    an error.
    """
    registry: list[tuple[str, str]] = []
    git: list[GitSource] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate(
            environment
        ):
            continue
        name = canonicalize_name(requirement.name)
        if requirement.url is not None:
            match = GIT_URL.match(requirement.url)
            if match is None:
                raise CheckFailed(f"not a commit-pinned git requirement: {line}")
            git.append(GitSource(name, match["repository"], match["commit"]))
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            raise CheckFailed(f"not an exact pin: {line}")
        registry.append((name, specifiers[0].version))
    return registry, git


def lock_registry_packages(lock: dict) -> dict[tuple[str, str], dict]:
    packages: dict[tuple[str, str], dict] = {}
    for package in lock["package"]:
        if "registry" in package.get("source", {}):
            key = (canonicalize_name(package["name"]), package["version"])
            packages[key] = package
    return packages


def wheel_tags(package: dict) -> list[frozenset[Tag]]:
    return [
        frozenset(
            parse_wheel_filename(urllib.parse.unquote(wheel["url"].rsplit("/", 1)[1]))[
                3
            ]
        )
        for wheel in package.get("wheels", [])
    ]


def needs_build(package: dict, compatible: frozenset[Tag]) -> bool:
    """Whether no wheel the lock pins for ``package`` installs here."""
    return not any(tags & compatible for tags in wheel_tags(package))


def sdist_of(package: dict) -> Sdist:
    sdist = package.get("sdist")
    if not sdist:
        raise CheckFailed(
            f"{package['name']}=={package['version']} has neither a compatible "
            "wheel nor an sdist in uv.lock"
        )
    algorithm, _, digest = sdist["hash"].partition(":")
    if algorithm != "sha256" or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise CheckFailed(f"unexpected sdist hash in uv.lock: {sdist['hash']}")
    return Sdist(package["name"], package["version"], sdist["url"], digest)


def plan(
    lock: dict,
    installed_export: str,
    build_export: str,
    compatible: frozenset[Tag],
    environment: dict[str, str] | None = None,
) -> Plan:
    """The sdists and git sources that locked installs build on this interpreter."""
    packages = lock_registry_packages(lock)
    registry, git = parse_export(installed_export, environment)
    build_registry, build_git = parse_export(build_export, environment)
    if build_git:
        raise CheckFailed(f"the build group has git requirements: {build_git}")
    for key in build_registry:
        if needs_build(packages[key], compatible):
            # A build backend built from source would itself be built with
            # whatever backends the index serves.
            raise CheckFailed(
                f"build dependency {key[0]}=={key[1]} has no wheel for this interpreter"
            )
    sdists = sorted(
        {
            sdist_of(packages[key])
            for key in registry
            if needs_build(packages[key], compatible)
        },
        key=lambda sdist: (sdist.name, sdist.version),
    )
    return Plan(tuple(sdists), tuple(sorted(set(git), key=lambda g: g.name)))


def parse_built(stderr: str) -> set[tuple[str, str]]:
    built = set()
    for line in stderr.splitlines():
        match = BUILT.match(line)
        if match:
            built.add((canonicalize_name(match["name"]), match["version"]))
    return built


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_url(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310
        destination.write_bytes(response.read())


def clean_environment() -> dict[str, str]:
    # The lock and the flags decide what uv does, not UV_* the runner set.
    return {
        key: value for key, value in os.environ.items() if not key.startswith("UV_")
    }


def _run(
    runner: Runner, command: Sequence[str], **kwargs
) -> subprocess.CompletedProcess:
    result = runner(
        list(command),
        capture_output=True,
        text=True,
        env=clean_environment(),
        **kwargs,
    )
    if result.returncode != 0:
        raise CheckFailed(
            f"`{' '.join(command)}` exited {result.returncode}:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    return result


def check(
    *,
    root: Path = ROOT,
    work: Path,
    uv: str = "uv",
    python: str = sys.executable,
    compatible: Iterable[Tag] | None = None,
    runner: Runner = subprocess.run,
    fetch: Fetcher = fetch_url,
    log: Callable[[str], None] = print,
) -> dict:
    compatible = frozenset(sys_tags() if compatible is None else compatible)
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    export = [
        *(uv, "export", "--no-config", "--project", str(root)),
        *("--locked", "--no-emit-project"),
    ]
    unhashed = [
        "--no-hashes",
        "--no-header",
        "--no-annotate",
        "--format",
        "requirements-txt",
    ]
    installed = _run(runner, [*export, "--all-extras", *unhashed], cwd=root).stdout
    build = _run(
        runner, [*export, "--only-group", BUILD_GROUP, *unhashed], cwd=root
    ).stdout
    expected = plan(lock, installed, build, compatible)
    log(
        f"{len(expected.sdists)} locked sdist(s) and {len(expected.git)} git "
        "source(s) build on this interpreter: "
        + ", ".join(
            [f"{s.name}=={s.version}" for s in expected.sdists]
            + [f"{g.name}@{g.commit}" for g in expected.git]
        )
    )

    build_constraints = work / "build-constraints.txt"
    _run(
        runner,
        [
            *export,
            "--only-group",
            BUILD_GROUP,
            "--format",
            "requirements-txt",
            "--output-file",
            str(build_constraints),
        ],
        cwd=root,
    )

    # What uv builds when it stages the locked set as the workflows do.
    locked = work / "locked-requirements.txt"
    _run(
        runner,
        [
            *export,
            "--all-extras",
            "--format",
            "requirements-txt",
            *(item for g in expected.git for item in ("--no-emit-package", g.name)),
            "--output-file",
            str(locked),
        ],
        cwd=root,
    )
    staged = _run(
        runner,
        [
            uv,
            "pip",
            "install",
            "--no-config",
            "--python",
            python,
            "--target",
            str(work / "site-packages"),
            "--no-cache",
            "--build-constraints",
            str(build_constraints),
            "--require-hashes",
            "--no-deps",
            "-r",
            str(locked),
        ],
        cwd=root,
    )
    observed = parse_built(staged.stderr)
    predicted = {(canonicalize_name(s.name), s.version) for s in expected.sdists}
    if observed != predicted:
        raise CheckFailed(
            "uv built a different set of sdists than uv.lock's wheel tags predict: "
            f"built {sorted(observed)}, predicted {sorted(predicted)}"
        )

    def build_strictly(source: Path, label: str) -> None:
        log(f"uv build --require-hashes: {label}")
        _run(
            runner,
            [
                uv,
                "build",
                "--no-config",
                "--python",
                python,
                "--wheel",
                "--no-cache",
                "--require-hashes",
                "--build-constraints",
                str(build_constraints),
                "--out-dir",
                str(work / "wheels"),
                str(source),
            ],
            cwd=root,
        )

    for sdist in expected.sdists:
        archive = work / "sdists" / sdist.filename
        archive.parent.mkdir(parents=True, exist_ok=True)
        fetch(sdist.url, archive)
        if sha256_file(archive) != sdist.sha256:
            raise CheckFailed(f"{sdist.url} does not match its uv.lock hash")
        build_strictly(archive, f"{sdist.name}=={sdist.version} (sdist)")
    for source in expected.git:
        checkout = work / "git" / source.name
        _run(runner, ["git", "init", "-q", str(checkout)])
        _run(
            runner,
            [
                "git",
                "-C",
                str(checkout),
                "fetch",
                "-q",
                "--depth=1",
                source.repository,
                source.commit,
            ],
        )
        _run(
            runner,
            ["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"],
        )
        head = _run(
            runner, ["git", "-C", str(checkout), "rev-parse", "HEAD"]
        ).stdout.strip()
        if head != source.commit:
            raise CheckFailed(
                f"{source.repository} checked out {head}, not {source.commit}"
            )
        build_strictly(checkout, f"{source.name}@{source.commit} (git)")
    build_strictly(root, "the project")

    # `uv sync` installs the project editable, which `uv build` cannot build.
    # Ask the locked backend itself what an editable build requires, and
    # require the build group to pin every one of those.
    hooks = work / "hooks"
    _run(
        runner,
        [uv, "venv", "--no-config", "--no-cache", "--python", python, str(hooks)],
    )
    hooks_python = str(hooks / "bin" / "python")
    _run(
        runner,
        [
            *(
                uv,
                "pip",
                "install",
                "--no-config",
                "--no-cache",
                "--python",
                hooks_python,
            ),
            *("--require-hashes", "--no-deps", "-r", str(build_constraints)),
        ],
    )
    probe = _run(runner, [hooks_python, "-I", "-c", EDITABLE_REQUIRES], cwd=root)
    editable = check_editable_requires(json.loads(probe.stdout), build)
    log(f"editable build requires, all locked: {', '.join(editable)}")

    return {
        "python": python,
        "sdists": [f"{s.name}=={s.version}" for s in expected.sdists],
        "git": [f"{g.name}@{g.commit}" for g in expected.git],
        "project": True,
        "editable_requires": editable,
    }


# Run by the locked build environment from the project root: the backend's
# static [build-system] requires plus its get_requires_for_build_editable().
EDITABLE_REQUIRES = """\
import importlib, json, tomllib
system = tomllib.load(open("pyproject.toml", "rb"))["build-system"]
module, _, attribute = system["build-backend"].partition(":")
backend = importlib.import_module(module)
for part in filter(None, attribute.split(".")):
    backend = getattr(backend, part)
hook = getattr(backend, "get_requires_for_build_editable", None)
print(json.dumps([*system.get("requires", []), *(hook() if hook else [])]))
"""


def check_editable_requires(requires: list[str], build_export: str) -> list[str]:
    """Every editable build requirement is a build-group pin that satisfies it."""
    pinned = dict(parse_export(build_export)[0])
    for text in requires:
        requirement = Requirement(text)
        name = canonicalize_name(requirement.name)
        if name not in pinned:
            raise CheckFailed(
                f"the editable build requires {text}, which the build group "
                "does not pin"
            )
        if not requirement.specifier.contains(pinned[name], prereleases=True):
            raise CheckFailed(
                f"the editable build requires {text}, but the build group pins "
                f"{name}=={pinned[name]}"
            )
    return sorted(requires)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--uv", default="uv")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="check-locked-builds-") as work:
        try:
            report = check(work=Path(work), uv=args.uv)
        except CheckFailed as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Encoder version provenance at any git ref, with no third-party imports.

The encoder version must name exactly one encoder state: an encoder-affecting
file may not change after the latest commit that increased the version. The
``--apply`` guard in ``cli.py`` and the provisioner enforce this at ``HEAD``.
This module checks the same rule at any ref, so that

- ``ci_parity`` can refuse an encoder pin whose code is newer than its version,
- the re-encode workflow can refuse an unversioned ``main`` before the encode
  job starts (and before the attempt budget counts the run), and
- the post-merge bump workflow can tell whether ``main`` needs a bump, and make
  it (axiom-encode#2048).

The bump leaves ``changelog.d/`` alone: those are towncrier fragments, which
CI checks with ``towncrier build --draft`` and nothing compiles yet.

It uses only the standard library, so a workflow can run it as a script with a
bare ``python3`` before any dependency is installed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

VERSION_FILES = (
    "pyproject.toml",
    "src/axiom_encode/__init__.py",
    "uv.lock",
)
VERSIONED_PREFIXES = ("src/axiom_encode/",)
VERSIONED_FILES = frozenset({"pyproject.toml", "uv.lock"})
CHANGELOG = "CHANGELOG.md"

_PYPROJECT_VERSION = re.compile(r'(?m)^version\s*=\s*"([^"]+)"\s*$')
_PACKAGE_VERSION = re.compile(r'(?m)^__version__\s*=\s*"([^"]+)"')
_LOCK_VERSION = re.compile(r'(?ms)^name = "axiom-encode"\nversion = "([^"]+)"')


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
    ).stdout


def _show(repo: Path, ref: str, path: str) -> str | None:
    try:
        return _git(repo, "show", "--end-of-options", f"{ref}:{path}", "--")
    except subprocess.CalledProcessError:
        return None


def _first(pattern: re.Pattern[str], text: str | None) -> str | None:
    match = pattern.search(text or "")
    return match.group(1) if match else None


def versions_at(repo: Path, ref: str) -> dict[str, str | None]:
    """The version each version file declares at ``ref``."""
    return {
        "pyproject": _first(_PYPROJECT_VERSION, _show(repo, ref, "pyproject.toml")),
        "package": _first(
            _PACKAGE_VERSION, _show(repo, ref, "src/axiom_encode/__init__.py")
        ),
        "lock": _first(_LOCK_VERSION, _show(repo, ref, "uv.lock")),
    }


def _numeric(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    return tuple(int(part) for part in parts)


def _increased(current: str | None, previous: str | None) -> bool:
    if not current:
        return False
    if not previous:
        return True
    current_key, previous_key = _numeric(current), _numeric(previous)
    if current_key is None or previous_key is None:
        return current > previous
    return current_key > previous_key


def _first_parent(repo: Path, commit: str) -> str | None:
    parts = _git(repo, "rev-list", "--parents", "-n", "1", commit).split()
    return parts[1] if len(parts) > 1 else None


def latest_version_commit(repo: Path, ref: str = "HEAD") -> str | None:
    """The latest commit reachable from ``ref`` that increased the version.

    Same rule as ``cli._latest_axiom_encode_version_commit``: prefer the bump
    to the version ``ref`` declares, else the latest increase of any kind.
    """
    ref_version = versions_at(repo, ref).get("pyproject")
    if not ref_version:
        return None
    fallback = None
    for commit in _git(repo, "log", "--format=%H", ref, "--", *VERSION_FILES).split():
        current = versions_at(repo, commit)
        if not current.get("pyproject") or not current.get("package"):
            continue
        parent = _first_parent(repo, commit)
        previous = versions_at(repo, parent).get("pyproject") if parent else None
        if not _increased(current.get("pyproject"), previous):
            continue
        if fallback is None:
            fallback = commit
        if current.get("pyproject") == ref_version:
            return commit
    return fallback


def is_versioned_path(path: str) -> bool:
    return path in VERSIONED_FILES or path.startswith(VERSIONED_PREFIXES)


def unversioned_changes(repo: Path, ref: str = "HEAD") -> list[str]:
    """Encoder-affecting paths that changed at ``ref`` after its latest bump.

    Raises ``ValueError`` when the version files disagree or no bump exists.
    """
    versions = versions_at(repo, ref)
    if not all(versions.values()) or len(set(versions.values())) != 1:
        raise ValueError(
            f"inconsistent encoder version metadata at {ref} "
            f"(pyproject={versions['pyproject']}, package={versions['package']}, "
            f"lock={versions['lock']})"
        )
    bump = latest_version_commit(repo, ref)
    if bump is None:
        raise ValueError(f"no committed encoder version bump is reachable from {ref}")
    changed = _git(
        repo, "diff", "--name-only", "--diff-filter=ACMRT", f"{bump}..{ref}"
    ).split()
    return [path for path in changed if is_versioned_path(path)]


def next_version(version: str) -> str:
    """Increase the last numeric component: ``0.2.2298`` becomes ``0.2.2299``."""
    key = _numeric(version)
    if key is None:
        raise ValueError(f"cannot increase non-numeric encoder version {version!r}")
    return ".".join(str(part) for part in (*key[:-1], key[-1] + 1))


def _replace_once(path: Path, pattern: re.Pattern[str], version: str) -> None:
    text = path.read_text()
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"{path} declares no encoder version")
    start, end = match.span(1)
    path.write_text(text[:start] + version + text[end:])


def _changelog_entry(repo: Path, ref: str, bump_commit: str | None) -> str:
    """One bullet per commit merged to ``ref`` since the previous bump."""
    span = f"{bump_commit}..{ref}" if bump_commit else ref
    subjects = _git(repo, "log", "--first-parent", "--format=%s", span)
    return "\n".join(f"- {subject}" for subject in subjects.splitlines() if subject)


def bump(repo: Path, ref: str = "HEAD") -> str:
    """Write the next version into the version files and a changelog header.

    Changes the working tree only; the caller commits it. Returns the version.
    """
    current = versions_at(repo, ref)["pyproject"]
    if not current:
        raise ValueError(f"pyproject.toml declares no version at {ref}")
    version = next_version(current)
    _replace_once(repo / "pyproject.toml", _PYPROJECT_VERSION, version)
    _replace_once(repo / "src/axiom_encode/__init__.py", _PACKAGE_VERSION, version)
    _replace_once(repo / "uv.lock", _LOCK_VERSION, version)
    entry = _changelog_entry(repo, ref, latest_version_commit(repo, ref))
    changelog = repo / CHANGELOG
    previous = changelog.read_text() if changelog.exists() else ""
    changelog.write_text(f"## {version}\n\n{entry}\n\n{previous}")
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser(
        "check", help="exit 1 when encoder files changed after the latest bump"
    )
    check.add_argument("--repo", type=Path, default=Path("."))
    check.add_argument("--ref", default="HEAD")
    check.add_argument(
        "--github-output",
        type=Path,
        help="write needs_bump=true|false here and exit 0 instead of failing",
    )
    do_bump = sub.add_parser(
        "bump", help="write the next version and a changelog header"
    )
    do_bump.add_argument("--repo", type=Path, default=Path("."))
    do_bump.add_argument("--ref", default="HEAD")
    args = parser.parse_args(argv)

    if args.command == "bump":
        print(bump(args.repo, args.ref))
        return 0
    try:
        changes = unversioned_changes(args.repo, args.ref)
    except ValueError as exc:
        print(f"encoder version provenance: {exc}", file=sys.stderr)
        return 1
    if args.github_output is not None:
        with args.github_output.open("a") as output:
            output.write(f"needs_bump={'true' if changes else 'false'}\n")
        return 0
    if changes:
        shown = ", ".join(changes[:8])
        more = f", and {len(changes) - 8} more" if len(changes) > 8 else ""
        print(
            f"encoder-affecting files changed at {args.ref} after the latest "
            f"encoder version bump: {shown}{more}. The post-merge version bump "
            "has not landed yet; re-dispatch once it has.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
import os
import re
import shutil
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
# The same patterns as cli's _extract_*_version, so both read one version.
_PACKAGE_VERSION = re.compile(r'(?m)^__version__\s*=\s*"([^"]+)"\s*$')
_LOCK_VERSION = re.compile(
    r'(?ms)^\[\[package\]\]\s*^name\s*=\s*"axiom-encode"\s*^version\s*=\s*"([^"]+)"'
)
_BUMP_BRANCH = "automation/encoder-version-"
# The bump workflow's GitHub App, as a pull request's author login.
BUMP_APP_LOGIN = "axiom-encoder-version-bump[bot]"
BUMP_WORKFLOW = ".github/workflows/encoder-version-bump.yml"
_PR_MERGE = re.compile(r"^Merge pull request #(\d+) from \S+?/(\S+)$")


def _git_environment() -> dict[str, str]:
    """The ambient environment minus GIT_* values, with history rewrites off.

    A GIT_DIR, GIT_CONFIG_* or similar value would read another repository or
    another history than the one at ``repo``. Replacement refs and grafts in
    ``repo`` itself would do the same: ``git replace --graft`` can drop a
    commit's parents, so an unversioned commit looks like its own bump. Every
    read here therefore sees the stored objects only, as ``reviewed_replacement``
    and ``retired_apply`` do. ``log.showSignature`` stays off, so a repository's
    ``gpg.program`` never runs on a read here.
    """
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(
        GIT_GRAFT_FILE=os.devnull,
        GIT_NO_REPLACE_OBJECTS="1",
        GIT_NO_LAZY_FETCH="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_COUNT="2",
        GIT_CONFIG_KEY_0="core.commitGraph",
        GIT_CONFIG_VALUE_0="false",
        GIT_CONFIG_KEY_1="log.showSignature",
        GIT_CONFIG_VALUE_1="false",
        GIT_TERMINAL_PROMPT="0",
    )
    return environment


def _git_binary() -> str:
    """git from the caller's PATH: a scrubbed environment may have none."""
    return shutil.which("git") or "git"


def _git(repo: Path, *args: str, environment: dict[str, str] | None = None) -> str:
    """Run git at ``repo`` under ``environment`` (default: ``_git_environment``).

    A caller with a stricter environment, such as the ``--apply`` guard's
    ``_read_only_git_environment``, passes its own.
    """
    return subprocess.run(
        [_git_binary(), "-C", str(repo), *args],
        capture_output=True,
        check=True,
        text=True,
        env=environment if environment is not None else _git_environment(),
    ).stdout


def show_file(repo: Path, ref: str, path: str) -> str | None:
    """``path`` at ``ref`` as stored, or None when it is missing."""
    return _show(repo, ref, path)


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
    if not parts or any(not part.isdecimal() for part in parts):
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

    Same choice as ``cli._latest_axiom_encode_version_commit`` and the
    provisioner: prefer the bump to the version ``ref`` declares, else the
    latest increase of any kind. ``main_history_changes`` adds two rules on
    ``ref``'s first-parent history that all three copies apply.
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


def _pyproject_versions(
    repo: Path, commits: list[str], environment: dict[str, str] | None = None
) -> list[str | None]:
    """The pyproject version at each commit, read in one ``git cat-file`` run."""
    if not commits:
        return []
    completed = subprocess.run(
        [_git_binary(), "-C", str(repo), "cat-file", "--batch"],
        input="".join(f"{commit}:pyproject.toml\n" for commit in commits).encode(),
        capture_output=True,
        check=True,
        env=environment if environment is not None else _git_environment(),
    )
    data, offset, versions = completed.stdout, 0, []
    for _ in commits:
        end = data.index(b"\n", offset)
        header = data[offset:end].decode().split()
        offset = end + 1
        if len(header) != 3:  # "<object> missing": no content follows
            versions.append(None)
            continue
        size = int(header[2])
        content = data[offset : offset + size]
        offset += size + 1  # every object's content, a blob or not, then LF
        if header[1] != "blob":
            versions.append(None)
            continue
        versions.append(_first(_PYPROJECT_VERSION, content.decode("utf-8", "replace")))
    return versions


def first_parent_versions(
    repo: Path, ref: str = "HEAD", environment: dict[str, str] | None = None
) -> list[tuple[str, str | None]]:
    """``(commit, version)`` for each first-parent commit of ``ref`` that changed
    pyproject.toml against its first parent, newest first.

    This is the version history of ``main`` itself: a merged branch's own
    commits are not on it, and a merge counts as one change to the version
    that ``main`` declares.
    """
    commits = _git(
        repo,
        "log",
        "--first-parent",
        "--format=%H",
        ref,
        "--",
        "pyproject.toml",
        environment=environment,
    ).split()
    return list(zip(commits, _pyproject_versions(repo, commits, environment)))


def _highest(versions: list[str | None]) -> str | None:
    numeric = [(key, v) for v in versions if v and (key := _numeric(v)) is not None]
    return max(numeric)[1] if numeric else None


def _first_parent_arrival(
    history: list[tuple[str, str | None]], version: str
) -> str | None:
    """The oldest first-parent commit at which ``main`` came to declare
    ``version``: every commit after it runs under that version, also when
    ``main`` left the version and declared it again later.

    ``history`` is newest first, so this walks it from the end.
    """
    for index, (commit, declared) in reversed(list(enumerate(history))):
        if declared != version:
            continue
        older = history[index + 1][1] if index + 1 < len(history) else None
        if older != version:
            return commit
    return None


def is_versioned_path(path: str) -> bool:
    return path in VERSIONED_FILES or path.startswith(VERSIONED_PREFIXES)


def unversioned_changes(repo: Path, ref: str = "HEAD") -> list[str]:
    """Encoder-affecting paths that changed at ``ref`` after its latest bump.

    Deletions and both sides of a move count, and ``-z`` keeps non-ASCII names
    unquoted, as in cli and the provisioner. Raises ``ValueError`` when the
    history is shallow, the version files disagree or no bump exists.
    """
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() == "true":
        # A shallow boundary commit reads as a bump and hides older changes.
        raise ValueError(f"{repo} is a shallow clone; the check needs full history")
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
    changed = set(_changed_since(repo, bump, ref))
    changed.update(main_history_changes(repo, ref, bump, versions["pyproject"]))
    return sorted(changed)


def main_history_changes(
    repo: Path,
    ref: str,
    bump: str,
    version: str,
    *,
    environment: dict[str, str] | None = None,
) -> set[str]:
    """What else needs a bump, judged on ``ref``'s first-parent history.

    One version must name one encoder state on main as well (axiom-encode#2051
    review). Two cases need a bump beyond the latest-bump rule, both from
    hand-picked versions. The ``--apply`` guard and the provisioner apply the
    same two rules.

    1. The head declares a version below one that main already declared, for
       example a merge that kept a pull request's older hand bump. Its number
       may already name another state: the version files need a bump.
    2. Main first reached this version at another commit than ``bump``, for
       example the bump workflow's commit when a merged branch carries its own
       hand bump to the same number. Encoder changes after that commit run
       under a number that already names the commit's state.

    A head version that is not numeric reads as below every numeric one, so
    rule 1 asks for a bump.
    """
    changed: set[str] = set()
    history = first_parent_versions(repo, ref, environment)
    highest = _highest([declared for _, declared in history])
    if highest is not None and _numeric(highest) > (_numeric(version) or ()):
        changed.update(VERSION_FILES)
    arrival = _first_parent_arrival(history, version)
    if arrival is not None and arrival != bump:
        changed.update(_changed_since(repo, arrival, ref, environment))
    return changed


def _changed_since(
    repo: Path, base: str, ref: str, environment: dict[str, str] | None = None
) -> list[str]:
    changed = _git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACDMRT",
        "-z",
        base,
        ref,
        environment=environment,
    ).split("\0")
    return [path for path in changed if path and is_versioned_path(path)]


def hand_picked_version(
    repo: Path,
    base: str,
    ref: str,
    *,
    author: str,
    head_repository: str,
    repository: str,
) -> str | None:
    """Why a pull request may not change the encoder version, or None.

    Only the bump workflow's App picks versions once that workflow is on the
    base branch. A pull request from anyone else must leave the version lines
    of ``base`` as they are: a hand bump can land below or beside a number the
    workflow already used. The author decides, not the branch name, which
    anyone can choose; a fork's pull request never counts as the App's.

    Raises ``ValueError`` when ``base`` or ``ref`` does not name a commit, so a
    checkout that cannot read the base refuses instead of passing as a base
    without the bump workflow.
    """
    for commit in (base, ref):
        try:
            _git(
                repo,
                "rev-parse",
                "--verify",
                "--quiet",
                "--end-of-options",
                f"{commit}^{{commit}}",
            )
        except subprocess.CalledProcessError:
            raise ValueError(f"{commit} does not name a commit in {repo}") from None
    if show_file(repo, base, BUMP_WORKFLOW) is None:
        return None  # Before axiom-encode#2048 lands, pull requests still bump.
    if author == BUMP_APP_LOGIN and head_repository == repository:
        return None
    before, after = versions_at(repo, base), versions_at(repo, ref)
    if before == after:
        return None
    return (
        f"this pull request changes the encoder version ({before['pyproject']} "
        f"to {after['pyproject']}); versions are assigned after merge by "
        f"{BUMP_WORKFLOW} (axiom-encode#2048). Take the base branch's "
        "pyproject.toml, src/axiom_encode/__init__.py and uv.lock version lines."
    )


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


def _changelog_line(subject: str, body: str) -> str | None:
    """A pull request merge reads as its title; a bump pull request is left out."""
    merge = _PR_MERGE.match(subject)
    if merge is None:
        return None if subject.startswith("Bump encoder version to ") else subject
    number, branch = merge.groups()
    if branch.startswith(_BUMP_BRANCH):
        return None
    title = next((line.strip() for line in body.splitlines() if line.strip()), "")
    return f"{title} (#{number})" if title else subject


def _changelog_entry(repo: Path, ref: str, bump_commit: str | None) -> str:
    """One bullet per change merged to ``ref`` since the previous bump."""
    span = f"{bump_commit}..{ref}" if bump_commit else ref
    records = _git(repo, "log", "--first-parent", "--format=%s%x00%b%x1e", span)
    lines = []
    for record in records.split("\x1e"):
        subject, _, body = record.strip("\n").partition("\0")
        line = _changelog_line(subject.strip(), body) if subject.strip() else None
        if line:
            lines.append(f"- {line}")
    return "\n".join(lines)


def bump(repo: Path, ref: str = "HEAD") -> str:
    """Write the next version into the version files and a changelog header.

    Changes the working tree only; the caller commits it. Returns the version.
    """
    current = versions_at(repo, ref)["pyproject"]
    if not current:
        raise ValueError(f"pyproject.toml declares no version at {ref}")
    # One more than the highest version main has ever declared, so a merge
    # that lowered the version never makes the bump reissue a used number.
    highest = _highest([current, *(v for _, v in first_parent_versions(repo, ref))])
    version = next_version(highest or current)
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
    pull_request = sub.add_parser(
        "pull-request",
        help="exit 1 when a pull request not opened by the bump App changes "
        "the encoder version",
    )
    pull_request.add_argument("--repo", type=Path, default=Path("."))
    pull_request.add_argument("--base", required=True)
    pull_request.add_argument("--ref", default="HEAD")
    pull_request.add_argument("--author", required=True)
    pull_request.add_argument("--head-repository", required=True)
    pull_request.add_argument("--repository", required=True)
    args = parser.parse_args(argv)

    if args.command == "bump":
        print(bump(args.repo, args.ref))
        return 0
    if args.command == "pull-request":
        try:
            reason = hand_picked_version(
                args.repo,
                args.base,
                args.ref,
                author=args.author,
                head_repository=args.head_repository,
                repository=args.repository,
            )
        except ValueError as exc:
            reason = str(exc)
        if reason:
            print(f"encoder version provenance: {reason}", file=sys.stderr)
            return 1
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

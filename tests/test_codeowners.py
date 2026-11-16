"""Pin code-owner review of everything under ``.github/``.

``main`` requires a code-owner review, with no administrator bypass, but
requires no other approval. A file that ``CODEOWNERS`` does not match
therefore needs no review at all. A workflow file names its own environment,
and an environment's branch rule limits the ref, not the workflow, so an
unowned workflow file can select a key-bearing environment and read its key
(``docs/merge-guard-key-protection.md``, R3 "Present exposure" and section 4,
"Complement").

``/.github/ @PavelMakarchuk @MaxGhenis`` closes that for file ownership: every
workflow (new, replaced or changed), any future composite action and
``CODEOWNERS`` itself has a code owner. This file fails if that stops being
true.

It is a tripwire, not the control. The control is branch protection plus
``CODEOWNERS``; a pull request can edit this test, but only together with a
``CODEOWNERS`` change that itself needs a code owner. It checks the file, not
the repository settings: the code-owner requirement, "Require approval of the
most recent reviewable push" and environment reviewers are admin steps
(section 5 of the proposal).

The matcher reads only the pattern forms this ``CODEOWNERS`` uses: a path
anchored at the repository root, with no wildcard, that names a file or (with
a trailing ``/``) a directory. Any other form is reported as a problem, so
that a pattern this file cannot evaluate never passes silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CODEOWNERS = ROOT / ".github" / "CODEOWNERS"
CODE_OWNERS = frozenset({"@PavelMakarchuk", "@MaxGhenis"})
GITHUB_DIRECTORY = "/.github/"
# The line that step 2 of the proposal's admin checklist greps for (grep -x).
GITHUB_DIRECTORY_LINE = "/.github/ @PavelMakarchuk @MaxGhenis"
# Paths that do not exist today. A new file under .github/ must have an owner
# on the day it is added, without a CODEOWNERS change of its own.
FUTURE_PATHS = (
    ".github/workflows/a-new-workflow.yml",
    ".github/workflows/a-new-workflow.yaml",
    ".github/workflows/nested/a-new-workflow.yml",
    ".github/actions/a-new-action/action.yml",
    ".github/dependabot.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/a-new-file",
)
_LITERAL_PATTERN = re.compile(r"/[A-Za-z0-9._/-]+")

Rule = tuple[int, str, frozenset[str]]


def parse(text: str) -> tuple[list[Rule], list[str]]:
    """Return the rules (line number, pattern, owners) and the unread lines."""

    rules: list[Rule] = []
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *owners = stripped.split()
        if (
            _LITERAL_PATTERN.fullmatch(pattern) is None
            or "//" in pattern
            or any(part in {".", ".."} for part in pattern.split("/"))
        ):
            problems.append(
                f"line {number}: pattern {pattern!r} is not a root-anchored "
                "literal path; this check cannot evaluate it"
            )
            continue
        if any(not owner.startswith("@") for owner in owners):
            problems.append(
                f"line {number}: {pattern!r} has a token that is not an owner "
                "(an inline comment or an email); this check cannot evaluate it"
            )
            continue
        rules.append((number, pattern, frozenset(owners)))
    return rules, problems


def matches(pattern: str, path: str) -> bool:
    """Whether a root-anchored literal pattern matches a repository path."""

    body = pattern.removeprefix("/")
    if body.endswith("/"):
        return path.startswith(body)
    # Without a trailing slash the pattern names a file, or a directory and
    # everything under it.
    return path == body or path.startswith(body + "/")


def owners_of(rules: list[Rule], path: str) -> frozenset[str]:
    """Return the owners of ``path``: the last matching pattern wins."""

    owners: frozenset[str] = frozenset()
    for _number, pattern, rule_owners in rules:
        if matches(pattern, path):
            owners = rule_owners
    return owners


def coverage_problems(text: str, paths: list[str]) -> list[str]:
    """Return every way a path under ``.github/`` can lack a code owner.

    ``paths`` are the repository paths to check. The entry checks cover the
    whole file: an entry anywhere that names fewer than both code owners is a
    problem too.
    """

    rules, problems = parse(text)
    if GITHUB_DIRECTORY_LINE not in text.splitlines():
        problems.append(f"the line {GITHUB_DIRECTORY_LINE!r} is missing")
    directory_lines = [
        number for number, pattern, _owners in rules if pattern == GITHUB_DIRECTORY
    ]
    if len(directory_lines) != 1:
        problems.append(
            f"{GITHUB_DIRECTORY!r} needs exactly one entry, found "
            f"{len(directory_lines)}"
        )
    for number, pattern, owners in rules:
        # A later entry replaces the owners of the paths it matches, and an
        # entry with no owner removes them. Every entry in this file names
        # both code owners, so no order of entries can leave a path with
        # fewer.
        if owners != CODE_OWNERS:
            problems.append(
                f"line {number}: {pattern!r} names {sorted(owners)}, not both "
                "code owners"
            )
        if (
            pattern != GITHUB_DIRECTORY
            and pattern.startswith(GITHUB_DIRECTORY)
            and directory_lines
            and number < directory_lines[0]
        ):
            problems.append(
                f"line {number}: {pattern!r} is above {GITHUB_DIRECTORY!r}, "
                "which would replace it (the last matching pattern wins)"
            )
    for path in paths:
        owners = owners_of(rules, path)
        if owners != CODE_OWNERS:
            problems.append(f"{path}: owners are {sorted(owners)}")
    return problems


def _text() -> str:
    return CODEOWNERS.read_text(encoding="utf-8")


def _files_under_dot_github() -> list[str]:
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / ".github").rglob("*")
        if path.is_file()
    )


def test_code_owners_own_every_file_under_dot_github_and_any_new_one():
    existing = _files_under_dot_github()
    assert ".github/CODEOWNERS" in existing
    assert [path for path in existing if path.startswith(".github/workflows/")]
    assert coverage_problems(_text(), [*existing, *FUTURE_PATHS]) == []


def test_the_line_the_admin_check_greps_for_is_present_once():
    assert _text().splitlines().count(GITHUB_DIRECTORY_LINE) == 1


def test_every_entry_names_both_code_owners():
    rules, problems = parse(_text())
    assert problems == []
    assert rules
    assert {owners for _number, _pattern, owners in rules} == {CODE_OWNERS}


def test_the_entry_does_not_reach_outside_dot_github():
    directory_only = [(1, GITHUB_DIRECTORY, CODE_OWNERS)]
    for path in (
        "src/axiom_encode/cli.py",
        ".githubx/workflows/a.yml",
        "docs/.github/workflows/a.yml",
        ".gitignore",
        "pyproject.toml",
    ):
        assert owners_of(directory_only, path) == frozenset(), path


BOTH = "@PavelMakarchuk @MaxGhenis"


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            f"/.github/CODEOWNERS {BOTH}\n/.github/workflows/ci.yml {BOTH}\n",
            id="main-before-this-change-only-named-files",
        ),
        pytest.param("", id="empty-file"),
        pytest.param(f"# /.github/ {BOTH}\n", id="the-line-commented-out"),
        pytest.param(f"/.github/workflows/ {BOTH}\n", id="workflows-only"),
        pytest.param("/.github/ @PavelMakarchuk\n", id="one-owner"),
        pytest.param("/.github/ @MaxGhenis @someone-else\n", id="another-owner"),
        pytest.param("/.github/\n", id="no-owner"),
        pytest.param(f".github/ {BOTH}\n", id="not-anchored"),
        pytest.param(f"/.github {BOTH}\n", id="no-trailing-slash"),
        pytest.param(f"/.GitHub/ {BOTH}\n", id="other-case"),
        pytest.param(
            "/.github/ @MaxGhenis @PavelMakarchuk\n",
            id="owners-in-another-order-the-admin-grep-would-miss",
        ),
        pytest.param(f"/.github/  {BOTH}\n", id="two-spaces-the-admin-grep-would-miss"),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/workflows/a-new-workflow.yml\n",
            id="a-later-entry-with-no-owner-removes-the-owners",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/workflows/ @PavelMakarchuk\n",
            id="a-later-directory-entry-with-one-owner",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/workflows/ci.yml @someone-else\n",
            id="a-later-entry-gives-a-workflow-to-someone-else",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/workflows/*.yml @someone-else\n",
            id="a-later-wildcard-entry",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n* @someone-else\n", id="a-later-catch-all-entry"
        ),
        pytest.param(
            f"/.github/ {BOTH}\n**/workflows/ @someone-else\n",
            id="a-later-double-star-entry",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/workflows/ci.yml {BOTH} # note\n",
            id="an-inline-comment-this-check-cannot-read",
        ),
        pytest.param(
            f"/.github/workflows/ci.yml {BOTH}\n/.github/ {BOTH}\n",
            id="the-directory-entry-below-a-specific-entry",
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/.github/ {BOTH}\n", id="the-directory-entry-twice"
        ),
        pytest.param(
            f"/.github/ {BOTH}\n/src/axiom_encode/cli.py @PavelMakarchuk\n",
            id="an-entry-elsewhere-loses-an-owner",
        ),
    ],
)
def test_the_tripwire_reports_each_way_to_lose_an_owner(text):
    paths = [".github/CODEOWNERS", ".github/workflows/ci.yml", *FUTURE_PATHS]
    assert coverage_problems(text, paths) != []


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(f"/.github/ {BOTH}\n", id="the-directory-entry-alone"),
        pytest.param(
            f"# comment\n\n/.github/ {BOTH}\n/.github/workflows/ci.yml {BOTH}\n"
            f"/.github/CODEOWNERS {BOTH}\n/src/axiom_encode/cli.py {BOTH}\n"
            f"/cmd/axiom-encode-apply-signer/ {BOTH}\n",
            id="with-more-specific-entries-below",
        ),
    ],
)
def test_the_tripwire_accepts_full_ownership(text):
    paths = [".github/CODEOWNERS", ".github/workflows/ci.yml", *FUTURE_PATHS]
    assert coverage_problems(text, paths) == []


@pytest.mark.parametrize(
    ("pattern", "path", "expected"),
    [
        ("/.github/", ".github/workflows/ci.yml", True),
        ("/.github/", ".github/CODEOWNERS", True),
        ("/.github/", ".github", False),
        ("/.github/", ".githubx/a.yml", False),
        ("/.github/workflows/ci.yml", ".github/workflows/ci.yml", True),
        ("/.github/workflows/ci.yml", ".github/workflows/ci.yml.bak", False),
        ("/.github/workflows/ci.yml", ".github/workflows/ci.yml/nested", True),
        ("/.github/workflows", ".github/workflows/ci.yml", True),
        (
            "/data/reviewed-rulespec-refs/",
            "data/reviewed-rulespec-refs/us/a.json",
            True,
        ),
        ("/go.mod", "go.mod", True),
        ("/go.mod", "cmd/go.mod", False),
    ],
)
def test_the_matcher_follows_root_anchored_literal_patterns(pattern, path, expected):
    assert matches(pattern, path) is expected

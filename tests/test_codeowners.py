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
# GitHub ignores a CODEOWNERS file over 3 MB. This file is a few kilobytes.
MAX_BYTES = 64 * 1024
# The entries on main before the directory entry: the signing workflows, the
# signer, supervisor, broker and trust bootstrap, the scripts the signing job
# runs, and the gates. Each must stay an entry with both owners. A pull
# request may add entries; it may not drop one of these without a reviewed
# change to this list.
REQUIRED_ENTRIES = (
    "/.github/workflows/targeted-signed-reencode.yml",
    "/.github/workflows/encoder-version-bump.yml",
    "/.github/workflows/ci.yml",
    "/src/axiom_encode/version_provenance.py",
    "/src/axiom_encode/prepare_signed_backfill.py",
    "/src/axiom_encode/reviewed_pr_admission.py",
    "/src/axiom_encode/content_merge_guard.py",
    "/src/axiom_encode/applied_manifest_v6.py",
    "/src/axiom_encode/cli.py",
    "/scripts/prepare_reviewed_pr_promotion.py",
    "/tests/test_reviewed_pr_admission.py",
    "/tests/test_content_merge_guard.py",
    "/tests/test_applied_manifest_v6.py",
    "/tests/test_prepare_reviewed_pr_promotion.py",
    "/tests/test_content_merge_guard_infrastructure.py",
    "/data/reviewed-rulespec-refs/",
    "/src/axiom_encode/corpus_artifact_transport.py",
    "/cmd/axiom-encode-apply-signer/",
    "/cmd/axiom-encode-signing-supervisor/",
    "/scripts/provision_verification_supervisor.py",
    "/src/axiom_encode/signing_broker.py",
    "/src/axiom_encode/_trusted_signing_bootstrap.py",
    "/go.mod",
    "/go.sum",
    "/scripts/prepare_signed_backfill.py",
    "/scripts/prepare_signed_queue.py",
    "/scripts/extract_repair_candidate.py",
    "/scripts/verify_repair_base_advance.py",
    "/scripts/materialize_corpus_release.py",
    "/scripts/provision_axiom_compose_runtime.py",
    "/scripts/prepare_retired_source_metadata.py",
    "/.github/workflows/pr-linkage.yml",
    "/src/axiom_encode/pr_linkage.py",
    "/.github/CODEOWNERS",
    "/src/axiom_encode/rulespec_admission_registry.py",
    "/.github/workflows/validation-effect.yml",
    "/scripts/check_validation_effect.py",
)

Rule = tuple[int, str, frozenset[str]]


def parse(text: str) -> tuple[list[Rule], list[str]]:
    """Return the rules (line number, pattern, owners) and the unread lines."""

    rules: list[Rule] = []
    problems: list[str] = []
    # GitHub's own reading of other line breaks and spaces (a form feed, a
    # lone carriage return, U+2028, a no-break space) is not documented, and
    # Python's ``splitlines`` and ``split`` treat them as separators. So the
    # file may hold printable ASCII and "\n" only, lines end at "\n" only,
    # and tokens are separated by spaces only.
    for number, line in enumerate(text.split("\n"), start=1):
        bad = sorted({char for char in line if not " " <= char <= "~"})
        if bad:
            problems.append(
                f"line {number}: characters {bad!r} are not printable ASCII; "
                "this check cannot evaluate the line"
            )
            continue
        stripped = line.strip(" ")
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *owners = [token for token in stripped.split(" ") if token]
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
    if GITHUB_DIRECTORY_LINE not in text.split("\n"):
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


def file_problems(path: Path) -> list[str]:
    """Return why ``path`` is not a small regular file that GitHub will read."""

    if path.is_symlink():
        return [f"{path.name} is a symbolic link"]
    if not path.is_file():
        return [f"{path.name} is not a regular file"]
    size = path.stat().st_size
    if size > MAX_BYTES:
        return [f"{path.name} has {size} bytes, more than {MAX_BYTES}"]
    return []


def path_problems(rules: list[Rule], root: Path) -> list[str]:
    """Return every entry whose pattern names no existing file or directory."""

    problems = []
    for number, pattern, _owners in rules:
        target = root / pattern.strip("/")
        if pattern.endswith("/"):
            exists = target.is_dir() and not target.is_symlink()
        else:
            exists = target.is_file() or target.is_dir()
        if not exists:
            problems.append(f"line {number}: {pattern!r} names no existing path")
    return problems


def required_entry_problems(rules: list[Rule]) -> list[str]:
    """Return every required entry that is missing or lost an owner."""

    patterns = {pattern for _number, pattern, _owners in rules}
    problems = [
        f"required entry {pattern!r} is missing"
        for pattern in REQUIRED_ENTRIES
        if pattern not in patterns
    ]
    for pattern in REQUIRED_ENTRIES:
        # A directory pattern is checked through a path inside it.
        path = pattern.strip("/") + ("/x" if pattern.endswith("/") else "")
        if pattern in patterns and owners_of(rules, path) != CODE_OWNERS:
            problems.append(f"required entry {pattern!r} lost an owner")
    return problems


def _text() -> str:
    # newline="" keeps "\r" visible: the default would translate it to "\n".
    assert file_problems(CODEOWNERS) == []
    with CODEOWNERS.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def test_codeowners_is_a_small_regular_file(tmp_path):
    assert file_problems(CODEOWNERS) == []
    real = tmp_path / "real"
    real.write_text(GITHUB_DIRECTORY_LINE + "\n", encoding="utf-8")
    assert file_problems(real) == []
    link = tmp_path / "CODEOWNERS"
    link.symlink_to(real)
    assert file_problems(link) != []
    assert file_problems(tmp_path / "absent") != []
    assert file_problems(tmp_path) != []
    large = tmp_path / "large"
    large.write_bytes(b"#" * MAX_BYTES + b"\n")
    assert file_problems(large) != []
    large.write_bytes(b"#" * (MAX_BYTES - 1) + b"\n")
    assert file_problems(large) == []
    assert MAX_BYTES * 40 < 3 * 1024 * 1024


def test_every_entry_names_an_existing_path(tmp_path):
    rules, problems = parse(_text())
    assert problems == []
    assert path_problems(rules, ROOT) == []
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("", encoding="utf-8")
    both = CODE_OWNERS
    assert path_problems([(1, "/src/a.py", both), (2, "/src/", both)], tmp_path) == []
    assert path_problems([(1, "/src", both)], tmp_path) == []
    for pattern in ("/src/b.py", "/src/a.py/", "/SRC/A.PY/", "/srcs/", "/a.py"):
        assert path_problems([(1, pattern, both)], tmp_path) != [], pattern


def test_the_required_entries_are_present_with_both_owners():
    rules, problems = parse(_text())
    assert problems == []
    assert len(REQUIRED_ENTRIES) == len(set(REQUIRED_ENTRIES)) == 37
    assert required_entry_problems(rules) == []
    kept = [rule for rule in rules if rule[1] != REQUIRED_ENTRIES[0]]
    assert required_entry_problems(kept) != []
    for pattern in (
        "/.github/workflows/targeted-signed-reencode.yml",
        "/src/axiom_encode/signing_broker.py",
        "/cmd/axiom-encode-apply-signer/",
        "/.github/CODEOWNERS",
    ):
        assert pattern in REQUIRED_ENTRIES
        dropped = [rule for rule in rules if rule[1] != pattern]
        assert required_entry_problems(dropped) != [], pattern
        weakened = [
            (
                number,
                name,
                frozenset({"@PavelMakarchuk"}) if name == pattern else owners,
            )
            for number, name, owners in rules
        ]
        assert required_entry_problems(weakened) != [], pattern


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(
            f"/.github/ {' '.join(sorted(CODE_OWNERS, reverse=True))}\x0c"
            "/.github/workflows/x.yml\n",
            id="form-feed-hides-an-entry-with-no-owner",
        ),
        pytest.param(
            "/.github/ @PavelMakarchuk @MaxGhenis\u2028/.github/workflows/\n",
            id="line-separator",
        ),
        pytest.param(
            "/.github/ @PavelMakarchuk @MaxGhenis\r/.github/workflows/\n",
            id="lone-carriage-return",
        ),
        pytest.param(
            "/.github/ @PavelMakarchuk @MaxGhenis\r\n", id="carriage-return-line-feed"
        ),
        pytest.param(
            "/.github/\u00a0@PavelMakarchuk @MaxGhenis\n", id="no-break-space"
        ),
        pytest.param("/.github/\t@PavelMakarchuk @MaxGhenis\n", id="tab"),
        pytest.param("/.github/ @PavelMakarchuk @MaxGhenis\x0b\n", id="vertical-tab"),
        pytest.param("/.github/ @PavelMakarchuk @MaxGhenis\x85\n", id="next-line"),
        pytest.param(
            "/.github/ @PavelMakarchuk @MaxGhenis\n\x1c/.github/workflows/\n",
            id="file-separator",
        ),
        pytest.param(
            "/.github/ @PavelMakarchuk @MaxGhenis\n# caf\u00e9\n",
            id="non-ascii-in-a-comment",
        ),
        pytest.param(
            "\ufeff/.github/ @PavelMakarchuk @MaxGhenis\n", id="byte-order-mark"
        ),
        pytest.param("/.github/ @PavelMakarchuk @MaxGhenis\x00\n", id="nul"),
    ],
)
def test_the_tripwire_refuses_characters_it_cannot_read(text):
    """Python would split these where GitHub may not, or the reverse."""

    _rules, problems = parse(text)
    assert problems != []
    assert coverage_problems(text, [".github/workflows/ci.yml"]) != []


def test_the_real_file_is_printable_ascii_with_line_feeds_only():
    assert all(char == "\n" or " " <= char <= "~" for char in _text())
    assert _text().endswith("\n")


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
    assert _text().split("\n").count(GITHUB_DIRECTORY_LINE) == 1


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

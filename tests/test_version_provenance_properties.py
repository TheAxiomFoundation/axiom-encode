"""Property tests for the encoder version rule.

The rule has three copies: ``version_provenance`` (the workflows and
``ci_parity``), the ``--apply`` guard in ``cli`` and the provisioner. They must
give one verdict on every history, and the rule must keep its promise that one
version names one encoder state on ``main``.

Invariants, for every history whose head declares one version in all three
version files:

1. The three copies give one verdict, and when they accept they name one bump.
2. Accepted commits on ``main``'s first-parent line that declare one version
   have one encoder state: the same ``src/axiom_encode``, ``pyproject.toml`` and
   ``uv.lock``.
3. A head below a version ``main`` declared, or with a version that is not
   numeric, needs a bump of all three version files.
4. A bump commit written by ``version_provenance.bump`` passes all three copies
   and declares a version above every one ``main`` declared. Intended limit:
   this holds for a numeric head. ``_increased`` compares a non-numeric version
   as text, so a bump from one can fail the workflow's own check, which leaves
   the head refused.

Histories are written with one ``git fast-import`` run per example, so commit
dates, merge parents and merge trees can be set freely.
"""

from __future__ import annotations

import importlib.util
import shutil
import string
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, example, given, note, settings
from hypothesis import strategies as st

from axiom_encode import cli, version_provenance

_SPEC = importlib.util.spec_from_file_location(
    "provision_verification_supervisor_for_properties",
    Path(__file__).parents[1] / "scripts" / "provision_verification_supervisor.py",
)
provisioner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = provisioner
_SPEC.loader.exec_module(provisioner)

# Decimal digits from other scripts, which int() reads, and digit-like
# characters that it refuses: superscripts, a circled digit and a fraction.
_DIGIT_LIKE = "".join(chr(code) for code in (0x663, 0xFF11, 0xB2, 0xB3, 0x2460, 0xBD))
_VERSION_PART = st.text(alphabet=string.digits + _DIGIT_LIKE + "rc-_+ ", max_size=4)
_VERSIONS = st.lists(_VERSION_PART, min_size=1, max_size=4).map(".".join)


@settings(max_examples=300, deadline=None)
@given(version=_VERSIONS)
def test_every_copy_reads_a_version_the_same_way(version: str):
    expected = cli._parse_numeric_version(version)
    assert version_provenance._numeric(version) == expected
    assert provisioner._parse_numeric_version(version) == expected


@settings(max_examples=300, deadline=None)
@given(
    current=st.one_of(st.none(), _VERSIONS), previous=st.one_of(st.none(), _VERSIONS)
)
def test_every_copy_judges_an_increase_the_same_way(
    current: str | None, previous: str | None
):
    now, before = {"pyproject": current}, {"pyproject": previous}
    expected = cli._axiom_encode_version_increased(now, before)
    assert version_provenance._increased(current, previous) is expected
    assert provisioner._encoder_version_increased(now, before) is expected


@settings(max_examples=200, deadline=None)
@given(version=_VERSIONS)
def test_a_version_is_never_an_increase_over_itself(version: str):
    same = {"pyproject": version}
    assert not version_provenance._increased(version, version)
    assert not cli._axiom_encode_version_increased(same, same)
    assert not provisioner._encoder_version_increased(same, same)


@settings(max_examples=200, deadline=None)
@given(parts=st.lists(st.integers(0, 10_000), min_size=1, max_size=4))
def test_next_version_is_always_an_increase(parts: list[int]):
    version = ".".join(str(part) for part in parts)
    following = version_provenance.next_version(version)
    assert version_provenance._increased(following, version)
    assert version_provenance._numeric(following) > tuple(parts)


_VERSION_FILE_LINES = st.sampled_from(
    [
        'version = "1.2"',
        'version="1.2"  ',
        '  version = "1.2"',
        'version = "1.2" # note',
        'version = ""',
        '__version__ = "3.4"',
        '__version__="3.4"\t',
        '__version__ = "3.4" # note',
        "[[package]]",
        'name = "axiom-encode"',
        'name = "other"',
        'source = { editable = "." }',
        "",
    ]
)
_VERSION_FILE_TEXT = st.lists(_VERSION_FILE_LINES, max_size=8).map("\n".join)


@settings(max_examples=300, deadline=None)
@given(text=_VERSION_FILE_TEXT)
def test_every_copy_extracts_one_version_from_a_file(text: str):
    for pattern, in_cli, in_provisioner in (
        (
            version_provenance._PYPROJECT_VERSION,
            cli._extract_pyproject_version,
            provisioner._extract_pyproject_version,
        ),
        (
            version_provenance._PACKAGE_VERSION,
            cli._extract_package_init_version,
            provisioner._extract_package_init_version,
        ),
        (
            version_provenance._LOCK_VERSION,
            cli._extract_uv_lock_version,
            provisioner._extract_uv_lock_version,
        ),
    ):
        expected = in_cli(text)
        assert version_provenance._first(pattern, text) == expected
        assert in_provisioner(text) == expected


_PATHS = st.one_of(
    st.sampled_from(
        ["pyproject.toml", "uv.lock", "CHANGELOG.md", "docs/x.md", "src/axiom_encode"]
    ),
    st.text(max_size=12).map("src/axiom_encode/".__add__),
    st.text(max_size=20),
)


@settings(max_examples=200, deadline=None)
@given(path=_PATHS)
def test_every_copy_counts_the_same_paths(path: str):
    expected = version_provenance.is_versioned_path(path)
    assert cli._is_axiom_encode_versioned_path(path) is expected
    assert provisioner._is_encoder_versioned_path(path) is expected


def test_every_copy_watches_the_same_version_files():
    assert (
        tuple(version_provenance.VERSION_FILES)
        == tuple(cli._AXIOM_ENCODE_VERSION_FILES)
        == tuple(provisioner._ENCODER_VERSION_FILES)
    )


@settings(max_examples=300, deadline=None)
@given(
    declared=st.lists(st.sampled_from(["1", "2", "3", None]), max_size=8),
    version=st.sampled_from(["1", "2", "3"]),
)
def test_the_arrival_is_the_oldest_commit_that_declares_the_version(
    declared: list[str | None], version: str
):
    history = [(f"c{index}", value) for index, value in enumerate(declared)]
    oldest = next((c for c, value in reversed(history) if value == version), None)
    assert version_provenance._first_parent_arrival(history, version) == oldest


# --- Generated histories ------------------------------------------------------

_BASE_DATE = 1_780_000_000
_VERSION_POOL = ("0.2.9", "0.2.10", "0.2.11", "0.2.12", "0.2.11rc1", "0.10.0")
_EXTRA_FILE = f"src/axiom_encode/donn{chr(0xE9)}es-{{}}.py"


@dataclass(frozen=True)
class _Tree:
    """One commit's files, each field an independent change."""

    version: str = "0.2.10"
    encoder: int = 0
    extra: int = 0
    lock_dep: int = 0
    py_dep: int = 0
    docs: int = 0

    def files(self) -> dict[str, str]:
        dependencies = f'dependencies = ["dep{self.py_dep}"]\n' if self.py_dep else ""
        lock_dep = (
            f'\n[[package]]\nname = "dep{self.lock_dep}"\nversion = "1"\n'
            if self.lock_dep
            else ""
        )
        files = {
            "pyproject.toml": (
                f'[project]\nname = "axiom-encode"\nversion = "{self.version}"\n'
                f"{dependencies}"
            ),
            "src/axiom_encode/__init__.py": f'__version__ = "{self.version}"\n',
            "src/axiom_encode/cli.py": f"MARKER = {self.encoder}\n",
            "uv.lock": (
                f'[[package]]\nname = "axiom-encode"\nversion = "{self.version}"\n'
                f'source = {{ editable = "." }}\n{lock_dep}'
            ),
            "docs.md": f"{self.docs}\n",
        }
        if self.extra:
            files[_EXTRA_FILE.format(self.extra)] = "EXTRA = 1\n"
        return files

    @property
    def encoder_state(self) -> tuple[object, ...]:
        return (self.version, self.encoder, self.extra, self.lock_dep, self.py_dep)


@dataclass(frozen=True)
class _Commit:
    mark: int
    parents: tuple[int, ...]
    tree: _Tree
    date: int


def _data(content: bytes) -> bytes:
    return f"data {len(content)}\n".encode() + content + b"\n"


class _History:
    """A commit graph in memory, written to a repository in one process."""

    def __init__(self) -> None:
        self.commits: dict[int, _Commit] = {}
        self.clock = _BASE_DATE
        self.changes = 0
        self.main = [self._commit((), _Tree(), self.clock)]
        self.shas: dict[int, str] = {}

    def _commit(self, parents: tuple[int, ...], tree: _Tree, date: int) -> int:
        mark = len(self.commits) + 1
        self.commits[mark] = _Commit(mark, parents, tree, date)
        return mark

    def tree(self, mark: int) -> _Tree:
        return self.commits[mark].tree

    def _edited(self, tree: _Tree, edit: tuple) -> _Tree:
        self.changes += 1
        kind = edit[0]
        if kind == "version":
            return replace(tree, version=edit[1])
        if kind == "revert_encoder":
            return replace(tree, encoder=0)
        if kind == "toggle_extra":
            return replace(tree, extra=0 if tree.extra else self.changes)
        return replace(tree, **{kind: self.changes})

    def _on_main(self, edit: tuple, date: int) -> None:
        tree = self._edited(self.tree(self.main[-1]), edit)
        self.main.append(self._commit((self.main[-1],), tree, date))

    def apply(self, operation: tuple) -> None:
        if operation[0] != "merge":
            self.clock += 60
            self._on_main(operation, self.clock)
            return
        _, back, side_edits, main_edits, versions_from, rest_from, side_later = (
            operation
        )
        start = self.main[max(0, len(self.main) - 1 - back)]
        side_date = self.clock + (1_000 if side_later else 1)
        main_date = self.clock + (1 if side_later else 1_000)
        self.clock += 2_000
        side, side_tree = start, self.tree(start)
        for offset, edit in enumerate(side_edits):
            side_tree = self._edited(side_tree, edit)
            side = self._commit((side,), side_tree, side_date + offset)
        for offset, edit in enumerate(main_edits):
            self._on_main(edit, main_date + offset)
        main_tree = self.tree(self.main[-1])
        source = {
            "main": main_tree,
            "side": side_tree,
        }.get(versions_from)
        if source is None:  # a position on main's first-parent line
            source = self.tree(self.main[versions_from % len(self.main)])
        rest = main_tree if rest_from == "main" else side_tree
        merged = replace(
            rest,
            version=source.version,
            lock_dep=source.lock_dep,
            py_dep=source.py_dep,
        )
        self.main.append(self._commit((self.main[-1], side), merged, self.clock))

    def _stream(self) -> bytes:
        out = bytearray()
        for commit in self.commits.values():
            out += f"commit refs/heads/main\nmark :{commit.mark}\n".encode()
            out += f"committer T <t@example.com> {commit.date} +0000\n".encode()
            out += _data(f"commit {commit.mark}".encode())
            if commit.parents:
                out += f"from :{commit.parents[0]}\n".encode()
                for parent in commit.parents[1:]:
                    out += f"merge :{parent}\n".encode()
            out += b"deleteall\n"
            for path, text in sorted(commit.tree.files().items()):
                out += f"M 100644 inline {path}\n".encode()
                out += _data(text.encode())
            out += b"\n"
        out += f"reset refs/heads/main\nfrom :{self.main[-1]}\n\n".encode()
        return bytes(out)

    def write(self, root: Path) -> Path:
        repo = root / "axiom-encode"
        marks = root / "marks"
        subprocess.run(
            ["git", "init", "-q", "-b", "main", str(repo)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "fast-import",
                "--quiet",
                f"--export-marks={marks}",
            ],
            input=self._stream(),
            check=True,
            capture_output=True,
        )
        for line in marks.read_text().splitlines():
            mark, sha = line.split()
            self.shas[int(mark.removeprefix(":"))] = sha
        return repo


_EDITS = st.one_of(
    st.tuples(st.just("version"), st.sampled_from(_VERSION_POOL)),
    st.sampled_from(
        [
            ("encoder",),
            ("revert_encoder",),
            ("toggle_extra",),
            ("lock_dep",),
            ("py_dep",),
            ("docs",),
        ]
    ),
)
_MERGES = st.tuples(
    st.just("merge"),
    st.integers(0, 4),  # how far back on main the side branch starts
    st.lists(_EDITS, min_size=1, max_size=3),  # the side branch's commits
    st.lists(_EDITS, max_size=2),  # main moves on while it is open
    st.one_of(st.sampled_from(["main", "side"]), st.integers(0, 12)),  # version files
    st.sampled_from(["main", "side"]),  # the merge's other files
    st.booleans(),  # the side branch's commits are newer than main's
)
_OPERATIONS = st.lists(st.one_of(_EDITS, _MERGES), min_size=1, max_size=7)

_REVIEW_NON_NUMERIC_HEAD = [("version", "0.2.11rc1")]
_REVIEW_LEFT_AND_DECLARED_AGAIN = [
    ("encoder",),
    ("version", "0.2.9"),
    ("version", "0.2.10"),
]
# main bumps to 0.2.11 and adds a dependency; a side branch from before the bump
# goes to 0.2.12 with newer commit dates; the merge restores the bump's files.
_REVIEW_RESTORING_MERGE = [
    (
        "merge",
        0,
        [("version", "0.2.12")],
        [("version", "0.2.11"), ("lock_dep",)],
        1,
        "main",
        True,
    )
]


@contextmanager
def _written(operations: list[tuple]) -> Iterator[tuple[_History, Path, Path]]:
    note(f"operations={operations!r}")
    history = _History()
    for operation in operations:
        history.apply(operation)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        root = Path(directory)
        yield history, history.write(root), root


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=True, text=True
    ).stdout.strip()


@dataclass(frozen=True)
class _Verdict:
    refuses: bool
    bump: str | None


def _verdicts(repo: Path, home: Path) -> dict[str, _Verdict]:
    """Each copy's verdict on HEAD, with the bump it measured from if it accepts."""
    changed = version_provenance.unversioned_changes(repo)
    provenance = _Verdict(
        bool(changed),
        None if changed else version_provenance.latest_version_commit(repo),
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            cli,
            "__version__",
            version_provenance.versions_at(repo, "HEAD")["pyproject"],
        )
        try:
            accepted = cli._require_axiom_encode_version_provenance(repo)
            guard = _Verdict(False, accepted["version_commit"])
        except RuntimeError:
            guard = _Verdict(True, None)
    try:
        bump = provisioner._require_snapshot_version_bump(
            Path(shutil.which("git")),
            repo / ".git",
            home,
            _git(repo, "rev-parse", "HEAD"),
        )
        provisioned = _Verdict(False, bump)
    except SystemExit:
        provisioned = _Verdict(True, None)
    return {
        "version_provenance": provenance,
        "guard": guard,
        "provisioner": provisioned,
    }


_HISTORY_SETTINGS = {
    "deadline": None,
    "database": None,
    "suppress_health_check": [HealthCheck.too_slow, HealthCheck.data_too_large],
}


@settings(max_examples=12, **_HISTORY_SETTINGS)
@given(operations=_OPERATIONS)
@example(operations=_REVIEW_NON_NUMERIC_HEAD)
@example(operations=_REVIEW_LEFT_AND_DECLARED_AGAIN)
@example(operations=_REVIEW_RESTORING_MERGE)
def test_the_copies_give_one_verdict_and_a_bump_clears_a_refusal(
    operations: list[tuple],
):
    with _written(operations) as (history, repo, root):
        verdicts = _verdicts(repo, root / "home")
        note(f"verdicts={verdicts!r}")
        assert len(set(verdicts.values())) == 1, verdicts

        declared = [history.tree(mark).version for mark in history.main]
        head_version = declared[-1]
        head_key = version_provenance._numeric(head_version)
        numeric = [
            key for key in map(version_provenance._numeric, declared) if key is not None
        ]
        changes = version_provenance.unversioned_changes(repo)
        if numeric and (head_key is None or max(numeric) > head_key):
            assert set(version_provenance.VERSION_FILES) <= set(changes)

        if not changes or head_key is None:
            return
        _git(repo, "checkout", "-q", "-f", "main")
        written = version_provenance.bump(repo)
        _git(repo, "add", "-A")
        _git(
            repo,
            "-c",
            "user.email=t@example.com",
            "-c",
            "user.name=T",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            f"Bump encoder version to {written}",
        )
        assert version_provenance._numeric(written) > max(numeric)
        verdicts = _verdicts(repo, root / "home")
        note(f"after the bump: {verdicts!r}")
        assert {verdict.refuses for verdict in verdicts.values()} == {False}, verdicts
        assert len(set(verdicts.values())) == 1, verdicts


@settings(max_examples=8, **_HISTORY_SETTINGS)
@given(operations=_OPERATIONS)
@example(operations=_REVIEW_LEFT_AND_DECLARED_AGAIN)
def test_one_version_names_one_encoder_state_on_main(operations: list[tuple]):
    with _written(operations) as (history, repo, _root):
        states: dict[str, tuple[object, ...]] = {}
        for mark in history.main:
            if version_provenance.unversioned_changes(repo, history.shas[mark]):
                continue
            tree = history.tree(mark)
            assert states.setdefault(tree.version, tree.encoder_state) == (
                tree.encoder_state
            ), f"{tree.version} names two accepted encoder states"

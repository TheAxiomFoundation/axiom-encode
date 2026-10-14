"""Signed SNAP queues built from a corpus checkout that keeps its bytes in lock files.

After axiom-corpus moved ``data/corpus/{sources,inventory,provisions,coverage}``
out of git, a checkout has ``.axiom/corpus-locks/`` and no provisions files.
``scripts/prepare_signed_queue.py`` lists the Oregon and Utah manual
directories from the checkout's locks as well as from disk, and places the
signed release's bytes before it hashes or reads them. These tests build each
corpus twice in real git repositories, before the switch (files tracked) and
after it (the same files moved into lock files, as ``corpus migrate`` does),
and check that both builds agree.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import axiom_encode.corpus_materialize as cm
import scripts.prepare_signed_queue as signed_queue
from scripts.prepare_signed_queue import (
    ALL_STATE_EXPECTED_COUNTS,
    ALL_STATE_INVENTORY_PATH,
    PRIORITY_CITATIONS,
    build_all_state_snap_queue,
    build_snap_queue,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    write_test_release_object,
)

GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(GIT is None, reason="git is required")
ROOT = Path(__file__).parents[1]
NAME = "test-signed-queue-release"
UT_BLOCK = PRIORITY_CITATIONS[0]
OR_PRIORITY_PAGES = (272, 273, 274)
OR_DIR = "data/corpus/provisions/us-or/manual"
UT_DIR = "data/corpus/provisions/us-ut/manual"
Scope = tuple[str, str, str]


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path_factory, monkeypatch):
    """Keep the real cache, R2 credentials and fetch switch out of every test."""

    monkeypatch.setenv("HOME", str(tmp_path_factory.mktemp("home")))
    for name in (
        cm.CACHE_ENV,
        cm.NO_FETCH_ENV,
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_ENDPOINT",
        "R2_ACCOUNT_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    cm._PRUNED_STAGING_DIRECTORIES.clear()


# ------------------------------------------------------------------- fixtures


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            GIT,
            "-c",
            "user.name=Axiom test",
            "-c",
            "user.email=test@axiom.invalid",
            "-C",
            str(repo),
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _record(citation: str, *, snap: bool = True) -> dict[str, object]:
    return {
        "body": "SNAP households qualify." if snap else "Cash assistance only.",
        "citation_path": citation,
        "heading": f"Heading for {citation}",
        "kind": "page",
    }


def _rows(records: list[dict[str, object]]) -> bytes:
    return "".join(json.dumps(r, sort_keys=True) + "\n" for r in records).encode()


def _scope_files(scope: Scope, provisions: bytes) -> dict[str, bytes]:
    jurisdiction, document_class, version = scope
    prefix = f"{jurisdiction}/{document_class}"
    return {
        f"data/corpus/provisions/{prefix}/{version}.jsonl": provisions,
        f"data/corpus/inventory/{prefix}/{version}.json": b"{}\n",
        f"data/corpus/coverage/{prefix}/{version}.json": b"{}\n",
        f"data/corpus/sources/{prefix}/{version}/fixture-source.txt": b"source\n",
    }


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


@dataclass
class Corpus:
    """One corpus repository, its signed release, and what that release signs."""

    root: Path
    release_scopes: list[Scope]
    signed: dict[str, bytes]
    unmaterialized: list[Scope] = field(default_factory=list)
    release_sha: str = ""

    @property
    def head(self) -> str:
        return _git(self.root, "rev-parse", "HEAD").strip()

    def sign(self) -> str:
        """Sign the release over ``signed`` at HEAD; leave the worktree as found."""

        saved = {
            path: (self.root / path).read_bytes()
            if (self.root / path).is_file()
            else None
            for path in self.signed
        }
        _write_files(self.root, self.signed)
        try:
            self.release_sha = write_test_release_object(
                self.root,
                NAME,
                self.release_scopes,
                git_commit=self.head,
                unmaterialized_scopes=self.unmaterialized,
            )
        finally:
            for path, data in saved.items():
                if data is None:
                    (self.root / path).unlink()
                else:
                    (self.root / path).write_bytes(data)
        return self.release_sha

    @property
    def release_object(self) -> Path:
        return self.root / "releases" / NAME / f"{self.release_sha}.json"


def _pre_switch_corpus(
    root: Path,
    contents: dict[Scope, bytes],
    *,
    release_scopes: list[Scope] | None = None,
    unmaterialized: list[Scope] = (),
    committed_overrides: dict[str, bytes] | None = None,
    tracked: dict[str, bytes] | None = None,
) -> Corpus:
    """A corpus that tracks its files, signed by a release bound to its HEAD.

    ``committed_overrides`` replaces some committed bytes after the release's
    bytes are chosen, so the release no longer matches the commit.
    """

    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    (root / ".gitignore").write_text("data/\n/releases/\n")
    scopes = list(contents)
    manifest = root / "manifests/releases" / f"{NAME}.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "name": NAME,
                "scopes": [
                    {"jurisdiction": j, "document_class": dc, "version": v}
                    for j, dc, v in scopes
                ],
            },
            indent=2,
        )
    )
    signed: dict[str, bytes] = {}
    for scope in scopes:
        signed.update(_scope_files(scope, contents[scope]))
    _write_files(root, {**signed, **(committed_overrides or {})})
    _write_files(root, tracked or {})
    _git(root, "add", ".")
    _git(root, "add", "-f", "data")
    _git(root, "commit", "--quiet", "-m", "pre-switch corpus")
    release_scopes = scopes if release_scopes is None else release_scopes
    corpus = Corpus(
        root,
        release_scopes,
        {
            path: data
            for scope in release_scopes
            for path, data in signed.items()
            if path in _scope_files(scope, b"")
        },
        list(unmaterialized),
    )
    corpus.sign()
    return corpus


def _switch(corpus: Corpus) -> None:
    """Move tracked protected files into lock files, as ``corpus migrate`` does.

    Each lock entry keeps the blob that held its bytes, so git history can
    serve them. The release is re-cut at the switch commit over the same
    bytes, and ``data/`` leaves the worktree.
    """

    root = corpus.root
    by_lock: dict[PurePosixPath, dict[str, cm.LockEntry]] = {}
    for line in _git(root, "ls-tree", "-r", "HEAD", "--", "data/corpus").splitlines():
        meta, path = line.split("\t", 1)
        blob = meta.split()[2]
        data = (root / path).read_bytes()
        lock_path = cm.lock_path_for(path)
        assert lock_path is not None, path
        by_lock.setdefault(lock_path, {})[path] = cm.LockEntry(
            hashlib.sha256(data).hexdigest(), len(data), blob
        )
    for lock_path, entries in by_lock.items():
        scope = (*lock_path.parts[-3:-1], lock_path.stem)
        target = root / lock_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(cm._canonical_lock_bytes(scope, entries))
    _git(root, "rm", "-r", "--cached", "--quiet", "data/corpus")
    _git(root, "add", str(cm.LOCK_ROOT))
    _git(root, "commit", "--quiet", "-m", "switch")
    shutil.rmtree(root / "data")
    corpus.sign()
    # Signing wrote and removed the release's files; drop the empty tree too.
    shutil.rmtree(root / "data", ignore_errors=True)


def _rulespec(base: Path, release_sha: str) -> tuple[Path, str]:
    rulespec = base / f"rulespec-{release_sha[:12]}"
    if not rulespec.exists():
        toolchain = rulespec / ".axiom/toolchain.toml"
        toolchain.parent.mkdir(parents=True)
        toolchain.write_text(
            "[toolchain]\n"
            f'axiom_corpus_release = "{NAME}"\n'
            f'axiom_corpus_release_content_sha256 = "{release_sha}"\n'
        )
        _git(rulespec, "init", "--quiet")
        _git(rulespec, "add", ".")
        _git(rulespec, "commit", "--quiet", "-m", "rulespec")
    return rulespec, _git(rulespec, "rev-parse", "HEAD").strip()


def _build_arguments(corpus: Corpus, base: Path, *, root: Path | None = None) -> dict:
    root = corpus.root if root is None else root
    rulespec, rulespec_ref = _rulespec(base, corpus.release_sha)
    public_key = base / "release-public-key"
    public_key.write_text(TEST_RELEASE_PUBLIC_KEY + "\n")
    return {
        "corpus_root": root,
        "rulespec_root": rulespec,
        "corpus_ref": _git(root, "rev-parse", "HEAD").strip(),
        "rulespec_ref": rulespec_ref,
        "rules_engine_ref": "c" * 40,
        "release_name": NAME,
        "release_content_sha256": corpus.release_sha,
        "release_object_path": corpus.release_object,
        "release_public_key_path": public_key,
        "state": "paused",
        "pause_reason": "Awaiting protected activation.",
    }


def _build(corpus: Corpus, base: Path, *, root: Path | None = None) -> dict:
    arguments = _build_arguments(corpus, base, root=root)
    return build_snap_queue(
        arguments.pop("corpus_root"), arguments.pop("rulespec_root"), **arguments
    )


def _outcome(build) -> tuple[str, object, str]:
    """("ok", payload, stderr) or ("error", message, stderr)."""

    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            return "ok", build(), stderr.getvalue()
        except ValueError as exc:
            return "error", str(exc), stderr.getvalue()


def _same_queue_but_refs(pre: dict, post: dict) -> None:
    """Two queues that differ only by the corpus commit and the release object."""

    assert pre["items"] == post["items"]
    assert pre["expected_counts"] == post["expected_counts"]
    strip = lambda q: {  # noqa: E731
        **q,
        "dispatch": {**q["dispatch"], "corpus_ref": None, "rulespec_ref": None},
        "release": {**q["release"], "content_sha256": None},
    }
    assert strip(pre) == strip(post)


def _snap_contents(
    or_scopes: dict[str, list[dict[str, object]]],
    ut_scopes: dict[str, list[dict[str, object]]],
) -> dict[Scope, bytes]:
    contents: dict[Scope, bytes] = {}
    for version, records in or_scopes.items():
        contents[("us-or", "manual", version)] = _rows(records)
    for version, records in ut_scopes.items():
        contents[("us-ut", "manual", version)] = _rows(records)
    return contents


def _expected_counts(contents: dict[Scope, bytes]) -> dict[str, int]:
    """The queue each fixture defines: one item per SNAP Oregon page, one per
    Utah topic (a block citation without ``/block-N``), plus the priority block."""

    or_items: set[str] = set()
    ut_items = {UT_BLOCK}
    for (jurisdiction, _, _), data in contents.items():
        for line in data.splitlines():
            record = json.loads(line)
            if "SNAP" not in record["body"]:
                continue
            citation = record["citation_path"]
            if jurisdiction == "us-or":
                or_items.add(citation)
            else:
                ut_items.add(signed_queue.BLOCK_SUFFIX_PATTERN.sub("", citation))
    return {
        "total": len(or_items) + len(ut_items),
        "us-or": len(or_items),
        "us-ut": len(ut_items),
    }


def _basic_contents() -> dict[Scope, bytes]:
    pages = [*OR_PRIORITY_PAGES, 5, 9]
    return _snap_contents(
        {
            "2026-07-16-or-notebook": [
                _record(f"us-or/manual/odhs/open/page-{n}") for n in pages[:3]
            ]
            + [_record("us-or/manual/odhs/open/page-1", snap=False)],
            "2026-07-20-or-supplement": [
                _record(f"us-or/manual/odhs/open/page-{n}") for n in pages[3:]
            ],
        },
        {
            "2026-05-27-ut-manuals": [
                _record(UT_BLOCK),
                _record("us-ut/manual/dws/eligibility/topic-a/block-1"),
                _record("us-ut/manual/dws/eligibility/topic-a/block-2"),
            ],
            "2026-07-13-recovery": [
                _record("us-ut/manual/dws/eligibility/topic-b/block-1"),
                _record("us-ut/manual/dws/eligibility/topic-c", snap=False),
            ],
        },
    )


def _provisions_on_disk(root: Path) -> dict[str, bytes]:
    base = root / "data/corpus/provisions"
    if not base.exists():
        return {}
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }


def _staging_leftovers(root: Path) -> list[Path]:
    staging = root / cm.FETCH_TEMP_DIR
    return sorted(staging.iterdir()) if staging.exists() else []


def _tree_snapshot(root: Path) -> dict[str, tuple[int, int]]:
    snapshot = {}
    for directory, names, files in os.walk(root):
        names[:] = [n for n in names if n != ".git"]
        for name in [*names, *files]:
            path = Path(directory) / name
            info = path.lstat()
            snapshot[path.relative_to(root).as_posix()] = (
                info.st_size,
                info.st_mtime_ns,
            )
    return snapshot


# ----------------------------------------------------------- Oregon and Utah


def test_lock_checkout_builds_the_queue_the_pre_switch_checkout_builds(
    tmp_path, monkeypatch, capsys
):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    pre = _build(corpus, tmp_path)
    pre_disk = _provisions_on_disk(corpus.root)

    _switch(corpus)
    assert cm.corpus_uses_lock_files(corpus.root)
    assert _provisions_on_disk(corpus.root) == {}
    capsys.readouterr()

    post = _build(corpus, tmp_path)

    _same_queue_but_refs(pre, post)
    assert post["dispatch"]["corpus_ref"] == corpus.head
    assert [item["citation"] for item in post["items"][:4]] == list(PRIORITY_CITATIONS)
    # Exactly the four manual provisions files, from git history, byte for byte.
    assert _provisions_on_disk(corpus.root) == pre_disk
    assert "placed 4 corpus release artifact(s)" in capsys.readouterr().err
    for path in pre_disk:
        assert (corpus.root / path).stat().st_nlink == 1
    assert not (corpus.root / "data/corpus/inventory").exists()
    assert not (corpus.root / "data/corpus/sources").exists()
    assert _staging_leftovers(corpus.root) == []
    # data/ is ignored, so the provenance check still sees a clean checkout.
    assert _git(corpus.root, "status", "--porcelain") == ""

    # A second build (the activation workflow regenerates the queue) reads
    # what is in place and places nothing.
    again = _build(corpus, tmp_path)
    assert again == post
    assert "placed" not in capsys.readouterr().err


def test_pre_switch_checkout_never_reaches_the_lock_code(tmp_path, monkeypatch):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    before = _tree_snapshot(corpus.root)

    def refuse(*args, **kwargs):
        raise AssertionError("a pre-switch checkout reached the lock-file code")

    for name in (
        "materialize_release_artifacts",
        "locked_directory_listing",
        "release_sources",
        "destination_state",
    ):
        monkeypatch.setattr(cm, name, refuse)
    monkeypatch.setattr(signed_queue, "_place_release_sources", refuse)

    payload = _build(corpus, tmp_path)

    assert len(payload["items"]) == _expected_counts(contents)["total"]
    assert _tree_snapshot(corpus.root) == before
    assert not (corpus.root / cm.FETCH_TEMP_DIR).exists()


def test_a_locked_scope_the_release_omits_fails_as_a_tracked_one_does(
    tmp_path, monkeypatch
):
    """The checkout's locks list the directory, not the release.

    A release that leaves out one of the checkout's Utah manual scopes fails
    the build before and after the switch, with the same message. Deriving
    the sources from the release alone would drop that scope silently.
    """

    contents = _basic_contents()
    omitted = ("us-ut", "manual", "2026-07-13-recovery")
    kept = [scope for scope in contents if scope != omitted]
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(
        tmp_path / "axiom-corpus", contents, release_scopes=kept
    )
    message = f"signed release omits queue source artifact: {UT_DIR}/{omitted[2]}.jsonl"
    with pytest.raises(ValueError, match=message):
        _build(corpus, tmp_path)

    _switch(corpus)
    with pytest.raises(ValueError, match=message):
        _build(corpus, tmp_path)
    # Nothing was placed: the omission is found before any write.
    assert _provisions_on_disk(corpus.root) == {}
    assert not (corpus.root / cm.FETCH_TEMP_DIR).exists()


def test_a_release_artifact_the_checkout_lacks_is_ignored_before_and_after(
    tmp_path, monkeypatch
):
    """A release may list an Oregon manual the corpus commit does not hold.

    A pre-switch build never sees it (it is not on disk); the lock-file build
    never lists it (no lock pins it). Both build the same queue.
    """

    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(
        tmp_path / "axiom-corpus",
        contents,
        unmaterialized=[("us-or", "manual", "2026-08-01-or-elsewhere")],
    )
    pre = _build(corpus, tmp_path)
    _switch(corpus)
    post = _build(corpus, tmp_path)

    _same_queue_but_refs(pre, post)
    assert not (corpus.root / OR_DIR / "2026-08-01-or-elsewhere.jsonl").exists()


def test_a_scope_reingested_after_the_release_fails_closed(tmp_path, monkeypatch):
    """The checkout's lock pins other bytes than the release: nothing is written."""

    contents = _basic_contents()
    path = f"{OR_DIR}/2026-07-20-or-supplement.jsonl"
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(
        tmp_path / "axiom-corpus",
        contents,
        committed_overrides={path: _rows([_record("us-or/manual/odhs/open/page-9")])},
    )
    with pytest.raises(ValueError, match=f"digest differs from signed release: {path}"):
        _build(corpus, tmp_path)

    _switch(corpus)
    with pytest.raises(ValueError, match="cannot place 1 of 4 queue source") as caught:
        _build(corpus, tmp_path)
    assert f"{path}: not placed: the checkout's lock" in str(caught.value)
    assert "pins other bytes" in str(caught.value)
    assert not (corpus.root / path).exists()
    # The files the lock does pin to the release's bytes are in place and exact.
    for placed, data in _provisions_on_disk(corpus.root).items():
        assert data == corpus.signed[placed]
    assert _staging_leftovers(corpus.root) == []


def test_a_present_file_with_other_bytes_fails_as_a_tampered_tracked_file_does(
    tmp_path, monkeypatch
):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    _switch(corpus)
    path = f"{UT_DIR}/2026-07-13-recovery.jsonl"
    edited = bytearray(corpus.signed[path])
    edited[0:1] = b"["  # same size, other bytes: present by size
    (corpus.root / path).parent.mkdir(parents=True)
    (corpus.root / path).write_bytes(bytes(edited))

    with pytest.raises(ValueError, match=f"digest differs from signed release: {path}"):
        _build(corpus, tmp_path)
    assert (corpus.root / path).read_bytes() == bytes(edited)


def test_a_stray_file_beside_the_locked_ones_fails_as_it_does_before_the_switch(
    tmp_path, monkeypatch
):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    stray = corpus.root / OR_DIR / "2026-09-01-or-draft.jsonl"
    stray.write_bytes(_rows([_record("us-or/manual/odhs/open/page-99")]))
    message = "release manifest is missing SNAP manual scopes: us-or/manual/2026-09-01-or-draft"
    with pytest.raises(ValueError, match=message):
        _build(corpus, tmp_path)

    _switch(corpus)
    stray.parent.mkdir(parents=True)
    stray.write_bytes(_rows([_record("us-or/manual/odhs/open/page-99")]))
    with pytest.raises(ValueError, match=message):
        _build(corpus, tmp_path)


def test_shallow_lock_checkout_needs_history_or_another_source(tmp_path, monkeypatch):
    """`actions/checkout` without fetch-depth: 0 gives a depth-1 clone.

    After the switch, the release commit's tree holds no provisions and the
    lock entries' blobs are in older commits, so a shallow clone can supply
    nothing from git. A full clone reads the blobs; a shallow one needs the
    cache or R2 (here the cache, which takes the same verified path).
    """

    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    pre = _build(corpus, tmp_path)
    _switch(corpus)
    url = corpus.root.resolve().as_uri()

    shallow = tmp_path / "shallow"
    subprocess.run(
        [GIT, "clone", "--quiet", "--depth", "1", url, str(shallow)], check=True
    )
    assert _git(shallow, "rev-parse", "--is-shallow-repository").strip() == "true"
    with pytest.raises(ValueError, match="cannot place 4 of 4 queue source") as caught:
        _build(corpus, tmp_path, root=shallow)
    assert "git: absent; git-lock: absent" in str(caught.value)
    assert "R2 read credentials" in str(caught.value)
    assert _provisions_on_disk(shallow) == {}
    assert _git(shallow, "status", "--porcelain") == ""

    cache = tmp_path / "corpus-cache"
    for data in corpus.signed.values():
        digest = hashlib.sha256(data).hexdigest()
        target = cache / "objects/sha256" / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    monkeypatch.setenv(cm.CACHE_ENV, str(cache))
    from_cache = _outcome(lambda: _build(corpus, tmp_path, root=shallow))
    assert from_cache[0] == "ok", from_cache[1]
    assert "from cache 4" in from_cache[2]
    _same_queue_but_refs(pre, from_cache[1])
    monkeypatch.delenv(cm.CACHE_ENV)

    full = tmp_path / "full"
    subprocess.run([GIT, "clone", "--quiet", url, str(full)], check=True)
    from_history = _outcome(lambda: _build(corpus, tmp_path, root=full))
    assert from_history[0] == "ok", from_history[1]
    assert "from git-lock 4" in from_history[2]
    _same_queue_but_refs(pre, from_history[1])


def test_no_fetch_switch_writes_nothing_and_requires_the_sources(tmp_path, monkeypatch):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    pre = _build(corpus, tmp_path)
    _switch(corpus)
    monkeypatch.setenv(cm.NO_FETCH_ENV, "1")

    with pytest.raises(ValueError, match="AXIOM_CORPUS_NO_FETCH is set, and 4 of 4"):
        _build(corpus, tmp_path)
    assert not (corpus.root / "data").exists()

    # With the files already in place (as `corpus fetch` leaves them), it builds.
    _write_files(
        corpus.root,
        {p: d for p, d in corpus.signed.items() if p.endswith(".jsonl")},
    )
    _same_queue_but_refs(pre, _build(corpus, tmp_path))
    assert not (corpus.root / cm.FETCH_TEMP_DIR).exists()


def test_symlinked_provisions_directory_fails_closed(tmp_path, monkeypatch):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    _switch(corpus)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (corpus.root / "data/corpus/provisions").mkdir(parents=True)
    (corpus.root / "data/corpus/provisions/us-ut").symlink_to(elsewhere)

    with pytest.raises(ValueError, match="cannot place") as caught:
        _build(corpus, tmp_path)
    assert "symlink" in str(caught.value)
    assert list(elsewhere.rglob("*")) == []


def test_an_invalid_lock_in_a_manual_directory_fails_the_listing(tmp_path, monkeypatch):
    contents = _basic_contents()
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    _switch(corpus)
    lock = corpus.root / cm.LOCK_ROOT / "us-or/manual/2026-07-20-or-supplement.json"
    lock.write_bytes(lock.read_bytes().replace(b"  ]\n}\n", b"]}\n"))
    _git(corpus.root, "commit", "--quiet", "-am", "corrupt a lock")
    corpus.sign()

    with pytest.raises(ValueError, match="is not a valid lock"):
        _build(corpus, tmp_path)
    assert _provisions_on_disk(corpus.root) == {}


def test_cli_keeps_stdout_for_the_queue_and_fails_closed_with_a_message(
    tmp_path, monkeypatch
):
    """The workflows redirect stdout into the queue file; notes go to stderr."""

    contents = _basic_contents()
    counts = _expected_counts(contents)
    monkeypatch.setattr(signed_queue, "EXPECTED_COUNTS", counts)
    corpus = _pre_switch_corpus(tmp_path / "axiom-corpus", contents)
    expected = _build(corpus, tmp_path)
    _switch(corpus)
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            GIT,
            "clone",
            "--quiet",
            "--depth",
            "1",
            corpus.root.resolve().as_uri(),
            str(shallow),
        ],
        check=True,
    )
    # The script runs in a child process, so give it the fixture's counts.
    runner = tmp_path / "run_prepare_signed_queue.py"
    runner.write_text(
        "import sys\n"
        f"sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'src')!r}]\n"
        "import scripts.prepare_signed_queue as q\n"
        f"q.EXPECTED_COUNTS = {counts!r}\n"
        "q.main()\n"
    )

    def run(root: Path) -> subprocess.CompletedProcess[str]:
        arguments = _build_arguments(corpus, tmp_path, root=root)
        return subprocess.run(
            [
                sys.executable,
                str(runner),
                "build-snap",
                str(root),
                str(arguments["rulespec_root"]),
                "--corpus-ref",
                arguments["corpus_ref"],
                "--rulespec-ref",
                arguments["rulespec_ref"],
                "--rules-engine-ref",
                arguments["rules_engine_ref"],
                "--release-name",
                NAME,
                "--release-content-sha256",
                corpus.release_sha,
                "--release-object",
                str(corpus.release_object),
                "--release-public-key",
                str(arguments["release_public_key_path"]),
                "--state",
                "paused",
                "--pause-reason",
                "Awaiting protected activation.",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    failed = run(shallow)
    assert failed.returncode == 1
    assert failed.stdout == ""
    assert "cannot place 4 of 4 queue source artifact(s)" in failed.stderr

    placed = run(corpus.root)
    assert placed.returncode == 0, placed.stderr
    _same_queue_but_refs(expected, json.loads(placed.stdout))
    assert "prepare_signed_queue: placed 4 corpus release artifact(s)" in placed.stderr


# ----------------------------------------------------------- all states


def _all_state_contents() -> tuple[list[dict], dict[Scope, bytes], list[Scope]]:
    jurisdictions = sorted(set(ALL_STATE_EXPECTED_COUNTS) - {"total"})
    states, contents, release_scopes = [], {}, []
    for jurisdiction in jurisdictions:
        version = f"test-{jurisdiction}-snap"
        inventory_scope = {
            "jurisdiction": jurisdiction,
            "document_class": "regulation",
            "version": version,
        }
        if jurisdiction == "us-ma":
            inventory_scope["version"] = "2026-07-17-ma-dta-snap-regulations"
            version = "2026-07-24-ma-dta-regulations-snap-current-union"
        entry = {
            "jurisdiction": jurisdiction,
            "queue_status": "published_current",
            "target_scope": inventory_scope,
        }
        scope = (jurisdiction, "regulation", version)
        records = [
            {
                "citation_label": f"{jurisdiction} SNAP",
                "citation_path": f"{jurisdiction}/regulation/snap",
                "kind": "document",
                "ordinal": 1,
            }
        ]
        if jurisdiction == "us-tx":
            # A state with a supporting manual scope as well.
            support = (jurisdiction, "manual", "test-us-tx-handbook")
            entry["supporting_scope"] = dict(
                zip(("jurisdiction", "document_class", "version"), support)
            )
            contents[support] = _rows(
                [
                    {
                        "citation_label": "TX handbook",
                        "citation_path": "us-tx/manual/handbook",
                        "kind": "document",
                        "ordinal": 1,
                    }
                ]
            )
            release_scopes.append(support)
        states.append(entry)
        contents[scope] = _rows(records)
        release_scopes.append(scope)
    return states, contents, release_scopes


def test_all_state_lock_checkout_builds_the_queue_the_pre_switch_checkout_builds(
    tmp_path, monkeypatch, capsys
):
    states, contents, _ = _all_state_contents()
    inventory = yaml.safe_dump({"program": "SNAP", "states": states}).encode()
    jurisdictions = sorted(set(ALL_STATE_EXPECTED_COUNTS) - {"total"})
    counts = {"total": 52, **dict.fromkeys(jurisdictions, 1), "us-tx": 2}
    monkeypatch.setattr(signed_queue, "ALL_STATE_EXPECTED_COUNTS", counts)
    monkeypatch.setattr(
        signed_queue,
        "ALL_STATE_INVENTORY_SHA256",
        hashlib.sha256(inventory).hexdigest(),
    )
    corpus = _pre_switch_corpus(
        tmp_path / "axiom-corpus",
        contents,
        tracked={ALL_STATE_INVENTORY_PATH: inventory},
    )

    def build(root: Path | None = None) -> dict:
        arguments = _build_arguments(corpus, tmp_path, root=root)
        return build_all_state_snap_queue(
            arguments.pop("corpus_root"), arguments.pop("rulespec_root"), **arguments
        )

    pre = build()
    _switch(corpus)
    capsys.readouterr()
    post = build()

    _same_queue_but_refs(pre, post)
    assert len(post["items"]) == 52
    assert "placed 52 corpus release artifact(s)" in capsys.readouterr().err
    assert (
        corpus.root
        / "data/corpus/provisions/us-ma/regulation"
        / "2026-07-24-ma-dta-regulations-snap-current-union.jsonl"
    ).is_file()
    assert _git(corpus.root, "status", "--porcelain") == ""
    assert build() == post
    assert "placed" not in capsys.readouterr().err


def test_all_state_lock_checkout_fails_closed_without_a_source(tmp_path, monkeypatch):
    states, contents, _ = _all_state_contents()
    inventory = yaml.safe_dump({"program": "SNAP", "states": states}).encode()
    monkeypatch.setattr(
        signed_queue,
        "ALL_STATE_INVENTORY_SHA256",
        hashlib.sha256(inventory).hexdigest(),
    )
    corpus = _pre_switch_corpus(
        tmp_path / "axiom-corpus",
        contents,
        tracked={ALL_STATE_INVENTORY_PATH: inventory},
    )
    _switch(corpus)
    shallow = tmp_path / "shallow"
    subprocess.run(
        [
            GIT,
            "clone",
            "--quiet",
            "--depth",
            "1",
            corpus.root.resolve().as_uri(),
            str(shallow),
        ],
        check=True,
    )
    arguments = _build_arguments(corpus, tmp_path, root=shallow)

    with pytest.raises(
        ValueError, match="cannot place 52 of 52 queue source"
    ) as caught:
        build_all_state_snap_queue(
            arguments.pop("corpus_root"), arguments.pop("rulespec_root"), **arguments
        )
    assert "... and 42 more" in str(caught.value)
    assert _provisions_on_disk(shallow) == {}
    assert _staging_leftovers(shallow) == []


# ------------------------------------------------------------- properties

PERTURBATIONS = ("none", "extra", "omit", "stray", "tamper")


@st.composite
def _snap_corpora(draw):
    """Random Oregon and Utah manual corpora and one perturbation.

    Records spread over 1-3 scopes per state; a citation may repeat across
    scopes; non-SNAP rows are noise. The perturbation: none; the release also
    lists a scope the corpus lacks (``extra``); the release omits one corpus
    scope (``omit``); an untracked file sits in a manual directory
    (``stray``); or one committed file differs from the signed bytes
    (``tamper``).
    """

    or_versions = [
        f"2026-07-{day:02d}-or" for day in range(1, 1 + draw(st.integers(1, 3)))
    ]
    ut_versions = [
        f"2026-06-{day:02d}-ut" for day in range(1, 1 + draw(st.integers(1, 3)))
    ]
    or_scopes = {version: [] for version in or_versions}
    ut_scopes = {version: [] for version in ut_versions}
    pages = sorted(
        set(OR_PRIORITY_PAGES) | draw(st.sets(st.integers(1, 400), max_size=6))
    )
    for page in pages:
        homes = draw(st.lists(st.sampled_from(or_versions), min_size=1, max_size=2))
        for version in homes:
            or_scopes[version].append(_record(f"us-or/manual/odhs/open/page-{page}"))
    for page in draw(st.sets(st.integers(401, 450), max_size=3)):
        or_scopes[draw(st.sampled_from(or_versions))].append(
            _record(f"us-or/manual/odhs/open/page-{page}", snap=False)
        )
    blocks = [UT_BLOCK]
    for topic in range(draw(st.integers(0, 4))):
        for block in range(1, 1 + draw(st.integers(1, 2))):
            blocks.append(f"us-ut/manual/dws/eligibility/topic-{topic}/block-{block}")
    for citation in blocks:
        ut_scopes[draw(st.sampled_from(ut_versions))].append(_record(citation))
    for scopes, prefix in (
        (or_scopes, "us-or/manual/x"),
        (ut_scopes, "us-ut/manual/x"),
    ):
        for version, records in scopes.items():
            if not records:
                records.append(_record(f"{prefix}/{version}-noise", snap=False))
    contents = _snap_contents(or_scopes, ut_scopes)
    perturbation = draw(st.sampled_from(PERTURBATIONS))
    target = draw(st.sampled_from(sorted(contents)))
    return contents, perturbation, target


_GIT_PROPERTY_SETTINGS = settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)


@_GIT_PROPERTY_SETTINGS
@given(_snap_corpora())
def test_property_lock_checkout_agrees_with_the_pre_switch_checkout(scenario):
    """Invariant: for every corpus, the build after the switch equals the build
    before it, or both fail. A successful lock-file build leaves only the
    release's bytes in place, keeps the checkout clean, and is idempotent."""

    contents, perturbation, target = scenario
    with tempfile.TemporaryDirectory() as directory, pytest.MonkeyPatch.context() as mp:
        base = Path(directory)
        mp.setattr(signed_queue, "EXPECTED_COUNTS", _expected_counts(contents))
        release_scopes, unmaterialized, overrides = list(contents), [], {}
        if perturbation == "extra":
            unmaterialized = [("us-or", "manual", "2026-12-31-or-not-in-corpus")]
        elif perturbation == "omit":
            release_scopes.remove(target)
        elif perturbation == "tamper":
            path = f"data/corpus/provisions/{target[0]}/manual/{target[2]}.jsonl"
            overrides = {
                path: contents[target] + _rows([_record(f"{target[0]}/tampered")])
            }
        stray = base / "axiom-corpus" / OR_DIR / "2026-09-30-stray.jsonl"
        corpus = _pre_switch_corpus(
            base / "axiom-corpus",
            contents,
            release_scopes=release_scopes,
            unmaterialized=unmaterialized,
            committed_overrides=overrides,
        )
        if perturbation == "stray":
            stray.write_bytes(_rows([_record("us-or/manual/odhs/open/page-999")]))
        pre = _outcome(lambda: _build(corpus, base))

        _switch(corpus)
        if perturbation == "stray":
            stray.parent.mkdir(parents=True)
            stray.write_bytes(_rows([_record("us-or/manual/odhs/open/page-999")]))
        post = _outcome(lambda: _build(corpus, base))

        assert (pre[0] == "ok") == (perturbation in {"none", "extra"}), pre[1]
        assert pre[0] == post[0], (perturbation, pre[1], post[1])
        if pre[0] == "error":
            if perturbation in {"omit", "stray"}:
                assert pre[1] == post[1]
            return
        _same_queue_but_refs(pre[1], post[1])
        sources = {
            p: d
            for p, d in corpus.signed.items()
            if p.startswith(
                (
                    "data/corpus/provisions/us-or/manual/",
                    "data/corpus/provisions/us-ut/manual/",
                )
            )
        }
        assert _provisions_on_disk(corpus.root) == sources
        assert _staging_leftovers(corpus.root) == []
        assert _git(corpus.root, "status", "--porcelain") == ""
        again = _outcome(lambda: _build(corpus, base))
        assert again[:2] == post[:2]
        assert "placed" not in again[2]

"""Placing verified corpus release artifacts in a lock-file corpus checkout."""

from __future__ import annotations

import errno
import hashlib
import importlib.util
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.error
from contextlib import nullcontext
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path, PurePosixPath

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import corpus_materialize as cm
from axiom_encode.corpus_release import (
    CorpusReleaseObjectError,
    VerifiedReleaseArtifact,
    verify_pinned_release_object_content,
)
from axiom_encode.corpus_resolver import (
    CorpusLayoutError,
    CorpusResolutionError,
    LocalCorpusRelease,
    UnmaterializedCorpusReleaseError,
    resolve_local_corpus_source,
)
from tests.release_object_fixtures import (
    TEST_RELEASE_PUBLIC_KEY,
    write_test_release_object,
)

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="git is required")
RELEASE = "test-release"
VERSION = "2026-01-02-title-7"
CITATION = "us/statute/7/2014/e"
BODY = "The standard deduction is $198."
PROVISIONS = f"data/corpus/provisions/us/statute/{VERSION}.jsonl"
INVENTORY = f"data/corpus/inventory/us/statute/{VERSION}.json"
STATUTE = ("us", "statute", VERSION)
# A second scope in another jurisdiction x document class: the resolver reads
# every release scope of the bucket it is asked about, and only those.
REG_VERSION = "2026-01-02-7-cfr-273"
REG_CITATION = "us/regulation/7/273/9"
REG_BODY = "Income eligibility standards apply to every household."
REG_PROVISIONS = f"data/corpus/provisions/us/regulation/{REG_VERSION}.jsonl"
REGULATION = ("us", "regulation", REG_VERSION)
SCOPE_TEXT = {STATUTE: (CITATION, BODY), REGULATION: (REG_CITATION, REG_BODY)}


@pytest.fixture(autouse=True)
def _isolated_environment(tmp_path_factory, monkeypatch):
    """Keep the real cache, R2 credentials and fetch switches out of every test."""

    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
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


# ---------------------------------------------------------------- fakes


def _artifact(
    path: str, data: bytes, cls: str = "provisions"
) -> VerifiedReleaseArtifact:
    return VerifiedReleaseArtifact(
        cls,
        path,
        hashlib.sha256(data).hexdigest(),
        len(data),
        1 if cls == "provisions" else None,
    )


class FakeSource:
    """Serves bytes by sha256 with a scripted behavior per artifact."""

    def __init__(self, label: str, behaviors: dict[str, tuple[str, bytes]]):
        self.label = label
        self.behaviors = behaviors
        self.opened: list[str] = []

    def open(self, artifact):
        self.opened.append(artifact.path)
        kind, data = self.behaviors.get(artifact.sha256, ("absent", b""))
        if kind == "absent":
            return None
        if kind == "raises":
            raise cm.SourceError("scripted failure")
        if kind == "correct":
            payload = data
        elif kind == "wrong_bytes":
            payload = bytes(b ^ 0xFF for b in data) if data else b"x"
        elif kind == "truncated":
            payload = data[:-1]
        elif kind == "oversized":
            payload = data + b"!"
        else:  # pragma: no cover - test bug
            raise AssertionError(kind)
        return iter([payload[i : i + 7] for i in range(0, len(payload), 7)] or [b""])


def _factory(*sources):
    return lambda: nullcontext(list(sources))


def _no_sources():
    raise AssertionError("sources must not be opened when nothing is to be placed")


def _leftovers(root: Path) -> list[Path]:
    """Files in the staging directory, and hidden or temporary files in a scope.

    A completed pass leaves none; axiom-corpus's `corpus lock` refuses a
    scope holding a hidden or `.corpus-fetch-` file.
    """

    corpus = root / "data" / "corpus"
    if not corpus.is_dir():
        return []
    return sorted(
        path
        for path in corpus.rglob("*")
        if not path.is_dir()
        and (
            path.relative_to(corpus).parts[0] == ".corpus-fetch-tmp"
            or path.name.startswith(".")
            or path.name.endswith(".part")
            or cm.FETCH_TEMP_MARKER in path.name
        )
    )


def _files(root: Path) -> list[Path]:
    """Every file under ``root/data``: none after a pass that placed nothing."""

    data = root / "data"
    return sorted(p for p in data.rglob("*") if not p.is_dir()) if data.is_dir() else []


# ------------------------------------------------------------- lock files


def _lock_bytes(lock_path: PurePosixPath, entries: list[dict[str, object]]) -> bytes:
    """Canonical lock bytes, laid out as axiom-corpus's serialize_lock writes them."""

    header = [
        f'  "schema_version": {json.dumps(cm.LOCK_SCHEMA_VERSION)},',
        f'  "jurisdiction": {json.dumps(lock_path.parts[-3])},',
        f'  "document_class": {json.dumps(lock_path.parts[-2])},',
        f'  "version": {json.dumps(lock_path.stem)},',
        '  "files": [',
    ]
    body = ",\n".join(
        "    " + json.dumps(entry, ensure_ascii=True, separators=(", ", ": "))
        for entry in sorted(entries, key=lambda entry: str(entry["path"]))
    )
    return ("{\n" + "\n".join(header) + "\n" + body + "\n  ]\n}\n").encode("ascii")


def _write_locks(
    root: Path,
    pins: dict[str, bytes | tuple[str, int]],
    blobs: dict[str, str] | None = None,
) -> None:
    """Write (replace) the scope locks of ``pins``: path -> bytes or (sha256, size)."""

    by_lock: dict[PurePosixPath, list[dict[str, object]]] = {}
    for path, pinned in pins.items():
        if isinstance(pinned, bytes):
            sha256, size = hashlib.sha256(pinned).hexdigest(), len(pinned)
        else:
            sha256, size = pinned
        entry: dict[str, object] = {"path": path, "sha256": sha256, "size": size}
        if blobs and path in blobs:
            entry["git_blob"] = blobs[path]
        lock_path = cm.lock_path_for(path)
        assert lock_path is not None, path
        by_lock.setdefault(lock_path, []).append(entry)
    for lock_path, entries in by_lock.items():
        target = root / lock_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(_lock_bytes(lock_path, entries))


def _repin(root: Path, path: str, data: bytes) -> None:
    """Point one existing lock entry at other bytes, as a re-ingest does."""

    lock_path = cm.lock_path_for(path)
    payload = json.loads((root / lock_path).read_bytes())
    for entry in payload["files"]:
        if entry["path"] == path:
            entry["sha256"] = hashlib.sha256(data).hexdigest()
            entry["size"] = len(data)
            entry.pop("git_blob", None)
    (root / lock_path).write_bytes(_lock_bytes(lock_path, payload["files"]))


# Bytes axiom-corpus's own `serialize_lock` wrote (corpus/out-of-git at
# ebec4224, src/axiom_corpus/corpus/corpus_locks.py) for these entries.
AXIOM_CORPUS_LOCK_PATH = PurePosixPath(f".axiom/corpus-locks/us/statute/{VERSION}.json")
AXIOM_CORPUS_LOCK_ENTRIES = [
    (f"data/corpus/coverage/us/statute/{VERSION}.json", b"{}\n", None),
    (f"data/corpus/inventory/us/statute/{VERSION}.json", b"{}\n", "a" * 40),
    (PROVISIONS, b'{"row": 1}\n', "b" * 40),
    (
        f"data/corpus/sources/us/statute/{VERSION}/a/source.html",
        b"<html></html>\n",
        None,
    ),
]
AXIOM_CORPUS_LOCK = b"""{
  "schema_version": "axiom-corpus/corpus-lock/v1",
  "jurisdiction": "us",
  "document_class": "statute",
  "version": "2026-01-02-title-7",
  "files": [
    {"path": "data/corpus/coverage/us/statute/2026-01-02-title-7.json", "sha256": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356", "size": 3},
    {"path": "data/corpus/inventory/us/statute/2026-01-02-title-7.json", "sha256": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356", "size": 3, "git_blob": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
    {"path": "data/corpus/provisions/us/statute/2026-01-02-title-7.jsonl", "sha256": "e321f5d57c4844eb66b8b6b82c2049cbefcc7f6d9775eb37d6e42117b6c08173", "size": 11, "git_blob": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
    {"path": "data/corpus/sources/us/statute/2026-01-02-title-7/a/source.html", "sha256": "b0693dc92f76e08bf1485b3dd9b514a2e31dfd6f39422a6b60edb722671dc98f", "size": 14}
  ]
}
"""


def test_lock_reader_agrees_with_axiom_corpus_serialize_lock():
    expected = {
        path: cm.LockEntry(hashlib.sha256(data).hexdigest(), len(data), blob)
        for path, data, blob in AXIOM_CORPUS_LOCK_ENTRIES
    }
    assert cm._parse_scope_lock(AXIOM_CORPUS_LOCK, AXIOM_CORPUS_LOCK_PATH) == expected
    entries = []
    for path, data, blob in AXIOM_CORPUS_LOCK_ENTRIES:
        entry = {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        if blob:
            entry["git_blob"] = blob
        entries.append(entry)
    # The test helper writes byte-for-byte what axiom-corpus writes.
    assert _lock_bytes(AXIOM_CORPUS_LOCK_PATH, entries) == AXIOM_CORPUS_LOCK


# ------------------------------------------------------------ unit tests


def test_first_verifying_source_wins_and_bad_sources_are_skipped(tmp_path):
    data = b'{"citation_path": "x"}\n'
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    bad = FakeSource("bad", {art.sha256: ("wrong_bytes", data)})
    good = FakeSource("good", {art.sha256: ("correct", data)})
    later = FakeSource("later", {art.sha256: ("correct", data)})

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(bad, good, later)
    )

    assert report.ok and report.materialized == {PROVISIONS: "good"}
    placed = tmp_path / PROVISIONS
    assert placed.read_bytes() == data
    assert placed.is_file() and not placed.is_symlink()
    assert placed.stat().st_nlink == 1
    assert later.opened == []
    assert _leftovers(tmp_path) == []


def test_unverified_bytes_are_never_placed(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    sources = [
        FakeSource("a", {art.sha256: ("wrong_bytes", data)}),
        FakeSource("b", {art.sha256: ("truncated", data)}),
        FakeSource("c", {art.sha256: ("oversized", data)}),
        FakeSource("d", {art.sha256: ("raises", data)}),
        FakeSource("e", {}),
    ]

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(*sources)
    )

    assert not report.ok
    reason = report.failed[PROVISIONS]
    for label in "abcde":
        assert f"{label}:" in reason
    assert "sha256 does not match" in reason
    assert "more than" in reason
    # No file, and no scope directory the attempt made; staging stays, empty.
    assert _files(tmp_path) == []
    assert not (tmp_path / "data" / "corpus" / "provisions").exists()
    assert (tmp_path / cm.FETCH_TEMP_DIR).is_dir()


def test_a_modified_file_the_lock_pins_is_left_untouched(tmp_path):
    data = b"release bytes\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local edit\n")

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    # Not a failure: the lock pins the release's bytes, the file is an edit.
    assert report.ok and "left untouched" in report.modified[PROVISIONS]
    assert "not yet locked" in report.modified[PROVISIONS]
    assert target.read_bytes() == b"local edit\n"


def test_present_files_open_no_source(tmp_path):
    """A present file is checked by size; no lock is needed to leave it alone."""

    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(data)

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    assert report.ok and report.present == [PROVISIONS] and not report.materialized


def test_verify_hashes_present_files(tmp_path):
    data = b"row one\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(b"row two\n")  # same size, other bytes

    assert cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources).ok
    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_no_sources, verify=True
    )
    assert "sha256 differs" in report.modified[PROVISIONS]
    assert target.read_bytes() == b"row two\n"


def test_symlinked_directory_on_the_path_is_refused(tmp_path):
    root = tmp_path / "corpus"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "data" / "corpus").mkdir(parents=True)
    (root / "data" / "corpus" / "provisions").symlink_to(elsewhere)
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(root, {PROVISIONS: data})

    report = cm.materialize_release_artifacts(
        root, [art], sources=_factory(FakeSource("s", {art.sha256: ("correct", data)}))
    )

    assert "symlink" in report.failed[PROVISIONS]
    assert list(elsewhere.rglob("*")) == []


def test_symlinked_destination_is_refused(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    outside = tmp_path / "outside.jsonl"
    outside.write_bytes(data)
    target.symlink_to(outside)

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    assert "symlink" in report.failed[PROVISIONS]


@pytest.mark.parametrize(
    "path",
    [
        "data/corpus/provisions/../x/y.jsonl",
        "data/corpus/other/us/statute/v.jsonl",
        "corpus/provisions/us/statute/v.jsonl",
        "data/corpus/provisions/us/statute/v\n.jsonl",
        "data/corpus/provisions//statute/v.jsonl",
    ],
)
def test_noncanonical_artifact_paths_are_refused(tmp_path, path):
    art = _artifact(path, b"x")
    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)
    assert "non-canonical" in report.failed[path]


def test_sources_artifacts_are_never_placed(tmp_path):
    """axiom-corpus fetches a scope's sources directory all or nothing."""

    path = f"data/corpus/sources/us/statute/{VERSION}/a/source.html"
    data = b"<html></html>\n"
    art = _artifact(path, data, "sources")
    _write_locks(tmp_path, {path: data})

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    assert "does not place sources" in report.failed[path]
    assert not (tmp_path / "data").exists()


def test_root_must_be_an_absolute_real_directory(tmp_path):
    with pytest.raises(cm.CorpusMaterializationError, match="absolute"):
        cm.materialize_release_artifacts(Path("relative"), [], sources=_no_sources)
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(cm.CorpusMaterializationError, match="real directory"):
        cm.materialize_release_artifacts(link, [], sources=_no_sources)


def _racing(target: Path, written: bytes, data: bytes):
    class Racing:
        label = "racing"

        def open(self, artifact):
            def chunks():
                target.write_bytes(written)  # another process wins the race
                yield data

            return chunks()

    return Racing()


def test_concurrent_writer_with_the_same_file_is_accepted(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    target = tmp_path / PROVISIONS

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(_racing(target, data, data))
    )

    assert report.ok and target.read_bytes() == data
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize("written", [b"different\n", b"ROW\n"])
def test_concurrent_writer_with_other_bytes_fails_closed(tmp_path, written):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    target = tmp_path / PROVISIONS

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(_racing(target, written, data))
    )

    # Another size, or the same size with other bytes: never accepted as placed.
    assert "left untouched" in report.modified[PROVISIONS]
    assert not report.materialized
    assert target.read_bytes() == written
    assert _leftovers(tmp_path) == []


def test_unwritable_destination_is_reported_not_raised(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    parent = (tmp_path / PROVISIONS).parent
    parent.mkdir(parents=True)
    parent.chmod(0o555)
    try:
        report = cm.materialize_release_artifacts(
            tmp_path,
            [art],
            sources=_factory(FakeSource("s", {art.sha256: ("correct", data)})),
        )
    finally:
        parent.chmod(0o755)
    assert "cannot link verified bytes into place" in report.failed[PROVISIONS]
    assert not (tmp_path / PROVISIONS).exists()
    assert _leftovers(tmp_path) == []


def test_source_failing_mid_stream_falls_through_to_the_next(tmp_path):
    data = b"0123456789" * 3
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})

    class Broken(io.BytesIO):
        def read(self, size=-1):
            if self.tell() >= 10:
                raise ConnectionResetError("peer reset")
            return super().read(min(size, 10))

    class Flaky:
        label = "flaky"

        def open(self, artifact):
            return cm._iter_handle(Broken(data))

    good = FakeSource("good", {art.sha256: ("correct", data)})
    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Flaky(), good)
    )

    assert report.materialized == {PROVISIONS: "good"}
    assert (tmp_path / PROVISIONS).read_bytes() == data
    assert _leftovers(tmp_path) == []


# ------------------------------------------------------------ lock gate


def test_placement_needs_the_checkout_lock_to_pin_the_release_bytes(tmp_path):
    data = b"release row\n"
    art = _artifact(PROVISIONS, data)
    source = FakeSource("s", {art.sha256: ("correct", data)})

    def run():
        return cm.materialize_release_artifacts(
            tmp_path, [art], sources=_factory(source), release_commit="c" * 40
        )

    report = run()
    assert (
        "has no lock for scope us/statute/2026-01-02-title-7"
        in (report.skipped[PROVISIONS])
    )
    _write_locks(tmp_path, {INVENTORY: b"{}\n"})
    assert "does not list this path" in run().skipped[PROVISIONS]
    _write_locks(tmp_path, {PROVISIONS: b"re-ingested row\n", INVENTORY: b"{}\n"})
    report = run()
    reason = report.skipped[PROVISIONS]
    assert "pins other bytes" in reason and "c" * 40 in reason
    # A skip is not a failure, touches nothing and opens no source.
    assert report.ok and not report.materialized and not report.failed
    assert source.opened == []
    assert not (tmp_path / "data").exists()

    _write_locks(tmp_path, {PROVISIONS: data, INVENTORY: b"{}\n"})
    assert run().materialized == {PROVISIONS: "s"}
    assert (tmp_path / PROVISIONS).read_bytes() == data


def test_a_file_the_lock_pins_to_other_bytes_is_skipped_and_kept(tmp_path):
    """A checkout fetched at a later corpus commit: its lock's bytes stay put."""

    release = b"release row\n"
    fetched = b"re-ingested row, longer\n"
    art = _artifact(PROVISIONS, release)
    _write_locks(tmp_path, {PROVISIONS: fetched})
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(fetched)

    for verify in (False, True):
        report = cm.materialize_release_artifacts(
            tmp_path, [art], sources=_no_sources, verify=verify
        )
        assert report.ok and "pins other bytes" in report.skipped[PROVISIONS]
        assert target.read_bytes() == fetched


def _mutated_lock(mutate) -> bytes:
    data = b"row\n"
    lock_path = cm.lock_path_for(PROVISIONS)
    payload = json.loads(
        _lock_bytes(
            lock_path,
            [
                {
                    "path": PROVISIONS,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            ],
        )
    )
    return mutate(payload)


def _dump(payload) -> bytes:
    return json.dumps(payload).encode()


def _with(**changes):
    return lambda payload: _dump({**payload, **changes})


def _with_entry(**changes):
    return lambda payload: _dump(
        {**payload, "files": [{**payload["files"][0], **changes}]}
    )


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (
            _with(schema_version="axiom-corpus/corpus-lock/v2"),
            "unsupported lock schema",
        ),
        (_with(jurisdiction="uk"), "scope does not match"),
        (_with(version="2026-01-03-title-7"), "scope does not match"),
        (_with(files=[]), "non-empty list"),
        (_with(files={}), "non-empty list"),
        (_with(extra=1), "lock keys"),
        (lambda p: _dump({k: v for k, v in p.items() if k != "version"}), "lock keys"),
        (lambda p: _dump({**p, "files": p["files"] * 2}), "lists a path twice"),
        (
            _with_entry(path=f"data/corpus/provisions/us/statute/{VERSION}x.jsonl"),
            "outside the scope",
        ),
        (_with_entry(size=True), "not a byte count"),
        (_with_entry(size=-1), "not a byte count"),
        (_with_entry(size="4"), "not a byte count"),
        (_with_entry(sha256="A" * 64), "not a sha256"),
        (_with_entry(git_blob="xyz"), "not an object id"),
        (_with_entry(git_blob="a" * 64), "not an object id"),
        (_with_entry(git_blob=None), "not an object id"),
        # Valid content in another layout, as axiom-corpus's parse_lock rejects.
        (_dump, "not in canonical form"),
        (
            lambda p: _lock_bytes(cm.lock_path_for(PROVISIONS), p["files"]).replace(
                b'  "version"', b'  "version": "other",\n  "version"', 1
            ),
            "not in canonical form",
        ),
        (_with_entry(mode="0644"), "each lock entry needs"),
        (lambda p: _dump({**p, "files": ["x"]}), "each lock entry needs"),
        (lambda p: _dump([p]), "lock keys"),
        (lambda p: b"{", "not a valid lock"),
        (lambda p: _dump(p).replace(b"us", "üs".encode()), "not a valid lock"),
        (lambda p: b"[" * 200_000, "not a valid lock"),
    ],
)
def test_an_invalid_lock_pins_nothing(tmp_path, mutate, reason):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    lock = tmp_path / cm.lock_path_for(PROVISIONS)
    lock.parent.mkdir(parents=True)
    lock.write_bytes(_mutated_lock(mutate))

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    assert reason in report.skipped[PROVISIONS]
    assert report.ok and not (tmp_path / "data").exists()


def test_an_unsorted_lock_or_an_invalid_scope_component_pins_nothing(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data, INVENTORY: b"{}\n"})
    lock = tmp_path / cm.lock_path_for(PROVISIONS)
    lines = lock.read_bytes().decode().split("\n")
    entry_lines = [i for i, line in enumerate(lines) if line.startswith("    {")]
    a, b = entry_lines
    lines[a], lines[b] = lines[b].rstrip(",") + ",", lines[a].rstrip(",")
    lock.write_text("\n".join(lines))
    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)
    assert "not in canonical form" in report.skipped[PROVISIONS]

    upper = "data/corpus/provisions/XX/statute/v1.jsonl"
    upper_lock = tmp_path / ".axiom/corpus-locks/XX/statute/v1.json"
    upper_lock.parent.mkdir(parents=True)
    upper_lock.write_bytes(
        _lock_bytes(
            PurePosixPath(".axiom/corpus-locks/XX/statute/v1.json"),
            [
                {
                    "path": upper,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            ],
        )
    )
    report = cm.materialize_release_artifacts(
        tmp_path, [_artifact(upper, data)], sources=_no_sources
    )
    assert "invalid scope component" in report.skipped[upper]
    assert not (tmp_path / "data").exists()


def test_an_indirect_or_oversized_lock_pins_nothing(tmp_path, monkeypatch):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)

    def skipped_because():
        report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)
        assert not (tmp_path / "data").exists()
        return report.skipped[PROVISIONS]

    real = tmp_path / "real-lock.json"
    _write_locks(tmp_path, {PROVISIONS: data})
    lock = tmp_path / cm.lock_path_for(PROVISIONS)
    real.write_bytes(lock.read_bytes())
    lock.unlink()
    lock.symlink_to(real)
    assert "is a symlink" in skipped_because()

    lock.unlink()
    lock.mkdir()
    assert "not a regular file" in skipped_because()

    lock.rmdir()
    _write_locks(tmp_path, {PROVISIONS: data})
    limit = cm.MAX_LOCK_FILE_BYTES
    monkeypatch.setattr(cm, "MAX_LOCK_FILE_BYTES", 16)
    assert "exceeds 16 bytes" in skipped_because()
    monkeypatch.setattr(cm, "MAX_LOCK_FILE_BYTES", limit)

    locks = tmp_path / ".axiom" / "corpus-locks"
    moved = tmp_path / "moved-locks"
    locks.rename(moved)
    locks.symlink_to(moved)
    assert "is a symlink" in skipped_because()


def test_git_lock_source_reads_only_a_blob_pinned_to_the_release_bytes(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)

    class Store:
        def __init__(self):
            self.opened: list[str] = []

        def open(self, name, size):
            self.opened.append(name)
            return iter([data])

    store = Store()
    assert cm.GitLockBlobSource(store, tmp_path).open(art) is None
    _write_locks(tmp_path, {PROVISIONS: b"other\n"}, {PROVISIONS: "b" * 40})
    assert cm.GitLockBlobSource(store, tmp_path).open(art) is None
    _write_locks(tmp_path, {PROVISIONS: data})
    assert cm.GitLockBlobSource(store, tmp_path).open(art) is None
    _write_locks(tmp_path, {PROVISIONS: data}, {PROVISIONS: "c" * 40})
    assert b"".join(cm.GitLockBlobSource(store, tmp_path).open(art)) == data
    assert store.opened == ["c" * 40]


def test_a_present_file_the_lock_pins_to_other_bytes_is_noted(tmp_path):
    """Same size, other bytes: nothing written, and the reason is kept for reads."""

    release = b"release row\n"
    reingested = b"re-ingest! \n"
    assert len(release) == len(reingested)
    art = _artifact(PROVISIONS, release)
    _write_locks(tmp_path, {PROVISIONS: reingested})
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(reingested)

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_no_sources, release_commit="c" * 40
    )

    assert report.present == [PROVISIONS] and report.ok
    assert "present by size, but" in report.notes[PROVISIONS]
    assert "pins other bytes" in report.unplaced[PROVISIONS]
    assert target.read_bytes() == reingested


def test_a_lock_that_changes_mid_placement_stops_it(tmp_path):
    data = b"release row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})

    class Repinning:
        label = "repinning"

        def open(self, artifact):
            def chunks():
                yield data
                # A `git pull` lands a re-ingest while the bytes are on the way.
                _repin(tmp_path, PROVISIONS, b"re-ingested row\n")

            return chunks()

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Repinning())
    )

    assert "during placement" in report.skipped[PROVISIONS]
    assert "pins other bytes" in report.skipped[PROVISIONS]
    assert not (tmp_path / PROVISIONS).exists()
    assert _files(tmp_path) == []


def test_a_scope_directory_another_process_removed_is_made_again(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    parent = (tmp_path / PROVISIONS).parent

    class Removing:
        label = "removing"

        def open(self, artifact):
            def chunks():
                yield data
                parent.rmdir()  # another process's failed attempt rolls back

            return chunks()

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Removing())
    )

    assert report.materialized == {PROVISIONS: "removing"}
    assert (tmp_path / PROVISIONS).read_bytes() == data
    assert _leftovers(tmp_path) == []


def test_a_failed_attempt_never_pulls_directories_from_a_concurrent_one(tmp_path):
    """B fails while A streams: A still places (staging stays; scope dirs return)."""

    import threading

    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    b_ensured, a_ensured, b_done = (threading.Event() for _ in range(3))

    class Absent:  # process B: no source holds the bytes
        label = "b"

        def open(self, artifact):
            b_ensured.set()
            assert a_ensured.wait(30)
            return None

    class Slow:  # process A: its source has them, slowly
        label = "a"

        def open(self, artifact):
            a_ensured.set()
            assert b_done.wait(30)
            return iter([data])

    results = {}

    def run(name, source, done=None):
        results[name] = cm.materialize_release_artifacts(
            tmp_path, [art], sources=_factory(source)
        )
        if done is not None:
            done.set()

    first = threading.Thread(target=run, args=("B", Absent(), b_done))
    first.start()
    assert b_ensured.wait(30)  # B has made the directories
    second = threading.Thread(target=run, args=("A", Slow()))
    second.start()
    first.join(60)
    second.join(60)

    assert "b: absent" in results["B"].failed[PROVISIONS]
    assert results["A"].materialized == {PROVISIONS: "a"}
    assert (tmp_path / PROVISIONS).read_bytes() == data
    assert _leftovers(tmp_path) == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_an_uninspectable_path_is_reported_not_raised(tmp_path, capsys):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    sealed = tmp_path / "data" / "corpus" / "provisions" / "us"
    sealed.mkdir(parents=True)
    sealed.chmod(0)
    try:
        report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)
    finally:
        sealed.chmod(0o755)
    assert "cannot inspect" in report.failed[PROVISIONS]


# ------------------------------------------------------------ staging


def test_temporary_bytes_stay_in_the_staging_directory(tmp_path):
    """Mid-stream, the only new file is a marked temporary outside every scope."""

    data = bytes(range(256)) * 4
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    seen: list[list[str]] = []

    class Probe:
        label = "probe"

        def open(self, artifact):
            def chunks():
                for start in range(0, len(data), 100):
                    yield data[start : start + 100]
                    seen.append(
                        sorted(
                            path.relative_to(tmp_path).as_posix()
                            for path in (tmp_path / "data").rglob("*")
                            if not path.is_dir()
                        )
                    )

            return chunks()

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Probe())
    )

    assert report.materialized == {PROVISIONS: "probe"}
    assert seen and all(len(files) == 1 for files in seen)
    (temporary,) = {PurePosixPath(files[0]) for files in seen}
    assert temporary.parent == cm.FETCH_TEMP_DIR
    assert temporary.name.startswith(cm.FETCH_TEMP_MARKER)
    assert _leftovers(tmp_path) == []
    assert (tmp_path / PROVISIONS).stat().st_nlink == 1


_KILLED_CHILD = r"""
import sys
from contextlib import nullcontext
from pathlib import Path

from axiom_encode import corpus_materialize as cm
from axiom_encode.corpus_release import VerifiedReleaseArtifact

root, path, sha256, size = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])


class Stall:
    label = "stall"

    def open(self, artifact):
        def chunks():
            yield b"partial"
            print("streaming", flush=True)
            sys.stdin.readline()  # SIGKILLed while waiting here
            yield b"never"

        return chunks()


cm.materialize_release_artifacts(
    root,
    [VerifiedReleaseArtifact("provisions", path, sha256, size, 1)],
    sources=lambda: nullcontext([Stall()]),
)
"""


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="needs SIGKILL")
def test_a_killed_placement_leaves_its_temporary_only_in_staging(tmp_path, monkeypatch):
    data = _rows_bytes()
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data, INVENTORY: b"{}\n"})
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _KILLED_CHILD,
            str(tmp_path),
            PROVISIONS,
            art.sha256,
            str(art.byte_count),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline() == b"streaming\n"
    finally:
        child.kill()
        child.wait(timeout=60)
        child.stdin.close()
        child.stdout.close()
    assert child.returncode == -signal.SIGKILL

    # The scope holds nothing `corpus lock` or signing would pick up ...
    assert list((tmp_path / PROVISIONS).parent.iterdir()) == []
    # ... and the leftover sits where axiom-corpus prunes, with its marker.
    (leftover,) = (tmp_path / cm.FETCH_TEMP_DIR).iterdir()
    assert leftover.name.startswith(cm.FETCH_TEMP_MARKER)
    # (Empty or partial: the kill lands before or after a buffered write.)
    assert b"partial".startswith(leftover.read_bytes())

    # The next pass places the file and keeps a fresh leftover ...
    good = FakeSource("good", {art.sha256: ("correct", data)})
    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_factory(good))
    assert report.materialized == {PROVISIONS: "good"}
    assert (tmp_path / PROVISIONS).read_bytes() == data
    assert leftover.exists()
    # ... and a later process deletes it once it is a day old.
    monkeypatch.setattr(cm, "STALE_FETCH_TEMP_SECONDS", -60)
    cm._PRUNED_STAGING_DIRECTORIES.clear()
    inventory = _artifact(INVENTORY, b"{}\n", "inventory")
    report = cm.materialize_release_artifacts(
        tmp_path,
        [inventory],
        sources=_factory(FakeSource("s", {inventory.sha256: ("correct", b"{}\n")})),
    )
    assert report.materialized == {INVENTORY: "s"}
    assert not leftover.exists()


def test_prune_deletes_only_stale_marker_files_once_per_process(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    stale = staging / f"{cm.FETCH_TEMP_MARKER}x.jsonl.1.abc"
    stale.write_bytes(b"x")
    other = staging / "keep.txt"
    other.write_bytes(b"k")
    target = tmp_path / "target"
    target.write_bytes(b"t")
    link = staging / f"{cm.FETCH_TEMP_MARKER}link"
    link.symlink_to(target)
    directory = staging / f"{cm.FETCH_TEMP_MARKER}dir"
    directory.mkdir()

    cm._prune_stale_fetch_temporaries(staging)
    assert stale.exists()  # a fresh file is some live placement's
    monkeypatch.setattr(cm, "STALE_FETCH_TEMP_SECONDS", -60)
    cm._prune_stale_fetch_temporaries(staging)
    assert stale.exists()  # once per process and directory
    cm._PRUNED_STAGING_DIRECTORIES.clear()
    cm._prune_stale_fetch_temporaries(staging)
    assert not stale.exists()
    assert other.exists() and link.is_symlink() and directory.is_dir()
    assert target.read_bytes() == b"t"


def test_a_symlinked_staging_directory_is_refused(tmp_path):
    root = tmp_path / "corpus"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "data" / "corpus").mkdir(parents=True)
    (root / cm.FETCH_TEMP_DIR).symlink_to(elsewhere)
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(root, {PROVISIONS: data})

    report = cm.materialize_release_artifacts(
        root, [art], sources=_factory(FakeSource("s", {art.sha256: ("correct", data)}))
    )

    assert "not a real directory" in report.failed[PROVISIONS]
    assert list(elsewhere.iterdir()) == []
    assert not (root / PROVISIONS).exists()


def _refuse_links(monkeypatch, code: int) -> None:
    def link(*_args, **_kwargs):
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(cm.os, "link", link)


@pytest.mark.skipif(
    cm._rename_noreplace() is None, reason="no no-replace rename on this platform"
)
def test_without_hard_links_a_no_replace_rename_places_the_file(tmp_path, monkeypatch):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    inventory = _artifact(INVENTORY, b"{}\n", "inventory")
    _write_locks(tmp_path, {PROVISIONS: data, INVENTORY: b"{}\n"})
    _refuse_links(monkeypatch, errno.EPERM)

    report = cm.materialize_release_artifacts(
        tmp_path,
        [art],
        sources=_factory(FakeSource("s", {art.sha256: ("correct", data)})),
    )

    assert report.materialized == {PROVISIONS: "s"}
    assert (tmp_path / PROVISIONS).read_bytes() == data
    assert (tmp_path / PROVISIONS).stat().st_nlink == 1
    assert _leftovers(tmp_path) == []

    # The rename never replaces a file that appears meanwhile.
    target = tmp_path / INVENTORY
    report = cm.materialize_release_artifacts(
        tmp_path,
        [inventory],
        sources=_factory(_racing(target, b"[1]\n", b"{}\n")),
    )
    assert "left untouched" in report.modified[INVENTORY]
    assert target.read_bytes() == b"[1]\n"
    assert _leftovers(tmp_path) == []


@pytest.mark.parametrize(
    ("code", "rename", "reason"),
    [
        (errno.ENOTSUP, None, "no no-replace rename"),
        (errno.EXDEV, "unused", "different filesystems"),
        (errno.EIO, "unused", "cannot link verified bytes into place"),
    ],
)
def test_placement_that_could_replace_a_file_fails_closed(
    tmp_path, monkeypatch, code, rename, reason
):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    _refuse_links(monkeypatch, code)
    if rename is None:
        monkeypatch.setattr(cm, "_rename_noreplace", lambda: None)

    report = cm.materialize_release_artifacts(
        tmp_path,
        [art],
        sources=_factory(FakeSource("s", {art.sha256: ("correct", data)})),
    )

    assert reason in report.failed[PROVISIONS]
    assert _files(tmp_path) == []
    assert not (tmp_path / PROVISIONS).parent.exists()


def test_a_failed_no_replace_rename_is_reported(tmp_path, monkeypatch):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    _write_locks(tmp_path, {PROVISIONS: data})
    _refuse_links(monkeypatch, errno.EMLINK)
    monkeypatch.setattr(cm, "_rename_noreplace", lambda: lambda _s, _t: errno.EACCES)

    report = cm.materialize_release_artifacts(
        tmp_path,
        [art],
        sources=_factory(FakeSource("s", {art.sha256: ("correct", data)})),
    )

    assert "no-replace rename failed" in report.failed[PROVISIONS]
    assert _files(tmp_path) == []
    assert not (tmp_path / PROVISIONS).parent.exists()


def test_fetch_switch_and_lock_detection(tmp_path):
    assert not cm.fetch_disabled({})
    assert not cm.fetch_disabled({cm.NO_FETCH_ENV: "0"})
    assert cm.fetch_disabled({cm.NO_FETCH_ENV: "1"})
    assert not cm.corpus_uses_lock_files(tmp_path)
    (tmp_path / ".axiom").mkdir()
    (tmp_path / ".axiom" / "corpus-locks").symlink_to(tmp_path)
    assert not cm.corpus_uses_lock_files(tmp_path)
    (tmp_path / ".axiom" / "corpus-locks").unlink()
    (tmp_path / ".axiom" / "corpus-locks").mkdir()
    assert cm.corpus_uses_lock_files(tmp_path)


@pytest.mark.parametrize(
    ("path", "lock"),
    [
        (f"data/corpus/provisions/us/statute/{VERSION}.jsonl", VERSION),
        (f"data/corpus/inventory/us/statute/{VERSION}.json", VERSION),
        (f"data/corpus/coverage/us/statute/{VERSION}.json", VERSION),
        (f"data/corpus/sources/us/statute/{VERSION}/a/b.html", VERSION),
        ("data/corpus/provisions/us/statute/v.json", None),
        ("data/corpus/inventory/us/statute/v.jsonl.bak", None),
        ("data/corpus/sources/us/statute/v", None),
        ("data/corpus/provisions/us/statute/.jsonl", None),
        ("data/corpus/anchors/us/statute/v.json", None),
    ],
)
def test_lock_path_for_maps_each_protected_path_to_its_scope_lock(path, lock):
    expected = (
        None if lock is None else cm.LOCK_ROOT / "us" / "statute" / f"{lock}.json"
    )
    assert cm.lock_path_for(path) == expected


def test_content_cache_source(tmp_path, monkeypatch):
    data = b"cached\n"
    art = _artifact(PROVISIONS, data)
    cache = tmp_path / "cache"
    monkeypatch.setenv(cm.CACHE_ENV, str(cache))
    assert cm.ContentCacheSource.from_environment() is None
    obj = cache / "objects" / "sha256" / art.sha256[:2] / art.sha256
    obj.parent.mkdir(parents=True)
    source = cm.ContentCacheSource.from_environment()
    assert source is not None and source.open(art) is None
    obj.write_bytes(data)
    assert b"".join(source.open(art)) == data
    obj.write_bytes(data + b"extra")
    with pytest.raises(cm.SourceError, match="cache object holds"):
        source.open(art)


# ------------------------------------------------------ property tests

BEHAVIORS = ("absent", "correct", "wrong_bytes", "truncated", "oversized", "raises")
_PROPERTY_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


@st.composite
def _scenarios(draw):
    count = draw(st.integers(min_value=1, max_value=5))
    payloads = draw(
        st.lists(
            st.binary(min_size=1, max_size=300),
            min_size=count,
            max_size=count,
            unique_by=lambda b: hashlib.sha256(b).hexdigest(),
        )
    )
    source_count = draw(st.integers(min_value=0, max_value=4))
    plan = [
        [draw(st.sampled_from(BEHAVIORS)) for _ in range(source_count)]
        for _ in payloads
    ]
    return payloads, plan


@_PROPERTY_SETTINGS
@given(_scenarios())
def test_property_placed_iff_some_source_verifies(scenario):
    """Fetch fidelity, fail-closed, source order and idempotence, for any sources."""

    payloads, plan = scenario
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        artifacts = [
            _artifact(f"data/corpus/provisions/j{i}/statute/v{i}.jsonl", data)
            for i, data in enumerate(payloads)
        ]
        _write_locks(root, {art.path: data for art, data in zip(artifacts, payloads)})
        sources = [
            FakeSource(
                f"s{index}",
                {
                    art.sha256: (plan[a][index], payloads[a])
                    for a, art in enumerate(artifacts)
                },
            )
            for index in range(len(plan[0]))
        ]

        report = cm.materialize_release_artifacts(
            root, artifacts, sources=_factory(*sources)
        )

        for a, art in enumerate(artifacts):
            behaviors = plan[a]
            target = root / art.path
            if "correct" in behaviors:
                assert target.read_bytes() == payloads[a]
                assert report.materialized[art.path] == f"s{behaviors.index('correct')}"
                assert target.stat().st_nlink == 1
            else:
                assert art.path in report.failed
                # A failed placement leaves no file and no directory behind.
                assert not target.exists()
                assert not target.parent.exists()
        assert _leftovers(root) == []
        if not report.materialized:
            assert _files(root) == []
        assert report.selected == len(artifacts)
        assert len(report.materialized) + len(report.failed) == len(artifacts)

        # Idempotence: nothing left to fetch means no source is opened again.
        if report.ok:
            again = cm.materialize_release_artifacts(
                root, artifacts, sources=_no_sources
            )
            assert again.ok and sorted(again.present) == sorted(
                a.path for a in artifacts
            )


LOCK_STATES = ("no_lock", "unlisted", "release", "other_size", "other_bytes")
DESTINATIONS = ("missing", "release", "lock_bytes", "stray")


@st.composite
def _checkouts(draw):
    count = draw(st.integers(min_value=1, max_value=4))
    payloads = draw(
        st.lists(
            st.binary(min_size=1, max_size=64),
            min_size=count,
            max_size=count,
            unique_by=lambda b: hashlib.sha256(b).hexdigest(),
        )
    )
    rows = [
        (
            draw(st.sampled_from(LOCK_STATES)),
            draw(st.sampled_from(DESTINATIONS)),
            draw(st.booleans()),
        )
        for _ in payloads
    ]
    return payloads, rows


@_PROPERTY_SETTINGS
@given(_checkouts(), st.booleans())
def test_property_protected_paths_hold_only_their_lock_bytes(checkout, verify):
    """What axiom-corpus relies on: encode never writes bytes a lock does not pin.

    For any lock (none, not listing the path, pinning the release's bytes,
    pinning other bytes of another or the same size), any file already there
    and any source: a file is placed iff the lock pins the release's bytes,
    the path is empty and a source verifies; an existing file never changes;
    and every artifact is present, placed, skipped or failed as specified.
    """

    payloads, rows = checkout
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        artifacts, before, pinned = [], {}, {}
        behaviors: dict[str, tuple[str, bytes]] = {}
        for i, (data, (lock_state, destination, source_ok)) in enumerate(
            zip(payloads, rows)
        ):
            art = _artifact(f"data/corpus/provisions/j{i}/statute/v{i}.jsonl", data)
            artifacts.append(art)
            other = {
                "other_size": data + b"~",
                "other_bytes": bytes(b ^ 0x01 for b in data),
            }.get(lock_state)
            sibling = f"data/corpus/inventory/j{i}/statute/v{i}.json"
            if lock_state == "unlisted":
                _write_locks(root, {sibling: b"{}\n"})
            elif lock_state != "no_lock":
                pinned[art.path] = data if lock_state == "release" else other
                _write_locks(root, {art.path: pinned[art.path], sibling: b"{}\n"})
            existing = {
                "missing": None,
                "release": data,
                "lock_bytes": other if other is not None else data + b"#",
                "stray": data + b"!!",
            }[destination]
            if existing is not None:
                (root / art.path).parent.mkdir(parents=True, exist_ok=True)
                (root / art.path).write_bytes(existing)
                before[art.path] = existing
            behaviors[art.sha256] = ("correct" if source_ok else "wrong_bytes", data)
        source = FakeSource("s", behaviors)

        report = cm.materialize_release_artifacts(
            root, artifacts, sources=_factory(source), verify=verify
        )

        opened = []
        for art, (lock_state, _destination, source_ok) in zip(artifacts, rows):
            target = root / art.path
            existing = before.get(art.path)
            same_size = existing is not None and len(existing) == art.byte_count
            same_bytes = existing is not None and hashlib.sha256(
                existing
            ).hexdigest() == (art.sha256)
            present = same_bytes or (same_size and not verify)
            pins_release = lock_state == "release"
            if existing is not None:
                assert target.read_bytes() == existing  # never replaced
            if present:
                assert art.path in report.present
                # A lock pinning other bytes is noted, never written through.
                assert (art.path in report.notes) == (
                    lock_state in {"other_size", "other_bytes"}
                )
            elif not pins_release:
                assert "not placed:" in report.skipped[art.path]
                assert target.exists() == (existing is not None)
            elif existing is not None:
                assert "left untouched" in report.modified[art.path]
            else:
                opened.append(art.path)
                if source_ok:
                    assert report.materialized[art.path] == "s"
                    # The contract: placed bytes are the bytes the lock pins.
                    assert target.read_bytes() == pinned[art.path]
                else:
                    assert art.path in report.failed and not target.exists()
        assert source.opened == opened
        assert (
            report.selected
            == len(artifacts)
            == (
                len(report.present)
                + len(report.materialized)
                + len(report.skipped)
                + len(report.modified)
                + len(report.failed)
            )
        )
        assert _leftovers(root) == []


@_PROPERTY_SETTINGS
@given(
    entries=st.dictionaries(
        st.text(alphabet="abcdefghij0123456789-._", min_size=1, max_size=12).filter(
            lambda s: s not in {".", ".."}
        ),
        st.tuples(
            st.binary(max_size=32),
            st.one_of(
                st.none(),
                st.text(alphabet="0123456789abcdef", min_size=40, max_size=40),
            ),
        ),
        min_size=1,
        max_size=6,
    )
)
def test_property_lock_reader_round_trips_canonical_locks(entries):
    lock_path = cm.lock_path_for(PROVISIONS)
    paths = {
        f"data/corpus/sources/us/statute/{VERSION}/{name}": value
        for name, value in entries.items()
    }
    written = []
    for path, (data, blob) in paths.items():
        entry = {
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }
        if blob:
            entry["git_blob"] = blob
        written.append(entry)

    parsed = cm._parse_scope_lock(_lock_bytes(lock_path, written), lock_path)

    assert parsed == {
        path: cm.LockEntry(hashlib.sha256(data).hexdigest(), len(data), blob)
        for path, (data, blob) in paths.items()
    }


_SEGMENT = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789-.", min_size=1, max_size=12
).filter(lambda s: s not in {".", ".."})


@given(j=_SEGMENT, dc=_SEGMENT, v=_SEGMENT, leaf=_SEGMENT)
def test_property_every_scope_file_maps_to_one_lock(j, dc, v, leaf):
    expected = cm.LOCK_ROOT / j / dc / f"{v}.json"
    for path in (
        f"data/corpus/provisions/{j}/{dc}/{v}.jsonl",
        f"data/corpus/inventory/{j}/{dc}/{v}.json",
        f"data/corpus/coverage/{j}/{dc}/{v}.json",
        f"data/corpus/sources/{j}/{dc}/{v}/{leaf}",
        f"data/corpus/sources/{j}/{dc}/{v}/{leaf}/{leaf}",
    ):
        assert cm.lock_path_for(path) == expected


# ---------------------------------------------------------------- SigV4


def test_sigv4_matches_the_aws_s3_get_object_example():
    """AWS S3 SigV4 documentation example: GET Object with a Range header."""

    headers = cm.sigv4_headers(
        "GET",
        "https://examplebucket.s3.amazonaws.com/test.txt",
        access_key_id="AKIAIOSFODNN7EXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        now=datetime(2013, 5, 24, tzinfo=UTC),
        region="us-east-1",
        headers={"Range": "bytes=0-9"},
    )

    assert headers["Authorization"] == (
        "AWS4-HMAC-SHA256 "
        "Credential=AKIAIOSFODNN7EXAMPLE/20130524/us-east-1/s3/aws4_request, "
        "SignedHeaders=host;range;x-amz-content-sha256;x-amz-date, "
        "Signature=f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41"
    )
    assert headers["x-amz-date"] == "20130524T000000Z"


_HEADER_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz-", min_size=1, max_size=10
).filter(lambda s: s not in {"host", "x-amz-date", "x-amz-content-sha256"})


@given(
    extra=st.dictionaries(
        _HEADER_NAME, st.text(alphabet="abc 123", max_size=10), max_size=4
    ),
    secret=st.text(alphabet="abcdefXYZ/+0123", min_size=1, max_size=20),
)
def test_property_sigv4_ignores_header_case_and_order_but_not_secrets(extra, secret):
    now = datetime(2026, 9, 29, 12, tzinfo=UTC)
    url = "https://acct.r2.cloudflarestorage.com/axiom-corpus/objects/sha256/ab/abc"

    def sign(headers, key=secret):
        return cm.sigv4_headers(
            "GET",
            url,
            access_key_id="k",
            secret_access_key=key,
            now=now,
            headers=headers,
        )["Authorization"]

    reordered = {name.upper(): value for name, value in reversed(list(extra.items()))}
    assert sign(extra) == sign(reordered)
    assert sign(extra) != sign(extra, key=secret + "x")


# ------------------------------------------------------------------ R2


class _Response(io.BytesIO):
    def __init__(self, data: bytes, length: str | None = None):
        super().__init__(data)
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = length


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://r2.test/x", code, "error", Message(), io.BytesIO()
    )


def _r2(opener, **kwargs):
    return cm.R2ObjectSource(
        cm.R2Credentials("https://acct.r2.cloudflarestorage.com", "key-id", "secret"),
        "axiom-corpus",
        opener=opener,
        clock=lambda: datetime(2026, 9, 29, tzinfo=UTC),
        sleep=lambda _seconds: None,
        **kwargs,
    )


def test_r2_source_requests_the_content_addressed_key_with_sigv4():
    data = b"from r2\n"
    art = _artifact(PROVISIONS, data)
    seen = []

    def opener(request, timeout):
        seen.append(request)
        return _Response(data, str(len(data)))

    assert b"".join(_r2(opener).open(art)) == data
    (request,) = seen
    assert request.full_url == (
        "https://acct.r2.cloudflarestorage.com/axiom-corpus/objects/sha256/"
        f"{art.sha256[:2]}/{art.sha256}"
    )
    auth = request.get_header("Authorization")
    assert auth.startswith(
        "AWS4-HMAC-SHA256 Credential=key-id/20260929/auto/s3/aws4_request"
    )
    assert "secret" not in auth


def test_r2_source_absent_retry_and_failure_paths():
    data = b"x\n"
    art = _artifact(PROVISIONS, data)
    assert (
        _r2(lambda r, timeout: (_ for _ in ()).throw(_http_error(404))).open(art)
        is None
    )

    calls = []

    def flaky(request, timeout):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(503)
        return _Response(data, str(len(data)))

    assert b"".join(_r2(flaky).open(art)) == data and len(calls) == 3

    with pytest.raises(cm.SourceError, match="HTTP 403"):
        _r2(lambda r, timeout: (_ for _ in ()).throw(_http_error(403))).open(art)
    with pytest.raises(cm.SourceError, match="HTTP 503"):
        _r2(lambda r, timeout: (_ for _ in ()).throw(_http_error(503))).open(art)
    with pytest.raises(cm.SourceError, match="object holds 99 bytes"):
        _r2(lambda r, timeout: _Response(data, "99")).open(art)

    import http.client

    garbled = []

    def bad_status(request, timeout):
        garbled.append(1)
        raise http.client.BadStatusLine("HTTP/9 ???")

    with pytest.raises(cm.SourceError, match="BadStatusLine"):
        _r2(bad_status).open(art)
    assert len(garbled) == 3  # retried like any transport failure


def test_r2_credentials_resolution(tmp_path):
    assert (
        cm.r2_credentials_from_environment({}, credential_path=tmp_path / "none")
        is None
    )
    stored = tmp_path / "r2.json"
    stored.write_text(
        json.dumps(
            {
                "access_key_id": "file-id",
                "secret_access_key": "file-secret",
                "account_id": "acct",
            }
        )
    )
    from_file = cm.r2_credentials_from_environment({}, credential_path=stored)
    assert from_file == cm.R2Credentials(
        "https://acct.r2.cloudflarestorage.com", "file-id", "file-secret"
    )
    assert "file-secret" not in repr(from_file)
    from_env = cm.r2_credentials_from_environment(
        {
            "R2_ACCESS_KEY_ID": "env-id",
            "R2_SECRET_ACCESS_KEY": "env-secret",
            "R2_ENDPOINT": "https://r2.test/",
        },
        credential_path=stored,
    )
    assert from_env == cm.R2Credentials("https://r2.test", "env-id", "env-secret")
    default = cm.r2_credentials_from_environment(
        {"R2_ACCESS_KEY_ID": "i", "R2_SECRET_ACCESS_KEY": "s"},
        credential_path=tmp_path / "none",
    )
    assert (
        default.endpoint
        == f"https://{cm.DEFAULT_R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    )
    assert (
        cm.r2_credentials_from_environment(
            {
                "R2_ACCESS_KEY_ID": "i",
                "R2_SECRET_ACCESS_KEY": "s",
                "R2_ENDPOINT": "http://r2.test",
            },
            credential_path=tmp_path / "none",
        )
        is None
    )


def test_release_sources_order_and_remote_switches(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    env = {
        cm.CACHE_ENV: str(cache),
        "R2_ACCESS_KEY_ID": "i",
        "R2_SECRET_ACCESS_KEY": "s",
    }

    def labels(**kwargs):
        with cm.release_sources(
            tmp_path, git_commit="a" * 40, r2_bucket="b", environ=env, **kwargs
        ) as sources:
            return [source.label for source in sources]

    assert labels() == ["cache", "git", "git-lock", "r2"]
    assert labels(remote=False) == ["cache", "git", "git-lock"]
    with cm.remote_fetch_disabled():
        assert labels() == ["cache", "git", "git-lock"]
    assert labels() == ["cache", "git", "git-lock", "r2"]


# ------------------------------------------------------------ git


def _git(repo: Path, *args: str, text: bool = True) -> str:
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
        text=text,
    ).stdout


def _rows_bytes(
    body: str = BODY,
    *,
    scope: tuple[str, str, str] = STATUTE,
    citation: str = CITATION,
) -> bytes:
    jurisdiction, document_class, version = scope
    row = {
        "id": f"row-{version}",
        "jurisdiction": jurisdiction,
        "document_class": document_class,
        "version": version,
        "source_path": f"sources/{jurisdiction}/{document_class}/{version}/source.xml",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
        "citation_path": citation,
        "body": body,
    }
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def _write_scope(root: Path, scope: tuple[str, str, str]) -> None:
    jurisdiction, document_class, version = scope
    citation, body = SCOPE_TEXT[scope]
    prefix = f"{jurisdiction}/{document_class}"
    for relative, data in (
        (
            f"provisions/{prefix}/{version}.jsonl",
            _rows_bytes(body, scope=scope, citation=citation),
        ),
        (f"inventory/{prefix}/{version}.json", b"{}\n"),
        (f"coverage/{prefix}/{version}.json", b"{}\n"),
        (f"sources/{prefix}/{version}/fixture-source.txt", b"source\n"),
    ):
        path = root / "data" / "corpus" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-f", "data")
    _git(root, "commit", "--quiet", "-m", message)
    return _git(root, "rev-parse", "HEAD").strip()


def _pre_switch_corpus(
    root: Path, scopes: tuple[tuple[str, str, str], ...] = (STATUTE,)
) -> tuple[str, str]:
    """A corpus repo whose protected files are tracked; returns (commit, release sha)."""

    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    (root / ".gitignore").write_text("data/\nreleases/\n")
    for scope in scopes:
        _write_scope(root, scope)
    _git(root, "add", ".gitignore")
    commit = _commit_all(root, "pre-switch corpus")
    release_sha = write_test_release_object(
        root, RELEASE, list(scopes), git_commit=commit
    )
    return commit, release_sha


def _switch(root: Path) -> str:
    """Move protected files into lock files (as `corpus migrate` does); returns commit."""

    tracked = _git(root, "ls-files", "data/corpus").splitlines()
    _write_locks(
        root,
        {path: (root / path).read_bytes() for path in tracked},
        {path: _git(root, "rev-parse", f"HEAD:{path}").strip() for path in tracked},
    )
    _git(root, "rm", "-r", "--cached", "--quiet", "data/corpus")
    _git(root, "add", str(cm.LOCK_ROOT))
    _git(root, "commit", "--quiet", "-m", "switch")
    shutil.rmtree(root / "data")
    return _git(root, "rev-parse", "HEAD").strip()


def _release(root: Path, sha: str) -> LocalCorpusRelease:
    return LocalCorpusRelease(root, RELEASE, sha, TEST_RELEASE_PUBLIC_KEY)


@needs_git
def test_lock_checkout_places_provisions_from_the_release_commit(tmp_path, capsys):
    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root)
    _switch(root)
    assert not (root / "data").exists()

    release = _release(root, release_sha)

    resolved = resolve_local_corpus_source(CITATION, release)
    assert resolved.body == BODY
    assert release.unplaced == {}
    placed = root / PROVISIONS
    assert placed.read_bytes() == _rows_bytes()
    assert placed.stat().st_nlink == 1
    assert "placed 1 corpus release artifact(s)" in capsys.readouterr().err
    # Only provisions: the resolver reads nothing else.
    assert not (root / "data" / "corpus" / "sources").exists()
    assert _leftovers(root) == []
    # data/ is ignored, so the checkout stays clean for provenance checks.
    assert _git(root, "status", "--porcelain") == ""

    # A second binding (a worker process) finds everything in place.
    again = _release(root, release_sha)
    assert again.content_sha256 == release.content_sha256
    assert "placed" not in capsys.readouterr().err


@needs_git
def test_a_scope_reingested_after_the_release_is_skipped_and_fails_only_when_read(
    tmp_path, capsys
):
    """The checkout's lock pins newer bytes: never write the release's over them."""

    root = tmp_path / "axiom-corpus"
    release_commit, release_sha = _pre_switch_corpus(root, (STATUTE, REGULATION))
    reingested = _rows_bytes("The standard deduction is $204 after a re-ingest.")
    (root / PROVISIONS).write_bytes(reingested)
    _commit_all(root, "re-ingest title 7")
    _switch(root)

    release = _release(root, release_sha)

    err = capsys.readouterr().err
    assert "placed 1 corpus release artifact(s)" in err
    assert "left 1 provisions artifact(s)" in err and release_commit in err
    assert set(release.unplaced) == {PROVISIONS}
    # The other scope reads; the re-ingested one fails when read, saying why.
    assert resolve_local_corpus_source(REG_CITATION, release).body == REG_BODY
    with pytest.raises(CorpusResolutionError, match="is missing") as caught:
        resolve_local_corpus_source(CITATION, release)
    assert "pins other bytes" in str(caught.value)
    assert release_commit in str(caught.value)
    # Nothing was written there: axiom-corpus sees the path as missing, not
    # as current, so `corpus fetch` fills it with the lock's bytes.
    assert not (root / PROVISIONS).exists()
    assert _leftovers(root) == []
    assert _git(root, "status", "--porcelain") == ""

    # After `corpus fetch` places the lock's bytes, binding keeps them.
    (root / PROVISIONS).parent.mkdir(parents=True)
    (root / PROVISIONS).write_bytes(reingested)
    again = _release(root, release_sha)
    assert (root / PROVISIONS).read_bytes() == reingested
    with pytest.raises(CorpusResolutionError, match="do not match") as caught:
        resolve_local_corpus_source(CITATION, again)
    assert "pins other bytes" in str(caught.value)

    # The documented way to read it: a worktree at the release's git.commit,
    # with the release object beside it.
    worktree = tmp_path / "axiom-corpus-at-release"
    _git(root, "worktree", "add", "--detach", "--quiet", str(worktree), release_commit)
    shutil.copytree(root / "releases", worktree / "releases")
    at_release = _release(worktree, release_sha)
    assert resolve_local_corpus_source(CITATION, at_release).body == BODY


@needs_git
def test_a_same_size_reingest_fails_on_read_with_the_reason(tmp_path):
    root = tmp_path / "axiom-corpus"
    release_commit, release_sha = _pre_switch_corpus(root)
    reingested = _rows_bytes("The standard deduction is $204.")
    assert len(reingested) == len(_rows_bytes())
    (root / PROVISIONS).write_bytes(reingested)
    _commit_all(root, "same-size re-ingest")
    _switch(root)
    (root / PROVISIONS).parent.mkdir(parents=True)
    (root / PROVISIONS).write_bytes(reingested)  # as `corpus fetch` leaves it

    release = _release(root, release_sha)

    with pytest.raises(CorpusResolutionError, match="do not match") as caught:
        resolve_local_corpus_source(CITATION, release)
    assert "present by size, but" in str(caught.value)
    assert release_commit in str(caught.value)
    assert (root / PROVISIONS).read_bytes() == reingested


@needs_git
def test_every_scope_skipped_still_binds_and_fails_on_read(tmp_path):
    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root)
    (root / PROVISIONS).write_bytes(
        _rows_bytes("A later, longer statement of the rule.")
    )
    _commit_all(root, "re-ingest")
    _switch(root)

    release = _release(root, release_sha)

    assert not (root / "data").exists()
    with pytest.raises(CorpusResolutionError, match="pins other bytes"):
        resolve_local_corpus_source(CITATION, release)


@needs_git
def test_release_cut_after_the_switch_reads_the_lock_git_blob(tmp_path):
    root = tmp_path / "axiom-corpus"
    _pre_switch_corpus(root)
    # Rebuild the release object over files still on disk, bound to the
    # post-switch commit, whose tree no longer contains data/corpus.
    shutil.rmtree(root / "releases")
    switched_tree_files = {
        p: (root / p).read_bytes() for p in _git(root, "ls-files", "data").splitlines()
    }
    post = _switch(root)
    for path, data in switched_tree_files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_bytes(data)
    release_sha = write_test_release_object(root, RELEASE, [STATUTE], git_commit=post)
    shutil.rmtree(root / "data")

    report = cm.materialize_release_artifacts(
        root.resolve(),
        [
            a
            for a in cm.load_pinned_release(root, RELEASE, release_sha)[1].artifacts
            if a.artifact_class in cm.PLACEABLE_ARTIFACT_CLASSES
        ],
        sources=lambda: cm.release_sources(
            root.resolve(), git_commit=post, r2_bucket="axiom-corpus"
        ),
    )

    assert report.ok and not report.skipped
    assert set(report.source_counts()) == {"git-lock"}
    assert (root / PROVISIONS).read_bytes() == _rows_bytes()


@needs_git
def test_lock_checkout_without_any_source_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root)
    _switch(root)
    shutil.rmtree(root / ".git")  # no history, no cache, no R2 credentials

    with pytest.raises(UnmaterializedCorpusReleaseError, match="Cannot place 1 of 1"):
        _release(root, release_sha)
    assert _files(root) == []

    monkeypatch.setenv(cm.NO_FETCH_ENV, "1")
    with pytest.raises(CorpusLayoutError, match="Canonical data/corpus/provisions"):
        _release(root, release_sha)


@needs_git
def test_a_locally_modified_provisions_file_fails_only_when_read(tmp_path, capsys):
    """Fresh extractor output not yet locked: other scopes still bind and read."""

    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root, (STATUTE, REGULATION))
    _switch(root)
    local = root / PROVISIONS
    local.parent.mkdir(parents=True)
    local.write_bytes(b"edited\n")

    release = _release(root, release_sha)

    assert "left 1 modified provisions file(s) untouched" in capsys.readouterr().err
    assert "left untouched" in release.unplaced[PROVISIONS]
    assert resolve_local_corpus_source(REG_CITATION, release).body == REG_BODY
    with pytest.raises(CorpusResolutionError, match="do not match") as caught:
        resolve_local_corpus_source(CITATION, release)
    assert "not yet locked" in str(caught.value)
    assert local.read_bytes() == b"edited\n"


@needs_git
def test_checkout_without_lock_files_is_never_written(tmp_path):
    """Pre-switch checkouts keep today's behavior: a missing file is an error."""

    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root)
    (root / PROVISIONS).unlink()

    release = _release(root, release_sha)
    with pytest.raises(Exception, match="Verified corpus release artifact is missing"):
        resolve_local_corpus_source(CITATION, release)
    assert not (root / PROVISIONS).exists()
    assert not (root / cm.FETCH_TEMP_DIR).exists()


@needs_git
def test_git_object_store_protocol(tmp_path):
    root = tmp_path / "axiom-corpus"
    commit, _ = _pre_switch_corpus(root)
    data = _rows_bytes()
    store = cm.GitObjectStore(root)
    try:
        assert b"".join(store.open(f"{commit}:{PROVISIONS}", len(data))) == data
        assert store.open(f"{commit}:data/corpus/missing.jsonl", 1) is None
        with pytest.raises(cm.SourceError, match="holds"):
            store.open(f"{commit}:{PROVISIONS}", len(data) + 1)
        with pytest.raises(cm.SourceError, match="not a blob"):
            store.open(f"{commit}:data/corpus", 1)
        # The process restarts cleanly after each refused object.
        assert b"".join(store.open(f"{commit}:{PROVISIONS}", len(data))) == data
        assert store.open(f"{commit}:{PROVISIONS}\n{commit}:x", len(data)) is None
        # Abandoning a stream mid-object resets the batch process too.
        stream = store.open(f"{commit}:{PROVISIONS}", len(data))
        stream.close()
        assert b"".join(store.open(f"{commit}:{PROVISIONS}", len(data))) == data
    finally:
        store.close()

    plain = tmp_path / "plain"
    plain.mkdir()
    missing = cm.GitObjectStore(plain)
    with pytest.raises(cm.SourceError, match="cannot read objects"):
        missing.open(f"{commit}:{PROVISIONS}", len(data))
    with pytest.raises(cm.SourceError, match="cannot read objects"):
        missing.open(f"{commit}:{PROVISIONS}", len(data))
    with pytest.raises(cm.SourceError, match="not on PATH"):
        cm.GitObjectStore(plain, git="").open("x", 1)


@needs_git
def test_trusted_supervisor_git_wrapper_allows_the_batch_reader(tmp_path):
    """Supervised validation reaches git only through this wrapper."""

    spec = importlib.util.spec_from_file_location(
        "provision_verification_supervisor",
        Path(__file__).parents[1] / "scripts" / "provision_verification_supervisor.py",
    )
    provisioner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provisioner)
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = provisioner._install_trusted_git_wrapper(
        tools,
        Path(sys.executable).resolve(),
        # The root-ownership policy (_resolve_trusted_git) is tested with the
        # provisioner; this test is about the wrapper's argument allowlist.
        Path(GIT).resolve(),
    )
    root = tmp_path / "axiom-corpus"
    commit, _ = _pre_switch_corpus(root)
    blob = _git(root, "rev-parse", f"{commit}:{PROVISIONS}").strip()
    data = _rows_bytes()

    store = cm.GitObjectStore(root, git=str(wrapper))
    try:
        assert b"".join(store.open(f"{commit}:{PROVISIONS}", len(data))) == data
        assert b"".join(store.open(blob, len(data))) == data
    finally:
        store.close()


# --------------------------------------------------------- release + CLI


@needs_git
def test_pinned_release_content_binding(tmp_path):
    root = tmp_path / "axiom-corpus"
    commit, release_sha = _pre_switch_corpus(root)
    resolved_root, verified = cm.load_pinned_release(root, RELEASE, release_sha)
    assert resolved_root == root.resolve()
    assert verified.git_commit == commit and verified.r2_bucket == "axiom-corpus"

    payload = json.loads(
        (root / "releases" / RELEASE / f"{release_sha}.json").read_text()
    )
    with pytest.raises(CorpusReleaseObjectError, match="pinned name"):
        verify_pinned_release_object_content(
            payload, name="other-release", content_sha256=release_sha
        )
    payload["content"]["artifacts"][0]["sha256"] = "0" * 64
    with pytest.raises(CorpusReleaseObjectError):
        verify_pinned_release_object_content(
            payload, name=RELEASE, content_sha256=release_sha
        )


@needs_git
def test_corpus_fetch_cli(tmp_path, capsys, monkeypatch):
    root = tmp_path / "axiom-corpus"
    release_commit, release_sha = _pre_switch_corpus(root)
    _switch(root)
    argv = [
        "--corpus-path",
        str(root),
        "--release",
        RELEASE,
        "--content-sha256",
        release_sha,
    ]

    assert cm.run_corpus_fetch([*argv, "--json", "--no-remote"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["materialized"] == 1 and payload["sources"] == {"git": 1}
    assert payload["skipped"] == {} and payload["failed"] == {}

    assert (
        cm.run_corpus_fetch([*argv, "--artifact-class", "inventory", "--verify"]) == 0
    )
    assert "1 placed" in capsys.readouterr().out
    assert (root / INVENTORY).read_text() == "{}\n"

    # Sources are fetched all or nothing, by axiom-corpus only.
    with pytest.raises(SystemExit):
        cm.run_corpus_fetch([*argv, "--artifact-class", "sources"])
    assert "invalid choice" in capsys.readouterr().err

    from axiom_encode import entrypoint

    monkeypatch.setattr(sys, "argv", ["axiom-encode", "corpus-fetch", *argv, "--json"])
    assert entrypoint.main() == 0
    assert json.loads(capsys.readouterr().out)["present"] == 1

    assert (
        cm.run_corpus_fetch(
            [
                "--corpus-path",
                str(root),
                "--release",
                RELEASE,
                "--content-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    assert "release object not found" in capsys.readouterr().err

    (root / PROVISIONS).write_bytes(b"edited\n")
    assert cm.run_corpus_fetch(argv) == 1
    captured = capsys.readouterr()
    assert "1 modified, 0 failed" in captured.out
    assert "left untouched" in captured.err
    assert cm.run_corpus_fetch([*argv, "--json"]) == 1
    assert PROVISIONS in json.loads(capsys.readouterr().out)["modified"]

    # A lock that pins other bytes: skipped, reported, and not a failure.
    (root / PROVISIONS).unlink()
    _repin(root, PROVISIONS, b"re-ingested\n")
    assert cm.run_corpus_fetch(argv) == 0
    captured = capsys.readouterr()
    assert "1 skipped, 0 modified, 0 failed" in captured.out
    assert "pins other bytes" in captured.err and release_commit in captured.err
    assert not (root / PROVISIONS).exists()
    assert cm.run_corpus_fetch([*argv, "--json"]) == 0
    assert PROVISIONS in json.loads(capsys.readouterr().out)["skipped"]

    with pytest.raises(SystemExit):
        cm.run_corpus_fetch(["--corpus-path", str(root), "--release", RELEASE])

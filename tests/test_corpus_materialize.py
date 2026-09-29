"""Placing verified corpus release artifacts in a lock-file corpus checkout."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.error
from contextlib import nullcontext
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path

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
PROTECTED = ("sources", "inventory", "provisions", "coverage")


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
    raise AssertionError("sources must not be opened when nothing is missing")


def _leftovers(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.part")]


# ------------------------------------------------------------ unit tests


def test_first_verifying_source_wins_and_bad_sources_are_skipped(tmp_path):
    data = b'{"citation_path": "x"}\n'
    art = _artifact(PROVISIONS, data)
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
    assert not (tmp_path / PROVISIONS).exists()
    assert _leftovers(tmp_path) == []


def test_existing_file_of_another_size_is_left_untouched(tmp_path):
    data = b"release bytes\n"
    art = _artifact(PROVISIONS, data)
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(b"local edit\n")

    report = cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources)

    assert "left untouched" in report.failed[PROVISIONS]
    assert target.read_bytes() == b"local edit\n"


def test_present_files_open_no_source(tmp_path):
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
    target = tmp_path / PROVISIONS
    target.parent.mkdir(parents=True)
    target.write_bytes(b"row two\n")  # same size, other bytes

    assert cm.materialize_release_artifacts(tmp_path, [art], sources=_no_sources).ok
    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_no_sources, verify=True
    )
    assert "sha256 differs" in report.failed[PROVISIONS]
    assert target.read_bytes() == b"row two\n"


def test_symlinked_directory_on_the_path_is_refused(tmp_path):
    root = tmp_path / "corpus"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (root / "data" / "corpus").mkdir(parents=True)
    (root / "data" / "corpus" / "provisions").symlink_to(elsewhere)
    data = b"row\n"
    art = _artifact(PROVISIONS, data)

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


def test_root_must_be_an_absolute_real_directory(tmp_path):
    with pytest.raises(cm.CorpusMaterializationError, match="absolute"):
        cm.materialize_release_artifacts(Path("relative"), [], sources=_no_sources)
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    with pytest.raises(cm.CorpusMaterializationError, match="real directory"):
        cm.materialize_release_artifacts(link, [], sources=_no_sources)


def test_concurrent_writer_with_the_same_file_is_accepted(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    target = tmp_path / PROVISIONS

    class Racing:
        label = "racing"

        def open(self, artifact):
            def chunks():
                target.write_bytes(data)  # another process wins the race
                yield data

            return chunks()

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Racing())
    )

    assert report.ok and target.read_bytes() == data
    assert _leftovers(tmp_path) == []


def test_concurrent_writer_with_other_bytes_fails_closed(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
    target = tmp_path / PROVISIONS

    class Racing:
        label = "racing"

        def open(self, artifact):
            def chunks():
                target.write_bytes(b"different\n")
                yield data

            return chunks()

    report = cm.materialize_release_artifacts(
        tmp_path, [art], sources=_factory(Racing())
    )

    assert "left untouched" in report.failed[PROVISIONS]
    assert target.read_bytes() == b"different\n"


def test_unwritable_destination_is_reported_not_raised(tmp_path):
    data = b"row\n"
    art = _artifact(PROVISIONS, data)
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
    assert "cannot write" in report.failed[PROVISIONS]
    assert not (tmp_path / PROVISIONS).exists()


def test_source_failing_mid_stream_falls_through_to_the_next(tmp_path):
    data = b"0123456789" * 3
    art = _artifact(PROVISIONS, data)

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


def test_r2_source_refuses_redirects():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requested: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            requested.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/elsewhere")
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        source = cm.R2ObjectSource(
            cm.R2Credentials(f"http://127.0.0.1:{server.server_port}", "id", "secret"),
            "axiom-corpus",
            sleep=lambda _s: None,
        )
        with pytest.raises(cm.SourceError, match="HTTP 302"):
            source.open(_artifact(PROVISIONS, b"x"))
    finally:
        server.shutdown()
        server.server_close()
    assert requested and all(not path.startswith("/elsewhere") for path in requested)


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


@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
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


@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(release=st.binary(min_size=1, max_size=200), local=st.binary(max_size=200))
def test_property_existing_bytes_are_never_clobbered(release, local):
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw).resolve()
        art = _artifact(PROVISIONS, release)
        target = root / PROVISIONS
        target.parent.mkdir(parents=True)
        target.write_bytes(local)
        source = FakeSource("s", {art.sha256: ("correct", release)})

        report = cm.materialize_release_artifacts(root, [art], sources=_factory(source))

        assert target.read_bytes() == local
        assert report.ok == (len(local) == len(release))
        assert source.opened == []


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


def _rows_bytes(body: str = BODY) -> bytes:
    row = {
        "id": f"row-{VERSION}",
        "jurisdiction": "us",
        "document_class": "statute",
        "version": VERSION,
        "source_path": f"sources/us/statute/{VERSION}/source.xml",
        "source_as_of": "2026-01-02",
        "expression_date": "2026-01-01",
        "citation_path": CITATION,
        "body": body,
    }
    return (json.dumps(row, sort_keys=True) + "\n").encode()


def _pre_switch_corpus(root: Path) -> tuple[str, str]:
    """A corpus repo whose protected files are tracked; returns (commit, release sha)."""

    root.mkdir(parents=True)
    _git(root, "init", "--quiet")
    (root / ".gitignore").write_text("data/\nreleases/\n")
    provisions = root / PROVISIONS
    provisions.parent.mkdir(parents=True)
    provisions.write_bytes(_rows_bytes())
    for artifact_class, relative, body in (
        ("inventory", f"inventory/us/statute/{VERSION}.json", "{}\n"),
        ("coverage", f"coverage/us/statute/{VERSION}.json", "{}\n"),
        ("sources", f"sources/us/statute/{VERSION}/fixture-source.txt", "source\n"),
    ):
        path = root / "data" / "corpus" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
    _git(root, "add", ".gitignore")
    _git(root, "add", "-f", "data")
    _git(root, "commit", "--quiet", "-m", "pre-switch corpus")
    commit = _git(root, "rev-parse", "HEAD").strip()
    release_sha = write_test_release_object(
        root, RELEASE, [("us", "statute", VERSION)], git_commit=commit
    )
    return commit, release_sha


def _switch(root: Path) -> str:
    """Move protected files into lock files (as `corpus migrate` does); returns commit."""

    tracked = [p for p in _git(root, "ls-files", "data/corpus").splitlines()]
    files = []
    for path in sorted(tracked):
        data = (root / path).read_bytes()
        files.append(
            {
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
                "git_blob": _git(root, "rev-parse", f"HEAD:{path}").strip(),
            }
        )
    lock = root / cm.LOCK_ROOT / "us" / "statute" / f"{VERSION}.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {"schema_version": "axiom-corpus/corpus-lock/v1", "files": files}, indent=1
        )
        + "\n"
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
    placed = root / PROVISIONS
    assert placed.read_bytes() == _rows_bytes()
    assert placed.stat().st_nlink == 1
    assert "placed 1 corpus release artifact(s)" in capsys.readouterr().err
    # Only provisions: the resolver reads nothing else.
    assert not (root / "data" / "corpus" / "sources").exists()
    # data/ is ignored, so the checkout stays clean for provenance checks.
    assert _git(root, "status", "--porcelain") == ""

    # A second binding (a worker process) finds everything in place.
    again = _release(root, release_sha)
    assert again.content_sha256 == release.content_sha256
    assert "placed" not in capsys.readouterr().err


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
    release_sha = write_test_release_object(
        root, RELEASE, [("us", "statute", VERSION)], git_commit=post
    )
    shutil.rmtree(root / "data")

    report = cm.materialize_release_artifacts(
        root.resolve(),
        [a for a in cm.load_pinned_release(root, RELEASE, release_sha)[1].artifacts],
        sources=lambda: cm.release_sources(
            root.resolve(), git_commit=post, r2_bucket="axiom-corpus"
        ),
    )

    assert report.ok
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

    monkeypatch.setenv(cm.NO_FETCH_ENV, "1")
    with pytest.raises(CorpusLayoutError, match="Canonical data/corpus/provisions"):
        _release(root, release_sha)


@needs_git
def test_lock_checkout_rejects_a_modified_provisions_file(tmp_path):
    root = tmp_path / "axiom-corpus"
    _, release_sha = _pre_switch_corpus(root)
    _switch(root)
    local = root / PROVISIONS
    local.parent.mkdir(parents=True)
    local.write_bytes(b"edited\n")

    with pytest.raises(UnmaterializedCorpusReleaseError, match="left untouched"):
        _release(root, release_sha)
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
    _, release_sha = _pre_switch_corpus(root)
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

    assert cm.run_corpus_fetch([*argv, "--artifact-class", "sources", "--verify"]) == 0
    assert "1 placed" in capsys.readouterr().out
    assert (
        root / f"data/corpus/sources/us/statute/{VERSION}/fixture-source.txt"
    ).read_text() == "source\n"

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
    assert "left untouched" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        cm.run_corpus_fetch(["--corpus-path", str(root), "--release", RELEASE])

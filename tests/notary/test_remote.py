import subprocess

import pytest

from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.remote import GitHubReader, RemoteRepository, _bounded

from .conftest import GitRepoBuilder
from .lineage_fixtures import LANE


class Response:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self.body, self.headers = status, body, headers or {}
        self.closed = False

    def iter_content(self, size):
        yield self.body

    def close(self):
        self.closed = True


def test_reader_binds_fixed_host_and_never_follows_authenticated_redirect(monkeypatch):
    reader = GitHubReader("fixture-never-a-real-token")
    calls = []

    def request(url, **kwargs):
        calls.append((url, kwargs))
        return Response(200, b'{"id":1}')

    monkeypatch.setattr(reader._session, "get", request)
    assert reader.get(f"/repos/{LANE}") == {"id": 1}
    assert calls[0][0] == "https://api.github.com/repos/" + LANE
    assert calls[0][1]["allow_redirects"] is False
    assert reader._session.trust_env is False


@pytest.mark.parametrize(
    "path",
    [
        "https://attacker.invalid",
        "/user",
        "/repos/x/../other",
        "/repos/x#fragment",
        "/repos/x\\other",
    ],
)
def test_api_path_cannot_select_another_origin(path):
    with pytest.raises(IdentityRefusal, match="remote_api_path"):
        GitHubReader().get(path)


def test_stream_limit_closes_response():
    response = Response(200, b"too-large")
    with pytest.raises(IdentityRefusal, match="response_too_large"):
        _bounded(response, 2)
    assert response.closed


@pytest.mark.parametrize(
    "location",
    [
        "http://x.blob.core.windows.net/a",
        "https://attacker.invalid/a",
        "https://x.blob.core.windows.net.attacker.invalid/a",
        "https://user:secret@x.blob.core.windows.net/a",
    ],
)
def test_artifact_redirect_is_constrained(monkeypatch, location):
    reader = GitHubReader()
    monkeypatch.setattr(
        reader, "_response", lambda _: Response(302, headers={"Location": location})
    )
    with pytest.raises(IdentityRefusal, match="artifact_redirect"):
        reader.archive(LANE, "123")


def bare_commit(remote, contents, *, parents=()):
    builder = GitRepoBuilder(remote.path)
    builder.git("config", "user.name", "Fixture")
    builder.git("config", "user.email", "fixture@example.invalid")
    tree = builder.mktree(
        [("100644", "blob", builder.hash_blob(contents), b"runme.py")]
    )
    options = [part for parent in parents for part in ("-p", parent)]
    commit = (
        builder.git("commit-tree", tree, *options, input_bytes=b"fixture\n")
        .decode()
        .strip()
    )
    builder.git("update-ref", "refs/heads/chain", commit)
    return commit


def test_remote_snapshot_reads_raw_data_without_execution(tmp_path):
    with RemoteRepository(LANE) as remote:
        source = b"raise RuntimeError('candidate code must not execute')\n"
        commit = bare_commit(remote, source)
        assert remote.snapshot(commit).blobs == {"runme.py": source}
        assert remote.chain_history(commit)[0].commit == commit
        path = remote.path
    assert not path.exists()


def test_chain_reader_rejects_merge_history():
    with RemoteRepository(LANE) as remote:
        root = bare_commit(remote, b"root")
        left = bare_commit(remote, b"left", parents=(root,))
        right = bare_commit(remote, b"right", parents=(root,))
        merge = bare_commit(remote, b"merge", parents=(left, right))
        with pytest.raises(IdentityRefusal, match="not_linear"):
            remote.chain_history(merge)


def test_fetch_has_clean_environment_and_does_not_echo_credentials(monkeypatch):
    with RemoteRepository(LANE, read_token="fixture-secret") as remote:

        def execute(command, **kwargs):
            assert "fixture-secret" not in " ".join(command)
            assert kwargs["env"]["GIT_CONFIG_GLOBAL"] == "/dev/null"
            assert kwargs["env"]["GIT_CONFIG_NOSYSTEM"] == "1"
            assert "GIT_SSH_COMMAND" not in kwargs["env"]
            return subprocess.CompletedProcess(
                command, 1, b"", b"sensitive diagnostic fixture-secret"
            )

        monkeypatch.setattr(subprocess, "run", execute)
        with pytest.raises(IdentityRefusal, match="^remote_git_failure$"):
            remote.fetch("refs/heads/main")


@pytest.mark.parametrize(
    "ref", ["--upload-pack=evil", "file:///tmp/repo", "refs/heads/../../other", "main"]
)
def test_fetch_ref_is_not_a_git_option_or_local_path(ref):
    with RemoteRepository(LANE) as remote:
        with pytest.raises(IdentityRefusal, match="remote_ref"):
            remote.fetch(ref)


@pytest.mark.parametrize("status", [404, 403, 500])
def test_api_failure_never_means_missing_bootstrap_ref(monkeypatch, status):
    reader = GitHubReader()
    monkeypatch.setattr(reader, "get", lambda _: {"full_name": LANE + "-notary"})
    monkeypatch.setattr(reader, "_response", lambda _: Response(status))
    with pytest.raises(IdentityRefusal, match="api_unavailable"):
        reader.optional_ref(LANE + "-notary", "chain")


def test_missing_repository_is_not_an_empty_chain(monkeypatch):
    reader = GitHubReader()

    def unavailable(_):
        raise IdentityRefusal("remote_api_unavailable")

    monkeypatch.setattr(reader, "collection", unavailable)
    with pytest.raises(IdentityRefusal, match="api_unavailable"):
        reader.optional_ref(LANE + "-notary", "chain")


def test_successful_complete_branch_list_can_establish_absence(monkeypatch):
    reader = GitHubReader()
    monkeypatch.setattr(
        reader, "_response", lambda _: Response(200, b'[{"name":"main"}]')
    )
    assert reader.optional_ref(LANE + "-notary", "chain") is None


def test_existing_chain_in_successful_branch_list_is_not_absent(monkeypatch):
    reader = GitHubReader()
    monkeypatch.setattr(
        reader, "_response", lambda _: Response(200, b'[{"name":"chain"}]')
    )
    assert reader.optional_ref(LANE + "-notary", "chain") == {"name": "chain"}

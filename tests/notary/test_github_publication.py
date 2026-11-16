from copy import deepcopy
from dataclasses import replace

import pytest

from axiom_encode.notary.github_publication import AdmissionChecks, ChainWriter
from axiom_encode.notary.identity import IdentityRefusal
from axiom_encode.notary.publication import plan_finalization, plan_publication
from axiom_encode.notary.remote import RemoteRepository

from .chain_fixtures import Epoch
from .lineage_fixtures import EPOCH, LANE


@pytest.fixture
def local(tmp_path):
    # The production factory has only a fixed GitHub origin. This test-only
    # replacement gives the same Git code a private bare local server.
    with RemoteRepository(LANE + "-notary") as server:
        origin = tmp_path / "origin.git"
        server._run("init", "--bare", str(origin), in_repo=False)

        def connect():
            repo = RemoteRepository(LANE + "-notary")
            repo._run("config", "remote.origin.url", str(origin))
            repo._run("config", "protocol.file.allow", "always")
            return repo

        writer = ChainWriter(LANE + "-notary", "fixture-only")
        writer._repository = connect
        epoch = Epoch.create()
        with connect() as repo:
            parent = None
            for index, snapshot in enumerate(epoch.history):
                parent = writer._commit(repo, snapshot.blobs, parent)
                epoch.history[index] = replace(snapshot, commit=parent)
            repo._run("push", "origin", parent + ":refs/heads/chain")
        yield writer, epoch, connect


def test_real_git_publication_finalization_and_idempotent_retry(local):
    writer, epoch, connect = local
    address, bundle, subject = epoch.receipt()
    plan = plan_publication(
        epoch.history, epoch.anchor, bundle, address, "receipt", epoch.active, subject
    )
    commit = writer.publish(plan, epoch.history[-1])
    with connect() as repo:
        assert repo.fetch("refs/heads/chain") == commit
        epoch.history.append(repo.snapshot(commit))
    retry = plan_publication(
        epoch.history, epoch.anchor, bundle, address, "receipt", epoch.active, subject
    )
    assert writer.publish(retry, epoch.history[-1]) == commit
    final = plan_finalization(epoch.history, epoch.anchor, address, subject)
    final_commit = writer.publish(final, epoch.history[-1])
    with connect() as repo:
        repo.fetch("refs/heads/chain")
        history = repo.chain_history(final_commit)
        assert history[-1].blobs["HEAD.json"] == final.files["HEAD.json"]
        assert len(history) == len(epoch.history) + 1


def test_competing_push_loses_exact_old_oid_cas(local):
    writer, epoch, connect = local
    address, bundle, subject = epoch.receipt()
    plan = plan_publication(
        epoch.history, epoch.anchor, bundle, address, "receipt", epoch.active, subject
    )
    original = writer._push
    winner = None

    def raced(repo, commit, expected):
        nonlocal winner
        with connect() as other:
            other.fetch("refs/heads/chain")
            winner = writer._commit(
                other,
                epoch.history[-1].blobs | {"1" * 64 + ".raw": b"competing"},
                expected,
            )
            original(other, winner, expected)
        original(repo, commit, expected)

    writer._push = raced
    with pytest.raises(IdentityRefusal, match="remote_git_failure"):
        writer.publish(plan, epoch.history[-1])
    with connect() as repo:
        assert repo.fetch("refs/heads/chain") == winner
        assert address + ".json" not in repo.snapshot(winner).blobs


def test_bootstrap_real_empty_repository_is_two_commits_and_cannot_repeat(tmp_path):
    with RemoteRepository(LANE + "-notary") as setup:
        origin = tmp_path / "empty.git"
        setup._run("init", "--bare", str(origin), in_repo=False)

    def connect():
        repo = RemoteRepository(LANE + "-notary")
        repo._run("config", "remote.origin.url", str(origin))
        repo._run("config", "protocol.file.allow", "always")
        repo._environment.update(
            GIT_AUTHOR_DATE="2026-09-20T12:00:00+00:00",
            GIT_COMMITTER_DATE="2026-09-20T12:00:00+00:00",
        )
        return repo

    writer = ChainWriter(LANE + "-notary", "fixture-only")
    writer._repository = connect
    epoch = Epoch.create()
    first = epoch.history[0].blobs
    final = {p: b for p, b in epoch.history[1].blobs.items() if p not in first}
    commit = writer.bootstrap(first, final)
    with connect() as repo:
        repo.fetch("refs/heads/chain")
        history = repo.chain_history(commit)
        assert len(history) == 2
        assert history[0].blobs == first
        assert history[1].blobs == epoch.history[1].blobs
    with pytest.raises(IdentityRefusal, match="publication_cas_lost"):
        writer.bootstrap(first, final)


class CheckAPI:
    def __init__(self):
        self.rows = []
        self.writes = []

    def request(self, method, path, token, *, body=None, status=200):
        assert token == "fixture-only" and path.startswith("/repos/" + LANE + "/")
        if method == "GET":
            return {"total_count": len(self.rows), "check_runs": deepcopy(self.rows)}
        self.writes.append((method, body))
        if method == "POST":
            assert status == 201
            row = body | {"id": 77, "app": {"id": 44}}
            self.rows.append(row)
        else:
            row = next(
                row for row in self.rows if str(row["id"]) == path.split("/")[-1]
            )
            row.update(body)
        return deepcopy(row)


@pytest.fixture
def checks():
    api = CheckAPI()
    return AdmissionChecks(api, LANE, 44, "Axiom notary admission", "fixture-only")


def test_required_check_create_update_and_invalidate(checks):
    head, first, second = "a" * 40, "b" * 64, "c" * 64
    assert checks.write(head, EPOCH, first, admitted=True) == "77"
    assert checks.target(head, EPOCH) == first
    assert checks.write(head, EPOCH, second, admitted=True) == "77"
    assert checks.target(head, EPOCH) == second
    checks.write(head, EPOCH, second, admitted=False)
    with pytest.raises(IdentityRefusal, match="not_admitted"):
        checks.target(head, EPOCH)
    assert len(checks.api.rows) == 1


def test_same_named_actions_check_never_supplies_authority(checks):
    head = "a" * 40
    checks.api.rows = [
        {
            "id": 4,
            "name": checks.name,
            "app": {"id": 15368},
            "head_sha": head,
            "external_id": f"axiom-notary:{EPOCH}:" + "b" * 64,
            "status": "completed",
            "conclusion": "success",
        }
    ]
    with pytest.raises(IdentityRefusal, match="missing"):
        checks.target(head, EPOCH)
    checks.write(head, EPOCH, "b" * 64, admitted=True)
    assert checks.target(head, EPOCH) == "b" * 64


def test_duplicate_app_checks_can_be_reconciled_and_invalidated(checks):
    head = "a" * 40
    checks.write(head, EPOCH, "b" * 64, admitted=True)
    checks.api.rows.append(deepcopy(checks.api.rows[0]) | {"id": 78})
    assert checks.target(head, EPOCH) == "b" * 64
    checks.api.rows[1]["external_id"] = f"axiom-notary:{EPOCH}:" + "c" * 64
    with pytest.raises(IdentityRefusal, match="ambiguous"):
        checks.target(head, EPOCH)
    checks.write(head, EPOCH, "c" * 64, admitted=False)
    assert all(row["conclusion"] == "action_required" for row in checks.api.rows)


def test_bootstrap_up_to_date_porcelain_is_not_a_cas_win():
    class Noop:
        def _run(self, *args):
            return (
                b"To fixture\n=\t"
                + b"a" * 40
                + b":refs/heads/chain\t[up to date]\nDone\n"
            )

    with pytest.raises(IdentityRefusal, match="cas_lost"):
        ChainWriter._push(Noop(), "a" * 40, None)


def test_void_sibling_does_not_invalidate_winning_check(checks):
    head, winner, loser = "a" * 40, "b" * 64, "c" * 64
    checks.write(head, EPOCH, winner, admitted=True)
    checks.invalidate(head, EPOCH, loser)
    assert checks.target(head, EPOCH) == winner
    checks.invalidate(head, EPOCH, winner)
    with pytest.raises(IdentityRefusal, match="not_admitted"):
        checks.target(head, EPOCH)

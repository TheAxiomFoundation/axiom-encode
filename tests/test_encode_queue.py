"""Central encode queue: state machine, atomic claim, lease, derived units."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path

import pytest
import yaml

from axiom_encode.encode_queue import build as queue_build
from axiom_encode.encode_queue import model
from axiom_encode.encode_queue.cli import run_queue
from axiom_encode.encode_queue.model import QueueError
from axiom_encode.encode_queue.service import EncodeQueue
from axiom_encode.encode_queue.store import (
    ApiResponse,
    GitHubApiError,
    GitHubBranchStore,
    GitHubUnknownResult,
    LocalFileStore,
    RevisionConflict,
    git_blob_sha,
)

START = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
V1 = "0.2.2399"
V2 = "0.2.2400"
BRANCH = "encoding-queue-state"
ME = {"machine": "m1", "person": "alice"}


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class FakeGitHub:
    """The GitHub calls the store makes, with a real expected-head check."""

    def __init__(self, clock: Clock, branch: str = BRANCH) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.blobs: dict[str, bytes] = {}
        self.trees: dict[str, dict[str, str]] = {"t0": {}}
        self.commits: dict[str, dict] = {
            "c0": {"tree": "t0", "parent": "", "message": "root"}
        }
        self.refs = {branch: "c0"}
        self.calls: list[str] = []
        self.writes = 0  # commits created
        self.write_requests = 0  # mutations sent, stale ones included

    def headers(self) -> dict[str, str]:
        return {"date": format_datetime(self.clock.now, usegmt=True)}

    def __call__(self, method: str, path: str, payload: dict | None) -> ApiResponse:
        with self.lock:
            if path == "graphql":
                self.calls.append("write")
                return self._mutate(payload["variables"]["input"])
            rest = path.split("/", 3)[3]
            self.calls.append(rest.split("/")[1] if "/" in rest else rest[:7])
            if rest.startswith("git/ref/heads/"):
                branch = rest.removeprefix("git/ref/heads/")
                if branch not in self.refs:
                    raise GitHubApiError(404, "Not Found")
                body = {"object": {"sha": self.refs[branch]}}
            elif rest.startswith("git/trees/"):
                commit = rest.removeprefix("git/trees/").split("?")[0]
                tree = self.trees[self.commits[commit]["tree"]]
                body = {
                    "tree": [
                        {"path": p, "type": "blob", "sha": s}
                        for p, s in sorted(tree.items())
                    ]
                }
            elif rest.startswith("git/blobs/"):
                data = self.blobs[rest.removeprefix("git/blobs/")]
                body = {"content": base64.b64encode(data).decode()}
            elif rest.startswith("commits?sha="):
                commit = rest.removeprefix("commits?sha=").split("&")[0]
                body = []
                while commit and len(body) < 30:
                    entry = self.commits[commit]
                    body.append(
                        {"sha": commit, "commit": {"message": entry["message"]}}
                    )
                    commit = entry["parent"]
            else:
                raise AssertionError(f"unexpected call {method} {path}")
            return ApiResponse(200, body, self.headers())

    def _mutate(self, request: dict) -> ApiResponse:
        self.write_requests += 1
        branch = request["branch"]["branchName"]
        head = self.refs[branch]
        if request["expectedHeadOid"] != head:
            error = {
                "type": "STALE_DATA",
                "message": f'Expected branch to point to "{request["expectedHeadOid"]}"'
                " but it did not. Pull and try again.",
            }
            return ApiResponse(200, {"errors": [error]}, self.headers())
        tree = dict(self.trees[self.commits[head]["tree"]])
        for addition in request["fileChanges"]["additions"]:
            data = base64.b64decode(addition["contents"])
            sha = git_blob_sha(data)
            self.blobs[sha] = data
            tree[addition["path"]] = sha
        for deletion in request["fileChanges"]["deletions"]:
            tree.pop(deletion["path"], None)
        self.writes += 1
        tree_id, commit_id = f"t{self.writes}", f"c{self.writes}"
        self.trees[tree_id] = tree
        message = request["message"]["headline"] + "\n\n" + request["message"]["body"]
        self.commits[commit_id] = {"tree": tree_id, "parent": head, "message": message}
        self.refs[branch] = commit_id
        body = {"data": {"createCommitOnBranch": {"commit": {"oid": commit_id}}}}
        return ApiResponse(200, body, self.headers())

    def document(self, path: str, branch: str = BRANCH) -> dict:
        tree = self.trees[self.commits[self.refs[branch]]["tree"]]
        return json.loads(self.blobs[tree[f"central-queue/{path}"]])

    def put(self, path: str, text: str, branch: str = BRANCH) -> None:
        """Edit the branch by hand, as anyone with write access can."""

        data = text.encode()
        sha = git_blob_sha(data)
        self.blobs[sha] = data
        head = self.refs[branch]
        tree = {**self.trees[self.commits[head]["tree"]], f"central-queue/{path}": sha}
        self.writes += 1
        self.trees[f"t{self.writes}"] = tree
        self.commits[f"c{self.writes}"] = {
            "tree": f"t{self.writes}",
            "parent": head,
            "message": "hand edit",
        }
        self.refs[branch] = f"c{self.writes}"


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def github(clock):
    return FakeGitHub(clock)


def github_store(github, **kwargs) -> GitHubBranchStore:
    kwargs.setdefault("sleep", lambda _seconds: None)
    return GitHubBranchStore("example/repo", api=github, **kwargs)


@pytest.fixture(params=["local", "github"])
def store(request, tmp_path, clock, github):
    if request.param == "local":
        return LocalFileStore(tmp_path / "queue", clock=clock)
    return github_store(github)


def make_queue(store) -> EncodeQueue:
    return EncodeQueue(store, sleep=lambda _seconds: None)


def cite(name: str, jurisdiction: str = "us-az") -> str:
    return f"{jurisdiction}/manual/des/{name}"


def units_document(package: str, citations: list[str], **extra) -> dict:
    return {
        "schema": model.UNITS_SCHEMA,
        "package": package,
        "built_at": model.format_time(START),
        "inputs": {},
        "on_main": 0,
        "units": [
            {"citation": c, "rank": rank} for rank, c in enumerate(citations, start=1)
        ],
        **extra,
    }


def seed(queue: EncodeQueue, packages: dict[str, list[str]]) -> None:
    """Packages in the given order, each with its units."""

    for package in packages:
        program, _, jurisdiction = package.partition("-")
        queue.add_package(program, jurisdiction, person="alice")
    queue.apply_build(
        {p: units_document(p, units) for p, units in packages.items()},
        person="alice",
    )


def claim(queue: EncodeQueue, machine: str = "m1", **kwargs) -> list[dict]:
    kwargs.setdefault("encoder_version", V1)
    return queue.claim(machine=machine, person="alice", **kwargs)


def states(queue: EncodeQueue, **kwargs) -> dict[str, str]:
    return {
        unit["citation"]: unit["state"]
        for package in queue.status(**kwargs)["packages"]
        for unit in package["units"]
    }


def ledger(queue: EncodeQueue, package: str = "snap-us-az") -> dict:
    document = queue._store.snapshot().read(model.ledger_doc(package))
    return (document or {"units": {}})["units"]


A, B, C = cite("a"), cite("b"), cite("c")

# ---------------------------------------------------------------- atomic claim


class _MeetAfterFirstRead:
    """Make two workers read the same revision before either commits."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self._barrier = threading.Barrier(2, timeout=10)
        self._seen: set[int] = set()

    def snapshot(self):
        snapshot = self._inner.snapshot()
        ident = threading.get_ident()
        if ident not in self._seen:
            self._seen.add(ident)
            self._barrier.wait()
        return snapshot

    def commit(self, *args):
        return self._inner.commit(*args)


def test_two_workers_racing_for_one_unit_get_it_once(store):
    seed(make_queue(store), {"snap-us-az": [A]})
    racing = _MeetAfterFirstRead(store)
    results: dict[str, list[dict]] = {}

    def worker(machine: str) -> None:
        results[machine] = claim(make_queue(racing), machine)

    threads = [threading.Thread(target=worker, args=(m,)) for m in ("m1", "m2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    won = [m for m, claimed in results.items() if claimed]
    assert len(results) == 2
    assert len(won) == 1
    assert results[won[0]][0]["citation"] == A
    status = make_queue(store).status()
    assert [row["owner"]["machine"] for row in status["active"]] == won


def test_five_machines_never_claim_a_unit_twice(store):
    citations = [cite(f"u{n:02d}") for n in range(30)]
    seed(make_queue(store), {"snap-us-az": citations})
    claimed: dict[str, list[str]] = {}
    errors: list[BaseException] = []

    def worker(machine: str) -> None:
        # A short real back-off, as in production, so no worker starves.
        queue = EncodeQueue(store, sleep=lambda seconds: time.sleep(seconds / 200))
        mine = claimed.setdefault(machine, [])
        try:
            while True:
                got = claim(queue, machine, count=2)
                if not got:
                    return
                mine.extend(unit["citation"] for unit in got)
        except BaseException as error:  # noqa: BLE001 - reported by the test
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(f"m{n}",)) for n in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    everything = [c for mine in claimed.values() for c in mine]
    assert sorted(everything) == citations
    assert len(set(everything)) == len(everything)


def test_commit_at_an_old_revision_is_a_conflict(store):
    first = store.snapshot()
    store.commit(first.revision, {"packages.json": model.empty_packages()}, "one")
    with pytest.raises(RevisionConflict):
        store.commit(first.revision, {"units/x.json": {}}, "two")
    assert store.snapshot().paths() == ["packages.json"]


def test_store_refuses_paths_outside_the_queue(store):
    revision = store.snapshot().revision
    for path in ("../x.json", "/x.json", "a/../x.json", ".git/x.json", "x.txt"):
        with pytest.raises(QueueError):
            store.commit(revision, {path: {}}, "bad")


def test_store_deletes_a_document(store):
    revision = store.commit(store.snapshot().revision, {"units/a.json": {}}, "add")
    store.commit(revision, {"units/a.json": None}, "delete")
    assert store.snapshot().paths() == []


# ----------------------------------------------------------------------- lease


def test_expired_lease_returns_the_unit_and_the_old_holder_is_told(store, clock):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    first = claim(queue, "m1", lease_seconds=600)[0]
    assert claim(queue, "m2") == []

    clock.advance(600 + model.LEASE_GRACE_SECONDS)
    assert claim(queue, "m2") == []  # still inside the grace period
    clock.advance(1)
    assert states(queue) == {A: "ready"}

    second = claim(queue, "m2")[0]
    assert second["lease_id"] != first["lease_id"]
    beat = queue.heartbeat({A: first["lease_id"]}, **ME)
    assert beat[A]["ok"] is False and "Stop this unit" in beat[A]["error"]
    with pytest.raises(QueueError, match="Stop this unit"):
        queue.report(A, "candidate", lease_id=first["lease_id"], **ME)
    assert states(queue) == {A: "reserved"}


def test_heartbeat_extends_the_lease_and_marks_encoding(store, clock):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A, B]})
    leases = {u["citation"]: u["lease_id"] for u in claim(queue, count=2)}
    clock.advance(1500)
    result = queue.heartbeat(leases, encoding=True, **ME)
    expiry = model.format_time(clock.now + timedelta(seconds=1800))
    assert result == {
        c: {"ok": True, "lease_expires_at": expiry, "superseded": False} for c in leases
    }
    clock.advance(1700)
    assert states(queue) == {A: "encoding", B: "encoding"}
    assert claim(queue, "m2") == []


def test_heartbeat_extends_the_valid_leases_when_one_is_lost(store, clock):
    """Review round 1, finding 5: a heartbeat is not all-or-nothing."""

    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A, B, C]})
    leases = {u["citation"]: u["lease_id"] for u in claim(queue, count=3)}
    queue.release(A, leases[A], **ME)
    clock.advance(1000)
    result = queue.heartbeat(leases, **ME)
    assert result[A]["ok"] is False
    expiry = model.format_time(clock.now + timedelta(seconds=1800))
    assert (
        result[B]
        == result[C]
        == {
            "ok": True,
            "lease_expires_at": expiry,
            "superseded": False,
        }
    )
    assert ledger(queue)[B]["lease_expires_at"] == expiry


def test_heartbeat_does_not_write_a_lease_with_more_than_half_left(github, clock):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    leases = {A: claim(queue)[0]["lease_id"]}
    writes = github.writes
    clock.advance(300)
    first = queue.heartbeat(leases, **ME)
    assert github.writes == writes
    assert first[A]["lease_expires_at"] == model.format_time(
        START + timedelta(seconds=1800)
    )
    clock.advance(700)
    queue.heartbeat(leases, **ME)
    assert github.writes == writes + 1


def test_a_slow_worker_keeps_its_result_when_nobody_claimed_again(store, clock):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease = claim(queue, lease_seconds=60)[0]
    clock.advance(4000)
    queue.report(A, "candidate", lease_id=lease["lease_id"], candidate_pr=7, **ME)
    assert states(queue) == {A: "candidate"}


def test_release_makes_the_unit_ready_at_once(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease = claim(queue, "m1")[0]
    assert queue.release(A, lease["lease_id"], **ME)["repeated"] is False
    assert states(queue) == {A: "ready"}
    # The same release again is the same answer, not an error.
    assert queue.release(A, lease["lease_id"], **ME)["repeated"] is True
    assert claim(queue, "m2")[0]["citation"] == A
    with pytest.raises(QueueError, match="Stop this unit"):
        queue.release(A, lease["lease_id"], **ME)


def test_only_the_owner_uses_a_lease(store):
    """Review round 1, finding 7: the lease id alone is not enough."""

    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease_id = claim(queue, "m1")[0]["lease_id"]
    other = {"machine": "m2", "person": "mallory"}
    with pytest.raises(QueueError, match="another owner"):
        queue.release(A, lease_id, **other)
    with pytest.raises(QueueError, match="another owner"):
        queue.report(A, "candidate", lease_id=lease_id, **other)
    assert queue.heartbeat({A: lease_id}, **other)[A]["ok"] is False
    with pytest.raises(QueueError, match="invalid lease id"):
        queue.release(A, "nope", **ME)
    assert states(queue) == {A: "reserved"}


def test_lease_bounds_and_owner_names_are_checked(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    for kwargs in (
        {"lease_seconds": 10},
        {"lease_seconds": 10**6},
        {"count": 0},
        {"count": 500},
        {"encoder_version": ""},
        {"encoder_version": "1\n2"},
        {"request_id": "x"},
    ):
        with pytest.raises(QueueError):
            claim(queue, **kwargs)
    with pytest.raises(QueueError):
        queue.claim(machine="bad name", person="alice", encoder_version=V1)
    assert states(queue) == {A: "ready"}


# ------------------------------------------------- server time (finding 4)


def test_lease_decisions_use_the_time_of_the_store_not_of_the_worker(github, clock):
    """A worker has no clock input at all: only the store's Date header counts."""

    queue_a = make_queue(github_store(github))
    queue_b = make_queue(github_store(github))
    seed(queue_a, {"snap-us-az": [A]})
    lease = claim(queue_a, "m1", lease_seconds=600)[0]
    assert lease["lease_expires_at"] == model.format_time(
        START + timedelta(seconds=600)
    )
    clock.advance(500)
    assert claim(queue_b, "m2") == []  # live by server time
    clock.advance(600)
    assert claim(queue_b, "m2")[0]["citation"] == A


def test_a_github_answer_without_a_date_is_refused(github):
    def no_date(method, path, payload):
        response = github(method, path, payload)
        return ApiResponse(response.status, response.body, {})

    with pytest.raises(QueueError, match="no clock"):
        GitHubBranchStore("example/repo", api=no_date).snapshot()


# -------------------------------------------------------------------- priority


def test_claim_follows_package_rank_then_unit_rank(store):
    queue = make_queue(store)
    nc1, nc2, ga1 = cite("1", "us-nc"), cite("2", "us-nc"), cite("1", "us-ga")
    seed(queue, {"snap-us-nc": [nc1, nc2], "snap-us-ga": [ga1]})
    claimed = claim(queue, count=3)
    assert [u["citation"] for u in claimed] == [nc1, nc2, ga1]
    assert claimed[2]["lease_id"].startswith("snap-us-ga:")


def test_people_add_and_reorder_packages(store):
    queue = make_queue(store)
    nc1, ga1 = cite("1", "us-nc"), cite("1", "us-ga")
    seed(queue, {"snap-us-nc": [nc1], "snap-us-ga": [ga1]})
    added = queue.add_package("medicaid", "us-fl", person="bob", before="snap-us-nc")
    assert added["rank"] == 1 and added["added_by"] == "bob"
    assert queue.move_package("snap-us-ga", person="bob", before="snap-us-nc") == [
        "medicaid-us-fl",
        "snap-us-ga",
        "snap-us-nc",
    ]
    # medicaid-us-fl has no units before a build, so the next unit is Georgia's.
    assert claim(queue)[0]["citation"] == ga1
    assert queue.move_package("medicaid-us-fl", person="bob")[-1] == "medicaid-us-fl"
    with pytest.raises(QueueError, match="already"):
        queue.add_package("snap", "us-ga", person="bob")
    with pytest.raises(QueueError):
        queue.move_package("snap-us-xx", person="bob")
    with pytest.raises(QueueError, match="longer than"):
        queue.add_package("wic", "us-ga", person="bob", note="n" * 501)
    packages = queue.status()["packages"]
    assert [p["rank"] for p in packages] == [1, 2, 3]
    assert packages[-1]["built"] is False


def test_claim_can_be_limited_to_one_package(store):
    queue = make_queue(store)
    nc1, ga1 = cite("1", "us-nc"), cite("1", "us-ga")
    seed(queue, {"snap-us-nc": [nc1], "snap-us-ga": [ga1]})
    assert claim(queue, package="snap-us-ga")[0]["citation"] == ga1
    with pytest.raises(QueueError):
        claim(queue, package="snap-us-xx")


# --------------------------------------------------------------------- blocked


def test_units_of_a_paused_or_blocked_package_are_never_claimed(store):
    queue = make_queue(store)
    tx1, mn1 = cite("1", "us-tx"), cite("1", "us-mn")
    seed(queue, {"snap-us-tx": [tx1], "snap-us-mn": [mn1]})
    with pytest.raises(QueueError, match="what unblocks"):
        queue.set_package_status("snap-us-tx", "blocked", person="alice")
    queue.set_package_status(
        "snap-us-tx", "blocked", person="alice", unblock="corpus release with TX"
    )
    queue.set_package_status("snap-us-mn", "paused", person="alice")
    assert claim(queue, count=5) == []
    status = queue.status()
    assert states(queue) == {tx1: "blocked", mn1: "blocked"}
    assert status["packages"][0]["units"][0]["unblock"] == "corpus release with TX"

    queue.set_package_status("snap-us-tx", "active", person="alice")
    assert [u["citation"] for u in claim(queue, count=5)] == [tx1]


def test_a_blocked_unit_is_never_claimed_until_a_person_unblocks_it(store, clock):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A, B]})
    lease = claim(queue)[0]
    with pytest.raises(QueueError, match="unblock is required"):
        queue.report(A, "blocked", lease_id=lease["lease_id"], **ME)
    queue.report(
        A,
        "blocked",
        lease_id=lease["lease_id"],
        unblock="engine: table lookup support",
        **ME,
    )
    queue.report(B, "blocked", person="bob", unblock="source page is empty")
    clock.advance(10**6)
    assert claim(queue, "m2", count=5, encoder_version=V2) == []
    row = queue.status()["packages"][0]["units"][0]
    assert row == {
        "citation": A,
        "state": "blocked",
        "unblock": "engine: table lookup support",
    }

    queue.report(A, "ready", person="bob")
    assert [u["citation"] for u in claim(queue, "m2", count=5)] == [A]


def test_a_unit_blocked_by_the_build_or_with_an_open_pr_is_never_claimed(store):
    queue = make_queue(store)
    queue.add_package("snap", "us-az", person="alice")
    queue.add_package("snap", "us-tx", person="alice")
    az = units_document("snap-us-az", [A, B])
    az["units"][0]["open_pr"] = 2100
    tx = units_document(
        "snap-us-tx", [cite("1", "us-tx")], blocked="corpus: not in the pin"
    )
    queue.apply_build({"snap-us-az": az, "snap-us-tx": tx}, person="alice")

    assert [u["citation"] for u in claim(queue, count=5)] == [B]
    status = queue.status()
    assert status["packages"][0]["units"][0] == {
        "citation": A,
        "state": "candidate",
        "candidate_pr": 2100,
    }
    assert status["packages"][1]["unblock"] == "corpus: not in the pin"
    assert status["packages"][1]["counts"]["blocked"] == 1


def test_refused_unit_waits_for_a_new_encoder_version_then_blocks(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    versions = ["0.2.1", "0.2.2", "0.2.3"]
    for attempt, version in enumerate(versions, start=1):
        lease_id = claim(queue, encoder_version=version)[0]["lease_id"]
        if attempt == 1:
            with pytest.raises(QueueError, match="category"):
                queue.report(A, "refused", lease_id=lease_id, **ME)
        result = queue.report(
            A,
            "refused",
            lease_id=lease_id,
            category="validation/numeric-occurrence",
            **ME,
        )
        assert result["attempts"] == attempt
        assert result["refused_versions"] == versions[:attempt]
        for old in versions[:attempt]:
            assert claim(queue, encoder_version=old) == []

    assert states(queue, encoder_version="0.2.9") == {A: "blocked"}
    assert claim(queue, encoder_version="0.2.9") == []
    row = queue.status()["packages"][0]["units"][0]
    assert "requeue" in row["unblock"]
    queue.report(A, "ready", person="bob")
    assert claim(queue, encoder_version="0.2.9")[0]["citation"] == A


def test_refusal_by_version_is_remembered_across_later_claims(store, clock):
    """Review round 1, finding 7: V1 never gets back a unit it was refused on."""

    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease = claim(queue, encoder_version=V1)[0]
    queue.report(A, "refused", lease_id=lease["lease_id"], category="review/no", **ME)

    # V2 claims it and releases it.
    lease = claim(queue, encoder_version=V2)[0]
    queue.release(A, lease["lease_id"], **ME)
    assert claim(queue, encoder_version=V1) == []
    assert ledger(queue)[A]["attempts"] == 1

    # V2 claims it again and its lease expires.
    claim(queue, encoder_version=V2)
    clock.advance(10**5)
    assert claim(queue, encoder_version=V1) == []
    assert states(queue, encoder_version=V1) == {A: "refused"}
    assert claim(queue, encoder_version=V2)[0]["citation"] == A


# ----------------------------------------------------------------- transitions


def test_landing_steps_follow_the_state_machine(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease = claim(queue)[0]
    with pytest.raises(QueueError, match="live lease"):
        queue.report(A, "candidate", person="bob")
    with pytest.raises(QueueError, match="cannot move"):
        queue.report(A, "signing", lease_id=lease["lease_id"], **ME)

    queue.report(A, "candidate", lease_id=lease["lease_id"], candidate_pr=41, **ME)
    assert claim(queue, "m2") == []
    with pytest.raises(QueueError, match="cannot set"):
        queue.report(A, "encoding", person="bob")
    queue.report(A, "signing", person="bob", signing_run=9001)
    queue.report(A, "in_train", person="bob", train_pr=77)
    landed = queue.report(A, "landed", person="bob")
    assert (landed["candidate_pr"], landed["signing_run"], landed["train_pr"]) == (
        41,
        9001,
        77,
    )
    assert landed["owner"] == ME
    with pytest.raises(QueueError, match="cannot move"):
        queue.report(A, "candidate", person="bob")
    with pytest.raises(QueueError, match="not a unit"):
        queue.report(cite("zzz"), "landed", person="bob")
    with pytest.raises(QueueError):
        queue.report(A, "landed")
    with pytest.raises(QueueError, match="longer than"):
        queue.report(B, "blocked", person="bob", unblock="x", note="n" * 501)
    with pytest.raises(QueueError, match="positive"):
        queue.report(A, "landed", person="bob", train_pr=-1)


def test_every_state_has_a_transition_row():
    assert set(model.TRANSITIONS) == set(model.STATES)
    for targets in model.TRANSITIONS.values():
        assert set(targets) <= set(model.STATES)
    assert model.TRANSITIONS[model.LANDED] == ()


# ------------------------------------------- idempotent requests (finding 1)


def test_the_same_claim_request_returns_the_same_units(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A, B, C]})
    first = claim(queue, count=2, request_id="request-0001")
    again = claim(queue, count=2, request_id="request-0001")
    assert again == first
    assert sorted(states(queue).values()) == ["ready", "reserved", "reserved"]
    # Another machine with the same request id gets its own units.
    assert claim(queue, "m2", request_id="request-0001")[0]["citation"] == C


def test_the_same_report_again_gets_the_same_answer(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease_id = claim(queue)[0]["lease_id"]
    first = queue.report(A, "candidate", lease_id=lease_id, candidate_pr=5, **ME)
    again = queue.report(A, "candidate", lease_id=lease_id, candidate_pr=5, **ME)
    assert again == {**first, "repeated": True}
    with pytest.raises(QueueError, match="already closed as candidate"):
        queue.report(A, "refused", lease_id=lease_id, category="x/y", **ME)


class _LoseAnswer:
    """The server acts on a write, then the client gets no answer."""

    def __init__(self, github: FakeGitHub, *, applied: bool) -> None:
        self.github = github
        self.applied = applied
        self.armed = True

    def __call__(self, method, path, payload):
        if path == "graphql" and self.armed:
            self.armed = False
            if self.applied:
                self.github(method, path, payload)
            raise GitHubUnknownResult("gh api timed out")
        return self.github(method, path, payload)


@pytest.mark.parametrize("applied", [True, False])
def test_a_lost_answer_to_a_claim_does_not_strand_the_unit(github, applied):
    """Review round 1, finding 1: the write landed, the answer did not."""

    seed(make_queue(github_store(github)), {"snap-us-az": [A, B]})
    writes = github.writes
    lossy = _LoseAnswer(github, applied=applied)
    queue = make_queue(github_store(lossy))

    claimed = claim(queue)
    assert [u["citation"] for u in claimed] == [A]
    entry = github.document("ledger/snap-us-az.json")["units"]
    assert list(entry) == [A]
    assert entry[A]["lease_id"] == claimed[0]["lease_id"]
    assert entry[A]["owner"] == ME
    # One commit when the first write was applied; one after the retry else.
    assert github.writes == writes + 1


def test_a_lost_answer_to_a_report_is_answered_from_the_branch(github):
    seed(make_queue(github_store(github)), {"snap-us-az": [A]})
    lease_id = claim(make_queue(github_store(github)))[0]["lease_id"]
    lossy = _LoseAnswer(github, applied=True)
    result = make_queue(github_store(lossy)).report(
        A, "candidate", lease_id=lease_id, **ME
    )
    assert result["state"] == "candidate"
    # The worker asks again (it never trusts a timeout): the same answer.
    again = make_queue(github_store(github)).report(
        A, "candidate", lease_id=lease_id, **ME
    )
    assert again["state"] == "candidate" and again["repeated"] is True


def test_a_write_refused_by_a_branch_rule_is_an_error_at_once(github):
    """Review round 1, finding 1: only a stale head is retried."""

    seed(make_queue(github_store(github)), {"snap-us-az": [A]})
    sent = []
    slept = []

    def ruled(method, path, payload):
        if path == "graphql":
            sent.append(1)
            error = {
                "type": "UNPROCESSABLE",
                "message": "Repository rule violations found: changes must be "
                "made through a pull request",
            }
            return ApiResponse(200, {"errors": [error]}, github.headers())
        return github(method, path, payload)

    queue = EncodeQueue(github_store(ruled, sleep=slept.append), sleep=slept.append)
    with pytest.raises(GitHubApiError, match="rule violations"):
        claim(queue)
    assert len(sent) == 1 and slept == []


@pytest.mark.parametrize(
    "limit",
    [
        GitHubApiError(403, "secondary rate limit", {"retry-after": "37"}),
        GitHubApiError(429, "too many requests", {"retry-after": "37"}),
        ApiResponse(
            200,
            {"errors": [{"type": "RATE_LIMITED", "message": "API rate limit"}]},
            {"retry-after": "37"},
        ),
    ],
    ids=["403", "429", "graphql"],
)
def test_a_rate_limit_waits_as_the_server_says_and_sends_again(github, limit):
    seed(make_queue(github_store(github)), {"snap-us-az": [A]})
    slept = []
    state = {"limited": 2}

    def limited(method, path, payload):
        if path == "graphql" and state["limited"]:
            state["limited"] -= 1
            if isinstance(limit, Exception):
                raise limit
            return limit
        return github(method, path, payload)

    queue = make_queue(github_store(limited, sleep=slept.append))
    assert claim(queue)[0]["citation"] == A
    assert slept == [37.0, 37.0]


def test_a_403_that_is_not_a_rate_limit_is_an_error(github):
    def forbidden(method, path, payload):
        raise GitHubApiError(403, "Resource not accessible")

    slept = []
    with pytest.raises(GitHubApiError, match="not accessible"):
        github_store(forbidden, sleep=slept.append).snapshot()
    assert slept == []


def test_a_missing_branch_is_an_error_not_an_empty_queue(github, clock):
    """Review round 1, finding 7."""

    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    del github.refs[BRANCH]
    with pytest.raises(QueueError, match="was not found"):
        queue.status()
    with pytest.raises(QueueError, match="was not found"):
        claim(queue)
    assert BRANCH not in github.refs  # the queue never creates the branch


# --------------------------------------- queue rows are not trusted (finding 2)


def test_the_target_path_is_computed_from_the_citation_never_stored(github):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    assert claim(queue)[0]["target_path"] == "us-az/policies/des/a.yaml"

    document = github.document("units/snap-us-az.json")
    document["units"][0]["target_path"] = "../../../../tmp/evil.yaml"
    github.put("units/snap-us-az.json", json.dumps(document))
    with pytest.raises(QueueError, match="unknown fields"):
        claim(queue, "m2")
    with pytest.raises(QueueError, match="unknown fields"):
        queue.status()


BAD_CITATIONS = [
    "us-az/manual/../../etc/passwd",
    "us-az/manual/des/a\nrm -rf",
    "us-az/manual/des/a;b",
    "us-az/manual/des/$(x)",
    "us-az/manual//a",
    "../us-az/manual/a",
    "/us-az/manual/a",
    "us-az/manual/ a",
    "us-az/manual/..",
    "us-az/a",
    "US-AZ/manual/a",
    "us-az/manual/" + "a" * 300,
]


@pytest.mark.parametrize("citation", BAD_CITATIONS)
def test_an_edited_citation_is_refused(github, citation):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    document = github.document("units/snap-us-az.json")
    document["units"][0]["citation"] = citation
    github.put("units/snap-us-az.json", json.dumps(document))
    with pytest.raises(QueueError, match="invalid citation"):
        claim(queue)
    with pytest.raises(QueueError, match="invalid citation"):
        model.check_citation(citation)


def test_real_citation_shapes_are_accepted():
    for citation in (
        "us/regulation/42/435/1000",
        "us-az/manual/des/faa5/na-medical-expenses-and-deduction/block-5",
        "us-ak/regulation/aac/title-7/chapter-45/7 AAC 45.010",
        "us-nc/manual/dhhs/fns/fns-340-deductions/page-23",
    ):
        assert model.check_citation(citation) == citation


def _edit_ledger(github, mutate) -> None:
    document = github.document("ledger/snap-us-az.json")
    mutate(document["units"][A])
    github.put("ledger/snap-us-az.json", json.dumps(document))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda e: e.update(lease_expires_at="2999-01-01T00:00:00Z"),
            "invalid lease time",
        ),
        (lambda e: e.update(lease_seconds=10**9), "lease must be"),
        (lambda e: e.update(heartbeat_at="2027-01-01T00:00:00Z"), "invalid lease time"),
        (lambda e: e.update(state="done"), "unknown state"),
        (lambda e: e.update(extra=1), "unknown fields"),
        (lambda e: e.update(attempts="many"), "wrong type"),
        (lambda e: e.update(attempts=True), "wrong type"),
        (lambda e: e.update(owner={"machine": "m1"}), "lacks fields"),
        (lambda e: e.update(lease_id="snap-us-nc:" + "0" * 32), "another package"),
        (lambda e: e.pop("lease_expires_at"), "lease without"),
        (lambda e: e.update(claimed_at="yesterday"), "invalid time"),
    ],
)
def test_an_edited_ledger_entry_is_refused(github, mutate, message):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A, B]})
    claim(queue)
    _edit_ledger(github, mutate)
    for action in (queue.status, lambda: claim(queue, "m2")):
        with pytest.raises(QueueError, match=message):
            action()


@pytest.mark.parametrize(
    ("path", "text"),
    [
        ("packages.json", "{not json"),
        ("packages.json", "[]"),
        ("packages.json", '{"schema": "other", "packages": []}'),
        ("packages.json", '{"schema": "axiom-encode/central-queue/packages/v1"}'),
        ("units/snap-us-az.json", '{"schema": "x"}'),
        ("units/snap-us-az.json", "null"),
        ("ledger/snap-us-az.json", '{"units": []}'),
    ],
)
def test_a_damaged_document_is_a_refusal_with_no_traceback(
    github, path, text, capsys, monkeypatch
):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    github.put(path, text)
    with pytest.raises(QueueError):
        queue.status()
    with pytest.raises(QueueError):
        claim(queue)

    monkeypatch.setattr(
        "axiom_encode.encode_queue.cli.GitHubBranchStore",
        lambda *args, **kwargs: github_store(github),
    )
    assert run_queue(["--repo", "example/repo", "status"]) == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("queue: refused: ")
    assert captured.err.count("\n") == 1 and "Traceback" not in captured.err


def test_a_closed_queue_refuses_every_operation(github):
    """A retired generation of the queue can never be written again."""

    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A]})
    document = github.document("packages.json")
    document["moved_to"] = "encoding-queue-state-2026-12"
    github.put("packages.json", json.dumps(document))
    for action in (
        lambda: claim(queue),
        queue.status,
        lambda: queue.add_package("snap", "us-nc", person="alice"),
    ):
        with pytest.raises(
            QueueError, match="moved to branch encoding-queue-state-2026-12"
        ):
            action()


# ---------------------------------------- one commit per batch (finding 3)


def test_a_batch_of_operations_is_one_commit_and_bad_ones_do_not_stop_it(github):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A, B, C]})
    leases = {u["citation"]: u["lease_id"] for u in claim(queue, count=2)}
    writes = github.writes
    results = queue.apply(
        [
            {"op": "report", "citation": A, "state": "candidate",
             "lease_id": leases[A], "candidate_pr": 3, **ME},
            {"op": "report", "citation": B, "state": "refused",
             "lease_id": leases[B], **ME},  # no category: refused, others go on
            {"op": "release", "citation": B, "lease_id": leases[B], **ME},
            {"op": "claim", "count": 2, "encoder_version": V1,
             "request_id": "request-0002", **ME},
            {"op": "heartbeat", "leases": {A: leases[A]}, **ME},
            {"op": "sudo"},
            {"op": "claim", "encoder_version": V1, "shell": "x", **ME},
        ]
    )  # fmt: skip
    assert [r["ok"] for r in results] == [True, False, True, True, True, False, False]
    assert github.writes == writes + 1
    assert [u["citation"] for u in results[3]["claimed"]] == [B, C]
    assert results[4]["leases"][A]["ok"] is False
    assert "unknown fields" in results[6]["error"]
    assert states(queue) == {A: "candidate", B: "reserved", C: "reserved"}
    with pytest.raises(QueueError):
        queue.apply([])
    with pytest.raises(QueueError):
        queue.apply([{"op": "claim"}] * 201)


def test_a_write_carries_only_the_ledger_shard_it_changed(github):
    queue = make_queue(github_store(github))
    seed(queue, {"snap-us-az": [A], "snap-us-nc": [cite("1", "us-nc")]})
    claim(queue, package="snap-us-nc")
    tree = github.trees[github.commits[github.refs[BRANCH]]["tree"]]
    assert "central-queue/ledger/snap-us-nc.json" in tree
    assert "central-queue/ledger/snap-us-az.json" not in tree
    before = dict(tree)
    claim(queue, "m2", package="snap-us-az")
    after = github.trees[github.commits[github.refs[BRANCH]]["tree"]]
    changed = {p for p in after if after[p] != before.get(p)}
    assert changed == {"central-queue/ledger/snap-us-az.json"}


def test_write_volume_for_five_machines_with_eight_workers(github, clock, tmp_path):
    """Review round 1, finding 3: writes depend on machines, not on workers.

    Each machine runs 8 workers and sends one batch per interval: the
    reports of the finished units, a heartbeat for the others, and one claim
    that refills its workers. Two machines always read the same revision, so
    half the batches lose a race and are sent again.
    """

    packages = {
        f"snap-us-{s}": [cite(f"u{n:03d}", f"us-{s}") for n in range(200)]
        for s in ("az", "nc", "ga")
    }
    seed(make_queue(github_store(github)), packages)
    github.calls.clear()
    writes, requests = github.writes, github.write_requests
    machines = [f"m{n}" for n in range(5)]
    queues = {
        m: make_queue(github_store(github, cache_dir=tmp_path / m)) for m in machines
    }
    held: dict[str, dict[str, str]] = {m: {} for m in machines}
    intervals = 20
    done = 0

    def flush(machine: str) -> None:
        nonlocal done
        owner = {"machine": machine, "person": "alice"}
        finished = dict(list(held[machine].items())[:4])  # 4 of 8 finish
        ops = [
            {
                "op": "report",
                "citation": c,
                "state": "candidate",
                "lease_id": i,
                **owner,
            }
            for c, i in finished.items()
        ]
        rest = {c: i for c, i in held[machine].items() if c not in finished}
        if rest:
            ops.append({"op": "heartbeat", "leases": rest, "encoding": True, **owner})
        ops.append(
            {"op": "claim", "count": 8 - len(rest), "encoder_version": V1, **owner}
        )
        results = queues[machine].apply(ops)
        assert all(r["ok"] for r in results)
        done += len(finished)
        held[machine] = {
            **rest,
            **{u["citation"]: u["lease_id"] for u in results[-1]["claimed"]},
        }

    for _interval in range(intervals):
        clock.advance(180)
        threads = [threading.Thread(target=flush, args=(m,)) for m in machines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

    batches = intervals * len(machines)
    commits = github.writes - writes
    sent = github.write_requests - requests
    assert commits == batches  # one commit per machine per interval
    assert done == 4 * (batches - len(machines))
    held_now = [c for m in machines for c in held[m]]
    assert len(held_now) == len(set(held_now)) == 40  # 5 machines x 8 workers
    # Requests per commit, stale retries included, in the worst case where
    # all five machines write in the same second.
    assert sent / commits <= 3.5
    assert len(github.calls) / commits <= 16
    # Per hour at one batch per machine per 180 s: 100 commits for 5 machines.
    assert 3600 / 180 * len(machines) == 100


def test_the_blob_cache_is_checked_against_its_sha(github, tmp_path):
    queue = make_queue(github_store(github, cache_dir=tmp_path / "cache"))
    seed(queue, {"snap-us-az": [A]})
    queue.status()
    cached = sorted((tmp_path / "cache").iterdir())
    assert cached
    for file in cached:
        file.write_text('{"schema": "edited"}')
    fresh = make_queue(github_store(github, cache_dir=tmp_path / "cache"))
    assert states(fresh) == {A: "ready"}  # edited cache files are not used


# ----------------------------------------------------------------------- build


def write_corpus(root: Path, rows: list[dict], documents: list[dict]) -> None:
    bundles = root / "manifests" / "program-bundles"
    bundles.mkdir(parents=True)
    (bundles / "snap.yaml").write_text(
        yaml.safe_dump(
            {
                "schema": "axiom-program-bundle/v2",
                "program": "snap",
                "as_of": "2026-10-06",
                "layers": [
                    {
                        "jurisdiction": "us-az",
                        "screener": documents[:1],
                        "full": documents,
                    }
                ],
            }
        )
    )
    provisions = root / "data" / "corpus" / "provisions" / "us-az" / "manual"
    provisions.mkdir(parents=True)
    (provisions / "faa5.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


DOC = "us-az/manual/des/faa5/medical-deduction"
OTHER_DOC = "us-az/manual/des/faa5/shelter-deduction"
MISSING_DOC = "us-az/manual/des/faa5/not-in-the-pin"


def corpus_rows() -> list[dict]:
    text = "The deduction is the amount over $35 each month. " * 6
    plain = "The household reports the expense to the office. " * 6
    return [
        {"citation_path": DOC, "heading": "Medical", "body": ""},
        {"citation_path": f"{DOC}/block-1", "heading": "Policy\tA\n", "body": plain},
        {"citation_path": f"{DOC}/block-2", "heading": "Amount", "body": text},
        {"citation_path": f"{DOC}/block-2/item-1", "heading": "Sub", "body": text},
        {"citation_path": f"{DOC}/block-3", "heading": "Examples", "body": text},
        {"citation_path": f"{DOC}/block-4", "heading": "Short", "body": "Too short."},
        {"citation_path": f"{DOC}/block-5", "heading": "Long", "body": text * 30},
        {"citation_path": f"{DOC}/block-6;rm", "heading": "Bad", "body": text},
        {"citation_path": f"{OTHER_DOC}/block-1", "heading": "Cap", "body": text},
        {"citation_path": f"{OTHER_DOC}/block-2", "heading": "Rule", "body": plain},
    ]


def bundle_documents() -> list[dict]:
    return [
        {"key": DOC, "citation_path": DOC, "scope": "in", "part": "Deductions"},
        {"key": OTHER_DOC, "citation_path": OTHER_DOC, "scope": "in", "part": "Other"},
        {"key": MISSING_DOC, "citation_path": MISSING_DOC, "scope": "in"},
        {"key": "out", "citation_path": f"{DOC}-out", "scope": "out"},
    ]


def write_module(rulespec: Path, path: str, citation: str | None = None) -> None:
    file = rulespec / path
    file.parent.mkdir(parents=True, exist_ok=True)
    cite_line = f"  corpus_citation_path: {citation}\n" if citation else ""
    file.write_text(f"format: rulespec/v1\nmodule:\n{cite_line}rules: []\n")


def derive(corpus: Path, rulespec: Path, open_prs: list[dict] | None = None) -> dict:
    modules, cited = queue_build.scan_rulespec(rulespec)
    return queue_build.derive_package(
        "snap",
        "us-az",
        bundle=queue_build.load_bundle(corpus, "snap"),
        corpus_index=queue_build.index_corpus(corpus, ["us-az"]),
        main_modules=modules,
        main_cited=cited,
        open_prs=queue_build.open_pr_paths(open_prs or []),
        now=START,
        inputs={"corpus_ref": "c", "rulespec_ref": "r"},
    )


def test_build_derives_units_from_bundle_corpus_main_and_open_prs(tmp_path):
    corpus, rulespec = tmp_path / "corpus", tmp_path / "rulespec"
    write_corpus(corpus, corpus_rows(), bundle_documents())
    rulespec.mkdir()
    write_module(rulespec, "us-az/policies/des/faa5/shelter-deduction/block-2.yaml")
    target = "us-az/policies/des/faa5/medical-deduction/block-1.yaml"
    document = derive(
        corpus,
        rulespec,
        [{"number": 2200, "files": [{"path": target}, {"path": "x.test.yaml"}]}],
    )

    # block-2 is kept over its child; Examples, too short, too long and the
    # row with an invalid citation are out; shelter block-2 is on main.
    # Money units in a computation part come first.
    assert [(u["rank"], u["citation"]) for u in document["units"]] == [
        (1, f"{DOC}/block-2"),
        (2, f"{OTHER_DOC}/block-1"),
        (3, f"{DOC}/block-1"),
    ]
    assert document["units"][2]["open_pr"] == 2200
    assert document["units"][2]["heading"] == "Policy A"
    assert all("target_path" not in unit for unit in document["units"])
    assert document["on_main"] == 1
    assert document["documents"] == {
        "in_scope": 3,
        "held": 2,
        "missing": [MISSING_DOC],
    }
    assert "blocked" not in document
    model.validate_units(document, "snap-us-az")


def test_build_counts_a_unit_of_overlapping_documents_once(tmp_path):
    """Review round 1, finding 7: `on_main` counted such units twice."""

    corpus, rulespec = tmp_path / "corpus", tmp_path / "rulespec"
    documents = [
        *bundle_documents(),
        {"key": "parent", "citation_path": "us-az/manual/des/faa5", "scope": "in"},
        {"key": "again", "citation_path": OTHER_DOC, "scope": "in"},
    ]
    write_corpus(corpus, corpus_rows(), documents)
    rulespec.mkdir()
    write_module(rulespec, "us-az/policies/des/faa5/shelter-deduction/block-2.yaml")
    write_module(rulespec, "us-az/policies/des/faa5/shelter-deduction/block-1.yaml")
    document = derive(corpus, rulespec)
    assert document["on_main"] == 2
    assert len(document["units"]) == 2


def test_a_unit_that_lands_on_main_disappears_from_the_queue(store, tmp_path):
    corpus, rulespec = tmp_path / "corpus", tmp_path / "rulespec"
    write_corpus(corpus, corpus_rows(), bundle_documents())
    rulespec.mkdir()
    queue = make_queue(store)
    queue.add_package("snap", "us-az", person="alice")
    queue.apply_build({"snap-us-az": derive(corpus, rulespec)}, person="alice")
    first = claim(queue)[0]
    assert first["citation"] == f"{DOC}/block-2"
    queue.report(first["citation"], "candidate", lease_id=first["lease_id"], **ME)
    assert len(states(queue)) == 4

    # The candidate merges: the module is on main at the next build.
    write_module(rulespec, first["target_path"], first["citation"])
    # Another module on main cites a whole document.
    write_module(rulespec, "us-az/policies/des/faa5/shelter.yaml", OTHER_DOC)
    result = queue.apply_build({"snap-us-az": derive(corpus, rulespec)}, person="alice")

    assert result["ledger_entries_removed"] == [first["citation"]]
    assert states(queue) == {f"{DOC}/block-1": "ready"}
    package = queue.status()["packages"][0]
    assert package["landed_on_main"] == 3
    claimed = claim(queue, "m2", count=5)
    assert [(u["citation"], u["target_path"]) for u in claimed] == [
        (
            f"{DOC}/block-1",
            "us-az/policies/des/faa5/medical-deduction/block-1.yaml",
        )
    ]


def test_build_never_removes_a_unit_under_a_live_lease(store, clock):
    """Review round 1, finding 6: the owner is told, the lease is kept."""

    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A, B]})
    lease_a, lease_b = (u["lease_id"] for u in claim(queue, count=2))
    result = queue.apply_build(
        {"snap-us-az": units_document("snap-us-az", [C])}, person="alice"
    )
    assert result["leases_superseded"] == [A, B]
    assert result["ledger_entries_removed"] == []
    active = queue.status()["active"]
    assert [(row["citation"], row["superseded"]) for row in active] == [
        (A, True),
        (B, True),
    ]

    # The owner learns it at the next heartbeat; the lease still extends.
    clock.advance(1000)
    beat = queue.heartbeat({A: lease_a}, **ME)[A]
    assert beat["ok"] is True and beat["superseded"] is True
    # Its report is answered, and the result is not recorded.
    assert queue.report(A, "candidate", lease_id=lease_a, **ME)["state"] == (
        "superseded"
    )
    assert A not in ledger(queue)
    # A lease that expires is removed by the next build.
    clock.advance(10**5)
    result = queue.apply_build(
        {"snap-us-az": units_document("snap-us-az", [C])}, person="alice"
    )
    assert result["ledger_entries_removed"] == [B]
    assert ledger(queue) == {}
    with pytest.raises(QueueError, match="not queued"):
        queue.apply_build({"snap-us-zz": units_document("snap-us-zz", [])}, person="a")
    del lease_b


def test_a_unit_that_comes_back_is_no_longer_superseded(store):
    queue = make_queue(store)
    seed(queue, {"snap-us-az": [A]})
    lease_id = claim(queue)[0]["lease_id"]
    queue.apply_build({"snap-us-az": units_document("snap-us-az", [])}, person="alice")
    queue.apply_build({"snap-us-az": units_document("snap-us-az", [A])}, person="alice")
    assert queue.heartbeat({A: lease_id}, **ME)[A]["superseded"] is False
    assert queue.report(A, "candidate", lease_id=lease_id, **ME)["state"] == "candidate"


def test_a_unit_belongs_to_the_highest_ranked_package_that_lists_it(store):
    queue = make_queue(store)
    shared = "us/regulation/42/435/1000"
    for program in ("medicaid", "chip"):
        queue.add_package(program, "us", person="alice")
    result = queue.apply_build(
        {
            "medicaid-us": units_document("medicaid-us", [shared]),
            "chip-us": units_document("chip-us", [shared, "us/regulation/42/457/1"]),
        },
        person="alice",
    )
    assert result["units"] == {"chip-us": 1, "medicaid-us": 1}
    claimed = claim(queue, count=5)
    assert [(u["package"], u["citation"]) for u in claimed] == [
        ("medicaid-us", shared),
        ("chip-us", "us/regulation/42/457/1"),
    ]


def test_build_says_what_unblocks_a_package_with_no_document_in_the_pin(tmp_path):
    corpus, rulespec = tmp_path / "corpus", tmp_path / "rulespec"
    write_corpus(corpus, [], bundle_documents())
    rulespec.mkdir()
    document = derive(corpus, rulespec)
    assert document["units"] == []
    assert document["blocked"].startswith("corpus: none of the 3 in-scope documents")
    with pytest.raises(QueueError, match="no program bundle"):
        queue_build.load_bundle(corpus, "medicaid")


# ------------------------------------------------------------------------- CLI


def run(capsys, store_dir: Path, *argv: str) -> tuple[int, object]:
    code = run_queue(["--store-dir", str(store_dir), *argv])
    out = capsys.readouterr().out
    try:
        return code, json.loads(out)
    except ValueError:
        return code, out


def test_cli_runs_the_whole_cycle_on_a_local_store(tmp_path, capsys):
    corpus, rulespec, store_dir = (tmp_path / n for n in ("corpus", "rules", "q"))
    write_corpus(corpus, corpus_rows(), bundle_documents())
    rulespec.mkdir()
    prs = tmp_path / "prs.json"
    prs.write_text("[]")
    who = ("--person", "alice")
    me = ("--machine", "m1", *who)

    code, _ = run(
        capsys, store_dir, "add", "--program", "snap", "--jurisdiction", "us-az", *who
    )
    assert code == 0
    code, built = run(
        capsys, store_dir, "build", "--corpus", str(corpus), "--rulespec",
        str(rulespec), "--open-prs", str(prs), "--corpus-ref", "abc", *who,
    )  # fmt: skip
    assert (code, built["units"]) == (0, {"snap-us-az": 4})

    code, claimed = run(
        capsys, store_dir, "claim", "--count", "2", "--request-id", "request-0003", *me
    )
    leases = {u["citation"]: u["lease_id"] for u in claimed["claimed"]}
    assert code == 0 and len(leases) == 2
    code, again = run(
        capsys, store_dir, "claim", "--count", "2", "--request-id", "request-0003", *me
    )
    assert again == claimed
    pairs = [f"{c}={i}" for c, i in leases.items()]
    code, beat = run(
        capsys, store_dir, "heartbeat", "--encoding", *me,
        *[x for p in pairs for x in ("--lease", p)],
    )  # fmt: skip
    assert code == 0 and all(r["ok"] for r in beat["leases"].values())

    first, second = leases
    code, _ = run(capsys, store_dir, "release", first, "--lease-id", leases[first], *me)
    assert code == 0
    code, reported = run(
        capsys, store_dir, "report", second, "--state", "refused", *me,
        "--lease-id", leases[second], "--category", "validation/tests-coverage",
    )  # fmt: skip
    assert (code, reported["state"]) == (0, "refused")

    ops = tmp_path / "ops.json"
    ops.write_text(
        json.dumps(
            [{"op": "claim", "encoder_version": V2, "machine": "m1", "person": "alice"}]
        )
    )
    code, applied = run(capsys, store_dir, "apply", "--ops", str(ops))
    assert code == 0 and applied["results"][0]["ok"] is True

    code, status = run(capsys, store_dir, "status", "--json")
    counts = status["packages"][0]["counts"]
    assert (counts["ready"], counts["refused"], counts["reserved"]) == (2, 1, 1)
    assert status["packages"][0]["inputs"]["corpus_ref"] == "abc"
    code, text = run(capsys, store_dir, "status")
    assert code == 0 and "snap-us-az" in text and "Encoding now:" in text

    # A refusal exits 1, prints one line and changes nothing.
    for argv in (
        ("release", first, "--lease-id", "nope", *me),
        ("release", first, "--lease-id", leases[second], "--machine", "m9", *who),
        ("move", "snap-us-zz", *who),
        ("apply", "--ops", str(tmp_path / "absent.json")),
        (
            "build",
            "--corpus",
            str(tmp_path / "absent"),
            "--rulespec",
            str(rulespec),
            *who,
        ),
        ("report", "us-az/manual/x/../y", "--state", "landed", *who),
    ):
        assert run_queue(["--store-dir", str(store_dir), *argv]) == 1
        captured = capsys.readouterr()
        assert captured.err.startswith("queue: refused: ")
        assert captured.err.count("\n") == 1


def test_cli_needs_exactly_one_backend(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AXIOM_ENCODE_QUEUE_DIR", raising=False)
    monkeypatch.delenv("AXIOM_ENCODE_QUEUE_REPO", raising=False)
    assert run_queue(["status"]) == 1
    assert run_queue(["--store-dir", str(tmp_path), "--repo", "a/b", "status"]) == 1
    assert "choose one backend" in capsys.readouterr().err


def test_axiom_encode_queue_is_dispatched_by_the_entrypoint(
    tmp_path, capsys, monkeypatch
):
    from axiom_encode import entrypoint

    monkeypatch.setattr(
        sys, "argv", ["axiom-encode", "queue", "--store-dir", str(tmp_path), "status"]
    )
    assert entrypoint.main() == 0
    assert "Central encode queue" in capsys.readouterr().out


# ---------------------------------------------------------------------- GitHub


def test_github_store_keeps_files_outside_its_prefix(github):
    github.blobs["old"] = b"{}"
    github.trees["t0"]["queues/us-snap-or-ut-pilot.json"] = "old"
    store = github_store(github)
    queue = make_queue(store)
    queue.add_package("snap", "us-az", person="alice")
    assert store.snapshot().paths() == ["packages.json"]
    tree = github.trees[github.commits[github.refs[BRANCH]]["tree"]]
    assert sorted(tree) == [
        "central-queue/packages.json",
        "queues/us-snap-or-ut-pilot.json",
    ]
    message = github.commits[github.refs[BRANCH]]["message"]
    assert message.startswith("queue: add package snap-us-az (alice)")
    assert "Queue-Request: " in message


def test_github_store_checks_its_arguments():
    for repo in ("norepo", "a/b/c", "/b"):
        with pytest.raises(QueueError):
            GitHubBranchStore(repo)
    with pytest.raises(QueueError):
        GitHubBranchStore("a/b", branch="--force")

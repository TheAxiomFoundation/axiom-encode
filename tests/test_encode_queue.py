"""Central encode queue: state machine, atomic claim, lease, derived units."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import threading
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from axiom_encode.encode_queue import build as queue_build
from axiom_encode.encode_queue import model
from axiom_encode.encode_queue.cli import run_queue
from axiom_encode.encode_queue.model import QueueError
from axiom_encode.encode_queue.service import EncodeQueue
from axiom_encode.encode_queue.store import (
    GitHubApiError,
    GitHubBranchStore,
    LocalFileStore,
    RevisionConflict,
)

START = datetime(2026, 11, 25, 12, 0, 0, tzinfo=UTC)
V1 = "0.2.2399"
V2 = "0.2.2400"


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: int) -> None:
        self.now += timedelta(seconds=seconds)


class FakeGitHub:
    """The Git Data API calls the store makes, with a real reference CAS."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.refs: dict[str, str] = {}
        self.blobs: dict[str, str] = {}
        self.trees: dict[str, dict[str, str]] = {}
        self.commits: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _sha(kind: str, value: object) -> str:
        text = kind + json.dumps(value, sort_keys=True)
        return hashlib.sha1(text.encode(), usedforsecurity=False).hexdigest()

    def _is_ancestor(self, ancestor: str, commit: str) -> bool:
        while commit:
            if commit == ancestor:
                return True
            parents = self.commits[commit]["parents"]
            commit = parents[0] if parents else ""
        return False

    def __call__(self, method: str, path: str, payload: dict | None):
        with self.lock:
            self.calls.append((method, path))
            _repos, _owner, _name, rest = path.split("/", 3)
            if method == "GET" and rest.startswith("git/ref/heads/"):
                branch = rest.removeprefix("git/ref/heads/")
                if branch not in self.refs:
                    raise GitHubApiError(404, "Not Found")
                return {"object": {"sha": self.refs[branch]}}
            if method == "GET" and rest.startswith("git/trees/"):
                commit = rest.removeprefix("git/trees/").split("?")[0]
                tree = self.trees[self.commits[commit]["tree"]]
                return {
                    "tree": [
                        {"path": p, "type": "blob", "sha": s}
                        for p, s in sorted(tree.items())
                    ]
                }
            if method == "GET" and rest.startswith("git/blobs/"):
                text = self.blobs[rest.removeprefix("git/blobs/")]
                return {"content": base64.b64encode(text.encode()).decode()}
            if method == "POST" and rest == "git/trees":
                base = payload.get("base_tree")
                tree = dict(self.trees[self.commits[base]["tree"]]) if base else {}
                for entry in payload["tree"]:
                    if "content" in entry:
                        sha = self._sha("blob", entry["content"])
                        self.blobs[sha] = entry["content"]
                        tree[entry["path"]] = sha
                    else:
                        assert entry["sha"] is None
                        tree.pop(entry["path"], None)
                sha = self._sha("tree", tree)
                self.trees[sha] = tree
                return {"sha": sha}
            if method == "POST" and rest == "git/commits":
                body = {"tree": payload["tree"], "parents": payload["parents"]}
                sha = self._sha("commit", {**body, "n": len(self.commits)})
                self.commits[sha] = body
                return {"sha": sha}
            if method == "PATCH" and rest.startswith("git/refs/heads/"):
                branch = rest.removeprefix("git/refs/heads/")
                assert payload["force"] is False
                if not self._is_ancestor(self.refs[branch], payload["sha"]):
                    raise GitHubApiError(422, "Update is not a fast forward")
                self.refs[branch] = payload["sha"]
                return {}
            if method == "POST" and rest == "git/refs":
                branch = payload["ref"].removeprefix("refs/heads/")
                if branch in self.refs:
                    raise GitHubApiError(422, "Reference already exists")
                self.refs[branch] = payload["sha"]
                return {}
            raise AssertionError(f"unexpected call {method} {path}")


@pytest.fixture(params=["local", "github"])
def store(request, tmp_path):
    if request.param == "local":
        return LocalFileStore(tmp_path / "queue")
    return GitHubBranchStore("example/repo", api=FakeGitHub())


@pytest.fixture
def clock():
    return Clock()


def make_queue(store, clock) -> EncodeQueue:
    return EncodeQueue(store, clock=clock, sleep=lambda _seconds: None)


def units_document(package: str, citations: list[str], **extra) -> dict:
    return {
        "schema": model.UNITS_SCHEMA,
        "package": package,
        "built_at": model.format_time(START),
        "inputs": {},
        "on_main": 0,
        "units": [
            {"citation": c, "target_path": f"{c}.yaml", "rank": rank}
            for rank, c in enumerate(citations, start=1)
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


def test_two_workers_racing_for_one_unit_get_it_once(store, clock):
    seed(make_queue(store, clock), {"snap-us-az": ["us-az/a"]})
    racing = _MeetAfterFirstRead(store)
    results: dict[str, list[dict]] = {}

    def worker(machine: str) -> None:
        results[machine] = claim(make_queue(racing, clock), machine)

    threads = [threading.Thread(target=worker, args=(m,)) for m in ("m1", "m2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    won = [m for m, claimed in results.items() if claimed]
    assert len(results) == 2
    assert len(won) == 1
    assert results[won[0]][0]["citation"] == "us-az/a"
    status = make_queue(store, clock).status()
    assert [row["owner"]["machine"] for row in status["active"]] == won


def test_five_machines_never_claim_a_unit_twice(store, clock):
    citations = [f"us-az/u{n:02d}" for n in range(30)]
    seed(make_queue(store, clock), {"snap-us-az": citations})
    claimed: dict[str, list[str]] = {}
    errors: list[BaseException] = []

    def worker(machine: str) -> None:
        # A short real back-off, as in production, so no worker starves.
        queue = EncodeQueue(
            store, clock=clock, sleep=lambda seconds: time.sleep(seconds / 50)
        )
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
        store.commit(first.revision, {"ledger.json": model.empty_ledger()}, "two")
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
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    first = claim(queue, "m1", lease_seconds=600)[0]
    assert claim(queue, "m2") == []

    clock.advance(600 + model.LEASE_GRACE_SECONDS)
    assert claim(queue, "m2") == []  # still inside the grace period
    clock.advance(1)
    assert states(queue) == {"us-az/a": "ready"}

    second = claim(queue, "m2")[0]
    assert second["lease_id"] != first["lease_id"]
    with pytest.raises(QueueError, match="Stop this unit"):
        queue.heartbeat({"us-az/a": first["lease_id"]})
    with pytest.raises(QueueError, match="Stop this unit"):
        queue.report("us-az/a", "candidate", lease_id=first["lease_id"])
    assert states(queue) == {"us-az/a": "reserved"}


def test_heartbeat_extends_the_lease_and_marks_encoding(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a", "us-az/b"]})
    leases = {u["citation"]: u["lease_id"] for u in claim(queue, count=2)}
    clock.advance(1500)
    expiries = queue.heartbeat(leases, encoding=True)
    assert set(expiries.values()) == {
        model.format_time(clock.now + timedelta(seconds=1800))
    }
    clock.advance(1700)
    assert states(queue) == {"us-az/a": "encoding", "us-az/b": "encoding"}
    assert claim(queue, "m2") == []


def test_a_slow_worker_keeps_its_result_when_nobody_claimed_again(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    lease = claim(queue, lease_seconds=60)[0]
    clock.advance(4000)
    queue.report("us-az/a", "candidate", lease_id=lease["lease_id"], candidate_pr=7)
    assert states(queue) == {"us-az/a": "candidate"}


def test_release_makes_the_unit_ready_at_once(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    lease = claim(queue, "m1")[0]
    queue.release("us-az/a", lease["lease_id"])
    assert states(queue) == {"us-az/a": "ready"}
    with pytest.raises(QueueError):
        queue.release("us-az/a", lease["lease_id"])
    assert claim(queue, "m2")[0]["citation"] == "us-az/a"


def test_lease_bounds_and_owner_names_are_checked(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    for kwargs in (
        {"lease_seconds": 10},
        {"lease_seconds": 10**6},
        {"count": 0},
        {"count": 500},
        {"encoder_version": ""},
    ):
        with pytest.raises(QueueError):
            claim(queue, **kwargs)
    with pytest.raises(QueueError):
        queue.claim(machine="bad name", person="alice", encoder_version=V1)
    assert states(queue) == {"us-az/a": "ready"}


# -------------------------------------------------------------------- priority


def test_claim_follows_package_rank_then_unit_rank(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-nc": ["us-nc/1", "us-nc/2"], "snap-us-ga": ["us-ga/1"]})
    order = [u["citation"] for u in claim(queue, count=3)]
    assert order == ["us-nc/1", "us-nc/2", "us-ga/1"]


def test_people_add_and_reorder_packages(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-nc": ["us-nc/1"], "snap-us-ga": ["us-ga/1"]})
    added = queue.add_package("medicaid", "us-fl", person="bob", before="snap-us-nc")
    assert added["rank"] == 1 and added["added_by"] == "bob"
    assert queue.move_package("snap-us-ga", person="bob", before="snap-us-nc") == [
        "medicaid-us-fl",
        "snap-us-ga",
        "snap-us-nc",
    ]
    # medicaid-us-fl has no units before a build, so the next unit is Georgia's.
    assert claim(queue)[0]["citation"] == "us-ga/1"
    assert queue.move_package("medicaid-us-fl", person="bob")[-1] == "medicaid-us-fl"
    with pytest.raises(QueueError, match="already"):
        queue.add_package("snap", "us-ga", person="bob")
    with pytest.raises(QueueError):
        queue.move_package("snap-us-xx", person="bob")
    packages = queue.status()["packages"]
    assert [p["rank"] for p in packages] == [1, 2, 3]
    assert packages[-1]["built"] is False


def test_claim_can_be_limited_to_one_package(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-nc": ["us-nc/1"], "snap-us-ga": ["us-ga/1"]})
    assert claim(queue, package="snap-us-ga")[0]["citation"] == "us-ga/1"
    with pytest.raises(QueueError):
        claim(queue, package="snap-us-xx")


# --------------------------------------------------------------------- blocked


def test_units_of_a_paused_or_blocked_package_are_never_claimed(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-tx": ["us-tx/1"], "snap-us-mn": ["us-mn/1"]})
    with pytest.raises(QueueError, match="what unblocks"):
        queue.set_package_status("snap-us-tx", "blocked", person="alice")
    queue.set_package_status(
        "snap-us-tx", "blocked", person="alice", unblock="corpus release with TX"
    )
    queue.set_package_status("snap-us-mn", "paused", person="alice")
    assert claim(queue, count=5) == []
    status = queue.status()
    assert states(queue) == {"us-tx/1": "blocked", "us-mn/1": "blocked"}
    assert status["packages"][0]["units"][0]["unblock"] == "corpus release with TX"

    queue.set_package_status("snap-us-tx", "active", person="alice")
    assert [u["citation"] for u in claim(queue, count=5)] == ["us-tx/1"]


def test_a_blocked_unit_is_never_claimed_until_a_person_unblocks_it(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a", "us-az/b"]})
    lease = claim(queue)[0]
    with pytest.raises(QueueError, match="what unblocks"):
        queue.report("us-az/a", "blocked", lease_id=lease["lease_id"])
    queue.report(
        "us-az/a",
        "blocked",
        lease_id=lease["lease_id"],
        unblock="engine: table lookup support",
    )
    queue.report("us-az/b", "blocked", person="bob", unblock="source page is empty")
    clock.advance(10**6)
    assert claim(queue, "m2", count=5, encoder_version=V2) == []
    row = queue.status()["packages"][0]["units"][0]
    assert row == {
        "citation": "us-az/a",
        "state": "blocked",
        "unblock": "engine: table lookup support",
    }

    queue.report("us-az/a", "ready", person="bob")
    assert [u["citation"] for u in claim(queue, "m2", count=5)] == ["us-az/a"]


def test_a_unit_blocked_by_the_build_or_with_an_open_pr_is_never_claimed(store, clock):
    queue = make_queue(store, clock)
    queue.add_package("snap", "us-az", person="alice")
    queue.add_package("snap", "us-tx", person="alice")
    az = units_document("snap-us-az", ["us-az/a", "us-az/b"])
    az["units"][0]["open_pr"] = 2100
    tx = units_document("snap-us-tx", ["us-tx/1"], blocked="corpus: not in the pin")
    queue.apply_build({"snap-us-az": az, "snap-us-tx": tx}, person="alice")

    assert [u["citation"] for u in claim(queue, count=5)] == ["us-az/b"]
    status = queue.status()
    assert status["packages"][0]["units"][0] == {
        "citation": "us-az/a",
        "state": "candidate",
        "candidate_pr": 2100,
    }
    assert status["packages"][1]["unblock"] == "corpus: not in the pin"
    assert status["packages"][1]["counts"]["blocked"] == 1


def test_refused_unit_waits_for_a_new_encoder_version_then_blocks(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    with pytest.raises(QueueError, match="category"):
        queue.report("us-az/a", "refused", lease_id=claim(queue)[0]["lease_id"])

    versions = ["0.2.1", "0.2.2", "0.2.3"]
    lease = queue.status()["active"][0]
    assert lease["citation"] == "us-az/a"
    lease_id = None
    for attempt, version in enumerate(versions, start=1):
        if attempt > 1:
            assert claim(queue, encoder_version=versions[attempt - 2]) == []
            lease_id = claim(queue, encoder_version=version)[0]["lease_id"]
        else:
            queue.release(
                "us-az/a",
                next(iter(_ledger(queue)["units"].values()))["lease_id"],
            )
            lease_id = claim(queue, encoder_version=version)[0]["lease_id"]
        result = queue.report(
            "us-az/a",
            "refused",
            lease_id=lease_id,
            category="validation/numeric-occurrence",
        )
        assert result["attempts"] == attempt
        assert result["encoder_version"] == version

    assert states(queue, encoder_version="0.2.9") == {"us-az/a": "blocked"}
    assert claim(queue, encoder_version="0.2.9") == []
    row = queue.status()["packages"][0]["units"][0]
    assert "requeue" in row["unblock"]
    queue.report("us-az/a", "ready", person="bob")
    assert claim(queue, encoder_version="0.2.9")[0]["citation"] == "us-az/a"


def _ledger(queue: EncodeQueue) -> dict:
    return queue._store.snapshot().read(model.LEDGER_DOC)


def test_release_keeps_the_count_of_refusals(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    lease = claim(queue, encoder_version=V1)[0]
    queue.report("us-az/a", "refused", lease_id=lease["lease_id"], category="review/no")
    lease = claim(queue, encoder_version=V2)[0]
    queue.release("us-az/a", lease["lease_id"])
    assert states(queue) == {"us-az/a": "ready"}
    assert _ledger(queue)["units"]["us-az/a"]["attempts"] == 1


# ----------------------------------------------------------------- transitions


def test_landing_steps_follow_the_state_machine(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a"]})
    lease = claim(queue)[0]
    with pytest.raises(QueueError, match="live lease"):
        queue.report("us-az/a", "candidate", person="bob")
    with pytest.raises(QueueError, match="cannot move"):
        queue.report("us-az/a", "signing", lease_id=lease["lease_id"])

    queue.report("us-az/a", "candidate", lease_id=lease["lease_id"], candidate_pr=41)
    assert claim(queue, "m2") == []
    with pytest.raises(QueueError, match="cannot set"):
        queue.report("us-az/a", "encoding", person="bob")
    queue.report("us-az/a", "signing", person="bob", signing_run=9001)
    queue.report("us-az/a", "in_train", person="bob", train_pr=77)
    landed = queue.report("us-az/a", "landed", person="bob")
    assert (landed["candidate_pr"], landed["signing_run"], landed["train_pr"]) == (
        41,
        9001,
        77,
    )
    assert landed["owner"] == {"machine": "m1", "person": "alice"}
    with pytest.raises(QueueError, match="cannot move"):
        queue.report("us-az/a", "candidate", person="bob")
    with pytest.raises(QueueError, match="not a unit"):
        queue.report("us-az/zzz", "landed", person="bob")
    with pytest.raises(QueueError, match="lease id or a person"):
        queue.report("us-az/a", "landed")


def test_every_state_has_a_transition_row():
    assert set(model.TRANSITIONS) == set(model.STATES)
    for targets in model.TRANSITIONS.values():
        assert set(targets) <= set(model.STATES)
    assert model.TRANSITIONS[model.LANDED] == ()


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
        {"citation_path": f"{DOC}/block-1", "heading": "Policy", "body": plain},
        {"citation_path": f"{DOC}/block-2", "heading": "Amount", "body": text},
        {"citation_path": f"{DOC}/block-2/item-1", "heading": "Sub", "body": text},
        {"citation_path": f"{DOC}/block-3", "heading": "Examples", "body": text},
        {"citation_path": f"{DOC}/block-4", "heading": "Short", "body": "Too short."},
        {"citation_path": f"{DOC}/block-5", "heading": "Long", "body": text * 30},
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
    cite = f"  corpus_citation_path: {citation}\n" if citation else ""
    file.write_text(f"format: rulespec/v1\nmodule:\n{cite}rules: []\n")


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

    # block-2 is kept over its child; Examples, too short and too long are out;
    # shelter block-2 is on main. Money units in a computation part come first.
    assert [(u["rank"], u["citation"]) for u in document["units"]] == [
        (1, f"{DOC}/block-2"),
        (2, f"{OTHER_DOC}/block-1"),
        (3, f"{DOC}/block-1"),
    ]
    assert document["units"][2]["open_pr"] == 2200
    assert document["units"][0]["target_path"] == (
        "us-az/policies/des/faa5/medical-deduction/block-2.yaml"
    )
    assert document["on_main"] == 1
    assert document["documents"] == {
        "in_scope": 3,
        "held": 2,
        "missing": [MISSING_DOC],
    }
    assert "blocked" not in document


def test_a_unit_that_lands_on_main_disappears_from_the_queue(store, clock, tmp_path):
    corpus, rulespec = tmp_path / "corpus", tmp_path / "rulespec"
    write_corpus(corpus, corpus_rows(), bundle_documents())
    rulespec.mkdir()
    queue = make_queue(store, clock)
    queue.add_package("snap", "us-az", person="alice")
    queue.apply_build({"snap-us-az": derive(corpus, rulespec)}, person="alice")
    first = claim(queue)[0]
    assert first["citation"] == f"{DOC}/block-2"
    queue.report(first["citation"], "candidate", lease_id=first["lease_id"])
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
    assert claim(queue, "m2", count=5) == [
        {
            "citation": f"{DOC}/block-1",
            "package": "snap-us-az",
            "target_path": "us-az/policies/des/faa5/medical-deduction/block-1.yaml",
            "lease_id": _ledger(queue)["units"][f"{DOC}/block-1"]["lease_id"],
            "lease_expires_at": model.format_time(START + timedelta(seconds=1800)),
        }
    ]


def test_build_keeps_a_live_lease_on_a_unit_that_left_the_membership(store, clock):
    queue = make_queue(store, clock)
    seed(queue, {"snap-us-az": ["us-az/a", "us-az/b"]})
    lease = claim(queue)[0]
    queue.apply_build(
        {"snap-us-az": units_document("snap-us-az", ["us-az/b"])}, person="alice"
    )
    assert "us-az/a" in _ledger(queue)["units"]
    queue.heartbeat({"us-az/a": lease["lease_id"]})
    with pytest.raises(QueueError, match="not a unit"):
        queue.report("us-az/a", "candidate", lease_id=lease["lease_id"])
    with pytest.raises(QueueError, match="not queued"):
        queue.apply_build({"snap-us-zz": units_document("snap-us-zz", [])}, person="a")


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

    assert (
        run(
            capsys,
            store_dir,
            "add",
            "--program",
            "snap",
            "--jurisdiction",
            "us-az",
            *who,
        )[0]
        == 0
    )
    code, built = run(
        capsys, store_dir, "build", "--corpus", str(corpus), "--rulespec",
        str(rulespec), "--open-prs", str(prs), "--corpus-ref", "abc", *who,
    )  # fmt: skip
    assert (code, built["units"]) == (0, {"snap-us-az": 4})

    code, claimed = run(
        capsys, store_dir, "claim", "--machine", "m1", "--count", "2", *who
    )
    leases = {u["citation"]: u["lease_id"] for u in claimed["claimed"]}
    assert code == 0 and len(leases) == 2
    pairs = [f"{c}={i}" for c, i in leases.items()]
    code, beat = run(
        capsys, store_dir, "heartbeat", "--encoding",
        *[x for p in pairs for x in ("--lease", p)],
    )  # fmt: skip
    assert code == 0 and set(beat["lease_expires_at"]) == set(leases)

    first, second = leases
    assert run(capsys, store_dir, "release", first, "--lease-id", leases[first])[0] == 0
    code, reported = run(
        capsys, store_dir, "report", second, "--state", "refused",
        "--lease-id", leases[second], "--category", "validation/tests-coverage",
    )  # fmt: skip
    assert (code, reported["state"]) == (0, "refused")

    code, status = run(capsys, store_dir, "status", "--json")
    counts = status["packages"][0]["counts"]
    assert (counts["ready"], counts["refused"]) == (3, 1)
    assert status["packages"][0]["inputs"]["corpus_ref"] == "abc"
    code, text = run(capsys, store_dir, "status")
    assert code == 0 and "snap-us-az" in text and "Encoding now:" in text

    # A refusal exits 1 and changes nothing.
    assert run(capsys, store_dir, "release", first, "--lease-id", "nope")[0] == 1
    assert run(capsys, store_dir, "move", "snap-us-zz", *who)[0] == 1


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


def test_github_store_creates_the_branch_once_and_keeps_other_files(clock):
    api = FakeGitHub()
    store = GitHubBranchStore("example/repo", branch="state", api=api)
    assert store.snapshot().revision == ""
    first = make_queue(store, clock)
    first.add_package("snap", "us-az", person="alice")
    head = api.refs["state"]
    assert list(api.trees[api.commits[head]["tree"]]) == ["central-queue/packages.json"]

    # A second first commit loses: the reference exists.
    with pytest.raises(RevisionConflict):
        store.commit("", {"ledger.json": model.empty_ledger()}, "late")
    # Files outside the prefix (the old dispatcher state) are not listed.
    tree = dict(api.trees[api.commits[head]["tree"]])
    tree["queues/us-snap-or-ut-pilot.json"] = "x"
    api.trees["t2"] = tree
    api.commits["c2"] = {"tree": "t2", "parents": [head]}
    api.refs["state"] = "c2"
    assert store.snapshot().paths() == ["packages.json"]
    first.add_package("snap", "us-nc", person="alice")
    final = api.trees[api.commits[api.refs["state"]]["tree"]]
    assert "queues/us-snap-or-ut-pilot.json" in final


def test_github_store_passes_other_api_errors_on():
    def api(method, path, payload):
        raise GitHubApiError(403, "Resource not accessible")

    store = GitHubBranchStore("example/repo", api=api)
    with pytest.raises(GitHubApiError):
        store.snapshot()
    for repo in ("norepo", "a/b/c", "/b"):
        with pytest.raises(QueueError):
            GitHubBranchStore(repo)

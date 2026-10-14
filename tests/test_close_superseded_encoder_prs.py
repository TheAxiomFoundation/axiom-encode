"""Tests for the person-run superseded encoder PR closer."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

import scripts.close_superseded_encoder_prs as closer
from scripts.close_superseded_encoder_prs import GitHubError

REPO = "TheAxiomFoundation/rulespec-us"
MEMO = "us/guidance/usda/fns/snap-obbb-alien-eligibility-implementation-memo"
COLA = "us/guidance/usda/fns/snap-fy2026-cola/page-1"


def _pull(
    number: int,
    citation: str,
    *,
    run_id: int | None = None,
    base: str = "main",
    created_at: str | None = None,
    draft: bool = True,
    title: str | None = None,
    head_repo: str = REPO,
    head: str | None = None,
) -> dict[str, Any]:
    run_id = run_id if run_id is not None else 35_000_000_000 + number
    return {
        "number": number,
        "state": "open",
        "draft": draft,
        "title": title or f"Add signed encoding manifest for {citation}",
        "created_at": created_at or f"2026-09-{10 + number % 20:02d}T00:00:00Z",
        "html_url": f"https://github.com/{REPO}/pull/{number}",
        "head": {
            "ref": head or f"axiom/signed-backfill-us-{run_id}-1",
            "sha": f"{number:040d}",
            "repo": {"full_name": head_repo},
        },
        "base": {"ref": base},
    }


def _manifest(citation: str, generated_at: str) -> dict[str, Any]:
    raw = json.dumps({"citation": citation, "generated_at": generated_at})
    return {"encoding": "base64", "content": base64.b64encode(raw.encode()).decode()}


class FakeGitHub:
    def __init__(self, pulls: list[dict[str, Any]]) -> None:
        self.pulls = pulls
        self.commits: dict[int, list[str]] = {}
        self.files: dict[int, list[dict[str, str]]] = {}
        self.contents: dict[tuple[str, str], dict[str, Any]] = {}
        self.writes: list[tuple[str, str, dict[str, Any]]] = []

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        params = params or {}
        if path == f"repos/{REPO}":
            return {"default_branch": "main"}
        if path == f"repos/{REPO}/pulls":
            return self.pulls if params.get("page") == "1" else []
        prefix = f"repos/{REPO}/pulls/"
        if path.startswith(prefix) and path.endswith("/commits"):
            number = int(path[len(prefix) : -len("/commits")])
            names = self.commits.get(number, [closer.BOT_AUTHOR])
            return [{"commit": {"author": {"name": name}}} for name in names]
        if path.startswith(prefix) and path.endswith("/files"):
            number = int(path[len(prefix) : -len("/files")])
            return self.files.get(number, []) if params.get("page") == "1" else []
        contents = f"repos/{REPO}/contents/"
        if path.startswith(contents):
            key = (path[len(contents) :], params["ref"])
            if key not in self.contents:
                raise GitHubError(404, "Not Found")
            return self.contents[key]
        raise AssertionError(f"unexpected GET {path}")

    def post(self, path: str, body: dict[str, Any]) -> Any:
        self.writes.append(("POST", path, body))
        return {}

    def patch(self, path: str, body: dict[str, Any]) -> Any:
        self.writes.append(("PATCH", path, body))
        return {}


def _plan(github: FakeGitHub) -> dict[int, closer.Decision]:
    decisions, _pulls = closer.plan(github, REPO, country="us")
    return {decision.number: decision for decision in decisions}


def test_older_pr_for_same_citation_and_base_is_closed() -> None:
    github = FakeGitHub([_pull(1370, MEMO), _pull(1371, MEMO), _pull(1351, COLA)])
    decisions = _plan(github)
    assert set(decisions) == {1370}
    assert decisions[1370].action == "close"
    assert "superseded by #1371" in decisions[1370].reason


def test_newest_pr_is_kept_among_many() -> None:
    github = FakeGitHub(
        [
            _pull(1372, MEMO, base="axiom/signed-backfill-us-35001504609-1"),
            _pull(1374, MEMO, base="axiom/signed-backfill-us-35001504609-1"),
            _pull(1375, MEMO, base="axiom/signed-backfill-us-35001504609-1"),
        ]
    )
    decisions = _plan(github)
    assert {n: d.action for n, d in decisions.items()} == {
        1372: "close",
        1374: "close",
    }
    assert all("#1375" in decision.reason for decision in decisions.values())


def test_different_base_is_not_a_supersession() -> None:
    # A stacked repair builds on the older PR rather than replacing it.
    github = FakeGitHub(
        [
            _pull(1371, MEMO, run_id=35001504609),
            _pull(1375, MEMO, base="axiom/signed-backfill-us-35001504609-1"),
        ]
    )
    assert _plan(github) == {}


def test_order_is_by_creation_time_not_number() -> None:
    github = FakeGitHub(
        [
            _pull(10, MEMO, created_at="2026-09-20T00:00:00Z"),
            _pull(11, MEMO, created_at="2026-09-19T00:00:00Z"),
        ]
    )
    decisions = _plan(github)
    assert set(decisions) == {11}
    assert "superseded by #10" in decisions[11].reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"title": f"Integrate signed {MEMO}"},
        {"head": "codex/snap-immigration-memo-final"},
        {"head": "axiom/signed-backfill-dk-35000001370-1"},
        {"head_repo": "someone/rulespec-us"},
    ],
)
def test_only_workflow_encoder_prs_are_considered(overrides: dict) -> None:
    github = FakeGitHub([_pull(1370, MEMO, **overrides), _pull(1371, MEMO)])
    assert _plan(github) == {}


def test_ready_for_review_pr_is_kept() -> None:
    github = FakeGitHub([_pull(1370, MEMO, draft=False), _pull(1371, MEMO)])
    decision = _plan(github)[1370]
    assert decision.action == "keep"
    assert "ready for review" in decision.reason


def test_pr_with_a_human_commit_is_kept() -> None:
    github = FakeGitHub([_pull(1370, MEMO), _pull(1371, MEMO)])
    github.commits[1370] = [closer.BOT_AUTHOR, "Max Ghenis"]
    decision = _plan(github)[1370]
    assert decision.action == "keep"
    assert "not made by axiom-encode-bot" in decision.reason


def test_base_of_an_open_stacked_pr_is_kept() -> None:
    older = _pull(1370, MEMO, run_id=34925528280)
    child = _pull(
        1380,
        "us/statute/7/2015/f",
        base="axiom/signed-backfill-us-34925528280-1",
    )
    github = FakeGitHub([older, _pull(1371, MEMO), child])
    decision = _plan(github)[1370]
    assert decision.action == "keep"
    assert "base of open #1380" in decision.reason


def test_stack_parent_stays_while_its_newest_child_is_open() -> None:
    parent = _pull(1370, MEMO, run_id=34925528280)
    first = _pull(1372, MEMO, base="axiom/signed-backfill-us-34925528280-1")
    second = _pull(1373, MEMO, base="axiom/signed-backfill-us-34925528280-1")
    github = FakeGitHub([parent, _pull(1371, MEMO), first, second])
    decisions = _plan(github)
    # 1372 is superseded by 1373 on the same base; 1373 keeps the parent open.
    assert decisions[1372].action == "close"
    assert decisions[1370].action == "keep"
    assert "base of open #1373" in decisions[1370].reason


def test_a_fully_superseded_stack_closes_together() -> None:
    path = ".axiom/encoding-manifests/us/guidance/usda/fns/memo.json"
    parent = _pull(1370, MEMO, run_id=34925528280)
    child = _pull(1372, MEMO, base="axiom/signed-backfill-us-34925528280-1")
    github = FakeGitHub([parent, _pull(1371, MEMO), child])
    github.files[1372] = [{"filename": path, "status": "modified"}]
    github.contents[(path, f"{1372:040d}")] = _manifest(
        MEMO, "2026-09-16T00:00:00+00:00"
    )
    github.contents[(path, "main")] = _manifest(MEMO, "2026-10-01T00:00:00+00:00")
    decisions = _plan(github)
    assert decisions[1370].action == "close"
    assert decisions[1372].action == "close"
    assert "newer signed encoding" in decisions[1372].reason


class TestLandedEncoding:
    PATH = ".axiom/encoding-manifests/us/guidance/usda/fns/memo.json"

    def _github(self, main_generated_at: str | None, main_citation: str = MEMO):
        github = FakeGitHub([_pull(1371, MEMO)])
        github.files[1371] = [
            {"filename": ".axiom/index/provisions_to_rules.json", "status": "modified"},
            {"filename": self.PATH, "status": "added"},
        ]
        github.contents[(self.PATH, f"{1371:040d}")] = _manifest(
            MEMO, "2026-09-15T17:30:00+00:00"
        )
        if main_generated_at is not None:
            github.contents[(self.PATH, "main")] = _manifest(
                main_citation, main_generated_at
            )
        return github

    def test_newer_encoding_on_main_supersedes(self) -> None:
        decision = _plan(self._github("2026-10-01T00:00:00Z"))[1371]
        assert decision.action == "close"
        assert "newer signed encoding" in decision.reason

    @pytest.mark.parametrize(
        "generated_at", ["2026-09-15T17:30:00+00:00", "2026-08-01T00:00:00+00:00"]
    )
    def test_same_or_older_encoding_on_main_does_not(self, generated_at) -> None:
        assert _plan(self._github(generated_at)) == {}

    def test_absent_manifest_on_main_does_not(self) -> None:
        assert _plan(self._github(None)) == {}

    def test_other_citation_at_the_path_does_not(self) -> None:
        github = self._github("2026-10-01T00:00:00Z", main_citation=COLA)
        assert _plan(github) == {}

    def test_naive_timestamps_are_not_compared(self) -> None:
        assert _plan(self._github("2026-10-01T00:00:00")) == {}


def test_dry_run_is_the_default_and_writes_nothing(capsys) -> None:
    github = FakeGitHub([_pull(1370, MEMO), _pull(1371, MEMO)])
    assert closer.main(["--country", "us"], github=github) == 0
    assert github.writes == []
    out = capsys.readouterr().out
    assert "close #1370" in out
    assert "dry run; pass --apply to close" in out


def test_apply_comments_then_closes_without_deleting_branches() -> None:
    github = FakeGitHub([_pull(1370, MEMO), _pull(1371, MEMO)])
    assert closer.main(["--country", "us", "--apply"], github=github) == 0
    assert [(method, path) for method, path, _body in github.writes] == [
        ("POST", f"repos/{REPO}/issues/1370/comments"),
        ("PATCH", f"repos/{REPO}/pulls/1370"),
    ]
    assert "superseded by #1371" in github.writes[0][2]["body"]
    assert github.writes[1][2] == {"state": "closed"}


def test_kept_prs_are_never_written() -> None:
    github = FakeGitHub([_pull(1370, MEMO, draft=False), _pull(1371, MEMO)])
    assert closer.main(["--apply"], github=github) == 0
    assert github.writes == []


def test_json_plan(capsys) -> None:
    github = FakeGitHub([_pull(1370, MEMO), _pull(1371, MEMO)])
    assert closer.main(["--json"], github=github) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == [
        {
            "number": 1370,
            "citation": MEMO,
            "action": "close",
            "reason": payload[0]["reason"],
            "url": f"https://github.com/{REPO}/pull/1370",
        }
    ]


def test_country_must_be_two_letters() -> None:
    with pytest.raises(SystemExit):
        closer.main(["--country", "us-tn"], github=FakeGitHub([]))

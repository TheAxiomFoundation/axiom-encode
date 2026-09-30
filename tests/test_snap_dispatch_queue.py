from __future__ import annotations

import base64
import copy
import importlib.util
import json
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "snap_dispatch_queue.py"
WORKFLOW = ROOT / ".github" / "workflows" / "snap-dispatch-queue.yml"
ENCODER_REPO = "TheAxiomFoundation/axiom-encode"
RULESPEC_REPO = "TheAxiomFoundation/rulespec-us"
TIP = "a" * 40
CORPUS_REF = "c" * 40
ENGINE_REF = "e" * 40
RELEASE = "us-rulespec-2026-08-08-obbb-alien-snap"
NOW = datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def _load():
    spec = importlib.util.spec_from_file_location("snap_dispatch_queue", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


q = _load()


# -- fixtures ---------------------------------------------------------------


def _page(citation: str, body: str | None, **extra: Any) -> dict[str, Any]:
    return {
        "citation_path": citation,
        "jurisdiction": citation.split("/", 1)[0],
        "heading": extra.pop("heading", citation.rsplit("/", 1)[-1]),
        "body": body,
        **extra,
    }


def _oregon_like() -> list[dict[str, Any]]:
    root = "us-or/manual/odhs/open"
    records = [_page(root, None, kind="document")]
    for number in (1, 2, 10):
        records.append(
            _page(
                f"{root}/page-{number}",
                f"SNAP rule text on page {number}. " * 40,
                parent_citation_path=root,
            )
        )
    records.append(
        _page(f"{root}/page-3", "TANF only. " * 100, parent_citation_path=root)
    )
    return records


def _utah_like() -> list[dict[str, Any]]:
    records = []
    for topic in ("100-general", "200-income"):
        base = f"us-ut/manual/dws/eligibility-manual/{topic}"
        records.append(_page(base, None, kind="document"))
        for block in (1, 2):
            records.append(
                _page(
                    f"{base}/block-{block}",
                    f"SNAP {topic} block {block}. ",
                    parent_citation_path=base,
                )
            )
    return records


def _toml_payload(text: str) -> dict[str, str]:
    return {"content": base64.b64encode(text.encode()).decode()}


def _toolchain(release: str = RELEASE) -> str:
    return f'[toolchain]\naxiom_corpus_release = "{release}"\n'


def _workflow_toolchain() -> str:
    return (
        "[workflow_toolchain]\n"
        f'axiom_corpus_ref = "{CORPUS_REF}"\n'
        f'axiom_rules_engine_ref = "{ENGINE_REF}"\n'
    )


def _state(
    citations: list[str], *, status: str = "pending", active: bool = True
) -> dict[str, Any]:
    return {
        "schema": q.SCHEMA,
        "queue_id": "us-snap-test",
        "state": "active" if active else "paused",
        "created_at": "2026-09-30T00:00:00Z",
        "updated_at": "2026-09-30T00:00:00Z",
        "settings": dict(q.DEFAULT_SETTINGS),
        "build": {"corpus_release": RELEASE, "corpus_ref": CORPUS_REF},
        "items": [
            {
                "citation": citation,
                "jurisdiction": "us-or",
                "label": citation,
                "chars": 100,
                "status": status,
                "attempts": [],
            }
            for citation in citations
        ],
    }


class FakeGitHub:
    """In-memory stand-in for the few GitHub endpoints the dispatcher uses."""

    def __init__(self) -> None:
        self.tip = TIP
        self.release = RELEASE
        self.runs: dict[int, dict[str, Any]] = {}
        self.jobs: dict[int, list[dict[str, Any]]] = {}
        self.pulls: dict[int, dict[str, Any]] = {}
        self.dispatches: list[dict[str, Any]] = []
        self.cancelled: list[int] = []
        self.return_run_id = True
        self.fail_dispatch_at: int | None = None
        self.fail_get: set[str] = set()
        self.next_id = 1000

    # helpers used by tests
    def finish(self, run_id: int, conclusion: str) -> None:
        self.runs[run_id].update(status="completed", conclusion=conclusion)

    def add_pull(self, run_id: int, *, state: str = "open", merged: bool = False):
        number = len(self.pulls) + 1
        self.pulls[number] = {
            "number": number,
            "html_url": f"https://github.com/{RULESPEC_REPO}/pull/{number}",
            "state": state,
            "merged_at": "2026-09-30T13:00:00Z" if merged else None,
            "head": {"ref": f"axiom/signed-backfill-us-{run_id}-1"},
        }
        return self.pulls[number]

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        params = params or {}
        for fragment in self.fail_get:
            if fragment in path:
                raise q.GitHubError(502, "bad gateway")
        if path == f"repos/{RULESPEC_REPO}/git/ref/heads/main":
            return {"object": {"sha": self.tip}}
        if path == f"repos/{RULESPEC_REPO}/contents/.axiom/toolchain.toml":
            return _toml_payload(_toolchain(self.release))
        if path == f"repos/{RULESPEC_REPO}/contents/.axiom/workflow-toolchain.toml":
            return _toml_payload(_workflow_toolchain())
        if path == f"repos/{ENCODER_REPO}/actions/workflows/{q.ENCODE_WORKFLOW}/runs":
            since = params["created"].removeprefix(">=")
            return {
                "workflow_runs": [
                    run for run in self.runs.values() if run["created_at"] >= since
                ]
            }
        match = re.fullmatch(rf"repos/{ENCODER_REPO}/actions/runs/(\d+)", path)
        if match:
            return self.runs[int(match.group(1))]
        match = re.fullmatch(rf"repos/{ENCODER_REPO}/actions/runs/(\d+)/jobs", path)
        if match:
            return {"jobs": self.jobs.get(int(match.group(1)), [])}
        if path == f"repos/{RULESPEC_REPO}/pulls":
            branch = params["head"].split(":", 1)[1]
            return [p for p in self.pulls.values() if p["head"]["ref"] == branch]
        match = re.fullmatch(rf"repos/{RULESPEC_REPO}/pulls/(\d+)", path)
        if match:
            return self.pulls[int(match.group(1))]
        raise AssertionError(f"unexpected GET {path} {params}")

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        if path.endswith(f"/actions/workflows/{q.ENCODE_WORKFLOW}/dispatches"):
            if self.fail_dispatch_at == len(self.dispatches):
                raise q.GitHubError(500, "dispatch failed")
            self.dispatches.append(body)
            run_id = self.next_id
            self.next_id += 1
            self.runs[run_id] = {
                "id": run_id,
                "status": "waiting",
                "conclusion": None,
                "run_attempt": 1,
                "created_at": q._iso(NOW),
                "display_title": q.RUN_TITLE_PREFIX + body["inputs"]["citation"],
                "triggering_actor": {"login": q.DISPATCH_ACTOR},
                "html_url": f"https://github.com/{ENCODER_REPO}/actions/runs/{run_id}",
            }
            return {"workflow_run_id": run_id} if self.return_run_id else None
        match = re.fullmatch(rf"repos/{ENCODER_REPO}/actions/runs/(\d+)/cancel", path)
        if match:
            run_id = int(match.group(1))
            self.cancelled.append(run_id)
            self.finish(run_id, "cancelled")
            return None
        raise AssertionError(f"unexpected POST {path}")


def _tick(state, github, *, now=NOW, dry_run=False):
    return q.tick(
        state,
        github=github,
        encoder_repo=ENCODER_REPO,
        now=now,
        dry_run=dry_run,
        sleep=lambda _seconds: None,
    )


def _run_id(item: dict[str, Any]) -> int:
    return item["attempts"][-1]["run_id"]


# -- unit selection ---------------------------------------------------------


def test_large_manual_splits_into_snap_pages_in_natural_order():
    units, stats = q.select_units(_oregon_like(), max_unit_chars=1_000)

    assert [unit["citation"].rsplit("/", 1)[-1] for unit in units] == [
        "page-1",
        "page-2",
        "page-10",
    ]
    assert stats["not_snap"] == 1


def test_small_topics_are_one_unit_each_with_their_blocks():
    units, _stats = q.select_units(_utah_like(), max_unit_chars=1_000)

    assert [unit["citation"] for unit in units] == [
        "us-ut/manual/dws/eligibility-manual/100-general",
        "us-ut/manual/dws/eligibility-manual/200-income",
    ]
    assert all(unit["chars"] > 0 for unit in units)


def test_nested_containers_descend_only_as_far_as_needed():
    records = [
        _page("us-mo/manual/dss/snap", None),
        _page("us-mo/manual/dss/snap/1105", None),
        _page("us-mo/manual/dss/snap/1105/a", "SNAP " * 300),
        _page("us-mo/manual/dss/snap/1105/b", "SNAP " * 300),
        _page("us-mo/manual/dss/snap/1110", None),
        _page("us-mo/manual/dss/snap/1110/a", "SNAP " * 20),
    ]

    units, _stats = q.select_units(records, max_unit_chars=2_000)

    assert [unit["citation"] for unit in units] == [
        "us-mo/manual/dss/snap/1105/a",
        "us-mo/manual/dss/snap/1105/b",
        "us-mo/manual/dss/snap/1110",
    ]


def test_empty_containers_are_skipped_and_oversize_leaves_are_flagged():
    records = [
        _page("us-ky/manual/snap", None),
        _page("us-ky/manual/snap/empty", None),
        _page("us-ky/manual/snap/huge", "SNAP " * 1_000),
    ]

    units, stats = q.select_units(records, max_unit_chars=100)

    assert [unit["citation"] for unit in units] == ["us-ky/manual/snap/huge"]
    assert units[0]["oversize"] is True
    assert stats["empty"] == 1
    assert stats["oversize"] == 1


def test_a_split_container_with_its_own_text_is_counted():
    records = [
        _page("us-xx/manual/ch", "SNAP chapter intro"),
        _page("us-xx/manual/ch/1", "SNAP " * 50),
        _page("us-xx/manual/ch/2", "SNAP " * 50),
    ]

    _units, stats = q.select_units(records, max_unit_chars=300)

    assert stats["container_text_dropped"] == 1


def test_table_of_contents_pages_are_skipped():
    contents = " ".join(f"SNAP section {n}{'.' * 40} {n}" for n in range(12))
    records = [
        _page("us-or/manual/open", None),
        _page("us-or/manual/open/page-2", contents),
        _page("us-or/manual/open/page-3", "SNAP rule text. " * 200),
    ]

    units, stats = q.select_units(records, max_unit_chars=1_000)

    assert [unit["citation"] for unit in units] == ["us-or/manual/open/page-3"]
    assert stats["table_of_contents"] == 1


def test_snap_marker_ignores_the_ordinary_word_snap():
    assert q._is_snap("Supplemental Nutrition Assistance Program")
    assert q._is_snap("SNAP households")
    assert not q._is_snap("a snapshot of the case")


# -- building ---------------------------------------------------------------


def _git_commit(repo: Path) -> str:
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "commit",
            "-q",
            "-m",
            "fixture",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _fixture_checkouts(
    tmp_path: Path,
    *,
    queue_status: str = "published_current",
    release_has_scope: bool = True,
    encoded: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus"
    scope = {
        "jurisdiction": "us-or",
        "document_class": "manual",
        "version": "2026-07-16-or-programs-eligibility-notebook",
    }
    provisions = corpus / q.PROVISIONS_DIR / "us-or" / "manual"
    provisions.mkdir(parents=True)
    (provisions / f"{scope['version']}.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in _oregon_like())
    )
    (corpus / "manifests" / "releases").mkdir(parents=True)
    (corpus / q.INVENTORY_PATH).write_text(
        yaml.safe_dump(
            {
                "states": [
                    {
                        "jurisdiction": "us-or",
                        "queue_status": queue_status,
                        "target_scope": scope,
                    }
                ]
            }
        )
    )
    (corpus / q.RELEASES_DIR / f"{RELEASE}.json").write_text(
        json.dumps({"name": RELEASE, "scopes": [scope] if release_has_scope else []})
    )
    corpus_ref = _git_commit(corpus)

    rulespec = tmp_path / "rulespec-us"
    (rulespec / ".axiom").mkdir(parents=True)
    (rulespec / ".axiom/toolchain.toml").write_text(_toolchain())
    (rulespec / ".axiom/workflow-toolchain.toml").write_text(
        _workflow_toolchain().replace(CORPUS_REF, corpus_ref)
    )
    manifests = rulespec / q.ENCODING_MANIFESTS_DIR
    manifests.mkdir(parents=True)
    for index, citation in enumerate(encoded):
        (manifests / f"{index}.json").write_text(json.dumps({"citation": citation}))
    _git_commit(rulespec)
    return corpus, rulespec


def test_build_makes_a_paused_queue_from_the_pinned_release(tmp_path):
    encoded = "us-or/manual/odhs/open/page-2"
    corpus, rulespec = _fixture_checkouts(tmp_path, encoded=(encoded,))

    state = q.build_queue(
        corpus,
        rulespec,
        queue_id="us-snap-or-pilot",
        jurisdictions=["us-or"],
        max_unit_chars=1_000,
        now=NOW,
    )

    assert state["state"] == "paused"
    assert state["build"]["corpus_release"] == RELEASE
    assert [(item["citation"], item["status"]) for item in state["items"]] == [
        ("us-or/manual/odhs/open/page-1", "pending"),
        (encoded, "done"),
        ("us-or/manual/odhs/open/page-10", "pending"),
    ]
    assert state["build"]["excluded"]["not_snap"] == 1


def test_build_refuses_sources_that_are_not_current(tmp_path):
    corpus, rulespec = _fixture_checkouts(tmp_path, queue_status="source_refetch")

    with pytest.raises(ValueError, match="not published_current"):
        q.build_queue(corpus, rulespec, queue_id="x", jurisdictions=["us-or"])


def test_build_refuses_scopes_outside_the_pinned_release(tmp_path):
    corpus, rulespec = _fixture_checkouts(tmp_path, release_has_scope=False)

    with pytest.raises(ValueError, match="is not in corpus release"):
        q.build_queue(corpus, rulespec, queue_id="x", jurisdictions=["us-or"])


def test_build_refuses_a_corpus_checkout_that_is_not_the_pin(tmp_path):
    corpus, rulespec = _fixture_checkouts(tmp_path)
    (rulespec / ".axiom/workflow-toolchain.toml").write_text(_workflow_toolchain())

    with pytest.raises(ValueError, match="but rulespec-us pins"):
        q.build_queue(corpus, rulespec, queue_id="x", jurisdictions=["us-or"])


def test_build_command_never_overwrites_a_queue(tmp_path):
    corpus, rulespec = _fixture_checkouts(tmp_path)
    args = [
        "build",
        "--corpus",
        str(corpus),
        "--rulespec",
        str(rulespec),
        "--state-dir",
        str(tmp_path / "state"),
        "--queue-id",
        "us-snap-or-pilot",
        "--jurisdictions",
        "us-or",
        "--max-unit-chars",
        "1000",
    ]

    assert q.main(args) == 0
    assert q.main(args) == 1
    saved = q.load_state(tmp_path / "state/queues/us-snap-or-pilot.json")
    assert len(saved["items"]) == 3


# -- validation -------------------------------------------------------------


def test_state_rejects_an_item_inside_another_item():
    state = _state(["us-or/a/1", "us-or/a/1-b", "us-or/a/1/z"])

    with pytest.raises(ValueError, match="is inside queue item us-or/a/1"):
        q.validate_state(state)


def test_state_rejects_unknown_statuses_and_duplicates():
    state = _state(["us-or/a"])
    state["items"][0]["status"] = "completed"
    with pytest.raises(ValueError, match="unknown status"):
        q.validate_state(state)

    state = _state(["us-or/a", "us-or/a"])
    with pytest.raises(ValueError, match="duplicate"):
        q.validate_state(state)


# -- ticking ----------------------------------------------------------------


def test_a_paused_queue_reconciles_but_never_dispatches():
    github = FakeGitHub()
    state = _state(["us-or/p/1"], active=False)

    result = _tick(state, github)

    assert github.dispatches == []
    assert result["hold"] == "queue is paused"


def test_active_queue_dispatches_up_to_max_in_flight_with_main_pins():
    github = FakeGitHub()
    state = _state([f"us-or/p/{n}" for n in range(1, 7)])

    result = _tick(state, github)

    assert len(github.dispatches) == 4
    assert github.dispatches[0] == {
        "ref": "main",
        "inputs": {
            "citation": "us-or/p/1",
            "country": "us",
            "rulespec_ref": TIP,
            "pr_base_branch": "main",
            "corpus_ref": CORPUS_REF,
            "rules_engine_ref": ENGINE_REF,
            "open_pr": "true",
        },
    }
    assert [item["status"] for item in state["items"]] == [
        "dispatched",
        "dispatched",
        "dispatched",
        "dispatched",
        "pending",
        "pending",
    ]
    assert all(_run_id(item) for item in state["items"][:4])
    assert [w["citation"] for w in result["waiting_for_approval"]] == []

    second = _tick(state, github)

    assert len(github.dispatches) == 4
    assert len(second["waiting_for_approval"]) == 4


def test_runs_are_found_by_title_when_dispatch_returns_no_id():
    github = FakeGitHub()
    github.return_run_id = False
    state = _state(["us-or/p/1"])

    _tick(state, github)

    assert _run_id(state["items"][0]) == 1000


def test_a_shortened_run_title_still_matches_one_run():
    github = FakeGitHub()
    github.return_run_id = False
    citation = "us-ut/manual/" + "x" * 150
    state = _state([citation])
    original_post = github.post

    def post(path, body=None):
        response = original_post(path, body)
        if body and "inputs" in body:
            run = github.runs[github.next_id - 1]
            run["display_title"] = run["display_title"][:150] + "…"
        return response

    github.post = post
    _tick(state, github)

    assert _run_id(state["items"][0]) == 1000


def test_success_with_an_open_pull_request_waits_for_review_then_merges():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    _tick(state, github)
    item = state["items"][0]
    github.finish(_run_id(item), "success")
    pull = github.add_pull(_run_id(item))

    _tick(state, github)
    assert item["status"] == "in_review"
    assert item["pr"]["url"] == pull["html_url"]

    pull["merged_at"] = "2026-09-30T14:00:00Z"
    pull["state"] = "closed"
    result = _tick(state, github)
    assert item["status"] == "done"
    assert any(event.startswith("done us-or/p/1") for event in result["events"])


def test_a_closed_unmerged_pull_request_blocks_the_item():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    _tick(state, github)
    item = state["items"][0]
    github.finish(_run_id(item), "success")
    github.add_pull(_run_id(item), state="closed")

    _tick(state, github)

    assert item["status"] == "blocked"
    assert "closed without merging" in item["note"]


def test_success_without_a_pull_request_is_blocked_for_a_person():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    _tick(state, github)
    github.finish(_run_id(state["items"][0]), "success")

    _tick(state, github)

    assert state["items"][0]["status"] == "blocked"
    assert "no RuleSpec pull request" in state["items"][0]["note"]


def test_failures_retry_once_then_block_with_the_failing_step():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    item = state["items"][0]
    failing_job = [
        {
            "name": "Queue protected signed RuleSpec re-encode",
            "conclusion": "failure",
            "steps": [
                {"name": "Checkout", "conclusion": "success"},
                {
                    "name": "Encode, review, validate, and apply",
                    "conclusion": "failure",
                },
            ],
        }
    ]

    _tick(state, github)
    github.finish(_run_id(item), "failure")
    github.jobs[_run_id(item)] = failing_job
    _tick(state, github)

    assert item["status"] == "dispatched"  # retried in the same tick
    assert len(github.dispatches) == 2
    github.finish(_run_id(item), "failure")
    github.jobs[_run_id(item)] = failing_job
    _tick(state, github)

    assert item["status"] == "blocked"
    assert item["note"].startswith("failed 2 times")
    assert "Encode, review, validate, and apply" in item["note"]
    assert len(github.dispatches) == 2


def test_an_exhausted_workflow_budget_blocks_without_counting():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    item = state["items"][0]
    _tick(state, github)
    github.finish(_run_id(item), "failure")
    github.jobs[_run_id(item)] = [
        {"name": q.BUDGET_JOB, "conclusion": "failure", "steps": []},
        {"name": "Queue protected signed RuleSpec re-encode", "conclusion": "skipped"},
    ]

    _tick(state, github)

    assert item["status"] == "blocked"
    assert "budget" in item["note"]
    assert q._counted(item) == 0


def test_a_waiting_run_on_a_stale_main_is_cancelled_and_redispatched():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    item = state["items"][0]
    _tick(state, github)
    first = _run_id(item)
    github.tip = "b" * 40

    _tick(state, github)

    assert github.cancelled == [first]
    assert item["attempts"][0]["result"] == "stale-cancelled"
    assert item["status"] == "dispatched"
    assert github.dispatches[-1]["inputs"]["rulespec_ref"] == "b" * 40
    assert q._counted(item) == 0
    assert q._cancellations(item) == 0


def test_repeated_cancellations_by_someone_else_block_the_item():
    github = FakeGitHub()
    state = _state(["us-or/p/1"])
    item = state["items"][0]
    _tick(state, github)
    for _ in range(q.MAX_CANCELLATIONS):
        github.finish(_run_id(item), "cancelled")
        _tick(state, github)

    assert item["status"] == "blocked"
    assert len(github.dispatches) == q.MAX_CANCELLATIONS


def test_a_dispatch_that_never_produces_a_run_is_requeued_after_a_while():
    github = FakeGitHub()
    github.return_run_id = False
    state = _state(["us-or/p/1"])
    item = state["items"][0]
    _tick(state, github)
    github.runs.clear()
    item["attempts"][0]["run_id"] = None

    _tick(state, github, now=NOW + timedelta(minutes=5))
    assert item["status"] == "dispatched"

    _tick(state, github, now=NOW + q.LOST_RUN_AFTER + timedelta(minutes=1))
    assert item["attempts"][0]["result"] == "lost"
    assert len(github.dispatches) == 2


def test_a_moved_corpus_release_holds_dispatch():
    github = FakeGitHub()
    github.release = "us-rulespec-2026-10-01-next"
    state = _state(["us-or/p/1"])

    result = _tick(state, github)

    assert github.dispatches == []
    assert "rebuild it" in result["hold"]


def test_a_failed_dispatch_keeps_what_was_already_sent():
    github = FakeGitHub()
    github.fail_dispatch_at = 2
    state = _state([f"us-or/p/{n}" for n in range(1, 5)])

    result = _tick(state, github)

    assert [item["status"] for item in state["items"]] == [
        "dispatched",
        "dispatched",
        "pending",
        "pending",
    ]
    assert "dispatching us-or/p/3 failed" in result["error"]


def test_one_unreadable_run_does_not_stop_the_tick():
    github = FakeGitHub()
    state = _state(["us-or/p/1", "us-or/p/2"])
    state["settings"]["max_in_flight"] = 1
    _tick(state, github)
    github.fail_get.add(f"/actions/runs/{_run_id(state['items'][0])}")

    result = _tick(state, github)

    assert any(event.startswith("could not check") for event in result["events"])
    assert state["items"][0]["status"] == "dispatched"


def test_a_quiet_tick_leaves_the_state_untouched():
    github = FakeGitHub()
    state = _state(["us-or/p/1"], active=False)
    before = copy.deepcopy(state)

    _tick(state, github)

    assert state == before


def test_dry_run_changes_nothing():
    github = FakeGitHub()
    state = _state(["us-or/p/1", "us-or/p/2"])
    before = copy.deepcopy(state)

    result = _tick(state, github, dry_run=True)

    assert github.dispatches == []
    assert state == before
    assert result["dispatched"] == ["us-or/p/1", "us-or/p/2"]


def test_tick_command_saves_state_and_fails_on_a_dispatch_error(tmp_path, monkeypatch):
    github = FakeGitHub()
    github.fail_dispatch_at = 1
    path = tmp_path / "queues" / "us-snap-test.json"
    q.write_state(path, _state(["us-or/p/1", "us-or/p/2"]))
    monkeypatch.setattr(q, "ApiGitHub", lambda _token: github)
    monkeypatch.setattr(q, "_now", lambda: NOW)
    monkeypatch.setattr(q.time, "sleep", lambda _seconds: None)
    monkeypatch.setenv("GH_TOKEN", "token")
    monkeypatch.setenv("GITHUB_REPOSITORY", ENCODER_REPO)
    summary = tmp_path / "summary.md"

    code = q.main(["tick", "--state-dir", str(tmp_path), "--summary", str(summary)])

    assert code == 1
    saved = q.load_state(path)
    assert [item["status"] for item in saved["items"]] == ["dispatched", "pending"]
    assert "dispatching us-or/p/2 failed" in summary.read_text()


def test_set_state_activates_and_pauses(tmp_path):
    path = tmp_path / "queues" / "us-snap-test.json"
    q.write_state(path, _state(["us-or/p/1"], active=False))

    assert (
        q.main(
            [
                "set-state",
                "--state-dir",
                str(tmp_path),
                "--queue-id",
                "us-snap-test",
                "active",
            ]
        )
        == 0
    )
    assert q.load_state(path)["state"] == "active"
    assert (
        q.main(
            [
                "set-state",
                "--state-dir",
                str(tmp_path),
                "--queue-id",
                "us-snap-test",
                "paused",
            ]
        )
        == 0
    )
    assert q.load_state(path)["state"] == "paused"


def test_summary_lists_counts_waiting_runs_and_changes():
    summary = q.render_summary(
        [
            {
                "queue_id": "us-snap-test",
                "state": "active",
                "rulespec_ref": TIP,
                "hold": None,
                "error": None,
                "dry_run": False,
                "dispatched": ["us-or/p/1"],
                "waiting_for_approval": [{"citation": "us-or/p/0", "url": "u"}],
                "events": ["done us-or/p/9: pr"],
                "counts": {
                    "pending": 3,
                    "dispatched": 1,
                    "in_review": 0,
                    "done": 1,
                    "blocked": 0,
                },
            }
        ]
    )

    assert "| 3 | 1 | 0 | 1 | 0 |" in summary
    assert "- `us-or/p/1`" in summary
    assert "- [us-or/p/0](u)" in summary
    assert "- done us-or/p/9: pr" in summary


# -- workflow ---------------------------------------------------------------


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_runs_hourly_on_main_one_tick_at_a_time():
    workflow = _workflow()
    triggers = workflow[True]  # PyYAML reads the bare `on` key as True

    assert triggers["schedule"] == [{"cron": "17 * * * *"}]
    assert workflow["concurrency"] == {
        "group": "snap-dispatch-queue",
        "cancel-in-progress": False,
    }
    assert workflow["permissions"] == {"actions": "write", "contents": "write"}
    assert workflow["jobs"]["queue"]["if"] == "github.ref == 'refs/heads/main'"


def test_workflow_passes_inputs_through_env_and_pins_actions():
    workflow = _workflow()
    steps = workflow["jobs"]["queue"]["steps"]

    for step in steps:
        assert "${{ inputs." not in step.get("run", "")
        uses = step.get("uses")
        if uses:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", uses), uses


def test_workflow_saves_state_even_when_the_tick_fails():
    steps = {step["name"]: step for step in _workflow()["jobs"]["queue"]["steps"]}

    assert steps["Save queue state"]["if"].startswith("${{ always()")

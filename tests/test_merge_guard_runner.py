"""The merge guard runner: shadow only, fail closed, inert until enabled.

API payloads are fixtures. ``open_pulls_page1.json`` is a recorded public
response (reduced); the evaluator's reads come from the ``World`` fake of
``test_content_merge_guard``. No test makes a network call.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

from axiom_encode import cli
from axiom_encode import content_merge_guard as guard
from axiom_encode import merge_guard_runner as runner
from tests.test_content_merge_guard import (
    COMPANION,
    NAME,
    PRIMARY,
    ROOT,
    B,
    H,
    P,
    T,
    build,
    commit,
    plain,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "merge-guard-runner.yml"
RECORDED = json.loads(
    (
        REPOSITORY_ROOT
        / "tests"
        / "fixtures"
        / "merge_guard_runner"
        / "open_pulls_page1.json"
    ).read_text("utf-8")
)
TOKEN = "ghs_" + "x" * 36
PULLS = ROOT + "/pulls?state=open&base=main&per_page=100&page="
SHADOW_LISTING = (
    f"{ROOT}/commits/{H}/check-runs"
    "?check_name=content-bound-merge-guard%20%28shadow%29&filter=latest&per_page=100"
)


def pulls(numbers):
    return [{"number": number, "state": "open"} for number in numbers]


class Writer:
    """A fake of the check-run POST that answers as GitHub does."""

    def __init__(self, **changes):
        self.calls = []
        self.changes = changes

    def __call__(self, path, body):
        self.calls.append((path, body))
        return {
            "id": 9001,
            "name": body["name"],
            "head_sha": body["head_sha"],
            "conclusion": body["conclusion"],
        } | self.changes


def shadow_world(world, existing=()):
    world.api[SHADOW_LISTING] = {
        "total_count": len(existing),
        "check_runs": list(existing),
    }
    return world


# --- names and modes ---------------------------------------------------------


def test_shadow_check_is_not_the_enforcing_check_and_is_always_neutral():
    assert runner.SHADOW_CHECK_NAME != runner.GUARD_CHECK_NAME
    assert runner.SHADOW_CONCLUSION == "neutral"
    assert runner.MODES == ("dry-run", "shadow")
    for verdict in (
        guard.Verdict(guard.PASS, guard.R_PASS),
        guard.Verdict(guard.FAIL, guard.R_NO_LANE),
        guard.Verdict(guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT),
    ):
        body = runner.shadow_check_body(verdict, H)
        assert body["name"] == runner.SHADOW_CHECK_NAME
        assert body["conclusion"] == "neutral"
        assert body["status"] == "completed"
        assert body["head_sha"] == H
        assert verdict.verdict in body["output"]["title"]
        assert verdict.reason in body["output"]["title"]
        assert "not a required check" in body["output"]["summary"]


def test_the_enforcing_check_name_is_never_written():
    source = (REPOSITORY_ROOT / "src/axiom_encode/merge_guard_runner.py").read_text()
    # One definition and no use: the name is reserved, not published.
    assert len(re.findall(r"\bGUARD_CHECK_NAME\b", source)) == 2
    assert "GUARD_CHECK_NAME: Final" in source


def test_shadow_body_refuses_a_missing_head_and_bounds_the_summary():
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "d" * 20000)
    with pytest.raises(runner.RunnerError):
        runner.shadow_check_body(verdict, "not-a-sha")
    with pytest.raises(runner.RunnerError):
        runner.shadow_check_body(verdict, None)
    summary = runner.shadow_check_body(verdict, H)["output"]["summary"]
    assert len(summary) < runner.MAX_SUMMARY_CHARS + 500
    assert "truncated" in summary


# --- the work list: pagination, truncation, errors ---------------------------


def test_recorded_open_pull_request_list_is_read_whole():
    page = RECORDED["page"]
    assert 0 < len(page) < 100
    api = {PULLS + "1": page}
    numbers = runner.open_pull_requests(NAME, api.__getitem__)
    assert numbers == [item["number"] for item in page]
    assert len(set(numbers)) == len(numbers)


def test_list_follows_full_pages_until_a_short_page():
    api = {
        PULLS + "1": pulls(range(1, 101)),
        PULLS + "2": pulls(range(101, 201)),
        PULLS + "3": pulls([201, 202, 203]),
    }
    assert runner.open_pull_requests(NAME, api.__getitem__) == list(range(1, 204))


def test_a_full_last_page_needs_an_empty_page_to_end():
    api = {PULLS + "1": pulls(range(1, 101)), PULLS + "2": []}
    assert runner.open_pull_requests(NAME, api.__getitem__) == list(range(1, 101))
    # Without the next page the list is not known to be complete.
    del api[PULLS + "2"]
    with pytest.raises(KeyError):
        runner.open_pull_requests(NAME, api.__getitem__)


def test_list_refuses_the_page_bound():
    api = {
        PULLS + str(page): pulls(range(page * 100, page * 100 + 100))
        for page in range(1, runner.MAX_PAGES + 1)
    }
    with pytest.raises(runner.RunnerError, match="page bound"):
        runner.open_pull_requests(NAME, api.__getitem__)


@pytest.mark.parametrize(
    "page",
    [
        {"message": "API rate limit exceeded"},
        None,
        "[]",
        pulls(range(1, 102)),
        [{"state": "open"}],
        [{"number": "7"}],
        [{"number": True}],
        [{"number": 0}],
        [7],
        pulls([7, 7]),
    ],
)
def test_list_refuses_an_invalid_page(page):
    with pytest.raises(runner.RunnerError):
        runner.open_pull_requests(NAME, {PULLS + "1": page}.__getitem__)


def test_list_refuses_a_number_repeated_on_a_later_page():
    api = {PULLS + "1": pulls(range(1, 101)), PULLS + "2": pulls([50])}
    with pytest.raises(runner.RunnerError):
        runner.open_pull_requests(NAME, api.__getitem__)


@pytest.mark.parametrize("name", ["axiom-encode", "rulespec-US", "", None, "../x"])
def test_list_refuses_a_repository_outside_the_rulespec_family(name):
    with pytest.raises(runner.RunnerError):
        runner.open_pull_requests(name, {}.__getitem__)


def test_sweep_whose_list_fails_is_a_run_error_not_an_empty_sweep():
    def fetch(path):
        raise runner.RunnerError("GitHub GET request failed: HTTPError")

    report, code = runner.run(NAME, None, runner.DRY_RUN, fetch=fetch)
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False
    assert report["results"] == []
    assert "HTTPError" in report["error"]


def test_sweep_of_an_empty_list_is_complete():
    report, code = runner.run(
        NAME, None, runner.DRY_RUN, fetch={PULLS + "1": []}.__getitem__
    )
    assert (code, report["complete"], report["results"]) == (0, True, [])
    assert report["sweep"] is True


# --- evaluation and reporting -------------------------------------------------


def test_dry_run_evaluates_and_writes_nothing():
    world = build()
    writer = Writer()
    report, code = runner.run(
        NAME, [42], runner.DRY_RUN, fetch=world.fetch, post=writer
    )
    assert code == 0 and report["complete"] is True
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == "pass"
    assert entry["publication"] == "none"
    assert writer.calls == []
    assert report["check_name"] is None
    assert not any("check-runs?check_name=content-bound" in p for p in world.calls)


def test_sweep_evaluates_each_listed_pull_request():
    world = build()
    world.api[PULLS + "1"] = pulls([42, 43])
    report, code = runner.run(NAME, None, runner.DRY_RUN, fetch=world.fetch)
    # 43 is not readable in the fake: an unreadable pull request is a run
    # error, and its verdict is fail, never "skipped".
    first, second = report["results"]
    assert first["verdict"]["verdict"] == "pass"
    assert second["verdict"]["verdict"] == "fail"
    assert second["verdict"]["reason"] == guard.R_READ_FAILED
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False


@pytest.mark.parametrize(
    ("world", "verdict"),
    [
        (lambda: build(), "pass"),
        (lambda: plain({"README.md": b"x\n"}), "not_applicable"),
        (lambda: plain({".github/workflows/x.yml": b"x\n"}), "fail"),
    ],
)
def test_shadow_writes_one_neutral_check_for_every_kind_of_verdict(world, verdict):
    world = shadow_world(world())
    writer = Writer()
    report, code = runner.run(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    assert code == 0 and report["complete"] is True
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == verdict
    assert entry["publication"] == "created"
    assert entry["check_run_id"] == 9001
    ((path, body),) = writer.calls
    assert path == ROOT + "/check-runs"
    assert body["name"] == runner.SHADOW_CHECK_NAME
    assert body["conclusion"] == "neutral"
    assert body["head_sha"] == H
    assert report["check_name"] == runner.SHADOW_CHECK_NAME


def test_truncated_tree_fails_closed_and_is_reported_as_fail():
    world = plain({"README.md": b"x\n"})
    world.tree(H, {"README.md": b"x\n"}, truncated=True)
    world = shadow_world(world)
    writer = Writer()
    report, _code = runner.run(
        NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer
    )
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == "fail"
    ((_path, body),) = writer.calls
    assert body["conclusion"] == "neutral"
    assert "fail" in body["output"]["title"]


def test_truncated_file_list_fails_closed():
    # The PR file list omits a file of the tree delta (a truncated listing).
    world = build()
    world.api[ROOT + "/pulls/42/files?per_page=100&page=1"] = world.api[
        ROOT + "/pulls/42/files?per_page=100&page=1"
    ][:-1]
    report, _code = runner.run(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert report["results"][0]["verdict"]["verdict"] == "fail"


def test_shadow_does_not_repeat_an_unchanged_verdict():
    world = plain({"README.md": b"x\n"})
    digest = runner.shadow_check_body(world.evaluate(), H)["external_id"]
    older = {"id": 5, "name": runner.SHADOW_CHECK_NAME, "external_id": "other"}
    newest = {
        "id": 6,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": digest,
        "status": "completed",
    }
    writer = Writer()
    report, code = runner.run(
        NAME,
        [42],
        runner.SHADOW,
        fetch=shadow_world(world, [newest, older]).fetch,
        post=writer,
    )
    assert code == 0
    assert report["results"][0]["publication"] == "unchanged"
    assert writer.calls == []


def test_shadow_reports_again_when_the_newest_check_differs():
    world = plain({"README.md": b"x\n"})
    digest = runner.shadow_check_body(world.evaluate(), H)["external_id"]
    same_but_older = {
        "id": 5,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": digest,
        "status": "completed",
    }
    newest = {"id": 6, "name": runner.SHADOW_CHECK_NAME, "external_id": "other"}
    other_name = {"id": 7, "name": "validate / validate", "external_id": digest}
    writer = Writer()
    report, _code = runner.run(
        NAME,
        [42],
        runner.SHADOW,
        fetch=shadow_world(world, [same_but_older, newest, other_name]).fetch,
        post=writer,
    )
    assert report["results"][0]["publication"] == "created"
    assert len(writer.calls) == 1


@pytest.mark.parametrize(
    "listing",
    [
        {"total_count": 2, "check_runs": []},
        {"total_count": "0", "check_runs": []},
        {"check_runs": []},
        {"total_count": 1, "check_runs": [{"name": "x"}]},
        {"total_count": 1, "check_runs": ["x"]},
        [],
        {"message": "Not Found"},
    ],
)
def test_incomplete_check_listing_is_a_publication_error(listing):
    world = plain({"README.md": b"x\n"})
    world.api[SHADOW_LISTING] = listing
    writer = Writer()
    report, code = runner.run(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False
    assert report["results"][0]["publication"] == "error"
    assert writer.calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"id": None},
        {"head_sha": B},
        {"name": runner.GUARD_CHECK_NAME},
        {"conclusion": "success"},
    ],
)
def test_unconfirmed_check_run_is_a_publication_error(changes):
    world = shadow_world(plain({"README.md": b"x\n"}))
    report, code = runner.run(
        NAME, [42], runner.SHADOW, fetch=world.fetch, post=Writer(**changes)
    )
    assert code == runner.EXIT_RUN_ERROR
    assert report["results"][0]["publication"] == "error"


def test_write_failure_is_a_publication_error():
    world = shadow_world(plain({"README.md": b"x\n"}))

    def post(path, body):
        raise runner.RunnerError("GitHub POST request failed: HTTPError")

    report, code = runner.run(NAME, [42], runner.SHADOW, fetch=world.fetch, post=post)
    assert code == runner.EXIT_RUN_ERROR
    assert report["results"][0]["publication_error"].endswith("HTTPError")


def test_evaluation_without_a_head_publishes_nothing():
    world = plain({"README.md": b"x\n"})
    world.api[ROOT + "/pulls/42"]["state"] = "closed"
    writer = Writer()
    report, code = runner.run(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    (entry,) = report["results"]
    assert entry["verdict"]["reason"] == guard.R_PR_IDENTITY
    assert entry["publication"] == "no_head"
    assert writer.calls == []
    # A closed pull request is a result, not a failed run.
    assert code == 0


def test_a_read_error_inside_an_evaluation_is_a_run_error():
    world = shadow_world(plain({"README.md": b"x\n"}))
    del world.api[f"{ROOT}/git/trees/{H}?recursive=1"]
    writer = Writer()
    report, code = runner.run(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == "fail"
    assert entry["verdict"]["reason"] == guard.R_READ_FAILED
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False
    # The fail verdict is still shown on the head that was read.
    assert entry["publication"] == "created"


@pytest.mark.parametrize(
    ("numbers", "mode", "post"),
    [
        ([42], "enforce", Writer()),
        ([42], "", Writer()),
        ([42], runner.SHADOW, None),
        ([], runner.DRY_RUN, None),
        ([0], runner.DRY_RUN, None),
        ([True], runner.DRY_RUN, None),
        (["42"], runner.DRY_RUN, None),
    ],
)
def test_run_refuses_unknown_modes_and_bad_requests(numbers, mode, post):
    world = build()
    report, code = runner.run(NAME, numbers, mode, fetch=world.fetch, post=post)
    assert code == runner.EXIT_RUN_ERROR
    assert report["results"] == [] and report["complete"] is False
    assert world.calls == []


def test_report_one_refuses_a_mode_that_is_not_dry_run_or_shadow():
    with pytest.raises(runner.RunnerError):
        runner.report_one(NAME, 42, "enforce", fetch=build().fetch, post=Writer())


# --- the HTTP layer -----------------------------------------------------------


class Response:
    def __init__(self, status, raw):
        self.status = status
        self._raw = raw

    def read(self, limit):
        return self._raw[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


class Opener:
    def __init__(self, status=200, raw=b"{}", error=None):
        self.requests = []
        self.status, self.raw, self.error = status, raw, error

    def open(self, request, timeout):
        self.requests.append(request)
        assert timeout == 30
        if self.error is not None:
            raise self.error
        return Response(self.status, self.raw)


def test_reader_sends_the_token_to_the_fixed_origin_only():
    opener = Opener(raw=b'{"ok": true}')
    fetch = runner.token_reader(TOKEN, opener=opener)
    assert fetch(ROOT + "/pulls/42") == {"ok": True}
    (request,) = opener.requests
    assert request.full_url == "https://api.github.com" + ROOT + "/pulls/42"
    assert request.get_method() == "GET"
    assert request.get_header("Authorization") == "Bearer " + TOKEN
    assert request.data is None


@pytest.mark.parametrize(
    "path",
    [
        "/repos/other-org/rulespec-us/pulls/1",
        "/user",
        "https://evil.example/repos/TheAxiomFoundation/x",
        ROOT + "/pulls/1#x",
        ROOT + "/../../other/pulls/1",
        ROOT + "/pulls/1\n",
        ROOT + "/pulls/1 x",
        None,
    ],
)
def test_reader_refuses_paths_outside_the_organisation(path):
    opener = Opener()
    with pytest.raises(runner.RunnerError):
        runner.token_reader(TOKEN, opener=opener)(path)
    assert opener.requests == []


@pytest.mark.parametrize("token", ["", "short", "has space " + "x" * 30, None])
def test_reader_refuses_an_absent_or_malformed_token(token):
    opener = Opener()
    with pytest.raises(runner.RunnerError, match=runner.TOKEN_ENV):
        runner.token_reader(token, opener=opener)(ROOT + "/pulls/1")
    assert opener.requests == []


def http_error(code):
    return urllib.error.HTTPError(
        "https://api.github.com/x", code, f"secret {TOKEN}", {}, io.BytesIO(b"")
    )


@pytest.mark.parametrize(
    "opener",
    [
        Opener(error=http_error(403)),
        Opener(error=http_error(404)),
        Opener(error=http_error(500)),
        Opener(error=urllib.error.URLError("unreachable")),
        Opener(error=TimeoutError()),
        Opener(status=204, raw=b""),
        Opener(status=301, raw=b"{}"),
        Opener(raw=b"not json"),
        Opener(raw=b""),
    ],
)
def test_reader_refuses_every_failed_read_without_echoing_the_token(opener):
    with pytest.raises(runner.RunnerError) as caught:
        runner.token_reader(TOKEN, opener=opener)(ROOT + "/pulls/1")
    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None


def test_reader_refuses_a_response_above_the_bound(monkeypatch):
    monkeypatch.setattr(runner, "MAX_RESPONSE_BYTES", 8)
    with pytest.raises(runner.RunnerError, match="bound"):
        runner.token_reader(TOKEN, opener=Opener(raw=b'{"a": "bcdefgh"}'))(
            ROOT + "/pulls/1"
        )


def test_default_opener_refuses_redirects_and_ignores_proxies(monkeypatch):
    built = []

    def build_opener(*handlers):
        built.append(handlers)
        return Opener()

    monkeypatch.setattr(runner.urllib.request, "build_opener", build_opener)
    runner.token_reader(TOKEN)(ROOT + "/pulls/1")
    ((proxy, redirect),) = built
    assert isinstance(proxy, runner.urllib.request.ProxyHandler)
    assert proxy.proxies == {}
    assert isinstance(redirect, runner._NoRedirect)


def test_writer_posts_check_runs_only_and_requires_created():
    opener = Opener(status=201, raw=b'{"id": 1}')
    post = runner.check_writer(TOKEN, opener=opener)
    assert post(ROOT + "/check-runs", {"name": "n"}) == {"id": 1}
    (request,) = opener.requests
    assert request.get_method() == "POST"
    assert json.loads(request.data) == {"name": "n"}
    assert request.get_header("Authorization") == "Bearer " + TOKEN
    with pytest.raises(runner.RunnerError):
        runner.check_writer(TOKEN, opener=Opener(status=200))(
            ROOT + "/check-runs", {"name": "n"}
        )


@pytest.mark.parametrize(
    "path",
    [
        ROOT + "/pulls/42/reviews",
        ROOT + "/check-runs/1",
        ROOT + "/statuses/" + "a" * 40,
        ROOT + "/branches/main/protection",
        "/repos/TheAxiomFoundation/axiom-encode/check-runs",
        "/repos/other-org/rulespec-us/check-runs",
        ROOT + "/check-runs?x=1",
    ],
)
def test_writer_refuses_every_other_write(path):
    opener = Opener(status=201)
    with pytest.raises(runner.RunnerError):
        runner.check_writer(TOKEN, opener=opener)(path, {})
    assert opener.requests == []


# --- the command --------------------------------------------------------------


def arguments(**changes):
    values = {"repository": NAME, "pr": 42, "sweep": False, "mode": "dry-run"}
    return argparse.Namespace(**(values | changes))


class Exploding:
    def open(self, request, timeout):
        raise AssertionError("no request may be made")


@pytest.mark.parametrize(
    "environ", [{}, {runner.TOKEN_ENV: ""}, {runner.TOKEN_ENV: "x"}]
)
def test_command_fails_closed_without_the_app_token(capsys, environ):
    code = runner.command(arguments(), environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False and report["results"] == []
    assert runner.TOKEN_ENV in report["error"]
    assert "guard GitHub App" in report["error"]


def test_command_prints_one_json_line_and_never_the_token(capsys, monkeypatch):
    world = shadow_world(plain({"README.md": b"x\n"}))
    writer = Writer()
    monkeypatch.setattr(runner, "token_reader", lambda token, opener=None: world.fetch)
    monkeypatch.setattr(runner, "check_writer", lambda token, opener=None: writer)
    code = runner.command(
        arguments(mode="shadow"), environ={runner.TOKEN_ENV: TOKEN}, opener=None
    )
    out = capsys.readouterr()
    assert code == 0
    assert out.out.count("\n") == 1
    assert TOKEN not in out.out and TOKEN not in out.err
    report = json.loads(out.out)
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["results"][0]["publication"] == "created"


def test_command_dry_run_builds_no_writer(capsys, monkeypatch):
    world = plain({"README.md": b"x\n"})
    monkeypatch.setattr(runner, "token_reader", lambda token, opener=None: world.fetch)

    def forbidden(token, opener=None):
        raise AssertionError("dry-run must not build a writer")

    monkeypatch.setattr(runner, "check_writer", forbidden)
    assert runner.command(arguments(), environ={runner.TOKEN_ENV: TOKEN}) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["publication"] == "none"


def test_command_sweep_passes_no_numbers(capsys, monkeypatch):
    api = {PULLS + "1": []}
    monkeypatch.setattr(
        runner, "token_reader", lambda token, opener=None: api.__getitem__
    )
    code = runner.command(
        arguments(pr=None, sweep=True), environ={runner.TOKEN_ENV: TOKEN}
    )
    report = json.loads(capsys.readouterr().out)
    assert (code, report["sweep"], report["complete"]) == (0, True, True)


def test_cli_registers_the_command_and_takes_no_token_argument(monkeypatch, capsys):
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["axiom-encode", "merge-guard-run", "--repository", NAME, "--sweep"],
    )
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == runner.EXIT_RUN_ERROR
    assert runner.TOKEN_ENV in json.loads(capsys.readouterr().out)["error"]
    for argv in (
        ["--repository", NAME],  # neither --pr nor --sweep
        ["--repository", NAME, "--pr", "1", "--sweep"],
        ["--repository", NAME, "--pr", "1", "--mode", "enforce"],
        ["--repository", NAME, "--pr", "1", "--token", TOKEN],
    ):
        monkeypatch.setattr(sys, "argv", ["axiom-encode", "merge-guard-run", *argv])
        with pytest.raises(SystemExit) as stop:
            cli.main()
        assert stop.value.code == 2
        capsys.readouterr()


# --- the workflow: inert until enabled ----------------------------------------


def workflow():
    return yaml.safe_load(WORKFLOW_PATH.read_text("utf-8"))


def test_workflow_has_no_trigger_but_a_manual_dispatch():
    document = workflow()
    # PyYAML reads the key ``on`` as the boolean True.
    triggers = document[True]
    assert set(triggers) == {"workflow_dispatch"}
    assert set(triggers["workflow_dispatch"]["inputs"]) == {"country", "pr", "mode"}
    mode = triggers["workflow_dispatch"]["inputs"]["mode"]
    assert tuple(mode["options"]) == runner.MODES
    assert mode["default"] == runner.DRY_RUN
    assert document["permissions"] == {"contents": "read"}


def test_workflow_gate_refuses_before_any_credential_is_in_reach():
    jobs = workflow()["jobs"]
    assert list(jobs) == ["gate", "run"]
    gate, run = jobs["gate"], jobs["run"]
    assert "environment" not in gate
    assert "secrets." not in json.dumps(gate)
    (step,) = gate["steps"]
    assert step["env"]["ENABLED"] == "${{ vars.MERGE_GUARD_RUNNER_ENABLED }}"
    assert step["env"]["REF"] == "${{ github.ref }}"
    script = step["run"]
    assert '[ "$REF" != "refs/heads/main" ]' in script
    assert '[ "$ENABLED" != "true" ]' in script
    assert script.count("exit 1") == 4
    assert run["needs"] == "gate"
    assert "if" not in run  # a failed gate skips it; nothing overrides that
    assert run["environment"] == "merge-guard"


def test_workflow_reads_only_the_named_app_credentials_and_fails_closed():
    text = WORKFLOW_PATH.read_text("utf-8")
    assert set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text)) == {"MERGE_GUARD_APP_KEY"}
    assert set(re.findall(r"vars\.([A-Za-z0-9_]+)", text)) == {
        "MERGE_GUARD_RUNNER_ENABLED",
        "MERGE_GUARD_APP_CLIENT_ID",
    }
    steps = workflow()["jobs"]["run"]["steps"]
    require = steps[0]
    assert require["name"] == "Require the guard App"
    assert "exit 1" in require["run"] and "::error::" in require["run"]
    token = next(step for step in steps if step.get("id") == "app-token")
    assert steps.index(token) > 0
    assert token["with"] == {
        "client-id": "${{ vars.MERGE_GUARD_APP_CLIENT_ID }}",
        "private-key": "${{ secrets.MERGE_GUARD_APP_KEY }}",
        "owner": "TheAxiomFoundation",
        "repositories": "rulespec-${{ inputs.country }}",
        "permission-checks": "${{ inputs.mode == 'shadow' && 'write' || 'read' }}",
        "permission-pull-requests": "read",
        "permission-contents": "read",
        "permission-actions": "read",
    }
    evaluate = next(step for step in steps if step.get("name") == "Evaluate and report")
    assert evaluate["env"][runner.TOKEN_ENV] == "${{ steps.app-token.outputs.token }}"
    assert "merge-guard-run" in evaluate["run"]
    # Inputs reach the shell through the environment, never by interpolation.
    for step in [*workflow()["jobs"]["gate"]["steps"], *steps]:
        assert "${{" not in step.get("run", "")


def test_workflow_runs_the_dispatched_main_commit_with_pinned_actions():
    steps = workflow()["jobs"]["run"]["steps"]
    checkout = next(step for step in steps if "checkout" in step.get("uses", ""))
    assert checkout["with"] == {
        "ref": "${{ github.sha }}",
        "persist-credentials": False,
    }
    for step in steps:
        if "uses" in step:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", step["uses"]), step["uses"]


def test_fixture_tree_and_modules_are_what_the_tests_assume():
    # Guards the imports above against a silent change of the shared fake.
    world = plain({P: PRIMARY, T: COMPANION})
    assert world.api[ROOT + "/pulls/42"]["head"]["sha"] == H
    assert commit(H)["sha"] == H

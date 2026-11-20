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
    M,
    P,
    T,
    build,
    commit,
    plain,
    v5_manifest,
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
CLIENT = "Iv23liGuardAppClient"
OUR_APP = {"id": 987654, "slug": "axiom-merge-guard", "client_id": CLIENT}


def sweep(*args, **kwargs):
    """``runner.run`` as the guard App whose client ID is ``CLIENT``."""
    kwargs.setdefault("app_client_id", CLIENT)
    return runner.run(*args, **kwargs)


PULLS = ROOT + "/pulls?state=open&base=main&per_page=100&page="
SHADOW_LISTING_PAGES = (
    f"{ROOT}/commits/{H}/check-runs"
    "?check_name=content-bound-merge-guard%20%28shadow%29"
    "&filter=all&per_page=100&page="
)
SHADOW_LISTING = SHADOW_LISTING_PAGES + "1"


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

    report, code = sweep(NAME, None, runner.DRY_RUN, fetch=fetch)
    assert code == runner.EXIT_RUN_ERROR
    assert report["complete"] is False
    assert report["results"] == []
    assert "HTTPError" in report["error"]


def test_sweep_of_an_empty_list_is_complete():
    report, code = sweep(
        NAME, None, runner.DRY_RUN, fetch={PULLS + "1": []}.__getitem__
    )
    assert (code, report["complete"], report["results"]) == (0, True, [])
    assert report["sweep"] is True


# --- evaluation and reporting -------------------------------------------------


def test_dry_run_evaluates_and_writes_nothing():
    world = build()
    writer = Writer()
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch, post=writer)
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
    report, code = sweep(NAME, None, runner.DRY_RUN, fetch=world.fetch)
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
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
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
    report, _code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
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
    report, _code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert report["results"][0]["verdict"]["verdict"] == "fail"


def test_shadow_does_not_repeat_an_unchanged_verdict():
    world = plain({"README.md": b"x\n"})
    digest = runner.shadow_check_body(world.evaluate(), H)["external_id"]
    older = {
        "id": 5,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": "other",
        "app": OUR_APP,
    }
    newest = {
        "id": 6,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": digest,
        "status": "completed",
        "app": OUR_APP,
    }
    writer = Writer()
    report, code = sweep(
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
        "app": OUR_APP,
    }
    newest = {
        "id": 6,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": "other",
        "app": OUR_APP,
    }
    other_name = {
        "id": 7,
        "name": "validate / validate",
        "external_id": digest,
        "status": "completed",
        "app": OUR_APP,
    }
    writer = Writer()
    report, _code = sweep(
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
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
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
    report, code = sweep(
        NAME, [42], runner.SHADOW, fetch=world.fetch, post=Writer(**changes)
    )
    assert code == runner.EXIT_RUN_ERROR
    assert report["results"][0]["publication"] == "error"


def test_write_failure_is_a_publication_error():
    world = shadow_world(plain({"README.md": b"x\n"}))

    def post(path, body):
        raise runner.RunnerError("GitHub POST request failed: HTTPError")

    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=post)
    assert code == runner.EXIT_RUN_ERROR
    assert report["results"][0]["publication_error"].endswith("HTTPError")


def test_evaluation_without_a_head_publishes_nothing():
    world = plain({"README.md": b"x\n"})
    world.api[ROOT + "/pulls/42"]["state"] = "closed"
    writer = Writer()
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
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
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
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
    report, code = sweep(NAME, numbers, mode, fetch=world.fetch, post=post)
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


def test_reader_sends_the_evaluators_compare_path():
    # ``compare/<base>...<head>`` holds three dots in one segment. It is not a
    # dot segment; a reader that refuses it fails every evaluation.
    opener = Opener(raw=b'{"status": "ahead"}')
    path = f"{ROOT}/compare/{B}...{H}"
    assert runner.token_reader(TOKEN, opener=opener)(path) == {"status": "ahead"}
    assert opener.requests[0].full_url == "https://api.github.com" + path


@pytest.mark.parametrize(
    "path",
    [
        ROOT + "/..",
        ROOT + "/./pulls/1",
        ROOT + "/pulls/..?x=1",
        ROOT + "/pulls/1?x=..",
        ROOT + "/%2e%2e/%2E%2E/other/pulls/1",
        ROOT + "/.%2e/pulls/1",
        ROOT + "/..%2f..%2fother/pulls/1",
        ROOT + "/compare/main...head",
        ROOT + f"/compare/{B}..{H}",
        ROOT + f"/compare/{B}....{H}",
        ROOT + f"/compare/{B}...{H}..",
        ROOT + f"/compare/{B}...{H}?x=..",
        ROOT + f"/compare/..{B}...{H}",
        ROOT + f"/compare/{B.upper()}...{H}",
        # Review round 2, P3-A: the encoded dot in upper and mixed case.
        ROOT + "/.%2E/pulls/1",
        ROOT + "/%2E%2E/pulls/1",
        ROOT + "/%2e%2E/pulls/1",
        ROOT + "/%2E/pulls/1",
        # P3-B: the compare exception is exactly 40 hex, three literal dots,
        # 40 hex. Short ids and a non-dot character in place of a dot are
        # refused.
        ROOT + "/compare/abc...def",
        ROOT + f"/compare/{B[:39]}...{H}",
        ROOT + f"/compare/{B}...{H[:39]}",
        ROOT + f"/compare/{B}x..{H}",
        ROOT + f"/compare/{B}..x{H}",
        # P3-C: a single-dot segment directly before a query or a ``;``
        # parameter is still a dot segment.
        ROOT + "/pulls/.?a",
        ROOT + "/pulls/.?",
        ROOT + "/.?a=1/pulls/1",
        ROOT + "/pulls/.;a/1",
        ROOT + "/pulls/.;",
    ],
)
def test_reader_refuses_dot_segments_plain_or_encoded(path):
    opener = Opener()
    with pytest.raises(runner.RunnerError, match="invalid GitHub API path"):
        runner.token_reader(TOKEN, opener=opener)(path)
    assert opener.requests == []


class ReadWriteOpener:
    """A fake of the HTTP opener that answers from a ``World``.

    A fake ``fetch`` replaces ``token_reader`` and with it the path filter of
    ``_request``. This fake sits one layer lower: every read and the one write
    pass the real ``token_reader``, ``check_writer`` and ``_request``.
    """

    ORIGIN = "https://api.github.com"

    def __init__(self, world):
        self.world = world
        self.paths = []
        self.posts = []

    def open(self, request, timeout):
        assert timeout == 30
        assert request.full_url.startswith(self.ORIGIN + "/")
        assert request.get_header("Authorization") == "Bearer " + TOKEN
        path = request.full_url[len(self.ORIGIN) :]
        if request.get_method() == "POST":
            body = json.loads(request.data)
            self.posts.append((path, body))
            created = {
                "id": 9001,
                "name": body["name"],
                "head_sha": body["head_sha"],
                "conclusion": body["conclusion"],
            }
            return Response(201, json.dumps(created).encode())
        assert request.get_method() == "GET" and request.data is None
        self.paths.append(path)
        try:
            value = self.world.fetch(path)
        except KeyError:
            raise http_error(404) from None
        return Response(200, json.dumps(value).encode())


# One passing v6 evaluation and one evaluation of the earlier (v5) lane: the
# two shapes of a pull request the guard lets through.
WHOLE_EVALUATIONS = [
    (build, ("pass", guard.R_PASS)),
    (
        lambda: plain({P: PRIMARY, T: COMPANION, M: v5_manifest(P, T)}),
        ("not_applicable", guard.R_EARLIER_LANE),
    ),
]


@pytest.mark.parametrize(("make", "expected"), WHOLE_EVALUATIONS)
def test_the_real_reader_carries_a_whole_evaluation(make, expected):
    # The class of fault "the real reader refuses a path the evaluator needs":
    # with a fake ``fetch`` the path filter of ``_request`` never runs. Here
    # the evaluator reads through ``token_reader`` and only the opener is fake.
    reference = make()
    wanted = reference.evaluate()
    assert (wanted.verdict, wanted.reason) == expected
    world = make()
    opener = ReadWriteOpener(world)
    verdict = guard.evaluate(NAME, 42, fetch=runner.token_reader(TOKEN, opener=opener))
    assert verdict.as_dict() == wanted.as_dict()
    # Every path the evaluator asked for reached the opener, in order.
    assert opener.paths == reference.calls
    assert f"{ROOT}/compare/{B}...{H}" in opener.paths
    assert opener.posts == []


@pytest.mark.parametrize(("make", "expected"), WHOLE_EVALUATIONS)
def test_the_command_evaluates_and_reports_through_the_real_reader(
    capsys, make, expected
):
    # The whole command in shadow mode: the evaluator's reads, the listing of
    # earlier shadow checks and the check-run write all pass ``_request``.
    world = shadow_world(make())
    opener = ReadWriteOpener(world)
    code = runner.command(
        arguments(mode="shadow"),
        environ={runner.TOKEN_ENV: TOKEN, runner.APP_CLIENT_ID_ENV: CLIENT},
        opener=opener,
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["complete"] is True
    (entry,) = report["results"]
    assert (entry["verdict"]["verdict"], entry["verdict"]["reason"]) == expected
    assert entry["publication"] == "created"
    assert SHADOW_LISTING in opener.paths
    ((path, body),) = opener.posts
    assert path == ROOT + "/check-runs" and body["head_sha"] == H


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
    body = runner.shadow_check_body(guard.Verdict(guard.FAIL, guard.R_NO_LANE), H)
    assert post(ROOT + "/check-runs", body) == {"id": 1}
    (request,) = opener.requests
    assert request.get_method() == "POST"
    assert json.loads(request.data) == body
    assert request.get_header("Authorization") == "Bearer " + TOKEN
    with pytest.raises(runner.RunnerError):
        runner.check_writer(TOKEN, opener=Opener(status=200))(
            ROOT + "/check-runs", body
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
    with pytest.raises(runner.RunnerError, match="check runs only"):
        # A valid shadow body: the path alone must refuse.
        runner.check_writer(TOKEN, opener=opener)(path, _body())
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


@pytest.mark.parametrize(
    "environ",
    [
        {"GITHUB_TOKEN": TOKEN},
        {"GH_TOKEN": TOKEN},
        {"GITHUB_TOKEN": TOKEN, "GH_TOKEN": TOKEN, runner.TOKEN_ENV: ""},
        {"MERGE_GUARD_APP_KEY": TOKEN, "ACTIONS_RUNTIME_TOKEN": TOKEN},
    ],
)
def test_command_takes_no_token_from_another_variable(capsys, monkeypatch, environ):
    # Review round 1, P3-5: "the App token is read from TOKEN_ENV only". A
    # well-formed token under another name does not start a run. It is set in
    # the injected mapping and in the process environment.
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    for mapping in (environ, None):
        code = runner.command(arguments(), environ=mapping, opener=Exploding())
        out = capsys.readouterr()
        report = json.loads(out.out)
        assert code == runner.EXIT_RUN_ERROR
        assert report["complete"] is False and report["results"] == []
        assert runner.TOKEN_ENV in report["error"]
        assert TOKEN not in out.out and TOKEN not in out.err


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
    # Review round 1, P3-1 and P3-4: the exact keys. No ``continue-on-error``
    # (a refused gate must fail the job, or the run job starts) and no other
    # key may appear on the job or on its one step.
    assert set(gate) == {"name", "if", "runs-on", "timeout-minutes", "steps"}
    assert gate["if"] == "github.repository == 'TheAxiomFoundation/axiom-encode'"
    assert "environment" not in gate
    assert "secrets." not in json.dumps(gate)
    (step,) = gate["steps"]
    assert set(step) == {"name", "env", "run"}
    assert set(run) == {
        "name",
        "needs",
        "runs-on",
        "timeout-minutes",
        "environment",
        "steps",
    }
    for job in (gate, run):
        for any_step in job["steps"]:
            assert "continue-on-error" not in any_step, any_step.get("name")
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
    require = steps[1]
    assert require["name"] == "Require the guard App"
    assert "exit 1" in require["run"] and "::error::" in require["run"]
    # Review round 1, P3-2: where the key is named. Twice in the file: as a
    # boolean in "Require the guard App" (the value never enters that step)
    # and as the input of the token action. No other step names a secret.
    assert text.count("secrets.") == 2
    assert text.count("secrets.MERGE_GUARD_APP_KEY") == 2
    assert require["env"] == {
        "APP_CLIENT_ID": "${{ vars.MERGE_GUARD_APP_CLIENT_ID }}",
        "APP_KEY_SET": "${{ secrets.MERGE_GUARD_APP_KEY != '' }}",
    }
    assert [step.get("name") for step in steps if "secrets." in json.dumps(step)] == [
        "Require the guard App",
        "Create the guard App token",
    ]
    assert "secrets." not in json.dumps(
        {key: value for key, value in workflow().items() if key != "jobs"}
    )
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
    # The job holds the App key: it restores no cache that another run wrote.
    setup_uv = next(step for step in steps if "setup-uv" in step.get("uses", ""))
    assert setup_uv["with"] == {"version": "0.11.7", "enable-cache": "false"}
    assert not any("cache" in step.get("uses", "").lower() for step in steps)


def test_workflow_installs_the_locked_encoder_without_dev_dependencies():
    # Review round 1, P3-3. The job later holds the App key: the install takes
    # the versions of the lock file and no development dependency.
    steps = workflow()["jobs"]["run"]["steps"]
    installs = [step for step in steps if "uv sync" in step.get("run", "")]
    assert installs == [
        {
            "name": "Install the locked encoder",
            "run": "uv sync --no-config --locked --python 3.13 --extra api --no-dev",
        }
    ]
    token = next(step for step in steps if step.get("id") == "app-token")
    assert steps.index(installs[0]) < steps.index(token)
    # No other step installs anything.
    for step in steps:
        if step is not installs[0]:
            assert not re.search(r"\b(uv|pip|npm|apt-get)\b", step.get("run", ""))


def test_fixture_tree_and_modules_are_what_the_tests_assume():
    # Guards the imports above against a silent change of the shared fake.
    world = plain({P: PRIMARY, T: COMPANION})
    assert world.api[ROOT + "/pulls/42"]["head"]["sha"] == H
    assert commit(H)["sha"] == H


# --- independent review round 1 ------------------------------------------------


def _digest_world():
    world = plain({"README.md": b"x\n"})
    return world, runner.shadow_check_body(world.evaluate(), H)["external_id"]


def _check(digest, **changes):
    return {
        "id": 6,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": digest,
        "status": "completed",
        "app": OUR_APP,
    } | changes


def _publication(world, existing, **kwargs):
    writer = Writer()
    report, _code = sweep(
        NAME,
        [42],
        runner.SHADOW,
        fetch=shadow_world(world, existing).fetch,
        post=writer,
        **kwargs,
    )
    return report["results"][0]["publication"], writer.calls


def _detailed_world():
    world = plain({".github/workflows/x.yml": b"x\n"})
    verdict = world.evaluate()
    assert verdict.verdict == "fail" and verdict.detail
    return world, verdict


def test_a_changed_detail_on_the_same_head_is_reported_again():
    # Review round 1, P3-6. Same verdict and reason, another detail: the
    # newest shadow check of ours no longer says what the guard decides.
    world, verdict = _detailed_world()
    earlier = guard.Verdict(
        verdict.verdict, verdict.reason, verdict.detail + " (earlier)", verdict.evidence
    )
    earlier_digest = runner.shadow_check_body(earlier, H)["external_id"]
    assert earlier_digest != runner.shadow_check_body(verdict, H)["external_id"]
    publication, calls = _publication(world, [_check(earlier_digest)])
    assert publication == "created" and len(calls) == 1
    # No detail against an empty detail is a change as well.
    bare = guard.Verdict(verdict.verdict, verdict.reason, "", verdict.evidence)
    publication, calls = _publication(
        world, [_check(runner.shadow_check_body(bare, H)["external_id"])]
    )
    assert publication == "created" and len(calls) == 1


def test_changed_evidence_alone_is_not_reported_again():
    # The documented limit of the digest (``_verdict_digest``): the evidence
    # is not hashed, so the summary of the newest check can show earlier
    # evidence. This test records that choice; it is not a guard.
    world, verdict = _detailed_world()
    earlier = guard.Verdict(
        verdict.verdict,
        verdict.reason,
        verdict.detail,
        verdict.evidence | {"base": "0" * 40, "reviewer_id": 1},
    )
    assert earlier.evidence != verdict.evidence
    publication, calls = _publication(
        world, [_check(runner.shadow_check_body(earlier, H)["external_id"])]
    )
    assert publication == "unchanged" and calls == []


def test_the_listing_of_earlier_shadow_checks_asks_for_every_run():
    # Review round 1, P3-7: ``filter=all``. Every shadow-named run is listed,
    # so ``total_count`` is the number of listed runs and the highest id of
    # ours is the newest. An older check of ours with this verdict and a
    # newer one with another verdict: the verdict is written again.
    world, digest = _digest_world()
    publication, calls = _publication(
        world, [_check(digest, id=5), _check("other", id=6)]
    )
    assert publication == "created" and len(calls) == 1
    listings = [path for path in world.calls if "check-runs?check_name=content" in path]
    assert listings == [SHADOW_LISTING]
    assert "&filter=all&" in SHADOW_LISTING and "filter=latest" not in SHADOW_LISTING


@pytest.mark.parametrize(
    "app",
    [
        {"id": 15368, "slug": "github-actions", "client_id": "Iv1.other"},
        {"id": 987654, "slug": "axiom-merge-guard"},
        {"id": 987654, "client_id": None},
        {"client_id": CLIENT.lower()},
        {},
        None,
        "x",
    ],
)
def test_a_shadow_named_check_of_another_app_does_not_suppress_the_verdict(app):
    # A pull request workflow can create a check with the shadow name and the
    # public verdict digest. Only a check of the guard App counts as ours.
    world, digest = _digest_world()
    publication, calls = _publication(world, [_check(digest, app=app)])
    assert publication == "created"
    assert len(calls) == 1


def test_a_forged_newer_check_does_not_hide_our_older_one():
    world, digest = _digest_world()
    forged = _check("other", id=9, app={"client_id": "Iv1.other"})
    publication, calls = _publication(world, [forged, _check(digest)])
    assert (publication, calls) == ("unchanged", [])


def test_without_the_app_client_id_every_verdict_is_written():
    world, digest = _digest_world()
    for client in (None, ""):
        publication, calls = _publication(world, [_check(digest)], app_client_id=client)
        assert publication == "created"
        assert len(calls) == 1
    assert not any("check_name=content-bound" in path for path in world.calls)


@pytest.mark.parametrize("status", ["in_progress", "queued", None])
def test_an_unfinished_check_of_ours_is_not_an_unchanged_verdict(status):
    world, digest = _digest_world()
    publication, _calls = _publication(world, [_check(digest, status=status)])
    assert publication == "created"


def test_a_check_of_ours_with_another_name_is_not_the_shadow_check():
    world, digest = _digest_world()
    other = _check(digest, id=9, name="some other check")
    publication, _calls = _publication(world, [other, _check("older", id=5)])
    assert publication == "created"


def test_an_unforeseen_evaluation_error_is_a_run_error(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(guard, "_evaluate", broken)
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=build().fetch)
    assert report["results"][0]["verdict"]["reason"] == guard.R_EVALUATION_ERROR
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False


def _body(**changes):
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE)
    return runner.shadow_check_body(verdict, H) | changes


@pytest.mark.parametrize(
    "body",
    [
        _body(name=runner.GUARD_CHECK_NAME),
        _body(name="validate / validate"),
        _body(conclusion="success"),
        _body(conclusion="failure"),
        _body(status="in_progress"),
        {k: v for k, v in _body().items() if k != "name"},
        {k: v for k, v in _body().items() if k != "conclusion"},
        {},
        None,
        [],
    ],
)
def test_the_writer_itself_refuses_everything_but_a_neutral_shadow_check(body):
    opener = Opener(status=201, raw=b'{"id": 1}')
    with pytest.raises(runner.RunnerError, match="shadow"):
        runner.check_writer(TOKEN, opener=opener)(ROOT + "/check-runs", body)
    assert opener.requests == []


def test_command_reads_the_app_client_id_from_the_environment(capsys, monkeypatch):
    world, digest = _digest_world()
    shadow_world(world, [_check(digest)])
    writer = Writer()
    monkeypatch.setattr(runner, "token_reader", lambda token, opener=None: world.fetch)
    monkeypatch.setattr(runner, "check_writer", lambda token, opener=None: writer)
    environ = {runner.TOKEN_ENV: TOKEN, runner.APP_CLIENT_ID_ENV: CLIENT}
    assert runner.command(arguments(mode="shadow"), environ=environ) == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["publication"] == (
        "unchanged"
    )
    assert runner.command(
        arguments(mode="shadow"), environ={runner.TOKEN_ENV: TOKEN}
    ) == (0)
    assert json.loads(capsys.readouterr().out)["results"][0]["publication"] == (
        "created"
    )


def test_workflow_gives_the_app_client_id_to_the_command():
    steps = workflow()["jobs"]["run"]["steps"]
    evaluate = next(step for step in steps if step.get("name") == "Evaluate and report")
    assert runner.APP_CLIENT_ID_ENV == "MERGE_GUARD_APP_CLIENT_ID"
    assert evaluate["env"][runner.APP_CLIENT_ID_ENV] == (
        "${{ vars.MERGE_GUARD_APP_CLIENT_ID }}"
    )


def gate(**changes):
    """Run the gate job's script as the runner does, with these values."""
    import subprocess

    (step,) = workflow()["jobs"]["gate"]["steps"]
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "REF": "refs/heads/main",
        "ENABLED": "true",
        "COUNTRY": "us",
        "PR": "",
    } | changes
    return subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True
    ).returncode


@pytest.mark.parametrize(
    "changes", [{}, {"PR": "12"}, {"COUNTRY": "nz", "PR": "999999999"}]
)
def test_gate_script_admits_a_valid_dispatch(changes):
    assert gate(**changes) == 0


@pytest.mark.parametrize(
    "changes",
    [
        # Multi-line values: a per-line match would admit these, and the token
        # action splits ``repositories`` on newlines and commas.
        {"COUNTRY": "us\naxiom-encode"},
        {"COUNTRY": "axiom-encode\nus"},
        {"COUNTRY": "us\n"},
        {"COUNTRY": "us,axiom-encode"},
        {"PR": "12\n--mode"},
        {"PR": "--sweep\n12"},
        {"PR": "12\n"},
        {"COUNTRY": "US"},
        {"COUNTRY": "usa"},
        {"COUNTRY": "u"},
        {"COUNTRY": ""},
        {"COUNTRY": "u s"},
        {"PR": "0"},
        {"PR": "012"},
        {"PR": "abc"},
        {"PR": "1234567890"},
        {"PR": "-1"},
        {"ENABLED": ""},
        {"ENABLED": "True"},
        {"ENABLED": "true\n"},
        {"ENABLED": "false"},
        {"REF": "refs/heads/feature"},
        {"REF": "refs/heads/main\nx"},
        {"REF": ""},
    ],
)
def test_gate_script_refuses(changes):
    assert gate(**changes) == 1


def test_design_names_the_enforcing_runner_as_its_own_change_before_step_4():
    doc = (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md").read_text()
    rows = [line for line in doc.splitlines() if line.startswith("| 3b |")]
    assert len(rows) == 1
    row = rows[0]
    assert "enforcing" in row.lower() and "`content-bound-merge-guard`" in row
    assert "every" in row and "trigger" in row.lower()
    assert "snapshot" in row  # the page-number sweep is fixed there
    flat = " ".join(doc.split())
    start = flat.index("4. **M** Add the guard's check")
    step = flat[start : flat.index("5. Run the current-order canary")]
    assert "change 3b" in step and "precondition" in step.lower()
    assert "WARNING" in step
    assert f"`{runner.SHADOW_CHECK_NAME}`" in step
    assert f"`{runner.GUARD_CHECK_NAME}`" in step


# --- follow-ups of #2257 --------------------------------------------------------

RUN_GATE_STEP = "Refuse again unless on main with the switch on"


def assert_c_locale_before_the_first_match(script):
    lines = [line.strip() for line in script.splitlines()]
    assert lines.count("export LC_ALL=C") == 1
    matches = [i for i, line in enumerate(lines) if "=~" in line]
    assert len(matches) == 2
    assert lines.index("export LC_ALL=C") < matches[0]
    # Nothing between the export and the matches sets a locale again.
    assert not any(
        name in line
        for line in lines[lines.index("export LC_ALL=C") + 1 :]
        for name in ("LC_", "LANG")
    )


def test_run_job_repeats_the_ref_and_switch_checks_before_any_credential():
    # R2-P3-1. A rerun of the run job alone reuses the earlier gate result, so
    # the run job makes the two checks itself, first.
    steps = workflow()["jobs"]["run"]["steps"]
    first = steps[0]
    assert first["name"] == RUN_GATE_STEP
    assert set(first) == {"name", "env", "run"}  # no ``if``, no continue-on-error
    assert first["env"] == {
        "REF": "${{ github.ref }}",
        "ENABLED": "${{ vars.MERGE_GUARD_RUNNER_ENABLED }}",
        "COUNTRY": "${{ inputs.country }}",
        "PR": "${{ inputs.pr }}",
    }
    script = first["run"]
    assert script.splitlines()[0] == "set -euo pipefail"
    assert '[ "$REF" != "refs/heads/main" ]' in script
    assert '[ "$ENABLED" != "true" ]' in script
    # Review round 1, P3-1: the two input checks of the gate, repeated.
    assert script.count("exit 1") == 4
    # Review round 2, P3-D: ``[a-z]`` and ``[0-9]`` are ranges of the
    # locale's collation in bash; the C locale is set before the first
    # pattern match, in the run job and in the gate step.
    assert_c_locale_before_the_first_match(script)
    (gate_step,) = workflow()["jobs"]["gate"]["steps"]
    assert_c_locale_before_the_first_match(gate_step["run"])
    # No credential is named before the repeated checks, and no step that
    # names one runs when they fail.
    assert "secrets." not in json.dumps(first)
    assert "MERGE_GUARD_APP" not in json.dumps(first)
    for step in steps[1:]:
        if "secrets." in json.dumps(step) or "app-token" in json.dumps(step):
            assert "if" not in step, step.get("name")
    assert "continue-on-error" not in workflow()["jobs"]["run"]


def run_gate(**changes):
    """Run the run job's repeated checks as the runner does."""
    import subprocess

    step = workflow()["jobs"]["run"]["steps"][0]
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "REF": "refs/heads/main",
        "ENABLED": "true",
        "COUNTRY": "us",
        "PR": "",
    } | changes
    return subprocess.run(
        ["bash", "-c", step["run"]], env=env, capture_output=True, text=True
    ).returncode


def test_run_job_checks_admit_main_with_the_switch_on():
    assert run_gate() == 0
    assert run_gate(PR="12") == 0
    assert run_gate(COUNTRY="nz", PR="999999999") == 0


# Review round 1, P3-1. The token step names ``rulespec-<country>`` and the
# token action splits its repository list on newlines and commas. The run job
# refuses these inputs itself; it does not rely on the gate job having run.
INVALID_INPUTS = [
    {"COUNTRY": "us\naxiom-encode"},
    {"COUNTRY": "axiom-encode\nus"},
    {"COUNTRY": "us\n"},
    {"COUNTRY": "us,axiom-encode"},
    {"COUNTRY": "US"},
    {"COUNTRY": "usa"},
    {"COUNTRY": "u"},
    {"COUNTRY": ""},
    {"COUNTRY": "u s"},
    {"PR": "12\n--mode"},
    {"PR": "--sweep\n12"},
    {"PR": "12\n"},
    {"PR": "0"},
    {"PR": "012"},
    {"PR": "abc"},
    {"PR": "1234567890"},
    {"PR": "-1"},
]


@pytest.mark.parametrize("changes", INVALID_INPUTS)
def test_run_job_refuses_the_inputs_the_gate_refuses(changes):
    assert gate(**changes) == 1
    assert run_gate(**changes) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"ENABLED": ""},
        {"ENABLED": "false"},
        {"ENABLED": "True"},
        {"ENABLED": "true\n"},
        {"REF": "refs/heads/feature"},
        {"REF": "refs/heads/main\nx"},
        {"REF": ""},
    ],
)
def test_run_job_checks_refuse_a_rerun_off_main_or_with_the_switch_off(changes):
    assert run_gate(**changes) == 1


DEEP_JSON = b"[" * 200_000 + b"]" * 200_000


def test_deeply_nested_json_is_a_failed_read_not_a_crash():
    # R2-P3-6. The premise first: this body is what raises RecursionError.
    with pytest.raises(RecursionError):
        json.loads(DEEP_JSON)
    read = runner.token_reader(TOKEN, opener=Opener(raw=DEEP_JSON))
    with pytest.raises(runner.RunnerError, match="not JSON"):
        read(ROOT + "/pulls/42")


@pytest.mark.parametrize("changes", [{"pr": None, "sweep": True}, {}])
def test_command_reports_deeply_nested_json_as_a_run_error(capsys, changes):
    code = runner.command(
        arguments(**changes),
        environ={runner.TOKEN_ENV: TOKEN},
        opener=Opener(raw=DEEP_JSON),
    )
    out = capsys.readouterr()
    assert code == runner.EXIT_RUN_ERROR
    assert out.out.count("\n") == 1 and TOKEN not in out.out
    report = json.loads(out.out)
    assert report["schema"] == runner.REPORT_SCHEMA
    assert report["complete"] is False
    assert report["error"]


def _evaluate_step():
    steps = workflow()["jobs"]["run"]["steps"]
    return next(step for step in steps if step.get("name") == "Evaluate and report")


def test_evaluate_step_sets_pipefail_before_the_pipe_to_tee():
    # R2-P3-7. Without pipefail the exit status of ``... | tee`` is tee's.
    script = _evaluate_step()["run"]
    assert script.splitlines()[0] == "set -euo pipefail"
    assert "| tee " in script
    assert "set +" not in script and "|| true" not in script
    assert "continue-on-error" not in _evaluate_step()
    assert _evaluate_step().get("shell", "bash") == "bash"


@pytest.mark.parametrize("pr", ["", "42"])
def test_evaluate_step_fails_when_the_runner_fails(tmp_path, pr):
    """Run the step's script with a runner that prints a report and exits 1."""
    import subprocess

    binary = tmp_path / ".venv" / "bin" / "axiom-encode"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho '{\"complete\":false}'\nexit 1\n")
    binary.chmod(0o755)
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "COUNTRY": "us",
        "PR": pr,
        "MODE": "dry-run",
        "RUNNER_TEMP": str(tmp_path),
    }
    done = subprocess.run(
        ["bash", "-c", _evaluate_step()["run"]],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    # The report still reaches the file that the upload step reads.
    report = (tmp_path / "merge-guard-report.json").read_text()
    assert json.loads(report) == {"complete": False}


def _shadow_run(run_id, **changes):
    return {
        "id": run_id,
        "name": runner.SHADOW_CHECK_NAME,
        "external_id": "other",
        "status": "completed",
        "app": {"id": 15368, "slug": "github-actions", "client_id": "Iv1.other"},
    } | changes


def _paged_shadow_world(world, runs, changes=None):
    """Serve ``runs`` as GitHub pages of 100 with the whole ``total_count``."""
    pages = [runs[i : i + 100] for i in range(0, len(runs), 100)] or [[]]
    if len(runs) % 100 == 0 and runs:
        pages.append([])
    for index, page in enumerate(pages, start=1):
        world.api[f"{SHADOW_LISTING_PAGES}{index}"] = {
            "total_count": len(runs),
            "check_runs": page,
        } | (changes or {}).get(index, {})
    return world


def test_more_than_one_page_of_shadow_named_checks_is_read_whole():
    # R2-P3-8. 250 checks with the shadow name on one head, ours among them on
    # the last page: the verdict is neither withheld nor written twice.
    world, digest = _digest_world()
    others = [_shadow_run(run_id) for run_id in range(1000, 1249)]
    ours = _check(digest, id=1249)
    writer = Writer()
    report, code = sweep(
        NAME,
        [42],
        runner.SHADOW,
        fetch=_paged_shadow_world(world, [*others, ours]).fetch,
        post=writer,
    )
    assert code == 0
    assert report["results"][0]["publication"] == "unchanged"
    assert writer.calls == []
    assert f"{SHADOW_LISTING_PAGES}3" in world.calls


def test_many_foreign_shadow_named_checks_do_not_withhold_the_verdict():
    world, _digest = _digest_world()
    others = [_shadow_run(run_id) for run_id in range(1000, 1150)]
    writer = Writer()
    report, code = sweep(
        NAME,
        [42],
        runner.SHADOW,
        fetch=_paged_shadow_world(world, others).fetch,
        post=writer,
    )
    assert code == 0
    assert report["results"][0]["publication"] == "created"
    assert len(writer.calls) == 1


def test_our_newest_check_on_a_later_page_decides():
    # The same verdict sits on page 1; a newer check of ours on page 2 differs.
    world, digest = _digest_world()
    runs = [
        _check(digest, id=1000),
        *[_shadow_run(run_id) for run_id in range(1001, 1100)],
        _check("other", id=1100),
    ]
    entry, calls = _publication_paged(world, runs)
    assert entry["publication"] == "created" and len(calls) == 1


def _publication_paged(world, runs, changes=None):
    writer = Writer()
    report, _code = sweep(
        NAME,
        [42],
        runner.SHADOW,
        fetch=_paged_shadow_world(world, runs, changes).fetch,
        post=writer,
    )
    return report["results"][0], writer.calls


def test_check_listing_past_its_page_bound_fails_closed():
    world, digest = _digest_world()
    count = runner.MAX_CHECK_RUN_PAGES * 100 + 1
    runs = [_check(digest, id=1000), *[_shadow_run(1000 + i) for i in range(1, count)]]
    entry, calls = _publication_paged(world, runs)
    assert entry["publication"] == "error"
    assert "page bound" in entry["publication_error"]
    assert calls == []
    assert f"{SHADOW_LISTING_PAGES}{runner.MAX_CHECK_RUN_PAGES + 1}" not in world.calls


def test_check_listing_at_its_page_bound_is_read_whole():
    world, digest = _digest_world()
    count = runner.MAX_CHECK_RUN_PAGES * 100
    runs = [*[_shadow_run(1000 + i) for i in range(count - 1)], _check(digest, id=9000)]
    entry, calls = _publication_paged(world, runs)
    assert entry["publication"] == "unchanged" and calls == []


@pytest.mark.parametrize(
    ("count", "changes"),
    [
        pytest.param(150, {2: {"total_count": 151}}, id="total-changes-between-pages"),
        pytest.param(150, {2: {"check_runs": []}}, id="short-page-before-the-total"),
        # 101 runs: page 2 holds one run, and it repeats an id of page 1.
        pytest.param(101, {2: {"check_runs": [_shadow_run(1000)]}}, id="repeated-id"),
        pytest.param(150, {2: {"check_runs": None}}, id="page-without-a-list"),
        # 101 runs, all on page 1: the count adds up, the page size does not.
        pytest.param(
            101,
            {1: {"check_runs": [_shadow_run(i) for i in range(1, 102)]}},
            id="page-above-the-page-size",
        ),
        pytest.param(150, {1: {"total_count": -1}}, id="negative-total"),
        # One run and the total True: True == 1, but it is not a count.
        pytest.param(1, {1: {"total_count": True}}, id="boolean-total"),
    ],
)
def test_a_paged_check_listing_that_does_not_add_up_fails_closed(count, changes):
    world, digest = _digest_world()
    runs = [_check(digest, id=999), *[_shadow_run(i) for i in range(1000, 999 + count)]]
    assert len(runs) == count
    entry, calls = _publication_paged(world, runs, changes)
    assert entry["publication"] == "error"
    assert calls == []


# Checklist step 1 of the design's section 8 (R2-P3-9). One place to update
# when the maintainers revise the step. Each phrase is matched in the step's
# text with runs of white space collapsed. This pins the wording of the
# precondition; it is not evidence that any mechanism closes the boundary.
STEP_1_START = "1. **M** Hard precondition"
STEP_1_END = "2. Merge changes 1-4"
STEP_1_REQUIRED = {
    "hold: before the key is stored or the runner enabled, shadow included": [
        "before anyone stores the App key",
        "`MERGE_GUARD_RUNNER_ENABLED`",
        "shadow mode included",
        "Recording the precondition in this design does not meet it",
        "The `main`-only branch rule alone does not meet this precondition",
        "fresh independent review verifies closure over the whole key-exposure set",
    ],
    "scope: the private key, not only a minted token": [
        "private key",
        "any of the App's installations",
        "for as long as the key stays valid",
    ],
    "scope: every job that can select the environment": [
        "every job, in any workflow",
        "select the `merge-guard` environment",
        "new, replaced or modified workflow",
        "dispatch input",
        "computes it in an expression",
        "the ability to land such a workflow on `main`",
        "not which workflow file or job selects it",
    ],
    "scope: everything such a job executes or leaves behind": [
        "the workflow steps and the actions they call",
        "`src/axiom_encode/entrypoint.py`",
        "`src/axiom_encode/__init__.py`",
        "`signing_broker` start-up",
        "directly or transitively",
        "`pyproject.toml`",
        "`uv.lock`",
        "build configuration",
        "build, install and interpreter start-up hooks",
        "every step before the token step",
        "anything it persists on the same runner",
    ],
    "mechanism and complement": [
        "mechanism",
        "complement",
        "(a)",
        "(b)",
        "(c)",
        "prevents other workflows from obtaining the key by selecting `merge-guard`",
        "no agent session approving",
    ],
    # Update this group, in a reviewed change, when Max chooses.
    "nothing is chosen yet": [
        "none of them chosen",
        "No option or complement has approval here",
        "This design chooses neither",
    ],
}


def _step_1(doc: str) -> str:
    flat = " ".join(doc.split())
    assert flat.count(STEP_1_START) == 1 and flat.count(STEP_1_END) == 1
    return flat[flat.index(STEP_1_START) : flat.index(STEP_1_END)]


def step_1_problems(step: str) -> list[str]:
    return [
        f"{group}: step 1 lost {phrase!r}"
        for group, phrases in STEP_1_REQUIRED.items()
        for phrase in phrases
        if phrase not in step
    ]


def _design() -> str:
    return (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md").read_text(
        encoding="utf-8"
    )


def test_design_checklist_step_1_keeps_its_scope_mechanism_and_hold():
    assert step_1_problems(_step_1(_design())) == []


@pytest.mark.parametrize(
    "phrase",
    [phrase for phrases in STEP_1_REQUIRED.values() for phrase in phrases],
)
def test_the_step_1_pin_reports_each_lost_element(phrase):
    step = _step_1(_design())
    assert phrase in step
    assert any(
        repr(phrase) in problem for problem in step_1_problems(step.replace(phrase, ""))
    )

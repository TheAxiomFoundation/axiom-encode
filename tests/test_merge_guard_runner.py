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
    B2,
    C1,
    CHECKS,
    COMPANION,
    H2,
    MAX,
    NAME,
    PAVEL,
    PRIMARY,
    REPO,
    ROOT,
    B,
    G,
    H,
    M,
    P,
    T,
    advance,
    build,
    commit,
    oid,
    plain,
    queue,
    review,
    second_read,
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
# The evidence every verdict of the evaluator carries: the judged pull request.
FOR_42 = {"pull_request": 42}


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
            "external_id": body["external_id"],
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
        guard.Verdict(guard.PASS, guard.R_PASS, "", FOR_42),
        guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", FOR_42),
        guard.Verdict(guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT, "", FOR_42),
    ):
        body = runner.shadow_check_body(verdict, H, 42)
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
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "d" * 20000, FOR_42)
    with pytest.raises(runner.RunnerError):
        runner.shadow_check_body(verdict, "not-a-sha", 42)
    with pytest.raises(runner.RunnerError):
        runner.shadow_check_body(verdict, None, 42)
    summary = runner.shadow_check_body(verdict, H, 42)["output"]["summary"]
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
    report, code = sweep(NAME, [42], runner.SHADOW, fetch=world.fetch, post=writer)
    (entry,) = report["results"]
    assert entry["verdict"]["verdict"] == "fail"
    assert entry["verdict"]["reason"] == guard.R_DELTA_INCOMPLETE
    # Review of head 36bab7533: the verdict stays ``fail`` AND the run is an
    # error. A truncated tree is not a completed evaluation.
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False
    assert report["error"] == "an evaluation could not read GitHub"
    ((_path, body),) = writer.calls
    assert body["conclusion"] == "neutral"
    assert "fail" in body["output"]["title"]


def test_truncated_file_list_fails_closed():
    # The PR file list omits a file of the tree delta (a truncated listing).
    world = build()
    world.api[ROOT + "/pulls/42/files?per_page=100&page=1"] = world.api[
        ROOT + "/pulls/42/files?per_page=100&page=1"
    ][:-1]
    report, code = sweep(NAME, [42], runner.DRY_RUN, fetch=world.fetch)
    assert report["results"][0]["verdict"]["verdict"] == "fail"
    assert report["results"][0]["verdict"]["reason"] == guard.R_DELTA_DISAGREES
    assert code == runner.EXIT_RUN_ERROR and report["complete"] is False


def test_shadow_does_not_repeat_an_unchanged_verdict():
    world = plain({"README.md": b"x\n"})
    digest = runner.shadow_check_body(world.evaluate(), H, 42)["external_id"]
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
    digest = runner.shadow_check_body(world.evaluate(), H, 42)["external_id"]
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
        # A single dot as a segment of the query (independent review of
        # a3c621254): only the plain ``segment == "."`` test refuses it.
        ROOT + "/pulls?x=/./y",
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
                "external_id": body["external_id"],
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
    body = runner.shadow_check_body(
        guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", FOR_42), H, 42
    )
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
    return world, runner.shadow_check_body(world.evaluate(), H, 42)["external_id"]


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
    earlier_digest = runner.shadow_check_body(earlier, H, 42)["external_id"]
    assert earlier_digest != runner.shadow_check_body(verdict, H, 42)["external_id"]
    publication, calls = _publication(world, [_check(earlier_digest)])
    assert publication == "created" and len(calls) == 1
    # No detail against an empty detail is a change as well.
    bare = guard.Verdict(verdict.verdict, verdict.reason, "", verdict.evidence)
    publication, calls = _publication(
        world, [_check(runner.shadow_check_body(bare, H, 42)["external_id"])]
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
        world, [_check(runner.shadow_check_body(earlier, H, 42)["external_id"])]
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
    verdict = guard.Verdict(guard.FAIL, guard.R_NO_LANE, "", FOR_42)
    return runner.shadow_check_body(verdict, H, 42) | changes


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


# The step of the design's activation checklist that makes the guard a merge
# requirement. Until the recommendation change it was step 4 ("Add the
# guard's check ... to the required checks"); the checklist is now the whole
# ordered runbook. Review round 1 of the recommendation (P1-4) gave the
# infrastructure lane its own step 10, so the step is number 11 (the ruleset
# becomes Active). Steps 1 and 2 keep their numbers: code comments name them.
#
# Each phrase carries the words before it that give it its meaning, so a
# negation in front of it ("whether or not the experiment passed") or another
# number ("steps 0-2 are done") is a lost phrase (review round 1, P2-8).
ACTIVE_STEP_START = "11. **M** (an organization owner) Switch the ruleset to `Active`"
ACTIVE_STEP_END = "12. Run the current-order canary"
ACTIVE_STEP_REQUIRED = [
    "Hard preconditions, all of them: steps 0-10 are done; change 3b is on `main`",
    "is the code at the pinned SHA; the experiment passed, as step 3 defines it;",
    "; the merge queue of step 8 is on, is still required at this moment",
    # Review round 3, P2-3 and P1-2: the bypass list of the queue ruleset and
    # the organization's "Runners" setting are preconditions of this step.
    "the merge-queue ruleset of `rulespec-us` has an empty bypass list",
    "**an organization owner has verified, at this moment, that the "
    'organization\'s "Runners" setting allows no repository-level self-hosted '
    "runner for `rulespec-us`**",
    '**If the "Runners" setting has another value, step 11 waits.**',
    "Changing it is an act on the whole organization, not on `rulespec-us`",
    "- Expected: a `rulespec-us` pull request merges only when its guard run is green.",
    "Run it before the switch for `merge_queue` and after it for both.",
    "shows `active`, an empty bypass list, `refs/heads/main` and the one workflow "
    "entry.",
    "the `bypass_actors` read of step 8 prints `[]`",
    "and Max has decided question 0 of §10 **and its lane is on at the pinned SHA** (step 10)",
    # Review round 1 of #2286, F1: the hold for the runner's inert path, the
    # one assumption that would fail open. Review round 1 of the
    # recommendation, P3-1: the key ``verdict`` alone is null in the inert
    # report too.
    "at the pin that is in the ruleset now, a pull request with no lane shows a "
    "failed `content-bound-merge-guard` job whose report has a non-null `verdict` "
    'and no `"inert"`',
    "**WARNING:** without the lane of question 0, an `Active` ruleset blocks every "
    "pin, ledger, workflow and tooling pull request on `rulespec-us`",
    "the rollback itself needs an organization owner",
    # P1-3 (b): the queue is a repository setting.
    "must contain `workflows` **and** `merge_queue`",
    "The merge queue is a repository setting of `rulespec-us`. Its administrators "
    "can turn it off",
    # P1-1: the rollback and its order.
    "- Rollback: set the enforcement back to `Evaluate`. Before step 13 this is free.",
    "**After step 13, only in the rollback order at the top of this checklist: "
    "first the writer off, then no signing run alive and the reader pin reverted, "
    "only then the ruleset.**",
    f"require the name `{runner.GUARD_CHECK_NAME}` only, and never "
    f"`{runner.SHADOW_CHECK_NAME}`, which is always neutral",
]


def _active_step(doc: str) -> str:
    flat = " ".join(doc.split())
    assert flat.count(ACTIVE_STEP_START) == 1 and flat.count(ACTIVE_STEP_END) == 1
    return flat[flat.index(ACTIVE_STEP_START) : flat.index(ACTIVE_STEP_END)]


def active_step_problems(step: str) -> list[str]:
    return [
        f"the Active step lost {phrase!r}"
        for phrase in ACTIVE_STEP_REQUIRED
        if phrase not in step
    ]


def test_design_names_the_enforcing_runner_as_its_own_change_before_the_active_step():
    doc = (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md").read_text()
    rows = [line for line in doc.splitlines() if line.startswith("| 3b |")]
    assert len(rows) == 1
    row = rows[0]
    assert "enforcing" in row.lower() and "`content-bound-merge-guard`" in row
    assert "every" in row and "trigger" in row.lower()
    assert "snapshot" in row  # the page-number sweep is fixed there
    assert active_step_problems(_active_step(doc)) == []


@pytest.mark.parametrize("phrase", ACTIVE_STEP_REQUIRED)
def test_the_active_step_pin_reports_each_lost_element(phrase):
    doc = (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md").read_text()
    step = _active_step(doc)
    assert phrase in step
    assert any(
        repr(phrase) in problem
        for problem in active_step_problems(step.replace(phrase, ""))
    )


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
# The key-exposure scope and the hold stay pinned although the step now
# recommends an option without a key: they bind both fallbacks and any App.
STEP_1_START = "1. **M** Hard precondition"
STEP_1_END = "2. **M** Merge the changes in this order"
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
    # Update this group, in a reviewed change, when Max accepts or rejects.
    "a recommendation exists, and nothing is chosen or accepted": [
        "None of them is chosen",
        "No option or complement has approval here",
        "**The recommendation is (g')**",
        "no guard App and no private key",
        "`docs/merge-guard-key-protection.md`",
        "Max has not accepted it",
    ],
    "the experiment is a precondition": [
        "The experiment is a precondition",
        "`docs/merge-guard-ruleset-experiment.md`",
        "nothing is chosen for activation",
    ],
    "the fallback is named": [
        "the named fallback is (b)+(d)",
        "non-exportable signing key",
        "the second fallback is (b) with the complement",
        "one approved but wrong change can leak the key for good",
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


# --- the decisive sentences of the three recommendation documents ----------------
#
# Review round 1 of the recommendation, P2-8. Each sentence must occur exactly
# once in its document, with runs of white space collapsed. A sentence is
# pinned with the words that give it its meaning, so the reviewer's mutants
# (a status that says "accepted", an agent that "may" run the experiment, a
# negated precondition, another step number) each lose a sentence. This pins
# wording; it is no evidence that a maintainer accepted or did anything.

DESIGN_DOC = "docs/content-bound-merge-guard.md"
KEY_PROTECTION_DOC = "docs/merge-guard-key-protection.md"
EXPERIMENT_DOC = "docs/merge-guard-ruleset-experiment.md"
DOC_SENTENCES = {
    DESIGN_DOC: {
        "nothing is chosen or accepted": [
            "Status: **design; maintainer acceptance and activation remain pending.**",
            "**Max has not accepted the recommendation, and nothing is chosen.** It "
            "holds only if the owner's experiment passes "
            "(`docs/merge-guard-ruleset-experiment.md`, not run).",
            "Two paths exist in code. Neither is active, and Max has accepted neither.",
            "**Max has not accepted that recommendation; until step 0 is done, every "
            "later step waits.**",
            "| Recommendation recorded, Max has not accepted yet: the ruleset, at a "
            "pinned SHA",
            "| Recommendation recorded, Max has not accepted yet: (g'), which creates "
            "no key and so needs no complement; first fallback (b)+(d); second "
            "fallback (b) with code-owner review of `.github/` and last-push approval. "
            "Max chooses.",
            "### The ruleset experiment (not run)",
        ],
        "how the decision is evidenced": [
            "A pull request comment from Max's account is **not sufficient evidence** "
            "of his decision",
            "An organization owner creates the `Evaluate` ruleset of step 6.",
            "**This checklist cannot verify who was at the keyboard**",
        ],
        "what an agent must not do": [
            "- **AGENT**: an agent session. It prepares pull requests and reads. It "
            "does not merge, mark ready, approve, change settings, dispatch signing "
            "runs or hold an `admin:org` token.",
            "and no stored secret, App or key. An agent must not run it.",
            "2. **M** Merge the changes in this order. Maintainers merge; agents may "
            "not.",
            "A wrong result is reported to a maintainer; the agent changes nothing.",
            "An agent may run these. Each is a GET with the token that agent "
            "sessions already have; none needs `admin:org`.",
            "An agent cannot make them: they need `admin:org`.",
            "No step of the experiment and no agent changes it.",
        ],
        # Review round 3, P1-3: the role and the mode of every step heading, by
        # value. An AGENT in the place of **M**, or `Active` in the place of
        # `Evaluate`, loses the sentence.
        "the role and the mode of each step": [
            "0. **M** (Max) Decide:",
            "1. **M** Hard precondition, before anyone stores the App key or sets "
            "`MERGE_GUARD_RUNNER_ENABLED`, shadow mode included:",
            "3. **M** (any organization owner) Run the experiment of "
            "`docs/merge-guard-ruleset-experiment.md`.",
            "4. AGENT, only if the results ask for it: a pull request with each "
            "named change of step 3, through a review cycle. **M** merges.",
            "5. AGENT and **M** Review the guard at the candidate commit, as §6 of "
            "`docs/merge-guard-key-protection.md` lists: an independent agent review "
            "and a maintainer's own reading of the workflow file and the eight "
            "bundle files.",
            "6. **M** (an organization owner) Create the organization ruleset in "
            "`Evaluate`, with the values of `docs/merge-guard-key-protection.md` §5: "
            "target `rulespec-us`, the branch `refs/heads/main` (named explicitly, not "
            '"the default branch"), the workflow '
            "`.github/workflows/content-bound-merge-guard.yml` of axiom-encode at the "
            "reviewed SHA, empty bypass list. This starts the shadow period.",
            "- Rollback: set the enforcement to `Disabled`. Free at this step",
            "7. AGENT and **M** Shadow period.",
            "8. **M** (a `rulespec-us` administrator) Turn on the merge queue for "
            "`rulespec-us` `main`, with the settings that rulespec-us#2160 lists: "
            "only merge non-failing pull requests, group size 1, build concurrency "
            "1, a status check timeout above 4 hours, merge method squash.",
            "The queue's status-check timeout has a maximum of 360 minutes [docs D7: "
            '`check_response_timeout_minutes`, "maximum: 360"]; set it to 360.',
            "9. Change 2b, then change 4. - AGENT prepares the pull request that sets "
            "`V6_READER_ENABLED = True`.",
            "**M** merges. The ruleset's pin does not move: the evaluator does not "
            "read this switch",
            "- AGENT then prepares the `rulespec-us` pull request that pins the "
            "encoder at that commit. **M** merges it.",
            "- The switch rule of step 2 governs: the writer stays off.",
            "1. **M** (Max) decides question 0: the proposed three-class lane of §3, "
            "another lane, or no lane.",
            "2. AGENT prepares the lane change, one pull request on axiom-encode:",
            "It goes through a review cycle. **M** merges.",
            "3. AGENT and **M** review the guard at the new commit, as in step 5, "
            "and record the new SHA and `BUNDLE_SHA256` in §11.",
            "4. **M** (an organization owner) moves the ruleset's pin to the new "
            "commit, **while the ruleset is still `Evaluate`**",
            "5. AGENT and **M** observe again, at the new pin:",
            "12. Run the current-order canary to completion: one promotion in "
            "today's order (`docs/reviewed-pr-promotion.md`) through the `Active` "
            "guard.",
            "13. Change 6, as one reviewed pull request. AGENT prepares it; it is not "
            "opened before steps 0-12 are done, because merging it takes effect.",
            "(`GUARD_WORKFLOW`: repository id, path, and the set of pinned SHAs, "
            "which holds exactly the SHA that the ruleset pins now), sets "
            "`V6_WRITER_ENABLED = True`, and updates `docs/reviewed-pr-promotion.md`. "
            "The lock itself is change 5c",
            "The ruleset's pin does **not** move for this change: the guard reads "
            "neither the switch nor `GUARD_WORKFLOW` [author-reported, activation "
            "preparation 2026-11-17]. **M** merges.",
            "14. The smallest live canary of the new order, on one real candidate.",
            "1. AGENT or contributor: local encode → candidate pull request.",
            "2. **M** (an enrolled dispatcher; it may be the candidate's author) "
            "dispatches `targeted-signed-reencode.yml`. The `production-signing` "
            "environment approval still applies to the run.",
            "**M** (a maintainer with write access who will **not** be the "
            "approver) marks it ready and squash-merges it there.",
            "5. **M** (the other enrolled owner) reads the green pull request and "
            "approves it on GitHub.",
            "6. **M** (a maintainer with write access) re-runs the failed guard job, "
            "or follows the procedure that step X3 found.",
            "7. If `main` moved, **M** (a maintainer who is not the approver) updates "
            "the branch first.",
            "Then **M** adds the pull request to the merge queue.",
            '15. "Continuous landing is on" when step 14 passed and the maintainers '
            "say so in §11.",
            "Steps 11 and 13 land as separate reviewed changes, step 11 first,",
        ],
        "the rollback order": [
            "**ROLLBACK ORDER. This rule binds every rollback below once step 13 has "
            "happened.**",
            "1. First set `V6_WRITER_ENABLED` back to `False`, in a reviewed change "
            "on axiom-encode `main` (the rollback of step 13).",
            # Review round 3, P1-1: signing runs that started before the
            # writer was switched off, and the missing transaction.
            "2. Then make every signing run and every signed v6 manifest harmless, "
            "in this order:",
            "- (i) **No signing run is alive.** Confirm that no run of "
            "`targeted-signed-reencode.yml` is queued, waiting for the "
            "`production-signing` approval or in progress. Read-only, as anyone: "
            "`gh run list --repo TheAxiomFoundation/axiom-encode --workflow "
            "targeted-signed-reencode.yml --limit 200 --json "
            "databaseId,status,headSha,createdAt --jq '[.[] | select(.status != "
            '"completed")]\'` must print `[]`',
            "so a run that started before the writer-off commit still has the "
            "writer on",
            "- (ii) **No older signing run is started again.** A re-run of an older "
            "signing run executes its original commit, where the writer is still on, "
            "and GitHub allows a re-run for 30 days [docs D9]. So no enrolled owner "
            "may re-run a signing run that was dispatched before the writer-off "
            "commit, or approve one for `production-signing`. This is a rule for "
            "people; nothing technical enforces it.",
            # Review round 4, P1-2: the rollback is a last resort and not
            # durable; P2-3: the survivors A1i, D1i, D2i, C3d; P3-1.
            "If the list is not empty, a maintainer lets each run end or cancels it, nobody approves it, and the read is repeated.",
            "The listing must reach back more than 30 days: raise `--limit`",
            "but only while the pin stays reverted, and after point 3 nothing technical keeps it reverted",
            "A signing run can be re-run, and a signed manifest can be pushed, after the read and before the ruleset changes.",
            "Leaving the revert out, which (iii) allows only with a complete inventory, leaves that gap open and depends on the rule of (ii).",
            "**After point 3 nothing technical protects `rulespec-us` from signed v6 bytes. This order is not a durable rollback.**",
            "The reader pin is content of a pull request: `repository-checks.yml` of `rulespec-us` reads `.axiom/workflow-toolchain.toml` from the checkout of the pull request itself",
            "`main` of `rulespec-us` requires 0 approvals and no code-owner review [live read 2026-11-18]. So any account with write access can move the reader pin forward again in the same pull request that lands signed v6 bytes [inferred]. Not tested: whether validation refuses a pull request that moves the pin and adds a manifest together.",
            "**So the recommended response to a guard problem after step 13 is not to weaken the ruleset.** In this order of preference:",
            "- (a) **Repair forward.** Move the pin to a fixed guard while the ruleset stays `Active`",
            "- (b) **Take the guard out of use, closed.** Do point 1 and keep the ruleset `Active`. A guard that cannot pass then blocks every merge of `rulespec-us`. That is the fail-closed state, and it holds until the repair.",
            "- (c) **Last resort: weaken the ruleset or the queue, in the order of points 1 to 3.** It reopens the exposure that existed before the guard, now for signed v6 bytes. While the ruleset is weak, three standing conditions hold. Each is a rule for people:",
            "- No pull request that pins `rulespec-us` to an encoder commit with the reader on merges while the ruleset is weak.",
            "- For 30 days after the writer-off commit, no enrolled owner re-runs or approves a signing run that was dispatched before that commit; or a maintainer deletes those runs, which removes the re-run",
            "Deleting a run is a maintainer's act; no agent does it.",
            "- Signed v6 bytes that never landed stay valid indefinitely: nothing in this design expires a signature [inferred].",
            "**Durable options, not designed:** enforced code-owner review of `.axiom/workflow-toolchain.toml` on `rulespec-us`, switched on before point 3",
            "or a new signing key, after which validation refuses every older signature [inferred].",
            "So (iii) reverts the reader pin in every case except the one its inventory proves: every signed v6 manifest that exists has landed or was never pushed.",
            "That order is then a last resort and not a durable rollback; the recommended response is to repair forward or to keep the ruleset `Active` |",
            "That order is a last resort and not a durable rollback: after it nothing technical protects `rulespec-us` from signed v6 bytes. The recommended response is to repair forward, or to keep the ruleset `Active` with the writer off.",
            "That order is a last resort and not a durable rollback: once the ruleset is weak, nothing technical protects `rulespec-us` from signed v6 bytes, because any write account can move the reader pin forward again in the pull request that lands them [inferred].",
            "is not complete, and only the reader pin revert applies.",
            "- (iii) **No signed v6 manifest can land: the reader is off.**",
            "3. Only then weaken the ruleset or the queue. **Between the check of "
            "point 2 and the change of point 3 GitHub offers no transaction.**",
            "The lock of the writer stops new signing runs only, and only at commits "
            "where the writer is off.",
            "Closing a pull request does not remove its signed bytes",
            "**This is required whenever a signed v6 manifest is outstanding or its disposition is unknown.**",
            "No shortcut exists for an emergency: a ruleset or queue that is "
            "weakened first is exactly the state that the switch rule describes.",
            "Do not disable the ruleset to land the revert",
            "**Before step 13 the ruleset rollback is free**",
            "- Rollback: turn the queue off. **WARNING:** allowed only while the "
            "ruleset is not `Active`. From step 11 on, a queue that is off silently "
            "reopens the stale-green window",
            "- Rollback: set the switch back, in a reviewed change. The ruleset stays "
            "`Active`. **This is point 1 of the rollback order at the top of this "
            "checklist, and it comes before any rollback of step 8 or step 11.**",
            "Every rollback therefore has an order: first the writer off, then no "
            "signing run alive and the reader pin reverted, only then the ruleset",
        ],
        "the experiment and its pass rule": [
            # Review round 4, P2-1: two adjusting classes, so that the rule
            # can be met by a step whose change is a procedure or a decision.
            '- **"The experiment passed" means exactly this:** every *blocking* step has the result `pass`; every step of the class *adjusting, run again* has either `pass`, or its named change merged, reviewed and the step run again with `pass`; and every step of the class *adjusting, no second run* has either `pass`, or its recorded result together with its named change reviewed and merged. An *informational* step needs a recorded result only.',
            "- **Why two adjusting classes.** For a step of the class *adjusting, no second run* the named change is a procedure or a decision, not a change of what the lab runs: the same step run again gives the same result. The rule therefore asks for the change, not for a second run.",
            "- **The experiment does not depend on any merge of step 2.** It uses only the two throwaway repositories, its own lab workflow files and the one lab ruleset; it reads no file of this stack. So an organization owner can run it at any time from now, before the merges.",
            "The one order rule: the experiment must have passed before step 5.",
            "The experiment of step 3 does not wait for these merges (step 3).",
            "A step that was not run counts as failed for this rule.",
            "The ruleset mechanism does not hold. Stop. Max decides on the fallback "
            "(b)+(d).",
            "No ruleset may name the runner, in `Evaluate` either, until the change "
            "is merged and X18 passes.",
            # Review round 3, P1-3 and P2-5: every row of the table, by value.
            "| Blocking | X1; X2 (a) to (e); X8 (a) and (b); X24; X15 (a) and (b); "
            'X15 (c), the part "the pull request merges only after the guard job has '
            'ended"; X5, the parts "a `merge_group` run starts" and "the '
            '`merge_group` run prints the marker of the pinned commit" |',
            '| Adjusting, run again | X18 (change: the runner loses its inert path, or reads the value that names the target); X11 (change: the workflow takes the bundle commit from the value that names the source commit); X5, the payload fields, and X15 (c), the identity of the group\'s check (change: the runner reads the real fields); X12 (change: the head binding of the runner and case 49); X14 (change: lock 3 of draft #2289 reads the real response shape; step 13 waits for it); X10 (change: `"protected": true` in the ruleset); X21 (change: the ruleset target, or a standing check of the default branch); the set-up record "the API accepts `sha` with no `ref`" (change, in three places together: the request body of `docs/merge-guard-key-protection.md` §5, its pin test, and lock 3 of draft #2289, which refuses an entry with `ref`) | The mechanism can hold, after the named change. Do the change through a review cycle, run the step again, and record both results |',
            '| Adjusting, no second run | X3 (change: the procedure after an approval becomes "close and reopen" or "update the branch", written into step 14 and into the table of human steps); X6 (change: the shadow period of steps 6 and 7 takes another form, named in the runbook); X20 (change: the queue trial of step 8 moves to the first `Active` landing, step 12. The same reviewed change replaces the trial clause of the step 11 precondition, and moves the stop rule for the inert path on a merge group from step 8 to step 12, before any second landing); X4 (change: none in the runner; a decision of Max: the fallback, which ends the ruleset path, or a read-only token, which is a stored secret. No such token is designed. It would have to meet the stored-credential precondition of step 11: for example a fine-grained read-only token of an account that is not enrolled, not an administrator and not an organization owner, which Max accepts by name) | The mechanism can hold with another procedure. Record the result, and do the named change of this checklist through a review cycle. No second run: it would give the same result |',
            "| Informational | X7, X9, X13, X16, X17, X19, X22; X8 (c); X15 (d); X23; "
            'the set-up records "the web form offers a SHA pin" and "how the web '
            'form names the target" | Record the result.',
            "- **X23 is informational by design.** The lab registers no self-hosted "
            "runner:",
            "No step of the experiment changes an organization setting.",
        ],
        "the inert-path hold": [
            "Every other unverified item of this section fails closed. This one would "
            "fail **open**:",
            "- **Hold for step 11 (the inert path).** At least one pull request with "
            "no lane must show a **failed** `content-bound-merge-guard` job under this "
            "`Evaluate` ruleset.",
            "The key `verdict` alone proves nothing: the inert report and every error "
            "report carry it with the value `null`",
            "stop, set the ruleset to `Disabled`, and do not go to step 11",
            "- **Stop rule for the inert path on a merge group.**",
            "means that the merge-group run took the source-repository path: stop, "
            "set the ruleset to `Disabled` (free at this step), and do not go to "
            "step 11.",
        ],
        "the lane goes live in its own step": [
            "10. **The infrastructure lane (question 0 of §10).**",
            "So it goes live here, while the ruleset is still in `Evaluate`, in this "
            "order:",
            'With "no lane" this checklist stops here: step 11 requires a lane that '
            "is on at the pinned SHA",
            "`INFRASTRUCTURE_LANE_ENABLED = True`; the lane's two success reasons "
            "added to the runner's `SUCCESS_VERDICTS`",
            "6. **The hold of step 7 is repeated at the new pin.**",
        ],
        "what Evaluate cannot show": [
            "So the queue merges the trial pull request when the group's "
            "`validate / validate` run is green, without waiting for the guard job.",
            "- **Established only by the lab, with its ruleset `Active`:** that the "
            "queue merges only after the guard job has ended",
            "**Stop rule:** if the pull request merges before the guard's merge-group "
            "job has ended, the ruleset result is not part of the queue's "
            "requirements. Set the ruleset back to `Evaluate` (free: step 13 has not "
            "happened) and do not go on.",
            "- **Cases 17 and 18 cannot be produced on a live pull request.**",
            "- Run cases 15 and 16 of §7 live on a throwaway `rulespec-us` pull "
            "request and confirm both results.",
        ],
        "the canary and the wait": [
            "(job success) **in the same run, with no re-run**",
            "**cannot be shown live on any of the eight candidates**",
            "which is **about 3 to 4 landings a day**",
            '**This design does not deliver "encode continuously without a '
            'per-candidate human approval".** After activation every candidate still '
            "needs one approval by an enrolled owner who is neither its author nor "
            "its dispatcher.",
        ],
        "the administrator residuals": [
            "- **The queue is a repository setting.**",
            "**But four repository settings stay in the administrator's hands, and "
            "each weakens the guard without touching the ruleset; a fifth, the name "
            "of the branch, is documented as closed.**",
            "| The guard workflow and the merge queue are both required on `main` | "
            "`gh api repos/TheAxiomFoundation/rulespec-us/rules/branches/main --jq "
            "'[.[].type]'` | a list that contains `workflows` and `merge_queue`.",
            "It detects; it prevents nothing.",
            # Review round 3, P2-3: a bypass actor on the queue ruleset.
            "- **The queue ruleset must have an empty bypass list.** An actor on the "
            "bypass list of that repository ruleset merges without the queue, so "
            "without the merge-group evaluation, and `rules/branches/main` still "
            "lists `merge_queue`",
            "must print `[]`. GitHub returns `bypass_actors` only to a caller with "
            "write access to the ruleset [docs D7]: an anonymous read, or a token "
            "without that access, cannot run this check, and `null` is not a pass.",
            "| Organization owner, a holder of the organization's ruleset "
            "permission, or a session with such a token | No |",
            # P2-1: the App case is not tested by any step.
            "That a check of the same name from another GitHub App cannot satisfy it "
            "is [inferred, untested]: the lab creates no App",
        ],
        # Review round 4, P1-1, P2-2, P2-4, P3-2, P3-3, P3-5.
        "stored credentials, installed Apps and Actions": [
            "| Write collaborator who reads a credential that is stored where a workflow of a branch can read it | **No**, when the credential belongs to an enrolled identity, an administrator, an organization owner, or an App that can write checks or contents |",
            "All three only while no stored credential gives the collaborator another identity (next row) |",
            'no merge and no review is needed [docs D23 for a repository secret: any user with write access to a repository can read every secret that is configured in it; the organization secret and "no merge is needed" are inferred]. The collaborator then acts as the credential\'s owner: every control of this design that rests on that identity is void for that collaborator [inferred from the documented secret semantics; not tested].',
            "That a workflow on a pushed branch is enough, with no merge, and that the same holds for an organization secret that the repository can use, is [inferred].",
            "**A removal alone does not last:** a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are [docs source D26]. It lasts only when an organization owner also restricts the installation of Apps by repository administrators, which the same text names. So the read of the installed Apps is repeated at each health check.",
            "- **An environment is no general exception** (review of head `dd8bf4fa2`).",
            "A fifth row, the installed Apps, needs the account of an organization owner; a session whose token belongs to a non-owner cannot run it and reports that.",
            "| The installed Apps are the recorded ones | `gh api --paginate orgs/TheAxiomFoundation/installations",
            "An App that is new, or that has `rulespec-us` again after its removal, goes to a maintainer |",
            "Also, a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are [docs source D26]; so removing an App from `rulespec-us` lasts only with that restriction.",
            "A deleted run loses its log, which is the record of who dispatched and who approved it [inferred]: save the output of this listing and the log of each run before the deletion.",
            "- **Change 4 has two modes, and the choice needs a maintainer's decision.**",
            "(1) *Pinned mode*, which `rulespec-us` uses today: a pin cannot move past the producer of any manifest that is already on `main`.",
            "So change 4 as a plain pin move fails CI.",
            "So change 4 in practice depends on rulespec-us#2059.",
            '(3) **In verifier-main mode `rulespec-us` verifies with encoder `main`. Then `V6_READER_ENABLED = True` on encoder `main` (change 2b) takes effect on `rulespec-us` when it merges; no separate reader pin exists to move or to revert; and the rollback "revert the reader pin" becomes "set `V6_READER_ENABLED` back to `False` on encoder `main`", a reviewed encoder change.**',
            "This step and the rollback order at the top are written for pinned mode; under verifier-main mode they need a reviewed rewrite first.",
            "The shared workflow refuses a pin pull request that changes anything beyond the pin files, so a waiver edit cannot ride with the pin move.",
            '**"Revert the reader pin" is the pinned-mode form.** If `rulespec-us` verifies with encoder `main` (verifier-main mode, checklist step 9), no reader pin exists, and the same act is a reviewed change on encoder `main` that sets `V6_READER_ENABLED` back to `False` [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim].',
            "GitHub cannot restrict a repository secret to one workflow file [inferred: no page read offers it]; an environment with required reviewers can hold a secret back [docs D24], unless an administrator bypasses or edits its rules (checklist step 11).",
            "**This exposure exists today and does not depend on the guard**: `rulespec-us` has the repository secret `BULK_ENCODE_TOKEN`",
            "Not read: its owner, its real scopes and whether it is still valid. It also touches today's promotion lane: whoever holds an enrolled owner's token can approve. Hard precondition of checklist steps 11 and 13;",
            "**no stored secret is a credential of an enrolled identity, an administrator, an organization owner or an App that can write checks or contents (an environment is an exception only by name and with verified controls), and an owner has read the installed Apps with their target repositories**",
            "Actions are enabled in `rulespec-us`; and Max has decided question 0 of §10",
            "- **Precondition check: stored credentials and installed Apps.**",
            "- **Must hold.** No stored secret is a credential of an enrolled identity, of an administrator of `rulespec-us`, of an organization owner, or of an App that can write checks or contents on `rulespec-us`.",
            "A name does not show whose credential a secret is: for each secret a maintainer states the owner, and the list of names with that statement is recorded in §11.",
            "Each installed App that can write checks on `rulespec-us` is removed from it, or Max accepts it by name as a residual risk:",
            "- **State on 2026-11-18** [live read 2026-11-18]: this precondition does not hold.",
            "The maintainer's token gets HTTP 403 there",
            '**Hard precondition: the check "Stored credentials and installed Apps" of step 11 is repeated now and holds.**',
            "Step E removes a stored token of `rulespec-us` that every write collaborator can read through a workflow of a branch",
            "| The stored secrets are the recorded ones | `gh api --paginate repos/TheAxiomFoundation/rulespec-us/actions/secrets",
            "exactly the names **and the `updated_at` of each** that §11 records",
            "| Actions are enabled in `rulespec-us` | `gh api repos/TheAxiomFoundation/rulespec-us/actions/permissions --jq .enabled` | `true`",
            "If a head with no guard run can merge while Actions are disabled, an administrator merges without the guard and enables Actions again; nothing blocks it, and the health check reads only the state at its own time [inferred].",
            "**Three more acts of an administrator are outside that count.** (6) **Installing a GitHub App.**",
            "A repository administrator can install an App that asks for no "
            'organization permission and not for "Repository administration", on the '
            "repositories that the administrator administers, unless an organization "
            "owner restricts it [docs source D26].",
            "An owner's read of the installed Apps is a precondition of `Active` (checklist step 11). (7) **Renaming the repository** [experiment X10]. (8) **Transferring the repository.**",
            "An organization owner can also rename a targeted branch without restriction [docs D22], which takes `main` out of a rule on `refs/heads/main`.",
            "Four rows need more than an anonymous read (the runners, the bypass list, the secret names and the Actions permission)",
            "Each is a read, so it is inside the AGENT role.",
            "whose credential a stored secret is, whether someone has already read a secret, and Actions that were disabled and enabled again between two checks.",
        ],
        # Review round 4, P2-3: the survivors G3i, J1i, H1i, H2d, H3d, I2d.
        "pins that review round 4 found missing": [
            "the maintainer's token has it, an anonymous read does not, and `null` is not a pass",
            "A same-named check that another GitHub App publishes is [inferred, untested]: the lab creates no App (§11).",
            "A field that arrives after those two pins makes every v6 unit fail at step 14 (closed) until both pins move again [inferred].",
            "steps 2 and 4 and, if Max accepted question 14, its change (step 0).",
            "If Max accepted question 14, that commit contains its change (step 0).",
            "What it cannot show: that the guard fails closed on a fault that did not happen.",
        ],
        "the decisions and their place": [
            # Review round 3, P2-4.
            "If Max accepts it, its change lands before the reviewed pin of step 5 "
            "and before the reader pin of step 9",
            "If Max declines it, nothing changes: the v6 shape stays as §5 gives it, "
            "and no step waits for it.",
            "After all six merges it prints five lines, with `False`, `False`, "
            "`False`, `None` and `None`.",
            "At no time may a line show `True` or a recorded identity.",
        ],
    },
    KEY_PROTECTION_DOC: {
        "nothing is chosen or accepted": [
            "Status: **recommendation; Max has not accepted it. Nothing is chosen.**",
            "It changes no setting, creates no App, environment, secret or ruleset, "
            "and satisfies no activation condition.",
            "- **Condition.** The recommendation holds only if a live experiment "
            "passes (`docs/merge-guard-ruleset-experiment.md`). Only an organization "
            "owner can run it. Until it passes, nothing is chosen for activation.",
            "- **Named fallback if the experiment fails, (b)+(d).**",
            "(4) A comment from Max's account is not sufficient evidence of this "
            "decision;",
            "Its bypass list is empty. It starts in `Evaluate` and moves to `Active` "
            "later.",
            "None of it is built or tested.",
            "Steps A, B and D are still open (read 2026-11-17).",
        ],
        "the key hold": [
            "Nobody may store a guard App key or set `MERGE_GUARD_RUNNER_ENABLED`, "
            "shadow mode included, until Max chooses a mechanism and any complement, "
            "maintainers implement them, and a fresh independent review verifies the "
            "whole key-exposure set (§6).",
            "Under (g') no key is ever created, so this hold is kept by never "
            "creating one. Under either fallback the hold applies in full.",
        ],
        "what an agent must not do": [
            "Agents perform none of these steps. Each needs repository or "
            "organization settings.",
            "An agent must not perform any step of this section: it must not create, "
            "edit or delete the ruleset, and it must not hold an `admin:org` token.",
            "Proposed standing rule: no agent session gets `admin:org`. GitHub cannot "
            "enforce that rule.",
            "It is the owner's own decision, outside the experiment, and no agent "
            "makes it.",
        ],
        # Review round 3, P1-3: the values of the web form, by value.
        "the ruleset form": [
            "- Ruleset name: `content-bound-merge-guard`.",
            "- Enforcement status: `Evaluate`.",
            "- Bypass list: empty. Add nothing.",
            "- Target repositories: selected repositories, `rulespec-us` only.",
            "- Target branches: the one ref `refs/heads/main`, as an include pattern.",
            'Do **not** choose the form\'s "default branch" target',
            '- Rule: "Require workflows to pass before merging". Repository '
            "`axiom-encode`. Workflow file "
            "`.github/workflows/content-bound-merge-guard.yml`. Pin: the reviewed "
            "commit SHA, all 40 characters. Select no other rule.",
            "It must show `evaluate`, an empty bypass list, the one repository, the "
            "one ref `refs/heads/main` and one `workflows` entry with the reviewed "
            "`sha`.",
            "must show `active`, an empty bypass list, the one repository, the one "
            "ref and the one workflow entry with the reviewed `sha`.",
        ],
        "the residual risks that decide": [
            "- **R4b. The inert path is the one assumption that fails open.**",
            "**So a merge queue on `rulespec-us` `main` is a precondition of "
            "`Active`**",
            "- (c) **The merge-group run narrows the window to seconds; it does not "
            "close it.**",
            "- **R6. Agents under both enrolled identities** (design §9). No option "
            "changes this.",
            "one session can author and dispatch as one owner and approve as the "
            "other, and every check passes. The guard records which accounts acted",
            "- **R12. What an administrator of `rulespec-us` can still change.**",
            "An administrator can turn it off with no review.",
            "**A bypass actor on the repository ruleset that holds the queue merges "
            "without the merge-group evaluation, and the rule still shows in "
            "`rules/branches/main`**",
            "The lock of draft #2289 cannot see it: it does not read bypass lists",
            "- **R13. After the v6 writer is on, a rollback of the ruleset is a last resort and is not durable; it has an order.**",
            '"Revert the reader pin" is the pinned-mode form: if `rulespec-us` verifies with encoder `main` (verifier-main mode, draft rulespec-us#2059), no reader pin exists, and the same act is a reviewed change that sets `V6_READER_ENABLED` back to `False` on encoder `main` [author-reported, pin trial 2026-11-18, rulespec-us#2304; interim]',
            "**A signing run that was dispatched before the writer-off commit still "
            "has the writer on.**",
            "So no enrolled owner may re-run such a run or approve it for "
            "`production-signing`; nothing technical enforces that.",
            "**GitHub offers no transaction between the check and the ruleset change.** Closing a pull request does not remove its signed bytes;",
            "**This order is not a durable rollback. Once the ruleset is weak, nothing technical protects `rulespec-us` from signed v6 bytes.**",
            "So any account with write access can move the reader pin forward again in the same pull request that lands signed v6 bytes [inferred]. Not tested: whether validation refuses a pull request that moves the pin and adds a manifest together.",
            "**The recommended response to a guard problem after step 13 is therefore not to weaken the ruleset**: (a) repair forward, with a pin move to a fixed guard while the ruleset is `Active`; or (b) turn the writer off and keep the ruleset `Active`, which blocks every merge of `rulespec-us` when the guard cannot pass (the fail-closed state).",
            "(c) Weakening the ruleset is the last resort. It reopens the exposure that existed before the guard, now for signed v6 bytes, and three standing conditions then hold as rules for people: no pin to an encoder commit with the reader on merges while the ruleset is weak; for 30 days after the writer-off commit no enrolled owner re-runs or approves an older signing run, or a maintainer deletes those runs after saving the run list and the logs [docs D25]; and signed v6 bytes that never landed stay valid indefinitely [inferred].",
            "**Durable options, not designed:** enforced code-owner review of `.axiom/workflow-toolchain.toml` on `rulespec-us` before the ruleset is weakened, or a new signing key.",
            "- **R14. A credential that is stored where a workflow of a branch can read it.**",
            "A write collaborator pushes a branch with a workflow that reads a repository secret, or an organization secret that the repository can use; no merge and no review is needed. The collaborator then acts as the owner of that credential.",
            "**If the owner is an enrolled identity, an administrator or an organization owner, every control of this design that rests on that identity is void for that collaborator** [inferred from the documented secret semantics; not tested].",
            "That GitHub cannot restrict a repository secret to one workflow file is [inferred]: no page read offers it.",
            "**Not read:** the owner of any of these credentials, the real scopes of `BULK_ENCODE_TOKEN` (only the sentence of the README), and whether it is still valid.",
            "**This exposure exists today, and it does not depend on the guard**: today's promotion lane also rests on the approval of an enrolled owner.",
            "- `rulespec-us`, repository secrets: `BULK_ENCODE_TOKEN` (created 2026-07-06), `OPENAI_API_KEY`, `SUPABASE_DB_URL`. No environment.",
            "If a head with no guard run can merge while Actions are disabled, an administrator merges without the guard: Actions off, merge, Actions on again. Nothing blocks it [inferred].",
            "must print `true` [live read 2026-11-18: `true`]. It is a row of the health check and a precondition of `Active` (design §8).",
            "Three more acts of an administrator are outside that count:",
            '- (f) **Installing a GitHub App.** A repository administrator can install a GitHub App in the organization when the App asks for no organization permission and not for the "Repository administration" permission, and only on repositories that this administrator administers; an organization owner can restrict that [docs source D26].',
            "**Also, a repository administrator can add an App that an organization owner has already installed to the repositories that this administrator administers, whatever the App's permissions are** [docs source D26]. So removing an installed App from `rulespec-us` does not last against one administrator. It lasts only when an organization owner restricts the installation of Apps by repository administrators, which the same text names.",
            "So a `rulespec-us` administrator alone can install an App that has the right to write checks there, which is the untested case of R5. Precondition of `Active`: an organization owner reads the installations (§5 step 11), and the read is repeated at each health check (design §8);",
            "and a `rulespec-us` administrator can add an App that an owner installed to `rulespec-us` again after its removal (R12 (f)).",
            "four Apps are installed; three of them have all repositories and `checks: write`",
            "- (g) **Renaming or transferring the repository.**",
            "An organization owner can also rename a targeted branch without restriction [docs D22], which takes `main` out of a rule on `refs/heads/main`.",
            "on, no rollback of the ruleset is durable.",
            "So the recommended response to a guard problem is to repair forward, or to keep the ruleset `Active` with the writer off, which stops new v6 signing; allowlisted and eligible earlier-lane (v5) changes still pass, and only a guard that cannot pass blocks every merge (§4.4, R13).",
            "(7) **A stored token exists today that every write collaborator of `rulespec-us` can read**: the repository secret `BULK_ENCODE_TOKEN`.",
            'This does not depend on the guard; step E of §5 "Now" removes it (§4.4, R14).',
            "(8) If step X8 (c) of the experiment shows that a head can merge while Actions are disabled, a `rulespec-us` administrator can merge without the guard by disabling Actions and enabling them again (R12 (d)).",
            "(9) The experiment needs no merge of this stack. An organization owner can run it now, before any review effort (§5).",
            "On 2026-11-18 that last precondition did not hold.",
            "Step E is a third exposure, of another kind: a stored token that a workflow of any branch of `rulespec-us` can read. It is open (read 2026-11-18).",
            "An owner who sets the ruleset to `Disabled` first creates exactly the "
            "forbidden state; nothing technical prevents it.",
            "therefore serial at about 5.4 to 7.8 hours each: **about 3 to 4 a day**",
            # P2-1: the App case is a residual, with its reason.
            "**X2 does not cover a same-named check that another GitHub App "
            "publishes.** That such a check cannot satisfy the rule is [inferred, "
            "untested]. The reason: the lab creates no App.",
        ],
        "the rollback order": [
            "**Once the v6 writer is on (activation checklist step 13), this "
            "rollback comes last.** First set `V6_WRITER_ENABLED` back to `False` in "
            "a reviewed change on encoder `main`. Then confirm that no run of "
            "`targeted-signed-reencode.yml` is queued, waiting for the "
            "`production-signing` approval or in progress: `gh run list --repo "
            "TheAxiomFoundation/axiom-encode --workflow targeted-signed-reencode.yml "
            "--limit 200 --json databaseId,status,headSha,createdAt --jq '[.[] | "
            'select(.status != "completed")]\'` must print `[]`; the listing must '
            "reach back more than 30 days, so raise `--limit` if it does not.",
            "A signing run that was dispatched before the writer-off commit still has the writer on, and a re-run of it executes that commit for 30 days [docs D9]: no enrolled owner may re-run or approve one.",
            "GitHub offers no transaction between the check and the ruleset change. Before checklist step 13 the rollback is free.",
            "**After checklist step 13 this rollback is a last resort, and it is not durable.** Once the ruleset is weak, nothing technical protects `rulespec-us` from signed v6 bytes: any account with write access can move the reader pin forward again in the pull request that lands them",
            "Do not weaken the ruleset for a guard problem. Repair forward (a pin move to a fixed guard while the ruleset is `Active`), or turn the writer off and keep the ruleset `Active`: that stops new v6 signing, while allowlisted and eligible v5 changes still pass [code: `SUCCESS_VERDICTS`]; if the guard cannot pass, it blocks every merge, which is the fail-closed state.",
            "If the ruleset is weakened in spite of that, the three standing conditions of R13 hold for as long as it is weak.",
            "E. Remove the stored token `BULK_ENCODE_TOKEN` of `rulespec-us` (R14). **Open** [live read 2026-11-18]. A maintainer finds the account that owns the token, revokes it in that account's token settings, and deletes the repository secret.",
            "Deleting the secret alone does not revoke a token that someone has already read.",
            "**The existing token is revoked in every case, also when a workflow still needs one**",
            "This exposure exists today and is independent of the guard: every write collaborator of `rulespec-us` can read the secret through a workflow of a branch [docs D23; inferred]. Not read: its owner, its real scopes and its validity.",
            "**The experiment does not wait for any merge.** It depends on no pull request of this stack:",
            "It must have passed before the review of §6 at the candidate SHA (checklist step 5).",
            "5. **After change 6 (activation checklist step 13) only: the lock "
            "constant first.**",
            "The set never keeps an older pin:",
        ],
        "the Active preconditions": [
            "Do this only after Max accepts, the experiment has passed as checklist "
            "step 3 defines it, and the review of §6 has read the workflow and the "
            "bundle at the candidate SHA.",
            "Immediately before the switch, the queue must still be required",
            "The owner's reads of step 9 and the bypass read of step 10 must also "
            "hold at that moment.",
            "A passed run there stops the activation.",
            # Review round 3, P1-2 and P2-3.
            "9. **Precondition of `Active`: an organization owner verifies the "
            "runner and rename settings (R12 (c), (e)).**",
            '**If the "Runners" setting has another value, the switch to `Active` '
            "waits.**",
            "10. **Precondition of `Active`: the merge-queue ruleset of `rulespec-us` "
            "has an empty bypass list (R12 (a)).**",
            "which must print `[]`. GitHub returns the field only to a caller with "
            "write access to the ruleset [docs D7]; `null` is not a pass.",
            # Review round 4, P1-1 and P2-4.
            "11. **Precondition of `Active` and of change 6: the stored secrets and the installed Apps (R14, R12 (f)).**",
            "**None of the secrets may be a credential of an enrolled identity, of an administrator of `rulespec-us`, of an organization owner, or of an App that can write checks or contents on `rulespec-us`. An environment is no general exception**",
            "A name does not show the owner of a credential: a maintainer states it for each secret, and the names with that statement go into the design's §11.",
            "Each installed App that can write checks on `rulespec-us` is removed from it, or Max accepts it by name as a residual (R5). A removal alone does not last against a repository administrator (R12 (f)): it needs the owner's restriction of installations too.",
            "`apply-signing` and `encoder-version-bump` of axiom-encode hold secrets with a branch policy and no required reviewers [live read 2026-11-18], so their secrets are judged like repository secrets",
            "So must the reads of step 11, and `gh api repos/TheAxiomFoundation/rulespec-us/actions/permissions --jq .enabled` must print `true` (R12 (d)).",
            "a read of the stored secrets and the installed Apps: no stored secret is a credential of an enrolled identity, an administrator, an organization owner or an App that can write checks or contents (R14, R12 (f); §5 step 11).",
        ],
    },
    EXPERIMENT_DOC: {
        "status and who": [
            "Status: **designed, not run.**",
            "Until an organization owner runs it and records the results, nothing is "
            "chosen for activation.",
            "An agent must not run it, and no agent session may receive a token with "
            "the `admin:org` scope for it.",
            "It touches neither `axiom-encode` nor any `rulespec-*` repository",
            "The owner's recorded output is the evidence; the agent adds none.",
        ],
        # Review round 3, P1-2: the experiment stays inside the lab.
        "what it creates and where it acts": [
            "It creates no stored secret, no GitHub App and no key. It registers no "
            "self-hosted runner. It changes no organization setting: the only "
            "organization-level object that it creates, edits or deletes is the "
            "ruleset `guard-lab`.",
            "Every step acts on the two lab repositories and the lab ruleset only.",
            "Class: informational. The lab registers no self-hosted runner and "
            "changes no organization setting.",
            "Record and stop; do not change any organization setting from the lab.",
            "Only then delete the organization ruleset `guard-lab`: `gh api -X DELETE "
            "orgs/TheAxiomFoundation/rulesets/<id>`.",
            "No organization setting was changed, so none is set back.",
        ],
        "the lab ruleset form": [
            "| Name | `guard-lab` |",
            "| Enforcement status | `Evaluate` (X6 changes it to `Active`) |",
            "| Bypass list | empty |",
            "| Target repositories | selected: `guard-lab-target` only |",
            "| Target branches | the one ref `refs/heads/main`, as an include "
            'pattern; not the form\'s "default branch" target |',
            '| Rule | "Require workflows to pass before merging": repository '
            "`guard-lab-source`, file `.github/workflows/lab-guard.yml`, pinned to "
            "the SHA of commit A |",
        ],
        "the steps that decide": [
            "(g') fails outright: use the fallback.",
            "- **If it fails.** This is the one unverified point that fails **open**.",
            "(g') must not go `Active`, and the runner needs a reviewed change",
            "A merge to axiom-encode `main` would then change the guard at once, and "
            "`main` needs no approval for most files. (g') fails: use the fallback.",
            "For (a) and (b) any author can cause that, so (g') fails: use the "
            "fallback.",
            "It prints `marker=A`: the source has commit B since X1 and the ruleset "
            "still pins commit A, so the SHA pin must hold for the `merge_group` run "
            "too.",
            "would change the guard at merge time at once. (g') fails: use the "
            "fallback.",
            "(a) or (b) fail: the merge-time evaluation does not bound the stale "
            "green; (g') is not acceptable for `Active`.",
            "if the pull request merges before the `lab-guard` job has ended, the "
            "ruleset result is not part of the queue's requirements, and (g') is not "
            "acceptable for `Active`.",
            "any author passes the guard by opening the pull request against another "
            "branch first. (g') fails: use the fallback.",
            "- **Not covered: a check that another GitHub App publishes.**",
            "- **Pass.** A `lab-guard` run starts with `github.event_name=merge_group` "
            "while the ruleset is in `Evaluate`. That is the one pass condition.",
            # Review round 4: P2-2, P2-3 (E1d, E2i, F1i), the coordinator's
            # addition and P3-3.
            "- **When.** At any time from now. **The experiment does not depend on any pull request of the merge-guard stack being merged**: it uses only the two throwaway repositories, its own lab workflow files (§1) and the one lab ruleset.",
            "It must have passed before checklist step 5 of `docs/content-bound-merge-guard.md` §8.",
            'A step of the class "adjusting, no second run" that fails does not stop the experiment: its result is recorded, and its named change is a procedure or a decision that a reviewed change of the checklist takes up.',
            "For (c) only a repository administrator can. Record it under R12 (d) of `docs/merge-guard-key-protection.md` §4.4: an administrator of the real repository can then merge without the guard by disabling Actions and enabling them again.",
            'The consequence is a named residual risk, a row of the design\'s health check (Actions are enabled) and an item of "What Max should know".',
            "An organization-level runner took a lab job; an owner reads the runner groups before anything else happens.",
            "this experiment must not change one.",
            "A marker other than `A`: the pin does not fix the file that a merge group runs,",
            "- Not covered: a transfer of the repository to another owner.",
            "- Anything about stored credentials. The lab stores no secret.",
            "That change must also replace the trial clause of the precondition of checklist step 11, which nothing can meet in this case, and move the stop rule for the inert path on a merge group from step 8 to step 12, before any second landing.",
            "No such token is designed. It would have to meet the stored-credential precondition of checklist step 11: not a credential of an enrolled identity, an administrator or an organization owner.",
        ],
    },
}
ALL_DOC_SENTENCES = [
    (name, sentence)
    for name, groups in DOC_SENTENCES.items()
    for sentences in groups.values()
    for sentence in sentences
]


def _flat(name: str) -> str:
    """The document on one line; the ``>`` of a block quote is not text."""
    text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
    return " ".join(re.sub(r"^> ?", "", text, flags=re.M).split())


def doc_sentence_problems(name: str, flat: str) -> list[str]:
    return [
        f"{name}: {group}: {flat.count(sentence)} times, not once: {sentence!r}"
        for group, sentences in DOC_SENTENCES[name].items()
        for sentence in sentences
        if flat.count(sentence) != 1
    ]


@pytest.mark.parametrize("name", sorted(DOC_SENTENCES))
def test_the_decisive_sentences_of_each_document_are_present_once(name):
    assert doc_sentence_problems(name, _flat(name)) == []


@pytest.mark.parametrize(("name", "sentence"), ALL_DOC_SENTENCES)
def test_the_sentence_pin_reports_each_lost_and_each_doubled_sentence(name, sentence):
    flat = _flat(name)
    for changed in (flat.replace(sentence, ""), flat + " " + sentence):
        problems = doc_sentence_problems(name, changed)
        assert len(problems) == 1 and repr(sentence) in problems[0]


# The class line of every step of the runbook, by value. A step that moves
# from blocking to informational, or a sub-part that loses its class, is a
# changed value here; checklist step 3 of the design gives the same classes
# in its table, which DOC_SENTENCES pins row by row.
EXPERIMENT_CLASSES = {
    "X6": "adjusting, no second run (the change is the form of the shadow "
    "period, design checklist steps 6 and 7).",
    "X1": "blocking.",
    "X11": "adjusting, run again.",
    "X18": "adjusting, run again; no ruleset may name the real runner until it passes.",
    "X12": "adjusting, run again.",
    "X4": "adjusting, no second run (the change is a decision, not code).",
    "X2": "blocking, each part.",
    "X3": "adjusting, no second run (the change is the procedure after an approval).",
    "X16": "informational.",
    "X13": "informational.",
    "X9": "informational.",
    "X8": "blocking for (a) and (b); informational for (c).",
    "X7 and X14": "X7 informational; X14 adjusting, run again.",
    "X5": 'blocking for "a `merge_group` run starts" and for the marker `A`; '
    "adjusting, run again, for the payload fields.",
    "X15": 'blocking for (a), (b) and for "merges only after the guard job has '
    "ended\" in (c); adjusting, run again, for the identity of the group's "
    "check in (c); "
    "informational for (d).",
    "X10": "adjusting, run again.",
    "X17": "informational.",
    "X19": "informational; it can change the pin procedure.",
    "X20": "adjusting, no second run (the change is the form of the queue trial, "
    "design "
    "checklist step 8).",
    "X21": "adjusting, run again.",
    "X22": "informational.",
    "X23": "informational. The lab registers no self-hosted runner and changes "
    "no organization setting.",
    "X24": "blocking.",
}


# A pin must fail when its sentence is deleted (the test above) and also when
# the decisive word of the sentence is inverted. This is a sample of such
# inversions, one for each kind of pin: a role, a mode, a form value, a class
# row, a rollback sentence, a precondition, the approval that remains, the
# key hold. The complete list is the reviewer's 53 doc mutants, run outside
# the test suite (review round 3 of the recommendation, P1-3).
DOC_INVERSIONS = [
    (DESIGN_DOC, "3. **M** (any organization owner) Run", "3. AGENT Run"),
    (DESIGN_DOC, "1. **M** (Max) decides question 0:", "1. AGENT decides question 0:"),
    (
        DESIGN_DOC,
        "8. **M** (a `rulespec-us` administrator) Turn on",
        "8. AGENT (with the maintainer's administrator token) Turn on",
    ),
    (
        DESIGN_DOC,
        "Create the organization ruleset in `Evaluate`,",
        "Create the organization ruleset in `Active`,",
    ),
    (DESIGN_DOC, "reviewed SHA, empty bypass list.", "reviewed SHA, a bypass list."),
    (DESIGN_DOC, "| Informational | X7, X9,", "| Informational | X24, X7, X9,"),
    (DESIGN_DOC, "X1; X2 (a) to (e); X8", "X1; X2 (a); X8"),
    (
        DESIGN_DOC,
        "**This is required whenever a signed v6 manifest is outstanding or its disposition is unknown.**",
        "In every other case closing the pull request is enough.",
    ),
    (DESIGN_DOC, "So no enrolled owner may re-run", "So an enrolled owner may re-run"),
    (DESIGN_DOC, "GitHub offers no transaction", "GitHub offers a transaction"),
    (DESIGN_DOC, "and do not go on.", "and go on to step 13."),
    (
        DESIGN_DOC,
        "every candidate still needs one approval",
        "no candidate needs an approval",
    ),
    (DESIGN_DOC, "5. **M** (the other enrolled owner) reads", "5. AGENT reads"),
    (
        DESIGN_DOC,
        "2. **M** (an enrolled dispatcher; it may be the candidate's author) "
        "dispatches",
        "2. AGENT dispatches",
    ),
    (DESIGN_DOC, "none needs `admin:org`.", "the owner checks need `admin:org`."),
    (
        DESIGN_DOC,
        "No step of the experiment changes an organization setting.",
        "A step of the experiment changes an organization setting.",
    ),
    (
        KEY_PROTECTION_DOC,
        "- Enforcement status: `Evaluate`.",
        "- Enforcement status: `Active`.",
    ),
    (
        KEY_PROTECTION_DOC,
        "- Bypass list: empty. Add nothing.",
        "- Bypass list: the owners.",
    ),
    (
        KEY_PROTECTION_DOC,
        "selected repositories, `rulespec-us` only.",
        "all repositories.",
    ),
    (
        KEY_PROTECTION_DOC,
        "Nobody may store a guard App key",
        "Maintainers may store a guard App key",
    ),
    (KEY_PROTECTION_DOC, "it does not close it.**", "it closes it.**"),
    (
        KEY_PROTECTION_DOC,
        "no enrolled owner may re-run or approve one",
        "an enrolled owner may re-run one",
    ),
    (
        EXPERIMENT_DOC,
        "| Bypass list | empty |",
        "| Bypass list | organization owners |",
    ),
    (EXPERIMENT_DOC, "Record and stop; do not change", "Record it; then change"),
    (EXPERIMENT_DOC, "It prints `marker=A`:", "It prints a marker:"),
    (
        EXPERIMENT_DOC,
        "It changes no organization setting:",
        "It changes one organization setting:",
    ),
    # Review round 4: one inversion for each new kind of pin.
    (DESIGN_DOC, "leaves that gap open and depends on", "closes that gap without"),
    (
        DESIGN_DOC,
        "an anonymous read does not, and `null` is not a pass",
        "an anonymous read does not, and `null` is a pass",
    ),
    (
        DESIGN_DOC,
        "A same-named check that another GitHub App publishes is [inferred, untested]",
        "A same-named check that another GitHub App publishes is [experiment X2]",
    ),
    (
        DESIGN_DOC,
        "This order is not a durable rollback.**",
        "This order is a durable rollback.**",
    ),
    (
        DESIGN_DOC,
        "is not to weaken the ruleset.** In this order",
        "is to weaken the ruleset first.** In this order",
    ),
    (DESIGN_DOC, "is void for that collaborator [inferred", "still holds [inferred"),
    (DESIGN_DOC, "this precondition does not hold.", "this precondition holds."),
    (
        DESIGN_DOC,
        "| Adjusting, no second run | X3 (change",
        "| Adjusting, no second run | X18, X3 (change",
    ),
    (
        KEY_PROTECTION_DOC,
        "**This order is not a durable rollback.",
        "**This order is a durable rollback.",
    ),
    (
        KEY_PROTECTION_DOC,
        "An environment is no general exception**",
        "An environment with required reviewers is an exception**",
    ),
    (
        KEY_PROTECTION_DOC,
        "revokes it in that account's token settings, and deletes",
        "keeps it, or deletes",
    ),
    (EXPERIMENT_DOC, "Record it under R12 (d)", "Record it under R1"),
    (
        EXPERIMENT_DOC,
        "does not depend on any pull request",
        "depends on every pull request",
    ),
    # Review round 5 and the pin trial.
    (
        DESIGN_DOC,
        "no separate reader pin exists to move or to revert",
        "the reader pin of step 9 still exists",
    ),
    (DESIGN_DOC, "**A removal alone does not last:**", "**A removal lasts:**"),
    (
        KEY_PROTECTION_DOC,
        "is a last resort and is not durable; it has an order.**",
        "is durable; it has an order.**",
    ),
]

# Wording that review round 4 removed (P1-2, P2-2). It must not come back.
REMOVED_WORDING = [
    "safe form",
    "The durable way",
    "record it under R1.",
]


@pytest.mark.parametrize("name", sorted(DOC_SENTENCES))
def test_the_wording_that_review_round_4_removed_stays_out(name):
    flat = _flat(name)
    assert [wording for wording in REMOVED_WORDING if wording in flat] == []


@pytest.mark.parametrize(("name", "old", "new"), DOC_INVERSIONS)
def test_the_sentence_pin_reports_an_inverted_decisive_word(name, old, new):
    flat = _flat(name)
    assert flat.count(old) == 1
    assert doc_sentence_problems(name, flat) == []
    assert doc_sentence_problems(name, flat.replace(old, new)) != []


def _experiment_steps(text: str) -> list[str]:
    """The ``Xn`` ids of the experiment, from the runbook's step headings."""
    ids = []
    for heading in re.findall(r"^### (X[0-9]+(?: and X[0-9]+)?)\. ", text, re.M):
        ids += heading.split(" and ")
    return ids


def test_every_experiment_step_is_defined_recorded_classified_and_not_run():
    design = (REPOSITORY_ROOT / DESIGN_DOC).read_text(encoding="utf-8")
    runbook = (REPOSITORY_ROOT / EXPERIMENT_DOC).read_text(encoding="utf-8")
    expected = [f"X{n}" for n in range(1, 25)]
    steps = _experiment_steps(runbook)
    assert sorted(steps, key=lambda step: int(step[1:])) == expected
    assert len(steps) == len(set(steps))
    # Each step of the runbook names its class in the paragraph after its
    # heading, and the class is pinned by value (review round 3, P1-3).
    classes = {
        heading: " ".join(line.split())
        for heading, line in re.findall(
            r"^### (X[0-9]+(?: and X[0-9]+)?)\. [^\n]*\n\nClass: (.*?)\n\n",
            runbook,
            re.M | re.S,
        )
    }
    assert classes == EXPERIMENT_CLASSES
    # The runbook's record table has a row for each step (X2 has five).
    record = runbook[runbook.index("## 4. Record") : runbook.index("## 5. What")]
    rows = re.findall(r"^\| (X[0-9]+)(?: \([a-e]\))? \|", record, re.M)
    assert sorted(set(rows), key=lambda step: int(step[1:])) == expected
    assert rows.count("X2") == 5
    # Every result cell of the runbook's table is empty: nothing is recorded.
    assert all(
        line.endswith("| | |") for line in record.splitlines() if line.startswith("| X")
    )
    # The design's table: every step, each with the result "not run". Update
    # this, in the reviewed change that copies the owner's record.
    section = design[
        design.index("### The ruleset experiment (not run)") : design.index(
            "### Other open points"
        )
    ]
    results = re.findall(r"^\| (X[0-9]+) \|.*\| ([^|]+) \|$", section, re.M)
    assert [step for step, _ in results] == expected
    assert {result.strip() for _, result in results} == {"not run"}
    # The pass rule of checklist step 3 sorts every step into a class.
    flat = " ".join(design.split())
    rule = flat[
        flat.index('- **"The experiment passed" means exactly this:**') : flat.index(
            "4. AGENT, only if the results ask for it"
        )
    ]
    classified = set(re.findall(r"\bX[0-9]+\b", rule))
    assert classified == set(expected)
    for heading in (
        "| Blocking |",
        "| Adjusting, run again |",
        "| Adjusting, no second run |",
        "| Informational |",
    ):
        assert rule.count(heading) == 1
    # Review round 4, P2-1: a step whose named change is a procedure or a
    # decision cannot pass in a second run, so its class asks for none.
    table = rule[rule.index("| Blocking |") : rule.index("| Informational |")]
    again, no_second = table.split("| Adjusting, no second run |")
    again = again[again.index("| Adjusting, run again |") :]
    assert set(re.findall(r"\bX[0-9]+\b", no_second)) == {"X3", "X4", "X6", "X20"}
    assert set(re.findall(r"\bX[0-9]+\b", again)) == {
        "X5",
        "X10",
        "X11",
        "X12",
        "X14",
        "X15",
        "X18",
        "X21",
    }
    no_second_classes = {
        step
        for heading, line in EXPERIMENT_CLASSES.items()
        if "no second run" in line
        for step in heading.split(" and ")
    }
    assert no_second_classes == {"X3", "X4", "X6", "X20"}


def test_the_ruleset_bodies_target_main_by_name_and_start_in_evaluate():
    # Review round 1 of the recommendation, P1-3 (a): ``~DEFAULT_BRANCH``
    # follows a changed default branch; the runner judges ``main`` only.
    bodies = []
    for name in (KEY_PROTECTION_DOC, EXPERIMENT_DOC):
        text = (REPOSITORY_ROOT / name).read_text(encoding="utf-8")
        for block in re.findall(r"```json\n(.*?)```", text, re.S):
            bodies.append(json.loads(block))
    assert len(bodies) == 2
    for body in bodies:
        assert body["conditions"]["ref_name"] == {
            "include": ["refs/heads/main"],
            "exclude": [],
        }
        assert body["enforcement"] == "evaluate" and body["bypass_actors"] == []
        assert body["target"] == "branch"
        assert set(body) == {
            "name",
            "target",
            "enforcement",
            "bypass_actors",
            "conditions",
            "rules",
        }
        assert set(body["conditions"]) == {"repository_name", "ref_name"}
        assert body["conditions"]["repository_name"]["exclude"] == []
        (rule,) = body["rules"]
        (entry,) = rule["parameters"]["workflows"]
        assert rule["type"] == "workflows" and set(entry) == {
            "repository_id",
            "path",
            "sha",
        }
    real, lab = bodies
    assert real["conditions"]["repository_name"]["include"] == ["rulespec-us"]
    assert real["name"] == "content-bound-merge-guard" and lab["name"] == "guard-lab"
    assert real["rules"][0]["parameters"]["workflows"][0]["path"] == (
        ".github/workflows/content-bound-merge-guard.yml"
    )
    # The source repository is axiom-encode, by id; the pin is a placeholder
    # for a commit SHA, never a branch.
    real_entry = real["rules"][0]["parameters"]["workflows"][0]
    assert real_entry["repository_id"] == 1125775893
    assert real_entry["sha"] == "<reviewed 40-hex commit SHA>"
    assert lab["rules"][0]["parameters"]["workflows"][0]["path"] == (
        ".github/workflows/lab-guard.yml"
    )
    assert lab["conditions"]["repository_name"]["include"] == ["guard-lab-target"]


def test_the_documents_state_the_waits_and_limits_of_the_runner_as_built():
    # The numbers the documents give for the wait come from the constants.
    assert runner.CI_WAIT_SECONDS == runner.MERGE_GROUP_WAIT_SECONDS == 5 * 3600
    design = _flat(DESIGN_DOC)
    key = _flat(KEY_PROTECTION_DOC)
    assert "for at most 18,000 seconds, five hours (`CI_WAIT_SECONDS`)" in design
    assert "**Job timeout.** `timeout-minutes` is 345." in design
    assert "`MAX_ENFORCE_REQUESTS` is 450" in design
    assert (
        "it waits for that check, at most 5 hours, with a poll every 180 seconds"
        in (key)
    )
    assert "The job timeout is 345 minutes" in key
    for text in (design, key, _flat(EXPERIMENT_DOC)):
        assert "60-minute wait" not in text and "at most 60 minutes" not in text


# --- the enforcing runner (change 3b) -------------------------------------------
#
# ``enforce`` judges the pull request or merge group of one Actions event. Its
# exit code is the job's conclusion: 0 only for a pass or not_applicable
# verdict that is bound to the event's head.

ENFORCE_ENV = {
    "GITHUB_REPOSITORY": REPO,
    "GITHUB_EVENT_NAME": "pull_request",
    runner.TOKEN_ENV: TOKEN,
}
V5_LANE = {P: PRIMARY, T: COMPANION, M: v5_manifest(P, T)}
CHECK_LISTING = f"{ROOT}/commits/{H}{CHECKS}"


def pr_event(head=H, number=42, repository=REPO, **changes):
    return {
        "action": "synchronize",
        "number": number,
        "pull_request": {"number": number, "head": {"sha": head}},
        "repository": {"full_name": repository},
    } | changes


def group_event(head=G, base=B, number=42, repository=REPO, **changes):
    return {
        "action": "checks_requested",
        "merge_group": {
            "head_sha": head,
            "base_sha": base,
            "head_ref": f"refs/heads/gh-readonly-queue/main/pr-{number}-{base}",
            "base_ref": "refs/heads/main",
        }
        | changes,
        "repository": {"full_name": repository},
    }


GROUP_RUN = "999"
GROUP_CHECK_LISTING = f"{ROOT}/commits/{G}{CHECKS}"
GROUP_RUN_PATH = f"{ROOT}/actions/runs/{GROUP_RUN}"
SOURCE = "TheAxiomFoundation/axiom-encode"


def group_ci(world, *, check=None, run=None, head=G):
    """The merge group's own run of the required check on ``head``: green.

    ``check`` and ``run`` are changes to the check run and to its workflow
    run, the shape of a ``merge_group`` run of ``repository-checks.yml``.
    """
    world.api[f"{ROOT}/commits/{head}{CHECKS}"] = {
        "total_count": 1,
        "check_runs": [
            {
                "id": 2,
                "name": "validate / validate",
                "app": {"id": 15368},
                "head_sha": head,
                "status": "completed",
                "conclusion": "success",
                "details_url": (
                    f"https://github.com/{REPO}/actions/runs/{GROUP_RUN}/job/2"
                ),
            }
            | (check or {})
        ],
    }
    world.api[GROUP_RUN_PATH] = {
        "id": int(GROUP_RUN),
        "event": "merge_group",
        "path": ".github/workflows/repository-checks.yml",
        "head_sha": head,
        "run_attempt": 1,
        "status": "completed",
        "conclusion": "success",
        "repository": {"full_name": REPO},
    } | (run or {})
    return world


def queued(world=None, **changes):
    """A merge group of one pull request whose own required check is green."""
    return group_ci(queue(build() if world is None else world, **changes))


def source_event(name="pull_request", repository=SOURCE, base=SOURCE):
    """An event of the encoder repository itself, as the inert path needs it."""
    if name == "merge_group":
        return group_event(repository=repository)
    event = pr_event(repository=repository)
    event["pull_request"]["base"] = {"repo": {"full_name": base}}
    return event


class Time:
    """An injected clock and sleep: sleeping moves the clock, nothing waits."""

    def __init__(self, advance=True):
        self.now = 1000.0
        self.sleeps = []
        self.advance = advance

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        if self.advance:
            self.now += seconds


def judge(world, event=..., name="pull_request", repository=REPO, time=None):
    """``runner.enforce`` over a fake world; no test sleeps or reads a clock."""
    time = time or Time()
    before = len(world.calls)
    report, code = runner.enforce(
        name,
        repository,
        pr_event() if event is ... else event,
        fetch=world.fetch,
        sleep=time.sleep,
        clock=time.clock,
    )
    assert report["schema"] == runner.ENFORCE_REPORT_SCHEMA
    assert report["requests"] == len(world.calls) - before
    assert code in (0, 1)
    return report, code


def reason_of(report):
    return (report["verdict"]["verdict"], report["verdict"]["reason"])


def pending_then(world, *, polls, then=None, head=H):
    """The required check on ``head`` is pending for ``polls`` reads, then ``then``.

    ``then`` are the changes to the green check run; ``None`` keeps it green.
    """
    listing = f"{ROOT}/commits/{head}{CHECKS}"
    world.ci(head, **(then or {}))
    settled = world.api[listing]
    world.ci(head, status="in_progress", conclusion=None)
    pending = world.api[listing]
    reads = []

    def read():
        reads.append(1)
        return pending if len(reads) <= polls else settled

    world.api[listing] = read
    return reads


# The exit code table: success only for the listed verdict and reason pairs.


def test_pass_and_not_applicable_end_the_job_with_success():
    for world, expected in (
        (build(), ("pass", guard.R_PASS)),
        (plain({"README.md": b"x\n"}), ("not_applicable", guard.R_NO_GUARDED_CONTENT)),
        (plain(V5_LANE), ("not_applicable", guard.R_EARLIER_LANE)),
    ):
        report, code = judge(world)
        assert (reason_of(report), code) == (expected, 0), report
        assert "error" not in report and "inert" not in report
        assert report["evaluations"] == 1 and report["ci_polls"] == 0
        assert report["pull_request"] == 42 and report["event_head"] == H


def test_a_fail_verdict_ends_the_job_with_failure():
    world = build()
    world.reviews([])
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_NO_APPROVAL), 1)
    # An unapproved v6 unit fails at step 5, before the evaluator looks at CI:
    # it does not wait.
    assert report["evaluations"] == 1 and report["ci_polls"] == 0
    assert not any("check-runs" in path for path in world.calls)


GUARD_REASONS = sorted(
    value
    for name, value in vars(guard).items()
    if name.startswith("R_") and isinstance(value, str)
)
SUCCESS = {
    ("pass", guard.R_PASS),
    ("not_applicable", guard.R_NO_GUARDED_CONTENT),
    ("not_applicable", guard.R_EARLIER_LANE),
}


def test_the_success_set_is_the_three_reasons_of_the_lanes_that_are_on():
    assert runner.SUCCESS_VERDICTS == SUCCESS
    assert len(GUARD_REASONS) > 40 and guard.R_CI in GUARD_REASONS
    # The proposed infrastructure lane is off; its success reasons are not
    # success reasons of the job until a reviewed change decides the lane.
    assert guard.INFRASTRUCTURE_LANE_ENABLED is False
    for verdict, reason in (
        ("pass", guard.R_CONTROL_PLANE_PASS),
        ("not_applicable", guard.R_INFRASTRUCTURE),
    ):
        assert reason in GUARD_REASONS
        assert (verdict, reason) not in runner.SUCCESS_VERDICTS


def control_plane_world():
    world = plain({".github/workflows/x.yml": b"x\n"})
    world.publish(H, B, [commit(H)])
    world.reviews([review(11, MAX, H)])
    return world


def test_a_verdict_of_the_switched_off_infrastructure_lane_fails_the_job(monkeypatch):
    # With the lane switched on in the test only, the evaluator itself gives
    # the lane's two success verdicts. The job fails for both.
    monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", True)
    report, code = judge(control_plane_world())
    assert reason_of(report) == ("pass", guard.R_CONTROL_PLANE_PASS)
    assert code == 1
    stand_aside = guard.Verdict(
        guard.NOT_APPLICABLE, guard.R_INFRASTRUCTURE, "", {"head": H}
    )
    monkeypatch.setattr(guard, "evaluate", lambda *_a, **_k: stand_aside)
    report, code = judge(build())
    assert reason_of(report) == ("not_applicable", guard.R_INFRASTRUCTURE)
    assert code == 1


@pytest.mark.parametrize("reason", [*GUARD_REASONS, "a_reason_added_later"])
@pytest.mark.parametrize(
    "verdict", ["pass", "not_applicable", "fail", "neutral", "success", "", None]
)
def test_exit_code_for_every_verdict_and_reason(monkeypatch, verdict, reason):
    judged = guard.Verdict(verdict, reason, "", {"pull_request": 42, "head": H})
    monkeypatch.setattr(guard, "evaluate", lambda *_a, **_k: judged)
    world = build()
    if reason == guard.R_CI:
        world.ci(H)  # settled at once; the second evaluation is the same fake
    report, code = judge(world)
    assert code == (0 if (verdict, reason) in SUCCESS else 1)
    assert report["verdict"]["reason"] == reason


@pytest.mark.parametrize("event_name", ["pull_request", "merge_group"])
def test_an_unforeseen_error_ends_the_job_with_failure(monkeypatch, event_name):
    def explode(*_args, **_kwargs):
        raise RuntimeError("unforeseen " + TOKEN)

    monkeypatch.setattr(guard, "evaluate", explode)
    monkeypatch.setattr(guard, "evaluate_merge_group", explode)
    event = pr_event() if event_name == "pull_request" else group_event()
    report, code = judge(queued(), event, event_name)
    assert code == 1 and report["verdict"] is None
    assert report["error"] == "unforeseen error: RuntimeError"


# The event head binding.


def test_a_verdict_for_another_head_than_the_event_head_fails():
    # The pull request is at H and would pass; this run's event (and so its
    # check) belongs to the older commit C1.
    world = build()
    assert world.evaluate().verdict == guard.PASS
    report, code = judge(world, pr_event(head=C1))
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)
    assert report["verdict"]["evidence"] == {
        "pull_request": 42,
        "event_head": C1,
        "evaluated_head": H,
        "evaluated_verdict": "pass",
        "evaluated_reason": guard.R_PASS,
    }
    assert report["evaluations"] == 1


def test_a_rerun_of_the_run_of_an_older_head_fails():
    # The run started for H. Since then the pull request moved to H2, which
    # is approved. A re-run replays the event of H.
    world = build()
    advance(world)
    world.reviews([review(12, MAX, H2)])
    assert world.evaluate().verdict == guard.PASS
    report, code = judge(world, pr_event(head=H))
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)
    report, code = judge(world, pr_event(head=H2))
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)


def test_a_stand_aside_verdict_for_another_head_fails_too():
    report, code = judge(plain({"README.md": b"x\n"}), pr_event(head=C1))
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)


def test_a_fail_for_another_head_is_reported_as_stale_and_does_not_wait():
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    time = Time()
    report, code = judge(world, pr_event(head=C1), time=time)
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)
    assert report["verdict"]["evidence"]["evaluated_reason"] == guard.R_CI
    assert time.sleeps == [] and report["ci_polls"] == 0


@pytest.mark.parametrize("verdict", ["pass", "not_applicable"])
@pytest.mark.parametrize("evidence", [{}, {"pull_request": 42}, {"head": None}])
def test_a_success_that_names_no_head_fails(monkeypatch, verdict, evidence):
    reason = guard.R_PASS if verdict == "pass" else guard.R_NO_GUARDED_CONTENT
    judged = guard.Verdict(verdict, reason, "", evidence)
    monkeypatch.setattr(guard, "evaluate", lambda *_a, **_k: judged)
    report, code = judge(build())
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)
    assert report["verdict"]["evidence"]["evaluated_head"] is None


def test_a_run_with_an_error_never_ends_with_success(monkeypatch):
    # The exit rule itself: a success verdict in a run that recorded an error
    # is a failed job.
    def errored(name, payload, fetch, sleep, clock, report):
        report["error"] = "a step of the run failed"
        return guard.Verdict(guard.PASS, guard.R_PASS, "", {"head": H})

    monkeypatch.setattr(runner, "_enforce_pull_request", errored)
    report, code = judge(build())
    assert reason_of(report) == ("pass", guard.R_PASS)
    assert code == 1 and report["error"] == "a step of the run failed"


def test_a_fail_before_the_evaluator_reads_a_head_keeps_its_reason():
    world = build()
    world.api[ROOT + "/pulls/42"]["state"] = "closed"
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_PR_IDENTITY), 1)


def test_the_head_moving_during_the_ci_wait_fails():
    # First evaluation: H, required check pending. During the wait the pull
    # request moves to an approved, green H2. The second evaluation passes
    # for H2, which is not the head of this run's event.
    world = build()
    time = Time()
    settled = dict(world.api)
    world.ci(H, status="in_progress", conclusion=None)

    def move(seconds):
        Time.sleep(time, seconds)
        world.api.update(settled)  # the old head's check has settled
        advance(world)
        world.reviews([review(12, MAX, H2)])

    report, code = runner.enforce(
        "pull_request",
        REPO,
        pr_event(),
        fetch=world.fetch,
        sleep=move,
        clock=time.clock,
    )
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)
    assert report["verdict"]["evidence"]["evaluated_verdict"] == "pass"
    assert report["evaluations"] == 2


# Waiting for the required check.


@pytest.mark.parametrize(
    ("world", "expected"),
    [
        (lambda: build(), ("pass", guard.R_PASS)),
        (lambda: plain(V5_LANE), ("not_applicable", guard.R_EARLIER_LANE)),
    ],
)
def test_a_pending_required_check_is_waited_for_then_judged_once(world, expected):
    world = world()
    # Pending at the first evaluation and at three polls; green at the fourth.
    pending_then(world, polls=4)
    time = Time()
    report, code = judge(world, time=time)
    assert (reason_of(report), code) == (expected, 0)
    assert report["evaluations"] == 2
    assert report["ci_polls"] == 4
    assert time.sleeps == [runner.CI_POLL_SECONDS] * 3
    assert report["waited_seconds"] == 3 * runner.CI_POLL_SECONDS


def test_a_required_check_that_stays_pending_times_out_and_fails():
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    time = Time()
    report, code = judge(world, time=time)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert report["evaluations"] == 2
    # One poll at the start and one after each sleep, until the wait is over.
    assert report["ci_polls"] == runner.MAX_CI_POLLS == 101
    assert report["waited_seconds"] == runner.CI_WAIT_SECONDS == 18000
    assert sum(time.sleeps) == runner.CI_WAIT_SECONDS
    assert set(time.sleeps) == {runner.CI_POLL_SECONDS} == {180}
    assert report["requests"] <= runner.MAX_ENFORCE_REQUESTS
    assert "error" not in report


def test_a_full_pull_request_wait_and_both_evaluations_fit_under_the_request_cap(
    monkeypatch,
):
    # The worst honest pull-request run (review round 2, P2-1 and P3-3): the
    # required check is pending at the first evaluation and at 100 polls, and
    # completed at the last poll the wait allows. Then the second evaluation
    # runs whole. A poll of this wait is one read.
    world = build()
    reads = pending_then(world, polls=runner.MAX_CI_POLLS)
    time = Time()
    started = []
    evaluate = guard.evaluate

    def counting(*args, **kwargs):
        started.append(len(world.calls))
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(guard, "evaluate", counting)
    report, code = judge(world, time=time)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    # Each evaluation is counted by itself: the first ends at the CI rule,
    # the polls follow, the second runs whole.
    first_start, second_start = started
    assert first_start == 0
    assert second_start - 101 * runner.CI_POLL_READS == 16
    assert report["requests"] - second_start == 25
    assert report["evaluations"] == 2
    assert report["ci_polls"] == runner.MAX_CI_POLLS == 101
    assert report["waited_seconds"] == runner.CI_WAIT_SECONDS
    # The listing is read once by the first evaluation, once by each poll
    # and twice by the second evaluation (its check and its recheck).
    assert len(reads) == 1 + 101 * runner.CI_POLL_READS + 2
    first, second = FIRST_EVALUATION_READS_UNTIL_CI, SECOND_EVALUATION_READS
    assert (first, second) == (16, 25)
    assert report["requests"] == first + 101 * runner.CI_POLL_READS + second == 142
    assert report["requests"] <= 2 * 30 + 101 == 161 <= runner.MAX_ENFORCE_REQUESTS


def test_the_wait_is_bounded_by_polls_when_the_clock_does_not_move():
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    time = Time(advance=False)
    report, code = judge(world, time=time)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert report["ci_polls"] == runner.MAX_CI_POLLS
    assert len(time.sleeps) == runner.MAX_CI_POLLS - 1
    assert report["evaluations"] == 2


def test_the_wait_is_bounded_by_the_clock_when_sleeps_are_long():
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    time = Time()

    def oversleep(seconds):
        time.sleeps.append(seconds)
        time.now += runner.CI_WAIT_SECONDS

    report, code = runner.enforce(
        "pull_request",
        REPO,
        pr_event(),
        fetch=world.fetch,
        sleep=oversleep,
        clock=time.clock,
    )
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert report["ci_polls"] == 2 and len(time.sleeps) == 1


@pytest.mark.parametrize(
    "then",
    [
        {"conclusion": "failure"},
        {"conclusion": "cancelled"},
        {"conclusion": "skipped"},
        {"conclusion": "neutral"},
        {"app": {"id": 1}},
    ],
)
def test_a_pending_required_check_that_ends_red_fails(then):
    world = build()
    pending_then(world, polls=3, then=then)
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert report["evaluations"] == 2 and report["ci_polls"] == 3


def test_case_22_green_on_an_older_head_only_fails_after_one_poll():
    # The listing for H holds a completed run of another head: settled, and
    # the second evaluation refuses it again.
    world = build()
    world.ci(H, head_sha=C1)
    time = Time()
    report, code = judge(world, time=time)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert (report["evaluations"], report["ci_polls"], time.sleeps) == (2, 1, [])


def test_case_22_no_required_check_on_the_head_is_waited_for():
    world = build()
    world.api[CHECK_LISTING] = {"total_count": 0, "check_runs": []}
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert report["ci_polls"] == runner.MAX_CI_POLLS


def test_one_pending_run_among_completed_runs_is_waited_for():
    world = build()
    green = world.api[CHECK_LISTING]["check_runs"][0]
    world.api[CHECK_LISTING] = {
        "total_count": 2,
        "check_runs": [green, green | {"id": 2, "status": "queued"}],
    }
    report, _code = judge(world)
    assert report["ci_polls"] == runner.MAX_CI_POLLS


def test_never_more_than_two_evaluations(monkeypatch):
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    evaluations = []
    evaluate = guard.evaluate

    def counted(*args, **kwargs):
        evaluations.append(args)
        return evaluate(*args, **kwargs)

    monkeypatch.setattr(guard, "evaluate", counted)
    report, code = judge(world)
    assert code == 1 and len(evaluations) == 2 == report["evaluations"]


def test_the_wait_never_produces_a_verdict():
    # The check settles green, and meanwhile the approval was dismissed: the
    # verdict is the second evaluation's, not "the check is green".
    world = build()
    pending_then(world, polls=2)
    time = Time()

    def dismiss(seconds):
        Time.sleep(time, seconds)
        world.reviews([review(11, MAX, H, state="DISMISSED")])

    report, code = runner.enforce(
        "pull_request",
        REPO,
        pr_event(),
        fetch=world.fetch,
        sleep=dismiss,
        clock=time.clock,
    )
    assert (reason_of(report), code) == (("fail", guard.R_NO_APPROVAL), 1)


@pytest.mark.parametrize(
    "listing",
    [
        None,
        [],
        {"message": "API rate limit exceeded"},
        {"total_count": 2, "check_runs": [{"status": "completed"}]},
        {"total_count": "1", "check_runs": [{"status": "completed"}]},
        {"total_count": 1, "check_runs": ["completed"]},
        {"total_count": 101, "check_runs": [{"status": "completed"}] * 101},
        {"check_runs": [{"status": "completed"}]},
    ],
)
def test_a_poll_that_cannot_be_read_whole_keeps_fail_and_is_a_run_error(listing):
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    pending = world.api[CHECK_LISTING]
    reads = []

    def read():
        reads.append(1)
        return pending if len(reads) == 1 else listing

    world.api[CHECK_LISTING] = read
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert "required check listing is incomplete" in report["error"]
    assert report["evaluations"] == 1


def test_a_poll_that_raises_keeps_fail_and_is_a_run_error():
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    pending = world.api[CHECK_LISTING]
    reads = []

    def read():
        reads.append(1)
        if len(reads) == 1:
            return pending
        raise runner.RunnerError("GitHub GET request failed: HTTPError")

    world.api[CHECK_LISTING] = read
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert "HTTPError" in report["error"] and report["evaluations"] == 1


# The merge group: evaluate as late as possible. The run waits until the
# group's own required check has completed with success, then evaluates once.

FIRST_EVALUATION_READ = ROOT + "/pulls/42"


def group_pending_then(world, *, polls, check=None, run=None):
    """The group's check is in progress for ``polls`` reads, then as given."""
    group_ci(world, check=check, run=run)
    settled = world.api[GROUP_CHECK_LISTING]
    group_ci(world, check={"status": "in_progress", "conclusion": None})
    pending = world.api[GROUP_CHECK_LISTING]
    group_ci(world, check=check, run=run)
    reads = []

    def read():
        reads.append(1)
        return pending if len(reads) <= polls else settled

    world.api[GROUP_CHECK_LISTING] = read
    return reads


def judge_group(world, time=None, sleep=None):
    time = time or Time()
    report, code = runner.enforce(
        "merge_group",
        REPO,
        group_event(),
        fetch=world.fetch,
        sleep=sleep or time.sleep,
        clock=time.clock,
    )
    assert report["requests"] == len(world.calls)
    return report, code


def test_a_merge_group_is_evaluated_only_after_its_required_check_completes():
    world = queued()
    group_pending_then(world, polls=3)
    time = Time()
    record = []
    fetch = world.fetch

    def recording(path):
        record.append(("read", path))
        return fetch(path)

    def sleeping(seconds):
        record.append(("sleep", seconds))
        time.sleep(seconds)

    world.fetch = recording
    report, code = judge_group(world, time, sleeping)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    # Three pending polls with a sleep after each, the fourth poll and its
    # read of the workflow run, and only then the first read of the
    # evaluation. No sleep follows it. After the last read of the evaluation
    # the group's check and its workflow run are read once more (review of
    # head 979124f10: the success must still be the same attempt); those two
    # reads are the last two of the run.
    poll, sleep = ("read", GROUP_CHECK_LISTING), ("sleep", 180)
    assert record[:9] == [
        poll,
        sleep,
        poll,
        sleep,
        poll,
        sleep,
        poll,
        ("read", GROUP_RUN_PATH),
        ("read", FIRST_EVALUATION_READ),
    ]
    assert record[-2:] == [poll, ("read", GROUP_RUN_PATH)]
    evaluation = record[8:-2]
    assert len(evaluation) == 29
    assert poll not in evaluation and sleep not in record[9:]
    assert ("read", GROUP_RUN_PATH) not in evaluation
    assert report["evaluations"] == 1 and report["ci_polls"] == 4
    assert report["group_ci_rechecks"] == 1
    assert report["waited_seconds"] == 3 * runner.MERGE_GROUP_POLL_SECONDS == 540


def test_a_merge_group_whose_check_is_complete_is_evaluated_at_once():
    world = queued()
    time = Time()
    report, code = judge_group(world, time)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert world.calls[:3] == [
        GROUP_CHECK_LISTING,
        GROUP_RUN_PATH,
        FIRST_EVALUATION_READ,
    ]
    assert time.sleeps == [] and report["ci_polls"] == 1


def test_a_dismissal_while_the_merge_group_validates_fails_the_job():
    # The approval is there when the group is created: an evaluation at that
    # time passes. It is dismissed while the group's check runs.
    world = queued()
    assert guard.evaluate_merge_group(NAME, G, B, 42, fetch=world.fetch).verdict == (
        guard.PASS
    )
    world.calls.clear()
    group_pending_then(world, polls=2)
    time = Time()

    def dismiss(seconds):
        time.sleep(seconds)
        world.reviews([review(11, MAX, H, state="DISMISSED")])

    report, code = judge_group(world, time, dismiss)
    assert (reason_of(report), code) == (("fail", guard.R_NO_APPROVAL), 1)
    assert report["evaluations"] == 1 and report["ci_polls"] == 3


def test_a_change_request_while_the_merge_group_validates_fails_the_job():
    world = queued()
    group_pending_then(world, polls=1)
    time = Time()

    def object_(seconds):
        time.sleep(seconds)
        world.reviews(
            [review(11, MAX, H), review(12, PAVEL, H, state="CHANGES_REQUESTED")]
        )

    report, code = judge_group(world, time, object_)
    assert (report["verdict"]["verdict"], code) == ("fail", 1)
    assert report["evaluations"] == 1


def no_evaluation(world, report, code):
    assert code == 1 and report["verdict"] is None and report["evaluations"] == 0
    assert FIRST_EVALUATION_READ not in world.calls
    assert not any("/pulls/" in path for path in world.calls)


def test_a_merge_group_check_that_never_completes_times_out_without_a_verdict():
    world = queued()
    group_ci(world, check={"status": "in_progress", "conclusion": None})
    time = Time()
    report, code = judge_group(world, time)
    no_evaluation(world, report, code)
    assert "did not complete within the wait" in report["error"]
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS == 101
    assert report["waited_seconds"] == runner.MERGE_GROUP_WAIT_SECONDS == 18000
    assert sum(time.sleeps) == runner.MERGE_GROUP_WAIT_SECONDS
    assert set(time.sleeps) == {runner.MERGE_GROUP_POLL_SECONDS} == {180}
    assert report["requests"] == 101


def test_a_merge_group_with_no_required_check_times_out_without_a_verdict():
    world = queued()
    world.api[GROUP_CHECK_LISTING] = {"total_count": 0, "check_runs": []}
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "did not complete within the wait" in report["error"]
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS


def test_the_merge_group_wait_is_bounded_by_polls_when_the_clock_does_not_move():
    world = queued()
    group_ci(world, check={"status": "queued", "conclusion": None})
    time = Time(advance=False)
    report, code = judge_group(world, time)
    no_evaluation(world, report, code)
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS
    assert len(time.sleeps) == runner.MAX_MERGE_GROUP_POLLS - 1


def test_the_merge_group_wait_is_bounded_by_the_clock_when_sleeps_are_long():
    world = queued()
    group_ci(world, check={"status": "queued", "conclusion": None})
    time = Time()

    def oversleep(seconds):
        time.sleeps.append(seconds)
        time.now += runner.MERGE_GROUP_WAIT_SECONDS

    report, code = judge_group(world, time, oversleep)
    no_evaluation(world, report, code)
    assert report["ci_polls"] == 2 and len(time.sleeps) == 1


def test_one_pending_group_check_among_completed_ones_is_waited_for():
    world = queued()
    green = world.api[GROUP_CHECK_LISTING]["check_runs"][0]
    world.api[GROUP_CHECK_LISTING] = {
        "total_count": 2,
        "check_runs": [green, green | {"id": 3, "status": "queued"}],
    }
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS
    # No listed check was read further while one of them was pending.
    assert GROUP_RUN_PATH not in world.calls


def test_a_completed_group_check_whose_run_is_not_completed_is_waited_for():
    # A check that says "completed" while its workflow run is in progress:
    # any workflow of the repository with ``checks: write`` can create one.
    world = queued()
    group_ci(world, run={"status": "in_progress", "conclusion": None})
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "did not complete within the wait" in report["error"]
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS
    # Two reads for every poll, and still under the request cap.
    assert report["requests"] == 2 * runner.MAX_MERGE_GROUP_POLLS == 202
    assert report["requests"] <= runner.MAX_ENFORCE_REQUESTS


SECOND_GROUP_RUN = "44150203164"
SECOND_GROUP_RUN_PATH = f"{ROOT}/actions/runs/{SECOND_GROUP_RUN}"


def two_group_checks(world, *, second_run=None):
    """Two listed checks on the group head, each with its own workflow run."""
    group_ci(world)
    first = world.api[GROUP_CHECK_LISTING]["check_runs"][0]
    second = first | {
        "id": 3,
        "details_url": (
            f"https://github.com/{REPO}/actions/runs/{SECOND_GROUP_RUN}/job/3"
        ),
    }
    world.api[GROUP_CHECK_LISTING] = {"total_count": 2, "check_runs": [first, second]}
    world.api[SECOND_GROUP_RUN_PATH] = (
        world.api[GROUP_RUN_PATH] | {"id": int(SECOND_GROUP_RUN)} | (second_run or {})
    )
    return world


def test_the_costliest_merge_group_wait_and_the_evaluation_fit_under_the_request_cap():
    # Review round 2, P3-3. The costliest poll: two listed checks, both
    # completed; the first one's workflow run is completed and green, the
    # second one's is in progress. That is three reads: the listing and two
    # workflow runs. Every poll of a full wait costs that, the last poll finds
    # the second run completed, and the evaluation follows.
    world = two_group_checks(queued())
    done = world.api[SECOND_GROUP_RUN_PATH]
    pending = done | {"status": "in_progress", "conclusion": None}
    reads = []

    def run():
        reads.append(1)
        return pending if len(reads) < runner.MAX_MERGE_GROUP_POLLS else done

    world.api[SECOND_GROUP_RUN_PATH] = run
    time = Time()
    report, code = judge_group(world, time)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["evaluations"] == 1
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS == 101
    assert report["waited_seconds"] == runner.MERGE_GROUP_WAIT_SECONDS
    assert runner.MERGE_GROUP_POLL_READS == 3
    # 101 polls and the one read after the evaluation, three reads each.
    assert runner.GROUP_CI_RECHECK_READS == 3 and report["group_ci_rechecks"] == 1
    assert world.calls.count(GROUP_CHECK_LISTING) == 102
    assert world.calls.count(GROUP_RUN_PATH) == 102
    assert len(reads) == 102
    assert report["requests"] == 3 * 101 + 29 + 3 == 335
    assert report["requests"] <= 3 * 101 + 30 + 3 == 336
    assert 336 <= runner.MAX_ENFORCE_REQUESTS


def test_the_costliest_merge_group_wait_that_never_completes_stays_under_the_cap():
    world = two_group_checks(
        queued(), second_run={"status": "in_progress", "conclusion": None}
    )
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "did not complete within the wait" in report["error"]
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS
    assert report["requests"] == 3 * 101 == 303 <= runner.MAX_ENFORCE_REQUESTS


@pytest.mark.parametrize("pending", [False, True])
def test_more_listed_group_checks_than_one_poll_reads_is_an_error(pending):
    # The bound that makes three reads the most one poll can cost. Three
    # listed checks are an error of the run at the first poll, completed or
    # not, with no workflow run read and no evaluation.
    world = two_group_checks(queued())
    listing = world.api[GROUP_CHECK_LISTING]
    third = listing["check_runs"][1] | {"id": 4}
    if pending:
        third |= {"status": "in_progress", "conclusion": None}
    world.api[GROUP_CHECK_LISTING] = {
        "total_count": 3,
        "check_runs": [*listing["check_runs"], third],
    }
    assert len(listing["check_runs"]) == runner.MAX_GROUP_CHECK_RUNS == 2
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "more checks with the required name" in report["error"]
    assert (report["ci_polls"], report["requests"]) == (1, 1)
    assert GROUP_RUN_PATH not in world.calls


def test_a_full_merge_group_wait_and_the_evaluation_fit_under_the_request_cap():
    # The usual full run: one listed check, so every poll makes two reads,
    # the last poll finds the run completed, and the evaluation follows.
    world = queued()
    group_ci(world, run={"status": "in_progress", "conclusion": None})
    pending = world.api[GROUP_RUN_PATH]
    group_ci(world)
    done = world.api[GROUP_RUN_PATH]
    reads = []

    def run():
        reads.append(1)
        return pending if len(reads) < runner.MAX_MERGE_GROUP_POLLS else done

    world.api[GROUP_RUN_PATH] = run
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["ci_polls"] == runner.MAX_MERGE_GROUP_POLLS
    assert report["requests"] == 2 * 101 + 29 + 2 == 233
    assert report["requests"] <= runner.MAX_ENFORCE_REQUESTS


def test_the_request_cap_bounds_the_merge_group_polls(monkeypatch):
    world = queued()
    group_ci(world, run={"status": "in_progress", "conclusion": None})
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", 40)
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert report["requests"] == 40 and "request cap" in report["error"]


@pytest.mark.parametrize(
    ("check", "run"),
    [
        ({"conclusion": "failure"}, {"conclusion": "failure"}),
        ({"conclusion": "cancelled"}, {"conclusion": "cancelled"}),
        ({"conclusion": "skipped"}, None),
        ({"conclusion": "neutral"}, None),
        ({"conclusion": None}, None),
        (None, {"conclusion": "failure"}),
        (None, {"conclusion": None}),
    ],
)
def test_a_merge_group_whose_required_check_did_not_succeed_fails_unevaluated(
    check, run
):
    world = queued()
    group_ci(world, check=check, run=run)
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "did not succeed" in report["error"]
    assert report["ci_polls"] == 1


OTHER_RUN = f"https://github.com/{REPO}/actions/runs/{GROUP_RUN}"


@pytest.mark.parametrize(
    ("check", "run"),
    [
        ({"name": "validate"}, None),
        ({"app": {"id": 1}}, None),
        ({"app": None}, None),
        ({"head_sha": H}, None),
        ({"details_url": None}, None),
        ({"details_url": OTHER_RUN}, None),
        ({"details_url": OTHER_RUN + "/job/x"}, None),
        ({"details_url": OTHER_RUN + "/job/2/x"}, None),
        ({"details_url": OTHER_RUN.replace(NAME, "rulespec-uk") + "/job/2"}, None),
        ({"details_url": OTHER_RUN.replace(GROUP_RUN, "0999") + "/job/2"}, None),
        ({"details_url": "x" + OTHER_RUN + "/job/2"}, None),
        ({"details_url": GROUP_RUN + "/job/2"}, None),
        ({"details_url": f"/{REPO}/actions/runs/{GROUP_RUN}/job/2"}, None),
        (None, {"id": 998}),
        (None, {"event": "pull_request"}),
        (None, {"event": "workflow_dispatch"}),
        (None, {"path": ".github/workflows/other.yml"}),
        (None, {"head_sha": H}),
        (None, {"repository": {"full_name": "TheAxiomFoundation/rulespec-uk"}}),
        (None, {"repository": None}),
    ],
)
def test_a_foreign_check_on_the_merge_group_head_fails_the_job_unevaluated(check, run):
    # A completed check with the required name that is not the group's own
    # run must not end the wait: the evaluation would run hours early.
    world = queued()
    group_ci(world, check=check, run=run)
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "not the merge group run of the required workflow" in report["error"]


def test_a_group_run_that_is_not_an_object_fails_the_job_unevaluated():
    world = queued()
    world.api[GROUP_RUN_PATH] = []
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "not the merge group run of the required workflow" in report["error"]


@pytest.mark.parametrize(
    "listing",
    [
        None,
        [],
        {"message": "API rate limit exceeded"},
        {"total_count": 2, "check_runs": [{"status": "completed"}]},
        {"total_count": "1", "check_runs": [{"status": "completed"}]},
        {"total_count": 1, "check_runs": ["completed"]},
        {"total_count": 101, "check_runs": [{"status": "completed"}] * 101},
        {"check_runs": [{"status": "completed"}]},
    ],
)
def test_a_merge_group_poll_that_cannot_be_read_whole_fails_unevaluated(listing):
    world = queued()
    group_ci(world, check={"status": "in_progress", "conclusion": None})
    pending = world.api[GROUP_CHECK_LISTING]
    reads = []

    def read():
        reads.append(1)
        return pending if len(reads) == 1 else listing

    world.api[GROUP_CHECK_LISTING] = read
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert "required check listing is incomplete" in report["error"]
    assert report["ci_polls"] == 2


@pytest.mark.parametrize(
    "error",
    [
        runner.RunnerError("GitHub GET request failed: HTTPError"),
        urllib.error.HTTPError("https://api.github.com", 500, "error", {}, None),
        KeyError("x"),
    ],
)
@pytest.mark.parametrize("at", ["listing", "run"])
def test_a_merge_group_poll_that_raises_fails_unevaluated(error, at):
    world = queued()

    def read():
        raise error

    world.api[GROUP_CHECK_LISTING if at == "listing" else GROUP_RUN_PATH] = read
    report, code = judge_group(world)
    no_evaluation(world, report, code)
    assert type(error).__name__ in report["error"] or "HTTPError" in report["error"]


def test_the_merge_group_evaluation_still_needs_the_pull_request_check_green():
    # The group's check is green; the pull request's own check on H is not.
    world = queued()
    world.ci(H, status="in_progress", conclusion=None)
    time = Time()
    report, code = judge_group(world, time)
    assert (reason_of(report), code) == (("fail", guard.R_CI), 1)
    assert time.sleeps == [] and report["ci_polls"] == 1


# Design section 7 in enforce mode.


def test_case_4_base_only_update_keeps_the_approval():
    world = build()
    advance(world, new_base=True)
    report, code = judge(world, pr_event(head=H2))
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["verdict"]["evidence"]["base"] == B2


def test_case_5_base_update_that_adds_a_workflow_voids_it():
    world = build()
    advance(world, new_base=True, extra={".github/workflows/x.yml": b"x\n"})
    report, code = judge(world, pr_event(head=H2))
    assert (report["verdict"]["verdict"], code) == ("fail", 1)


def test_case_10_a_dismissed_approval_fails_the_next_evaluation():
    world = build()
    assert judge(world)[1] == 0
    world.calls.clear()
    world.reviews([review(11, MAX, H, state="DISMISSED")])
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_NO_APPROVAL), 1)


def test_case_18_a_dismissal_during_the_evaluation_fails():
    world = build()
    second_read(
        world,
        ROOT + "/pulls/42/reviews?per_page=100&page=1",
        [review(11, MAX, H, state="DISMISSED")],
    )
    report, code = judge(world)
    assert (reason_of(report), code) == (("fail", guard.R_CHANGED), 1)


def test_case_25_docs_only_is_not_applicable_and_passes_the_job():
    report, code = judge(plain({"README.md": b"docs\n"}))
    assert (reason_of(report), code) == (
        ("not_applicable", guard.R_NO_GUARDED_CONTENT),
        0,
    )


def test_case_24_a_merge_group_of_two_units_fails_the_job():
    world = queued(group_tree=oid("tree-of-two-units"))
    report, code = judge(world, group_event(), "merge_group")
    assert (reason_of(report), code) == (("fail", guard.R_MERGE_GROUP_TREE), 1)


# Case 17: a failure at each read of a passing run fails the job.


class Failing:
    """``world.fetch`` that raises at the read with index ``index``."""

    def __init__(self, world, index):
        self.world, self.index, self.calls = world, index, world.calls

    def fetch(self, path):
        if len(self.world.calls) == self.index:
            self.world.calls.append(path)
            raise urllib.error.HTTPError(
                "https://api.github.com", 500, "error", {}, None
            )
        return self.world.fetch(path)


def passing_runs():
    def waited():
        world = build()
        pending_then(world, polls=2)
        return world

    return {
        "pull_request": (build, ..., "pull_request"),
        "pull_request_v5": (lambda: plain(V5_LANE), ..., "pull_request"),
        "pull_request_waited": (waited, ..., "pull_request"),
        "merge_group": (queued, group_event(), "merge_group"),
    }


PASSING_RUNS = passing_runs()


def run_length(key):
    make, event, name = PASSING_RUNS[key]
    report, code = judge(make(), event, name)
    assert code == 0
    return report["requests"]


# The number of reads of each passing run. Constants, not computed at import:
# a change that breaks a passing run then fails tests, not the collection.
PASSING_RUN_LENGTHS = {
    "pull_request": 25,
    "pull_request_v5": 14,
    "pull_request_waited": 43,
    # One poll (the listing and the workflow run), the evaluation of 29, and
    # the listing and the workflow run once more after it.
    "merge_group": 33,
}
READ_INDEXES = [
    (key, index)
    for key, length in PASSING_RUN_LENGTHS.items()
    for index in range(length)
]


def test_the_passing_runs_have_the_measured_number_of_reads():
    assert set(PASSING_RUN_LENGTHS) == set(PASSING_RUNS)
    assert {key: run_length(key) for key in PASSING_RUNS} == PASSING_RUN_LENGTHS
    assert len(READ_INDEXES) == 25 + 14 + 43 + 33


@pytest.mark.parametrize(("key", "index"), READ_INDEXES)
def test_case_17_a_failed_read_at_any_index_fails_the_job(key, index):
    make, event, name = PASSING_RUNS[key]
    world = Failing(make(), index)
    report, code = judge(world, event, name)
    assert code == 1, report
    assert "error" in report
    if key == "merge_group" and index < 2:
        # A failed poll of the merge group wait: no evaluation, no verdict.
        assert report["verdict"] is None and report["evaluations"] == 0
    elif key == "merge_group" and index >= 31:
        # A failed read of the group's check AFTER the evaluation: the
        # verdict that was reached stays in the report, and the job fails.
        assert report["verdict"]["verdict"] == "pass"
        assert report["group_ci_rechecks"] == 1
    else:
        assert report["verdict"]["verdict"] == "fail"
    assert report["requests"] == index + 1 or report["evaluations"] == 2


# Request volume: measured, and capped.

MEASURED_REQUESTS = {
    "v6 unit, pass": 25,
    "v5 lane, one unit": 14,
    "v5 lane, five units": 26,
    "no lane, fail": 5,
    "allowlist only": 5,
    "v6 unit, unapproved": 15,
    "control plane, pass (lane switched on in the test)": 17,
    "merge group of a v6 unit, pass (2 reads of one poll, 29, then 2 again)": 33,
}
MERGE_GROUP_LABEL = (
    "merge group of a v6 unit, pass (2 reads of one poll, 29, then 2 again)"
)
# The largest evaluation any fixture makes. MAX_ENFORCE_REQUESTS leaves room
# for two of them and every poll of one full wait.
LARGEST_EVALUATION = 30
# Reads of the first evaluation of the v6 fixture until it finds the required
# check pending and stops with ``ci_not_green``, and of its second, whole
# evaluation. (The review of head 979124f10 measured 16 and 25; an earlier
# revision stated 12 and 29, which only has the same sum.)
FIRST_EVALUATION_READS_UNTIL_CI = 16
SECOND_EVALUATION_READS = 25


def five_v5_units():
    files = {}
    for index in range(5):
        primary = f"us-az/policies/train/unit-{index}.yaml"
        companion = f"us-az/policies/train/unit-{index}.test.yaml"
        content = {primary: PRIMARY, companion: COMPANION}
        files |= content | {
            f".axiom/encoding-manifests/us-az/policies/train/unit-{index}.json": (
                v5_manifest(primary, companion, content=content)
            )
        }
    return plain(files)


def test_request_volume_per_evaluation_is_measured_and_bounded(monkeypatch):
    def unapproved():
        world = build()
        world.reviews([])
        return world

    def control_plane():
        monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", True)
        return control_plane_world()

    measured = {}
    for label, make, event, name in (
        ("v6 unit, pass", build, ..., "pull_request"),
        ("v5 lane, one unit", lambda: plain(V5_LANE), ..., "pull_request"),
        ("v5 lane, five units", five_v5_units, ..., "pull_request"),
        (
            "no lane, fail",
            lambda: plain({".github/workflows/x.yml": b"x\n"}),
            ...,
            "pull_request",
        ),
        ("allowlist only", lambda: plain({"README.md": b"x\n"}), ..., "pull_request"),
        ("v6 unit, unapproved", unapproved, ..., "pull_request"),
        (
            "control plane, pass (lane switched on in the test)",
            control_plane,
            ...,
            "pull_request",
        ),
        (MERGE_GROUP_LABEL, queued, group_event(), "merge_group"),
    ):
        report, _code = judge(make(), event, name)
        assert report["evaluations"] == 1, label
        measured[label] = report["requests"]
        monkeypatch.setattr(guard, "INFRASTRUCTURE_LANE_ENABLED", False)
    assert measured == MEASURED_REQUESTS
    evaluations = measured | {MERGE_GROUP_LABEL: measured[MERGE_GROUP_LABEL] - 4}
    assert max(evaluations.values()) == 29 <= LARGEST_EVALUATION
    # Pull request: two evaluations and every poll of one full wait fit under
    # the cap. Merge group: every poll of one full wait at its highest cost
    # (three reads, review round 2 P3-3), the one evaluation and the second
    # read of the group's check after it fit under it.
    # The cap is well under the documented 1,000 requests per hour.
    pull_request_worst = (
        2 * LARGEST_EVALUATION + runner.MAX_CI_POLLS * runner.CI_POLL_READS
    )
    merge_group_worst = (
        runner.MAX_MERGE_GROUP_POLLS * runner.MERGE_GROUP_POLL_READS
        + LARGEST_EVALUATION
        + runner.GROUP_CI_RECHECK_READS
    )
    assert (pull_request_worst, merge_group_worst) == (161, 336)
    assert runner.GROUP_CI_RECHECK_READS == runner.MERGE_GROUP_POLL_READS
    assert max(pull_request_worst, merge_group_worst) <= runner.MAX_ENFORCE_REQUESTS
    assert runner.MAX_MERGE_GROUP_POLLS == 101 and runner.MAX_CI_POLLS == 101
    assert (runner.CI_POLL_READS, runner.MERGE_GROUP_POLL_READS) == (1, 3)
    assert runner.MERGE_GROUP_POLL_READS == 1 + runner.MAX_GROUP_CHECK_RUNS
    # Sized for a later batch of 10 units: tests/test_merge_guard_request_budget.py.
    assert runner.MAX_ENFORCE_REQUESTS == 450
    # The polls of one run in any one hour: 20 for either event. One hour
    # can hold BOTH evaluations of a pull-request run (a short wait: the
    # review of head 979124f10 measured 61 reads in 3,420 simulated seconds),
    # so the hourly maximum is 20 + 2 x 30 = 80 for a pull request, and
    # 60 + 30 + 3 = 93 for a merge group (design section 8).
    assert 3600 // runner.MERGE_GROUP_POLL_SECONDS == 20
    assert 3600 // runner.CI_POLL_SECONDS == 20
    assert 20 * runner.CI_POLL_READS + 2 * LARGEST_EVALUATION == 80
    assert (
        20 * runner.MERGE_GROUP_POLL_READS
        + LARGEST_EVALUATION
        + runner.GROUP_CI_RECHECK_READS
        == 93
    )
    # A v5 lane costs three reads per unit: its manifest and two module files.
    assert (26 - 14) / 4 == 3


def test_the_request_cap_fails_the_run_closed(monkeypatch):
    needed = judge(build())[0]["requests"]
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", needed)
    assert judge(build())[1] == 0
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", needed - 1)
    world = build()
    report, code = runner.enforce("pull_request", REPO, pr_event(), fetch=world.fetch)
    assert code == 1
    assert report["verdict"]["verdict"] == "fail"
    assert report["requests"] == needed - 1 == len(world.calls)
    assert "error" in report


def test_the_request_cap_also_bounds_the_polls(monkeypatch):
    world = build()
    world.ci(H, status="in_progress", conclusion=None)
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", 40)
    report, code = judge(world)
    assert code == 1 and report["requests"] == 40
    assert "request cap" in report["error"]
    assert report["evaluations"] == 1


# The event: name, repository and payload.


@pytest.mark.parametrize(
    "event_name",
    [
        "pull_request_target",
        "pull_request_review",
        "push",
        "workflow_dispatch",
        "schedule",
        "PULL_REQUEST",
        "pull_request ",
        "",
        None,
        ["pull_request"],
    ],
)
def test_any_other_event_fails_without_a_read(event_name):
    world = build()
    report, code = judge(world, pr_event() | group_event(), event_name)
    assert code == 1 and world.calls == [] and report["verdict"] is None
    assert "not pull_request or merge_group" in report["error"]


@pytest.mark.parametrize(
    "repository",
    [
        "TheAxiomFoundation/axiom-encode",
        "TheAxiomFoundation/rulespec-US",
        "TheAxiomFoundation/rulespec",
        "TheAxiomFoundation/rulespec-us/x",
        "theaxiomfoundation/rulespec-us",
        "Other/rulespec-us",
        "rulespec-us",
        "",
        None,
    ],
)
def test_a_repository_outside_the_rulespec_family_fails_without_a_read(repository):
    world = build()
    report, code = judge(world, pr_event(repository=repository), repository=repository)
    assert code == 1 and world.calls == [] and report["verdict"] is None
    assert "not a RuleSpec repository" in report["error"]


@pytest.mark.parametrize(
    "event",
    [
        pr_event(repository="TheAxiomFoundation/rulespec-uk"),
        pr_event(repository=None),
        {k: v for k, v in pr_event().items() if k != "repository"},
        pr_event(number=43) | {"pull_request": pr_event()["pull_request"]},
        pr_event() | {"number": "42"},
        pr_event() | {"number": None},
        {k: v for k, v in pr_event().items() if k != "number"},
        pr_event(number=0),
        pr_event(number=-1),
        pr_event(number=True),
        pr_event(number="42"),
        pr_event(number=42.0),
        pr_event(head="main"),
        pr_event(head=H[:39]),
        pr_event(head=H.upper()),
        pr_event(head=None),
        pr_event() | {"pull_request": None},
        pr_event() | {"pull_request": {"number": 42}},
        pr_event() | {"pull_request": {"number": 42, "head": H}},
        {k: v for k, v in pr_event().items() if k != "pull_request"},
        group_event(),
        [],
        None,
        "pull_request",
    ],
)
def test_a_malformed_pull_request_event_fails_without_a_read(event):
    world = build()
    report, code = judge(world, event)
    assert code == 1 and world.calls == [] and report["verdict"] is None
    assert report["error"]


@pytest.mark.parametrize(
    "changes",
    [
        {"head_ref": "refs/heads/main"},
        {"head_ref": f"refs/heads/gh-readonly-queue/release/pr-42-{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B}-extra"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B}/x"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B}\n"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B[:39]}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B.upper()}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-0-{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-042-{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-43-{B}"},
        {"head_ref": f"refs/heads/gh-readonly-queue/main/pr-42-{B}-pr-43-{B}"},
        {"head_ref": f"gh-readonly-queue/main/pr-42-{B}"},
        {"head_ref": f"refs/tags/gh-readonly-queue/main/pr-42-{B}"},
        {"head_ref": f"x/refs/heads/gh-readonly-queue/main/pr-42-{B}"},
        {"head_ref": None},
        {"head_ref": 42},
        {"base_ref": "refs/heads/release"},
        {"base_ref": "main"},
        {"base_ref": None},
        {"head_sha": "main"},
        {"head_sha": None},
        {"base_sha": B[:39]},
        {"base_sha": None},
    ],
)
def test_a_merge_group_event_must_name_one_pull_request_queued_to_main(changes):
    world = queued()
    report, code = judge(world, group_event(**changes), "merge_group")
    assert code == 1 and world.calls == [] and report["verdict"] is None
    assert report["error"]


@pytest.mark.parametrize(
    "event",
    [
        {k: v for k, v in group_event().items() if k != "merge_group"},
        group_event() | {"merge_group": None},
        group_event(repository="TheAxiomFoundation/rulespec-uk"),
        pr_event(),
    ],
)
def test_a_malformed_merge_group_event_fails_without_a_read(event):
    world = queued()
    report, code = judge(world, event, "merge_group")
    assert code == 1 and world.calls == [] and report["verdict"] is None


def test_a_merge_group_event_passes_for_the_one_approved_unit():
    world = queued()
    report, code = judge(world, group_event(), "merge_group")
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["event_head"] == G and report["pull_request"] == 42
    assert report["verdict"]["evidence"]["merge_group"]["head"] == G


def test_the_merge_group_number_comes_from_the_head_ref():
    # The ref names pull request 43, which the fake cannot read.
    world = queued()
    report, code = judge(world, group_event(number=43), "merge_group")
    assert (reason_of(report), code) == (("fail", guard.R_READ_FAILED), 1)
    assert world.calls == [GROUP_CHECK_LISTING, GROUP_RUN_PATH, ROOT + "/pulls/43"]


@pytest.mark.parametrize(
    "evidence",
    [
        {"pull_request": 42, "merge_group": {"head": H, "base": B}},
        {"pull_request": 42, "merge_group": {"head": G, "base": B2}},
        {"pull_request": 43, "merge_group": {"head": G, "base": B}},
        {"pull_request": 42, "merge_group": None},
        {"pull_request": 42},
    ],
)
def test_a_success_that_is_not_bound_to_the_event_merge_group_fails(
    monkeypatch, evidence
):
    judged = guard.Verdict(guard.PASS, guard.R_PASS, "", evidence)
    monkeypatch.setattr(guard, "evaluate_merge_group", lambda *_a, **_k: judged)
    report, code = judge(queued(), group_event(), "merge_group")
    assert (reason_of(report), code) == (("fail", runner.R_EVENT_HEAD_STALE), 1)


def test_a_bound_merge_group_success_is_passed_through(monkeypatch):
    evidence = {"pull_request": 42, "merge_group": {"head": G, "base": B}}
    judged = guard.Verdict(guard.PASS, guard.R_PASS, "", evidence)
    monkeypatch.setattr(guard, "evaluate_merge_group", lambda *_a, **_k: judged)
    assert judge(queued(), group_event(), "merge_group")[1] == 0


# The event file.


def event_file(tmp_path, raw):
    path = tmp_path / "event.json"
    path.write_bytes(raw if isinstance(raw, bytes) else json.dumps(raw).encode())
    return str(path)


def test_event_file_is_read_as_one_strict_json_object(tmp_path):
    assert runner.read_event(event_file(tmp_path, pr_event())) == pr_event()


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not json",
        b"[]",
        b"null",
        b'"pull_request"',
        b'{"number": 42, "number": 43}',
        b'{"pull_request": {"number": 1, "number": 42}}',
        b'{"number": NaN}',
        b'{"number": Infinity}',
        b'{"number": 42} trailing',
        b'{"number": "\xff"}',
        b"\xef\xbb\xbf{}",
        DEEP_JSON,
        b'{"a":' * 100_000 + b"1" + b"}" * 100_000,
    ],
    ids=[
        "empty",
        "not-json",
        "array",
        "null",
        "string",
        "duplicate-key",
        "nested-duplicate-key",
        "nan",
        "infinity",
        "trailing-text",
        "invalid-utf8",
        "byte-order-mark",
        "deep-arrays",
        "deep-objects",
    ],
)
def test_event_file_refuses_anything_else(tmp_path, raw):
    with pytest.raises(runner.RunnerError, match="not strict JSON"):
        runner.read_event(event_file(tmp_path, raw))


def test_event_file_refuses_a_payload_above_its_bound(tmp_path, monkeypatch):
    raw = json.dumps(pr_event()).encode()
    monkeypatch.setattr(runner, "MAX_EVENT_BYTES", len(raw))
    assert runner.read_event(event_file(tmp_path, raw)) == pr_event()
    monkeypatch.setattr(runner, "MAX_EVENT_BYTES", len(raw) - 1)
    with pytest.raises(runner.RunnerError, match="exceeds its bound"):
        runner.read_event(event_file(tmp_path, raw))


@pytest.mark.parametrize("path", [None, "", 7, "missing.json", "."])
def test_event_file_refuses_an_absent_or_unreadable_path(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(runner.RunnerError):
        runner.read_event(path)


# The command: inputs from the runner environment only; reads only.


class WorldOpener:
    """The HTTP layer over a fake world; records every request it is sent."""

    def __init__(self, world):
        self.world = world
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        assert request.full_url.startswith("https://api.github.com/repos/")
        path = request.full_url.removeprefix("https://api.github.com")
        return Response(200, json.dumps(self.world.fetch(path)).encode())


def run_enforce_command(capsys, tmp_path, world, event=None, **environ):
    opener = WorldOpener(world)
    time = Time()
    values = ENFORCE_ENV | {
        "GITHUB_EVENT_PATH": event_file(
            tmp_path, pr_event() if event is None else event
        )
    }
    code = runner.enforce_command(
        environ=values | environ, opener=opener, sleep=time.sleep, clock=time.clock
    )
    out = capsys.readouterr()
    assert out.out.count("\n") == 1 and out.out.endswith("\n")
    assert TOKEN not in out.out and TOKEN not in out.err
    return code, json.loads(out.out), opener


@pytest.mark.parametrize(
    ("world", "event", "name", "code", "verdict"),
    [
        (lambda: build(), None, "pull_request", 0, "pass"),
        (lambda: plain(V5_LANE), None, "pull_request", 0, "not_applicable"),
        (
            lambda: plain({"README.md": b"x\n"}),
            None,
            "pull_request",
            0,
            "not_applicable",
        ),
        (
            lambda: plain({".github/workflows/x.yml": b"x\n"}),
            None,
            "pull_request",
            1,
            "fail",
        ),
        (queued, group_event(), "merge_group", 0, "pass"),
        (
            lambda: queued(group_tree=oid("other")),
            group_event(),
            "merge_group",
            1,
            "fail",
        ),
    ],
)
def test_command_judges_the_event_and_sends_only_get_requests(
    capsys, tmp_path, monkeypatch, world, event, name, code, verdict
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("enforce mode must not build a writer")

    monkeypatch.setattr(runner, "check_writer", forbidden)
    got, report, opener = run_enforce_command(
        capsys, tmp_path, world(), event, GITHUB_EVENT_NAME=name
    )
    assert (got, report["verdict"]["verdict"]) == (code, verdict)
    assert report["schema"] == runner.ENFORCE_REPORT_SCHEMA
    assert report["event"] == name and report["repository"] == REPO
    assert opener.requests and report["requests"] == len(opener.requests)
    for request in opener.requests:
        assert request.get_method() == "GET"
        assert request.data is None
        assert request.get_header("Authorization") == "Bearer " + TOKEN


def test_command_waits_through_the_http_layer(capsys, tmp_path):
    world = build()
    pending_then(world, polls=3)
    code, report, opener = run_enforce_command(capsys, tmp_path, world)
    assert code == 0 and report["evaluations"] == 2 and report["ci_polls"] == 3
    assert {request.get_method() for request in opener.requests} == {"GET"}


SOURCE_ENV = {"GITHUB_REPOSITORY": SOURCE, "GITHUB_EVENT_NAME": "pull_request"}


def run_source_command(capsys, tmp_path, event, **environ):
    """The command where ``GITHUB_REPOSITORY`` names the encoder repository."""
    values = SOURCE_ENV | {"GITHUB_EVENT_PATH": event_file(tmp_path, event)} | environ
    code = runner.enforce_command(environ=values, opener=Exploding())
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    return code, json.loads(out)


@pytest.mark.parametrize("name", ["pull_request", "merge_group"])
def test_command_is_inert_in_the_source_repository(capsys, tmp_path, name):
    # No token and an opener that refuses every request: the source
    # repository's own runs make no network read.
    code, report = run_source_command(
        capsys, tmp_path, source_event(name), GITHUB_EVENT_NAME=name
    )
    assert code == 0
    assert report == {
        "schema": runner.ENFORCE_REPORT_SCHEMA,
        "event": name,
        "repository": SOURCE,
        "verdict": None,
        "requests": 0,
        "inert": "source_repository",
    }


OTHER_REPOSITORIES = [
    REPO,
    "TheAxiomFoundation/rulespec-uk",
    "theaxiomfoundation/axiom-encode",
    "TheAxiomFoundation/axiom-encode-fork",
    "Other/axiom-encode",
    "axiom-encode",
    "",
    None,
    7,
    ["TheAxiomFoundation/axiom-encode"],
]


@pytest.mark.parametrize("repository", OTHER_REPOSITORIES)
@pytest.mark.parametrize("name", ["pull_request", "merge_group"])
def test_the_inert_path_fails_when_the_event_names_another_repository(
    capsys, tmp_path, name, repository
):
    # F1 of review round 1: GITHUB_REPOSITORY says "axiom-encode" and the
    # event belongs to a RuleSpec repository. That must never be a success.
    code, report = run_source_command(
        capsys,
        tmp_path,
        source_event(name, repository=repository),
        GITHUB_EVENT_NAME=name,
        **{runner.TOKEN_ENV: TOKEN},
    )
    assert code == 1 and "inert" not in report and report["requests"] == 0
    assert "another repository" in report["error"]


@pytest.mark.parametrize("base", OTHER_REPOSITORIES)
def test_the_inert_path_fails_when_the_pull_request_is_of_another_repository(
    capsys, tmp_path, base
):
    # ``repository.full_name`` agrees; the pull request's own base does not.
    code, report = run_source_command(capsys, tmp_path, source_event(base=base))
    assert code == 1 and "inert" not in report
    assert "belongs to another repository" in report["error"]


@pytest.mark.parametrize(
    "event",
    [
        {k: v for k, v in source_event().items() if k != "repository"},
        source_event() | {"repository": None},
        source_event() | {"repository": SOURCE},
        source_event() | {"repository": {"name": "axiom-encode"}},
        {k: v for k, v in source_event().items() if k != "pull_request"},
        source_event() | {"pull_request": None},
        source_event() | {"pull_request": {"number": 42, "head": {"sha": H}}},
        source_event() | {"pull_request": {"base": None}},
        source_event() | {"pull_request": {"base": {"repo": None}}},
        source_event() | {"pull_request": {"base": {"repo": SOURCE}}},
        source_event() | {"pull_request": {"base": {"repo": {}}}},
        source_event("merge_group"),
    ],
)
def test_the_inert_path_fails_on_an_event_that_does_not_name_the_source(
    capsys, tmp_path, event
):
    code, report = run_source_command(capsys, tmp_path, event)
    assert code == 1 and "inert" not in report and report["error"]


@pytest.mark.parametrize(
    "raw", [b"not json", b'{"number": 42, "number": 42}', b"[]", b"", DEEP_JSON]
)
def test_the_inert_path_fails_on_an_event_that_is_not_strict_json(
    capsys, tmp_path, raw
):
    code, report = run_source_command(capsys, tmp_path, raw)
    assert code == 1 and "inert" not in report
    assert "not strict JSON" in report["error"]


@pytest.mark.parametrize("path", [None, "", "missing.json", "."])
def test_the_inert_path_fails_without_a_readable_event(
    capsys, tmp_path, monkeypatch, path
):
    monkeypatch.chdir(tmp_path)
    environ = dict(SOURCE_ENV)
    if path is not None:
        environ["GITHUB_EVENT_PATH"] = path
    code = runner.enforce_command(environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and "inert" not in report and report["error"]


def test_the_inert_path_fails_on_an_oversized_event(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MAX_EVENT_BYTES", 16)
    code, report = run_source_command(capsys, tmp_path, source_event())
    assert code == 1 and "exceeds its bound" in report["error"]


@pytest.mark.parametrize(
    "name",
    [None, "push", "pull_request_target", "workflow_dispatch", "PULL_REQUEST", ""],
)
def test_the_inert_path_fails_on_another_event(capsys, tmp_path, name):
    environ = {} if name is None else {"GITHUB_EVENT_NAME": name}
    values = (
        {"GITHUB_REPOSITORY": SOURCE}
        | {"GITHUB_EVENT_PATH": event_file(tmp_path, source_event())}
        | environ
    )
    code = runner.enforce_command(environ=values, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and "inert" not in report
    assert "not pull_request or merge_group" in report["error"]


def test_the_inert_path_fails_on_an_unforeseen_error(capsys, tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("unforeseen " + TOKEN)

    monkeypatch.setattr(runner, "_require_source_event", explode)
    code, report = run_source_command(capsys, tmp_path, source_event())
    assert code == 1 and "inert" not in report
    assert report["error"] == "unforeseen error: RuntimeError"


@pytest.mark.parametrize(
    "repository",
    [
        "theaxiomfoundation/axiom-encode",
        "TheAxiomFoundation/Axiom-Encode",
        "TheAxiomFoundation/axiom-encode ",
        "TheAxiomFoundation/axiom-encode-fork",
        "TheAxiomFoundation/axiom-encode/x",
        "Other/axiom-encode",
        "axiom-encode",
        "TheAxiomFoundation/axiom-rules-engine",
        "TheAxiomFoundation/.github",
        "",
        None,
    ],
)
def test_only_the_exact_source_repository_is_inert(capsys, tmp_path, repository):
    environ = dict(ENFORCE_ENV) | {
        "GITHUB_EVENT_PATH": event_file(tmp_path, pr_event(repository=repository))
    }
    if repository is None:
        del environ["GITHUB_REPOSITORY"]
    else:
        environ["GITHUB_REPOSITORY"] = repository
    code = runner.enforce_command(environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and "inert" not in report
    assert "not a RuleSpec repository" in report["error"]


@pytest.mark.parametrize("token", [None, "", "x", "has space " + "x" * 30])
def test_command_fails_closed_without_the_job_token(capsys, tmp_path, token):
    environ = dict(ENFORCE_ENV) | {
        "GITHUB_EVENT_PATH": event_file(tmp_path, pr_event())
    }
    if token is None:
        del environ[runner.TOKEN_ENV]
    else:
        environ[runner.TOKEN_ENV] = token
    code = runner.enforce_command(environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["verdict"] is None and report["requests"] == 0
    assert runner.TOKEN_ENV in report["error"]


@pytest.mark.parametrize(
    "raw", [b"not json", b'{"number": 42, "number": 42}', b"[]", b""]
)
def test_command_fails_closed_on_a_malformed_event_without_a_request(
    capsys, tmp_path, raw
):
    environ = ENFORCE_ENV | {"GITHUB_EVENT_PATH": event_file(tmp_path, raw)}
    code = runner.enforce_command(environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and report["verdict"] is None
    assert "not strict JSON" in report["error"]


def test_command_fails_closed_on_an_oversized_event(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "MAX_EVENT_BYTES", 16)
    environ = ENFORCE_ENV | {"GITHUB_EVENT_PATH": event_file(tmp_path, pr_event())}
    code = runner.enforce_command(environ=environ, opener=Exploding())
    assert code == 1
    assert "exceeds its bound" in json.loads(capsys.readouterr().out)["error"]


def test_command_fails_closed_without_an_event_path(capsys):
    code = runner.enforce_command(environ=dict(ENFORCE_ENV), opener=Exploding())
    assert code == 1
    assert "GITHUB_EVENT_PATH" in json.loads(capsys.readouterr().out)["error"]


@pytest.mark.parametrize("event_name", [None, "push", "pull_request_target"])
def test_command_fails_closed_on_another_event(capsys, tmp_path, event_name):
    environ = ENFORCE_ENV | {
        "GITHUB_EVENT_PATH": event_file(tmp_path, pr_event()),
        "GITHUB_EVENT_NAME": event_name,
    }
    if event_name is None:
        del environ["GITHUB_EVENT_NAME"]
    code = runner.enforce_command(environ=environ, opener=Exploding())
    report = json.loads(capsys.readouterr().out)
    assert code == 1 and "not pull_request or merge_group" in report["error"]


def test_command_reports_a_failed_read_as_fail_and_a_run_error(capsys, tmp_path):
    code = runner.enforce_command(
        environ=ENFORCE_ENV | {"GITHUB_EVENT_PATH": event_file(tmp_path, pr_event())},
        opener=Opener(error=http_error(403)),
    )
    out = capsys.readouterr()
    report = json.loads(out.out)
    assert code == 1 and TOKEN not in out.out
    assert report["verdict"]["reason"] == guard.R_READ_FAILED
    assert report["error"] == "an evaluation could not read GitHub"


def test_enforce_has_no_argument_that_selects_a_pull_request_or_a_mode():
    import inspect

    assert list(inspect.signature(runner.enforce_command).parameters) == [
        "environ",
        "opener",
        "sleep",
        "clock",
    ]
    source = inspect.getsource(runner.enforce_command) + inspect.getsource(
        runner.enforce
    )
    assert "check_writer" not in source and "post" not in source.lower()
    assert "POST" not in source


def test_cli_registers_merge_guard_enforce_without_arguments(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("GITHUB_REPOSITORY", "TheAxiomFoundation/axiom-encode")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    monkeypatch.setenv("GITHUB_EVENT_PATH", event_file(tmp_path, source_event()))
    monkeypatch.delenv(runner.TOKEN_ENV, raising=False)
    monkeypatch.setattr(sys, "argv", ["axiom-encode", "merge-guard-enforce"])
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == 0
    assert json.loads(capsys.readouterr().out)["inert"] == "source_repository"
    monkeypatch.setenv("GITHUB_REPOSITORY", REPO)
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == 1
    assert runner.TOKEN_ENV in json.loads(capsys.readouterr().out)["error"]
    for argv in (
        ["--pr", "42"],
        ["--repository", NAME],
        ["--mode", "enforce"],
        ["--token", TOKEN],
        ["42"],
    ):
        monkeypatch.setattr(sys, "argv", ["axiom-encode", "merge-guard-enforce", *argv])
        with pytest.raises(SystemExit) as stop:
            cli.main()
        assert stop.value.code == 2
        capsys.readouterr()


def test_every_read_of_every_lane_is_a_path_the_reader_sends():
    for make, event, name in PASSING_RUNS.values():
        world = make()
        judge(world, event, name)
        opener = Opener(raw=b"{}")
        fetch = runner.token_reader(TOKEN, opener=opener)
        for path in world.calls:
            fetch(path)
        assert len(opener.requests) == len(world.calls)

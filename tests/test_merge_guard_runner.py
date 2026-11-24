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
    assert runner.MAX_ENFORCE_REQUESTS == 350
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

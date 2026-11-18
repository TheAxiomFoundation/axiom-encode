"""Runner for the content-bound merge guard: evaluate and report.

This is change 3 and change 3b of ``docs/content-bound-merge-guard.md``
section 8. It runs ``content_merge_guard.evaluate`` on pull requests of one
RuleSpec repository and reports the verdict. Two callers exist.

The shadow runner (``command``, ``axiom-encode merge-guard-run``, workflow
``.github/workflows/merge-guard-runner.yml``) runs as the guard GitHub App:

- ``dry-run`` (the default) reads and prints. It writes nothing to GitHub.
- ``shadow`` also writes one check run per judged head, named
  ``SHADOW_CHECK_NAME``, whose conclusion is always ``neutral``. The verdict is
  in the title and the summary. A shadow check cannot block a merge and must
  never be made a required check: ``neutral`` satisfies a required check, so
  requiring it would require nothing.
- The one writer, ``check_writer``, refuses every body that is not a
  completed, neutral check named ``SHADOW_CHECK_NAME``. ``GUARD_CHECK_NAME``
  is the name reserved for the enforcing check; no code here writes it.

The enforcing runner (``enforce``, ``enforce_command``, ``axiom-encode
merge-guard-enforce``, workflow
``.github/workflows/content-bound-merge-guard.yml``) judges the one pull
request or merge group of the Actions event that started it. It reads with the
job's read-only token and writes nothing: no code path of ``enforce`` builds
``check_writer`` or sends a request with a body. Its result is the exit code
of the process, which is the conclusion of the Actions job: 0 only for a
``pass`` or ``not_applicable`` verdict whose bindings hold, 1 for everything
else. It is required nowhere until an organization owner creates a ruleset
that names the workflow. For a merge group it first waits until the group's
own required check has completed and evaluates after that, so that the
verdict is as late as the job can make it.

Nothing here carries authority. The list of open pull requests is only a work
list: every verdict comes from the evaluator's own live reads. A read that
fails, is truncated or reaches its page bound is an error of the run. It is
never reported as "no pull requests" or "nothing to check".

The token is read from the environment variable ``TOKEN_ENV`` only. It is
never accepted on the command line and never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Final

from axiom_encode import content_merge_guard as guard
from axiom_encode.reviewed_pr_admission import MAX_RESPONSE_BYTES, _NoRedirect

TOKEN_ENV: Final = "MERGE_GUARD_TOKEN"
# The guard App's client ID (not a secret). With it the runner can tell its
# own earlier shadow check from one that another publisher gave the same name.
APP_CLIENT_ID_ENV: Final = "MERGE_GUARD_APP_CLIENT_ID"
# The enforcing check of a later change. Reserved: never published here.
GUARD_CHECK_NAME: Final = "content-bound-merge-guard"
SHADOW_CHECK_NAME: Final = "content-bound-merge-guard (shadow)"
# ``neutral`` is the only conclusion a shadow check may carry. It is not
# ``success`` (nobody reads a green tick as an approval) and not ``failure``
# (no red cross on pull requests the guard does not gate yet).
SHADOW_CONCLUSION: Final = "neutral"
DRY_RUN: Final = "dry-run"
SHADOW: Final = "shadow"
MODES: Final = (DRY_RUN, SHADOW)
REPORT_SCHEMA: Final = "axiom-encode/merge-guard-runner-report/v1"
MAX_PAGES: Final = 10
# Pages of 100 that ``_already_reported`` reads of one head's shadow-named
# check runs before it refuses.
MAX_CHECK_RUN_PAGES: Final = 10
MAX_SUMMARY_CHARS: Final = 4000
EXIT_OK: Final = 0
EXIT_RUN_ERROR: Final = 1
# The socket timeout of one request. It bounds each blocking operation of the
# request, not the request as a whole.
REQUEST_TIMEOUT_SECONDS: Final = 30

# --- the enforcing runner (change 3b) ----------------------------------------
ENFORCE_REPORT_SCHEMA: Final = "axiom-encode/merge-guard-enforce-report/v1"
EVENT_PULL_REQUEST: Final = "pull_request"
EVENT_MERGE_GROUP: Final = "merge_group"
INERT_SOURCE_REPOSITORY: Final = "source_repository"
# The run's check attaches to the head of the event that started the run. A
# verdict for another head must never satisfy it.
R_EVENT_HEAD_STALE: Final = "event_head_is_stale"
# The verdicts that end the job with success, by verdict AND reason. A new
# success reason of the evaluator does not pass here until a reviewed change
# adds it to this set. The two success reasons of the proposed infrastructure
# lane (``guard.R_CONTROL_PLANE_PASS``, ``guard.R_INFRASTRUCTURE``) are not in
# it: that lane is a proposal and is switched off. A verdict with one of them
# ends the job with failure. They are added in the reviewed change that
# decides the lane, not before.
SUCCESS_VERDICTS: Final = frozenset(
    {
        (guard.PASS, guard.R_PASS),
        (guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT),
        (guard.NOT_APPLICABLE, guard.R_EARLIER_LANE),
    }
)
# The guard run starts together with ``validate / validate`` and nothing
# starts it again when that check finishes. So a ``ci_not_green`` verdict
# waits: poll the required check on the event head every ``CI_POLL_SECONDS``
# until every listed run is completed, for at most ``CI_WAIT_SECONDS``, then
# evaluate once more. The wait only delays; it never produces a verdict.
#
# ``CI_WAIT_SECONDS`` is five hours, so the job outlasts normal CI. Measured
# read-only on 2026-11-17 (design section 8, "Measured duration of the
# required check"): of the 40 newest successful ``pull_request`` runs of the
# required workflow on rulespec-us, 38 took 172 to 227 minutes. A wait
# shorter than that ends every signed head with ``ci_not_green`` and a manual
# re-run. The poll interval is the merge group's: 101 polls of one listing
# read each.
CI_POLL_SECONDS: Final = 180
CI_WAIT_SECONDS: Final = 5 * 3600
MAX_CI_POLLS: Final = CI_WAIT_SECONDS // CI_POLL_SECONDS + 1
# Reads of one poll of the pull-request wait: the listing. The poll does not
# verify the listed runs; the evaluation that follows does.
CI_POLL_READS: Final = 1
# A merge group is judged as late as the job can do it. Its run starts when
# the group is created, and the group's own ``validate / validate`` run can
# take hours; a verdict from the start of that time would not see an approval
# that is dismissed while the group validates. So the run first waits until
# the required check on the GROUP head has completed, and only then evaluates,
# once. The wait gives no verdict. When it ends without a completed check,
# when the check did not succeed, or when a poll fails, the run is an error
# and nothing is evaluated.
#
# ``MERGE_GROUP_WAIT_SECONDS`` is five hours. rulespec-us#2160 reports (its
# author's figures, not measured here) 2.5 to 4 hours of validation for each
# group and asks for a queue status-check timeout above 4 hours. The job's
# ``timeout-minutes`` must exceed the longest run of either event (one
# evaluation, the wait, its last poll interval and last poll, one more
# evaluation), and a GitHub-hosted job ends after 6 hours at the latest.
#
# A poll of this wait is one listing read, plus one read of the workflow run
# of each listed check once all of them say ``completed``. The listing may
# hold at most ``MAX_GROUP_CHECK_RUNS`` checks; more is an error of the run.
# So a poll costs at most ``MERGE_GROUP_POLL_READS`` = 3 reads (review round
# 2, P3-3: two listed checks, the first with a completed run, the second with
# a run in progress), and 101 polls and the largest measured evaluation (29
# reads, 30 with headroom) are at most 3 * 101 + 30 = 333 requests.
MERGE_GROUP_POLL_SECONDS: Final = 180
MERGE_GROUP_WAIT_SECONDS: Final = 5 * 3600
MAX_MERGE_GROUP_POLLS: Final = MERGE_GROUP_WAIT_SECONDS // MERGE_GROUP_POLL_SECONDS + 1
MAX_GROUP_CHECK_RUNS: Final = 2
MERGE_GROUP_POLL_READS: Final = 1 + MAX_GROUP_CHECK_RUNS
# The workflow run event of the required check on a merge group head.
MERGE_GROUP_RUN_EVENT: Final = "merge_group"
# One event payload. GitHub documents "Payloads are capped at 25 MB"
# (https://docs.github.com/en/webhooks/webhook-events-and-payloads); a
# pull_request or merge_group payload is far smaller. A larger file fails.
MAX_EVENT_BYTES: Final = 4 * 1024 * 1024
# Hard cap of GitHub requests in one enforce run: the worst case of either
# event, with headroom. The largest evaluation measured on fixtures is 29
# requests (``tests/test_merge_guard_runner.py``, "request volume"), counted
# as 30. Pull request: two evaluations and every poll of one full wait are
# 2 * 30 + 101 * 1 = 161 requests. Merge group: every poll at its highest
# cost and the one evaluation are 3 * 101 + 30 = 333 requests (see above).
# Both worst cases are driven by tests. Past the cap every read fails, so a
# verdict is ``fail`` and a wait is an error. A full wait takes five hours,
# so in any one hour a run makes at most 20 polls: 20 + 30 = 50 requests for
# a pull request (an hour that holds an evaluation), 60 + 30 = 90 for a merge
# group. GitHub documents the job token's limit as 1,000 requests per hour
# per repository
# (https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).
MAX_ENFORCE_REQUESTS: Final = 350

_API: Final = "https://api.github.com"
_REPOSITORY = re.compile(r"rulespec-[a-z]{2}(-[a-z0-9]+)*")
_TOKEN = re.compile(r"[A-Za-z0-9_.-]{20,512}")
_SHA = re.compile(r"[0-9a-f]{40}")
# The temporary branch of a merge group that holds one pull request to
# ``main``: ``refs/heads/gh-readonly-queue/main/pr-<number>-<40 hex>``.
_QUEUE_REF = re.compile(
    r"refs/heads/gh-readonly-queue/main/pr-([1-9][0-9]{0,8})-[0-9a-f]{40}"
)
_COMPARE_SEGMENT = re.compile(r"[0-9a-f]{40}\.\.\.[0-9a-f]{40}")
_CHECK_RUNS_PATH = re.compile(
    r"/repos/TheAxiomFoundation/rulespec-[a-z]{2}(-[a-z0-9]+)*/check-runs"
)
_RUN_ID = re.compile(r"[1-9][0-9]{0,19}")

# Verdict reasons that mean "the evaluation could not read", not "the pull
# request is refused". The verdict is still ``fail``; the run is also an error.
_UNREAD_REASONS: Final = frozenset({guard.R_READ_FAILED, guard.R_EVALUATION_ERROR})

Fetch = Callable[[str], object]
Post = Callable[[str, dict], object]
Sleep = Callable[[float], object]
Clock = Callable[[], float]


class RunnerError(Exception):
    """The run could not establish or report a verdict. Never a verdict."""


def _strict_int(value: object) -> bool:
    return type(value) is int and value > 0


def _request(
    method: str,
    path: str,
    token: str,
    body: dict | None,
    opener,
    expected_status: int,
) -> object:
    """One bounded request to the fixed GitHub API origin, with ``token``."""
    if (
        not isinstance(path, str)
        or not path.startswith("/repos/TheAxiomFoundation/")
        or any(c in path for c in ("#", "\\", "\r", "\n", " "))
        or "%2e" in path.lower()
        or any(
            segment == "."
            # No ".." anywhere, as before, with one exception: the
            # evaluator's ``compare/<40 hex>...<40 hex>`` segment. Refusing
            # it would fail every evaluation at its second read.
            or (".." in segment and _COMPARE_SEGMENT.fullmatch(segment) is None)
            for segment in path.split("/")
        )
        # Review round 2, P3-C: the same test on the path part alone, so a
        # query or a ``;`` parameter after a single dot does not hide the
        # dot segment (``/x/.?a``, ``/x/.;a/y``). The whole-string test
        # above stays: it also refuses ".." inside a query.
        or any(
            segment.partition(";")[0] == "."
            for segment in path.partition("?")[0].split("/")
        )
    ):
        raise RunnerError("invalid GitHub API path")
    if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
        raise RunnerError(f"{TOKEN_ENV} is absent or malformed")
    data = None
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "axiom-encode-merge-guard-runner",
        "Authorization": "Bearer " + token,
    }
    if body is not None:
        data = json.dumps(body, sort_keys=True).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        _API + path, data=data, headers=headers, method=method
    )
    if opener is None:
        # No environment proxy and no redirect: the origin is not an input.
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect()
        )
    try:
        with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            status = response.status
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except RunnerError:
        raise
    except Exception as exc:  # HTTP error, rate limit, redirect, network
        # Only the exception type: a message can echo request headers.
        raise RunnerError(
            f"GitHub {method} request failed: {type(exc).__name__}"
        ) from None
    if status != expected_status:
        raise RunnerError(f"GitHub {method} request did not succeed")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RunnerError("GitHub response exceeds its bound")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        # RecursionError: JSON nested deeper than the interpreter's limit. It
        # is not a ValueError; uncaught, it would end the command without a
        # report.
        raise RunnerError("GitHub response is not JSON") from None


def token_reader(token: str, *, opener=None) -> Fetch:
    """The evaluator's ``fetch``: bounded GET reads with the token."""

    def fetch(path: str) -> object:
        return _request("GET", path, token, None, opener, 200)

    return fetch


def check_writer(token: str, *, opener=None) -> Post:
    """The one write this module makes: create a check run."""

    def post(path: str, body: dict) -> object:
        if _CHECK_RUNS_PATH.fullmatch(path) is None:
            raise RunnerError("the runner writes shadow check runs only")
        if (
            not isinstance(body, dict)
            or body.get("name") != SHADOW_CHECK_NAME
            or body.get("conclusion") != SHADOW_CONCLUSION
            or body.get("status") != "completed"
        ):
            raise RunnerError("the runner writes a neutral shadow check only")
        return _request("POST", path, token, body, opener, 201)

    return post


def open_pull_requests(repository: str, fetch: Fetch) -> list[int]:
    """Numbers of the open pull requests to ``main``. A work list only.

    Refuses a page that is not a list, a page above the page size, an entry
    without a number, a repeated number and the page bound. An empty first
    page is the only way to return no pull requests.
    """
    if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
        raise RunnerError("invalid repository")
    root = f"/repos/{guard.ORGANISATION}/{repository}"
    numbers: list[int] = []
    for page in range(1, MAX_PAGES + 1):
        items = fetch(f"{root}/pulls?state=open&base=main&per_page=100&page={page}")
        if not isinstance(items, list) or len(items) > 100:
            raise RunnerError("pull request list page is invalid")
        for item in items:
            number = item.get("number") if isinstance(item, dict) else None
            if not _strict_int(number) or number in numbers:
                raise RunnerError("pull request list entry is invalid")
            numbers.append(number)
        if len(items) < 100:
            return numbers
    raise RunnerError("pull request list reached its page bound")


def _verdict_digest(verdict: guard.Verdict) -> str:
    """The identity of a shadow check: verdict, reason and detail.

    The evidence is not part of it. It names the ``main`` tip, so hashing it
    would write a new check on every open pull request each time ``main``
    moves, and every check counts toward the bound of ``_already_reported``.
    Consequence: when only the evidence changes on one head (another
    approving owner, a rerun of CI), the write is skipped and the summary of
    the newest shadow check still shows the earlier evidence. The report the
    command prints always carries the current evidence.
    """
    core = json.dumps(
        [verdict.verdict, verdict.reason, verdict.detail],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(core.encode("utf-8")).hexdigest()


def shadow_check_body(verdict: guard.Verdict, head: str) -> dict:
    """The check run a shadow evaluation writes. Always ``neutral``."""
    if not isinstance(head, str) or _SHA.fullmatch(head) is None:
        raise RunnerError("verdict names no head commit")
    text = json.dumps(verdict.as_dict(), sort_keys=True, indent=2)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n... (truncated)"
    return {
        "name": SHADOW_CHECK_NAME,
        "head_sha": head,
        "status": "completed",
        "conclusion": SHADOW_CONCLUSION,
        "external_id": _verdict_digest(verdict),
        "output": {
            "title": f"shadow verdict: {verdict.verdict} ({verdict.reason})",
            "summary": (
                "Shadow mode. This check is informational: it is not a "
                "required check, it cannot block this pull request, and its "
                "conclusion is always neutral. It shows what the "
                "content-bound merge guard would decide for this head.\n\n"
                "```json\n" + text + "\n```"
            ),
        },
    }


def _already_reported(
    root: str, head: str, digest: str, fetch: Fetch, app_client_id: str | None
) -> bool:
    """Whether the guard App's newest shadow check on ``head`` has this verdict.

    Only a check that the guard App itself published counts. Any workflow of
    the RuleSpec repository can create a check with the shadow name and the
    (public) verdict digest; trusting it would let a pull request suppress its
    own verdict. Without the App's client ID nothing counts and the verdict is
    written again.
    """
    if not isinstance(app_client_id, str) or not app_client_id:
        return False
    name = urllib.parse.quote(SHADOW_CHECK_NAME, safe="")
    # "List check runs for a Git reference"
    # (https://docs.github.com/en/rest/checks/runs#list-check-runs-for-a-git-reference):
    # ``per_page`` is at most 100, ``page`` selects a page, the response
    # carries ``total_count`` and ``check_runs``, and ``filter`` is ``latest``
    # (the default, "returns the most recent check runs") or ``all``. The
    # document does not say how ``latest`` groups runs. The reader asks for
    # ``all``: every shadow-named run of the head is listed, ``total_count``
    # has one meaning (the number of listed runs) and the newest run of the
    # App is the one with the highest id. With ``latest`` the page arithmetic
    # below would depend on whether ``total_count`` counts the runs before or
    # after the filter for runs an App creates [check live].
    # The price: every shadow-named run on the head counts toward the bound,
    # the App's own earlier ones included. The reader follows the pages up to
    # ``MAX_CHECK_RUN_PAGES`` and refuses past that bound: a listing it did
    # not read whole never suppresses a verdict and never selects the
    # "newest" check.
    listing_path = (
        f"{root}/commits/{head}/check-runs"
        f"?check_name={name}&filter=all&per_page=100&page="
    )
    runs: list[dict] = []
    seen: set[int] = set()
    expected: int | None = None
    for page in range(1, MAX_CHECK_RUN_PAGES + 1):
        listing = fetch(f"{listing_path}{page}")
        page_runs = listing.get("check_runs") if isinstance(listing, dict) else None
        if (
            not isinstance(page_runs, list)
            or len(page_runs) > 100
            or type(listing.get("total_count")) is not int
        ):
            raise RunnerError("check run listing is incomplete")
        if expected is None:
            expected = listing["total_count"]
        elif listing["total_count"] != expected:
            raise RunnerError("check run listing changed while it was read")
        for run in page_runs:
            if (
                not isinstance(run, dict)
                or not _strict_int(run.get("id"))
                or run["id"] in seen
            ):
                raise RunnerError("check run listing entry is invalid")
            seen.add(run["id"])
            runs.append(run)
        if len(runs) == expected:
            break
        if len(runs) > expected or len(page_runs) < 100:
            raise RunnerError("check run listing is incomplete")
    else:
        raise RunnerError("check run listing reached its page bound")
    ours = []
    for run in runs:
        app = run.get("app")
        if (
            run.get("name") == SHADOW_CHECK_NAME
            and isinstance(app, dict)
            and app.get("client_id") == app_client_id
        ):
            ours.append(run)
    if not ours:
        return False
    newest = max(ours, key=lambda run: run["id"])
    return newest.get("external_id") == digest and newest.get("status") == "completed"


def report_one(
    repository: str,
    number: int,
    mode: str,
    *,
    fetch: Fetch,
    post: Post | None,
    app_client_id: str | None = None,
) -> dict:
    """Evaluate one pull request and, in shadow mode, publish the verdict.

    The verdict is the evaluator's. ``publication`` says what was written:
    ``none`` (dry run), ``created``, ``unchanged`` (the newest shadow check
    already carries this verdict), ``no_head`` (the evaluation failed before
    it could name a head, so there is no commit to report on) or ``error``.
    """
    verdict = guard.evaluate(repository, number, fetch=fetch)
    entry = {"pull_request": number, "verdict": verdict.as_dict()}
    if mode == DRY_RUN:
        entry["publication"] = "none"
        return entry
    if mode != SHADOW or post is None:
        raise RunnerError("unsupported mode")
    head = verdict.evidence.get("head")
    if not isinstance(head, str) or _SHA.fullmatch(head) is None:
        entry["publication"] = "no_head"
        return entry
    root = f"/repos/{guard.ORGANISATION}/{repository}"
    try:
        body = shadow_check_body(verdict, head)
        if _already_reported(root, head, body["external_id"], fetch, app_client_id):
            entry["publication"] = "unchanged"
            return entry
        created = post(root + "/check-runs", body)
        if (
            not isinstance(created, dict)
            or not _strict_int(created.get("id"))
            or created.get("head_sha") != head
            or created.get("name") != SHADOW_CHECK_NAME
            or created.get("conclusion") != SHADOW_CONCLUSION
        ):
            raise RunnerError("GitHub did not confirm the shadow check run")
        entry["publication"] = "created"
        entry["check_run_id"] = created["id"]
    except RunnerError as exc:
        entry["publication"] = "error"
        entry["publication_error"] = str(exc)
    return entry


def run(
    repository: str,
    numbers: list[int] | None,
    mode: str,
    *,
    fetch: Fetch,
    post: Post | None = None,
    app_client_id: str | None = None,
) -> tuple[dict, int]:
    """Evaluate ``numbers``, or every open pull request to ``main``.

    Returns the report and the exit code. The exit code is about the run, not
    the verdicts: ``EXIT_OK`` when every pull request was evaluated and every
    intended publication happened, ``EXIT_RUN_ERROR`` otherwise. A ``fail``
    verdict is a result, not a run error, unless its reason is a failed read.
    """
    report: dict = {
        "schema": REPORT_SCHEMA,
        "repository": f"{guard.ORGANISATION}/{repository}",
        "mode": mode,
        "check_name": SHADOW_CHECK_NAME if mode == SHADOW else None,
        "sweep": numbers is None,
        "complete": False,
        "results": [],
    }
    try:
        if mode not in MODES:
            raise RunnerError("unsupported mode")
        if mode == SHADOW and post is None:
            raise RunnerError("shadow mode has no check writer")
        if not isinstance(repository, str) or _REPOSITORY.fullmatch(repository) is None:
            raise RunnerError("invalid repository")
        if numbers is None:
            numbers = open_pull_requests(repository, fetch)
        elif not numbers or any(not _strict_int(number) for number in numbers):
            raise RunnerError("invalid pull request number")
        for number in numbers:
            report["results"].append(
                report_one(
                    repository,
                    number,
                    mode,
                    fetch=fetch,
                    post=post,
                    app_client_id=app_client_id,
                )
            )
    except RunnerError as exc:
        report["error"] = str(exc)
        return report, EXIT_RUN_ERROR
    if any(entry["publication"] == "error" for entry in report["results"]):
        report["error"] = "a shadow check run was not published"
        return report, EXIT_RUN_ERROR
    if any(
        entry["verdict"]["reason"] in _UNREAD_REASONS for entry in report["results"]
    ):
        # The verdict is ``fail`` (closed), and the run says it could not read.
        report["error"] = "an evaluation could not read GitHub"
        return report, EXIT_RUN_ERROR
    report["complete"] = True
    return report, EXIT_OK


def command(args, *, environ=None, opener=None) -> int:
    """``axiom-encode merge-guard-run``: print one report as one JSON line.

    Exit 0 when the run completed, 1 when it could not evaluate or publish
    (also when the App token is absent). (2 is a usage error.) Verdicts are
    in the report; they do not change the exit code.
    """
    environ = os.environ if environ is None else environ
    token = environ.get(TOKEN_ENV, "")
    numbers = None if args.pr is None else [args.pr]
    if _TOKEN.fullmatch(token) is None:
        # Fail closed and say why. No anonymous fallback: an unauthenticated
        # sweep would hit the public rate limit and report partial results.
        report = {
            "schema": REPORT_SCHEMA,
            "repository": f"{guard.ORGANISATION}/{args.repository}",
            "mode": args.mode,
            "check_name": SHADOW_CHECK_NAME if args.mode == SHADOW else None,
            "sweep": numbers is None,
            "complete": False,
            "results": [],
            "error": (
                f"{TOKEN_ENV} is absent or malformed: the runner needs an "
                "installation token of the guard GitHub App"
            ),
        }
        code = EXIT_RUN_ERROR
    else:
        report, code = run(
            args.repository,
            numbers,
            args.mode,
            fetch=token_reader(token, opener=opener),
            post=check_writer(token, opener=opener) if args.mode == SHADOW else None,
            app_client_id=environ.get(APP_CLIENT_ID_ENV) or None,
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return code


# --- the enforcing runner (change 3b) ----------------------------------------
#
# Why the result can never be ``neutral`` (or any conclusion other than
# success or failure that this code chooses): the enforcing runner creates no
# check run. GitHub derives the check from the Actions job, and the job's
# conclusion comes from the exit code of this process: 0 is ``success``, any
# other code is ``failure``. A cancelled or timed-out job is ``cancelled``,
# which does not satisfy a required workflow. The workflow has no ``if`` and
# no ``continue-on-error``, so no step can turn a non-zero code into a skip or
# a success (``tests/test_merge_guard_enforce_workflow.py``).


class _CountedFetch:
    """``fetch`` with a count and a hard cap of requests for one run."""

    def __init__(self, fetch: Fetch, cap: int):
        self._fetch = fetch
        self._cap = cap
        self.count = 0

    def __call__(self, path: str) -> object:
        if self.count >= self._cap:
            raise RunnerError("the run reached its request cap")
        self.count += 1
        return self._fetch(path)


def _event_object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RunnerError(f"event payload has no {label} object")
    return value


def _event_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise RunnerError(f"event payload {label} is not an exact commit")
    return value


def read_event(path: object) -> dict:
    """The event payload of the run: one bounded, strict JSON object."""
    if not isinstance(path, str) or not path:
        raise RunnerError("GITHUB_EVENT_PATH is absent")
    try:
        with open(path, "rb") as stream:
            raw = stream.read(MAX_EVENT_BYTES + 1)
    except OSError as exc:
        raise RunnerError(
            f"event payload is unreadable: {type(exc).__name__}"
        ) from None
    if len(raw) > MAX_EVENT_BYTES:
        raise RunnerError("event payload exceeds its bound")
    try:
        # Refuses invalid UTF-8, a duplicate key, a non-finite number, JSON
        # nested past the recursion limit and anything that is not an object.
        return guard._parse_json_object(raw)
    except Exception as exc:
        raise RunnerError(
            f"event payload is not strict JSON: {type(exc).__name__}"
        ) from None


def _required_check_runs(root: str, head: str, fetch: Fetch) -> list[dict]:
    """The check runs named like the required check on ``head``, read whole."""
    name = urllib.parse.quote(guard.REQUIRED_CHECK_NAME, safe="")
    listing = fetch(
        f"{root}/commits/{head}/check-runs?check_name={name}&filter=latest&per_page=100"
    )
    runs = listing.get("check_runs") if isinstance(listing, dict) else None
    if (
        not isinstance(runs, list)
        or len(runs) > 100
        or type(listing.get("total_count")) is not int
        or listing["total_count"] != len(runs)
        or any(not isinstance(run, dict) for run in runs)
    ):
        raise RunnerError("required check listing is incomplete")
    return runs


def _ci_settled(root: str, head: str, fetch: Fetch) -> bool:
    """Whether every required-named check run on ``head`` is completed.

    At least one must exist. This decides only when to evaluate again; the
    evaluation makes its own reads and decides the verdict. The match is by
    name only: any completed check with the required name ends the wait. For
    a pull request that is harmless, because the evaluation that follows
    verifies the check itself and gives ``fail`` for a foreign one.
    """
    runs = _required_check_runs(root, head, fetch)
    return bool(runs) and all(run.get("status") == "completed" for run in runs)


def _group_ci_succeeded(root: str, group_head: str, fetch: Fetch) -> bool:
    """Whether the merge group's own required check has completed.

    For a merge group an early end of the wait is not harmless: the
    evaluation would then run long before the merge. So a name is not enough
    here. Every listed check must be the Actions check of a ``merge_group``
    run of the required workflow on ``group_head``; a listed check that is
    anything else is an error of the run. A check that says ``completed``
    while its workflow run does not is waited for.

    Returns False while the check is absent or not completed. Once it has
    completed, anything but ``success``, of the check or of its workflow run,
    is an error of the run: the job fails and nothing is evaluated. A red
    group therefore never has a green guard job, whatever the repository's own
    requirement of that check does.
    """
    runs = _required_check_runs(root, group_head, fetch)
    if len(runs) > MAX_GROUP_CHECK_RUNS:
        # Bounds the reads of one poll (``MERGE_GROUP_POLL_READS``).
        raise RunnerError(
            "more checks with the required name on the merge group head than "
            "one poll reads"
        )
    if not runs or any(run.get("status") != "completed" for run in runs):
        return False
    repository = root.removeprefix("/repos/")
    prefix = f"https://github.com/{repository}/actions/runs/"
    foreign = RunnerError(
        "a check with the required name on the merge group head is not the "
        "merge group run of the required workflow"
    )
    for run in runs:
        app = run.get("app")
        details = run.get("details_url")
        run_id, _, job = (
            details.removeprefix(prefix).partition("/job/")
            if isinstance(details, str) and details.startswith(prefix)
            else ("", "", "")
        )
        if (
            run.get("name") != guard.REQUIRED_CHECK_NAME
            or not isinstance(app, dict)
            or app.get("id") != guard.REQUIRED_CHECK_APP_ID
            or run.get("head_sha") != group_head
            or _RUN_ID.fullmatch(run_id) is None
            or _RUN_ID.fullmatch(job) is None
        ):
            raise foreign
        workflow = fetch(f"{root}/actions/runs/{run_id}")
        named = workflow.get("repository") if isinstance(workflow, dict) else None
        if (
            not isinstance(workflow, dict)
            or str(workflow.get("id")) != run_id
            or workflow.get("event") != MERGE_GROUP_RUN_EVENT
            or workflow.get("path") != guard.REQUIRED_CHECK_WORKFLOW_PATH
            or workflow.get("head_sha") != group_head
            or not isinstance(named, dict)
            or named.get("full_name") != repository
        ):
            raise foreign
        if workflow.get("status") != "completed":
            return False
        if run.get("conclusion") != "success" or workflow.get("conclusion") != (
            "success"
        ):
            raise RunnerError("the required check of the merge group did not succeed")
    return True


def _wait_for_group_ci(
    root: str, group_head: str, fetch: Fetch, sleep: Sleep, clock: Clock, report: dict
) -> None:
    """Return when the merge group's required check has completed.

    Raises ``RunnerError`` when it has not within ``MERGE_GROUP_WAIT_SECONDS``
    on the injected clock or ``MAX_MERGE_GROUP_POLLS`` polls. Records the
    polls and the seconds waited in ``report``.
    """
    start = clock()
    while True:
        report["ci_polls"] += 1
        try:
            settled = _group_ci_succeeded(root, group_head, fetch)
        finally:
            report["waited_seconds"] = max(0, int(clock() - start))
        if settled:
            return
        if (
            clock() - start >= MERGE_GROUP_WAIT_SECONDS
            or report["ci_polls"] >= MAX_MERGE_GROUP_POLLS
        ):
            raise RunnerError(
                "the required check of the merge group did not complete within the wait"
            )
        sleep(MERGE_GROUP_POLL_SECONDS)


def _wait_for_ci(
    root: str, head: str, fetch: Fetch, sleep: Sleep, clock: Clock
) -> tuple[int, int]:
    """Poll until the required check on ``head`` settles or the wait ends.

    Returns the seconds waited and the number of polls. Bounded twice: by
    ``CI_WAIT_SECONDS`` on the injected clock and by ``MAX_CI_POLLS``.
    """
    start = clock()
    polls = 0
    while polls < MAX_CI_POLLS:
        polls += 1
        if _ci_settled(root, head, fetch):
            break
        if clock() - start >= CI_WAIT_SECONDS or polls >= MAX_CI_POLLS:
            break
        sleep(CI_POLL_SECONDS)
    return max(0, int(clock() - start)), polls


def _bind_event_head(verdict: guard.Verdict, event_head: str) -> guard.Verdict:
    """Refuse a verdict that the evaluator reached for another head.

    A ``fail`` that names no head (the evaluation stopped before it read one)
    stays as it is. Every other verdict must name the event head: a head that
    moved, or a re-run of the run of an older head, gets ``fail``.
    """
    head = verdict.evidence.get("head")
    if head == event_head:
        return verdict
    if verdict.verdict == guard.FAIL and head is None:
        return verdict
    return guard.Verdict(
        guard.FAIL,
        R_EVENT_HEAD_STALE,
        "the verdict is for another head than the head of this run's event",
        {
            "pull_request": verdict.evidence.get("pull_request"),
            "event_head": event_head,
            "evaluated_head": head,
            "evaluated_verdict": verdict.verdict,
            "evaluated_reason": verdict.reason,
        },
    )


def _enforce_pull_request(
    name: str, payload: dict, fetch: Fetch, sleep: Sleep, clock: Clock, report: dict
) -> guard.Verdict:
    pull = _event_object(payload.get("pull_request"), "pull_request")
    number = pull.get("number")
    if not _strict_int(number) or type(payload.get("number")) is not int:
        raise RunnerError("event payload names no pull request number")
    if payload["number"] != number:
        raise RunnerError("event payload names two pull request numbers")
    event_head = _event_sha(
        _event_object(pull.get("head"), "pull request head").get("sha"), "head"
    )
    report["pull_request"] = number
    report["event_head"] = event_head
    report["evaluations"] = 1
    verdict = _bind_event_head(guard.evaluate(name, number, fetch=fetch), event_head)
    if verdict.verdict == guard.FAIL and verdict.reason == guard.R_CI:
        root = f"/repos/{guard.ORGANISATION}/{name}"
        try:
            waited, polls = _wait_for_ci(root, event_head, fetch, sleep, clock)
        except RunnerError as exc:
            # The verdict stays ``fail`` and the run says why it stopped.
            report["error"] = "waiting for the required check failed: " + str(exc)
            return verdict
        except Exception as exc:  # an injected reader's own error type
            report["error"] = (
                "waiting for the required check failed: " + type(exc).__name__
            )
            return verdict
        report["waited_seconds"] = waited
        report["ci_polls"] = polls
        # Exactly one more evaluation, after the wait ended either way. Its
        # verdict is final; a second ``ci_not_green`` does not wait again.
        report["evaluations"] = 2
        verdict = _bind_event_head(
            guard.evaluate(name, number, fetch=fetch), event_head
        )
    return verdict


def _enforce_merge_group(
    name: str, payload: dict, fetch: Fetch, sleep: Sleep, clock: Clock, report: dict
) -> guard.Verdict:
    group = _event_object(payload.get("merge_group"), "merge_group")
    group_head = _event_sha(group.get("head_sha"), "merge group head")
    group_base = _event_sha(group.get("base_sha"), "merge group base")
    head_ref = group.get("head_ref")
    match = _QUEUE_REF.fullmatch(head_ref) if isinstance(head_ref, str) else None
    if match is None:
        raise RunnerError(
            "merge group head_ref is not the queue branch of one pull request to main"
        )
    if group.get("base_ref") != "refs/heads/main":
        raise RunnerError("merge group base_ref is not refs/heads/main")
    number = int(match.group(1))
    report["pull_request"] = number
    report["event_head"] = group_head
    # First the wait, then the one evaluation. A wait that fails or ends
    # without a completed check raises: the run is an error with no verdict.
    _wait_for_group_ci(
        f"/repos/{guard.ORGANISATION}/{name}", group_head, fetch, sleep, clock, report
    )
    report["evaluations"] = 1
    verdict = guard.evaluate_merge_group(
        name, group_head, group_base, number, fetch=fetch
    )
    bound = verdict.evidence.get("merge_group")
    if (verdict.verdict, verdict.reason) in SUCCESS_VERDICTS and (
        not isinstance(bound, dict)
        or bound.get("head") != group_head
        or bound.get("base") != group_base
        or verdict.evidence.get("pull_request") != number
    ):
        return guard.Verdict(
            guard.FAIL,
            R_EVENT_HEAD_STALE,
            "the verdict is not bound to the merge group of this run's event",
            {"pull_request": number, "event_head": group_head},
        )
    return verdict


def enforce(
    event_name: object,
    repository: object,
    payload: dict,
    *,
    fetch: Fetch,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> tuple[dict, int]:
    """Judge the pull request or merge group of one Actions event.

    ``repository`` is ``GITHUB_REPOSITORY`` (``TheAxiomFoundation/rulespec-us``)
    and ``payload`` the parsed event. Returns the report and the exit code:
    ``EXIT_OK`` only for a verdict of ``SUCCESS_VERDICTS`` that is bound to
    the event's head, in a run with no error. Every other outcome, every read
    error included, is ``EXIT_RUN_ERROR``. Reads only; there is no writer.
    """
    counted = _CountedFetch(fetch, MAX_ENFORCE_REQUESTS)
    report: dict = {
        "schema": ENFORCE_REPORT_SCHEMA,
        "event": event_name if isinstance(event_name, str) else None,
        "repository": repository if isinstance(repository, str) else None,
        "verdict": None,
        "requests": 0,
        "evaluations": 0,
        "ci_polls": 0,
        "waited_seconds": 0,
    }
    verdict = None
    try:
        prefix = guard.ORGANISATION + "/"
        name = (
            repository.removeprefix(prefix)
            if isinstance(repository, str) and repository.startswith(prefix)
            else None
        )
        if name is None or _REPOSITORY.fullmatch(name) is None:
            raise RunnerError("the repository is not a RuleSpec repository")
        if not isinstance(payload, dict):
            raise RunnerError("event payload is not an object")
        named = _event_object(payload.get("repository"), "repository")
        if named.get("full_name") != repository:
            raise RunnerError("event payload names another repository")
        if event_name == EVENT_PULL_REQUEST:
            verdict = _enforce_pull_request(
                name, payload, counted, sleep, clock, report
            )
        elif event_name == EVENT_MERGE_GROUP:
            verdict = _enforce_merge_group(name, payload, counted, sleep, clock, report)
        else:
            raise RunnerError("the event is not pull_request or merge_group")
    except RunnerError as exc:
        report["error"] = str(exc)
    except Exception as exc:  # fail closed on anything unforeseen
        report["error"] = "unforeseen error: " + type(exc).__name__
    report["requests"] = counted.count
    if verdict is not None:
        report["verdict"] = verdict.as_dict()
        if verdict.reason in _UNREAD_REASONS and "error" not in report:
            report["error"] = "an evaluation could not read GitHub"
    if (
        verdict is not None
        and "error" not in report
        and (verdict.verdict, verdict.reason) in SUCCESS_VERDICTS
    ):
        return report, EXIT_OK
    return report, EXIT_RUN_ERROR


def _require_source_event(event_name: object, payload: dict) -> None:
    """Refuse the inert path unless the event, too, is the encoder's own.

    The inert path is the one place where this command ends with success
    without an evaluation. It must not rest on ``GITHUB_REPOSITORY`` alone:
    no live ruleset run has shown what that variable is when an organization
    ruleset runs the workflow for a RuleSpec repository. So the event payload
    must agree, in every field that names the repository:

    - the event is ``pull_request`` or ``merge_group``, the workflow's two
      triggers;
    - ``repository.full_name`` is the encoder repository;
    - for ``pull_request``, the pull request's own base repository
      (``pull_request.base.repo.full_name``) is the encoder repository. A
      pull request of a RuleSpec repository names that repository there.

    A ``merge_group`` payload has no second field that names a repository.

    This narrows the assumption and does not remove it. If a ruleset run for
    a RuleSpec repository carried the encoder repository BOTH in
    ``GITHUB_REPOSITORY`` and in these payload fields, the job would pass
    with no evaluation. The activation checklist therefore holds the ruleset
    in Evaluate mode until a pull request with no lane shows a FAILED guard
    job.
    """
    if event_name not in (EVENT_PULL_REQUEST, EVENT_MERGE_GROUP):
        raise RunnerError("the event is not pull_request or merge_group")
    named = _event_object(payload.get("repository"), "repository")
    if named.get("full_name") != guard.ENCODER_REPOSITORY:
        raise RunnerError(
            "GITHUB_REPOSITORY names the source repository and the event "
            "payload names another repository"
        )
    if event_name == EVENT_PULL_REQUEST:
        pull = _event_object(payload.get("pull_request"), "pull_request")
        base = _event_object(pull.get("base"), "pull request base")
        base_repository = _event_object(base.get("repo"), "pull request base repo")
        if base_repository.get("full_name") != guard.ENCODER_REPOSITORY:
            raise RunnerError(
                "GITHUB_REPOSITORY names the source repository and the pull "
                "request of the event belongs to another repository"
            )


def enforce_command(
    *,
    environ=None,
    opener=None,
    sleep: Sleep = time.sleep,
    clock: Clock = time.monotonic,
) -> int:
    """``axiom-encode merge-guard-enforce``: the required workflow's command.

    Every input comes from the Actions runner environment:
    ``GITHUB_REPOSITORY``, ``GITHUB_EVENT_NAME``, ``GITHUB_EVENT_PATH`` and the
    job token in ``TOKEN_ENV``. No argument selects a pull request, a
    repository or a mode. Prints one JSON report line. Exit 0 only when the
    verdict lets the head merge; 1 otherwise.

    In ``TheAxiomFoundation/axiom-encode`` itself, where the workflow's
    triggers also fire on the encoder's own pull requests, the command is
    inert: it makes no network read and exits 0. It is inert only when
    ``GITHUB_REPOSITORY`` AND the event payload name the encoder repository
    (``_require_source_event``); when they disagree, or the event cannot be
    read, it exits 1.
    """
    environ = os.environ if environ is None else environ
    repository = environ.get("GITHUB_REPOSITORY")
    event_name = environ.get("GITHUB_EVENT_NAME")
    if repository == guard.ENCODER_REPOSITORY:
        report = {
            "schema": ENFORCE_REPORT_SCHEMA,
            "event": event_name if isinstance(event_name, str) else None,
            "repository": repository,
            "verdict": None,
            "requests": 0,
        }
        try:
            _require_source_event(
                event_name, read_event(environ.get("GITHUB_EVENT_PATH"))
            )
        except RunnerError as exc:
            report["error"] = str(exc)
            code = EXIT_RUN_ERROR
        except Exception as exc:  # fail closed on anything unforeseen
            report["error"] = "unforeseen error: " + type(exc).__name__
            code = EXIT_RUN_ERROR
        else:
            report["inert"] = INERT_SOURCE_REPOSITORY
            code = EXIT_OK
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
        return code
    token = environ.get(TOKEN_ENV, "")
    try:
        if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
            # Fail closed and say why. No anonymous fallback.
            raise RunnerError(
                f"{TOKEN_ENV} is absent or malformed: the enforcing runner "
                "needs the job token"
            )
        payload = read_event(environ.get("GITHUB_EVENT_PATH"))
    except RunnerError as exc:
        report = {
            "schema": ENFORCE_REPORT_SCHEMA,
            "event": event_name if isinstance(event_name, str) else None,
            "repository": repository if isinstance(repository, str) else None,
            "verdict": None,
            "requests": 0,
            "error": str(exc),
        }
        code = EXIT_RUN_ERROR
    else:
        report, code = enforce(
            event_name,
            repository,
            payload,
            fetch=token_reader(token, opener=opener),
            sleep=sleep,
            clock=clock,
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return code

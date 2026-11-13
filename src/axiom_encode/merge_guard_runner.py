"""Runner for the content-bound merge guard: evaluate and report. SHADOW ONLY.

This is change 3 of ``docs/content-bound-merge-guard.md`` section 8: the code
that the workflow ``.github/workflows/merge-guard-runner.yml`` calls. It runs
``content_merge_guard.evaluate`` on pull requests of one RuleSpec repository,
authenticated as the guard GitHub App, and reports the verdict.

What it can and cannot do:

- ``dry-run`` (the default) reads and prints. It writes nothing to GitHub.
- ``shadow`` also writes one check run per judged head, named
  ``SHADOW_CHECK_NAME``, whose conclusion is always ``neutral``. The verdict is
  in the title and the summary. A shadow check cannot block a merge and must
  never be made a required check: ``neutral`` satisfies a required check, so
  requiring it would require nothing.
- There is no enforcing mode. ``GUARD_CHECK_NAME`` is the name reserved for
  the enforcing check of a later change; this module refuses to publish it.

Nothing here carries authority. The list of open pull requests is only a work
list: every verdict comes from the evaluator's own live reads. A read that
fails, is truncated or reaches its page bound is an error of the run. It is
never reported as "no pull requests" or "nothing to check".

The App token is read from the environment variable ``TOKEN_ENV`` only. It is
never accepted on the command line and never printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Final

from axiom_encode import content_merge_guard as guard
from axiom_encode.reviewed_pr_admission import MAX_RESPONSE_BYTES, _NoRedirect

TOKEN_ENV: Final = "MERGE_GUARD_TOKEN"
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
MAX_SUMMARY_CHARS: Final = 4000
EXIT_OK: Final = 0
EXIT_RUN_ERROR: Final = 1

_API: Final = "https://api.github.com"
_REPOSITORY = re.compile(r"rulespec-[a-z]{2}(-[a-z0-9]+)*")
_TOKEN = re.compile(r"[A-Za-z0-9_.-]{20,512}")
_SHA = re.compile(r"[0-9a-f]{40}")
_CHECK_RUNS_PATH = re.compile(
    r"/repos/TheAxiomFoundation/rulespec-[a-z]{2}(-[a-z0-9]+)*/check-runs"
)

# Verdict reasons that mean "the evaluation could not read", not "the pull
# request is refused". The verdict is still ``fail``; the run is also an error.
_UNREAD_REASONS: Final = frozenset({guard.R_READ_FAILED, guard.R_EVALUATION_ERROR})

Fetch = Callable[[str], object]
Post = Callable[[str, dict], object]


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
    """One bounded request to the fixed GitHub API origin, as the App."""
    if (
        not isinstance(path, str)
        or not path.startswith("/repos/TheAxiomFoundation/")
        or any(c in path for c in ("#", "\\", "\r", "\n", " "))
        or ".." in path
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
        with opener.open(request, timeout=30) as response:
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
    except ValueError:
        raise RunnerError("GitHub response is not JSON") from None


def token_reader(token: str, *, opener=None) -> Fetch:
    """The evaluator's ``fetch``: bounded GET reads with the App token."""

    def fetch(path: str) -> object:
        return _request("GET", path, token, None, opener, 200)

    return fetch


def check_writer(token: str, *, opener=None) -> Post:
    """The one write this module makes: create a check run."""

    def post(path: str, body: dict) -> object:
        if _CHECK_RUNS_PATH.fullmatch(path) is None:
            raise RunnerError("the runner writes check runs only")
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


def _already_reported(root: str, head: str, digest: str, fetch: Fetch) -> bool:
    """Whether the newest shadow check on ``head`` carries this verdict."""
    name = urllib.parse.quote(SHADOW_CHECK_NAME, safe="")
    listing = fetch(
        f"{root}/commits/{head}/check-runs?check_name={name}&filter=latest&per_page=100"
    )
    runs = listing.get("check_runs") if isinstance(listing, dict) else None
    if (
        not isinstance(runs, list)
        or type(listing.get("total_count")) is not int
        or listing["total_count"] != len(runs)
    ):
        raise RunnerError("check run listing is incomplete")
    ours = []
    for run in runs:
        if not isinstance(run, dict) or not _strict_int(run.get("id")):
            raise RunnerError("check run listing entry is invalid")
        if run.get("name") == SHADOW_CHECK_NAME:
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
        if _already_reported(root, head, body["external_id"], fetch):
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
                report_one(repository, number, mode, fetch=fetch, post=post)
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
        )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return code

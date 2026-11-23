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
  the enforcing check of a later change (3b of the design). The one writer,
  ``check_writer``, refuses every body that is not a completed, neutral check
  named ``SHADOW_CHECK_NAME``, so this module cannot publish the real name.

Nothing here carries authority. The list of open pull requests is only a work
list: every verdict comes from the evaluator's own live reads. A read that
fails, is truncated or reaches its page bound is an error of the run. It is
never reported as "no pull requests" or "nothing to check", and an evaluation
that could not read everything it needed is never reported as a completed
run: its verdict is ``fail``, the report says ``complete: false`` and the
exit code is ``EXIT_RUN_ERROR`` (``_evaluation_incomplete``).

``complete: true`` means: every pull request of the work list this run
obtained was evaluated from complete reads, and every intended check was
written. It does not mean that the work list held every open pull request.
The list is read page by page while pull requests open and close; a pull
request that moves across a page boundary between two page reads can be
missing from the list (or be refused as a repeated number). A sweep is
therefore not a census, and the next sweep reads the list again.

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

_API: Final = "https://api.github.com"
_REPOSITORY = re.compile(r"rulespec-[a-z]{2}(-[a-z0-9]+)*")
# The token is opaque. GitHub changes its format and length (40 characters
# for an earlier installation token, about 520 for a stateless one, and it
# says to expect more), so nothing here depends on either. Two tests only:
# the characters are those of an HTTP ``Bearer`` credential (RFC 6750
# ``b64token``: ASCII letters, digits and ``-._~+/=``), so the value cannot
# split or fold the ``Authorization`` header (no space, no control character,
# nothing outside ASCII); and the length is inside transport bounds. The
# lower bound refuses an empty or placeholder value before any request; the
# upper bound is above any header a server accepts and is not a token format
# rule.
MIN_TOKEN_CHARS: Final = 20
MAX_TOKEN_CHARS: Final = 16 * 1024
_TOKEN = re.compile(r"[A-Za-z0-9._~+/=-]{%d,%d}" % (MIN_TOKEN_CHARS, MAX_TOKEN_CHARS))
_SHA = re.compile(r"[0-9a-f]{40}")
# ``_external_id``: ``pr-<number>:<sha256 hex>``.
_EXTERNAL_ID = re.compile(r"pr-[1-9][0-9]*:[0-9a-f]{64}")
_COMPARE_SEGMENT = re.compile(r"[0-9a-f]{40}\.\.\.[0-9a-f]{40}")
_CHECK_RUNS_PATH = re.compile(
    r"/repos/TheAxiomFoundation/rulespec-[a-z]{2}(-[a-z0-9]+)*/check-runs"
)

# Verdict reasons that mean "the evaluation could not read", not "the pull
# request is refused". The verdict is still ``fail``; the run is also an error.
_UNREAD_REASONS: Final = guard.INCOMPLETE_READ_REASONS
# The reasons of a COMPLETED evaluation, by verdict. An evaluation counts as
# completed only when its verdict and reason are listed here and it carries
# no incomplete-read mark. The list is closed on purpose: a reason the
# evaluator gains later is an error of the run until a reviewed change puts
# it on one side (``tests/test_merge_guard_runner.py`` requires every reason
# of the evaluator to be in exactly one of the two sets).
_DECIDED_REASONS: Final = {
    guard.PASS: frozenset({guard.R_PASS, guard.R_CONTROL_PLANE_PASS}),
    guard.NOT_APPLICABLE: frozenset(
        {guard.R_NO_GUARDED_CONTENT, guard.R_EARLIER_LANE, guard.R_INFRASTRUCTURE}
    ),
    guard.FAIL: frozenset(
        {
            guard.R_SIGNED_BACKFILL,
            guard.R_GUARDED_LEAF,
            guard.R_MANIFEST_REMOVED,
            guard.R_NO_LANE,
            guard.R_UNATTRIBUTED,
            guard.R_CONTENT_NOT_COVERED,
            guard.R_UNKNOWN_SCHEMA,
            guard.R_MIXED_SCHEMAS,
            guard.R_MULTIPLE_V6,
            guard.R_UNSUPPORTED_REPOSITORY,
            guard.R_PR_IDENTITY,
            guard.R_PR_DRAFT,
            guard.R_BASE_NOT_CURRENT,
            guard.R_TOO_MANY_MANIFESTS,
            guard.R_MANIFEST_UNPARSEABLE,
            guard.R_MANIFEST_SHAPE,
            guard.R_MANIFEST_CANDIDATE,
            guard.R_CITATION,
            guard.R_MANIFEST_PATH,
            guard.R_SCOPE_PATHS,
            guard.R_SCOPE_NOT_NEW,
            guard.R_APPLIED_FILES,
            guard.R_RUN,
            guard.R_DISPATCHER_MISMATCH,
            guard.R_DISPATCHER_NOT_ENROLLED,
            guard.R_CHANGES_REQUESTED,
            guard.R_NO_APPROVAL,
            guard.R_NOT_ENROLLED,
            guard.R_IS_AUTHOR,
            guard.R_IS_DISPATCHER,
            guard.R_AUTHORED_COMMITS,
            guard.R_NOT_CONTENT_BOUND,
            guard.R_NO_COUNTING_APPROVAL,
            guard.R_CI,
            guard.R_CHANGED,
        }
    ),
}

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


def _evaluation_incomplete(verdict: object) -> bool:
    """Whether a verdict (as a dict) is NOT the result of a completed evaluation.

    Fail closed: completed means a known verdict with a reason from the
    closed list of decisions and no incomplete-read mark of the evaluator.
    Everything else (a truncated tree, a listing past its page bound, a
    failed request, an unknown reason, a verdict of an unexpected shape) is
    an evaluation that could not read, and so an error of the run.
    """
    if not isinstance(verdict, dict):
        return True
    kind = verdict.get("verdict")
    reason = verdict.get("reason")
    evidence = verdict.get("evidence")
    decided = _DECIDED_REASONS.get(kind) if isinstance(kind, str) else None
    return (
        decided is None
        or not isinstance(reason, str)
        or reason not in decided
        or reason in _UNREAD_REASONS
        or not isinstance(evidence, dict)
        or guard.EVIDENCE_INCOMPLETE_READ in evidence
    )


def _external_id(number: int, verdict: guard.Verdict) -> str:
    """The identity of a shadow check: the pull request and its verdict."""
    return f"{_external_id_prefix(number)}{_verdict_digest(number, verdict)}"


def _external_id_prefix(number: int) -> str:
    if not _strict_int(number):
        raise RunnerError("a shadow check needs the pull request it judges")
    return f"pr-{number}:"


def _verdict_digest(number: int, verdict: guard.Verdict) -> str:
    """Digest of the pull request number, the verdict, reason and detail.

    The verdict is for a pull request, not for a commit: the evaluator reads
    the reviews and the CI run of ONE pull request. Two pull requests can
    have the same head commit and different verdicts, and a check run
    attaches to the commit. So the number is part of the identity (and of
    the title): without it the two verdicts would overwrite each other's
    "unchanged" state on every sweep and nobody could tell which pull request
    a check on the shared commit is about.

    The evidence is not part of it. It names the ``main`` tip, so hashing it
    would write a new check on every open pull request each time ``main``
    moves, and every check counts toward the bound of ``_already_reported``.
    Consequence: when only the evidence changes on one head (another
    approving owner, a rerun of CI), the write is skipped and the summary of
    the newest shadow check still shows the earlier evidence. The report the
    command prints always carries the current evidence.
    """
    core = json.dumps(
        [number, verdict.verdict, verdict.reason, verdict.detail],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(core.encode("utf-8")).hexdigest()


def shadow_check_body(verdict: guard.Verdict, head: str, number: int) -> dict:
    """The check run a shadow evaluation writes. Always ``neutral``.

    ``number`` is the pull request that was evaluated. It must be the one the
    verdict names: a body never carries the verdict of another pull request.
    """
    if not isinstance(head, str) or _SHA.fullmatch(head) is None:
        raise RunnerError("verdict names no head commit")
    if not _strict_int(number) or verdict.evidence.get("pull_request") != number:
        raise RunnerError("verdict is not for the pull request of the check")
    text = json.dumps(verdict.as_dict(), sort_keys=True, indent=2)
    if len(text) > MAX_SUMMARY_CHARS:
        text = text[:MAX_SUMMARY_CHARS] + "\n... (truncated)"
    return {
        "name": SHADOW_CHECK_NAME,
        "head_sha": head,
        "status": "completed",
        "conclusion": SHADOW_CONCLUSION,
        "external_id": _external_id(number, verdict),
        "output": {
            "title": (
                f"shadow verdict for #{number}: {verdict.verdict} ({verdict.reason})"
            ),
            "summary": (
                "Shadow mode. This check is informational: it is not a "
                "required check, it cannot block this pull request, and its "
                "conclusion is always neutral. It shows what the "
                "content-bound merge guard would decide for pull request "
                f"#{number} at this head. The verdict is for that pull "
                "request only: another pull request with the same head "
                "commit has its own shadow check on this commit.\n\n"
                "```json\n" + text + "\n```"
            ),
        },
    }


def _already_reported(
    root: str,
    head: str,
    number: int,
    external_id: str,
    fetch: Fetch,
    app_client_id: str | None,
) -> bool:
    """Whether the guard App's newest shadow check of pull request ``number``
    on ``head`` has this verdict.

    A check of ours that names ANOTHER pull request on the same commit is not
    looked at: it neither suppresses this verdict nor is it "the newest". A
    check of ours whose ``external_id`` names no pull request in the exact
    form of ``_external_id`` is kept in the comparison: it can make the
    verdict be written again and can never suppress it.

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
    prefix = _external_id_prefix(number)
    ours = []
    for run in runs:
        app = run.get("app")
        identity = run.get("external_id")
        if (
            run.get("name") != SHADOW_CHECK_NAME
            or not isinstance(app, dict)
            or app.get("client_id") != app_client_id
        ):
            continue
        if (
            isinstance(identity, str)
            and _EXTERNAL_ID.fullmatch(identity) is not None
            and not identity.startswith(prefix)
        ):
            continue  # ours, for another pull request with this head
        ours.append(run)
    if not ours:
        return False
    newest = max(ours, key=lambda run: run["id"])
    return (
        newest.get("external_id") == external_id and newest.get("status") == "completed"
    )


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
        body = shadow_check_body(verdict, head, number)
        if _already_reported(
            root, head, number, body["external_id"], fetch, app_client_id
        ):
            entry["publication"] = "unchanged"
            return entry
        created = post(root + "/check-runs", body)
        if (
            not isinstance(created, dict)
            or not _strict_int(created.get("id"))
            or created.get("head_sha") != head
            or created.get("name") != SHADOW_CHECK_NAME
            or created.get("conclusion") != SHADOW_CONCLUSION
            or created.get("external_id") != body["external_id"]
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
    the verdicts: ``EXIT_OK`` when every pull request of the work list was
    evaluated from complete reads and every intended publication happened,
    ``EXIT_RUN_ERROR`` otherwise. A ``fail`` verdict is a result, not a run
    error, unless the evaluation behind it could not read everything
    (``_evaluation_incomplete``): then the verdict stays ``fail`` AND the run
    is an error with ``complete: false``. ``complete`` is about the work list
    the run obtained (module docstring), not about every open pull request.
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
    if any(_evaluation_incomplete(entry.get("verdict")) for entry in report["results"]):
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

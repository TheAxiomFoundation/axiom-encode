#!/usr/bin/env python3
"""Fail-fast budget for repeated failed re-encode dispatches of one citation.

Motivation (axiom-encode#1495): individual citations burned 20-31 targeted
re-encode runs in a single week at API prices, all of them ad hoc
dispatches retrying a persistently failing gate. This guard runs as a
cheap unprotected job before the production-signing environment gate and
before any model call. When the same citation has already failed
``attempt_budget`` consecutive times (no intervening success) inside the
lookback window, an ad hoc dispatch is blocked with triage instructions.

Queue-driven dispatches never reach this script in CI — the
attempt_budget job's own condition skips them so the guard can never
stall a tranche. The ``queue_id`` report-only branch below survives as
defense in depth for odd dispatch paths and local use.

Repair replays are report-only for the numeric budget. Their prior-run
identity and preserved candidate are authenticated by the protected resolver
before any model call, so they must be able to reach that resolver even when
fresh attempts are exhausted.

A second rule covers repair replays too: when the two most recent consecutive
model-bearing failures for the citation ended with the same final rule-code
set, another dispatch is refused. The set comes from each failed run's own
diagnostics artifact (``generated/<lane>/final-rejected-candidate/issues.json``).
A run without that artifact, or with any final issue that names no validator
rule code (a YAML parse, compile or expected-value error), has no known set
and never triggers the rule. An exact-citation entry in
``REPEATED_FAILURE_OVERRIDE_JSON`` or ``ATTEMPT_BUDGET_OVERRIDE=true`` lifts it.

The guard is a cost damper, not a security gate: if the GitHub API cannot
be reached it fails open with a loud warning. Stdlib-only because it runs
on a bare runner before any toolchain setup.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_BUDGET = 3
DEFAULT_LOOKBACK_DAYS = 7
# Conclusions that consumed model spend and count against the budget.
FAILURE_CONCLUSIONS = {"failure", "timed_out"}
# Conclusions that neither consume budget nor reset the streak.
NEUTRAL_CONCLUSIONS = {
    "cancelled",
    "skipped",
    "neutral",
    "action_required",
    "startup_failure",
    "stale",
}
MAX_RUN_PAGES = 5
RUNS_PER_PAGE = 100
# Job name of the protected encode job; a counted failure must have reached
# it (not skipped) to prove the run actually spent model tokens. Runs this
# guard itself blocked conclude "failure" with the encode job skipped and
# must not feed back into the streak.
ENCODE_JOB_NAME = "Queue protected signed RuleSpec re-encode"
ENCODE_STEP_NAME = "Encode, review, validate, and apply"
# At most this many per-run jobs lookups per evaluation; beyond the cap a
# failure is counted without verification (conservative toward blocking).
MAX_JOB_LOOKUPS = 10
# Consecutive failures with one final rule-code set that refuse the next run.
REPEATED_FAILURE_LIMIT = 2
# Failure diagnostics: the artifact holds one tar of the packaged directory.
FAILURE_ARTIFACT_PREFIX = "targeted-reencode-failure"
FAILURE_ARCHIVE_NAME = "targeted-reencode-failure.tar"
FINAL_ISSUES_PATTERN = re.compile(
    r"generated/[A-Za-z0-9._-]+/final-rejected-candidate/issues\.json"
)
# The packaging step caps generated diagnostics at 64 MiB before tar and zip.
MAX_ARTIFACT_BYTES = 96 * 1024 * 1024
MAX_ISSUES_BYTES = 4 * 1024 * 1024
# Wall-clock budget for reading rule codes, well inside the job's 10 minutes;
# past it the rule stays silent (fails open) instead of timing out the job.
RULE_CODE_LOOKUP_SECONDS = 240
# "<module path>: <stage>: [<rule code>] <message>". Only validator rules
# carry a code; output defects without one (a YAML parse error, a compile
# error, a wrong expected value) leave the run's set unknown, because a fresh
# generation often clears them.
BRACKETED_RULE_CODE = re.compile(
    r"(?:\S+: )?[a-z][a-z-]*: \[(?P<code>[a-z][a-z0-9-]*(?::[a-z0-9-]+)*)\]"
)


@dataclass
class Decision:
    """Outcome of evaluating the failed-attempt streak for a citation."""

    streak: int
    budget: int
    counted_runs: list[dict] = field(default_factory=list)

    @property
    def exhausted(self) -> bool:
        return self.streak >= self.budget


def run_citation(run_name: object) -> str | None:
    """Extract the citation from a targeted re-encode run name.

    Run names look like ``Targeted signed RuleSpec re-encode
    [<queue>:<item>:<sha>] <citation>``; citations are corpus paths and
    never contain ``"] "``, so the segment after the last ``"] "`` is the
    citation.
    """
    if not isinstance(run_name, str) or "] " not in run_name:
        return None
    return run_name.rsplit("] ", 1)[-1]


def evaluate_attempt_budget(
    runs: list[dict],
    *,
    citation: str,
    budget: int,
    current_run_id: int,
    spent_model_tokens: Callable[[dict], bool] | None = None,
) -> Decision:
    """Count consecutive completed failures for ``citation``.

    ``runs`` may arrive in any order; they are sorted newest-first here.
    The streak counts failure conclusions from the most recent completed
    run backwards and stops at the first success. Neutral conclusions
    (cancelled and friends) are skipped without resetting the streak.

    ``spent_model_tokens`` classifies whether a run actually reached the
    encode job. It filters both directions: failure runs that never
    reached a model call — above all, runs this guard itself blocked — do
    not extend the streak, and phantom successes that never ran encode
    (e.g. a queue run re-run where encode requires attempt 1, leaving
    only successful/skipped jobs) do not reset it. Both are neutral.
    """
    decision = Decision(streak=0, budget=budget)
    matching = [
        run
        for run in runs
        if run_citation(run.get("name")) == citation
        and run.get("id") != current_run_id
        and run.get("status") == "completed"
    ]
    matching.sort(key=lambda run: str(run.get("created_at") or ""), reverse=True)
    for run in matching:
        conclusion = run.get("conclusion")
        if conclusion == "success":
            if spent_model_tokens is not None and not spent_model_tokens(run):
                continue
            break
        if conclusion in NEUTRAL_CONCLUSIONS:
            continue
        if conclusion in FAILURE_CONCLUSIONS:
            if spent_model_tokens is not None and not spent_model_tokens(run):
                continue
            decision.streak += 1
            decision.counted_runs.append(
                {
                    "id": run.get("id"),
                    "run_attempt": run.get("run_attempt", 1),
                    "created_at": run.get("created_at"),
                    "conclusion": conclusion,
                    "html_url": run.get("html_url"),
                }
            )
            continue
        # Unknown conclusion values are treated as neutral so that new
        # GitHub statuses cannot spuriously block encodes.
    return decision


def issue_rule_code(issue: object) -> str | None:
    """The rule code a final validator issue names, or ``None`` if unknown."""

    if not isinstance(issue, str):
        return None
    coded = BRACKETED_RULE_CODE.match(issue)
    return coded["code"] if coded else None


def rule_codes_from_issue_payloads(payloads: list[object]) -> frozenset[str] | None:
    """Union of rule codes across lanes; ``None`` unless every issue is named."""

    codes: set[str] = set()
    for payload in payloads:
        issues = payload.get("issues") if isinstance(payload, dict) else None
        if not isinstance(issues, list) or not issues:
            return None
        for issue in issues:
            code = issue_rule_code(issue)
            if code is None:
                return None
            codes.add(code)
    return frozenset(codes) if codes else None


def rule_codes_from_failure_artifact(zip_path: str) -> frozenset[str] | None:
    """Read final rule codes from a downloaded failure diagnostics artifact.

    Streams the packaged tar without extracting anything to disk and reads
    only bounded ``final-rejected-candidate/issues.json`` regular files.
    """

    payloads: list[object] = []
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != [FAILURE_ARCHIVE_NAME]:
            return None
        with (
            archive.open(FAILURE_ARCHIVE_NAME) as stream,
            tarfile.open(fileobj=stream, mode="r|") as packaged,
        ):
            for member in packaged:
                name = member.name.removeprefix("./")
                if FINAL_ISSUES_PATTERN.fullmatch(name) is None:
                    continue
                if not member.isreg() or member.size > MAX_ISSUES_BYTES:
                    return None
                handle = packaged.extractfile(member)
                if handle is None:
                    return None
                try:
                    payloads.append(json.loads(handle.read(MAX_ISSUES_BYTES + 1)))
                except (ValueError, UnicodeDecodeError, RecursionError):
                    return None
    if not payloads:
        return None
    return rule_codes_from_issue_payloads(payloads)


@dataclass
class RepeatedFailure:
    """Consecutive failures that ended with one final rule-code set."""

    rule_codes: frozenset[str]
    runs: list[dict]


def evaluate_repeated_failure(
    counted_runs: list[dict],
    rule_codes: Callable[[dict], frozenset[str] | None],
    *,
    limit: int = REPEATED_FAILURE_LIMIT,
) -> RepeatedFailure | None:
    """Whether the newest ``limit`` counted failures share a final rule set.

    ``counted_runs`` is the newest-first failure streak from
    :func:`evaluate_attempt_budget`. An unknown set on any of those runs
    leaves the rule silent.
    """

    recent = counted_runs[:limit]
    if len(recent) < limit:
        return None
    sets = []
    for run in recent:
        codes = rule_codes(run)
        if not codes:
            return None
        sets.append(codes)
    if any(codes != sets[0] for codes in sets[1:]):
        return None
    return RepeatedFailure(rule_codes=sets[0], runs=recent)


def make_encode_job_checker(
    fetch_jobs: Callable[[int], list[dict]],
    *,
    max_lookups: int = MAX_JOB_LOOKUPS,
    encode_job_name: str = ENCODE_JOB_NAME,
    encode_step_name: str = ENCODE_STEP_NAME,
) -> Callable[[dict], bool]:
    """Build a ``spent_model_tokens`` classifier backed by the jobs API.

    A run counts as model-spending only when its encode job contains the
    actual encode step with a non-skipped conclusion. Looking only at the
    enclosing job is insufficient: trusted-repair and other preflight steps
    can fail that job before the model runs. Past ``max_lookups`` (or on
    lookup errors) the classification is conservative toward blocking in both
    directions: unverified failures count against the budget and
    unverified successes stay neutral rather than resetting the streak.
    Bounded so a long history cannot stall the guard.
    """
    lookups = 0

    def _unverified(run: dict) -> bool:
        return run.get("conclusion") != "success"

    def spent_model_tokens(run: dict) -> bool:
        nonlocal lookups
        run_id = run.get("id")
        if not isinstance(run_id, int) or lookups >= max_lookups:
            return _unverified(run)
        lookups += 1
        try:
            jobs = fetch_jobs(run_id)
        except Exception:
            return _unverified(run)
        encode_jobs = [
            job
            for job in jobs
            if isinstance(job, dict) and job.get("name") == encode_job_name
        ]
        if not encode_jobs:
            return False
        encode_steps = [
            step
            for job in encode_jobs
            for step in (job.get("steps") or [])
            if isinstance(step, dict) and step.get("name") == encode_step_name
        ]
        return any(step.get("conclusion") != "skipped" for step in encode_steps)

    return spent_model_tokens


def _fetch_completed_runs(
    *,
    repo: str,
    workflow_file: str,
    token: str,
    lookback_days: int,
) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    runs: list[dict] = []
    for page in range(1, MAX_RUN_PAGES + 1):
        query = urllib.parse.urlencode(
            {
                "status": "completed",
                "created": f">={since}",
                "per_page": str(RUNS_PER_PAGE),
                "page": str(page),
            }
        )
        url = (
            f"https://api.github.com/repos/{repo}/actions/workflows/"
            f"{workflow_file}/runs?{query}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "axiom-encode-attempt-budget",
            },
        )
        payload = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    payload = json.load(response)
                break
            except (urllib.error.URLError, TimeoutError, ValueError):
                if attempt == 0:
                    time.sleep(2)
                else:
                    raise
        page_runs = (payload or {}).get("workflow_runs") or []
        runs.extend(run for run in page_runs if isinstance(run, dict))
        if len(page_runs) < RUNS_PER_PAGE:
            break
    return runs


def _fetch_run_jobs(*, repo: str, token: str, run_id: int) -> list[dict]:
    url = f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/jobs?per_page=30"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "axiom-encode-attempt-budget",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.load(response)
    jobs = (payload or {}).get("jobs") or []
    return [job for job in jobs if isinstance(job, dict)]


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def _fetch_failure_rule_codes(
    *, repo: str, token: str, run: dict, deadline: float | None = None
) -> frozenset[str] | None:
    """Final rule codes of one failed run, or ``None`` when unavailable.

    ``deadline`` is a :func:`time.monotonic` instant after which the download
    stops and the run's set is unknown.
    """

    if deadline is None:
        deadline = time.monotonic() + RULE_CODE_LOOKUP_SECONDS
    run_id = run.get("id")
    run_attempt = run.get("run_attempt") or 1
    if not isinstance(run_id, int) or not isinstance(run_attempt, int):
        return None
    name = f"{FAILURE_ARTIFACT_PREFIX}-{run_id}-{run_attempt}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "axiom-encode-attempt-budget",
    }
    query = urllib.parse.urlencode({"name": name, "per_page": "10"})
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/runs/{run_id}/artifacts?{query}",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        listing = json.load(response)
    artifacts = [
        artifact
        for artifact in (listing or {}).get("artifacts") or []
        if isinstance(artifact, dict)
        and artifact.get("name") == name
        and artifact.get("expired") is False
    ]
    if len(artifacts) != 1:
        return None
    size = artifacts[0].get("size_in_bytes")
    download_url = artifacts[0].get("archive_download_url")
    if (
        not isinstance(size, int)
        or size > MAX_ARTIFACT_BYTES
        or not isinstance(download_url, str)
        or not download_url.startswith("https://api.github.com/")
    ):
        return None
    # The API answers with a redirect to signed blob storage. Follow it by
    # hand so the bearer token never leaves api.github.com.
    opener = urllib.request.build_opener(_NoRedirect)
    try:
        with opener.open(
            urllib.request.Request(download_url, headers=headers), timeout=30
        ):
            return None
    except urllib.error.HTTPError as redirect:
        location = redirect.headers.get("Location")
        redirect.close()
        if redirect.code not in {301, 302, 303, 307, 308} or not location:
            return None
    if not location.startswith("https://") or time.monotonic() > deadline:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        zip_path = os.path.join(scratch, "artifact.zip")
        with (
            urllib.request.urlopen(
                urllib.request.Request(
                    location, headers={"User-Agent": headers["User-Agent"]}
                ),
                timeout=60,
            ) as response,
            open(zip_path, "wb") as sink,
        ):
            written = 0
            while chunk := response.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_ARTIFACT_BYTES or time.monotonic() > deadline:
                    return None
                sink.write(chunk)
        return rule_codes_from_failure_artifact(zip_path)


def repeated_failure_override(raw: str, *, citation: str, now: datetime) -> bool:
    """Whether an expiring exact-citation entry lifts the repeated-failure rule."""

    if not raw.strip():
        return False
    configuration = json.loads(raw)
    if not isinstance(configuration, dict):
        raise ValueError("repeated-failure override must be an object")
    if citation not in configuration:
        return False
    entry = configuration[citation]
    if not isinstance(entry, dict) or set(entry) != {"expires_at", "reason"}:
        raise ValueError("repeated-failure override requires expires_at and reason")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError("repeated-failure override requires a triage reason")
    if not isinstance(entry["expires_at"], str):
        raise ValueError("repeated-failure override expiration must be ISO")
    expiration = datetime.fromisoformat(entry["expires_at"])
    if expiration.tzinfo is None or expiration.utcoffset() is None:
        raise ValueError("repeated-failure override expiration needs a timezone")
    return expiration > now


def _append_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def configured_citation_budget(
    raw: str, *, citation: str, fallback: int, now: datetime
) -> int:
    """Resolve an expiring exact-citation limit without disabling enforcement."""
    if not raw.strip():
        return fallback
    configuration = json.loads(raw)
    if not isinstance(configuration, dict):
        raise ValueError("citation budget configuration must be an object")
    if citation not in configuration:
        return fallback
    entry = configuration[citation]
    if not isinstance(entry, dict) or set(entry) != {"budget", "expires_at", "reason"}:
        raise ValueError("citation budget requires budget, expires_at and reason")
    budget = entry["budget"]
    if type(budget) is not int or budget < 1:
        raise ValueError("citation budget must be a positive integer")
    if not isinstance(entry["reason"], str) or not entry["reason"].strip():
        raise ValueError("citation budget requires a triage reason")
    if not isinstance(entry["expires_at"], str):
        raise ValueError("citation budget expiration must be an ISO datetime")
    expiration = datetime.fromisoformat(entry["expires_at"])
    if expiration.tzinfo is None or expiration.utcoffset() is None:
        raise ValueError("citation budget expiration must include a timezone")
    return budget if expiration > now else fallback


def _render_summary(
    decision: Decision, *, citation: str, enforced: bool, blocked: bool
) -> list[str]:
    lines = ["### Failed-attempt budget", ""]
    lines.append(
        f"- Citation: `{citation}` — {decision.streak} consecutive failed "
        f"run(s) in the lookback window (budget {decision.budget})."
    )
    for run in decision.counted_runs:
        lines.append(
            f"  - {run.get('created_at')} — {run.get('conclusion')} — "
            f"{run.get('html_url')}"
        )
    if blocked:
        lines.append(
            "- **Blocked before the signing environment and any model call.** "
            "Triage the failing gate first (see axiom-encode#1494 for why "
            "regeneration alone rarely fixes these). To proceed anyway, set "
            "an expiring exact-citation limit in repository variable "
            "`ATTEMPT_BUDGET_BY_CITATION_JSON` (see "
            "`docs/targeted-reencode-attempt-budget.md`). This retains the "
            "other citations' limits. The repository-wide alternatives are "
            "`ATTEMPT_BUDGET_OVERRIDE=true` or raising `ATTEMPT_BUDGET`; "
            "remove temporary settings after the reviewed retry."
        )
    elif not enforced:
        lines.append(
            "- Report-only: queue dispatch, authenticated repair path, "
            "deterministic migration, or override; "
            "not blocking."
        )
    return lines


def is_retired_source_metadata_plan(source_bundle_json: str) -> bool:
    """Recognize the model-free lane; the protected resolver validates it."""

    try:
        payload = json.loads(source_bundle_json)
    except (ValueError, RecursionError):
        return False
    return isinstance(payload, dict) and payload.get("schema_version") in (
        "axiom-encode/retired-source-metadata-migration-plan/v1",
        "axiom-encode/retired-source-metadata-migration-plan/v2",
    )


def _render_repeated_failure(
    repeated: RepeatedFailure, *, citation: str, blocked: bool
) -> list[str]:
    codes = ", ".join(f"`{code}`" for code in sorted(repeated.rule_codes))
    lines = [
        "### Repeated failure",
        "",
        f"- Citation: `{citation}` — the last {len(repeated.runs)} model-bearing "
        f"runs failed with the same final rule codes: {codes}.",
    ]
    for run in repeated.runs:
        lines.append(f"  - {run.get('created_at')} — {run.get('html_url')}")
    if blocked:
        lines.append(
            "- **Blocked before the signing environment and any model call**, "
            "including repair replays. Another attempt rarely clears the same "
            "rules: fix the generator (prompt, repair guidance or deterministic "
            "repair) or a wrong validator rule first, or make sure the review "
            "finding addresses these rules. To retry anyway, add an expiring "
            "entry for this exact citation to the repository variable "
            "`REPEATED_FAILURE_OVERRIDE_JSON` (see "
            "`docs/targeted-reencode-attempt-budget.md`)."
        )
    else:
        lines.append(
            "- Report-only: queue dispatch, deterministic migration, or override; "
            "not blocking."
        )
    return lines


def main() -> int:
    citation = os.environ.get("CITATION", "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GH_TOKEN", "")
    workflow_file = os.environ.get("WORKFLOW_FILE", "targeted-signed-reencode.yml")
    queue_id = os.environ.get("QUEUE_ID", "").strip()
    repair_run_id = os.environ.get("REPAIR_RUN_ID", "").strip()
    metadata_migration = is_retired_source_metadata_plan(
        os.environ.get("SOURCE_BUNDLE_JSON", "")
    )
    override = os.environ.get("ATTEMPT_BUDGET_OVERRIDE", "").strip().lower()
    try:
        budget = int(os.environ.get("ATTEMPT_BUDGET", str(DEFAULT_BUDGET)))
    except ValueError:
        budget = DEFAULT_BUDGET
    if budget < 1:
        budget = DEFAULT_BUDGET
    try:
        budget = configured_citation_budget(
            os.environ.get("ATTEMPT_BUDGET_BY_CITATION_JSON", ""),
            citation=citation,
            fallback=budget,
            now=datetime.now(timezone.utc),
        )
    except (ValueError, TypeError):
        print(
            "attempt-budget: invalid citation budget configuration; "
            "retaining the repository-wide budget.",
            file=sys.stderr,
        )
    try:
        lookback_days = int(os.environ.get("LOOKBACK_DAYS", str(DEFAULT_LOOKBACK_DAYS)))
    except ValueError:
        lookback_days = DEFAULT_LOOKBACK_DAYS
    try:
        current_run_id = int(os.environ.get("GITHUB_RUN_ID", "0"))
    except ValueError:
        current_run_id = 0

    if not citation or not repo or not token:
        print(
            "attempt-budget: missing CITATION/GITHUB_REPOSITORY/GH_TOKEN; "
            "failing open.",
            file=sys.stderr,
        )
        return 0

    try:
        runs = _fetch_completed_runs(
            repo=repo,
            workflow_file=workflow_file,
            token=token,
            lookback_days=lookback_days,
        )
    except Exception as error:  # noqa: BLE001 - fail open on API trouble
        print(
            f"attempt-budget: could not list workflow runs ({error}); failing open.",
            file=sys.stderr,
        )
        return 0

    decision = evaluate_attempt_budget(
        runs,
        citation=citation,
        budget=budget,
        current_run_id=current_run_id,
        spent_model_tokens=make_encode_job_checker(
            lambda run_id: _fetch_run_jobs(repo=repo, token=token, run_id=run_id)
        ),
    )
    enforced = (
        queue_id == ""
        and repair_run_id == ""
        and not metadata_migration
        and override != "true"
    )
    blocked = enforced and decision.exhausted
    print(
        f"attempt-budget: citation={citation} streak={decision.streak} "
        f"budget={budget} enforced={enforced} blocked={blocked}"
    )
    if decision.counted_runs or blocked or override == "true":
        _append_summary(
            _render_summary(
                decision, citation=citation, enforced=enforced, blocked=blocked
            )
        )

    try:
        repeat_overridden = repeated_failure_override(
            os.environ.get("REPEATED_FAILURE_OVERRIDE_JSON", ""),
            citation=citation,
            now=datetime.now(timezone.utc),
        )
    except (ValueError, TypeError):
        repeat_overridden = False
        print(
            "attempt-budget: invalid repeated-failure override configuration; "
            "keeping the repeated-failure rule.",
            file=sys.stderr,
        )
    repeat_enforced = (
        queue_id == ""
        and not metadata_migration
        and override != "true"
        and not repeat_overridden
    )

    lookup_deadline = time.monotonic() + RULE_CODE_LOOKUP_SECONDS

    def rule_codes(run: dict) -> frozenset[str] | None:
        try:
            return _fetch_failure_rule_codes(
                repo=repo, token=token, run=run, deadline=lookup_deadline
            )
        except Exception as error:  # noqa: BLE001 - fail open on API trouble
            print(
                f"attempt-budget: could not read rule codes for run "
                f"{run.get('id')} ({error}); not counting it.",
                file=sys.stderr,
            )
            return None

    repeated = (
        evaluate_repeated_failure(decision.counted_runs, rule_codes)
        if decision.streak >= REPEATED_FAILURE_LIMIT
        else None
    )
    repeat_blocked = repeat_enforced and repeated is not None
    if repeated is not None:
        print(
            "attempt-budget: repeated final rule codes "
            f"{sorted(repeated.rule_codes)} enforced={repeat_enforced} "
            f"blocked={repeat_blocked}"
        )
        _append_summary(
            _render_repeated_failure(
                repeated, citation=citation, blocked=repeat_blocked
            )
        )
        if repeat_blocked:
            print(
                "::error title=Repeated final rule codes::The last "
                f"{len(repeated.runs)} model-bearing runs for {citation} failed "
                f"with {', '.join(sorted(repeated.rule_codes))}; see "
                "REPEATED_FAILURE_OVERRIDE_JSON in "
                "docs/targeted-reencode-attempt-budget.md"
            )
    if override == "true":
        print("attempt-budget: override set by dispatcher; not blocking.")
    if repair_run_id and not repeat_blocked:
        print(
            "attempt-budget: repair replay is authenticated by the protected "
            "resolver; not blocking."
        )
    if metadata_migration:
        print(
            "attempt-budget: retired source metadata migration makes no model call; "
            "not blocking."
        )
    return 1 if blocked or repeat_blocked else 0


if __name__ == "__main__":
    sys.exit(main())

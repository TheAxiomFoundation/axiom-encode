#!/usr/bin/env python3
"""Close draft encoder PRs that a newer PR or a landed encoding supersedes.

The targeted signed re-encode workflow opens one draft RuleSpec PR per run,
on branch ``axiom/signed-backfill-<country>-<run_id>-<attempt>`` with title
``Add signed encoding manifest for <citation>``. Retries for one citation
leave the older drafts open, and a draft stays open after its citation lands
on the default branch through another PR.

This script lists those open drafts and plans to close a draft when either:

- a newer open encoder PR names the same citation and targets the same base
  branch (the newest one is kept), or
- the default branch now holds a signed manifest for the same citation whose
  ``generated_at`` is later than the draft's own manifest.

It keeps a PR that is not a draft any more, that carries a commit not both
authored and committed by ``axiom-encode-bot``, whose branch is the base of an
open PR that stays open (a stacked repair), or whose RuleSpec CI passes while
the newer PR's does not. It prints each PR's CI state. It is a dry run by
default: it prints the plan and changes nothing. With ``--apply`` it closes
each planned PR (``--only`` limits that to the numbers reviewed in the dry
run) and comments why. It never deletes branches, so a closed PR can be
reopened. A person runs it; the encode workflow does not.

Authentication: ``GH_TOKEN`` or ``GITHUB_TOKEN``, else ``gh auth token``.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

API_VERSION = "2022-11-28"
BOT_AUTHOR = "axiom-encode-bot"
BOT_EMAIL = "axiom-encode-bot@users.noreply.github.com"
MANIFEST_PREFIX = ".axiom/encoding-manifests/"
TITLE_PATTERN = re.compile(r"Add signed encoding manifest for (?P<citation>\S+)")
COUNTRY_PATTERN = re.compile(r"[a-z]{2}")


class GitHub(Protocol):
    def get(self, path: str, params: dict[str, str] | None = None) -> Any: ...

    def post(self, path: str, body: dict[str, Any]) -> Any: ...

    def patch(self, path: str, body: dict[str, Any]) -> Any: ...


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


class ApiGitHub:
    """Minimal GitHub REST client over urllib, authenticated by a token."""

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        self._token = token
        self._api_url = api_url.rstrip("/")

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(
            f"{self._api_url}/{path.lstrip('/')}",
            data=data,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": "axiom-encode-close-superseded-prs",
                **({"Content-Type": "application/json"} if data is not None else {}),
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise GitHubError(exc.code, detail) from None
        return json.loads(raw) if raw else None

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        return self._request("GET", f"{path}{query}", None)

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("POST", path, body)

    def patch(self, path: str, body: dict[str, Any]) -> Any:
        return self._request("PATCH", path, body)


def _paged(github: GitHub, path: str, params: dict[str, str]) -> list[Any]:
    items: list[Any] = []
    for page in range(1, 51):
        batch = github.get(path, {**params, "per_page": "100", "page": str(page)})
        if not isinstance(batch, list):
            raise GitHubError(0, f"{path} did not return a list")
        items.extend(batch)
        if len(batch) < 100:
            break
    return items


@dataclass(frozen=True)
class EncoderPr:
    number: int
    citation: str
    base: str
    head: str
    head_sha: str
    created_at: str
    draft: bool
    url: str


@dataclass(frozen=True)
class Decision:
    number: int
    citation: str
    action: str  # "close" or "keep"
    reason: str
    url: str


def encoder_prs(
    pulls: list[dict[str, Any]], *, repo: str, country: str
) -> list[EncoderPr]:
    """Open PRs that the targeted signed re-encode workflow opened."""

    branch = re.compile(rf"axiom/signed-backfill-{re.escape(country)}-\d+-\d+")
    found: list[EncoderPr] = []
    for pull in pulls:
        head = pull.get("head") or {}
        title = TITLE_PATTERN.fullmatch(str(pull.get("title") or ""))
        if (
            pull.get("state") != "open"
            or title is None
            or branch.fullmatch(str(head.get("ref") or "")) is None
            or (head.get("repo") or {}).get("full_name") != repo
        ):
            continue
        found.append(
            EncoderPr(
                number=int(pull["number"]),
                citation=title["citation"],
                base=str((pull.get("base") or {}).get("ref") or ""),
                head=str(head["ref"]),
                head_sha=str(head.get("sha") or ""),
                created_at=str(pull.get("created_at") or ""),
                draft=bool(pull.get("draft")),
                url=str(pull.get("html_url") or ""),
            )
        )
    return found


def _contents_json(
    github: GitHub, repo: str, path: str, ref: str
) -> dict[str, Any] | None:
    try:
        payload = github.get(
            f"repos/{repo}/contents/{urllib.parse.quote(path)}", {"ref": ref}
        )
    except GitHubError as exc:
        if exc.status == 404:
            return None
        raise
    if not isinstance(payload, dict) or payload.get("encoding") != "base64":
        return None
    try:
        parsed = json.loads(base64.b64decode(payload.get("content") or ""))
    except (ValueError, UnicodeDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def landed_newer_encoding(
    github: GitHub, repo: str, pr: EncoderPr, default_branch: str
) -> str | None:
    """Why the default branch holds a newer signed encoding of the citation."""

    files = _paged(github, f"repos/{repo}/pulls/{pr.number}/files", {})
    for changed in files:
        path = str(changed.get("filename") or "")
        if not (path.startswith(MANIFEST_PREFIX) and path.endswith(".json")):
            continue
        if changed.get("status") not in {"added", "modified"}:
            continue
        ours = _contents_json(github, repo, path, pr.head_sha)
        if not ours or ours.get("citation") != pr.citation:
            continue
        theirs = _contents_json(github, repo, path, default_branch)
        if not theirs or theirs.get("citation") != pr.citation:
            continue
        ours_at = _generated_at(ours)
        theirs_at = _generated_at(theirs)
        if ours_at is not None and theirs_at is not None and theirs_at > ours_at:
            return (
                f"`{default_branch}` holds a newer signed encoding of "
                f"`{pr.citation}` (`{path}` generated {theirs_at.isoformat()}; "
                f"this PR's was generated {ours_at.isoformat()})"
            )
    return None


def _generated_at(manifest: dict[str, Any]) -> datetime | None:
    value = manifest.get("generated_at")
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def only_bot_commits(github: GitHub, repo: str, pr: EncoderPr) -> bool:
    """Every commit was authored and committed by the workflow's bot identity.

    A person who amends, rebases or cherry-picks a bot commit keeps the bot as
    author but becomes the committer, so both identities are checked.
    """

    commits = _paged(github, f"repos/{repo}/pulls/{pr.number}/commits", {})

    def is_bot(person: object) -> bool:
        return (
            isinstance(person, dict)
            and person.get("name") == BOT_AUTHOR
            and person.get("email") == BOT_EMAIL
        )

    return bool(commits) and all(
        is_bot((commit.get("commit") or {}).get("author"))
        and is_bot((commit.get("commit") or {}).get("committer"))
        for commit in commits
    )


def ci_state(github: GitHub, repo: str, sha: str) -> str:
    """Summarize the head commit's check runs.

    ``failure`` if any run failed, ``pending`` if any is unfinished,
    ``incomplete`` if any was cancelled or went stale, ``success`` if all
    passed or were skipped, and ``none`` without check runs.
    """

    runs: list[dict[str, Any]] = []
    for page in range(1, 11):
        payload = github.get(
            f"repos/{repo}/commits/{sha}/check-runs",
            {"per_page": "100", "page": str(page)},
        )
        batch = (payload or {}).get("check_runs") or []
        runs.extend(run for run in batch if isinstance(run, dict))
        if len(batch) < 100:
            break
    if not runs:
        return "none"
    conclusions = {run.get("conclusion") for run in runs}
    if conclusions & {"failure", "timed_out", "action_required", "startup_failure"}:
        return "failure"
    if any(run.get("status") != "completed" for run in runs):
        return "pending"
    if conclusions - {"success", "neutral", "skipped"}:
        return "incomplete"
    return "success"


def plan(
    github: GitHub, repo: str, *, country: str
) -> tuple[list[Decision], list[dict[str, Any]]]:
    """Decide, for every open encoder PR, whether it is superseded."""

    pulls = _paged(github, f"repos/{repo}/pulls", {"state": "open"})
    default_branch = str(github.get(f"repos/{repo}")["default_branch"])
    prs = encoder_prs(pulls, repo=repo, country=country)

    groups: dict[tuple[str, str], list[EncoderPr]] = defaultdict(list)
    for pr in prs:
        groups[(pr.citation, pr.base)].append(pr)
    reasons: dict[int, str] = {}
    superseded_by: dict[int, EncoderPr] = {}
    for members in groups.values():
        members.sort(key=lambda pr: (pr.created_at, pr.number))
        newest = members[-1]
        for pr in members[:-1]:
            superseded_by[pr.number] = newest
            reasons[pr.number] = (
                f"superseded by #{newest.number}, a newer open encoder PR for "
                f"`{pr.citation}` on base `{pr.base}`"
            )
    for pr in prs:
        if pr.number not in reasons:
            landed = landed_newer_encoding(github, repo, pr, default_branch)
            if landed:
                reasons[pr.number] = landed

    states: dict[str, str] = {}

    def state_of(pr: EncoderPr) -> str:
        if pr.head_sha not in states:
            states[pr.head_sha] = ci_state(github, repo, pr.head_sha)
        return states[pr.head_sha]

    keep: dict[int, str] = {}
    for pr in prs:
        if pr.number not in reasons:
            continue
        newer = superseded_by.get(pr.number)
        reasons[pr.number] += f" (RuleSpec CI: {state_of(pr)}"
        reasons[pr.number] += (
            f"; #{newer.number}: {state_of(newer)})" if newer is not None else ")"
        )
        if not pr.draft:
            keep[pr.number] = "marked ready for review"
        elif not only_bot_commits(github, repo, pr):
            keep[pr.number] = f"has commits not authored and committed by {BOT_AUTHOR}"
        elif (
            newer is not None
            and state_of(pr) == "success"
            and state_of(newer) != "success"
        ):
            keep[pr.number] = f"its RuleSpec CI passes and #{newer.number}'s does not"

    # A stacked repair builds on its base PR's branch: keep a base while any
    # PR on top of it stays open.
    closing = {number for number in reasons if number not in keep}
    by_head = {pr.head: pr for pr in prs}
    changed = True
    while changed:
        changed = False
        for pull in pulls:
            child = int(pull["number"])
            parent = by_head.get(str((pull.get("base") or {}).get("ref") or ""))
            if (
                parent is not None
                and parent.number in closing
                and child not in closing
                and pull.get("state") == "open"
            ):
                closing.discard(parent.number)
                keep[parent.number] = f"base of open #{child}"
                changed = True

    decisions = []
    for pr in sorted(prs, key=lambda pr: pr.number):
        if pr.number in closing:
            decisions.append(
                Decision(pr.number, pr.citation, "close", reasons[pr.number], pr.url)
            )
        elif pr.number in reasons:
            decisions.append(
                Decision(
                    pr.number,
                    pr.citation,
                    "keep",
                    f"{reasons[pr.number]}; kept: {keep[pr.number]}",
                    pr.url,
                )
            )
    return decisions, pulls


def comment_body(decision: Decision) -> str:
    return (
        f"Closing as superseded: {decision.reason}.\n\n"
        "Closed by `scripts/close_superseded_encoder_prs.py` in axiom-encode. "
        "The branch is kept; reopen this PR if it should stay."
    )


def apply(
    github: GitHub,
    repo: str,
    decisions: list[Decision],
    *,
    only: set[int] | None = None,
) -> list[str]:
    """Close each planned PR (restricted to ``only`` when given), then comment.

    Closing first means a failed close never leaves a "closing" comment on an
    open PR.
    """

    done = []
    for decision in decisions:
        if decision.action != "close":
            continue
        if only is not None and decision.number not in only:
            continue
        github.patch(f"repos/{repo}/pulls/{decision.number}", {"state": "closed"})
        github.post(
            f"repos/{repo}/issues/{decision.number}/comments",
            {"body": comment_body(decision)},
        )
        done.append(f"closed #{decision.number}")
    return done


def _numbers(raw: str) -> set[int]:
    try:
        numbers = {int(part) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected PR numbers like 1370,1372") from exc
    if not numbers or any(number < 1 for number in numbers):
        raise argparse.ArgumentTypeError("expected PR numbers like 1370,1372")
    return numbers


def _token() -> str:
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        return subprocess.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "no GitHub token: set GH_TOKEN or GITHUB_TOKEN, or log in to gh"
        ) from exc


def main(argv: list[str] | None = None, github: GitHub | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--country", default="us", help="two-letter country code")
    parser.add_argument(
        "--repo",
        help="RuleSpec repository (default TheAxiomFoundation/rulespec-<country>)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="apply",
        action="store_false",
        default=False,
        help="print the plan and change nothing (default)",
    )
    mode.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        help="close and comment on every PR the plan closes",
    )
    parser.add_argument(
        "--only",
        type=_numbers,
        help="with --apply, act only on these PR numbers from the reviewed dry run "
        "(comma-separated); a number the new plan no longer closes is skipped",
    )
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    args = parser.parse_args(argv)
    if COUNTRY_PATTERN.fullmatch(args.country) is None:
        parser.error("--country must be a two-letter lowercase country code")
    repo = args.repo or f"TheAxiomFoundation/rulespec-{args.country}"
    client = github if github is not None else ApiGitHub(_token())

    decisions, _pulls = plan(client, repo, country=args.country)
    if args.json:
        print(json.dumps([decision.__dict__ for decision in decisions], indent=2))
    else:
        for decision in decisions:
            print(
                f"{decision.action} #{decision.number} {decision.citation}: "
                f"{decision.reason}"
            )
        closing = sum(1 for decision in decisions if decision.action == "close")
        print(
            f"{closing} to close, {len(decisions) - closing} superseded but kept"
            f"{'' if args.apply else ' (dry run; pass --apply to close)'}"
        )
    if args.apply:
        for line in apply(client, repo, decisions, only=args.only):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())

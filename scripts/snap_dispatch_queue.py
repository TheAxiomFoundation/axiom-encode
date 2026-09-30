#!/usr/bin/env python3
"""Build SNAP encoding queues and dispatch them on a schedule.

Queue state lives on the ``encoding-queue-state`` branch, one JSON file per
queue under ``queues/``. Each hourly tick first reconciles the encode runs it
dispatched earlier, then sends the next pending items to
``targeted-signed-reencode.yml`` pinned to the current rulespec-us ``main``
tip, exactly as ad hoc encodes are dispatched.

The dispatcher adds no trust checks of its own. The encode workflow's
``production-signing`` approval and the review of each draft RuleSpec pull
request remain the gates; the dispatcher only decides what to send next and
records what happened.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

SCHEMA = "axiom-encode/snap-dispatch-queue/v1"
STATUSES = ("pending", "dispatched", "in_review", "done", "blocked")
QUEUE_STATES = ("paused", "active")
QUEUE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
JURISDICTION_PATTERN = re.compile(r"^us-[a-z]{2}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
API_VERSION = "2026-03-10"

ENCODE_WORKFLOW = "targeted-signed-reencode.yml"
BUDGET_JOB = "Enforce failed-attempt budget"
RUN_TITLE_PREFIX = "Targeted signed RuleSpec re-encode [adhoc:adhoc:adhoc] "
DISPATCH_ACTOR = "github-actions[bot]"
INVENTORY_PATH = "manifests/state-snap-manual-agent-queue.yaml"
RELEASES_DIR = "manifests/releases"
PROVISIONS_DIR = "data/corpus/provisions"
ENCODING_MANIFESTS_DIR = ".axiom/encoding-manifests"

DEFAULT_SETTINGS = {
    "country": "us",
    "pr_base_branch": "main",
    "max_in_flight": 4,
    "max_attempts": 2,
}
# Largest source unit, in characters of source text, sent as one item. Units
# are the largest subtrees at or under this size, so an item never contains
# another item and a whole manual is never sent as one encode.
DEFAULT_MAX_UNIT_CHARS = 20_000
# A dispatched run that still cannot be found after this long is treated as
# never created, and its item goes back to pending.
LOST_RUN_AFTER = timedelta(minutes=30)
# Cancellations by someone else before an item is blocked instead of retried.
# The dispatcher's own stale-base cancellations never count.
MAX_CANCELLATIONS = 3

SNAP_MARKERS = (
    re.compile(r"\bSNAP\b"),
    re.compile(r"supplemental nutrition assistance|food stamp", re.IGNORECASE),
    # Integrated manuals head shared policy with the programs it covers; in
    # the Utah manual that heading is "All Programs".
    re.compile(r"\bAll Programs\b"),
)
# Tables of contents are dense with dot leaders ("Introduction....... 12");
# policy text almost never has them. In the Oregon notebook, contents pages
# have 24+ leader runs and policy pages 0-2.
TOC_LEADER = re.compile(r"\.{8,}")
TOC_MIN_LEADERS = 10
# The encode workflow requires rulespec main to still be at the pinned tip
# when it starts (checkout verification) and again when it opens the pull
# request, so a run whose main moved fails at one of these steps.
STALE_BASE_STEPS = frozenset(
    {
        "Verify immutable checkout identities",
        "Push lane branch and open draft pull request",
    }
)
# Consecutive runs lost to a moving main before an item is set aside.
MAX_STALE = 6
COUNTED_FAILURES = frozenset({"failure", "timed_out", "startup_failure"})


# -- time -------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


# -- building ---------------------------------------------------------------


def natural_key(citation: str) -> tuple[Any, ...]:
    """Sort key that puts page-2 before page-10."""

    return tuple(
        (0, int(chunk), "") if chunk.isdigit() else (1, 0, chunk)
        for chunk in re.split(r"(\d+)", citation)
        if chunk
    )


def _is_snap(text: str) -> bool:
    return any(marker.search(text) for marker in SNAP_MARKERS)


def _is_table_of_contents(text: str) -> bool:
    return len(TOC_LEADER.findall(text)) >= TOC_MIN_LEADERS


def _label(record: dict[str, Any]) -> str:
    for field in ("heading", "citation_label", "citation_path"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def select_units(
    records: list[dict[str, Any]], *, max_unit_chars: int
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, list[str]]]:
    """Pick non-overlapping source units from one corpus scope.

    A unit is the largest subtree whose source text fits in ``max_unit_chars``.
    Records without their own body are containers; the encoder composes their
    text from descendants, so a small container (a Utah manual topic) is one
    unit, while a large one (a whole Oregon manual) is split into its children.

    Returns the units, counts of what was left out, and the left-out
    citations by reason so a person can check them.
    """

    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        citation = record.get("citation_path")
        if not isinstance(citation, str) or not citation:
            raise ValueError("corpus record lacks citation_path")
        if citation in by_path:
            raise ValueError(f"duplicate corpus citation_path: {citation}")
        by_path[citation] = record

    parent: dict[str, str | None] = {}
    for citation, record in by_path.items():
        declared = record.get("parent_citation_path")
        if isinstance(declared, str) and declared in by_path:
            parent[citation] = declared
            continue
        parts = citation.split("/")
        parent[citation] = next(
            (
                "/".join(parts[:end])
                for end in range(len(parts) - 1, 0, -1)
                if "/".join(parts[:end]) in by_path
            ),
            None,
        )

    children: dict[str, list[str]] = {citation: [] for citation in by_path}
    roots: list[str] = []
    for citation, parent_citation in parent.items():
        if parent_citation is None:
            roots.append(citation)
        else:
            children[parent_citation].append(citation)

    def own_text(citation: str) -> str:
        body = by_path[citation].get("body")
        return body if isinstance(body, str) else ""

    # Subtree sizes, children before parents.
    size: dict[str, int] = {}
    for root in roots:
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            citation, expanded = stack.pop()
            if expanded:
                size[citation] = len(own_text(citation)) + sum(
                    size[child] for child in children[citation]
                )
            else:
                stack.append((citation, True))
                stack.extend((child, False) for child in children[citation])
    if len(size) != len(by_path):
        raise ValueError("corpus scope has records outside its citation tree")

    def subtree_text(citation: str) -> str:
        parts: list[str] = []
        stack = [citation]
        while stack:
            current = stack.pop()
            record = by_path[current]
            parts.append(_label(record))
            parts.append(own_text(current))
            stack.extend(children[current])
        return "\n".join(parts)

    # Walk the tree in reading order and classify each candidate unit.
    candidates: list[tuple[str, str]] = []  # (citation, reason or "snap")
    container_text_dropped = 0
    stack = sorted(roots, key=natural_key, reverse=True)
    while stack:
        citation = stack.pop()
        if size[citation] == 0:
            candidates.append((citation, "empty"))
            continue
        if size[citation] > max_unit_chars and children[citation]:
            if own_text(citation).strip():
                container_text_dropped += 1
            stack.extend(sorted(children[citation], key=natural_key, reverse=True))
            continue
        text = subtree_text(citation)
        if _is_table_of_contents(text):
            candidates.append((citation, "table_of_contents"))
        elif _is_snap(text):
            candidates.append((citation, "snap"))
        else:
            candidates.append((citation, "not_snap"))

    # A PDF page between two SNAP pages is almost always the same SNAP rule
    # running across a page break, even when it never names the program.
    kept = {citation for citation, reason in candidates if reason == "snap"}
    siblings: dict[str | None, list[tuple[str, str]]] = {}
    for citation, reason in candidates:
        if reason in {"snap", "not_snap"}:
            siblings.setdefault(parent[citation], []).append((citation, reason))
    between: set[str] = set()
    for run in siblings.values():
        for index in range(1, len(run) - 1):
            citation, reason = run[index]
            if (
                reason == "not_snap"
                and by_path[citation].get("kind") == "page"
                and run[index - 1][1] == "snap"
                and run[index + 1][1] == "snap"
            ):
                between.add(citation)
    kept |= between

    units: list[dict[str, Any]] = []
    excluded: dict[str, list[str]] = {
        "empty": [],
        "table_of_contents": [],
        "not_snap": [],
    }
    oversize_count = 0
    for citation, reason in candidates:
        if citation not in kept:
            excluded[reason].append(citation)
            continue
        record = by_path[citation]
        oversize = size[citation] > max_unit_chars
        oversize_count += int(oversize)
        units.append(
            {
                "citation": citation,
                "jurisdiction": record.get("jurisdiction"),
                "label": _label(record),
                "chars": size[citation],
                **({"oversize": True} if oversize else {}),
                **(
                    {"kept_because": "between SNAP pages"}
                    if citation in between
                    else {}
                ),
            }
        )
    stats = {
        **{reason: len(citations) for reason, citations in excluded.items()},
        "between_snap_pages": len(between),
        "oversize": oversize_count,
        "container_text_dropped": container_text_dropped,
    }
    return units, stats, excluded


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"corpus row is not an object at {path}:{line_number}")
            records.append(record)
    return records


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def encoded_citations(rulespec_root: Path) -> set[str]:
    """Corpus citations that already have a signed manifest on this checkout."""

    citations: set[str] = set()
    for path in sorted((rulespec_root / ENCODING_MANIFESTS_DIR).rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        citation = payload.get("citation") if isinstance(payload, dict) else None
        if isinstance(citation, str) and citation:
            citations.add(citation)
    return citations


def read_toolchain(rulespec_root: Path) -> dict[str, str]:
    toolchain = tomllib.loads(
        (rulespec_root / ".axiom/toolchain.toml").read_text(encoding="utf-8")
    )["toolchain"]
    workflow = tomllib.loads(
        (rulespec_root / ".axiom/workflow-toolchain.toml").read_text(encoding="utf-8")
    )["workflow_toolchain"]
    return _toolchain_pins(toolchain, workflow)


def _toolchain_pins(
    toolchain: dict[str, Any], workflow: dict[str, Any]
) -> dict[str, str]:
    pins = {
        "corpus_release": toolchain.get("axiom_corpus_release"),
        "corpus_ref": workflow.get("axiom_corpus_ref"),
        "rules_engine_ref": workflow.get("axiom_rules_engine_ref"),
    }
    if not isinstance(pins["corpus_release"], str) or not pins["corpus_release"]:
        raise ValueError("rulespec toolchain has no axiom_corpus_release")
    for name in ("corpus_ref", "rules_engine_ref"):
        if not isinstance(pins[name], str) or not SHA_PATTERN.fullmatch(pins[name]):
            raise ValueError(f"rulespec workflow toolchain has no valid {name}")
    return pins


def build_queue(
    corpus_root: Path,
    rulespec_root: Path,
    *,
    queue_id: str,
    jurisdictions: list[str],
    max_unit_chars: int = DEFAULT_MAX_UNIT_CHARS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build a paused queue from the corpus release rulespec-us main pins."""

    import yaml

    if not QUEUE_ID_PATTERN.fullmatch(queue_id):
        raise ValueError(f"invalid queue id: {queue_id}")
    if not jurisdictions:
        raise ValueError("at least one jurisdiction is required")
    for jurisdiction in jurisdictions:
        if not JURISDICTION_PATTERN.fullmatch(jurisdiction):
            raise ValueError(f"invalid jurisdiction: {jurisdiction}")

    pins = read_toolchain(rulespec_root)
    corpus_head = _git_head(corpus_root)
    if corpus_head != pins["corpus_ref"]:
        raise ValueError(
            f"corpus checkout is at {corpus_head}, but rulespec-us pins "
            f"{pins['corpus_ref']}"
        )
    release = json.loads(
        (corpus_root / RELEASES_DIR / f"{pins['corpus_release']}.json").read_text(
            encoding="utf-8"
        )
    )
    release_scopes = {
        (scope["jurisdiction"], scope["document_class"], scope["version"])
        for scope in release.get("scopes", [])
    }
    inventory = yaml.safe_load(
        (corpus_root / INVENTORY_PATH).read_text(encoding="utf-8")
    )
    entries = {entry["jurisdiction"]: entry for entry in inventory["states"]}

    encoded = encoded_citations(rulespec_root)
    scopes: list[dict[str, str]] = []
    excluded: dict[str, int] = {}
    excluded_citations: dict[str, list[str]] = {}
    items: dict[str, dict[str, Any]] = {}
    for jurisdiction in jurisdictions:
        entry = entries.get(jurisdiction)
        if entry is None:
            raise ValueError(f"{jurisdiction} is not in the SNAP source inventory")
        if entry.get("queue_status") != "published_current":
            raise ValueError(
                f"{jurisdiction} SNAP sources are {entry.get('queue_status')!r}, "
                "not published_current"
            )
        for field in ("target_scope", "supporting_scope"):
            scope = entry.get(field)
            if not scope:
                continue
            key = (scope["jurisdiction"], scope["document_class"], scope["version"])
            if key not in release_scopes:
                raise ValueError(
                    f"{jurisdiction} {field} {'/'.join(key)} is not in corpus "
                    f"release {pins['corpus_release']}"
                )
            path = corpus_root / PROVISIONS_DIR / key[0] / key[1] / f"{key[2]}.jsonl"
            units, stats, left_out = select_units(
                _read_jsonl(path), max_unit_chars=max_unit_chars
            )
            for name, count in stats.items():
                excluded[name] = excluded.get(name, 0) + count
            for reason, citations in left_out.items():
                excluded_citations.setdefault(reason, []).extend(citations)
            scopes.append(
                {
                    "jurisdiction": key[0],
                    "document_class": key[1],
                    "version": key[2],
                    "items": len(units),
                }
            )
            for unit in units:
                if unit["citation"] in encoded:
                    unit["status"] = "done"
                    unit["note"] = "already encoded on rulespec-us main"
                else:
                    unit["status"] = "pending"
                unit["attempts"] = []
                items.setdefault(unit["citation"], unit)

    ordered = sorted(
        items.values(),
        key=lambda item: (item["jurisdiction"] or "", natural_key(item["citation"])),
    )
    stamp = _iso(now or _now())
    state = {
        "schema": SCHEMA,
        "queue_id": queue_id,
        "state": "paused",
        "created_at": stamp,
        "updated_at": stamp,
        "settings": dict(DEFAULT_SETTINGS),
        "build": {
            "jurisdictions": jurisdictions,
            "corpus_release": pins["corpus_release"],
            "corpus_ref": pins["corpus_ref"],
            "rulespec_ref": _git_head(rulespec_root),
            "max_unit_chars": max_unit_chars,
            "scopes": scopes,
            "excluded": excluded,
            "excluded_citations": excluded_citations,
        },
        "items": ordered,
    }
    validate_state(state)
    return state


# -- validation -------------------------------------------------------------


def validate_state(state: dict[str, Any]) -> None:
    if state.get("schema") != SCHEMA:
        raise ValueError(f"queue state schema must be {SCHEMA}")
    if not QUEUE_ID_PATTERN.fullmatch(str(state.get("queue_id", ""))):
        raise ValueError("queue state has an invalid queue_id")
    if state.get("state") not in QUEUE_STATES:
        raise ValueError(f"queue state must be one of {QUEUE_STATES}")
    settings = state.get("settings")
    if not isinstance(settings, dict):
        raise ValueError("queue state has no settings")
    for name in ("max_in_flight", "max_attempts"):
        value = settings.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"settings.{name} must be a positive integer")
    if settings.get("pr_base_branch") != "main":
        raise ValueError("settings.pr_base_branch must be main")
    items = state.get("items")
    if not isinstance(items, list):
        raise ValueError("queue state has no items list")
    seen: set[str] = set()
    for item in items:
        citation = item.get("citation")
        if not isinstance(citation, str) or not citation:
            raise ValueError("queue item lacks a citation")
        if citation in seen:
            raise ValueError(f"duplicate queue item: {citation}")
        seen.add(citation)
        if item.get("status") not in STATUSES:
            raise ValueError(f"{citation} has unknown status {item.get('status')!r}")
        if not isinstance(item.get("attempts"), list):
            raise ValueError(f"{citation} has no attempts list")
        requeued_after = item.get("requeued_after", 0)
        if (
            not isinstance(requeued_after, int)
            or isinstance(requeued_after, bool)
            or not 0 <= requeued_after <= len(item["attempts"])
        ):
            raise ValueError(f"{citation} has an invalid requeued_after")
    for citation in seen:
        parts = citation.split("/")
        for end in range(1, len(parts)):
            ancestor = "/".join(parts[:end])
            if ancestor in seen:
                raise ValueError(
                    f"queue item {citation} is inside queue item {ancestor}"
                )


def load_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    validate_state(state)
    return state


def write_state(path: Path, state: dict[str, Any]) -> None:
    validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


# -- GitHub -----------------------------------------------------------------


class GitHubError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


class GitHub(Protocol):
    def get(self, path: str, params: dict[str, str] | None = None) -> Any: ...

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any: ...


class ApiGitHub:
    """Minimal GitHub REST client over urllib, authenticated by a token."""

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        self._token = token
        self._api_url = api_url.rstrip("/")

    def _request(self, method: str, path: str, body: bytes | None) -> Any:
        request = urllib.request.Request(
            f"{self._api_url}/{path.lstrip('/')}",
            data=body,
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "X-GitHub-Api-Version": API_VERSION,
                **({"Content-Type": "application/json"} if body is not None else {}),
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

    def post(self, path: str, body: dict[str, Any] | None = None) -> Any:
        payload = json.dumps(body).encode("utf-8") if body is not None else b""
        return self._request("POST", path, payload)


@dataclass(frozen=True)
class Target:
    """Where encodes land and which pins the next dispatch uses."""

    encoder_repo: str
    rulespec_repo: str
    rulespec_ref: str
    corpus_release: str
    corpus_ref: str
    rules_engine_ref: str


def _contents(github: GitHub, repo: str, path: str, ref: str) -> str:
    payload = github.get(f"repos/{repo}/contents/{path}", {"ref": ref})
    return base64.b64decode(payload["content"]).decode("utf-8")


def resolve_target(github: GitHub, encoder_repo: str, country: str) -> Target:
    rulespec_repo = f"TheAxiomFoundation/rulespec-{country}"
    tip = github.get(f"repos/{rulespec_repo}/git/ref/heads/main")["object"]["sha"]
    if not SHA_PATTERN.fullmatch(tip):
        raise ValueError(f"rulespec main tip is not a commit SHA: {tip}")
    pins = _toolchain_pins(
        tomllib.loads(_contents(github, rulespec_repo, ".axiom/toolchain.toml", tip))[
            "toolchain"
        ],
        tomllib.loads(
            _contents(github, rulespec_repo, ".axiom/workflow-toolchain.toml", tip)
        )["workflow_toolchain"],
    )
    return Target(
        encoder_repo=encoder_repo,
        rulespec_repo=rulespec_repo,
        rulespec_ref=tip,
        **pins,
    )


def _run_url(repo: str, run_id: int) -> str:
    return f"https://github.com/{repo}/actions/runs/{run_id}"


# Run listings are paged 100 at a time; ten pages cover about a week of
# ad hoc encodes, far more than one tick needs.
MAX_RUN_PAGES = 10


def dispatcher_runs(
    github: GitHub, repo: str, *, since: datetime
) -> list[dict[str, Any]]:
    """Encode runs this dispatcher (the Actions bot) created since ``since``."""

    runs: list[dict[str, Any]] = []
    for page in range(1, MAX_RUN_PAGES + 1):
        payload = github.get(
            f"repos/{repo}/actions/workflows/{ENCODE_WORKFLOW}/runs",
            {
                "event": "workflow_dispatch",
                "created": f">={_iso(since)}",
                "per_page": "100",
                "page": str(page),
            },
        )
        batch = payload.get("workflow_runs", [])
        runs.extend(
            run
            for run in batch
            if (run.get("triggering_actor") or {}).get("login") == DISPATCH_ACTOR
            and str(run.get("display_title") or "").startswith(RUN_TITLE_PREFIX)
        )
        if len(batch) < 100:
            break
    return runs


def match_run(
    runs: list[dict[str, Any]], citation: str, *, claimed: set[int]
) -> dict[str, Any] | None:
    """The earliest unclaimed run whose title names ``citation``."""

    expected = f"{RUN_TITLE_PREFIX}{citation}"
    matches = [
        run
        for run in runs
        if run.get("id") not in claimed and run.get("display_title") == expected
    ]
    return min(matches, key=lambda run: (run["created_at"], run["id"]), default=None)


def find_run(
    github: GitHub,
    repo: str,
    citation: str,
    *,
    since: datetime,
    claimed: set[int],
) -> dict[str, Any] | None:
    """The earliest unclaimed dispatcher run for ``citation`` since ``since``."""

    return match_run(
        dispatcher_runs(github, repo, since=since), citation, claimed=claimed
    )


@dataclass(frozen=True)
class Failure:
    budget_exhausted: bool
    step: str | None
    where: str


def describe_failure(github: GitHub, repo: str, run_id: int) -> Failure:
    """Which job and step a failed encode run stopped at."""

    jobs = github.get(
        f"repos/{repo}/actions/runs/{run_id}/jobs", {"per_page": "100"}
    ).get("jobs", [])
    budget_exhausted = any(
        job.get("name") == BUDGET_JOB and job.get("conclusion") == "failure"
        for job in jobs
    )
    for job in jobs:
        if job.get("conclusion") not in COUNTED_FAILURES:
            continue
        steps = [
            step.get("name", "?")
            for step in job.get("steps") or []
            if step.get("conclusion") == "failure"
        ]
        name = job.get("name", "?")
        if steps:
            return Failure(budget_exhausted, steps[0], f"{name}: {steps[0]}")
        return Failure(budget_exhausted, None, name)
    return Failure(budget_exhausted, None, "no failed step reported")


def rejected_by(github: GitHub, repo: str, run_id: int) -> str | None:
    """Who rejected the run's signing approval, if anyone did."""

    for approval in github.get(f"repos/{repo}/actions/runs/{run_id}/approvals"):
        if approval.get("state") == "rejected":
            return (approval.get("user") or {}).get("login") or "a reviewer"
    return None


def find_pull_request(
    github: GitHub, rulespec_repo: str, country: str, run_id: int, run_attempt: int
) -> dict[str, Any] | None:
    owner = rulespec_repo.split("/", 1)[0]
    branch = f"axiom/signed-backfill-{country}-{run_id}-{run_attempt}"
    pulls = github.get(
        f"repos/{rulespec_repo}/pulls",
        {"head": f"{owner}:{branch}", "state": "all", "per_page": "10"},
    )
    return pulls[0] if pulls else None


# -- ticking ----------------------------------------------------------------


def _latest(item: dict[str, Any]) -> dict[str, Any]:
    return item["attempts"][-1]


def _current_attempts(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Attempts since the item was last requeued by a person."""

    return item["attempts"][item.get("requeued_after", 0) :]


def _counted(item: dict[str, Any]) -> int:
    return sum(1 for attempt in _current_attempts(item) if attempt.get("counted"))


def _cancellations(item: dict[str, Any]) -> int:
    return sum(
        1 for attempt in _current_attempts(item) if attempt.get("result") == "cancelled"
    )


def _stale(item: dict[str, Any]) -> int:
    """Runs in a row lost to a moving rulespec main."""

    count = 0
    for attempt in reversed(_current_attempts(item)):
        if attempt.get("result") not in {"stale-cancelled", "stale-base"}:
            break
        count += 1
    return count


def _requeue_after_stale(item: dict[str, Any], events: list[str]) -> None:
    if _stale(item) >= MAX_STALE:
        item["status"] = "blocked"
        item["note"] = (
            f"rulespec main moved during {MAX_STALE} runs in a row; requeue it "
            "when main is quieter"
        )
        events.append(f"blocked {item['citation']}: {item['note']}")
    else:
        item["status"] = "pending"


def _requeue_after_cancel(item: dict[str, Any], events: list[str]) -> None:
    if _cancellations(item) >= MAX_CANCELLATIONS:
        item["status"] = "blocked"
        item["note"] = f"cancelled {MAX_CANCELLATIONS} times without finishing"
        events.append(f"blocked {item['citation']}: {item['note']}")
    else:
        item["status"] = "pending"


def _apply_pull_request(
    item: dict[str, Any], pull: dict[str, Any], events: list[str]
) -> None:
    item["pr"] = {"number": pull["number"], "url": pull["html_url"]}
    if pull.get("merged_at"):
        item["status"] = "done"
        item["note"] = "RuleSpec pull request merged"
        events.append(f"done {item['citation']}: {pull['html_url']}")
    elif pull.get("state") == "closed":
        item["status"] = "blocked"
        item["note"] = "RuleSpec pull request closed without merging"
        events.append(f"blocked {item['citation']}: {item['note']}")
    else:
        item["status"] = "in_review"


def reconcile_dispatched(
    item: dict[str, Any],
    *,
    github: GitHub,
    target: Target,
    settings: dict[str, Any],
    claimed: set[int],
    now: datetime,
    events: list[str],
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Advance one dispatched item. Returns its run when it is still waiting."""

    attempt = _latest(item)
    repo = target.encoder_repo
    if attempt.get("run_id") is None:
        dispatched_at = _parse_iso(attempt["dispatched_at"])
        run = find_run(
            github,
            repo,
            item["citation"],
            since=dispatched_at - timedelta(minutes=1),
            claimed=claimed,
        )
        if run is None:
            if now - dispatched_at > LOST_RUN_AFTER:
                attempt["result"] = "lost"
                attempt["note"] = "no run appeared for this dispatch"
                item["status"] = "pending"
                events.append(f"requeued {item['citation']}: dispatch produced no run")
            return None
        attempt["run_id"] = run["id"]
        attempt["run_url"] = _run_url(repo, run["id"])
        claimed.add(run["id"])
    try:
        run = github.get(f"repos/{repo}/actions/runs/{attempt['run_id']}")
    except GitHubError as exc:
        if exc.status != 404:
            raise
        attempt["result"] = "lost"
        attempt["note"] = "the run no longer exists"
        item["status"] = "pending"
        events.append(f"requeued {item['citation']}: its run was deleted")
        return None
    stale = (
        attempt.get("rulespec_ref") is not None
        and attempt["rulespec_ref"] != target.rulespec_ref
    )
    if run.get("status") != "completed":
        if not stale:
            return run
        # The run will fail when it checks rulespec main again, after its
        # model spend if it has already started, so stop it now.
        if dry_run:
            events.append(f"would cancel {attempt['run_url']}: rulespec main moved")
            return run
        try:
            github.post(f"repos/{repo}/actions/runs/{attempt['run_id']}/cancel")
        except GitHubError as exc:
            events.append(f"could not cancel {attempt['run_url']}: {exc}")
            return run
        attempt["result"] = "stale-cancelled"
        attempt["note"] = "rulespec main moved before the run finished"
        events.append(f"cancelled {attempt['run_url']}: rulespec main moved")
        _requeue_after_stale(item, events)
        return None
    attempt["run_attempt"] = run.get("run_attempt", 1)
    conclusion = run.get("conclusion")
    if conclusion == "success":
        attempt["result"] = "success"
        pull = find_pull_request(
            github,
            target.rulespec_repo,
            settings["country"],
            attempt["run_id"],
            attempt["run_attempt"],
        )
        if pull is None:
            item["status"] = "blocked"
            item["note"] = "run succeeded but no RuleSpec pull request was found"
            events.append(f"blocked {item['citation']}: {item['note']}")
        else:
            _apply_pull_request(item, pull, events)
        return None
    if conclusion in COUNTED_FAILURES:
        failure = describe_failure(github, repo, attempt["run_id"])
        attempt["result"] = conclusion
        attempt["note"] = failure.where
        reviewer = rejected_by(github, repo, attempt["run_id"])
        if reviewer is not None:
            item["status"] = "blocked"
            item["note"] = f"signing approval rejected by {reviewer}"
        elif failure.budget_exhausted:
            item["status"] = "blocked"
            item["note"] = "the encode workflow's failed-attempt budget is used up"
        elif stale and failure.step in STALE_BASE_STEPS:
            attempt["result"] = "stale-base"
            events.append(f"retrying {item['citation']}: rulespec main moved mid-run")
            _requeue_after_stale(item, events)
            return None
        else:
            attempt["counted"] = True
            if _counted(item) >= settings["max_attempts"]:
                item["status"] = "blocked"
                item["note"] = f"failed {_counted(item)} times; last at {failure.where}"
            else:
                item["status"] = "pending"
                events.append(f"retrying {item['citation']}: failed at {failure.where}")
                return None
        events.append(f"blocked {item['citation']}: {item['note']}")
        return None
    attempt["result"] = "cancelled"
    attempt["note"] = f"run concluded {conclusion}"
    _requeue_after_cancel(item, events)
    return None


def reconcile_in_review(
    item: dict[str, Any], *, github: GitHub, target: Target, events: list[str]
) -> None:
    pull = github.get(f"repos/{target.rulespec_repo}/pulls/{item['pr']['number']}")
    _apply_pull_request(item, pull, events)


def tick(
    state: dict[str, Any],
    *,
    github: GitHub,
    encoder_repo: str,
    now: datetime | None = None,
    dry_run: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    claimed_elsewhere: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Reconcile finished runs, then dispatch the next pending items.

    ``claimed_elsewhere`` holds run ids other queues have recorded, so no
    queue ever adopts or matches another queue's run.
    """

    now = now or _now()
    before = json.dumps(state, sort_keys=True)
    settings = state["settings"]
    target = resolve_target(github, encoder_repo, settings["country"])
    events: list[str] = []
    waiting: list[dict[str, Any]] = []
    claimed = set(claimed_elsewhere) | recorded_run_ids(state)
    error = None

    for item in state["items"]:
        try:
            if item["status"] == "dispatched":
                run = reconcile_dispatched(
                    item,
                    github=github,
                    target=target,
                    settings=settings,
                    claimed=claimed,
                    now=now,
                    events=events,
                    dry_run=dry_run,
                )
                if run is not None and run.get("status") == "waiting":
                    waiting.append(
                        {"citation": item["citation"], "url": run["html_url"]}
                    )
            elif item["status"] == "in_review":
                reconcile_in_review(item, github=github, target=target, events=events)
        except GitHubError as exc:
            events.append(f"could not check {item['citation']}: {exc}")
        except Exception as exc:  # noqa: BLE001 - one bad item must not stop the rest
            events.append(f"could not check {item['citation']}: {exc!r}")
            error = f"unexpected error checking {item['citation']}: {exc!r}"

    dispatched: list[dict[str, Any]] = []
    hold = None
    if state["state"] != "active":
        hold = "queue is paused"
    elif target.corpus_release != state["build"]["corpus_release"]:
        hold = (
            f"rulespec-us main now pins corpus release {target.corpus_release}, "
            f"but this queue was built from {state['build']['corpus_release']}; "
            "it needs rebuilding from the new release before dispatching more"
        )
    else:
        # A tick whose state push failed leaves runs nobody recorded; adopt
        # them instead of dispatching their items a second time.
        recorded = [
            _parse_iso(attempt["dispatched_at"])
            for item in state["items"]
            for attempt in item["attempts"]
        ]
        since = max(recorded, default=_parse_iso(state["created_at"]))
        unrecorded = dispatcher_runs(
            github, encoder_repo, since=since - timedelta(minutes=5)
        )
        in_flight = sum(1 for item in state["items"] if item["status"] == "dispatched")
        for item in state["items"]:
            if in_flight >= settings["max_in_flight"]:
                break
            if item["status"] != "pending":
                continue
            orphan = match_run(unrecorded, item["citation"], claimed=claimed)
            if orphan is not None:
                claimed.add(orphan["id"])
                item["status"] = "dispatched"
                item["attempts"].append(
                    {
                        "dispatched_at": orphan["created_at"],
                        "rulespec_ref": None,
                        "run_id": orphan["id"],
                        "run_url": _run_url(encoder_repo, orphan["id"]),
                        "note": "adopted a run this dispatcher started but "
                        "never recorded",
                    }
                )
                events.append(f"adopted unrecorded run {orphan['html_url']}")
                in_flight += 1
                continue
            in_flight += 1
            if not dry_run:
                try:
                    response = github.post(
                        f"repos/{encoder_repo}/actions/workflows/"
                        f"{ENCODE_WORKFLOW}/dispatches",
                        {
                            "ref": "main",
                            "inputs": {
                                "citation": item["citation"],
                                "country": settings["country"],
                                "rulespec_ref": target.rulespec_ref,
                                "pr_base_branch": settings["pr_base_branch"],
                                "corpus_ref": target.corpus_ref,
                                "rules_engine_ref": target.rules_engine_ref,
                                "open_pr": "true",
                            },
                        },
                    )
                except GitHubError as exc:
                    error = error or f"dispatching {item['citation']} failed: {exc}"
                    break
                item["status"] = "dispatched"
                item.pop("note", None)
                attempt = {
                    "dispatched_at": _iso(now),
                    "rulespec_ref": target.rulespec_ref,
                    "corpus_ref": target.corpus_ref,
                    "rules_engine_ref": target.rules_engine_ref,
                    "run_id": None,
                }
                run_id = (
                    response.get("workflow_run_id")
                    if isinstance(response, dict)
                    else None
                )
                if isinstance(run_id, int) and run_id not in claimed:
                    attempt["run_id"] = run_id
                    attempt["run_url"] = _run_url(encoder_repo, run_id)
                    claimed.add(run_id)
                item["attempts"].append(attempt)
            dispatched.append(item)

    if dispatched and not dry_run:
        # Runs appear a few seconds after dispatch; attach what shows up now
        # and let the next tick find the rest.
        for delay in (5, 10, 20):
            unresolved = [i for i in dispatched if _latest(i).get("run_id") is None]
            if not unresolved:
                break
            sleep(delay)
            for item in unresolved:
                try:
                    run = find_run(
                        github,
                        encoder_repo,
                        item["citation"],
                        since=now - timedelta(minutes=1),
                        claimed=claimed,
                    )
                except GitHubError:
                    run = None
                if run is not None:
                    _latest(item)["run_id"] = run["id"]
                    _latest(item)["run_url"] = _run_url(encoder_repo, run["id"])
                    claimed.add(run["id"])

    if not dry_run and json.dumps(state, sort_keys=True) != before:
        state["updated_at"] = _iso(now)
    return {
        "queue_id": state["queue_id"],
        "state": state["state"],
        "rulespec_ref": target.rulespec_ref,
        "hold": hold,
        "error": error,
        "dry_run": dry_run,
        "dispatched": [item["citation"] for item in dispatched],
        "waiting_for_approval": waiting,
        "events": events,
        "counts": status_counts(state),
    }


def requeue(state: dict[str, Any], citation: str, *, now: datetime) -> None:
    """Send a blocked item again with a fresh retry and cancellation budget."""

    item = next((i for i in state["items"] if i["citation"] == citation), None)
    if item is None:
        raise ValueError(f"{citation} is not in queue {state['queue_id']}")
    if item["status"] != "blocked":
        raise ValueError(f"{citation} is {item['status']}, not blocked")
    item["status"] = "pending"
    item["requeued_after"] = len(item["attempts"])
    item["note"] = f"requeued {_iso(now)} after: {item.get('note', 'blocked')}"
    state["updated_at"] = _iso(now)


def recorded_run_ids(state: dict[str, Any]) -> set[int]:
    return {
        attempt["run_id"]
        for item in state["items"]
        for attempt in item["attempts"]
        if attempt.get("run_id") is not None
    }


def overlapping_items(
    new: dict[str, Any], existing: Iterable[dict[str, Any]]
) -> list[str]:
    """Citations in ``new`` equal to, inside, or containing an existing item."""

    taken = {item["citation"] for state in existing for item in state["items"]}
    clashes = []
    for item in new["items"]:
        parts = item["citation"].split("/")
        prefixes = {"/".join(parts[:end]) for end in range(1, len(parts) + 1)}
        if prefixes & taken or any(
            other.startswith(f"{item['citation']}/") for other in taken
        ):
            clashes.append(item["citation"])
    return clashes


# -- reporting --------------------------------------------------------------


def status_counts(state: dict[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(STATUSES, 0)
    for item in state["items"]:
        counts[item["status"]] += 1
    return counts


def render_summary(results: Iterable[dict[str, Any]]) -> str:
    lines = ["## SNAP dispatch queue", ""]
    for result in results:
        counts = result["counts"]
        lines.append(f"### {result['queue_id']} ({result['state']})")
        lines.append("")
        lines.append("| " + " | ".join(STATUSES) + " |")
        lines.append("|" + " --- |" * len(STATUSES))
        lines.append("| " + " | ".join(str(counts[s]) for s in STATUSES) + " |")
        lines.append("")
        verb = "Would dispatch" if result["dry_run"] else "Dispatched"
        if result["dispatched"]:
            lines.append(f"{verb} against rulespec-us `{result['rulespec_ref']}`:")
            lines.extend(f"- `{citation}`" for citation in result["dispatched"])
            lines.append("")
        if result["hold"]:
            lines.append(f"Not dispatching: {result['hold']}.")
            lines.append("")
        if result["error"]:
            lines.append(f"**Error:** {result['error']}")
            lines.append("")
        if result["waiting_for_approval"]:
            lines.append("Waiting for `production-signing` approval:")
            lines.extend(
                f"- [{run['citation']}]({run['url']})"
                for run in result["waiting_for_approval"]
            )
            lines.append("")
        if result["events"]:
            lines.append("Changes this tick:")
            lines.extend(f"- {event}" for event in result["events"])
            lines.append("")
    return "\n".join(lines) + "\n"


# -- command line -----------------------------------------------------------


def _queue_path(state_dir: Path, queue_id: str) -> Path:
    if not QUEUE_ID_PATTERN.fullmatch(queue_id):
        raise ValueError(f"invalid queue id: {queue_id}")
    return state_dir / "queues" / f"{queue_id}.json"


def _queue_paths(state_dir: Path, queue_id: str | None) -> list[Path]:
    if queue_id:
        path = _queue_path(state_dir, queue_id)
        if not path.exists():
            raise ValueError(f"no queue named {queue_id} in {state_dir}")
        return [path]
    return sorted((state_dir / "queues").glob("*.json"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="build a new paused queue")
    build.add_argument("--corpus", type=Path, required=True)
    build.add_argument("--rulespec", type=Path, required=True)
    build.add_argument("--state-dir", type=Path, required=True)
    build.add_argument("--queue-id", required=True)
    build.add_argument("--jurisdictions", required=True)
    build.add_argument("--max-unit-chars", type=int, default=DEFAULT_MAX_UNIT_CHARS)

    tick_parser = commands.add_parser("tick", help="reconcile and dispatch")
    tick_parser.add_argument("--state-dir", type=Path, required=True)
    tick_parser.add_argument("--queue-id")
    tick_parser.add_argument("--dry-run", action="store_true")
    tick_parser.add_argument("--summary", type=Path)

    set_state = commands.add_parser("set-state", help="activate or pause a queue")
    set_state.add_argument("--state-dir", type=Path, required=True)
    set_state.add_argument("--queue-id", required=True)
    set_state.add_argument("state", choices=QUEUE_STATES)

    requeue_parser = commands.add_parser("requeue", help="retry a blocked item")
    requeue_parser.add_argument("--state-dir", type=Path, required=True)
    requeue_parser.add_argument("--queue-id", required=True)
    requeue_parser.add_argument("--citation", required=True)

    validate = commands.add_parser("validate", help="validate queue state files")
    validate.add_argument("paths", type=Path, nargs="+")

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            path = _queue_path(args.state_dir, args.queue_id)
            if path.exists():
                raise ValueError(f"{path} already exists; builds never overwrite")
            state = build_queue(
                args.corpus,
                args.rulespec,
                queue_id=args.queue_id,
                jurisdictions=[
                    j.strip() for j in args.jurisdictions.split(",") if j.strip()
                ],
                max_unit_chars=args.max_unit_chars,
            )
            existing = [
                load_state(other) for other in _queue_paths(args.state_dir, None)
            ]
            clashes = overlapping_items(state, existing)
            if clashes:
                raise ValueError(
                    f"{len(clashes)} items overlap an existing queue, e.g. {clashes[0]}"
                )
            write_state(path, state)
            print(json.dumps({"queue_id": args.queue_id, **status_counts(state)}))
        elif args.command == "tick":
            token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
            repo = os.environ.get("GITHUB_REPOSITORY")
            if not token or not repo:
                raise ValueError("tick needs GH_TOKEN and GITHUB_REPOSITORY")
            github = ApiGitHub(token)
            everything = {
                path: load_state(path) for path in _queue_paths(args.state_dir, None)
            }
            results = []
            for path in _queue_paths(args.state_dir, args.queue_id):
                state = everything[path]
                elsewhere = frozenset(
                    run_id
                    for other_path, other in everything.items()
                    if other_path != path
                    for run_id in recorded_run_ids(other)
                )
                try:
                    results.append(
                        tick(
                            state,
                            github=github,
                            encoder_repo=repo,
                            dry_run=args.dry_run,
                            claimed_elsewhere=elsewhere,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - keep ticking other queues
                    results.append(
                        {
                            "queue_id": state["queue_id"],
                            "state": state["state"],
                            "rulespec_ref": None,
                            "hold": None,
                            "error": f"tick failed: {exc!r}",
                            "dry_run": args.dry_run,
                            "dispatched": [],
                            "waiting_for_approval": [],
                            "events": [],
                            "counts": status_counts(state),
                        }
                    )
                finally:
                    if not args.dry_run:
                        write_state(path, state)
            summary = render_summary(results)
            if args.summary:
                with args.summary.open("a", encoding="utf-8") as stream:
                    stream.write(summary)
            print(summary)
            if any(result["error"] for result in results):
                return 1
        elif args.command == "set-state":
            path = _queue_path(args.state_dir, args.queue_id)
            state = load_state(path)
            state["state"] = args.state
            state["updated_at"] = _iso(_now())
            write_state(path, state)
        elif args.command == "requeue":
            path = _queue_path(args.state_dir, args.queue_id)
            state = load_state(path)
            requeue(state, args.citation, now=_now())
            write_state(path, state)
        elif args.command == "validate":
            for path in args.paths:
                load_state(path)
    except (ValueError, GitHubError, OSError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

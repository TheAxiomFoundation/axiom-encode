"""Check the encoding linkage lines every axiom-encode pull request carries.

An encoder fix used to name neither the encoding it fixes nor the failing run
that showed the defect, so later analyses had to guess the link from titles
and changed files. Every pull request body now carries two lines::

    Encoding-Target: us/regulation/7/273/7, us/statute/26/21
    Fixes-Run: 42409703021, local:1a2b3c4d

or, for work that fixes no particular encoding or run, an explicit ``none``::

    Encoding-Target: none
    Fixes-Run: none

``Encoding-Target`` lists corpus citation paths (``<jurisdiction>/<kind>/...``,
as the targeted signed re-encode takes them). ``Fixes-Run`` lists run ids:

- a ``targeted-signed-reencode.yml`` run of this repository, as its numeric id
  or its ``https://github.com/<repo>/actions/runs/<id>`` URL (normalized to
  the id);
- ``local:<id>`` for a local encode, where ``<id>`` is the ``encoding_runs``
  id that ``encoding_db`` logs (8 lowercase hex characters, or a full UUID).

Each line appears exactly once, values are comma-separated, a value appears
once, and ``none`` stands alone. A line may sit in a list item or have its key
in bold. Lines inside HTML comments, fenced code blocks and indented (four
space) code blocks do not count, so the pull request template's guidance and
quoted examples are ignored.

With ``--verify-runs`` each signed run id is looked up through the GitHub REST
API (read-only). The run must exist in the repository, come from
``.github/workflows/targeted-signed-reencode.yml`` as a ``workflow_dispatch``
on ``main`` (so its run name is the reviewed one), and its run name must end
with a citation that ``Encoding-Target`` lists. Local run ids live only in a
local database and are checked for format only.

The lookup fails closed. A run id the API does not find, a run of another
workflow, and an API outage (after retries) all fail the check. The job is
cheap to rerun, and a pass that skipped the lookup would make the linkage
counts inexact, which is what this check exists to prevent.

Pull requests opened by the encoder version-bump App from this repository are
exempt, as in ``version_provenance.hand_picked_version``.

This module uses only the standard library, so CI runs it with ``python3 -I``
without installing the project. CI runs the base branch's copy under
``pull_request_target``, so a pull request cannot loosen the check that judges
it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

TARGET_KEY = "Encoding-Target"
RUN_KEY = "Fixes-Run"
NONE = "none"

# The bump workflow's GitHub App, as a pull request's author login. The same
# value as version_provenance.BUMP_APP_LOGIN; a test keeps them equal.
BUMP_APP_LOGIN = "axiom-encoder-version-bump[bot]"
SIGNED_WORKFLOW_PATH = ".github/workflows/targeted-signed-reencode.yml"
MAX_RUNS = 20
RATE_LIMIT_WAIT = 60.0
# All lookups together, inside the workflow's ten-minute job timeout, so a
# slow API ends in a fail-closed error rather than a cancelled job.
VERIFY_DEADLINE = 180.0

_LIST_MARKER = r"(?:(?:[-*+]|[0-9]{1,9}[.)])[ \t]+)?"
# A fence opens after up to three spaces, or after a (possibly nested) list
# marker; `prefix` is then the list item's content column.
_FENCE = re.compile(
    r"^(?P<prefix> {0,3}| *(?:[-*+]|[0-9]{1,9}[.)])[ \t]+)(`{3,}|~{3,})"
)
_COMMENT_BLOCK = re.compile(rf" {{0,3}}{_LIST_MARKER}<!--")
_CODE_SPAN = re.compile(r"(`+)(?!`).*?(?<!`)\1(?!`)")
_CLOSING_FENCE = re.compile(r"( *)(`{3,}|~{3,})[ \t]*")
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\((\S+)\)")
_BOLD = r"(?:\*\*|__)?"
# Up to three spaces of indent (four is a code block), an optional list marker
# (bullet or number) and bold around the key: `- **Encoding-Target:** none`
# counts. _clean strips bold, backticks, a link and a hard break from values.
_LINE = re.compile(
    rf"^ {{0,3}}(?:(?:[-*+]|[0-9]{{1,9}}[.)])[ \t]+)?{_BOLD}"
    rf"({re.escape(TARGET_KEY)}|{re.escape(RUN_KEY)}){_BOLD}[ \t]*:{_BOLD}(.*)$",
    re.IGNORECASE,
)
_JURISDICTION = r"[a-z]{2}(?:-[a-z0-9]{2,})?"
_SEGMENT = r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9_-])?"
CITATION = re.compile(rf"{_JURISDICTION}(?:/{_SEGMENT}){{2,}}")
SIGNED_RUN = re.compile(r"[1-9][0-9]{0,19}")
LOCAL_RUN = re.compile(
    r"local:(?:[0-9a-f]{8}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)
_RUN_URL = re.compile(
    r"https://github\.com/(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)"
    r"/actions/runs/(?P<id>[0-9]+)(?:/attempts/[0-9]+)?(?:/job/[0-9]+)?/?"
    r"(?:[?#]\S*)?"
)
# run-name: "Targeted signed RuleSpec re-encode [<queue>:<item>:<sha>] <citation>"
_RUN_TITLE_CITATION = re.compile(r"\]\s+(\S+)\s*$")


@dataclass
class Linkage:
    """The parsed lines. An empty list means ``none``."""

    targets: list[str] = field(default_factory=list)
    signed_runs: list[str] = field(default_factory=list)
    local_runs: list[str] = field(default_factory=list)


@dataclass
class Result:
    linkage: Linkage | None
    errors: list[str]


def _mask_code_spans(line: str) -> str:
    """The line with inline code spans blanked, same length, for searching."""
    return _CODE_SPAN.sub(lambda match: " " * len(match.group(0)), line)


def _visible_lines(body: str, errors: list[str]) -> list[str]:
    """The body's lines outside HTML comments and fenced code blocks.

    One pass, so a `<!--` inside a fence does not hide the rest of the body
    and a fence inside a comment does not open a code block. A line that
    starts with `<!--` (after up to three spaces or a list marker) hides
    everything up to the first `-->`, or to the end of the body, as GitHub
    does. A `<!--` later in a line must close on the same line: GitHub may
    hide following lines too, depending on paragraph rules this parser does
    not model, so an unclosed one is an error (fail closed). `<!--` and `-->`
    inside inline code spans are literal text. A fence (also after a list
    marker) closes only on a bare fence line, as in CommonMark.
    """
    lines: list[str] = []
    fence: tuple[str, int] | None = None  # (fence characters, content column)
    in_comment = False
    raw = body.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for number, line in enumerate(raw, start=1):
        if fence is not None:
            fence_chars, column = fence
            indent = len(line) - len(line.lstrip(" "))
            if line.strip() and indent < column:
                # Less indented than its list item: the item, and with it
                # the fence, has ended; this line is ordinary text.
                fence = None
            else:
                match = _CLOSING_FENCE.fullmatch(line)
                if (
                    match
                    and len(match.group(1)) <= column + 3
                    and match.group(2)[0] == fence_chars[0]
                    and len(match.group(2)) >= len(fence_chars)
                ):
                    fence = None
                continue
        visible = ""
        rest = line
        block = None if in_comment else _COMMENT_BLOCK.match(line)
        if block is not None:
            in_comment = True
            rest = line[block.end() :]
        while rest:
            if in_comment:
                end = rest.find("-->")
                if end < 0:
                    rest = ""
                else:
                    in_comment = False
                    rest = rest[end + 3 :]
                continue
            masked = _mask_code_spans(rest)
            start = masked.find("<!--")
            if start < 0:
                visible += rest
                break
            end = masked.find("-->", start + 4)
            if end < 0:
                errors.append(
                    f"line {number}: an HTML comment opens mid-line and does not "
                    "close on that line; close it with `-->` on the same line, "
                    "or start the line with `<!--`"
                )
                visible += rest[:start]
                break
            visible += rest[:start]
            rest = rest[end + 3 :]
        if visible.strip() == "" and line.strip() != "":
            continue
        match = _FENCE.match(visible)
        if match and visible == line:
            fence = (match.group(2), len(match.group("prefix")))
            continue
        lines.append(visible)
    return lines


def _clean(part: str) -> str:
    """One value without the markdown an author may wrap it in."""
    part = part.strip()
    part = re.sub(r"\\$", "", part).strip()  # a backslash hard break
    link = _MARKDOWN_LINK.fullmatch(part)
    if link is not None:
        part = link.group(1)
    for _ in range(2):
        part = part.strip().strip("`").strip()
        part = re.sub(r"^(?:\*\*|__)|(?:\*\*|__)$", "", part).strip()
        if part.startswith("<") and part.endswith(">") and "://" in part:
            part = part[1:-1]
    return part


def _values(raw: str, key: str, errors: list[str]) -> list[str] | None:
    """Split a line's value; None after an error, [] for ``none``."""
    parts = [_clean(part) for part in raw.split(",")]
    if parts == [""]:
        errors.append(f"{key}: no value; write `{key}: {NONE}` if there is none")
        return None
    if any(part == "" for part in parts):
        errors.append(f"{key}: empty entry in {raw.strip()!r}")
        return None
    lowered = [part.lower() for part in parts]
    if NONE in lowered:
        if len(parts) > 1:
            errors.append(f"{key}: `{NONE}` must stand alone, got {raw.strip()!r}")
            return None
        return []
    duplicates = sorted({part for part in parts if parts.count(part) > 1})
    if duplicates:
        errors.append(f"{key}: listed more than once: {', '.join(duplicates)}")
        return None
    return parts


def _target(value: str, errors: list[str]) -> str | None:
    if value.endswith((".yaml", ".yml")):
        errors.append(
            f"{TARGET_KEY}: {value!r} is a file path; give the corpus citation "
            "path without the extension, for example us/regulation/7/273/7"
        )
        return None
    if CITATION.fullmatch(value) is None:
        errors.append(
            f"{TARGET_KEY}: {value!r} is not a corpus citation path "
            "(<jurisdiction>/<kind>/<id>..., for example us/regulation/7/273/7)"
        )
        return None
    return value


def _run(
    value: str, repository: str | None, errors: list[str]
) -> tuple[str, str] | None:
    """('signed', id) or ('local', value), or None after an error."""
    if LOCAL_RUN.fullmatch(value):
        return ("local", value)
    url = _RUN_URL.fullmatch(value)
    if url is not None:
        if repository is not None and url.group("repo").lower() != repository.lower():
            errors.append(
                f"{RUN_KEY}: {value!r} is a run of {url.group('repo')}, "
                f"not of {repository}"
            )
            return None
        value = url.group("id")
    if SIGNED_RUN.fullmatch(value):
        return ("signed", value)
    errors.append(
        f"{RUN_KEY}: {value!r} is neither a targeted-signed-reencode run id "
        "(digits, or its actions/runs URL) nor a local run id "
        "(local:<8 hex characters or a UUID>)"
    )
    return None


def parse(body: str | None, repository: str | None = None) -> Result:
    """Parse and check the format of a pull request body's linkage lines."""
    errors: list[str] = []
    found: dict[str, list[str]] = {TARGET_KEY: [], RUN_KEY: []}
    for line in _visible_lines(body or "", errors):
        match = _LINE.match(line)
        if match is None:
            continue
        written = match.group(1)
        key = TARGET_KEY if written.lower() == TARGET_KEY.lower() else RUN_KEY
        if written != key:
            errors.append(f"write the key as `{key}:`, not `{written}:`")
        found[key].append(match.group(2))
    for key, raws in found.items():
        if not raws:
            errors.append(
                f"missing `{key}:` line; add it to the pull request body "
                f"(use `{key}: {NONE}` for work that fixes no particular "
                f"{'encoding' if key == TARGET_KEY else 'run'})"
            )
        elif len(raws) > 1:
            errors.append(f"`{key}:` appears {len(raws)} times; write it once")
    if errors:
        return Result(None, errors)

    linkage = Linkage()
    targets = _values(found[TARGET_KEY][0], TARGET_KEY, errors)
    runs = _values(found[RUN_KEY][0], RUN_KEY, errors)
    for value in targets or []:
        if (citation := _target(value, errors)) is not None:
            linkage.targets.append(citation)
    for value in runs or []:
        parsed = _run(value, repository, errors)
        if parsed is None:
            continue
        kind, run_id = parsed
        bucket = linkage.signed_runs if kind == "signed" else linkage.local_runs
        if run_id in bucket:
            errors.append(f"{RUN_KEY}: run {run_id} is listed more than once")
        else:
            bucket.append(run_id)
    if len(linkage.signed_runs) + len(linkage.local_runs) > MAX_RUNS:
        errors.append(f"{RUN_KEY}: at most {MAX_RUNS} runs per pull request")
    if (
        not errors
        and (linkage.signed_runs or linkage.local_runs)
        and not linkage.targets
    ):
        errors.append(
            f"{RUN_KEY} names a run but {TARGET_KEY} is `{NONE}`; "
            "list the citation the run encoded"
        )
    return Result(None if errors else linkage, errors)


def is_exempt(author: str, head_repository: str, repository: str) -> bool:
    """The version-bump App's own pull requests, from this repository."""
    return author == BUMP_APP_LOGIN and head_repository == repository


Fetch = Callable[[str], tuple[int, dict | None]]


def github_fetch(
    token: str,
    api_url: str = "https://api.github.com",
    sleep: Callable[[float], None] = time.sleep,
) -> Fetch:
    """A read-only GET returning (status, json); status 0 on a network error."""

    def fetch(path: str) -> tuple[int, dict | None]:
        request = urllib.request.Request(
            f"{api_url.rstrip('/')}/{path.lstrip('/')}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {token}",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "axiom-encode-pr-linkage",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            # A rate limit answers 403 or 429. Wait out a short retry-after
            # (at most RATE_LIMIT_WAIT seconds) and report 429 so the caller
            # retries; a longer wait fails closed after the retries.
            if error.code == 429 or (
                error.code == 403
                and (
                    error.headers.get("retry-after")
                    or error.headers.get("x-ratelimit-remaining") == "0"
                )
            ):
                try:
                    wait = float(error.headers.get("retry-after") or "")
                except ValueError:
                    wait = 0.0
                if 0 < wait <= RATE_LIMIT_WAIT:
                    sleep(wait)
                return 429, None
            return error.code, None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return 0, None

    return fetch


def _transient(status: int) -> bool:
    return status == 0 or status == 429 or status >= 500


def verify_signed_runs(
    linkage: Linkage,
    repository: str,
    fetch: Fetch,
    *,
    attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    deadline: float = VERIFY_DEADLINE,
) -> list[str]:
    """Errors for signed run ids that are not this repository's signed runs."""
    errors: list[str] = []
    started = clock()
    for run_id in linkage.signed_runs:
        if clock() - started > deadline:
            errors.append(
                f"{RUN_KEY}: run lookups passed the {deadline:.0f}s deadline before "
                f"run {run_id}; the check fails closed, so rerun this job"
            )
            break
        status, payload = 0, None
        for attempt in range(attempts):
            status, payload = fetch(f"repos/{repository}/actions/runs/{run_id}")
            if not _transient(status):
                break
            if attempt + 1 < attempts:
                sleep(2.0 * (attempt + 1))
        if status == 404:
            errors.append(f"{RUN_KEY}: run {run_id} does not exist in {repository}")
            continue
        if status != 200 or not isinstance(payload, dict):
            errors.append(
                f"{RUN_KEY}: could not look up run {run_id} through the GitHub "
                f"API (status {status or 'network error'}); the check fails "
                "closed, so rerun this job once the API answers"
            )
            continue
        path = str(payload.get("path") or "").split("@", 1)[0]
        if path != SIGNED_WORKFLOW_PATH:
            errors.append(
                f"{RUN_KEY}: run {run_id} is a run of {path or 'an unknown workflow'}, "
                f"not of {SIGNED_WORKFLOW_PATH}; use local:<id> for a local run"
            )
            continue
        if (
            payload.get("event") != "workflow_dispatch"
            or payload.get("head_branch") != "main"
        ):
            errors.append(
                f"{RUN_KEY}: run {run_id} is not a workflow_dispatch run on main "
                f"(event {payload.get('event')!r}, branch "
                f"{payload.get('head_branch')!r}), so its run name is not the "
                "reviewed one"
            )
            continue
        title = str(payload.get("display_title") or payload.get("name") or "")
        match = _RUN_TITLE_CITATION.search(title)
        if match is None:
            errors.append(
                f"{RUN_KEY}: cannot read the citation from run {run_id}'s "
                f"name {title!r}"
            )
            continue
        if match.group(1) not in linkage.targets:
            errors.append(
                f"{RUN_KEY}: run {run_id} encoded {match.group(1)}, which "
                f"{TARGET_KEY} does not list"
            )
    return errors


def check(
    body: str | None,
    *,
    author: str = "",
    head_repository: str = "",
    repository: str = "",
    fetch: Fetch | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Linkage | None, list[str], bool]:
    """(linkage, errors, exempt) for one pull request."""
    if repository and is_exempt(author, head_repository, repository):
        return None, [], True
    result = parse(body, repository or None)
    if result.linkage is None or fetch is None:
        return result.linkage, result.errors, False
    errors = verify_signed_runs(result.linkage, repository, fetch, sleep=sleep)
    return (None if errors else result.linkage), errors, False


def _from_event(path: Path) -> dict[str, str]:
    event = json.loads(path.read_text(encoding="utf-8"))
    pull = event.get("pull_request")
    if not isinstance(pull, dict):
        raise SystemExit("error: the event has no pull_request")
    return {
        "body": pull.get("body") or "",
        "author": (pull.get("user") or {}).get("login") or "",
        "head_repository": ((pull.get("head") or {}).get("repo") or {}).get("full_name")
        or "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Check a pull request body's {TARGET_KEY} and {RUN_KEY} lines."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--event", type=Path, help="a pull_request event payload ($GITHUB_EVENT_PATH)"
    )
    source.add_argument("--body-file", type=Path, help="a file holding the PR body")
    parser.add_argument("--author", default="")
    parser.add_argument("--head-repository", default="")
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY", "TheAxiomFoundation/axiom-encode"),
    )
    parser.add_argument(
        "--verify-runs",
        action="store_true",
        help="look up signed run ids through the GitHub API (needs GITHUB_TOKEN)",
    )
    args = parser.parse_args(argv)

    if args.event is not None:
        pull = _from_event(args.event)
    else:
        pull = {
            "body": args.body_file.read_text(encoding="utf-8"),
            "author": args.author,
            "head_repository": args.head_repository,
        }
    fetch = None
    if args.verify_runs:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            print("error: --verify-runs needs GITHUB_TOKEN", file=sys.stderr)
            return 2
        fetch = github_fetch(
            token, os.environ.get("GITHUB_API_URL", "https://api.github.com")
        )
    linkage, errors, exempt = check(
        pull["body"],
        author=pull["author"],
        head_repository=pull["head_repository"],
        repository=args.repository,
        fetch=fetch,
    )
    if exempt:
        print(f"exempt: opened by {BUMP_APP_LOGIN} from {args.repository}")
        return 0
    if errors:
        for error in errors:
            print(f"::error::{error}")
        print(
            f"Fix the `{TARGET_KEY}:` / `{RUN_KEY}:` lines in the pull request "
            "body (format: .github/pull_request_template.md); editing the body "
            "reruns this check.",
            file=sys.stderr,
        )
        return 1
    assert linkage is not None
    print(f"{TARGET_KEY}: {', '.join(linkage.targets) or NONE}")
    runs = linkage.signed_runs + linkage.local_runs
    print(f"{RUN_KEY}: {', '.join(runs) or NONE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

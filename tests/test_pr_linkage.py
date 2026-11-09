"""The Encoding-Target / Fixes-Run lines every pull request body carries."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode import pr_linkage, version_provenance
from axiom_encode.pr_linkage import Linkage, check, parse, verify_signed_runs

ROOT = Path(__file__).resolve().parents[1]
REPO = "TheAxiomFoundation/axiom-encode"
SIGNED = pr_linkage.SIGNED_WORKFLOW_PATH


def _body(target: str, run: str, extra: str = "") -> str:
    return (
        f"## Summary\n\nFix it.\n\nEncoding-Target: {target}\nFixes-Run: {run}\n{extra}"
    )


def _run_payload(citation: str, path: str = SIGNED) -> dict:
    return {
        "path": path,
        "event": "workflow_dispatch",
        "head_branch": "main",
        "display_title": (
            f"Targeted signed RuleSpec re-encode [adhoc:adhoc:adhoc] {citation}"
        ),
    }


def _fetch(
    responses: dict[str, tuple[int, dict | None]], calls: list[str] | None = None
):
    def fetch(path: str):
        if calls is not None:
            calls.append(path)
        run_id = path.rsplit("/", 1)[1]
        return responses.get(run_id, (404, None))

    return fetch


# --- parsing -----------------------------------------------------------------


def test_valid_single_values():
    result = parse(_body("us/regulation/7/273/7", "42409703021"), REPO)
    assert result.errors == []
    assert result.linkage == Linkage(
        targets=["us/regulation/7/273/7"], signed_runs=["42409703021"]
    )


def test_none_is_explicit_and_empty():
    result = parse(_body("none", "none"))
    assert result.errors == []
    assert result.linkage == Linkage()


def test_none_is_case_insensitive_but_must_stand_alone():
    assert parse(_body("None", "NONE")).errors == []
    errors = parse(_body("none, us/statute/26/21", "none")).errors
    assert any("must stand alone" in error for error in errors)


def test_multiple_values_and_mixed_run_kinds():
    result = parse(
        _body(
            "us/regulation/7/273/7, us-tn/manual/dhs/snap/24-22/page-14",
            f"42409703021, local:1a2b3c4d, https://github.com/{REPO}/actions/runs/42536997513/attempts/2, "
            "local:0f1e2d3c-4b5a-6978-8091-a2b3c4d5e6f7",
        ),
        REPO,
    )
    assert result.errors == []
    assert result.linkage.targets == [
        "us/regulation/7/273/7",
        "us-tn/manual/dhs/snap/24-22/page-14",
    ]
    assert result.linkage.signed_runs == ["42409703021", "42536997513"]
    assert result.linkage.local_runs == [
        "local:1a2b3c4d",
        "local:0f1e2d3c-4b5a-6978-8091-a2b3c4d5e6f7",
    ]


def test_backticked_values_are_accepted():
    result = parse(_body("`us/statute/26/21`", "`local:1a2b3c4d`"))
    assert result.errors == []
    assert result.linkage.targets == ["us/statute/26/21"]


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ("No lines here.", "missing `Encoding-Target:`"),
        ("Encoding-Target: none\n", "missing `Fixes-Run:`"),
        ("Fixes-Run: none\n", "missing `Encoding-Target:`"),
        ("", "missing `Encoding-Target:`"),
    ],
)
def test_missing_lines(body, fragment):
    result = parse(body)
    assert result.linkage is None
    assert any(fragment in error for error in result.errors)


def test_none_body_is_missing():
    assert parse(None).linkage is None


@pytest.mark.parametrize(
    ("target", "run", "fragment"),
    [
        ("", "none", "no value"),
        ("us/statute/26/21,", "none", "empty entry"),
        ("us/statute/26/21, us/statute/26/21", "none", "more than once"),
        ("us/statutes/26/21.yaml", "none", "is a file path"),
        ("us", "none", "not a corpus citation path"),
        ("us/statute", "none", "not a corpus citation path"),
        ("US/statute/26/21", "none", "not a corpus citation path"),
        ("us/statute/../21", "none", "not a corpus citation path"),
        ("us/statute/26 21", "none", "not a corpus citation path"),
        ("REPLACE-with-citation-paths-or-none", "none", "not a corpus citation path"),
        ("us/statute/26/21", "", "no value"),
        ("us/statute/26/21", "run 123", "neither"),
        ("us/statute/26/21", "0123", "neither"),
        ("us/statute/26/21", "local:XYZ", "neither"),
        ("us/statute/26/21", "local:1a2b3c4", "neither"),
        ("us/statute/26/21", "local:1a2b3c4d, local:1a2b3c4d", "more than once"),
        (
            "us/statute/26/21",
            f"42409703021, https://github.com/{REPO}/actions/runs/42409703021",
            "listed more than once",
        ),
        (
            "us/statute/26/21",
            "https://github.com/someone/else/actions/runs/42409703021",
            "is a run of someone/else",
        ),
        ("none", "42409703021", "Encoding-Target is `none`"),
        ("none", "local:1a2b3c4d", "Encoding-Target is `none`"),
    ],
)
def test_malformed_values(target, run, fragment):
    result = parse(_body(target, run), REPO)
    assert result.linkage is None
    assert any(fragment in error for error in result.errors), result.errors


def test_a_line_written_twice_is_refused():
    body = _body("none", "none", extra="Encoding-Target: us/statute/26/21\n")
    errors = parse(body).errors
    assert any("appears 2 times" in error for error in errors)


def test_a_key_in_the_wrong_case_is_refused_with_its_spelling():
    errors = parse("encoding-target: none\nFixes-Run: none\n").errors
    assert errors == ["write the key as `Encoding-Target:`, not `encoding-target:`"]


def test_lines_in_html_comments_and_code_fences_do_not_count():
    template = (ROOT / ".github" / "pull_request_template.md").read_text()
    # The template's guidance mentions both keys inside a comment; only its
    # placeholders count, and they fail until replaced.
    assert parse(template).linkage is None
    filled = template.replace(
        "Encoding-Target: REPLACE-with-citation-paths-or-none", "Encoding-Target: none"
    ).replace("Fixes-Run: REPLACE-with-run-ids-or-none", "Fixes-Run: none")
    assert parse(filled).errors == []

    fenced = "```text\nEncoding-Target: us/statute/26/21\nFixes-Run: 1\n```\n"
    assert "missing" in parse(fenced).errors[0]
    commented = "<!-- Encoding-Target: none\nFixes-Run: none -->"
    assert "missing" in parse(commented).errors[0]
    unclosed = "Encoding-Target: none\nFixes-Run: none\n<!-- trailing"
    assert parse(unclosed).errors == []
    assert parse(_body("none", "none").replace("\n", "\r\n")).errors == []


# --- exemption ---------------------------------------------------------------


def test_the_bump_app_login_matches_the_version_rule():
    assert pr_linkage.BUMP_APP_LOGIN == version_provenance.BUMP_APP_LOGIN


def test_the_bump_app_from_this_repository_is_exempt():
    linkage, errors, exempt = check(
        "no lines at all",
        author=pr_linkage.BUMP_APP_LOGIN,
        head_repository=REPO,
        repository=REPO,
    )
    assert (linkage, errors, exempt) == (None, [], True)


@pytest.mark.parametrize(
    ("author", "head"),
    [
        (pr_linkage.BUMP_APP_LOGIN, "fork/axiom-encode"),
        ("axiom-encoder-version-bump", REPO),
        ("PavelMakarchuk", REPO),
    ],
)
def test_only_the_bump_app_from_this_repository_is_exempt(author, head):
    linkage, errors, exempt = check(
        "no lines at all", author=author, head_repository=head, repository=REPO
    )
    assert not exempt
    assert errors


# --- run verification --------------------------------------------------------


def test_a_signed_run_for_a_listed_citation_passes():
    calls: list[str] = []
    linkage, errors, _ = check(
        _body("us/regulation/7/273/7", "42409703021, local:1a2b3c4d"),
        repository=REPO,
        fetch=_fetch(
            {"42409703021": (200, _run_payload("us/regulation/7/273/7"))}, calls
        ),
    )
    assert errors == []
    assert linkage.signed_runs == ["42409703021"]
    # Local runs are format-checked only.
    assert calls == [f"repos/{REPO}/actions/runs/42409703021"]


def test_an_unknown_run_fails():
    _, errors, _ = check(
        _body("us/statute/26/21", "42409703021"), repository=REPO, fetch=_fetch({})
    )
    assert errors == [f"Fixes-Run: run 42409703021 does not exist in {REPO}"]


def test_a_run_of_another_workflow_fails():
    payload = _run_payload("us/statute/26/21", path=".github/workflows/ci.yml")
    _, errors, _ = check(
        _body("us/statute/26/21", "1"),
        repository=REPO,
        fetch=_fetch({"1": (200, payload)}),
    )
    assert "is a run of .github/workflows/ci.yml" in errors[0]


def test_a_run_whose_citation_is_not_listed_fails():
    _, errors, _ = check(
        _body("us/statute/26/21", "1"),
        repository=REPO,
        fetch=_fetch({"1": (200, _run_payload("us/statute/26/32"))}),
    )
    assert errors == [
        "Fixes-Run: run 1 encoded us/statute/26/32, which Encoding-Target does not list"
    ]


def test_a_run_name_without_a_citation_fails():
    payload = {
        **_run_payload("x/y/z"),
        "display_title": "Targeted signed RuleSpec re-encode",
    }
    _, errors, _ = check(
        _body("us/statute/26/21", "1"),
        repository=REPO,
        fetch=_fetch({"1": (200, payload)}),
    )
    assert "cannot read the citation" in errors[0]


@pytest.mark.parametrize("status", [0, 429, 500, 502, 403, 401])
def test_an_api_failure_fails_closed(status):
    calls: list[str] = []
    sleeps: list[float] = []
    _, errors, _ = check(
        _body("us/statute/26/21", "1"),
        repository=REPO,
        fetch=_fetch({"1": (status, None)}, calls),
        sleep=sleeps.append,
    )
    assert len(errors) == 1
    assert "fails closed" in errors[0]
    transient = status in (0, 429, 500, 502)
    assert len(calls) == (3 if transient else 1)
    assert len(sleeps) == (2 if transient else 0)


def test_a_transient_failure_then_success_passes():
    answers = iter([(503, None), (200, _run_payload("us/statute/26/21"))])
    linkage = Linkage(targets=["us/statute/26/21"], signed_runs=["1"])
    errors = verify_signed_runs(
        linkage, REPO, lambda path: next(answers), sleep=lambda seconds: None
    )
    assert errors == []


def test_too_many_runs_are_refused():
    runs = ", ".join(str(n) for n in range(1, pr_linkage.MAX_RUNS + 2))
    errors = parse(_body("us/statute/26/21", runs)).errors
    assert any("at most" in error for error in errors)


# --- command line and workflow ----------------------------------------------


def _event(tmp_path: Path, body: str | None, login: str = "someone") -> Path:
    path = tmp_path / "event.json"
    path.write_text(
        json.dumps(
            {
                "pull_request": {
                    "body": body,
                    "user": {"login": login},
                    "head": {"repo": {"full_name": REPO}},
                }
            }
        )
    )
    return path


def _run_module(*args: str) -> subprocess.CompletedProcess:
    # As CI runs it: isolated, standard library only, by path.
    return subprocess.run(
        [sys.executable, "-I", str(ROOT / "src/axiom_encode/pr_linkage.py"), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_passes_a_valid_event(tmp_path):
    done = _run_module(
        "--event", str(_event(tmp_path, _body("none", "none"))), "--repository", REPO
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Encoding-Target: none" in done.stdout


def test_cli_fails_a_missing_body(tmp_path):
    done = _run_module("--event", str(_event(tmp_path, None)), "--repository", REPO)
    assert done.returncode == 1
    assert "::error::missing `Encoding-Target:`" in done.stdout


def test_cli_exempts_the_bump_app(tmp_path):
    event = _event(tmp_path, None, login=pr_linkage.BUMP_APP_LOGIN)
    done = _run_module("--event", str(event), "--repository", REPO)
    assert done.returncode == 0
    assert "exempt" in done.stdout


def test_cli_verify_runs_needs_a_token(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    body = tmp_path / "body.md"
    body.write_text(_body("none", "none"))
    assert pr_linkage.main(["--body-file", str(body), "--verify-runs"]) == 2


def test_the_workflow_runs_the_module_on_every_body_change():
    workflow = yaml.safe_load((ROOT / ".github/workflows/pr-linkage.yml").read_text())
    triggers = workflow[True]  # PyYAML reads the `on:` key as True.
    assert "pull_request" not in triggers
    assert set(triggers["pull_request_target"]["types"]) >= {
        "opened",
        "edited",
        "synchronize",
        "reopened",
    }
    assert "merge_group" in triggers
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    steps = workflow["jobs"]["linkage"]["steps"]
    check_step = next(
        step
        for step in steps
        if step.get("name") == "Check Encoding-Target and Fixes-Run"
    )
    assert "python3 -I src/axiom_encode/pr_linkage.py" in check_step["run"]
    assert "--verify-runs" in check_step["run"]
    assert "$GITHUB_EVENT_PATH" in check_step["run"]
    # The body never passes through an expression into the shell.
    assert "github.event.pull_request.body" not in json.dumps(workflow)
    assert set(check_step["env"]) == {"GITHUB_TOKEN"}


# --- review-fix cycle 1 ------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "- Encoding-Target: none\n- Fixes-Run: none\n",
        "* **Encoding-Target:** none\n* **Fixes-Run:** none\n",
        "**Encoding-Target**: none\n__Fixes-Run__: none\n",
        "   Encoding-Target: none\nFixes-Run: none\n",
    ],
)
def test_list_and_bold_markup_counts(body):
    assert parse(body).errors == []


def test_an_indented_code_block_does_not_count():
    body = "    Encoding-Target: none\n    Fixes-Run: none\n"
    assert "missing" in parse(body).errors[0]
    tabbed = "\tEncoding-Target: none\n\tFixes-Run: none\n"
    assert "missing" in parse(tabbed).errors[0]


def test_an_unclosed_comment_inside_a_fence_hides_nothing():
    body = "```html\n<!-- start\n```\n\nEncoding-Target: none\nFixes-Run: none\n"
    assert parse(body).errors == []


def test_a_fence_inside_a_comment_opens_no_code_block():
    body = "<!--\n```\n-->\nEncoding-Target: none\nFixes-Run: none\n"
    assert parse(body).errors == []


def test_text_around_an_inline_comment_still_counts():
    body = "Encoding-Target: none <!-- generic -->\nFixes-Run: <!-- x --> none\n"
    assert parse(body).errors == []


@pytest.mark.parametrize(
    "url",
    [
        f"https://github.com/{REPO}/actions/runs/42409703021/job/123456",
        f"https://github.com/{REPO}/actions/runs/42409703021?pr=2156",
        f"https://github.com/{REPO}/actions/runs/42409703021/attempts/2/job/9#step:3:1",
    ],
)
def test_run_urls_copied_from_the_ui_are_normalized(url):
    result = parse(_body("us/statute/26/21", url), REPO)
    assert result.errors == []
    assert result.linkage.signed_runs == ["42409703021"]


@pytest.mark.parametrize(
    ("event", "branch"),
    [("push", "main"), ("workflow_dispatch", "feature/x"), (None, None)],
)
def test_a_signed_run_off_main_or_not_dispatched_fails(event, branch):
    payload = {
        **_run_payload("us/statute/26/21"),
        "event": event,
        "head_branch": branch,
    }
    _, errors, _ = check(
        _body("us/statute/26/21", "1"),
        repository=REPO,
        fetch=_fetch({"1": (200, payload)}),
    )
    assert "not a workflow_dispatch run on main" in errors[0]


def test_a_secondary_rate_limit_403_is_retried(monkeypatch):
    import email.message
    import urllib.error

    headers = email.message.Message()
    headers["retry-after"] = "60"

    def raise_403(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "rate", headers, None)

    monkeypatch.setattr(pr_linkage.urllib.request, "urlopen", raise_403)
    waits: list[float] = []
    fetch = pr_linkage.github_fetch("t", sleep=waits.append)
    assert fetch("repos/x/y/actions/runs/1") == (429, None)
    assert waits == [60.0]
    headers.replace_header("retry-after", "3600")
    assert fetch("repos/x/y/actions/runs/1") == (429, None)
    assert waits == [60.0]  # too long to wait: fail closed after the retries

    plain = email.message.Message()

    def raise_plain_403(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "forbidden", plain, None)

    monkeypatch.setattr(pr_linkage.urllib.request, "urlopen", raise_plain_403)
    assert pr_linkage.github_fetch("t")("repos/x/y/actions/runs/1") == (403, None)


def test_the_workflow_runs_only_the_base_branch_checker():
    # pull_request_target holds a token for the base repository, so no step
    # may check out or run the pull request's code.
    text = (ROOT / ".github/workflows/pr-linkage.yml").read_text()
    workflow = yaml.safe_load(text)
    steps = workflow["jobs"]["linkage"]["steps"]
    checkouts = [step for step in steps if "uses" in step]
    assert len(checkouts) == 1
    assert "actions/checkout@" in checkouts[0]["uses"]
    assert checkouts[0]["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkouts[0]["with"]["persist-credentials"] is False
    assert checkouts[0]["with"]["sparse-checkout"] == "src/axiom_encode/pr_linkage.py"
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["jobs"]["linkage"]["timeout-minutes"] * 60 > (
        # The deadline is checked before each run; one run's lookup can then
        # take three attempts of a 20 s timeout and a rate-limit wait, plus
        # the 2 s and 4 s backoffs.
        pr_linkage.VERIFY_DEADLINE + 3 * (20 + pr_linkage.RATE_LIMIT_WAIT) + 6
    )
    for step in steps:
        assert "${{ github.event.pull_request." not in step.get("run", "")
    for forbidden in ("head.sha", "head.ref", "merge_commit_sha", "refs/pull/"):
        assert forbidden not in text
    queue = next(
        step for step in steps if step.get("if") == "github.event_name == 'merge_group'"
    )
    assert "uses" not in queue
    assert all(
        step.get("if")
        in (
            "github.event_name == 'pull_request_target'",
            "github.event_name == 'merge_group'",
        )
        for step in steps
    )


def test_every_automation_that_opens_a_pull_request_here_writes_both_lines():
    # A pull request this repository's workflows open against this repository
    # must pass the check, except the version-bump App's, which is exempt.
    opened = []
    for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
        if path.name == "encoder-version-bump.yml":
            continue
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                run = step.get("run", "")
                if "gh pr create" in run or (
                    re.search(r"repos/\$\{?GITHUB_REPOSITORY\}?/pulls\b(?!/)", run)
                    and re.search(r"POST|-X\s*POST", run)
                ):
                    opened.append((path.name, job_name, run))
    assert {name for name, _, _ in opened} == {
        "finalize-signed-snap-queue.yml",
        "prepare-all-state-snap-queue.yml",
    }
    for name, job_name, run in opened:
        assert "Encoding-Target: none" in run, (name, job_name)
        assert "Fixes-Run: none" in run, (name, job_name)


# --- review-fix cycle 2 ------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "**Encoding-Target: none**\n**Fixes-Run: none**\n",
        "1. Encoding-Target: none\n2) Fixes-Run: none\n",
        "Encoding-Target: none\\\nFixes-Run: none\n",
    ],
)
def test_more_markdown_forms_count(body):
    assert parse(body).errors == []


@pytest.mark.parametrize(
    "value",
    [
        f"[run](https://github.com/{REPO}/actions/runs/42409703021)",
        f"<https://github.com/{REPO}/actions/runs/42409703021>",
        "**42409703021**",
    ],
)
def test_linked_run_values_are_unwrapped(value):
    result = parse(_body("us/statute/26/21", value), REPO)
    assert result.errors == []
    assert result.linkage.signed_runs == ["42409703021"]


def test_a_fence_with_an_info_string_does_not_close_a_block():
    body = "```python\nx\n```python\nEncoding-Target: none\nFixes-Run: none\n"
    assert "missing" in parse(body).errors[0]


def test_an_unclosed_comment_marker_in_prose_is_literal():
    body = "Strip `<!--` blocks first.\n\nEncoding-Target: none\nFixes-Run: none\n"
    assert parse(body).errors == []
    closed_later = "<!-- a\nEncoding-Target: none\n-->\nFixes-Run: none\n"
    assert "missing `Encoding-Target:`" in parse(closed_later).errors[0]


def test_a_comment_block_opened_at_a_line_start_hides_the_rest():
    # GitHub hides an unclosed `<!--` block to the end of the body.
    body = "<!-- note\nEncoding-Target: none\nFixes-Run: none\n"
    assert "missing" in parse(body).errors[0]
    body = "   <!-- note\nEncoding-Target: none\nFixes-Run: none\n"
    assert "missing" in parse(body).errors[0]


def test_inline_markers_on_different_lines_hide_nothing():
    body = (
        "Handles `<!--` here.\nEncoding-Target: none\nFixes-Run: none\n"
        "And `-->` there.\n"
    )
    assert parse(body).errors == []


def test_run_lookups_stop_at_the_deadline():
    now = [0.0]
    calls: list[str] = []

    def fetch(path):
        calls.append(path)
        now[0] += 100.0
        return 200, _run_payload("us/statute/26/21")

    linkage = Linkage(targets=["us/statute/26/21"], signed_runs=["1", "2", "3"])
    errors = verify_signed_runs(
        linkage, REPO, fetch, sleep=lambda s: None, clock=lambda: now[0], deadline=150
    )
    assert len(calls) == 2
    assert errors == [
        "Fixes-Run: run lookups passed the 150s deadline before run 3; "
        "the check fails closed, so rerun this job"
    ]

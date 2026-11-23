"""The enforcing runner: the group's validation at success, whole inert events.

Regression and hostile cases for the review of head ``979124f10``:

1. a merge group passed after its validation had turned red, because the
   group's required check was read only before the evaluation;
2. in the encoder repository itself a ``merge_group`` event that held nothing
   but the repository name ended the job with success.

API payloads are fixtures: the ``World`` fake of ``test_content_merge_guard``
and the helpers of ``test_merge_guard_runner``. No test makes a network call,
sleeps or reads a clock.
"""

from __future__ import annotations

import copy
import urllib.error

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import merge_guard_runner as runner
from tests.test_content_merge_guard import REPO, ROOT, B, G, H, build, oid, plain, queue
from tests.test_merge_guard_runner import (
    FIRST_EVALUATION_READ,
    GROUP_CHECK_LISTING,
    GROUP_RUN,
    GROUP_RUN_PATH,
    REPOSITORY_ROOT,
    SOURCE,
    V5_LANE,
    Time,
    five_v5_units,
    group_event,
    judge_group,
    pr_event,
    queued,
    reason_of,
    run_enforce_command,
    run_source_command,
    source_event,
)
from tests.test_merge_guard_runner_reads import INCOMPLETE_OUTCOMES

OTHER_RUN = "1000"
OTHER_RUN_PATH = f"{ROOT}/actions/runs/{OTHER_RUN}"
LAST_EVALUATION_READ_COUNT = 2 + 29  # one poll of two reads, then the evaluation


def http_error(code):
    def fail():
        raise urllib.error.HTTPError("https://api.github.com/x", code, "x", {}, None)

    return fail


def network_error():
    raise urllib.error.URLError("connection reset")


def timed_out():
    raise TimeoutError("timed out")


def runner_error():
    raise runner.RunnerError("GitHub GET request failed: HTTPError")


def listing_of(world):
    return copy.deepcopy(world.api[GROUP_CHECK_LISTING])


def check(world, **changes):
    """Replace the one listed check of the group head by a changed copy."""
    listing = listing_of(world)
    listing["check_runs"][0] |= changes
    world.api[GROUP_CHECK_LISTING] = listing


def workflow_run(world, **changes):
    world.api[GROUP_RUN_PATH] = world.api[GROUP_RUN_PATH] | changes


def other_run(world, **changes):
    """The check now points at another workflow run, green unless changed."""
    check(
        world,
        details_url=f"https://github.com/{REPO}/actions/runs/{OTHER_RUN}/job/2",
    )
    world.api[OTHER_RUN_PATH] = (
        world.api[GROUP_RUN_PATH] | {"id": int(OTHER_RUN)} | changes
    )


def second_check(world, **changes):
    listing = listing_of(world)
    first = listing["check_runs"][0]
    second = first | {
        "id": 3,
        "details_url": f"https://github.com/{REPO}/actions/runs/{OTHER_RUN}/job/3",
    }
    world.api[OTHER_RUN_PATH] = world.api[GROUP_RUN_PATH] | {"id": int(OTHER_RUN)}
    world.api[GROUP_CHECK_LISTING] = {
        "total_count": 2,
        "check_runs": [first, second | changes],
    }


def third_check(world):
    second_check(world)
    listing = listing_of(world)
    listing["check_runs"].append(listing["check_runs"][1] | {"id": 4})
    listing["total_count"] = 3
    world.api[GROUP_CHECK_LISTING] = listing


def set_api(path, value):
    def change(world):
        world.api[path] = value

    return change


def listing_change(**changes):
    def change(world):
        world.api[GROUP_CHECK_LISTING] = listing_of(world) | changes

    return change


# What happens to the group's validation between the read before the
# evaluation and the read after it. Every one of these must fail the job.
BETWEEN_THE_TWO_READS = {
    # --- the review's case and its variants
    "the same check turns red": lambda w: check(w, conclusion="failure"),
    "a second attempt fails (the review's probe)": lambda w: (
        check(w, id=3, conclusion="failure"),
        workflow_run(w, run_attempt=2, conclusion="failure"),
    ),
    "a second attempt is in progress": lambda w: (
        check(w, id=3, status="in_progress", conclusion=None),
        workflow_run(w, run_attempt=2, status="in_progress", conclusion=None),
    ),
    "a second attempt is queued": lambda w: check(
        w, id=3, status="queued", conclusion=None
    ),
    "the check is completed and its run was restarted": lambda w: workflow_run(
        w, run_attempt=2, status="in_progress", conclusion=None
    ),
    "a second attempt succeeded": lambda w: (
        check(w, id=3),
        workflow_run(w, run_attempt=2),
    ),
    "the same check, the run is at attempt 2 and green": lambda w: workflow_run(
        w, run_attempt=2
    ),
    "the same check, the run turned red": lambda w: workflow_run(
        w, conclusion="failure"
    ),
    "the same check, the run was cancelled": lambda w: workflow_run(
        w, conclusion="cancelled"
    ),
    # --- replaced
    "another workflow run id, green": other_run,
    "another workflow run id, red": lambda w: other_run(w, conclusion="failure"),
    "another check run id with the same name and run": lambda w: check(w, id=3),
    "one more green check with the required name": second_check,
    "one more check, in progress": lambda w: second_check(
        w, status="in_progress", conclusion=None
    ),
    "one more check, red": lambda w: second_check(w, conclusion="failure"),
    "two more checks": third_check,
    # --- deleted or not readable whole
    "the check is deleted": set_api(
        GROUP_CHECK_LISTING, {"total_count": 0, "check_runs": []}
    ),
    "pagination: more runs than the page lists": listing_change(total_count=2),
    "pagination: 101 listed": lambda w: w.api.__setitem__(
        GROUP_CHECK_LISTING,
        {"total_count": 101, "check_runs": listing_of(w)["check_runs"] * 101},
    ),
    "the listing has no total_count": lambda w: w.api.__setitem__(
        GROUP_CHECK_LISTING, {"check_runs": listing_of(w)["check_runs"]}
    ),
    "the listing has no check_runs": listing_change(check_runs=None),
    "the listing is a list": set_api(GROUP_CHECK_LISTING, []),
    "the listing is null": set_api(GROUP_CHECK_LISTING, None),
    "a listed check is not an object": listing_change(check_runs=["x"]),
    "the workflow run is not an object": set_api(GROUP_RUN_PATH, []),
    "the workflow run is null": set_api(GROUP_RUN_PATH, None),
    # --- failed requests
    "rate limit on the listing (403)": set_api(GROUP_CHECK_LISTING, http_error(403)),
    "rate limit on the listing (429)": set_api(GROUP_CHECK_LISTING, http_error(429)),
    "server error on the listing": set_api(GROUP_CHECK_LISTING, http_error(502)),
    "network error on the listing": set_api(GROUP_CHECK_LISTING, network_error),
    "timeout on the listing": set_api(GROUP_CHECK_LISTING, timed_out),
    "the reader refuses the listing": set_api(GROUP_CHECK_LISTING, runner_error),
    "rate limit on the workflow run": set_api(GROUP_RUN_PATH, http_error(403)),
    "network error on the workflow run": set_api(GROUP_RUN_PATH, network_error),
    "the workflow run is gone (404)": set_api(GROUP_RUN_PATH, http_error(404)),
    # --- foreign or of another shape
    "the check is now of another app": lambda w: check(w, app={"id": 987654}),
    "the check has another name": lambda w: check(w, name="validate"),
    "the check is on another head": lambda w: check(w, head_sha=H),
    "the check has no id": lambda w: check(w, id=None),
    "the check id is text": lambda w: check(w, id="2"),
    "the check id is a boolean": lambda w: check(w, id=True),
    "the run has no attempt": lambda w: workflow_run(w, run_attempt=None),
    "the run attempt is text": lambda w: workflow_run(w, run_attempt="1"),
    "the run attempt is a boolean": lambda w: workflow_run(w, run_attempt=True),
    "the run attempt is 0": lambda w: workflow_run(w, run_attempt=0),
    "the run is of another event": lambda w: workflow_run(w, event="pull_request"),
    "the run is of another workflow": lambda w: workflow_run(
        w, path=".github/workflows/other.yml"
    ),
    "the run is on another head": lambda w: workflow_run(w, head_sha=H),
    "the run is of another repository": lambda w: workflow_run(
        w, repository={"full_name": "TheAxiomFoundation/rulespec-uk"}
    ),
    "the check is skipped": lambda w: check(w, conclusion="skipped"),
    "the check is neutral": lambda w: check(w, conclusion="neutral"),
    "the check is cancelled": lambda w: check(w, conclusion="cancelled"),
    "the check has no conclusion": lambda w: check(w, conclusion=None),
}


def changing_during_the_evaluation(world, change, at=FIRST_EVALUATION_READ):
    """Apply ``change`` to the world at the first read of the evaluation."""
    fetch = world.fetch
    state = {"changed": False}

    def changing(path):
        if path == at and not state["changed"]:
            state["changed"] = True
            change(world)
        return fetch(path)

    world.fetch = changing
    return state


def assert_failed_after_the_evaluation(world, report, code):
    assert code == 1, report
    assert report["error"]
    # The evaluation ran once and reached its verdict; the verdict stays in
    # the report and the job still fails.
    assert report["evaluations"] == 1 and report["group_ci_rechecks"] == 1
    assert reason_of(report) == ("pass", guard.R_PASS)
    assert world.calls.count(GROUP_CHECK_LISTING) == 2


def test_the_review_probe_a_group_whose_validation_turned_red_fails():
    # The reviewer's probe: ``queued()`` and ``judge_group()``, the group's
    # validation replaced by a failed second attempt at the first read of the
    # evaluation. It returned exit 0, ``pass``, group CI ``failure`` and one
    # listing read.
    world = queued()
    state = changing_during_the_evaluation(
        world, BETWEEN_THE_TWO_READS["a second attempt fails (the review's probe)"]
    )
    report, code = judge_group(world)
    assert state["changed"]
    assert world.api[GROUP_CHECK_LISTING]["check_runs"][0]["conclusion"] == "failure"
    assert_failed_after_the_evaluation(world, report, code)
    assert "did not succeed" in report["error"]


@pytest.mark.parametrize("label", sorted(BETWEEN_THE_TWO_READS))
def test_no_change_of_the_group_validation_during_the_evaluation_passes(label):
    world = queued()
    state = changing_during_the_evaluation(world, BETWEEN_THE_TWO_READS[label])
    report, code = judge_group(world)
    assert state["changed"]
    assert_failed_after_the_evaluation(world, report, code)


@pytest.mark.parametrize("label", sorted(BETWEEN_THE_TWO_READS))
def test_no_change_of_the_group_validation_passes_a_stand_aside_group(label):
    # The same for a group whose verdict is ``not_applicable`` (earlier lane).
    world = queued(plain(V5_LANE))
    state = changing_during_the_evaluation(world, BETWEEN_THE_TWO_READS[label])
    report, code = judge_group(world)
    assert state["changed"] and code == 1 and report["error"]
    assert reason_of(report) == ("not_applicable", guard.R_EARLIER_LANE)
    assert report["group_ci_rechecks"] == 1


@pytest.mark.parametrize("label", sorted(BETWEEN_THE_TWO_READS))
def test_a_change_at_the_last_read_of_the_evaluation_is_still_seen(label):
    # The change comes as late as it can: at the last read the evaluation
    # makes (its recheck of the pull request's own CI run).
    world = queued()
    reference = queued()
    judge_group(reference)
    last = reference.calls[LAST_EVALUATION_READ_COUNT - 1]
    assert reference.calls[LAST_EVALUATION_READ_COUNT] == GROUP_CHECK_LISTING
    count = {"reads": 0}
    fetch = world.fetch

    def changing(path):
        count["reads"] += 1
        if count["reads"] == LAST_EVALUATION_READ_COUNT:
            assert path == last
            BETWEEN_THE_TWO_READS[label](world)
        return fetch(path)

    world.fetch = changing
    report, code = judge_group(world)
    assert_failed_after_the_evaluation(world, report, code)


def test_an_unchanged_group_validation_passes_with_exactly_one_more_read():
    world = queued()
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["group_ci_rechecks"] == 1 and "error" not in report
    # The last two reads of the run are the listing and the workflow run:
    # nothing is read, slept or evaluated after them.
    assert world.calls[-2:] == [GROUP_CHECK_LISTING, GROUP_RUN_PATH]
    assert world.calls[:2] == [GROUP_CHECK_LISTING, GROUP_RUN_PATH]
    assert world.calls.count(GROUP_CHECK_LISTING) == 2
    assert world.calls.count(GROUP_RUN_PATH) == 2
    assert len(world.calls) == 2 + 29 + 2 == report["requests"]


def test_two_listed_checks_in_another_order_are_the_same_validation():
    # The identity is a set: the order of a listing is not a change.
    world = queued()
    second_check(world)

    def reorder(w):
        listing = listing_of(w)
        listing["check_runs"].reverse()
        w.api[GROUP_CHECK_LISTING] = listing

    state = changing_during_the_evaluation(world, reorder)
    report, code = judge_group(world)
    assert state["changed"]
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert len(world.calls) == 3 + 29 + 3 == report["requests"]


def test_two_listed_checks_with_one_identity_are_foreign():
    world = queued()
    listing = listing_of(world)
    listing["check_runs"] *= 2
    listing["total_count"] = 2
    world.api[GROUP_CHECK_LISTING] = listing
    report, code = judge_group(world)
    assert code == 1 and report["verdict"] is None and report["evaluations"] == 0
    assert "not the merge group run" in report["error"]


def test_a_failing_verdict_needs_no_second_read():
    world = queued()
    world.reviews([])
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("fail", guard.R_NO_APPROVAL), 1)
    assert report["group_ci_rechecks"] == 0
    assert world.calls.count(GROUP_CHECK_LISTING) == 1


def test_a_success_outside_the_success_set_is_rechecked_and_still_fails(monkeypatch):
    # The control-plane pass is not in ``SUCCESS_VERDICTS``. It is rechecked
    # like every verdict that is not ``fail`` and then fails the job anyway.
    def control_plane(name, head, base, number, *, fetch):
        return guard.Verdict(
            guard.PASS,
            guard.R_CONTROL_PLANE_PASS,
            "",
            {"pull_request": number, "merge_group": {"head": head, "base": base}},
        )

    monkeypatch.setattr(guard, "evaluate_merge_group", control_plane)
    world = queued()
    report, code = judge_group(world)
    assert code == 1 and report["group_ci_rechecks"] == 1


def test_a_passing_verdict_with_the_incomplete_mark_fails_the_job(monkeypatch):
    def marked(name, head, base, number, *, fetch):
        return guard.Verdict(
            guard.PASS,
            guard.R_PASS,
            "",
            {
                "pull_request": number,
                "merge_group": {"head": head, "base": base},
                guard.EVIDENCE_INCOMPLETE_READ: True,
            },
        )

    monkeypatch.setattr(guard, "evaluate_merge_group", marked)
    report, code = judge_group(queued())
    assert code == 1 and report["error"] == "an evaluation could not read GitHub"


def test_the_request_cap_reached_at_the_second_read_fails_the_job(monkeypatch):
    world = queued()
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", LAST_EVALUATION_READ_COUNT)
    report, code = judge_group(world)
    assert code == 1 and "request cap" in report["error"]
    assert reason_of(report) == ("pass", guard.R_PASS)
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", LAST_EVALUATION_READ_COUNT + 1)
    report, code = judge_group(queued())
    assert code == 1 and "request cap" in report["error"]
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", LAST_EVALUATION_READ_COUNT + 2)
    report, code = judge_group(queued())
    assert code == 0


def test_the_second_read_does_not_wait_or_sleep():
    world = queued()
    changing_during_the_evaluation(
        world, BETWEEN_THE_TWO_READS["a second attempt is in progress"]
    )
    time = Time()
    report, code = judge_group(world, time)
    assert code == 1 and time.sleeps == []
    assert "no longer completed" in report["error"]
    assert report["ci_polls"] == 1 and report["waited_seconds"] == 0


@pytest.mark.parametrize(
    "first",
    [None, frozenset(), set(), [], (), "x", {(2, GROUP_RUN, 1)}, [(2, GROUP_RUN, 1)]],
)
def test_the_comparison_refuses_a_first_identity_that_is_not_a_filled_frozenset(
    first,
):
    world = queued()
    with pytest.raises(runner.RunnerError, match="not the successful run"):
        runner._require_same_group_ci(ROOT, G, world.fetch, first)


def test_the_identity_is_the_check_the_run_and_the_attempt():
    world = queued()
    identity = runner._group_ci_succeeded(ROOT, G, world.fetch)
    assert identity == frozenset({(2, GROUP_RUN, 1)})
    runner._require_same_group_ci(ROOT, G, world.fetch, identity)
    for other in (
        frozenset({(3, GROUP_RUN, 1)}),
        frozenset({(2, OTHER_RUN, 1)}),
        frozenset({(2, GROUP_RUN, 2)}),
        frozenset({(2, GROUP_RUN, 1), (3, OTHER_RUN, 1)}),
    ):
        with pytest.raises(runner.RunnerError, match="not the successful run"):
            runner._require_same_group_ci(ROOT, G, world.fetch, other)


def test_the_command_fails_a_replaced_group_validation_through_the_http_layer(
    capsys, tmp_path
):
    world = queued()
    changing_during_the_evaluation(
        world, BETWEEN_THE_TWO_READS["a second attempt succeeded"]
    )
    code, report, opener = run_enforce_command(
        capsys, tmp_path, world, group_event(), GITHUB_EVENT_NAME="merge_group"
    )
    assert code == 1 and "not the successful run" in report["error"]
    assert report["verdict"]["verdict"] == "pass"
    assert {request.get_method() for request in opener.requests} == {"GET"}
    assert report["requests"] == len(opener.requests) == 2 + 29 + 2


# --- group membership: a limit, recorded (non-blocking finding) ---------------


def test_a_group_with_an_earlier_queued_merge_as_parent_and_one_tree_passes():
    # This test records a limit; it is not a guard. The evaluator binds the
    # CONTENT of the group (its tree is the tree of the judged head, on the
    # judged base). It reads no parent of the group commit, so it cannot tell
    # how many pull requests the queue put into the group when the others
    # add nothing to the tree. Activation therefore requires a queue with a
    # group size of one (design section 6 and case 24).
    world = queued()
    world.api[f"{ROOT}/git/commits/{G}"]["parents"] = [
        {"sha": oid("an-earlier-queued-merge")},
        {"sha": H},
    ]
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert "group size of one" in " ".join(guard.evaluate_merge_group.__doc__.split())


# --- 2. the inert path needs a whole event ------------------------------------


def group_of_source(**changes):
    return group_event(repository=SOURCE, **changes)


def without(mapping, key):
    return {name: value for name, value in mapping.items() if name != key}


INCOMPLETE_SOURCE_GROUP_EVENTS = {
    "the review's probe: only the repository": {"repository": {"full_name": SOURCE}},
    "merge_group is null": group_of_source() | {"merge_group": None},
    "merge_group is a list": group_of_source() | {"merge_group": []},
    "merge_group is text": group_of_source() | {"merge_group": "x"},
    "merge_group is empty": group_of_source() | {"merge_group": {}},
    "merge_group is true": group_of_source() | {"merge_group": True},
    "no head_sha": group_of_source(head_sha=None),
    "head_sha is a branch": group_of_source(head_sha="main"),
    "head_sha is short": group_of_source(head_sha=G[:39]),
    "head_sha is upper case": group_of_source(head_sha=G.upper()),
    "head_sha has a newline": group_of_source(head_sha=G + "\n"),
    "no base_sha": group_of_source(base_sha=None),
    "base_sha is a number": group_of_source(base_sha=7),
    "no head_ref": group_of_source(head_ref=None),
    "head_ref is main": group_of_source(head_ref="refs/heads/main"),
    "head_ref is a tag": group_of_source(
        head_ref=f"refs/tags/gh-readonly-queue/main/pr-42-{B}"
    ),
    "head_ref names no pull request": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/main/{B}"
    ),
    "head_ref has pull request 0": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/main/pr-0-{B}"
    ),
    "head_ref has a short commit": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/main/pr-42-{B[:39]}"
    ),
    "head_ref has a trailing newline": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/main/pr-42-{B}\n"
    ),
    "head_ref has no branch": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/pr-42-{B}"
    ),
    "head_ref is of another branch than base_ref": group_of_source(
        head_ref=f"refs/heads/gh-readonly-queue/release/pr-42-{B}"
    ),
    "head_ref is a number": group_of_source(head_ref=42),
    "no base_ref": group_of_source(base_ref=None),
    "base_ref is a bare name": group_of_source(base_ref="main"),
    "base_ref is a tag": group_of_source(base_ref="refs/tags/main"),
    "base_ref is empty": group_of_source(base_ref="refs/heads/"),
    "base_ref has a space": group_of_source(base_ref="refs/heads/ma in"),
    "a pull_request payload under the merge_group name": source_event("pull_request"),
    "no repository": without(group_of_source(), "repository"),
    "repository is null": group_of_source() | {"repository": None},
    "repository is the target": group_event(),
    "repository in another case": group_of_source()
    | {"repository": {"full_name": SOURCE.lower()}},
}


@pytest.mark.parametrize("label", sorted(INCOMPLETE_SOURCE_GROUP_EVENTS))
def test_the_inert_path_fails_on_an_incomplete_merge_group_event(
    capsys, tmp_path, label
):
    code, report = run_source_command(
        capsys,
        tmp_path,
        INCOMPLETE_SOURCE_GROUP_EVENTS[label],
        GITHUB_EVENT_NAME="merge_group",
    )
    assert code == 1 and "inert" not in report and report["error"]
    assert report["verdict"] is None and report["requests"] == 0


def pull_of_source(**changes):
    event = source_event("pull_request")
    event["pull_request"] |= changes
    return event


INCOMPLETE_SOURCE_PULL_EVENTS = {
    "only the repository": {"repository": {"full_name": SOURCE}},
    "only the repository and the base": {
        "repository": {"full_name": SOURCE},
        "pull_request": {"base": {"repo": {"full_name": SOURCE}}},
    },
    "no payload number": without(source_event("pull_request"), "number"),
    "payload number is text": source_event("pull_request") | {"number": "42"},
    "payload number differs": source_event("pull_request") | {"number": 43},
    "payload number is true": source_event("pull_request") | {"number": True},
    "no pull request number": pull_of_source(number=None),
    "pull request number is 0": pull_of_source(number=0),
    "pull request number is negative": pull_of_source(number=-42),
    "pull request number is a float": pull_of_source(number=42.0),
    "no head": pull_of_source(head=None),
    "head is a commit id, not an object": pull_of_source(head=H),
    "head has no sha": pull_of_source(head={}),
    "head sha is a branch": pull_of_source(head={"sha": "main"}),
    "head sha is short": pull_of_source(head={"sha": H[:39]}),
    "head sha is upper case": pull_of_source(head={"sha": H.upper()}),
    "pull_request is null": source_event("pull_request") | {"pull_request": None},
    "pull_request is a list": source_event("pull_request") | {"pull_request": []},
    "no base": pull_of_source(base=None),
    "base repo is null": pull_of_source(base={"repo": None}),
    "base repo is the target": pull_of_source(base={"repo": {"full_name": REPO}}),
    "a merge_group payload under the pull_request name": group_of_source(),
}


@pytest.mark.parametrize("label", sorted(INCOMPLETE_SOURCE_PULL_EVENTS))
def test_the_inert_path_fails_on_an_incomplete_pull_request_event(
    capsys, tmp_path, label
):
    code, report = run_source_command(
        capsys,
        tmp_path,
        INCOMPLETE_SOURCE_PULL_EVENTS[label],
        GITHUB_EVENT_NAME="pull_request",
    )
    assert code == 1 and "inert" not in report and report["error"]
    assert report["verdict"] is None and report["requests"] == 0


@pytest.mark.parametrize(
    ("name", "event"),
    [
        ("pull_request", source_event("pull_request")),
        ("merge_group", source_event("merge_group")),
        (
            "merge_group",
            group_of_source(
                head_ref=f"refs/heads/gh-readonly-queue/release/2.x/pr-7-{B}",
                base_ref="refs/heads/release/2.x",
            ),
        ),
    ],
)
def test_a_whole_event_of_the_source_repository_is_still_inert(
    capsys, tmp_path, name, event
):
    code, report = run_source_command(capsys, tmp_path, event, GITHUB_EVENT_NAME=name)
    assert code == 0 and report["inert"] == runner.INERT_SOURCE_REPOSITORY
    assert report["requests"] == 0 and "error" not in report


@pytest.mark.parametrize(
    "event",
    [
        {"repository": {"full_name": REPO}},
        group_event() | {"merge_group": None},
        group_event() | {"merge_group": {}},
        group_event(head_sha=None),
        group_event(base_sha=None),
        group_event(head_ref=None),
        group_event(base_ref=None),
        without(group_event(), "merge_group"),
        pr_event(),
    ],
)
def test_the_target_repository_fails_on_an_incomplete_merge_group_event(event):
    # The other repository case: a RuleSpec repository. No read, no verdict.
    world = queue(build())
    report, code = runner.enforce(
        "merge_group", REPO, event, fetch=world.fetch, sleep=Time().sleep
    )
    assert code == 1 and world.calls == [] and report["verdict"] is None
    assert report["error"]


# --- request budget with the second read ---------------------------------------


def test_a_merge_group_of_a_five_unit_train_costs_four_reads_above_its_pull_request():
    # The design's arithmetic for a train: a pull request evaluation of
    # ``11 + 3u`` reads, and four more for the group (two commits, the
    # ancestry and the ``main`` tip).
    train = five_v5_units()
    alone = len(train.calls)
    assert train.evaluate().verdict == "not_applicable"
    pull_request_reads = len(train.calls) - alone
    assert pull_request_reads == 11 + 3 * 5 == 26
    world = queued(five_v5_units())
    report, code = judge_group(world)
    assert code == 0 and reason_of(report) == ("not_applicable", guard.R_EARLIER_LANE)
    assert report["requests"] == 2 + (15 + 3 * 5) + 2 == 34


def test_the_request_cap_still_covers_every_documented_worst_case():
    cap = runner.MAX_ENFORCE_REQUESTS
    polls = runner.MAX_MERGE_GROUP_POLLS
    recheck = runner.GROUP_CI_RECHECK_READS
    assert (cap, polls, runner.MAX_CI_POLLS, recheck) == (350, 101, 101, 3)
    # Pull request: two evaluations and a full wait. Unchanged by the second
    # read, which belongs to the merge-group run only.
    assert 2 * 30 + 101 == 161 <= cap
    assert 2 * (11 + 3 * 37) + 101 == 345 <= cap < 2 * (11 + 3 * 38) + 101
    # Merge group, every poll at three reads.
    assert 3 * polls + 30 + recheck == 336 <= cap
    assert 3 * polls + (15 + 3 * 9) + recheck == 348 <= cap
    assert 3 * polls + (15 + 3 * 10) + recheck == 351 > cap
    # Merge group, one listed check: 100 polls of one read, the last of two.
    assert 100 + 2 + (15 + 3 * 77) + 2 == 350 <= cap
    assert 100 + 2 + (15 + 3 * 78) + 2 > cap


# --- the design says what the code does (doc pins) ----------------------------

DESIGN_REQUIRED = {
    # On this branch the recommendation's checklist replaces the list that
    # change 3b marked superseded; the four rules stand at its head.
    "four rules of the checklist": [
        "**Four rules that no reordering of this list may break**",
        "**The gate is the pinned workflow, not a status check by name.**",
        "Nothing in this list requires a status check named `content-bound-merge-guard`.",
        "**No removal of the pre-sign approval before the queue is verified.**",
        "Change 6 (step 13) comes after the ruleset is `Active` (step 11)",
        "the merge queue is required on `main` with a group size of 1 (step 8)",
        "has shown the guard's `merge_group` job rejecting a dismissed approval",
        "**A live test that a same-named check cannot satisfy the rule**",
        "**One failed canary for each trigger** before `Active`",
        "**the second canary: in that trial the merge group of a pull request "
        "with no lane showed a failed `content-bound-merge-guard` job of the "
        "`merge_group` event",
        "**Hard precondition: a live `merge_group` run of the guard has rejected "
        "a dismissed approval**",
    ],
    "inert path": [
        "The payload must also be a whole event of its kind",
        "a `merge_group` object with an exact `head_sha` and `base_sha`",
        "**Activation hold, one canary for each trigger:**",
        "the merge group of such a pull request must show a **failed** job",
        'A passed job, or a report with `"inert"`, for either trigger stops the',
    ],
    "same attempt": [
        "**The same successful attempt, read again after the evaluation.**",
        "check run id, the id of its workflow run and that run's `run_attempt`",
        "at another attempt (a green one too)",
        "The second read does not wait",
    ],
    "group membership": [
        "**Group size 1 is a required queue setting, not something the evaluator proves.**",
        "It does not establish the group's **membership**",
        "a larger size is a reason not to activate",
        "the evaluator reads and judges that one pull request only",
        '`grouping_strategy: "ALLGREEN"`, `max_entries_to_merge: 1`, '
        "`min_entries_to_merge: 1` and `max_entries_to_build: 1`",
        "**The evaluator does not enforce them and cannot prove a group to be a "
        "single pull request**",
        "membership is not verified, the queue's group size of 1 is a required setting",
    ],
    "request budget": [
        "3 x 101 + 30 + 3 = 336",
        "16 + 101 + 25 = 142",
        "20 + 2 x 30 = 80",
        "60 + 30 + 3 = 93",
        "the limit is reached at N = 12",
        "19,260 s, 321 minutes",
        "3 x 101 + (15 + 3u) + 3 <= 350 admits about 9",
        "100 + 2 + (15 + 3u) + 2 <= 350 admits about 77",
        "2 x (11 + 3u) + 101 <= 350",
    ],
}


def design_text() -> str:
    return " ".join(
        (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md")
        .read_text(encoding="utf-8")
        .replace("\n> ", "\n")
        .replace("\n>\n", "\n")
        .split()
    )


def design_problems(text: str) -> list[str]:
    return [
        f"{group}: the design lost {phrase!r}"
        for group, phrases in DESIGN_REQUIRED.items()
        for phrase in phrases
        if text.count(phrase) != 1
    ]


def test_the_design_states_the_fixes_of_this_review():
    assert design_problems(design_text()) == []


@pytest.mark.parametrize(
    "phrase", [phrase for phrases in DESIGN_REQUIRED.values() for phrase in phrases]
)
def test_the_design_pin_reports_each_lost_statement(phrase):
    text = design_text()
    assert text.count(phrase) == 1
    assert any(
        repr(phrase) in problem for problem in design_problems(text.replace(phrase, ""))
    )


def test_the_design_no_longer_states_the_figures_the_review_corrected():
    text = design_text()
    for stale in (
        "12 + 101 + 29 = 142\n",
        "3 x 101 + 30 = 333 |",
        "at most 50 requests for a pull request and 90 for a merge group",
        "20 + 30 = 50. The job token",
    ):
        assert stale.strip() not in text or "earlier revision" in text
    assert "20 + 30 = 50" not in text
    assert "= 333 |" not in text


def test_the_workflow_header_names_both_canaries_and_the_second_read():
    header = " ".join(
        line.lstrip("# ")
        for line in (
            REPOSITORY_ROOT / ".github" / "workflows" / "content-bound-merge-guard.yml"
        )
        .read_text(encoding="utf-8")
        .split("\njobs:")[0]
        .splitlines()
    )
    for phrase in (
        "Activation hold, one canary for each trigger",
        "a pull request with no lane AND a merge group of such a pull request "
        "must each show a FAILED job",
        "the payload is a whole event of its kind",
        "it reads that run once more and ends with success only when it is still "
        "the same successful attempt",
    ):
        assert phrase in " ".join(header.split()), phrase


# --- no incomplete read ends the enforcing job with success --------------------


def enforce_cases():
    worlds = {
        "pull_request v6": (build, ..., "pull_request"),
        "pull_request v5": (lambda: plain(V5_LANE), ..., "pull_request"),
        "pull_request allowlist": (
            lambda: plain({"README.md": b"x\n"}),
            ...,
            "pull_request",
        ),
        "merge_group v6": (queued, group_event(), "merge_group"),
        "merge_group v5": (
            lambda: queued(plain(V5_LANE)),
            group_event(),
            "merge_group",
        ),
    }
    cases = []
    for label, (make, event, name) in worlds.items():
        answers = []
        world = make()
        fetch = world.fetch

        def recording(path, fetch=fetch, answers=answers):
            answers.append(fetch(path))
            return answers[-1]

        world.fetch = recording
        _report, code = runner.enforce(
            name,
            REPO,
            pr_event() if event is ... else event,
            fetch=world.fetch,
            sleep=Time().sleep,
            clock=Time().clock,
        )
        assert code == 0, label
        for index, answer in enumerate(answers):
            for outcome, change in INCOMPLETE_OUTCOMES.items():
                try:
                    applies = change(copy.deepcopy(answer)) is not None
                except OSError:
                    applies = True
                if "tree" in outcome and not (
                    isinstance(answer, dict) and isinstance(answer.get("tree"), list)
                ):
                    # A commit object has a ``tree`` too; only a tree listing
                    # can be truncated.
                    applies = False
                if applies:
                    cases.append((label, index, outcome))
    return worlds, cases


ENFORCE_WORLDS, ENFORCE_INJECTED = enforce_cases()


def test_the_enforce_injection_matrix_covers_every_read_of_every_passing_run():
    reads = {}
    for label, index, _outcome in ENFORCE_INJECTED:
        reads[label] = max(reads.get(label, 0), index + 1)
    assert reads == {
        "pull_request v6": 25,
        "pull_request v5": 14,
        "pull_request allowlist": 5,
        "merge_group v6": 33,
        "merge_group v5": 22,
    }
    for outcome in INCOMPLETE_OUTCOMES:
        assert any(case[2] == outcome for case in ENFORCE_INJECTED), outcome


@pytest.mark.parametrize(("label", "index", "outcome"), ENFORCE_INJECTED)
def test_no_incomplete_read_ends_the_enforcing_job_with_success(label, index, outcome):
    """Read ``index`` of a passing run is incomplete: exit 1 and an error.

    The reads of the wait, of the evaluation and of the second read of the
    group's check are all covered. The job never ends with success, and the
    report always says that the run had an error (not only a ``fail``).
    """
    make, event, name = ENFORCE_WORLDS[label]
    world = make()
    change = INCOMPLETE_OUTCOMES[outcome]
    count = {"reads": 0}
    fetch = world.fetch

    def injected(path):
        position = count["reads"]
        count["reads"] += 1
        value = fetch(path)
        if position != index:
            return value
        replaced = change(copy.deepcopy(value))
        assert replaced is not None
        return None if replaced == "null" else replaced

    time = Time()
    report, code = runner.enforce(
        name,
        REPO,
        pr_event() if event is ... else event,
        fetch=injected,
        sleep=time.sleep,
        clock=time.clock,
    )
    assert count["reads"] > index
    assert code == 1, report
    assert report["error"], report


@pytest.mark.parametrize(
    "commit_object",
    [{"sha": G, "tree": {"sha": "not-a-tree"}}, {"sha": G}, [], None],
)
def test_a_group_commit_that_cannot_be_read_whole_carries_the_mark(commit_object):
    world = queued()
    world.api[f"{ROOT}/git/commits/{G}"] = commit_object
    verdict = guard.evaluate_merge_group("rulespec-us", G, B, 42, fetch=world.fetch)
    assert (verdict.verdict, verdict.reason) == ("fail", guard.R_MERGE_GROUP_TREE)
    assert verdict.evidence[guard.EVIDENCE_INCOMPLETE_READ] is True
    report, code = judge_group(queued_with(commit_object))
    assert code == 1 and report["error"] == "an evaluation could not read GitHub"


def queued_with(commit_object):
    world = queued()
    world.api[f"{ROOT}/git/commits/{G}"] = commit_object
    return world


def test_a_group_tree_that_differs_is_a_decision_without_the_mark():
    world = queued(group_tree=oid("tree-of-two-units"))
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("fail", guard.R_MERGE_GROUP_TREE), 1)
    assert guard.EVIDENCE_INCOMPLETE_READ not in report["verdict"]["evidence"]
    assert "error" not in report


def test_an_unforeseen_error_of_the_group_evaluation_carries_the_mark(monkeypatch):
    def broken(*_args, **_kwargs):
        raise RuntimeError("unforeseen")

    monkeypatch.setattr(guard, "_evaluate_merge_group", broken)
    verdict = guard.evaluate_merge_group("rulespec-us", G, B, 42, fetch=lambda _p: {})
    assert (verdict.verdict, verdict.reason) == ("fail", guard.R_EVALUATION_ERROR)
    assert verdict.evidence[guard.EVIDENCE_INCOMPLETE_READ] is True


def test_a_stale_event_head_is_a_decision_of_the_runner_not_a_failed_read():
    world = build()
    report, code = runner.enforce(
        "pull_request",
        REPO,
        pr_event(head=oid("an-older-head")),
        fetch=world.fetch,
        sleep=Time().sleep,
        clock=Time().clock,
    )
    assert code == 1 and reason_of(report) == ("fail", runner.R_EVENT_HEAD_STALE)
    assert "error" not in report


def test_a_ci_verdict_from_an_incomplete_read_is_not_waited_for():
    # ``ci_not_green`` starts the wait only as a decision. A listing that
    # does not add up is an incomplete read: one evaluation, no poll, an
    # error. (Before, the run waited and judged again.)
    world = build()
    listing = f"{ROOT}/commits/{H}/check-runs?check_name=validate%20%2F%20validate"
    listing += "&filter=latest&per_page=100"
    world.api[listing]["total_count"] = 2
    time = Time()
    report, code = runner.enforce(
        "pull_request",
        REPO,
        pr_event(),
        fetch=world.fetch,
        sleep=time.sleep,
        clock=time.clock,
    )
    assert code == 1 and reason_of(report) == ("fail", guard.R_CI)
    assert report["error"] == "an evaluation could not read GitHub"
    assert (report["evaluations"], report["ci_polls"], time.sleeps) == (1, 0, [])
    assert "no wait and no second evaluation" in design_text()


@pytest.mark.parametrize(
    "change",
    [
        lambda w: check(w, id=None),
        lambda w: check(w, id="2"),
        lambda w: check(w, id=True),
        lambda w: check(w, id=0),
        lambda w: check(w, id=2.0),
        lambda w: check(w, **{"id": [2]}),
        lambda w: workflow_run(w, run_attempt=None),
        lambda w: workflow_run(w, run_attempt="1"),
        lambda w: workflow_run(w, run_attempt=True),
        lambda w: workflow_run(w, run_attempt=0),
        lambda w: workflow_run(w, run_attempt=1.0),
        lambda w: w.api[GROUP_RUN_PATH].pop("run_attempt"),
        lambda w: w.api[GROUP_CHECK_LISTING]["check_runs"][0].pop("id"),
    ],
)
def test_a_group_check_whose_identity_cannot_be_compared_is_never_evaluated(change):
    # From the first read on: a check without a numeric id, or a run without
    # a numeric attempt, would compare equal to itself at the second read and
    # say nothing. It is foreign: an error, no evaluation.
    world = queued()
    change(world)
    report, code = judge_group(world)
    assert code == 1 and report["verdict"] is None and report["evaluations"] == 0
    assert "not the merge group run" in report["error"]
    assert report["group_ci_rechecks"] == 0


def test_a_group_whose_tree_is_the_head_of_its_last_member_judges_that_member_only():
    # This test records a limit; it is not a guard (coordination note of
    # 2026-11-22, reproduced on the writer stack). Pull request 42 already
    # contains an earlier queued pull request 41. The group for 42 has the
    # tree of 42's head on the judged base: it passes, and no read names 41.
    world = queued()
    report, code = judge_group(world)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert not any("/pulls/41" in path for path in world.calls)
    doc = " ".join(guard.evaluate_merge_group.__doc__.split())
    assert "not its membership" in doc

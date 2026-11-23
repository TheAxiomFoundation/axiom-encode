"""The enforcing runner: the request cap of one run, sized for a batch of 10.

``MAX_ENFORCE_REQUESTS`` bounds a runaway run. It is 450 so that the run of
a later batch of 10 v6 units fits (issue #2301). The batch evaluator is not
on this branch: draft #2309 measures its costliest evaluation as
``20 + 10u`` reads for ``u`` units. These tests therefore do two things.

1. They measure what THIS branch's evaluator and runner cost, with the
   costliest single unit (the fixture of the other files is not that one).
2. They drive the real runner and the real evaluator through runs whose
   evaluation costs exactly what #2309 measures for 10 units (120 reads for
   the pull request, 124 for its merge group). The evaluation is made that
   large by a means this branch has: further green checks with the required
   name on the pull request head, two reads each. No evaluator is replaced.

API payloads are fixtures: the ``World`` fake of ``test_content_merge_guard``
and the helpers of ``test_merge_guard_runner``. No test makes a network call,
sleeps or reads a clock.
"""

from __future__ import annotations

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import merge_guard_runner as runner
from tests.test_content_merge_guard import (
    B2,
    C1,
    CHECKS,
    CI_RUN,
    MAX,
    PAVEL,
    REPO,
    ROOT,
    WEB_FLOW,
    B,
    H,
    build,
    commit,
    oid,
    review,
)
from tests.test_merge_guard_runner import (
    REPOSITORY_ROOT,
    SECOND_GROUP_RUN_PATH,
    Time,
    judge,
    judge_group,
    queued,
    reason_of,
    two_group_checks,
)

CAP = 450
EARLIER_CAP = 350
POLLS = 101
APPROVED = oid("approved-before-the-head")
HEAD_CHECKS = f"{ROOT}/commits/{H}{CHECKS}"


# What draft #2309 measures for a batch of ``units`` v6 units (its
# ``test_the_worst_case_evaluation_costs_20_plus_10_requests_per_unit``), and
# what this branch's ``evaluate_merge_group`` adds to an evaluation.
def batch_evaluation(units: int) -> int:
    return 20 + 10 * units


MERGE_GROUP_READS = 4


def pull_request_run(evaluation: int) -> int:
    """Two whole evaluations and every poll of one full wait."""
    return 2 * evaluation + runner.MAX_CI_POLLS * runner.CI_POLL_READS


def merge_group_run(evaluation: int) -> int:
    """Every poll at its highest cost, one evaluation, the second read."""
    return (
        runner.MAX_MERGE_GROUP_POLLS * runner.MERGE_GROUP_POLL_READS
        + evaluation
        + MERGE_GROUP_READS
        + runner.GROUP_CI_RECHECK_READS
    )


def honest_merge_group_run(evaluation: int) -> int:
    """One listed check: 100 polls of one read, the last of two."""
    return (runner.MAX_MERGE_GROUP_POLLS - 1) + 2 + evaluation + MERGE_GROUP_READS + 2


# --- worlds --------------------------------------------------------------------


def single_unit(*, candidate_base_behind_main=True, checks=1):
    """One v6 unit at its costliest for this branch's evaluator.

    The approval is on a commit that is not the head (three reads: the
    ancestry, the approved tree, the fork tree). With
    ``candidate_base_behind_main`` the manifest's candidate base is an
    ancestor of ``main`` and not its tip (one read). The signing run is on
    an encoder commit behind the encoder's tip (the fixture has that
    already). ``checks`` green checks with the required name are on the
    head, each from its own ``pull_request`` run of the required workflow
    (one read of each run in each of the two passes).
    """

    def older_base(payload):
        payload["attestation_scope"]["candidate"]["base_sha"] = B2

    world = build(mutate=older_base if candidate_base_behind_main else None)
    if candidate_base_behind_main:
        world.ancestry(B2, B, B2)
    # A commit between the candidate and the head with the head's tree.
    world.tree(APPROVED, world.trees[H])
    world.publish(
        H,
        B,
        [
            commit(C1, PAVEL),
            commit(APPROVED, PAVEL, WEB_FLOW),
            commit(H, PAVEL, WEB_FLOW),
        ],
    )
    world.ancestry(B, APPROVED, B)
    world.reviews([review(11, MAX, APPROVED)])
    listing = world.api[HEAD_CHECKS]
    first = listing["check_runs"][0]
    workflow = world.api[f"{ROOT}/actions/runs/{CI_RUN}"]
    for index in range(1, checks):
        run_id = str(5000 + index)
        listing["check_runs"].append(
            first
            | {
                "id": 100 + index,
                "details_url": (
                    f"https://github.com/{REPO}/actions/runs/{run_id}/job/1"
                ),
            }
        )
        world.api[f"{ROOT}/actions/runs/{run_id}"] = workflow | {"id": int(run_id)}
    listing["total_count"] = checks
    return world


def sized(evaluation: int):
    """A single unit whose pull-request evaluation costs ``evaluation`` reads.

    The costliest single unit costs 29 (28 with the candidate base at the
    ``main`` tip); each further check adds 2.
    """
    behind = (evaluation - 29) % 2 == 0
    further, odd = divmod(evaluation - (29 if behind else 28), 2)
    assert odd == 0 and further >= 0
    return single_unit(candidate_base_behind_main=behind, checks=1 + further)


def evaluation_reads(world) -> int:
    before = len(world.calls)
    verdict = world.evaluate()
    assert (verdict.verdict, verdict.reason) == (guard.PASS, guard.R_PASS), verdict
    return len(world.calls) - before


def every_poll_at_three_reads(world):
    """The costliest wait: two listed group checks, the second run pending
    until the last poll the wait allows."""
    world = two_group_checks(queued(world))
    done = world.api[SECOND_GROUP_RUN_PATH]
    pending = done | {"status": "in_progress", "conclusion": None}
    reads = []

    def run():
        reads.append(1)
        return pending if len(reads) < runner.MAX_MERGE_GROUP_POLLS else done

    world.api[SECOND_GROUP_RUN_PATH] = run
    return world


def pending_for_a_full_wait(world):
    """The required check on the head is pending at the first evaluation and
    at 100 polls, and as ``world`` has it at the last poll."""
    settled = world.api[HEAD_CHECKS]
    first = settled["check_runs"][0]
    pending = {
        "total_count": 1,
        "check_runs": [first | {"status": "in_progress", "conclusion": None}],
    }
    reads = []

    def read():
        reads.append(1)
        return pending if len(reads) <= runner.MAX_CI_POLLS else settled

    world.api[HEAD_CHECKS] = read
    return world


# --- what this branch costs ----------------------------------------------------


def test_the_costliest_single_unit_costs_29_reads_and_33_in_a_merge_group():
    # The fixture of the other files costs 25 and 29. It is approved on its
    # head (no read) and its candidate base is the ``main`` tip (no read).
    assert evaluation_reads(build()) == 25
    assert evaluation_reads(single_unit(candidate_base_behind_main=False)) == 28
    assert evaluation_reads(single_unit()) == 29
    # One poll of two reads, the evaluation, the second read of two.
    report, code = judge_group(queued(single_unit()))
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["requests"] == 2 + (29 + MERGE_GROUP_READS) + 2 == 37
    # #2309 adds one read for each unit (the tree of its candidate base), so
    # its figure for one unit is this one plus 1.
    assert batch_evaluation(1) == 29 + 1


def test_each_further_check_with_the_required_name_costs_two_reads():
    # Not bounded by the evaluator below one page of 100 checks: each listed
    # check is one more workflow run to read in each of the two passes.
    assert evaluation_reads(single_unit(checks=2)) == 29 + 2
    assert evaluation_reads(single_unit(checks=11)) == 29 + 20
    for evaluation in (120, 124, 144, 145):
        assert evaluation_reads(sized(evaluation)) == evaluation


def test_the_whole_runs_of_the_costliest_single_unit_fit_under_the_cap():
    world = every_poll_at_three_reads(single_unit())
    report, code = judge_group(world, Time())
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["ci_polls"] == POLLS
    assert report["requests"] == 3 * 101 + 33 + 3 == 339 == merge_group_run(29)
    assert report["requests"] <= EARLIER_CAP <= runner.MAX_ENFORCE_REQUESTS

    world = pending_for_a_full_wait(single_unit())
    report, code = judge(world, time=Time())
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert (report["evaluations"], report["ci_polls"]) == (2, POLLS)
    # The first evaluation stops at the listing of the required check: every
    # read of the first pass but the workflow run, none of the second pass.
    assert report["requests"] == 20 + 101 + 29 == 150 <= pull_request_run(29) == 159


# --- a batch of 10 -------------------------------------------------------------


def test_the_cap_and_the_arithmetic_of_the_runs():
    assert runner.MAX_ENFORCE_REQUESTS == CAP == 450
    assert (runner.MAX_CI_POLLS, runner.MAX_MERGE_GROUP_POLLS) == (POLLS, POLLS)
    assert (runner.CI_POLL_READS, runner.MERGE_GROUP_POLL_READS) == (1, 3)
    assert runner.GROUP_CI_RECHECK_READS == 3
    # units: evaluation, pull-request run, merge group at three reads a
    # poll, merge group with one listed check.
    table = {
        units: (
            batch_evaluation(units),
            pull_request_run(batch_evaluation(units)),
            merge_group_run(batch_evaluation(units)),
            honest_merge_group_run(batch_evaluation(units)),
        )
        for units in (1, 2, 10)
    }
    assert table == {
        1: (30, 161, 340, 138),
        2: (40, 181, 350, 148),
        10: (120, 341, 430, 228),
    }
    # The binding run is the merge group at three reads a poll. The earlier
    # cap covered 2 units and no more; this one covers 10 with 20 to spare
    # and, by the same row, 12 and no more.
    assert merge_group_run(batch_evaluation(2)) == EARLIER_CAP
    assert merge_group_run(batch_evaluation(3)) > EARLIER_CAP
    assert CAP - merge_group_run(batch_evaluation(10)) == 20
    assert merge_group_run(batch_evaluation(12)) == 450 == CAP
    assert merge_group_run(batch_evaluation(13)) > CAP
    assert pull_request_run(batch_evaluation(15)) == 441 <= CAP
    assert pull_request_run(batch_evaluation(16)) > CAP
    # An earlier-lane train: ``11 + 3u`` reads, ``15 + 3u`` in a merge group.
    assert 2 * (11 + 3 * 54) + 101 == 447 <= CAP < 2 * (11 + 3 * 55) + 101
    assert 3 * POLLS + (15 + 3 * 43) + 3 == 450 <= CAP
    assert 3 * POLLS + (15 + 3 * 44) + 3 > CAP
    assert 100 + 2 + (15 + 3 * 110) + 2 == 449 <= CAP < 100 + 2 + (15 + 3 * 111) + 2


def test_one_hour_of_one_run_and_the_documented_limit_of_the_job_token():
    limit = 1000  # "1,000 requests per hour per repository" (GitHub docs)
    polls_an_hour = 3600 // runner.CI_POLL_SECONDS
    assert polls_an_hour == 3600 // runner.MERGE_GROUP_POLL_SECONDS == 20
    pull_request_hour = polls_an_hour * runner.CI_POLL_READS + 2 * batch_evaluation(10)
    merge_group_hour = (
        polls_an_hour * runner.MERGE_GROUP_POLL_READS
        + batch_evaluation(10)
        + MERGE_GROUP_READS
        + runner.GROUP_CI_RECHECK_READS
    )
    assert (pull_request_hour, merge_group_hour) == (260, 187)
    # Three pull-request runs of a batch of 10 in their costliest hour fit
    # the limit; a fourth does not. Of single units: 12 (80 each).
    assert 3 * pull_request_hour == 780 <= limit < 4 * pull_request_hour == 1040
    assert 12 * 80 <= limit < 13 * 80
    # The cap is for a run, not for an hour. Two runs that both reach it in
    # one hour fit the limit and three do not; so it was at the earlier cap.
    assert 2 * CAP == 900 <= limit < 3 * CAP
    assert 2 * EARLIER_CAP <= limit < 3 * EARLIER_CAP


def test_the_merge_group_run_of_a_ten_unit_evaluation_fits_under_the_cap():
    # The binding run. Every poll of a full wait costs three reads; then an
    # evaluation of 124 reads (#2309: 20 + 10 x 10, and 4 for the group);
    # then the second read of the group's check, three reads.
    world = every_poll_at_three_reads(sized(batch_evaluation(10)))
    time = Time()
    report, code = judge_group(world, time)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert "error" not in report
    assert (report["evaluations"], report["group_ci_rechecks"]) == (1, 1)
    assert report["ci_polls"] == POLLS
    assert report["waited_seconds"] == runner.MERGE_GROUP_WAIT_SECONDS
    assert report["requests"] == 3 * 101 + (24 + 10 * 10) + 3 == 430
    assert report["requests"] == merge_group_run(batch_evaluation(10))
    assert EARLIER_CAP < report["requests"] <= runner.MAX_ENFORCE_REQUESTS


def test_the_pull_request_run_of_a_ten_unit_evaluation_fits_under_the_cap():
    world = pending_for_a_full_wait(sized(batch_evaluation(10)))
    report, code = judge(world, time=Time())
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert "error" not in report
    assert (report["evaluations"], report["ci_polls"]) == (2, POLLS)
    assert report["waited_seconds"] == runner.CI_WAIT_SECONDS
    # The first evaluation stops at the listing of the required check (19
    # reads here: ``sized(120)`` has its candidate base at the ``main`` tip).
    assert report["requests"] == 19 + 101 + 120 == 240
    assert report["requests"] <= pull_request_run(batch_evaluation(10)) == 341
    assert 341 <= runner.MAX_ENFORCE_REQUESTS


def test_the_run_that_needs_the_whole_cap_passes_and_one_request_more_fails():
    # 20 requests above the binding run of 10 units: exactly the cap.
    world = every_poll_at_three_reads(sized(batch_evaluation(10) + 20))
    report, code = judge_group(world, Time())
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["requests"] == 430 + 20 == runner.MAX_ENFORCE_REQUESTS
    assert "error" not in report

    # One request more. The run needs 451: the last read of the second read
    # of the group's check is refused before it is sent. The evaluation had
    # passed; the job fails, with the reason every capped run has.
    world = every_poll_at_three_reads(sized(batch_evaluation(10) + 21))
    report, code = judge_group(world, Time())
    assert code == 1
    assert report["error"] == "the run reached its request cap"
    assert reason_of(report) == ("pass", guard.R_PASS)
    assert report["requests"] == len(world.calls) == runner.MAX_ENFORCE_REQUESTS


@pytest.mark.parametrize("above", [4, 5, 60])
def test_a_cap_reached_inside_the_evaluation_is_a_failed_read(above):
    # 4 requests above the cap or more: the refused read is one of the
    # evaluation's own. The verdict is ``fail`` with the incomplete-read
    # mark, the run is an error, and nothing is read past the cap.
    world = every_poll_at_three_reads(sized(batch_evaluation(10) + 20 + above))
    report, code = judge_group(world, Time())
    assert code == 1 and report["verdict"]["verdict"] == "fail"
    assert report["verdict"]["evidence"][guard.EVIDENCE_INCOMPLETE_READ] is True
    assert report["error"] == "an evaluation could not read GitHub"
    assert report["group_ci_rechecks"] == 0
    assert report["requests"] == len(world.calls) == runner.MAX_ENFORCE_REQUESTS


# --- the cap on the pull-request path (review round 1, P2-1) --------------------
#
# The literal boundary 449/450/451 cannot be reached on the pull-request path
# with these fixtures: one listing holds at most 100 checks, so the largest
# evaluation is 29 + 2 x 99 = 227 reads and the largest run is
# 20 + 101 + 227 = 348 (driven below). So the tests of this path set the cap
# (``monkeypatch``) around the need of one run. That run is the 10-unit one:
# a first evaluation of 19 reads that stops at the pending check, 101 polls
# of one read, and a second, whole evaluation of 120 reads: 240 requests.
# What they isolate is that ONE budget covers the first evaluation, every
# poll and the second evaluation of a pull-request run.

PR_FIRST, PR_POLLS, PR_SECOND = 19, 101, 120
PR_NEED = PR_FIRST + PR_POLLS + PR_SECOND


def pull_request_run_under(monkeypatch, cap):
    monkeypatch.setattr(runner, "MAX_ENFORCE_REQUESTS", cap)
    world = pending_for_a_full_wait(sized(batch_evaluation(10)))
    report, code = judge(world, time=Time())
    # Nothing is sent past the cap, whatever the outcome.
    assert report["requests"] == len(world.calls) <= cap
    return report, code


def test_the_largest_pull_request_run_of_these_fixtures_is_below_the_cap():
    world = sized(29 + 2 * 99)
    assert len(world.api[HEAD_CHECKS]["check_runs"]) == 100  # one full listing
    world = pending_for_a_full_wait(world)
    report, code = judge(world, time=Time())
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert report["requests"] == 20 + 101 + 227 == 348 < runner.MAX_ENFORCE_REQUESTS


def test_a_pull_request_run_passes_at_its_exact_need_and_fails_one_below(
    monkeypatch,
):
    assert PR_NEED == 240
    report, code = pull_request_run_under(monkeypatch, PR_NEED)
    assert (reason_of(report), code) == (("pass", guard.R_PASS), 0)
    assert "error" not in report
    assert (report["requests"], report["evaluations"]) == (PR_NEED, 2)
    assert report["ci_polls"] == PR_POLLS

    # One request less: the last read of the second evaluation is refused.
    report, code = pull_request_run_under(monkeypatch, PR_NEED - 1)
    assert code == 1
    assert report["requests"] == PR_NEED - 1
    assert report["verdict"]["verdict"] == "fail"
    assert report["verdict"]["evidence"][guard.EVIDENCE_INCOMPLETE_READ] is True
    assert report["error"] == "an evaluation could not read GitHub"


@pytest.mark.parametrize(
    "cap",
    [
        PR_FIRST + PR_POLLS,  # the first read of the second evaluation
        PR_FIRST + PR_POLLS + 1,
        PR_SECOND,  # the need of the second evaluation alone
        PR_SECOND + 1,
        PR_SECOND + PR_POLLS,  # the second evaluation and the polls alone
        200,
        PR_NEED - PR_FIRST,  # everything but the first evaluation
        PR_NEED - 2,
    ],
)
def test_the_second_evaluation_of_a_pull_request_run_has_no_budget_of_its_own(
    monkeypatch, cap
):
    # A cap between what the run used before its second evaluation (120) and
    # its whole need (240). The second evaluation alone needs 120, so it
    # would fit a budget that started again; it must not.
    assert PR_FIRST + PR_POLLS <= cap < PR_NEED and cap >= PR_SECOND
    report, code = pull_request_run_under(monkeypatch, cap)
    assert code == 1
    assert report["requests"] == cap
    assert (report["evaluations"], report["ci_polls"]) == (2, PR_POLLS)
    assert report["verdict"]["verdict"] == "fail"
    assert report["verdict"]["evidence"][guard.EVIDENCE_INCOMPLETE_READ] is True
    assert report["error"] == "an evaluation could not read GitHub"


@pytest.mark.parametrize("polls_sent", [0, 1, 50, 100])
def test_each_poll_of_a_pull_request_wait_uses_the_budget(monkeypatch, polls_sent):
    # The cap is reached inside the wait: the first evaluation and exactly
    # ``polls_sent`` polls are sent, the next poll is refused. The wait ends
    # with the cap's own error, and no second evaluation is made.
    cap = PR_FIRST + polls_sent
    report, code = pull_request_run_under(monkeypatch, cap)
    assert code == 1
    assert report["requests"] == cap
    assert report["evaluations"] == 1
    assert reason_of(report) == ("fail", guard.R_CI)
    assert report["error"] == (
        "waiting for the required check failed: the run reached its request cap"
    )


def test_a_cap_inside_the_first_evaluation_of_a_pull_request_run_fails_it(
    monkeypatch,
):
    report, code = pull_request_run_under(monkeypatch, PR_FIRST - 1)
    assert code == 1
    assert (report["requests"], report["evaluations"]) == (PR_FIRST - 1, 1)
    assert report["verdict"]["evidence"][guard.EVIDENCE_INCOMPLETE_READ] is True
    assert report["error"] == "an evaluation could not read GitHub"


def test_the_cap_admits_no_batch_by_itself():
    # The second lock stays: the success set holds no batch reason, and this
    # branch's evaluator refuses a second v6 unit.
    assert runner.SUCCESS_VERDICTS == frozenset(
        {
            (guard.PASS, guard.R_PASS),
            (guard.NOT_APPLICABLE, guard.R_NO_GUARDED_CONTENT),
            (guard.NOT_APPLICABLE, guard.R_EARLIER_LANE),
        }
    )
    assert not any(
        "multi_unit" in reason for _verdict, reason in runner.SUCCESS_VERDICTS
    )
    assert not hasattr(guard, "MAX_V6_UNITS")
    assert guard.R_MULTIPLE_V6 == "multiple_v6_manifests"


# --- the design says what the code does (doc pins) ----------------------------

DESIGN_BUDGET = [
    "**Request budget for a batch (cap 450).**",
    "`MAX_ENFORCE_REQUESTS` is 450 (it was 350, and 250 before that,",
    "| 1 | 30 | 2 x 30 + 101 = 161 | 3 x 101 + 34 + 3 = 340 | 100 + 2 + 34 + 2 = 138 |",
    "| 2 | 40 | 2 x 40 + 101 = 181 | 3 x 101 + 44 + 3 = 350 | 100 + 2 + 44 + 2 = 148 |",
    "| 10 | 120 | 2 x 120 + 101 = 341 | 3 x 101 + 124 + 3 = **430** | "
    "100 + 2 + 124 + 2 = 228 |",
    "The cap of 450 leaves 20 requests over the binding run of 10 units",
    "On this branch one v6 unit costs at most 29 reads for a pull request and "
    "33 for its merge group",
    "each further check with the required name on the pull request head costs 2 reads",
    "The cap admits no batch by itself.",
    "20 + 2 x 120 = 260",
    "60 + 124 + 3 = 187",
    "three such runs fit one hour (780) and a fourth does not (1,040)",
    "two runs that both reach the cap in one hour use 900 of the 1,000, and "
    "three exceed it",
    "A read that the limit refuses is a failed read: the run fails closed.",
    "a mean of about 10 seconds for each request",
    "2 x (11 + 3u) + 101 <= 450",
    "3 x 101 + (15 + 3u) + 3 <= 450 admits about 43",
    "100 + 2 + (15 + 3u) + 2 <= 450 admits about 110",
]


def design_text() -> str:
    return " ".join(
        (REPOSITORY_ROOT / "docs" / "content-bound-merge-guard.md")
        .read_text(encoding="utf-8")
        .split()
    )


def budget_problems(text: str) -> list[str]:
    problems = [
        f"lost or doubled: {sentence}"
        for sentence in DESIGN_BUDGET
        if text.count(sentence) != 1
    ]
    problems += [
        f"stale: {stale}"
        for stale in ("<= 350", "cap of 350", "`MAX_ENFORCE_REQUESTS` is 350")
        if stale in text
    ]
    return problems


def test_the_design_states_the_budget_these_tests_measure():
    assert budget_problems(design_text()) == []


@pytest.mark.parametrize("sentence", DESIGN_BUDGET)
def test_the_budget_pin_reports_each_lost_and_each_doubled_sentence(sentence):
    text = design_text()
    assert budget_problems(text.replace(sentence, "")) == [
        f"lost or doubled: {sentence}"
    ]
    assert f"lost or doubled: {sentence}" in budget_problems(text + " " + sentence)


KEY_PROTECTION_BUDGET = [
    "the cap of 450 requests for a run admits about 54 units",
    "whose costliest hour is 260 requests: three such runs fit the limit and a "
    "fourth does not",
]


def test_the_key_protection_document_states_the_same_cap():
    text = " ".join(
        (REPOSITORY_ROOT / "docs" / "merge-guard-key-protection.md")
        .read_text(encoding="utf-8")
        .split()
    )
    assert [text.count(sentence) for sentence in KEY_PROTECTION_BUDGET] == [1, 1]
    assert "cap of 350" not in text


def test_the_budget_pin_reports_the_figures_of_the_earlier_cap():
    text = design_text()
    assert budget_problems(text + " 2 x (11 + 3u) + 101 <= 350") == ["stale: <= 350"]
    assert budget_problems(text + " the cap of 350 admits") == ["stale: cap of 350"]

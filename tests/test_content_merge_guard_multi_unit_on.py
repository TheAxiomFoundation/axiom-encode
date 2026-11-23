"""The single-unit suite of the merge guard, run again with the multi-unit
switch on.

``tests/test_content_merge_guard.py`` is the specification of the single-unit
v6 lane and of every other lane. That file is unchanged and passes with the
committed switch (off). This module imports all of its tests and runs them
with ``reviewed_pr_admission.MULTI_UNIT_V6_ENABLED`` patched to ``True``, so
each of them must pass with the switch on too.

It is also a differential test. Every call of ``guard.evaluate`` that one of
those tests makes is evaluated twice over the same GitHub responses: with the
switch on (the result the test sees) and, replayed read by read, with the
switch off. The two verdicts must be equal as whole dicts (verdict, reason,
detail, evidence), and the switch-off evaluation must make the reads of the
switch-on one in the same order, with one kind of read left out: the tree of
a candidate's base commit, which only the candidate rule of the multi-unit
form reads. Designed differences, counted apart:

- a pull request with two or more v6 manifests: off it is
  ``multiple_v6_manifests``, on it is judged as a batch; in that suite it
  never passes;
- a refusal by the candidate rule of the multi-unit form
  (``v6_candidate_holds_unjudged_change``), which the switch-off evaluation
  does not apply: none occurs in that suite, and the test below requires
  that.
"""

from __future__ import annotations

import copy

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import reviewed_pr_admission as admission
from tests import test_content_merge_guard as single
from tests.test_content_merge_guard import *  # noqa: F403 (the suite itself)

COUNTS = {
    "identical": 0,
    "batch": 0,
    "candidate_rule": 0,
    "extra_read_failed": 0,
    "extra_reads": 0,
}
_evaluate_once = guard.evaluate


class _Replay:
    """Serve the recorded reads of one evaluation, in order, and nothing else.

    A recorded read of a Git tree that the replaying evaluation does not ask
    for is skipped and counted: the switch-on evaluation reads the tree of a
    candidate's base commit, the switch-off one does not.
    """

    def __init__(self, recorded):
        self.recorded, self.position, self.skipped = recorded, 0, []

    def __call__(self, path):
        while (
            self.position < len(self.recorded)
            and self.recorded[self.position][0] != path
            and "/git/trees/" in self.recorded[self.position][0]
        ):
            self.skipped.append(self.recorded[self.position][0])
            self.position += 1
        if self.position >= len(self.recorded):
            raise AssertionError("the switch-off evaluation made a further read")
        expected, kind, value = self.recorded[self.position]
        self.position += 1
        if path != expected:
            raise AssertionError(f"read {path} where the other read {expected}")
        if kind == "raise":
            raise value
        return copy.deepcopy(value)

    def finish(self):
        while self.position < len(self.recorded) and (
            "/git/trees/" in self.recorded[self.position][0]
        ):
            self.skipped.append(self.recorded[self.position][0])
            self.position += 1
        return self.position == len(self.recorded)


def _both_states(repository, pr, *, fetch=guard.github_json):
    recorded = []

    def recording(path):
        try:
            value = fetch(path)
        except BaseException as exc:
            recorded.append((path, "raise", exc))
            raise
        recorded.append((path, "return", copy.deepcopy(value)))
        return value

    assert admission.MULTI_UNIT_V6_ENABLED is True
    on = _evaluate_once(repository, pr, fetch=recording)
    replay = _Replay(recorded)
    admission.MULTI_UNIT_V6_ENABLED = False
    try:
        off = _evaluate_once(repository, pr, fetch=replay)
    finally:
        admission.MULTI_UNIT_V6_ENABLED = True
    if off.reason == guard.R_MULTIPLE_V6:
        # The designed difference. In this suite such a head never passes.
        assert on.verdict == guard.FAIL, on
        assert on.reason != guard.R_MULTIPLE_V6
        COUNTS["batch"] += 1
    elif on.reason == guard.R_CANDIDATE_UNJUDGED:
        COUNTS["candidate_rule"] += 1
    else:
        if on.as_dict() != off.as_dict():
            # The one other way the switch-on evaluation may differ: its extra
            # read (a candidate base tree) failed. It then refuses where the
            # switch-off evaluation, which does not make that read, goes on.
            assert on.verdict == guard.FAIL and replay.skipped, (on, off)
            COUNTS["extra_read_failed"] += 1
        else:
            assert replay.finish(), "the switch-off evaluation read less"
            COUNTS["identical"] += 1
            COUNTS["extra_reads"] += len(replay.skipped)
    return on


@pytest.fixture(autouse=True)
def multi_unit_switch_on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    monkeypatch.setattr(guard, "evaluate", _both_states)


# The two tests of the single-unit suite that the switch changes by design.
# Each is replaced here by its on-state statement.


def test_case_20_two_v6_manifests():  # noqa: F811 (replaces the imported test)
    """Off: ``multiple_v6_manifests``. On: the second manifest is the first
    one's bytes at another position, so it is not at the canonical path."""
    world = single.build(
        head_extra={single.M.replace("block-7", "block-8"): single.manifest_bytes()}
    )
    single.fails(world, guard.R_MANIFEST_PATH)


def test_f5_candidate_base_must_be_an_ancestor_of_the_base():  # noqa: F811
    """As in the single-unit suite, with the one thing the switch-on
    evaluation reads in addition: the tree of the candidate's base commit,
    which that suite's fixture does not hold."""

    def other(payload):
        payload["attestation_scope"]["candidate"]["base_sha"] = single.B2

    world = single.build(mutate=other)
    world.tree(single.B2, single.BASE_FILES)
    for status, fork in (
        ("diverged", single.C1),
        ("behind", single.B2),
        ("ahead", single.C1),
    ):
        world.ancestry(single.B2, single.B, fork, status=status)
        single.fails(world, guard.R_MANIFEST_CANDIDATE)
    world.ancestry(single.B2, single.B, single.B2)
    single.expect(world, guard.PASS, guard.R_PASS)


def test_zz_the_differential_covered_the_suite():
    """Runs last in this module (pytest keeps definition order)."""
    print(COUNTS)
    assert COUNTS["identical"] >= 400, COUNTS
    assert COUNTS["batch"] >= 1, COUNTS
    # No evaluation of that suite is refused by the candidate rule, and none
    # differs because the extra read failed.
    assert COUNTS["candidate_rule"] == 0 and COUNTS["extra_read_failed"] == 0, COUNTS
    assert COUNTS["extra_reads"] >= 1, COUNTS

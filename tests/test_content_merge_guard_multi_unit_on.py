"""The single-unit suite of the merge guard, run again with the multi-unit
switch on: a batch of one is a single unit.

``tests/test_content_merge_guard.py`` is the specification of the single-unit
v6 lane and of every other lane. This module imports all of its tests and
runs them with ``reviewed_pr_admission.MULTI_UNIT_V6_ENABLED`` patched to
``True``, so each of them must pass unchanged with the switch on.

It is also a differential test. Every call of ``guard.evaluate`` that one of
those tests makes is evaluated twice over the same GitHub responses: with the
switch on (the result the test sees) and, replayed read by read, with the
switch off. The two verdicts must be equal as whole dicts (verdict, reason,
detail, evidence) and the two evaluations must make the same reads in the
same order. The one designed difference is a pull request with two or more
v6 manifests: off it is ``multiple_v6_manifests``, on it is judged as a
batch; those calls are counted apart and never pass in that suite.
"""

from __future__ import annotations

import copy

import pytest

from axiom_encode import content_merge_guard as guard
from axiom_encode import reviewed_pr_admission as admission
from tests import test_content_merge_guard as single
from tests.test_content_merge_guard import *  # noqa: F403 (the suite itself)

COUNTS = {"identical": 0, "batch": 0}
_evaluate_once = guard.evaluate


class _Replay:
    """Serve the recorded reads of one evaluation, in order, and nothing else."""

    def __init__(self, recorded):
        self.recorded, self.position = recorded, 0

    def __call__(self, path):
        if self.position >= len(self.recorded):
            raise AssertionError("the switch-off evaluation made a further read")
        expected, kind, value = self.recorded[self.position]
        self.position += 1
        if path != expected:
            raise AssertionError(f"read {path} where the other read {expected}")
        if kind == "raise":
            raise value
        return copy.deepcopy(value)


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
    else:
        assert on.as_dict() == off.as_dict()
        assert replay.position == len(recorded), "the switch-off evaluation read less"
        COUNTS["identical"] += 1
    return on


@pytest.fixture(autouse=True)
def multi_unit_switch_on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_V6_ENABLED", True)
    monkeypatch.setattr(guard, "evaluate", _both_states)


# The one test of the single-unit suite whose expectation the switch changes
# by design. It is replaced here by its on-state statement.


def test_case_20_two_v6_manifests():  # noqa: F811 (replaces the imported test)
    """Off: ``multiple_v6_manifests``. On: the second manifest is the first
    one's bytes at another position, so it is not at the canonical path."""
    world = single.build(
        head_extra={single.M.replace("block-7", "block-8"): single.manifest_bytes()}
    )
    single.fails(world, guard.R_MANIFEST_PATH)


def test_zz_the_differential_covered_the_suite():
    """Runs last in this module (pytest keeps definition order)."""
    print(COUNTS)
    assert COUNTS["identical"] >= 400, COUNTS
    assert COUNTS["batch"] >= 1, COUNTS

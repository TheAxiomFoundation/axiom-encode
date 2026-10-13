"""Throwaway: fails on a workflow run's first attempt, passes when the job is re-run."""

import os


def test_fails_on_the_first_attempt_only():
    assert os.environ.get("GITHUB_RUN_ATTEMPT", "1") != "1"

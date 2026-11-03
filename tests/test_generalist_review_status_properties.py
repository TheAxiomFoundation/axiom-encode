"""Reviewer status never reads as approval unless a scored review passed."""

from types import SimpleNamespace

from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.harness.encoding_db import ReviewResult, ReviewResults
from axiom_encode.harness.evals import generalist_review_snapshot

metrics = st.builds(
    SimpleNamespace,
    generalist_review_score=st.none() | st.floats(0, 10, allow_nan=False),
    generalist_review_pass=st.none() | st.booleans(),
    generalist_review_issues=st.lists(st.text(max_size=8), max_size=2),
    generalist_review_prompt_sha256=st.none() | st.just("a" * 64),
    generalist_review_status=st.none()
    | st.sampled_from(["passed", "failed", "skipped", "unavailable"]),
    generalist_review_skip_reason=st.none() | st.sampled_from(["user_requested"]),
)


@given(metrics)
def test_only_a_scored_passing_review_is_reported_as_passed(recorded):
    snapshot = generalist_review_snapshot(recorded)

    scored_pass = (
        recorded.generalist_review_score is not None
        and recorded.generalist_review_pass is True
    )
    assert (snapshot["status"] == "passed") is scored_pass
    assert (snapshot["passed"] is True) is scored_pass
    if recorded.generalist_review_score is None:
        assert snapshot["status"] in {"skipped", "unavailable", "not_recorded"}
    if snapshot["status"] != "skipped":
        assert snapshot["skip_reason"] is None


@given(recorded=metrics, other_reviews_pass=st.booleans())
def test_aggregate_review_never_passes_on_an_unpassed_generalist(
    recorded, other_reviews_pass
):
    snapshot = generalist_review_snapshot(recorded)
    results = ReviewResults(
        reviews=[
            ReviewResult(reviewer="formula", passed=other_reviews_pass, items_checked=1)
        ],
        oracle_context={"generalist_review": snapshot},
    )

    assert results.passed is (snapshot["status"] == "passed" and other_reviews_pass)

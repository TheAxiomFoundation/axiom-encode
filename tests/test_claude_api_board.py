"""The board admits API contestants and excludes context infrastructure errors."""

import pytest

from axiom_encode.harness import eval_board
from tests.test_eval_board import CASE_IDENTITIES, _payload, _result, _write_payload


@pytest.mark.parametrize("overflow", [False, True])
def test_board_accepts_claude_api_and_does_not_score_context_overflow(
    tmp_path, overflow
):
    row = _result(
        "api",
        CASE_IDENTITIES[0],
        backend="claude-api",
        model="claude-opus-5-5",
        success=not overflow,
        failure_kind="error" if overflow else None,
        metrics=None if overflow else "default",
        error="Claude API context overflow (prompt_chars=1200000)"
        if overflow
        else None,
    )
    payload = _payload(
        [("api", "claude-api", "claude-opus-5-5")], [row], complete=False
    )
    path = _write_payload(tmp_path, "api.json", payload)
    board = eval_board.fold_eval_board([path], allow_partial=True)
    [stats] = board.runners
    assert stats.cases_run == (0 if overflow else 1)
    if overflow:
        assert stats.infrastructure_error_count == 1


@pytest.mark.parametrize("overflow", [False, True])
def test_board_reports_api_admission_not_applicable(tmp_path, overflow):
    from axiom_encode.harness.admission import not_applicable_admission_result

    row = _result(
        "api",
        CASE_IDENTITIES[0],
        backend="claude-api",
        model="claude-opus-5-5",
        success=not overflow,
        failure_kind="error" if overflow else None,
        metrics=None if overflow else "default",
        error="Claude API context overflow (prompt_chars=1200000)"
        if overflow
        else None,
    )
    row["admission_score"] = not_applicable_admission_result().to_dict()
    from tests.test_eval_board import _execution_identity

    execution = _execution_identity()
    execution["admission_score"] = {"enabled": True, "axiom_compose": None}
    payload = _payload(
        [("api", "claude-api", "claude-opus-5-5")],
        [row],
        complete=False,
        execution_identity=execution,
    )
    path = _write_payload(tmp_path, "api.json", payload)
    board = eval_board.fold_eval_board([path], allow_partial=True)
    [stats] = board.runners
    assert stats.admission_not_applicable_count == 1
    assert stats.admission_case_count == 0
    assert "N/A" in eval_board.render_eval_board_markdown(board)

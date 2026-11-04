"""Regressions from PR 1658's independent review; no network or optional SDK."""

import json

import pytest

from axiom_encode.judges import ScreenPolicy, Verdict, statutory_fidelity_screen
from axiom_encode.run_log import RunLogWriter
from tests import test_statutory_fidelity_screen as t


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in t._SCREEN_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "field",
    [
        "model",
        "model_key_only",
        "model_key_lowercase",
        "foreign_model_key",
        "probability_label",
        "invalid_probability_label",
        "choice",
    ],
)
def test_server_strings_never_record_credentials(monkeypatch, tmp_path, field):
    response = t.recorded_response(t.RECORDED_ORIGINAL)
    if field == "model":
        response.model = "jev-1 " + t.LEAKY_TEXT
    elif field == "model_key_only":
        response.model = "jev-" + t.SECRET
    elif field == "model_key_lowercase":
        response.model = "jev-" + t.SECRET.lower()
    elif field == "foreign_model_key":
        response.model = "gpt-" + t.SECRET
    elif field == "choice":
        response.answers["verdict"].choice = t.LEAKY_TEXT
    else:
        response.answers["verdict"].probabilities = {
            t.LEAKY_TEXT: "invalid" if field == "invalid_probability_label" else 0.5
        }
    event = t._screen(monkeypatch, response, policy=ScreenPolicy(mode="cascade"))
    writer = RunLogWriter("server-strings", log_dir=tmp_path)
    assert event.emit(writer) is not None
    serialized = "\n".join(
        [
            json.dumps(event.to_dict()),
            repr(event),
            (tmp_path / "server-strings.jsonl").read_text(),
        ]
    ).lower()
    for forbidden in (t.SECRET, "Authorization", "Bearer", "x-api-key"):
        assert forbidden.lower() not in serialized
    assert event.verdict == Verdict.ERROR
    assert event.extra["screen"]["cascade"]["request_referee"] is True
    assert t.validate_event_dict(event.to_dict()) == []


def test_unrecognized_choice_diagnostic_is_fixed_even_for_a_custom_client():
    call = t._call_from(t.RECORDED_ORIGINAL)
    call.answers["verdict"] = t.ChoiceAnswer(
        choice=t.LEAKY_TEXT, confidence=0.5, probabilities={"pass": 0.5, "flag": 0.5}
    )
    event = statutory_fidelity_screen.run(
        "provision", "rules: []", client=t.FakeSystemOneClient(call)
    )
    assert t.SECRET.lower() not in json.dumps(event.to_dict()).lower()
    assert "authorization" not in event.judge_error.message.lower()
    assert event.judge_error.type == "unrecognized_verdict"


@pytest.mark.parametrize(
    "probabilities",
    [
        {},
        {"unexpected": 0.5},
        {"pass": 1.0},
        {"pass": 0.5, "flag": 0.5, "unexpected": 0.0},
    ],
)
def test_choice_requires_exactly_the_requested_probability_labels(
    monkeypatch, probabilities
):
    response = t.recorded_response(t.RECORDED_ORIGINAL)
    response.answers["verdict"].probabilities = probabilities
    event = t._screen(monkeypatch, response, policy=ScreenPolicy(mode="cascade"))
    assert event.verdict == Verdict.ERROR
    assert event.judge_error.type == "schema_error"
    assert event.extra["screen"]["cascade"]["request_referee"] is True


@pytest.mark.parametrize(
    "confidence",
    [None, "high", float("nan"), float("inf"), float("-inf"), -0.1, 1.1, "missing"],
)
def test_choice_requires_a_finite_confidence_in_range(monkeypatch, confidence):
    response = t.recorded_response(t.RECORDED_ORIGINAL)
    response.answers["verdict"].confidence = confidence
    if confidence == "missing":
        del response.answers["verdict"].confidence
    event = t._screen(monkeypatch, response, policy=ScreenPolicy(mode="cascade"))
    assert event.verdict == Verdict.ERROR
    assert event.judge_error.type == "schema_error"
    assert event.extra["screen"]["cascade"]["request_referee"] is True


@pytest.mark.parametrize(
    ("probability", "threshold", "request_referee"),
    [
        (0.33334, 0.333333, True),
        (0.333333, 0.333333, True),
        (0.33336, 0.33337, False),
        (0.249999, 0.25, False),
        (0.00004, 0.000033, True),
    ],
)
def test_cascade_and_findings_use_unrounded_probabilities(
    monkeypatch, probability, threshold, request_referee
):
    row = {**t.RECORDED_ORIGINAL, "amount_mismatch": probability}
    policy = ScreenPolicy(mode="cascade", thresholds={"amount_mismatch": threshold})
    event = t._screen(monkeypatch, row, policy=policy)
    assert event.extra["screen"]["cascade"]["request_referee"] is request_referee
    assert bool(event.findings) is request_referee
    assert event.extra["screen"]["probabilities"]["amount_mismatch"] == probability
    if request_referee:
        assert event.findings[0].probability == probability
        # The canonical evidence is presentation; its existing format is preserved.
        assert event.to_dict()["findings"][0]["evidence"] == (
            f"probability={probability:.4f}"
        )

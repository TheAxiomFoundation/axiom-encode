"""Tests for the TypeSafe System One statutory-fidelity screen.

No network. System One responses are replayed from the 2026-09-17 pilot's
recorded answers for one pair (run ``939be8a7``,
``us-co:regulations/10-ccr-2506-1/4.803.1``): the clean original artifact and
the same artifact with one planted amount change (``15`` to ``19``). A fake
``typesafe_sdk`` module stands in for the SDK, so the real client code path
runs end to end.

Load-bearing invariants:

* a screen failure is a visible ``judge_error`` event, never a pass, and always
  requests the referee (fail closed);
* findings carry a kind and a probability and never a fabricated locator;
* the referee is skipped only in cascade mode and only below threshold;
* nothing from an SDK exception, the key, or a header can reach a run log.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

from axiom_encode.constants import (
    DEFAULT_JUDGE_SCREEN_MODE,
    DEFAULT_JUDGE_SCREEN_THRESHOLD,
)
from axiom_encode.judges import (
    JUDGE_STAGE,
    CascadeDecision,
    JudgeEvent,
    JudgeStage,
    ScreenPolicy,
    SystemOneCall,
    SystemOneClient,
    TokenCounts,
    Verdict,
    cli_commands,
    model_family,
    statutory_fidelity,
    statutory_fidelity_screen,
    truncate_provision,
    validate_event_dict,
)
from axiom_encode.judges.statutory_fidelity import FIDELITY_KINDS
from axiom_encode.judges.system_one import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulQuestion,
)
from axiom_encode.run_log import RunLogWriter, StageStatus, iter_events

SECRET = "ts-live-SECRET-0123456789abcdef"
LEAKY_TEXT = (
    f"POST https://api.typesafe.ai/v1/systemone failed; "
    f"Authorization: Bearer {SECRET}; x-api-key: {SECRET}"
)

# Recorded 2026-09-17 (jev-axiom-planted.json, run 939be8a7, kind "amount").
RECORDED_ORIGINAL = {
    "model": "jev-1.13.0",
    "in": 1099,
    "out": 128,
    "verdict": "pass",
    "p_flag": 0.32,
    "confidence": 0.36,
    "faithful": 0.88,
    "unrepresented_clause": 0.5,
    "untraceable_branch": 0.34,
    "boundary_direction": 0.15,
    "amount_mismatch": 0.06,
}
RECORDED_PLANTED_AMOUNT = {
    "model": "jev-1.13.0",
    "in": 1099,
    "out": 128,
    "verdict": "flag",
    "p_flag": 1.0,
    "confidence": 1.0,
    "faithful": 0.04,
    "unrepresented_clause": 0.56,
    "untraceable_branch": 0.52,
    "boundary_direction": 0.91,
    "amount_mismatch": 0.97,
}

_SCREEN_ENV = (
    "TYPESAFE_API_KEY",
    "AXIOM_GENERATOR_MODEL",
    "AXIOM_JUDGE_PROVISION_CHARS",
    "AXIOM_JUDGE_SCREEN_MODE",
    "AXIOM_JUDGE_SCREEN_MODEL",
    "AXIOM_JUDGE_SCREEN_THRESHOLD",
    "AXIOM_JUDGE_SCREEN_TIMEOUT_SECONDS",
    "AXIOM_JUDGE_SCREEN_MAX_RETRIES",
    *(f"AXIOM_JUDGE_SCREEN_THRESHOLD_{kind.upper()}" for kind in FIDELITY_KINDS),
)


@pytest.fixture(autouse=True)
def _clean_screen_environment(monkeypatch):
    for name in _SCREEN_ENV:
        monkeypatch.delenv(name, raising=False)


# -- fake SDK -------------------------------------------------------------


class FakeTypeSafeError(Exception):
    pass


class FakeTypeSafeAPIError(FakeTypeSafeError):
    pass


class TypeSafeRateLimitError(FakeTypeSafeAPIError):
    pass


class TypeSafeAuthenticationError(FakeTypeSafeAPIError):
    pass


class TypeSafeInternalServerError(FakeTypeSafeAPIError):
    pass


class TypeSafeAPIConnectionError(FakeTypeSafeError, ConnectionError):
    pass


class TypeSafeAPITimeoutError(TypeSafeAPIConnectionError, TimeoutError):
    pass


class _Usage:
    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _NoulAnswer:
    def __init__(self, noul):
        self.noul = noul


class _ChoiceAnswer:
    def __init__(self, choice, confidence, probabilities):
        self.choice = choice
        self.confidence = confidence
        self.probabilities = probabilities


class _Response:
    def __init__(self, model, usage, answers):
        self.model = model
        self.usage = usage
        self.answers = answers


def recorded_response(row: dict, **overrides) -> _Response:
    """Rebuild an SDK-shaped response from one recorded pilot row."""

    answers = {
        "verdict": _ChoiceAnswer(
            row["verdict"],
            row["confidence"],
            {"pass": round(1 - row["p_flag"], 4), "flag": row["p_flag"]},
        ),
        "faithful": _NoulAnswer(row["faithful"]),
    }
    for kind in FIDELITY_KINDS:
        answers[kind] = _NoulAnswer(row[kind])
    answers.update(overrides.pop("answers", {}))
    for name in overrides.pop("drop", ()):
        answers.pop(name)
    return _Response(
        overrides.pop("model", row["model"]),
        _Usage(row["in"], row["out"]),
        answers,
    )


def install_fake_typesafe(monkeypatch, responses, *, init_error=None):
    """Install a fake ``typesafe_sdk`` whose ``system_one`` replays ``responses``.

    Each item is a response object (returned) or an Exception (raised).
    Returns a dict recording what the client passed to the SDK.
    """

    seen: dict = {"closed": 0, "calls": []}
    queue = list(responses)
    mod = types.ModuleType("typesafe_sdk")

    class Noul:
        def __init__(self, *, instructions=None, criteria=None):
            self.instructions = instructions
            self.criteria = criteria

    class Choice:
        def __init__(self, *, criteria, instructions=None):
            self.instructions = instructions
            self.criteria = criteria

    class RetryPolicy:
        def __init__(self, max_retries=2):
            self.max_retries = max_retries

    class TypeSafeClient:
        def __init__(self, *, api_key=None, model=None, retry=None, timeout=None):
            if init_error is not None:
                raise init_error
            seen["init"] = {
                "api_key": api_key,
                "model": model,
                "max_retries": getattr(retry, "max_retries", None),
                "timeout": timeout,
            }

        def system_one(self, state, questions, **kwargs):
            seen["calls"].append({"state": state, "questions": questions})
            item = queue.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        def close(self):
            seen["closed"] += 1

    mod.Noul = Noul
    mod.Choice = Choice
    mod.RetryPolicy = RetryPolicy
    mod.TypeSafeClient = TypeSafeClient
    monkeypatch.setitem(sys.modules, "typesafe_sdk", mod)
    return seen


def _client(**kwargs) -> SystemOneClient:
    kwargs.setdefault("api_key", SECRET)
    kwargs.setdefault("generator_model", "gpt-5.6-terra")
    return SystemOneClient(**kwargs)


def _screen(monkeypatch, row_or_item, *, policy=None, **run_kwargs) -> JudgeEvent:
    item = (
        recorded_response(row_or_item) if isinstance(row_or_item, dict) else row_or_item
    )
    install_fake_typesafe(monkeypatch, [item])
    run_kwargs.setdefault("citation", "us-co:regulations/10-ccr-2506-1/4.803.1")
    return statutory_fidelity_screen.run(
        "provision text", "rules: []", client=_client(), policy=policy, **run_kwargs
    )


class FakeSystemOneClient:
    """A SystemOneClient stand-in that returns a preset SystemOneCall."""

    def __init__(self, call: SystemOneCall):
        self._call = call
        self.provision_chars = 24_000
        self.generator_model = "gpt-5.6-terra"
        self.seen: dict = {}

    def call(self, *, state, questions) -> SystemOneCall:
        self.seen = {"state": dict(state), "questions": dict(questions)}
        return self._call


def _call_from(row: dict) -> SystemOneCall:
    answers: dict = {
        "verdict": ChoiceAnswer(
            choice=row["verdict"],
            confidence=row["confidence"],
            probabilities={"pass": round(1 - row["p_flag"], 4), "flag": row["p_flag"]},
        ),
        "faithful": row["faithful"],
    }
    for kind in FIDELITY_KINDS:
        answers[kind] = float(row[kind])
    return SystemOneCall(
        answers=answers,
        model=row["model"],
        family="typesafe",
        tokens=TokenCounts(row["in"], row["out"]),
        latency_ms=190,
    )


# -- family classification ------------------------------------------------


def test_model_family_classifies_jev_as_typesafe():
    assert model_family("jev-1.13.0") == "typesafe"
    assert model_family("jev-latest") == "typesafe"
    assert model_family("JEV-2.0.0") == "typesafe"
    # Existing families are unchanged.
    assert model_family("claude-haiku-4-5-20251001") == "anthropic"
    assert model_family("gpt-5.6-terra") == "openai"
    assert model_family("mystery-model") == "unknown"


def test_screen_family_passes_the_cross_family_guard_and_is_recorded(monkeypatch):
    client = _client()
    assert client.cross_family_problem(None) is None
    assert client.cross_family_problem("jev-1.13.0") is None
    event = _screen(monkeypatch, RECORDED_ORIGINAL)
    attrs = event.to_dict()["attrs"]
    assert attrs["judge_model"] == "jev-1.13.0"
    assert attrs["judge_model_family"] == "typesafe"
    assert attrs["generator_model"] == "gpt-5.6-terra"


@pytest.mark.parametrize("generator", ["jev-1.13.0", "my-inhouse-wrapper"])
def test_generator_sharing_or_hiding_its_family_is_guarded(monkeypatch, generator):
    seen = install_fake_typesafe(monkeypatch, [recorded_response(RECORDED_ORIGINAL)])
    call = _client(generator_model=generator).call(
        state={"a": 1}, questions={"q": NoulQuestion(instructions="x")}
    )
    assert not call.ok
    assert call.error.type == "cross_family_guard"
    # Guarded before any request is made.
    assert seen["calls"] == []


@pytest.mark.parametrize("responding", ["mystery-1", "gpt-5.6-terra", "claude-opus-5"])
def test_responding_model_outside_the_typesafe_family_is_guarded(
    monkeypatch, responding
):
    install_fake_typesafe(
        monkeypatch, [recorded_response(RECORDED_ORIGINAL, model=responding)]
    )
    call = _client().call(
        state={"a": 1}, questions=statutory_fidelity_screen.build_questions()
    )
    assert not call.ok
    assert call.error.type == "cross_family_guard"
    assert call.answers is None
    # The spend still happened and stays visible for cost control.
    assert call.tokens.input == 1099
    assert call.model == responding


# -- client: recorded response mapping ------------------------------------


def test_client_maps_recorded_response_and_records_model_latency_usage(monkeypatch):
    seen = install_fake_typesafe(monkeypatch, [recorded_response(RECORDED_ORIGINAL)])
    questions = statutory_fidelity_screen.build_questions()
    call = _client(timeout=12, max_retries=1).call(
        state={"citation": "c"}, questions=questions
    )
    assert call.ok
    assert call.model == "jev-1.13.0"
    assert call.family == "typesafe"
    assert call.tokens == TokenCounts(1099, 128)
    assert call.latency_ms >= 0
    verdict = call.answers["verdict"]
    assert isinstance(verdict, ChoiceAnswer)
    assert verdict.choice == "pass"
    assert verdict.confidence == 0.36
    assert verdict.probabilities == {"pass": 0.68, "flag": 0.32}
    assert call.answers["amount_mismatch"] == 0.06
    assert call.answers["boundary_direction"] == 0.15

    assert seen["init"] == {
        "api_key": SECRET,
        "model": None,
        "max_retries": 1,
        "timeout": 12.0,
    }
    assert seen["closed"] == 1
    sent = seen["calls"][0]
    assert sent["state"] == {"citation": "c"}
    assert set(sent["questions"]) == {"verdict", "faithful", *FIDELITY_KINDS}
    assert type(sent["questions"]["verdict"]).__name__ == "Choice"
    assert set(sent["questions"]["verdict"].criteria) == {"pass", "flag"}
    assert type(sent["questions"]["amount_mismatch"]).__name__ == "Noul"


def test_questions_are_one_choice_and_one_noul_per_fidelity_kind():
    questions = statutory_fidelity_screen.build_questions()
    assert isinstance(questions["verdict"], ChoiceQuestion)
    assert set(questions["verdict"].criteria) == {"pass", "flag"}
    for kind in (
        "unrepresented_clause",
        "untraceable_branch",
        "boundary_direction",
        "amount_mismatch",
    ):
        assert isinstance(questions[kind], NoulQuestion)
    assert tuple(FIDELITY_KINDS) == (
        "unrepresented_clause",
        "untraceable_branch",
        "boundary_direction",
        "amount_mismatch",
    )


def test_client_pins_a_configured_model(monkeypatch):
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_MODEL", "jev-1.13.0")
    seen = install_fake_typesafe(monkeypatch, [recorded_response(RECORDED_ORIGINAL)])
    _client().call(state={}, questions={"q": NoulQuestion(instructions="x")})
    assert seen["init"]["model"] == "jev-1.13.0"


# -- client: fail closed ---------------------------------------------------


def test_missing_key_is_fail_closed(monkeypatch):
    seen = install_fake_typesafe(monkeypatch, [recorded_response(RECORDED_ORIGINAL)])
    call = SystemOneClient(generator_model="gpt-5.6-terra").call(
        state={}, questions={"q": NoulQuestion(instructions="x")}
    )
    assert not call.ok
    assert call.error.type == "missing_api_key"
    assert seen["calls"] == []


def test_missing_sdk_is_fail_closed(monkeypatch):
    # ``None`` in sys.modules makes ``import typesafe_sdk`` raise ImportError.
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    call = _client().call(state={}, questions={"q": NoulQuestion(instructions="x")})
    assert not call.ok
    assert call.error.type == "sdk_missing"
    assert "axiom-encode[typesafe]" in call.error.message


def test_empty_questions_is_fail_closed(monkeypatch):
    install_fake_typesafe(monkeypatch, [])
    call = _client().call(state={}, questions={})
    assert not call.ok
    assert call.error.type == "empty_questions"


@pytest.mark.parametrize(
    "exc",
    [
        TypeSafeRateLimitError(LEAKY_TEXT),
        TypeSafeAuthenticationError(LEAKY_TEXT),
        TypeSafeInternalServerError(LEAKY_TEXT),
        TypeSafeAPIConnectionError(LEAKY_TEXT),
        TypeSafeAPITimeoutError(LEAKY_TEXT),
        RuntimeError(LEAKY_TEXT),
    ],
)
def test_sdk_exceptions_map_to_a_judge_error_by_class_name_only(monkeypatch, exc):
    seen = install_fake_typesafe(monkeypatch, [exc])
    call = _client().call(state={}, questions={"q": NoulQuestion(instructions="x")})
    assert not call.ok
    assert call.answers is None
    assert call.error.type == type(exc).__name__
    assert type(exc).__name__ in call.error.message
    for forbidden in (SECRET, "Authorization", "Bearer", "x-api-key", "api.typesafe"):
        assert forbidden not in call.error.message
        assert forbidden not in call.error.type
    assert call.latency_ms >= 0
    assert seen["closed"] == 1


def test_client_construction_failure_is_an_error_not_a_raise(monkeypatch):
    install_fake_typesafe(monkeypatch, [], init_error=FakeTypeSafeError(LEAKY_TEXT))
    call = _client().call(state={}, questions={"q": NoulQuestion(instructions="x")})
    assert not call.ok
    assert call.error.type == "FakeTypeSafeError"
    assert SECRET not in call.error.message


@pytest.mark.parametrize(
    "response",
    [
        recorded_response(RECORDED_ORIGINAL, drop=("amount_mismatch",)),
        recorded_response(
            RECORDED_ORIGINAL, answers={"amount_mismatch": _NoulAnswer("high")}
        ),
        recorded_response(RECORDED_ORIGINAL, answers={"verdict": _NoulAnswer(0.4)}),
        recorded_response(
            RECORDED_ORIGINAL,
            answers={"verdict": _ChoiceAnswer("pass", 0.5, {"pass": "likely"})},
        ),
        _Response("jev-1.13.0", _Usage(5, 1), None),
    ],
)
def test_missing_or_malformed_answers_are_schema_errors(monkeypatch, response):
    install_fake_typesafe(monkeypatch, [response])
    call = _client().call(
        state={}, questions=statutory_fidelity_screen.build_questions()
    )
    assert not call.ok
    assert call.error.type == "schema_error"
    assert call.answers is None


def test_client_repr_and_call_result_never_carry_the_key(monkeypatch):
    install_fake_typesafe(monkeypatch, [recorded_response(RECORDED_ORIGINAL)])
    client = _client()
    assert SECRET not in repr(client)
    call = client.call(state={}, questions={"q": NoulQuestion(instructions="x")})
    assert SECRET not in repr(call)


# -- stage: verdict mapping ------------------------------------------------


def test_recorded_clean_original_passes_with_no_findings(monkeypatch):
    event = _screen(monkeypatch, RECORDED_ORIGINAL)
    assert event.stage == JudgeStage.STATUTORY_FIDELITY_SCREEN
    assert event.verdict == Verdict.PASS
    assert event.findings == []
    assert event.advisory is True
    assert event.confidence == 0.36
    screen = event.extra["screen"]
    assert screen["probabilities"] == {
        "unrepresented_clause": 0.5,
        "untraceable_branch": 0.34,
        "boundary_direction": 0.15,
        "amount_mismatch": 0.06,
    }
    assert screen["p_faithful"] == 0.88
    assert screen["verdict_probabilities"] == {"pass": 0.68, "flag": 0.32}
    assert validate_event_dict(event.to_dict()) == []


def test_recorded_planted_amount_flags_with_probability_and_no_locators(monkeypatch):
    event = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT, rule_path="co/4.803.1.yaml")
    assert event.verdict == Verdict.FLAG
    # Only the two pilot-validated kinds report; both crossed the placeholder.
    assert [f.kind for f in event.findings] == ["boundary_direction", "amount_mismatch"]
    amount = event.findings[1]
    assert amount.probability == 0.97
    # No fabricated clause reference or rule path, and the finding says so.
    assert amount.clause_ref == ""
    assert amount.rule_path == ""
    assert "probabilities only" in amount.explanation
    assert "does not locate" in amount.explanation

    payload = event.to_dict()
    assert payload["status"] == "passed"  # advisory: a flag never gates
    finding = payload["findings"][1]
    assert finding["code"] == "amount_mismatch"
    assert finding["locator"] is None
    assert finding["evidence"] == "probability=0.9700"
    assert validate_event_dict(payload) == []


def test_record_only_kinds_never_report_unless_configured(monkeypatch):
    # unrepresented_clause scored 0.56 here, above the placeholder, but the
    # pilot could not validate that kind, so by default it is recorded only.
    event = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT)
    assert "unrepresented_clause" not in [f.kind for f in event.findings]
    assert event.extra["screen"]["probabilities"]["unrepresented_clause"] == 0.56

    policy = ScreenPolicy.from_env(
        {"AXIOM_JUDGE_SCREEN_THRESHOLD_UNREPRESENTED_CLAUSE": "0.5"}
    )
    event = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT, policy=policy)
    assert "unrepresented_clause" in [f.kind for f in event.findings]


def test_choice_flag_without_a_kind_over_threshold_flags_without_findings(monkeypatch):
    row = {**RECORDED_ORIGINAL, "verdict": "flag", "p_flag": 0.61}
    event = _screen(monkeypatch, row)
    assert event.verdict == Verdict.FLAG
    assert event.findings == []
    assert validate_event_dict(event.to_dict()) == []


def test_kind_over_threshold_flags_even_when_the_choice_says_pass(monkeypatch):
    row = {**RECORDED_ORIGINAL, "amount_mismatch": 0.9}
    event = _screen(monkeypatch, row)
    assert event.verdict == Verdict.FLAG
    assert [f.kind for f in event.findings] == ["amount_mismatch"]


def test_unrecognized_choice_is_an_error_never_a_pass(monkeypatch):
    response = recorded_response(
        RECORDED_ORIGINAL,
        answers={"verdict": _ChoiceAnswer("maybe", 0.5, {"maybe": 1.0})},
    )
    event = _screen(monkeypatch, response)
    assert event.verdict == Verdict.ERROR
    assert event.passed is False
    assert event.judge_error.type == "unrecognized_verdict"
    assert validate_event_dict(event.to_dict()) == []


def test_client_failure_is_an_error_event_never_a_pass(monkeypatch):
    event = _screen(monkeypatch, TypeSafeRateLimitError(LEAKY_TEXT))
    assert event.verdict == Verdict.ERROR
    assert event.passed is False
    payload = event.to_dict()
    assert payload["status"] == "error"
    assert payload["attrs"]["judge_error"]["type"] == "TypeSafeRateLimitError"
    assert payload["reason_code"] == "TypeSafeRateLimitError"
    assert validate_event_dict(payload) == []


def test_screen_sends_the_same_truncated_provision_window_as_the_referee():
    provision = "HEAD" + ("x" * 60_000) + "TAILBOUNDARY"
    fake = FakeSystemOneClient(_call_from(RECORDED_ORIGINAL))
    fake.provision_chars = 1_000
    statutory_fidelity_screen.run(provision, "rules: []", citation="c", client=fake)
    state = fake.seen["state"]
    assert state["provision_text_verbatim"] == truncate_provision(provision, 1_000)
    assert state["provision_text_verbatim"].endswith("TAILBOUNDARY")
    assert state["generated_rulespec_artifact"] == "rules: []"
    assert state["citation"] == "c"


def test_event_records_latency_usage_policy_and_a_prompt_digest():
    fake = FakeSystemOneClient(_call_from(RECORDED_ORIGINAL))
    event = statutory_fidelity_screen.run("prov", "rule", citation="c", client=fake)
    assert event.tokens == TokenCounts(1099, 128)
    screen = event.extra["screen"]
    assert screen["latency_ms"] == 190
    assert screen["mode"] == "advisory"
    assert screen["threshold_source"] == "placeholder"
    assert screen["questions_version"] == statutory_fidelity_screen.QUESTIONS_VERSION
    assert len(event.judge_prompt_sha256) == 64
    # The digest binds the state actually sent.
    other = statutory_fidelity_screen.run("prov", "rule 2", citation="c", client=fake)
    assert other.judge_prompt_sha256 != event.judge_prompt_sha256


def test_screen_never_produces_the_needs_review_label(monkeypatch):
    flagged = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT)
    errored = _screen(monkeypatch, RuntimeError("boom"))
    assert statutory_fidelity.needs_review_label(flagged) is None
    assert statutory_fidelity.needs_review_label(errored) is None


# -- cascade policy --------------------------------------------------------


def test_policy_defaults_are_advisory_with_the_placeholder_threshold():
    policy = ScreenPolicy.from_env({})
    assert DEFAULT_JUDGE_SCREEN_MODE == "advisory"
    assert policy.mode == "advisory"
    assert policy.cascade is False
    assert policy.thresholds == {
        "amount_mismatch": DEFAULT_JUDGE_SCREEN_THRESHOLD,
        "boundary_direction": DEFAULT_JUDGE_SCREEN_THRESHOLD,
    }
    assert policy.to_dict()["threshold_source"] == "placeholder"
    assert ScreenPolicy() == ScreenPolicy.from_env({})


def test_policy_reads_mode_and_thresholds_from_the_environment():
    policy = ScreenPolicy.from_env(
        {
            "AXIOM_JUDGE_SCREEN_MODE": " Cascade ",
            "AXIOM_JUDGE_SCREEN_THRESHOLD": "0.4",
            "AXIOM_JUDGE_SCREEN_THRESHOLD_BOUNDARY_DIRECTION": "0.2",
        }
    )
    assert policy.cascade is True
    assert policy.thresholds == {"amount_mismatch": 0.4, "boundary_direction": 0.2}
    assert policy.to_dict()["threshold_source"] == "env"
    assert policy.with_mode("advisory").cascade is False
    assert policy.with_mode("advisory").thresholds == policy.thresholds


@pytest.mark.parametrize(
    "environ",
    [
        {"AXIOM_JUDGE_SCREEN_MODE": "gate"},
        {"AXIOM_JUDGE_SCREEN_THRESHOLD": "high"},
        {"AXIOM_JUDGE_SCREEN_THRESHOLD": "1.5"},
        {"AXIOM_JUDGE_SCREEN_THRESHOLD_AMOUNT_MISMATCH": "-0.1"},
    ],
)
def test_invalid_policy_configuration_is_refused(environ):
    with pytest.raises(ValueError):
        ScreenPolicy.from_env(environ)


def test_policy_refuses_unknown_kinds_and_an_untriggerable_cascade():
    with pytest.raises(ValueError):
        ScreenPolicy(thresholds={"style": 0.5})
    # A cascade with nothing to trigger on would skip the referee every time.
    with pytest.raises(ValueError):
        ScreenPolicy(mode="cascade", thresholds={})
    assert ScreenPolicy(mode="advisory", thresholds={}).thresholds == {}


def test_advisory_mode_always_requests_the_referee(monkeypatch):
    policy = ScreenPolicy()
    clean = _screen(monkeypatch, RECORDED_ORIGINAL, policy=policy)
    planted = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT, policy=policy)
    assert statutory_fidelity_screen.cascade_decision(clean, policy) == CascadeDecision(
        True, "advisory_mode", ()
    )
    decision = statutory_fidelity_screen.cascade_decision(planted, policy)
    assert decision.request_referee is True
    assert decision.reason == "advisory_mode"
    assert set(decision.triggered) == {"amount_mismatch", "boundary_direction"}


def test_cascade_mode_skips_the_referee_only_below_threshold(monkeypatch):
    policy = ScreenPolicy(mode="cascade")
    clean = _screen(monkeypatch, RECORDED_ORIGINAL, policy=policy)
    below = statutory_fidelity_screen.cascade_decision(clean, policy)
    assert below == CascadeDecision(False, "below_threshold", ())
    assert clean.extra["screen"]["cascade"] == below.to_dict()

    planted = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT, policy=policy)
    above = statutory_fidelity_screen.cascade_decision(planted, policy)
    assert above.request_referee is True
    assert above.reason == "threshold_exceeded"
    assert set(above.triggered) == {"amount_mismatch", "boundary_direction"}


def test_cascade_threshold_is_inclusive_and_per_kind(monkeypatch):
    row = {**RECORDED_ORIGINAL, "boundary_direction": 0.25, "amount_mismatch": 0.24}
    policy = ScreenPolicy(mode="cascade")
    event = _screen(monkeypatch, row, policy=policy)
    decision = statutory_fidelity_screen.cascade_decision(event, policy)
    assert decision.request_referee is True
    assert decision.triggered == ("boundary_direction",)

    strict = ScreenPolicy(
        mode="cascade", thresholds={"amount_mismatch": 0.3, "boundary_direction": 0.3}
    )
    event = _screen(monkeypatch, row, policy=strict)
    assert statutory_fidelity_screen.cascade_decision(
        event, strict
    ).request_referee is (False)


def test_the_choice_verdict_alone_never_triggers_the_cascade(monkeypatch):
    row = {**RECORDED_ORIGINAL, "verdict": "flag", "p_flag": 0.9}
    policy = ScreenPolicy(mode="cascade")
    event = _screen(monkeypatch, row, policy=policy)
    assert event.verdict == Verdict.FLAG
    decision = statutory_fidelity_screen.cascade_decision(event, policy)
    assert decision == CascadeDecision(False, "below_threshold", ())


def test_an_errored_screen_always_requests_the_referee(monkeypatch):
    policy = ScreenPolicy(mode="cascade")
    event = _screen(monkeypatch, TypeSafeAPITimeoutError(LEAKY_TEXT), policy=policy)
    decision = statutory_fidelity_screen.cascade_decision(event, policy)
    assert decision == CascadeDecision(True, "screen_error", ())
    assert event.extra["screen"]["cascade"]["request_referee"] is True


def test_an_incomplete_screen_requests_the_referee():
    policy = ScreenPolicy(mode="cascade")
    event = JudgeEvent(
        stage=JudgeStage.STATUTORY_FIDELITY_SCREEN,
        verdict=Verdict.PASS,
        extra={"screen": {"probabilities": {"amount_mismatch": 0.01}}},
    )
    decision = statutory_fidelity_screen.cascade_decision(event, policy)
    assert decision.request_referee is True
    assert decision.reason == "screen_incomplete"


def test_cascade_decision_refuses_events_from_other_stages():
    referee = JudgeEvent(stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.PASS)
    with pytest.raises(ValueError):
        statutory_fidelity_screen.cascade_decision(referee, ScreenPolicy())


# -- run log ---------------------------------------------------------------


def test_screen_event_emits_into_the_canonical_run_log(monkeypatch, tmp_path):
    event = _screen(monkeypatch, RECORDED_PLANTED_AMOUNT, run_id="run-screen")
    writer = RunLogWriter("run-screen", log_dir=tmp_path)
    assert event.emit(writer) is not None and writer.last_error is None

    stored = list(iter_events(tmp_path / "run-screen.jsonl"))[0]
    assert stored.stage == JUDGE_STAGE
    assert stored.status == StageStatus.passed.value
    assert stored.attrs["judge_stage"] == "statutory_fidelity_screen"
    assert stored.attrs["judge_model"] == "jev-1.13.0"
    assert stored.attrs["judge_model_family"] == "typesafe"
    assert stored.attrs["advisory"] is True
    assert stored.attrs["tokens"] == {"input": 1099, "output": 128}
    assert stored.attrs["screen"]["probabilities"]["amount_mismatch"] == 0.97
    assert stored.attrs["screen"]["cascade"]["reason"] == "advisory_mode"
    assert [f.code for f in stored.findings] == [
        "boundary_direction",
        "amount_mismatch",
    ]
    assert stored.findings[1].evidence == "probability=0.9700"
    assert stored.findings[1].locator is None
    assert validate_event_dict(stored.to_dict()) == []


def test_no_key_or_authorization_header_can_reach_a_run_log(monkeypatch, tmp_path):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    install_fake_typesafe(
        monkeypatch,
        [
            recorded_response(RECORDED_PLANTED_AMOUNT),
            TypeSafeAuthenticationError(LEAKY_TEXT),
            TypeSafeAPIConnectionError(LEAKY_TEXT),
        ],
    )
    writer = RunLogWriter("run-secrets", log_dir=tmp_path)
    serialized = []
    for _ in range(3):
        # Default client: the key comes from the environment, as in production.
        event = statutory_fidelity_screen.run(
            "prov", "rule", citation="c", run_id="run-secrets"
        )
        assert event.emit(writer) is not None
        serialized.append(json.dumps(event.to_dict(), default=str))
        serialized.append(repr(event))
    serialized.append((tmp_path / "run-secrets.jsonl").read_text(encoding="utf-8"))

    assert len(list(iter_events(tmp_path / "run-secrets.jsonl"))) == 3
    for text in serialized:
        for forbidden in (SECRET, "Authorization", "Bearer", "x-api-key"):
            assert forbidden not in text


# -- CLI -------------------------------------------------------------------


class _StubSource:
    body = "authoritative source"
    requested = "us/statute/26/1"

    def to_attestation(self):
        return {"requested_corpus_citation_path": self.requested}


def _cli_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    rule_file = tmp_path / "rule.yaml"
    rule_file.write_text("rules: []\n", encoding="utf-8")
    values = dict(
        root=tmp_path,
        corpus_path=tmp_path,
        corpus_citation_path="us/statute/26/1",
        rule_file=rule_file,
        rule_path=None,
        run_id="run-cli",
        log_dir=tmp_path / "logs",
        json=True,
        screen=True,
        screen_mode=None,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def cli(monkeypatch):
    """Bind the CLI to a stub corpus source, a fake screen, and a spy referee."""

    state = {"row": RECORDED_ORIGINAL, "referee_calls": 0}
    monkeypatch.setattr(
        cli_commands, "_load_bound_source", lambda args, citation: _StubSource()
    )
    monkeypatch.setattr(
        statutory_fidelity_screen,
        "SystemOneClient",
        lambda: FakeSystemOneClient(_call_from(state["row"])),
    )

    def fake_referee(provision_text, generated_rule, **kwargs):
        state["referee_calls"] += 1
        return JudgeEvent(stage=JudgeStage.STATUTORY_FIDELITY, verdict=Verdict.PASS)

    monkeypatch.setattr(statutory_fidelity, "run", fake_referee)
    return state


def test_parser_wires_the_screen_command_and_flags():
    parser = argparse.ArgumentParser()
    cli_commands.register_judge_subparsers(parser.add_subparsers(dest="command"))
    common = [
        "--root",
        "r",
        "--corpus-path",
        "c",
        "--corpus-citation-path",
        "us/statute/26/1",
        "--rule-file",
        "x.yaml",
    ]
    assert "judge-fidelity-screen" in cli_commands.COMMANDS
    plain = parser.parse_args(["judge-fidelity", *common])
    assert plain.screen is False and plain.screen_mode is None
    screened = parser.parse_args(
        ["judge-fidelity", *common, "--screen", "--screen-mode", "cascade"]
    )
    assert screened.screen is True and screened.screen_mode == "cascade"
    alone = parser.parse_args(["judge-fidelity-screen", *common])
    assert alone.command == "judge-fidelity-screen"
    with pytest.raises(SystemExit):
        parser.parse_args(["judge-fidelity", *common, "--screen-mode", "gate"])


def test_judge_fidelity_without_screen_is_unchanged(cli, tmp_path, capsys):
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path, screen=False))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attrs"]["judge_stage"] == "statutory_fidelity"
    assert cli["referee_calls"] == 1
    events = list(iter_events(tmp_path / "logs" / "run-cli.jsonl"))
    assert [e.attrs["judge_stage"] for e in events] == ["statutory_fidelity"]


def test_advisory_screen_records_itself_and_still_runs_the_referee(
    cli, tmp_path, capsys
):
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cascade"] == {
        "request_referee": True,
        "reason": "advisory_mode",
        "triggered": [],
    }
    assert payload["screen"]["attrs"]["judge_stage"] == "statutory_fidelity_screen"
    assert payload["referee"]["attrs"]["judge_stage"] == "statutory_fidelity"
    assert payload["referee"]["attrs"]["screen_cascade"]["reason"] == "advisory_mode"
    assert cli["referee_calls"] == 1
    events = list(iter_events(tmp_path / "logs" / "run-cli.jsonl"))
    assert [e.attrs["judge_stage"] for e in events] == [
        "statutory_fidelity_screen",
        "statutory_fidelity",
    ]
    assert events[0].attrs["source_attestation"] == {
        "requested_corpus_citation_path": "us/statute/26/1"
    }


def test_cascade_screen_below_threshold_skips_the_referee(
    cli, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_MODE", "cascade")
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["referee"] is None
    assert payload["cascade"]["request_referee"] is False
    assert payload["cascade"]["reason"] == "below_threshold"
    assert cli["referee_calls"] == 0
    events = list(iter_events(tmp_path / "logs" / "run-cli.jsonl"))
    assert [e.attrs["judge_stage"] for e in events] == ["statutory_fidelity_screen"]


def test_cascade_screen_above_threshold_requests_the_referee(cli, tmp_path, capsys):
    cli["row"] = RECORDED_PLANTED_AMOUNT
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path, screen_mode="cascade"))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cascade"]["reason"] == "threshold_exceeded"
    assert payload["referee"] is not None
    assert cli["referee_calls"] == 1


def test_cascade_screen_error_still_runs_the_referee(
    cli, tmp_path, capsys, monkeypatch
):
    from axiom_encode.judges.run_log import JudgeError

    failed = SystemOneCall(
        answers=None,
        model=None,
        family=None,
        tokens=TokenCounts(),
        latency_ms=3,
        error=JudgeError(type="TypeSafeRateLimitError", message="request raised"),
    )
    monkeypatch.setattr(
        statutory_fidelity_screen,
        "SystemOneClient",
        lambda: FakeSystemOneClient(failed),
    )
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path, screen_mode="cascade"))
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["screen"]["status"] == "error"
    assert payload["cascade"]["reason"] == "screen_error"
    assert payload["referee"] is not None
    assert cli["referee_calls"] == 1


def test_invalid_screen_policy_exits_before_any_judge_runs(
    cli, tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv("AXIOM_JUDGE_SCREEN_THRESHOLD", "very high")
    status = cli_commands.cmd_judge_fidelity(_cli_args(tmp_path))
    assert status == 2
    assert "invalid screen policy" in capsys.readouterr().err
    assert cli["referee_calls"] == 0


def test_dispatch_routes_the_standalone_screen_command(cli, tmp_path, capsys):
    cli["row"] = RECORDED_PLANTED_AMOUNT
    args = _cli_args(tmp_path, command="judge-fidelity-screen")
    assert cli_commands.dispatch(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["attrs"]["judge_stage"] == "statutory_fidelity_screen"
    assert payload["attrs"]["verdict"] == "flag"
    assert payload["attrs"]["screen"]["cascade"]["request_referee"] is True
    assert cli["referee_calls"] == 0
    events = list(iter_events(tmp_path / "logs" / "run-cli.jsonl"))
    assert len(events) == 1


def test_standalone_screen_text_output_names_the_cascade(cli, tmp_path, capsys):
    args = _cli_args(tmp_path, json=False, screen_mode="cascade")
    assert cli_commands.cmd_judge_fidelity_screen(args) == 0
    out = capsys.readouterr().out
    assert "stage=statutory_fidelity_screen" in out
    assert "cascade: referee skipped (below_threshold" in out


# -- documentation contract ------------------------------------------------

_DOC = Path(__file__).parents[1] / "docs" / "judge-stages.md"
_TRAINING_SENTENCE = (
    "TypeSafe describes its training as reinforcement learning for\n"
    "calibrated decisions; reward and data undisclosed."
)


def test_doc_marks_the_threshold_as_a_placeholder_owned_by_the_verifier_track():
    text = _DOC.read_text(encoding="utf-8")
    assert f"`{DEFAULT_JUDGE_SCREEN_THRESHOLD}` (placeholder)" in text
    assert "benchmarks/verifier/" in text
    assert "is the authority for the real threshold" in text


def test_doc_makes_no_training_claim_beyond_the_permitted_sentence():
    text = _DOC.read_text(encoding="utf-8")
    assert _TRAINING_SENTENCE in text
    remainder = text.replace(_TRAINING_SENTENCE, "")
    for word in ("trained", "training", "reinforcement", "fine-tun"):
        assert word not in remainder.lower()
    assert "—" not in text  # no em-dashes

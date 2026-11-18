"""Messages runner contracts, with no live Anthropic requests."""

import asyncio
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import evals
from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.pricing import (
    anthropic_refusal_is_billed,
    estimate_usage_cost_usd,
)


def _response(stop_reason="end_turn", text="module: synthetic", **kwargs):
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="private"),
            SimpleNamespace(type="text", text=text),
        ],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=20,
            cache_creation_input_tokens=10,
            reasoning_output_tokens=7,
        ),
        stop_reason=stop_reason,
        model="served-claude-model",
        _request_id="req-synthetic",
        **kwargs,
    )


def _mock_anthropic(response=None, *, error=None, events=(), stall=False):
    calls, clients = [], []

    class Stream:
        async def __aenter__(self):
            if error is not None:
                raise error
            return self

        async def __aexit__(self, *_args):
            return False

        async def __aiter__(self):
            if stall:
                await asyncio.sleep(30)
            for event in events:
                yield event

        async def get_final_message(self):
            return response

    class Client:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.closed = False
            self.messages = self
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            self.closed = True

        def stream(self, **kwargs):
            calls.append(kwargs)
            return Stream()

    class APITimeoutError(Exception):
        pass

    return SimpleNamespace(
        AsyncAnthropic=Client,
        APITimeoutError=APITimeoutError,
        __version__="mock-anthropic-version",
        calls=calls,
        clients=clients,
    )


def _run(
    prompt="test prompt",
    response=None,
    *,
    sdk=None,
    model="claude-opus-5-5",
    effort="high",
):
    sdk = sdk or _mock_anthropic(response or _response())
    with (
        patch.dict(sys.modules, {"anthropic": sdk}),
        patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "synthetic-api-key",
                "ANTHROPIC_BASE_URL": "https://redirect.invalid",
                "AXIOM_API_ENCODER_EFFORT": effort,
            },
        ),
    ):
        result = evals._run_claude_api_prompt_eval(
            evals.EvalRunnerSpec("api", "claude-api", model),
            SimpleNamespace(root=Path("synthetic-workspace")),
            prompt,
        )
    return result, sdk


@settings(max_examples=30, deadline=None)
@given(st.text())
def test_request_preserves_any_prompt_with_neutral_system(prompt):
    """I1: exact user bytes and no credit-ineligible system identity."""
    result, sdk = _run(prompt)
    assert result.error is None
    request = sdk.calls[0]
    assert request["messages"] == [{"role": "user", "content": prompt}]
    system = request["system"]
    assert (
        "you are claude code, anthropic's official cli for claude."
        not in system.lower()
    )
    assert (
        "you are a claude agent, built on anthropic's claude agent sdk."
        not in system.lower()
    )
    assert (
        result.trace["system_prompt_sha256"]
        == hashlib.sha256(system.encode()).hexdigest()
    )
    assert request["max_tokens"] == 64_000
    assert request["output_config"] == {"effort": "high"}
    client_options = sdk.clients[0].kwargs
    assert client_options["api_key"] == "synthetic-api-key"
    assert client_options["base_url"] == "https://api.anthropic.com"
    assert client_options["max_retries"] == 0
    assert sdk.clients[0].closed
    assert "synthetic-api-key" not in repr(result)
    assert "redirect.invalid" not in repr(result)


@settings(max_examples=30, deadline=None)
@given(st.text())
def test_identical_inputs_build_identical_request_payloads(prompt):
    """I9: request content has no clock-, workspace-, or random-dependent data."""
    _, first = _run(prompt)
    _, second = _run(prompt)
    assert first.calls == second.calls


def test_prompt_dispatch_selects_claude_api_runner():
    runner = evals.EvalRunnerSpec("api", "claude-api", "claude-opus-5-5")
    workspace = SimpleNamespace(root=Path("synthetic-workspace"))
    response = evals.EvalPromptResponse(text="synthetic: true", duration_ms=0)
    with (
        patch.object(evals, "_remaining_eval_case_budget_seconds", return_value=None),
        patch.object(
            evals, "_run_claude_api_prompt_eval", return_value=response
        ) as run,
    ):
        assert evals._run_prompt_eval(runner, workspace, "exact prompt") is response
    run.assert_called_once_with(runner, workspace, "exact prompt")


@pytest.mark.parametrize(
    "stop_reason",
    [
        "end_turn",
        "stop_sequence",
        "refusal",
        "max_tokens",
        "model_context_window_exceeded",
        "pause_turn",
        "tool_use",
        "unknown",
        None,
    ],
)
@pytest.mark.parametrize("text", ["", " \n", "synthetic: true"])
def test_only_finished_nonempty_responses_yield_artifacts(stop_reason, text):
    """I3: exhaust all known stops, an unknown stop, and absent stop metadata."""
    result, _ = _run(response=_response(stop_reason, text))
    valid = stop_reason in {"end_turn", "stop_sequence"} and bool(text.strip())
    assert (result.error is None) == valid
    assert result.text == (text if valid else "")
    assert result.trace["stop_reason"] == stop_reason


@settings(max_examples=30, deadline=None)
@given(st.text(), st.sampled_from(["http", "stop"]))
def test_context_overflow_is_infrastructure_without_text(prompt, source):
    """I2: API context overflow cannot leave a partial artifact to score."""
    if source == "http":
        error = type("BadRequestError", (Exception,), {"status_code": 400})(
            "prompt is too long: input length exceeds the context window"
        )
        sdk = _mock_anthropic(error=error)
    else:
        sdk = _mock_anthropic(_response("model_context_window_exceeded"))
    result, _ = _run(prompt, sdk=sdk)
    assert result.text == ""
    assert "context overflow" in result.error.lower()
    assert result.trace["infrastructure_failure"] == "context-overflow"
    assert result.trace["prompt_chars"] == len(prompt)
    assert str(len(prompt)) in result.error


def test_usage_pricing_and_trace_metadata():
    result, _ = _run()
    assert result.tokens == TokenUsage(130, 50, 20, 10, 7)
    assert result.trace["uncached_input_tokens"] == 100
    assert result.estimated_cost_usd == estimate_usage_cost_usd(
        "claude-opus-5-5", result.tokens
    )
    assert result.trace["provider"] == "anthropic"
    assert result.trace["backend"] == "messages"
    assert result.trace["model"] == "claude-opus-5-5"
    assert result.trace["served_model"] == "served-claude-model"
    assert result.trace["effort"] == "high"
    assert result.trace["anthropic_sdk_version"] == "mock-anthropic-version"
    assert result.trace["request_id"] == "req-synthetic"
    assert result.trace["timed_out"] is False


def test_sdk_model_dump_usage_captures_nested_reasoning_tokens():
    response = _response()
    response.usage = SimpleNamespace(
        model_dump=lambda: {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 10,
            "output_tokens_details": {"reasoning_tokens": 7},
        }
    )
    result, _ = _run(response=response)
    assert result.error is None
    assert result.tokens == TokenUsage(130, 50, 20, 10, 7)
    assert result.trace["uncached_input_tokens"] == 100
    assert result.estimated_cost_usd == estimate_usage_cost_usd(
        "claude-opus-5-5", result.tokens
    )


def test_haiku_cost_stays_unknown():
    result, _ = _run(model="claude-haiku-5-5")
    assert result.tokens is not None
    assert result.estimated_cost_usd is None


@pytest.mark.parametrize(
    "effort, expected", [(" MAX ", "max"), ("", "high"), ("low", "low")]
)
def test_effort_resolution_matches_encoder(effort, expected):
    result, sdk = _run(effort=effort)
    assert result.error is None
    assert result.trace["effort"] == expected
    assert sdk.calls[0]["output_config"] == {"effort": expected}


def test_invalid_effort_fails_before_constructing_client():
    result, sdk = _run(effort="not-an-effort")
    assert result.text == ""
    assert "AXIOM_API_ENCODER_EFFORT" in result.error
    assert "not a valid effort" in result.error
    assert sdk.clients == []


def test_refusal_category_survives_streamed_delta():
    sdk = _mock_anthropic(
        _response("refusal"),
        events=[
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_details={"category": "bio"}),
            )
        ],
    )
    result, _ = _run(sdk=sdk)
    assert result.text == ""
    assert result.trace["refusal_category"] == "bio"
    assert "bio" in result.error


@pytest.mark.parametrize(
    "category",
    [
        "bio",
        "frontier_llm",
        "reasoning_extraction",
        "cyber",
        "general_harms",
        None,
        "a_future_category",
    ],
)
@pytest.mark.parametrize("shape", ["dict", "object"])
@pytest.mark.parametrize("delivery", ["final", "stream", "absent"])
def test_pre_output_refusal_cost_matches_existing_billing_policy(
    category, shape, delivery
):
    details = (
        {"category": category}
        if shape == "dict"
        else SimpleNamespace(category=category)
    )
    response = _response(
        "refusal", "", stop_details=details if delivery == "final" else None
    )
    response.usage = SimpleNamespace(input_tokens=100, output_tokens=0)
    events = (
        [
            SimpleNamespace(
                type="message_delta", delta=SimpleNamespace(stop_details=details)
            )
        ]
        if delivery == "stream"
        else ()
    )
    result, _ = _run(sdk=_mock_anthropic(response, events=events))

    seen_category = category if delivery != "absent" else None
    billed = anthropic_refusal_is_billed(seen_category, 0)
    if delivery == "absent" or billed is None:
        expected = None
    elif billed:
        expected = estimate_usage_cost_usd("claude-opus-5-5", result.tokens)
    else:
        expected = 0.0
    assert result.text == ""
    assert result.error is not None
    assert result.tokens == TokenUsage(input_tokens=100, output_tokens=0)
    assert result.trace["refusal_category"] == seen_category
    assert result.estimated_cost_usd == expected


@settings(max_examples=60, deadline=None)
@given(
    model=st.sampled_from(["claude-opus-5-5", "claude-sonnet-5-5"]),
    category=st.one_of(
        st.none(),
        st.sampled_from(
            [
                "bio",
                "frontier_llm",
                "reasoning_extraction",
                "cyber",
                "general_harms",
                "a_future_category",
            ]
        ),
    ),
    shape=st.sampled_from(["dict", "object"]),
    delivery=st.sampled_from(["final", "stream", "absent"]),
    input_counts=st.lists(st.integers(0, 10_000), min_size=3, max_size=3),
    output_tokens=st.one_of(st.just(0), st.integers(1, 10_000)),
)
def test_refusal_cost_invariant_matches_existing_billing_policy(
    model, category, shape, delivery, input_counts, output_tokens
):
    """Every refusal uses the existing policy across categories and usage."""
    details = (
        {"category": category}
        if shape == "dict"
        else SimpleNamespace(category=category)
    )
    response = _response(
        "refusal", "", stop_details=details if delivery == "final" else None
    )
    uncached, read, create = input_counts
    response.usage = SimpleNamespace(
        input_tokens=uncached,
        output_tokens=output_tokens,
        cache_read_input_tokens=read,
        cache_creation_input_tokens=create,
    )
    events = (
        [
            SimpleNamespace(
                type="message_delta", delta=SimpleNamespace(stop_details=details)
            )
        ]
        if delivery == "stream"
        else ()
    )
    result, _ = _run(sdk=_mock_anthropic(response, events=events), model=model)

    seen_category = category if delivery != "absent" else None
    billed = anthropic_refusal_is_billed(seen_category, output_tokens)
    expected = estimate_usage_cost_usd(model, result.tokens)
    if billed is None or (output_tokens == 0 and delivery == "absent"):
        expected = None
    elif not billed:
        expected = 0.0
    assert result.text == ""
    assert result.error is not None
    assert result.tokens == TokenUsage(
        input_tokens=sum(input_counts),
        output_tokens=output_tokens,
        cache_read_tokens=read,
        cache_creation_tokens=create,
    )
    assert result.trace["refusal_category"] == seen_category
    assert result.estimated_cost_usd == expected


def test_sdk_exception_class_and_message_are_reported_without_text():
    result, _ = _run(sdk=_mock_anthropic(error=RuntimeError("synthetic failure")))
    assert result.text == ""
    assert "RuntimeError" in result.error
    assert "synthetic failure" in result.error
    assert result.trace["exception_class"] == "RuntimeError"
    assert result.trace["exception_message"] == "synthetic failure"


def test_sdk_exception_never_echoes_the_key():
    result, _ = _run(
        sdk=_mock_anthropic(error=RuntimeError("SDK failed with synthetic-api-key"))
    )
    assert "synthetic-api-key" not in repr(result)
    assert result.trace["exception_class"] == "RuntimeError"
    assert "SDK failed" in result.trace["exception_message"]


def test_sdk_exception_reading_final_content_returns_no_text():
    class ResponseWithFailingContent:
        def __getattr__(self, name):
            return getattr(_response(), name)

        @property
        def content(self):
            raise RuntimeError("SDK content accessor failed")

    result, _ = _run(sdk=_mock_anthropic(ResponseWithFailingContent()))
    assert result.text == ""
    assert result.trace["exception_class"] == "RuntimeError"
    assert "SDK content accessor failed" in result.error


def test_budget_expiring_during_preparation_sends_no_request():
    with patch.object(
        evals, "_remaining_eval_case_budget_seconds", side_effect=[1, 1, -1, -1]
    ):
        result, sdk = _run()
    assert result.text == ""
    assert result.timed_out
    assert result.timeout_stage == "case_budget"
    assert sdk.calls == []


def test_case_budget_bounds_the_stream_and_reports_timeout_fields():
    sdk = _mock_anthropic(_response(), stall=True)
    with evals._active_eval_case_budget(0.01):
        result, _ = _run(sdk=sdk)
    assert result.text == ""
    assert result.timed_out
    assert result.timeout_stage == "case_budget"
    assert result.timeout_reason == "wall"
    assert result.timeout_attempts == 1
    assert 0 < result.timeout_seconds <= 0.01
    assert sdk.clients[0].kwargs["timeout"] <= 0.01
    assert result.trace["timed_out"] is True
    assert result.trace["timeout_stage"] == "case_budget"


def test_expired_budget_sends_no_request():
    with (
        patch.object(evals, "_remaining_eval_case_budget_seconds", return_value=-1),
        evals._active_eval_case_budget(1),
    ):
        result, sdk = _run()
    assert result.text == ""
    assert result.timed_out
    assert result.timeout_stage == "case_budget"
    assert sdk.calls == []


def test_missing_key_does_not_construct_a_client():
    sdk = _mock_anthropic(_response())
    with (
        patch.dict(sys.modules, {"anthropic": sdk}),
        patch.dict("os.environ", {}, clear=True),
    ):
        result = evals._run_claude_api_prompt_eval(
            evals.EvalRunnerSpec("api", "claude-api", "claude-opus-5-5"),
            SimpleNamespace(root=Path("synthetic-workspace")),
            "prompt",
        )
    assert result.text == ""
    assert "ANTHROPIC_API_KEY" in result.error
    assert sdk.clients == []


@settings(max_examples=60, deadline=None)
@given(
    model=st.sampled_from(["claude-opus-5-5", "claude-sonnet-5-5"]),
    counts=st.lists(st.integers(0, 100_000), min_size=5, max_size=5),
    coordinate=st.integers(0, 4),
    increase=st.integers(0, 100_000),
)
def test_cost_is_nonnegative_zero_and_monotone_for_disjoint_token_counts(
    model, counts, coordinate, increase
):
    """I4: grow each billed token population, keeping cached input within total."""

    def cost(parts):
        uncached, output, read, create, reasoning = parts
        return estimate_usage_cost_usd(
            model,
            TokenUsage(
                input_tokens=uncached + read + create,
                output_tokens=output,
                cache_read_tokens=read,
                cache_creation_tokens=create,
                reasoning_output_tokens=reasoning,
            ),
        )

    before = cost(counts)
    larger = list(counts)
    larger[coordinate] += increase
    assert before >= 0
    assert cost([0] * 5) == 0
    assert cost(larger) >= before


@settings(max_examples=30, deadline=None)
@given(
    model=st.sampled_from(["claude-opus-5-5", "claude-sonnet-5-5"]),
    counts=st.lists(st.integers(0, 100_000), min_size=5, max_size=5),
    coordinate=st.integers(0, 4),
    increase=st.integers(1, 100_000),
)
def test_provider_token_counters_normalize_and_price_monotonically(
    model, counts, coordinate, increase
):
    """I4: each independent SDK counter grows without reducing the estimate."""

    def cost(parts):
        uncached, output, read, create, reasoning = parts
        response = _response()
        response.usage = SimpleNamespace(
            input_tokens=uncached,
            output_tokens=output,
            cache_read_input_tokens=read,
            cache_creation_input_tokens=create,
            reasoning_output_tokens=reasoning,
        )
        result, _ = _run(response=response, model=model)
        assert result.error is None
        assert result.tokens == TokenUsage(
            input_tokens=uncached + read + create,
            output_tokens=output,
            cache_read_tokens=read,
            cache_creation_tokens=create,
            reasoning_output_tokens=reasoning,
        )
        assert result.estimated_cost_usd == estimate_usage_cost_usd(
            model, result.tokens
        )
        return result.estimated_cost_usd

    before = cost(counts)
    larger = list(counts)
    larger[coordinate] += increase
    assert before >= 0
    assert cost([0] * 5) == 0
    assert cost(larger) >= before

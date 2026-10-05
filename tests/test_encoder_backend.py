"""
Tests for encoder backend abstraction.

Updated for self-contained backends (no plugin dependencies).
"""

import hashlib
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

# Import what we're testing
from axiom_encode.harness.backends import (
    AgentSDKBackend,
    ClaudeCodeBackend,
    CodexCLIBackend,
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
)
from axiom_encode.harness.encoding_db import TokenUsage
from axiom_encode.harness.pricing import estimate_usage_cost_usd
from axiom_encode.prompts import ENCODER_PROMPT, get_encoder_prompt
from axiom_encode.prompts.encoder import (
    _FORMULA_PROTOCOL,
    _TESTS_PROTOCOL,
    _assemble,
)


def test_assembled_prompt_has_no_duplicated_or_glued_blocks():
    """Regression guards for the protocol-block assembly (codex review).

    The prompt is composed by concatenating named protocol blocks; these
    assert the *assembled* string is clean, since a bad block boundary or a
    mis-partitioned block is invisible when you only inspect the blocks.
    """
    # Output-shape and prose-ban guidance must appear exactly once.
    assert ENCODER_PROMPT.count("Emit only RuleSpec YAML") == 1
    assert ENCODER_PROMPT.count("Do not emit Python code, markdown fences") == 1
    # No duplicated heading from a block that re-states the core header.
    assert "Hard requirements:Hard requirements:" not in ENCODER_PROMPT
    assert ENCODER_PROMPT.count("Hard requirements:") == 1
    # No block boundary glued two sentences together (the SOURCE_SCOPE /
    # composition seam previously produced "proxy.- If source text").
    assert "proxy.- If source text" not in ENCODER_PROMPT


def test_formula_and_tests_protocols_are_distinct_and_populated():
    assert _FORMULA_PROTOCOL.strip(), "_FORMULA_PROTOCOL must not be empty"
    assert "Put formulas under" in _FORMULA_PROTOCOL
    assert "min(uncapped_amount, max(cap_a," in _FORMULA_PROTOCOL
    # Test guidance lives in the tests protocol, formula guidance does not.
    assert "Emit only RuleSpec YAML" in _TESTS_PROTOCOL
    assert "Put formulas under" not in _TESTS_PROTOCOL


def test_assemble_strips_blocks_and_cannot_glue_or_stack():
    assembled = _assemble("- alpha statement.", "  - beta statement.  ", "", "- gamma.")
    assert assembled == "- alpha statement.\n\n- beta statement.\n\n- gamma."
    # No empty block leaves a triple newline, and no boundary glues words.
    assert "\n\n\n" not in assembled
    assert "statement.- beta" not in assembled


@pytest.fixture(autouse=True)
def mock_sdk_env(tmp_path):
    """Provide API key for AgentSDKBackend tests."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        yield


class TestEncoderBackendInterface:
    """Test the abstract backend interface."""

    def test_backend_is_abstract(self):
        """EncoderBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EncoderBackend()

    def test_request_dataclass(self):
        """EncoderRequest holds encoding inputs."""
        req = EncoderRequest(
            citation="26 USC 32",
            source_text="The earned income tax credit...",
            output_path=Path("/tmp/test.yaml"),
        )
        assert req.citation == "26 USC 32"
        assert req.source_text.startswith("The earned")
        assert req.output_path == Path("/tmp/test.yaml")

    def test_response_dataclass(self):
        """EncoderResponse holds encoding outputs."""
        resp = EncoderResponse(
            rulespec_content="eitc:\n  entity: TaxUnit",
            success=True,
            error=None,
            duration_ms=1500,
        )
        assert resp.success
        assert "eitc" in resp.rulespec_content
        assert resp.duration_ms == 1500


def _assert_encoder_prompt_topics(prompt: str) -> None:
    """Check durable prompt themes without freezing every wording tweak."""
    required_topics = [
        "module.proof_validation.required: true",
        "Do not emit `source_url`",
        "every source-stated amount, rate, threshold, cap, and limit",
        '"per taxpayer per beneficiary"',
        "Do not treat the final interval row as open-ended",
        "Do not invent synthetic row-number constants for text-label tables",
        "Do not use an `indexed_by` table with numeric keys merely to encode rows whose",
        "Use a negative sentinel such as `-1`",
        "source text `133%` should be",
        "kind: derived_relation",
        '"This source is about SNAP" is not enough',
        "Any rule that uses `entity: <filtered-entity>`",
        "Treat stale unit-scoped imports for that base as",
        "self-employment, wage, net-earnings, compensation, remuneration",
        "earned income of an individual shall be",
        "Never drop the jurisdiction prefix",
        "Importing an adjacent upstream output only as proof",
        "is not an executable dependency",
        "RuleSpec document-root `inputs`",
        "never nested under `module`",
        "purpose-limited replacement rate",
        "not as `section_<cited>_*`",
        "predicate for the excepted category",
        "`subject to`, or `notwithstanding`",
        "`subject to paragraph (c)`",
        "toggle each gate at least once",
        "Validation fails if a direct local `#input.*_exception_applies`",
        "Compute each expected `output:` value by evaluating the emitted RuleSpec",
        "flat\n  threshold with a percentage of excess income",
        "`clause_ii_provides_otherwise`",
        "paragraph_d_2_methodology_limit_satisfied",
        "dependency graph remains acyclic",
        "copied context file already exports the operative legal condition",
        "Do not\n  recreate it as a local factual input",
        "Axiom formulas have no date literal type",
        "Never use `post_YYYY`, `pre_YYYY`, `after_YYYY`, `before_YYYY`",
        "Do not write `else if` or `elif`",
        "min(source_amount, cap)",
        "compute the uncapped base amount separately",
        "min(uncapped_amount, max(cap_a,",
        "Do not\n  reconstruct the cited section's amount locally",
        "Only include `blocked_by` entries when you know the exact RuleSpec output",
        "us:statutes/us-ca/17000",
        "direct release-bound corpus source text",
    ]
    for topic in required_topics:
        assert topic in prompt


def test_generic_encoder_prompt_includes_durable_rule_spec_guidance():
    prompt = get_encoder_prompt(
        citation="26 USC 63(c)(5)",
        output_path="statutes/26/63/c/5.yaml",
        corpus_citation_path="us/statute/26/63",
    )

    _assert_encoder_prompt_topics(ENCODER_PROMPT)
    _assert_encoder_prompt_topics(prompt)
    assert "For 26 USC 1402(a)(12)" not in ENCODER_PROMPT
    normalized_prompt = " ".join(prompt.split())
    assert "Include `us/statute/26/63` in `module.source_verification`." in prompt
    assert (
        "module.source_verification.corpus_citation_path: us/statute/26/63"
        in normalized_prompt
    )
    assert (
        "Use that exact same `us/statute/26/63` value in every source-backed proof"
        in prompt
    )
    assert "do not append subsection markers or other path segments" in prompt
    assert "Never emit `corpus_citation_paths`" in normalized_prompt
    assert "corpus resolver under this one canonical path" in normalized_prompt
    assert "Target citation/source id: 26 USC 63(c)(5)" in prompt
    assert "Expected output path: statutes/26/63/c/5.yaml" in prompt


def test_complete_source_prompt_requires_explicit_branch_inventory():
    prompt = get_encoder_prompt(
        citation="42 USC 1437c-1",
        output_path="statutes/42/1437c-1.yaml",
        corpus_citation_path="us/statute/42/1437c–1",
        require_complete_source_unit=True,
    )
    normalized_prompt = " ".join(prompt.split())

    assert "inventory every top-level structural branch" in normalized_prompt
    assert "Do not return while any branch is absent" in normalized_prompt
    assert "absolute output path preserves the branch label" in normalized_prompt


class TestClaudeCodeBackend:
    """Test the Claude Code CLI backend (subprocess approach)."""

    def test_backend_inherits_interface(self):
        """ClaudeCodeBackend implements EncoderBackend."""
        backend = ClaudeCodeBackend()
        assert isinstance(backend, EncoderBackend)

    def test_encode_calls_subprocess(self):
        """encode() calls claude CLI via subprocess."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert mock_run.called
            # Should use 'claude' command
            cmd = mock_run.call_args[0][0]
            assert "claude" in cmd

    def test_encode_uses_embedded_prompt(self):
        """encode() uses embedded encoder prompt (no plugin agent)."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            cmd = mock_run.call_args[0][0]
            # Should NOT include --agent or --plugin-dir (self-contained)
            assert "--agent" not in cmd
            assert "--plugin-dir" not in cmd
            # Should include --print and -p flags
            assert "--print" in cmd
            assert "-p" in cmd

    def test_encode_confines_claude_to_output_directory(self, tmp_path):
        backend = ClaudeCodeBackend(cwd=tmp_path / "mutable-checkout")
        output = tmp_path / "isolated-output" / "rule.yaml"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="rules: []", stderr="", returncode=0)
            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output,
                )
            )

        cmd = mock_run.call_args[0][0]
        kwargs = mock_run.call_args.kwargs
        assert "bypassPermissions" not in cmd
        assert cmd[cmd.index("--permission-mode") + 1] == "acceptEdits"
        assert cmd[cmd.index("--tools") + 1] == "Read,Write,Edit"
        assert "--safe-mode" in cmd
        assert "--no-session-persistence" in cmd
        assert kwargs["cwd"] == output.parent.resolve()

    def test_predict_returns_scores(self):
        """predict() returns score predictions."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"rulespec_reviewer": 8.0, "formula_reviewer": 7.5, "confidence": 0.8}',
                stderr="",
                returncode=0,
            )

            scores = backend.predict(
                citation="26 USC 32",
                source_text="EITC rules...",
            )

            assert scores.rulespec_reviewer == 8.0
            assert scores.confidence == 0.8


class TestAgentSDKBackend:
    """Test the Claude API backend (anthropic SDK)."""

    def test_backend_inherits_interface(self):
        """AgentSDKBackend implements EncoderBackend."""
        backend = AgentSDKBackend()
        assert isinstance(backend, EncoderBackend)

    def test_requires_api_key(self):
        """AgentSDKBackend requires ANTHROPIC_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove API key from env
            import os

            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AgentSDKBackend(api_key=None)

    @staticmethod
    def _stream_anthropic(
        response, *, events=(), on_final=None, stream_error=None, close_error=None
    ):
        """Mock ``anthropic`` whose AsyncAnthropic().messages.stream() yields response.

        Iterating the stream yields ``events``. Returns the module mock and the
        stream() kwargs seen. The clients the backend opened are on
        ``mock_anthropic.clients``; each one's ``closed`` turns true when its
        ``async with`` block exits, which then raises ``close_error`` if set.
        """

        calls = []
        clients = []

        class _Stream:
            async def __aiter__(self):
                for event in events:
                    yield event

            async def get_final_message(self):
                if on_final is not None:
                    on_final()
                return response

        class _StreamManager:
            async def __aenter__(self):
                return _Stream()

            async def __aexit__(self, *exc):
                return False

        class _Client:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.closed = False
                self.messages = Mock()
                self.messages.stream = self._stream
                clients.append(self)

            def _stream(self, **kwargs):
                calls.append(kwargs)
                if stream_error is not None:
                    raise stream_error
                return _StreamManager()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                self.closed = True
                if close_error is not None:
                    raise close_error
                return False

        mock_anthropic = Mock()
        mock_anthropic.AsyncAnthropic = _Client
        mock_anthropic.clients = clients
        return mock_anthropic, calls

    @staticmethod
    def _response(
        blocks, *, stop_reason="end_turn", stop_details=None, output_tokens=50
    ):
        response = Mock()
        response.content = blocks
        response.usage = Mock(input_tokens=100, output_tokens=output_tokens)
        response.stop_reason = stop_reason
        response.stop_details = stop_details
        return response

    @staticmethod
    def _request(tmp_path):
        return EncoderRequest(
            citation="26 USC 1",
            source_text="Test",
            output_path=tmp_path / "missing.yaml",
        )

    @classmethod
    async def _encode(cls, response, tmp_path, *, events=(), **backend_kwargs):
        """Run one encode against ``response``; return it and the module mock."""
        mock_anthropic, _ = cls._stream_anthropic(response, events=events)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="k", **backend_kwargs).encode_async(
                cls._request(tmp_path)
            )
        return resp, mock_anthropic

    @pytest.mark.asyncio
    async def test_encode_async(self, tmp_path, monkeypatch):
        """encode_async() streams from the anthropic SDK and keeps text blocks only."""
        monkeypatch.delenv("AXIOM_API_ENCODER_EFFORT", raising=False)
        backend = AgentSDKBackend(api_key="test-key")
        response = self._response(
            [
                Mock(type="thinking", thinking=""),
                Mock(type="text", text="test:\n  entity: TaxUnit"),
            ]
        )
        mock_anthropic, calls = self._stream_anthropic(response)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(self._request(tmp_path))

        assert resp.success
        assert resp.error is None
        assert resp.tokens is not None
        assert resp.rulespec_content == "test:\n  entity: TaxUnit"
        assert calls[0]["model"] == "claude-opus-5-5"
        assert calls[0]["max_tokens"] == AgentSDKBackend.MAX_TOKENS
        assert calls[0]["max_tokens"] > 21_333  # needs streaming
        assert calls[0]["output_config"] == {"effort": "high"}
        # One client per encode, and it is closed when the encode returns.
        assert [client.kwargs for client in mock_anthropic.clients] == [
            {"api_key": "test-key"}
        ]
        assert mock_anthropic.clients[0].closed

    @pytest.mark.asyncio
    async def test_encode_async_prices_the_call_and_records_a_trace(self, tmp_path):
        from axiom_encode.harness.pricing import estimate_usage_cost_usd

        resp, _ = await self._encode(
            self._response([Mock(type="text", text="x: 1")]), tmp_path, effort="max"
        )

        assert resp.success
        # Opus 5.5 at $4 / $20 per million: 100 input + 50 output tokens.
        assert resp.cost_usd == pytest.approx(100 * 4.0 / 1e6 + 50 * 20.0 / 1e6)
        assert resp.cost_usd == estimate_usage_cost_usd("claude-opus-5-5", resp.tokens)
        assert resp.trace == {
            "provider": "anthropic",
            "backend": "api",
            "model": "claude-opus-5-5",
            "stop_reason": "end_turn",
            "effort": "max",
        }

    @pytest.mark.asyncio
    async def test_encode_async_unpriced_model_reports_no_cost(self, tmp_path):
        resp, _ = await self._encode(
            self._response([Mock(type="text", text="x: 1")]),
            tmp_path,
            model="claude-unpriced-test-model",
        )

        assert resp.success
        assert resp.tokens is not None
        assert resp.cost_usd is None
        assert resp.trace["model"] == "claude-unpriced-test-model"

    @pytest.mark.asyncio
    async def test_encode_async_effort_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "xhigh")
        mock_anthropic, calls = self._stream_anthropic(
            self._response([Mock(type="text", text="x: 1")])
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            await AgentSDKBackend(api_key="k").encode_async(self._request(tmp_path))
            await AgentSDKBackend(api_key="k", effort="medium").encode_async(
                self._request(tmp_path)
            )
        assert calls[0]["output_config"] == {"effort": "xhigh"}
        assert calls[1]["output_config"] == {"effort": "medium"}

    def test_effort_levels_are_the_api_levels(self):
        from axiom_encode.constants import DEFAULT_API_ENCODER_EFFORT

        assert AgentSDKBackend.EFFORTS == {"low", "medium", "high", "xhigh", "max"}
        assert DEFAULT_API_ENCODER_EFFORT in AgentSDKBackend.EFFORTS

    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_every_api_effort_level_is_accepted(self, effort, monkeypatch):
        monkeypatch.delenv("AXIOM_API_ENCODER_EFFORT", raising=False)
        assert AgentSDKBackend(api_key="k", effort=effort).effort == effort
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", effort)
        assert AgentSDKBackend(api_key="k").effort == effort

    def test_effort_is_normalized_and_blank_means_default(self, monkeypatch):
        monkeypatch.delenv("AXIOM_API_ENCODER_EFFORT", raising=False)
        assert AgentSDKBackend(api_key="k", effort=" XHigh ").effort == "xhigh"
        assert AgentSDKBackend(api_key="k").effort == "high"
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "")
        assert AgentSDKBackend(api_key="k").effort == "high"
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "  ")
        assert AgentSDKBackend(api_key="k").effort == "high"

    def test_blank_effort_argument_defers_to_environment(self, monkeypatch):
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "low")
        assert AgentSDKBackend(api_key="k", effort="  ").effort == "low"
        assert AgentSDKBackend(api_key="k", effort="").effort == "low"
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "turbo")
        with pytest.raises(ValueError, match=r"^AXIOM_API_ENCODER_EFFORT='turbo' "):
            AgentSDKBackend(api_key="k", effort=" ")

    @pytest.mark.parametrize("effort", ["hgih", "minimal", "none", "ultra", "1"])
    def test_invalid_effort_argument_raises(self, effort, monkeypatch):
        monkeypatch.delenv("AXIOM_API_ENCODER_EFFORT", raising=False)
        with pytest.raises(ValueError, match=rf"^effort='{effort}' is not a valid"):
            AgentSDKBackend(api_key="k", effort=effort)

    def test_invalid_effort_environment_value_raises(self, monkeypatch):
        monkeypatch.setenv("AXIOM_API_ENCODER_EFFORT", "turbo")
        with pytest.raises(
            ValueError,
            match=r"^AXIOM_API_ENCODER_EFFORT='turbo' is not a valid effort; "
            r"use one of \['high', 'low', 'max', 'medium', 'xhigh'\]$",
        ):
            AgentSDKBackend(api_key="k")
        # An explicit valid effort wins over a bad environment value.
        assert AgentSDKBackend(api_key="k", effort="low").effort == "low"

    @pytest.mark.asyncio
    async def test_encode_async_refusal_is_a_failed_encode(self, tmp_path):
        response = self._response(
            [], stop_reason="refusal", stop_details={"category": "cyber"}
        )
        mock_anthropic, _ = self._stream_anthropic(response)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="k").encode_async(
                self._request(tmp_path)
            )
        assert not resp.success
        assert resp.rulespec_content == ""
        assert "declined" in resp.error and "cyber" in resp.error

    @pytest.mark.asyncio
    async def test_encode_async_refusal_reads_stop_details_object(self, tmp_path):
        # Newer SDKs type stop_details as a model object, not a dict.
        details = SimpleNamespace(
            type="refusal", category="bio", explanation="declined by classifier"
        )
        resp, _ = await self._encode(
            self._response(
                [Mock(type="text", text="partial: 1")],
                stop_reason="refusal",
                stop_details=details,
            ),
            tmp_path,
        )
        assert not resp.success
        assert resp.rulespec_content == ""
        assert resp.error == "model declined the encode request (category: bio)"

    @pytest.mark.parametrize(
        "stop_details",
        [
            None,  # neither the final message nor the stream carried them
            {"type": "refusal", "category": None, "explanation": None},
            SimpleNamespace(type="refusal", category=None, explanation=None),
        ],
        ids=["none", "dict-without-category", "object-without-category"],
    )
    @pytest.mark.asyncio
    async def test_encode_async_refusal_without_category(self, tmp_path, stop_details):
        resp, _ = await self._encode(
            self._response([], stop_reason="refusal", stop_details=stop_details),
            tmp_path,
        )
        assert not resp.success
        assert resp.rulespec_content == ""
        assert resp.error == "model declined the encode request"
        assert resp.trace["refusal_category"] is None

    @pytest.mark.asyncio
    async def test_encode_async_refusal_reads_stop_details_from_the_stream(
        self, tmp_path
    ):
        # anthropic 0.83.0 leaves stop_details off the final message; the
        # message_delta event still carries them, as a plain dict.
        events = [
            SimpleNamespace(type="message_start"),
            SimpleNamespace(type="text", text="ignored"),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(
                    stop_reason="refusal",
                    stop_details={
                        "type": "refusal",
                        "category": "bio",
                        "explanation": "declined",
                    },
                ),
            ),
            SimpleNamespace(type="message_stop"),
        ]
        resp, _ = await self._encode(
            self._response([], stop_reason="refusal", output_tokens=0),
            tmp_path,
            events=events,
        )
        assert not resp.success
        assert resp.error == "model declined the encode request (category: bio)"
        assert resp.trace["refusal_category"] == "bio"
        # A bio refusal is billed even before any output.
        assert resp.cost_usd == pytest.approx(100 * 4.0 / 1e6)

    @pytest.mark.asyncio
    async def test_encode_async_final_stop_details_win_over_the_stream(self, tmp_path):
        events = [
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(
                    stop_reason="refusal", stop_details={"category": "bio"}
                ),
            )
        ]
        resp, _ = await self._encode(
            self._response(
                [],
                stop_reason="refusal",
                stop_details=SimpleNamespace(type="refusal", category="cyber"),
                output_tokens=0,
            ),
            tmp_path,
            events=events,
        )
        assert resp.error == "model declined the encode request (category: cyber)"
        assert resp.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_unbilled_refusal_reports_its_usage_at_no_cost(self, tmp_path):
        # Anthropic's example refusal: a cyber decline before any output. It is
        # not billed, but usage still carries its token counts.
        resp, _ = await self._encode(
            self._response(
                [],
                stop_reason="refusal",
                stop_details={
                    "type": "refusal",
                    "category": "cyber",
                    "explanation": "declined",
                },
                output_tokens=0,
            ),
            tmp_path,
        )
        assert not resp.success
        assert (resp.tokens.input_tokens, resp.tokens.output_tokens) == (100, 0)
        assert resp.cost_usd == 0.0
        assert resp.trace["refusal_category"] == "cyber"

    @pytest.mark.parametrize("output_tokens", [0, 1, 50])
    @pytest.mark.parametrize("delivery", ["final", "stream", "absent"])
    @pytest.mark.parametrize("shape", ["dict", "object"])
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
    @pytest.mark.asyncio
    async def test_refusal_cost_invariants(
        self, tmp_path, category, shape, delivery, output_tokens
    ):
        """Exhaustive over categories x stop_details shapes x sources x output.

        A refusal always fails and reports its usage. Its cost is the full
        estimate when Anthropic bills it (after any output, or before output in
        a billed category), 0.0 when it does not (a free or null category), and
        None when it came before any output without stop_details or in a
        category Anthropic's billing table does not list, so the bill is
        unknown.
        """
        details = (
            {"type": "refusal", "category": category}
            if shape == "dict"
            else SimpleNamespace(type="refusal", category=category)
        )
        events = ()
        if delivery == "stream":
            events = [
                SimpleNamespace(
                    type="message_delta",
                    delta=SimpleNamespace(stop_reason="refusal", stop_details=details),
                )
            ]
        resp, _ = await self._encode(
            self._response(
                [],
                stop_reason="refusal",
                stop_details=details if delivery == "final" else None,
                output_tokens=output_tokens,
            ),
            tmp_path,
            events=events,
        )

        known = delivery != "absent"
        billed = {"bio", "frontier_llm", "reasoning_extraction"}
        free = {"cyber", "general_harms", None}
        full = estimate_usage_cost_usd(
            "claude-opus-5-5",
            TokenUsage(input_tokens=100, output_tokens=output_tokens),
        )
        if output_tokens > 0 or (known and category in billed):
            expected = full
        elif known and category in free:
            expected = 0.0
        else:
            expected = None
        assert not resp.success
        assert resp.rulespec_content == ""
        assert (resp.tokens.input_tokens, resp.tokens.output_tokens) == (
            100,
            output_tokens,
        )
        assert resp.cost_usd == expected
        seen = category if known else None
        assert resp.trace["refusal_category"] == seen
        assert resp.error == "model declined the encode request" + (
            f" (category: {seen})" if seen else ""
        )

    @pytest.mark.asyncio
    async def test_encode_async_truncation_is_a_failed_encode(self, tmp_path):
        # A truncated RuleSpec must never be returned as content, even if it
        # looks well-formed so far.
        response = self._response(
            [Mock(type="text", text="test:\n  entity: Tax")], stop_reason="max_tokens"
        )
        mock_anthropic, _ = self._stream_anthropic(response)
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="k").encode_async(
                self._request(tmp_path)
            )
        assert not resp.success
        assert resp.rulespec_content == ""
        assert "max_tokens" in resp.error

    @pytest.mark.asyncio
    async def test_encode_async_context_window_exceeded_is_a_failed_encode(
        self, tmp_path
    ):
        resp, _ = await self._encode(
            self._response(
                [Mock(type="text", text="test:\n  entity: Tax")],
                stop_reason="model_context_window_exceeded",
            ),
            tmp_path,
        )
        assert not resp.success
        assert resp.rulespec_content == ""
        assert "context window" in resp.error
        assert "100 input tokens" in resp.error
        assert "max_tokens" not in resp.error
        assert resp.trace["stop_reason"] == "model_context_window_exceeded"

    @pytest.mark.parametrize(
        "blocks",
        [
            [],
            [Mock(type="thinking", thinking="")],
            [Mock(type="thinking", thinking=""), Mock(type="text", text=" \n\t")],
        ],
        ids=["no-blocks", "thinking-only", "blank-text"],
    )
    @pytest.mark.asyncio
    async def test_encode_async_empty_end_turn_is_a_failed_encode(
        self, tmp_path, blocks
    ):
        resp, _ = await self._encode(self._response(blocks), tmp_path)
        assert not resp.success
        assert resp.rulespec_content == ""
        assert resp.error == "response had no RuleSpec text (stop_reason: end_turn)"

    @pytest.mark.parametrize(
        ("stop_reason", "stop_details", "blocks"),
        [
            # A refusal after some output bills that output and the input.
            ("refusal", {"category": "cyber"}, [Mock(type="text", text="a:")]),
            ("max_tokens", None, [Mock(type="text", text="a: 1")]),
            ("model_context_window_exceeded", None, [Mock(type="text", text="a")]),
            ("end_turn", None, [Mock(type="thinking", thinking="")]),
        ],
        ids=["refusal-after-output", "max-tokens", "context-window", "empty"],
    )
    @pytest.mark.asyncio
    async def test_failed_encode_still_reports_usage_cost_and_trace(
        self, tmp_path, stop_reason, stop_details, blocks
    ):
        # Each of these calls is billed for its 100 input and 50 output tokens
        # though the encode failed, so the usage, cost and trace must survive
        # into the failed response.
        resp, mock_anthropic = await self._encode(
            self._response(blocks, stop_reason=stop_reason, stop_details=stop_details),
            tmp_path,
        )
        assert not resp.success
        assert resp.tokens is not None
        assert (resp.tokens.input_tokens, resp.tokens.output_tokens) == (100, 50)
        assert resp.cost_usd == pytest.approx(0.0014)
        assert resp.trace["provider"] == "anthropic"
        assert resp.trace["stop_reason"] == stop_reason
        assert mock_anthropic.clients[0].closed

    @pytest.mark.parametrize(
        "stop_reason",
        [
            "end_turn",
            "stop_sequence",
            "refusal",
            "max_tokens",
            "model_context_window_exceeded",
        ],
    )
    @pytest.mark.parametrize(
        "blocks",
        [
            [],
            [Mock(type="thinking", thinking="")],
            [Mock(type="text", text="  \n")],
            [Mock(type="thinking", thinking=""), Mock(type="text", text="a: 1")],
            [Mock(type="text", text="a:"), Mock(type="text", text=" 1")],
        ],
        ids=["none", "thinking", "blank", "thinking+text", "split-text"],
    )
    @pytest.mark.asyncio
    async def test_encode_outcome_invariants(self, tmp_path, stop_reason, blocks):
        """Exhaustive over stop reasons x content shapes.

        A response succeeds exactly when the model finished on its own and left
        non-blank text; a failed response never carries RuleSpec content; and
        once the API answered, tokens, cost and trace are always reported.
        """
        resp, _ = await self._encode(
            self._response(blocks, stop_reason=stop_reason), tmp_path
        )
        text = "".join(
            block.text for block in blocks if getattr(block, "type", None) == "text"
        )
        finished = stop_reason in ("end_turn", "stop_sequence")
        assert resp.success == (finished and bool(text.strip()))
        if resp.success:
            assert resp.error is None
            assert resp.rulespec_content == text
        else:
            assert resp.error
            assert resp.rulespec_content == ""
        assert resp.tokens is not None and resp.cost_usd is not None
        assert resp.trace["stop_reason"] == stop_reason

    @pytest.mark.asyncio
    async def test_failure_while_closing_still_reports_the_answer(self, tmp_path):
        mock_anthropic, _ = self._stream_anthropic(
            self._response([Mock(type="text", text="a: 1")]),
            close_error=RuntimeError("connection pool close failed"),
        )
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="k").encode_async(
                self._request(tmp_path)
            )
        assert not resp.success
        assert resp.error == "connection pool close failed"
        assert (resp.tokens.input_tokens, resp.tokens.output_tokens) == (100, 50)
        assert resp.cost_usd == pytest.approx(0.0014)
        assert resp.trace["stop_reason"] == "end_turn"
        assert mock_anthropic.clients[0].closed

    def test_default_model_is_current_opus(self):
        from axiom_encode.constants import DEFAULT_MODEL

        assert DEFAULT_MODEL == "claude-opus-5-5"
        assert AgentSDKBackend(api_key="k").model == "claude-opus-5-5"

    @pytest.mark.asyncio
    async def test_encode_batch_parallel(self):
        """encode_batch() runs multiple encodings in parallel."""
        backend = AgentSDKBackend(api_key="test-key")

        requests = [
            EncoderRequest(
                citation=f"26 USC {i}",
                source_text=f"Statute {i}",
                output_path=Path(f"/tmp/test{i}.yaml"),
            )
            for i in range(5)
        ]

        with patch.object(backend, "encode_async") as mock_encode:
            mock_encode.return_value = EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            responses = await backend.encode_batch(requests, max_concurrent=3)

            # All 5 should complete
            assert len(responses) == 5
            # encode_async should be called 5 times
            assert mock_encode.call_count == 5

    @pytest.mark.asyncio
    async def test_encode_batch_respects_concurrency_limit(self):
        """encode_batch() respects max_concurrent limit."""
        backend = AgentSDKBackend(api_key="test-key")
        concurrent_count = 0
        max_seen = 0

        async def track_concurrency(req):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            import asyncio

            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=10,
            )

        with patch.object(backend, "encode_async", side_effect=track_concurrency):
            requests = [
                EncoderRequest(
                    citation=f"26 USC {i}",
                    source_text=f"Statute {i}",
                    output_path=Path(f"/tmp/test{i}.yaml"),
                )
                for i in range(10)
            ]

            await backend.encode_batch(requests, max_concurrent=3)

            # Should never exceed max_concurrent
            assert max_seen <= 3


class TestClaudeCodeBackendAdditional:
    """Additional tests for ClaudeCodeBackend to cover missing lines."""

    def test_encode_with_nonzero_returncode(self):
        """Test encode returns error when CLI returns non-zero."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="Error: failed to encode",
                stderr="",
                returncode=1,
            )

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert not resp.success
            assert resp.error is not None
            assert resp.rulespec_content == ""

    def test_encode_reads_file_when_exists(self, tmp_path):
        """Test encode reads from output_path when file exists."""
        backend = ClaudeCodeBackend()
        output_path = tmp_path / "output.yaml"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="CLI output ignored",
                stderr="",
                returncode=0,
            )
            # Create the file as if Claude wrote it
            output_path.write_text("file_var:\n  entity: TaxUnit\n")

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output_path,
                )
            )

            assert resp.success
            assert "file_var" in resp.rulespec_content

    def test_predict_no_json_in_output(self):
        """Test predict returns defaults when no JSON found in output."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="No JSON here, just plain text",
                stderr="",
                returncode=0,
            )

            scores = backend.predict("26 USC 1", "Statute text")
            # Should return defaults on error
            assert scores.confidence == 0.3

    def test_predict_returns_defaults_on_exception(self):
        """Test predict returns defaults when exception occurs."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("unexpected error")

            scores = backend.predict("26 USC 1", "Statute text")
            assert scores.confidence == 0.3

    def test_run_claude_code_uses_print_flag(self):
        """Test _run_claude_code uses --print flag (self-contained, no plugin)."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="test", stderr="", returncode=0)

            backend._run_claude_code("test prompt")

            cmd = mock_run.call_args[0][0]
            assert "--print" in cmd
            assert "--plugin-dir" not in cmd
            # Current Claude CLI versions validate --mcp-config as a record
            # with an mcpServers key; a bare "{}" fails every invocation.
            assert cmd[cmd.index("--mcp-config") + 1] == '{"mcpServers": {}}'

    def test_run_claude_code_timeout(self):
        """Test _run_claude_code handles timeout."""
        import subprocess

        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)

            output, code = backend._run_claude_code("test", timeout=300)
            assert "Timeout" in output
            assert code == 1

    def test_run_claude_code_file_not_found(self):
        """Test _run_claude_code handles missing CLI."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output, code = backend._run_claude_code("test")
            assert "not found" in output
            assert code == 1

    def test_run_claude_code_generic_error(self):
        """Test _run_claude_code handles generic exception."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            output, code = backend._run_claude_code("test")
            assert "Error" in output
            assert code == 1


class TestAgentSDKBackendAdditional:
    """Additional tests for AgentSDKBackend to cover missing lines."""

    @pytest.mark.asyncio
    async def test_encode_async_with_usage(self):
        """Test encode_async captures token usage."""
        backend = AgentSDKBackend(api_key="test-key")

        mock_response = TestAgentSDKBackend._response(
            [Mock(type="text", text="encoded")]
        )
        mock_anthropic, _ = TestAgentSDKBackend._stream_anthropic(mock_response)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/nonexistent.yaml"),
                )
            )

            assert resp.success
            assert resp.tokens is not None
            assert resp.tokens.input_tokens == 100
            assert resp.tokens.output_tokens == 50

    @pytest.mark.asyncio
    async def test_encode_async_ignores_stale_output_file(self, tmp_path):
        """A file left at output_path by an earlier run never overrides the reply.

        The backend gives the model no tools, so it cannot have written the file.
        """
        backend = AgentSDKBackend(api_key="test-key")

        output_path = tmp_path / "output.yaml"
        output_path.write_text("stale_file:\n  entity: TaxUnit\n")

        mock_response = TestAgentSDKBackend._response(
            [Mock(type="text", text="fresh_reply:\n  entity: TaxUnit")]
        )
        mock_anthropic, _ = TestAgentSDKBackend._stream_anthropic(mock_response)

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=output_path,
                )
            )

        assert resp.success
        assert resp.rulespec_content == "fresh_reply:\n  entity: TaxUnit"
        assert output_path.read_text() == "stale_file:\n  entity: TaxUnit\n"

    @pytest.mark.asyncio
    async def test_encode_async_stale_file_cannot_rescue_an_empty_reply(self, tmp_path):
        output_path = tmp_path / "output.yaml"
        output_path.write_text("stale_file:\n  entity: TaxUnit\n")
        mock_anthropic, _ = TestAgentSDKBackend._stream_anthropic(
            TestAgentSDKBackend._response([Mock(type="thinking", thinking="")])
        )

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="test-key").encode_async(
                EncoderRequest(
                    citation="26 USC 1", source_text="Test", output_path=output_path
                )
            )

        assert not resp.success
        assert resp.rulespec_content == ""
        assert "no RuleSpec text" in resp.error

    @pytest.mark.asyncio
    async def test_encode_async_reads_output_file_written_during_request(
        self, tmp_path
    ):
        output_path = tmp_path / "output.yaml"
        output_path.write_text("stale_file:\n  entity: TaxUnit\n")

        def write_during_request():
            output_path.write_text("fresh_file:\n  entity: TaxUnit\n")
            # Kernels stamp mtimes from a coarse clock that can trail
            # time.time() by a few milliseconds; push the stamp clear of the
            # request start so the test cannot flake on that.
            later = time.time() + 5
            os.utime(output_path, (later, later))

        mock_anthropic, _ = TestAgentSDKBackend._stream_anthropic(
            TestAgentSDKBackend._response([Mock(type="text", text="reply: 1")]),
            on_final=write_during_request,
        )

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await AgentSDKBackend(api_key="test-key").encode_async(
                EncoderRequest(
                    citation="26 USC 1", source_text="Test", output_path=output_path
                )
            )

        assert resp.success
        assert resp.rulespec_content == "fresh_file:\n  entity: TaxUnit\n"

    @pytest.mark.asyncio
    async def test_encode_async_import_error(self):
        """Test encode_async handles missing anthropic import."""
        backend = AgentSDKBackend(api_key="test-key")

        import builtins

        orig_import = builtins.__import__

        def no_anthropic_import(name, *args, **kwargs):
            if name == "anthropic":
                raise ImportError("No module named 'anthropic'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_anthropic_import):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert not resp.success
            assert resp.error.startswith(
                "anthropic SDK not installed; install axiom-encode[api]"
            )
            assert resp.tokens is None and resp.cost_usd is None

    @pytest.mark.asyncio
    async def test_encode_async_other_import_error_is_not_a_missing_sdk(self):
        mock_anthropic, calls = TestAgentSDKBackend._stream_anthropic(None)
        with (
            patch.dict("sys.modules", {"anthropic": mock_anthropic}),
            patch(
                "axiom_encode.harness.backends.get_encoder_prompt",
                side_effect=ImportError("cannot import name 'x'"),
            ),
        ):
            resp = await AgentSDKBackend(api_key="test-key").encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

        assert not resp.success
        assert resp.error == "cannot import name 'x'"
        assert calls == []

    @pytest.mark.asyncio
    async def test_encode_async_generic_error(self):
        """Test encode_async handles generic exception."""
        backend = AgentSDKBackend(api_key="test-key")

        mock_anthropic, _ = TestAgentSDKBackend._stream_anthropic(
            None, stream_error=RuntimeError("Connection failed")
        )

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

        assert not resp.success
        assert "Connection failed" in resp.error
        # No answer, so nothing was billed, but the client is still closed.
        assert resp.tokens is None and resp.cost_usd is None
        assert resp.trace["provider"] == "anthropic"
        assert resp.trace["stop_reason"] is None
        assert mock_anthropic.clients[0].closed


class TestAgentSDKPrediction:
    """Test AgentSDKBackend.predict() method."""

    def test_predict_returns_default_scores(self):
        """Test predict returns default PredictionScores."""
        backend = AgentSDKBackend(api_key="test-key")
        scores = backend.predict("26 USC 1", "Statute text")
        assert scores.confidence == 0.5


class TestCodexCLIBackend:
    """Test the Codex CLI backend."""

    def test_encode_calls_codex_exec(self):
        backend = CodexCLIBackend(cwd=Path("/tmp/work"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"type":"item.completed","item":{"type":"agent_message","text":"test:\\n  entity: TaxUnit"}}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":2}}',
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/output/test.yaml"),
                    model="gpt-5.4",
                )
            )

            cmd = mock_run.call_args[0][0]
            assert Path(cmd[0]).name == "codex"
            assert cmd[1:3] == ["exec", "--json"]
            assert "--model" in cmd
            assert "gpt-5.4" in cmd
            assert "--add-dir" not in cmd
            assert cmd[cmd.index("-C") + 1] == str(Path("/tmp/output").resolve())
            assert mock_run.call_args.kwargs["cwd"] == Path("/tmp/output").resolve()

    def test_trusted_subscription_records_pinned_cli_provenance(
        self, tmp_path, monkeypatch
    ):
        binary = tmp_path / "codex"
        binary.write_bytes(b"pinned-codex")
        monkeypatch.setattr(
            "axiom_encode.harness.backends.resolve_codex_cli", lambda: str(binary)
        )
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "runtime-home"))
        digest = hashlib.sha256(b"pinned-codex").hexdigest()
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_VERSION", "codex-cli 0.test")
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_SHA256", digest)
        version, digest = CodexCLIBackend._trusted_cli_provenance()
        assert version == "codex-cli 0.test"
        assert digest == hashlib.sha256(b"pinned-codex").hexdigest()

    def test_trusted_subscription_executes_bound_cli_not_path_decoy(
        self, tmp_path, monkeypatch
    ):
        trusted = tmp_path / "trusted-codex"
        trusted.write_text("#!/bin/sh\nexit 0\n")
        trusted.chmod(0o700)
        decoy_dir = tmp_path / "decoy-bin"
        decoy_dir.mkdir()
        decoy_marker = tmp_path / "decoy-executed"
        decoy = decoy_dir / "codex"
        decoy.write_text(f"#!/bin/sh\ntouch {decoy_marker}\nexit 99\n")
        decoy.chmod(0o700)
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", str(tmp_path / "runtime-home"))
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_BIN", str(trusted))
        monkeypatch.setenv("PATH", f"{decoy_dir}{os.pathsep}{os.environ['PATH']}")

        output, returncode = CodexCLIBackend()._run_codex_exec(
            "prompt", "gpt-test", 10, tmp_path
        )

        assert returncode == 0, output
        assert not decoy_marker.exists()

    def test_encode_parses_jsonl_usage(self):
        backend = CodexCLIBackend(cwd=Path("/tmp/work"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"type":"event_msg","payload":{"type":"token_count","info":{"total_token_usage":{"reasoning_output_tokens":7}}}}\n{"type":"item.completed","item":{"type":"agent_message","text":"test:\\n  entity: TaxUnit"}}\n{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5,"cached_input_tokens":2}}',
                stderr="",
                returncode=0,
            )

            response = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test statute",
                    output_path=Path("/tmp/output/test.yaml"),
                    model="gpt-5.4",
                )
            )

            assert response.success
            assert response.tokens is not None
            assert response.tokens.input_tokens == 10
            assert response.tokens.output_tokens == 5
            assert response.tokens.cache_read_tokens == 2
            assert response.tokens.reasoning_output_tokens == 7
            assert response.trace is not None
            assert response.trace["provider"] == "openai"


class TestBackendContract:
    """Test that each backend returns the shared response contract."""

    def test_both_backends_return_encoder_response(self):
        """Both backends return EncoderResponse from encode()."""
        # This ensures the abstraction is clean

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            cli_backend = ClaudeCodeBackend()
            cli_resp = cli_backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert isinstance(cli_resp, EncoderResponse)

    def test_sync_wrapper_for_sdk_backend(self):
        """AgentSDKBackend.encode() provides sync wrapper."""
        backend = AgentSDKBackend(api_key="test-key")

        with patch.object(backend, "encode_async") as mock_async:
            mock_async.return_value = EncoderResponse(
                rulespec_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            # Sync encode() should work
            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    source_text="Test",
                    output_path=Path("/tmp/test.yaml"),
                )
            )

            assert isinstance(resp, EncoderResponse)


class TestCodexAuthPreflight:
    """Codex-backend auth preflight helpers (encode#1054)."""

    def test_auth_path_defaults_to_home_codex(self):
        from axiom_encode.codex_cli import codex_auth_json_path

        with patch.dict(os.environ, {}, clear=True):
            assert codex_auth_json_path() == Path.home() / ".codex" / "auth.json"

    def test_trusted_runtime_resolution_uses_supervisor_bound_path(self, monkeypatch):
        from axiom_encode.codex_cli import resolve_codex_cli

        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_RUNTIME", "1")
        monkeypatch.setenv("CODEX_HOME", "/protected/runtime-codex-home")
        monkeypatch.setenv("AXIOM_ENCODE_CODEX_BIN", "/hostile/override")
        monkeypatch.setenv("AXIOM_ENCODE_TRUSTED_CODEX_BIN", "/trusted/bin/codex")
        assert resolve_codex_cli() == "/trusted/bin/codex"

    def test_auth_path_honors_codex_home(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_json_path

        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            assert codex_auth_json_path() == tmp_path / "auth.json"

    def test_auth_error_when_no_file_and_no_key(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            error = codex_auth_error()
        assert error is not None
        assert "Codex backend requires authentication" in error
        assert str(tmp_path / "auth.json") in error

    def test_auth_ok_when_file_present(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        (tmp_path / "auth.json").write_text('{"OPENAI_API_KEY": "sk-test"}\n')
        with patch.dict(os.environ, {"CODEX_HOME": str(tmp_path)}, clear=True):
            assert codex_auth_error() is None

    def test_auth_ok_when_openai_api_key_set(self, tmp_path):
        from axiom_encode.codex_cli import codex_auth_error

        # No auth.json, but OPENAI_API_KEY is enough for the Codex CLI.
        with patch.dict(
            os.environ,
            {"CODEX_HOME": str(tmp_path), "OPENAI_API_KEY": "sk-test"},
            clear=True,
        ):
            assert codex_auth_error() is None

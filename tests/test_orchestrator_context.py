"""Tests for encoding context injection in Orchestrator prompts."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac.harness.orchestrator import Backend, Orchestrator, SubsectionTask


@pytest.fixture
def cli_orchestrator(temp_db_path):
    """CLI-backend orchestrator with a temp DB."""
    return Orchestrator(backend=Backend.CLI, db_path=temp_db_path)


@pytest.fixture
def api_orchestrator():
    """API-backend orchestrator (no DB needed for this test)."""
    return Orchestrator(backend=Backend.API, api_key="test-key")


@pytest.fixture
def openai_orchestrator(temp_db_path):
    """OpenAI Responses backend orchestrator."""
    return Orchestrator(
        backend=Backend.OPENAI,
        api_key="openai-test-key",
        db_path=temp_db_path,
    )


class TestBuildContextSection:
    def test_cli_backend_returns_context(self, cli_orchestrator):
        result = cli_orchestrator._build_context_section()
        assert "Past encoding reference" in result
        assert "encoding_runs" in result
        assert "sqlite3" in result
        assert ".rac files" in result

    def test_cli_backend_includes_db_path(self, cli_orchestrator):
        result = cli_orchestrator._build_context_section()
        assert str(cli_orchestrator.encoding_db.db_path) in result

    def test_api_backend_returns_empty(self, api_orchestrator):
        result = api_orchestrator._build_context_section()
        assert result == ""

    def test_cli_no_db_uses_default_path(self):
        orch = Orchestrator(backend=Backend.CLI)
        result = orch._build_context_section()
        assert "encodings.db" in result
        assert "Past encoding reference" in result


class TestContextInPrompts:
    def test_subsection_prompt_includes_context_cli(self, cli_orchestrator):
        task = SubsectionTask(
            subsection_id="(a)",
            title="Allowance of credit",
            file_name="a.rac",
            dependencies=[],
        )
        prompt = cli_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" in prompt

    def test_fallback_prompt_includes_context_cli(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" in prompt

    def test_subsection_prompt_no_context_api(self, api_orchestrator):
        task = SubsectionTask(
            subsection_id="(a)",
            title="Allowance of credit",
            file_name="a.rac",
            dependencies=[],
        )
        prompt = api_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" not in prompt

    def test_fallback_prompt_no_context_api(self, api_orchestrator):
        prompt = api_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" not in prompt


class TestCriticalRulesInPrompts:
    """Tests that the 4 P0 rules appear in both prompt builders."""

    def _make_task(self):
        return SubsectionTask(
            subsection_id="(a)",
            title="Allowance of credit",
            file_name="a.rac",
            dependencies=[],
        )

    def test_subsection_prompt_has_compile_preflight(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "COMPILE CHECK" in prompt
        assert "python -m rac.test_runner" in prompt

    def test_subsection_prompt_has_write_tests(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "WRITE TESTS" in prompt
        assert ".rac.test" in prompt

    def test_subsection_prompt_api_requests_two_file_bundle(self, openai_orchestrator):
        prompt = openai_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "=== FILE: a.rac ===" in prompt
        assert "=== FILE: a.rac.test ===" in prompt

    def test_subsection_prompt_has_parent_imports(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "PARENT IMPORTS FROM CHILDREN" in prompt
        assert "from ./{child}" in prompt

    def test_subsection_prompt_has_indexed_by(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "INDEXED_BY FOR INFLATION" in prompt
        assert "indexed_by:" in prompt

    def test_subsection_prompt_has_cross_statute_import_rule(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "CROSS-STATUTE DEFINITIONS MUST BE IMPORTED" in prompt

    def test_subsection_prompt_requires_status_and_four_space_rac_format(
        self, cli_orchestrator
    ):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "Every `.rac` file MUST include `status:` explicitly" in prompt
        assert "Use 4 spaces for fields under each definition" in prompt
        assert "Quote all `description:` and `label:` string values" in prompt

    def test_subsection_prompt_forbids_branch_local_assignment_blocks(
        self, cli_orchestrator
    ):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "selected_threshold = threshold_joint" in prompt
        assert "must NOT assign helper variables inside branches" in prompt
        assert "Do NOT write `status in [joint, surviving_spouse]`" in prompt

    def test_subsection_prompt_prefers_precise_nested_xml_text(
        self, cli_orchestrator, monkeypatch
    ):
        task = SubsectionTask(
            subsection_id="h/4/A",
            title="In general",
            file_name="h/4/A.rac",
            dependencies=[],
        )
        monkeypatch.setattr(
            cli_orchestrator,
            "_lookup_precise_subsection_text",
            lambda citation, subsection_id: "(A) Exact nested text from XML.",
        )
        prompt = cli_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 24",
            output_path=Path("/tmp/test"),
            statute_text="WRONG FULL TEXT",
        )
        assert "(A) Exact nested text from XML." in prompt
        assert "WRONG FULL TEXT" not in prompt

    def test_fallback_prompt_has_compile_preflight(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "COMPILE CHECK" in prompt
        assert "python -m rac.test_runner" in prompt

    def test_fallback_prompt_has_write_tests(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "WRITE TESTS" in prompt
        assert ".rac.test" in prompt

    def test_fallback_prompt_has_parent_imports(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "PARENT IMPORTS FROM CHILDREN" in prompt

    def test_fallback_prompt_has_indexed_by(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "INDEXED_BY FOR INFLATION" in prompt
        assert "indexed_by:" in prompt

    def test_fallback_prompt_has_cross_statute_import_rule(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "CROSS-STATUTE DEFINITIONS MUST BE IMPORTED" in prompt

    def test_fallback_prompt_requires_status_and_four_space_rac_format(
        self, cli_orchestrator
    ):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "Every `.rac` file MUST include `status:` explicitly" in prompt
        assert "Use 4 spaces for fields under each definition" in prompt
        assert "Quote all `description:` and `label:` string values" in prompt

    def test_fallback_prompt_forbids_branch_local_assignment_blocks(
        self, cli_orchestrator
    ):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "selected_threshold = threshold_joint" in prompt
        assert "must NOT assign helper variables inside branches" in prompt
        assert "Do NOT write `status in [joint, surviving_spouse]`" in prompt


class TestExtractSubsectionText:
    """Tests for _extract_subsection_text."""

    SAMPLE_STATUTE = (
        "(a) In general. A credit is allowed. "
        "(b) Applicable percentage. The percentage is 35 percent. "
        "(c) Dollar limit. The amount shall not exceed $3,000. "
        "(d) Earned income limitation. Reduced by earned income."
    )

    def test_extracts_first_subsection(self, api_orchestrator):
        result = api_orchestrator._extract_subsection_text(self.SAMPLE_STATUTE, "a")
        assert result is not None
        assert "(a) In general" in result
        assert "credit is allowed" in result
        # Should NOT contain (b) content
        assert "Applicable percentage" not in result

    def test_extracts_middle_subsection(self, api_orchestrator):
        result = api_orchestrator._extract_subsection_text(self.SAMPLE_STATUTE, "b")
        assert result is not None
        assert "(b) Applicable percentage" in result
        assert "35 percent" in result
        assert "Dollar limit" not in result

    def test_extracts_last_subsection(self, api_orchestrator):
        result = api_orchestrator._extract_subsection_text(self.SAMPLE_STATUTE, "d")
        assert result is not None
        assert "(d) Earned income limitation" in result

    def test_handles_parenthesized_id(self, api_orchestrator):
        result = api_orchestrator._extract_subsection_text(self.SAMPLE_STATUTE, "(b)")
        assert result is not None
        assert "Applicable percentage" in result

    def test_returns_none_for_missing_subsection(self, api_orchestrator):
        result = api_orchestrator._extract_subsection_text(self.SAMPLE_STATUTE, "z")
        assert result is None

    def test_returns_none_for_empty_text(self, api_orchestrator):
        assert api_orchestrator._extract_subsection_text("", "a") is None
        assert api_orchestrator._extract_subsection_text(None, "a") is None

    def test_returns_none_for_empty_id(self, api_orchestrator):
        assert api_orchestrator._extract_subsection_text("text", "") is None

    def test_numeric_subsection_ids(self, api_orchestrator):
        text = "(1) First item. (2) Second item. (3) Third item."
        result = api_orchestrator._extract_subsection_text(text, "2")
        assert result is not None
        assert "(2) Second item" in result
        assert "First item" not in result
        assert "Third item" not in result

    def test_truncates_long_text(self, api_orchestrator):
        long_text = "(a) " + "x" * 20000 + " (b) short."
        result = api_orchestrator._extract_subsection_text(long_text, "a")
        assert result is not None
        assert len(result) <= 15100  # 15000 + "... [truncated]"
        assert result.endswith("... [truncated]")


class TestCliUsageParsing:
    def test_run_via_cli_parses_structured_usage(self, cli_orchestrator, monkeypatch):
        payload = json.dumps(
            {
                "type": "result",
                "result": 'status: encoded\n\ncredit:\n    from 1998-01-01: 1000\n',
                "is_error": False,
                "total_cost_usd": 0.123,
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 7,
                    "cache_read_input_tokens": 5,
                    "cache_creation_input_tokens": 3,
                },
            }
        )

        def fake_run(*args, **kwargs):
            return SimpleNamespace(stdout=payload, stderr="", returncode=0)

        monkeypatch.setattr("subprocess.run", fake_run)

        result = asyncio.run(cli_orchestrator._run_via_cli("prompt"))

        assert result["text"].startswith("status: encoded")
        assert result["tokens"].input_tokens == 12
        assert result["tokens"].output_tokens == 7
        assert result["tokens"].cache_read_tokens == 5
        assert result["tokens"].cache_creation_tokens == 3
        assert result["cost"] == 0.123

    def test_run_via_openai_responses_parses_usage_and_reasoning(
        self, openai_orchestrator, monkeypatch
    ):
        payload = {
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Need one file only."}],
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": 'status: encoded\n\ncredit:\n    from 1998-01-01: 1000\n',
                        }
                    ],
                },
            ],
            "usage": {
                "input_tokens": 21,
                "input_tokens_details": {"cached_tokens": 8},
                "output_tokens": 34,
                "output_tokens_details": {"reasoning_tokens": 13},
            },
        }

        class FakeResponse:
            status_code = 200
            headers = {"x-request-id": "req_123"}

            def json(self):
                return payload

        def fake_post(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr("requests.post", fake_post)

        result = asyncio.run(
            openai_orchestrator._run_via_openai_responses(
                "full prompt", "system prompt", "user prompt"
            )
        )

        assert result["text"].startswith("status: encoded")
        assert result["tokens"].input_tokens == 21
        assert result["tokens"].cache_read_tokens == 8
        assert result["tokens"].output_tokens == 34
        assert result["tokens"].reasoning_output_tokens == 13
        assert result["trace"]["provider"] == "openai"
        assert result["trace"]["backend"] == "openai"
        assert result["trace"]["request_id"] == "req_123"

    def test_materialize_agent_artifact_writes_response_content(
        self, openai_orchestrator, tmp_path
    ):
        from autorac.harness.orchestrator import AgentRun, Phase

        expected = tmp_path / "26" / "24" / "a.rac"
        agent_run = AgentRun(
            agent_type="encoder",
            prompt="Encode 26 USC 24(a)",
            phase=Phase.ENCODING,
            result='status: encoded\n\ncredit:\n    from 1998-01-01: 1000\n',
        )

        wrote = openai_orchestrator._materialize_agent_artifact(agent_run, expected)

        assert wrote is True
        assert expected.exists()
        assert "status: encoded" in expected.read_text()

    def test_materialize_agent_artifact_writes_companion_test_bundle(
        self, openai_orchestrator, tmp_path
    ):
        from autorac.harness.orchestrator import AgentRun, Phase

        expected = tmp_path / "26" / "24" / "a.rac"
        agent_run = AgentRun(
            agent_type="encoder",
            prompt="Encode 26 USC 24(a)",
            phase=Phase.ENCODING,
            result=(
                "=== FILE: a.rac ===\n"
                'status: encoded\n\n"""\n26 USC 24(a)\n"""\n\n'
                "credit_amount:\n    from 1998-01-01: 1000\n"
                "=== FILE: a.rac.test ===\n"
                "credit_amount:\n"
                '  - name: "base case"\n'
                "    period: 1998-01\n"
                "    inputs: {}\n"
                "    expect: 1000\n"
            ),
        )

        wrote = openai_orchestrator._materialize_agent_artifact(agent_run, expected)

        assert wrote is True
        assert expected.exists()
        assert expected.with_suffix(".rac.test").exists()
        assert "credit_amount:" in expected.with_suffix(".rac.test").read_text()


class TestPathResolution:
    def test_find_statute_root_for_temp_output_tree(self, cli_orchestrator, tmp_path):
        output_path = tmp_path / "26" / "24" / "a"
        output_path.mkdir(parents=True)

        assert cli_orchestrator._find_statute_root(output_path) == tmp_path

    def test_temp_output_nested_leaf_uses_full_citation_prefix(
        self, cli_orchestrator, tmp_path
    ):
        rac_file = tmp_path / "26" / "24" / "a" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text("status: encoded\n")

        assert cli_orchestrator._artifact_relative_path(rac_file) == "26/24/a/a.rac"
        assert cli_orchestrator._citation_from_rac_file(rac_file) == "26 USC 24(a)"

    def test_fetch_statute_text_prefers_exact_xml_subsection(
        self, openai_orchestrator, monkeypatch
    ):
        monkeypatch.setattr(
            "autorac.harness.orchestrator.find_citation_text",
            lambda citation, xml_root: "(a) Allowance of credit. Exact subsection text.",
        )

        result = openai_orchestrator._fetch_statute_text("26 USC 24(a)")

        assert result == "(a) Allowance of credit. Exact subsection text."


class TestReviewContext:
    def test_run_reviews_parallel_inlines_file_context(
        self, openai_orchestrator, tmp_path, monkeypatch
    ):
        from autorac.harness.orchestrator import AgentRun, Phase

        rac_file = tmp_path / "26" / "24" / "a" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text("status: encoded\namount:\n    from 2018-01-01: 2000\n")
        rac_file.with_suffix(".rac.test").write_text(
            'amount:\n  - name: "base case"\n    period: 2018-01\n    inputs: {}\n    expect: 2000\n'
        )

        prompts = []

        async def fake_run_agent(agent_key, prompt, phase):
            prompts.append(prompt)
            return AgentRun(
                agent_type=agent_key,
                prompt=prompt,
                phase=phase,
                result="ok",
            )

        monkeypatch.setattr(openai_orchestrator, "_run_agent", fake_run_agent)
        monkeypatch.setattr(
            openai_orchestrator,
            "_fetch_statute_text",
            lambda citation: "(a) Allowance of credit. $2,000 per qualifying child.",
        )

        runs = asyncio.run(
            openai_orchestrator._run_reviews_parallel(
                "26 USC 24(a)",
                "PE: UNTESTED",
                [rac_file],
            )
        )

        assert len(runs) == 4
        assert any("## Files Under Review" in prompt for prompt in prompts)
        assert any("amount:" in prompt for prompt in prompts)
        assert any("## Companion Test File: a.rac.test" in prompt for prompt in prompts)
        assert any("$2,000 per qualifying child" in prompt for prompt in prompts)


class TestSubsectionPromptUsesSpecificText:
    """Tests that _build_subsection_prompt injects subsection-specific text."""

    STATUTE = (
        "(a) Credit allowed. There is allowed a credit. "
        "(b) Rate. The rate is 35 percent. "
        "(c) Limit. Not more than $3,000."
    )

    def test_prompt_contains_subsection_text_not_full(self, api_orchestrator):
        task = SubsectionTask(
            subsection_id="b",
            title="Rate",
            file_name="b.rac",
            dependencies=[],
        )
        prompt = api_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text=self.STATUTE,
        )
        assert "Statute text for subsection (b)" in prompt
        assert "35 percent" in prompt
        # Should NOT have the full-text fallback label
        assert "Full statute text (excerpt)" not in prompt

    def test_prompt_falls_back_when_subsection_not_found(self, api_orchestrator):
        task = SubsectionTask(
            subsection_id="z",
            title="Nonexistent",
            file_name="z.rac",
            dependencies=[],
        )
        prompt = api_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text=self.STATUTE,
        )
        assert "Full statute text (excerpt)" in prompt


class TestBuildAggregatorPrompt:
    """Tests for _build_aggregator_prompt."""

    def test_contains_root_file_name(self, api_orchestrator):
        tasks = [
            SubsectionTask(subsection_id="a", title="Credit", file_name="a.rac"),
            SubsectionTask(subsection_id="b", title="Rate", file_name="b.rac"),
        ]
        prompt = api_orchestrator._build_aggregator_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            tasks=tasks,
        )
        assert "21.rac" in prompt
        assert "ROOT aggregator" in prompt

    def test_lists_all_children(self, api_orchestrator):
        tasks = [
            SubsectionTask(subsection_id="a", title="Credit", file_name="a.rac"),
            SubsectionTask(subsection_id="b", title="Rate", file_name="b.rac"),
            SubsectionTask(subsection_id="c", title="Limit", file_name="c.rac"),
        ]
        prompt = api_orchestrator._build_aggregator_prompt(
            citation="26 USC 32",
            output_path=Path("/tmp/test"),
            tasks=tasks,
        )
        assert "a.rac" in prompt
        assert "b.rac" in prompt
        assert "c.rac" in prompt
        assert "32.rac" in prompt

    def test_includes_dsl_cheatsheet(self, api_orchestrator):
        tasks = [
            SubsectionTask(subsection_id="a", title="Credit", file_name="a.rac"),
        ]
        prompt = api_orchestrator._build_aggregator_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            tasks=tasks,
        )
        assert "RAC DSL quick reference" in prompt

    def test_no_duplicate_logic_instruction(self, api_orchestrator):
        tasks = [
            SubsectionTask(subsection_id="a", title="Credit", file_name="a.rac"),
        ]
        prompt = api_orchestrator._build_aggregator_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            tasks=tasks,
        )
        assert "NOT duplicate" in prompt or "NOT re-encode" in prompt


class TestAggregatorWaveInEncoding:
    """Tests that _run_encoding_parallel adds an aggregator wave."""

    @pytest.fixture
    def analysis_json(self):
        return """Some analysis text.
<!-- STRUCTURED_OUTPUT
{"subsections": [
  {"id": "a", "title": "Credit", "disposition": "ENCODE", "file": "a.rac"},
  {"id": "b", "title": "Rate", "disposition": "ENCODE", "file": "b.rac"},
  {"id": "c", "title": "Limit", "disposition": "ENCODE", "file": "c.rac"}
],
 "dependencies": {"b": ["a"]},
 "encoding_order": ["a", "c", "b"]}
-->"""

    def test_parse_and_waves(self, api_orchestrator, analysis_json):
        """Verify parse + wave computation works for aggregator test setup."""
        tasks = api_orchestrator._parse_analyzer_output(analysis_json)
        assert len(tasks) == 3
        waves = api_orchestrator._compute_waves(tasks)
        # a and c have no deps -> wave 0, b depends on a -> wave 1
        assert len(waves) == 2

    def test_parallel_wave_logs_runs_and_compile_checks(
        self, cli_orchestrator, analysis_json, tmp_path, monkeypatch
    ):
        """Completed subsection runs should be logged before the whole encode finishes."""
        from autorac.harness.orchestrator import AgentRun, Phase

        output_path = tmp_path / "26" / "21"
        output_path.mkdir(parents=True)

        async def fake_run_agent(agent_key, prompt, phase):
            if "One file: a.rac" in prompt:
                (output_path / "a.rac").write_text("amount:\n    from 2024-01-01: 0\n")
            elif "One file: c.rac" in prompt:
                (output_path / "c.rac").write_text("limit:\n    from 2024-01-01: 0\n")
            elif "One file: b.rac" in prompt:
                (output_path / "b.rac").write_text("rate:\n    from 2024-01-01: 0\n")
            elif "ROOT aggregator" in prompt:
                (output_path / "21.rac").write_text("credit:\n    from 2024-01-01: 0\n")
            return AgentRun(
                agent_type=agent_key,
                prompt=prompt,
                phase=phase,
                result="ok",
            )

        monkeypatch.setattr(cli_orchestrator, "_run_agent", fake_run_agent)
        monkeypatch.setattr(
            cli_orchestrator,
            "_run_compile_gate",
            lambda path: SimpleNamespace(
                passed=True,
                issues=[],
                raw_output=f"Compiled {path.name}",
                error=None,
            ),
        )

        cli_orchestrator.encoding_db.start_session(session_id="wave-log")
        runs = asyncio.run(
            cli_orchestrator._run_encoding_parallel(
                "26 USC 21",
                output_path,
                "(a) Credit. (b) Rate. (c) Limit.",
                analysis_json,
                session_id="wave-log",
                max_concurrent=1,
            )
        )

        assert len(runs) == 4
        events = cli_orchestrator.encoding_db.get_session_events("wave-log")
        agent_ends = [e for e in events if e.event_type == "agent_end"]
        compile_events = [
            e
            for e in events
            if e.event_type == "provenance_validation"
            and e.metadata.get("validation_type") == "compile"
        ]
        assert len(agent_ends) == 4
        assert len(compile_events) == 4

    def test_compile_gate_failure_stops_before_dependent_wave(
        self, cli_orchestrator, tmp_path, monkeypatch
    ):
        """A failed compile in wave 0 should stop dependent later waves."""
        from autorac.harness.orchestrator import AgentRun, Phase

        analysis_json = """<!-- STRUCTURED_OUTPUT
{"subsections": [
  {"id": "a", "title": "Credit", "disposition": "ENCODE", "file": "a.rac"},
  {"id": "b", "title": "Rate", "disposition": "ENCODE", "file": "b.rac"}
],
 "dependencies": {"b": ["a"]},
 "encoding_order": ["a", "b"]}
-->"""

        output_path = tmp_path / "26" / "21"
        output_path.mkdir(parents=True)
        prompts_seen = []

        async def fake_run_agent(agent_key, prompt, phase):
            prompts_seen.append(prompt)
            if "One file: a.rac" in prompt:
                (output_path / "a.rac").write_text("amount:\n    from 2024-01-01: 0\n")
            return AgentRun(
                agent_type=agent_key,
                prompt=prompt,
                phase=phase,
                result="ok",
            )

        def fake_compile_gate(path):
            if path.name == "a.rac":
                return SimpleNamespace(
                    passed=False,
                    issues=["Compilation failed: unexpected token"],
                    raw_output=None,
                    error="unexpected token",
                )
            return SimpleNamespace(passed=True, issues=[], raw_output="ok", error=None)

        monkeypatch.setattr(cli_orchestrator, "_run_agent", fake_run_agent)
        monkeypatch.setattr(cli_orchestrator, "_run_compile_gate", fake_compile_gate)

        cli_orchestrator.encoding_db.start_session(session_id="wave-stop")
        with pytest.raises(RuntimeError, match="Wave compile gate failed"):
            asyncio.run(
                cli_orchestrator._run_encoding_parallel(
                    "26 USC 21",
                    output_path,
                    "(a) Credit. (b) Rate.",
                    analysis_json,
                    session_id="wave-stop",
                    max_concurrent=1,
                )
            )

        assert any("One file: a.rac" in prompt for prompt in prompts_seen)
        assert not any("One file: b.rac" in prompt for prompt in prompts_seen)

    def test_compile_gate_failure_triggers_targeted_repair(
        self, cli_orchestrator, tmp_path, monkeypatch
    ):
        """A compile failure should trigger a focused repair attempt before aborting."""
        from autorac.harness.orchestrator import AgentRun, Phase

        analysis_json = """<!-- STRUCTURED_OUTPUT
{"subsections": [
  {"id": "a", "title": "Credit", "disposition": "ENCODE", "file": "a.rac"}
],
 "dependencies": {},
 "encoding_order": ["a"]}
-->"""

        output_path = tmp_path / "26" / "21"
        output_path.mkdir(parents=True)
        prompts_seen = []
        compile_calls = 0

        async def fake_run_agent(agent_key, prompt, phase):
            nonlocal compile_calls
            prompts_seen.append(prompt)
            if "Fix the compile failure" in prompt:
                (output_path / "a.rac").write_text(
                    '"""\n(a) Credit.\n"""\n\nstatus: encoded\n\namount:\n    description: "Fixed"\n    unit: USD\n    from 2024-01-01: 0\n'
                )
            else:
                (output_path / "a.rac").write_text(
                    '"""\n(a) Credit.\n"""\n\nstatus: encoded\n\namount:\n    description: Broken\n    unit: USD\n    from 2024-01-01: 0\n'
                )
            (output_path / "a.rac.test").write_text(
                'amount:\n    - name: "ok"\n      period: 2024-01\n      inputs: {}\n      expect: 0\n'
            )
            return AgentRun(
                agent_type=agent_key,
                prompt=prompt,
                phase=phase,
                result="ok",
            )

        def fake_compile_gate(path):
            nonlocal compile_calls
            compile_calls += 1
            if compile_calls == 1:
                return SimpleNamespace(
                    passed=False,
                    issues=["Compilation failed: line 6, col 18: unexpected token: IDENT"],
                    raw_output=None,
                    error="unexpected token",
                )
            return SimpleNamespace(passed=True, issues=[], raw_output="ok", error=None)

        monkeypatch.setattr(cli_orchestrator, "_run_agent", fake_run_agent)
        monkeypatch.setattr(cli_orchestrator, "_run_compile_gate", fake_compile_gate)
        monkeypatch.setattr(
            cli_orchestrator,
            "_lookup_precise_subsection_text",
            lambda citation, subsection_id: "(a) Credit.",
        )

        cli_orchestrator.encoding_db.start_session(session_id="wave-repair")
        runs = asyncio.run(
            cli_orchestrator._run_encoding_parallel(
                "26 USC 21",
                output_path,
                "(a) Credit.",
                analysis_json,
                session_id="wave-repair",
                max_concurrent=1,
            )
        )

        assert len(runs) == 2
        assert any("Fix the compile failure" in prompt for prompt in prompts_seen)
        events = cli_orchestrator.encoding_db.get_session_events("wave-repair")
        compile_events = [
            e
            for e in events
            if e.event_type == "provenance_validation"
            and e.metadata.get("validation_type") == "compile"
        ]
        assert len(compile_events) == 2


class TestOutputPathInference:
    def test_find_statute_root_for_temp_output_path(self, cli_orchestrator):
        output_path = Path("/tmp/autorac-openai-irc24-reencode-v5/26/24")
        assert (
            cli_orchestrator._find_statute_root(output_path)
            == Path("/tmp/autorac-openai-irc24-reencode-v5")
        )


class TestLogAgentRunNoTruncation:
    """Tests that _log_agent_run does not truncate prompt or result content."""

    def test_long_prompt_not_truncated(self, cli_orchestrator):
        """Prompts longer than 2000 chars should be stored in full."""
        from autorac.harness.orchestrator import AgentRun, Phase

        long_prompt = "x" * 5000
        agent_run = AgentRun(
            agent_type="encoder",
            prompt=long_prompt,
            phase=Phase.ENCODING,
            result="short result",
        )

        cli_orchestrator.encoding_db.start_session(session_id="trunc-test")
        cli_orchestrator._log_agent_run("trunc-test", agent_run)

        events = cli_orchestrator.encoding_db.get_session_events("trunc-test")
        start_event = [e for e in events if e.event_type == "agent_start"][0]
        assert len(start_event.content) == 5000

    def test_long_result_not_truncated(self, cli_orchestrator):
        """Results longer than 2000 chars should be stored in full."""
        from autorac.harness.orchestrator import AgentRun, Phase

        long_result = "y" * 5000
        agent_run = AgentRun(
            agent_type="rac_reviewer",
            prompt="short prompt",
            phase=Phase.REVIEW,
            result=long_result,
        )

        cli_orchestrator.encoding_db.start_session(session_id="trunc-test-2")
        cli_orchestrator._log_agent_run("trunc-test-2", agent_run)

        events = cli_orchestrator.encoding_db.get_session_events("trunc-test-2")
        end_event = [e for e in events if e.event_type == "agent_end"][0]
        assert len(end_event.content) == 5000


class TestProvenanceLogging:
    def test_log_agent_run_persists_assistant_message(self, cli_orchestrator):
        """Assistant responses should be logged as first-class session events."""
        from autorac.harness.orchestrator import AgentMessage, AgentRun, Phase, TokenUsage

        agent_run = AgentRun(
            agent_type="encoder",
            prompt="Encode 26 USC 24(a)",
            phase=Phase.ENCODING,
            result="Created a.rac",
            total_tokens=TokenUsage(input_tokens=123, output_tokens=45),
            provider_trace={
                "provider": "anthropic",
                "backend": "cli",
                "model": "claude-opus-test",
                "raw_output": "Created a.rac",
            },
        )
        agent_run.messages.append(
            AgentMessage(
                role="assistant",
                content="Created a.rac",
                summary="Completed successfully",
                tokens=TokenUsage(input_tokens=123, output_tokens=45),
            )
        )

        cli_orchestrator.encoding_db.start_session(session_id="prov-agent")
        cli_orchestrator._log_agent_run("prov-agent", agent_run)

        events = cli_orchestrator.encoding_db.get_session_events("prov-agent")
        assistant_event = [e for e in events if e.event_type == "agent_assistant"][0]
        sidecar_event = [e for e in events if e.event_type == "provenance_sidecar"][0]
        assert assistant_event.content == "Created a.rac"
        assert assistant_event.metadata["summary"] == "Completed successfully"
        assert assistant_event.metadata["tokens"]["input"] == 123
        assert assistant_event.metadata["tokens"]["output"] == 45
        assert assistant_event.metadata["tokens"]["cache_read"] == 0
        assert assistant_event.metadata["tokens"]["cache_creation"] == 0
        assert assistant_event.metadata["tokens"]["reasoning_output"] == 0
        assert sidecar_event.metadata["backend"] == "cli"
        assert "Provider sidecar trace for encoder" in sidecar_event.content
        assert '"provider": "anthropic"' in sidecar_event.content

    def test_log_agent_run_promotes_provider_reasoning(self, cli_orchestrator):
        """Provider-exposed reasoning should be logged as structured provenance."""
        from autorac.harness.orchestrator import AgentRun, Phase, TokenUsage

        agent_run = AgentRun(
            agent_type="encoder",
            prompt="Encode 26 USC 24(a)",
            phase=Phase.ENCODING,
            result="Created a.rac",
            total_tokens=TokenUsage(input_tokens=123, output_tokens=45),
            provider_trace={
                "provider": "openai",
                "backend": "codex-cli",
                "model": "gpt-5.4",
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "item_4",
                            "type": "reasoning",
                            "text": "Need to align subsection (a) with the section 151 child count.",
                        },
                    }
                ],
            },
        )

        cli_orchestrator.encoding_db.start_session(session_id="prov-reasoning")
        cli_orchestrator._log_agent_run("prov-reasoning", agent_run)

        events = cli_orchestrator.encoding_db.get_session_events("prov-reasoning")
        reasoning_event = [e for e in events if e.event_type == "provenance_reasoning"][0]
        assert reasoning_event.content.startswith("Need to align subsection (a)")
        assert reasoning_event.metadata["provider"] == "openai"
        assert reasoning_event.metadata["backend"] == "codex-cli"
        assert reasoning_event.metadata["source"] == "events.item.completed"
        assert reasoning_event.metadata["item_id"] == "item_4"

    def test_log_analysis_provenance_emits_plan(self, cli_orchestrator):
        """Structured analyzer output should become a normalized provenance plan."""
        analysis_text = """Analysis
<!-- STRUCTURED_OUTPUT
{"subsections": [
  {"id": "a", "title": "Allowance of credit", "disposition": "ENCODE", "file": "a.rac"},
  {"id": "b", "title": "Limitations", "disposition": "ENCODE", "file": "b.rac"}
],
 "dependencies": {"b": ["a"]},
 "encoding_order": ["a", "b"]}
-->"""

        cli_orchestrator.encoding_db.start_session(session_id="prov-plan")
        cli_orchestrator._log_analysis_provenance(
            "prov-plan", "26 USC 24", analysis_text
        )

        events = cli_orchestrator.encoding_db.get_session_events("prov-plan")
        plan_event = [e for e in events if e.event_type == "provenance_plan"][0]
        assert "Encoding plan for 26 USC 24" in plan_event.content
        assert len(plan_event.metadata["tasks"]) == 2
        assert plan_event.metadata["tasks"][1]["dependencies"] == ["a"]

    def test_log_artifact_provenance_flags_ungrounded_numeric_literals(
        self, cli_orchestrator, tmp_path
    ):
        """Artifact provenance should distinguish grounded from ungrounded values."""
        from autorac.harness.orchestrator import Phase

        rac_file = tmp_path / "statute" / "26" / "24" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '''# 26 USC 24(a)
"""
(a) Allowance of credit. There shall be allowed a credit of $1,000.
"""
status: encoded

ctc_base_amount:
  entity: TaxUnit
  period: Year
  dtype: Money
  from 2018-01-01: 1000
  from 2025-01-01: 2200
'''
        )

        cli_orchestrator.encoding_db.start_session(session_id="prov-artifact")
        cli_orchestrator._log_artifact_provenance_records(
            "prov-artifact", [rac_file], Phase.ENCODING
        )

        events = cli_orchestrator.encoding_db.get_session_events("prov-artifact")
        artifact_event = [e for e in events if e.event_type == "provenance_artifact"][0]
        grounded = artifact_event.metadata["grounded_values"]
        ungrounded = artifact_event.metadata["ungrounded_values"]

        assert artifact_event.metadata["relative_artifact_path"] == "26/24/a.rac"
        assert [item["value"] for item in grounded] == [1000.0]
        assert [item["value"] for item in ungrounded] == [2200.0]
        assert "Ungrounded values: 2200" in artifact_event.content

    def test_log_artifact_provenance_derives_citation_for_temp_output(
        self, cli_orchestrator, tmp_path
    ):
        """Temp output roots should still resolve a legal citation and relative path."""
        from autorac.harness.orchestrator import Phase

        rac_file = tmp_path / "26" / "25B" / "c" / "1.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '''# 26 USC 25B(c)(1)
"""
(1) In general.-- The term "eligible individual" means any individual if such
individual has attained the age of 18 as of the close of the taxable year.
"""
status: encoded

age_threshold:
    from 2002-01-01: 18
'''
        )

        cli_orchestrator.encoding_db.start_session(session_id="prov-temp-artifact")
        cli_orchestrator._log_artifact_provenance_records(
            "prov-temp-artifact", [rac_file], Phase.ENCODING
        )

        events = cli_orchestrator.encoding_db.get_session_events("prov-temp-artifact")
        artifact_event = [e for e in events if e.event_type == "provenance_artifact"][0]

        assert artifact_event.metadata["relative_artifact_path"] == "26/25B/c/1.rac"
        assert artifact_event.metadata["citation"] == "26 USC 25B(c)(1)"
        assert "Citation: 26 USC 25B(c)(1)" in artifact_event.content

    def test_log_review_provenance_extracts_reviewer_result(self, cli_orchestrator):
        """Reviewer JSON should be normalized into provenance review records."""
        from autorac.harness.orchestrator import AgentRun, Phase

        review_run = AgentRun(
            agent_type="parameter_reviewer",
            prompt="Review the file",
            phase=Phase.REVIEW,
            result=(
                '{"score": 4.5, "passed": false, '
                '"issues": ["Value 2200 not found in source"], '
                '"reasoning": "Ungrounded amendment amount."}'
            ),
        )

        cli_orchestrator.encoding_db.start_session(session_id="prov-review")
        cli_orchestrator._log_review_provenance("prov-review", review_run)

        events = cli_orchestrator.encoding_db.get_session_events("prov-review")
        review_event = [e for e in events if e.event_type == "provenance_review"][0]
        assert review_event.metadata["agent_type"] == "parameter_reviewer"
        assert review_event.metadata["score"] == 4.5
        assert review_event.metadata["passed"] is False
        assert review_event.metadata["issues"] == ["Value 2200 not found in source"]

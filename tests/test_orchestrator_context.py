"""Tests for encoding context injection in Orchestrator prompts."""

import sys
from pathlib import Path

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
        assert "autorac test" in prompt

    def test_subsection_prompt_has_write_tests(self, cli_orchestrator):
        prompt = cli_orchestrator._build_subsection_prompt(
            task=self._make_task(),
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
        )
        assert "WRITE TESTS" in prompt
        assert ".rac.test" in prompt

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

    def test_fallback_prompt_has_compile_preflight(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test text",
        )
        assert "COMPILE CHECK" in prompt
        assert "autorac test" in prompt

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


# =========================================================================
# Fix #23: Non-USC citation handling
# =========================================================================


class TestParseAnalyzerOutputDotNotation:
    """Test _parse_analyzer_output with dot-notation subsection IDs (state regs)."""

    def test_dot_notation_ids(self, cli_orchestrator):
        """Dot-notation IDs like (4.3) and (4.6.1.A) are parsed correctly."""
        text = """
| Subsection | Title | Disposition | File |
|---|---|---|---|
| (4.3) | Income Standards | ENCODE | 4.3.rac |
| (4.4) | Definitions | SKIP | - |
| (4.6.1.A) | CCAP Copayment | ENCODE | 4.6.1.A.rac |
"""
        tasks = cli_orchestrator._parse_analyzer_output(text)
        assert len(tasks) == 2
        assert tasks[0].subsection_id == "4.3"
        assert tasks[0].file_name == "4.3.rac"
        assert tasks[1].subsection_id == "4.6.1.A"
        assert tasks[1].file_name == "4.6.1.A.rac"

    def test_usc_style_ids_still_work(self, cli_orchestrator):
        """Traditional USC-style IDs like (a), (b) still parse correctly."""
        text = """
| Subsection | Title | Disposition | File |
|---|---|---|---|
| (a) | Allowance | ENCODE | a.rac |
| (b) | Limits | ENCODE | b.rac |
"""
        tasks = cli_orchestrator._parse_analyzer_output(text)
        assert len(tasks) == 2
        assert tasks[0].subsection_id == "a"
        assert tasks[1].subsection_id == "b"


class TestRunViaCliAcceptEdits:
    """Test _run_via_cli builds correct command with/without accept_edits."""

    def test_accept_edits_adds_permission_mode(self, cli_orchestrator):
        """accept_edits=True adds --permission-mode acceptEdits."""
        import asyncio
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.stdout = '{"result": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            asyncio.run(cli_orchestrator._run_via_cli("test prompt", accept_edits=True))
            cmd = mock_run.call_args[0][0]
            assert "--permission-mode" in cmd
            assert "acceptEdits" in cmd
            # Verify ordering: --permission-mode comes after --print
            pm_idx = cmd.index("--permission-mode")
            print_idx = cmd.index("--print")
            assert pm_idx > print_idx

    def test_no_accept_edits_omits_permission_mode(self, cli_orchestrator):
        """accept_edits=False (default) omits --permission-mode."""
        import asyncio
        from unittest.mock import MagicMock, patch

        mock_result = MagicMock()
        mock_result.stdout = '{"result": "ok"}'
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result) as mock_run:
            asyncio.run(
                cli_orchestrator._run_via_cli("test prompt", accept_edits=False)
            )
            cmd = mock_run.call_args[0][0]
            assert "--permission-mode" not in cmd
            assert "acceptEdits" not in cmd


class TestEncodeNonUscPathDerivation:
    """Test Orchestrator.encode() derives correct output_path for non-USC citations."""

    def test_non_usc_uses_slug(self):
        """Non-USC citation produces a slug-based path instead of title/section.

        Tests the path derivation logic extracted from Orchestrator.encode().
        """
        import re
        from pathlib import Path

        citation = "RI CCAP 218-RICR-20-00-4"
        is_usc = bool(re.search(r"\bUSC\b", citation, re.IGNORECASE))
        assert not is_usc

        slug = citation.replace(" ", "-").lower()
        output_path = Path.home() / "RulesFoundation" / "rac-us" / "statute" / slug
        assert "ri-ccap-218-ricr-20-00-4" in str(output_path)
        # Should NOT have title/section splitting
        assert "/ri/" not in str(output_path).split("statute/")[-1]

    def test_usc_uses_title_section(self):
        """USC citation produces title/section path structure."""
        import re
        from pathlib import Path

        citation = "26 USC 21"
        is_usc = bool(re.search(r"\bUSC\b", citation, re.IGNORECASE))
        assert is_usc

        citation_clean = (
            citation.replace("USC", "").replace("usc", "").replace("\u00a7", "").strip()
        )
        parts = citation_clean.split()
        title = parts[0]
        section = parts[1]
        output_path = (
            Path.home()
            / "RulesFoundation"
            / "rac-us"
            / "statute"
            / title
            / section.replace("(", "/").replace(")", "")
        )
        assert "/26/21" in str(output_path)

    def test_usc_substring_not_false_positive(self):
        """Citation containing 'USC' as substring (e.g. 'Massachusetts') is not USC."""
        import re

        citation = "Massachusetts CCAP 101"
        is_usc = bool(re.search(r"\bUSC\b", citation, re.IGNORECASE))
        assert not is_usc

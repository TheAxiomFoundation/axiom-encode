"""Tests for external dependency resolution (stub creation).

Tests the _scan_unresolved_imports, _citation_from_path,
_extract_rac_content, and _resolve_external_dependencies methods.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac.harness.orchestrator import (
    AgentRun,
    Orchestrator,
    Phase,
)


@pytest.fixture
def orchestrator():
    """Create an Orchestrator with CLI backend for testing."""
    with patch.object(Orchestrator, "_build_context_section", return_value=""):
        return Orchestrator(model="test", backend="cli")


@pytest.fixture
def rac_tree(tmp_path):
    """Create a mock rac-us statute tree with 'statute' as root."""
    statute = tmp_path / "statute"
    statute.mkdir()
    return statute


@pytest.fixture
def dir_21(rac_tree):
    """Create statute/26/21/ directory."""
    d = rac_tree / "26" / "21"
    d.mkdir(parents=True)
    return d


class TestScanUnresolvedImports:
    def test_no_imports(self, orchestrator, dir_21):
        """Files with no imports return empty list."""
        (dir_21 / "a.rac").write_text(
            "# 26 USC 21(a)\nstatus: encoded\nsome_var:\n    entity: TaxUnit\n"
        )
        result = orchestrator._scan_unresolved_imports(dir_21)
        assert result == []

    def test_resolved_import(self, orchestrator, rac_tree, dir_21):
        """Imports that resolve to existing files are not returned."""
        # Create the target file with the variable
        target = rac_tree / "26" / "62" / "62.rac"
        target.parent.mkdir(parents=True)
        target.write_text(
            "# 26 USC 62\nstatus: stub\nadjusted_gross_income:\n    entity: TaxUnit\n"
        )

        # Create file that imports from it
        (dir_21 / "a.rac").write_text(
            "# 26 USC 21(a)\nsome_var:\n    imports:\n        - 26/62#adjusted_gross_income\n"
        )

        result = orchestrator._scan_unresolved_imports(dir_21)
        assert len(result) == 0

    def test_unresolved_import_detected(self, orchestrator, dir_21):
        """Imports to nonexistent files are detected."""
        (dir_21 / "a.rac").write_text(
            "# 26 USC 21(a)\nsome_var:\n    imports:\n        - 26/9999#fake_var\n"
        )
        result = orchestrator._scan_unresolved_imports(dir_21)
        assert len(result) == 1
        assert result[0][0] == "26/9999"
        assert result[0][1] == "fake_var"

    def test_deduplicates(self, orchestrator, dir_21):
        """Same import in multiple files is only returned once."""
        (dir_21 / "a.rac").write_text(
            "x:\n    imports:\n        - 26/62#adjusted_gross_income\n"
        )
        (dir_21 / "b.rac").write_text(
            "y:\n    imports:\n        - 26/62#adjusted_gross_income\n"
        )
        result = orchestrator._scan_unresolved_imports(dir_21)
        assert len(result) == 1


class TestCitationFromPath:
    def test_simple_section(self, orchestrator):
        assert orchestrator._citation_from_path("26/62") == "26 USC 62"

    def test_subsection(self, orchestrator):
        assert orchestrator._citation_from_path("26/21/b/1/C") == "26 USC 21(b)(1)(C)"

    def test_two_parts(self, orchestrator):
        assert orchestrator._citation_from_path("26/7703") == "26 USC 7703"

    def test_deep_subsection(self, orchestrator):
        assert orchestrator._citation_from_path("26/32/c/2") == "26 USC 32(c)(2)"


class TestExtractRacContent:
    def test_raw_content(self, orchestrator):
        response = "# 26 USC 62\nstatus: stub\nagi:\n    entity: TaxUnit\n"
        result = orchestrator._extract_rac_content(response)
        assert result.startswith("# 26 USC 62")
        assert "status: stub" in result

    def test_code_fence(self, orchestrator):
        response = "Here is the stub:\n```yaml\n# 26 USC 62\nstatus: stub\n```\n"
        result = orchestrator._extract_rac_content(response)
        assert result.startswith("# 26 USC 62")

    def test_preamble_then_content(self, orchestrator):
        response = "I'll create a stub.\n\n# 26 USC 62\nstatus: stub\n"
        result = orchestrator._extract_rac_content(response)
        assert result.startswith("# 26 USC 62")

    def test_no_rac_content(self, orchestrator):
        response = "I'm not sure what to do here."
        result = orchestrator._extract_rac_content(response)
        assert result is None

    def test_ansi_codes_stripped(self, orchestrator):
        response = "\x1b[32m# 26 USC 62\x1b[0m\nstatus: stub\n"
        result = orchestrator._extract_rac_content(response)
        assert result.startswith("# 26 USC 62")

    def test_rac_keywords_fallback(self, orchestrator):
        """When no # header, falls back to RAC keyword detection."""
        response = "Here is the stub content:\n\nstatus: stub\nsome_var:\n    entity: TaxUnit\n"
        result = orchestrator._extract_rac_content(response)
        assert result is not None
        assert result.startswith("status: stub")

    def test_hash_without_space(self, orchestrator):
        """Lines starting with # (no space) are still detected."""
        response = "Preamble text\n#26 USC 62\nstatus: stub\n"
        result = orchestrator._extract_rac_content(response)
        assert "#26 USC 62" in result

    def test_empty_response(self, orchestrator):
        assert orchestrator._extract_rac_content("") is None
        assert orchestrator._extract_rac_content("   \n  ") is None


class TestResolveExternalDependencies:
    @pytest.mark.asyncio
    async def test_no_unresolved_returns_empty(self, orchestrator, dir_21):
        """When no unresolved imports, returns empty list."""
        (dir_21 / "a.rac").write_text("# no imports\nstatus: encoded\n")

        result = await orchestrator._resolve_external_dependencies(dir_21)
        assert result == []

    @pytest.mark.asyncio
    async def test_creates_stub_for_unresolved(self, orchestrator, dir_21):
        """Creates stub file for unresolved import."""
        (dir_21 / "a.rac").write_text("x:\n    imports:\n        - 26/9999#some_var\n")

        stub_content = "# 26 USC 9999\nstatus: stub\nsome_var:\n    entity: TaxUnit\n"
        mock_run = AgentRun(
            agent_type="stub_generator",
            prompt="",
            phase=Phase.RESOLVE_EXTERNALS,
            result=stub_content,
        )

        with (
            patch.object(
                orchestrator,
                "_run_agent",
                new_callable=AsyncMock,
                return_value=mock_run,
            ),
            patch.object(
                orchestrator, "_fetch_statute_text", return_value="Some statute text"
            ),
        ):
            result = await orchestrator._resolve_external_dependencies(dir_21)

        assert len(result) == 1
        assert result[0].exists()
        content = result[0].read_text()
        assert "some_var:" in content


class TestPhaseEnum:
    def test_resolve_externals_phase_exists(self):
        assert Phase.RESOLVE_EXTERNALS.value == "resolve_externals"

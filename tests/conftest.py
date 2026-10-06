"""
Pytest fixtures for axiom_encode tests.
"""

import sys
import tempfile
from pathlib import Path

import pytest

# Add src to path for imports - make src accessible as 'axiom_encode'
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from axiom_encode import (
    EncodingDB,
    EncodingRun,
    PipelineResult,
    ReviewResult,
    ReviewResults,
    ValidationResult,
    __version__,
)
from axiom_encode import signing_broker as _signing_broker
from tests.release_object_fixtures import TEST_RELEASE_PUBLIC_KEY
from tests.signing_broker_fixtures import SigningBrokerFixture


@pytest.fixture(autouse=True)
def _trusted_test_corpus_release_broker(monkeypatch):
    """Install the verification broker used by signed corpus fixtures."""

    broker = SigningBrokerFixture(
        corpus_release_public_key=TEST_RELEASE_PUBLIC_KEY,
    )
    monkeypatch.setattr(_signing_broker, "_active_broker", broker)
    monkeypatch.setattr(_signing_broker, "_active_broker_pid", None)


@pytest.fixture(autouse=True)
def _pinned_test_encoder_execution_identity(monkeypatch):
    """Keep signed-manifest fixtures independent of the dirty test checkout."""

    monkeypatch.setattr(
        "axiom_encode.cli._current_guard_encoder_execution_identity",
        lambda: {
            "repository": "github.com/TheAxiomFoundation/axiom-encode",
            "commit": "a" * 40,
            "version": __version__,
        },
    )


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_experiments.db"


@pytest.fixture
def experiment_db(temp_db_path):
    """Create a temporary experiment database."""
    return EncodingDB(temp_db_path)


@pytest.fixture
def sample_review_results():
    """Sample review results for testing."""
    return ReviewResults(
        reviews=[
            ReviewResult(
                reviewer="rulespec_reviewer",
                passed=True,
                items_checked=10,
                items_passed=8,
            ),
            ReviewResult(
                reviewer="formula_reviewer",
                passed=True,
                items_checked=10,
                items_passed=7,
            ),
            ReviewResult(
                reviewer="parameter_reviewer",
                passed=True,
                items_checked=10,
                items_passed=8,
            ),
            ReviewResult(
                reviewer="integration_reviewer",
                passed=True,
                items_checked=10,
                items_passed=8,
            ),
        ],
        policyengine_match=0.90,
    )


@pytest.fixture
def sample_encoding_run(sample_review_results):
    """Create a sample encoding run."""
    run = EncodingRun(
        file_path="/path/to/statute.yaml",
        citation="26 USC 32",
        agent_type="axiom_encode:encoder",
        agent_model="claude-opus-4-6",
        rulespec_content="# EITC\nEarnedIncome:\n  dtype: Money\n",
        source_text="Sample statute text for EITC",
        review_results=sample_review_results,
    )
    run.total_duration_ms = 4500
    return run


@pytest.fixture
def mock_validation_result():
    """Create a mock validation result that passes."""
    return PipelineResult(
        results={
            "ci": ValidationResult(
                validator_name="ci",
                passed=True,
                score=None,
                issues=[],
                duration_ms=100,
            ),
            "rulespec_reviewer": ValidationResult(
                validator_name="rulespec_reviewer",
                passed=True,
                score=8.0,
                issues=[],
                duration_ms=500,
            ),
            "formula_reviewer": ValidationResult(
                validator_name="formula_reviewer",
                passed=True,
                score=7.5,
                issues=[],
                duration_ms=500,
            ),
            "parameter_reviewer": ValidationResult(
                validator_name="parameter_reviewer",
                passed=True,
                score=8.5,
                issues=[],
                duration_ms=500,
            ),
            "integration_reviewer": ValidationResult(
                validator_name="integration_reviewer",
                passed=True,
                score=8.0,
                issues=[],
                duration_ms=500,
            ),
            "policyengine": ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=0.95,
                issues=[],
                duration_ms=1000,
            ),
        },
        total_duration_ms=2000,
        all_passed=True,
    )


@pytest.fixture
def mock_failing_validation_result():
    """Create a mock validation result that fails."""
    return PipelineResult(
        results={
            "ci": ValidationResult(
                validator_name="ci",
                passed=False,
                score=None,
                issues=["Parse error: unexpected token"],
                duration_ms=100,
                error="Parse error: unexpected token",
            ),
            "rulespec_reviewer": ValidationResult(
                validator_name="rulespec_reviewer",
                passed=True,
                score=5.0,
                issues=["Missing citation reference"],
                duration_ms=500,
            ),
            "formula_reviewer": ValidationResult(
                validator_name="formula_reviewer",
                passed=True,
                score=4.5,
                issues=["Formula logic incorrect"],
                duration_ms=500,
            ),
            "parameter_reviewer": ValidationResult(
                validator_name="parameter_reviewer",
                passed=True,
                score=6.0,
                issues=[],
                duration_ms=500,
            ),
            "integration_reviewer": ValidationResult(
                validator_name="integration_reviewer",
                passed=True,
                score=5.5,
                issues=[],
                duration_ms=500,
            ),
        },
        total_duration_ms=1500,
        all_passed=False,
    )

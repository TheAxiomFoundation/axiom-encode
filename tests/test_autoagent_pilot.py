"""Tests for the autoresearch-style pilot helpers."""

from autorac.harness.autoagent_pilot import (
    AUTOAGENT_PILOT_MANIFESTS,
)
from autorac.harness.autoresearch_pilot import (
    AUTORESEARCH_PILOT_MANIFESTS,
    autorac_repo_root,
    extract_primary_runner_summary,
    pilot_editable_paths,
    pilot_manifest_paths,
    program_path,
    score_readiness_summary,
)


def test_pilot_manifest_paths_resolve_existing_files():
    manifests = pilot_manifest_paths()

    assert len(manifests) == 3
    assert all(path.exists() for path in manifests)
    assert manifests[0].name == "uk_wave18_remaining_repair.yaml"


def test_autoagent_alias_points_at_same_manifest_set():
    assert AUTOAGENT_PILOT_MANIFESTS == AUTORESEARCH_PILOT_MANIFESTS


def test_pilot_editable_paths_point_at_prompt_surface():
    paths = pilot_editable_paths()

    assert paths == [
        autorac_repo_root() / "src/autorac/harness/eval_prompt_surface.py"
    ]


def test_program_path_resolves_autoresearch_program():
    assert program_path() == autorac_repo_root() / "autoresearch/program.md"


def test_score_readiness_summary_rewards_ready_runner():
    ready_score = score_readiness_summary(
        {
            "ready": True,
            "compile_pass_rate": 1.0,
            "ci_pass_rate": 1.0,
            "zero_ungrounded_rate": 1.0,
            "generalist_review_pass_rate": 1.0,
            "mean_estimated_cost_usd": 0.2,
        }
    )
    degraded_score = score_readiness_summary(
        {
            "ready": False,
            "compile_pass_rate": 0.9,
            "ci_pass_rate": 0.8,
            "zero_ungrounded_rate": 1.0,
            "generalist_review_pass_rate": 0.6,
            "mean_estimated_cost_usd": 0.2,
        }
    )

    assert ready_score > degraded_score
    assert ready_score == 99.98


def test_extract_primary_runner_summary_returns_first_runner():
    runner, summary = extract_primary_runner_summary(
        {
            "readiness": {
                "codex-gpt-5.4": {"ready": True},
                "claude-opus": {"ready": False},
            }
        }
    )

    assert runner == "codex-gpt-5.4"
    assert summary == {"ready": True}

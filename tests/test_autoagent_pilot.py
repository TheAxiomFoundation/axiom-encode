"""Tests for the autoresearch-style pilot helpers."""

from autorac.harness.autoagent_pilot import (
    AUTOAGENT_PILOT_MANIFESTS,
)
from autorac.harness.autoresearch_pilot import (
    AUTORESEARCH_FINAL_REVIEW_MANIFESTS,
    AUTORESEARCH_PILOT_MANIFESTS,
    autorac_repo_root,
    build_mutation_prompt,
    extract_autoresearch_score,
    extract_primary_runner_summary,
    final_review_manifest_paths,
    load_autoresearch_report,
    pilot_editable_paths,
    pilot_manifest_paths,
    program_path,
    score_readiness_summary,
    seed_legislation_cache,
    shared_legislation_cache_root,
    should_keep_candidate,
    sync_legislation_cache,
)


def test_pilot_manifest_paths_resolve_existing_files():
    manifests = pilot_manifest_paths()

    assert len(manifests) == 5
    assert all(path.exists() for path in manifests)
    assert manifests[0].name == "uk_wave18_remaining_repair.yaml"
    assert manifests[-2].name == "uk_autoresearch_partner_disjunction.yaml"
    assert manifests[-1].name == "uk_autoresearch_semantic_margin.yaml"


def test_autoagent_alias_points_at_same_manifest_set():
    assert AUTOAGENT_PILOT_MANIFESTS == AUTORESEARCH_PILOT_MANIFESTS


def test_final_review_manifest_paths_resolve_existing_files():
    manifests = final_review_manifest_paths()

    assert len(manifests) == len(AUTORESEARCH_FINAL_REVIEW_MANIFESTS)
    assert all(path.exists() for path in manifests)
    assert manifests[0].name == "uk_autoresearch_final_review.yaml"


def test_pilot_editable_paths_point_at_prompt_surface():
    paths = pilot_editable_paths()

    assert paths == [
        autorac_repo_root() / "src/autorac/harness/eval_prompt_surface.py"
    ]


def test_program_path_resolves_autoresearch_program():
    assert program_path() == autorac_repo_root() / "autoresearch/program.md"


def test_shared_legislation_cache_root_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "shared-cache"
    monkeypatch.setenv("AUTORAC_SHARED_LEGISLATION_CACHE", str(target))

    assert shared_legislation_cache_root() == target.resolve()


def test_seed_legislation_cache_copies_existing_local_cache(monkeypatch, tmp_path):
    search_root = tmp_path / "tmp"
    prior_run = search_root / "autorac-prior-run"
    cache_dir = prior_run / "_legislation_gov_uk_cache" / "uksi-2002-1792-2025-03-31"
    source_dir = prior_run / "_legislation_gov_uk" / "uksi-2002-1792-2025-03-31"
    cache_dir.mkdir(parents=True)
    source_dir.mkdir(parents=True)
    (cache_dir / "source.akn").write_text("cached akn\n")
    (cache_dir / "source.xml").write_text("cached xml\n")
    (source_dir / "source.akn").write_text("fetched akn\n")
    (source_dir / "source.xml").write_text("fetched xml\n")

    shared_root = tmp_path / "shared"
    run_root = tmp_path / "run"

    copied = seed_legislation_cache(
        run_root,
        shared_root=shared_root,
        search_root=search_root,
    )

    assert copied["_legislation_gov_uk"] == 2
    assert copied["_legislation_gov_uk_cache"] == 2
    assert (
        run_root / "_legislation_gov_uk_cache" / "uksi-2002-1792-2025-03-31" / "source.akn"
    ).read_text() == "cached akn\n"


def test_sync_legislation_cache_promotes_new_files_into_shared_root(tmp_path):
    run_root = tmp_path / "run"
    shared_root = tmp_path / "shared"
    source_dir = run_root / "_legislation_gov_uk_cache" / "uksi-2002-1792-2025-03-31"
    source_dir.mkdir(parents=True)
    (source_dir / "source.akn").write_text("new akn\n")
    (source_dir / "source.xml").write_text("new xml\n")

    synced = sync_legislation_cache(run_root, shared_root=shared_root)

    assert synced["_legislation_gov_uk_cache"] == 2
    assert (
        shared_root
        / "_legislation_gov_uk_cache"
        / "uksi-2002-1792-2025-03-31"
        / "source.xml"
    ).read_text() == "new xml\n"


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


def test_load_autoresearch_report_and_extract_score(tmp_path):
    report_path = tmp_path / "report.json"
    report_path.write_text(
        """{
  "aggregate_score": 99.5,
  "results": []
}
"""
    )

    report = load_autoresearch_report(report_path)

    assert extract_autoresearch_score(report) == 99.5


def test_should_keep_candidate_uses_strict_improvement_by_default():
    assert should_keep_candidate(99.0, 99.1) is True
    assert should_keep_candidate(99.0, 99.0) is False
    assert should_keep_candidate(99.0, 98.9) is False


def test_should_keep_candidate_can_keep_on_tie():
    assert should_keep_candidate(99.0, 99.0, keep_on_tie=True) is True


def test_build_mutation_prompt_mentions_only_editable_file():
    prompt = build_mutation_prompt(
        editable_relpath="src/autorac/harness/eval_prompt_surface.py",
        program_relpath="program.md",
        baseline_report_relpath="baseline-report.json",
    )

    assert "src/autorac/harness/eval_prompt_surface.py" in prompt
    assert "program.md" in prompt
    assert "baseline-report.json" in prompt
    assert "Do not create, delete, or rename files." in prompt
    assert "separate holdout final-review set" in prompt
    assert "baseline training report is already fully ready" in prompt
    assert "Do not make naming-only, readability-only, or token-count-only edits" in prompt
    assert "Target at most one concrete failure cluster per iteration" in prompt
    assert "preserve that as a real disjunction" in prompt

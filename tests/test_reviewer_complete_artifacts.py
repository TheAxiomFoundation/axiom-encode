"""Reviewers must see executable rules and tests beyond the old excerpt limits."""

import hashlib
import json
from pathlib import Path

import pytest

from axiom_encode.harness import validator_pipeline


@pytest.mark.parametrize(
    "reviewer", ["generalist-reviewer", "rulespec-reviewer", "Formula Reviewer"]
)
def test_reviewer_receives_complete_rules_and_companion_tests(
    tmp_path: Path, monkeypatch, reviewer: str
):
    root = tmp_path / "rulespec-us"
    rules_file = root / "us/statutes/example/lifetime.yaml"
    rules_file.parent.mkdir(parents=True)
    rules = (
        "format: rulespec/v1\n"
        + "# preceding rule context\n" * 350
        + "rules:\n  - name: total\n    kind: derived\n    versions:\n"
        + "      - formula: sum_top_n_over_periods(net_value, selected_count)\n"
    )
    tests = (
        "# preceding fixture context\n" * 150
        + "cases:\n  - name: final history case\n    expected:\n"
        + "      us:statutes/example/lifetime#total: 7\n"
    )
    assert rules.index("sum_top_n_over_periods") > 6000
    assert tests.index("final history case") > 3000
    rules_file.write_text(rules)
    rules_file.with_suffix(".test.yaml").write_text(tests)
    captured = []

    def capture_review(prompt, **_kwargs):
        captured.append(prompt)
        return json.dumps({"score": 10, "passed": True, "issues": []}), 0

    monkeypatch.setattr(validator_pipeline, "run_claude_code", capture_review)
    pipeline = validator_pipeline.ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
    )
    result = pipeline._run_reviewer(reviewer, rules_file)

    assert len(captured) == 1
    assert rules in captured[0]
    assert tests in captured[0]
    assert result.prompt_sha256 == hashlib.sha256(captured[0].encode()).hexdigest()

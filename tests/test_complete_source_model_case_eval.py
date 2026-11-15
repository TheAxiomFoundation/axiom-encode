"""Unresolved source demands stay with the model repair loop."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode.harness import evals

BOUNDARY = (
    "[complete-source-unit:tests] Companion tests do not exercise every "
    "source-stated boundary input; missing: (a)=500."
)


@pytest.mark.parametrize(
    "issue",
    [
        BOUNDARY,
        "[complete-source-unit:tests] Direct nonnegative clamp companion evidence "
        "is missing for `max(0, income)` in `benefit`.",
        "[complete-source-unit:tests] Companion tests do not demonstrate "
        "formula branch (b).",
    ],
)
@pytest.mark.parametrize("complete", [False, True])
def test_gap_feedback_never_appends_expected_cases(tmp_path, issue, complete):
    module = tmp_path / "artifact.yaml"
    module.write_text("format: rulespec/v1\n")
    companion = tmp_path / "artifact.test.yaml"
    companion.write_text("# Model-stated expectations.\ncases: []\n")
    before = (module.read_bytes(), companion.read_bytes())
    metrics = SimpleNamespace(ci_issues=[issue])
    with (
        patch.object(evals, "evaluate_artifact", return_value=metrics) as evaluate,
        patch.object(evals, "_apply_generated_eval_repairs", return_value=[]) as repair,
    ):
        result = evals._evaluate_generated_artifact_with_repairs(
            rulespec_file=module,
            policy_repo_root=tmp_path / "rulespec-us" / "us",
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=complete,
        )
    assert result is metrics
    assert result.ci_issues == [issue]
    assert evaluate.call_count == repair.call_count == 1
    assert (module.read_bytes(), companion.read_bytes()) == before


def test_existing_artifact_repairs_still_revalidate_without_rewriting_cases(tmp_path):
    module = tmp_path / "artifact.yaml"
    module.write_text("before\n")
    companion = tmp_path / "artifact.test.yaml"
    companion.write_text("# Keep the model's values and formatting.\n[]\n")
    original_cases = companion.read_bytes()
    initial = SimpleNamespace(ci_issues=[BOUNDARY])
    repaired = SimpleNamespace(ci_issues=[BOUNDARY])
    calls = []

    def repair(**_kwargs):
        calls.append(module.read_text())
        if len(calls) == 1:
            module.write_text("repaired proof import\n")
            return ["proof_import:x"]
        return []

    with (
        patch.object(evals, "evaluate_artifact", side_effect=[initial, repaired]),
        patch.object(evals, "_apply_generated_eval_repairs", side_effect=repair),
    ):
        result = evals._evaluate_generated_artifact_with_repairs(
            rulespec_file=module,
            policy_repo_root=tmp_path,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            source_text="Source body",
            local_corpus_release=object(),
            require_complete_source_unit=True,
        )
    assert calls == ["before\n", "repaired proof import\n"]
    assert result is repaired
    assert result.ci_issues == [BOUNDARY]
    assert companion.read_bytes() == original_cases

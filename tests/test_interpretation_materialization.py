"""Raw interpretation constraints survive artifact repair and retry overlays."""

from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.evals import (
    ValidationRetryCandidate,
    _materialize_eval_artifact,
    _overlay_validation_retry_candidate,
)
from axiom_encode.harness.proof_validator import (
    find_interpretation_yaml_issues,
    validate_rulespec_proofs,
)

CITATION = "zz/manual/fictional-lantern/review-schedule"
EXCERPT = "Review supplies at the start of each period."


def _content(question="'1e5'"):
    return f"""format: rulespec/v1
module:
  inputs:
    - name: supplied_review_count
      entity: Person
      dtype: Count
      period: Month
  interpretations:
    - id: weekly-review
      question: {question}
      texts:
        - corpus_citation_path: {CITATION}
          excerpt: {EXCERPT}
      reading: Review supplies once a week.
      alternatives:
        - id: daily-review
          reading: Review supplies every day.
      basis: The fictional schedule supports weekly reviews.
      status: open
      affects: [weekly_review_rate]
rules:
  - name: weekly_review_rate
    kind: parameter
    dtype: Float
    versions:
      - effective_from: '2024-01-01'
        formula: '0.2'
"""


INVALID_REPLACEMENTS = [
    ("  interpretations:\n", "  interpretations: []\n  interpretations:\n"),
    ("      question: '1e5'\n", "      question: first\n      question: '1e5'\n"),
    ("      question: '1e5'\n", "      question: ! '1e5'\n"),
    ("      question: '1e5'\n", "      question: !local ordinary prose\n"),
    ("      question: '1e5'\n", "      question: 1e5\n"),
    ("    - id: weekly-review\n", "    - id: ! weekly-review\n"),
    ("  interpretations:\n", "  interpretations: !\n"),
]
VALID_STRINGS = [
    (f"'{value}'" if style == "quoted" else f"!!str {value}", value)
    for value in ["1e5", "-0o17", "+.5", "2e10"]
    for style in ["quoted", "tagged"]
]


def _materialize(tmp_path: Path, content: str, delivery: str) -> str:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    target = artifact_root / "section.yaml"
    response = content
    workspace_root = None
    if delivery == "bundle":
        response = "=== FILE: section.yaml ===\n" + content
    elif delivery == "workspace":
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir()
        (workspace_root / target.name).write_text(content, encoding="utf-8")
        response = "Wrote the requested file."
    assert _materialize_eval_artifact(
        response,
        target,
        workspace_root=workspace_root,
        artifact_root=artifact_root,
    )
    return target.read_text(encoding="utf-8")


@pytest.mark.parametrize("replacement", INVALID_REPLACEMENTS)
@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_preserves_invalid_raw_interpretations(
    tmp_path, replacement, delivery
):
    content = _content().replace(*replacement, 1)
    issues = find_interpretation_yaml_issues(content)
    assert issues
    materialized = _materialize(tmp_path, content, delivery)
    assert materialized == content
    assert find_interpretation_yaml_issues(materialized) == issues


@pytest.mark.parametrize(("scalar", "value"), VALID_STRINGS)
@pytest.mark.parametrize("delivery", ["response", "bundle", "workspace"])
def test_materialization_repairs_module_inputs_without_changing_string_types(
    tmp_path, scalar, value, delivery
):
    materialized = _materialize(tmp_path, _content(scalar), delivery)
    payload = yaml.safe_load(materialized)
    assert "inputs" not in payload["module"]
    assert payload["inputs"][0]["name"] == "supplied_review_count"
    assert payload["module"]["interpretations"][0]["question"] == value
    assert find_interpretation_yaml_issues(materialized) == []
    assert validate_rulespec_proofs(
        materialized, source_texts={CITATION: EXCERPT}
    ).passed


@pytest.mark.parametrize("replacement", INVALID_REPLACEMENTS)
@pytest.mark.parametrize("invalid_side", ["generated", "preserved"])
def test_retry_overlay_rejects_raw_interpretations_before_rewriting(
    tmp_path, replacement, invalid_side
):
    invalid = _content().replace(*replacement, 1)
    generated = invalid if invalid_side == "generated" else _content()
    preserved = invalid if invalid_side == "preserved" else _content()
    target = tmp_path / "section.yaml"
    target.write_text(generated, encoding="utf-8")
    candidate = ValidationRetryCandidate(rulespec=preserved)

    with pytest.raises(
        ValueError,
        match=f"{invalid_side} repair overlay has invalid interpretation syntax",
    ):
        _overlay_validation_retry_candidate(
            target, artifact_root=tmp_path, candidate=candidate
        )

    assert target.read_text(encoding="utf-8") == generated
    assert candidate.rulespec == preserved


@pytest.mark.parametrize(("scalar", "value"), VALID_STRINGS)
def test_retry_overlay_preserves_valid_interpretation_strings(tmp_path, scalar, value):
    content = _content(scalar)
    target = tmp_path / "section.yaml"
    target.write_text(content, encoding="utf-8")
    _overlay_validation_retry_candidate(
        target,
        artifact_root=tmp_path,
        candidate=ValidationRetryCandidate(rulespec=content),
    )
    rewritten = target.read_text(encoding="utf-8")
    assert (
        yaml.safe_load(rewritten)["module"]["interpretations"][0]["question"] == value
    )
    assert find_interpretation_yaml_issues(rewritten) == []
    assert validate_rulespec_proofs(rewritten, source_texts={CITATION: EXCERPT}).passed

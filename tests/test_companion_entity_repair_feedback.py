"""Cross-entity companion failures must be actionable encoder repair feedback."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import axiom_encode.cli as cli_module
from axiom_encode.harness.evals import (
    _eval_artifact_validation_error,
    _format_validation_retry_feedback,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline


def test_cross_entity_companion_ci_failure_reaches_encoder_repair_prompt(
    monkeypatch, tmp_path
):
    fixture_root = Path(__file__).parent / "fixtures" / "companion_declared_entities"
    content_root = tmp_path / "rulespec-us" / "us-nm"
    rules_file = content_root / "statutes" / "7-2-39.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text((fixture_root / "nm.yaml").read_text())
    rules_file.with_suffix(".test.yaml").write_text(
        (fixture_root / "nm.test.yaml").read_text()
    )
    input_name = "claimable_as_dependent_on_another_return"
    output_name = "nm_certain_dependents_person_admission"
    input_key = f"us-nm:statutes/7-2-39#input.{input_name}"
    output_key = f"us-nm:statutes/7-2-39#{output_name}"
    compiled_payload = {
        "program": {
            "parameters": [],
            "relations": [],
            "derived": [
                {
                    "id": output_key,
                    "name": output_name,
                    "entity": "TaxUnit",
                    "expr": {"kind": "input", "name": input_key},
                }
            ],
        }
    }
    pipeline = ValidatorPipeline(
        policy_repo_path=content_root,
        axiom_rules_path=tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    def compile_fixture(_rules_file, compiled_path):
        compiled_path.write_text(json.dumps(compiled_payload))
        return subprocess.CompletedProcess([], 0, "", ""), compiled_payload

    def execute_fixture(command, *, input, **_kwargs):
        request = json.loads(input)
        query_id = request["queries"][0]["entity_id"]
        facts = request["dataset"]["inputs"]
        if not any(
            fact["name"] == input_key
            and fact["entity"] == "TaxUnit"
            and fact["entity_id"] == query_id
            for fact in facts
        ):
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                f"missing input {input_name} for entity {query_id}",
            )
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "results": [
                        {
                            **query,
                            "outputs": {
                                output_key: {"kind": "judgment", "outcome": "holds"}
                            },
                        }
                        for query in request["queries"]
                    ]
                }
            ),
            "",
        )

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", compile_fixture)
    monkeypatch.setattr(pipeline, "_axiom_rules_binary", lambda: tmp_path / "engine")
    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.subprocess.run", execute_fixture
    )

    ci_result = pipeline._run_rulespec_ci(rules_file)
    assert not ci_result.passed
    finding = next(
        issue
        for issue in ci_result.issues
        if input_name in issue
        and output_name in issue
        and "Person" in issue
        and "TaxUnit" in issue
        and "relation" in issue
    )
    metrics = SimpleNamespace(
        compile_pass=True,
        ci_pass=ci_result.passed,
        compile_issues=[],
        ci_issues=ci_result.issues,
    )
    validation_error = _eval_artifact_validation_error(metrics)
    assert validation_error == "Generated RuleSpec failed CI validation"
    failed_attempt = cli_module._FailedEncodeAttempt(
        result=SimpleNamespace(metrics=metrics),
        error=validation_error,
        validation_issues=cli_module._validation_retry_issue_snapshot(
            metrics.compile_issues, metrics.ci_issues
        ),
    )
    feedback = cli_module._encode_validation_retry_feedback([failed_attempt])
    assert finding in feedback
    rendered_prompt = _format_validation_retry_feedback(feedback)
    assert json.dumps(finding, ensure_ascii=False) in rendered_prompt
    assert "Correct every listed issue in this candidate" in rendered_prompt


@pytest.mark.parametrize("case_input", [None, "", {}])
def test_empty_input_missing_fact_remains_an_execution_finding(
    monkeypatch, tmp_path, case_input
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=tmp_path,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline.subprocess.run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, "", "missing input `claimant` for entity case"
        ),
    )
    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={"input": case_input, "output": {"eligibility": "holds"}},
        case_name="empty-input",
        case_index=0,
        period={"period_kind": "month", "start": "2026-01-01", "end": "2026-01-31"},
        output_names=["eligibility"],
        derived_by_key={"eligibility": {"entity": "Household"}},
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        module_target=None,
        declared_relation_names=set(),
    )
    assert outputs is None
    assert len(issues) == 1
    assert "execution failed: missing input `claimant`" in issues[0]

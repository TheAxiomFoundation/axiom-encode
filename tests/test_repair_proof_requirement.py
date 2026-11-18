"""Partial repair omissions must not switch off retained proof checks."""

from copy import deepcopy

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.harness.proof_validator import find_rulespec_proof_issues
from axiom_encode.harness.validator_pipeline import find_test_input_assignment_issues

ABSENT = object()


def documents(tmp_path, module=ABSENT, retained_proof=None):
    retained = {
        "format": "rulespec/v1",
        "module": {"proof_validation": retained_proof or {"required": True}},
        "inputs": [
            {"name": "fact", "entity": "Person", "dtype": "Integer", "period": "Year"}
        ],
        "rules": [
            {
                "name": "result",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Year",
                "versions": [{"effective_from": "2025-01-01", "formula": "fact + 1"}],
            }
        ],
    }
    generated = {"format": "rulespec/v1", "rules": []}
    if module is not ABSENT:
        generated["module"] = deepcopy(module)
    path = tmp_path / "repair.yaml"
    path.write_text(yaml.safe_dump(generated))
    candidate = evals.ValidationRetryCandidate(rulespec=yaml.safe_dump(retained))
    return path, candidate


@pytest.mark.parametrize(
    "module",
    [ABSENT, {}, {"proof_validation": {}}, {"proof_validation": {"required": True}}],
)
def test_omission_keeps_real_proof_and_input_checks_enabled(tmp_path, module):
    path, candidate = documents(tmp_path, module)
    repairs = evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    content = path.read_text()
    assert yaml.safe_load(content)["module"]["proof_validation"]["required"] is True
    assert bool(repairs.count("module:proof_validation.required")) == (
        module != {"proof_validation": {"required": True}}
    )
    assert find_rulespec_proof_issues(content)
    cases = [
        {"name": "missing fact", "period": "2025", "input": {}, "output": {"result": 1}}
    ]
    assert find_test_input_assignment_issues(content, cases)


@pytest.mark.parametrize("value", [False, None, 0, 1, "true", "false", [], {}])
def test_explicit_downgrade_refused_without_writes(tmp_path, value):
    path, candidate = documents(tmp_path, {"proof_validation": {"required": value}})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="cannot disable"):
        evals._overlay_validation_retry_candidate(
            path, artifact_root=tmp_path, candidate=candidate
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "module",
    [
        None,
        [],
        "module",
        {"proof_validation": None},
        {"proof_validation": []},
        {"proof_validation": True},
    ],
)
def test_malformed_generated_requirement_refused(tmp_path, module):
    path, candidate = documents(tmp_path, module)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="repair cannot"):
        evals._overlay_validation_retry_candidate(
            path, artifact_root=tmp_path, candidate=candidate
        )
    assert path.read_bytes() == before


def test_only_required_flag_restored_not_old_source_or_summary(tmp_path):
    module = {
        "summary": "new summary",
        "source_verification": {"source_sha256": "new"},
        "proof_validation": {"extra": "retained generated setting"},
    }
    path, candidate = documents(tmp_path, module)
    evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert yaml.safe_load(path.read_text())["module"] == {
        **module,
        "proof_validation": {**module["proof_validation"], "required": True},
    }


@pytest.mark.parametrize("proof", [ABSENT, {}, {"required": False}])
def test_non_strict_base_does_not_invent_requirement(tmp_path, proof):
    path, candidate = documents(tmp_path, {})
    retained = yaml.safe_load(candidate.rulespec)
    if proof is ABSENT:
        retained["module"].pop("proof_validation")
    else:
        retained["module"]["proof_validation"] = proof
    candidate = evals.ValidationRetryCandidate(rulespec=yaml.safe_dump(retained))
    evals._overlay_validation_retry_candidate(
        path, artifact_root=tmp_path, candidate=candidate
    )
    assert "proof_validation" not in yaml.safe_load(path.read_text())["module"]


@pytest.mark.parametrize(
    "proof", [None, [], True, {"required": None}, {"required": 1}, {"required": "true"}]
)
def test_malformed_retained_requirement_is_invalid_base(tmp_path, proof):
    path, candidate = documents(tmp_path, {})
    retained = yaml.safe_load(candidate.rulespec)
    retained["module"]["proof_validation"] = proof
    candidate = evals.ValidationRetryCandidate(rulespec=yaml.safe_dump(retained))
    before = path.read_bytes()
    with pytest.raises(evals._PreservedRepairOverlayError, match="proof_validation"):
        evals._overlay_validation_retry_candidate(
            path, artifact_root=tmp_path, candidate=candidate
        )
    assert path.read_bytes() == before

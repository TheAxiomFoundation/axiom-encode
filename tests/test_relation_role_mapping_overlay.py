"""Normalize synthetic input declarations before retained-candidate repairs."""

import subprocess

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.prompts.encoder import RELATION_ROLES_SHAPE

LABEL = "# Synthetic encoder test input; not a RuleSpec module.\n"


def overlay_payload(formula, mapping):
    declaration = {"dtype": "Integer", "entity": "Person"}
    return {
        "format": "rulespec/v1",
        "inputs": {"amount": declaration}
        if mapping
        else [{"name": "amount", **declaration}],
        "rules": [
            {
                "name": "result",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            }
        ],
    }


@pytest.mark.parametrize("retained_mapping", [False, True])
@pytest.mark.parametrize("repair_mapping", [False, True])
def test_a2_paths_repair_mapping_normalizes_both_overlay_sides(
    tmp_path, retained_mapping, repair_mapping
):
    candidate = evals.ValidationRetryCandidate(
        rulespec=LABEL + yaml.safe_dump(overlay_payload("amount", retained_mapping))
    )
    assert evals._repair_overlay_candidate_base_issue(candidate) is None
    artifact_root = tmp_path / "generated"
    module = artifact_root / "section.yaml"
    module.parent.mkdir()
    module.write_text(
        LABEL + yaml.safe_dump(overlay_payload("amount + amount", repair_mapping))
    )
    assert (
        evals._overlay_validation_retry_candidate(
            module, artifact_root=artifact_root, candidate=candidate
        )
        == ()
    )
    merged = yaml.safe_load(module.read_text())
    assert merged["inputs"] == [
        {"name": "amount", "dtype": "Integer", "entity": "Person"}
    ]
    assert merged["rules"][0]["versions"][0]["formula"] == "amount + amount"


@pytest.mark.parametrize(
    "inputs",
    [None, "amount", 7, {"amount": "Integer"}, {7: {}}, {"amount": {"name": "other"}}],
)
def test_overlay_keeps_refusing_invalid_input_declarations(inputs):
    candidate = evals.ValidationRetryCandidate(
        rulespec=LABEL
        + yaml.safe_dump({"format": "rulespec/v1", "inputs": inputs, "rules": []})
    )
    assert evals._repair_overlay_candidate_base_issue(candidate) is not None


def test_a2_mapping_candidate_repair_compiles(tmp_path, real_engine):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/overlay.yaml"
    module.parent.mkdir(parents=True)
    candidate = evals.ValidationRetryCandidate(
        rulespec=LABEL + yaml.safe_dump(overlay_payload("amount", True))
    )
    module.write_text(LABEL + yaml.safe_dump(overlay_payload("amount + amount", True)))
    evals._overlay_validation_retry_candidate(
        module, artifact_root=root, candidate=candidate
    )
    compiled = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(module),
            "--rulespec-root",
            str(root),
            "--output",
            str(tmp_path / "compiled.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert compiled.returncode == 0, compiled.stderr


def test_worked_relation_role_shape_teaches_list_inputs():
    shape = yaml.safe_load(
        RELATION_ROLES_SHAPE.split("```yaml\n", 1)[1].split("```", 1)[0]
    )
    assert isinstance(shape["inputs"], list)
    assert [entry["name"] for entry in shape["inputs"]] == [
        "meal_group",
        "household_exception_may_apply",
        "snap_household_unit",
        "ineligible_for_participation",
    ]

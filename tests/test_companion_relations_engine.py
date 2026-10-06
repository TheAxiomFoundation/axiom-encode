"""Exercise companion request binding with the real strict-binding engine."""

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from axiom_encode.cli import _execute_rulespec_test_file

_LOCAL_ENGINE_DIR = Path(
    "/Users/maxghenis/TheAxiomFoundation/_worktrees/engine-findings"
)


@pytest.fixture
def strict_engine_binary():
    binary = Path(
        os.environ.get(
            "AXIOM_STRICT_ENGINE_BINARY",
            _LOCAL_ENGINE_DIR / "bin-relation-binding",
        )
    )
    if not binary.is_file() or not os.access(binary, os.X_OK):
        pytest.skip("set AXIOM_STRICT_ENGINE_BINARY to a strict-binding engine binary")
    return binary


@pytest.mark.parametrize(
    "slot_entities", [["TaxUnit", "Person"], ["Person", "TaxUnit"]]
)
@pytest.mark.parametrize("declaration_matches_usage", [True, False])
def test_companion_request_binds_and_matches_real_engine_reference(
    tmp_path,
    monkeypatch,
    strict_engine_binary,
    slot_entities,
    declaration_matches_usage,
):
    # This is an engineering fixture, not an encoding of a tax provision.
    repo = tmp_path / "rulespec-us"
    program_file = repo / "us/statutes/1/companion.yaml"
    program_file.parent.mkdir(parents=True)
    program_file.write_text(
        """format: rulespec/v1
rules:
  - name: qualifying_child_of_tax_unit
    kind: data_relation
    data_relation:
      arity: 2
      arguments: SLOT_ENTITIES
  - name: qualifying_child
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: is_eligible
  - name: credit
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: case_value + count_where(qualifying_child_of_tax_unit, qualifying_child) * 1500
""".replace("SLOT_ENTITIES", json.dumps(slot_entities))
    )
    module = "us:statutes/1/companion"
    test_file = program_file.with_suffix(".test.yaml")
    test_file.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "one_eligible_child",
                    "period": 2026,
                    "input": {
                        f"{module}#input.case_value": 0,
                        f"{module}#relation.qualifying_child_of_tax_unit": [
                            {f"{module}#input.is_eligible": True}
                        ],
                    },
                    "output": {f"{module}#credit": 1500},
                }
            ],
            sort_keys=False,
        )
    )
    calls = []
    real_run = subprocess.run

    def capture_run(command, **kwargs):
        # Observe the producer's actual request and real engine response. Every
        # compile/run still executes the binary; no evaluator is substituted.
        result = real_run(command, **kwargs)
        if (
            command[1] == "compile"
            and result.returncode == 0
            and not declaration_matches_usage
        ):
            # Historical typed artifacts may retain declarations that disagree
            # with their executable slots. Keep the real compiled expressions,
            # changing only that declaration to exercise usage precedence.
            artifact_path = Path(command[command.index("--output") + 1])
            artifact = json.loads(artifact_path.read_text())
            for relation in artifact["program"]["relations"]:
                relation["slot_entities"].reverse()
            artifact_path.write_text(json.dumps(artifact))
        if command[1] == "run-compiled":
            calls.append((command, json.loads(kwargs["input"]), result))
        return result

    monkeypatch.setattr("axiom_encode.cli.subprocess.run", capture_run)
    compiled_cache = {}
    result = _execute_rulespec_test_file(
        test_file,
        binary=strict_engine_binary,
        axiom_rules_path=tmp_path,
        env=os.environ.copy(),
        rulespec_roots=[repo],
        tmp_path=tmp_path,
        compiled_cache=compiled_cache,
        policy_repo_path=repo / "us",
    )
    assert result == {"cases": 1, "compiled": 1, "failures": []}
    artifact = compiled_cache[program_file][1]
    assert artifact["program"]["relations"][0]["slot_entities"] == (
        slot_entities if declaration_matches_usage else list(reversed(slot_entities))
    )
    assert len(calls) == 1
    command, request, execution = calls[0]
    assert "relation_binding" not in request
    assert "--relation-binding" not in command
    response = json.loads(execution.stdout)
    assert response["metadata"]["relation_binding"] == "strict"
    assert {record["entity"] for record in request["dataset"]["inputs"]} == {
        "TaxUnit",
        "Person",
    }

    reference = copy.deepcopy(request)
    ids_by_kind = {
        record["entity"]: record["entity_id"] for record in request["dataset"]["inputs"]
    }
    for relation in reference["dataset"]["relations"]:
        relation["tuple"] = [ids_by_kind[kind] for kind in slot_entities]
    reference_result = real_run(
        command,
        input=json.dumps(reference),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert reference_result.returncode == 0, reference_result.stderr
    reference_response = json.loads(reference_result.stdout)
    assert response["results"] == reference_response["results"]
    output = next(iter(response["results"][0]["outputs"].values()))
    assert float(output["value"]["value"]) == 1500

    # Reproduce the pre-fix producer's exact tuple/label choices. Strict binding
    # must reject this, even when its tuple happens to follow a reverse declaration.
    legacy = copy.deepcopy(request)
    for relation in legacy["dataset"]["relations"]:
        relation["tuple"] = ["related_0", "case"]
    for record in legacy["dataset"]["inputs"]:
        if record["entity"] == "Person":
            record["entity_id"] = "related_0"
        record["entity"] = "Entity"
    legacy_result = real_run(
        command,
        input=json.dumps(legacy),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert legacy_result.returncode != 0
    assert "strict dataset relation entity validation failed" in legacy_result.stderr
    assert "found `Entity`" in legacy_result.stderr

    # The pre-strict main engine silently accepted the old owner-first mismatch
    # and returned zero. Keep that before/after evidence when the binary exists.
    main_binary = Path(
        os.environ.get(
            "AXIOM_MAIN_ENGINE_BINARY", _LOCAL_ENGINE_DIR / "bin-main-5a29e03"
        )
    )
    if slot_entities == ["TaxUnit", "Person"] and main_binary.is_file():
        old_result = real_run(
            [str(main_binary), *command[1:]],
            input=json.dumps(legacy),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert old_result.returncode == 0, old_result.stderr
        old_output = next(
            iter(json.loads(old_result.stdout)["results"][0]["outputs"].values())
        )
        assert float(old_output["value"]["value"]) == 0

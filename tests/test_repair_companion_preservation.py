"""A policy proof repair must not silently weaken retained test coverage."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.prepare_signed_backfill import (
    immutable_atomic_source_contract,
    repair_execution_metadata,
    resolve_atomic_repair_mode,
    split_atomic_source_input,
)
from axiom_encode.repair_companion import preserved_companion_record


@pytest.fixture
def retained():
    return [
        {
            "name": "repair_ei_threshold_entered",
            "period": "2025",
            "input": {"entered": True, "income": 82500},
            "output": {
                "deduction_capped": 0,
                "deductions_total": 0,
                "income_before_adjustments": 82500,
                "repayment": 0,
                "net_income": 82500,
                "net_income_signed": 82500,
            },
        },
        {"name": "zero", "period": "2025", "input": {}, "output": {"x": 0}},
    ]


def test_current_failure_drops_five_outputs(retained):
    proposed = copy.deepcopy(retained)
    proposed[0]["output"] = {"net_income_signed": 82500}
    with pytest.raises(ValueError, match="facts or assertions"):
        preserved_companion_record(yaml.safe_dump(retained), yaml.safe_dump(proposed))


def test_reordering_and_presentation_preserve_byte_binding(retained):
    original = yaml.safe_dump(retained)
    proposed = yaml.safe_dump(list(reversed(retained)), default_flow_style=True)
    record = preserved_companion_record(original, proposed)
    assert record["cases"] == 2
    assert record["assertions"] == 7
    assert (
        record["original_tests_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    )
    assert record["final_tests_sha256"] == hashlib.sha256(proposed.encode()).hexdigest()


@pytest.mark.parametrize(
    "mutation",
    ["case", "input", "period", "expected", "bool", "new_output", "duplicate"],
)
def test_unauthorized_changes_refuse(retained, mutation):
    proposed = copy.deepcopy(retained)
    if mutation == "case":
        proposed.pop()
    elif mutation == "input":
        proposed[0]["input"]["income"] = 82501
    elif mutation == "period":
        proposed[0]["period"] = "2026"
    elif mutation == "expected":
        proposed[0]["output"]["net_income"] = 1
    elif mutation == "bool":
        proposed[0]["output"]["repayment"] = False
    elif mutation == "new_output":
        proposed[0]["output"]["unreviewed"] = 1
    else:
        proposed.append(copy.deepcopy(proposed[0]))
    with pytest.raises(ValueError):
        preserved_companion_record(yaml.safe_dump(retained), yaml.safe_dump(proposed))


def test_duplicate_yaml_key_refuses(retained):
    original = yaml.safe_dump(retained)
    proposed = original.replace("repayment: 0", "repayment: 1\n    repayment: 0")
    with pytest.raises(ValueError):
        preserved_companion_record(original, proposed)


def test_container_change_refuses(retained):
    with pytest.raises(ValueError):
        preserved_companion_record(
            yaml.safe_dump(retained), yaml.safe_dump({"cases": retained})
        )


def descriptor(**changes):
    return json.dumps(
        {
            "schema": "axiom-encode/atomic-source-transaction/v7",
            "transaction": [],
            "preserve_repair_companion": True,
            **changes,
        }
    )


def test_preservation_descriptor_keeps_historical_source_identity():
    assert immutable_atomic_source_contract(descriptor()) == split_atomic_source_input(
        "[]"
    )
    selection = resolve_atomic_repair_mode(
        descriptor(),
        {
            "REPAIR_RUN_ID": "41127692566",
            "CITATION": "ca/policy/cra/t1-2025/t1-income-net-taxable-income/step-3",
            "EXISTING_SIGNED_IMPORTS_JSON": '["ca/policies/step-2.yaml"]',
        },
    )
    assert selection == {
        "mode": "full_artifact",
        "tests_only": False,
        "preserve_repair_companion": True,
    }
    assert (
        repair_execution_metadata(
            descriptor(), "41127692566", "full_artifact", "false", "true"
        )["preserve_repair_companion"]
        is True
    )


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_preservation_flag_is_exact_true(value):
    with pytest.raises(ValueError):
        split_atomic_source_input(descriptor(preserve_repair_companion=value))


@pytest.mark.parametrize(
    "changes",
    [{"transaction": {}}, {"extra": True}, {"transaction": ["ca/policy/other"]}],
)
def test_preservation_descriptor_is_closed(changes):
    with pytest.raises(ValueError):
        split_atomic_source_input(descriptor(**changes))


@pytest.mark.parametrize(
    "field", ["DEPENDENT_CITATION", "QUEUE_ID", "REPLACE_LEGACY_RULESPEC_PATH"]
)
def test_preservation_refuses_other_lanes(field):
    with pytest.raises(ValueError):
        resolve_atomic_repair_mode(descriptor(), {"REPAIR_RUN_ID": "123", field: "x"})


@pytest.mark.parametrize("selected", ["", "false", "1"])
def test_missing_or_downgraded_execution_selection_refuses(selected):
    with pytest.raises(ValueError):
        repair_execution_metadata(
            descriptor(), "123", "full_artifact", "false", selected
        )


def test_default_metadata_has_no_new_field():
    assert repair_execution_metadata("[]", "123", "full_artifact", "false") == {
        "requested_mode": "legacy_inference",
        "mode": "full_artifact",
        "tests_only": False,
    }


def test_final_gate_uses_original_seed_after_late_output_loss(retained, tmp_path):
    from axiom_encode import cli
    from axiom_encode.harness.evals import ValidationRetryCandidate

    original = yaml.safe_dump(retained)
    seed = ValidationRetryCandidate(rulespec="format: rulespec/v1\n", tests=original)
    result = SimpleNamespace()
    setattr(result, cli._PRESERVED_REPAIR_COMPANION_ATTR, seed)
    target = tmp_path / "step-3.test.yaml"
    target.write_text(original)
    assert cli._preserved_repair_companion_record(result, target)["assertions"] == 7
    weakened = copy.deepcopy(retained)
    weakened[0]["output"] = {"net_income_signed": 82500}
    target.write_text(yaml.safe_dump(weakened))
    with pytest.raises(RuntimeError, match="Cannot apply repair"):
        cli._preserved_repair_companion_record(result, target)
    # No opt-in baseline: ordinary source-backed repairs keep their behavior.
    assert cli._preserved_repair_companion_record(SimpleNamespace(), target) is None


def test_mode_requires_authenticated_seed_and_full_source(retained):
    from axiom_encode import cli
    from axiom_encode.harness.evals import ValidationRetryCandidate

    seed = ValidationRetryCandidate(
        rulespec="format: rulespec/v1\n", tests=yaml.safe_dump(retained)
    )
    args = SimpleNamespace(
        preserve_repair_companion=True,
        apply=True,
        require_complete_source_unit=True,
        citation="ca/policy/example/step-3",
        repair_candidate_path=Path("policies/example/step-3.yaml"),
    )
    cli._validate_preserved_repair_companion(args, seed)
    with pytest.raises(ValueError):
        cli._validate_preserved_repair_companion(args, None)
    args.require_complete_source_unit = False
    with pytest.raises(ValueError):
        cli._validate_preserved_repair_companion(args, seed)
    args.require_complete_source_unit = True
    args.repair_candidate_tests_only = True
    with pytest.raises(ValueError):
        cli._validate_preserved_repair_companion(args, seed)


@pytest.mark.parametrize("mutation", ["missing", "seed", "final", "count", "unknown"])
def test_packaging_refuses_downgraded_or_misbound_signed_record(retained, mutation):
    from axiom_encode.repair_companion import verify_preservation_receipt

    raw = yaml.safe_dump(retained)
    record = {
        **preserved_companion_record(raw, raw),
        "original_rulespec_sha256": "a" * 64,
    }
    expected = {
        key: record[key]
        for key in (
            "original_rulespec_sha256",
            "original_tests_sha256",
            "final_tests_sha256",
        )
    }
    verify_preservation_receipt(record, **expected)
    if mutation == "missing":
        record = None
    elif mutation == "seed":
        record["original_tests_sha256"] = "b" * 64
    elif mutation == "final":
        record["final_tests_sha256"] = "b" * 64
    elif mutation == "count":
        record["cases"] = True
    else:
        record["ignored"] = True
    with pytest.raises(ValueError):
        verify_preservation_receipt(record, **expected)


def test_v7_authenticates_old_source_transaction_without_relabeling():
    from scripts.extract_repair_candidate import _repair_lane_for_atomic_source

    assert _repair_lane_for_atomic_source(
        {"atomic_source_input": "[]"}, descriptor()
    ) == ("target", ["target"])
    with pytest.raises(ValueError, match="preservation mode"):
        _repair_lane_for_atomic_source(
            {"atomic_source_input": descriptor()}, descriptor()
        )
    metadata = {
        "atomic_source_input": descriptor(),
        "repair_execution": repair_execution_metadata(
            descriptor(), "123", "full_artifact", "false", "true"
        ),
    }
    assert _repair_lane_for_atomic_source(metadata, descriptor()) == (
        "target",
        ["target"],
    )


def test_final_receipt_hashes_crlf_bytes_without_normalization(retained, tmp_path):
    from axiom_encode import cli
    from axiom_encode.harness.evals import ValidationRetryCandidate

    original = yaml.safe_dump(retained)
    seed = ValidationRetryCandidate(rulespec="format: rulespec/v1\n", tests=original)
    result = SimpleNamespace()
    setattr(result, cli._PRESERVED_REPAIR_COMPANION_ATTR, seed)
    raw = original.replace("\n", "\r\n").encode()
    target = tmp_path / "companion.test.yaml"
    target.write_bytes(raw)
    record = cli._preserved_repair_companion_record(result, target)
    assert record["final_tests_sha256"] == hashlib.sha256(raw).hexdigest()
    assert (
        record["original_tests_sha256"] == hashlib.sha256(original.encode()).hexdigest()
    )


@pytest.mark.parametrize(
    "mutation", [None, "record", "signature", "unsigned", "root", "duplicate"]
)
def test_packaging_checks_actual_signed_manifest_bytes(retained, monkeypatch, mutation):
    from axiom_encode import cli
    from axiom_encode.repair_companion import verified_preservation_manifest
    from tests.eval_evidence_fixtures import (
        TEST_APPLY_PRIVATE_KEY_B64,
        TEST_APPLY_PUBLIC_KEY_B64,
    )
    from tests.signing_broker_fixtures import SigningBrokerFixture

    broker = SigningBrokerFixture(
        apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
        apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
    )
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: broker)
    text = yaml.safe_dump(retained)
    record = {
        **preserved_companion_record(text, text),
        "original_rulespec_sha256": "a" * 64,
    }
    expected = {
        key: record[key]
        for key in (
            "original_rulespec_sha256",
            "original_tests_sha256",
            "final_tests_sha256",
        )
    }
    payload = {
        "schema_version": "axiom-encode/applied-rulespec/v5",
        "tool": "axiom-encode encode --apply",
        "repair_companion_preservation": record,
    }
    cli._sign_applied_encoding_manifest(payload, broker)
    if mutation == "record":
        record["cases"] += 1
    elif mutation == "signature":
        payload["signature"]["value"] = "invalid"
    elif mutation == "unsigned":
        del payload["signature"]
    elif mutation == "root":
        monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: None)
    raw = json.dumps(payload).encode()
    if mutation == "duplicate":
        raw = raw[:-1] + b', "tool": "axiom-encode encode --apply"}'
    if mutation is None:
        assert (
            verified_preservation_manifest(raw, **expected)
            == hashlib.sha256(raw).hexdigest()
        )
    else:
        with pytest.raises(ValueError):
            verified_preservation_manifest(raw, **expected)


@pytest.mark.parametrize("late_failure", [False, True])
def test_guard_emits_preservation_only_after_all_checks_pass(
    tmp_path, monkeypatch, capsys, late_failure
):
    from axiom_encode import cli, repair_companion

    repo = tmp_path / "rulespec-ca"
    repo.mkdir()
    expectation = ["ca/policy/example/step-3", "a" * 64, "b" * 64]
    evidence = {"verified": "exact"}
    called = []
    monkeypatch.setattr(
        cli, "_resolve_canonical_rulespec_checkout", lambda *a, **k: repo
    )
    monkeypatch.setattr(
        cli,
        "guard_generated_change_issues",
        lambda *a, **k: ["late unrelated failure"] if late_failure else [],
    )

    def checked(root, *values):
        assert root == repo and list(values) == expectation
        called.append(True)
        return evidence

    monkeypatch.setattr(repair_companion, "guard_preservation_evidence", checked)
    args = SimpleNamespace(
        repo=repo,
        base_ref="base",
        head_ref="HEAD",
        corpus_path=tmp_path,
        json=True,
        preserved_repair_companion=expectation,
    )
    with pytest.raises(SystemExit) as exit_info:
        cli.cmd_guard_generated(args)
    report = json.loads(capsys.readouterr().out)
    assert exit_info.value.code == int(late_failure)
    assert bool(called) is not late_failure
    if late_failure:
        assert "repair_companion_preservation" not in report
    else:
        assert report["repair_companion_preservation"] == evidence


def test_guard_workflow_routes_expectation_through_supervisor_only():
    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/targeted-signed-reencode.yml"
        ).read_text()
    )
    steps = next(
        job["steps"]
        for job in workflow["jobs"].values()
        if any(
            step.get("id") == "verify_generated_provenance"
            for step in job.get("steps", [])
        )
    )
    step = next(
        step for step in steps if step.get("id") == "verify_generated_provenance"
    )
    script = step["run"]
    assert 'if [ "$PRESERVE_REPAIR_COMPANION" = true ]' in script
    assert 'preservation_args=(--preserved-repair-companion "$CITATION"' in script
    assert "-- /opt/axiom-verification/axiom-encode guard-generated" in script
    assert '"${preservation_args[@]}"' in script
    assert (
        "steps.repair_candidate.outputs.rulespec_sha256"
        in step["env"]["REPAIR_CANDIDATE_RULESPEC_SHA256"]
    )


@pytest.mark.parametrize(
    "mutation",
    [None, "manifest", "companion", "seed", "failed_guard", "missing_record"],
)
def test_actual_packaging_preservation_block_binds_verified_bytes(
    tmp_path, monkeypatch, mutation
):
    import os
    import sys
    import textwrap

    workflow = yaml.safe_load(
        (
            Path(__file__).parents[1] / ".github/workflows/targeted-signed-reencode.yml"
        ).read_text()
    )
    script = next(
        step["run"]
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if "expected_preservation = {" in step.get("run", "")
    )
    start = (
        script.index("                  guard = json.loads(")
        if "                  guard = json.loads(" in script
        else script.index("        guard = json.loads(")
    )
    end = script.index(
        "\n",
        script.index(
            'preservation_digests[relative_path] = expected_preservation["manifest_sha256"]',
            start,
        ),
    )
    block = textwrap.dedent(script[start:end])
    manifest_path = Path(".axiom/encoding-manifests/ca/policies/example.json")
    companion = Path("ca/policies/example.test.yaml")
    raw = b'{"signed": "verified by supervisor"}'
    tests = b"original companion\n"
    seed = "a" * 64
    monkeypatch.setenv("REPAIR_CANDIDATE_RULESPEC_SHA256", seed)
    monkeypatch.setenv("REPAIR_CANDIDATE_TESTS_SHA256", "b" * 64)
    record = {
        "schema": "axiom-encode/verified-repair-companion/v1",
        "citation": "ca/policy/example",
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "companion_path": companion.as_posix(),
        "original_rulespec_sha256": seed,
        "original_tests_sha256": "b" * 64,
        "final_tests_sha256": hashlib.sha256(tests).hexdigest(),
    }
    guard = {"passed": True, "issues": [], "repair_companion_preservation": record}
    if mutation == "manifest":
        raw += b"changed"
    elif mutation == "companion":
        tests += b"changed"
    elif mutation == "seed":
        monkeypatch.setenv("REPAIR_CANDIDATE_RULESPEC_SHA256", "c" * 64)
    elif mutation == "failed_guard":
        guard["passed"] = False
    elif mutation == "missing_record":
        del guard["repair_companion_preservation"]
    (tmp_path / "guard-generated.json").write_text(json.dumps(guard))
    monkeypatch.setattr(
        sys, "argv", ["packaging", str(tmp_path / "context-manifest.json")]
    )
    scope = dict(
        json=json,
        Path=Path,
        sys=sys,
        hashlib=hashlib,
        os=os,
        citation="ca/policy/example",
        relative_path=manifest_path,
        captured_manifests={manifest_path: raw},
        companion=companion,
        companion_bytes=tests,
        preservation_digests={},
    )
    if mutation is None:
        exec(
            compile(block, "actual workflow packaging preservation block", "exec"),
            scope,
        )
        assert scope["preservation_digests"] == {
            manifest_path: hashlib.sha256(raw).hexdigest()
        }
    else:
        with pytest.raises(SystemExit, match="Preservation evidence"):
            exec(
                compile(block, "actual workflow packaging preservation block", "exec"),
                scope,
            )

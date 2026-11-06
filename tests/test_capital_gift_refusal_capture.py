"""Synthetic materializer around historical actual records; no fresh authority."""

import copy
import json
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import capital_gift_capture as c
from tests import test_capital_gift_execution_capture as positive_fixture
from tests.test_capital_gift_refusals import REFUSALS


@pytest.fixture
def mixed(tmp_path, monkeypatch):
    cases = copy.deepcopy(positive_fixture.CASES)
    calls = copy.deepcopy(positive_fixture.ACTUAL["calls"])
    cases.append(REFUSALS["separately_executed_positive"])
    calls.append(REFUSALS["separately_executed_positive_call"])
    cases[1:1] = REFUSALS["cases"]
    calls[1:1] = REFUSALS["calls"]
    from axiom_encode.harness import same_case_execution as s

    original, snapshot = tmp_path / "rulespec-ca", tmp_path / "snapshot" / "rulespec-ca"
    relative = "ca/policies/cra/t1-2025/capital-gains-on-gifts.yaml"
    files = []
    for name, raw in [
        (relative, positive_fixture.STATIC["policy_yaml"].encode()),
        (
            relative.replace(".yaml", ".test.yaml"),
            yaml.safe_dump(cases, sort_keys=False).encode(),
        ),
    ]:
        for root in (original, snapshot):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        files.append(s.ClosureFile(name, original / name, raw))
    binary, artifact = tmp_path / "binary", tmp_path / "compiled.json"
    binary.write_bytes(b"synthetic binary; no actual engine authority")
    artifact.write_text(json.dumps(positive_fixture.ACTUAL["artifact"]))
    cap = s.ExecutionCapture(
        original,
        snapshot,
        relative,
        tuple(files),
        binary,
        s.digest(binary.read_bytes()),
        positive_fixture.ENGINE_PIN,
        0,
        "synthetic-head",
    )
    source = (
        (positive_fixture.F / "t1170-authenticated-body.txt").read_text(),
        positive_fixture.STATIC["p113_body"],
        ("synthetic resolver",),
    )
    monkeypatch.setattr(c, "_sources", lambda release: source)
    monkeypatch.setattr(
        s, "_materialized_capture", lambda obj, ref=weakref.ref(cap): obj is ref()
    )
    assert c.activate(SimpleNamespace(local_corpus_release=object()), cap)
    cap._capital_gift_owned = True
    cap.compiled(artifact)
    assert cap._capital_gift_context is not None
    return cap, cases, calls


def stage_all(cap, cases, calls):
    for i, (case, call) in enumerate(zip(cases, calls, strict=True), 1):
        if "expected_refusal" in case:
            assert cap.stage_refusal(
                case,
                call["request"],
                call["returncode"],
                call["stdout"],
                call["stderr"],
                cap.binary,
                cap.artifact,
                i,
            )
        else:
            cap.stage(
                case,
                call["request"],
                json.loads(call["stdout"]),
                cap.binary,
                cap.artifact,
                i,
            )


def test_private_mixed_ordinals_keep_refusals_out_of_successful_records(mixed):
    cap, cases, calls = mixed
    stage_all(cap, cases, calls)
    evidence = cap.evidence()
    assert len(evidence.records) == 83
    assert len(cap._capital_gift_refusals) == 18
    observed = c.observation(evidence, positive_fixture.STATIC["policy_yaml"])
    assert observed is not None
    assert len(observed.refusals) == 18
    assert len(observed.positive.records) == 83
    assert sum(len(r.outputs) for r in observed.positive.records) == 362
    assert (
        c.observation(replace(evidence), positive_fixture.STATIC["policy_yaml"]) is None
    )


@pytest.mark.parametrize(
    "mutation", ["ordinal", "case", "error", "owner", "positive_stage"]
)
def test_refusal_staging_does_not_use_generic_fallback(mixed, mutation):
    cap, cases, calls = mixed
    case, call, ordinal = copy.deepcopy(cases[1]), copy.deepcopy(calls[1]), 2
    if mutation == "ordinal":
        ordinal = 1
    elif mutation == "case":
        case["name"] = "different raw identity"
    elif mutation == "error":
        call["stderr"] = "division by zero\n"
    elif mutation == "owner":
        cap._capital_gift_context = replace(cap._capital_gift_context)
    elif mutation == "positive_stage":
        cap.stage(case, call["request"], {}, cap.binary, cap.artifact, ordinal)
        assert cap.records == []
        assert cap._capital_gift_refusals == []
        return
    assert not cap.stage_refusal(
        case,
        call["request"],
        call["returncode"],
        call["stdout"],
        call["stderr"],
        cap.binary,
        cap.artifact,
        ordinal,
    )
    assert cap.records == []
    assert cap._capital_gift_refusals == []


def test_cli_actual_request_builder_and_mixed_result_contract(mixed, monkeypatch):
    from axiom_encode import cli

    cap, cases, calls = mixed
    artifact = positive_fixture.ACTUAL["artifact"]
    derived = {r["id"]: r for r in artifact["program"]["derived"]}
    parameters = {r["id"]: r for r in artifact["program"]["parameters"]}
    by_request = {json.dumps(r["request"], sort_keys=True): r for r in calls}

    def run(command, **kwargs):
        assert command[1] == "run-compiled"
        request = json.loads(kwargs["input"])
        record = by_request[json.dumps(request, sort_keys=True)]
        return SimpleNamespace(
            **{k: record[k] for k in ("returncode", "stdout", "stderr")}
        )

    monkeypatch.setattr(cli.subprocess, "run", run)
    for index, case in enumerate(cases, 1):
        failures = cli._execute_rulespec_test_case(
            cap.original_root / cap.root_relative.replace(".yaml", ".test.yaml"),
            case,
            case_name=case["name"],
            compiled_path=cap.artifact,
            binary=cap.binary,
            axiom_rules_path=cap.original_root,
            env={},
            parameter_by_id=parameters,
            derived_ids=set(derived),
            derived_by_id=derived,
            declared_relation_names={
                r["name"] for r in artifact["program"]["relations"]
            },
            declared_relation_slots=cli._rulespec_declared_relation_slots(artifact),
            compiled_relations=artifact["program"]["relations"],
            policy_repo_path=cap.original_root / "ca",
            rulespec_roots=(),
            execution_capture=cap,
            case_index=index,
        )
        assert failures == [], (index, failures)
    assert (
        c.observation(cap.evidence(), positive_fixture.STATIC["policy_yaml"])
        is not None
    )


def test_pipeline_actual_request_builder_and_mixed_result_contract(mixed, monkeypatch):
    from axiom_encode.harness import validator_pipeline as v
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    cap, cases, calls = mixed
    by_request = {json.dumps(r["request"], sort_keys=True): r for r in calls}

    def run(command, **kwargs):
        assert command[1] == "run-compiled"
        record = by_request[json.dumps(json.loads(kwargs["input"]), sort_keys=True)]
        return SimpleNamespace(
            **{k: record[k] for k in ("returncode", "stdout", "stderr")}
        )

    pipe = ValidatorPipeline(
        policy_repo_path=cap.original_root / "ca",
        axiom_rules_path=cap.original_root,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipe, "_axiom_rules_binary", lambda: cap.binary)
    monkeypatch.setattr(pipe, "_rulespec_engine_env", lambda: {})
    monkeypatch.setattr(v.subprocess, "run", run)
    issues = pipe._run_rulespec_test_cases(
        rules_file=cap.original_root / cap.root_relative,
        compiled_path=cap.artifact,
        compiled_payload=positive_fixture.ACTUAL["artifact"],
        cases=cases,
        execution_capture=cap,
    )
    assert issues == []
    assert (
        c.observation(cap.evidence(), positive_fixture.STATIC["policy_yaml"])
        is not None
    )


def test_assignment_guard_uses_authenticated_target_and_preserves_default(mixed):
    from axiom_encode.harness.validator_pipeline import (
        find_test_input_assignment_issues,
    )

    cap, cases, _ = mixed
    policy = positive_fixture.STATIC["policy_yaml"]
    issues = find_test_input_assignment_issues(policy, cases, execution_capture=cap)
    assert issues == []
    unauthenticated = find_test_input_assignment_issues(policy, cases)
    assert len(unauthenticated) == 18
    assert all("authenticated refusal context" in issue for issue in unauthenticated)
    forged = copy.deepcopy(cases)
    forged[1]["expected_refusal"]["output"] = (
        forged[1]["expected_refusal"]["output"] + "_other"
    )
    assert any(
        "authenticated refusal context" in issue
        for issue in find_test_input_assignment_issues(
            policy, forged, execution_capture=cap
        )
    )


def test_named_target_dependencies_keep_all_branch_inputs():
    from axiom_encode.harness.validator_pipeline import (
        _required_inputs_for_test_outputs,
    )

    arguments = dict(
        symbol_inputs={
            "target": {"condition", "both_branch_fact"},
            "other": {"unrelated"},
        },
        symbol_dependencies={"target": set(), "other": set()},
        symbol_versions={"target": None, "other": None},
        period_bounds=None,
    )
    assert _required_inputs_for_test_outputs(None, **arguments) == {
        "condition",
        "both_branch_fact",
        "unrelated",
    }
    assert (
        _required_inputs_for_test_outputs(
            None, selected_output_names=("target",), **arguments
        )
        == _required_inputs_for_test_outputs({"target": 0}, **arguments)
        == {"condition", "both_branch_fact"}
    )


def test_cli_refusal_snapshot_uses_scoped_strict_copy(mixed, monkeypatch, tmp_path):
    from contextlib import contextmanager

    from axiom_encode import cli
    from axiom_encode.harness import same_case_execution as s

    cap, _, _ = mixed
    original = SimpleNamespace(
        local_corpus_release=object(), require_complete_source_unit=False
    )
    seen = []

    @contextmanager
    def snapshot(pipe, program):
        assert pipe is not original
        assert pipe.local_corpus_release is original.local_corpus_release
        assert pipe.require_complete_source_unit is True
        seen.append(program)
        yield None

    monkeypatch.setattr(s, "compilation_snapshot", snapshot)
    with pytest.raises(ValueError, match="snapshot/source binding unavailable"):
        cli._execute_rulespec_test_file(
            cap.original_root / cap.root_relative.replace(".yaml", ".test.yaml"),
            binary=cap.binary,
            pipeline=original,
            axiom_rules_path=cap.original_root,
            env={},
            rulespec_roots=(),
            tmp_path=tmp_path,
            compiled_cache={},
            policy_repo_path=cap.original_root / "ca",
        )
    assert seen == [cap.original_root / cap.root_relative]
    assert original.require_complete_source_unit is False


def test_pipeline_failure_clears_owned_positive_and_refusal_evidence(
    mixed, monkeypatch
):
    from axiom_encode.harness import validator_pipeline as v

    cap, cases, calls = mixed
    case, call = cases[1], calls[1]
    assert cap.stage_refusal(
        case,
        call["request"],
        call["returncode"],
        call["stdout"],
        call["stderr"],
        cap.binary,
        cap.artifact,
        2,
    )
    stage_all(cap, cases[:1], calls[:1])
    assert cap.records and cap._capital_gift_refusals
    response = json.loads(calls[0]["stdout"])
    output = next(iter(response["results"][0]["outputs"].values()))
    output["value"]["value"] = "999999"
    pipe = v.ValidatorPipeline(
        policy_repo_path=cap.original_root / "ca",
        axiom_rules_path=cap.original_root,
        local_corpus_release=None,
        enable_oracles=False,
    )
    monkeypatch.setattr(pipe, "_axiom_rules_binary", lambda: cap.binary)
    monkeypatch.setattr(pipe, "_rulespec_engine_env", lambda: {})
    monkeypatch.setattr(
        v.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout=json.dumps(response), stderr=""
        ),
    )
    issues = pipe._run_rulespec_test_cases(
        rules_file=cap.original_root / cap.root_relative,
        compiled_path=cap.artifact,
        compiled_payload=positive_fixture.ACTUAL["artifact"],
        cases=cases[:1],
        execution_capture=cap,
    )
    assert issues
    assert cap.records == [] and cap._capital_gift_refusals == []
    assert cap._capital_gift_context is None
    assert c.observation(cap.evidence(), positive_fixture.STATIC["policy_yaml"]) is None

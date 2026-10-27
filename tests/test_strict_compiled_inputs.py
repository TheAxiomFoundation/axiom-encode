"""Sparse companions rely on reached strict reads, never manufactured defaults."""

import copy
import json
import os
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.cli import (
    _complete_missing_local_test_inputs,
    _try_repair_generated_test_input_assignments_for_apply,
)
from axiom_encode.harness.strict_compiled_inputs import (
    compiled_program_has_strict_input_reads,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline,
    find_test_input_assignment_issues,
)

TARGET = "us:policies/example/lazy_inputs"


def artifact():
    expr = {"kind": "input", "name": "amount"}
    return {
        "artifact_format_version": 2,
        "engine_version": "0.2.2",
        "metadata": {},
        "program": {
            "parameters": [],
            "relations": [],
            "derived": [
                {
                    "name": "result",
                    "entity": "Person",
                    "dtype": "decimal",
                    "period": "Month",
                    "semantics": "scalar",
                    "expr": expr,
                    "versions": [
                        {
                            "effective_from": "2026-01-01",
                            "effective_to": None,
                            "semantics": "scalar",
                            "expr": copy.deepcopy(expr),
                        }
                    ],
                }
            ],
        },
    }


def test_recognizes_strict_program():
    assert compiled_program_has_strict_input_reads(artifact())


@pytest.mark.parametrize("location", ["root", "inactive", "imported"])
@pytest.mark.parametrize(
    "expr",
    [
        {
            "kind": "input_or_else",
            "name": "amount",
            "default": {"kind": "integer", "value": 0},
        },
        {"kind": "future_operator", "name": "amount"},
        {"kind": "input", "name": "amount", "default": 0},
    ],
)
def test_unknown_or_default_reads_anywhere_refuse(location, expr):
    data = artifact()
    if location == "root":
        data["program"]["derived"][0]["expr"] = expr
    elif location == "inactive":
        data["program"]["derived"][0]["versions"].append(
            {"effective_from": "2099-01-01", "semantics": "scalar", "expr": expr}
        )
    else:
        rule = copy.deepcopy(data["program"]["derived"][0])
        rule.update(name="other:module#result", expr=expr)
        data["program"]["derived"].append(rule)
    assert not compiled_program_has_strict_input_reads(data)


@pytest.mark.parametrize(
    "value", [None, {}, {"program": {}}, {"artifact_format_version": True}, []]
)
def test_absent_or_unknown_artifact_is_not_evidence(value):
    assert not compiled_program_has_strict_input_reads(value)


def test_unknown_relation_predicate_and_parameter_expression_refuse():
    data = artifact()
    data["program"]["relations"] = [
        {
            "name": "r",
            "arity": 2,
            "slot_entities": ["Person", "Person"],
            "predicate": {"kind": "input_or_else"},
        }
    ]
    assert not compiled_program_has_strict_input_reads(data)
    data = artifact()
    data["program"]["parameters"] = [
        {
            "name": "p",
            "versions": [
                {
                    "effective_from": "2026-01-01",
                    "values": {"0": {"kind": "input_or_else", "default": 0}},
                }
            ],
        }
    ]
    assert not compiled_program_has_strict_input_reads(data)


def declaration():
    return {
        "format": "rulespec/v1",
        "module": {"proof_validation": {"required": True}},
        "inputs": [
            {"name": "selected", "entity": "Person", "dtype": "Boolean"},
            {"name": "amount", "entity": "Person", "dtype": "Integer"},
            {"name": "unused", "entity": "Person", "dtype": "Integer"},
        ],
        "rules": [
            {
                "name": "result",
                "kind": "derived",
                "entity": "Person",
                "dtype": "Integer",
                "period": "Month",
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": "if selected: amount else: unused",
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "mode",
    ["valid", "false_valid", "missing_amount", "flipped_branch", "missing_selector"],
)
def test_normal_ci_executes_sparse_cases_and_apply_does_not_fill(
    tmp_path, real_engine, monkeypatch, mode
):
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/lazy_inputs.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(yaml.safe_dump(declaration()))
    facts = {f"{TARGET}#input.selected": True}
    if mode != "missing_amount":
        facts[f"{TARGET}#input.amount"] = 7
    if mode == "false_valid":
        facts = {f"{TARGET}#input.selected": False, f"{TARGET}#input.unused": 7}
    elif mode == "flipped_branch":
        facts[f"{TARGET}#input.selected"] = False
    elif mode == "missing_selector":
        del facts[f"{TARGET}#input.selected"]
    case = {
        "name": "sparse",
        "period": "2026-01",
        "input": facts,
        "output": {f"{TARGET}#result": 7},
    }
    test_file = module.with_suffix(".test.yaml")
    test_file.write_text(yaml.safe_dump([case]))
    before = test_file.read_bytes()
    # The no-evidence utility remains conservative about unselected branches.
    assert find_test_input_assignment_issues(module.read_text(), [case])
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    calls = []
    original = pipeline._run_rulespec_test_cases

    def run(**kwargs):
        calls.append(kwargs["cases"])
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "_run_rulespec_test_cases", run)
    result = pipeline._run_rulespec_ci(module)
    assert calls == [[case]]
    assert not any("Test input assignment missing:" in i for i in result.issues)
    runtime_errors = [i for i in result.issues if "missing input" in i.lower()]
    assert bool(runtime_errors) == (mode not in {"valid", "false_valid"})
    # Other source/proof gates may reject this deliberately minimal fixture.
    validation = SimpleNamespace(results={"ci": result})
    assert not _complete_missing_local_test_inputs(
        rules_file=module,
        test_file=test_file,
        repo_path=root / "us",
        relative_output=module.relative_to(root / "us"),
        validation=validation,
    )
    generated = SimpleNamespace(output_file=str(module))
    assert (
        _try_repair_generated_test_input_assignments_for_apply(
            generated,
            output_root=root / "us",
            policy_repo_path=root / "us",
            issues=result.issues,
        )
        == []
    )
    assert test_file.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    ["schema", "program", "derived", "version", "parameter_index", "malformed_kind"],
)
def test_unknown_executable_shapes_keep_conservative_check(mutation):
    data = artifact()
    if mutation == "schema":
        data["artifact_format_version"] = 3
    elif mutation == "program":
        data["program"]["future_executable"] = {"kind": "input_or_else"}
    elif mutation == "derived":
        data["program"]["derived"][0]["default"] = 0
    elif mutation == "version":
        data["program"]["derived"][0]["versions"][0]["fallback"] = 0
    elif mutation == "parameter_index":
        data["program"]["parameters"] = [
            {
                "name": "p",
                "indexed_by": {"kind": "input_or_else"},
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "values": {"0": {"kind": "integer", "value": 1}},
                    }
                ],
            }
        ]
    else:
        data["program"]["derived"][0]["expr"]["kind"] = []
    assert not compiled_program_has_strict_input_reads(data)


def test_judgment_and_lte_have_strict_children():
    data = artifact()
    rule = data["program"]["derived"][0]
    rule["semantics"] = "judgment"
    rule["expr"] = {
        "kind": "comparison",
        "op": "lte",
        "left": {"kind": "input", "name": "amount"},
        "right": {"kind": "literal", "value": {"kind": "integer", "value": 5}},
    }
    rule["versions"][0]["semantics"] = "judgment"
    rule["versions"][0]["expr"] = copy.deepcopy(rule["expr"])
    assert compiled_program_has_strict_input_reads(data)
    rule["versions"][0]["expr"]["left"] = {
        "kind": "input_or_else",
        "name": "amount",
        "default": {"kind": "integer", "value": 0},
    }
    assert not compiled_program_has_strict_input_reads(data)


def test_default_compile_rejection_and_ir_static_fallback(
    tmp_path, real_engine, monkeypatch
):
    root = tmp_path / "rulespec-us"
    source = root / "us/policies/example/default_input.yaml"
    source.parent.mkdir(parents=True)
    content = declaration()
    content["rules"][0]["default"] = 17
    source.write_text(yaml.safe_dump(content))
    compiled = tmp_path / "compiled.json"
    completed = subprocess.run(
        [
            str(real_engine.binary),
            "compile",
            "--program",
            str(source),
            "--output",
            str(compiled),
            "--rulespec-root",
            str(root),
        ],
        cwd=real_engine.root,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "express the fallback explicitly in the formula" in completed.stderr

    # RuleSpec deliberately rejects defaults. Separately exercise conservative
    # CI handling of a structural IR fixture, not a claimed compiler output.
    payload = artifact()
    expr = {
        "kind": "input_or_else",
        "name": "amount",
        "default": {"kind": "integer", "value": 0},
    }
    rule = payload["program"]["derived"][0]
    rule.update(expr=expr, dtype="integer")
    rule["versions"][0]["expr"] = copy.deepcopy(expr)
    compiled.write_text(json.dumps(payload))
    assert not compiled_program_has_strict_input_reads(payload)
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/lazy_inputs.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(yaml.safe_dump(declaration()))
    case = {
        "name": "missing",
        "period": "2026-01",
        "input": {},
        "output": {f"{TARGET}#result": 0},
    }
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump([case]))
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    # Inject only at the compile seam to test the conservative fallback.
    def compile_owned(_source, output):
        output.write_bytes(compiled.read_bytes())
        return subprocess.CompletedProcess([], 0, "", ""), payload

    monkeypatch.setattr(pipeline, "_compile_rulespec_to_artifact", compile_owned)
    calls = []
    monkeypatch.setattr(
        pipeline, "_run_rulespec_test_cases", lambda **kw: calls.append(kw) or []
    )
    result = pipeline._run_rulespec_ci(module)
    call_count = len(calls)
    assert call_count == 0
    assert any("Test input assignment missing:" in issue for issue in result.issues)


def test_missing_related_child_fact_remains_runtime_failure(
    tmp_path, real_engine, monkeypatch
):
    from tests.test_derived_relation_sum import declaration as relation_declaration

    target = "us:policies/example/relation_sum"
    root = tmp_path / "rulespec-us"
    module = root / "us/policies/example/relation_sum.yaml"
    module.parent.mkdir(parents=True)
    content = relation_declaration()
    content["module"] = {"proof_validation": {"required": True}}
    module.write_text(yaml.safe_dump(content))
    cases = [
        {
            "name": "missing_child",
            "period": "2026-01",
            "input": {
                f"{target}#relation.members": [{f"{target}#input.amount": 7}, {}]
            },
            "output": {f"{target}#total": 14},
        }
    ]
    test = module.with_suffix(".test.yaml")
    test.write_text(yaml.safe_dump(cases))
    before = test.read_bytes()
    pipeline = ValidatorPipeline(
        policy_repo_path=root / "us",
        axiom_rules_path=real_engine.root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    calls = []
    original = pipeline._run_rulespec_test_cases

    def run(**kw):
        calls.append(kw["cases"])
        return original(**kw)

    monkeypatch.setattr(pipeline, "_run_rulespec_test_cases", run)
    result = pipeline._run_rulespec_ci(module)
    assert calls == [cases]
    assert any("missing input" in issue.lower() for issue in result.issues)
    assert not any("Test input assignment missing:" in issue for issue in result.issues)
    assert not _complete_missing_local_test_inputs(
        rules_file=module,
        test_file=test,
        repo_path=root / "us",
        relative_output=module.relative_to(root / "us"),
        validation=SimpleNamespace(results={"ci": result}),
    )
    assert test.read_bytes() == before

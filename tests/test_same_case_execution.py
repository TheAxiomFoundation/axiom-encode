"""Execution evidence is exact-case, source-bound and never counterfactual."""

import copy
import json
from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

URI = "ca:policies/test/provider#reported"
ROOT = "ca:policies/test/consumer"


@pytest.fixture
def world(tmp_path):
    formula = "reported + local_amount"
    excerpt = "Add: reported + local_amount."
    rule = {
        "name": "out",
        "kind": "derived",
        "entity": "Person",
        "period": "Year",
        "dtype": "Money",
        "unit": "CAD",
        "versions": [
            {
                "effective_from": "2031-01-01",
                "effective_to": "2031-12-31",
                "formula": formula,
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": "ca/policy/test",
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
    }
    content = yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "module": {},
            "imports": [URI],
            "inputs": [
                {
                    "name": "local_amount",
                    "dtype": "Money",
                    "entity": "Person",
                    "period": "Year",
                    "unit": "CAD",
                }
            ],
            "rules": [rule],
        }
    )
    relative = "ca/policies/test/consumer.yaml"
    original = tmp_path / "original" / relative
    snapshot = tmp_path / "snapshot"
    original.parent.mkdir(parents=True)
    (snapshot / relative).parent.mkdir(parents=True)
    original.write_text(content)
    (snapshot / relative).write_text(content)
    case = {
        "name": "sum",
        "period": 2031,
        "input": {ROOT + "#input.local_amount": 7},
        "output": {ROOT + "#out": 10},
    }
    query = {
        "entity_id": "alice",
        "period": {
            "period_kind": "tax_year",
            "start": "2031-01-01",
            "end": "2031-12-31",
        },
        "outputs": [ROOT + "#out"],
    }

    def item(ref, value):
        return {
            "id": ref,
            "entity_id": "alice",
            "kind": "scalar",
            "dtype": "decimal",
            "unit": "CAD",
            "value": {"kind": "decimal", "value": str(value)},
        }

    response = {
        "metadata": {
            "requested_mode": "explain",
            "actual_mode": "explain",
            "fallback_reason": None,
        },
        "results": [
            {
                "entity_id": "alice",
                "period": query["period"],
                "outputs": {ROOT + "#out": item(ROOT + "#out", 10)},
                "trace": {URI: item(URI, 3), ROOT + "#out": item(ROOT + "#out", 10)},
            }
        ],
    }
    request = {"mode": "explain", "queries": [query], "dataset": {}}
    artifact = {
        "program": {
            "derived": [
                {
                    "id": ref,
                    "name": ref.split("#")[1],
                    "entity": "Person",
                    "period": "Year",
                    "dtype": "decimal",
                    "unit": "CAD",
                    "semantics": "scalar",
                }
                for ref in [URI, ROOT + "#out"]
            ]
        }
    }
    evidence = e.SameCaseEvidence(
        (e.ClosureFile(relative, original, content.encode()),),
        relative,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(artifact).encode(),
        (e.CaseExecution(e.case_key(case), json.dumps(request), json.dumps(response)),),
    )
    return SimpleNamespace(
        content=content,
        case=case,
        rule=rule,
        excerpt=excerpt,
        evidence=evidence,
        response=response,
        request=request,
        artifact=artifact,
    )


def changed(world, response=None, artifact=None):
    record = replace(
        world.evidence.records[0], response_json=json.dumps(response or world.response)
    )
    return replace(
        world.evidence,
        artifact_bytes=json.dumps(artifact or world.artifact).encode(),
        records=(record,),
    )


def test_exact_case_values_and_integer_money(world):
    assert world.evidence.values(world.case, world.content) == (
        {"reported": Decimal(3)},
        {"out": Decimal(10)},
    )
    world.response["results"][0]["trace"][URI]["value"] = {
        "kind": "integer",
        "value": 3,
    }
    assert changed(world).values(world.case, world.content)[0] == {
        "reported": Decimal(3)
    }


@pytest.mark.parametrize(
    "wire",
    [
        {"kind": "integer", "value": True},
        {"kind": "decimal", "value": 3.0},
        {"kind": "decimal", "value": "NaN"},
        {"kind": "decimal", "value": "Infinity"},
    ],
)
def test_invalid_numeric_trace_cannot_supply_import(world, wire):
    world.response["results"][0]["trace"][URI]["value"] = wire
    assert changed(world).values(world.case, world.content)[0] == {}


@pytest.mark.parametrize(
    "field,value",
    [("unit", "USD"), ("dtype", "bool"), ("kind", "judgment"), ("entity_id", "bob")],
)
def test_foreign_trace_fields_refuse(world, field, value):
    world.response["results"][0]["trace"][URI][field] = value
    got = changed(world).values(world.case, world.content)
    assert got is None or not got[0]


def test_judgment_cannot_be_currency(world):
    row = world.artifact["program"]["derived"][0]
    row.update(dtype="bool", unit=None, semantics="judgment")
    node = world.response["results"][0]["trace"][URI]
    node.update(kind="judgment", outcome="holds", unit=None)
    node.pop("dtype")
    assert changed(world).values(world.case, world.content)[0] == {"reported": True}
    node["unit"] = "CAD"
    assert changed(world).values(world.case, world.content)[0] == {}


@pytest.mark.parametrize("change", ["fact", "period", "output", "name"])
def test_no_counterfactual_or_case_reuse(world, change):
    case = copy.deepcopy(world.case)
    if change == "fact":
        case["input"][ROOT + "#input.local_amount"] += 1
    elif change == "period":
        case["period"] = 2032
    elif change == "output":
        case["output"][ROOT + "#out"] = 11
    else:
        case["name"] = "other"
    assert world.evidence.values(case, world.content) is None


def test_case_identity_preserves_types():
    assert len({e.case_key({"x": x}) for x in [True, 1, 1.0, "1"]}) == 4
    assert e.case_key({"x": Decimal("1")}) is None
    assert e.case_key({"x": float("nan")}) is None
    assert e.case_key({1: "x"}) is None


def test_snapshot_mutation_invalidates_binding(world):
    assert world.evidence.bound(world.content, ())
    world.evidence.files[0].original.write_text(world.content + "\n")
    assert not world.evidence.bound(world.content, ())


@pytest.mark.parametrize("error", [ValueError, OSError])
def test_unsupported_snapshot_does_not_catch_body_errors(monkeypatch, error):
    monkeypatch.setattr(e, "_snapshot_plan", lambda *a: None)
    with pytest.raises(error, match="caller"):
        with e.compilation_snapshot(None, None) as snapshot:
            assert snapshot is None
            raise error("caller")


def test_no_clone_without_complete_source_or_pin():
    assert (
        e._snapshot_plan(SimpleNamespace(require_complete_source_unit=False), None)
        is None
    )
    assert (
        e._snapshot_plan(
            SimpleNamespace(
                require_complete_source_unit=True, _declared_engine_pin=lambda: None
            ),
            None,
        )
        is None
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"format: rulespec/v1\nmodule: {}\nimports: x",
        b"format: rulespec/v1\nmodule: {}\nmodule: {}",
        b"format: rulespec/v1\nmodule: &x {}\nimports: *x",
    ],
)
def test_unsupported_source_shapes_refuse(raw):
    with pytest.raises((ValueError, yaml.YAMLError)):
        e._payload(raw)


def test_inline_import_list_is_not_omitted():
    assert e._payload(
        b'format: rulespec/v1\nmodule: {}\nimports: ["ca:policies/x#out"]'
    )["imports"] == ["ca:policies/x#out"]


def test_same_case_replay_uses_private_caches(world, monkeypatch):
    observed = []
    context = dict(
        represented_annual_spans=set(),
        measurement_roles=object(),
        represented_measurement_spans=set(),
        measurement_witnesses=set(),
    )

    def witness(branch, **kwargs):
        for key, value in context.items():
            assert kwargs[key] is value
        observed.append(kwargs)
        assert kwargs["dependency_cache"] == {id(world.case): {"out": Decimal(10)}}
        return {("out", "case:" + str(id(world.case)))}

    monkeypatch.setattr(s, "_formula_branch_test_witnesses", witness)
    kwargs = dict(
        content=world.content,
        principal_rules={"out": world.rule},
        rule_names={"out"},
        asserted_by_rule={"out": [world.case]},
        formula_environment={},
        **context,
    )
    assert e.formula_witnesses(world.evidence, None, **kwargs)
    assert kwargs["formula_environment"] == {}
    assert "dependency_cache" not in kwargs
    assert len(observed) == 1
    mutated = copy.deepcopy(world.case)
    mutated["input"][ROOT + "#input.local_amount"] = 8
    kwargs["asserted_by_rule"] = {"out": [mutated]}
    assert not e.formula_witnesses(world.evidence, None, **kwargs)


def test_real_source_operation_still_checks_operator(world):
    branch = s.SourceStructureBranch(
        (), "formula-clause", "sum", world.excerpt, 0, len(world.excerpt)
    )

    def run():
        return e.formula_witnesses(
            world.evidence,
            branch,
            content=world.content,
            principal_rules={"out": world.rule},
            rule_names={"out"},
            asserted_by_rule={"out": [world.case]},
            formula_environment={},
            corpus_citation_path="ca/policy/test",
            source_text=world.excerpt,
            extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )

    assert run()
    # Correct observed subtraction result is still not evidence of source addition.
    world.rule["versions"][0]["formula"] = "reported - local_amount"
    world.case["output"][ROOT + "#out"] = -4
    world.response["results"][0]["trace"][ROOT + "#out"]["value"]["value"] = "-4"
    world.evidence = replace(
        changed(world),
        records=(
            e.CaseExecution(
                e.case_key(world.case),
                json.dumps(world.request),
                json.dumps(world.response),
            ),
        ),
    )
    assert not run()


@pytest.mark.parametrize(
    "failure",
    [OSError("write failed"), __import__("subprocess").TimeoutExpired("clone", 1)],
)
def test_setup_failure_declines_but_does_not_catch_body(monkeypatch, failure):
    monkeypatch.setattr(e, "_snapshot_plan", lambda *a: (1,))

    def fail(*args):
        raise failure

    monkeypatch.setattr(e, "_materialize_snapshot", fail)
    with e.compilation_snapshot(None, None) as capture:
        assert capture is None
    with pytest.raises(ValueError, match="caller"):
        with e.compilation_snapshot(None, None):
            raise ValueError("caller")


@pytest.mark.parametrize(
    "shape",
    [
        {"rules": [None]},
        {"rules": None},
        {
            "rules": [
                {
                    "name": "out",
                    "kind": "derived",
                    "versions": [{"formula": "reported + local_amount"}],
                    "metadata": None,
                }
            ]
        },
        {
            "rules": [
                {
                    "name": "out",
                    "kind": "derived",
                    "versions": [{"formula": "reported + local_amount"}],
                    "metadata": {"proof": None},
                }
            ]
        },
        {
            "rules": [
                {
                    "name": "out",
                    "kind": "derived",
                    "versions": [{"formula": "reported + local_amount"}],
                    "metadata": {
                        "proof": {"atoms": [{"kind": "formula", "source": None}]}
                    },
                }
            ]
        },
    ],
)
def test_malformed_eligibility_does_not_replace_normal_diagnostics(shape):
    pipe = SimpleNamespace(
        _source_texts_for_rulespec_content=lambda _: None,
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {},
    )
    assert not e._has_consumer(pipe, "", shape, {"reported"})


def test_orchestration_preserves_original_repository_context(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from axiom_encode.harness import validator_pipeline as v

    original = tmp_path / "original/ca"
    snapshot = tmp_path / "snapshot"
    original.mkdir(parents=True)
    sibling = original / "sibling.yaml"
    sibling.write_text("must remain visible")
    root = original / "root.yaml"
    root.write_text("original")
    capture = SimpleNamespace(
        snapshot_root=snapshot,
        root_relative="ca/root.yaml",
        clone_seconds=0.1,
        files=(),
        records=[],
    )

    @contextmanager
    def materialized(*args):
        yield capture

    monkeypatch.setattr(v, "compilation_snapshot", materialized)
    pipe = object.__new__(v.ValidatorPipeline)
    pipe.policy_repo_path = original

    def ordinary(file, *, execution_capture=None):
        assert pipe.policy_repo_path == original
        assert file == root
        assert sibling.read_text() == "must remain visible"
        assert execution_capture is capture
        return SimpleNamespace(details={})

    pipe._run_rulespec_ci_impl = ordinary
    assert (
        pipe._run_rulespec_ci(root).details["same_case_snapshot"]["clone_seconds"]
        == 0.1
    )


def test_capture_requires_pinned_same_artifact_explain_and_matching_query(
    world, tmp_path
):
    binary = tmp_path / "binary"
    binary.write_bytes(b"pinned executable")
    artifact = tmp_path / "compiled.json"
    artifact.write_bytes(world.evidence.artifact_bytes)
    capture = e.ExecutionCapture(
        tmp_path,
        world.evidence.snapshot_root,
        world.evidence.root_relative,
        world.evidence.files,
        binary,
        e.digest(binary.read_bytes()),
        "c" * 40,
        0.0,
        "a" * 40,
    )
    capture.compiled(artifact)
    capture.stage(world.case, world.request, world.response, binary, artifact)
    assert capture.evidence() is not None
    capture.records.clear()
    bad = copy.deepcopy(world.response)
    bad["metadata"]["actual_mode"] = "fast"
    capture.stage(world.case, world.request, bad, binary, artifact)
    assert capture.evidence() is None
    bad = copy.deepcopy(world.response)
    bad["results"][0]["period"]["start"] = "2032-01-01"
    capture.stage(world.case, world.request, bad, binary, artifact)
    assert capture.evidence() is None
    binary.write_bytes(b"changed")
    capture.stage(world.case, world.request, world.response, binary, artifact)
    assert capture.evidence() is None


def test_any_companion_failure_clears_invocation_evidence(world, tmp_path):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    pipe = object.__new__(ValidatorPipeline)
    pipe.policyengine_rule_hint = None
    pipe.policy_repo_path = tmp_path
    capture = SimpleNamespace(records=[world.evidence.records[0]])
    issues = pipe._run_rulespec_test_cases(
        rules_file=tmp_path / "unused.yaml",
        compiled_path=tmp_path / "unused.json",
        compiled_payload=world.artifact,
        cases=[
            {
                "name": "bad",
                "period": 2031,
                "input": {},
                "output": {"unknown_output": 0},
            }
        ],
        execution_capture=capture,
    )
    assert issues
    assert capture.records == []


def test_unreached_import_node_is_not_target_reachability(world, monkeypatch):
    world.rule["versions"][0]["formula"] = "local_amount + 3"
    calls = []
    monkeypatch.setattr(
        s, "_formula_branch_test_witnesses", lambda *a, **k: calls.append(k)
    )
    assert not e.formula_witnesses(
        world.evidence,
        None,
        content=world.content,
        principal_rules={"out": world.rule},
        rule_names={"out"},
        asserted_by_rule={"out": [world.case]},
        formula_environment={},
    )
    assert not calls


def test_missing_trace_and_duplicate_case_cannot_supply_evidence(world):
    world.response["results"][0]["trace"].pop(URI)
    assert changed(world).values(world.case, world.content)[0] == {}
    duplicate = replace(world.evidence, records=world.evidence.records * 2)
    assert duplicate.values(world.case, world.content) is None


def test_unhashable_citation_declines_before_mapping_lookup():
    payload = {
        "rules": [
            {
                "name": "out",
                "kind": "derived",
                "versions": [{"formula": "reported + value"}],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "kind": "formula",
                                "source": {"corpus_citation_path": [], "excerpt": "x"},
                            }
                        ]
                    }
                },
            }
        ]
    }
    pipe = SimpleNamespace(
        _source_texts_for_rulespec_content=lambda _: None,
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {"x": "x"},
    )
    assert not e._has_consumer(pipe, "", payload, {"reported"})


def test_new_invocation_and_new_compile_have_no_prior_evidence(world, tmp_path):
    binary = tmp_path / "binary"
    binary.write_bytes(b"pinned")
    artifact = tmp_path / "compiled"
    artifact.write_bytes(world.evidence.artifact_bytes)

    def fresh():
        return e.ExecutionCapture(
            tmp_path,
            world.evidence.snapshot_root,
            world.evidence.root_relative,
            world.evidence.files,
            binary,
            e.digest(binary.read_bytes()),
            "a" * 40,
            0.0,
            "b" * 40,
        )

    first = fresh()
    first.compiled(artifact)
    first.stage(world.case, world.request, world.response, binary, artifact)
    assert first.evidence() is not None
    second = fresh()
    second.compiled(artifact)
    assert second.evidence() is None
    first.compiled(artifact)
    assert first.evidence() is None


def test_actual_output_mismatch_discards_all_tentative_records(world, tmp_path):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    pipe = object.__new__(ValidatorPipeline)
    pipe.policy_repo_path = tmp_path
    pipe.require_complete_source_unit = False
    pipe.policyengine_rule_hint = None
    pipe._axiom_rules_binary = lambda: tmp_path / "unused-binary"
    wrong = {
        "kind": "scalar",
        "id": ROOT + "#out",
        "dtype": "decimal",
        "unit": "CAD",
        "value": {"kind": "decimal", "value": "11"},
    }
    calls = []

    def execute(**kwargs):
        calls.append(kwargs)
        return {ROOT + "#out": wrong}, []

    pipe._run_rulespec_derived_test_case = execute
    capture = SimpleNamespace(records=[world.evidence.records[0]])
    issues = pipe._run_rulespec_test_cases(
        rules_file=tmp_path / "missing.yaml",
        compiled_path=tmp_path / "missing.json",
        compiled_payload=world.artifact,
        cases=[{**world.case, "period": world.request["queries"][0]["period"]}],
        execution_capture=capture,
    )
    assert len(calls) == 1
    assert any("expected" in issue for issue in issues), issues
    assert capture.records == []


@pytest.mark.parametrize("wrong_row", [False, True], ids=["listed", "wrong-row-listed"])
def test_same_case_reader_refuses_singleton_scalar_credit(world, wrong_row):
    # Rebind the exact mutated case: stale receipt rejection cannot explain this.
    case = copy.deepcopy(world.case)
    case["output"][ROOT + "#out"] = [10]
    if wrong_row:
        case["tables"] = {"Person": [{"local_amount": 8}]}
    record = replace(world.evidence.records[0], case_sha256=e.case_key(case))
    evidence = replace(world.evidence, records=(record,))
    assert evidence.values(case, world.content) is not None
    asserted = s._test_case_asserted_output_value(case, "out")
    assert asserted == [10]
    assert not s._asserted_formula_runtime_values_equal(
        world.rule, Decimal(10), asserted
    )
    branch = s.SourceStructureBranch(
        (), "formula-clause", "sum", world.excerpt, 0, len(world.excerpt)
    )
    assert not e.formula_witnesses(
        evidence,
        branch,
        content=world.content,
        principal_rules={"out": world.rule},
        rule_names={"out"},
        asserted_by_rule={"out": [case]},
        formula_environment={},
        corpus_citation_path="ca/policy/test",
        source_text=world.excerpt,
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )

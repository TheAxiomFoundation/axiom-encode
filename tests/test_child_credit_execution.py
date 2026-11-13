"""Actual historical NS calls; synthetic closure identities are not admission."""

import copy
import json
from pathlib import Path

import pytest

from axiom_encode.bound_child_credit_form import bind_child_credit_form
from axiom_encode.harness.child_credit_execution import observe_child_credit
from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    SameCaseEvidence,
    case_key,
)

ROOT = "ca/policies/cra/t1-2025/ns479-credits.yaml"
CITATION = "ca/policy/cra/t1-2025/ns479-credits"
FIXTURE = Path(__file__).parent / "fixtures/child_credit/observed-graph.json"


def data():
    return json.loads(FIXTURE.read_text())


def evidence(tmp_path, d):
    original = tmp_path / "original" / ROOT
    original.parent.mkdir(parents=True)
    original.write_text(d["content"])
    snapshot = tmp_path / "snapshot"
    target = snapshot / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(d["content"])
    return SameCaseEvidence(
        (ClosureFile(ROOT, original, d["content"].encode()),),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        d["engine_pin"],
        json.dumps(d["artifact"]).encode(),
        tuple(
            CaseExecution(
                r["case_sha256"],
                json.dumps(r["request"]),
                json.dumps(r["response"]),
                r["case_index"],
            )
            for r in d["records"]
        ),
    )


def observe(tmp_path, d):
    binding = bind_child_credit_form(d["content"], d["source"], CITATION)
    assert binding is not None
    return observe_child_credit(
        binding, evidence(tmp_path, d), d["content"], d["cases"]
    )


def test_actual_17_cases65_outputs(tmp_path):
    d = data()
    assert len(d["cases"]) == 17
    assert sum(len(c["output"]) for c in d["cases"]) == 65
    result = observe(tmp_path, d)
    assert result is not None
    assert len(result.duty_pairs) == 7
    assert len(result.refund_cases) == 3
    assert any(
        "@" not in name and entity != "ns479-claimant"
        for name, entity in result.provider_instances
    )


@pytest.mark.parametrize(
    "defect",
    [
        "pin",
        "extra_unit",
        "builtin",
        "duplicate_unit",
        "cad_float",
        "slot",
        "provider",
        "compiled_bool",
        "extra_derived",
        "parameter",
        "case_identity",
        "case_expected",
        "actual_kind",
        "actual_bool",
        "foreign_id",
        "trace_key",
        "missing_provider",
        "extra_trace",
        "cross_row_dependency",
        "response",
        "expression",
        "frame",
        "metadata",
        "relation_reverse",
        "relation_duplicate",
        "raw_fact",
        "raw_row",
        "missing_record",
        "compiled_version",
        "parameter_read",
    ],
)
def test_tampering_refuses(tmp_path, defect):
    d = data()
    program = d["artifact"]["program"]
    record = d["records"][0]
    request = record["request"]
    result = record["response"]["results"][0]
    trace = result["trace"]
    foreign = next(k for k in trace if "@entity:" in k)
    if defect == "pin":
        d["engine_pin"] = "0" * 40
    elif defect == "extra_unit":
        program["units"].append({"name": "OTHER", "kind": "count"})
    elif defect == "builtin":
        program["units"][0]["minor_units"] = 0
    elif defect == "duplicate_unit":
        program["units"].append(copy.deepcopy(program["units"][-1]))
    elif defect == "cad_float":
        program["units"][-1]["minor_units"] = 2.0
    elif defect == "slot":
        program["relations"][0]["slot_entities"].reverse()
    elif defect == "provider":
        next(r for r in program["derived"] if r["expr"]["kind"] == "sum_related")[
            "expr"
        ]["current_slot"] = 0
    elif defect == "compiled_bool":
        next(
            r
            for r in program["derived"]
            if r["name"] == "children_expense_table_completion_required"
        )["expr"]["right"]["value"]["value"] = 1
    elif defect == "extra_derived":
        program["derived"].append(copy.deepcopy(program["derived"][0]))
    elif defect == "parameter":
        program["parameters"][0]["versions"][0]["values"]["0"]["value"] = 501
    elif defect == "case_identity":
        d["cases"][0]["name"] += "changed"
    elif defect == "case_expected":
        d["cases"][0]["output"][next(iter(d["cases"][0]["output"]))] = 999
        record["case_sha256"] = case_key(d["cases"][0])
    elif defect == "actual_kind":
        next(f for f in request["dataset"]["inputs"] if f["entity"] == "SourceEntry")[
            "entity"
        ] = "Person"
    elif defect == "actual_bool":
        request["dataset"]["inputs"][0]["value"]["value"] = 0
    elif defect == "foreign_id":
        trace[foreign]["entity_id"] = "unrelated"
    elif defect == "trace_key":
        trace[foreign + "00"] = trace.pop(foreign)
    elif defect == "missing_provider":
        trace.pop(foreign)
    elif defect == "extra_trace":
        trace[foreign + "00"] = copy.deepcopy(trace[foreign])
    elif defect == "cross_row_dependency":
        node = next(
            v
            for k, v in trace.items()
            if "@entity:" in k and v["name"] == "per_child_column_f"
        )
        node["dependencies"][0] = node["dependencies"][0][:-2] + "32"
    elif defect == "response":
        next(
            v
            for v in result["outputs"].values()
            if v.get("value", {}).get("kind") == "decimal"
        )["value"]["value"] = "999"
    elif defect == "expression":
        trace[foreign]["executed_expression"] = "0"
    elif defect == "frame":
        request["queries"][0]["period"]["end"] = "2025-06-30"
    elif defect == "metadata":
        record["response"]["metadata"]["other"] = True
    elif defect == "relation_reverse":
        request["dataset"]["relations"][0]["tuple"].reverse()
    elif defect == "relation_duplicate":
        request["dataset"]["relations"].append(
            copy.deepcopy(request["dataset"]["relations"][0])
        )
    elif defect == "raw_fact":
        request["dataset"]["inputs"][-1]["value"]["value"] = 1
    elif defect == "raw_row":
        d["cases"][0]["input"][
            next(k for k in d["cases"][0]["input"] if "#relation." in k)
        ][0][
            next(
                k
                for k in d["cases"][0]["input"][
                    next(k for k in d["cases"][0]["input"] if "#relation." in k)
                ][0]
                if "eligible_expenses" in k
            )
        ] = 901
        record["case_sha256"] = case_key(d["cases"][0])
    elif defect == "missing_record":
        d["records"].pop()
    elif defect == "compiled_version":
        program["derived"][0]["versions"][0]["effective_to"] = "2026-12-31"
    elif defect == "parameter_read":
        next(v for v in trace.values() if v.get("parameter_reads"))["parameter_reads"][
            0
        ]["value"]["value"] = 501
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize("indices", [(12, 13), (10, 11), (14, 15, 16), (7, 8, 9)])
def test_coherent_omitted_roles_refuse(tmp_path, indices):
    d = data()
    d["cases"] = [c for i, c in enumerate(d["cases"]) if i not in indices]
    d["records"] = [r for i, r in enumerate(d["records"]) if i not in indices]
    for i, r in enumerate(d["records"], 1):
        r["case_index"] = i
    assert observe(tmp_path, d) is None


def partial_identity(d, *, assistance=100, others=200, only_b=False, only_bc=False):
    """Synthetic adversarial capture derived from actual case7, not new runtime."""
    case = copy.deepcopy(d["cases"][6])
    record = copy.deepcopy(d["records"][6])
    prefix = "ca:policies/cra/t1-2025/ns479-credits#"
    case["name"] = "synthetic-partial-domain"
    names = (
        ["eligible_expenses_column_b"]
        if only_b
        else [
            "eligible_expenses_column_b",
            "assistance_column_c",
            "claimed_by_others_column_e",
        ]
    )
    if only_bc:
        names = ["eligible_expenses_column_b", "assistance_column_c"]
    values = {
        "eligible_expenses_column_b": 800,
        "assistance_column_c": assistance,
        "claimed_by_others_column_e": others,
    }
    case["output"] = {prefix + n: values[n] for n in names}
    case["input"][prefix + "input.assistance_received_or_receivable_c"] = assistance
    case["input"][prefix + "input.amount_from_d_claimed_by_others_e"] = others
    record["request"]["queries"][0]["outputs"] = list(case["output"])
    for f in record["request"]["dataset"]["inputs"]:
        if f["name"] == prefix + "input.assistance_received_or_receivable_c":
            f["value"] = {"kind": "integer", "value": assistance}
        if f["name"] == prefix + "input.amount_from_d_claimed_by_others_e":
            f["value"] = {"kind": "integer", "value": others}
    result = record["response"]["results"][0]
    for bucket in ("outputs", "trace"):
        result[bucket] = {
            k: v for k, v in result[bucket].items() if k in case["output"]
        }
        for k, v in result[bucket].items():
            v["value"] = {"kind": "integer", "value": values[k.removeprefix(prefix)]}
    record["case_index"] = len(d["cases"]) + 1
    record["case_sha256"] = case_key(case)
    d["cases"].append(case)
    d["records"].append(record)


@pytest.mark.parametrize(
    "assistance,others", [(900, 200), (100, 600), (-1, 200), (100, -1)]
)
def test_partial_identity_domain_is_checked_without_negative_outputs(
    tmp_path, assistance, others
):
    d = data()
    partial_identity(d, assistance=assistance, others=others, only_bc=assistance > 800)
    assert observe(tmp_path, d) is None


def test_valid_partial_identity_and_unreached_domain_do_not_require_other_facts(
    tmp_path,
):
    d = data()
    partial_identity(d)
    assert observe(tmp_path / "valid", d) is not None
    d = data()
    partial_identity(d, assistance=900, others=600, only_b=True)
    assert observe(tmp_path / "unreached", d) is not None


@pytest.mark.parametrize("which", [8, 9, 11, 13, 15])
def test_missing_pair_member_with_remaining_output_coverage_refuses(tmp_path, which):
    d = data()
    d["cases"].pop(which)
    d["records"].pop(which)
    for i, r in enumerate(d["records"], 1):
        r["case_index"] = i
    assert observe(tmp_path, d) is None


def test_not_evaluated_branch_metadata_is_not_unchecked_authority(tmp_path):
    d = data()
    trace = d["records"][0]["response"]["results"][0]["trace"]
    next(v for v in trace.values() if v.get("not_evaluated_dependencies"))[
        "not_evaluated_dependencies"
    ][0]["dependency"] = "foreign"
    assert observe(tmp_path, d) is None


def test_remove_cap_distribution_contrast_coherently_refuses(tmp_path):
    d = data()
    d["cases"].pop(1)
    d["records"].pop(1)
    for i, record in enumerate(d["records"], 1):
        record["case_index"] = i
    assert observe(tmp_path, d) is None

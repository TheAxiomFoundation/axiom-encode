"""Actual engine records, synthetic local evidence wrappers; no native authority."""

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.bound_retirement_transfer_form import bind_retirement_transfer_form
from axiom_encode.harness import retirement_transfer_execution as subject
from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    ExecutionCapture,
    SameCaseEvidence,
    case_key,
    digest,
)

FIXTURE = Path(__file__).parent / "fixtures/retirement_transfer/observed-graph.json"
ROOT = "ca/policies/revenu-quebec/tp1-2025/schedule-q-retirement-income-transfer.yaml"
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-q-retirement-income-transfer"
PREFIX = "ca:policies/revenu-quebec/tp1-2025/schedule-q-retirement-income-transfer#"


def data():
    return json.loads(FIXTURE.read_text())


def evidence(tmp_path, d):
    original = tmp_path / "original"
    snapshot = tmp_path / "snapshot"
    files = []
    rawcases = (
        d["companion_raw"] if d["cases"] == data()["cases"] else json.dumps(d["cases"])
    )
    for relative, raw in [
        (ROOT, d["content"].encode()),
        (ROOT.replace(".yaml", ".test.yaml"), rawcases.encode()),
    ]:
        for root in (original, snapshot):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        files.append(ClosureFile(relative, original / relative, raw))
    return SameCaseEvidence(
        tuple(files),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        d["engine_pin"],
        json.dumps(d["artifact"]).encode(),
        tuple(
            CaseExecution(
                case_key(c),
                json.dumps(r["request"]),
                json.dumps(r["response"]),
                r["case_index"],
            )
            for c, r in zip(d["cases"], d["records"], strict=True)
        ),
    )


def observe(tmp_path, d):
    binding = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    assert binding is not None
    return subject.observe_retirement_transfer(
        binding, evidence(tmp_path, d), d["content"], d["cases"]
    )


def test_actual26_complete_graph_and99_outputs(tmp_path):
    d = data()
    assert len(d["cases"]) == 26
    assert sum(len(c["output"]) for c in d["cases"]) == 99
    assert observe(tmp_path, d) is not None


def test_compiled_union():
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    subject._compiled(b, d["content"], d["artifact"], PREFIX)


def test_each_actual_record():
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    inputs, rules, expressions, parameters = subject._compiled(
        b, d["content"], d["artifact"], PREFIX
    )
    for i, (case, r) in enumerate(zip(d["cases"], d["records"], strict=True), 1):
        record = CaseExecution(
            case_key(case), json.dumps(r["request"]), json.dumps(r["response"]), i
        )
        subject._record(
            case,
            record,
            i,
            inputs,
            rules,
            expressions,
            parameters,
            PREFIX,
            dict(b.rule_roles),
            case["period"],
        )


def omit(d, indices):
    """Coherent synthetic subset; request/response values stay actual."""
    d["cases"] = [c for i, c in enumerate(d["cases"], 1) if i not in indices]
    d["records"] = [r for i, r in enumerate(d["records"], 1) if i not in indices]
    for i, r in enumerate(d["records"], 1):
        r["case_index"] = i


@pytest.mark.parametrize("missing", [3, 5, 7, 9, 4, 6, 8, 10, 15, 20, 21, 22])
def test_coherent_missing_role_refused(tmp_path, missing):
    d = data()
    omit(d, {missing})
    assert observe(tmp_path, d) is None


def test_all_provider_permission_cases_cannot_be_replaced_by_local_pairs(tmp_path):
    d = data()
    omit(d, {4, 6, 8, 10})
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "defect",
    [
        "extra_unit",
        "builtin",
        "cad_float",
        "duplicate_unit",
        "unknown_pin",
        "format_bool",
        "catalog_missing_year",
        "catalog_year_alias",
        "catalog_extra",
        "parameter_date",
        "parameter_year_bool",
        "derived_extra",
        "derived_missing",
        "version_expr",
        "expr",
        "effective_date",
        "relation_slot",
        "relation_kind",
        "provider_requested",
    ],
)
def test_compiled_identity_refusals(tmp_path, defect):
    d = data()
    a = d["artifact"]
    p = a["program"]
    if defect == "extra_unit":
        p["units"].append({"name": "unused", "kind": "count"})
    elif defect == "builtin":
        p["units"][0]["minor_units"] = 0
    elif defect == "cad_float":
        p["units"][-1]["minor_units"] = 2.0
    elif defect == "duplicate_unit":
        p["units"].append(copy.deepcopy(p["units"][-1]))
    elif defect == "unknown_pin":
        d["engine_pin"] = "c" * 40
    elif defect == "format_bool":
        a["artifact_format_version"] = True
    elif defect == "catalog_missing_year":
        a["metadata"]["input_catalog"] = [
            x for x in a["metadata"]["input_catalog"] if x["slot"] != "source_year"
        ]
    elif defect == "catalog_year_alias":
        next(x for x in a["metadata"]["input_catalog"] if x["slot"] == "source_year")[
            "request_names"
        ].append("foreign#input.source_year")
    elif defect == "catalog_extra":
        a["metadata"]["input_catalog"].append(
            {
                "slot": "unused",
                "canonical_request_name": PREFIX + "input.unused",
                "request_names": [PREFIX + "input.unused"],
            }
        )
    elif defect == "parameter_date":
        p["parameters"][2]["versions"][0]["values"]["2025"]["value"] = "1962-01-01"
    elif defect == "parameter_year_bool":
        p["parameters"][1]["versions"][0]["values"]["0"]["value"] = True
    elif defect == "derived_extra":
        p["derived"].append(copy.deepcopy(p["derived"][0]))
    elif defect == "derived_missing":
        p["derived"].pop()
    elif defect == "version_expr":
        p["derived"][0]["versions"][0]["expr"] = {
            "kind": "literal",
            "value": {"kind": "decimal", "value": "4000"},
        }
    elif defect == "expr":
        p["derived"][0]["expr"] = {
            "kind": "literal",
            "value": {"kind": "decimal", "value": "4000"},
        }
    elif defect == "effective_date":
        p["derived"][0]["versions"][0]["effective_to"] = "2026-12-31"
    elif defect == "relation_slot":
        for r in p["derived"]:
            if r["expr"].get("kind") == "sum_related":
                r["expr"]["current_slot"] = 0
                r["expr"]["related_slot"] = 1
    elif defect == "relation_kind":
        p["relations"][0]["slot_entities"][0] = "Family"
    elif defect == "provider_requested":
        for r in p["derived"]:
            if r["expr"].get("kind") == "sum_related":
                r["expr"]["value"] = {
                    "kind": "input",
                    "name": "schedule_q_requested_retirement_income_transfer_line_22",
                }
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_kind",
        "bool_int",
        "bool_zero",
        "date_text",
        "date_float",
        "source_year_override",
        "foreign_input",
        "missing_reached",
        "duplicate_fact",
        "period",
        "assessment",
        "output_missing",
        "response_value",
        "response_judgment",
        "raw_case_date",
        "case_index",
        "parameter_read",
        "skipped_reason",
        "missing_trace",
        "extra_trace",
    ],
)
def test_actual_record_refusals(tmp_path, defect):
    d = data()
    r = d["records"][0]
    q = r["request"]
    out = r["response"]["results"][0]
    if defect == "wrong_kind":
        q["dataset"]["inputs"][0]["entity"] = "Family"
    elif defect in ("bool_int", "bool_zero"):
        f = next(
            f
            for f in q["dataset"]["inputs"]
            if f["value"]["kind"] == "bool"
            and f["value"]["value"] == (defect == "bool_int")
        )
        f["value"]["value"] = 1 if defect == "bool_int" else 0
    elif defect == "date_text":
        next(f for f in q["dataset"]["inputs"] if f["value"]["kind"] == "date")[
            "value"
        ]["kind"] = "text"
    elif defect == "date_float":
        next(f for f in q["dataset"]["inputs"] if f["value"]["kind"] == "date")[
            "value"
        ]["value"] = 1961.0
    elif defect == "source_year_override":
        f = copy.deepcopy(q["dataset"]["inputs"][0])
        f["name"] = PREFIX + "input.source_year"
        f["value"] = {"kind": "integer", "value": 2024}
        q["dataset"]["inputs"].append(f)
    elif defect == "foreign_input":
        q["dataset"]["inputs"][0]["name"] = (
            "foreign#input.schedule_q_retirement_income_line_122_line_10"
        )
    elif defect == "missing_reached":
        q["dataset"]["inputs"].pop(0)
    elif defect == "duplicate_fact":
        q["dataset"]["inputs"].append(copy.deepcopy(q["dataset"]["inputs"][0]))
    elif defect == "period":
        q["queries"][0]["period"]["start"] = "2025-02-01"
        out["period"]["start"] = "2025-02-01"
    elif defect == "assessment":
        q["queries"][0]["assessment_date"] = "2026-01-01"
        out["assessment_date"] = "2026-01-01"
    elif defect == "output_missing":
        q["queries"][0]["outputs"].pop()
    elif defect == "response_value":
        out["outputs"][PREFIX + "schedule_q_retirement_income_transferred_line_245"][
            "value"
        ]["value"] = "18001"
    elif defect == "response_judgment":
        out["outputs"][PREFIX + "retirement_income_transfer_permission_applies"][
            "outcome"
        ] = "not_holds"
    elif defect == "raw_case_date":
        d["cases"][0]["input"][PREFIX + "input.transferor_birth_date"] = "1961-01-01"
    elif defect == "case_index":
        r["case_index"] = 2
    elif defect == "parameter_read":
        out["trace"][PREFIX + "retirement_income_transfer_permission_applies"][
            "parameter_reads"
        ][0]["value"]["value"] = 2024
    elif defect == "skipped_reason":
        d["records"][2]["response"]["results"][0]["trace"][
            PREFIX + "retirement_income_transfer_permission_applies"
        ]["not_evaluated_dependencies"][0]["reason"] = "branch_not_selected"
    elif defect == "missing_trace":
        out["trace"].pop(next(iter(out["trace"])))
    elif defect == "extra_trace":
        out["trace"]["foreign"] = copy.deepcopy(next(iter(out["trace"].values())))
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "defect",
    [
        "reverse_tuple",
        "self_edge",
        "extra_edge",
        "duplicate_edge",
        "missing_edge",
        "provider_kind",
        "provider_id",
        "node_id",
        "trace_hex",
        "dependency",
        "skipped_provider",
        "skipped_parameter",
    ],
)
def test_actual_provider_refusals(tmp_path, defect):
    d = data()
    r = d["records"][1]
    q = r["request"]
    out = r["response"]["results"][0]
    edge = q["dataset"]["relations"][0]
    key = next(k for k in out["trace"] if "@entity:" in k)
    node = out["trace"][key]
    if defect == "reverse_tuple":
        edge["tuple"].reverse()
    elif defect == "self_edge":
        edge["tuple"][0] = edge["tuple"][1]
    elif defect == "extra_edge":
        q["dataset"]["relations"].append(
            {
                "name": edge["name"],
                "tuple": ["unrelated", edge["tuple"][1]],
                "interval": edge["interval"],
            }
        )
    elif defect == "duplicate_edge":
        q["dataset"]["relations"].append(copy.deepcopy(edge))
    elif defect == "missing_edge":
        q["dataset"]["relations"] = []
    elif defect == "provider_kind":
        q["dataset"]["inputs"][0]["entity"] = "Family"
    elif defect == "provider_id":
        node["entity_id"] = "other"
    elif defect == "node_id":
        node["id"] = key
    elif defect == "trace_hex":
        out["trace"][key.upper()] = out["trace"].pop(key)
    elif defect == "dependency":
        out["trace"][PREFIX + "retirement_income_received_line123"]["dependencies"] = [
            PREFIX + "schedule_q_retirement_income_transferred_line_245"
        ]
    elif defect in ("skipped_provider", "skipped_parameter"):
        out = d["records"][3]["response"]["results"][0]
        n = next(
            n
            for k, n in out["trace"].items()
            if k.startswith(
                PREFIX
                + (
                    "schedule_q_retirement_income_transferred_line_245"
                    if defect == "skipped_provider"
                    else "retirement_income_transfer_permission_applies"
                )
                + "@entity:"
            )
        )
        n["not_evaluated_dependencies"][0]["dependency"] = "foreign"
    assert observe(tmp_path, d) is None


def test_missing_unselected_rl1_is_genuine_positive(tmp_path):
    d = data()
    assert (
        PREFIX + "input.quebec_rl1_box_e_from_included_compensation"
        not in d["cases"][0]["input"]
    )
    assert observe(tmp_path, d) is not None


def test_positive_transfer_zero_denominator_corrupt_graph_refused(tmp_path):
    d = data()
    r = next(
        r
        for r in d["artifact"]["program"]["derived"]
        if r["name"] == "schedule_q_retirement_income_transferred_line_245"
    )
    for parent in [r, r["versions"][0]]:
        parent["expr"] = {
            "kind": "literal",
            "value": {"kind": "decimal", "value": "100"},
        }
    assert observe(tmp_path, d) is None


def test_coherent_reordering_preserves_complete_witnesses(tmp_path):
    d = data()
    d["cases"].reverse()
    d["records"].reverse()
    for i, r in enumerate(d["records"], 1):
        r["case_index"] = i
    assert observe(tmp_path, d) is not None


@pytest.mark.parametrize("missing", [18, 19, 26])
def test_coherent_zero_transfer_role_omission(tmp_path, missing):
    d = data()
    omit(d, {missing})
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize("partial", [False, True])
def test_actual_negative_net_domain_refused_even_partial(partial):
    d = data()
    negative = d["actual_domain_controls"]["negative_net_domain_unresolved"]
    case = copy.deepcopy(d["cases"][18])
    case["input"][PREFIX + "input.schedule_q_retirement_income_line_122_line_10"] = 900
    case["output"] = {
        PREFIX + "schedule_q_eligible_retirement_income_line_20": -100,
        PREFIX + "schedule_q_retirement_income_transferred_line_245": -50,
        PREFIX + "schedule_q_quebec_withholding_transferred_line_451_1": 0,
    }
    request = copy.deepcopy(negative["request"])
    response = json.loads(negative["stdout"])
    if partial:
        # Synthetic projection of the unchanged actual negative-net record;
        # both independently reached arithmetic nodes and values preserved.
        case["output"] = {
            PREFIX + "schedule_q_eligible_retirement_income_line_20": -100
        }
        request["queries"][0]["outputs"] = list(case["output"])
        r = response["results"][0]
        r["outputs"] = {k: v for k, v in r["outputs"].items() if k in case["output"]}
        r["trace"] = {
            k: v
            for k, v in r["trace"].items()
            if k
            in (
                PREFIX + "schedule_q_eligible_retirement_income_line_20",
                PREFIX + "schedule_q_retirement_income_deductions_line_18",
            )
        }
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    args = subject._compiled(b, d["content"], d["artifact"], PREFIX)
    record = CaseExecution(
        case_key(case), json.dumps(request), json.dumps(response), 19
    )
    with pytest.raises(ValueError, match="negative reached net income"):
        subject._record(
            case, record, 19, *args, PREFIX, dict(b.rule_roles), case["period"]
        )


def drop_output(d, index, name):
    """Synthetic coherent output/trace projection, no invented value."""
    case = d["cases"][index - 1]
    record = d["records"][index - 1]
    key = PREFIX + name
    del case["output"][key]
    record["request"]["queries"][0]["outputs"].remove(key)
    result = record["response"]["results"][0]
    del result["outputs"][key]
    keep = set(result["outputs"])
    pending = list(keep)
    while pending:
        for dependency in result["trace"][pending.pop()].get("dependencies", []):
            if dependency not in keep:
                keep.add(dependency)
                pending.append(dependency)
    result["trace"] = {k: v for k, v in result["trace"].items() if k in keep}


@pytest.mark.parametrize("role", ["recipient_income", "recipient_withholding"])
@pytest.mark.parametrize("index", [4, 6, 8, 10])
def test_provider_permission_output_cannot_borrow_other_output(tmp_path, role, index):
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    drop_output(d, index, dict(b.rule_roles)[role])
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize("role", ["enclosure", "identification"])
def test_each_document_principal_needs_own_pair(tmp_path, role):
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    drop_output(d, 21, dict(b.rule_roles)[role])
    assert observe(tmp_path, d) is None


def test_all_zero_net_no_transfer_actual_results_are_zero(tmp_path):
    d = data()
    observed = observe(tmp_path, d)
    assert observed is not None
    assert set(observed.zero_transfer_cases) == {
        case_key(d["cases"][i - 1]) for i in (18, 19, 26)
    }


def test_companion_bytes_and_index_bind_actual_record(tmp_path):
    d = data()
    e = evidence(tmp_path, d)
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    assert (
        subject.observe_retirement_transfer(b, e, d["content"], d["cases"]) is not None
    )
    altered = copy.deepcopy(d["cases"])
    altered[0]["name"] = "different"
    assert subject.observe_retirement_transfer(b, e, d["content"], altered) is None
    path = e.snapshot_root / ROOT
    path.write_text(d["content"] + "\n")
    assert subject.observe_retirement_transfer(b, e, d["content"], d["cases"]) is None


def test_withholding_duty_requires_principal_pair(tmp_path):
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    drop_output(d, 11, dict(b.rule_roles)["withholding_duty"])
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "role",
    [
        "recipient_income",
        "recipient_withholding",
        "recipient_income_duty",
        "recipient_withholding_duty",
    ],
)
def test_recipient_requested_pair_is_required_for_each_output(tmp_path, role):
    d = data()
    b = bind_retirement_transfer_form(d["content"], d["source"], CITATION)
    name = dict(b.rule_roles)[role]
    for index in (
        (24, 25) if role in ("recipient_income", "recipient_withholding") else (25, 26)
    ):
        drop_output(d, index, name)
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize("where", ["dataset", "query"])
def test_unknown_transport_cannot_be_silently_ignored(tmp_path, where):
    d = data()
    request = d["records"][0]["request"]
    if where == "dataset":
        request["dataset"]["tables"] = [{"name": "override", "rows": [{"value": 1}]}]
    else:
        request["queries"][0]["version_override"] = "2026-01-01"
    assert observe(tmp_path, d) is None


def capture(tmp_path, d, *, seed=True, companion_raw=None):
    original = tmp_path / "original"
    snapshot = tmp_path / "snapshot"
    files = []
    if companion_raw is None:
        companion_raw = (
            d["companion_raw"].encode()
            if d["cases"] == data()["cases"]
            else json.dumps(d["cases"]).encode()
        )
    for relative, raw in [
        (ROOT, d["content"].encode()),
        (ROOT.replace(".yaml", ".test.yaml"), companion_raw),
    ]:
        for root in (original, snapshot):
            p = root / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(raw)
        files.append(ClosureFile(relative, original / relative, raw))
    binary = tmp_path / "engine"
    binary.write_bytes(b"synthetic-unit-test-binary")
    c = ExecutionCapture(
        original,
        snapshot,
        ROOT,
        tuple(files),
        binary,
        digest(binary.read_bytes()),
        d["engine_pin"],
        0,
        "a" * 40,
    )
    if seed:
        c._retirement_transfer_seed = subject._capture_source_binding(
            d["content"], {CITATION: d["source"]}, ROOT
        )
        assert c._retirement_transfer_seed is not None
    artifact = tmp_path / "compiled.json"
    artifact.write_text(json.dumps(d["artifact"]))
    c.compiled(artifact)
    return c


def stage(c, d, index=0):
    r = d["records"][index]
    c.stage(
        d["cases"][index],
        r["request"],
        r["response"],
        c.binary,
        c.artifact,
        case_index=r["case_index"],
    )


def test_private_q_context_accepts_all_actual26_and_recompile_clears(tmp_path):
    d = data()
    c = capture(tmp_path, d)
    assert c._retirement_transfer_context is not None
    for i in range(26):
        stage(c, d, i)
    assert len(c.records) == 26
    e = c.evidence()
    assert e is not None
    b = subject._capture_source_binding(d["content"], {CITATION: d["source"]}, ROOT)
    assert (
        subject.observe_retirement_transfer(b, e, d["content"], d["cases"]) is not None
    )
    c.compiled(c.artifact)
    assert not c.records and c._retirement_transfer_context is not None


def test_unseeded_old_strict_route_remains18_local_only(tmp_path):
    d = data()
    c = capture(tmp_path, d, seed=False)
    for i in range(26):
        stage(c, d, i)
    assert len(c.records) == 18


def test_q_seed_is_not_a_constructor_option(tmp_path):
    d = data()
    c = capture(tmp_path, d)
    assert (
        "_retirement_transfer_seed"
        not in __import__("inspect").signature(ExecutionCapture).parameters
    )
    c._child_credit_seed = object()
    c.compiled(c.artifact)
    assert c._retirement_transfer_context is None
    stage(c, d, 0)
    assert not c.records


@pytest.mark.parametrize(
    "defect",
    [
        "compiled_slot",
        "compiled_catalog",
        "compiled_unit",
        "compiled_format",
        "unsupported_pin",
        "forged_seed",
        "missing_companion",
        "duplicate_cases",
    ],
)
def test_invalid_context_never_falls_back_to_same_entity_case(tmp_path, defect):
    d = data()
    if defect == "compiled_slot":
        next(
            r
            for r in d["artifact"]["program"]["derived"]
            if r["expr"].get("kind") == "sum_related"
        )["expr"]["current_slot"] = 0
    elif defect == "compiled_catalog":
        d["artifact"]["metadata"]["input_catalog"][0]["request_names"].append("foreign")
    elif defect == "compiled_unit":
        d["artifact"]["program"]["units"][-1]["minor_units"] = 0
    elif defect == "compiled_format":
        d["artifact"]["artifact_format_version"] = True
    elif defect == "unsupported_pin":
        d["engine_pin"] = "0" * 40
    elif defect == "duplicate_cases":
        d["cases"].append(copy.deepcopy(d["cases"][0]))
    c = capture(tmp_path, d)
    if defect == "forged_seed":
        c._retirement_transfer_seed = replace(
            c._retirement_transfer_seed, content_sha256="0" * 64
        )
        c.compiled(c.artifact)
    elif defect == "missing_companion":
        c.files = c.files[:1]
        c.compiled(c.artifact)
    assert c._retirement_transfer_context is None
    stage(c, d, 0)
    assert not c.records and c.evidence() is None


@pytest.mark.parametrize(
    "defect",
    [
        "case_name",
        "case_index",
        "case_bool_index",
        "actual_kind",
        "foreign_trace",
        "relation_reverse",
        "pins",
        "assessment",
        "binary",
        "artifact",
        "original",
        "snapshot",
        "companion",
        "context_root",
        "context_pin",
        "context_hash",
    ],
)
def test_capture_boundaries_refuse(tmp_path, defect):
    d = data()
    c = capture(tmp_path, d)
    r = d["records"][1]
    if defect == "case_name":
        d["cases"][1]["name"] += "changed"
    elif defect == "case_index":
        r["case_index"] = 1
    elif defect == "case_bool_index":
        r["case_index"] = True
    elif defect == "actual_kind":
        next(f for f in r["request"]["dataset"]["inputs"] if f["entity"] == "Person")[
            "entity"
        ] = "Family"
    elif defect == "foreign_trace":
        trace = r["response"]["results"][0]["trace"]
        key = next(k for k in trace if "@entity:" in k)
        trace[key]["entity_id"] = "unrelated"
    elif defect == "relation_reverse":
        r["request"]["dataset"]["relations"][0]["tuple"].reverse()
    elif defect == "pins":
        r["request"]["pins"] = []
    elif defect == "assessment":
        r["request"]["queries"][0]["assessment_date"] = "2025-12-31"
        r["response"]["results"][0]["assessment_date"] = "2025-12-31"
    elif defect == "binary":
        c.binary.write_bytes(b"changed")
    elif defect == "artifact":
        c.artifact.write_text(c.artifact.read_text() + " ")
    elif defect == "original":
        c.files[0].original.write_text(d["content"] + "\n")
    elif defect == "snapshot":
        (c.snapshot_root / ROOT).write_text(d["content"] + "\n")
    elif defect == "companion":
        c.files[1].original.write_bytes(b"[]")
    elif defect == "context_root":
        c._retirement_transfer_context = replace(
            c._retirement_transfer_context, root_relative="ca/policies/other.yaml"
        )
    elif defect == "context_pin":
        c._retirement_transfer_context = replace(
            c._retirement_transfer_context, engine_pin="0" * 40
        )
    elif defect == "context_hash":
        c._retirement_transfer_context = replace(
            c._retirement_transfer_context, artifact_sha256="0" * 64
        )
    stage(c, d, 1)
    assert not c.records


@pytest.mark.parametrize(
    "defect", ["absent", "wrong_hash", "wrong_path", "alias_path", "imports"]
)
def test_no_source_or_path_metadata_grants_eligibility(defect):
    d = data()
    source = {CITATION: d["source"]}
    path = ROOT
    content = d["content"]
    if defect == "absent":
        source = {}
    elif defect == "wrong_hash":
        source[CITATION] += "\n"
    elif defect == "wrong_path":
        path = ROOT.replace("schedule-q-retirement-income-transfer", "other")
    elif defect == "alias_path":
        path = ROOT.replace("ca/", "ca//", 1)
    elif defect == "imports":
        content = content + "\nimports: [foreign]\n"
    assert subject._capture_source_binding(content, source, path) is None


@pytest.mark.parametrize(
    "raw",
    [
        b"- &cycle {name: cycle, input: {self: *cycle}}\n",
        b"- name: one\n  name: two\n",
        b"- name: bad\n  input: {x: .nan}\n",
    ],
)
def test_ordinary_companion_unsafe_structures_refuse_context(tmp_path, raw):
    d = data()
    c = capture(tmp_path, d, companion_raw=raw)
    assert c._retirement_transfer_context is None
    stage(c, d, 0)
    assert not c.records


def q_snapshot_world(tmp_path):
    """Local Git fixture, no authenticated checkout/binary claim."""
    import subprocess
    from types import SimpleNamespace

    d = data()
    root = tmp_path / "rulespec-ca"
    target = root / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(d["content"])
    companion = target.with_suffix(".test.yaml")
    companion.write_text(d["companion_raw"])
    sibling = root / "ca/policies/unrelated.yaml"
    sibling.write_text("format: rulespec/v1\nrules: []\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    binary = tmp_path / "fixture-engine"
    binary.write_bytes(b"not executed")
    pipe = SimpleNamespace(
        require_complete_source_unit=True,
        _declared_engine_pin=lambda: SimpleNamespace(sha=d["engine_pin"]),
        policy_repo_path=root / "ca",
        rulespec_dependency_roots=(),
        _rulespec_compile_roots=lambda: [root],
        _rulespec_test_path=lambda _: companion,
        _complete_source_unit_import_bindings=lambda _: (),
        _source_texts_for_rulespec_content=lambda _: {CITATION: d["source"]},
        _proof_source_texts_for_rulespec_content=lambda *a, **k: {
            CITATION: d["source"]
        },
        _axiom_rules_binary=lambda: binary,
    )
    return pipe, target


def test_q_snapshot_mints_only_exact_two_file_source_bound_context(tmp_path):
    from axiom_encode.harness import same_case_execution as same

    pipe, target = q_snapshot_world(tmp_path)
    with same.compilation_snapshot(pipe, target) as c:
        assert c is not None and c._retirement_transfer_seed is not None
        assert c._child_credit_seed is None
        assert {f.relative for f in c.files} == {
            ROOT,
            ROOT.replace(".yaml", ".test.yaml"),
        }
        assert not (c.snapshot_root / "ca/policies/unrelated.yaml").exists()


@pytest.mark.parametrize(
    "change",
    [
        "unknown_module",
        "unbound_graph",
        "missing_pin",
        "dependencies",
        "not_git",
        "symlink",
        "missing_companion",
        "wrong_source",
    ],
)
def test_q_snapshot_plan_stays_closed(tmp_path, change):
    from axiom_encode.harness import same_case_execution as same

    pipe, target = q_snapshot_world(tmp_path)
    if change == "unknown_module":
        target.write_text(
            "format: rulespec/v1\nmodule: {summary: Unrelated local module}\nrules: []\n"
        )
    elif change == "unbound_graph":
        target.write_text(data()["content"].replace("min(", "max(", 1))
    elif change == "missing_pin":
        pipe._declared_engine_pin = lambda: None
    elif change == "dependencies":
        pipe.rulespec_dependency_roots = (tmp_path / "rulespec-us",)
    elif change == "not_git":
        __import__("shutil").rmtree(pipe.policy_repo_path.parent / ".git")
    elif change == "symlink":
        other = tmp_path / "other.yaml"
        other.write_text(data()["content"])
        target.unlink()
        target.symlink_to(other)
    elif change == "missing_companion":
        target.with_suffix(".test.yaml").unlink()
    elif change == "wrong_source":
        pipe._proof_source_texts_for_rulespec_content = lambda *a, **k: {
            CITATION: data()["source"] + "changed"
        }
    assert same._snapshot_plan(pipe, target) is None


def test_source_change_between_plan_and_materialization_has_no_fallback(tmp_path):
    from axiom_encode.harness import same_case_execution as same

    pipe, target = q_snapshot_world(tmp_path)
    plan = same._snapshot_plan(pipe, target)
    assert plan is not None
    pipe._proof_source_texts_for_rulespec_content = lambda *a, **k: {
        CITATION: data()["source"] + "changed"
    }
    destination = tmp_path / "snapshot"
    destination.mkdir()
    assert same._materialize_snapshot(pipe, target, plan, str(destination)) is None


def test_context_does_not_accept_different_raw_companion_even_if_case_values_match(
    tmp_path,
):
    d = data()
    c = capture(tmp_path, d)
    assert c._retirement_transfer_context is not None
    c._retirement_transfer_context = replace(
        c._retirement_transfer_context, case_keys=("f" * 64,) * 26
    )
    stage(c, d, 0)
    assert not c.records


def test_coherently_replaced_context_companion_is_not_actual_closure(tmp_path):
    d = data()
    c = capture(tmp_path, d)
    changed = copy.deepcopy(d["cases"])
    changed[0]["name"] = "different-raw-companion"
    c._retirement_transfer_context = subject._bind_capture_context(
        c._retirement_transfer_seed,
        d["content"],
        ROOT,
        d["engine_pin"],
        c.artifact.read_bytes(),
        json.dumps(changed).encode(),
    )
    assert c._retirement_transfer_context is not None and c.fresh()
    r = d["records"][0]
    c.stage(changed[0], r["request"], r["response"], c.binary, c.artifact, case_index=1)
    assert not c.records


@pytest.mark.parametrize("kind", ["extra", "duplicate"])
def test_private_q_context_requires_exact_two_file_closure(tmp_path, kind):
    d = data()
    c = capture(tmp_path, d)
    if kind == "duplicate":
        c.files += c.files[:1]
    else:
        relative = "ca/policies/extra.yaml"
        raw = b"format: rulespec/v1\nrules: []\n"
        for root in (c.original_root, c.snapshot_root):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        c.files += (ClosureFile(relative, c.original_root / relative, raw),)
    assert c.fresh()
    stage(c, d, 0)
    assert not c.records


def test_boolean_case_index_cannot_alias_matching_first_case(tmp_path):
    d = data()
    c = capture(tmp_path, d)
    r = d["records"][0]
    c.stage(
        d["cases"][0],
        r["request"],
        r["response"],
        c.binary,
        c.artifact,
        case_index=True,
    )
    assert not c.records

"""Actual unsigned82-call fixture; standalone validation only, no capture grant."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_capital_gift_operands import bind_capital_gift
from axiom_encode.harness.capital_gift_execution import (
    ENGINE_PIN,
    PREFIX,
    _compiled,
    _record,
    capital_gift_case_key,
)

F = Path(__file__).parent / "fixtures"
STATIC = json.loads((F / "t1170-static.json").read_text())
EXECUTION = json.loads((F / "t1170-execution.json").read_text())
CASES = yaml.safe_load(EXECUTION["companion_yaml"])
CASES = CASES["cases"] if isinstance(CASES, dict) else CASES


@pytest.fixture(scope="module")
def bound():
    b = bind_capital_gift(
        STATIC["policy_yaml"],
        (F / "t1170-authenticated-body.txt").read_text(),
        STATIC["p113_body"],
    )
    assert b is not None
    return b, *_compiled(b, EXECUTION["artifact"], ENGINE_PIN)


@pytest.mark.parametrize("index", range(82))
def test_actual_typed_record_and_provider_closure(bound, index):
    b, derived, parameters = bound
    record = _record(b, derived, parameters, CASES[index], EXECUTION["calls"][index])
    assert len(record.outputs) == len(CASES[index]["output"])
    assert record.root_entity_id == CASES[index]["entity_id"]


@pytest.mark.parametrize(
    "mutation",
    [
        "pin",
        "format_bool",
        "units_extra",
        "units_precision",
        "units_duplicate",
        "builtin",
        "catalog_extra",
        "catalog_alias",
        "catalog_missing_index",
        "relation_direction",
        "rule_extra",
        "rule_missing",
        "expr",
        "version_expr",
        "date_value",
        "source",
    ],
)
def test_compiled_union_and_type_profile(bound, mutation):
    b, _, _ = bound
    a = copy.deepcopy(EXECUTION["artifact"])
    pin = ENGINE_PIN
    if mutation == "pin":
        pin = "unknown"
    elif mutation == "format_bool":
        a["artifact_format_version"] = True
    elif mutation == "units_extra":
        a["program"]["units"].append({"name": "EXTRA", "kind": "count"})
    elif mutation == "units_precision":
        a["program"]["units"][-1]["minor_units"] = 2.0
    elif mutation == "units_duplicate":
        a["program"]["units"].append(copy.deepcopy(a["program"]["units"][-1]))
    elif mutation == "builtin":
        a["program"]["units"][0]["minor_units"] = 0
    elif mutation == "catalog_extra":
        a["metadata"]["input_catalog"].append({"slot": "unknown"})
    elif mutation == "catalog_alias":
        a["metadata"]["input_catalog"][0]["request_names"].append("alias")
    elif mutation == "catalog_missing_index":
        a["metadata"]["input_catalog"] = [
            v for v in a["metadata"]["input_catalog"] if v["slot"] != "t1170_form_year"
        ]
    elif mutation == "relation_direction":
        a["program"]["relations"][0]["slot_entities"].reverse()
    elif mutation == "rule_extra":
        a["program"]["derived"].append(copy.deepcopy(a["program"]["derived"][0]))
    elif mutation == "rule_missing":
        a["program"]["derived"].pop()
    elif mutation == "expr":
        a["program"]["derived"][0]["expr"] = {
            "kind": "literal",
            "value": {"kind": "decimal", "value": "0"},
        }
    elif mutation == "version_expr":
        a["program"]["derived"][0]["versions"][0]["expr"] = {
            "kind": "literal",
            "value": {"kind": "decimal", "value": "0"},
        }
    elif mutation == "date_value":
        next(
            v
            for v in a["program"]["parameters"]
            if v["name"] == "t1170_form_year_start"
        )["versions"][0]["values"]["2025"] = {"kind": "text", "value": "2025-01-01"}
    else:
        a["program"]["module"]["source_verification"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        _compiled(b, a, pin)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_kind",
        "bool_integer",
        "unknown_input",
        "parameter_override",
        "assessment",
        "nested_table",
        "case_name",
        "Date_to_Text",
        "period",
        "extra_query",
        "relation_duplicate",
        "relation_reverse",
        "provider_id",
        "trace_extra",
        "trace_absent",
        "trace_value",
        "trace_id",
        "trace_expression",
        "trace_dependencies",
        "trace_skipped",
        "trace_parameter",
        "response_mode",
        "nonzero",
    ],
)
def test_actual_request_record_counterexamples(bound, mutation):
    b, derived, parameters = bound
    index = (
        8
        if mutation == "Date_to_Text"
        else 70
        if mutation.startswith("relation") or mutation == "provider_id"
        else 0
    )
    if mutation == "Date_to_Text":
        index = next(
            i
            for i, c in enumerate(CASES)
            if any(k.endswith("t1170_reported_bond_maturity_date") for k in c["output"])
        )
    case = copy.deepcopy(CASES[index])
    call = copy.deepcopy(EXECUTION["calls"][index])
    request = call["request"]
    response = json.loads(call["stdout"])
    result = response["results"][0]
    trace = result["trace"]
    if mutation == "wrong_kind":
        request["dataset"]["inputs"][0]["entity"] = "Person"
    elif mutation == "bool_integer":
        fact = next(
            f for f in request["dataset"]["inputs"] if f["value"]["kind"] == "bool"
        )
        fact["value"]["value"] = 1
    elif mutation == "unknown_input":
        request["dataset"]["inputs"][0]["name"] += "_other"
    elif mutation == "parameter_override":
        fact = copy.deepcopy(request["dataset"]["inputs"][0])
        fact["name"] = PREFIX + "input.t1170_form_year"
        request["dataset"]["inputs"].append(fact)
    elif mutation == "assessment":
        request["queries"][0]["assessment_date"] = "2025-12-31"
    elif mutation == "nested_table":
        request["dataset"]["tables"] = []
    elif mutation == "case_name":
        # Stored raw identity must change; record validation alone is not live context.
        old = capital_gift_case_key(case, b)
        case["name"] += " changed"
        assert capital_gift_case_key(case, b) != old
        return
    elif mutation == "Date_to_Text":
        key = next(
            k for k in case["output"] if k.endswith("t1170_reported_bond_maturity_date")
        )
        case["output"][key] = str(case["output"][key])
    elif mutation == "period":
        request["queries"][0]["period"]["end"] = "2025-06-30"
    elif mutation == "extra_query":
        request["queries"].append(copy.deepcopy(request["queries"][0]))
    elif mutation == "relation_duplicate":
        request["dataset"]["relations"].append(
            copy.deepcopy(request["dataset"]["relations"][0])
        )
    elif mutation == "relation_reverse":
        request["dataset"]["relations"][0]["tuple"].reverse()
    elif mutation == "provider_id":
        request["dataset"]["relations"][0]["tuple"][0] = "unrelated"
    elif mutation == "trace_extra":
        trace["unrelated"] = copy.deepcopy(next(iter(trace.values())))
    elif mutation == "trace_absent":
        trace.pop(next(iter(trace)))
    elif mutation == "trace_value":
        next(v for v in trace.values() if v.get("value", {}).get("kind") == "decimal")[
            "value"
        ]["value"] = "99999"
    elif mutation == "trace_id":
        next(iter(trace.values()))["id"] += "_other"
    elif mutation == "trace_expression":
        next(iter(trace.values()))["executed_expression"] = "0"
    elif mutation == "trace_dependencies":
        next(v for v in trace.values() if v["dependencies"])["dependencies"] = []
    elif mutation == "trace_skipped":
        next(v for v in trace.values() if "not_evaluated_dependencies" in v)[
            "not_evaluated_dependencies"
        ] = []
    elif mutation == "trace_parameter":
        next(v for v in trace.values() if "parameter_reads" in v)["parameter_reads"][0][
            "value"
        ]["value"] = "1900"
    elif mutation == "response_mode":
        response["metadata"]["actual_mode"] = "fast"
    else:
        call["returncode"] = 1
    call["stdout"] = json.dumps(response)
    with pytest.raises((ValueError, KeyError)):
        _record(b, derived, parameters, case, call)


def observation(bound, cases=None, calls=None, artifact=None):
    from axiom_encode.harness.capital_gift_execution import observe_capital_gift

    b, _, _ = bound
    return observe_capital_gift(
        b,
        STATIC["policy_yaml"],
        (F / "t1170-authenticated-body.txt").read_text(),
        STATIC["p113_body"],
        artifact or EXECUTION["artifact"],
        calls if calls is not None else EXECUTION["calls"],
        yaml.safe_dump(cases, sort_keys=False)
        if cases is not None
        else EXECUTION["companion_yaml"],
        ENGINE_PIN,
    )


def test_complete_positive_observation_keeps_refusal_gap_explicit(bound):
    result = observation(bound)
    assert result is not None
    assert len(result.records) == 82
    assert sum(len(r.outputs) for r in result.records) == 361
    assert len(result.root_role_pairs) == 42
    assert result.pending_refusal_contract == (
        "direct_gift_principal",
        "direct_prior_exchange_principal",
        "direct_partnership_principal",
        "provider_refusal_lineage",
    )


@pytest.mark.parametrize(
    ("output", "blocked"),
    [
        ("t1170_enclosure_required", "not_holds"),
        ("t1170_additional_sheet_required", "not_holds"),
        ("t1170_fts_t691_reporting_required", "not_holds"),
        ("t1170_partnership_special_calculation_required", "not_holds"),
        ("t1170_gift_within_form_scope", False),
        ("t1170_exchange_extension_conditions_met", "not_holds"),
    ],
)
def test_coherent_case_and_record_omission_reaches_role_requirement(
    bound, output, blocked
):
    from axiom_encode.harness.capital_gift_execution import _positive_coverage

    keep = [
        i for i, c in enumerate(CASES) if c["output"].get(PREFIX + output) != blocked
    ]
    assert 0 < len(keep) < 82
    cases = [CASES[i] for i in keep]
    calls = [EXECUTION["calls"][i] for i in keep]
    b, derived, parameters = bound
    # These are explicitly reconstructed unsigned test scenarios; every retained
    # exact record passes before the required role-pair guard is exercised.
    records = tuple(
        _record(b, derived, parameters, c, a) for c, a in zip(cases, calls, strict=True)
    )
    assert len(records) == len(cases)
    with pytest.raises(ValueError, match="missing root-role"):
        _positive_coverage(b, records, cases)
    assert observation(bound, cases, calls) is None


@pytest.mark.parametrize(
    "category", ["shares_units", "bonds_other", "ecological_land", "partnership"]
)
def test_each_provider_pooling_distinction_is_required(bound, category):
    from axiom_encode.harness.capital_gift_execution import _positive_coverage

    relation = (
        PREFIX
        + "relation.t1170_"
        + category
        + ("_exchange_for_return" if category == "partnership" else "_row_for_return")
    )
    keep = [i for i, c in enumerate(CASES) if len(c["input"].get(relation, [])) < 2]
    assert len(keep) < 82
    cases = [CASES[i] for i in keep]
    calls = [EXECUTION["calls"][i] for i in keep]
    b, derived, parameters = bound
    records = tuple(
        _record(b, derived, parameters, c, a) for c, a in zip(cases, calls, strict=True)
    )
    with pytest.raises(ValueError, match="per-entry-before-aggregate"):
        _positive_coverage(b, records, cases)


def test_root_pair_alpha_renaming_does_not_relax_actual_record_identity(bound):
    from axiom_encode.harness.capital_gift_execution import _root_pairs

    b, derived, parameters = bound
    cases = [CASES[i] for i in (49, 50)]
    records = [
        _record(b, derived, parameters, CASES[i], EXECUTION["calls"][i])
        for i in (49, 50)
    ]
    assert cases[0]["entity_id"] != cases[1]["entity_id"]
    pairs = _root_pairs(records, cases)
    assert len(pairs) == 1
    assert pairs[0][2:] == (records[0].case_key, records[1].case_key)
    changed = copy.deepcopy(EXECUTION["calls"][49])
    changed["request"]["queries"][0]["entity_id"] = cases[1]["entity_id"]
    with pytest.raises(ValueError):
        _record(b, derived, parameters, cases[0], changed)


def test_root_pair_does_not_generalize_to_related_provider_graph(bound):
    from axiom_encode.harness.capital_gift_execution import _root_pairs

    b, derived, parameters = bound
    cases = [CASES[i] for i in (70, 72)]
    records = [
        _record(b, derived, parameters, CASES[i], EXECUTION["calls"][i])
        for i in (70, 72)
    ]
    assert _root_pairs(records, cases) == ()


@pytest.mark.parametrize(
    "mutation",
    [
        "tag_string",
        "unknown_date_path",
        "nan",
        "cycle",
        "float_bool",
        "ordinal_name",
        "date_to_string",
    ],
)
def test_private_typed_case_identity_is_not_generic_serialization(bound, mutation):
    from datetime import date

    b, _, _ = bound
    case = copy.deepcopy(CASES[0])
    original = capital_gift_case_key(case, b)
    if mutation == "tag_string":
        case["name"] = '["date","2025-01-01"]'
    elif mutation == "ordinal_name":
        case["name"] += " another"
    elif mutation == "float_bool":
        case["input"][PREFIX + "input.t1170_qualified_donee"] = 1.0
    elif mutation == "date_to_string":
        case = copy.deepcopy(
            next(c for c in CASES if any(type(v) is date for v in c["output"].values()))
        )
        original = capital_gift_case_key(case, b)
        for k, v in case["output"].items():
            if type(v) is date:
                case["output"][k] = v.isoformat()
    elif mutation == "unknown_date_path":
        case["name"] = date(2025, 1, 1)
    elif mutation == "nan":
        case["input"][PREFIX + "input.t1170_proceeds_of_disposition_column_2"] = float(
            "nan"
        )
    else:
        case["cycle"] = case
    if mutation in {"unknown_date_path", "nan", "cycle"}:
        with pytest.raises(ValueError):
            capital_gift_case_key(case, b)
    else:
        assert capital_gift_case_key(case, b) != original


def test_artifact_summary_and_fast_metadata_are_not_ignored(bound):
    for field in ("summary", "blockers"):
        a = copy.deepcopy(EXECUTION["artifact"])
        if field == "summary":
            a["program"]["module"]["summary"] += " altered"
        else:
            a["metadata"]["fast_path"]["blockers"].pop()
        assert observation(bound, artifact=a) is None

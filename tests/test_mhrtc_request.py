"""Closed request transport tests; no engine, source authority or coverage."""

import copy
import json
from decimal import Decimal

import pytest
import yaml

from axiom_encode.harness import mhrtc_request as m
from axiom_encode.harness.mhrtc_program_contract import (
    INPUTS,
    POLICY_PATH,
    RULES,
    UNITS,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

PREFIX = "ca:policies/cra/t1-2025/multigenerational-home-renovation-tax-credit#"
OUTPUT = PREFIX + "federal_2025_mhrtc_total_qualifying_expenditures_line_45354"
CREDIT = PREFIX + "multigenerational_home_renovation_tax_credit_line_45355"


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "rulespec-ca"
    policy = root / POLICY_PATH
    policy.parent.mkdir(parents=True)
    policy.write_text(
        yaml.safe_dump(
            {"format": "rulespec/v1", "inputs": INPUTS, "rules": RULES, "units": UNITS}
        )
    )
    pipe = ValidatorPipeline(
        root / "ca",
        tmp_path / "unused-engine",
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )
    # This is a deliberately small transport IR, not a claimed compiled M program.
    compiled = {
        "program": {
            "derived": [
                {
                    "id": PREFIX + r["name"],
                    "name": r["name"],
                    "entity": r["entity"],
                    "dtype": r["dtype"].lower(),
                }
                for r in RULES
                if r["kind"] == "derived"
            ],
            "parameters": [
                {"id": PREFIX + r["name"], "name": r["name"]}
                for r in RULES
                if r["kind"] == "parameter"
            ],
            "relations": [
                {
                    "name": PREFIX + "relation." + r["name"],
                    "arity": r["data_relation"]["arity"],
                    "slot_entities": r["data_relation"]["arguments"],
                }
                for r in RULES
                if r["kind"] == "data_relation"
            ],
        }
    }
    case = {
        "name": "explicit transport",
        "entity_id": "person-1",
        "period": {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
        "input": {PREFIX + "input.filing_claimant_key": 1},
        "tables": {
            "Claim": [
                {
                    "id": "claim-1",
                    PREFIX + "input.claimant_key": 1,
                    PREFIX + "input.dwelling_land_area_hectares": 0.5,
                }
            ]
        },
        "relations": [
            {
                "name": PREFIX + "relation.claim_records",
                "tuple": ["claim-1", "person-1"],
            }
        ],
        "output": {OUTPUT: 40000, CREDIT: 5800},
    }
    monkeypatch.setattr(
        pipe, "_axiom_rules_binary", lambda: pytest.fail("no engine resolution")
    )
    return case, compiled, pipe, policy


def plan(fixture):
    case, compiled, pipe, policy = fixture
    return m.reconstruct_mhrtc_request(
        case, compiled, pipe=pipe, rules_file=policy, case_index=1
    )


def test_exact_ordinary_dataset_query_and_output_order(fixture):
    result = plan(fixture)
    assert result.parameter_outputs == ()
    assert result.derived_outputs == (OUTPUT, CREDIT)
    request = result.request
    assert request["mode"] == "explain"
    assert request["queries"] == [
        {
            "entity_id": "person-1",
            "period": fixture[0]["period"],
            "outputs": [OUTPUT, CREDIT],
        }
    ]
    assert request["dataset"]["inputs"][0] == {
        "name": PREFIX + "input.filing_claimant_key",
        "entity": "Family",
        "entity_id": "person-1",
        "interval": {"start": "2025-01-01", "end": "2025-12-31"},
        "value": {"kind": "integer", "value": 1},
    }
    assert request["dataset"]["relations"][0]["tuple"] == ["claim-1", "person-1"]
    fixture[0]["output"] = {CREDIT: 5800, OUTPUT: 40000}
    assert plan(fixture).request["queries"][0]["outputs"] == [CREDIT, OUTPUT]


def test_root_mutation_is_reconstructed_not_borrowed(fixture):
    old = plan(fixture).request
    case = fixture[0]
    case["entity_id"] = "different-root"
    case["relations"][0]["tuple"][1] = "different-root"
    new = plan(fixture).request
    assert new != old
    assert new["queries"][0]["entity_id"] == "different-root"
    assert new["dataset"]["inputs"][0]["entity_id"] == "different-root"
    assert new["dataset"]["relations"][0]["tuple"][1] == "different-root"


def test_exact_decimal_companion_scalar_reaches_wire_without_float_rounding(fixture):
    case, compiled, pipe, policy = fixture
    text = yaml.safe_dump([case], sort_keys=False).replace(
        "0.5", "0.50000000000000000001"
    )
    parsed = m.parse_mhrtc_companion(text)[0]
    assert parsed["tables"]["Claim"][0][
        PREFIX + "input.dwelling_land_area_hectares"
    ] == Decimal("0.50000000000000000001")
    result = m.reconstruct_mhrtc_request(
        parsed, compiled, pipe=pipe, rules_file=policy, case_index=1
    )
    assert result.request["dataset"]["inputs"][-1]["value"] == {
        "kind": "decimal",
        "value": "0.50000000000000000001",
    }
    json.dumps(result.request, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "period",
        "period_alias",
        "extra_field",
        "root_type",
        "row_alias",
        "anonymous_row",
        "unknown_input",
        "computed_input",
        "wrong_entity",
        "bool_integer",
        "nonfinite",
        "unknown_output",
        "output_alias",
        "output_rowlist",
        "mixed_entity",
        "reversed_edge",
        "unknown_endpoint",
        "edge_alias",
        "extra_edge_field",
        "parameter_graph",
        "typed_date",
    ],
)
def test_closed_case_mutations_reject(fixture, mutation):
    case = fixture[0]
    row = case["tables"]["Claim"][0]
    if mutation == "period":
        case["period"]["end"] = "2026-12-31"
    elif mutation == "period_alias":
        case["period"] = "2025"
    elif mutation == "extra_field":
        case["expected_refusal"] = {}
    elif mutation == "root_type":
        case["entity_id"] = 1
    elif mutation == "row_alias":
        row["entity_id"] = row.pop("id")
    elif mutation == "anonymous_row":
        row.pop("id")
    elif mutation == "unknown_input":
        case["input"][PREFIX + "input.opaque_eligibility"] = True
    elif mutation == "computed_input":
        case["input"][OUTPUT] = 40000
    elif mutation == "wrong_entity":
        case["input"][PREFIX + "input.claimant_key"] = 1
    elif mutation == "bool_integer":
        case["input"][PREFIX + "input.filing_claimant_key"] = True
    elif mutation == "nonfinite":
        row[PREFIX + "input.dwelling_land_area_hectares"] = float("inf")
    elif mutation == "unknown_output":
        case["output"] = {PREFIX + "opaque_output": 1}
    elif mutation == "output_alias":
        case["output"] = {
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354": 1
        }
    elif mutation == "output_rowlist":
        case["output"][OUTPUT] = [40000]
    elif mutation == "mixed_entity":
        case["output"][PREFIX + "line_one"] = 40000
    elif mutation == "reversed_edge":
        case["relations"][0]["tuple"].reverse()
    elif mutation == "unknown_endpoint":
        case["relations"][0]["tuple"][0] = "absent"
    elif mutation == "edge_alias":
        case["relations"][0]["name"] = "claim_records"
    elif mutation == "extra_edge_field":
        case["relations"][0]["trusted"] = True
    elif mutation == "parameter_graph":
        case["output"] = {next(iter(m._PUBLIC_PARAMETERS)): 50000}
    elif mutation == "typed_date":
        row[PREFIX + "input.completion_date"] = "2025-02-30"
    with pytest.raises(ValueError):
        plan(fixture)


def test_parameter_only_is_no_runtime_classification(fixture):
    case = fixture[0]
    case.pop("tables")
    case.pop("relations")
    case["input"] = {}
    case["output"] = {
        PREFIX
        + "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation": 50000,
        PREFIX + "federal_2025_mhrtc_rate": 0.145,
        PREFIX + "federal_2025_mhrtc_maximum_credit_per_renovation": 7250,
    }
    result = plan(fixture)
    assert result.request is None and result.derived_outputs == ()
    assert result.parameter_outputs == tuple(case["output"])
    # Classification never asserts that expected parameter values have passed.
    case["output"][PREFIX + "federal_2025_mhrtc_rate"] = 999
    assert plan(fixture).request is None


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "wrong_entity",
        "wrong_module",
        "missing_parameter",
        "relation_slots",
        "duplicate_relation",
    ],
)
def test_compiled_role_checks_fail_closed(fixture, mutation):
    case, compiled, _, _ = fixture
    if mutation == "duplicate":
        compiled["program"]["derived"].append(
            copy.deepcopy(compiled["program"]["derived"][0])
        )
    elif mutation == "wrong_entity":
        compiled["program"]["derived"][0]["entity"] = "OtherClaim"
    elif mutation == "wrong_module":
        compiled["program"]["derived"][0]["id"] = "other:policy#value"
    elif mutation == "relation_slots":
        compiled["program"]["relations"][0]["slot_entities"] = ["Family", "Claim"]
    elif mutation == "duplicate_relation":
        compiled["program"]["relations"].append(
            copy.deepcopy(compiled["program"]["relations"][0])
        )
    elif mutation == "missing_parameter":
        case.pop("tables")
        case.pop("relations")
        case["input"] = {}
        case["output"] = {p: 1 for p in m._PUBLIC_PARAMETERS}
        compiled["program"]["parameters"] = []
    with pytest.raises(ValueError):
        plan(fixture)


@pytest.mark.parametrize(
    "text",
    [
        "- name: one\n  name: two\n",
        "- &a {name: one}\n- *a\n",
        "- {<<: {name: one}}\n",
        "{}",
        "[]",
    ],
)
def test_duplicate_keys_aliases_merges_and_bad_root_yaml_reject(text):
    with pytest.raises((ValueError, yaml.YAMLError)):
        m.parse_mhrtc_companion(text)


def test_duplicate_case_names_reject(fixture):
    with pytest.raises(ValueError, match="duplicate case"):
        m.parse_mhrtc_companion(yaml.safe_dump([fixture[0], copy.deepcopy(fixture[0])]))


@pytest.mark.parametrize("index", [True, 0, -1, 1025, "1"])
def test_invalid_ordinals_reject(fixture, index):
    case, compiled, pipe, policy = fixture
    with pytest.raises(ValueError, match="ordinal"):
        m.reconstruct_mhrtc_request(
            case, compiled, pipe=pipe, rules_file=policy, case_index=index
        )

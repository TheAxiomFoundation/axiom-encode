"""Mutation controls for the static M reference, never runtime/source clearance."""

from __future__ import annotations

import copy

import pytest
import yaml

from axiom_encode.harness.bound_mhrtc_program import bind_mhrtc_program
from axiom_encode.harness.mhrtc_native_envelope import (
    INPUT_DESCRIPTIONS,
    MODULE,
    RULE_METADATA,
    RULE_SOURCES,
)
from axiom_encode.harness.mhrtc_program_contract import (
    CITATION,
    INPUTS,
    POLICY_PATH,
    RULES,
    SOURCE_TEXT,
    UNITS,
)


@pytest.fixture
def policy():
    return copy.deepcopy(
        {
            "format": "rulespec/v1",
            "imports": [],
            "module": MODULE,
            "units": UNITS,
            "inputs": [
                {**i, "description": INPUT_DESCRIPTIONS[i["name"]]} for i in INPUTS
            ],
            "rules": [
                {
                    **r,
                    "source": RULE_SOURCES[r["name"]],
                    "metadata": RULE_METADATA[r["name"]],
                }
                for r in RULES
            ],
        }
    )


def bind(policy, source=SOURCE_TEXT, **kwargs):
    return bind_mhrtc_program(yaml.safe_dump(policy, sort_keys=False), source, **kwargs)


def rule(policy, name):
    return next(item for item in policy["rules"] if item["name"] == name)


def test_match_is_static_evidence_with_unresolved_authority(policy):
    result = bind(policy)
    assert result is not None
    assert len(result.rule_groups) == 99
    assert len(policy["inputs"]) == 129
    assert len({group for group, *_ in result.source_spans}) == 18
    assert result.unresolved_obligations == (
        "independent source semantic review",
        "fresh compiler/runtime binding",
        "complete-source occurrence/witness coverage",
    )
    assert not hasattr(result, "covers_gate")
    assert not hasattr(result, "numeric_owner")
    for _, start, end, excerpt in result.source_spans:
        assert SOURCE_TEXT[start:end] == excerpt


def test_whitespace_parentheses_and_declaration_order_are_not_hash_allowlist(policy):
    original = bind(policy)
    rule(policy, "qualifying_individual")["versions"][0]["formula"] = (
        "(individual_age_at_year_end >= 65) or "
        "((individual_age_at_year_end >= 18) and individual_dtc)"
    )
    policy["rules"].reverse()
    policy["inputs"].reverse()
    result = bind(policy)
    assert result is not None
    assert result.content_sha256 != original.content_sha256
    assert dict(result.expressions) == dict(original.expressions)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_module",
        "disabled_proof",
        "source_hash",
        "description",
        "missing_atoms",
        "atom_path",
        "atom_kind",
        "excerpt",
        "citation",
        "extra_metadata",
        "removed_diagnostic",
    ],
)
def test_native_metadata_is_required_and_bound_without_stripping(policy, mutation):
    item = rule(policy, "other_allocated_local_capacity")
    atom = item["metadata"]["proof"]["atoms"][0]
    if mutation == "missing_module":
        policy.pop("module")
    elif mutation == "disabled_proof":
        policy["module"]["proof_validation"]["required"] = False
    elif mutation == "source_hash":
        policy["module"]["source_verification"]["source_sha256"] = "0" * 64
    elif mutation == "description":
        policy["inputs"][0]["description"] = "Precomputed overall entitlement"
    elif mutation == "missing_atoms":
        item["metadata"]["proof"]["atoms"] = []
    elif mutation == "atom_path":
        atom["path"] = "unrelated"
    elif mutation == "atom_kind":
        atom["kind"] = "definition"
    elif mutation == "excerpt":
        atom["source"]["excerpt"] += " Not in source."
    elif mutation == "citation":
        atom["source"]["corpus_citation_path"] += "/unrelated"
    elif mutation == "extra_metadata":
        item["metadata"]["claim_full_coverage"] = True
    else:
        extra = copy.deepcopy(item)
        extra["name"] = "claim_arithmetic_domain_valid"
        policy["rules"].append(extra)
    assert bind(policy) is None


@pytest.mark.parametrize("index", range(99))
def test_every_rule_is_closed_and_required(policy, index):
    policy["rules"].pop(index)
    assert bind(policy) is None


@pytest.mark.parametrize("index", range(129))
def test_every_input_role_is_closed_and_typed(policy, index):
    policy["inputs"][index]["entity"] = "UnrelatedEntity"
    assert bind(policy) is None


@pytest.mark.parametrize(
    ("name", "formula"),
    [
        ("qualifying_individual", "individual_age_at_year_end >= 18"),
        ("eligible_claimant", "canada_resident_full_year"),
        ("eligible_dwelling", "dwelling_is_housing_unit and dwelling_in_canada"),
        (
            "receipt_basic_conditions",
            "expense_reasonable and expense_directly_attributable",
        ),
        ("paid_after_source_cutoff", "record_paid_date >= expense_start_cutoff[0]"),
        (
            "other_paid_after_source_cutoff",
            "other_record_paid_date > expense_start_cutoff[0]",
        ),
        ("owned_claim_records", "record_claimant_key != filing_claimant_identity"),
        (
            "line_three",
            "min(line_one, federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation)",
        ),
        ("selected_claim_capacity", "allocated_local_capacity"),
        ("federal_2025_mhrtc_rate", "0.15"),
        ("federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation", "50001"),
        ("federal_2025_mhrtc_maximum_credit_per_renovation", "7500"),
        (
            "federal_2025_mhrtc_total_qualifying_expenditures_line_45354",
            "total_qualifying_expenditures",
        ),
        (
            "selected_renovation_sheet_count",
            "if selected_qualifying_expenditures > 0: 1 else: 0",
        ),
        ("qualifying_individual", "max(individual_age_at_year_end, 65)"),
        ("qualifying_individual", "individual_age_at_year_end / 65"),
        ("qualifying_individual", "__import__('os').system('false')"),
        ("qualifying_individual", "individual_age_at_year_end >= 65 # ignored"),
        ("paid_after_source_cutoff", "record_paid_date > expense_start_cutoff[False]"),
        ("line_one", "sum(qualifying_receipts.amount_including_taxes, 0)"),
        ("line_one", "sum(qualifying_receipts.amount_including_taxes, start=0)"),
        ("line_one", "sum(qualifying_receipts.amount_including_taxes.real)"),
        ("line_one", "sum([amount_including_taxes])"),
        ("line_three", "min(line_one - line_two)"),
        ("retain_supporting_documents", "True"),
        ("retain_supporting_documents", "true\nfalse"),
        ("line_three", "line_three = 50000"),
    ],
)
def test_changed_semantics_and_unknown_grammar_fail(policy, name, formula):
    rule(policy, name)["versions"][0]["formula"] = formula
    assert bind(policy) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "entity",
        "slot",
        "direction",
        "unit",
        "period",
        "date",
        "source_date",
        "land",
        "bool_key",
        "float_key",
        "string_key",
        "extra_version",
        "extra_rule",
        "extra_input",
        "opaque_input",
        "imports",
        "module",
        "unit_bool",
        "version_bool",
    ],
)
def test_declarations_types_and_headers_are_closed(policy, mutation):
    target = rule(policy, "qualifying_individual")
    if mutation == "entity":
        target["entity"] = "Family"
    elif mutation == "slot":
        rule(policy, "receipts")["data_relation"]["arguments"].reverse()
    elif mutation == "direction":
        rule(policy, "qualifying_receipts")["derived_relation"]["current_slot"] = 0
    elif mutation == "unit":
        rule(policy, "line_one")["unit"] = "USD"
    elif mutation == "period":
        target["period"] = "Month"
    elif mutation == "date":
        target["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "source_date":
        rule(policy, "expense_start_cutoff")["versions"][0]["values"][0] = "2023-01-01"
    elif mutation == "land":
        rule(policy, "general_land_limit_hectares")["versions"][0]["values"][0] = 1.24
    elif mutation.endswith("_key"):
        key = {"bool_key": False, "float_key": 0.0, "string_key": "0"}[mutation]
        rule(policy, "expense_start_cutoff")["versions"][0]["values"] = {
            key: "2022-12-31"
        }
    elif mutation == "extra_version":
        target["versions"].append(copy.deepcopy(target["versions"][0]))
    elif mutation == "extra_rule":
        policy["rules"].append(copy.deepcopy(target))
    elif mutation == "extra_input":
        policy["inputs"].append(copy.deepcopy(policy["inputs"][0]))
    elif mutation == "opaque_input":
        policy["inputs"].append(
            {
                "name": "eligible",
                "entity": "Claim",
                "dtype": "Boolean",
                "period": "Year",
            }
        )
        target["versions"][0]["formula"] = "eligible"
    elif mutation == "imports":
        policy["imports"] = [{"path": "unrelated.yaml"}]
    elif mutation == "module":
        policy["module"] = {"summary": "trust me"}
    elif mutation == "unit_bool":
        policy["units"][0]["minor_units"] = True
    elif mutation == "version_bool":
        rule(policy, "receipts")["data_relation"]["arity"] = True
    assert bind(policy) is None


@pytest.mark.parametrize(
    "addition",
    [
        "format: rulespec/v1\n",
        "units: []\n",
        "extra: &x []\n",
        "extra: &x [*x]\n",
        "<<: {}\n",
        "? [x, y]\n: 1\n",
    ],
)
def test_duplicate_yaml_aliases_merges_and_unhashable_keys_fail(policy, addition):
    content = yaml.safe_dump(policy, sort_keys=False) + addition
    assert bind_mhrtc_program(content, SOURCE_TEXT) is None


def test_coherent_candidate_and_source_mutation_does_not_redefine_reference(policy):
    cap = rule(policy, "federal_2025_mhrtc_qualifying_expenditure_limit_per_renovation")
    cap["versions"][0]["formula"] = "60000"
    changed_source = SOURCE_TEXT.replace("50,000", "60,000")
    assert bind(policy, changed_source) is None


@pytest.mark.parametrize(
    "source",
    [SOURCE_TEXT + "\n", SOURCE_TEXT.replace("14.5%", "15%"), SOURCE_TEXT[1:], ""],
)
def test_source_is_whole_literal_and_not_candidate_summary(policy, source):
    assert bind(policy, source) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"citation": CITATION + "/subsection"},
        {"policy_path": "other.yaml"},
        {"policy_path": "/" + POLICY_PATH},
    ],
)
def test_exact_canonical_location(policy, kwargs):
    assert bind(policy, **kwargs) is None


def test_parameter_zero_key_remains_integer(policy):
    for name in ("expense_start_cutoff", "other_expense_start_cutoff"):
        assert list(rule(policy, name)["versions"][0]["values"]) == [0]
        assert type(next(iter(rule(policy, name)["versions"][0]["values"]))) is int
    assert bind(policy) is not None


@pytest.mark.parametrize("rate", ["0.1449999999999999999", "0.1450000000000000001"])
def test_decimal_formula_precision_is_not_python_float_equality(policy, rate):
    rule(policy, "federal_2025_mhrtc_rate")["versions"][0]["formula"] = rate
    assert bind(policy) is None


@pytest.mark.parametrize("value", ["0.50000000000000001", "0.49999999999999999"])
def test_yaml_parameter_precision_is_not_python_float_equality(policy, value):
    raw = yaml.safe_dump(policy, sort_keys=False)
    assert "0.5" in raw
    changed = raw.replace("0.5", value, 1)
    assert bind_mhrtc_program(changed, SOURCE_TEXT) is None

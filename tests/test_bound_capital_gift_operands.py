"""Unsigned whole-unit static contract tests; no runtime/corpus admission claim."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_capital_gift_operands import (
    CapitalGiftBinding,
    bind_capital_gift,
)

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = json.loads((FIXTURES / "t1170-static.json").read_text())
SOURCE = (FIXTURES / "t1170-authenticated-body.txt").read_text()
POLICY = yaml.safe_load(FIXTURE["policy_yaml"])
GUIDE = FIXTURE["p113_body"]


def payload():
    return copy.deepcopy(POLICY)


def bind(value, source=SOURCE, guide=GUIDE):
    return bind_capital_gift(yaml.safe_dump(value, sort_keys=False), source, guide)


def rule(value, name):
    return next(r for r in value["rules"] if r["name"] == name)


def formula(value, name, old, new):
    item = rule(value, name)["versions"][0]
    assert old in item["formula"]
    item["formula"] = item["formula"].replace(old, new)


def test_actual_whole_model_has_complete_static_partition():
    result = bind_capital_gift(FIXTURE["policy_yaml"], SOURCE, GUIDE)
    assert isinstance(result, CapitalGiftBinding)
    assert len(result.inputs) == 42
    assert len(result.rules) == 66
    assert len(result.source.records) == 16
    assert len(result.guide_spans) == 6
    assert result.bound(FIXTURE["policy_yaml"], SOURCE, GUIDE)
    by_name = {r.name: r for r in result.rules}
    gain = by_name["federal_2025_t1170_gift_capital_gain_column_5"]
    assert gain.entity == "SourceEntry"
    assert gain.expression[0] == "match_true"
    assert "t1170_gift_within_form_scope" in gain.dependencies
    total = by_name["federal_2025_t1170_total_gain_eligible_zero_inclusion"]
    assert total.entity == "Person"
    assert total.dependencies == (
        "t1170_bonds_other_zero_inclusion_line_68250",
        "t1170_shares_units_zero_inclusion_line_68230",
    )
    assert by_name["t1170_form_year_start"].expression == (
        "date",
        2025,
        "2025-01-01",
        "t1170_form_year",
    )


def test_binds_semantics_not_arbitrary_fixture_bytes_or_order():
    value = payload()
    value["module"]["summary"] = "Different non-operative summary."
    value["rules"].reverse()
    value["inputs"].reverse()
    for item in value["inputs"]:
        del item["description"]
    for item in value["rules"]:
        item["metadata"]["proof"]["atoms"].reverse()
    formula(
        value,
        "federal_2025_t1170_gain_eligible_zero_inclusion_column_7",
        " * ",
        "  *  ",
    )
    formula(
        value,
        "t1170_ecological_land_classification_matches",
        '"land_interest"',
        "'land_interest'",
    )
    assert bind(value) is not None


@pytest.mark.parametrize("name", [i["name"] for i in POLICY["inputs"]])
@pytest.mark.parametrize(
    "mutation",
    ["missing", "duplicate", "wrong_type", "wrong_entity", "negation", "unknown_key"],
)
def test_every_factual_role_is_typed_complete_and_corroborated(name, mutation):
    value = payload()
    item = next(i for i in value["inputs"] if i["name"] == name)
    if mutation == "missing":
        value["inputs"].remove(item)
    elif mutation == "duplicate":
        value["inputs"].append(copy.deepcopy(item))
    elif mutation == "wrong_type":
        item["dtype"] = "Text" if item["dtype"] != "Text" else "Boolean"
    elif mutation == "wrong_entity":
        item["entity"] = "Person" if item["entity"] != "Person" else "SourceEntry"
    elif mutation == "negation":
        item["description"] = "Not: " + item["description"]
    else:
        item["default"] = False
    assert bind(value) is None


@pytest.mark.parametrize("name", [r["name"] for r in POLICY["rules"]])
@pytest.mark.parametrize("mutation", ["missing", "extra", "proof", "executable"])
def test_every_declaration_and_source_owner_is_required(name, mutation):
    value = payload()
    item = rule(value, name)
    if mutation == "missing":
        value["rules"].remove(item)
    elif mutation == "extra":
        extra = copy.deepcopy(item)
        extra["name"] += "_unowned"
        value["rules"].append(extra)
    elif mutation == "proof":
        item["metadata"]["proof"]["atoms"].pop(0)
    elif item["kind"] == "data_relation":
        item["data_relation"]["arguments"].reverse()
    elif "values" in item["versions"][0]:
        item["versions"][0]["values"] = {2025: "2024-12-31"}
    else:
        item["versions"][0]["formula"] = "false"
    assert bind(value) is None


@pytest.mark.parametrize(
    ("name", "old", "new"),
    [
        (
            "federal_2025_t1170_gift_capital_gain_column_5",
            "t1170_adjusted_cost_base_column_3 - t1170_outlays_and_expenses_column_4",
            "t1170_outlays_and_expenses_column_4 - t1170_adjusted_cost_base_column_3",
        ),
        (
            "federal_2025_t1170_gain_eligible_zero_inclusion_column_7",
            "t1170_eligible_amount_of_gift_column_6 / t1170_proceeds_of_disposition_column_2",
            "t1170_proceeds_of_disposition_column_2 / t1170_eligible_amount_of_gift_column_6",
        ),
        (
            "t1170_gift_within_form_scope",
            "t1170_form_year_end[t1170_form_year]",
            "period_end",
        ),
        (
            "t1170_ecological_interest_subtype_matches",
            '== "Quebec" and',
            '== "Quebec" or',
        ),
        ("t1170_ecological_interest_subtype_matches", ">=", "<="),
        (
            "t1170_ecological_donee_category_matches",
            "and t1170_ecological_donee_approved_in_respect_of_gift",
            "or t1170_ecological_donee_approved_in_respect_of_gift",
        ),
        (
            "t1170_ecological_source_conditions_met",
            "and t1170_ecological_donee_not_private_foundation",
            "",
        ),
        (
            "t1170_prior_exchange_within_zero_inclusion_scope",
            "days_between(t1170_exchange_date, t1170_disposition_date)",
            "days_between(t1170_disposition_date, t1170_exchange_date)",
        ),
        (
            "t1170_prior_exchange_gain_at_zero_inclusion",
            "t1170_prior_exchange_capital_gain",
            "federal_2025_t1170_gift_capital_gain_column_5",
        ),
        (
            "t1170_partnership_exchange_source_conditions_met",
            "t1170_partnership_exchange_right_at_issue",
            "t1170_exchange_right_at_issue",
        ),
        ("t1170_partnership_special_capital_gain_per_entry", "min(", "max("),
        ("t1170_partnership_special_capital_gain_per_entry", "max(", "min("),
        (
            "federal_2025_t1170_total_gain_eligible_zero_inclusion",
            "t1170_bonds_other_zero_inclusion_line_68250",
            "t1170_ecological_land_zero_inclusion_line_68252",
        ),
        (
            "t1170_partnership_gain_for_schedule3_line17400",
            "t1170_partnership_special_capital_gain_per_entry",
            "federal_2025_t1170_taxable_capital_gain_from_gift",
        ),
        (
            "federal_2025_t1170_shares_and_units_schedule_3_line_13200",
            "t1170_shares_units_row_for_return",
            "t1170_bonds_other_row_for_return",
        ),
        (
            "t1170_additional_sheet_required",
            "needs_t1170_additional_space",
            "completing_t1170",
        ),
        (
            "t1170_fts_gain_for_t691_part1_line25",
            "t1170_completed_schedule3_fts_gain_line68231",
            "t1170_prior_exchange_capital_gain",
        ),
    ],
)
def test_source_semantic_mutations_refuse_with_unchanged_valid_proofs(name, old, new):
    value = payload()
    formula(value, name, old, new)
    assert bind(value) is None


@pytest.mark.parametrize(
    "suffix", ["\n    false => 0", "\n    _ => 0", "\n# permissive", "\ntrue"]
)
def test_no_match_principal_does_not_gain_fallback(suffix):
    value = payload()
    rule(value, "t1170_prior_exchange_gain_at_zero_inclusion")["versions"][0][
        "formula"
    ] += suffix
    assert bind(value) is None


@pytest.mark.parametrize("bad", [True, 1, 1.0, [], {}, None, "true"])
def test_proof_required_is_exact_boolean_true(bad):
    if bad is True:
        assert bind(payload())
        return
    value = payload()
    value["module"]["proof_validation"]["required"] = bad
    assert bind(value) is None


@pytest.mark.parametrize("bad", [True, 2.0, "2", None])
def test_cad_minor_units_exact_integer(bad):
    value = payload()
    value["units"][0]["minor_units"] = bad
    assert bind(value) is None


@pytest.mark.parametrize(
    "bad",
    [
        {"2025": "2025-01-01"},
        {2025.0: "2025-01-01"},
        {True: "2025-01-01"},
        {2025: True},
        {2025: 20250101},
        {2025: "2025-01-01", 2026: "2026-01-01"},
    ],
)
def test_date_constant_has_exact_integer_key_and_date_value(bad):
    value = payload()
    rule(value, "t1170_form_year_start")["versions"][0]["values"] = bad
    assert bind(value) is None


@pytest.mark.parametrize("target", ["t1170_form_year_start", "t1170_form_year_end"])
@pytest.mark.parametrize("mutation", ["index", "period", "input_override", "alias"])
def test_date_fixed_context_not_caller_or_frame_override(target, mutation):
    value = payload()
    item = rule(value, target)
    if mutation == "index":
        item["indexed_by"] = "t1170_year_acquired_column_1"
    elif mutation == "period":
        item["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "alias":
        item["versions"][0] = {
            "formula": "t1170_disposition_date",
            "effective_from": "2025-01-01",
            "effective_to": "2025-12-31",
        }
    else:
        value["inputs"].append(
            {
                "name": "t1170_form_year",
                "entity": "SourceEntry",
                "dtype": "Integer",
                "period": "Year",
            }
        )
    assert bind(value) is None


@pytest.mark.parametrize("field", ["path", "kind", "source"])
@pytest.mark.parametrize("bad", [[], {}, None, 1, True])
def test_malformed_proof_refuses_without_exception(field, bad):
    value = payload()
    value["rules"][0]["metadata"]["proof"]["atoms"][0][field] = bad
    assert bind(value) is None


def test_bogus_or_role_swapped_proofs_and_wrong_period_anchor_refuse():
    for mutation in ("excerpt", "wrong_role", "year_only", "duplicate", "extra"):
        value = payload()
        atoms = value["rules"][0]["metadata"]["proof"]["atoms"]
        if mutation == "excerpt":
            atoms[0]["source"]["excerpt"] = "nonexistent"
        elif mutation == "wrong_role":
            atoms[0]["source"] = copy.deepcopy(
                rule(value, "t1170_fts_gain_for_t691_part1_line25")["metadata"][
                    "proof"
                ]["atoms"][0]["source"]
            )
        elif mutation == "year_only":
            atoms[1]["source"]["excerpt"] = "2025"
        elif mutation == "duplicate":
            atoms.append(copy.deepcopy(atoms[0]))
        else:
            atoms.append(
                {
                    "path": "versions[0].formula",
                    "kind": "formula",
                    "source": {"corpus_citation_path": "other", "excerpt": "0.50"},
                }
            )
        assert bind(value) is None, mutation


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("within 30 days", "within 31 days"),
        ("other qualified donee in 2025", "other qualified donee in 2024"),
        (
            "rest is subject to the inclusion rate of 50%",
            "rest is subject to the inclusion rate of 60%",
        ),
        (
            ".nruter\nruoy\nhtiw\nmrof\nsiht\nedulcnI",
            ".nruter\nruoy\nhtiw\nmrof\nsiht\nedulcnI\nDo not file",
        ),
        (
            ".3\neludehcS\nfo\n3\ntraP\nni\n00231",
            ".3\neludehcS\nfo\n3\ntraP\nni\n00351",
        ),
    ],
)
def test_source_hash_and_all_proofs_coherently_rebound_cannot_change_ownership(
    old, new
):
    assert old in SOURCE
    source = SOURCE.replace(old, new)
    value = payload()
    value["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    for item in value["rules"]:
        for atom in item["metadata"]["proof"]["atoms"]:
            if atom["source"]["corpus_citation_path"].endswith(
                "/capital-gains-on-gifts"
            ):
                atom["source"]["excerpt"] = atom["source"]["excerpt"].replace(old, new)
    assert bind(value, source) is None


def test_guide_qualification_or_citation_mutation_refuses():
    assert (
        bind(
            payload(),
            guide=GUIDE.replace("not less than 100 years", "not less than 10 years"),
        )
        is None
    )
    value = payload()
    atom = rule(value, "t1170_prior_exchange_gain_at_zero_inclusion")["metadata"][
        "proof"
    ]["atoms"][1]
    atom["source"]["corpus_citation_path"] += "/other"
    assert bind(value) is None


def test_contrary_input_description_cannot_reinterpret_preserved_semantic_name():
    value = payload()
    a = next(
        i for i in value["inputs"] if i["name"] == "t1170_prior_exchange_capital_gain"
    )
    b = next(
        i
        for i in value["inputs"]
        if i["name"] == "t1170_partnership_exchange_gain_otherwise_determined"
    )
    a["description"] = b["description"]
    assert bind(value) is None


@pytest.mark.parametrize(
    "raw",
    [
        "null",
        "[]",
        "format: rulespec/v1\nformat: rulespec/v1",
        "x: &x [*x]",
        "x: {[]: true}",
    ],
)
def test_duplicate_alias_cycle_unhashable_and_nonobject_policy_refuse(raw):
    assert bind_capital_gift(raw, SOURCE, GUIDE) is None


@pytest.mark.parametrize(
    "mutation",
    [
        "absent_required",
        "false_required",
        "empty_description",
        "float_formula",
        "nan_formula",
        "inf_formula",
        "source_hash",
        "proof_formula_reversal",
    ],
)
def test_additional_closed_contract_edges(mutation):
    value = payload()
    if mutation == "absent_required":
        value["module"].pop("proof_validation")
    elif mutation == "false_required":
        value["module"]["proof_validation"]["required"] = False
    elif mutation == "empty_description":
        value["inputs"][0]["description"] = ""
    elif mutation in {"float_formula", "nan_formula", "inf_formula"}:
        text = {
            "float_formula": "2025.0",
            "nan_formula": "nan",
            "inf_formula": "1e309",
        }[mutation]
        rule(value, "t1170_form_year")["versions"][0]["formula"] = text
    elif mutation == "source_hash":
        value["module"]["source_verification"]["source_sha256"] = "0" * 64
    else:
        a = rule(value, "t1170_reported_bond_issuer")
        b = rule(value, "t1170_reported_land_address_or_legal_description")
        a["versions"], b["versions"] = b["versions"], a["versions"]
        a["metadata"], b["metadata"] = b["metadata"], a["metadata"]
    assert bind(value) is None

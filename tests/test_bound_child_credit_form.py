"""Unsigned diagnostic static graph, not execution or source-completeness proof."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_child_credit_form import bind_child_credit_form

FIXTURE = Path(__file__).parent / "fixtures/child_credit"
CONTENT = (FIXTURE / "policy.yaml").read_text()
SOURCE = (FIXTURE / "source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/ns479-credits"


def check(payload):
    return bind_child_credit_form(
        yaml.safe_dump(payload, sort_keys=False), SOURCE, CITATION
    )


def test_complete_original_graph_and_no_execution_authority():
    bound = bind_child_credit_form(CONTENT, SOURCE, CITATION)
    assert bound is not None
    assert len(bound.rule_roles) == 21 and len(bound.input_roles) == 10
    assert bound.relation_arguments == ("SourceEntry", "Person")
    assert bound.bound(CONTENT, SOURCE, CITATION)
    assert not replace(bound, content_sha256="0" * 64).bound(CONTENT, SOURCE, CITATION)
    assert not hasattr(bound, "owns_condition")


def test_declaration_order_and_expression_spacing_do_not_define_roles():
    p = yaml.safe_load(CONTENT)
    p["rules"].reverse()
    p["inputs"].reverse()
    next(r for r in p["rules"] if r["name"] == "per_child_column_f")["versions"][0][
        "formula"
    ] = "( per_child_column_d  - claimed_by_others_column_e )"
    assert check(p) is not None


@pytest.mark.parametrize("index", range(21))
def test_every_rule_required_and_extra_rejected(index):
    p = yaml.safe_load(CONTENT)
    p["rules"].pop(index)
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    extra = deepcopy(p["rules"][index])
    extra["name"] = "extra"
    p["rules"].append(extra)
    assert check(p) is None


@pytest.mark.parametrize("index", range(10))
def test_every_input_required_and_contradictory_description_rejected(index):
    p = yaml.safe_load(CONTENT)
    p["inputs"].pop(index)
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["inputs"][index]["description"] = "Not " + p["inputs"][index]["description"]
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["inputs"][index]["description"] = p["inputs"][(index + 1) % 10]["description"]
    assert check(p) is None


@pytest.mark.parametrize(
    "name,formula",
    [
        ("eligible_expenses_column_b", "assistance_received_or_receivable_c"),
        ("assistance_column_c", "eligible_expenses_b"),
        (
            "per_child_column_d",
            "min(assistance_column_c - eligible_expenses_column_b, ns479_2025_child_sports_arts_expense_cap)",
        ),
        (
            "per_child_column_d",
            "max(0, min(eligible_expenses_column_b - assistance_column_c, ns479_2025_child_sports_arts_expense_cap))",
        ),
        ("per_child_column_f", "claimed_by_others_column_e - per_child_column_d"),
        ("per_child_column_f", "per_child_column_d - assistance_column_c"),
        (
            "ns479_children_sports_arts_expenses_line_62410",
            "sum(annual_child_row_for_claimant.per_child_column_d)",
        ),
        (
            "ns479_children_sports_arts_column_d_capped_expenses",
            "min(sum(annual_child_row_for_claimant.eligible_expenses_column_b), ns479_2025_child_sports_arts_expense_cap)",
        ),
        (
            "children_expense_table_completion_required",
            "ns479_volunteer_firefighters_ground_search_rescue_credit_eligible",
        ),
        ("additional_sheet_attachment_required", "not needs_additional_space"),
        (
            "paper_return_form_attachment_required",
            "completing_ns479 or filing_paper_return",
        ),
        (
            "conditional_credit_excess_over_taxes",
            "if nova_scotia_credits_line_47900 >= taxes_to_pay:\n  nova_scotia_credits_line_47900 - taxes_to_pay\nelse:\n  0",
        ),
    ],
)
def test_ordered_role_and_obligation_mutations(name, formula):
    p = yaml.safe_load(CONTENT)
    next(r for r in p["rules"] if r["name"] == name)["versions"][0]["formula"] = formula
    assert check(p) is None


def test_coherent_entered_reversal_preserves_proofs_but_refused():
    p = yaml.safe_load(CONTENT)
    a = next(r for r in p["rules"] if r["name"] == "eligible_expenses_column_b")
    b = next(r for r in p["rules"] if r["name"] == "assistance_column_c")
    a["versions"][0]["formula"], b["versions"][0]["formula"] = (
        b["versions"][0]["formula"],
        a["versions"][0]["formula"],
    )
    assert check(p) is None


@pytest.mark.parametrize(
    "field,value", [("required", 1), ("required", False), ("required", None)]
)
def test_strict_proof_type(field, value):
    p = yaml.safe_load(CONTENT)
    p["module"]["proof_validation"][field] = value
    assert check(p) is None


@pytest.mark.parametrize("value", [True, 2.0, "2", None])
def test_cad_minor_units_exact_integer(value):
    p = yaml.safe_load(CONTENT)
    p["units"][0]["minor_units"] = value
    assert check(p) is None


@pytest.mark.parametrize("value", [True, 0.0878, "0.0879 + 0", None])
def test_rate_exact_typed_value(value):
    p = yaml.safe_load(CONTENT)
    p["rules"][2]["versions"][0]["formula"] = value
    assert check(p) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("effective_from", True),
        ("effective_from", 2025.0),
        ("effective_from", "2025-02-01"),
        ("effective_to", None),
        ("effective_to", "2026-12-31"),
    ],
)
def test_closed_date_endpoints(field, value):
    p = yaml.safe_load(CONTENT)
    p["rules"][0]["versions"][0][field] = value
    assert check(p) is None


@pytest.mark.parametrize("field,value", [("kind", []), ("path", {}), ("source", None)])
def test_malformed_proof_refuses_without_exception(field, value):
    p = yaml.safe_load(CONTENT)
    p["rules"][0]["metadata"]["proof"]["atoms"][0][field] = value
    assert check(p) is None


def test_endpoint_year_only_and_missing_operation_proof_refused():
    p = yaml.safe_load(CONTENT)
    r = p["rules"][0]
    for a in r["metadata"]["proof"]["atoms"]:
        if a["kind"] == "effective_period":
            a["source"]["excerpt"] = "2025"
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["rules"][0]["metadata"]["proof"]["atoms"].pop(0)
    assert check(p) is None


def test_relation_direction_and_extra_semantics_refused():
    p = yaml.safe_load(CONTENT)
    p["rules"][3]["data_relation"]["arguments"].reverse()
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["rules"][3]["data_relation"]["max_cardinality"] = 6
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["inputs"][0]["default"] = False
    assert check(p) is None
    p = yaml.safe_load(CONTENT)
    p["rules"][0]["versions"][0]["override"] = True
    assert check(p) is None


def test_duplicate_yaml_and_wrong_source_refuse():
    assert bind_child_credit_form(CONTENT + "\nimports: []\n", SOURCE, CITATION) is None
    assert bind_child_credit_form(CONTENT, SOURCE + "\nDo not claim.", CITATION) is None
    p = yaml.safe_load(CONTENT)
    p["rules"].append(deepcopy(p["rules"][0]))
    assert check(p) is None


def test_bare_physical_header_is_real_endpoint_anchor():
    import hashlib

    from axiom_encode.child_credit_source import recognize_child_credit_form

    source = SOURCE.split("\n\n", 1)[1]
    form = recognize_child_credit_form(source, CITATION)
    assert form is not None
    header = source[slice(*form.records[0].span)]
    p = yaml.safe_load(CONTENT)
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    for rule in p["rules"]:
        for atom in rule["metadata"]["proof"]["atoms"]:
            if atom["kind"] == "effective_period":
                atom["source"]["excerpt"] = header
    assert bind_child_credit_form(yaml.safe_dump(p), source, CITATION) is not None


def test_optional_description_absence_not_factual_authentication():
    p = yaml.safe_load(CONTENT)
    for item in p["inputs"]:
        item.pop("description")
    assert check(p) is not None
    p["inputs"][0]["name"] = "not_eligible"
    assert check(p) is None


@pytest.mark.parametrize("index", range(21))
def test_unknown_rule_semantic_fields_refused(index):
    p = yaml.safe_load(CONTENT)
    p["rules"][index]["condition"] = "false"
    assert check(p) is None

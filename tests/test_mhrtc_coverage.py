"""Pure observation controls; fabricated views never confer source authority."""

import copy
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.harness import mhrtc_coverage as m
from axiom_encode.harness.mhrtc_program_contract import SOURCE_TEXT
from tests.test_bound_mhrtc_program import policy as native_policy  # noqa: F401
from tests.test_mhrtc_request import fixture as transport_fixture  # noqa: F401
from tests.test_mhrtc_results import (
    parameters,
    sample,  # noqa: F401
)


@pytest.fixture
def outer(transport_fixture, native_policy):  # noqa: F811
    """Real binder/request/result paths; only engine outputs are synthetic.

    These simple public objects deliberately demonstrate no private provenance.
    The observer must retain missing roles even when transport checks pass.
    """
    case, payload, pipe, policy_file = transport_fixture
    content = yaml.safe_dump(native_policy, sort_keys=False)
    policy_file.write_text(content)
    param_payload, expected = parameters()
    payload["program"]["parameters"] = param_payload["program"]["parameters"]
    for row in payload["program"]["derived"]:
        if row["name"] in (m._TOTAL, m._CREDIT):
            row.update(
                dtype="decimal",
                unit="CAD",
                period="Year",
                semantics="scalar",
                versions=[
                    {"effective_from": "2025-01-01", "effective_to": "2025-12-31"}
                ],
            )
    second = copy.deepcopy(case)
    second["name"] = "second public fixture"
    parameter = {
        "name": "static parameters",
        "entity_id": "parameter-filer",
        "period": copy.deepcopy(case["period"]),
        "input": {},
        "output": {
            key: float(value) if type(value) is Decimal else value
            for key, value in expected.items()
        },
    }
    companion = yaml.safe_dump([case, second, parameter], sort_keys=False)
    frozen = m.parse_mhrtc_companion(companion)
    receipts = []
    for ordinal, current in enumerate(frozen[:2], 1):
        plan = m.reconstruct_mhrtc_request(
            current, payload, pipe=pipe, rules_file=policy_file, case_index=ordinal
        )
        query = plan.request["queries"][0]
        outputs = {
            key: {
                "kind": "scalar",
                "name": key.removeprefix(m._PREFIX),
                "id": key,
                "dtype": "decimal",
                "unit": "CAD",
                "value": {"kind": "decimal", "value": str(value)},
            }
            for key, value in current["output"].items()
        }
        response = {
            "metadata": {
                "requested_mode": "explain",
                "actual_mode": "explain",
                "fallback_reason": None,
            },
            "results": [
                {
                    "entity_id": query["entity_id"],
                    "period": query["period"],
                    "outputs": outputs,
                    "trace": {
                        key: {**copy.deepcopy(node), "entity_id": query["entity_id"]}
                        for key, node in outputs.items()
                    },
                }
            ],
        }
        request_json, stdout = json.dumps(plan.request), json.dumps(response)
        record = SimpleNamespace(
            case_index=ordinal, request_json=request_json, response_json=stdout
        )
        receipts.append(
            SimpleNamespace(
                ordinal=ordinal,
                record=record,
                principal=SimpleNamespace(request_json=request_json, stdout=stdout),
                auxiliary=(),
            )
        )
    context = SimpleNamespace(
        content=content.encode(),
        source=SOURCE_TEXT,
        companion=companion.encode(),
        rebuilt=json.dumps(payload).encode(),
        principal_ordinals=(1, 2),
        parameter_ordinals=(3,),
    )
    return context, tuple(receipts), pipe, policy_file


def run_outer(outer):
    context, receipts, pipe, policy_file = outer
    return m.observe_mhrtc_coverage(
        context, receipts, pipe=pipe, rules_file=policy_file
    )


def test_outer_real_reconstruction_accepts_transport_not_authority(outer):
    result = run_outer(outer)
    assert set(m._UNFINISHED) <= set(result.missing)
    assert {w.role for w in result.witnesses} == {
        "parameters.source_constants",
        "rate.nonzero",
    }
    assert (
        next(
            w for w in result.witnesses if w.role == "parameters.source_constants"
        ).axis
        == "static_no_runtime"
    )


@pytest.fixture
def outer_aux(outer):
    """Nonempty reconstructed auxiliary transport, synthetic engine only."""
    context, _, pipe, policy_file = outer
    payload = json.loads(context.rebuilt)
    cases = yaml.safe_load(context.companion)
    for index, case in enumerate(cases[:2]):
        case["tables"]["Claim"][0][m._INPUT + "renovation_key"] = 101
        case["tables"]["OtherClaim"] = [
            {
                "id": "other",
                m._INPUT + "other_renovation_key": 101,
                m._INPUT + "other_claimant_key": 2,
                m._INPUT + "other_actual_claimed_expenses": 42000
                if index == 0
                else 12000,
            }
        ]
        case["relations"].append(
            {
                "name": m._PREFIX + "relation.candidate_other_claims",
                "tuple": ["other", "claim-1"],
            }
        )
        amount = 8000 if index == 0 else 38000
        case["output"] = {
            m._PREFIX + m._TOTAL: amount,
            m._PREFIX + m._CREDIT: amount * 0.145,
        }
    auxiliary_names = (
        "other_claim_source_eligibility",
        "other_selected_for_individual",
        "other_allocated_local_capacity",
    )
    for row in payload["program"]["derived"]:
        if row["name"] in auxiliary_names:
            money = row["name"] == auxiliary_names[-1]
            row.update(
                dtype="decimal" if money else "judgment",
                unit="CAD" if money else None,
                period="Year",
                semantics="scalar" if money else "judgment",
                versions=[
                    {"effective_from": "2025-01-01", "effective_to": "2025-12-31"}
                ],
            )
    context.rebuilt = json.dumps(payload).encode()
    context.companion = yaml.safe_dump(cases, sort_keys=False).encode()
    frozen = m.parse_mhrtc_companion(context.companion.decode())

    def synthetic(plan, values):
        query = plan.request["queries"][0]
        nodes = {}
        for key, value in values.items():
            node = {
                "id": key,
                "name": key.removeprefix(m._PREFIX),
                "unit": None if type(value) is str else "CAD",
            }
            if type(value) is str:
                node.update(kind="judgment", outcome=value)
            else:
                node.update(
                    kind="scalar",
                    dtype="decimal",
                    value={"kind": "decimal", "value": str(value)},
                )
            nodes[key] = node
        response = {
            "metadata": {
                "requested_mode": "explain",
                "actual_mode": "explain",
                "fallback_reason": None,
            },
            "results": [
                {
                    "entity_id": query["entity_id"],
                    "period": query["period"],
                    "outputs": nodes,
                    "trace": {
                        key: {**copy.deepcopy(node), "entity_id": query["entity_id"]}
                        for key, node in nodes.items()
                    },
                }
            ],
        }
        return SimpleNamespace(
            request_json=json.dumps(plan.request), stdout=json.dumps(response)
        )

    receipts = []
    for ordinal, case in enumerate(frozen[:2], 1):
        plan = m.reconstruct_mhrtc_request(
            case, payload, pipe=pipe, rules_file=policy_file, case_index=ordinal
        )
        principal = synthetic(plan, case["output"])
        aux = []
        for kind in ("qualification", "capacity"):
            plan = m.reconstruct_mhrtc_other_claim_request(
                case,
                payload,
                pipe=pipe,
                rules_file=policy_file,
                case_index=ordinal,
                other_claim_id="other",
                witness_kind=kind,
            )
            values = (
                {m._PREFIX + name: "holds" for name in auxiliary_names[:2]}
                if kind == "qualification"
                else {m._PREFIX + auxiliary_names[-1]: 42000}
            )
            aux.append(synthetic(plan, values))
        record = SimpleNamespace(
            case_index=ordinal,
            request_json=principal.request_json,
            response_json=principal.stdout,
        )
        receipts.append(
            SimpleNamespace(
                ordinal=ordinal,
                record=record,
                principal=principal,
                auxiliary=(("other", *aux),),
            )
        )
    return context, tuple(receipts), pipe, policy_file


def test_outer_nonempty_auxiliary_preserves_actual_target_and_trace(outer_aux):
    observed = run_outer(outer_aux)
    assert "shared.selected_positive" in roles(observed)
    assert m._UNFINISHED[6] in observed.missing


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong_target",
        "reorder_kind",
        "crosscase_qualification",
        "crosscase_capacity",
        "swap_principal",
    ],
)
def test_outer_nonempty_auxiliary_rejects_crosscase_or_missing(outer_aux, mutation):
    context, receipts, _, _ = outer_aux
    target, qualification, capacity = receipts[0].auxiliary[0]
    if mutation == "missing":
        receipts[0].auxiliary = ()
    elif mutation == "wrong_target":
        receipts[0].auxiliary = (("foreign", qualification, capacity),)
    elif mutation == "reorder_kind":
        receipts[0].auxiliary = ((target, capacity, qualification),)
    elif mutation == "crosscase_qualification":
        receipts[0].auxiliary = ((target, receipts[1].auxiliary[0][1], capacity),)
    elif mutation == "crosscase_capacity":
        receipts[0].auxiliary = ((target, qualification, receipts[1].auxiliary[0][2]),)
    else:
        receipts[0].principal = receipts[1].principal
    with pytest.raises(ValueError):
        run_outer(outer_aux)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "order",
        "duplicate",
        "list",
        "dict_record",
        "bool_ordinal",
        "wrong_index",
        "bool_index",
        "parameter_as_runtime",
        "parameter_partition",
    ],
)
def test_outer_exact_record_and_parameter_partition(outer, mutation):
    context, receipts, pipe, path = outer
    if mutation == "missing":
        receipts = receipts[:-1]
    elif mutation == "order":
        receipts = receipts[::-1]
    elif mutation == "duplicate":
        receipts = (receipts[0], receipts[0])
    elif mutation == "list":
        receipts = list(receipts)
    elif mutation == "dict_record":
        receipts = ({"ordinal": 1}, receipts[1])
    elif mutation == "bool_ordinal":
        receipts[0].ordinal = True
    elif mutation == "wrong_index":
        receipts[0].record.case_index = 2
    elif mutation == "bool_index":
        receipts[0].record.case_index = True
    elif mutation == "parameter_as_runtime":
        context.principal_ordinals = (1, 2, 3)
    else:
        context.parameter_ordinals = ()
    with pytest.raises(ValueError):
        run_outer((context, receipts, pipe, path))


@pytest.mark.parametrize(
    "mutation",
    [
        "frozen_fact",
        "record_request",
        "fresh_request",
        "record_response",
        "expected",
        "parameter_literal",
        "foreign_aux",
        "extra_aux",
        "source",
        "policy",
    ],
)
def test_outer_same_facts_and_typed_results_required(outer, mutation):
    context, receipts, _, _ = outer
    if mutation in ("frozen_fact", "expected"):
        cases = yaml.safe_load(context.companion)
        if mutation == "frozen_fact":
            cases[0]["input"][m._INPUT + "filing_claimant_key"] = 2
        else:
            cases[0]["output"][m._PREFIX + m._CREDIT] = 1
        context.companion = yaml.safe_dump(cases).encode()
    elif mutation == "record_request":
        receipts[0].record.request_json = "{}"
    elif mutation == "fresh_request":
        receipts[0].principal.request_json = "{}"
    elif mutation == "record_response":
        receipts[0].record.response_json = "{}"
    elif mutation == "parameter_literal":
        payload = json.loads(context.rebuilt)
        payload["program"]["parameters"][0]["versions"][0]["values"]["0"]["value"] = (
            50001
        )
        context.rebuilt = json.dumps(payload).encode()
    elif mutation == "foreign_aux":
        receipts[0].auxiliary = (("foreign", None, None),)
    elif mutation == "extra_aux":
        receipts[0].auxiliary = (("other", None, None), ("other", None, None))
    elif mutation == "source":
        context.source += "changed"
    else:
        context.content = b"format: rulespec/v1"
    with pytest.raises(ValueError):
        run_outer(outer)


def view():
    parent = {
        "id": "claim",
        m._INPUT + "claimant_key": 1,
        m._INPUT + "individual_key": 10,
        m._INPUT + "renovation_key": 101,
    }
    return {
        "ordinal": 9,
        "case": {
            "name": "not an authority label",
            "input": {},
            "tables": {"Claim": [parent]},
            "relations": [],
        },
        "root": "filer",
        "trace": {},
        "owned": [parent],
        "edges": set(),
        "auxiliary": [],
    }


def roles(result):
    return {w.role for w in result.witnesses}


def test_empty_or_unfinished_suite_cannot_claim_complete():
    result = m._recognize([])
    assert not result.witnesses
    assert set(m._UNFINISHED) <= set(result.missing)
    assert "expense.cutoff_rejected" in result.missing
    assert "documents.paper" in result.missing
    assert "land.below" not in result.missing
    assert "reimbursement.full" not in result.missing
    assert not hasattr(result, "covers_gate")


def test_exact_axis_pair_does_not_ignore_unrelated_fact_or_type():
    a = view()
    a["owned"][0][m._INPUT + "canada_resident_full_year"] = True
    b = copy.deepcopy(a)
    b["ordinal"] = 10
    b["owned"][0][m._INPUT + "canada_resident_full_year"] = False

    def left(v, r):
        return m._fact(r, "canada_resident_full_year") is True

    def right(v, r):
        return m._fact(r, "canada_resident_full_year") is False

    args = ([a, b], "Claim", ("canada_resident_full_year",), left, right)
    assert m._pairs(*args)
    b["owned"][0][m._INPUT + "unrelated_fact"] = 1
    assert not m._pairs(*args)
    b["owned"][0].pop(m._INPUT + "unrelated_fact")
    b["owned"][0][m._INPUT + "claimant_key"] = True
    assert not m._pairs(*args)


def test_one_eligibility_axis_cannot_clear_conjunctive_role():
    a = view()
    a["owned"][0][m._INPUT + "canada_resident_full_year"] = True
    a["trace"][("eligible_claimant", "claim")] = "holds"
    a["trace"][("claim_source_eligibility", "claim")] = "holds"
    b = copy.deepcopy(a)
    b["ordinal"] = 10
    b["owned"][0][m._INPUT + "canada_resident_full_year"] = False
    b["trace"][("eligible_claimant", "claim")] = "not_holds"
    b["trace"][("claim_source_eligibility", "claim")] = "not_holds"
    result = m._recognize([a, b])
    assert m._UNFINISHED[0] in result.missing
    assert m._UNFINISHED[0] + ":canada_resident_full_year" not in result.missing
    assert m._UNFINISHED[0] + ":secondary_kitchen" in result.missing


def test_event_pattern_cannot_mix_date_and_payer():
    parent = {m._INPUT + "completion_date": "2025-10-01"}
    expense = {
        m._INPUT + "paid_event_exists": True,
        m._INPUT + "incurred_event_exists": True,
        m._INPUT + "paid_date": "2025-06-01",
        m._INPUT + "incurred_date": "2022-12-31",
        m._INPUT + "paid_by_claimant": False,
        m._INPUT + "incurred_by_claimant": True,
    }
    assert not m._event_valid(expense, parent, "paid")
    assert not m._event_valid(expense, parent, "incurred")
    expense[m._INPUT + "incurred_date"] = "2023-01-01"
    assert m._event_valid(expense, parent, "incurred")


def test_order_equivalence_preserves_identity_keys_and_requires_real_permutation():
    a = view()
    second = copy.deepcopy(a["owned"][0])
    second.update(id="claim2", **{m._INPUT + "renovation_key": 102})
    a["owned"].append(second)
    a["trace"].update(
        {
            ("selected_for_individual", "claim"): "holds",
            ("selected_for_individual", "claim2"): "not_holds",
            (m._TOTAL, "filer"): Decimal(40000),
            (m._CREDIT, "filer"): Decimal(5800),
        }
    )
    # view's owned list and table list are distinct containers.
    a["case"]["tables"]["Claim"] = a["owned"]
    b = copy.deepcopy(a)
    b["ordinal"] = 10
    b["case"]["tables"]["Claim"].reverse()
    captured = {}
    m._order_and_foreign_refinements(
        [a, b], lambda role, axis, matches: captured.update({role: matches})
    )
    assert captured[m._UNFINISHED[5]]
    b["case"]["tables"]["Claim"][0][m._INPUT + "individual_key"] = 11
    m._order_and_foreign_refinements(
        [a, b], lambda role, axis, matches: captured.update({role: matches})
    )
    assert not captured[m._UNFINISHED[5]]


@pytest.mark.parametrize(
    "mutation",
    [
        "owned",
        "zero_totals",
        "zero_foreign",
        "ineligible",
        "unselected",
        "history",
        "owned_ineligible",
        "owned_unselected",
        "other_claim",
    ],
)
def test_foreign_subtree_cannot_mutate_owned_graph(mutation):
    a = view()
    a["case"]["input"][m._INPUT + "filing_claimant_key"] = 1
    a["trace"].update(
        {
            (m._TOTAL, "filer"): Decimal(40000),
            (m._CREDIT, "filer"): Decimal(5800),
            ("selected_total_qualifying_expenditures", "filer"): Decimal(40000),
        }
    )
    b = copy.deepcopy(a)
    b["ordinal"] = 10
    a["case"]["tables"]["Expense"] = [{"id": "owned-expense"}]
    b["case"]["tables"]["Expense"] = [{"id": "owned-expense"}]
    foreign = copy.deepcopy(b["owned"][0])
    foreign.update(id="foreign", **{m._INPUT + "claimant_key": 2})
    for row in (foreign, a["owned"][0], b["owned"][0]):
        row[m._INPUT + "completion_date"] = "2025-10-01"
        row[m._INPUT + "canada_resident_full_year"] = True
    b["trace"].update(
        {
            ("claim_source_eligibility", "claim"): "holds",
            ("selected_for_individual", "claim"): "holds",
            ("selected_qualifying_expenditures", "claim"): Decimal(40000),
        }
    )
    expense = {
        "id": "foreign-expense",
        m._INPUT + "amount_including_taxes": 40000,
        m._INPUT + "reimbursement_received_or_receivable": 0,
        m._INPUT + "expense_reasonable": True,
        m._INPUT + "expense_directly_attributable": True,
        m._INPUT + "medical_credit_claimed": False,
        m._INPUT + "home_accessibility_credit_claimed": False,
        m._INPUT + "paid_event_exists": True,
        m._INPUT + "paid_by_claimant": True,
        m._INPUT + "paid_date": "2025-06-01",
    }
    for item in (a, b):
        owned_expense = copy.deepcopy(expense)
        owned_expense["id"] = "owned-expense"
        item["case"]["tables"]["Expense"] = [owned_expense]
        item["case"]["relations"].append(
            {
                "name": m._PREFIX + "relation.receipts",
                "tuple": ["owned-expense", "claim"],
            }
        )
        item["edges"].add(("receipts", ("owned-expense", "claim")))
        item["case"]["tables"]["QualifyingIndividual"] = [
            {
                "id": "selection",
                m._INPUT + "selection_individual_key": 10,
                m._INPUT + "selected_renovation_key": 101,
            }
        ]
        item["case"]["relations"].append(
            {
                "name": m._PREFIX + "relation.claim_selection_records",
                "tuple": ["selection", "claim"],
            }
        )
        item["edges"].add(("claim_selection_records", ("selection", "claim")))
    b["case"]["relations"].append(
        {
            "name": m._PREFIX + "relation.claim_selection_records",
            "tuple": ["selection", "foreign"],
        }
    )
    b["edges"].add(("claim_selection_records", ("selection", "foreign")))
    b["case"]["tables"]["Expense"].append(expense)
    b["case"]["relations"].append(
        {
            "name": m._PREFIX + "relation.receipts",
            "tuple": ["foreign-expense", "foreign"],
        }
    )
    b["edges"].add(("receipts", ("foreign-expense", "foreign")))
    b["case"]["tables"]["Claim"].append(foreign)
    edge = {"name": m._PREFIX + "relation.claim_records", "tuple": ["foreign", "filer"]}
    b["case"]["relations"].append(edge)
    b["edges"].add(("claim_records", ("foreign", "filer")))
    captured = {}
    m._order_and_foreign_refinements(
        [a, b], lambda role, axis, matches: captured.update({role: matches})
    )
    assert captured[m._UNFINISHED[7]]
    if mutation == "owned":
        b["case"]["tables"]["Claim"][0][m._INPUT + "renovation_key"] = 999
    elif mutation == "ineligible":
        foreign[m._INPUT + "canada_resident_full_year"] = False
    elif mutation == "unselected":
        b["edges"].remove(("claim_selection_records", ("selection", "foreign")))
    elif mutation == "history":
        b["edges"].add(("prior_renovations", ("history", "foreign")))
        b["case"]["tables"]["History"] = [
            {
                "id": "history",
                m._INPUT + "historical_individual_key": 10,
                m._INPUT + "historical_renovation_key": 100,
            }
        ]
    elif mutation == "owned_ineligible":
        b["trace"][("claim_source_eligibility", "claim")] = "not_holds"
    elif mutation == "owned_unselected":
        b["trace"][("selected_for_individual", "claim")] = "not_holds"
    elif mutation == "other_claim":
        b["edges"].add(("candidate_other_claims", ("other", "foreign")))
    elif mutation == "zero_foreign":
        expense[m._INPUT + "amount_including_taxes"] = 0
    else:
        for item in (a, b):
            item["trace"][(m._TOTAL, "filer")] = Decimal(0)
            item["trace"][(m._CREDIT, "filer")] = Decimal(0)
    m._order_and_foreign_refinements(
        [a, b], lambda role, axis, matches: captured.update({role: matches})
    )
    assert not captured[m._UNFINISHED[7]]


def test_name_and_ordinal_do_not_select_semantic_role():
    a = view()
    a["trace"][("qualifying_individual", "claim")] = "holds"
    b = copy.deepcopy(a)
    b["case"]["name"] = "blocking age17"
    b["ordinal"] = 120
    assert roles(m._recognize([a])) == roles(m._recognize([b]))
    b["trace"][("qualifying_individual", "claim")] = "not_holds"
    assert roles(m._recognize([a])) != roles(m._recognize([b]))


def test_foreign_entity_trace_cannot_witness_current_claim():
    a = view()
    a["trace"][("qualifying_individual", "other-claim")] = "holds"
    assert "eligibility.qualifying_individual.holds" not in roles(m._recognize([a]))


def test_reimbursement_needs_reached_financial_nodes():
    a = view()
    a["trace"][("line_one", "claim")] = Decimal(40000)
    assert "reimbursement.partial" not in roles(m._recognize([a]))
    a["trace"][("line_two", "claim")] = Decimal(10000)
    a["trace"][("line_three", "claim")] = Decimal(30000)
    assert "reimbursement.partial" in roles(m._recognize([a]))


@pytest.mark.parametrize(
    "allocation,capacity,role",
    [
        (12000, 12000, "minister.lower_allocation"),
        (50000, 40000, "minister.capacity_below_allocation"),
    ],
)
def test_minister_allocation_requires_actual_decision_and_observed_capacity(
    allocation, capacity, role
):
    a = view()
    a["owned"][0].update(
        {
            m._INPUT + "actual_minister_decision_exists": True,
            m._INPUT + "allocated_expenditure_portion": allocation,
        }
    )
    a["trace"].update(
        {
            ("actual_allocation_applies", "claim"): "holds",
            ("line_three", "claim"): Decimal(40000),
            ("allocated_local_capacity", "claim"): Decimal(capacity),
        }
    )
    assert role in roles(m._recognize([a]))
    a["owned"][0][m._INPUT + "actual_minister_decision_exists"] = False
    assert role not in roles(m._recognize([a]))


@pytest.mark.parametrize("paper", [True, False])
def test_documents_use_factual_paper_choice_and_all_three_outputs(paper):
    a = view()
    a["case"]["input"][m._INPUT + "filing_paper_return"] = paper
    for rule, value in (
        ("attach_schedule_to_return", "holds" if paper else "not_holds"),
        ("retain_supporting_documents", "holds"),
        ("include_supporting_documents_with_return", "not_holds"),
    ):
        a["trace"][(rule, "filer")] = value
    role = "documents.paper" if paper else "documents.electronic"
    assert role in roles(m._recognize([a]))
    a["case"]["input"][m._INPUT + "filing_paper_return"] = int(paper)
    assert role not in roles(m._recognize([a]))


def test_sheet_count_is_owned_claim_indicator_not_family_scalar():
    a = view()
    a["trace"][("additional_renovation_sheet_required", "filer")] = "not_holds"
    a["trace"][("selected_renovation_sheet_count", "filer")] = 1
    assert "sheet.not_holds" not in roles(m._recognize([a]))
    a["trace"][("selected_renovation_sheet_count", "claim")] = 1
    assert "sheet.not_holds" in roles(m._recognize([a]))


def test_cutoff_reject_uses_date_filter_not_basic_receipt_false():
    a = view()
    expense = {
        "id": "expense",
        m._INPUT + "paid_date": "2022-12-31",
        m._INPUT + "incurred_date": "2022-12-31",
    }
    a["case"]["tables"]["Expense"] = [expense]
    a["edges"].add(("receipts", ("expense", "claim")))
    a["trace"].update(
        {
            ("receipt_basic_conditions", "expense"): "holds",
            ("paid_after_source_cutoff", "expense"): "not_holds",
            ("incurred_after_source_cutoff", "expense"): "not_holds",
            ("line_one", "claim"): Decimal(0),
        }
    )
    assert "expense.cutoff_rejected" in roles(m._recognize([a]))
    a["edges"].clear()
    assert "expense.cutoff_rejected" not in roles(m._recognize([a]))


def test_land_building_only_does_not_supply_threshold_witness():
    a = view()
    a["owned"][0].update(
        {
            m._INPUT + "dwelling_land_scope_key": 1,
            m._INPUT + "dwelling_land_area_hectares": Decimal(".5"),
        }
    )
    expense = {
        "id": "e",
        m._INPUT + "expense_is_land": False,
        m._INPUT + "expense_land_scope_key": 1,
    }
    a["case"]["tables"]["Expense"] = [expense]
    a["edges"].add(("receipts", ("e", "claim")))
    a["trace"][("land_extent_eligible", "claim")] = "holds"
    assert "land.at" not in roles(m._recognize([a]))
    expense[m._INPUT + "expense_is_land"] = True
    assert "land.at" in roles(m._recognize([a]))


def test_typed_trace_from_existing_result_fixture(sample):  # noqa: F811
    payload, request, response, _ = sample
    result = m._trace(payload, request, response)
    assert result[(m._CREDIT, "p")] == Decimal(7250)


@pytest.mark.parametrize("change", ["entity", "key", "unit", "dtype", "value"])
def test_trace_identity_and_wire_mutation_rejected(sample, change):  # noqa: F811
    payload, request, response, _ = sample
    trace = response["results"][0]["trace"]
    key = next(iter(trace))
    if change == "key":
        trace[key + "@entity:70"] = trace.pop(key)
    elif change == "entity":
        trace[key]["entity_id"] = "foreign"
    elif change == "unit":
        trace[key]["unit"] = "USD"
    elif change == "dtype":
        trace[key]["dtype"] = "integer"
    else:
        trace[key]["value"]["value"] = "NaN"
    with pytest.raises(ValueError):
        m._trace(payload, request, response)


def test_trace_not_evaluated_marker_is_not_an_observation(sample):  # noqa: F811
    payload, request, response, _ = sample
    response["results"][0]["trace"] = {
        "not_evaluated_dependencies": {"reason": "short_circuit"}
    }
    with pytest.raises(ValueError):
        m._trace(payload, request, response)


def test_missing_qualification_receipt_never_defaults_ineligible(monkeypatch):
    monkeypatch.setattr(m, "mhrtc_otherclaim_obligations", lambda *a, **k: ("other",))
    case = {"tables": {"OtherClaim": [{"id": "other"}]}}
    with pytest.raises(ValueError, match="qualification"):
        m._auxiliary(
            case, {}, (("other", None, None),), pipe=None, rules_file=None, ordinal=1
        )


def test_ineligible_auxiliary_never_clamps_or_requires_capacity(monkeypatch):
    monkeypatch.setattr(m, "_trace", lambda *args: {})
    monkeypatch.setattr(m, "mhrtc_otherclaim_obligations", lambda *a, **k: ("other",))
    monkeypatch.setattr(
        m,
        "reconstruct_mhrtc_other_claim_request",
        lambda *a, **k: SimpleNamespace(request={}),
    )
    monkeypatch.setattr(
        m,
        "observe_mhrtc_outputs",
        lambda *a: (
            (m._PREFIX + "other_claim_source_eligibility", "not_holds"),
            (m._PREFIX + "other_selected_for_individual", "holds"),
        ),
    )
    case = {
        "tables": {
            "OtherClaim": [
                {"id": "other", m._INPUT + "other_actual_claimed_expenses": 42000}
            ]
        }
    }
    receipt = SimpleNamespace(request_json="{}", stdout="{}")
    result = m._auxiliary(
        case, {}, (("other", receipt, None),), pipe=None, rules_file=None, ordinal=1
    )
    assert [(r.target, r.role) for r in result] == [
        ("other", "shared.source_ineligible")
    ]


def test_trace_canonical_suffix_still_requires_declared_entity_kind(sample):  # noqa: F811
    payload, request, response, _ = sample
    trace = response["results"][0]["trace"]
    key = next(iter(trace))
    node = trace.pop(key)
    node["entity_id"] = "e"
    trace[key + "@entity:65"] = node
    request["dataset"] = {"inputs": [{"entity": "Expense", "entity_id": "e"}]}
    with pytest.raises(ValueError, match="entity kind"):
        m._trace(payload, request, response)


@pytest.mark.parametrize(
    "selected,amount,limit,accepted",
    [
        (True, 42000, 42000, True),
        (True, 0, 0, True),
        (True, 42001, 42000, False),
        (True, -1, 42000, False),
        (True, 0, None, False),
        (False, 0, None, True),
        (False, 1, None, False),
        (False, -1, None, False),
        (False, 0, 42000, False),
    ],
)
def test_auxiliary_selected_claim_domain_has_no_clamp(
    monkeypatch, selected, amount, limit, accepted
):
    monkeypatch.setattr(m, "mhrtc_otherclaim_obligations", lambda *a, **k: ("other",))
    monkeypatch.setattr(
        m,
        "reconstruct_mhrtc_other_claim_request",
        lambda *a, **k: SimpleNamespace(request={}),
    )

    def observed(payload, request, response):
        if response["kind"] == "capacity":
            return ((m._PREFIX + "other_allocated_local_capacity", Decimal(limit)),)
        return (
            (m._PREFIX + "other_claim_source_eligibility", "holds"),
            (
                m._PREFIX + "other_selected_for_individual",
                "holds" if selected else "not_holds",
            ),
        )

    monkeypatch.setattr(m, "observe_mhrtc_outputs", observed)
    monkeypatch.setattr(m, "_trace", lambda *args: {})
    case = {
        "tables": {
            "OtherClaim": [
                {"id": "other", m._INPUT + "other_actual_claimed_expenses": amount}
            ]
        }
    }
    q = SimpleNamespace(request_json="{}", stdout='{"kind":"qualification"}')
    cap = (
        None
        if limit is None
        else SimpleNamespace(request_json="{}", stdout='{"kind":"capacity"}')
    )
    if accepted:
        result = m._auxiliary(
            case, {}, (("other", q, cap),), pipe=None, rules_file=None, ordinal=1
        )
        assert len(result) == 1
        assert (
            case["tables"]["OtherClaim"][0][m._INPUT + "other_actual_claimed_expenses"]
            == amount
        )
    else:
        with pytest.raises(ValueError):
            m._auxiliary(
                case, {}, (("other", q, cap),), pipe=None, rules_file=None, ordinal=1
            )


def test_eligibility_contrast_rejects_identical_off_axis_blocker():
    a = view()
    a["owned"][0][m._INPUT + "canada_resident_full_year"] = True
    a["owned"][0][m._INPUT + "dwelling_in_canada"] = False
    a["trace"][("eligible_claimant", "claim")] = "holds"
    a["trace"][("claim_source_eligibility", "claim")] = "not_holds"
    b = copy.deepcopy(a)
    b["owned"][0][m._INPUT + "canada_resident_full_year"] = False
    b["trace"][("eligible_claimant", "claim")] = "not_holds"
    result = m._recognize([a, b])
    assert m._UNFINISHED[0] + ":canada_resident_full_year" in result.missing

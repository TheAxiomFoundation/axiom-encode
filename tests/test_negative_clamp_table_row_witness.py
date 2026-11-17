"""A sole table row can witness a direct nonnegative clamp.

Entity outputs such as ``Payment`` state their facts in a ``tables`` row and
assert a row-ordered list. The clamp check must read that row, and must still
refuse every case that does not execute and assert the clamp.
"""

from __future__ import annotations

import copy
import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us/guidance/example/fee"
MODULE = "us:policies/example/fee"
SOURCE = "Qualifying fee payments are an allowable expense."
EXTRACT = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)
GROUND = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)


def _payload(entity):
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "fee_qualifies",
                "entity": entity,
                "dtype": "Boolean",
                "period": "Month",
            },
            {
                "name": "fee_amount",
                "entity": entity,
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
            },
        ],
        "rules": [
            {
                "name": "allowable_fee",
                "kind": "derived",
                "entity": entity,
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": SOURCE,
                                },
                            }
                        ]
                    }
                },
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": ("if fee_qualifies: max(0, fee_amount) else: 0"),
                    }
                ],
            },
            {
                "name": "fee_counts",
                "kind": "derived",
                "entity": entity,
                "dtype": "Boolean",
                "period": "Month",
                "versions": [
                    {"effective_from": "2026-01-01", "formula": "fee_qualifies"}
                ],
            },
        ],
    }


def _facts(qualifies, amount):
    return {
        f"{MODULE}#input.fee_qualifies": qualifies,
        f"{MODULE}#input.fee_amount": amount,
    }


def _scalar_case(name, qualifies, amount, expected):
    return {
        "name": name,
        "period": "2026-01",
        "input": _facts(qualifies, amount),
        "output": {f"{MODULE}#allowable_fee": expected},
    }


def _row_case(name, qualifies, amount, expected):
    return {
        "name": name,
        "period": "2026-01",
        "input": {},
        "tables": {"Payment": [_facts(qualifies, amount)]},
        "output": {f"{MODULE}#allowable_fee": [expected]},
    }


def _base_cases(build):
    return [
        build("qualifying", True, 250, 250),
        build("not_qualifying", False, 250, 0),
        build("zero", True, 0, 0),
    ]


def _clamp_issues(entity, cases, *, payload=None):
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload or _payload(entity), sort_keys=False),
        SOURCE,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=EXTRACT,
        extract_numeric_grounding_occurrences=GROUND,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [issue for issue in result.issues if "Direct nonnegative clamp" in issue]


def _clamp_refused(entity, cases, *, payload=None):
    return bool(_clamp_issues(entity, cases, payload=payload))


def test_scalar_case_control_witnesses_the_clamp():
    cases = _base_cases(_scalar_case)
    assert _clamp_refused("Household", cases)
    cases.append(_scalar_case("negative", True, -25, 0))
    assert not _clamp_refused("Household", cases)


def test_sole_table_row_witnesses_the_clamp():
    cases = _base_cases(_row_case)
    cases.append(_row_case("negative", True, -25, 0))
    assert not _clamp_refused("Payment", cases)


def test_sole_table_row_witnesses_the_clamp_without_an_input_key():
    negative = _row_case("negative", True, -25, 0)
    del negative["input"]
    assert not _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_table_row_repeated_in_flat_input_witnesses_the_clamp():
    negative = _row_case("negative", True, -25, 0)
    negative["input"] = copy.deepcopy(negative["tables"]["Payment"][0])
    assert not _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_table_rows_without_a_below_zero_case_are_refused():
    assert _clamp_refused("Payment", _base_cases(_row_case))


def test_zero_valued_table_row_is_refused():
    assert _clamp_refused("Payment", [_row_case("zero", True, 0, 0)])


def test_below_zero_table_row_in_the_other_branch_is_refused():
    cases = _base_cases(_row_case)
    cases.append(_row_case("negative_not_qualifying", False, -25, 0))
    assert _clamp_refused("Payment", cases)


def test_below_zero_table_row_that_does_not_assert_the_clamp_rule_is_refused():
    negative = _row_case("negative", True, -25, 0)
    negative["output"] = {f"{MODULE}#fee_counts": [True]}
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_table_row_with_a_wrong_assertion_is_refused():
    cases = _base_cases(_row_case)
    cases.append(_row_case("negative", True, -25, -25))
    assert _clamp_refused("Payment", cases)


def _two_rows(case):
    case["tables"]["Payment"].append(_facts(True, 250))
    case["output"] = {f"{MODULE}#allowable_fee": [0, 250]}


def _other_entity_row(case):
    case["tables"] = {
        "Payment": [_facts(True, 0)],
        "OtherPayment": [_facts(True, -25)],
    }


def _flat_input_differs_from_row(case):
    case["input"] = _facts(True, 0)


def _flat_input_adds_a_fact(case):
    case["input"] = {f"{MODULE}#input.fee_qualifies": True}


def _nested_expectation(case):
    case["output"] = {f"{MODULE}#allowable_fee": [[0]]}


def _row_entity_is_not_the_rule_entity(case):
    case["tables"] = {"OtherPayment": case["tables"]["Payment"]}


@pytest.mark.parametrize(
    "unbind",
    [
        _two_rows,
        _other_entity_row,
        _flat_input_differs_from_row,
        _flat_input_adds_a_fact,
        _nested_expectation,
        _row_entity_is_not_the_rule_entity,
    ],
)
def test_below_zero_table_row_without_an_exact_binding_is_refused(unbind):
    negative = _row_case("negative", True, -25, 0)
    unbind(negative)
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_first_row_of_two_with_a_one_item_assertion_is_refused():
    negative = _row_case("negative", True, -25, 0)
    negative["tables"]["Payment"].append(_facts(True, 250))
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_sole_row_with_a_two_item_assertion_is_refused():
    negative = _row_case("negative", True, -25, 0)
    negative["output"] = {f"{MODULE}#allowable_fee": [0, 0]}
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_row_asserted_with_another_entity_output_is_refused():
    payload = _payload("Payment")
    payload["rules"][1]["entity"] = "Household"
    negative = _row_case("negative", True, -25, 0)
    negative["output"][f"{MODULE}#fee_counts"] = [True]
    cases = [*_base_cases(_row_case), negative]
    assert _clamp_refused("Payment", cases, payload=payload)
    # Control: the sibling assertion binds when it has the row's entity.
    assert not _clamp_refused("Payment", cases)


def test_row_witnesses_only_the_clamp_on_its_selected_path():
    payload = _payload("Payment")
    payload["inputs"].append({**payload["inputs"][1], "name": "other_amount"})
    payload["rules"][0]["versions"][0]["formula"] = (
        "if fee_qualifies: max(0, fee_amount) else: max(0, other_amount)"
    )
    negative = _row_case("negative", True, -5, 0)
    negative["tables"]["Payment"][0][f"{MODULE}#input.other_amount"] = -5
    issues = _clamp_issues("Payment", [negative], payload=payload)
    assert len(issues) == 1
    assert "`max(0, other_amount)`" in issues[0]
    assert "`max(0, fee_amount)`" not in issues[0]


@pytest.mark.parametrize("flat_amount", [-25, "-25"])
def test_below_zero_flat_input_cannot_borrow_a_row_assertion(flat_amount):
    row_amount = 25 if flat_amount == -25 else -25
    negative = _row_case("negative", True, row_amount, 0)
    negative["input"] = _facts(True, flat_amount)
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_flat_input_cannot_borrow_a_sibling_row_assertion():
    negative = _scalar_case("negative", True, -25, 0)
    negative["tables"] = {"Payment": [_facts(True, 25)]}
    negative["output"][f"{MODULE}#fee_counts"] = [True]
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_row_identity_keys_do_not_break_a_repeated_flat_input():
    negative = _row_case("negative", True, -25, 0)
    negative["input"] = _facts(True, -25)
    negative["tables"]["Payment"][0].update(id="row-1", payment_id="payment-1")
    assert not _clamp_refused("Payment", [*_base_cases(_row_case), negative])


@pytest.mark.parametrize("flat_input", [{}, None])
def test_scalar_clamp_assertion_beside_a_row_assertion_is_refused(flat_input):
    # One list assertion makes the runner compare every output as a list, so
    # the scalar assertion of the clamped rule fails there.
    negative = _row_case("negative", True, -25, 0)
    negative["output"] = {
        f"{MODULE}#allowable_fee": 0,
        f"{MODULE}#fee_counts": [True],
    }
    if flat_input is None:
        del negative["input"]
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_row_clamp_assertion_beside_a_scalar_assertion_is_refused():
    negative = _row_case("negative", True, -25, 0)
    negative["output"][f"{MODULE}#fee_counts"] = True
    assert _clamp_refused("Payment", [*_base_cases(_row_case), negative])


def test_below_zero_flat_input_beside_another_entity_row_assertion_is_refused():
    payload = _payload("Household")
    payload["rules"][1]["entity"] = "Payment"
    negative = _scalar_case("negative", True, -25, 0)
    negative["tables"] = {"Payment": [_facts(True, 25)]}
    negative["output"][f"{MODULE}#fee_counts"] = [True]
    cases = [*_base_cases(_scalar_case), negative]
    assert _clamp_refused("Household", cases, payload=payload)


def test_below_zero_flat_input_beside_an_unknown_row_assertion_is_refused():
    negative = _scalar_case("negative", True, -25, 0)
    negative["output"]["us:policies/example/other#unknown"] = [1]
    assert _clamp_refused("Household", [*_base_cases(_scalar_case), negative])


def _no_tables(case, payload):
    del case["tables"]


def _second_row(case, payload):
    case["tables"]["Payment"].append(_facts(True, 250))


def _gate_false(case, payload):
    case["input"] = _facts(False, -25)
    case["tables"] = {"Payment": [_facts(False, -25)]}


def _rules_without_entity(case, payload):
    for rule in payload["rules"]:
        del rule["entity"]


@pytest.mark.parametrize(
    "malform",
    [_no_tables, _second_row, _gate_false, _rules_without_entity],
)
def test_scalar_clamp_assertion_beside_a_sibling_row_assertion_is_refused(malform):
    payload = _payload("Payment")
    negative = _scalar_case("negative", True, -25, 0)
    negative["tables"] = {"Payment": [_facts(True, -25)]}
    negative["output"][f"{MODULE}#fee_counts"] = [True]
    malform(negative, payload)
    cases = [*_base_cases(_row_case), negative]
    assert _clamp_refused("Payment", cases, payload=payload)

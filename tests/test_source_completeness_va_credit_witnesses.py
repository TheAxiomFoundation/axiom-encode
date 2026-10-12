"""Virginia credit witnesses use benefit arithmetic, not subdivision locators."""

from copy import deepcopy
from functools import partial

import pytest

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as vp

CITATION = "us-va/statute/58.1/58.1-339.8"
EXTRACT = partial(
    vp.extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
)

# Exact computational sentences retained by signed encode 38131544124.
B2 = (
    "For taxable years beginning on and after January 1, 2006, any individual or "
    "married individuals filing jointly, eligible for a tax credit pursuant to "
    "§ 32 of the Internal Revenue Code, may for the taxable year, in lieu of the "
    "credit authorized under subdivision 1, claim a nonrefundable credit against "
    "the tax imposed pursuant to § 58.1-320 in an amount equal to 20 percent of "
    "the credit claimed by the individual or married individuals for federal "
    "individual income taxes pursuant to § 32 of the Internal Revenue Code for "
    "the taxable year."
)
B3_2022 = (
    "For taxable years beginning on and after January 1, 2022, but before January "
    "1, 2025 any individual or married individuals filing jointly, eligible for "
    "a tax credit pursuant to § 32 of the Internal Revenue Code, may for the "
    "taxable year, in lieu of the credit authorized under subdivision 1 or 2, "
    "claim a refundable credit against the tax imposed pursuant to § 58.1-320 "
    "in an amount equal to 15 percent of the credit claimed by the individual "
    "or married individuals for federal individual income taxes pursuant to "
    "§ 32 of the Internal Revenue Code for the taxable year."
)
B3_2025 = (
    "For taxable years beginning on and after January 1, 2025 but before January "
    "1, 2027, any individual or married individuals filing jointly may, for the "
    "taxable year, in lieu of the credit authorized under subdivision 1 or 2, "
    "claim a refundable credit against the tax imposed pursuant to § 58.1-320 "
    "in an amount equal to 20 percent of the credit claimed by the individual "
    "or married individuals for federal individual income taxes pursuant to "
    "§ 32 of the Internal Revenue Code for the taxable year."
)
COMPUTATIONS = (
    pytest.param(B2, "2006-01-01", None, "2006-01-01", 0.20, 300, id="B2"),
    pytest.param(
        B3_2022,
        "2022-01-01",
        "2024-12-31",
        "2024-01-01",
        0.15,
        300,
        id="B3-2022-2024",
    ),
    pytest.param(
        B3_2025,
        "2025-01-01",
        "2026-12-31",
        "2025-01-01",
        0.20,
        400,
        id="B3-2025-2026",
    ),
)


def _fixture(clause, start, end, period, rate, result):
    formula = (
        f"min(virginia_income_tax_liability, {rate} * federal_credit_claimed)"
        if clause == B2
        else f"{rate} * federal_credit_claimed"
    )
    version = {"effective_from": start, "formula": formula}
    if end is not None:
        version["effective_to"] = end
    rule = {
        "name": "virginia_credit",
        "kind": "derived",
        "entity": "TaxUnit",
        "dtype": "Money",
        "period": "Year",
        "unit": "USD",
        "versions": [version],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": clause,
                        },
                    }
                ]
            }
        },
    }
    case = {
        "name": "credit arithmetic",
        "period": period,
        "input": {
            "federal_credit_claimed": 2000,
            "virginia_income_tax_liability": 300,
        },
        "output": {"virginia_credit": result},
    }
    return rule, case


def _witnesses(clause, rule, case):
    branch = sc.SourceStructureBranch(
        ("b", "2" if clause == B2 else "3"),
        "formula-clause",
        "Virginia credit computation",
        clause,
        0,
        len(clause),
    )
    return sc._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules={rule["name"]: rule},
        rule_names={rule["name"]},
        # Production _companion_test_issues supplies only cases asserting this
        # principal output. Expected-value checks belong to case validation.
        asserted_by_rule={
            rule["name"]: [case]
            if rule["name"] in sc._test_case_output_names(case)
            else []
        },
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=clause,
    )


@pytest.mark.parametrize("clause,start,end,period,rate,result", COMPUTATIONS)
def test_credit_amount_and_percentage_supply_formula_witness(
    clause, start, end, period, rate, result
):
    rule, case = _fixture(clause, start, end, period, rate, result)
    execution = sc._case_formula_execution(rule, case)
    assert execution is not None and execution.evaluated_value is not None
    assert sc._formula_execution_runtime_value(execution) == result
    assert _witnesses(clause, rule, case)


@pytest.mark.parametrize("clause,start,end,period,rate,result", COMPUTATIONS)
@pytest.mark.parametrize("change", ("wrong_rate", "no_assertion"))
def test_locator_cleanup_does_not_discharge_real_computation(
    clause, start, end, period, rate, result, change
):
    rule, case = _fixture(clause, start, end, period, rate, result)
    if change == "wrong_rate":
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            str(rate), "0.25"
        )
        case["output"]["virginia_credit"] = 300 if clause == B2 else 500
    else:
        case["output"] = {}
    assert not _witnesses(clause, rule, case)


@pytest.mark.parametrize("clause,start,end,period,rate,result", COMPUTATIONS)
def test_credit_witness_requires_case_in_statutory_window(
    clause, start, end, period, rate, result
):
    rule, case = _fixture(clause, start, end, period, rate, result)
    before = deepcopy(case)
    before["period"] = f"{int(start[:4]) - 1}-01-01"
    assert not _witnesses(clause, rule, before)
    if end is not None:
        after = deepcopy(case)
        after["period"] = f"{int(end[:4]) + 1}-01-01"
        assert not _witnesses(clause, rule, after)

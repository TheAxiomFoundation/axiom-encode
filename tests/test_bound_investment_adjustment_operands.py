from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)

FIXTURES = Path(__file__).parent / "fixtures"
CONTENT = (FIXTURES / "schedule-n-static.yaml").read_text()
PARENT = (FIXTURES / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (FIXTURES / "schedule-n-guide.txt").read_text()


def bind(doc):
    return bind_investment_adjustment(
        yaml.safe_dump(doc, sort_keys=False), PARENT, GUIDE
    )


def test_complete_static_graph_keeps_explicit_upstream_boundary():
    result = bind_investment_adjustment(CONTENT, PARENT, GUIDE)
    assert result is not None
    assert len(result.rule_expressions) == 19
    assert result.residuals == ()
    assert "return252_is_already_computed_supplied_fact" in result.upstream_boundaries
    assert result.bound(CONTENT, PARENT, GUIDE)


@pytest.mark.parametrize("index", range(19))
def test_each_formula_omission_or_substitution_fails(index):
    doc = yaml.safe_load(CONTENT)
    doc["rules"][index]["versions"][0]["formula"] = "0"
    assert bind(doc) is None


@pytest.mark.parametrize("index", range(21))
def test_each_input_role_type_mismatch_fails(index):
    doc = yaml.safe_load(CONTENT)
    doc["inputs"][index]["dtype"] = "Text"
    assert bind(doc) is None


@pytest.mark.parametrize("index", range(19))
def test_each_formula_proof_required(index):
    doc = yaml.safe_load(CONTENT)
    doc["rules"][index]["metadata"]["proof"]["atoms"].pop(0)
    assert bind(doc) is None


@pytest.mark.parametrize("index", range(19))
def test_each_closed_effective_period_required(index):
    doc = yaml.safe_load(CONTENT)
    doc["rules"][index]["versions"][0]["effective_to"] = "2026-12-31"
    assert bind(doc) is None


@pytest.mark.parametrize(
    "old,new",
    [
        (" == true or net_capital_loss", " == true and net_capital_loss"),
        (
            "max(0, schedule_n_investment_expenses_line_18 - schedule_n_investment_income_line_36)",
            "max(0, schedule_n_investment_income_line_36 - schedule_n_investment_expenses_line_18)",
        ),
        (
            "schedule_n_unused_adjustment_prior_years_line_70 + schedule_n_investment_expense_adjustment_line_260 + schedule_n_other_investment_expense_adjustment_line_276",
            "schedule_n_unused_adjustment_prior_years_line_70 + schedule_n_investment_expense_adjustment_line_260",
        ),
    ],
)
def test_source_arithmetic_and_presence_not_interchangeable(old, new):
    doc = yaml.safe_load(CONTENT)
    matches = 0
    for rule in doc["rules"]:
        formula = rule["versions"][0]["formula"]
        if old in formula:
            rule["versions"][0]["formula"] = formula.replace(old, new)
            matches += 1
    assert matches == 1
    assert bind(doc) is None


def test_unrecognized_source_or_guide_cannot_bind():
    assert bind_investment_adjustment(CONTENT, PARENT + "except", GUIDE) is None
    assert (
        bind_investment_adjustment(CONTENT, PARENT, GUIDE.replace("either", "both", 1))
        is None
    )


def test_no_duplicate_rule_or_input_and_no_extra_executable_fields():
    for field in ("inputs", "rules"):
        doc = yaml.safe_load(CONTENT)
        doc[field][-1] = doc[field][0]
        assert bind(doc) is None
    doc = yaml.safe_load(CONTENT)
    doc["rules"][1]["versions"][0]["otherwise"] = "true"
    assert bind(doc) is None
    doc = yaml.safe_load(CONTENT)
    doc["module"]["proof_validation"]["required"] = 1
    assert bind(doc) is None


def test_duplicate_yaml_and_false_declared_entity_refuse():
    assert (
        bind_investment_adjustment(CONTENT + "\nformat: rulespec/v1\n", PARENT, GUIDE)
        is None
    )
    doc = yaml.safe_load(CONTENT)
    doc["rules"][1]["entity"] = "Family"
    assert bind(doc) is None


@pytest.mark.parametrize("precision", [2.0, "2", True, False])
def test_currency_precision_requires_exact_integer(precision):
    doc = yaml.safe_load(CONTENT)
    doc["units"][0]["minor_units"] = precision
    assert bind(doc) is None


@pytest.mark.parametrize("index", range(21))
@pytest.mark.parametrize(
    "description", ["", "This amount excludes the named source role."]
)
def test_changed_input_description_is_not_a_trusted_role(index, description):
    doc = yaml.safe_load(CONTENT)
    doc["inputs"][index]["description"] = description
    assert bind(doc) is None


def test_presence_description_cannot_borrow_another_boolean_role():
    doc = yaml.safe_load(CONTENT)
    doc["inputs"][-1]["description"] = doc["inputs"][-2]["description"]
    assert bind(doc) is None


@pytest.mark.parametrize("row", [14, 16, 30, 34, 70])
def test_source_specific_upstream_boundary_cannot_be_erased(row):
    from axiom_encode.harness.bound_investment_adjustment_operands import (
        INPUT_ROWS,
        PREFIX,
    )

    doc = yaml.safe_load(CONTENT)
    target = next(i for i in doc["inputs"] if i["name"] == PREFIX + INPUT_ROWS[row])
    target["description"] = doc["inputs"][0]["description"]
    assert bind(doc) is None


def test_entered_copy_cannot_be_silently_capped_or_gated():
    for formula in (
        "schedule_n_maximum_allowable_return_252",
        "match schedule_n_entered_return252_within_source_upper_conditions:\n    true => schedule_n_amount_from_return_line_252_line_78",
    ):
        doc = yaml.safe_load(CONTENT)
        rule = next(
            r
            for r in doc["rules"]
            if r["name"] == "schedule_n_adjustment_used_line_252"
        )
        rule["versions"][0]["formula"] = formula
        assert bind(doc) is None


def test_upper_conditions_report_cannot_gain_unproved_nonnegative_requirement():
    doc = yaml.safe_load(CONTENT)
    rule = next(
        r
        for r in doc["rules"]
        if r["name"] == "schedule_n_entered_return252_within_source_upper_conditions"
    )
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        "if ", "if schedule_n_amount_from_return_line_252_line_78 >= 0 and ", 1
    )
    assert bind(doc) is None


def test_line14_contract_subtracts_expenses_before_applying_half_rate():
    # The guide defines an upstream supplied amount; this binder does not
    # calculate it from raw return241 or infer the three expense amounts.
    from decimal import Decimal

    source_steps = (
        "Subtract the total from the amount on line 241 of your return. "
        "• Multiply the result by 50%."
    )
    assert source_steps in GUIDE
    return241 = Decimal("1000")
    specified_expenses = sum(map(Decimal, ("100", "50", "50")))
    correct_supplied_line14 = (return241 - specified_expenses) * Decimal("0.5")
    wrong_order_line14 = return241 * Decimal("0.5") - specified_expenses
    assert correct_supplied_line14 == Decimal("400")
    assert wrong_order_line14 == Decimal("300")

    doc = yaml.safe_load(CONTENT)
    name = "schedule_n_exploration_development_expenses_line_14"
    target = next(i for i in doc["inputs"] if i["name"] == name)
    target["description"] = (
        "Already calculated line14 result under guide260: 50% of (return241 "
        "minus the sum of the three specified Quebec expense amounts); "
        "supplied result, not raw241."
    )
    assert bind(doc) is not None
    assert not any(r["name"] == name for r in doc["rules"])
    line18 = next(
        r for r in doc["rules"] if r["name"] == "schedule_n_investment_expenses_line_18"
    )
    assert name in line18["versions"][0]["formula"]
    target["description"] = (
        "Already calculated line14 result under guide260: 50% of return241 "
        "minus the three specified Quebec expense amounts; supplied result, not raw241."
    )
    assert bind(doc) is None

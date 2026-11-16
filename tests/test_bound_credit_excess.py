import pytest

from axiom_encode.harness.bound_credit_excess import parse_credit_excess


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_actual_dsl_excess_shape(separator):
    result = parse_credit_excess(f"if credit > taxes: credit - taxes{separator}else: 0")
    assert result is not None
    assert (result.credit, result.taxes) == ("credit", "taxes")


@pytest.mark.parametrize(
    "formula",
    [
        "if credit > taxes: credit - taxes # else: 0",
        "if credit >= taxes: credit - taxes else: 0",
        "if credit > taxes: taxes - credit else: 0",
        "if credit > taxes: credit - taxes else: False",
        "if credit > taxes: credit - taxes else: 1",
        "if credit > taxes: credit - taxes; unrelated else: 0",
        "if credit > taxes: credit - taxes else: 0; unrelated",
        "if credit > taxes > other: credit - taxes else: 0",
        "if credit > taxes: credit - taxes else: -0",
        "if credit > taxes: credit - taxes else: float(0)",
        "if credit > credit: credit - credit else: 0",
        "if credit > taxes: credit - other else: 0",
    ],
)
def test_only_exact_branch_topology(formula):
    assert parse_credit_excess(formula) is None

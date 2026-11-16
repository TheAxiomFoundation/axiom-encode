import pytest

from axiom_encode.harness import source_completeness as s


@pytest.mark.parametrize(
    "text",
    [
        "annuity payments taxable under paragraph 56(1)(d) minus the capital portion deducted under paragraph 60(a)",
        "amounts included under subsection 12(1) plus income from property",
        "payment taxable under paragraph 6(1)(a) minus the amount withheld",
        "income included under subsection 56(1) minus payments",
        "annuity payments taxable under paragraph 56(1)(d)\nminus the capital portion",
    ],
)
def test_referenced_monetary_operation(text):
    assert s._REFERENCED_MONETARY_ARITHMETIC_EXPRESSION.search(text)
    assert s.source_states_explicit_computation(text)


@pytest.mark.parametrize(
    "text",
    [
        "See paragraph 56(1)(d) minus the capital portion",
        "payments taxable under paragraph 56(1)(d) and the capital portion",
        "payments taxable under paragraph 56(1)(d) minus administrative processing time",
        "payments listed under paragraph 56(1)(d) minus the capital portion",
        "payments taxable under paragraph 56(1)(d); minus the capital portion",
        "payments taxable under paragraph 56(1)(d). Minus the capital portion",
        "payments taxable under paragraph 56(1)(d) minus the number of forms",
        "payments taxable under paragraph 56(1)(d) less often than payments",
        "The paragraph 56(1)(d) heading is labelled minus the capital portion",
        "payments taxable under paragraph 56(1)(d)(e)(f)(g) minus income",
        "payments taxable under paragraph 56(1)(d) except income",
        "payments taxable under paragraph 56(1)(d) minus (see the capital portion)",
    ],
)
def test_incomplete_referenced_monetary_operation(text):
    assert not s._REFERENCED_MONETARY_ARITHMETIC_EXPRESSION.search(text)


def test_annuity_without_column_operator():
    text = "annuity payments taxable under paragraph 56(1)(d) minus the capital portion deducted under paragraph 60(a)"
    assert s.source_states_explicit_computation(text)
    assert not s.source_states_explicit_computation(text.replace(" minus ", " and "))

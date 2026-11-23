import re
from pathlib import Path

import pytest

from axiom_encode.harness.bound_investment_adjustment_operands import (
    bind_investment_adjustment,
)
from axiom_encode.harness.investment_adjustment_numeric import (
    investment_adjustment_numeric_owners,
    match_investment_adjustment_numeric_owner,
)

F = Path(__file__).parent / "fixtures"
CONTENT = (F / "schedule-n-static.yaml").read_text()
PARENT = (F / "schedule-n-authenticated-body.txt").read_text()
GUIDE = (F / "schedule-n-guide.txt").read_text()
BOUND = bind_investment_adjustment(CONTENT, PARENT, GUIDE)


def match(start, end, literal):
    return match_investment_adjustment_numeric_owner(
        BOUND, CONTENT, PARENT, GUIDE, start, end, literal
    )


def test_complete_ordered_inventory_not_only_current_missing_values():
    owners = investment_adjustment_numeric_owners(BOUND, CONTENT, PARENT, GUIDE)
    assert len(owners) == 83
    assert [(o.span, o.literal) for o in owners] == [
        ((m.start(), m.end()), m.group()) for m in re.finditer(r"\d+", PARENT)
    ]
    assert len({o.span for o in owners}) == 83


def test_same_spelling_is_not_the_same_source_role():
    assert match(588, 590, "50").role == "substantive_half_rate"
    assert match(1971, 1973, "50").role == "printed_row_graph_locator"
    assert match(88, 90, "12").role == "noncomputational_identifier"
    assert match(138, 140, "12").role == "printed_row_graph_locator"
    assert [
        match(a, b, "0").owners for a, b in [(915, 916), (1851, 1852), (1918, 1919)]
    ] == [(40,), (60,), (64,)]
    assert match(2148, 2151, "252").owners == (70, 76)
    assert match(2260, 2263, "252").owners == (78,)


@pytest.mark.parametrize(
    "start,end,literal",
    [
        (587, 591, "50%"),
        (589, 590, "0"),
        (588, 591, "50%"),
        (588, 590, "50.0"),
        (78, 850, "50"),
        (True, 590, "50"),
        (588, 590, 50),
        (588, 590, "+50"),
    ],
)
def test_no_enclosing_span_normalization_or_digit_substring_credit(start, end, literal):
    assert match(start, end, literal) is None


@pytest.mark.parametrize("mutation", ["source", "formula", "guide", "unbound"])
def test_full_static_binding_is_required_even_for_noncomputational_locators(mutation):
    content, parent, guide, bound = CONTENT, PARENT, GUIDE, BOUND
    if mutation == "source":
        parent = parent.replace("multiplied by 50%", "multiplied by 55%")
    elif mutation == "formula":
        content = content.replace("0.5", "0.6")
    elif mutation == "guide":
        guide += " "
    elif mutation == "unbound":
        bound = None
    with pytest.raises(ValueError):
        investment_adjustment_numeric_owners(bound, content, parent, guide)

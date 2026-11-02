"""Properties of the lexical formula-reference collector.

Fixture repair trusts these answers to decide which facts a formula reads and
whether an output depends on a lifetime reduction, so they must not depend on
comments, quoted text or spacing, and a call must never read as a fact.
"""

import keyword

from hypothesis import given
from hypothesis import strategies as st

from axiom_encode.rulespec_formula_identifiers import (
    _CALLABLES,
    _PERIOD_SYMBOLS,
    _RESERVED,
    formula_calls_lifetime_reduction,
    formula_reference_identifiers,
)

LIFETIME_REDUCTIONS = (
    "sum_over_periods",
    "max_over_periods",
    "count_over_periods",
    "sum_top_n_over_periods",
)

fact_names = st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True).filter(
    lambda name: (
        name not in _CALLABLES
        and name not in _RESERVED
        and name not in _PERIOD_SYMBOLS
        and not keyword.iskeyword(name)
    )
)
spacing = st.sampled_from(["", " ", "  ", "\n"])
inert_text = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\n"),
    max_size=30,
)


@st.composite
def formulas(draw):
    """Small formulas mixing facts, lifetime calls and other known calls."""
    facts = draw(st.lists(fact_names, min_size=1, max_size=4))
    terms = []
    for fact in facts:
        shape = draw(st.sampled_from(["fact", "lifetime", "call"]))
        if shape == "lifetime":
            terms.append(f"{draw(st.sampled_from(LIFETIME_REDUCTIONS))}({fact})")
        elif shape == "call":
            terms.append(f"{draw(st.sampled_from(['max', 'min', 'floor']))}({fact})")
        else:
            terms.append(fact)
    return " + ".join(terms)


@given(formula=formulas(), comment=inert_text)
def test_comments_never_change_references_or_lifetime_routing(formula, comment):
    commented = f"{formula} # {comment}"
    assert formula_reference_identifiers(commented) == formula_reference_identifiers(
        formula
    )
    assert formula_calls_lifetime_reduction(
        commented
    ) is formula_calls_lifetime_reduction(formula)


@given(formula=formulas(), quoted=inert_text.filter(lambda text: '"' not in text))
def test_quoted_text_never_adds_references_or_lifetime_calls(formula, quoted):
    with_label = f'{formula} + len("{quoted}")'
    assert formula_reference_identifiers(with_label) == formula_reference_identifiers(
        formula
    )
    assert formula_calls_lifetime_reduction(
        with_label
    ) is formula_calls_lifetime_reduction(formula)


@given(
    reduction=st.sampled_from(LIFETIME_REDUCTIONS),
    fact=fact_names,
    gap=spacing,
)
def test_a_lifetime_call_is_never_a_fact_and_a_bare_name_is_never_a_call(
    reduction, fact, gap
):
    call = f"{reduction}{gap}({fact})"
    assert formula_calls_lifetime_reduction(call)
    assert formula_reference_identifiers(call) == {fact}

    bare = f"{reduction} + {fact}"
    assert not formula_calls_lifetime_reduction(bare)
    assert formula_reference_identifiers(bare) == {reduction, fact}


@given(formula=formulas())
def test_routing_matches_the_presence_of_a_lifetime_call(formula):
    assert formula_calls_lifetime_reduction(formula) is any(
        f"{reduction}(" in formula for reduction in LIFETIME_REDUCTIONS
    )

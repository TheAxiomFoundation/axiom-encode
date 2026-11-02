"""A typed factual Text intervention corroborates its exact residence predicate."""

from copy import deepcopy

import pytest

from axiom_encode.harness import source_completeness as c
from tests.test_source_owned_residence_applicability import (
    CITATION,
    EXCLUSION,
    SELECTOR,
    SOURCE,
    branch,
    fixture,
)

INPUT = "province_or_territory_of_residence_at_year_end"


def setup():
    principal, cases = fixture()
    helper = deepcopy(principal)
    helper.update(name=SELECTOR, dtype="Judgment", period="Year")
    helper["versions"][0]["formula"] = f'{INPUT} == "Quebec"'
    for case, place, value in zip(cases, ["Nova Scotia", "Quebec"], [False, True]):
        case["input"] = {INPUT: place}
        case["output"][SELECTOR] = "holds" if value else "not_holds"
    declaration = dict(name=INPUT, dtype="Text", period="Year", entity="Person")
    return principal, helper, declaration, cases


def collect(principal, helper, declaration, cases):
    rules = {"credit": principal, SELECTOR: helper}
    asserted = {name: cases for name in rules}
    witnesses = c._toggled_formula_boolean_selectors(
        rules, asserted_by_rule=asserted, formula_environment={}
    )
    return c._exception_witnesses_for_branch(
        branch(SOURCE),
        source_text=SOURCE,
        corpus_citation_path=CITATION,
        principal_rules=rules,
        principal_rule_paths={name: {()} for name in rules},
        asserted_by_rule=asserted,
        toggled_exception_selectors=witnesses,
        input_declarations={INPUT: declaration},
        formula_environment={},
        extract_numeric_occurrences=lambda _text: (),
    )


def test_actual_derived_residence_pair():
    args = setup()
    found = collect(*args)
    assert any(w.rule_name == "credit" and w.active_value and w.zeroes for w in found)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_place",
        "compound",
        "negated",
        "opaque",
        "helper_dtype",
        "helper_period",
        "input_dtype",
        "input_period",
        "input_entity",
        "helper_entity",
        "wrong_helper_assertion",
        "missing_helper_assertion",
        "wrong_principal",
        "missing_principal",
        "extra_input",
        "unknown",
        "case_mismatch",
        "wrong_formula_proof",
        "missing_date_proof",
        "bare_date",
        "superseded",
        "cross_year",
        "bypass",
        "shadow",
        "different_keys",
    ],
)
def test_reject_unowned_text_intervention(mutation):
    p, h, d, cases = setup()
    if mutation == "wrong_place":
        h["versions"][0]["formula"] = f'{INPUT} == "Ontario"'
    elif mutation == "compound":
        h["versions"][0]["formula"] += " and true"
    elif mutation == "negated":
        h["versions"][0]["formula"] = f'not ({INPUT} != "Quebec")'
    elif mutation == "opaque":
        h["versions"][0]["formula"] = "eligible"
    elif mutation == "helper_dtype":
        h["dtype"] = "Boolean"
    elif mutation == "helper_period":
        h["period"] = "Month"
    elif mutation == "input_dtype":
        d["dtype"] = "Integer"
    elif mutation == "input_period":
        d["period"] = "Month"
    elif mutation == "input_entity":
        d["entity"] = "Household"
    elif mutation == "helper_entity":
        h["entity"] = "Household"
    elif mutation == "wrong_helper_assertion":
        cases[0]["output"][SELECTOR] = "holds"
    elif mutation == "missing_helper_assertion":
        for case in cases:
            del case["output"][SELECTOR]
    elif mutation == "wrong_principal":
        cases[1]["output"]["credit"] = 10
    elif mutation == "missing_principal":
        for case in cases:
            del case["output"]["credit"]
    elif mutation == "extra_input":
        cases[0]["input"]["other"] = False
        cases[1]["input"]["other"] = True
    elif mutation == "unknown":
        cases[0]["input"][INPUT] = "Atlantis"
    elif mutation == "case_mismatch":
        cases[0]["input"][INPUT] = "nova scotia"
    elif mutation == "wrong_formula_proof":
        h["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = EXCLUSION.replace(
            "Quebec", "Ontario"
        )
    elif mutation == "missing_date_proof":
        h["metadata"]["proof"]["atoms"] = h["metadata"]["proof"]["atoms"][:1]
    elif mutation == "bare_date":
        h["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = "for 2025"
    elif mutation == "superseded":
        h["versions"].append(
            dict(
                effective_from="2025-07-01", effective_to="2025-12-31", formula="false"
            )
        )
    elif mutation == "cross_year":
        for case in cases:
            case["period"]["end"] = "2026-12-31"
    elif mutation == "bypass":
        p["versions"][0]["formula"] = "1600"
    elif mutation == "shadow":
        for case in cases:
            case["input"][SELECTOR] = case["name"] == "excluded"
    elif mutation == "different_keys":
        cases[0]["output"]["extra"] = 0
    assert not any(w.rule_name == "credit" for w in collect(p, h, d, cases))


def test_reversed_equality_operands():
    p, h, d, cases = setup()
    h["versions"][0]["formula"] = f'"Quebec" == {INPUT}'
    assert any(w.rule_name == "credit" for w in collect(p, h, d, cases))


@pytest.mark.parametrize(
    "name",
    ["province_of_birth", "favorite_province", "province_or_territory_of_residence"],
)
def test_unrelated_or_unqualified_text_fact_cannot_borrow_helper_identity(
    name, monkeypatch
):
    monkeypatch.setitem(setup.__globals__, "INPUT", name)
    assert not any(w.rule_name == "credit" for w in collect(*setup()))


@pytest.mark.parametrize(
    "description",
    [
        "Province of birth",
        "Favorite province",
        "Province or territory of residence",
        "Province of residence at year end unless disabled",
        42,
    ],
)
def test_conflicting_declared_description_rejected(description):
    p, h, d, cases = setup()
    d["description"] = description
    assert not any(w.rule_name == "credit" for w in collect(p, h, d, cases))


def test_matching_declared_residence_description():
    p, h, d, cases = setup()
    d["description"] = "Province or territory of residence at year end."
    assert any(w.rule_name == "credit" for w in collect(p, h, d, cases))


@pytest.mark.parametrize("prefix", ["non_", "not_"])
def test_positive_equality_cannot_borrow_negative_selector_polarity(
    prefix, monkeypatch
):
    monkeypatch.setitem(setup.__globals__, "SELECTOR", prefix + SELECTOR)
    p, h, d, cases = setup()
    p["versions"][0]["formula"] = f"if {SELECTOR}: 1600 else: 0"
    cases[0]["output"]["credit"] = 0
    cases[1]["output"]["credit"] = 1600
    assert not any(w.rule_name == "credit" for w in collect(p, h, d, cases))

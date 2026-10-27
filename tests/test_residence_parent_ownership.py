"""Factual residence definition owns its parent, not a monetary child gate."""

from copy import deepcopy
from dataclasses import replace

import pytest

from axiom_encode.harness import source_completeness as c
from tests.test_derived_residence_interventions import INPUT, setup
from tests.test_source_owned_residence_applicability import CITATION, SOURCE

PARENT = "If you were a resident of Ontario at the end of the year, follow the instructions that apply to your situation:"
NAME = "resident_of_ontario_at_year_end"


def fixture():
    _, helper, declaration, _ = setup()
    helper["name"] = NAME
    helper["versions"][0]["formula"] = f'{INPUT} == "Ontario"'
    helper["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = PARENT
    return helper, declaration


def resolve(helper, declaration, source=SOURCE, excerpt=PARENT):
    return c._source_owned_residence_parent_clause(
        excerpt,
        rule=helper,
        version_index=0,
        source_text=source,
        branches=c.recognize_source_structure(source),
        corpus_citation_path=CITATION,
        input_declarations={INPUT: declaration},
    )


def issues(helper, declaration, source=SOURCE, extra=None):
    rules = {helper["name"]: helper, **(extra or {})}
    return c._opaque_same_source_condition_input_issues(
        {"rules": list(rules.values()), "inputs": [declaration]},
        source_text=source,
        branches=c.recognize_source_structure(source),
        principal_rules=rules,
        corpus_citation_path=CITATION,
    )


def test_actual_parent_fact_does_not_acquire_child_threshold():
    h, d = fixture()
    result = resolve(h, d)
    assert result is not None
    assert result.text == PARENT
    assert SOURCE[result.start : result.end] == PARENT
    assert "$200" not in result.text
    assert issues(h, d) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "money",
        "wrong_name",
        "compound",
        "negation",
        "wrong_literal",
        "wrong_fact",
        "wrong_year",
        "wrong_date_proof",
        "wrong_formula_proof",
        "missing_child",
        "wrong_child_layout",
        "duplicate",
        "quoted",
        "unclosed_quote",
        "preposed",
        "trailing",
        "wrong_referral",
        "foreign_header",
        "unknown_header",
    ],
)
def test_unknown_or_restrictive_parent_not_narrowed(mutation):
    h, d = fixture()
    source = SOURCE
    excerpt = PARENT
    if mutation == "money":
        h["dtype"] = "Money"
    elif mutation == "wrong_name":
        h["name"] = "eligible_ontario"
    elif mutation == "compound":
        h["versions"][0]["formula"] += " and minimum_tax_is_payable"
    elif mutation == "negation":
        h["versions"][0]["formula"] = f'not ({INPUT} != "Ontario")'
    elif mutation == "wrong_literal":
        h["versions"][0]["formula"] = f'{INPUT} == "Quebec"'
    elif mutation == "wrong_fact":
        d["description"] = "Province of birth"
    elif mutation == "wrong_year":
        h["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "wrong_date_proof":
        h["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = "for 2025"
    elif mutation == "wrong_formula_proof":
        h["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "Ontario"
    elif mutation == "missing_child":
        source = source.replace(PARENT + "\n– If", PARENT + "\nDo calculate")
    elif mutation == "wrong_child_layout":
        source = source.replace(PARENT + "\n– If", PARENT + "\nUnknown If")
    elif mutation == "duplicate":
        source += "\n" + PARENT
    elif mutation == "quoted":
        source = source.replace(PARENT, '"' + PARENT + '"')
    elif mutation == "unclosed_quote":
        source = '"\n' + source
    elif mutation in {"preposed", "trailing", "wrong_referral"}:
        excerpt = {
            "preposed": PARENT.replace(
                "If you were", "If you pay minimum tax and were"
            ),
            "trailing": PARENT[:-1] + " unless disabled:",
            "wrong_referral": PARENT.replace(
                "follow the instructions that apply to your situation",
                "claim the benefit",
            ),
        }[mutation]
        source = source.replace(PARENT, excerpt)
        h["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = excerpt
    elif mutation == "foreign_header":
        source = "Form T9999 instructions\n" + source
    elif mutation == "unknown_header":
        source = source.replace(
            "Protected B when completed", "Another form restrictions"
        )
    assert resolve(h, d, source, excerpt) is None


def test_money_dispatcher_and_dead_helper_do_not_borrow_factual_clearance():
    h, d = fixture()
    bad = deepcopy(h)
    bad.update(name="credit", dtype="Money")
    bad["versions"][0]["formula"] = "if minimum_tax_is_payable: 100 else: 0"
    assert resolve(bad, d) is None
    found = issues(h, d, extra={"credit": bad})
    assert found and "`credit`" in found[0]
    assert NAME not in found[0]


def test_wrong_threshold_monetary_rule_retains_full_child_obligation():
    h, d = fixture()
    bad = deepcopy(h)
    bad.update(name="credit", dtype="Money")
    bad["versions"][0]["formula"] = "if amount <= 201: 100 else: 0"
    assert resolve(bad, d) is None
    clauses, ambiguous = c._source_condition_clauses_owned_by_excerpt(
        PARENT,
        rule=bad,
        source_text=SOURCE,
        branches=c.recognize_source_structure(SOURCE),
        corpus_citation_path=CITATION,
    )
    assert not ambiguous and any("$200 or less" in x.text for x in clauses)
    assert issues(h, d, extra={"credit": bad})


def test_child_bullet_outside_owned_note_cannot_corroborate_parent():
    h, d = fixture()
    branches = c.recognize_source_structure(SOURCE)
    parent_end = SOURCE.index(PARENT) + len(PARENT)
    shortened = tuple(
        replace(b, end=parent_end) if b.path == ("1",) else b for b in branches
    )
    assert (
        c._source_owned_residence_parent_clause(
            PARENT,
            rule=h,
            version_index=0,
            source_text=SOURCE,
            branches=shortened,
            corpus_citation_path=CITATION,
            input_declarations={INPUT: d},
        )
        is None
    )

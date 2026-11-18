"""``income test`` names a test; it is not an income fact for gates-collapsed.

The gate matcher refuses every source gate that holds an income fact of
unknown polarity, because no input name can corroborate "income exists" and
"income is absent" at once. The words ``income test`` (``the gross income
test``, ``the net income tests``) state neither: they name a test that a unit
passes. Read as an income fact, such a gate could match no input, so no
encoding of the sentence could pass.

The exemption is the exact phrase ``income test`` or ``income tests`` in
normalized clause text. All source text here is synthetic.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_gate_predicate_polarities,
    _source_gate_semantic_tokens,
    analyze_complete_source_unit,
)

CITATION = "us-zz/manual/abc/benefit-determination/block-1"
CLAUSE = (
    "When the unit passes the gross income test, and the net income test the "
    "agency performs the final benefit test."
)
GROSS = "unit_passes_gross_income_test"
NET = "unit_passes_net_income_test"


def _candidate(clause: str, names: tuple[str, ...], formula: str | None) -> str:
    document = {
        "format": "rulespec/v1",
        "module": {
            "proof_validation": {"required": True},
            "source_verification": {"corpus_citation_path": CITATION},
            "summary": clause,
        },
        "inputs": [
            {"name": name, "entity": "Household", "dtype": "Boolean", "period": "Month"}
            for name in names
        ],
        "rules": [
            {
                "name": "final_benefit_test_triggered",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Judgment",
                "period": "Month",
                "source": "Manual block 1",
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "path": "versions[0].formula",
                                "kind": "condition",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": clause,
                                },
                            }
                        ]
                    }
                },
                "versions": [
                    {
                        "effective_from": "2020-01-01",
                        "formula": formula or " and ".join(names),
                    }
                ],
            }
        ],
    }
    return yaml.safe_dump(document, sort_keys=False, width=10**6)


def _companion_tests(names: tuple[str, ...]) -> list[dict]:
    cases = []
    for index, false_name in enumerate((None, *names)):
        cases.append(
            {
                "name": f"case_{index}",
                "period": "2024-01",
                "input": {name: name != false_name for name in names},
                "output": {
                    "final_benefit_test_triggered": (
                        "holds" if false_name is None else "not_holds"
                    )
                },
            }
        )
    return cases


def _collapsed(
    clause: str = CLAUSE,
    names: tuple[str, ...] = (GROSS, NET),
    formula: str | None = None,
):
    profile = vp._numeric_profile_for_citation_path(CITATION)
    analysis = analyze_complete_source_unit(
        _candidate(clause, names, formula),
        clause,
        corpus_citation_path=CITATION,
        test_cases=_companion_tests(names),
        extract_numeric_occurrences=functools.partial(
            vp.extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            vp.extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=vp.extract_named_scalar_occurrences,
        numeric_value_is_grounded=vp.numeric_value_is_grounded,
    )
    return [
        issue
        for issue in analysis.issues
        if getattr(issue, "check", None)
        == CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED.id
    ]


# --- The accepted forms -----------------------------------------------------


def test_a_candidate_with_one_input_for_each_named_test_passes():
    assert _collapsed() == []


@pytest.mark.parametrize(
    "clause",
    [
        CLAUSE.replace("income test", "income tests"),
        CLAUSE.replace("gross income test", "Gross Income Test"),
        CLAUSE.replace("net income test", "NET INCOME   TEST"),
        CLAUSE.replace("gross income test", "gross income tests"),
    ],
)
def test_plural_case_and_spacing_variants_pass(clause):
    assert _collapsed(clause) == []


def test_two_canonical_imports_named_for_the_tests_pass():
    names = ("gross_income_test", "net_income_test")
    assert _collapsed(names=names) == []


# --- The test condition itself is still demanded ----------------------------


@pytest.mark.parametrize(
    ("names", "formula"),
    [
        # One aggregate input for both tests.
        (("unit_passes_income_tests",), None),
        # One of the two tests is ignored.
        ((GROSS,), None),
        ((GROSS, NET), GROSS),
        # Two inputs that do not name the tests.
        (("application_is_complete", "interview_is_done"), None),
        (("unit_passes_test_one", "unit_passes_test_two"), None),
        # A name must carry the word "income" of the test that it stands for.
        (("unit_passes_gross_test", "unit_passes_net_test"), None),
        ((GROSS, "unit_passes_net_test"), None),
        # An income fact or an income amount is not the test.
        (("unit_has_gross_income", "unit_has_net_income"), None),
        (("gross_income", "net_income"), None),
        # The consequence of the clause is not one of its two conditions.
        ((GROSS, "agency_performs_final_benefit_test"), None),
    ],
)
def test_a_candidate_that_does_not_encode_both_tests_stays_refused(names, formula):
    assert len(_collapsed(names=names, formula=formula)) == 1


@pytest.mark.parametrize(
    "names",
    [
        ("unit_fails_gross_income_test", "unit_fails_net_income_test"),
        ("no_gross_income_test", "no_net_income_test"),
        ("unit_does_not_pass_gross_income_test", NET),
    ],
)
def test_an_input_name_with_a_negation_before_the_test_stays_refused(names):
    assert len(_collapsed(names=names)) == 1


# --- Other wording keeps the income-fact reading ----------------------------


@pytest.mark.parametrize(
    ("phrase", "names"),
    [
        ("income-test", (GROSS, NET)),
        ("income testing", (GROSS, NET)),
        ("income tested", (GROSS, NET)),
        ("income testable", (GROSS, NET)),
        ("income test's", (GROSS, NET)),
        ("income, test", (GROSS, NET)),
        ("income (test)", (GROSS, NET)),
        (
            "income limit",
            ("unit_meets_gross_income_limit", "unit_meets_net_income_limit"),
        ),
        (
            "income standard",
            ("unit_meets_gross_income_standard", "unit_meets_net_income_standard"),
        ),
        (
            "earnings test",
            ("unit_passes_gross_earnings_test", "unit_passes_net_earnings_test"),
        ),
    ],
)
def test_other_phrases_after_income_stay_refused(phrase, names):
    clause = CLAUSE.replace("income test", phrase)
    assert clause != CLAUSE
    assert len(_collapsed(clause, names)) == 1


@pytest.mark.parametrize(
    ("clause", "names"),
    [
        (
            "When the gross income is verified, and the net income is verified the "
            "agency performs the final benefit test.",
            ("gross_income_is_verified", "net_income_is_verified"),
        ),
        (
            "When the unit states the gross income of the unit, and the net income "
            "of the unit the agency performs the final benefit test.",
            ("unit_states_gross_income_of_unit", "unit_states_net_income_of_unit"),
        ),
        (
            "When the gross income exceeds the standard, and the net income exceeds "
            "the standard the agency performs the final benefit test.",
            ("gross_income_exceeds_standard", "net_income_exceeds_standard"),
        ),
        (
            "When the unit has a gross countable income, and a net countable income "
            "the agency performs the final benefit test.",
            ("unit_gross_countable_income", "unit_net_countable_income"),
        ),
    ],
)
def test_income_followed_by_a_fact_word_stays_refused(clause, names):
    assert len(_collapsed(clause, names)) == 1


def test_a_negated_test_in_the_source_stays_refused():
    clause = CLAUSE.replace("passes", "does not pass")
    assert len(_collapsed(clause)) == 1


def test_a_line_break_inside_the_phrase_stays_refused():
    # The gate splitter reads each source line by itself, so the first line
    # ends at the word "income" and keeps the income-fact reading.
    clause = CLAUSE.replace("gross income test", "gross income\ntest")
    assert len(_collapsed(clause)) == 1


# --- A real income condition in the same sentence ---------------------------

WITH_INCOME = (
    "When the unit passes the gross income test, and has income below the "
    "standard the agency performs the final benefit test."
)


def test_a_real_income_condition_keeps_its_own_gate():
    # Two inputs named for tests do not cover the income condition.
    assert len(_collapsed(WITH_INCOME)) == 1
    assert len(_collapsed(WITH_INCOME, (GROSS,))) == 1
    assert len(_collapsed(WITH_INCOME, (GROSS, "interview_is_done"))) == 1
    # The income condition needs an input that states the comparison.
    assert _collapsed(WITH_INCOME, (GROSS, "unit_has_income_below_standard")) == []


def test_an_income_fact_in_the_same_condition_as_a_named_test_stays_refused():
    clause = CLAUSE.replace(
        "gross income test", "gross income test with countable income"
    )
    assert len(_collapsed(clause)) == 1
    clause = CLAUSE.replace("gross income test", "income and the gross income test")
    assert len(_collapsed(clause)) == 1


# --- The reading itself -----------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "passes the gross income test",
        "passes the net income tests",
        "the Income Test is met",
        "meets the income  test",
        "na_gross_income_test_passes",
        "(income test)",
        "income test, and",
    ],
)
def test_the_exact_phrase_has_no_income_polarity(text):
    assert _source_gate_predicate_polarities(text) == {}
    tokens = _source_gate_semantic_tokens(text.replace("_", " "))
    assert "income" in tokens
    assert "unknown_income" not in tokens
    assert "no_income" not in tokens


@pytest.mark.parametrize(
    "text",
    [
        "passes the gross income-test",
        "gross income testing",
        "income tested program",
        "income testers",
        "income test's result",
        "income test-related",
        "income, test",
        "income (test)",
        "income limit",
        "income is",
        "income of the unit",
        "income exceeds the limit",
        "countable income",
        "passes the earnings test",
        "non-income test",
        "does not pass the gross income test",
        "never passes the income test",
        "passes the income test and reports income",
        "income test documentation",
        "tax income test",
        "it is not true that the unit passes the income test yet income",
    ],
)
def test_every_other_form_keeps_the_unknown_income_reading(text):
    assert _source_gate_predicate_polarities(text) == {"income": None}
    assert "unknown_income" in _source_gate_semantic_tokens(text)


@pytest.mark.parametrize(
    ("text", "negative"),
    [
        ("has no income test", True),
        ("has zero income tests", True),
        ("without income test", True),
        ("has income test", False),
        ("receives gross income test", False),
    ],
)
def test_an_explicit_income_fact_reading_wins_over_the_phrase(text, negative):
    assert _source_gate_predicate_polarities(text) == {"income": negative}


def test_each_income_word_is_read_by_itself():
    text = "passes the gross income test with income below the standard"
    assert _source_gate_predicate_polarities(text) == {"income": None}
    text = "passes the income test without income"
    assert _source_gate_predicate_polarities(text) == {"income": True}
    text = "it is not true that the unit passes the income test"
    assert _source_gate_predicate_polarities(text) == {}
    text = "not the case that the unit has income below the income test"
    assert _source_gate_predicate_polarities(text) == {"income": True}

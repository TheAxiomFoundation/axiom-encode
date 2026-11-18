"""A positive statement that a unit passes a named income test, in gates-collapsed.

The gate matcher refuses every source gate that holds an income fact of
unknown polarity. The words ``income test`` (``the gross income test``) get
that reading, although they name a test and state nothing about income.

This file pins the one exception. A gate gets an income-test marker only when
its whole text is a closed, positive form: ``the <subject> passes the [<0-2
words>] income test`` as the first words of a plain ``if`` / ``when``
condition, or a second verb phrase or object joined to it by ``and``. An input
gets the marker only when its whole name (and description, if any) is such a
form. A marked gate is matched only by a marked input that names the test with
the same words. Every other text keeps the reading of ``main``: the gate can
match no input.

All source text here is synthetic.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    _SOURCE_GATE_INCOME_TEST_BLOCKED_SLOT_WORDS,
    _SOURCE_GATE_INCOME_TEST_CONTINUATION,
    _SOURCE_GATE_INCOME_TEST_STATEMENT,
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_conjunctive_fact_gates,
    _source_gate_income_test_input_markers,
    _source_gate_income_test_markers,
    _source_gate_predicate_polarities,
    _source_gate_semantic_tokens,
    _terminal_gate_evidence,
    analyze_complete_source_unit,
)

CITATION = "us-zz/manual/abc/benefit-determination/block-1"
LEAD = "The agency performs the final benefit test"
CLAUSE = f"{LEAD} when the unit passes the gross income test and the net income test."
GROSS = "unit_passes_gross_income_test"
NET = "unit_passes_net_income_test"
RESIDENT = "unit_is_resident"
GROSS_MARKERS = frozenset({"income_test_name:gross income test"})
NET_MARKERS = frozenset({"income_test_name:net income test"})


def _candidate(
    clause: str,
    names: tuple[str, ...],
    formula: str | None,
    descriptions: dict[str, str],
) -> str:
    document = {
        "format": "rulespec/v1",
        "module": {
            "proof_validation": {"required": True},
            "source_verification": {"corpus_citation_path": CITATION},
            "summary": clause,
        },
        "inputs": [
            {
                "name": name,
                "entity": "Household",
                "dtype": "Boolean",
                "period": "Month",
                **({"description": descriptions[name]} if name in descriptions else {}),
            }
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
    return [
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
        for index, false_name in enumerate((None, *names))
    ]


def _collapsed(
    clause: str = CLAUSE,
    names: tuple[str, ...] = (GROSS, NET),
    formula: str | None = None,
    descriptions: dict[str, str] | None = None,
):
    profile = vp._numeric_profile_for_citation_path(CITATION)
    analysis = analyze_complete_source_unit(
        _candidate(clause, names, formula, descriptions or {}),
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


def _refused(*args, **kwargs) -> bool:
    return len(_collapsed(*args, **kwargs)) == 1


def _statement(text: str):
    return _source_gate_income_test_markers(text, (_SOURCE_GATE_INCOME_TEST_STATEMENT,))


# --- The accepted forms -----------------------------------------------------


def test_one_input_for_each_named_test_passes():
    assert _collapsed() == []


@pytest.mark.parametrize(
    "names",
    [
        ("gross_income_test_passes", "net_income_test_passes"),
        ("gross_income_test_passed", "net_income_test_is_met"),
        ("household_meets_gross_income_test", "case_passes_net_income_test"),
        ("passes_gross_income_test", "satisfies_the_net_income_test"),
        ("unit_gross_income_test_passed", "the_household_net_income_test_met"),
    ],
)
def test_other_positive_input_forms_pass(names):
    assert _collapsed(names=names) == []


@pytest.mark.parametrize(
    "clause",
    [
        CLAUSE.replace("gross income test", "Gross Income Test"),
        CLAUSE.replace("net income test", "NET INCOME   TEST"),
        CLAUSE.replace("when the unit", "if the unit"),
        CLAUSE.replace("and the net", "and passes the net"),
        CLAUSE.replace("and the net", "and the unit passes the net"),
        CLAUSE.replace("and the net", "and meets the net"),
        CLAUSE[:-1] + " and the unit is a resident.",
    ],
)
def test_variants_of_the_positive_statement_pass(clause):
    assert clause != CLAUSE
    names = (GROSS, NET, RESIDENT) if "resident" in clause else (GROSS, NET)
    assert _collapsed(clause, names) == []


def test_an_input_names_the_test_with_the_words_of_the_source():
    program = CLAUSE.replace("the gross", "the ABC gross").replace(
        "the net", "its ABC net"
    )
    named = ("abc_gross_income_test_passes", "unit_passes_abc_net_income_test")
    assert _collapsed(program, named) == []
    assert _refused(program)
    assert _refused(names=named)
    assert _refused(program, ("xyz_gross_income_test_passes", named[1]))
    assert _refused(program, ("abc_other_gross_income_test_passes", named[1]))
    plural = CLAUSE.replace("income test", "income tests")
    several = ("unit_passes_gross_income_tests", "net_income_tests_are_met")
    assert _collapsed(plural, several) == []
    assert _refused(plural)
    assert _refused(names=several)


def test_a_named_test_next_to_another_fact_passes_with_both_inputs():
    clause = f"{LEAD} when the unit passes the income test and the unit is a resident."
    assert _collapsed(clause, ("unit_passes_income_test", RESIDENT)) == []


@pytest.mark.parametrize(
    "verb",
    ["passes", "pass", "passed", "meets", "meet", "met"]
    + ["satisfies", "satisfy", "satisfied"],
)
def test_each_allowed_verb(verb):
    assert _collapsed(CLAUSE.replace("passes", verb)) == []
    assert _statement(f"the unit {verb} the gross income test") == GROSS_MARKERS
    assert (
        _source_gate_income_test_input_markers(f"unit_{verb}_gross_income_test", "")
        == GROSS_MARKERS
    )


@pytest.mark.parametrize(
    "head",
    ["applicant", "applicants", "case", "cases", "claimant", "claimants"]
    + ["families", "family", "group", "groups", "household", "households"]
    + ["individual", "individuals", "person", "persons", "taxpayer", "taxpayers"]
    + ["unit", "units"],
)
def test_each_allowed_subject_noun(head):
    assert _collapsed(CLAUSE.replace("the unit", f"the {head}")) == []
    assert _statement(f"the abc budgetary {head} passes the gross income test")
    assert (
        _source_gate_income_test_input_markers(f"{head}_passes_gross_income_test", "")
        == GROSS_MARKERS
    )


@pytest.mark.parametrize("determiner", ["the", "a", "an", "each", "every"])
def test_each_allowed_subject_determiner(determiner):
    assert _collapsed(CLAUSE.replace("the unit", f"{determiner} unit")) == []


@pytest.mark.parametrize("determiner", ["the", "a", "an", "its", "their"])
def test_each_allowed_test_determiner(determiner):
    assert _collapsed(CLAUSE.replace("the gross", f"{determiner} gross")) == []
    assert _collapsed(CLAUSE.replace("the net", f"{determiner} net")) == []


@pytest.mark.parametrize("result", ["passes", "pass", "passed", "met", "satisfied"])
@pytest.mark.parametrize("auxiliary", ["", "is_", "are_", "was_", "were_"])
def test_each_allowed_result_word_of_an_input_name(result, auxiliary):
    name = f"gross_income_test_{auxiliary}{result}"
    assert _source_gate_income_test_input_markers(name, "") == GROSS_MARKERS
    assert _collapsed(names=(name, NET)) == []


@pytest.mark.parametrize(
    "description",
    [
        "The unit passes the gross income test.",
        "Whether the unit passes the gross income test",
        "True if the household meets the gross income test.",
        "True when the gross income test is met",
        "Gross income test passed",
    ],
)
def test_a_description_that_states_the_same_test_passes(description):
    assert _collapsed(descriptions={GROSS: description}) == []


# --- What a candidate must still encode -------------------------------------


@pytest.mark.parametrize(
    ("names", "formula"),
    [
        (("unit_passes_income_tests",), None),
        ((GROSS,), None),
        ((GROSS, NET), GROSS),
        (("application_is_complete", "interview_is_done"), None),
        (("unit_passes_test_one", "unit_passes_test_two"), None),
        (("unit_passes_gross_test", "unit_passes_net_test"), None),
        ((GROSS, "unit_passes_net_test"), None),
        ((GROSS, "agency_performs_final_benefit_test"), None),
        # Each gate needs an input that names its OWN test.
        (("income_test_one", "income_test_two"), None),
        (("gross_income_test", "gross_income_test_2"), None),
        (("gross_income_test", "net_income_test"), None),
        ((GROSS, "final_income_test"), None),
        ((GROSS, "unit_passes_final_income_test"), None),
        ((GROSS, "household_passes_gross_income_test"), None),
        ((GROSS, "unit_passes_income_test"), None),
        (("unit_passes_income_test", "unit_meets_income_test"), None),
        # An input that names another test, period, program or subject's test.
        (("na_gross_income_test_passes", "na_net_income_test_passes"), None),
        (("spouse_gross_income_test_passed", NET), None),
        (("prior_gross_income_test_passed", NET), None),
        (("unit_passes_snap_gross_income_test", NET), None),
        (("unit_passes_gross_monthly_income_test", NET), None),
        (("agency_gross_income_test_passes", NET), None),
        # An income fact or an income amount is not the test.
        (("unit_has_gross_income", "unit_has_net_income"), None),
        (("unit_has_gross_income", NET), None),
        (("gross_income", "net_income"), None),
    ],
)
def test_a_candidate_that_does_not_name_both_tests_stays_refused(names, formula):
    assert _refused(names=names, formula=formula)


def test_a_gate_without_a_modifier_needs_an_input_without_one():
    clause = f"{LEAD} when the unit passes the income test and the unit is a resident."
    assert _refused(clause, (GROSS, RESIDENT))
    assert _refused(clause, ("unit_has_income", RESIDENT))
    assert _refused(clause, ("unit_passes_income_test",))
    assert _refused(clause, ("unit_passes_income_test", "interview_is_done"))


@pytest.mark.parametrize(
    "name",
    [
        "gross_income_test_failed",
        "gross_income_test_not_passed",
        "gross_income_test_not_met",
        "gross_income_test_is_not_met",
        "gross_income_test_never_passed",
        "gross_income_test_waived",
        "gross_income_test_applies",
        "gross_income_test_result",
        "gross_income_test_passed_previously",
        "unit_ineligible_under_gross_income_test",
        "unit_fails_gross_income_test",
        "unit_does_not_pass_gross_income_test",
        "unit_almost_passes_gross_income_test",
        "unit_never_passes_gross_income_test",
        "unit_is_exempt_from_gross_income_test",
        "unit_is_denied_under_gross_income_test",
        "unit_unable_to_pass_gross_income_test",
        "unit_takes_gross_income_test",
        "no_gross_income_test",
        "no_unit_passes_gross_income_test",
        "unit_passes_no_gross_income_test",
        "unit_passes_all_but_gross_income_test",
        "unit_passes_other_than_gross_income_test",
        "unit_passes_gross_income_testing",
        "unit_passes_gross_income_test2",
        "unit_passes_gross_income_test_and_has_income",
        "unit_passes_gross_earned_income_income_test",
        "unit_passes_gross_earnings_test",
        "gross_income_test_amount_is_met",
        "former_unit_passes_gross_income_test",
        "false_that_unit_passes_gross_income_test",
        "doubtful_unit_passes_gross_income_test",
        "budgetary_unit_passes_gross_income_test",
        "unit_passes_one_two_gross_income_test",
        "one_two_gross_income_test_passes",
        "a_b_c_gross_income_test_passes",
    ],
)
def test_an_input_name_that_is_not_a_positive_statement_stays_refused(name):
    assert _source_gate_income_test_input_markers(name, "") == frozenset()
    assert _refused(names=(name, NET))


@pytest.mark.parametrize(
    "description",
    [
        "The gross income test is failed",
        "The gross income test is not met.",
        "The unit does not pass the gross income test.",
        "Whether the unit fails the gross income test",
        "The unit is exempt from the gross income test",
        "The unit passes the gross income test or is exempt",
        "The unit passes the net income test",
        "The unit passes the income test",
        "Gross income of the unit",
        "A status",
        "False if the unit passes the gross income test",
        "Not true that the unit passes the gross income test",
        "Unless the unit passes the gross income test",
        "Doubtful the unit passes the gross income test",
        "False that unit passes the gross income test",
        "The unit passes whether the gross income test",
        "The unit passes the gross income test whether",
    ],
)
def test_a_description_that_does_not_state_the_same_test_stays_refused(description):
    assert _source_gate_income_test_input_markers(GROSS, description) == frozenset()
    assert _refused(descriptions={GROSS: description})


# --- Source gates that are not a positive statement (all as on main) --------


@pytest.mark.parametrize(
    "clause",
    [
        # A sentential negation around the statement.
        f"{LEAD} when it is not true that the unit passes the gross income test "
        "and the net income test.",
        f"{LEAD} when it is not the case that the unit passes the gross income "
        "test and the net income test.",
        # Other verbs and negators.
        CLAUSE.replace("passes", "is failing"),
        CLAUSE.replace("passes", "is exempt from"),
        CLAUSE.replace("passes", "is unable to pass"),
        CLAUSE.replace("passes", "is ineligible under"),
        CLAUSE.replace("passes", "is denied under"),
        CLAUSE.replace("passes", "flunks"),
        CLAUSE.replace("passes", "fails"),
        CLAUSE.replace("passes", "hardly passes"),
        CLAUSE.replace("passes", "almost passes"),
        CLAUSE.replace("passes", "never passes"),
        CLAUSE.replace("passes", "does not pass"),
        CLAUSE.replace("passes", "has passed"),
        CLAUSE.replace("passes", "must pass"),
        CLAUSE.replace("passes", "takes"),
        CLAUSE.replace("passes", "is subject to"),
        f"{LEAD} when the unit passes neither the gross income test nor the net "
        "income test.",
        f"{LEAD} when the unit passes no gross income test and the net income test.",
        # A negation or another fact before the first statement.
        f"{LEAD} when the unit does not file a report and pass the gross income "
        "test and the net income test.",
        f"{LEAD} when the unit does not file nor passes the gross income test "
        "and the net income test.",
        f"{LEAD} when it is not true that the applicant files and the unit passes "
        "the gross income test and the net income test.",
        f"{LEAD} when the unit is a resident and the unit passes the gross income "
        "test and the net income test.",
        f"{LEAD} when passes the gross income test and the net income test.",
        # An exception after the statements.
        CLAUSE[:-1] + " unless the unit is exempt.",
        CLAUSE[:-1] + " except when the unit is exempt.",
        # Other cues.
        CLAUSE.replace("when the unit", "unless the unit"),
        CLAUSE.replace("when the unit", "provided that the unit"),
        "Unless the unit passes the gross income test and the net income test, "
        "the agency performs the final benefit test.",
        # A subject that is not an allowed noun directly before the verb.
        CLAUSE.replace("the unit", "no unit"),
        CLAUSE.replace("the unit", "neither unit"),
        CLAUSE.replace("the unit", "the unit never"),
        CLAUSE.replace("the unit", "the agency"),
        CLAUSE.replace("the unit", "it"),
        # Words after the test name.
        CLAUSE.replace("gross income test", "gross income test amount"),
        CLAUSE.replace("gross income test", "gross income test without income"),
        CLAUSE.replace("net income test.", "net income test amount."),
        CLAUSE.replace("net income test.", "net income test it fails."),
        # The second object needs the first statement directly before it.
        f"{LEAD} when the unit is a resident and the net income test.",
        f"{LEAD} when the unit passes the gross income test nor the net income test.",
        # Other determiners before the test name.
        CLAUSE.replace("the gross", "no gross"),
        CLAUSE.replace("the gross", "neither gross"),
        CLAUSE.replace("the gross", "every gross"),
        CLAUSE.replace("and the net", "and not the net"),
        CLAUSE.replace("and the net", "and no net"),
        # More than two words before "income test".
        CLAUSE.replace("the gross", "the other than gross"),
        CLAUSE.replace("the gross", "the one two gross"),
    ],
)
def test_a_gate_that_is_not_a_positive_statement_stays_refused(clause):
    assert clause != CLAUSE
    assert _refused(clause)


@pytest.mark.parametrize(
    "phrase",
    [
        "income-test",
        "income testing",
        "income tested",
        "income testable",
        "income testers",
        "income test's",
        "income test’s",
        "income test2",
        "income test-related",
        "income, test",
        "income (test)",
        "income limit",
        "income standard",
        "earnings test",
        "5income test",
        "'income test",
        "’income test",
        "non-income test",
        "income income test",
    ],
)
def test_other_wording_around_income_stays_refused(phrase):
    assert _refused(CLAUSE.replace("income test", phrase))
    assert _refused(CLAUSE.replace("gross income test", f"gross {phrase}"))
    assert _refused(CLAUSE.replace("net income test", f"net {phrase}"))
    assert _statement(f"the unit passes the gross {phrase}") is None


@pytest.mark.parametrize("boundary", ["; ", ". ", " but ", ", ", " ", "\n", " and "])
def test_a_statement_after_other_text_in_the_same_gate_stays_refused(boundary):
    clause = (
        f"{LEAD} when the unit is a resident{boundary}the unit passes the gross "
        "income test and the net income test."
    )
    assert _refused(clause)
    assert _refused(clause, (RESIDENT, GROSS, NET))


def test_the_real_shape_with_the_consequence_after_the_test_stays_refused():
    # The clause has no boundary between the last test name and the main
    # clause, so the last gate holds the words of the consequence. That gate
    # is not a closed form. The first gate is matched; the second is not.
    clause = (
        "When the unit passes the gross income test, and the net income test the "
        "agency performs the final benefit test."
    )
    assert _refused(clause)
    (_, first), (_, second) = _source_conjunctive_fact_gates(clause)
    assert first >= GROSS_MARKERS
    assert not any(word.startswith("income_test_name:") for word in second)
    clause = (
        "When the unit passes the gross income test and the net income test, the "
        "agency performs the final benefit test."
    )
    assert _refused(clause)


@pytest.mark.parametrize("word", ["abc2", "non-cash", "unit's", "x’s", "a1"])
def test_a_slot_word_is_letters_only(word):
    assert _statement(f"the unit passes the {word} income test") is None
    assert _statement(f"the {word} unit passes the gross income test") is None
    name = "unit_passes_" + word.replace("-", "_") + "_income_test"
    if not word.isalnum():
        return
    assert _source_gate_income_test_input_markers(name, "") == frozenset()
    assert _refused(CLAUSE.replace("the gross", f"the {word}"), (name, NET))


def test_a_line_break_inside_the_statement_stays_refused():
    assert _refused(CLAUSE.replace("gross income test", "gross income\ntest"))
    assert _refused(CLAUSE.replace("passes the gross", "passes\nthe gross"))


# --- Real income facts ------------------------------------------------------


@pytest.mark.parametrize(
    ("condition", "names"),
    [
        # A real income fact in the same gate as a named test.
        (
            "the unit receives earned income counted in the gross income test "
            "and the unit is a resident",
            (GROSS, RESIDENT),
        ),
        (
            "the unit has income while it passes the income test and the unit is "
            "a resident",
            ("unit_passes_income_test", RESIDENT),
        ),
        (
            "the unit has income while it passes the income test and the unit is "
            "a resident",
            ("unit_has_income", RESIDENT),
        ),
        (
            "the unit passes the income test without income and the unit is a resident",
            ("unit_has_no_income", RESIDENT),
        ),
        (
            "the unit passes the income test without income and the unit is a resident",
            ("unit_passes_income_test", RESIDENT),
        ),
        (
            "the unit passes the gross income test with countable income and the "
            "net income test",
            (GROSS, NET),
        ),
        # An input named for a test does not state an income fact.
        (
            "the unit has income and the unit is a resident",
            ("unit_passes_income_test", RESIDENT),
        ),
        (
            "the unit has income below the standard and the unit is a resident",
            ("unit_income_test_below_standard", RESIDENT),
        ),
        (
            "the unit has income below the standard and the unit is a resident",
            ("unit_passes_income_test", RESIDENT),
        ),
        (
            "the unit passes the gross income test and has income below the standard",
            (GROSS, NET),
        ),
        (
            "the unit passes the gross income test and has income below the standard",
            (GROSS, "unit_passes_income_test"),
        ),
        # An input named for a test does not state another test.
        (
            "the unit meets the resource test and the unit is a resident",
            (GROSS, RESIDENT),
        ),
    ],
)
def test_income_facts_and_named_tests_do_not_stand_for_each_other(condition, names):
    assert _refused(f"{LEAD} when {condition}.", names)


def test_a_real_income_condition_next_to_a_named_test_needs_its_own_input():
    clause = (
        f"{LEAD} when the unit passes the gross income test and has income below "
        "the standard."
    )
    assert _collapsed(clause, (GROSS, "unit_has_income_below_standard")) == []


# --- The closed list for the open word slots --------------------------------

BLOCKED_SLOT_WORDS = (
    ["absence", "almost", "barely", "but", "cannot", "denied", "devoid"]
    + ["document", "documentation", "earning", "earnings", "evidence", "except"]
    + ["exempt", "fail", "failed", "failing", "fails", "half", "hardly", "if"]
    + ["income", "ineligible", "lack", "lacked", "lacking", "lacks", "nearly"]
    + ["neither", "never", "nil", "no", "non", "nor", "not", "only", "other"]
    + ["partial", "partly", "proof", "record", "report", "reported", "reporting"]
    + ["reports", "scarcely", "tax", "taxed", "taxes", "than", "unable", "unless"]
    + ["without", "zero"]
)
# For these words main has its own reading before the marker is consulted: it
# cuts the clause at ``except`` / ``unless``, and it reads ``no income`` and
# the like as an explicit "no income" fact. The end-to-end result for them is
# the result of main; the form-level assertions below still pin each word.
MAIN_READS_BEFORE_THE_MARKER = frozenset(
    {"except", "unless", "no", "nil", "zero", "without"}
    | {"lack", "lacked", "lacking", "lacks"}
)


def test_the_blocked_slot_words_are_exactly_the_pinned_list():
    assert frozenset({*BLOCKED_SLOT_WORDS, "0"}) == (
        _SOURCE_GATE_INCOME_TEST_BLOCKED_SLOT_WORDS
    )


@pytest.mark.parametrize("word", BLOCKED_SLOT_WORDS)
def test_a_blocked_word_in_any_open_slot_keeps_the_gate_refused(word):
    # The subject slot and the modifier slot of the source statement.
    assert _statement(f"the {word} unit passes the gross income test") is None
    assert _statement(f"the unit passes the {word} income test") is None
    assert _statement(f"the unit passes the {word} gross income test") is None
    assert (
        _source_gate_income_test_markers(
            f"the {word} income test", (_SOURCE_GATE_INCOME_TEST_CONTINUATION,)
        )
        is None
    )
    # The same slots of an input name, in both name forms.
    for name in (
        f"unit_passes_{word}_income_test",
        f"unit_passes_{word}_gross_income_test",
        f"{word}_income_test_passes",
        f"{word}_gross_income_test_passes",
        f"gross_{word}_income_test_passes",
    ):
        assert _source_gate_income_test_input_markers(name, "") == frozenset()
    if word in MAIN_READS_BEFORE_THE_MARKER:
        return
    # End to end: without the block each pair below would match.
    assert _refused(CLAUSE.replace("the unit", f"the {word} unit"))
    assert _refused(
        CLAUSE.replace("the gross", f"the {word}"),
        (f"unit_passes_{word}_income_test", NET),
    )
    assert _refused(
        CLAUSE.replace("the net", f"the {word}"),
        (GROSS, f"{word}_income_test_passes"),
    )


# --- The readings of main are unchanged -------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "passes the gross income test",
        "the unit passes the gross income test",
        "na gross income test passes",
        "it is not true that the unit passes the income test",
        "passes the gross income test with income below the standard",
    ],
)
def test_the_income_reading_itself_is_unchanged(text):
    assert _source_gate_predicate_polarities(text) == {"income": None}
    tokens = _source_gate_semantic_tokens(text)
    assert "unknown_income" in tokens
    assert not any(token.startswith("income_test_name:") for token in tokens)


def test_a_marked_input_keeps_its_polarity_conflict():
    evidence = _terminal_gate_evidence(
        {
            "inputs": [
                {"name": GROSS, "entity": "Household", "dtype": "Boolean"},
                {"name": "unit_has_income", "entity": "Household", "dtype": "Boolean"},
            ]
        }
    )
    assert evidence[GROSS].polarity_conflict
    assert "unknown_income" in evidence[GROSS].predicates
    assert evidence[GROSS].predicates >= GROSS_MARKERS
    assert not any(
        word.startswith("income_test_name:")
        for word in evidence["unit_has_income"].predicates
    )


def test_the_markers_of_the_two_gates():
    (_, first), (_, second) = _source_conjunctive_fact_gates(CLAUSE)
    assert {w for w in first if w.startswith("income_test_name:")} == GROSS_MARKERS
    assert {w for w in second if w.startswith("income_test_name:")} == NET_MARKERS
    assert "unknown_income" in first
    assert "unknown_income" in second

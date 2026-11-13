"""Bulleted alternatives are not simultaneous facts for the gates-collapsed check.

The fixtures are real rejected candidates of three Arizona utility units; see
``tests/fixtures/az-utility-alternative-lists/README.md``.
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_conjunctive_fact_gates,
    analyze_complete_source_unit,
)

FIXTURES = Path(__file__).parent / "fixtures" / "az-utility-alternative-lists"
UNIT = "us-az/manual/des/faa5/na-utility-expenses-and-allowances"

LIFELINE = (
    "When the budgetary unit participates in the Lifeline Assistance Program, a "
    "telephone expense is allowable when one or more of the following apply:\n\n"
    "● The discount program does not pay the expense in full\n\n"
    "● Extra airtime is purchased after exhausting free minutes"
)
SHARED = (
    "The budgetary unit is entitled to a utility allowance based on utility "
    "expenses verified when either situation occurs:\n\n"
    "● More than one budgetary unit lives in the same residence and shares "
    "utility expenses.\n\n"
    "● One budgetary unit shares a housing unit with others who do not receive NA."
)
HEATING = (
    "Any of the following utility expenses are considered heating sources when "
    "used to provide heat for the home:\n\n"
    "● Electricity\n\n● Gas or propane\n\n● Coal, wood, or oil\n\n"
    "Any of the following are considered cooling sources:\n\n"
    "● Central air conditioners\n\n● Evaporative coolers\n\n"
    "● Room air conditioners\n\n"
    "The cost for any of the following supplemental sources of heating or cooling "
    "do not qualify for the SUA:\n"
)
WOOD = (
    "The budgetary unit is eligible for SUA when all of the following occur:\n\n"
    "● The wood, which is purchased or billed separate from or itemized with rent "
    "or mortgage payments\n\n"
    "● The wood is used to produce heat."
)


def _collapsed(
    directory: str,
    block: int,
    content: str | None = None,
    source: str | None = None,
):
    folder = FIXTURES / directory
    citation = f"{UNIT}/block-{block}"
    profile = vp._numeric_profile_for_citation_path(citation)
    analysis = analyze_complete_source_unit(
        content or (folder / "candidate.yaml").read_text(),
        source or (folder / "source.txt").read_text(),
        corpus_citation_path=citation,
        test_cases=yaml.safe_load((folder / "candidate.test.yaml").read_text()),
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


@pytest.mark.parametrize(
    ("directory", "block"),
    [
        ("block-14-e9371ed3", 14),
        ("block-14-66be7798", 14),
        ("block-3-07b0428f", 3),
        ("block-3-9dd68cae", 3),
        ("block-4-170e8d74", 4),
        ("block-4-28d570f4", 4),
    ],
)
def test_candidates_that_encode_alternatives_as_alternatives_pass(directory, block):
    assert _collapsed(directory, block) == []


def test_all_of_the_following_list_still_passes_with_separate_facts():
    assert _collapsed("control-block-16-858a1c04", 16) == []


def test_merged_all_of_the_following_facts_still_fail():
    folder = FIXTURES / "control-block-16-858a1c04"
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "wood_sua_conditions_met",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    merged = [
        rule
        for rule in payload["rules"]
        if "same_wood_used_to_produce_heat" in rule["versions"][0]["formula"]
    ]
    assert [rule["name"] for rule in merged] == ["wood_expense_sua_condition"]
    merged[0]["versions"][0]["formula"] = "wood_sua_conditions_met"

    issues = _collapsed(
        "control-block-16-858a1c04", 16, yaml.safe_dump(payload, sort_keys=False)
    )

    assert len(issues) == 1
    assert "`wood_expense_sua_condition` versions[0]" in issues[0]
    assert "when all of the following occur:" in issues[0]
    assert issues[0].expected == "2 distinct gate inputs"
    assert issues[0].found.startswith("0 of 2 source gates")


def test_merged_shared_meter_conjunction_still_fails_and_names_its_clause():
    folder = FIXTURES / "block-4-28d570f4"
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "evaluated_unit_metered_status",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    rule = payload["rules"][-1]
    assert rule["name"] == "separate_housing_shared_meter_utility_allowance_path"
    rule["versions"][0]["formula"] = (
        "evaluated_unit_utility_expense_verified and evaluated_unit_metered_status"
    )

    issues = _collapsed("block-4-28d570f4", 4, yaml.safe_dump(payload, sort_keys=False))

    assert len(issues) == 1
    message = issues[0]
    # Only the rule with the real conjunction; the alternative paths stay clean.
    assert message.count(" versions[") == 1
    assert "`separate_housing_shared_meter_utility_allowance_path` versions[0]" in (
        message
    )
    assert (
        '[source clause "When budgetary units that incur utility expenses live '
        "in separate housing units" in message
    )
    assert "states 2 facts by their source words: fact 1 (" in message
    assert "gives 1 of 2 facts its own input" in message
    assert "no input name matches the source words of fact 2]" in message
    assert message.expected == "2 distinct gate inputs"
    assert message.found == (
        "1 of 2 source gates with their own terminal input on the "
        "least-supported formula path"
    )


@pytest.mark.parametrize("clause", [LIFELINE, SHARED, HEATING])
def test_bullets_under_an_alternative_chapeau_are_not_conjunctive_gates(clause):
    assert _source_conjunctive_fact_gates(clause) == ()


@pytest.mark.parametrize(
    "clause",
    [
        WOOD,
        LIFELINE.replace(
            "one or more of the following apply", "all of the following apply"
        ),
        SHARED.replace("either situation occurs", "both situations occur"),
        # Words outside the closed grammar keep the conservative reading.
        SHARED.replace("either situation occurs", "the following situations occur"),
        # A conjunction inside one alternative is still counted.
        LIFELINE.replace(
            "● Extra airtime is purchased after exhausting free minutes",
            "● The parent is eligible and the child is disabled",
        ),
        # A proviso after the list is part of the condition.
        LIFELINE + "\n\nprovided that the parent is eligible and the child is disabled",
        # A conjunction in the chapeau itself is still counted.
        LIFELINE.replace(
            "when one or more",
            "when the parent is eligible and the child is disabled and one or more",
        ),
    ],
)
def test_conjunctions_near_an_alternative_list_are_still_gates(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


def test_a_condition_ends_at_the_next_sentence_paragraph():
    first_list = HEATING.split("Any of the following are considered cooling")[0]
    assert _source_conjunctive_fact_gates(first_list) == ()
    assert _source_conjunctive_fact_gates(HEATING) == ()
    assert _source_conjunctive_fact_gates(LIFELINE + "\n\n" + NEXT_SENTENCE) == ()


ANY_OF = "The unit is eligible when any of the following apply:"
ONE_OR_MORE = "The unit is eligible when one or more of the following apply:"
ALL_OF = "The unit is eligible when all of the following apply:"
EMPLOYED = "● The parent is employed"
DISABLED = "● The child is disabled"
EXEMPT = "● The household is exempt"
ELDERLY = "● The household is elderly"
NEXT_SENTENCE = "A utility allowance is allowed regardless of the percentage paid."


def _list(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def test_the_closed_alternative_shapes_drop_their_bullets():
    assert _source_conjunctive_fact_gates(_list(ANY_OF, EMPLOYED, DISABLED)) == ()
    assert _source_conjunctive_fact_gates(_list(ONE_OR_MORE, EMPLOYED, DISABLED)) == ()
    # The grammar ignores letter case.
    assert (
        _source_conjunctive_fact_gates(
            _list(ANY_OF.replace("any of the", "ANY Of The"), EMPLOYED, DISABLED)
        )
        == ()
    )
    # A wrapped bullet without a conjunction is still one alternative.
    assert (
        _source_conjunctive_fact_gates(
            _list(ANY_OF, "● The parent is\nemployed full time", DISABLED)
        )
        == ()
    )


@pytest.mark.parametrize(
    "clause",
    [
        # P1-a: a nested chapeau bullet starts a conjunctive sublist.
        _list(
            ANY_OF,
            EXEMPT,
            "● The household meets all of the following:",
            EMPLOYED,
            DISABLED,
        ),
        _list(
            LIFELINE.split("\n\n")[0],
            EXEMPT,
            "● The household meets these requirements:",
            EMPLOYED,
            DISABLED,
        ),
        # P1-a: a second, conjunctive list follows the alternatives.
        _list(
            ONE_OR_MORE,
            EXEMPT,
            ELDERLY,
            "and all of the following apply:",
            EMPLOYED,
            DISABLED,
        ),
        "\n".join(
            [
                ONE_OR_MORE,
                EXEMPT,
                ELDERLY,
                "All of the following must also apply:",
                EMPLOYED,
                DISABLED,
            ]
        ),
        # A bullet that ends with a conjunction makes the list conjunctive.
        _list(ANY_OF, EMPLOYED + "; and", DISABLED),
        _list(ANY_OF, EMPLOYED + " as well as the child is disabled", EXEMPT),
        _list(ANY_OF, "● Both the parent is employed, the child is disabled", EXEMPT),
    ],
)
def test_one_alternative_chapeau_governs_only_its_own_bullet_run(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


@pytest.mark.parametrize(
    "chapeau",
    [
        # P1-b: the quantifier does not govern the list of conditions.
        "The unit is eligible when any of the following persons meets all of "
        "the following requirements:",
        "The unit is eligible when it receives any of the following benefits "
        "and meets these requirements:",
        "The unit is eligible when it does not meet any of the following:",
        "The unit is eligible when it is not the case that any of the following apply:",
        "The unit is eligible when the requirements for any of the following "
        "programs are all met:",
        "The unit is eligible when any of the following persons meets the "
        "requirements below:",
        "Any of the following expenses are deductible when both conditions are met:",
        "Any of the following expenses are deductible when the following "
        "conditions are met:",
        "Any of the following expenses are deductible when these facts are true:",
        "The unit isn't eligible when any of the following apply:",
        "The unit is eligible when one or more of the following occurs "
        "regarding the parent or the child:",
        "The unit is eligible when none of the following apply:",
        "The unit is eligible when each of the following is true:",
        "The unit is eligible when either of the following apply:",
        "The unit is eligible when at least two of the following apply:",
        # The quantifier must be in the last sentence of the line.
        "Any of the following documents are proof. The unit is eligible when "
        "the parent qualifies:",
        "The parent qualifies when employed. The unit is eligible when any of "
        "the following apply:",
        # The quantifier shape must be the whole line.
        "The unit is eligible. Any of the following expenses are deductible "
        "when the parent qualifies:",
        # The chapeau must end with a colon.
        "The unit is eligible when any of the following apply.",
        "The unit is eligible when any of the following apply",
    ],
)
def test_a_chapeau_outside_the_closed_grammar_keeps_its_bullets(chapeau):
    assert len(_source_conjunctive_fact_gates(_list(chapeau, EMPLOYED, DISABLED))) >= 2


@pytest.mark.parametrize(
    "clause",
    [
        # Only the line that holds the cue, directly above the list, counts.
        "The unit is eligible when all of the following apply.\n"
        "Any of the following documents may be used as proof:\n\n"
        + _list(EMPLOYED, DISABLED),
        _list(ALL_OF, "Any of the following apply:", EMPLOYED, DISABLED),
        _list(ANY_OF, "The facts below are required:", EMPLOYED, DISABLED),
        _list(
            "Any of the following are proof:",
            "● Wages",
            "The unit is eligible when the following occur:",
            EMPLOYED,
            DISABLED,
        ),
        # A bullet is never a chapeau.
        _list(
            "Any of the following apply:",
            "● Heat, when the parent is employed",
            DISABLED,
            "● The household is poor",
        ),
        # P2-d: the same bullet text under another chapeau, before or after.
        _list("Any of the following are proof:", EMPLOYED, ALL_OF, EMPLOYED, DISABLED),
        _list(ALL_OF, EMPLOYED, DISABLED, "Any of the following apply:", EMPLOYED),
    ],
)
def test_the_chapeau_is_the_cue_line_directly_above_the_list(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


@pytest.mark.parametrize(
    "clause",
    [
        # P1-c: the paragraph after the list continues the condition.
        _list(
            ONE_OR_MORE,
            EXEMPT,
            ELDERLY,
            "In addition, all of the following must apply:",
            EMPLOYED,
            DISABLED,
        ),
        _list(
            ANY_OF,
            EXEMPT,
            "The parent must be employed and the child must be disabled.",
        ),
        _list(ANY_OF, EMPLOYED, DISABLED, "The household also is poor."),
        # P1-c: no cut under a chapeau that is not an alternative chapeau.
        _list(ALL_OF.replace("apply", "occur"), EMPLOYED, "The child is disabled."),
        _list(
            ALL_OF.replace("apply", "occur"),
            EMPLOYED,
            EXEMPT,
            "The child is disabled and the household is poor.",
        ),
        _list(
            "The unit is eligible when the parent is employed:",
            "● Wages",
            "The child is disabled and the household is poor.",
        ),
        # The cut needs a blank line, a capital and a final period or colon.
        _list(ANY_OF, EMPLOYED, DISABLED + "\n" + NEXT_SENTENCE),
        _list(ANY_OF, EMPLOYED, DISABLED, NEXT_SENTENCE[0].lower() + NEXT_SENTENCE[1:]),
        _list(ANY_OF, EMPLOYED, DISABLED, NEXT_SENTENCE.rstrip(".")),
    ],
)
def test_text_after_the_list_keeps_the_conservative_reading(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


def test_a_wrapped_bullet_keeps_its_conjunction():
    wrapped = _list(
        ANY_OF, "● The parent is employed\nand the child is disabled", EXEMPT
    )
    joined = wrapped.replace("employed\nand", "employed and")
    assert len(_source_conjunctive_fact_gates(joined)) >= 2
    assert _source_conjunctive_fact_gates(wrapped) == _source_conjunctive_fact_gates(
        joined
    )


WOOD_BULLETS = WOOD.split("\n\n")[1:]
WOOD_LEAD = "The budgetary unit is eligible for SUA when "


@pytest.mark.parametrize(
    ("lead", "body", "prefix"),
    [
        (
            "any of the following occur:",
            [
                "● The budgetary unit is exempt",
                "● The budgetary unit meets all of the following:",
                *WOOD_BULLETS,
            ],
            "",
        ),
        (
            "any of the following fuels meets all of the following requirements:",
            WOOD_BULLETS,
            "",
        ),
        (
            "one or more of the following occur:",
            [
                "● The budgetary unit is exempt",
                "● The budgetary unit is elderly",
                "In addition, all of the following must occur:",
                *WOOD_BULLETS,
            ],
            "",
        ),
        (
            "one or more of the following occur:",
            [
                "● The budgetary unit is exempt",
                "● The budgetary unit is elderly",
                "and all of the following occur:",
                *WOOD_BULLETS,
            ],
            "",
        ),
        (
            "all of the following occur:",
            WOOD_BULLETS[::-1],
            "Any of the following are heating facts:\n\n" + WOOD_BULLETS[1] + "\n\n",
        ),
        (
            "all of the following occur:",
            [WOOD_BULLETS[0], WOOD_BULLETS[1][2:]],
            "",
        ),
        ("it does not meet any of the following:", WOOD_BULLETS, ""),
        (
            "it uses any of the following fuels and both conditions below are met:",
            WOOD_BULLETS,
            "",
        ),
    ],
)
def test_a_merged_conjunctive_list_near_alternative_words_still_fails(
    lead, body, prefix
):
    folder = FIXTURES / "control-block-16-858a1c04"
    source = (folder / "source.txt").read_text()
    assert source.startswith(WOOD)
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "wood_sua_conditions_met",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    rule = payload["rules"][0]
    rule["versions"][0]["formula"] = "wood_sua_conditions_met"
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = WOOD_LEAD + lead

    issues = _collapsed(
        "control-block-16-858a1c04",
        16,
        yaml.safe_dump(payload, sort_keys=False),
        source=prefix + _list(WOOD_LEAD + lead, *body) + source[len(WOOD) :],
    )

    assert len(issues) == 1


def test_merged_facts_under_a_true_alternative_list_pass():
    folder = FIXTURES / "control-block-16-858a1c04"
    source = (folder / "source.txt").read_text()
    payload = yaml.safe_load((folder / "candidate.yaml").read_text())
    template = payload["inputs"][0]
    payload["inputs"].append(
        {
            "name": "wood_sua_conditions_met",
            "entity": template["entity"],
            "dtype": "Boolean",
            "period": template["period"],
        }
    )
    lead = WOOD_LEAD + "one or more of the following occur:"
    rule = payload["rules"][0]
    rule["versions"][0]["formula"] = "wood_sua_conditions_met"
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = lead

    assert (
        _collapsed(
            "control-block-16-858a1c04",
            16,
            yaml.safe_dump(payload, sort_keys=False),
            source=_list(lead, *WOOD_BULLETS) + source[len(WOOD) :],
        )
        == []
    )

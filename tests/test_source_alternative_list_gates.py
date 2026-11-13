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


def _collapsed(directory: str, block: int, content: str | None = None):
    folder = FIXTURES / directory
    citation = f"{UNIT}/block-{block}"
    profile = vp._numeric_profile_for_citation_path(citation)
    analysis = analyze_complete_source_unit(
        content or (folder / "candidate.yaml").read_text(),
        (folder / "source.txt").read_text(),
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
    later = (
        "The unit is eligible when the parent is employed:\n\n"
        "● Wages\n\n"
        "The child is disabled and the household is poor."
    )
    assert _source_conjunctive_fact_gates(first_list) == ()
    assert _source_conjunctive_fact_gates(later) == ()
    # An unbulleted lowercase continuation is not a new sentence paragraph.
    assert (
        len(
            _source_conjunctive_fact_gates(
                later.replace("The child is disabled", "the child is disabled")
            )
        )
        >= 2
    )

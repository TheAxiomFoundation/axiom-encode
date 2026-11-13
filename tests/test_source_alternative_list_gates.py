"""Bulleted alternatives are not simultaneous facts for the gates-collapsed check.

The fixtures are real rejected candidates of three Arizona utility units; see
``tests/fixtures/az-utility-alternative-lists/README.md``.
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.source_completeness import (
    CHECK_SOURCE_EXPLICIT_CONDITIONS_COLLAPSED,
    _source_conjunctive_fact_gates,
    _without_alternative_bullets_or_later_paragraphs,
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
HEATING = (FIXTURES / "block-3-07b0428f" / "source.txt").read_text()
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


LIFELINE_CHAPEAU, SHARED_CHAPEAU, HEATING_CHAPEAU = (
    clause.split("\n\n")[0] for clause in (LIFELINE, SHARED, HEATING)
)
HEATING_REST = HEATING[HEATING.index("Any of the following are considered cooling") :]
EMPLOYED = "● The parent is employed"
DISABLED = "● The child is disabled"
EXEMPT = "● The household is exempt"
ELDERLY = "● The household is elderly"


def _list(*paragraphs: str) -> str:
    return "\n\n".join(paragraphs)


def _body_is_unchanged(clause: str) -> bool:
    """Run the helper as `_source_conjunctive_fact_gates` does, on the first cue."""

    cue = re.search(r"\b(?:if|when|provided\s+that|unless)\b", clause, re.IGNORECASE)
    assert cue is not None
    body = clause[cue.end() :]
    return (
        _without_alternative_bullets_or_later_paragraphs(clause, body, cue.end())
        == body
    )


def _swap(chapeau: str, old: str, new: str) -> str:
    assert old in chapeau
    return chapeau.replace(old, new)


@pytest.mark.parametrize(
    "clause",
    [
        _list(LIFELINE_CHAPEAU, EMPLOYED, DISABLED),
        _list(SHARED_CHAPEAU, EXEMPT, ELDERLY),
        "\n".join([LIFELINE_CHAPEAU, EMPLOYED, DISABLED]),
        # The closed forms after the cue.
        _list(
            _swap(
                LIFELINE_CHAPEAU,
                "when one or more of the following apply",
                "if any of the following are met",
            ),
            EMPLOYED,
            DISABLED,
        ),
        _list(
            _swap(
                SHARED_CHAPEAU, "either situation occurs", "any of the following occur"
            ),
            EXEMPT,
            ELDERLY,
        ),
        # Typographic spaces are plain spaces.
        _list(
            _swap(LIFELINE_CHAPEAU, "is allowable", "is allowable"),
            EMPLOYED.replace("● ", "● "),
            DISABLED,
        ),
        # A preceding paragraph does not change the chapeau.
        _list("The child is disabled.", LIFELINE_CHAPEAU, EMPLOYED, DISABLED),
    ],
)
def test_the_allowlisted_alternative_lists_drop_their_bullets(clause):
    assert len(_source_conjunctive_fact_gates(clause)) < 2


def test_the_heating_list_needs_the_exact_rest_of_its_unit():
    assert _source_conjunctive_fact_gates(HEATING) == ()
    assert _source_conjunctive_fact_gates(HEATING + "\n") == ()
    first_list = HEATING[: -len(HEATING_REST)].rstrip()
    assert _source_conjunctive_fact_gates(first_list) == ()
    for changed in (
        # Other text after the list may continue the condition.
        _list(first_list, "In addition, all of the following must occur:")
        + "\n\n"
        + _list(EMPLOYED, DISABLED),
        HEATING
        + "\n\n"
        + _list("All of the following must occur:", EMPLOYED, DISABLED),
        HEATING.replace("● Fans", "● Fans and heat lamps"),
        # No blank line before the rest.
        first_list + "\n" + HEATING_REST,
    ):
        assert len(_source_conjunctive_fact_gates(changed)) >= 2, changed


@pytest.mark.parametrize(
    "chapeau",
    [
        # Reviewer shapes: a main clause outside the allowlist.
        "The unit is eligible when any of the following apply:",
        "The unit is eligible when one or more of the following apply:",
        "The unit is eligible when either situation occurs:",
        "The unit is eligible when the worker cannot verify that any of the "
        "following apply:",
        "The unit is eligible when it isn’t shown that any of the following apply:",
        "The unit is eligible when the unit fails to show that any of the "
        "following are true:",
        "The unit is eligible when the worker cannot confirm either situation occurs:",
        "Any of the following costs are deductible when:",
        "Any of the following fuels qualifies when it meets the two tests listed:",
        "Any of the following utility expenses are considered heating sources "
        "when the unit:",
        # Text between the cue and the quantifier, or after the verb.
        _swap(
            LIFELINE_CHAPEAU,
            "when one or more",
            "when the worker cannot verify that one or more",
        ),
        _swap(
            SHARED_CHAPEAU,
            "when either",
            "when the worker cannot rule out that either",
        ),
        _swap(LIFELINE_CHAPEAU, "apply:", "apply to each member:"),
        _swap(LIFELINE_CHAPEAU, "apply:", "apply"),
        LIFELINE_CHAPEAU + " Both facts below are needed.",
        _swap(LIFELINE_CHAPEAU, "apply:", "apply."),
        _swap(HEATING_CHAPEAU, "for the home:", "for the home when:"),
        # Another quantifier or cue.
        _swap(LIFELINE_CHAPEAU, "one or more of", "all of"),
        _swap(LIFELINE_CHAPEAU, "one or more of", "any one of"),
        _swap(LIFELINE_CHAPEAU, "one or more of", "at least two of"),
        _swap(SHARED_CHAPEAU, "either situation occurs", "both situations occur"),
        # A main clause that is not the real one, word for word.
        _swap(LIFELINE_CHAPEAU, "is allowable", "is not allowable"),
        _swap(LIFELINE_CHAPEAU, "is allowable", "isn’t allowable"),
        _swap(LIFELINE_CHAPEAU, "a telephone expense", "no telephone expense"),
        _swap(LIFELINE_CHAPEAU, "allowable", "allow​able"),
        LIFELINE_CHAPEAU.upper(),
        "The child is disabled. " + LIFELINE_CHAPEAU,
        # The chapeau is one physical line.
        _swap(LIFELINE_CHAPEAU, "following apply", "following\napply"),
        _swap(LIFELINE_CHAPEAU, "a telephone", "a\ntelephone"),
    ],
)
def test_a_chapeau_outside_the_allowlist_keeps_its_bullets(chapeau):
    clause = _list(chapeau, EMPLOYED, DISABLED)
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


@pytest.mark.parametrize(
    "clause",
    [
        # A line wrap can hide a negation on the line before.
        "A unit qualifies that does not show\n"
        + _list(HEATING_CHAPEAU, EMPLOYED, DISABLED),
        "It is false that\n"
        + _list(LIFELINE_CHAPEAU[0].lower() + LIFELINE_CHAPEAU[1:], EMPLOYED, DISABLED),
        "The child is disabled.\n" + _list(LIFELINE_CHAPEAU, EMPLOYED, DISABLED),
        # Only blank lines may stand between the chapeau and the list.
        _list(LIFELINE_CHAPEAU, "The facts below are required:", EMPLOYED, DISABLED),
        # The condition cue is on another line than the chapeau.
        _list(
            "The unit is eligible when all of the following apply.",
            HEATING_CHAPEAU,
            EMPLOYED,
            DISABLED,
        ),
    ],
)
def test_the_chapeau_is_the_cue_line_after_a_blank_line(clause):
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


@pytest.mark.parametrize(
    "items",
    [
        # A conjunction or a requirement word in a bullet.
        [EMPLOYED + " and the child is disabled", EXEMPT],
        [EXEMPT + ";", EMPLOYED + "; and", DISABLED + "."],
        [EMPLOYED, "● and the child is disabled"],
        [EMPLOYED + " & the child is disabled", EXEMPT],
        [EMPLOYED + " together with the child is disabled", EXEMPT],
        [EMPLOYED + " so long as the child is disabled", EXEMPT],
        [EMPLOYED + " provided the child is disabled", EXEMPT],
        ["● The parent must be employed", EXEMPT],
        ["● The parent is required to be employed", EXEMPT],
        [EMPLOYED + " (the child is disabled)", EXEMPT],
        [EMPLOYED + ";", DISABLED + ";"],
        ["● Employment: the parent is employed", EXEMPT],
        # A nested chapeau.
        [EXEMPT, "● The household meets the following:", EMPLOYED, DISABLED],
        [
            EXEMPT,
            "● The household meets the tests below",
            "  " + EMPLOYED,
            "  " + DISABLED,
        ],
        # Another glyph or list marker.
        [EMPLOYED, DISABLED.replace("●", "•")],
        [EXEMPT, ELDERLY, "1. The parent is employed", "2. The child is disabled"],
        # One bullet is not a list.
        [EMPLOYED, "The child is disabled."],
        # Text after the list.
        [EMPLOYED, DISABLED, "so long as the household is billed for the expense"],
        [EMPLOYED, DISABLED, "The household is required to be billed for the expense."],
        [EMPLOYED, DISABLED, "The household has to be billed for the expense."],
        [
            EXEMPT,
            ELDERLY,
            "The following requirements apply as well:",
            EMPLOYED,
            DISABLED,
        ],
        [
            EXEMPT,
            ELDERLY,
            "Furthermore, the following conditions apply:",
            EMPLOYED,
            DISABLED,
        ],
        [EXEMPT, ELDERLY, "when all of the following apply:", EMPLOYED, DISABLED],
        [EMPLOYED, DISABLED, "unless the household is billed and the child is absent"],
    ],
)
def test_a_list_outside_the_allowlist_keeps_every_gate(items):
    assert len(_source_conjunctive_fact_gates(_list(LIFELINE_CHAPEAU, *items))) >= 2


@pytest.mark.parametrize(
    "lines",
    [
        # Single-newline text: a continuation line is never joined to a bullet.
        [
            EXEMPT,
            "● The household meets these tests",
            "o The parent is employed",
            "o The child is disabled",
        ],
        [
            EXEMPT,
            "● The household meets these tests:",
            "o The parent is employed",
            "o The child is disabled",
        ],
        [
            EXEMPT,
            "● The participant is a student who",
            "- is enrolled half time",
            "- is employed 20 hours",
        ],
        [
            EXEMPT,
            "● The household meets these tests",
            "(a) the parent is employed",
            "(b) the child is disabled",
        ],
        [
            EXEMPT,
            "● The household meets the following:",
            "the parent is employed",
            "the child is disabled",
        ],
        [EMPLOYED, DISABLED, "so long as the household is billed for the expense"],
        [EXEMPT, ELDERLY, "All of the following must also apply:", EMPLOYED, DISABLED],
        [EMPLOYED + "\twhile the child is disabled", EXEMPT],
        ["● The parent is employed", "while the child is disabled", EXEMPT],
    ],
)
def test_a_continuation_line_keeps_every_gate(lines):
    clause = "\n".join([LIFELINE_CHAPEAU, *lines])
    assert len(_source_conjunctive_fact_gates(clause)) >= 2


@pytest.mark.parametrize(
    "clause",
    [
        # These have fewer than two gates on main too, so the count cannot
        # show the refusal; the body must come back unchanged.
        _list(
            _swap(LIFELINE_CHAPEAU, "allowable when", "allowable unless"),
            EMPLOYED,
            DISABLED,
        ),
        _list(
            _swap(LIFELINE_CHAPEAU, "allowable when", "allowable except when"),
            EMPLOYED,
            DISABLED,
        ),
        _list(LIFELINE_CHAPEAU, EMPLOYED, DISABLED, HEATING_REST),
        _list(
            HEATING[: -len(HEATING_REST)].rstrip(),
            "The home is required to be heated.",
        ),
        _list(HEATING_CHAPEAU, "● Electricity"),
        _list(HEATING_CHAPEAU, "● Electricity", "  ● Gas or propane"),
        _list(HEATING_CHAPEAU, "● Electricity", "• Gas or propane"),
        _list(HEATING_CHAPEAU, "● Electricity", "● Gas, when used for heat"),
    ],
)
def test_a_refused_shape_leaves_the_body_unchanged(clause):
    assert _body_is_unchanged(clause)


def test_the_real_bullet_with_its_own_conjunction_stays_in_the_body():
    first, second = SHARED.split("\n\n")[1:]
    assert " and " in first
    body = _without_alternative_bullets_or_later_paragraphs(SHARED, SHARED, 0)
    assert body == SHARED_CHAPEAU + "\n" + first
    # The same bullet does not open the list to other conjunctions.
    nested = _list(SHARED_CHAPEAU, first, "● The household meets the following:")
    assert _without_alternative_bullets_or_later_paragraphs(nested, nested, 0) == (
        nested
    )
    other = _list(SHARED_CHAPEAU, first.replace("shares", "splits"), second)
    assert _without_alternative_bullets_or_later_paragraphs(other, other, 0) == other


def test_the_body_must_be_the_untouched_rest_of_the_text():
    clause = _list(LIFELINE_CHAPEAU, EMPLOYED, DISABLED)
    start = clause.index(" the budgetary unit")
    body = clause[start:]
    assert (
        _without_alternative_bullets_or_later_paragraphs(clause, body, start)
        == (body.split("\n")[0])
    )
    # An earlier step cut the body: the text goes on, so nothing is dropped.
    longer = clause + "\n\nunless the parent is absent and the child is absent"
    assert _without_alternative_bullets_or_later_paragraphs(longer, body, start) == (
        body
    )
    # An earlier step rewrote a later line of the body.
    rewritten = body.replace(DISABLED, DISABLED + " and the parent is absent")
    assert rewritten != body
    assert (
        _without_alternative_bullets_or_later_paragraphs(clause, rewritten, start)
        == rewritten
    )
    # The body does not start at the cue position.
    assert (
        _without_alternative_bullets_or_later_paragraphs(clause, body, start + 1)
        == body
    )
    assert len(_source_conjunctive_fact_gates(longer)) >= 2


WOOD_BULLETS = WOOD.split("\n\n")[1:]
WOOD_LEAD = "The budgetary unit is eligible for SUA when "
UNIT_EXEMPT = "● The budgetary unit is exempt"
UNIT_ELDERLY = "● The budgetary unit is elderly"


def _merged_wood_issues(source_head: str, excerpt: str):
    folder = FIXTURES / "control-block-16-858a1c04"
    source = (folder / "source.txt").read_text()
    assert source.startswith(WOOD) and excerpt in source_head
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
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = excerpt
    return _collapsed(
        "control-block-16-858a1c04",
        16,
        yaml.safe_dump(payload, sort_keys=False),
        source=source_head + source[len(WOOD) :],
    )


@pytest.mark.parametrize(
    ("lead", "body"),
    [
        ("all of the following occur:", _list("", *WOOD_BULLETS)),
        # Without the real main clause even a plain alternative list is kept.
        ("one or more of the following occur:", _list("", *WOOD_BULLETS)),
        (
            "the worker cannot verify that any of the following occur:",
            _list("", *WOOD_BULLETS),
        ),
        (
            "it isn’t verified that any of the following occur:",
            _list("", *WOOD_BULLETS),
        ),
        (
            "the worker cannot rule out that either situation occurs:",
            _list("", *WOOD_BULLETS),
        ),
        (
            "any of the following occur:",
            "\n"
            + "\n".join(
                [
                    UNIT_EXEMPT,
                    "● The budgetary unit meets the following:",
                    *(bullet.replace("●", "o") for bullet in WOOD_BULLETS),
                ]
            ),
        ),
        (
            "any of the following occur:",
            _list(
                "",
                UNIT_EXEMPT,
                "● The budgetary unit meets the tests below",
                *("  " + bullet for bullet in WOOD_BULLETS),
            ),
        ),
        (
            "one or more of the following occur:",
            _list(
                "",
                UNIT_EXEMPT,
                UNIT_ELDERLY,
                "The budgetary unit is required to meet the following conditions:",
                *WOOD_BULLETS,
            ),
        ),
        (
            "one or more of the following occur:",
            _list(
                "",
                UNIT_EXEMPT,
                UNIT_ELDERLY,
                "The following requirements apply as well:",
                *WOOD_BULLETS,
            ),
        ),
        (
            "one or more of the following occur:",
            _list(
                "",
                UNIT_EXEMPT,
                WOOD_BULLETS[0],
                "The wood is required to be used to produce heat.",
            ),
        ),
    ],
)
def test_a_merged_conjunctive_list_near_alternative_words_still_fails(lead, body):
    assert len(_merged_wood_issues(WOOD_LEAD + lead + body, WOOD_LEAD + lead)) == 1


@pytest.mark.parametrize(
    ("chapeau", "excerpt"),
    [
        (
            "Any of the following wood costs qualify the budgetary unit for SUA when:",
            None,
        ),
        (
            "Any of the following budgetary units is eligible for SUA when it "
            "meets the two tests listed:",
            None,
        ),
        (
            "A budgetary unit is eligible for SUA that does not show\n"
            "any of the following when the application is filed:",
            "any of the following when the application is filed:",
        ),
    ],
)
def test_a_merged_condition_list_under_a_subject_quantifier_still_fails(
    chapeau, excerpt
):
    issues = _merged_wood_issues(_list(chapeau, *WOOD_BULLETS), excerpt or chapeau)
    assert len(issues) == 1

"""Two short encoder prompt bullets and the validator behavior they rely on.

1. Never import from a module that declares ``module.kind: composition``. The
   rules engine refuses such a module on its atomic surface, so no test here
   can show the refusal without the engine; the tests pin the text only.
2. A count of persons or things that the source treats as one quantity,
   compared with a source-stated number, is a ``dtype: Count`` value with
   companion cases at the number and next to it. The tests pin the text and
   show that a Boolean input cannot exercise the source boundary and that a
   Count input with a case exactly at the number does. A count of listed
   conditions is outside the bullet; one Count input for such a list is
   refused.

Both prompt copies (``prompts/encoder.py`` and ``harness/evals.py``) carry the
same two bullets.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import evals
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from axiom_encode.prompts.encoder import (
    _NAMING_PROTOCOL,
    _TESTS_PROTOCOL,
    ENCODER_PROMPT,
)

COMPOSITION_IMPORT = (
    "- Never import from a module that declares `module.kind: composition`. "
    "In a file you write, the rules engine loads each import target as an "
    "atomic module, and an atomic module must not declare `module.kind`, so "
    "the file fails to compile. Treat the exports of such a module as not "
    "available for import; no other import rule changes."
)
COUNT_LISTED_CONDITIONS = (
    "This bullet does not apply to a count of listed conditions "
    '("at least 2 of the following"): each listed condition stays its own '
    "input."
)
COUNT_THRESHOLD_SCOPE = (
    "This bullet does not change how N itself is named, or the rules for a "
    "boundary on an imported derived output."
)
COUNT_THRESHOLD = (
    "- When a source condition compares a whole-number count of persons or "
    "things that the source treats as one quantity (for example members, "
    "persons, or months) with a source-stated number N "
    '("N or more", "at least N", "more than N"), the formula compares a '
    "`dtype: Count` value with N: a local `dtype: Count` input when no import "
    "or relation supplies the count, not a `dtype: Boolean` input that "
    "restates the comparison. Give companion cases with that count exactly at "
    "N and at the adjacent whole number on the other side of the comparison "
    '(N plus 1 for "more than N", N minus 1 for "N or more" and "at least '
    'N"), each asserting the affected principal output. '
    f"{COUNT_LISTED_CONDITIONS} Where another rule allows "
    '"a fact predicate or derived condition" for such a comparison, use the '
    f"derived condition over the count. {COUNT_THRESHOLD_SCOPE}"
)

# Each bullet with its block, the bullet before it and the bullet after it.
BULLETS = [
    pytest.param(
        COMPOSITION_IMPORT,
        _NAMING_PROTOCOL,
        "- Never introduce an import cycle.",
        "- Never create a derived rule whose formula references that same rule's name.",
        id="composition-import",
    ),
    pytest.param(
        COUNT_THRESHOLD,
        _TESTS_PROTOCOL,
        "- When a cap is encoded in its own derived rule because the provision also",
        "- Never turn an imported derived rule into a fabricated",
        id="count-threshold",
    ),
]


def _flat(text: str) -> str:
    return " ".join(text.split())


def _bullet(flat_text: str, start: str) -> str:
    begin = flat_text.index(start)
    return flat_text[begin : flat_text.index(" - ", begin + 2)]


def _eval_prompt(tmp_path) -> str:
    source = tmp_path / "source.txt"
    source.write_text("The annual allowance is 100.\n")
    workspace = evals.EvalWorkspace(tmp_path, source, tmp_path / "manifest.json")
    return _flat(
        evals._build_eval_prompt(
            "us/guidance/example/allowance",
            "cold",
            workspace,
            [],
            "policies/example/allowance.yaml",
            include_tests=True,
        )
    )


@pytest.mark.parametrize(("text", "block", "before", "after"), BULLETS)
def test_encoder_prompt_carries_the_guidance_bullet(text, block, before, after):
    flat_block = _flat(block)
    assert _bullet(flat_block, text[:60]) == text
    assert flat_block.index(before) < flat_block.index(text) < flat_block.index(after)
    # The bullet directly follows its neighbor: no other bullet lies between.
    assert _bullet(flat_block, before) + " " + text in flat_block
    prompt = _flat(ENCODER_PROMPT)
    assert flat_block in prompt
    assert prompt.count(text) == 1


@pytest.mark.parametrize(("text", "block", "before", "after"), BULLETS)
def test_eval_prompt_carries_the_same_guidance_bullet(
    text, block, before, after, tmp_path
):
    prompt = _eval_prompt(tmp_path)
    assert prompt.count(text) == 1
    assert prompt.index(before) < prompt.index(text) < prompt.index(after)
    assert _bullet(prompt, before) + " " + text in prompt


def test_neither_prompt_tells_which_text_a_list_condition_excerpt_quotes(tmp_path):
    # A bullet that told the encoder to quote the list item, not the chapeau,
    # was withdrawn: with item-only excerpts the gates-collapsed check does
    # not see a dropped item of a conjunctive list.
    for prompt in (_flat(ENCODER_PROMPT), _eval_prompt(tmp_path)):
        assert "the line that introduces the list" not in prompt
        assert "not the chapeau" not in prompt


def test_composition_bullet_names_only_the_import_of_a_composition_module():
    # One prohibition and its reason; nothing about boundary inputs, deferral
    # or proof atoms, which their own rules keep.
    assert COMPOSITION_IMPORT.endswith("no other import rule changes.")
    # The engine claim covers a file the encoder writes, not every load.
    assert "In a file you write, the rules engine loads each import target" in (
        COMPOSITION_IMPORT
    )
    lowered = COMPOSITION_IMPORT.lower()
    for word in ("boundary", "defer", "proof", "cycle", "prefix", "hash", "every"):
        assert word not in lowered, word


def test_count_bullet_keeps_the_neighboring_rules():
    assert COUNT_THRESHOLD.endswith(COUNT_THRESHOLD_SCOPE)
    # The naming rule it refers to still offers both forms, in these words.
    assert (
        "Encode the whole source-stated condition as a fact predicate or "
        "derived condition"
    ) in _flat(_NAMING_PROTOCOL)
    # The imported-output boundary rule stays as it is.
    assert (
        "choose clearly below/above-boundary inputs instead of an exact boundary case."
    ) in _flat(_TESTS_PROTOCOL)
    # It does not say how the number is declared, and claims no check.
    for word in ("parameter", "literal", "named scalar", "validator", "check"):
        assert word not in COUNT_THRESHOLD, word


def test_count_bullet_excludes_a_count_of_listed_conditions():
    assert COUNT_THRESHOLD.count(COUNT_LISTED_CONDITIONS) == 1
    assert "that the source treats as one quantity" in COUNT_THRESHOLD
    examples = COUNT_THRESHOLD[
        COUNT_THRESHOLD.index("(for example") : COUNT_THRESHOLD.index(") with a")
    ]
    assert examples == "(for example members, persons, or months"
    assert "items" not in COUNT_THRESHOLD


# --- Validator behavior on constructed sources -----------------------------

CITATION = "us/guidance/example/lending"
MODULE = "us:policies/example/lending"


def _issues(payload: dict, source: str, cases: list[dict]) -> list[str]:
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [str(issue) for issue in result.issues]


def _source_atom(excerpt: str, kind: str = "formula") -> dict:
    return {
        "path": "versions[0].formula",
        "kind": kind,
        "source": {"corpus_citation_path": CITATION, "excerpt": excerpt},
    }


def _input(name: str, dtype: str) -> dict:
    return {"name": name, "entity": "Household", "dtype": dtype, "period": "Month"}


BOOLEAN_INPUT = _input("lending_unit_has_enough_members", "Boolean")
COUNT_INPUT = _input("lending_unit_member_count", "Count")
BOUNDARY_UNEXERCISED = (
    "[complete-source-unit:tests] Companion tests do not exercise every "
    "source-stated boundary input; missing: source unit=4."
)
NUMERIC_RECALL = "[complete-source-unit:numeric-recall]"
GATES_COLLAPSED = "[complete-source-unit:source-explicit-conditions]"


def _count_payload(
    source: str,
    count_input: dict,
    formula: str,
    *,
    threshold: int | None = 4,
    excerpt: str | None = None,
) -> dict:
    excerpt = source if excerpt is None else excerpt
    rules = []
    if threshold is not None:
        rules.append(
            {
                "name": "group_loan_member_threshold",
                "kind": "parameter",
                "dtype": "Count",
                "metadata": {"proof": {"atoms": [_source_atom(excerpt, "parameter")]}},
                "versions": [
                    {"effective_from": "2026-01-01", "formula": str(threshold)}
                ],
            }
        )
    rules.append(
        {
            "name": "group_loan_available",
            "kind": "derived",
            "entity": "Household",
            "dtype": "Judgment",
            "period": "Month",
            "metadata": {"proof": {"atoms": [_source_atom(excerpt)]}},
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }
    )
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [count_input],
        "rules": rules,
    }


def _count_case(input_name: str, value: object, holds: bool) -> dict:
    return {
        "name": f"{input_name}_{value}",
        "period": "2026-01",
        "input": {f"{MODULE}#input.{input_name}": value},
        "output": {f"{MODULE}#group_loan_available": "holds" if holds else "not_holds"},
    }


BOOLEAN_CASES = [
    _count_case(BOOLEAN_INPUT["name"], True, True),
    _count_case(BOOLEAN_INPUT["name"], False, False),
]

# The source wording, the comparison, and the adjacent value on the other side.
COUNT_FORMS = [
    pytest.param(
        "A lending unit with at least 4 members may take a group loan.",
        ">=",
        3,
        id="at-least",
    ),
    pytest.param(
        "A lending unit with more than 4 members may take a group loan.",
        ">",
        5,
        id="more-than",
    ),
]


def _holds(operator: str, value: int) -> bool:
    return value >= 4 if operator == ">=" else value > 4


@pytest.mark.parametrize(("source", "operator", "adjacent"), COUNT_FORMS)
def test_boolean_input_cannot_exercise_a_count_boundary(source, operator, adjacent):
    payload = _count_payload(source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"])
    assert _issues(payload, source, BOOLEAN_CASES) == [BOUNDARY_UNEXERCISED]


@pytest.mark.parametrize(("source", "operator", "adjacent"), COUNT_FORMS)
def test_count_input_with_cases_at_and_next_to_the_number_is_clean(
    source, operator, adjacent
):
    formula = f"{COUNT_INPUT['name']} {operator} group_loan_member_threshold"
    payload = _count_payload(source, COUNT_INPUT, formula)
    cases = [
        _count_case(COUNT_INPUT["name"], value, _holds(operator, value))
        for value in (4, adjacent)
    ]
    assert _issues(payload, source, cases) == []


@pytest.mark.parametrize(("source", "operator", "adjacent"), COUNT_FORMS)
def test_count_input_without_a_case_at_the_number_is_refused(
    source, operator, adjacent
):
    formula = f"{COUNT_INPUT['name']} {operator} group_loan_member_threshold"
    payload = _count_payload(source, COUNT_INPUT, formula)
    cases = [
        _count_case(COUNT_INPUT["name"], value, _holds(operator, value))
        for value in (1, 9)
    ]
    assert _issues(payload, source, cases) == [BOUNDARY_UNEXERCISED]


def test_or_more_wording_makes_no_boundary_obligation():
    # The bullet names "N or more" for the modelling only: the boundary check
    # does not read that wording, so neither form is refused for it.
    source = "A lending unit with 4 or more members may take a group loan."
    payload = _count_payload(source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"])
    assert _issues(payload, source, BOOLEAN_CASES) == []
    formula = f"{COUNT_INPUT['name']} >= group_loan_member_threshold"
    payload = _count_payload(source, COUNT_INPUT, formula)
    cases = [_count_case(COUNT_INPUT["name"], value, value >= 4) for value in (1, 9)]
    assert _issues(payload, source, cases) == []


def test_how_the_number_passes_numeric_recall_is_not_the_bullets_subject():
    # Known open point, pinned as it is on main. A digit N written as a
    # formula literal has no named scalar and fails numeric recall; a named
    # parameter passes. The naming protocol tells the encoder not to create a
    # small-number parameter for a number that only qualifies a local factual
    # condition. The bullet says nothing about N's form.
    source = "A lending unit with at least 4 members may take a group loan."
    cases = [_count_case(COUNT_INPUT["name"], value, value >= 4) for value in (3, 4)]
    literal = _count_payload(
        source, COUNT_INPUT, f"{COUNT_INPUT['name']} >= 4", threshold=None
    )
    issues = _issues(literal, source, cases)
    assert len(issues) == 1
    assert issues[0].startswith(NUMERIC_RECALL)
    assert "numeric value 4 has no named scalar representation" in issues[0]
    named = _count_payload(
        source, COUNT_INPUT, f"{COUNT_INPUT['name']} >= group_loan_member_threshold"
    )
    assert _issues(named, source, cases) == []
    # A Boolean input without a parameter fails the same recall check, so the
    # Count form is not what makes the number need a name.
    boolean = _count_payload(
        source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"], threshold=None
    )
    issues = _issues(boolean, source, BOOLEAN_CASES)
    assert len(issues) == 2
    assert issues[0].startswith(NUMERIC_RECALL)
    assert issues[1] == BOUNDARY_UNEXERCISED


def test_small_word_number_stays_a_literal_in_the_comparison():
    source = "A lending unit active for more than one month may take a group loan."
    payload = _count_payload(
        source, COUNT_INPUT, f"{COUNT_INPUT['name']} > 1", threshold=None
    )
    cases = [
        _count_case(COUNT_INPUT["name"], 1, False),
        _count_case(COUNT_INPUT["name"], 2, True),
    ]
    assert _issues(payload, source, cases) == []
    boolean = _count_payload(
        source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"], threshold=None
    )
    assert _issues(boolean, source, BOOLEAN_CASES) == [
        BOUNDARY_UNEXERCISED.replace("=4.", "=1.")
    ]


LISTED_CONDITIONS_SOURCE = (
    "A lending unit may take a group loan when it meets at least 2 of the "
    "following conditions:\n\n"
    "● The unit has a tool steward.\n\n"
    "● The unit has paid its dues.\n\n"
    "● The unit has a storage room."
)


def test_one_count_input_for_listed_conditions_is_refused():
    # "At least 2 of the following" counts listed conditions, which the
    # bullet excludes. One Count input for the list hides each condition.
    count_input = _input("lending_unit_conditions_met_count", "Count")
    payload = _count_payload(
        LISTED_CONDITIONS_SOURCE,
        count_input,
        f"{count_input['name']} >= group_loan_member_threshold",
        threshold=2,
        excerpt=LISTED_CONDITIONS_SOURCE.split("\n\n")[0],
    )
    cases = [_count_case(count_input["name"], value, value >= 2) for value in (1, 2)]
    issues = _issues(payload, LISTED_CONDITIONS_SOURCE, cases)
    assert any(issue.startswith(GATES_COLLAPSED) for issue in issues)

"""Three short encoder prompt bullets and the validator behavior they rely on.

1. Never import from a module that declares ``module.kind: composition``. The
   rules engine refuses such a module on its atomic surface, so no test here
   can show the refusal without the engine; the tests pin the text only.
2. A formula proof excerpt for a condition that a list item states quotes the
   item, not the list chapeau. The tests pin the text and show, on a
   constructed source, that the chapeau excerpt gives a gates-collapsed finding
   and the item excerpt does not.
3. A count compared with a source-stated number is a ``dtype: Count`` value
   with companion cases at the number and next to it. The tests pin the text
   and show that a Boolean input cannot exercise the source boundary and that
   a Count input with a case exactly at the number does.

Both prompt copies (``prompts/encoder.py`` and ``harness/evals.py``) carry the
same three bullets.
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
    _COMPLETE_SOURCE_UNIT_PROTOCOL,
    _CORE_CONTRACT,
    _NAMING_PROTOCOL,
    _TESTS_PROTOCOL,
    ENCODER_PROMPT,
)

COMPOSITION_IMPORT = (
    "- Never import from a module that declares `module.kind: composition`. "
    "The rules engine loads every import target as an atomic module, and an "
    "atomic module must not declare `module.kind`, so the importing file fails "
    "to compile. Treat the exports of such a module as not available for "
    "import; no other import rule changes."
)
CHAPEAU_EXCERPT_SCOPE = (
    "This bullet does not change a requirement to quote a chapeau that itself "
    "states applicability, temporal scope, or a shared computation."
)
CHAPEAU_EXCERPT = (
    "- When a list item states the condition of a rule, a `source.excerpt` on "
    "that rule's `versions[N].formula` proof atoms quotes that list item, not "
    "the line that introduces the list (the chapeau). The validator reads an "
    "excerpt taken from the chapeau together with the list items that follow "
    "it and can count those items as separate simultaneous conditions of that "
    f"rule. {CHAPEAU_EXCERPT_SCOPE}"
)
COUNT_THRESHOLD_SCOPE = (
    "This bullet does not change how N itself is named, or the rules for a "
    "boundary on an imported derived output."
)
COUNT_THRESHOLD = (
    "- When a source condition compares a whole-number count (for example "
    "members, persons, items, or months) with a source-stated number N "
    '("N or more", "at least N", "more than N"), the formula compares a '
    "`dtype: Count` value with N: a local `dtype: Count` input when no import "
    "or relation supplies the count, not a `dtype: Boolean` input that "
    "restates the comparison. Give companion cases with that count exactly at "
    "N and at the adjacent whole number on the other side of the comparison "
    '(N plus 1 for "more than N", N minus 1 for "N or more" and "at least '
    'N"), each '
    "asserting the affected principal output. Where another rule allows "
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
        CHAPEAU_EXCERPT,
        _CORE_CONTRACT,
        "- For source-backed proof atoms, `source.corpus_citation_path` is sufficient.",
        "- For imported proof support, put `import:` at the proof atom top level",
        id="chapeau-excerpt",
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


def test_composition_bullet_names_only_the_import_of_a_composition_module():
    # One prohibition and its reason; nothing about boundary inputs, deferral
    # or proof atoms, which their own rules keep.
    assert COMPOSITION_IMPORT.endswith("no other import rule changes.")
    lowered = COMPOSITION_IMPORT.lower()
    for word in ("boundary", "defer", "proof", "cycle", "prefix", "hash"):
        assert word not in lowered, word


def test_chapeau_bullet_keeps_the_required_parent_chapeau_excerpt():
    # Complete-source-unit mode requires an exact parent-chapeau excerpt for a
    # chapeau that states applicability, temporal scope or a shared
    # computation. The new bullet names those three purposes as out of scope.
    required = _flat(_COMPLETE_SOURCE_UNIT_PROTOCOL)
    assert (
        "When a parent chapeau states applicability, temporal scope, or a "
        "shared computation, bind one consuming executable rule to that exact "
        "parent path and include an exact parent-chapeau proof excerpt."
    ) in required
    assert CHAPEAU_EXCERPT.endswith(CHAPEAU_EXCERPT_SCOPE)
    assert CHAPEAU_EXCERPT.count("applicability, temporal scope, or a shared") == 1
    # It tells which text an excerpt quotes; it does not ask for an excerpt.
    assert "Add `source.excerpt` only for" in _flat(_CORE_CONTRACT)
    assert "must include" not in CHAPEAU_EXCERPT
    assert "always" not in CHAPEAU_EXCERPT.lower()


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
    # It does not say how the number is declared.
    for word in ("parameter", "literal", "named scalar"):
        assert word not in COUNT_THRESHOLD, word


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


def _source_atom(excerpt: str | None, kind: str = "formula") -> dict:
    source = {"corpus_citation_path": CITATION}
    if excerpt is not None:
        source["excerpt"] = excerpt
    return {"path": "versions[0].formula", "kind": kind, "source": source}


CHAPEAU = (
    "Lending units may be eligible to receive any of the following credits "
    "from their dues:"
)
ITEM = (
    "A repair credit equal to 10% of its rental income when the lending unit "
    "has rental income."
)
LIST_SOURCE = (
    f"{CHAPEAU}\n\n● {ITEM}\n\n"
    "● Seasonal self-employment loss credit. (See the credit list.)\n\n"
    "● Storm repair credit. (See the credit list.)"
)


def _list_payload(atoms: list[dict]) -> dict:
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "lending_unit_has_rental_income",
                "entity": "Household",
                "dtype": "Boolean",
                "period": "Month",
            },
            {
                "name": "rental_income",
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
            },
        ],
        "rules": [
            {
                "name": "repair_credit_rate",
                "kind": "parameter",
                "dtype": "Rate",
                "metadata": {"proof": {"atoms": [_source_atom(ITEM, "parameter")]}},
                "versions": [{"effective_from": "2026-01-01", "formula": "0.10"}],
            },
            {
                "name": "repair_credit",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
                "metadata": {"proof": {"atoms": atoms}},
                "versions": [
                    {
                        "effective_from": "2026-01-01",
                        "formula": (
                            "if lending_unit_has_rental_income: "
                            "repair_credit_rate * rental_income else: 0"
                        ),
                    }
                ],
            },
        ],
    }


def _list_case(name: str, has_income: bool, expected: int) -> dict:
    return {
        "name": name,
        "period": "2026-01",
        "input": {
            f"{MODULE}#input.lending_unit_has_rental_income": has_income,
            f"{MODULE}#input.rental_income": 200,
        },
        "output": {f"{MODULE}#repair_credit": expected},
    }


LIST_CASES = [_list_case("income", True, 20), _list_case("no_income", False, 0)]
GATES_COLLAPSED = "[complete-source-unit:source-explicit-conditions]"


def test_item_excerpt_for_a_list_item_condition_is_clean():
    assert _issues(_list_payload([_source_atom(ITEM)]), LIST_SOURCE, LIST_CASES) == []


@pytest.mark.parametrize(
    "atoms",
    [
        pytest.param([_source_atom(CHAPEAU)], id="chapeau-only"),
        pytest.param(
            [_source_atom(CHAPEAU, "condition"), _source_atom(ITEM)],
            id="chapeau-and-item",
        ),
    ],
)
def test_chapeau_excerpt_reads_the_next_item_as_a_second_condition(atoms):
    issues = _issues(_list_payload(atoms), LIST_SOURCE, LIST_CASES)
    assert len(issues) == 1
    assert issues[0].startswith(GATES_COLLAPSED)
    assert "for 2 source gates" in issues[0]


def test_chapeau_citation_without_an_excerpt_is_clean():
    # The check reads excerpts only. A citation-only atom beside the item
    # excerpt does not pull the chapeau's clause in.
    atoms = [_source_atom(None, "condition"), _source_atom(ITEM)]
    assert _issues(_list_payload(atoms), LIST_SOURCE, LIST_CASES) == []


BOOLEAN_INPUT = {
    "name": "lending_unit_has_enough_members",
    "entity": "Household",
    "dtype": "Boolean",
    "period": "Month",
}
COUNT_INPUT = {
    "name": "lending_unit_member_count",
    "entity": "Household",
    "dtype": "Count",
    "period": "Month",
}
BOUNDARY_UNEXERCISED = (
    "[complete-source-unit:tests] Companion tests do not exercise every "
    "source-stated boundary input; missing: source unit=4."
)


def _count_payload(source: str, count_input: dict, formula: str) -> dict:
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [count_input],
        "rules": [
            {
                "name": "group_loan_member_threshold",
                "kind": "parameter",
                "dtype": "Count",
                "metadata": {"proof": {"atoms": [_source_atom(source, "parameter")]}},
                "versions": [{"effective_from": "2026-01-01", "formula": "4"}],
            },
            {
                "name": "group_loan_available",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Judgment",
                "period": "Month",
                "metadata": {"proof": {"atoms": [_source_atom(source)]}},
                "versions": [{"effective_from": "2026-01-01", "formula": formula}],
            },
        ],
    }


def _count_case(input_name: str, value: object, holds: bool) -> dict:
    return {
        "name": f"{input_name}_{value}",
        "period": "2026-01",
        "input": {f"{MODULE}#input.{input_name}": value},
        "output": {f"{MODULE}#group_loan_available": "holds" if holds else "not_holds"},
    }


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
    cases = [
        _count_case(BOOLEAN_INPUT["name"], True, True),
        _count_case(BOOLEAN_INPUT["name"], False, False),
    ]
    assert _issues(payload, source, cases) == [BOUNDARY_UNEXERCISED]


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


def test_count_comparison_with_a_small_literal_needs_no_parameter():
    # "more than one" keeps its number in the comparison; the bullet does not
    # change how the number is named.
    source = "A lending unit active for more than one month may take a group loan."
    payload = _count_payload(source, COUNT_INPUT, f"{COUNT_INPUT['name']} > 1")
    del payload["rules"][0]
    cases = [
        _count_case(COUNT_INPUT["name"], 1, False),
        _count_case(COUNT_INPUT["name"], 2, True),
    ]
    assert _issues(payload, source, cases) == []
    boolean = _count_payload(source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"])
    del boolean["rules"][0]
    cases = [
        _count_case(BOOLEAN_INPUT["name"], True, True),
        _count_case(BOOLEAN_INPUT["name"], False, False),
    ]
    assert _issues(boolean, source, cases) == [
        BOUNDARY_UNEXERCISED.replace("=4.", "=1.")
    ]

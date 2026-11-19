"""Two short encoder prompt bullets and the validator behavior they rely on.

1. Never import from a module that declares ``module.kind: composition``. The
   rules engine refuses such a module on its atomic surface, so no test here
   can show the refusal without the engine; the tests pin the text only.
2. A count of persons or things that the source treats as one quantity,
   compared with a source-stated number through the wording "at least N",
   "more than N" or "no more than N", is a ``dtype: Count`` value in one
   comparison, with companion cases at the number and next to it. The tests
   pin the text and, on constructed sources, the wording table: for these
   three wordings a Count input passes the boundary check and the pair check
   and a Boolean input is refused; for "N or more", "N or fewer" and "fewer
   than N" the pair check refuses the Count input, so the bullet does not
   name them. A separate equality test before the comparison leaves the
   boundary unexercised. A count of listed conditions is outside the bullet.

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
    "This bullet does not change how N is named, or the rules for a "
    "boundary on an imported derived output."
)
COUNT_SINGLE_COMPARISON = (
    "The threshold is one comparison of the count with N, with no separate "
    "test of equality with N."
)
COUNT_THRESHOLD = (
    "- When a source condition compares a count of persons or things that the "
    "source treats as one quantity (for example members or months) with a "
    'source-stated number N ("at least N", "more than N", "no more than N"), '
    "the formula compares a `dtype: Count` value with N: a local "
    "`dtype: Count` input when no import or relation supplies the count, not "
    "a `dtype: Boolean` input that restates the comparison. "
    f"{COUNT_SINGLE_COMPARISON} Give companion cases with that count exactly "
    "at N and at the adjacent number on the other side "
    '(N minus 1 for "at least N", N plus 1 for the other two), each '
    f"asserting the affected principal output. {COUNT_LISTED_CONDITIONS} "
    'Where another rule allows "a fact predicate or derived condition" for '
    "such a comparison, use the derived condition over the count. "
    f"{COUNT_THRESHOLD_SCOPE}"
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
    assert examples == "(for example members or months"
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
COUNT_INPUT = _input("lending_unit_members", "Count")
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


# The wording table. Each row: the wording, the comparison, the adjacent count
# on the other side, and whether a Count input with cases at N and at that
# adjacent count passes the pair check on a conditional sentence. Both checks
# read the wordings the bullet names; the pair check reads no interval from
# the other three, so no numeric case pair can satisfy it there.
WORDINGS = [
    pytest.param("at least {n}", ">=", -1, True, id="at-least"),
    pytest.param("more than {n}", ">", 1, True, id="more-than"),
    pytest.param("no more than {n}", "<=", 1, True, id="no-more-than"),
    pytest.param("{n} or more", ">=", -1, False, id="or-more"),
    pytest.param("{n} or fewer", "<=", 1, False, id="or-fewer"),
    pytest.param("fewer than {n}", "<", -1, False, id="fewer-than"),
]
NUMBER_FORMS = [pytest.param("4", id="digit"), pytest.param("four", id="word")]
RELATIVE_CLAUSE = "A lending unit with {wording} members may take a group loan."
CONDITIONAL = (
    "A lending unit may take a group loan when the lending unit has {wording} members."
)
PAIR_MISSING = (
    "[complete-source-unit:tests] Source-stated exceptions or applicability "
    "conditions require paired positive/blocking cases"
)
_COMPARE = {
    ">=": lambda value: value >= 4,
    ">": lambda value: value > 4,
    "<=": lambda value: value <= 4,
    "<": lambda value: value < 4,
}


def _checks(issues: list[str]) -> set[str]:
    found = set()
    for issue in issues:
        if issue.startswith(PAIR_MISSING):
            found.add("pair")
        elif "source-stated boundary input" in issue:
            found.add("boundary")
        else:
            found.add(issue)
    return found


def _count_form_checks(source: str, operator: str, adjacent: int) -> set[str]:
    formula = f"{COUNT_INPUT['name']} {operator} group_loan_member_threshold"
    cases = [
        _count_case(COUNT_INPUT["name"], value, _COMPARE[operator](value))
        for value in sorted((4, 4 + adjacent))
    ]
    return _checks(_issues(_count_payload(source, COUNT_INPUT, formula), source, cases))


def _boolean_form_checks(source: str) -> set[str]:
    payload = _count_payload(source, BOOLEAN_INPUT, BOOLEAN_INPUT["name"])
    return _checks(_issues(payload, source, BOOLEAN_CASES))


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "named"), WORDINGS)
def test_wording_table_for_a_count_input(wording, operator, adjacent, named, number):
    phrase = wording.format(n=number)
    # No wording refuses the Count form on the boundary check.
    relative = RELATIVE_CLAUSE.format(wording=phrase)
    assert _count_form_checks(relative, operator, adjacent) == set()
    conditional = CONDITIONAL.format(wording=phrase)
    expected = set() if named else {"pair"}
    assert _count_form_checks(conditional, operator, adjacent) == expected


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "named"), WORDINGS)
def test_wording_table_for_a_boolean_input(wording, operator, adjacent, named, number):
    # The mirror image: a Boolean input is refused for exactly the wordings
    # the bullet names, and passes for the others.
    phrase = wording.format(n=number)
    relative = _boolean_form_checks(RELATIVE_CLAUSE.format(wording=phrase))
    conditional = _boolean_form_checks(CONDITIONAL.format(wording=phrase))
    assert relative == ({"boundary"} if named else set())
    assert conditional == ({"pair"} if named else set())


def test_count_bullet_names_exactly_the_wordings_the_count_form_passes():
    listed = COUNT_THRESHOLD[
        COUNT_THRESHOLD.index("number N (") : COUNT_THRESHOLD.index("), the formula")
    ]
    assert listed == 'number N ("at least N", "more than N", "no more than N"'
    for param in WORDINGS:
        wording, _operator, _adjacent, named = param.values
        quoted = '"' + wording.format(n="N") + '"'
        assert (quoted in COUNT_THRESHOLD) is named, wording
    assert '(N minus 1 for "at least N", N plus 1 for the other two)' in (
        COUNT_THRESHOLD
    )


# --- One comparison, no equality branch ------------------------------------

FEE_INPUT = {
    "name": "group_fee_amount",
    "entity": "Household",
    "dtype": "Money",
    "unit": "USD",
    "period": "Month",
}
THRESHOLD = "group_loan_member_threshold"
SINGLE_COMPARISON = f"if lending_unit_members > {THRESHOLD}: group_fee_amount else: 0"
EQUALITY_BRANCH_FIRST = (
    f"if lending_unit_members == {THRESHOLD}: 0 else: "
    f"(if lending_unit_members > {THRESHOLD}: group_fee_amount else: 0)"
)
EQUALITY_IN_THE_CONDITION = (
    f"if lending_unit_members == {THRESHOLD} or lending_unit_members < {THRESHOLD}: "
    "0 else: group_fee_amount"
)
FEE_SOURCES = [
    pytest.param(
        "A lending unit with more than 4 members pays the group fee.",
        id="relative-clause",
    ),
    pytest.param(
        "A lending unit pays the group fee when the lending unit has more than "
        "4 members.",
        id="conditional",
    ),
]


def _fee_checks(source: str, formula: str) -> set[str]:
    payload = _count_payload(source, COUNT_INPUT, formula)
    payload["inputs"].append(FEE_INPUT)
    payload["rules"][1].update({"name": "group_fee", "dtype": "Money", "unit": "USD"})
    cases = [
        {
            "name": f"members_{value}",
            "period": "2026-01",
            "input": {
                f"{MODULE}#input.lending_unit_members": value,
                f"{MODULE}#input.group_fee_amount": 30,
            },
            "output": {f"{MODULE}#group_fee": 30 if value > 4 else 0},
        }
        for value in (3, 4, 5)
    ]
    return _checks(_issues(payload, source, cases))


def test_count_bullet_asks_for_one_comparison():
    assert COUNT_THRESHOLD.count(COUNT_SINGLE_COMPARISON) == 1


@pytest.mark.parametrize("source", FEE_SOURCES)
def test_one_comparison_of_the_count_exercises_the_boundary(source):
    assert _fee_checks(source, SINGLE_COMPARISON) == set()


@pytest.mark.parametrize("source", FEE_SOURCES)
@pytest.mark.parametrize(
    "formula",
    [
        pytest.param(EQUALITY_BRANCH_FIRST, id="equality-branch-first"),
        pytest.param(EQUALITY_IN_THE_CONDITION, id="equality-in-the-condition"),
    ],
)
def test_separate_equality_test_leaves_the_boundary_unexercised(source, formula):
    # The three cases give the same outputs as with one comparison, but the
    # case at the number never reaches the `>` comparison.
    assert _fee_checks(source, formula) == {"boundary"}


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

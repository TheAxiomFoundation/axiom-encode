"""Count thresholds in the boundary and exception-pair test checks (#2305).

Constructed sources only. Each test names the encoding it expects the two
checks to accept, and the wrong encodings of the same sentence that they must
still refuse.
"""

from __future__ import annotations

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us/guidance/example/lending"
MODULE = "us:policies/example/lending"
OUTPUT = "group_loan_available"
THRESHOLD = "group_loan_member_threshold"

_GROUNDING = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)


def _interval(
    text: str, reader=None
) -> tuple[float | None, bool, float | None, bool] | None:
    """The interval the pair check reads for one condition."""

    reader = reader or sc._exception_condition_interval
    interval = reader(
        sc.authoritative_numeric_recall_text(text),
        extract_numeric_occurrences=_GROUNDING,
    )
    if interval is None:
        return None
    return (
        interval.lower.value if interval.lower is not None else None,
        interval.lower_inclusive,
        interval.upper.value if interval.upper is not None else None,
        interval.upper_inclusive,
    )


def _floor(value: float, *, inclusive: bool = True):
    return (value, inclusive, None, False)


def _ceiling(value: float, *, inclusive: bool = True):
    return (None, False, value, inclusive)


def _issues(payload: dict, source: str, cases: list[dict]) -> list[str]:
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=_GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [str(issue) for issue in result.issues]


def _checks(issues: list[str]) -> set[str]:
    found = set()
    for issue in issues:
        if "require paired positive/blocking cases" in issue:
            found.add("pair")
        elif "source-stated boundary input" in issue:
            found.add("boundary")
        else:
            found.add(issue)
    return found


def _atom(excerpt: str, kind: str = "formula") -> dict:
    return {
        "path": "versions[0].formula",
        "kind": kind,
        "source": {"corpus_citation_path": CITATION, "excerpt": excerpt},
    }


def _input(name: str, dtype: str) -> dict:
    declaration = {
        "name": name,
        "entity": "Household",
        "dtype": dtype,
        "period": "Month",
    }
    if dtype == "Money":
        declaration["unit"] = "USD"
    return declaration


def _payload(
    source: str,
    inputs: list[dict],
    formula: str,
    *,
    threshold: float | None = None,
    threshold_dtype: str = "Count",
) -> dict:
    rules = []
    if threshold is not None:
        parameter = {
            "name": THRESHOLD,
            "kind": "parameter",
            "dtype": threshold_dtype,
            "metadata": {"proof": {"atoms": [_atom(source, "parameter")]}},
            "versions": [{"effective_from": "2026-01-01", "formula": str(threshold)}],
        }
        if threshold_dtype == "Money":
            parameter["unit"] = "USD"
        rules.append(parameter)
    rules.append(
        {
            "name": OUTPUT,
            "kind": "derived",
            "entity": "Household",
            "dtype": "Judgment",
            "period": "Month",
            "metadata": {"proof": {"atoms": [_atom(source)]}},
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }
    )
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": inputs,
        "rules": rules,
    }


def _case(inputs: dict[str, object], holds: bool) -> dict:
    return {
        "name": "case_" + "_".join(f"{key}_{value}" for key, value in inputs.items()),
        "period": "2026-01",
        "input": {f"{MODULE}#input.{key}": value for key, value in inputs.items()},
        "output": {f"{MODULE}#{OUTPUT}": "holds" if holds else "not_holds"},
    }


_COMPARE = {
    ">=": lambda value, number: value >= number,
    ">": lambda value, number: value > number,
    "<=": lambda value, number: value <= number,
    "<": lambda value, number: value < number,
}
_OTHER_STRICTNESS = {">=": ">", ">": ">=", "<=": "<", "<": "<="}


def _count_checks(
    source: str,
    operator: str,
    values: tuple[float, ...],
    *,
    name: str = "lending_unit_members",
    number: float = 4,
    dtype: str = "Count",
    threshold: float | None = 4,
    compared_with: str | None = None,
    extra_cases: tuple[dict, ...] = (),
) -> set[str]:
    right = compared_with or (THRESHOLD if threshold is not None else str(number))
    payload = _payload(
        source,
        [_input(name, dtype)],
        f"{name} {operator} {right}",
        threshold=threshold,
        threshold_dtype=dtype if dtype == "Money" else "Count",
    )
    compared = threshold if compared_with is None and threshold is not None else number
    if compared_with is not None and compared_with.lstrip("-").isdigit():
        compared = float(compared_with)
    cases = [
        _case({name: value}, _COMPARE[operator](value, compared)) for value in values
    ]
    return _checks(_issues(payload, source, [*cases, *extra_cases]))


def _boolean_checks(source: str, name: str = "lending_unit_has_enough_members"):
    payload = _payload(source, [_input(name, "Boolean")], name, threshold=4)
    cases = [_case({name: True}, True), _case({name: False}, False)]
    return _checks(_issues(payload, source, cases))


# --- 1. The interval of a wording -------------------------------------------

READ = [
    pytest.param("when the unit has four or more members", _floor(4), id="or-more"),
    pytest.param("when the unit has 4 or more members", _floor(4), id="or-more-digit"),
    pytest.param("4 members or fewer", _ceiling(4), id="unit-or-fewer"),
    pytest.param("$4,650 or less", _ceiling(4650), id="or-less"),
    pytest.param("members numbering 4 or greater", _floor(4), id="or-greater"),
    pytest.param("a score of 4 or higher", _floor(4), id="or-higher"),
    pytest.param("a score of 4 or lower", _ceiling(4), id="or-lower"),
    pytest.param("age 60 or older", _floor(60), id="or-older"),
    pytest.param("age 17 or younger", _ceiling(17), id="or-younger"),
    pytest.param("21 years of age and under", _ceiling(21), id="and-under"),
    pytest.param("age 65 and over", _floor(65), id="and-over"),
    pytest.param("a score of 4 and above", _floor(4), id="and-above"),
    pytest.param("a score of 4 or below", _ceiling(4), id="or-below"),
    pytest.param("fifteen (15) or more days", _floor(15), id="restated-number"),
    pytest.param("fewer than 4 members", _ceiling(4, inclusive=False), id="fewer-than"),
    pytest.param("no fewer than 4 members", _floor(4), id="no-fewer-than"),
    pytest.param("not fewer than 4 members", _floor(4), id="not-fewer-than"),
    pytest.param("two or more of the daily meals", _floor(2), id="or-more-of"),
]


@pytest.mark.parametrize(("text", "expected"), READ)
def test_interval_is_read_for_a_postfix_or_fewer_than_wording(text, expected):
    assert _interval(text) == expected


@pytest.mark.parametrize(("text", "expected"), READ)
def test_only_the_pair_check_reads_the_new_wordings(text, expected):
    # The shared parser is not changed, so the boundary check, the formula
    # branches and every other reader keep their reading.
    assert _interval(text, sc._formula_interval_from_text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("at least 4 members", _floor(4)),
        ("more than 4 members", _floor(4, inclusive=False)),
        ("no more than 4 members", _ceiling(4)),
        ("less than 4 members", _ceiling(4, inclusive=False)),
        ("between 2 and 4 members", (2, True, 4, True)),
    ],
)
def test_pair_check_keeps_the_reading_of_the_shared_parser(text, expected):
    assert _interval(text) == expected
    assert _interval(text, sc._formula_interval_from_text) == expected


UNREAD = [
    # "One or more" means "any".
    pytest.param("one or more of the following", id="one-or-more"),
    # A floor and a ceiling this parser does not compose.
    pytest.param("3 or more but fewer than 6 members", id="floor-and-ceiling"),
    pytest.param("4 or more members and 2 or fewer vehicles", id="two-limits"),
    # A second comparison, or a preposition, after the "or".
    pytest.param("20 days and over 65 years", id="and-over-number"),
    pytest.param("5 members or under the plan", id="preposition"),
    # A modal or a negation in front of the number.
    pytest.param("the unit must have four or more members", id="modal"),
    pytest.param("the unit does not have four or more members", id="negation"),
    pytest.param("no unit with four or more members", id="no"),
    # More than three words between the number and its comparison.
    pytest.param("4 members of the lending unit or more", id="too-far"),
    # Not a comparison.
    pytest.param("4 or 5 members", id="alternative-numbers"),
    pytest.param("more or less 4 members", id="more-or-less"),
]


@pytest.mark.parametrize("text", UNREAD)
def test_interval_stays_unread(text):
    assert _interval(text) is None


def test_or_more_before_a_second_comparison_keeps_the_reading_of_main():
    # Unchanged: only the prefix comparison is read.
    for text, expected in (
        ("$500 or more than $1,000", _floor(1000, inclusive=False)),
        ("3 or more but less than 6 members", _ceiling(6, inclusive=False)),
    ):
        assert _interval(text) == expected
        assert _interval(text, sc._formula_interval_from_text) == expected


POSTFIX_WORDS = [
    ("or more", True),
    ("or greater", True),
    ("or higher", True),
    ("or less", False),
    ("or fewer", False),
    ("or lower", False),
    ("or older", True),
    ("and older", True),
    ("or over", True),
    ("and over", True),
    ("or above", True),
    ("and above", True),
    ("or younger", False),
    ("and younger", False),
    ("or under", False),
    ("and under", False),
    ("or below", False),
    ("and below", False),
]


@pytest.mark.parametrize(("words", "is_floor"), POSTFIX_WORDS)
def test_each_postfix_wording_that_states_a_limit_has_an_interval(words, is_floor):
    # The words by which `_STATED_NUMERIC_LIMIT` calls a clause a numeric
    # limit after its number. Each must give an interval, or the pair check
    # owes a case pair that no numeric pair can pay.
    text = f"when the unit has 7 {words}"
    assert sc._STATED_NUMERIC_LIMIT.search(text) is not None
    assert _interval(text) == (_floor(7) if is_floor else _ceiling(7))


# --- 2. The wording table at the two checks ---------------------------------

# The wording, the comparison, the adjacent count on the other side, and
# whether the boundary check reads the wording. The boundary check is not
# changed here: for the last five wordings a relative clause creates no
# boundary obligation, as before.
WORDINGS = [
    pytest.param("at least {n}", ">=", -1, True, id="at-least"),
    pytest.param("more than {n}", ">", 1, True, id="more-than"),
    pytest.param("no more than {n}", "<=", 1, True, id="no-more-than"),
    pytest.param("{n} or more", ">=", -1, False, id="or-more"),
    pytest.param("{n} or fewer", "<=", 1, False, id="or-fewer"),
    pytest.param("fewer than {n}", "<", -1, False, id="fewer-than"),
    pytest.param("no fewer than {n}", ">=", -1, False, id="no-fewer-than"),
    pytest.param("{n} or greater", ">=", -1, False, id="or-greater"),
]
NUMBER_FORMS = [pytest.param("4", id="digit"), pytest.param("four", id="word")]
RELATIVE = "A lending unit with {wording} members may take a group loan."
CONDITIONAL = (
    "A lending unit may take a group loan when the lending unit has {wording} members."
)


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "boundary"), WORDINGS)
def test_count_input_with_cases_at_and_next_to_the_number_passes(
    wording, operator, adjacent, boundary, number
):
    values = tuple(sorted((4, 4 + adjacent)))
    for shape in (RELATIVE, CONDITIONAL):
        source = shape.format(wording=wording.format(n=number))
        assert _count_checks(source, operator, values) == set(), shape


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "boundary"), WORDINGS)
def test_boolean_restatement_is_refused_by_the_pair_check_for_every_wording(
    wording, operator, adjacent, boundary, number
):
    phrase = wording.format(n=number)
    assert _boolean_checks(CONDITIONAL.format(wording=phrase)) == {"pair"}
    relative = _boolean_checks(RELATIVE.format(wording=phrase))
    assert relative == ({"boundary"} if boundary else set())


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "boundary"), WORDINGS)
def test_other_strictness_is_refused_by_the_pair_check_for_every_wording(
    wording, operator, adjacent, boundary, number
):
    # `>` for "N or more", `<` for "N or fewer": the case at N and the case
    # next to it give the same output, so nothing shows the boundary.
    values = tuple(sorted((4, 4 + adjacent)))
    phrase = wording.format(n=number)
    wrong = _OTHER_STRICTNESS[operator]
    assert _count_checks(CONDITIONAL.format(wording=phrase), wrong, values) == {"pair"}
    relative = _count_checks(RELATIVE.format(wording=phrase), wrong, values)
    assert relative == ({"boundary"} if boundary else set())


@pytest.mark.parametrize("number", NUMBER_FORMS)
@pytest.mark.parametrize(("wording", "operator", "adjacent", "boundary"), WORDINGS)
def test_boundary_check_is_unchanged_for_a_relative_clause(
    wording, operator, adjacent, boundary, number
):
    source = RELATIVE.format(wording=wording.format(n=number))
    expected = {"boundary"} if boundary else set()
    assert _count_checks(source, operator, (1, 9)) == expected


# --- 3. A numeric pair is linked by the number, not by the input name -------

NAMES = [
    "lending_unit_member_count",
    "lending_unit_members_count",
    "lending_unit_members",
    "member_count",
    "unit_member_count",
    "number_of_lending_unit_members",
    "lending_unit_size",
    "persons_in_lending_unit",
    "headcount",
]
# "Is eligible ... when": the pair check reads that the condition enables the
# output. "May take ... when" gives it no direction.
AT_LEAST_FOUR = (
    "A lending unit is eligible for a group loan when the lending unit has at "
    "least 4 members."
)
NO_DIRECTION = CONDITIONAL.format(wording="at least 4")


@pytest.mark.parametrize("name", NAMES)
def test_pair_at_the_stated_number_passes_for_any_input_name(name):
    assert _count_checks(AT_LEAST_FOUR, ">=", (3, 4), name=name) == set()


@pytest.mark.parametrize("name", ["lending_unit_size", "headcount"])
def test_number_link_still_needs_the_boundary_and_the_right_comparison(name):
    # The other strictness: the pair at 3 and 4 does not toggle the output,
    # and the pair at 4 and 5 does not cross "at least 4".
    assert "pair" in _count_checks(AT_LEAST_FOUR, ">", (3, 4), name=name)
    assert "pair" in _count_checks(AT_LEAST_FOUR, ">", (4, 5), name=name)
    # A pair that crosses the interval with no case on the number.
    assert "pair" in _count_checks(AT_LEAST_FOUR, ">=", (1, 9), name=name)
    assert "pair" in _count_checks(AT_LEAST_FOUR, ">=", (3, 5), name=name)
    # The effect reversed: the loan is available below 4 members.
    assert "pair" in _count_checks(AT_LEAST_FOUR, "<", (3, 4), name=name)


def test_number_link_is_off_where_the_check_reads_no_effect_direction():
    # For "may take ... when" the pair check does not read whether the
    # condition enables or blocks. A reversed pair on an input whose name
    # shares words with the condition passes there (as before this change).
    # The link by number must not extend that to every input name.
    assert _count_checks(NO_DIRECTION, "<", (3, 4), name="lending_unit_members") == (
        set()
    )
    assert _count_checks(NO_DIRECTION, "<", (3, 4), name="headcount") == {"pair"}
    assert _count_checks(NO_DIRECTION, ">=", (3, 4), name="headcount") == {"pair"}


def test_number_link_needs_the_input_compared_with_the_stated_number():
    # The formula compares the input with 3, not with the source's 4. The
    # pair 3/4 crosses "at least 4" and sits on 4, but nothing ties this
    # input to the source's number.
    checks = _count_checks(
        AT_LEAST_FOUR,
        ">",
        (3, 4),
        name="headcount",
        threshold=None,
        compared_with="3",
    )
    assert "pair" in checks


def test_number_link_needs_a_dtype_that_fits_the_number():
    # "4 members" is not a currency amount: a Money input is not linked.
    assert _count_checks(AT_LEAST_FOUR, ">=", (3, 4), name="grant_total", dtype="Money")
    money = (
        "A lending unit is eligible for a group loan when the lending unit "
        "holds at least $40 in dues."
    )
    assert (
        _count_checks(
            money,
            ">=",
            (39, 40),
            name="paid_dues_balance",
            dtype="Money",
            number=40,
            threshold=40,
        )
        == set()
    )
    # "$40" is a currency amount: a Count input is not linked.
    assert "pair" in _count_checks(
        money, ">=", (39, 40), name="headcount", number=40, threshold=40
    )


def test_number_link_needs_a_one_sided_limit():
    # "Between 2 and 4" has two bounds; a pair on one of them is not linked.
    source = (
        "A lending unit is eligible for a group loan when the lending unit has "
        "between 2 and 4 members."
    )
    assert "pair" in _count_checks(source, "<=", (4, 5), name="headcount")
    # The pair 1/2 sits on the floor and crosses it; the ceiling is not shown.
    assert "pair" in _count_checks(
        source, ">=", (1, 2), name="headcount", number=2, threshold=2
    )


TWO_LIMITS_SAME_NUMBER = (
    "A lending unit is eligible for a group loan when the lending unit has at "
    "least 4 members.\n\n"
    "A lending unit is eligible for a group loan when the lending unit owns at "
    "least 4 tools."
)


def _two_limit_checks(first: str, second: str | None) -> set[str]:
    inputs = [_input(first, "Count")]
    formula = f"{first} >= {THRESHOLD}"
    cases = [_case({first: 3}, False), _case({first: 4}, True)]
    if second is not None:
        inputs.append(_input(second, "Count"))
        formula = f"{first} >= {THRESHOLD} or {second} >= {THRESHOLD}"
        cases = [
            _case({first: 3, second: 0}, False),
            _case({first: 4, second: 0}, True),
            _case({first: 0, second: 3}, False),
            _case({first: 0, second: 4}, True),
        ]
    payload = _payload(TWO_LIMITS_SAME_NUMBER, inputs, formula, threshold=4)
    return _checks(_issues(payload, TWO_LIMITS_SAME_NUMBER, cases))


def test_number_stated_twice_falls_back_to_the_name_test():
    # One pair on one input must not pay for both conditions: with the
    # number stated twice the link by number is off and the name decides.
    assert "pair" in _two_limit_checks("headcount", None)
    assert "pair" in _two_limit_checks("lending_unit_members", None)
    assert "pair" in _two_limit_checks("headcount", "equipment_total")
    assert "pair" not in _two_limit_checks("lending_unit_members", "lending_unit_tools")


# --- 4. A comparison behind a distant negation ------------------------------

NEGATED = (
    "A tenant unit is not eligible for a group loan when the tenant unit has "
    "not paid rent for more than one consecutive month."
)


def _negated_boundary(text: str):
    return sc._negated_comparison_boundary_interval(
        sc.authoritative_numeric_recall_text(text),
        extract_numeric_occurrences=_GROUNDING,
    )


def _negated_checks(
    operator: str,
    values: tuple[int, ...],
    *,
    name: str = "unpaid_rent_month_count",
    source: str = NEGATED,
    number: int = 1,
) -> set[str]:
    return _count_checks(
        source, operator, values, name=name, number=number, threshold=None
    )


def test_negated_comparison_is_not_given_a_direction():
    # "has not paid rent for more than one month" can mean unpaid for over a
    # month, or not (paid for over a month). The interval stays unread.
    condition = sc._source_exception_condition_text(NEGATED)
    assert condition.startswith("when the tenant unit has not paid rent")
    assert _interval(condition) is None
    negated = _negated_boundary(condition)
    assert negated is not None
    assert (negated.lower.value, negated.lower_inclusive, negated.upper) == (
        1,
        False,
        None,
    )


@pytest.mark.parametrize(
    "name", ["unpaid_rent_month_count", "consecutive_months_rent_unpaid", "arrears"]
)
def test_count_pair_on_the_boundary_of_a_negated_comparison_passes(name):
    # Either side may be the exception; the boundary lies between 1 and 2.
    assert _negated_checks(">", (1, 2), name=name) == set()
    assert _negated_checks("<=", (1, 2), name=name) == set()


@pytest.mark.parametrize(
    # The second name shares words with the condition and passes the name
    # test; the first is linked by the number alone.
    "name",
    ["unpaid_rent_month_count", "consecutive_months_rent_unpaid"],
)
def test_negated_comparison_still_refuses_another_boundary(name):
    # `>= 1` puts the boundary between 0 and 1: wrong under both readings.
    assert _negated_checks(">=", (0, 1), name=name) == {"pair"}
    assert _negated_checks("<", (0, 1), name=name) == {"pair"}
    # No case on the stated number.
    assert _negated_checks(">", (0, 5), name=name) == {"pair"}
    assert _negated_checks("<=", (0, 5), name=name) == {"pair"}
    # `> 2` with cases at 2 and 3: the pair misses the stated number.
    assert _negated_checks(">", (2, 3), name=name, number=2) == {"pair"}


@pytest.mark.parametrize(
    "source",
    [
        # Two negated comparisons.
        "when the tenant unit has not paid rent for more than one month and has "
        "not filed for more than one month.",
        # A modal in front of the negation.
        "when the tenant unit may have not paid rent for more than one "
        "consecutive month.",
        # A second negation between the verb and the comparison.
        "when the tenant unit has not paid or not offered rent for more than one "
        "consecutive month.",
        # A negation in the clause in front of the negated verb.
        "when no adult of the tenant unit has not paid rent for more than one "
        "consecutive month.",
        # The comparison is too far from the negated verb.
        "when the tenant unit has not paid any part of the rent that the lease "
        "sets for the unit for more than one consecutive month.",
        # No negated verb.
        "when the tenant unit has paid rent for more than one consecutive month.",
    ],
)
def test_other_negated_shapes_stay_unread(source):
    assert _negated_boundary(source) is None


def test_negated_reader_declines_an_adjacent_negation_the_parser_reads():
    # "is not more than 4" is a ceiling the interval parser reads itself.
    source = CONDITIONAL.format(wording="not more than 4")
    condition = sc._source_exception_condition_text(source)
    assert _negated_boundary(condition) is None
    assert _interval(condition) == _ceiling(4)
    assert _count_checks(source, "<=", (4, 5)) == set()
    assert _count_checks(source, "<", (4, 5)) == {"pair"}


# --- 5. A relative clause inside a conditional ------------------------------

WHO_HAS = (
    "When a tenant who has been absent for more than one consecutive month "
    "returns, the tenant is eligible for a group loan."
)


@pytest.mark.parametrize(
    "name",
    [
        "consecutive_months_absent",
        "tenant_absence_months",
        "tenant_absent_month_count",
        "time_away",
    ],
)
def test_who_has_clause_passes_with_a_count_input_of_any_name(name):
    kwargs = {"name": name, "number": 1, "threshold": None}
    assert _count_checks(WHO_HAS, ">", (1, 2), **kwargs) == set()
    # Wrong encodings of the same sentence stay refused.
    assert _count_checks(WHO_HAS, ">=", (0, 1), **kwargs) == {"boundary", "pair"}
    assert "boundary" in _count_checks(WHO_HAS, ">", (0, 5), **kwargs)
    assert "boundary" in _count_checks(WHO_HAS, "<=", (1, 2), **kwargs)


def test_who_has_clause_refuses_a_boolean_restatement():
    name = "tenant_absent_more_than_one_consecutive_month"
    payload = _payload(WHO_HAS, [_input(name, "Boolean")], name)
    cases = [_case({name: True}, True), _case({name: False}, False)]
    assert _checks(_issues(payload, WHO_HAS, cases)) == {"boundary", "pair"}

"""A `min(amount, cap)` cap is witnessed by a case at the cap.

Arizona DES FAA5 medical block 8 (local run f61921aa) says "Up to $40 per day
for lodging is allowed when the participant must travel over 50 miles". The
candidate gates on `travel_distance_miles > threshold` and caps the cost with
`min(max(0, daily_lodging_cost), medical_lodging_daily_cap)`. Its companion
tests set `daily_lodging_cost` to 30, 40 and 60 under qualifying conditions,
but the boundary check still refused it with `missing: source unit=40`: the
comparison gate made the capped cost drop out of the rule's selector names, so
no case could reach the cap reader.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    _numeric_profile_for_citation_path,
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

FIXTURE = Path(__file__).parent / "fixtures" / "az_batch3_medical_block_8"
PROVENANCE = json.loads((FIXTURE / "provenance.json").read_text())
CITATION = PROVENANCE["citation"]
PREFIX = "us-az:policies/des/faa5/na-medical-expenses-and-deduction/block-8"
OUTPUT = f"{PREFIX}#allowable_daily_medical_lodging_cost"
COST = f"{PREFIX}#input.daily_lodging_cost"
CAP_CASE = "qualifying_lodging_exactly_at_daily_cap"
BOUNDARY_FINDING = "missing: source unit=40"


def _read(name: str, key: str) -> str:
    text = (FIXTURE / name).read_text()
    assert hashlib.sha256(text.encode()).hexdigest() == PROVENANCE[key]["sha256"]
    return text


CANDIDATE = _read("candidate.yaml", "candidate")
CASES = yaml.safe_load(_read("candidate.test.yaml", "companion"))
SOURCE = _read("source.txt", "source")


def _issues(
    candidate: str = CANDIDATE,
    cases: list | None = None,
    source: str | None = None,
) -> list[str]:
    profile = _numeric_profile_for_citation_path(CITATION)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate,
        extract_named_scalars=extract_named_scalar_occurrences,
    )
    analysis = sc.analyze_complete_source_unit(
        candidate,
        SOURCE if source is None else source,
        corpus_citation_path=CITATION,
        test_cases=CASES if cases is None else cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile=profile
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile=profile
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
        artifact_numeric_values=tuple(value for _name, value in bindings),
        artifact_numeric_bindings=bindings,
    )
    return list(analysis.issues)


def _boundary_refused(issues: list[str]) -> bool:
    return any(BOUNDARY_FINDING in issue for issue in issues)


def _cap_case() -> dict:
    return next(case for case in CASES if case["name"] == CAP_CASE)


def _without_cap_case() -> list[dict]:
    return [case for case in CASES if case["name"] != CAP_CASE]


def test_fixture_is_the_recorded_refusal():
    assert "$40 per day" in SOURCE
    assert BOUNDARY_FINDING in PROVENANCE["recorded_issue"]
    assert _cap_case()["input"][COST] == 40
    assert _cap_case()["output"][OUTPUT] == 40


def test_case_at_the_cap_witnesses_the_source_boundary():
    issues = _issues()
    assert not _boundary_refused(issues)
    assert issues == []


@pytest.mark.parametrize(
    "cap_formula",
    [
        "min(medical_lodging_daily_cap, max(0, daily_lodging_cost))",
        "min(daily_lodging_cost, medical_lodging_daily_cap)",
    ],
)
def test_operand_order_and_clamp_do_not_matter(cap_formula):
    candidate = CANDIDATE.replace(
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)", cap_formula
    )
    assert candidate != CANDIDATE
    assert not _boundary_refused(_issues(candidate))


def test_cases_below_and_above_the_cap_alone_are_still_refused():
    assert _boundary_refused(_issues(cases=_without_cap_case()))


def test_a_blocked_case_at_the_cap_is_still_refused():
    blocked = json.loads(json.dumps(_cap_case()))
    blocked["input"][f"{PREFIX}#input.participant_has_lodging_cost"] = False
    blocked["output"][OUTPUT] = 0
    assert _boundary_refused(_issues(cases=[*_without_cap_case(), blocked]))


def test_a_cap_on_another_value_is_still_refused():
    candidate = CANDIDATE.replace(
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)",
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap + 5)",
    )
    assert candidate != CANDIDATE
    assert _boundary_refused(_issues(candidate))


CAP_FORMULA = "min(max(0, daily_lodging_cost), medical_lodging_daily_cap)"
GATE = "travel_distance_miles > medical_lodging_travel_distance_threshold:"


def _with_outputs(cases: list[dict], outputs: dict[str, float]) -> list[dict]:
    cases = copy.deepcopy(cases)
    for case in cases:
        if case["name"] in outputs:
            case["output"][OUTPUT] = outputs[case["name"]]
    return cases


def _with_reimbursement_input(candidate: str) -> str:
    return candidate.replace(
        "rules:\n",
        "  - name: reimbursement\n    entity: Person\n    dtype: Money\n"
        "    unit: USD\n    period: Day\nrules:\n",
        1,
    )


def _scaled_amount_never_reaches_the_cap():
    candidate = CANDIDATE.replace(
        CAP_FORMULA, "min(max(0, daily_lodging_cost) * 0.5, medical_lodging_daily_cap)"
    )
    cases = _with_outputs(
        CASES,
        {
            "qualifying_lodging_below_daily_cap": 15,
            CAP_CASE: 20,
            "qualifying_lodging_above_daily_cap": 30,
        },
    )
    return candidate, cases


def _net_amount_never_reaches_the_cap():
    candidate = _with_reimbursement_input(CANDIDATE).replace(
        CAP_FORMULA,
        "min(daily_lodging_cost - reimbursement, medical_lodging_daily_cap)",
    )
    cases = []
    for case in copy.deepcopy(CASES):
        if case["name"] == "qualifying_lodging_above_daily_cap":
            continue
        if COST in case["input"]:
            case["input"][f"{PREFIX}#input.reimbursement"] = 0
        if case["name"] == CAP_CASE:
            case["input"][f"{PREFIX}#input.reimbursement"] = 10
            case["output"][OUTPUT] = 30
        cases.append(case)
    return candidate, cases


def _cap_only_in_the_gate():
    candidate = CANDIDATE.replace(
        GATE,
        GATE[:-1] + " and min(daily_lodging_cost, medical_lodging_daily_cap) > 0:",
    ).replace(CAP_FORMULA, "max(0, daily_lodging_cost)")
    cases = _with_outputs(CASES, {"qualifying_lodging_above_daily_cap": 60})
    return candidate, cases


def _cap_term_multiplied_by_zero():
    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "max(0, daily_lodging_cost)"
        " + 0 * min(daily_lodging_cost, medical_lodging_daily_cap)",
    )
    cases = _with_outputs(CASES, {"qualifying_lodging_above_daily_cap": 60})
    return candidate, cases


@pytest.mark.parametrize(
    "mutant",
    [
        _scaled_amount_never_reaches_the_cap,
        _net_amount_never_reaches_the_cap,
        _cap_only_in_the_gate,
        _cap_term_multiplied_by_zero,
    ],
)
def test_a_case_at_40_that_does_not_reach_an_applied_cap_is_still_refused(mutant):
    candidate, cases = mutant()
    assert candidate != CANDIDATE
    assert any(case["input"].get(COST) == 40 for case in cases)
    assert _boundary_refused(_issues(candidate, cases))


def _with_input(candidate: str, name: str, dtype: str = "Money") -> str:
    return candidate.replace(
        "rules:\n",
        f"  - name: {name}\n    entity: Person\n    dtype: {dtype}\n"
        "    unit: USD\n    period: Day\nrules:\n",
        1,
    )


def _with_input_value(cases: list[dict], name: str, value: float) -> list[dict]:
    cases = copy.deepcopy(cases)
    for case in cases:
        if COST in case["input"]:
            case["input"][f"{PREFIX}#input.{name}"] = value
    return cases


def _cap_from_an_input():
    """The cap uses a case input, not the source-bound parameter."""

    candidate = _with_input(CANDIDATE, "daily_cap_input").replace(
        CAP_FORMULA, "min(max(0, daily_lodging_cost), daily_cap_input)"
    )
    return candidate, _with_input_value(CASES, "daily_cap_input", 40)


def _amount_just_above_the_cap():
    """The capped amount is 40.000001, not the cap."""

    cases = copy.deepcopy(CASES)
    _cap_case_of(cases)["input"][COST] = 40.000001
    return CANDIDATE, cases


def _boundary_value_only_after_scaling():
    """The amount reaches 40 only as 20 * 2; no input is at the source value."""

    candidate = _with_input(CANDIDATE, "share", "Decimal").replace(
        CAP_FORMULA,
        "min(max(0, daily_lodging_cost) * share, medical_lodging_daily_cap)",
    )
    cases = _with_input_value(CASES, "share", 1)
    cap_case = _cap_case_of(cases)
    cap_case["input"][f"{PREFIX}#input.share"] = 2
    cap_case["input"][COST] = 20
    return candidate, cases


def _outer_plateau_below_the_cap():
    """An outer `min(..., 30)` makes the output flat below the cap."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "min(min(max(0, daily_lodging_cost), medical_lodging_daily_cap), 30)",
    )
    cases = _with_outputs(
        CASES, {CAP_CASE: 30, "qualifying_lodging_above_daily_cap": 30}
    )
    return candidate, cases


def _flat_just_below_the_cap():
    """The output is already at the cap at 39, so 40 is no kink.

    The leaf is a `max` of two `min` calls, so the whole-leaf rule also refuses
    it.
    """

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "max(min(max(0, daily_lodging_cost), medical_lodging_daily_cap),"
        " min(max(0, daily_lodging_cost) + 1, medical_lodging_daily_cap))",
    )
    cases = _with_outputs(CASES, {"qualifying_lodging_below_daily_cap": 31})
    return candidate, cases


def _output_still_rising_above_the_cap():
    """The clamp applies inside a larger leaf; the output rises above 40 too."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        f"{CAP_FORMULA} + max(0, daily_lodging_cost) - medical_lodging_daily_cap",
    )
    cases = _with_outputs(
        CASES,
        {
            "qualifying_lodging_below_daily_cap": 20,
            "qualifying_lodging_above_daily_cap": 60,
        },
    )
    return candidate, cases


def _output_flat_on_both_sides_of_the_cap():
    """The clamp applies inside a larger leaf; the output is 40 at 39 too."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        f"{CAP_FORMULA} + max(0, medical_lodging_daily_cap - daily_lodging_cost)",
    )
    cases = _with_outputs(CASES, {"qualifying_lodging_below_daily_cap": 40})
    return candidate, cases


def _plateau_from_an_input_limit():
    """The cap term is zeroed; an input limit of 40 makes the plateau."""

    candidate = _with_input(CANDIDATE, "budget_limit").replace(
        CAP_FORMULA,
        "min(max(0, daily_lodging_cost), medical_lodging_daily_cap) * 0"
        " + min(max(0, daily_lodging_cost), budget_limit)",
    )
    return candidate, _with_input_value(CASES, "budget_limit", 40)


def _plateau_from_an_input_limit_with_another_cap_use():
    """As above, plus ``+ cap - 40``, which is zero at the cap.

    Lowering the cap wherever the parameter appears changes this unrelated
    term, so a check that lowers it there takes the zeroed clamp for the cause
    of the plateau.
    """

    candidate, cases = _plateau_from_an_input_limit()
    return (
        candidate.replace(
            "budget_limit)", "budget_limit) + medical_lodging_daily_cap - 40"
        ),
        cases,
    )


def _output_flat_on_both_sides_of_a_whole_leaf_cap():
    """The leaf is the `min`, but its amount is 40 or more at 39 and at 41."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "min(max(daily_lodging_cost, 2 * medical_lodging_daily_cap"
        " - daily_lodging_cost), medical_lodging_daily_cap)",
    )
    cases = _with_outputs(CASES, {"qualifying_lodging_below_daily_cap": 40})
    return candidate, cases


def _output_falling_above_a_whole_leaf_cap():
    """The leaf is the `min`, but its amount is 39 at 39 and at 41."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "min(min(daily_lodging_cost, 2 * medical_lodging_daily_cap"
        " - daily_lodging_cost), medical_lodging_daily_cap)",
    )
    cases = _with_outputs(CASES, {"qualifying_lodging_above_daily_cap": 20})
    return candidate, cases


def _cap_of_45_with_the_amount_shifted_by_5():
    """The cap is `cap + 5` = 45, which the source does not state.

    The amount is `cost + 5`, so it reaches 45 at a cost of 40 and the output
    kinks there, but a $45 cap cannot witness the $40 boundary.
    """

    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        "min(max(0, daily_lodging_cost) + 5, medical_lodging_daily_cap + 5)",
    )
    cases = _with_outputs(
        CASES,
        {
            "qualifying_lodging_below_daily_cap": 35,
            CAP_CASE: 45,
            "qualifying_lodging_above_daily_cap": 45,
        },
    )
    return candidate, cases


def _cap_case_of(cases: list[dict]) -> dict:
    return next(case for case in cases if case["name"] == CAP_CASE)


@pytest.mark.parametrize(
    "mutant",
    [
        _cap_from_an_input,
        _amount_just_above_the_cap,
        _boundary_value_only_after_scaling,
        _outer_plateau_below_the_cap,
        _flat_just_below_the_cap,
        _output_still_rising_above_the_cap,
        _output_flat_on_both_sides_of_the_cap,
        _output_flat_on_both_sides_of_a_whole_leaf_cap,
        _output_falling_above_a_whole_leaf_cap,
        _cap_of_45_with_the_amount_shifted_by_5,
        _plateau_from_an_input_limit,
        _plateau_from_an_input_limit_with_another_cap_use,
    ],
)
def test_each_cap_witness_condition_is_required(mutant):
    candidate, cases = mutant()
    assert candidate != CANDIDATE or cases != CASES
    assert _boundary_refused(_issues(candidate, cases))


def test_another_use_of_the_cap_does_not_make_a_zeroed_clamp_the_cause():
    candidate, cases = _plateau_from_an_input_limit_with_another_cap_use()
    assert "budget_limit) + medical_lodging_daily_cap - 40" in candidate
    assert _issues(candidate, cases) == [
        "[complete-source-unit:tests] Companion tests do not exercise every"
        f" source-stated boundary input; {BOUNDARY_FINDING}."
    ]


AMOUNT = "max(0, daily_lodging_cost)"
CLAMP = f"min({AMOUNT}, medical_lodging_daily_cap)"
BUDGET = f"min({AMOUNT}, budget_limit)"
PRINCIPAL_RULE = "  - name: allowable_daily_medical_lodging_cost\n"


def _with_parameter(candidate: str, name: str, formula: str) -> str:
    assert PRINCIPAL_RULE in candidate
    return candidate.replace(
        PRINCIPAL_RULE,
        f"  - name: {name}\n    kind: parameter\n    dtype: Money\n"
        "    period: Day\n    unit: USD\n    versions:\n"
        "      - effective_from: '2026-08-01'\n        formula: |-\n"
        f"          {formula}\n" + PRINCIPAL_RULE,
        1,
    )


def _budget_leaf(leaf: str, parameter: tuple[str, str] | None = None):
    """A leaf that adds an input limit of 40 to a term that is zero for every cost."""

    candidate = _with_input(CANDIDATE, "budget_limit").replace(CAP_FORMULA, leaf)
    if parameter is not None:
        candidate = _with_parameter(candidate, *parameter)
    return candidate, _with_input_value(CASES, "budget_limit", 40)


@pytest.mark.parametrize(
    ("leaf", "parameter"),
    [
        pytest.param(f"{CLAMP} - {CLAMP} + {BUDGET}", None, id="m1-same-clamp-cancels"),
        pytest.param(
            f"{CLAMP} + max({AMOUNT}, medical_lodging_daily_cap)"
            f" - {AMOUNT} - medical_lodging_daily_cap + {BUDGET}",
            None,
            id="m2-min-plus-max-identity",
        ),
        pytest.param(
            f"max({CLAMP}, 0) - {CLAMP} + {BUDGET}", None, id="m9-cancel-inside-max"
        ),
        pytest.param(
            f"2 * {CLAMP} - {CLAMP} - {CLAMP} + {BUDGET}",
            None,
            id="m10-doubled-and-halved",
        ),
        pytest.param(
            f"{CLAMP} - min({AMOUNT}, cap_alias) + {BUDGET}",
            ("cap_alias", "medical_lodging_daily_cap"),
            id="m4b-alias-parameter-cancels",
        ),
        pytest.param(
            f"{CLAMP} - min({AMOUNT}, other_forty) + {BUDGET}",
            ("other_forty", "40"),
            id="m5-second-parameter-cancels",
        ),
        pytest.param(
            f"{CLAMP} - min({AMOUNT}, 40) + {BUDGET}", None, id="m6-literal-cancels"
        ),
    ],
)
def test_a_cancelled_clamp_inside_a_larger_leaf_is_not_credited(leaf, parameter):
    """The cap clamp is cancelled for every cost; the input limit makes the plateau.

    Lowering the cap in the credited `min` call alone still changes these
    leaves, because the cancelling term keeps its value. So only a leaf that is
    the `min` call itself is credited.
    """

    candidate, cases = _budget_leaf(leaf, parameter)
    assert CLAMP in candidate
    assert _issues(candidate, cases) == [
        "[complete-source-unit:tests] Companion tests do not exercise every"
        f" source-stated boundary input; {BOUNDARY_FINDING}."
    ]


@pytest.mark.parametrize(
    "leaf",
    [f"max({CAP_FORMULA}, {CAP_FORMULA})", f"{CAP_FORMULA} * 1"],
    ids=["duplicate-clamp-in-max", "clamp-times-one"],
)
def test_a_cap_clamp_inside_a_larger_leaf_fails_closed(leaf):
    """Accepted trade-off: only a leaf that is the `min` call itself is credited."""

    candidate = CANDIDATE.replace(CAP_FORMULA, leaf)
    assert candidate != CANDIDATE
    assert _boundary_refused(_issues(candidate))


GATES = (
    "participant_has_lodging_cost"
    " and lodging_is_to_obtain_medical_treatment_or_services"
    " and participant_must_travel"
    " and travel_distance_miles > medical_lodging_travel_distance_threshold"
)
CAP_BRANCH = f"if {GATES}:\n            {CAP_FORMULA}\n          else: 0"
SWITCH = f"{PREFIX}#cost_within_switch"


def _branch_switch_one_step_above_the_cap():
    """At a cost of 40 the leaf is the cap; at 41 another branch returns an input.

    An asserted Boolean dependency `cost <= switch_point` (40) gates the cap
    branch, and an `elif` returns the input `fallback_amount` (40). The output
    is 39, 40, 40 at costs of 39, 40 and 41, but at 41 the plateau comes from
    `fallback_amount`, not from the cap.
    """

    candidate = _with_input(_with_input(CANDIDATE, "switch_point"), "fallback_amount")
    assert CAP_BRANCH in candidate
    candidate = candidate.replace(
        CAP_BRANCH,
        f"if {GATES} and cost_within_switch:\n            {CAP_FORMULA}\n"
        f"          elif {GATES}:\n            fallback_amount\n          else: 0",
    ).replace(
        PRINCIPAL_RULE,
        "  - name: cost_within_switch\n    kind: derived\n    entity: Person\n"
        "    dtype: Boolean\n    period: Day\n    versions:\n"
        "      - effective_from: '2026-08-01'\n        formula: |-\n"
        "          daily_lodging_cost <= switch_point\n" + PRINCIPAL_RULE,
        1,
    )
    cases = _with_input_value(CASES, "switch_point", 40)
    cases = _with_input_value(cases, "fallback_amount", 40)
    for case in cases:
        if COST in case["input"]:
            case["output"][SWITCH] = case["input"][COST] <= 40
    return candidate, cases


def test_another_branch_one_step_above_the_cap_is_not_a_kink():
    candidate, cases = _branch_switch_one_step_above_the_cap()
    assert "elif" in candidate and "cost_within_switch:" in candidate
    assert _cap_case_of(cases)["output"][SWITCH] is True
    assert _issues(candidate, cases) == [
        "[complete-source-unit:tests] Companion tests do not exercise every"
        f" source-stated boundary input; {BOUNDARY_FINDING}."
    ]


def test_the_cap_is_measured_with_the_arithmetic_that_executes_the_formula():
    """A float evaluator reads this cap as $40; the formula executes $45."""

    shift = "5 * (1 + floor(0.3 - 0.1 - 0.2))"
    candidate = CANDIDATE.replace(
        CAP_FORMULA,
        f"min(max(0, daily_lodging_cost + {shift}), "
        f"medical_lodging_daily_cap + {shift})",
    )
    assert candidate != CANDIDATE
    cases = _with_outputs(
        CASES,
        {
            "qualifying_lodging_below_daily_cap": 35,
            CAP_CASE: 45,
            "qualifying_lodging_above_daily_cap": 45,
        },
    )
    issues = _issues(candidate, cases)
    assert len(issues) == 1 and BOUNDARY_FINDING in issues[0]


_FLOAT_ZERO = "5 * floor(0.3 - 0.1 - 0.2)"


@pytest.mark.parametrize(
    ("leaf", "outputs"),
    [
        pytest.param(
            f"min((daily_lodging_cost - 39) * (45 + {_FLOAT_ZERO}),"
            " medical_lodging_daily_cap)",
            (-405, 40, 40),
            id="amount-only",
        ),
        pytest.param(
            "min(min(max(0, daily_lodging_cost), medical_lodging_daily_cap),"
            f" medical_lodging_daily_cap + 5 + {_FLOAT_ZERO})",
            (30, 40, 40),
            id="cap-only",
        ),
    ],
)
def test_each_operand_is_measured_with_the_executing_arithmetic(leaf, outputs):
    """One operand alone read in float arithmetic would credit these leaves."""

    candidate = CANDIDATE.replace(CAP_FORMULA, leaf)
    assert candidate != CANDIDATE
    below, at, above = outputs
    cases = _with_outputs(
        CASES,
        {
            "qualifying_lodging_below_daily_cap": below,
            CAP_CASE: at,
            "qualifying_lodging_above_daily_cap": above,
        },
    )
    issues = _issues(candidate, cases)
    assert len(issues) == 1 and BOUNDARY_FINDING in issues[0]


# The cap is bound to its own source occurrence, not only to its number.

CAP_AND_GATE_SENTENCE = (
    "Up to $40 per day for lodging is allowed when the participant "
    "must travel over 50 miles to obtain medical treatments or services."
)
CAP_SENTENCE = "Up to $40 per day for lodging is allowed."
CAP_EXCERPT = 'excerpt: "Up to $40 per day for lodging is allowed"'
THRESHOLD_EXCERPT = "the participant must travel over 50 miles"
DISTANCE = f"{PREFIX}#input.travel_distance_miles"
MISSING_40 = [
    "[complete-source-unit:tests] Companion tests do not exercise every"
    f" source-stated boundary input; {BOUNDARY_FINDING}."
]
BOTH_MISSING_40 = [MISSING_40[0].replace("unit=40.", "unit=40, source unit=40.")]


def _refusals(monkeypatch, candidate, cases, source) -> tuple[list[str], list[str]]:
    """Issues with every guard, then with the stated-once guard switched off.

    Two boundaries of 40 are refused first because the unit states the number
    twice. The second result shows what the binding rules alone do.
    """

    issues = _issues(candidate, cases, source)
    with monkeypatch.context() as patch:
        patch.setattr(sc, "_boundary_value_is_stated_once", lambda *_a, **_k: True)
        return issues, _issues(candidate, cases, source)


# A condition value of 40 also refuses the lodging cap of 40: the binding
# rules do not credit a cap number that any condition reads by value.
REFUSED_BY_EACH_GUARD = (BOTH_MISSING_40, BOTH_MISSING_40)


def _second_forty(
    threshold_sentence: str,
    *,
    gate: str,
    distances: dict[int, int],
    threshold_first: bool = False,
):
    """Restate the fixture so a second obligation also has the number 40.

    The lodging cap keeps its own sentence. `threshold_sentence` states the
    travel gate with the value 40, `gate` is the comparison the candidate
    uses, and `distances` moves the companion distance inputs.
    """

    sentences = (
        f"{threshold_sentence} {CAP_SENTENCE}"
        if threshold_first
        else f"{CAP_SENTENCE} {threshold_sentence}"
    )
    assert CAP_AND_GATE_SENTENCE in SOURCE
    source = SOURCE.replace(CAP_AND_GATE_SENTENCE, sentences)
    candidate = (
        CANDIDATE.replace(CAP_AND_GATE_SENTENCE, sentences)
        .replace(THRESHOLD_EXCERPT, threshold_sentence.rstrip("."))
        .replace("\n          50\n", "\n          40\n")
        .replace("travel_distance_miles >", f"travel_distance_miles {gate}")
        .replace(
            PROVENANCE["source"]["sha256"],
            hashlib.sha256(source.encode()).hexdigest(),
        )
    )
    assert CAP_EXCERPT in candidate and CAP_AND_GATE_SENTENCE not in candidate
    assert f"travel_distance_miles {gate} medical_lodging" in candidate
    cases = copy.deepcopy(CASES)
    for case in cases:
        if DISTANCE in case["input"]:
            case["input"][DISTANCE] = distances[case["input"][DISTANCE]]
    return candidate, cases, source


def _distance_inputs(cases: list[dict]) -> set[int]:
    return {case["input"][DISTANCE] for case in cases if DISTANCE in case["input"]}


MORE_THAN_40_MILES = (
    "Travel distance is more than 40 miles for lodging to be allowable."
)
UNTESTED_LOWER_GATE = {49: 38, 50: 39, 51: 51}


@pytest.mark.parametrize(
    ("threshold_sentence", "gate", "distances", "threshold_first"),
    [
        pytest.param(
            MORE_THAN_40_MILES, ">=", UNTESTED_LOWER_GATE, False, id="lower-limit"
        ),
        pytest.param(
            "Travel distance is less than 40 miles for lodging to be allowable.",
            "<=",
            {49: 45, 50: 41, 51: 30},
            False,
            id="upper-limit",
        ),
        pytest.param(
            "Travel cost is more than $40 for lodging to be allowable.",
            ">=",
            UNTESTED_LOWER_GATE,
            False,
            id="both-in-usd",
        ),
        pytest.param(
            MORE_THAN_40_MILES, ">=", UNTESTED_LOWER_GATE, True, id="threshold-first"
        ),
    ],
)
def test_the_cap_case_does_not_witness_another_threshold_with_the_same_number(
    monkeypatch, threshold_sentence, gate, distances, threshold_first
):
    """The wrong inclusive gate pays at 40, and no case sets the gate input to 40."""

    candidate, cases, source = _second_forty(
        threshold_sentence,
        gate=gate,
        distances=distances,
        threshold_first=threshold_first,
    )
    assert source.count("40") == 2
    assert 40 not in _distance_inputs(cases)
    assert _cap_case_of(cases)["input"][COST] == 40
    # The unit states 40 as a boundary twice, so the cap path credits neither.
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD


def test_the_other_threshold_is_witnessed_by_its_own_cases():
    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">", distances={49: 39, 50: 40, 51: 41}
    )
    assert {40, 41} <= _distance_inputs(cases)
    assert _issues(candidate, cases, source) == []


def test_a_cap_excerpt_that_states_the_number_twice_binds_no_occurrence():
    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    both_sentences = f"{CAP_SENTENCE} {MORE_THAN_40_MILES}"
    candidate = candidate.replace(CAP_EXCERPT, f'excerpt: "{both_sentences}"')
    assert CAP_EXCERPT not in candidate
    # Neither 40 is credited now: the cap has no single occurrence, and no
    # case sets the travel distance to 40.
    assert _issues(candidate, cases, source) == BOTH_MISSING_40


def test_a_cap_excerpt_that_occurs_twice_in_the_source_binds_no_occurrence():
    repeated = f"{CAP_AND_GATE_SENTENCE} {CAP_SENTENCE}"
    source = SOURCE.replace(CAP_AND_GATE_SENTENCE, repeated)
    candidate = CANDIDATE.replace(CAP_AND_GATE_SENTENCE, repeated).replace(
        PROVENANCE["source"]["sha256"], hashlib.sha256(source.encode()).hexdigest()
    )
    assert source.count("Up to $40 per day for lodging is allowed") == 2
    # Binding the first place alone would credit one of the two $40 caps.
    assert _issues(candidate, source=source) == BOTH_MISSING_40


def test_a_cap_without_its_own_source_excerpt_binds_no_occurrence():
    """A literal cap has no proof excerpt that says which 40 it is."""

    candidate = CANDIDATE.replace(CAP_FORMULA, "min(max(0, daily_lodging_cost), 40)")
    assert candidate != CANDIDATE
    assert _boundary_refused(_issues(candidate))


def test_a_cap_excerpt_from_another_source_binds_no_occurrence():
    cap_atom = f"corpus_citation_path: {CITATION}\n              {CAP_EXCERPT}"
    assert CANDIDATE.count(cap_atom) == 1
    candidate = CANDIDATE.replace(
        cap_atom, f"corpus_citation_path: {CITATION}-other\n              {CAP_EXCERPT}"
    )
    assert _boundary_refused(_issues(candidate))


def test_a_computed_cap_binds_no_occurrence():
    """A name inside a larger cap expression does not say which 40 the cap is."""

    candidate = CANDIDATE.replace(
        CAP_FORMULA, "min(max(0, daily_lodging_cost), medical_lodging_daily_cap * 1)"
    )
    assert candidate != CANDIDATE
    assert _issues(candidate) == MISSING_40


# The bound occurrence must be a cap obligation and nothing else.

ALT = f"{PREFIX}#input.alt"
INCLUSIVE_GATES = GATES.replace("> medical", ">= medical")
TRAVEL_THRESHOLD = "medical_lodging_travel_distance_threshold"


def _alternate_cap_branch(cap_name: str, candidate: str, cases: list[dict]):
    """Add a branch on a Boolean input that caps the cost with `cap_name`.

    The branch has its own cases below, at and above the cap. The first
    branch keeps the lodging cap, so the lodging $40 stays witnessed.
    """

    branch = f"if {INCLUSIVE_GATES}:\n            {CAP_FORMULA}\n          else: 0"
    assert candidate.count(branch) == 1
    candidate = _with_input(candidate, "alt", "Boolean").replace(
        branch,
        f"if {INCLUSIVE_GATES} and alt:\n"
        f"            min(max(0, daily_lodging_cost), {cap_name})\n"
        f"          elif {INCLUSIVE_GATES}:\n            {CAP_FORMULA}\n"
        "          else: 0",
    )
    cases = copy.deepcopy(cases)
    alternate_cases = []
    for case in cases:
        if COST not in case["input"]:
            continue
        case["input"][ALT] = False
        if case["name"] in {
            "qualifying_lodging_below_daily_cap",
            CAP_CASE,
            "qualifying_lodging_above_daily_cap",
        }:
            alternate = copy.deepcopy(case)
            alternate["name"] += "_alt"
            alternate["input"][ALT] = True
            alternate_cases.append(alternate)
    assert len(alternate_cases) == 3
    return candidate, [*cases, *alternate_cases]


def test_a_cap_that_a_condition_also_reads_is_not_a_cap_obligation(monkeypatch):
    """The gate threshold used as a second cap does not test the gate."""

    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    candidate, cases = _alternate_cap_branch(TRAVEL_THRESHOLD, candidate, cases)
    assert 40 not in _distance_inputs(cases)
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD


def test_a_cap_whose_occurrence_another_parameter_claims_is_not_credited(
    monkeypatch,
):
    """A second parameter of 40 proved by the gate sentence is the gate's 40."""

    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    gate_sentence = MORE_THAN_40_MILES.rstrip(".")
    assert candidate.count(f'excerpt: "{gate_sentence}"') == 1
    candidate = candidate.replace(
        PRINCIPAL_RULE,
        "  - name: alt_cap\n    kind: parameter\n    dtype: Money\n"
        "    period: Day\n    unit: USD\n    metadata:\n      proof:\n"
        "        atoms:\n          - path: versions[0].formula\n"
        "            kind: amount\n            source:\n"
        f"              corpus_citation_path: {CITATION}\n"
        f'              excerpt: "{gate_sentence}"\n'
        "    versions:\n      - effective_from: '2026-08-01'\n"
        "        formula: |-\n          40\n" + PRINCIPAL_RULE,
        1,
    )
    candidate, cases = _alternate_cap_branch("alt_cap", candidate, cases)
    assert 40 not in _distance_inputs(cases)
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD

    # The gate parameter now reads `alt_cap` and has no formula proof of its
    # own. The gate still reads the cap, through that parameter.
    threshold_rule = (
        f'excerpt: "{gate_sentence}"\n    versions:\n'
        "      - effective_from: '2026-08-01'\n        formula: |-\n          40\n"
        "  - name: alt_cap\n"
    )
    threshold_atom = (
        "          - path: versions[0].formula\n            kind: parameter\n"
    )
    assert candidate.count(threshold_rule) == 1
    assert candidate.count(threshold_atom) == 1
    aliased = candidate.replace(
        threshold_rule, threshold_rule.replace("          40\n", "          alt_cap\n")
    ).replace(threshold_atom, threshold_atom.replace(".formula", ".effective_from"))
    assert _refusals(monkeypatch, aliased, cases, source) == REFUSED_BY_EACH_GUARD


def _two_forties_with_the_threshold_as_a_second_cap():
    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    candidate, cases = _alternate_cap_branch(TRAVEL_THRESHOLD, candidate, cases)
    return candidate, cases, source


GATE_COMPARISON = f"travel_distance_miles >= {TRAVEL_THRESHOLD}"


def _helper_rule(name: str, dtype: str, formula: str) -> str:
    return (
        f"  - name: {name}\n    kind: derived\n    entity: Person\n"
        f"    dtype: {dtype}\n    period: Day\n    versions:\n"
        "      - effective_from: '2026-08-01'\n        formula: |-\n"
        f"          {formula}\n"
    )


@pytest.mark.parametrize(
    ("helper", "formula", "gate", "helper_output"),
    [
        pytest.param(
            "excess_miles",
            f"travel_distance_miles - {TRAVEL_THRESHOLD}",
            "excess_miles >= 0",
            lambda miles: miles - 40,
            id="difference",
        ),
        pytest.param(
            "threshold_plus_zero",
            f"{TRAVEL_THRESHOLD} + 0",
            "travel_distance_miles >= threshold_plus_zero",
            lambda miles: 40,
            id="threshold-plus-zero",
        ),
        pytest.param(
            "excess_miles",
            f"max(0, travel_distance_miles - {TRAVEL_THRESHOLD} + 1)",
            "excess_miles > 0",
            lambda miles: max(0, miles - 39),
            id="positive-part",
        ),
        pytest.param(
            "gap",
            f"{TRAVEL_THRESHOLD} - travel_distance_miles",
            "not (gap > 0)",
            lambda miles: 40 - miles,
            id="negated-gap",
        ),
    ],
)
def test_a_threshold_behind_a_helper_rule_is_still_a_condition_value(
    monkeypatch, helper, formula, gate, helper_output
):
    """The gate reads the threshold through a derived rule that computes with it."""

    candidate, cases, source = _two_forties_with_the_threshold_as_a_second_cap()
    assert candidate.count(GATE_COMPARISON) == 2
    candidate = candidate.replace(GATE_COMPARISON, gate).replace(
        PRINCIPAL_RULE, _helper_rule(helper, "Decimal", formula) + PRINCIPAL_RULE, 1
    )
    cases = copy.deepcopy(cases)
    for case in cases:
        if DISTANCE in case["input"]:
            case["output"][f"{PREFIX}#{helper}"] = helper_output(
                case["input"][DISTANCE]
            )
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD


GATE_SENTENCE = MORE_THAN_40_MILES.rstrip(".")
GATE_PARAMETER_ATOM = (
    "          - path: versions[0].formula\n            kind: parameter\n"
)


def _alt_cap_parameter(excerpt: str = GATE_SENTENCE) -> str:
    return (
        "  - name: alt_cap\n    kind: parameter\n    dtype: Money\n"
        "    period: Day\n    unit: USD\n    metadata:\n      proof:\n"
        "        atoms:\n          - path: versions[0].formula\n"
        "            kind: amount\n            source:\n"
        f"              corpus_citation_path: {CITATION}\n"
        f'              excerpt: "{excerpt}"\n'
        "    versions:\n      - effective_from: '2026-08-01'\n"
        "        formula: |-\n          40\n"
    )


def _gate_proof_moved_off_the_formula(candidate: str) -> str:
    assert candidate.count(GATE_PARAMETER_ATOM) == 1
    return candidate.replace(
        GATE_PARAMETER_ATOM, GATE_PARAMETER_ATOM.replace(".formula", ".effective_from")
    )


def _gate_proof_without_the_number(candidate: str) -> str:
    gate_excerpt = f'kind: parameter\n            source:\n              corpus_citation_path: {CITATION}\n              excerpt: "{GATE_SENTENCE}"'
    assert candidate.count(gate_excerpt) == 1
    return candidate.replace(
        gate_excerpt,
        gate_excerpt.replace(GATE_SENTENCE, "miles for lodging to be allowable"),
    )


@pytest.mark.parametrize(
    "hide_the_gate_claim",
    [_gate_proof_moved_off_the_formula, _gate_proof_without_the_number],
)
def test_a_condition_value_equal_to_the_cap_needs_no_proof_claim(
    monkeypatch, hide_the_gate_claim
):
    """The gate parameter is 40 but its proof no longer claims the gate's number.

    A second parameter `alt_cap` of 40, proved by the gate sentence, caps one
    branch. No other parameter's proof holds that number now, so only the
    gate parameter's value shows that the number is a comparison threshold.
    """

    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    candidate = hide_the_gate_claim(
        candidate.replace(PRINCIPAL_RULE, _alt_cap_parameter() + PRINCIPAL_RULE, 1)
    )
    candidate, cases = _alternate_cap_branch("alt_cap", candidate, cases)
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD


def test_a_literal_condition_value_equal_to_the_cap_is_refused(monkeypatch):
    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    candidate = candidate.replace(
        PRINCIPAL_RULE, _alt_cap_parameter() + PRINCIPAL_RULE, 1
    )
    candidate, cases = _alternate_cap_branch("alt_cap", candidate, cases)
    assert candidate.count(GATE_COMPARISON) == 2
    candidate = _gate_proof_moved_off_the_formula(candidate).replace(
        GATE_COMPARISON, "travel_distance_miles >= 40"
    )
    assert _refusals(monkeypatch, candidate, cases, source) == REFUSED_BY_EACH_GUARD


def test_the_whole_sentence_as_the_proof_of_both_parameters_is_credited():
    """The natural encoding: the cap and the threshold both quote the sentence."""

    whole = f'excerpt: "{CAP_AND_GATE_SENTENCE}"'
    candidate = CANDIDATE.replace(CAP_EXCERPT, whole).replace(
        f'excerpt: "{THRESHOLD_EXCERPT}"', whole
    )
    assert candidate.count(whole) == 3
    assert _issues(candidate) == []
    assert _issues(candidate, _without_cap_case()) == MISSING_40


def test_a_second_boundary_of_the_same_number_stops_the_cap_credit(monkeypatch):
    """Only the stated-once guard refuses here: no condition reads a 40."""

    parking = " Parking of up to $40 per day is not covered."
    source = SOURCE.replace(CAP_AND_GATE_SENTENCE, CAP_AND_GATE_SENTENCE + parking)
    candidate = CANDIDATE.replace(
        f"{CAP_AND_GATE_SENTENCE}\n", f"{CAP_AND_GATE_SENTENCE}{parking}\n"
    ).replace(
        PROVENANCE["source"]["sha256"], hashlib.sha256(source.encode()).hexdigest()
    )
    assert source.count("$40") == 2 and candidate.count(parking) == 1
    assert _refusals(monkeypatch, candidate, CASES, source) == (
        BOTH_MISSING_40,
        MISSING_40,
    )


# The parts of the binding, each alone.

UNIT = "Up to $40 per day for lodging. Meals: $40 per day. Over 50 miles."
LODGING_40 = (UNIT.index("$40") + 1, UNIT.index("$40") + 3)


def _parameter(value: str, *excerpts: str, path: str = CITATION) -> dict:
    return {
        "kind": "parameter",
        "versions": [{"formula": value}],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "source": {"corpus_citation_path": path, "excerpt": excerpt},
                    }
                    for excerpt in excerpts
                ]
            }
        },
    }


def _derived(formula: str) -> dict:
    return {"kind": "derived", "versions": [{"formula": formula}]}


def _cap_is_bound(
    parameters: dict[str, dict],
    environment: dict[str, float],
    principal: str = "if miles > limit:\n  min(cost, cap)\nelse: 0",
    **derived: str,
) -> bool:
    parameters = {
        "cap": _parameter("40", "Up to $40 per day for lodging"),
        "limit": _parameter("50", "Over 50 miles"),
        **parameters,
    }
    binding = sc._minimum_cap_source_binding(
        parameters,
        principal_rules={
            "out": _derived(principal),
            **{name: _derived(formula) for name, formula in derived.items()},
        },
        source_text=UNIT,
        corpus_citation_path=CITATION,
    )
    return sc._minimum_cap_is_bound_to_source_span(
        sc._parse_formula_expression("cap"),
        40.0,
        LODGING_40,
        cap_binding=binding,
        environment={"cap": 40, "limit": 50, **environment},
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text,
            profile=_numeric_profile_for_citation_path(CITATION),
        ),
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_the_synthetic_cap_is_bound():
    assert _cap_is_bound({}, {})


def test_a_repeated_cap_excerpt_has_no_single_place():
    assert UNIT.count("$40 per day") == 2
    assert not _cap_is_bound({"cap": _parameter("40", "$40 per day")}, {})


def test_a_cap_proof_that_states_the_value_twice_has_no_single_number():
    both = "Up to $40 per day for lodging. Meals: $40 per day"
    assert not _cap_is_bound({"cap": _parameter("40", both)}, {})
    assert not _cap_is_bound(
        {
            "cap": _parameter(
                "40", "Up to $40 per day for lodging", "Meals: $40 per day"
            )
        },
        {},
    )


def test_a_cap_name_in_a_condition_is_refused_whatever_value_it_resolves_to():
    gate_reads_cap = "if miles > cap:\n  min(cost, cap)\nelse: 0"
    assert not _cap_is_bound({}, {}, principal=gate_reads_cap)
    assert not _cap_is_bound({}, {"cap": 41}, principal=gate_reads_cap)
    assert _cap_is_bound({}, {"cap": 41})


def test_another_parameter_with_no_known_value_claims_the_number():
    unused = {"unused": _parameter("50", "Up to $40 per day for lodging. Meals")}
    assert _cap_is_bound(unused, {"unused": 50})
    assert not _cap_is_bound(unused, {})
    assert not _cap_is_bound(unused, {"unused": 40})


def test_each_place_of_a_repeated_excerpt_is_a_claim():
    assert UNIT.count("$40 per day") == 2
    meals = {"meals": _parameter("40", "$40 per day")}
    assert not _cap_is_bound(meals, {"meals": 40})
    assert _cap_is_bound(
        {"meals": _parameter("40", "Meals: $40 per day")}, {"meals": 40}
    )


def test_a_condition_parameter_with_no_known_value_is_a_use():
    assert _cap_is_bound({}, {"limit": 50})
    assert not _cap_is_bound({}, {"limit": None})


@pytest.mark.parametrize(
    ("principal", "derived"),
    [
        pytest.param(
            "min(cost, cap)", {"allowed": "miles >= cap"}, id="comparison-line-only"
        ),
        pytest.param(
            "if allowed:\n  min(cost, cap)\nelse: 0",
            {"allowed": "holds(far)", "far": "positive(miles - cap)"},
            id="header-with-no-operator",
        ),
        pytest.param(
            "if miles > limit:\n  min(cost, cap)\nelse: 0",
            {"gap": "max(0, miles - cap + 1)", "second": "if gap: 1\nelse: 0"},
            id="through-a-computing-rule",
        ),
        pytest.param(
            "if miles > 40:\n  min(cost, cap)\nelse: 0", {}, id="literal-in-header"
        ),
        pytest.param(
            "min(cost, cap)", {"allowed": "miles >= 40.0"}, id="literal-on-line"
        ),
        pytest.param(
            "if allowed:\n  min(cost, cap)\nelse: 0",
            {"allowed": "holds(far)", "far": "positive(miles - 40)"},
            id="literal-through-a-rule",
        ),
    ],
)
def test_each_way_a_condition_reads_the_cap_number_is_refused(principal, derived):
    assert _cap_is_bound({}, {}, principal="min(cost, cap)")
    assert not _cap_is_bound({}, {}, principal=principal, **derived)


def test_a_second_boundary_with_the_same_number_is_not_stated_once():
    profile = _numeric_profile_for_citation_path(CITATION)
    branch = sc.SourceStructureBranch((), "source-unit", "", UNIT, 0, len(UNIT))
    forties = [
        occurrence
        for occurrence in extract_typed_numeric_occurrences_from_text(
            UNIT, profile=profile
        )
        if occurrence.value == 40
    ]
    fifty = next(
        occurrence
        for occurrence in extract_typed_numeric_occurrences_from_text(
            UNIT, profile=profile
        )
        if occurrence.value == 50
    )
    assert len(forties) == 2

    def stated_once(boundary, boundaries):
        return sc._boundary_value_is_stated_once(
            branch,
            boundary,
            [(branch, other) for other in boundaries],
            source_text=UNIT,
            numeric_value_is_grounded=numeric_value_is_grounded,
        )

    assert stated_once(forties[0], [forties[0], fifty])
    assert stated_once(forties[0], [forties[0], forties[0], fifty])
    assert not stated_once(forties[0], [*forties, fifty])
    assert not stated_once(forties[1], [*forties, fifty])
    moved = sc.SourceStructureBranch((), "source-unit", "", UNIT, 5, len(UNIT) + 5)
    assert not sc._boundary_value_is_stated_once(
        moved,
        forties[0],
        [(moved, forties[0])],
        source_text=UNIT,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_known_gap_a_gate_case_at_40_is_credited_for_the_cap_with_the_same_number():
    """Pins the older selector path, which this change leaves as it is.

    Known gap of the selector path, tracked in a follow-up issue. That path
    grounds a comparison case by its number only. So a travel-distance case
    at 40 is credited for the lodging $40 cap too, and the candidate passes
    with no case at the cap. When the selector path binds occurrences, this
    result must become the one missing-40 finding.
    """

    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">", distances={49: 39, 50: 40, 51: 41}
    )
    without_cap_case = [case for case in cases if case["name"] != CAP_CASE]
    assert 40 in _distance_inputs(without_cap_case)
    assert 40 not in {
        case["input"][COST] for case in without_cap_case if COST in case["input"]
    }
    assert _issues(candidate, without_cap_case, source) == []


def test_known_gap_a_gate_behind_a_helper_rule_is_not_asked_for_its_boundary():
    """Pins the older selector path, which this change leaves as it is.

    Known gap of the selector path, tracked in a follow-up issue. The gate
    `travel_distance_miles >= threshold` is in a Boolean helper rule, and the
    principal rule reads the helper. The source says "more than 40 miles", no
    case sets the distance to 40, and the wrong inclusive gate still passes.
    The min-cap path credits neither 40 here, so this result comes from the
    selector path alone. When that path binds occurrences, this result must
    become a missing-40 finding.
    """

    candidate, cases, source = _second_forty(
        MORE_THAN_40_MILES, gate=">=", distances=UNTESTED_LOWER_GATE
    )
    assert candidate.count(GATE_COMPARISON) == 1
    candidate = candidate.replace(GATE_COMPARISON, "lodging_allowed").replace(
        PRINCIPAL_RULE,
        _helper_rule("lodging_allowed", "Boolean", GATE_COMPARISON) + PRINCIPAL_RULE,
        1,
    )
    cases = copy.deepcopy(cases)
    for case in cases:
        if DISTANCE in case["input"]:
            case["output"][f"{PREFIX}#lodging_allowed"] = case["input"][DISTANCE] >= 40
    assert _distance_inputs(cases) == {38, 39, 51}
    assert _issues(candidate, cases, source) == []

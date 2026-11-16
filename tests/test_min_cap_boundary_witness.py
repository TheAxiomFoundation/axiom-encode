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


def _issues(candidate: str = CANDIDATE, cases: list | None = None) -> list[str]:
    profile = _numeric_profile_for_citation_path(CITATION)
    bindings = sc.collect_artifact_numeric_bindings(
        candidate,
        extract_named_scalars=extract_named_scalar_occurrences,
    )
    analysis = sc.analyze_complete_source_unit(
        candidate,
        SOURCE,
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

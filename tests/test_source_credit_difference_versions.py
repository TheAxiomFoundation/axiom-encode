"""A good credit witness cannot excuse another version or formula alternative."""

from copy import deepcopy
from textwrap import indent

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from tests.test_source_credit_difference_conditions import (
    CLAUSE,
    _derived_credit_candidate,
    _public_gate,
    _with_credit_result_alias,
)


def _append_proved_version(rule, formula, year):
    index = len(rule["versions"])
    rule["versions"].append({"effective_from": f"{year}-01-01", "formula": formula})
    atoms = rule.get("metadata", {}).get("proof", {}).get("atoms", [])
    for atom in tuple(atoms):
        if atom.get("path") == "versions[0].formula":
            atoms.append({**deepcopy(atom), "path": f"versions[{index}].formula"})


def _with_wages(payload, cases):
    payload["inputs"].append(
        {"name": "wages", "description": "Unrelated eligible wages.", "dtype": "Money"}
    )
    for case in cases:
        case["input"]["wages"] = 50


def _evaluated_outputs(payload, case):
    return sc._case_dependency_environment(
        {rule["name"]: rule for rule in payload["rules"]},
        case,
        formula_environment={},
        require_asserted_value=False,
    )


@pytest.mark.parametrize(
    "target,formula,wrong_result",
    [
        ("state_credit", "amount_1 + wages", 850),
        (
            "child_care_credit",
            "if resident_has_federal_tax_liability: "
            "state_credit + federal_credit else: state_credit",
            1200,
        ),
    ],
    ids=["wage-added-operand", "summed-credits"],
)
@pytest.mark.parametrize("assert_wrong_future_case", [False, True])
def test_round5_valid_2026_credit_witness_does_not_excuse_invalid_2027_version(
    target, formula, wrong_result, assert_wrong_future_case
):
    payload, cases = _derived_credit_candidate("proof")
    _with_wages(payload, cases)
    rule = next(rule for rule in payload["rules"] if rule["name"] == target)
    _append_proved_version(rule, formula, 2027)

    future_case = deepcopy(cases[0])
    future_case.update(name="wrong_future_credit", period="2027-01-01")
    evaluated = _evaluated_outputs(payload, future_case)
    assert evaluated["child_care_credit"] == wrong_result
    assert future_case["input"]["amount_1"] - future_case["input"]["amount_2"] == 800
    if assert_wrong_future_case:
        future_case["output"] = {
            name: float(value) for name, value in evaluated.items()
        }
        cases.append(future_case)

    assert _public_gate(payload, "E. " + CLAUSE, cases)


def _nested_alternatives(good, last):
    return (
        "if branch_choice_0:\n"
        + indent(good, "  ")
        + "\nelse:\n  if branch_choice_1:\n"
        + indent(good, "    ")
        + "\n  else:\n"
        + indent(last, "    ")
    )


def _literal_credit_parameter(payload, cases, target):
    """Keep upstream fixture amounts faithful to a literal local parameter."""
    target["kind"] = "parameter"
    for version in target["versions"]:
        version["formula"] = "2"
    state, federal = (2, 1) if target["name"] == "state_credit" else (1, 2)
    for case in cases:
        case["input"].update(amount_1=state, amount_2=federal)
        case["output"].update(
            state_credit=state,
            federal_credit=federal,
            child_care_credit=(
                state - federal
                if case["input"]["resident_has_federal_tax_liability"]
                else state
            ),
        )


def _colliding_credit_input(payload, target):
    role = "federal" if target["name"].startswith("federal") else "state"
    description = (
        "The amount of the state child care credit for which the taxpayer is eligible."
        if role == "state"
        else "The federal credit for child and dependent care expenses the taxpayer "
        "is able to deduct from federal tax liability."
    )
    payload["inputs"].append(
        {"name": target["name"], "dtype": "Money", "description": description}
    )


@pytest.mark.parametrize("assert_wrong_future_case", [False, True])
def test_parameter_input_collision_cannot_hide_wrong_future_credit(
    assert_wrong_future_case,
):
    payload, cases = _derived_credit_candidate("proof")
    parameter = payload["rules"][1]
    _literal_credit_parameter(payload, cases, parameter)
    _append_proved_version(parameter, "3", 2027)
    _colliding_credit_input(payload, parameter)
    assert all("state_credit" not in case["input"] for case in cases)

    future = deepcopy(cases[0])
    future["period"] = "2027-01-01"
    evaluated = sc._case_dependency_environment(
        {rule["name"]: rule for rule in payload["rules"] if rule["kind"] == "derived"},
        future,
        formula_environment=sc._constant_rule_environment(payload),
        require_asserted_value=False,
    )
    assert evaluated["child_care_credit"] == 2
    assert future["input"]["amount_1"] - future["input"]["amount_2"] == 1
    if assert_wrong_future_case:
        future["output"].update(state_credit=3, child_care_credit=2)
        cases.append(future)

    issues = _public_gate(payload, "E. " + CLAUSE, cases)
    assert any(
        "[complete-source-unit:declaration-collision]" in issue
        and "state_credit" in issue
        for issue in issues
    ), issues


@pytest.mark.parametrize("site", ["state", "state_parameter", "state_alias"])
def test_local_operand_input_collision_is_explicit_even_without_poison(site):
    payload, cases, target, _good, _wrong = _versioned_candidate(site, 2)
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()
    if site.endswith("parameter"):
        _literal_credit_parameter(payload, cases, target)
    _colliding_credit_input(payload, target)

    issues = _public_gate(payload, "E. " + CLAUSE, cases)
    assert any(
        "[complete-source-unit:declaration-collision]" in issue
        and target["name"] in issue
        for issue in issues
    ), issues


def _versioned_candidate(site, version_count):
    payload, cases = _derived_credit_candidate("proof")
    _with_wages(payload, cases)
    for index in range(2):
        name = f"branch_choice_{index}"
        payload["inputs"].append({"name": name, "dtype": "Boolean"})
        for case in cases:
            case["input"][name] = True

    if site == "root":
        target = payload["rules"][0]
        wrong = (
            "if resident_has_federal_tax_liability: "
            "state_credit + federal_credit else: state_credit"
        )
    elif site == "result_alias":
        payload, cases = _with_credit_result_alias(payload, cases)
        target = payload["rules"][-1]
        # The shared fixture aliases metadata shallowly. Later proof atoms
        # belong to this dependency's versions, not the single-version root.
        target["metadata"] = deepcopy(target["metadata"])
        wrong = "state_credit + federal_credit"
    else:
        index = 1 if site.startswith("state") else 2
        target = payload["rules"][index]
        wrong = f"amount_{index} + wages"
        if site.endswith("alias"):
            # Poison the last link, beyond the source-qualified operand and
            # another neutral identity alias. All valid witnesses reach it.
            target["versions"][0]["formula"] = f"operand_alias_{index}_1"
            for link in (1, 2):
                name = f"operand_alias_{index}_{link}"
                formula = f"operand_alias_{index}_2" if link == 1 else f"amount_{index}"
                target = {
                    "name": name,
                    "kind": "derived",
                    "dtype": "Money",
                    "versions": [{"effective_from": "2026-01-01", "formula": formula}],
                }
                payload["rules"].append(target)
                for case in cases:
                    case["output"][name] = case["input"][f"amount_{index}"]

    good = target["versions"][0]["formula"]
    for index in range(1, version_count):
        _append_proved_version(target, good, 2026 + index)
    return payload, cases, target, good, wrong


@given(
    site=st.sampled_from(
        [
            "root",
            "state",
            "federal",
            "state_alias",
            "federal_alias",
            "result_alias",
            "state_parameter",
            "federal_parameter",
        ]
    ),
    layout=st.sampled_from(["version", "alternative"]),
    version_count=st.integers(min_value=2, max_value=4),
    poison=st.integers(min_value=0, max_value=3),
    collision=st.booleans(),
)
@example(site="root", layout="version", version_count=2, poison=0, collision=False)
@example(site="state", layout="version", version_count=2, poison=0, collision=False)
@example(site="federal", layout="version", version_count=3, poison=1, collision=False)
@example(
    site="state_alias", layout="version", version_count=3, poison=1, collision=False
)
@example(
    site="federal_alias", layout="version", version_count=2, poison=0, collision=False
)
@example(
    site="result_alias", layout="version", version_count=4, poison=2, collision=False
)
@example(site="root", layout="alternative", version_count=2, poison=0, collision=False)
@example(site="state", layout="alternative", version_count=2, poison=0, collision=False)
@example(
    site="federal", layout="alternative", version_count=3, poison=1, collision=False
)
@example(
    site="state_alias", layout="alternative", version_count=3, poison=0, collision=False
)
@example(
    site="federal_alias",
    layout="alternative",
    version_count=2,
    poison=1,
    collision=False,
)
@example(
    site="result_alias",
    layout="alternative",
    version_count=4,
    poison=0,
    collision=False,
)
@example(site="state", layout="version", version_count=2, poison=0, collision=True)
@example(
    site="state_alias", layout="version", version_count=3, poison=1, collision=True
)
@example(
    site="state_parameter", layout="version", version_count=2, poison=0, collision=True
)
@example(
    site="federal_parameter",
    layout="version",
    version_count=3,
    poison=1,
    collision=True,
)
@example(
    site="state_parameter",
    layout="alternative",
    version_count=2,
    poison=0,
    collision=True,
)
@settings(deadline=None, max_examples=20)
def test_round5_any_poisoned_credit_version_or_alternative_rejects(
    site, layout, version_count, poison, collision
):
    payload, cases, target, good, wrong = _versioned_candidate(site, version_count)
    # The candidate has a valid earlier witness before one version or one
    # otherwise-unexecuted nested alternative is poisoned.
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()
    if site.endswith("parameter"):
        _literal_credit_parameter(payload, cases, target)
        good, wrong = "2", "3"
    if layout == "version":
        index = 1 + poison % (version_count - 1)
        formula = wrong
    else:
        index = poison % version_count
        formula = _nested_alternatives(good, wrong)
    target["versions"][index]["formula"] = formula
    if collision:
        _colliding_credit_input(payload, target)

    # The existing 2026 assertions remain faithful even when the poisoned
    # version is 2026: its nested bad alternative has never been selected.
    evaluated = _evaluated_outputs(payload, cases[0])
    assert evaluated["child_care_credit"] == cases[0]["output"]["child_care_credit"]
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_round5_faithful_multiversion_nested_credit_alternatives_accept():
    payload, cases, root, good, _wrong = _versioned_candidate("root", 3)
    for version in root["versions"]:
        version["formula"] = _nested_alternatives(good, good)
    for year in (2027, 2028):
        for original in tuple(cases[:3]):
            case = deepcopy(original)
            case["period"] = f"{year}-01-01"
            cases.append(case)
    # The source's liability=false fallback remains the state amount rather
    # than the difference, including in each later version.
    assert _evaluated_outputs(payload, cases[1])["child_care_credit"] == 1000
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()


def test_round5_valid_credit_rule_does_not_excuse_another_source_owned_formula():
    payload, cases = _derived_credit_candidate("proof")
    additional = deepcopy(payload["rules"][0])
    additional["name"] = "unfaithful_additional_credit"
    additional["versions"][0]["formula"] = (
        "if resident_has_federal_tax_liability: "
        "state_credit + federal_credit else: state_credit"
    )
    payload["rules"].append(additional)
    for case in cases:
        state = case["input"]["amount_1"]
        federal = case["input"]["amount_2"]
        case["output"][additional["name"]] = (
            state + federal
            if case["input"]["resident_has_federal_tax_liability"]
            else state
        )
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_round5_operand_proof_cannot_exclude_a_nonidentity_root_from_validation():
    payload, cases = _derived_credit_candidate("proof")
    root = payload["rules"][0]
    operand_atom = payload["rules"][1]["metadata"]["proof"]["atoms"][0]
    root_atom = root["metadata"]["proof"]["atoms"][0]
    root_atom["source"]["excerpt"] = operand_atom["source"]["excerpt"]
    _append_proved_version(
        root,
        "if resident_has_federal_tax_liability: "
        "state_credit + federal_credit else: state_credit",
        2027,
    )
    # Original 2026 assertions remain accurate; the narrowed proof must not
    # turn this result formula into an operand exemption for its 2027 sum.
    assert _evaluated_outputs(payload, cases[0])["child_care_credit"] == 800
    assert _public_gate(payload, "E. " + CLAUSE, cases)


@pytest.mark.parametrize("proof", ["missing", "other_credit"])
def test_round5_each_operand_version_needs_its_own_unambiguous_binding(proof):
    payload, cases = _derived_credit_candidate("proof")
    operand = payload["rules"][1]
    _append_proved_version(operand, "amount_1", 2027)
    atoms = operand["metadata"]["proof"]["atoms"]
    if proof == "missing":
        atoms.pop()
    else:
        atoms[-1]["source"]["excerpt"] = payload["rules"][2]["metadata"]["proof"][
            "atoms"
        ][0]["source"]["excerpt"]
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_round5_input_declaration_cannot_hide_invalid_local_operand_versions():
    payload, cases = _derived_credit_candidate("proof")
    _with_wages(payload, cases)
    operand = payload["rules"][1]
    _append_proved_version(operand, "amount_1 + wages", 2027)
    payload["inputs"].append(
        {
            "name": "state_credit",
            "dtype": "Money",
            "description": "The amount of the state child care credit for which the taxpayer is eligible.",
        }
    )
    assert _public_gate(payload, "E. " + CLAUSE, cases)


def test_round5_expired_local_alias_version_does_not_overlap_credit_formula():
    payload, cases = _derived_credit_candidate("proof")
    _with_wages(payload, cases)
    operand = payload["rules"][1]
    operand["versions"][0]["formula"] = "intermediate_state_amount"
    payload["rules"].append(
        {
            "name": "intermediate_state_amount",
            "kind": "derived",
            "dtype": "Money",
            "versions": [
                {
                    "effective_from": "2025-01-01",
                    "effective_to": "2025-12-31",
                    "formula": "amount_1 + wages",
                },
                {"effective_from": "2026-01-01", "formula": "amount_1"},
            ],
        }
    )
    for case in cases:
        case["output"]["intermediate_state_amount"] = case["input"]["amount_1"]
    # Both source-owned rules start in 2026, after the neutral alias's invalid
    # dependency formula expired. It has no source claim of its own to validate.
    assert _evaluated_outputs(payload, cases[0])["child_care_credit"] == 800
    assert _public_gate(payload, "E. " + CLAUSE, cases) == ()

"""Shared membership predicates belong to each enumerated subject, not a fourth fact."""

import pytest

from axiom_encode.harness.source_completeness import (
    _shared_membership_list_body,
    _source_conjunctive_fact_gates,
    _terminal_gate_evidence,
    _terminal_names_corroborate_source_gates,
)

SOURCE = """The child support expense is not allowed when all of the following are in the same budgetary unit:

● The parent paying the child support.

● The child for whom the child support is being paid.

● The parent receiving the child support.
"""


def test_shared_membership_list_has_three_members_not_four_facts():
    gates = _source_conjunctive_fact_gates(SOURCE)
    assert len(gates) == 3
    assert all(
        "budgetary" in predicates and "unit" in predicates for _, predicates in gates
    )
    assert any("paying" in predicates for _, predicates in gates)
    assert any("receiving" in predicates for _, predicates in gates)


@pytest.mark.parametrize("extra", ["if eligible", "and eligible", "unless exempt"])
def test_modified_header_is_not_silently_discarded(extra):
    text = SOURCE.replace("budgetary unit:", f"budgetary unit {extra}:")
    assert _source_conjunctive_fact_gates(text) != _source_conjunctive_fact_gates(
        SOURCE
    )


def test_additional_member_stays_required():
    gates = _source_conjunctive_fact_gates(
        SOURCE + "\n● The spouse receiving income.\n"
    )
    assert len(gates) == 4


def test_unbulleted_conditions_are_unchanged():
    gates = _source_conjunctive_fact_gates(
        "Allowed when the parent is eligible and the child is disabled."
    )
    assert len(gates) == 2


@pytest.mark.parametrize(
    "body",
    [
        "all of the following are in the same budgetary unit:\nThe parent.\nThe child.",
        "all of the following are in the same budgetary unit:\n● The parent.\n● The child.\nOnly if eligible.",
        "any of the following are in the same budgetary unit:\n● The parent.\n● The child.",
        "all of the following are not in the same budgetary unit:\n● The parent.\n● The child.",
        "all of the following are in the same budgetary unit if eligible:\n● The parent.\n● The child.",
        "all of the following are in the same budgetary unit:\n● The parent; unless exempt.\n● The child.",
    ],
)
def test_unrecognized_list_shapes_are_byte_identical(body):
    assert _shared_membership_list_body(body) == body


@pytest.mark.parametrize(
    "suffix,accepted",
    [
        ("is_in_same_budgetary_unit", True),
        ("", False),
        ("is_in_different_budgetary_unit", False),
        ("is_outside_same_budgetary_unit", False),
        ("is_not_in_same_budgetary_unit", False),
        ("is_never_in_same_budgetary_unit", False),
        ("or_spouse_is_in_same_budgetary_unit", False),
        ("or_anyone_is_in_same_budgetary_unit", False),
    ],
)
def test_every_terminal_must_preserve_shared_membership(suffix, accepted):
    names = frozenset(
        f"payment_{role}_{suffix}".rstrip("_")
        for role in ("paying_parent", "child", "receiving_parent")
    )
    evidence = _terminal_gate_evidence(
        {"inputs": [{"name": name, "dtype": "Boolean"} for name in names]}
    )
    assert (
        _terminal_names_corroborate_source_gates(
            names, _source_conjunctive_fact_gates(SOURCE), terminal_evidence=evidence
        )
        is accepted
    )


@pytest.mark.parametrize(
    "description",
    [
        "The participant is not in the same budgetary unit.",
        "The participant is never in the same budgetary unit.",
        "The participant is in the same household.",
    ],
)
def test_description_cannot_contradict_affirmative_name(description):
    names = frozenset(
        f"payment_{role}_is_in_same_budgetary_unit"
        for role in ("paying_parent", "child", "receiving_parent")
    )
    evidence = _terminal_gate_evidence(
        {
            "inputs": [
                {
                    "name": name,
                    "dtype": "Boolean",
                    "description": description,
                }
                for name in names
            ]
        }
    )
    assert not _terminal_names_corroborate_source_gates(
        names, _source_conjunctive_fact_gates(SOURCE), terminal_evidence=evidence
    )


@pytest.mark.parametrize(
    "extra",
    [
        ". The parent must be employed",
        " who is employed",
        " if employed",
    ],
)
def test_extra_bullet_conditions_keep_original_parser(extra):
    body = SOURCE.split("when ", 1)[1].replace(
        "The parent paying the child support.",
        f"The parent paying the child support{extra}.",
    )
    assert _shared_membership_list_body(body) == body

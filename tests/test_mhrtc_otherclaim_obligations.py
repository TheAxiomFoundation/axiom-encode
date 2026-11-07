"""Structural witness selection never substitutes for source qualification."""

import copy

import pytest

from axiom_encode.harness.mhrtc_otherclaim_obligations import (
    mhrtc_otherclaim_obligations,
)
from tests.test_mhrtc_otherclaim_request import PREFIX, other_fixture  # noqa: F401
from tests.test_mhrtc_request import fixture as principal_fixture  # noqa: F401


def obligations(fixture):
    case, payload, pipe, policy = fixture
    return mhrtc_otherclaim_obligations(
        case, payload, pipe=pipe, rules_file=policy, case_index=1
    )


def test_relevant_target(other_fixture):  # noqa: F811
    assert obligations(other_fixture) == ("other-1",)


@pytest.mark.parametrize(
    "change", ["other_renovation", "self", "unlinked", "other_filer"]
)
def test_irrelevant_targets_not_invented(other_fixture, change):  # noqa: F811
    case = other_fixture[0]
    row = case["tables"]["OtherClaim"][0]
    if change == "other_renovation":
        row[PREFIX + "input.other_renovation_key"] = 202
    elif change == "self":
        row[PREFIX + "input.other_claimant_key"] = 1
    elif change == "unlinked":
        case["relations"].pop()
    else:
        case["tables"]["Claim"][0][PREFIX + "input.claimant_key"] = 3
    assert obligations(other_fixture) == ()


@pytest.mark.parametrize(
    "where,name",
    [
        ("Claim", "claimant_key"),
        ("Claim", "renovation_key"),
        ("OtherClaim", "other_claimant_key"),
        ("OtherClaim", "other_renovation_key"),
    ],
)
def test_linked_missing_fact_not_silently_skipped(other_fixture, where, name):  # noqa: F811
    other_fixture[0]["tables"][where][0].pop(PREFIX + "input." + name)
    with pytest.raises(ValueError):
        obligations(other_fixture)


def test_missing_filer_not_silently_skipped(other_fixture):  # noqa: F811
    other_fixture[0]["input"].pop(PREFIX + "input.filing_claimant_key")
    with pytest.raises(ValueError):
        obligations(other_fixture)


def test_duplicate_target_rejected(other_fixture):  # noqa: F811
    rows = other_fixture[0]["tables"]["OtherClaim"]
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError):
        obligations(other_fixture)


def test_same_target_multiple_parent_edges_yields_one_obligation(other_fixture):  # noqa: F811
    case = other_fixture[0]
    parent = copy.deepcopy(case["tables"]["Claim"][0])
    parent["id"] = "claim-2"
    case["tables"]["Claim"].append(parent)
    case["relations"].extend(
        [
            {
                "name": PREFIX + "relation.claim_records",
                "tuple": ["claim-2", case["entity_id"]],
            },
            {
                "name": PREFIX + "relation.candidate_other_claims",
                "tuple": ["other-1", "claim-2"],
            },
        ]
    )
    assert obligations(other_fixture) == ("other-1",)


def test_unlinked_incomplete_row_does_not_invent_association(other_fixture):  # noqa: F811
    case = other_fixture[0]
    case["relations"].pop()
    case["tables"]["OtherClaim"][0].pop(PREFIX + "input.other_claimant_key")
    assert obligations(other_fixture) == ()


def test_multiple_targets_preserve_frozen_table_order(other_fixture):  # noqa: F811
    case = other_fixture[0]
    row = copy.deepcopy(case["tables"]["OtherClaim"][0])
    row["id"] = "other-2"
    row[PREFIX + "input.other_claimant_key"] = 3
    case["tables"]["OtherClaim"].insert(0, row)
    case["relations"].append(
        {
            "name": PREFIX + "relation.candidate_other_claims",
            "tuple": ["other-2", "claim-1"],
        }
    )
    assert obligations(other_fixture) == ("other-2", "other-1")


def test_duplicate_linked_parent_row_rejected(other_fixture):  # noqa: F811
    parents = other_fixture[0]["tables"]["Claim"]
    parents.append(copy.deepcopy(parents[0]))
    with pytest.raises(ValueError):
        obligations(other_fixture)

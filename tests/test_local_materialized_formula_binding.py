"""Normal materializer spellings retain exact source-owned formula contracts."""

import ast
from copy import deepcopy

import pytest

from axiom_encode.harness.local_pair_source_evidence import matches_expected_formula
from axiom_encode.harness.validator_pipeline import _repair_else_if_formula
from tests.test_local_vehicle_date_evidence import DATA

PAIRS = [
    (r["name"], v["formula"], _repair_else_if_formula(v["formula"]))
    for r in DATA["rules"]
    for v in r.get("versions", [])
    if isinstance(v.get("formula"), str)
    and _repair_else_if_formula(v["formula"]) != v["formula"]
]


@pytest.mark.parametrize("name,original,materialized", PAIRS)
def test_exact_materializer_alternative(name, original, materialized):
    assert matches_expected_formula(original, original, ast_equivalent=True)
    assert matches_expected_formula(materialized, original, ast_equivalent=True)
    assert matches_expected_formula(materialized, original)
    for bad in [
        materialized + "; 0",
        materialized + " # comment",
        materialized + " else: 0",
        materialized.replace("if ", "if not ", 1),
        materialized.replace("else:", "else: 1 +", 1),
        materialized.encode(),
        ast.parse(original),
        None,
        1,
        True,
    ]:
        assert not matches_expected_formula(bad, original, ast_equivalent=True), name


def test_materializer_does_not_mutate_input():
    before = deepcopy(DATA)
    for _, original, materialized in PAIRS:
        assert matches_expected_formula(materialized, original)
    assert DATA == before


@pytest.mark.parametrize("use_materialized", [False, True])
def test_complete_materialized_bindings(use_materialized):
    """Exercise full source/proof closure, including downstream AST consumers."""
    from types import SimpleNamespace

    from axiom_encode.harness import local_pair_source_evidence as pair
    from axiom_encode.harness.local_prior_form_evidence import (
        _SPANS,
        prior_form_reference_binding,
    )
    from axiom_encode.harness.local_vehicle_date_evidence import (
        _ROLES,
        vehicle_date_role_binding,
    )
    from tests.test_local_vehicle_date_evidence import BODY, CITE

    data = deepcopy(DATA)
    if use_materialized:
        for rule in data["rules"]:
            for version in rule.get("versions", []):
                if isinstance(version.get("formula"), str):
                    version["formula"] = _repair_else_if_formula(version["formula"])

    def branch(start, end):
        return SimpleNamespace(start=start, end=end, text=BODY[start:end])

    for profile in pair._PROFILES.values():
        assert pair.bind_pair_role(data, branch(*profile[:2]), BODY, CITE, profile[2])
    for name, profile in pair._SIGNED_PROFILES.items():
        assert pair.bind_signed_role(data, branch(*profile[0]), BODY, CITE, name)
    assert pair.bind_chart_duty(data, branch(775, 1075), BODY, CITE)
    for span in pair._REMAINING_CHART_SPANS.values():
        assert pair.bind_remaining_chart_role(data, branch(*span), BODY, CITE)
    assert pair.bind_expense_group(data, branch(2877, 3984), BODY, CITE)
    case = {
        "period": {
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        }
    }
    for row in _ROLES:
        assert vehicle_date_role_binding(data, BODY, CITE, case, (row[0], row[0] + 4))
    for start in _SPANS:
        assert prior_form_reference_binding(data, BODY, CITE, case, (start, start + 4))

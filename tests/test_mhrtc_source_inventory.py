"""Real production extraction controls; static roles are not source credit."""

from dataclasses import replace
from functools import partial

import pytest
import yaml

from axiom_encode.harness import mhrtc_source_inventory as m
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.bound_mhrtc_program import bind_mhrtc_program
from axiom_encode.harness.mhrtc_program_contract import CITATION, SOURCE_TEXT
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests.test_bound_mhrtc_program import policy  # noqa: F401


@pytest.fixture
def inventory(policy):  # noqa: F811
    content = yaml.safe_dump(policy, sort_keys=False)
    bound = bind_mhrtc_program(content, SOURCE_TEXT)
    assert bound is not None
    branches = s.recognize_source_structure(SOURCE_TEXT, corpus_citation_path=CITATION)
    printed = s._printed_operation_context(
        SOURCE_TEXT,
        branches,
        CITATION,
        consumer_content=content,
        proof_sources={CITATION: SOURCE_TEXT},
    )
    formulas = s._source_formula_branches(
        SOURCE_TEXT,
        printed_operations=printed,
        corpus_citation_path=CITATION,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
    )
    shared = dict(
        raw_source_text=SOURCE_TEXT,
        branches=branches,
        active_branches=branches,
        deferred_paths=set(),
        formula_branches=formulas,
        deferred_clauses=[],
        corpus_citation_path=CITATION,
    )
    controls = s._source_control_branches(
        SOURCE_TEXT,
        **shared,
        extract_numeric_occurrences=partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        numeric_exception_controls_only=False,
    )
    exceptions = s._source_exception_branches(
        SOURCE_TEXT, **shared, numeric_limits_only=False
    )
    return (
        bound,
        content,
        dict(
            formulas=tuple(formulas),
            controls=tuple(controls),
            exceptions=tuple(exceptions),
        ),
    )


def test_production_inventory_has_exact_static_roles(inventory):
    bound, content, branches = inventory
    roles = m.mhrtc_branch_roles(bound, content, SOURCE_TEXT, **branches)
    assert len(roles) == 30
    assert sum(r.inventory_kind == "formula_branches" for r in roles) == 9
    assert sum(r.inventory_kind == "control_branches" for r in roles) == 11
    assert sum(r.inventory_kind == "exception_branches" for r in roles) == 10
    assert any(r.classification == "informational_no_computation" for r in roles)
    assert not hasattr(roles[0], "covers_gate")
    reversed_branches = {k: v[::-1] for k, v in branches.items()}
    assert (
        m.mhrtc_branch_roles(bound, content, SOURCE_TEXT, **reversed_branches) == roles
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", 303),
        ("end", 349),
        ("text", "invented"),
        ("kind", "exception-clause"),
        ("label", "different"),
        ("path", ("other",)),
        ("structural_numeric_spans", ((303, 304),)),
        ("computation_chapeaux", ("invented",)),
        ("required_principal_output", "invented"),
        ("start", 302.0),
        ("path", []),
    ],
)
def test_every_typed_branch_field_bound(inventory, field, value):
    bound, content, branches = inventory
    branches["formulas"] = (
        replace(branches["formulas"][0], **{field: value}),
        *branches["formulas"][1:],
    )
    with pytest.raises(ValueError):
        m.mhrtc_branch_roles(bound, content, SOURCE_TEXT, **branches)


@pytest.mark.parametrize(
    "mode", ["missing", "duplicate", "replace", "wrong_category", "dict"]
)
def test_complete_multiset_and_branch_category_required(inventory, mode):
    bound, content, branches = inventory
    original = branches["controls"]
    if mode == "missing":
        branches["controls"] = original[:-1]
    elif mode == "duplicate":
        branches["controls"] = (*original, original[0])
    elif mode == "replace":
        branches["controls"] = (*original[:-1], original[0])
    elif mode == "wrong_category":
        branches["controls"] = original[:-1]
        branches["exceptions"] += (original[-1],)
    else:
        branches["controls"] = ({}, *original[1:])
    with pytest.raises(ValueError):
        m.mhrtc_branch_roles(bound, content, SOURCE_TEXT, **branches)


def test_changed_source_or_unbound_policy_rejected(inventory):
    bound, content, branches = inventory
    with pytest.raises(ValueError):
        m.mhrtc_branch_roles(None, content, SOURCE_TEXT, **branches)
    with pytest.raises(ValueError):
        m.mhrtc_branch_roles(bound, content, SOURCE_TEXT + "changed", **branches)

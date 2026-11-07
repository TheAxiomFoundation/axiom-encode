"""Exact source-coordinate controls; no runtime/source-authority claim."""

from dataclasses import replace

import pytest
import yaml

from axiom_encode.harness import mhrtc_numeric as m
from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.bound_mhrtc_program import bind_mhrtc_program
from axiom_encode.harness.mhrtc_program_contract import CITATION, SOURCE_TEXT
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests.test_bound_mhrtc_program import policy  # noqa: F401


@pytest.fixture
def inputs(policy):  # noqa: F811
    content = yaml.safe_dump(policy, sort_keys=False)
    bound = bind_mhrtc_program(content, SOURCE_TEXT)
    assert bound is not None
    recall = sc.authoritative_numeric_recall_text(
        SOURCE_TEXT, corpus_citation_path=CITATION
    )
    occurrences = tuple(
        extract_typed_numeric_inventory_occurrences_from_text(recall, profile="legacy")
    )
    return bound, content, SOURCE_TEXT, occurrences


def test_exact_complete_recall_and_source_mapping(inputs):
    roles = m.mhrtc_numeric_roles(*inputs)
    assert len(roles) == 32
    assert sum(r.recall_span != r.source_span for r in roles) == 23
    assert {r.source_span[0] - r.recall_span[0] for r in roles} == {0, 11, 22}
    compound = next(r for r in roles if r.literal == "×\n\n8")
    standalone = next(r for r in roles if r.source_span == (7535, 7536))
    assert compound != standalone
    assert compound.role == standalone.role == "printed_line_locator"
    assert not hasattr(compound, "covers_gate")


@pytest.mark.parametrize(
    "field,value",
    [
        ("start", 2370),
        ("end", 2374),
        ("raw", "14.50"),
        ("value", 0.15),
        ("source_value", 14.5),
        ("has_rate_context", False),
        ("has_temporal_context", True),
        ("has_structural_context", True),
        ("requires_rate_context", False),
        ("is_word_number", True),
        ("alternative_values", (0.145,)),
        ("requires_month_context", True),
        ("has_rate_context", 1),
        ("value", float("nan")),
        ("alternative_values", []),
    ],
)
def test_every_typed_occurrence_field_is_bound(inputs, field, value):
    bound, content, source, occurrences = inputs
    changed = (replace(occurrences[0], **{field: value}), *occurrences[1:])
    with pytest.raises(ValueError):
        m.mhrtc_numeric_roles(bound, content, source, changed)


@pytest.mark.parametrize(
    "mode", ["missing", "duplicate", "same_count_wrong_repeat", "dict"]
)
def test_complete_multiset_required(inputs, mode):
    bound, content, source, occurrences = inputs
    if mode == "missing":
        changed = occurrences[:-1]
    elif mode == "duplicate":
        changed = (*occurrences, occurrences[0])
    elif mode == "same_count_wrong_repeat":
        changed = (*occurrences[:-1], occurrences[0])
    else:
        changed = ({"value": 0.145}, *occurrences[1:])
    with pytest.raises(ValueError):
        m.mhrtc_numeric_roles(bound, content, source, changed)


def test_order_irrelevant_but_raw_offsets_not_recall_offsets(inputs):
    bound, content, source, occurrences = inputs
    assert m.mhrtc_numeric_roles(
        bound, content, source, occurrences[::-1]
    ) == m.mhrtc_numeric_roles(*inputs)
    roles = m.mhrtc_numeric_roles(*inputs)
    changed = tuple(
        replace(o, start=r.source_span[0], end=r.source_span[1])
        for o, r in zip(occurrences, roles, strict=True)
    )
    with pytest.raises(ValueError):
        m.mhrtc_numeric_roles(bound, content, source, changed)


def test_unknown_source_origin_rejected(inputs, monkeypatch):
    monkeypatch.setattr(sc._NumericRecallSourceMap, "source_span", lambda *args: None)
    with pytest.raises(ValueError, match="noncontiguous"):
        m.mhrtc_numeric_roles(*inputs)


def test_bound_policy_and_source_required(inputs):
    bound, content, source, occurrences = inputs
    with pytest.raises(ValueError):
        m.mhrtc_numeric_roles(None, content, source, occurrences)
    with pytest.raises(ValueError):
        m.mhrtc_numeric_roles(bound, content, source + "changed", occurrences)


def test_static_role_inventory_may_not_omit_occurrence(inputs, monkeypatch):
    monkeypatch.setattr(m, "_RAW_ROLES", m._RAW_ROLES[:-1])
    with pytest.raises(ValueError, match="unowned"):
        m.mhrtc_numeric_roles(*inputs)

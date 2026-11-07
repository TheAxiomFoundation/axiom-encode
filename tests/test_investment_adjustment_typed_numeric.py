from dataclasses import replace

import pytest

from axiom_encode.harness.investment_adjustment_numeric import (
    match_investment_adjustment_typed_numeric_owner,
)
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
)
from tests.test_investment_adjustment_numeric import BOUND, CONTENT, GUIDE, PARENT


def match(occurrence):
    return match_investment_adjustment_typed_numeric_owner(
        BOUND, CONTENT, PARENT, GUIDE, occurrence
    )


def compound():
    return next(
        o
        for o in extract_typed_numeric_inventory_occurrences_from_text(PARENT)
        if o.start == 574
    )


def test_actual_extractor_streams_have_exact_source_roles():
    inventory = extract_typed_numeric_inventory_occurrences_from_text(PARENT)
    grounding = extract_typed_numeric_occurrences_from_text(PARENT)
    assert len(inventory) == 30 and len(grounding) == 33
    assert all(match(o) is not None for o in inventory + grounding)
    owner = match(compound())
    assert owner.span == (574, 591)
    assert owner.owners == ("resource_recovery_rate", "line28actualnonzero")
    assert owner.role == "substantive_half_rate"


@pytest.mark.parametrize(
    "change",
    [
        {"start": 573},
        {"end": 590},
        {"raw": "50%"},
        {"start": 588, "end": 591, "raw": "50%"},
        {"value": 50.0},
        {"value": True},
        {"source_value": 0.5},
        {"source_value": 50},
        {"has_rate_context": False},
        {"requires_rate_context": False},
        {"has_temporal_context": True},
        {"has_structural_context": True},
        {"is_word_number": True},
        {"requires_month_context": True},
        {"alternative_values": (50.0,)},
        {"alternative_values": []},
        {"start": True},
    ],
)
def test_compound_metadata_cannot_be_expanded_normalized_or_retyped(change):
    assert match(replace(compound(), **change)) is None


def test_identical_printed_50_is_not_a_half_rate():
    item = next(
        o
        for o in extract_typed_numeric_occurrences_from_text(PARENT)
        if o.start == 1971
    )
    assert match(item).role == "printed_row_graph_locator"
    assert match(replace(item, value=0.5, has_rate_context=True)) is None

"""Technical selector-matching regressions, not a signed legal encoding."""

import pytest

from axiom_encode.harness import source_completeness as sc

CONDITION = (
    "wenn der Körper- und Gesundheitszustand von dem für das Lebensalter "
    "typischen Zustand abweicht."
)
SELECTOR = "body_or_health_condition_deviates_from_age_typical_condition"


@pytest.mark.parametrize("source_negative", [False, True])
@pytest.mark.parametrize("selector_negative", [False, True])
def test_age_relative_comparison_preserves_both_negations(
    source_negative, selector_negative
):
    source = (
        CONDITION.replace("abweicht", "nicht abweicht")
        if source_negative
        else CONDITION
    )
    selector = (
        SELECTOR.replace("deviates", "not_deviates") if selector_negative else SELECTOR
    )
    assert sc._source_exception_selector_is_relevant(source, selector)
    assert sc._source_exception_selector_active_value(source, selector) == (
        source_negative == selector_negative
    )


@pytest.mark.parametrize(
    "selector",
    [
        SELECTOR + "_and_vehicle_is_blue",
        SELECTOR + "_and_" + SELECTOR,
        SELECTOR.replace("deviates", "does_not_deviate") + "_and_" + SELECTOR,
        SELECTOR.replace("deviates", "not_not_deviates"),
        SELECTOR.replace("age_typical", "population_typical"),
        SELECTOR.replace("deviates", "equals"),
        SELECTOR.replace("body_or_health", "tax_or_income"),
        "clinical_assessment_forecasts_age_atypical_body_or_health_condition",
        "impairment_present",
    ],
)
def test_other_predicates_are_not_age_relative_comparisons(selector):
    assert not sc._source_exception_selector_is_relevant(CONDITION, selector)


@pytest.mark.parametrize(
    "source",
    [
        CONDITION[:-1] + " und eine Bescheinigung vorliegt.",
        CONDITION[:-1] + " oder eine Bescheinigung vorliegt.",
        CONDITION.replace("Lebensalter", "Einkommen"),
        CONDITION.replace("abweicht", "übereinstimmt"),
        "wenn eine Beeinträchtigung nach Satz 1 zu erwarten ist.",
    ],
)
def test_other_or_additional_source_conditions_remain_unmatched(source):
    assert not sc._source_exception_selector_is_relevant(source, SELECTOR)

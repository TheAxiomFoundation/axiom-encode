"""Predicate correspondence only; these tests do not establish legal coverage."""

import pytest

from axiom_encode.harness import source_completeness as sc

CONDITION = "wenn eine Beeinträchtigung nach Satz 1 zu erwarten ist."


@pytest.mark.parametrize("source_negative", [False, True])
@pytest.mark.parametrize("selector_negative", [False, True])
@pytest.mark.parametrize("copula", ["", "is_"])
def test_expected_impairment_preserves_polarity(
    source_negative, selector_negative, copula
):
    source = (
        CONDITION.replace("zu erwarten", "nicht zu erwarten")
        if source_negative
        else CONDITION
    )
    selector = (
        "impairment_" + copula + ("not_" if selector_negative else "") + "expected"
    )
    assert sc._source_exception_selector_is_relevant(source, selector)
    assert sc._source_exception_selector_active_value(source, selector) == (
        source_negative == selector_negative
    )


@pytest.mark.parametrize(
    "selector",
    [
        "clinical_assessment_forecasts_sensory_impairment",
        "clinical_assessment_forecasts_age_atypical_body_or_health_condition",
        "sensory_impairment_expected",
        "impairment_present",
        "impairment_assessment_record_exists",
        "impairment_expected_and_vehicle_is_blue",
        "impairment_not_not_expected",
    ],
)
def test_expected_impairment_does_not_accept_subtypes_or_evidence_flags(selector):
    assert not sc._source_exception_selector_is_relevant(CONDITION, selector)


@pytest.mark.parametrize(
    "source",
    [
        CONDITION[:-1] + " und eine Bescheinigung vorliegt.",
        CONDITION[:-1] + " oder eine Bescheinigung vorliegt.",
        CONDITION.replace("eine Beeinträchtigung", "eine schwere Beeinträchtigung"),
        CONDITION.replace("zu erwarten ist", "vorliegt"),
        CONDITION.replace("eine Beeinträchtigung", "keine Beeinträchtigung").replace(
            "zu erwarten", "nicht zu erwarten"
        ),
    ],
)
def test_expected_impairment_does_not_ignore_added_source_conditions(source):
    assert not sc._source_exception_selector_is_relevant(source, "impairment_expected")

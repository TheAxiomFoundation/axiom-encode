"""Source-condition boundaries must preserve independent documentary duties."""

import pytest

from axiom_encode.harness import source_completeness as sc

PENSION = (
    "wenn dem Kind wegen seiner Behinderung nach den gesetzlichen Vorschriften "
    "Renten oder andere laufende Bezüge zustehen, durch den Rentenbescheid "
    "oder einen entsprechenden Bescheid,"
)
SELECTOR = "pension_or_ongoing_payment_is_stated_as_due_because_of_disability"
MEDICAL = (
    "Aus der Bescheinigung bzw. dem Gutachten muss Folgendes hervorgehen:\n"
    "− Vorliegen der Behinderung,\n"
    "− Beginn der Behinderung, soweit das Kind das 25. Lebensjahr vollendet hat, und\n"
    "− Auswirkungen der Behinderung auf die Erwerbsfähigkeit des Kindes."
)


@pytest.mark.parametrize("text", [PENSION, PENSION.upper(), PENSION.replace(" ", "\n")])
def test_complete_disability_pension_condition_links_translated_selector(text):
    assert sc._source_exception_selector_is_relevant(text, SELECTOR)


@pytest.mark.parametrize(
    "text",
    [
        PENSION.replace("wegen seiner Behinderung ", ""),
        PENSION.replace("nach den gesetzlichen Vorschriften ", ""),
        PENSION.replace("wegen seiner Behinderung", "wegen seines Alters"),
        PENSION.replace("zustehen", "nicht zustehen"),
        PENSION + " sofern der Antrag gestellt wurde",
        "wenn das Kind im Inland wohnt",
    ],
)
def test_pension_translation_does_not_hide_other_conditions(text):
    assert not sc._source_exception_selector_is_relevant(text, SELECTOR)


@pytest.mark.parametrize(
    "selector",
    [
        "pension_or_ongoing_payment_is_not_stated_as_due_because_of_disability",
        "pension_or_ongoing_payment_is_stated_as_due_because_of_age",
        "pension_or_ongoing_payment_is_stated_as_due",
        SELECTOR + "_and_application_filed",
        "document_is_accepted",
    ],
)
def test_pension_translation_requires_complete_corresponding_selector(selector):
    assert not sc._source_exception_selector_is_relevant(PENSION, selector)


@pytest.mark.parametrize("bullet", ["−", "–", "-"])
@pytest.mark.parametrize("age", [18, 25, 27])
def test_medical_onset_condition_ends_before_independent_effects_item(bullet, age):
    source = MEDICAL.replace("−", bullet).replace("25", str(age))
    assert sc._source_exception_condition_text(source) == (
        f"soweit das Kind das {age}. Lebensjahr vollendet hat"
    )


@pytest.mark.parametrize(
    "text",
    [
        MEDICAL.replace(
            "vollendet hat, und", "vollendet hat und ein Antrag gestellt wurde, und"
        ),
        MEDICAL.replace("Vorliegen der Behinderung", "Vorliegen einer Krankheit"),
        MEDICAL + " Eine weitere Voraussetzung ist die Antragstellung.",
        MEDICAL.replace("− Auswirkungen", "und Auswirkungen"),
    ],
)
def test_medical_condition_specialization_does_not_trim_changed_source(text):
    assert sc._source_exception_condition_text(text) != (
        "soweit das Kind das 25. Lebensjahr vollendet hat"
    )
    assert "Auswirkungen" in sc._source_exception_condition_text(text)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "no_pair",
        "wrong_output",
        "two_inputs",
        "wrong_period",
        "wrong_path",
        "constant",
    ],
)
def test_pension_pair_still_requires_executed_single_change_and_source_path(mutation):
    import functools

    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        extract_typed_numeric_inventory_occurrences_from_text,
        numeric_value_is_grounded,
    )

    source = "(1) " + PENSION
    cases = [
        {
            "period": "2025-07-01",
            "input": {SELECTOR: value, "unrelated": False},
            "output": {"result": value},
        }
        for value in (False, True)
    ]
    formula = SELECTOR
    path = "(1)"
    if mutation == "no_pair":
        cases = cases[:1]
    elif mutation == "wrong_output":
        cases[1]["output"]["result"] = False
    elif mutation == "two_inputs":
        cases[1]["input"]["unrelated"] = True
    elif mutation == "wrong_period":
        cases[1]["period"] = "2025-08-01"
    elif mutation == "wrong_path":
        path = "(2)"
    elif mutation == "constant":
        formula = f"{SELECTOR} or true"
        cases[0]["output"]["result"] = True
    content = f"""format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: de/guidance/bzst-dakg-2025/a-19-2/document-1
rules:
  - name: result
    kind: derived
    source: de/guidance/bzst-dakg-2025/a-19-2/document-1{path}
    versions:
      - formula: '{formula}'
"""
    result = sc.analyze_complete_source_unit(
        content,
        source,
        corpus_citation_path="de/guidance/bzst-dakg-2025/a-19-2/document-1",
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="de-DE"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    missing = any("[complete-source-unit:tests]" in issue for issue in result.issues)
    assert missing is (mutation is not None)

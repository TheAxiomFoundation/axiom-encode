"""Actual complete-source analysis, with normal numeric/proof adapters."""

import functools
import hashlib

import pytest
import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import validator_pipeline as v
from tests.test_worksheet_country_evidence import SOURCE, c, certify, fixture, owns


def analyze(payload, cases):
    payload = {**payload, "format": "rulespec/v1"}
    content = yaml.safe_dump(payload)
    bindings = sc.collect_artifact_numeric_bindings(
        content,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        imported_symbol_contents=(),
    )
    return sc.analyze_complete_source_unit(
        content,
        SOURCE,
        corpus_citation_path=c.CITE,
        source_context={c.CITE: SOURCE},
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            v.extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_values=tuple(x[1] for x in bindings),
        artifact_numeric_bindings=bindings,
    ).issues


def country_findings(issues):
    return [
        x
        for x in issues
        if "[complete-source-unit:source-explicit-conditions]" in x and c.OWNER in x
    ]


def test_normal_analyzer_country_evidence_is_local():
    payload, cases = fixture()
    issues = analyze(payload, cases)
    assert not country_findings(issues)
    assert issues  # This tiny contract does not implement the whole worksheet.
    assert any(
        "[complete-source-unit:tests]" in x or "[complete-source-unit:numeric" in x
        for x in issues
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "owner_antecedent_only",
        "threshold_other_clause",
        "wrong_amount_type",
        "wrong_owner_entity",
        "missing_assertion",
        "wrong_threshold",
        "negative_description",
    ],
)
def test_normal_analyzer_rejected_certificate_retains_country_obligation(mutation):
    payload, cases = fixture()
    owner, parameter = payload["rules"]
    if mutation == "owner_antecedent_only":
        owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = c.ANTECEDENT
    elif mutation == "threshold_other_clause":
        parameter["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "If the total non-business income taxes you paid to all foreign countries is $200 or less"
        )
    elif mutation == "wrong_amount_type":
        payload["inputs"][1]["unit"] = "USD"
    elif mutation == "wrong_owner_entity":
        owner["entity"] = "Household"
    elif mutation == "missing_assertion":
        cases[0]["output"] = {}
    elif mutation == "wrong_threshold":
        parameter["versions"][0]["formula"] = 201
    elif mutation == "negative_description":
        payload["inputs"][1]["description"] = "has no income"
    assert country_findings(analyze(payload, cases))


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("value", [[], {}, None, True, 887.0])
def test_malformed_exact_span_is_false(field, value):
    payload, cases = fixture()
    evidence = certify(payload, cases)
    assert not owns(evidence, payload, cases, **{field: value})


@pytest.mark.parametrize(
    "anchor",
    [
        "As a result, you do not have to complete this form.",
        "the amount of the federal foreign non-business income tax credit",
        "This form does not apply to residents of Quebec.",
        "If you must pay minimum tax and you were a resident of Manitoba,",
    ],
)
def test_injected_source_restriction_in_preceding_context_rejects(anchor):
    assert SOURCE.count(anchor) == 1
    source = SOURCE.replace(
        anchor, "Only residents with a disabled spouse qualify.\n" + anchor
    )
    payload, cases = fixture()
    payload["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    assert c.certify_worksheet_country(
        payload, test_cases=cases, source_text=source, corpus_citation_path=c.CITE
    ).unresolved

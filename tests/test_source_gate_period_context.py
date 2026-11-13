from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.validator_pipeline import ValidatorPipeline

SOURCE = "If the person is a resident and the person is a citizen, the credit applies."
CITATION = "xx/policy/example/credit"


def _issues(formula, *, alias=False, extra_input=None):
    def rule(name, formula):
        return {
            "name": name,
            "kind": "derived",
            "dtype": "Judgment",
            "entity": "Person",
            "period": "Year",
            "source": SOURCE,
            "versions": [{"effective_from": "2025-01-01", "formula": formula}],
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "formula",
                            "source": {
                                "corpus_citation_path": CITATION,
                                "excerpt": SOURCE,
                            },
                        }
                    ]
                }
            },
        }

    inputs = [
        {
            "name": "person_is_resident",
            "description": "The person is a resident.",
            "dtype": "Boolean",
        },
        {
            "name": "person_is_citizen",
            "description": "The person is a citizen.",
            "dtype": "Boolean",
        },
        {
            "name": "reference_date",
            "description": "Explicit reference date.",
            "dtype": "Date",
        },
    ]
    if extra_input:
        inputs.append(extra_input)
    for item in inputs:
        item.update(entity="Person", period="Year")
    rules = [rule("credit_applies", formula)]
    if alias:
        rules.append(rule("credit_reporting_required", "credit_applies"))
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": inputs,
        "rules": rules,
    }
    pipeline = ValidatorPipeline(
        Path("/tmp/rulespec-xx/xx"),
        Path("/tmp/unused"),
        enable_oracles=False,
        local_corpus_release=None,
        require_complete_source_unit=True,
    )
    issues = pipeline._complete_source_unit_issues(
        yaml.safe_dump(payload),
        validation_source_texts={CITATION: SOURCE},
        test_cases=[],
    )
    return [str(i) for i in issues if "source-explicit-conditions" in str(i)]


@pytest.mark.parametrize("intrinsic", ["period_start", "period_end"])
@pytest.mark.parametrize("alias", [False, True])
def test_scalar_context_preserves_resolved_direct_and_alias_fact_gates(
    intrinsic, alias
):
    assert not _issues(
        f"person_is_resident and person_is_citizen and reference_date == {intrinsic}",
        alias=alias,
    )


@pytest.mark.parametrize(
    "term",
    [
        "unknown_period_end",
        "Period_end",
        "period_end() and reference_date == period_end",
        "period_end and reference_date == period_end",
    ],
)
def test_unknown_call_or_mixed_judgment_positions_remain_closed(term):
    assert _issues(f"person_is_resident and person_is_citizen and {term}", alias=True)


def test_context_does_not_supply_missing_source_fact_even_with_shadow_description():
    assert _issues(
        "person_is_resident and reference_date == period_end",
        extra_input={
            "name": "period_end",
            "dtype": "Boolean",
            "description": "The person is a citizen.",
        },
    )
    assert _issues("reference_date == period_start and reference_date == period_end")
    assert _issues("period_end == period_end")


def test_bare_judgment_name_is_not_removed_as_scalar_context():
    assert not _issues(
        "person_is_resident and period_end",
        extra_input={
            "name": "period_end",
            "dtype": "Boolean",
            "description": "The person is a citizen.",
        },
    )

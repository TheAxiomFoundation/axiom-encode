"""A descriptive resource type does not identify the claimant as a Family."""

import pytest
import yaml

from axiom_encode.harness import validator_pipeline as v

QUOTE = (
    "Insurable earnings (QPIP) and pensionable earnings (QPP) of a person "
    "responsible for a family-type resource\nor an intermediate resource "
    "(line 8 of form LM-53-V)"
)


def document(excerpt):
    return yaml.safe_dump(
        {
            "format": "rulespec/v1",
            "rules": [
                {
                    "name": "resource_operator_earnings",
                    "kind": "derived",
                    "entity": "Person",
                    "dtype": "Money",
                    "unit": "CAD",
                    "period": "Year",
                    "versions": [{"effective_from": "2025-01-01", "formula": "amount"}],
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {
                                    "path": "versions[0].formula",
                                    "kind": "formula",
                                    "source": {"excerpt": excerpt},
                                }
                            ]
                        }
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "text", [QUOTE, QUOTE.upper(), QUOTE.replace("responsible for", "responsible\nfor")]
)
def test_explicit_person_resource_description_is_not_family_scope(text):
    assert v._classify_source_scope(text) is None
    assert v._classify_source_unit_entity(text) is None
    assert v.find_source_scope_consistency_issues(document(text)) == []


@pytest.mark.parametrize(
    "text,entity",
    [
        ("Family income determines eligibility.", "family"),
        ("Household resources determine eligibility.", "household"),
    ],
)
def test_genuine_unit_requirement_survives_resource_description(text, entity):
    combined = QUOTE + ". " + text
    assert v._classify_source_scope(combined) == "unit"
    assert v._classify_source_unit_entity(combined) == entity
    assert len(v.find_source_scope_consistency_issues(document(combined))) == 1


@pytest.mark.parametrize(
    "text",
    [
        "A family-type resource has income requirements.",
        "A person works for a family-type resource with income requirements.",
        "A person responsible for a family benefit meets resource requirements.",
        "A person responsible for family income meets eligibility requirements.",
    ],
)
def test_incomplete_or_different_description_keeps_unit_signal(text):
    assert v._classify_source_scope(text) == "unit"
    assert v._classify_source_unit_entity(text) == "family"


def test_genuine_mixed_person_and_family_scope_remains_mixed():
    text = QUOTE + ". A person is eligible only if Family income meets the requirement."
    assert v._classify_source_scope(text) is None
    assert v._classify_source_unit_entity(text) == "family"


def test_resource_modifier_mask_preserves_offsets_and_unrelated_words():
    text = QUOTE + ". Family income is tested."
    masked = v._mask_person_resource_type_description(text)
    start = text.index("family-type")
    assert masked == text[:start] + " " * len("family") + text[start + len("family") :]
    assert len(masked) == len(text)
    assert [i for i, c in enumerate(masked) if c == "\n"] == [
        i for i, c in enumerate(text) if c == "\n"
    ]

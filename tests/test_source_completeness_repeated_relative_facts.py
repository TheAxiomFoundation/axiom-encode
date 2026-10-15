"""Restrictive property qualifiers retain independent source fact gates."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc

CITATION = "us-co/statute/39/39-1-103"
SOURCE = """(b) For property tax years beginning on or after January 1, 1999, if the actual
use as of the assessment date of any real property that is located in a limited
gaming district but that is not used for limited gaming is used as residential real
property, the real property shall be classified as residential real property, and the
assessing officer shall determine the actual value of said real property as of the
assessment date by applying the market approach to appraisal."""
FACTS = (
    "property_located_in_limited_gaming_district",
    "property_not_used_for_limited_gaming",
    "property_used_as_residential_real_property",
)


def _payload(formula):
    return {
        "inputs": [
            {"name": name, "dtype": "Boolean", "description": name.replace("_", " ")}
            for name in FACTS
        ],
        "rules": [
            {
                "name": "property_classified_as_residential",
                "kind": "derived",
                "dtype": "Judgment",
                "source": CITATION,
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
                "versions": [{"formula": formula}],
            }
        ],
    }


def _issues(formula):
    payload = _payload(formula)
    return sc._opaque_same_source_condition_input_issues(
        payload,
        source_text=SOURCE,
        branches=sc.recognize_source_structure(SOURCE, corpus_citation_path=CITATION),
        principal_rules={rule["name"]: rule for rule in payload["rules"]},
        corpus_citation_path=CITATION,
    )


@pytest.mark.parametrize("source", [SOURCE, " ".join(SOURCE.split())])
def test_repeated_relative_qualifiers_retain_district_gaming_and_use_facts(source):
    gates = sc._source_conjunctive_fact_gates(source)
    assert len(gates) == 3
    assert {"located", "limited", "gaming", "district"} <= gates[0][1]
    assert gates[1][1] == frozenset({"used", "limited", "gaming"})
    assert {"used", "residential", "real", "property"} <= gates[2][1]
    assert not _issues(" and ".join(FACTS))


@pytest.mark.parametrize("missing", FACTS)
def test_repeated_relative_omission_is_rejected_on_executed_formula(missing):
    formula = " and ".join(FACTS)
    omitted = " and ".join(name for name in FACTS if name != missing)
    assert not _issues(formula)
    assert _issues(omitted)
    present = dict.fromkeys(FACTS, True)
    absent = {**present, missing: False}
    for expression, environment, expected in (
        (formula, present, True),
        (formula, absent, False),
        (omitted, absent, True),
    ):
        execution = sc._execute_formula_text(
            expression, environment=environment, constant_environment={}
        )
        assert execution is not None
        assert sc._formula_execution_runtime_value(execution) is expected


@settings(max_examples=30, deadline=None)
@given(breaks=st.sets(st.integers(min_value=1, max_value=90), max_size=45))
def test_repeated_relative_fact_gates_ignore_physical_wraps(breaks):
    flat = " ".join(SOURCE.split())
    wrapped = "".join(
        word + ("\n" if index in breaks else " ")
        for index, word in enumerate(flat.split(), start=1)
    ).strip()
    assert sc._source_conjunctive_fact_gates(wrapped) == (
        sc._source_conjunctive_fact_gates(flat)
    )
    assert len(sc._source_conjunctive_fact_gates(wrapped)) == 3


@pytest.mark.parametrize(
    "body",
    [
        "the owner knows that property is located in a district but that it is residential",
        "no property that is located in a district but that is not used for gaming is residential",
        "neither property that is located in a district but that is not used for gaming is residential",
        "property that is located in a district or zone but that is not used for gaming is residential",
        "property that is located in a district but that is not used for gaming or commerce is residential",
        "property that is located in a district but that is owned by a person who is blind is residential",
        "property that is located in a district but that is taxed because it is owned by a spouse is residential",
        "property that is located in a district but that is owned by a spouse is considered a building that is residential",
        "property that is located in a district but that may be used for gaming is residential",
        "property that is located in a district but that is not used for gaming",
        'property that is described as "located in a district but that is not used for gaming" is residential',
    ],
)
def test_repeated_relative_boundaries_reject_ambiguous_scopes(body):
    assert sc._source_gate_restrictive_repeated_relative_but_boundaries(body) == ()

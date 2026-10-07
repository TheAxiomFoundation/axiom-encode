"""Glossary labels cannot exempt source-stated operative duration boundaries."""

from __future__ import annotations

import functools
import random

import pytest

from axiom_encode.harness import source_completeness as completeness
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_numeric_recall_structural_properties import (
    PROFILES,
    US,
    _inventory,
    _recall_issues,
)

SUBDIVISIONS = ("(d)(5)", "(d)(5)(A)", "(d)(5)(A)(i)")
OPERATIVE_PREFIXES = (
    "A household is eligible if it contains ",
    "An applicant qualifies if they belong to ",
    "Applicants receive SNAP benefits if they are ",
    "Qualification requires ",
    "Qualification requires:\n",
    "Eligibility requires:\n",
    "To qualify, applicants must include:\n",
    "A household is eligible:\n",
    # A new line authenticates the apparent heading, but it cannot discard a
    # policy effect that precedes it in the same source branch.
    "A household is eligible if its members include:\n",
    "A household is eligible if "
    + "the relevant immigration status has been documented and verified " * 4
    + "it contains:\n",
)


def _source(subdivisions, *, prefix="", value=1):
    return (
        f"{prefix}Parolees Paroled into the U.S. under section 212{subdivisions} "
        f"of the INA for a period of at least {value} years."
    )


def _root(source):
    return completeness.SourceStructureBranch(
        (), "source-unit", "source unit", source, 0, len(source)
    )


def _bounds(source, profile):
    return [
        (occurrence.value, occurrence.raw)
        for _branch, occurrence in completeness._source_boundary_obligations(
            (_root(source),),
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text,
                profile=profile,
            ),
        )
    ]


def _boundary_test_issues(source, profile, *, value=1, witnessed=False):
    rule_name = "household_eligible"
    principal_rules = {
        rule_name: {
            "name": rule_name,
            "kind": "derived",
            "dtype": "Judgment",
            "versions": [{"formula": f"parole_years >= {value}"}],
        }
    }
    cases = (
        {
            "name": "duration endpoint" if witnessed else "above duration endpoint",
            "period": "2026-01-01",
            "input": {"parole_years": value if witnessed else value + 1},
            "output": {rule_name: "holds"},
        },
    )
    issues = completeness._companion_test_issues(
        principal_rules,
        parameter_rules={},
        principal_rule_paths={rule_name: {()}},
        principal_formula_clause_rules={},
        formula_branches=(),
        branches=(_root(source),),
        source_text=source,
        corpus_citation_path=US,
        deferred_paths=set(),
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text,
            profile=profile,
        ),
        numeric_value_is_grounded=numeric_value_is_grounded,
        formula_environment={},
        source_bound_constant_occurrences={},
        declared_input_names={"parole_years"},
    )
    return [issue for issue in issues if "source-stated boundary input" in issue]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("prefix", OPERATIVE_PREFIXES)
def test_operative_prefix_before_parolee_heading_keeps_duration_boundary(
    profile, subdivisions, prefix
):
    source = _source(subdivisions, prefix=prefix)
    assert [(item.value, item.raw) for item in _inventory(source, US, profile)] == [
        (1, "1")
    ]
    assert _recall_issues(source, US, profile)
    assert not _recall_issues(source, US, profile, (1,))
    assert _bounds(source, profile) == [(1, "1")]


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("prefix", OPERATIVE_PREFIXES)
@pytest.mark.parametrize("witnessed", (False, True), ids=("missing", "present"))
def test_operative_parolee_duration_requires_an_executed_boundary_case(
    profile, subdivisions, prefix, witnessed
):
    source = _source(subdivisions, prefix=prefix)
    assert (
        bool(_boundary_test_issues(source, profile, witnessed=witnessed))
        is not witnessed
    )


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("subdivisions", SUBDIVISIONS)
@pytest.mark.parametrize("heading_prefix", ("", "Alien Group Description:\n"))
def test_actual_parolee_glossary_heading_keeps_scalar_without_boundary_case(
    profile, subdivisions, heading_prefix
):
    source = _source(subdivisions, prefix=heading_prefix)
    assert [(item.value, item.raw) for item in _inventory(source, US, profile)] == [
        (1, "1")
    ]
    assert _recall_issues(source, US, profile)
    assert not _recall_issues(source, US, profile, (1,))
    assert _bounds(source, profile) == []
    assert _boundary_test_issues(source, profile) == []


@pytest.mark.parametrize("profile", PROFILES)
def test_generated_glossary_boundary_exemption_requires_a_standalone_heading(profile):
    """Changing the cited subdivision or duration never waives a policy bound."""

    generator = random.Random(1779)
    values = (1, 1.5, *generator.sample(range(2, 100), 4))
    for subdivisions in SUBDIVISIONS:
        for value in values:
            glossary = _source(subdivisions, value=value)
            assert _bounds(glossary, profile) == []
            assert _boundary_test_issues(glossary, profile, value=value) == []
            for prefix in OPERATIVE_PREFIXES:
                operative = _source(subdivisions, prefix=prefix, value=value)
                assert _bounds(operative, profile) == [(value, str(value))]
                assert _boundary_test_issues(operative, profile, value=value)
                assert not _boundary_test_issues(
                    operative, profile, value=value, witnessed=True
                )

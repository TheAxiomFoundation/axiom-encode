"""Source-boundary feedback keeps raw excerpts and stored numeric values."""

from __future__ import annotations

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness.source_completeness import (
    MISSING_BOUNDARY_DETAIL_MARKER,
    _boundary_value_text,
    analyze_complete_source_unit,
)
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

CITATION = "us-zz/statute/9"


SOURCE = (
    "(a) A household whose countable income is less than $500 and whose "
    "housing cost is low is eligible.\n"
    "(b) Housing cost is low if the rent is less than $500.\n"
)


def _rule(name, formula, excerpt, *, parameter=False):
    return {
        "name": name,
        "kind": "parameter" if parameter else "derived",
        "dtype": "Money" if parameter else "Judgment",
        **({} if parameter else {"entity": "Household", "period": "Month"}),
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "amount" if parameter else "formula",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                ]
            }
        },
        "versions": [{"effective_from": "2026-01-01", "formula": formula}],
    }


def _payload(*, direct_rent=False, wrong_parameter=False, literal=False):
    income_threshold = "rent_limit" if wrong_parameter else "income_limit"
    if literal:
        income_threshold = "500"
    return {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {"name": name, "entity": "Household", "dtype": "Money", "period": "Month"}
            for name in ("countable_income", "rent")
        ],
        "rules": [
            _rule("income_limit", "500", "less than $500 and", parameter=True),
            _rule("rent_limit", "500", "rent is less than $500", parameter=True),
            _rule(
                "housing_cost_low",
                "rent < rent_limit",
                "Housing cost is low if the rent is less than $500.",
            ),
            _rule(
                "household_eligible",
                f"countable_income <= {income_threshold} and "
                + ("rent < rent_limit" if direct_rent else "housing_cost_low"),
                SOURCE.splitlines()[0].removeprefix("(a) "),
            ),
        ],
    }


def _case(name, income, rent, **output):
    return {
        "name": name,
        "period": "2026-01",
        "input": {"rent": rent, "countable_income": income},
        "output": output,
    }


CASES = [
    _case("low", 100, 100, household_eligible="holds", housing_cost_low="holds"),
    _case(
        "high", 900, 900, household_eligible="not_holds", housing_cost_low="not_holds"
    ),
    _case("rent boundary", 100, 500, housing_cost_low="not_holds"),
    _case(
        "rent blocks",
        100,
        900,
        household_eligible="not_holds",
        housing_cost_low="not_holds",
    ),
]


def _analyze(payload, source=SOURCE, cases=CASES):
    return analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_source_anchor_keeps_threshold_after_citation_cleanup():
    source = SOURCE.replace(
        "(a) A household",
        "(a) Under Public Law 115-97, Public Law 116-136, Public Law 117-169, "
        "and Public Law 118-5, a household",
    )
    payload = _payload()
    payload["rules"][-1]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
        source.splitlines()[0].removeprefix("(a) ")
    )
    result = _analyze(payload, source)
    [issue] = [item for item in result.issues if "boundary input; missing:" in item]
    context = issue.split("source `", 1)[1].split("`", 1)[0]
    assert "countable income is less than $500" in context


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=100, deadline=None)
def test_boundary_text_round_trips_the_stored_number_as_a_yaml_number(value):
    parsed = yaml.safe_load(_boundary_value_text(value))
    assert type(parsed) in {int, float}
    assert float(parsed) == value


@pytest.mark.parametrize(
    ("source_amount", "value", "rendered"),
    [
        ("$1,000,000", 1_000_000, "1000000"),
        ("$1,234,567", 1_234_567, "1234567"),
        ("$12,345.67", 12345.67, "12345.67"),
    ],
)
def test_boundary_label_list_details_and_typed_fields_keep_exact_value(
    source_amount, value, rendered
):
    source = f"(a) A household whose income is less than {source_amount} is eligible.\n"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [
            {
                "name": "income",
                "entity": "Household",
                "dtype": "Money",
                "period": "Month",
            }
        ],
        "rules": [
            _rule("limit", str(value), f"less than {source_amount}", parameter=True),
            _rule("eligible", "income < limit", source.removeprefix("(a) ").strip()),
        ],
    }
    cases = [
        {
            "name": name,
            "period": "2026-01",
            "input": {"income": number},
            "output": {"eligible": outcome},
        }
        for name, number, outcome in [
            ("low", 0, "holds"),
            ("high", value * 2, "not_holds"),
        ]
    ]
    result = _analyze(payload, source, cases)
    [issue] = [item for item in result.issues if "boundary input; missing:" in item]
    label = f"(a)={rendered}"
    assert f"missing: {label}." in issue
    assert f"{MISSING_BOUNDARY_DETAIL_MARKER} {label} at" in issue
    assert f"compared directly with {rendered} `income`" in issue
    assert issue.expected == f"case at {label}"
    assert issue.found == f"no case at {label}"
    [obligation] = result.companion_test_obligations
    assert obligation.label == label
    assert float(yaml.safe_load(rendered)) == float(value)
    assert not isinstance(yaml.safe_load(rendered), str)

import functools

import pytest
import yaml

from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)


def _analysis(mutation=None):
    citation = "us/guidance/example/household-benefit"
    source = "For households of one and two persons, the benefit is 8 percent of income, rounded to the nearest whole dollar."

    def rule(name, formula, kind="derived"):
        return {
            "name": name,
            "kind": kind,
            "dtype": "Money" if kind == "derived" else "Decimal",
            "unit": "USD",
            "entity": "Household",
            "period": "Month",
            "metadata": {
                "proof": {
                    "atoms": [
                        {
                            "path": "versions[0].formula",
                            "kind": "formula",
                            "source": {
                                "corpus_citation_path": citation,
                                "excerpt": source,
                            },
                        }
                    ]
                }
            },
            "versions": [{"effective_from": "2026-01-01", "formula": formula}],
        }

    raw = rule("raw_benefit", "income * rate")
    benefit = rule(
        "benefit",
        "if household_size == lower or household_size == upper:\n  raw_benefit\nelse: 0",
    )
    benefit["rounding"] = "half_up"
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "units": [{"name": "USD", "kind": "currency", "minor_units": 0}],
        "inputs": [
            {
                "name": "income",
                "dtype": "Money",
                "entity": "Household",
                "period": "Month",
            },
            {
                "name": "household_size",
                "dtype": "Count",
                "entity": "Household",
                "period": "Month",
            },
        ],
        "rules": [
            rule("rate", "0.08", "parameter"),
            rule("lower", "1", "parameter"),
            rule("upper", "2", "parameter"),
            raw,
            benefit,
        ],
    }
    cases = [
        {
            "name": str(i),
            "period": "2026-01",
            "input": {"income": income, "household_size": 2},
            "output": {"raw_benefit": value, "benefit": rounded},
        }
        for i, (income, value, rounded) in enumerate(
            [(1256.125, "100.49", 100), (1256.25, "100.50", 101)]
        )
    ]
    if mutation == "unasserted":
        for case in cases:
            case["output"].pop("raw_benefit")
    elif mutation == "tampered":
        for case in cases:
            case["output"]["raw_benefit"] = "99.1"
    elif mutation == "wrong_period":
        raw["versions"][0]["effective_from"] = "2027-01-01"
    elif mutation == "rounded_dependency":
        raw["rounding"] = "floor"
    elif mutation == "unreached":
        benefit["versions"][0]["formula"] = (
            "if household_size == lower or household_size == upper:\n  income\nelse: 0"
        )
    elif mutation == "wrong_rate":
        raw["versions"][0]["formula"] = "income * 0.09"
    elif mutation == "opaque":
        payload["imports"] = ["us:provider#opaque"]
        raw["versions"][0]["formula"] = "opaque * rate"
        raw["metadata"]["proof"]["atoms"].append(
            {
                "path": "versions[0].formula",
                "kind": "import",
                "import": {
                    "target": "us:provider#opaque",
                    "output": "opaque",
                    "hash": "sha256:" + "a" * 64,
                },
            }
        )
    if mutation == "transitive_opaque":
        payload["imports"] = ["us:provider#opaque"]
        bridge = rule("bridge", "opaque")
        bridge["metadata"]["proof"]["atoms"].append(
            {
                "path": "versions[0].formula",
                "kind": "import",
                "import": {
                    "target": "us:provider#opaque",
                    "output": "opaque",
                    "hash": "sha256:" + "a" * 64,
                },
            }
        )
        payload["rules"].append(bridge)
        raw["versions"][0]["formula"] = "bridge * rate"
        for case in cases:
            case["output"]["bridge"] = case["input"]["income"]
    return s.analyze_complete_source_unit(
        yaml.safe_dump(payload),
        source,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
        extract_numeric_grounding_occurrences=functools.partial(
            extract_typed_numeric_occurrences_from_text, profile="legacy"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_source_clause_combines_guard_with_replayed_asserted_intermediate():
    result = _analysis()
    assert not result.issues, result.issues


@pytest.mark.parametrize(
    "mutation",
    [
        "unasserted",
        "tampered",
        "wrong_period",
        "rounded_dependency",
        "unreached",
        "wrong_rate",
        "opaque",
    ],
)
def test_source_clause_does_not_borrow_unverified_or_unreached_arithmetic(mutation):
    result = _analysis(mutation)
    assert any("complete-source-unit:tests" in issue for issue in result.issues), (
        result.issues
    )


def test_source_clause_rejects_transitive_opaque_assertion_bridge():
    result = _analysis("transitive_opaque")
    assert any(
        "do not demonstrate formula branch" in issue for issue in result.issues
    ), result.issues

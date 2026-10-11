"""New identity-rate credit requires the exact source-owned computation."""

import functools

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)

PAGE = "us-fl/manual/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
NAMESPACE = "us-fl:policies/dcf/ess-program-policy-manual/2000-coverage-groups/page-5"
SOURCE = (
    "The SFU must meet the net income limit of 100% of the federal poverty "
    "level for the AG size and the asset limit of $4,500."
)
INVENTORY = functools.partial(
    extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)
GROUNDING = functools.partial(
    extract_typed_numeric_occurrences_from_text, profile="legacy"
)


def _issues(formula, at, *, rate=1, identity=None, citation=PAGE):
    def atom(text):
        return {
            "kind": "formula",
            "path": "versions[0].formula",
            "source": {"corpus_citation_path": citation, "excerpt": text},
        }

    constants = {"net_income_ratio": rate, "asset_limit": 4500}
    if identity is not None:
        constants["identity"] = identity
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": citation}},
        "inputs": [
            {
                "name": name,
                "entity": "Household",
                "dtype": "Money",
                "unit": "USD",
                "period": "Month",
            }
            for name in (
                "federal_poverty_level",
                "other_level",
                "sfu_net_income",
                "spouse_net_income",
                "sfu_countable_assets",
            )
        ],
        "rules": [
            {
                "name": name,
                "kind": "parameter",
                "dtype": "Money" if name == "asset_limit" else "Rate",
                "metadata": {"proof": {"atoms": [atom(SOURCE.rstrip("."))]}},
                "versions": [{"effective_from": "0001-01-01", "formula": str(value)}],
            }
            for name, value in constants.items()
        ]
        + [
            {
                "name": "meets_limits",
                "kind": "derived",
                "entity": "Household",
                "dtype": "Judgment",
                "period": "Month",
                "metadata": {"proof": {"atoms": [atom(SOURCE.rstrip("."))]}},
                "versions": [{"effective_from": "0001-01-01", "formula": formula}],
            }
        ],
    }
    cases = [
        {
            "name": name,
            "period": "2026-05",
            "input": {
                "federal_poverty_level": 1000,
                "other_level": 2000,
                "sfu_net_income": income,
                "spouse_net_income": income - 1000,
                "sfu_countable_assets": 4500,
            },
            "output": {f"{NAMESPACE}#meets_limits": expected},
        }
        for name, income, expected in (
            ("at_the_limits", at, "holds"),
            ("above_the_net_limit", at + 1, "not_holds"),
        )
    ]
    result = sc.analyze_complete_source_unit(
        yaml.safe_dump(payload, sort_keys=False),
        SOURCE,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=INVENTORY,
        extract_numeric_grounding_occurrences=GROUNDING,
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )
    return [
        issue
        for issue in result.issues
        if "Companion tests do not demonstrate formula branch" in issue
    ]


@pytest.mark.parametrize(
    ("threshold", "at"),
    (
        ("federal_poverty_level + net_income_ratio * federal_poverty_level", 2000),
        ("net_income_ratio * federal_poverty_level - 100", 900),
        ("net_income_ratio * federal_poverty_level * 2", 2000),
        ("net_income_ratio * federal_poverty_level / 2", 500),
        ("net_income_ratio * other_level", 2000),
    ),
)
@pytest.mark.parametrize(
    "citation", (PAGE, "us/statute/7/2014", "de/manual/benefits/example")
)
def test_identity_retry_rejects_unstated_arithmetic_and_wrong_base(
    threshold, at, citation
):
    formula = f"sfu_net_income <= ({threshold}) and sfu_countable_assets <= asset_limit"
    assert len(_issues(formula, at, citation=citation)) == 1


def test_identity_retry_cannot_borrow_an_unrelated_one_for_a_wrong_rate():
    formula = (
        "sfu_net_income <= identity * net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    assert len(_issues(formula, 2000, rate=2, identity=1)) == 1


@settings(max_examples=30, deadline=None)
@given(rate=st.integers(min_value=2, max_value=100))
def test_identity_retry_requires_the_rate_on_the_source_base_to_match(rate):
    formula = (
        "sfu_net_income <= identity * net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    assert not _matches(
        formula, {"net_income_ratio": rate, "identity": 1, "asset_limit": 4500}
    )


def _matches(formula, environment, *, source=SOURCE):
    branch = sc.SourceStructureBranch((), "formula", "limit", source, 0, len(source))
    execution = sc._FormulaExecution((), formula, None, False, environment)
    return sc._formula_execution_matches_source_branch(
        execution,
        branch,
        interval=None,
        formula_environment={},
        extract_numeric_occurrences=INVENTORY,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


@settings(max_examples=60, deadline=None)
@given(extra=st.integers(min_value=1, max_value=10000))
def test_identity_retry_never_credits_an_added_source_unstated_term(extra):
    formula = (
        f"sfu_net_income <= net_income_ratio * federal_poverty_level + {extra} "
        "and sfu_countable_assets <= asset_limit"
    )
    assert not _matches(formula, {"net_income_ratio": 1, "asset_limit": 4500})


def test_identity_retry_does_not_transfer_the_income_rate_to_assets():
    formula = (
        "sfu_countable_assets <= net_income_ratio * federal_poverty_level "
        "and sfu_net_income <= asset_limit"
    )
    assert not _matches(formula, {"net_income_ratio": 1, "asset_limit": 4500})


def test_identity_retry_does_not_use_a_spouses_income_for_the_sfu_limit():
    formula = (
        "spouse_net_income <= net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    assert len(_issues(formula, 2000)) == 1


@pytest.mark.parametrize(
    ("income", "assets"),
    (
        ("other_net_income", "sfu_countable_assets"),
        ("spouse_net_income", "sfu_countable_assets"),
        ("gross_net_income", "sfu_countable_assets"),
        ("sfu_net_income", "spouse_assets"),
        ("sfu_net_income", "sfu_uncountable_assets"),
    ),
)
def test_identity_retry_keeps_all_operative_subject_qualifiers(income, assets):
    formula = (
        f"{income} <= net_income_ratio * federal_poverty_level "
        f"and {assets} <= asset_limit"
    )
    assert not _matches(formula, {"net_income_ratio": 1, "asset_limit": 4500})


@pytest.mark.parametrize("actor", ("spouse of the SFU", "other SFU", "former SFU"))
def test_identity_retry_binds_the_exact_source_actor(actor):
    formula = (
        "sfu_net_income <= net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    source = SOURCE.replace("The SFU", f"The {actor}")
    assert not _matches(
        formula, {"net_income_ratio": 1, "asset_limit": 4500}, source=source
    )


def test_identity_retry_cannot_borrow_a_separate_actors_asset_limit():
    formula = (
        "sfu_net_income <= net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    source = SOURCE.replace(
        "and the asset limit", ". The spouse must meet the asset limit"
    )
    assert not _matches(
        formula, {"net_income_ratio": 1, "asset_limit": 4500}, source=source
    )


@pytest.mark.parametrize(
    ("subject", "value"), (("sfu_net_income", 1), ("sfu_countable_assets", 4500))
)
def test_identity_retry_cannot_replace_a_source_subject_with_a_constant(subject, value):
    formula = (
        "sfu_net_income <= net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    assert not _matches(
        formula, {"net_income_ratio": 1, "asset_limit": 4500, subject: value}
    )


@pytest.mark.parametrize("base", ("poverty", "ag_size", "level_for_ag_size"))
def test_identity_retry_binds_the_whole_primary_percentage_base(base):
    formula = (
        f"sfu_net_income <= net_income_ratio * {base} "
        "and sfu_countable_assets <= asset_limit"
    )
    assert not _matches(formula, {"net_income_ratio": 1, "asset_limit": 4500})


@pytest.mark.parametrize(
    "base",
    ("poverty_level", "fpl", "federal_poverty_level_for_ag_size", "fpl_for_ag_size"),
)
def test_identity_retry_accepts_complete_base_concepts_and_source_qualifiers(base):
    formula = (
        f"sfu_net_income <= net_income_ratio * {base} "
        "and sfu_countable_assets <= asset_limit"
    )
    assert _matches(formula, {"net_income_ratio": 1, "asset_limit": 4500})


@pytest.mark.parametrize("extra", ("identity <= 1", "other_level <= 4500", "1 <= 1"))
def test_identity_retry_rejects_unrelated_comparisons(extra):
    formula = (
        "sfu_net_income <= net_income_ratio * federal_poverty_level "
        f"and sfu_countable_assets <= asset_limit and {extra}"
    )
    assert not _matches(
        formula, {"net_income_ratio": 1, "asset_limit": 4500, "identity": 1}
    )


@pytest.mark.parametrize(
    "citation", (PAGE, "us/statute/7/2014", "de/manual/benefits/example")
)
def test_exact_source_owned_identity_product_is_demonstrated(citation):
    formula = (
        "sfu_net_income <= net_income_ratio * federal_poverty_level "
        "and sfu_countable_assets <= asset_limit"
    )
    assert _issues(formula, 1000, citation=citation) == []


def test_folded_match_credit_is_preserved(monkeypatch):
    calls = []

    def leaf_match(*args, keep_identity_factors, **kwargs):
        calls.append(keep_identity_factors)
        return not keep_identity_factors

    monkeypatch.setattr(sc, "_formula_execution_leaf_matches_source_branch", leaf_match)
    assert _matches("sfu_net_income <= federal_poverty_level + 1", {})
    assert calls == [False]

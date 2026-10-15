"""Retain paired coverage for the verbatim manual-condition review probes.

The exemption is bound to two complete instructions at their canonical source.
When clause extraction separates added qualifications, the complete source body
must still authenticate the instruction; an altered body keeps its obligation.
"""

from __future__ import annotations

import functools
import itertools
import keyword

import pytest
from hypothesis import Phase, example, given, settings
from hypothesis import strategies as st

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness.validator_pipeline import (
    extract_named_scalar_occurrences,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.manual_procedure_differential import (
    analyze_with_manual_procedure_differential,
    manual_procedure_requires_paired_witness,
)

MANUAL_CITATION = (
    "us-fl/manual/dcf/ess-program-policy-manual/2600-calculation-of-benefits/page-8"
)
QUANTITY_SEQUENCE = (
    "when gross income is determined, that income is compared to the allowable "
    "gross income limit for the assistance group's size."
)
CORPUS_CLAUSES = (
    "Chapter: 2600 Calculation of Benefits Program: FS 1 2610.0000 Food Stamps "
    "When the eligibility specialist has determined available income as per "
    "Chapters 1800 and 2400, the policy in this chapter must be used to perform "
    "the budgets and tests to determine eligibility for benefits and the benefit "
    "amount. 2610.0100 BUDGETS AND TEST CALCULATIONS (FS) Each program has "
    "budgets and tests that must be executed in order to determine eligibility.",
    "Refer to Appendix A-1 for the 165% Need Standard Table. 2610.0104.03 Gross "
    "Income Test (FS) When gross income is determined, that income is compared "
    "to the allowable gross income limit for the assistance group's size.",
)


def _analyze(
    source,
    formula,
    cases,
    *,
    rule_name="result",
    description="",
    citation="de/statute/estg/32a",
    mark_source=True,
):
    source_path = citation + ("(1)" if mark_source else "")
    content = f"""\
format: rulespec/v1
module:
  source_verification:
    corpus_citation_path: {citation}
rules:
  - name: {rule_name}
    kind: derived
    description: {description}
    source: {source_path}
    versions:
      - formula: '{formula}'
"""
    return analyze_with_manual_procedure_differential(
        content,
        f"(1) {source}" if mark_source else source,
        corpus_citation_path=citation,
        test_cases=cases,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
        extract_named_scalars=extract_named_scalar_occurrences,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def _case(inputs, output, *, rule_name="result"):
    return {
        "name": str(inputs),
        "period": "2026",
        "input": inputs,
        "output": {rule_name: output},
    }


def _paired_issue(result):
    return any("paired positive/blocking" in issue for issue in result.issues)


def _diagnostic_case(name, inputs, output):
    return {
        "name": str(name),
        "period": "2026",
        "input": inputs,
        "output": {"result": output},
    }


def _diagnostic_probes():
    """Keep the earlier review runners' literal sources and companion grids."""

    both_source = "If both earned and unearned income are greater than or equal to the income threshold, the applicant is ineligible."
    object_source = "If income is greater than or equal to both earned and unearned income threshold, the applicant is ineligible."
    probes = [
        (
            "shared_subject_projection",
            both_source,
            "if earned >= income_threshold and unearned_income >= income_threshold: false else: true",
            [
                _diagnostic_case(
                    (a, b),
                    {
                        "earned": a,
                        "unearned_income": b,
                        "earned_income": 99,
                        "income_threshold": 100,
                    },
                    not (a >= 100 and b >= 100),
                )
                for a, b in itertools.product((99, 100, 101), repeat=2)
            ],
            True,
        ),
        (
            "shared_object_singular_projection",
            object_source,
            "if income >= earned and income >= unearned_income_threshold: false else: true",
            [
                _diagnostic_case(
                    (a, b),
                    {
                        "income": 100,
                        "earned": a,
                        "unearned_income_threshold": b,
                        "earned_income_threshold": 200,
                    },
                    not (100 >= a and 100 >= b),
                )
                for a, b in itertools.product((99, 100, 101), repeat=2)
            ],
            False,
        ),
        (
            "shared_object_plural_projection",
            "If income is greater than or equal to both earned and unearned income thresholds, the applicant is ineligible.",
            "if income >= earned and income >= unearned_income_thresholds: false else: true",
            [
                _diagnostic_case(
                    (a, b),
                    {
                        "income": 100,
                        "earned": a,
                        "unearned_income_thresholds": b,
                        "earned_income_threshold": 200,
                        "unearned_income_threshold": 200,
                    },
                    not (100 >= a and 100 >= b),
                )
                for a, b in itertools.product((99, 100, 101), repeat=2)
            ],
            False,
        ),
    ]
    literal_cases = [
        _diagnostic_case(assets, {"income": 99, "assets": assets}, assets < 100)
        for assets in (99, 100)
    ]
    probes.extend(
        (
            (
                "both_literal_ge_projection",
                "If both income and assets are greater than or equal to 100, the applicant is ineligible.",
                "if assets >= 100: false else: true",
                literal_cases,
                False,
            ),
            (
                "both_literal_equals_or_exceeds",
                "If both income and assets are equal to or exceed 100, the applicant is ineligible.",
                "if assets >= 100: false else: true",
                literal_cases,
                True,
            ),
            (
                "compound_literal_suffix_projection",
                "If total earned and unearned income is equal to or exceeds 100, the applicant is ineligible.",
                "if unearned_income >= 100: false else: true",
                [
                    _diagnostic_case(
                        value,
                        {
                            "total_earned_and_unearned_income": 99,
                            "unearned_income": value,
                        },
                        value < 100,
                    )
                    for value in (99, 100)
                ],
                True,
            ),
        )
    )
    for connector in ("and", "or"):
        source = f"If income is greater than or equal to the income threshold {connector} the applicant is a resident, the applicant is ineligible."
        for formula, kind in (
            ("not is_resident", "bool-only"),
            (
                "if other_income >= income_threshold "
                + connector
                + " is_resident: false else: true",
                "other-operand",
            ),
        ):
            cases = []
            for x in (99, 100, 101):
                for resident in (False, True):
                    blocked = (
                        resident
                        if kind == "bool-only"
                        else (
                            x >= 100 and resident
                            if connector == "and"
                            else x >= 100 or resident
                        )
                    )
                    cases.append(
                        _diagnostic_case(
                            (x, resident),
                            {
                                "income": x,
                                "income_threshold": 100,
                                "other_income": x,
                                "is_resident": resident,
                            },
                            not blocked,
                        )
                    )
            probes.append((f"{connector}_{kind}", source, formula, cases, False))
        literal_source = f"If income is greater than or equal to 100 {connector} the applicant is a resident, the applicant is ineligible."
        for comparison in ("income > 100", "income >= 101"):
            cases = [
                _diagnostic_case(
                    (x, resident),
                    {"income": x, "is_resident": resident},
                    not (
                        x > 100 and resident
                        if connector == "and"
                        else x > 100 or resident
                    ),
                )
                for x in (99, 100, 101, 102)
                for resident in (False, True)
            ]
            probes.append(
                (
                    f"literal_{connector}_{comparison}",
                    literal_source,
                    f"if {comparison} {connector} is_resident: false else: true",
                    cases,
                    False,
                )
            )
    probes.append(
        (
            "ordinary_side_equality_two_points",
            "If income is equal to or exceeds the income threshold, the applicant is ineligible.",
            "if income >= income_threshold and income != income_threshold: false else: true",
            [
                _diagnostic_case(
                    value, {"income": value, "income_threshold": 100}, output
                )
                for value, output in ((100, True), (101, False))
            ],
            True,
        )
    )
    source = "If income is greater than or equal to the income threshold and the applicant is a resident, the applicant is ineligible."
    probes.append(
        (
            "resident_only_toggle_at_equality",
            source,
            "if income >= income_threshold and is_resident: false else: true",
            [
                _diagnostic_case(
                    (x, resident),
                    {"income": x, "income_threshold": 100, "is_resident": resident},
                    not (x >= 100 and resident),
                )
                for x, resident in ((100, True), (100, False))
            ],
            False,
        )
    )
    probes.append(
        (
            "disjunct_without_assets_boundary",
            "If income is greater than or equal to the income threshold or assets are greater than or equal to the asset threshold, the applicant is ineligible.",
            "if income >= income_threshold or assets >= asset_threshold: false else: true",
            [
                _diagnostic_case(
                    (income, 99),
                    {
                        "income": income,
                        "income_threshold": 100,
                        "assets": 99,
                        "asset_threshold": 100,
                        "is_resident": False,
                    },
                    not (income >= 100 or 99 >= 100),
                )
                for income in (99, 100, 101)
            ],
            False,
        )
    )
    for mirrored in (False, True):
        probes.append(
            (
                "equality_only_three_points_mirrored"
                if mirrored
                else "equality_only_three_points",
                "If the income threshold is less than or equal to income, the applicant is ineligible."
                if mirrored
                else "If income is greater than or equal to the income threshold, the applicant is ineligible.",
                "if income == income_threshold: false else: true",
                [
                    _diagnostic_case(
                        income,
                        {"income": income, "income_threshold": 100},
                        income != 100,
                    )
                    for income in (99, 100, 101)
                ],
                True,
            )
        )
    probes.append(
        (
            "dishonest_both_four_combinations_formula_output",
            "If both income and assets are greater than or equal to the income threshold, the applicant is ineligible.",
            "if (income >= income_threshold and assets >= income_threshold) or (income < income_threshold and assets < income_threshold): false else: true",
            [
                _diagnostic_case(
                    (income, assets),
                    {"income": income, "assets": assets, "income_threshold": 100},
                    not (
                        (income >= 100 and assets >= 100)
                        or (income < 100 and assets < 100)
                    ),
                )
                for income, assets in ((100, 100), (99, 100), (100, 99), (99, 99))
            ],
            True,
        )
    )
    for kind, connector, permission in itertools.product(
        ("resident", "disabled"), ("and", "or"), (False, True)
    ):
        flag = "is_" + kind
        household = "household_" + flag
        source = (
            f"If the applicant is a {kind} {connector} the household is a {kind}, "
            "the credit is " + ("allowed." if permission else "not allowed.")
        )
        if kind == "disabled":
            source = source.replace("is a disabled", "is disabled")
        cases = [
            _diagnostic_case(
                (a, b),
                {flag: a, household: b},
                (a and b if connector == "and" else a or b) is permission,
            )
            for a, b in itertools.product((False, True), repeat=2)
        ]
        dishonest_cases = [
            {**case, "output": {"result": case["input"][flag] is permission}}
            for case in cases
        ]
        probes.append(
            (
                f"dropped_household_{kind}_{connector}_{permission}",
                source,
                f"if {flag}: {'true' if permission else 'false'} else: {'false' if permission else 'true'}",
                dishonest_cases,
                False,
            )
        )
    return tuple(probes)


DIAGNOSTIC_PROBES = _diagnostic_probes()


@pytest.mark.parametrize(
    "label, source, formula, cases, expected_paired_issue",
    DIAGNOSTIC_PROBES,
    ids=[probe[0] for probe in DIAGNOSTIC_PROBES],
)
def test_verbatim_diagnostic_probes_preserve_baseline_witness_behavior(
    label, source, formula, cases, expected_paired_issue
):
    result = _analyze(source, formula, cases)
    assert manual_procedure_requires_paired_witness(source)
    assert expected_paired_issue is not None, (
        f"Immutable-parent replay required: {label}"
    )
    assert _paired_issue(result) is expected_paired_issue


@pytest.mark.parametrize(
    "label, source, formula, cases, expected_paired_issue",
    DIAGNOSTIC_PROBES,
    ids=[probe[0] for probe in DIAGNOSTIC_PROBES],
)
def test_verbatim_diagnostic_case_grids_keep_their_paired_issue(
    label, source, formula, cases, expected_paired_issue
):
    result = _analyze(source, formula, cases)
    assert manual_procedure_requires_paired_witness(source)
    if expected_paired_issue:
        assert _paired_issue(result), label
    else:
        # #1920: restore absolute rejection when the baseline witness is fixed.
        assert result.issues == result.empty_list_issues, label


@pytest.mark.parametrize(
    "tail",
    (
        "Only for households with an elderly member.",
        "The household must have a disabled member.",
        "This comparison applies only to households with an elderly member.",
        "That income is compared only for residents.",
        "If the applicant is resident, the test applies.",
    ),
)
@pytest.mark.parametrize("include_sequence", (False, True))
def test_verbatim_sentence_boundary_diagnostics_preserve_baseline_extraction(
    tail, include_sequence
):
    source = QUANTITY_SEQUENCE + " " + tail if include_sequence else tail
    cases = [
        {
            "name": "only enabled case",
            "input": {"is_disabled": True, "countable_income": 1000},
            "output": {"gross_income_test_applies": True},
        }
    ]
    result = _analyze(
        source,
        "true",
        cases,
        rule_name="gross_income_test_applies",
        citation="us-fl/manual/test/section-1",
        mark_source=False,
    )
    assert manual_procedure_requires_paired_witness(source)
    expected_paired_issue = include_sequence or tail.startswith("If ")
    assert _paired_issue(result) is expected_paired_issue


@pytest.mark.parametrize(
    "tail",
    (
        "Only for households with an elderly member.",
        "The household must have a disabled member.",
        "If the applicant is resident, the test applies.",
        "The benefit amount is zero.",
        "Review the application.",
    ),
)
@pytest.mark.parametrize("separator", ("; ", ". ", ", ", " and ", " -- ", "\n"))
def test_verbatim_tail_delimiters_keep_preexisting_paired_issue(tail, separator):
    source = QUANTITY_SEQUENCE.rstrip(".") + separator + tail
    result = _analyze(
        source,
        "true",
        [
            {
                "name": "only enabled case",
                "input": {"is_disabled": True, "countable_income": 1000},
                "output": {"gross_income_test_applies": True},
            }
        ],
        rule_name="gross_income_test_applies",
        citation="us-fl/manual/test/section-1",
        mark_source=False,
    )
    assert manual_procedure_requires_paired_witness(source)
    assert _paired_issue(result)


def _analyze_diagnostic_rules(source, rules, cases):
    citation = "us-fl/manual/test/section-1"
    content = (
        "format: rulespec/v1\nmodule:\n  source_verification:\n"
        f"    corpus_citation_path: {citation}\nrules:\n"
    )
    for name, path, formula in rules:
        content += (
            f"  - name: {name}\n    kind: derived\n    source: {citation}{path}\n"
            f'    versions: [{{formula: "{formula}"}}]\n'
        )
    return analyze_with_manual_procedure_differential(
        content,
        source,
        corpus_citation_path=citation,
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


def _diagnostic_rule_probes():
    numbered = (
        "Assistance groups must meet the gross income standards with the following exceptions:\n"
        "1. Households with an elderly member.\n"
        "2. Households with a disabled member."
    )
    examples = (
        (
            "determination_actual_decision",
            "When the eligibility specialist has determined that the applicant is disabled, apply the earned income deduction.",
        ),
        (
            "determination_joined_age_applicability",
            "When the eligibility specialist has determined income and the household includes a member over 60, subtract $100 from countable income.",
        ),
        (
            "exclusion_embedded_disability_applicability",
            "When applying the gross income test, the income of a person who qualifies only if disabled is excluded from this calculation.",
        ),
        ("exception_items_no_cues", numbered),
    )
    probes = []
    for label, source in examples:
        for formula in ("true", "is_disabled", "countable_income - 100"):
            if formula == "countable_income - 100" and "joined" not in label:
                continue
            rules = (
                [
                    ("households_with_elderly_member", "(1)", formula),
                    ("households_with_disabled_member", "(2)", formula),
                ]
                if "items" in label
                else [("earned_income_deduction_applies", "", formula)]
            )
            for include_header_rule in (False, True) if "items" in label else (False,):
                actual_rules = (
                    [("gross_income_test_applies", "", formula), *rules]
                    if include_header_rule
                    else rules
                )
                expected = 900 if formula == "countable_income - 100" else True
                probes.append(
                    (
                        f"{label}_{formula}_{include_header_rule}",
                        source,
                        actual_rules,
                        [
                            {
                                "name": "only enabled case",
                                "input": {
                                    "is_disabled": True,
                                    "countable_income": 1000,
                                },
                                "output": {
                                    name: expected for name, _, _ in actual_rules
                                },
                            }
                        ],
                    )
                )
    examples = (
        (
            "prefix_subject_to",
            "Subject to the household having a disabled member, " + QUANTITY_SEQUENCE,
            [("gross_income_test_applies", "", "true")],
        ),
        (
            "prefix_operative_standard",
            "The gross income standard applies only to households with an elderly member; "
            + QUANTITY_SEQUENCE,
            [("gross_income_test_applies", "", "true")],
        ),
        (
            "prefix_operative_standard_joined",
            "The gross income standard applies only to households with an elderly member, and "
            + QUANTITY_SEQUENCE,
            [("gross_income_test_applies", "", "true")],
        ),
        (
            "inline_parenthetic_item",
            "Assistance groups must meet the standards with the following exceptions:\n1. Households with an elderly member; and (2) Households with a disabled member.",
            [
                ("gross_income_test_applies", "", "true"),
                ("elderly_exception", "(1)", "is_elderly"),
            ],
        ),
        (
            "multi_sentence_item",
            "Assistance groups must meet the standards with the following exceptions:\n1. Households with an elderly member. That member must also be disabled.\n2. Households with a resident member.",
            [
                ("gross_income_test_applies", "", "true"),
                ("elderly_exception", "(1)", "is_elderly"),
                ("resident_exception", "(2)", "is_resident"),
            ],
        ),
    )
    for label, source, rules in examples:
        values = (
            ((False, False), (True, False), (False, True))
            if "\n" in source
            else ((True, True),)
        )
        cases = [
            {
                "name": str((elderly, resident)),
                "period": "2026",
                "input": {
                    "is_elderly": elderly,
                    "is_resident": resident,
                    "is_disabled": True,
                },
                "output": {
                    name: elderly
                    if formula == "is_elderly"
                    else resident
                    if formula == "is_resident"
                    else True
                    for name, _, formula in rules
                },
            }
            for elderly, resident in values
        ]
        probes.append((label, source, rules, cases))
    return tuple(probes)


DIAGNOSTIC_RULE_PROBES = _diagnostic_rule_probes()


@pytest.mark.parametrize(
    "label, source, rules, cases",
    DIAGNOSTIC_RULE_PROBES,
    ids=[probe[0] for probe in DIAGNOSTIC_RULE_PROBES],
)
def test_verbatim_rule_diagnostics_keep_preexisting_paired_issue(
    label, source, rules, cases
):
    result = _analyze_diagnostic_rules(source, rules, cases)
    assert manual_procedure_requires_paired_witness(source)
    assert _paired_issue(result)


@pytest.mark.parametrize(
    "source, condition",
    (
        (
            "If income is greater than or equal to the income threshold, the applicant is ineligible.",
            "income == income_threshold",
        ),
        (
            "If the income threshold is less than or equal to income, the applicant is ineligible.",
            "income == income_threshold",
        ),
    ),
)
def test_round5_equality_only_formula_keeps_inclusive_obligation(source, condition):
    cases = [
        _case({"income": income, "income_threshold": 100}, income != 100)
        for income in (99, 100)
    ]
    assert _paired_issue(_analyze(source, f"if {condition}: false else: true", cases))


def test_round5_sparse_boolean_pairs_keep_compound_obligation():
    source = "If both income and assets are greater than or equal to the income threshold, the applicant is ineligible."
    formula = (
        "if (income >= income_threshold and assets >= income_threshold) or "
        "(income < income_threshold and assets < income_threshold): false else: true"
    )
    cases = [
        _case(
            {"income": income, "assets": assets, "income_threshold": 100},
            not ((income >= 100 and assets >= 100) or (income < 100 and assets < 100)),
        )
        for income, assets in ((100, 100), (99, 100), (100, 99))
    ]
    assert _paired_issue(_analyze(source, formula, cases))


@pytest.mark.parametrize("honest", (False, True))
def test_round5_negative_principal_preserves_source_output_polarity(honest):
    rule_name = "applicant_is_ineligible"
    source = "If the applicant is disabled, the applicant is ineligible."
    formula = (
        "if is_disabled: true else: false"
        if honest
        else "if is_disabled: false else: true"
    )
    cases = [
        _case(
            {"is_disabled": disabled},
            disabled if honest else not disabled,
            rule_name=rule_name,
        )
        for disabled in (False, True)
    ]
    result = _analyze(
        source,
        formula,
        cases,
        rule_name=rule_name,
        description="The applicant is ineligible.",
    )
    assert _paired_issue(result) is not honest


# The source wording is retained exactly, including unsupported grammar.
REVIEW_CLAUSES = (
    "When the eligibility specialist has determined that the applicant is disabled, apply the earned income deduction.",
    "When the eligibility specialist has determined income and the household includes a member over 60, subtract $100 from countable income.",
    "When the applicant is determined eligible, apply the earned income deduction.",
    "When the worker has determined income, apply the earned income deduction.",
    "Subject to the household having a disabled member, " + QUANTITY_SEQUENCE,
    "Only for households with an elderly member, " + QUANTITY_SEQUENCE,
    "For residents, " + QUANTITY_SEQUENCE,
    "Provided that the household has a disabled member, " + QUANTITY_SEQUENCE,
    "The gross income standard applies only to households with an elderly member; "
    + QUANTITY_SEQUENCE,
    "The gross income standard applies only to households with an elderly member, and "
    + QUANTITY_SEQUENCE,
    "The gross income standard applies only to households with an elderly member, or "
    + QUANTITY_SEQUENCE,
    "Subject to the household having a disabled member; " + QUANTITY_SEQUENCE,
    "For residents; " + QUANTITY_SEQUENCE,
    "(Subject to the household having a disabled member), " + QUANTITY_SEQUENCE,
    "If the household has a disabled member, " + QUANTITY_SEQUENCE,
    QUANTITY_SEQUENCE + " Only for households with an elderly member.",
    QUANTITY_SEQUENCE + " The household must have a disabled member.",
    QUANTITY_SEQUENCE + " If the applicant is resident, the test applies.",
    QUANTITY_SEQUENCE + " The benefit amount is zero.",
    QUANTITY_SEQUENCE + " Review the application.",
    "Assistance groups must meet the standards with the following exceptions:\n1. Households with an elderly member; and (2) Households with a disabled member.",
    "Assistance groups must meet the standards with the following exceptions:\n1. Households with an elderly member. That member must also be disabled.\n2. Households with a resident member.",
    "Assistance groups must meet the gross income standards with the following exceptions:\n1. Households with an elderly member.\n2. Households with a disabled member.",
    "If the applicant qualifies, the credit is not allowed, provided income does not exceed the income threshold.",
    "The individual (and spouse or children if required) is ineligible, provided income does not exceed the income threshold.",
    "If income is equal to or exceeds the income threshold, the applicant is ineligible.",
    "If the applicant qualifies, the worker reviews the record, provided income does not exceed the income threshold.",
    "If the applicant qualifies, the applicant who is ineligible receives the credit, provided income does not exceed the income threshold.",
    "If the applicant qualifies, the credit is not allowed to expire, provided income does not exceed the income threshold.",
    "When applying the gross income test, the income of a person who qualifies only if disabled is excluded from this calculation.",
    "When applying the gross income test, the income of a person who qualifies if disabled is excluded from this calculation.",
    "When applying the gross income test, the income of a person who qualifies is excluded from this calculation.",
    "When applying the gross income test, the income of a person unless resident is excluded from this calculation.",
    "When applying the gross income test, the income of a person when eligible is excluded from this calculation.",
    "When applying the gross income test, the income of the elderly or disabled person and their spouse is excluded from this calculation.",
    "When applying the gross income test, the income of boarders is excluded from this determination.",
    "When applying the gross income test, the income of the others is excluded from this test.",
    "When applying the gross income test, the income of the others is excluded from this calculation.",
    "If income is greater than or equal to the income threshold and the applicant is a resident, the applicant is ineligible.",
    "If income is greater than or equal to the income threshold or the applicant is a resident, the applicant is ineligible.",
    "If income is greater than or equal to the income threshold or assets are greater than or equal to the asset threshold, the applicant is ineligible.",
    "If both income and assets are greater than or equal to the income threshold, the applicant is ineligible.",
    "If the applicant is disabled, the credit is allowed, provided income does not exceed the income threshold.",
    "Unless the applicant is disabled, the credit is allowed, provided income does not exceed the income threshold.",
    "If total earned and unearned income is greater than or equal to the income threshold, the applicant is ineligible.",
    "If income is greater than or equal to the earned and unearned income threshold, the applicant is ineligible.",
    "If both earned and unearned income are greater than or equal to the income threshold, the applicant is ineligible.",
    "If income is greater than or equal to both earned and unearned income threshold, the applicant is ineligible.",
    "If income is greater than or equal to both earned and unearned income thresholds, the applicant is ineligible.",
    "If both earned and unearned income are greater than or equal to both gross and net income threshold, the applicant is ineligible.",
    "If the applicant is a resident and the household is a resident, the credit is allowed.",
    "If the applicant is disabled and the household is disabled, the credit is allowed.",
    "If the applicant qualifies and the household qualifies, the credit is allowed.",
    "If income is greater than or equal to the income threshold, the applicant is ineligible.",
    "If the income threshold is less than or equal to income, the applicant is ineligible.",
    "If the applicant is disabled, the applicant is ineligible.",
)


@pytest.mark.parametrize("clause", REVIEW_CLAUSES)
def test_verbatim_review_conditions_keep_their_obligation(clause):
    assert not sc._source_condition_is_manual_procedure(clause)
    assert manual_procedure_requires_paired_witness(
        clause, corpus_citation_path=MANUAL_CITATION
    )


@pytest.mark.parametrize("clause", REVIEW_CLAUSES)
def test_verbatim_review_conditions_keep_issue_without_paired_cases(clause):
    result = _analyze(clause, "true", [_case({}, True)], citation=MANUAL_CITATION)
    assert _paired_issue(result)


@pytest.mark.parametrize(
    "label, source, formula, cases, expected_paired_issue",
    DIAGNOSTIC_PROBES,
    ids=[probe[0] for probe in DIAGNOSTIC_PROBES],
)
def test_verbatim_diagnostic_conditions_keep_issue_without_paired_cases(
    label, source, formula, cases, expected_paired_issue
):
    result = _analyze(source, formula, cases[:1], citation=MANUAL_CITATION)
    assert _paired_issue(result), label


@pytest.mark.parametrize("connector", ("and", "or", "then"))
@pytest.mark.parametrize("space", ("", " "))
def test_round1_attached_inline_exception_marker_keeps_header_issue(connector, space):
    source = (
        "Assistance groups must meet the gross income standards with the following exceptions:\n"
        "1. Households with an elderly member.\n"
        "2. Households with a disabled member."
    ).replace(".\n2. ", f"; {connector}2.{space}")
    result = _analyze(
        source,
        "true",
        [_case({}, True)],
        citation=MANUAL_CITATION,
        mark_source=False,
    )
    assert any("following exceptions" in issue for issue in result.issues)


@pytest.mark.parametrize("clause", CORPUS_CLAUSES)
def test_complete_corpus_instructions_have_no_paired_issue(clause):
    result = _analyze(
        clause,
        "true",
        [_case({}, True)],
        citation=MANUAL_CITATION,
        mark_source=False,
    )
    assert not _paired_issue(result)


@pytest.mark.parametrize("clause", CORPUS_CLAUSES)
@pytest.mark.parametrize(
    "prefix",
    (
        "For residents; ",
        "For residents. ",
        "For residents.\n",
        "Only for households with an elderly member; ",
        "Only for households with an elderly member. ",
        "Only for households with an elderly member.\n",
        "Subject to the household having a disabled member; ",
        "If the applicant is disabled; ",
    ),
)
def test_leading_qualification_keeps_full_corpus_instruction_issue(clause, prefix):
    result = _analyze(
        prefix + clause,
        "true",
        [_case({}, True)],
        citation=MANUAL_CITATION,
        mark_source=False,
    )
    assert _paired_issue(result)


@pytest.mark.parametrize("clause", CORPUS_CLAUSES)
@pytest.mark.parametrize(
    "suffix",
    (
        " Only for households with an elderly member.",
        "\nOnly for households with an elderly member.",
        " For residents.",
        "\nFor residents.",
        " This comparison applies only to households with an elderly member.",
        " If the applicant is disabled, the applicant is ineligible.",
        " Unless the household is a resident, the applicant is ineligible.",
    ),
)
def test_following_qualification_keeps_full_corpus_instruction_obligation(
    clause, suffix
):
    text = clause + suffix
    assert manual_procedure_requires_paired_witness(
        text, corpus_citation_path=MANUAL_CITATION
    )
    assert manual_procedure_requires_paired_witness(
        clause, corpus_citation_path=MANUAL_CITATION, source_text=text
    )
    result = _analyze(
        text,
        "true",
        [_case({}, True)],
        citation=MANUAL_CITATION,
        mark_source=False,
    )
    assert _paired_issue(result)


_SYMBOLIC_OPERANDS = st.from_regex(r"[a-z][a-z_]{0,15}", fullmatch=True).filter(
    lambda name: (
        not keyword.iskeyword(name)
        and name not in {"unless", "provided", "when", "only", "except"}
    )
)
_NUMERIC_AMOUNTS = st.one_of(
    st.integers(min_value=-1000, max_value=1_000_000),
    st.decimals(
        min_value=0,
        max_value=1_000_000,
        places=2,
        allow_nan=False,
        allow_infinity=False,
    ),
)


@settings(
    max_examples=64,
    deadline=None,
    phases=(Phase.explicit, Phase.generate),
    database=None,
)
@example(
    clause=CORPUS_CLAUSES[0],
    condition="if the applicant is disabled",
    position="before",
    separator=", ",
)
@example(
    clause=CORPUS_CLAUSES[1],
    condition="unless the applicant is a resident",
    position="after",
    separator="; ",
)
@example(
    clause=CORPUS_CLAUSES[0],
    condition="provided the applicant qualifies",
    position="before",
    separator=". ",
)
@example(
    clause=CORPUS_CLAUSES[1],
    condition="when the household is eligible",
    position="after",
    separator=" and ",
)
@example(
    clause=CORPUS_CLAUSES[0],
    condition="only for households with an elderly member",
    position="before",
    separator="\n",
)
@example(
    clause=CORPUS_CLAUSES[1],
    condition="except for households with a disabled member",
    position="after",
    separator=", ",
)
@given(
    clause=st.sampled_from(CORPUS_CLAUSES),
    condition=st.one_of(
        st.sampled_from(
            (
                "if the applicant is disabled",
                "unless the applicant is a resident",
                "provided the applicant qualifies",
                "when the household is eligible",
                "only for households with an elderly member",
                "except for households with a disabled member",
                "income is greater than the threshold",
                "income is less than or equal to the limit",
                "income equals the threshold",
                "income exceeds the limit",
                "income >= income_threshold",
                "assets < asset_threshold",
                "the benefit amount is $100",
                "the household has a disabled member",
            )
        ),
        st.builds(
            lambda operand, operator, amount: f"{operand} {operator} {amount}",
            _SYMBOLIC_OPERANDS,
            st.sampled_from(("<", "<=", "==", "!=", ">=", ">")),
            _NUMERIC_AMOUNTS,
        ),
        st.builds(lambda amount: f"the benefit amount is ${amount}", _NUMERIC_AMOUNTS),
    ),
    position=st.sampled_from(("before", "after")),
    separator=st.sampled_from((", ", "; ", ". ", " and ", "\n")),
)
def test_comparison_or_condition_never_loses_its_obligation(
    clause, condition, position, separator
):
    text = (
        condition + separator + clause
        if position == "before"
        else clause + separator + condition
    )
    assert not sc._source_condition_is_manual_procedure(text)
    assert manual_procedure_requires_paired_witness(
        text, corpus_citation_path=MANUAL_CITATION
    )
    assert manual_procedure_requires_paired_witness(
        clause, corpus_citation_path=MANUAL_CITATION, source_text=text
    )

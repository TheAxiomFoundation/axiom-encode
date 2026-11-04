from copy import deepcopy
from functools import partial

import pytest
import yaml

from axiom_encode.harness import source_completeness as c
from axiom_encode.harness import validator_pipeline as v

CITATION = "ca/policy/example/annual-tax-credit"
HEADER = (
    "Use this form to calculate the foreign non-business income tax credit for 2025"
)
CLAUSE = "The amount is total foreign taxes paid for 2025 divided by 2."
SOURCE = HEADER + ".\n" + CLAUSE
EXTRACT = partial(
    v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
)


def fixture():
    rule = {
        "name": "annual_result",
        "kind": "derived",
        "entity": "Person",
        "dtype": "Decimal",
        "period": "Year",
        "unit": "USD",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": "taxes / 2",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": f"versions[0].{field}",
                        "kind": "formula" if field == "formula" else "definition",
                        "source": {
                            "corpus_citation_path": CITATION,
                            "excerpt": excerpt,
                        },
                    }
                    for field, excerpt in [
                        ("formula", CLAUSE),
                        ("effective_from", HEADER),
                        ("effective_to", HEADER),
                    ]
                ]
            }
        },
    }
    case = {
        "name": "annual",
        "period": "2025-06-01",
        "input": {"taxes": 100},
        "output": {"annual_result": 50},
    }
    return rule, case


def witnesses(rule, case, source=SOURCE, clause=CLAUSE):
    start = source.index(clause)
    branch = c.SourceStructureBranch(
        (), "formula-clause", "annual", clause, start, start + len(clause)
    )
    covered = set()
    found = c._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules={rule["name"]: rule},
        rule_names={rule["name"]},
        asserted_by_rule={rule["name"]: [case]},
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=source,
        represented_annual_spans=covered,
    )
    return found, covered, branch


def test_executed_full_year_representation_covers_only_in_clause_year():
    rule, case = fixture()
    found, covered, branch = witnesses(rule, case)
    assert found
    assert covered == {
        (SOURCE.index("2025", len(HEADER)), SOURCE.index("2025", len(HEADER)) + 4)
    }
    assert (SOURCE.index("2025"), SOURCE.index("2025") + 4) not in covered
    assert branch.text == SOURCE[branch.start : branch.end]
    # Context-free callers still require the year and divisor.
    assert {
        x.value
        for x in c._formula_branch_computation_occurrences(
            branch, interval=None, extract_numeric_occurrences=EXTRACT
        )
    } == {2025, 2}


@pytest.mark.parametrize(
    "change",
    [
        "missing_from",
        "missing_to",
        "one_day",
        "wrong_year",
        "cross_year",
        "no_period",
        "outside_period",
        "wrong_result",
        "no_assertion",
        "missing_date_proof",
        "wrong_citation",
        "fabricated_date_excerpt",
        "unrelated_formula_proof",
        "duplicate_start",
    ],
)
def test_insufficient_annual_evidence_never_represents_year(change):
    rule, case = fixture()
    version = rule["versions"][0]
    if change == "missing_from":
        version.pop("effective_from")
    elif change == "missing_to":
        version.pop("effective_to")
    elif change == "one_day":
        version["effective_from"] = version["effective_to"] = "2025-06-01"
    elif change == "wrong_year":
        version.update(effective_from="2024-01-01", effective_to="2024-12-31")
    elif change == "cross_year":
        version["effective_to"] = "2026-12-31"
    elif change == "no_period":
        case.pop("period")
    elif change == "outside_period":
        case["period"] = "2026-01-01"
    elif change == "wrong_result":
        case["output"]["annual_result"] = 51
    elif change == "no_assertion":
        case["output"] = {}
    elif change == "missing_date_proof":
        rule["metadata"]["proof"]["atoms"].pop()
    elif change == "wrong_citation":
        rule["metadata"]["proof"]["atoms"][-1]["source"]["corpus_citation_path"] = (
            "ca/policy/elsewhere"
        )
    elif change == "fabricated_date_excerpt":
        rule["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = (
            HEADER + " invented"
        )
    elif change == "unrelated_formula_proof":
        rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = HEADER
    elif change == "duplicate_start":
        rule["versions"].append(deepcopy(version))
    _, covered, _ = witnesses(rule, case)
    assert not covered


def test_correct_dates_and_assertion_do_not_hide_wrong_arithmetic():
    rule, case = fixture()
    rule["versions"][0]["formula"] = "taxes * 2"
    case["output"]["annual_result"] = 200
    found, covered, _ = witnesses(rule, case)
    assert not found and not covered


@pytest.mark.parametrize(
    "clause",
    [
        "Divide total foreign taxes paid for 2024 by 2.",
        "Divide income by 2025.",
        "Pay $2025.",
        "Divide total foreign taxes paid for birth year 2025 by 2.",
        "Divide total foreign taxes paid before 2025 by 2.",
        'Divide "total foreign taxes paid for 2025" by 2.',
    ],
)
def test_other_numeric_roles_remain_obligations(clause):
    rule, case = fixture()
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    _, covered, _ = witnesses(rule, case, HEADER + ".\n" + clause, clause)
    assert not covered


def test_global_accounting_requires_executed_owner_and_retains_header_occurrence():
    rule, case = fixture()
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": CITATION}},
        "inputs": [{"name": "taxes", "dtype": "Decimal", "entity": "Person"}],
        "rules": [rule],
    }
    kwargs = dict(
        corpus_citation_path=CITATION,
        extract_numeric_occurrences=EXTRACT,
        extract_named_scalars=v.extract_named_scalar_occurrences,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        artifact_numeric_values=(2,),
    )
    positive = c.analyze_complete_source_unit(
        yaml.safe_dump(payload), SOURCE, test_cases=[case], **kwargs
    )
    negative = c.analyze_complete_source_unit(
        yaml.safe_dump(payload), SOURCE, test_cases=[], **kwargs
    )
    assert (
        positive.covered_source_numeric_occurrence_count
        == negative.covered_source_numeric_occurrence_count + 1
    )
    assert any(
        "numeric-recall" in issue and "2025" in issue for issue in positive.issues
    )
    assert not any(
        "do not demonstrate formula branch" in issue for issue in positive.issues
    )


def test_actual_ontario_denominator_preserves_the_rate_obligation():
    clause = (
        "Divide it by the total foreign taxes paid for 2025 (to get this total, "
        "add the amount on line 7 of Part 4 of\nForm T691, Alternative Minimum Tax, "
        "divided by 66.6666% and the amount on line 8 of Part 4 of Form T691)."
    )
    rule, case = fixture()
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    rule["versions"][0]["formula"] = "paid / (amt / 0.666666 + special)"
    case["input"] = {"paid": 3, "amt": 0.666666, "special": 1}
    case["output"]["annual_result"] = 1.5
    found, covered, branch = witnesses(rule, case, HEADER + ".\n" + clause, clause)
    assert found and len(covered) == 1
    # Even a correctly asserted different rate cannot discharge the source rate.
    rule["versions"][0]["formula"] = "paid / (amt / 0.333333 + special)"
    case["output"]["annual_result"] = 1
    found, covered, _ = witnesses(rule, case, HEADER + ".\n" + clause, clause)
    assert not found and not covered


def test_structural_cleanup_and_identical_money_year_do_not_expand_coverage():
    rule, case = fixture()
    source = HEADER + ".\n1. " + CLAUSE + "\nThe payment is $2025."
    found, covered, branch = witnesses(rule, case, source)
    assert found and len(covered) == 1
    mask = c.authoritative_numeric_recall_text(c._mask_numeric_spans(source, covered))
    assert "$2025" in mask
    assert mask.count("2025") == 2  # header plus genuine monetary operand
    assert len(c.authoritative_numeric_recall_text(source)) < len(source)
    assert source[branch.start : branch.end] == CLAUSE


@pytest.mark.parametrize(
    "suffix", [" dollars", " days", " months", " people", " times"]
)
def test_year_shaped_noncalendar_quantities_are_not_annual_qualifiers(suffix):
    rule, case = fixture()
    clause = CLAUSE.replace("2025 divided", "2025" + suffix + " divided")
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    _, covered, _ = witnesses(rule, case, HEADER + ".\n" + clause, clause)
    assert not covered


def test_unused_dated_decoy_does_not_supply_an_undated_owner():
    rule, case = fixture()
    decoy = deepcopy(rule)
    decoy["name"] = "unused_annual_decoy"
    rule["versions"][0].pop("effective_to")
    source = SOURCE
    start = source.index(CLAUSE)
    branch = c.SourceStructureBranch(
        (), "formula-clause", "annual", CLAUSE, start, len(source)
    )
    covered = set()
    found = c._formula_branch_test_witnesses(
        branch,
        corpus_citation_path=CITATION,
        principal_rules={rule["name"]: rule, decoy["name"]: decoy},
        rule_names={rule["name"], decoy["name"]},
        asserted_by_rule={rule["name"]: [case], decoy["name"]: []},
        extract_numeric_occurrences=EXTRACT,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        formula_environment={},
        dependency_cache={},
        execution_cache={},
        source_text=source,
        represented_annual_spans=covered,
    )
    assert not found and not covered


def test_identical_year_valued_arithmetic_operand_in_same_clause_is_not_hidden():
    rule, case = fixture()
    clause = CLAUSE[:-1] + " plus 2025."
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    source = HEADER + ".\n1. " + clause
    found, covered, _ = witnesses(rule, case, source, clause)
    assert not found and not covered  # omitted monetary addition remains an obligation
    rule["versions"][0]["formula"] = "taxes / 2 + 2025"
    case["output"]["annual_result"] = 2075
    found, covered, branch = witnesses(rule, case, source, clause)
    assert found and len(covered) == 1
    masked = c._mask_numeric_spans(source, covered)
    assert "plus 2025" in masked
    assert source[branch.start : branch.end] == clause


@pytest.mark.parametrize(
    "start,end",
    [
        ("2025-07-01", "2025-12-31"),
        ("2025-07-01", "2025-07-31"),
        ("2025-01-01", "2025-12-31"),
    ],
)
def test_superseding_or_overlapping_version_cannot_claim_full_year(start, end):
    rule, case = fixture()
    rule["versions"].append(
        {"effective_from": start, "effective_to": end, "formula": "taxes / 3"}
    )
    _, covered, _ = witnesses(rule, case)
    assert not covered


@pytest.mark.parametrize(
    "prefix,suffix",
    [
        ('"\n', ""),
        ("“\n", ""),
        ("'\n", ""),
        ("‘\n", ""),
        ("«\n", ""),
        ("“\n", '"'),
        ('"\n', "”"),
        ("[\n", ")"),
        ("(\n", ""),
        ("", "]"),
    ],
)
def test_unbalanced_quote_or_bracket_context_cannot_authorize_year(prefix, suffix):
    rule, case = fixture()
    _, covered, _ = witnesses(rule, case, prefix + SOURCE + suffix)
    assert not covered


@pytest.mark.parametrize(
    "context",
    [
        "Taxpayer's instructions.\n",
        "Taxpayer’s instructions.\n",
        "Taxpayers' instructions.\n",
        "Taxpayers’ instructions.\n",
        "Enter “0” when instructed.\n",
        'Enter "0" when instructed.\n',
        "Enter '0' when instructed.\n",
        "Enter ‘0’ when instructed.\n",
        "See [the instructions (below)].\n",
    ],
)
def test_balanced_typography_and_legitimate_apostrophes_preserve_evidence(context):
    rule, case = fixture()
    found, covered, _ = witnesses(rule, case, context + SOURCE)
    assert found and len(covered) == 1

"""Exact source cardinality owns count and, independently, annual period."""

from pathlib import Path

import pytest

from axiom_encode.harness import source_completeness as c
from axiom_encode.harness import validator_pipeline as v

SOURCE = (Path(__file__).parent / "fixtures/t2036_operand_gates/source.txt").read_text()
CLAUSE = "If you paid tax to more than one jurisdiction in 2025, calculate this amount according to note (3) of Form T2209."
HEADER = (
    "Use this form to calculate the foreign non-business income tax credit for 2025"
)
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
NAME = "jurisdictions_tax_paid_count"


def fixture():
    rule = {
        "name": "net_income",
        "kind": "derived",
        "dtype": "Money",
        "entity": "Person",
        "period": "Year",
        "versions": [
            {
                "effective_from": "2025-01-01",
                "effective_to": "2025-12-31",
                "formula": f"if {NAME} > 1: allocation - deduction else: 0",
            }
        ],
        "metadata": {
            "proof": {
                "atoms": [
                    {
                        "path": "versions[0].formula",
                        "kind": "formula",
                        "source": {"corpus_citation_path": CITATION, "excerpt": CLAUSE},
                    },
                    *[
                        {
                            "path": "versions[0]." + field,
                            "kind": "definition",
                            "source": {
                                "corpus_citation_path": CITATION,
                                "excerpt": HEADER,
                            },
                        }
                        for field in ("effective_from", "effective_to")
                    ],
                ]
            }
        },
    }
    cases = [
        {
            "name": str(n),
            "period": {
                "period_kind": "tax_year",
                "start": "2025-01-01",
                "end": "2025-12-31",
            },
            "input": {NAME: n, "allocation": 110000, "deduction": 10000},
            "output": {"net_income": 100000 if n == 2 else 0},
        }
        for n in (1, 2)
    ]
    inputs = {
        NAME: {"name": NAME, "dtype": "Integer", "entity": "Person", "period": "Year"}
    }
    return rule, cases, inputs


def run(rule, cases, inputs, source=SOURCE, clause=CLAUSE, environment=None):
    start = source.index(clause)
    branch = c.SourceStructureBranch(
        ("3",), "exception-clause", "(3)", clause, start, start + len(clause)
    )
    by = {"net_income": cases}
    witnesses = c._toggled_formula_numeric_selectors(
        {"net_income": rule}, asserted_by_rule=by, formula_environment=environment or {}
    )
    spans = set()
    missing = c._unwitnessed_exception_branches(
        (branch,),
        source_text=source,
        corpus_citation_path=CITATION,
        principal_rules={"net_income": rule},
        principal_rule_paths={"net_income": {("3",)}},
        asserted_by_rule=by,
        toggled_exception_selectors=witnesses,
        extract_numeric_occurrences=v.extract_typed_numeric_occurrences_from_text,
        input_declarations=inputs,
        formula_environment=environment or {},
        represented_annual_spans=spans,
    )
    return missing, spans


def test_actual_source_count_and_annual_occurrence_have_separate_evidence():
    rule, cases, inputs = fixture()
    missing, spans = run(rule, cases, inputs)
    assert not missing
    assert spans == {(4269, 4273)}
    assert SOURCE[4269:4273] == "2025"
    assert rule["versions"][0]["formula"].find("> 1") >= 0


def test_nonannual_cardinality_does_not_record_year():
    rule, cases, inputs = fixture()
    clause = CLAUSE.replace(" in 2025", "")
    source = SOURCE.replace(CLAUSE, clause)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    inputs[NAME]["period"] = "Month"
    missing, spans = run(rule, cases, inputs, source, clause)
    assert not missing
    assert spans == set()


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_event",
        "wrong_dtype",
        "wrong_entity",
        "wrong_period_type",
        "wrong_period",
        "crossyear",
        "missing_end",
        "wrong_proof",
        "bare_date",
        "wrong_date_source",
        "superseded",
        "opaque",
        "dead",
        "wrong_threshold",
        "wrong_assertion",
        "extra_input",
        "alias_collision",
        "negative_count",
    ],
)
def test_unowned_or_uncorroborated_count_never_records_annual_span(mutation):
    rule, cases, inputs = fixture()
    if mutation == "wrong_event":
        inputs[NAME]["name"] = "jurisdictions_tax_payable_count"
        inputs = {"jurisdictions_tax_payable_count": inputs[NAME]}
    elif mutation == "wrong_dtype":
        inputs[NAME]["dtype"] = "Decimal"
    elif mutation == "wrong_entity":
        inputs[NAME]["entity"] = "Household"
    elif mutation == "wrong_period_type":
        inputs[NAME]["period"] = "Month"
    elif mutation == "wrong_period":
        for case in cases:
            case["period"] = "2024"
    elif mutation == "crossyear":
        for case in cases:
            case["period"]["end"] = "2026-12-31"
    elif mutation == "missing_end":
        for case in cases:
            del case["period"]["end"]
    elif mutation == "wrong_proof":
        rule["metadata"]["proof"]["atoms"][0]["source"]["corpus_citation_path"] = (
            "ca/policy/other"
        )
    elif mutation == "bare_date":
        rule["metadata"]["proof"]["atoms"][1]["source"]["excerpt"] = "2025"
    elif mutation == "wrong_date_source":
        rule["metadata"]["proof"]["atoms"][1]["source"]["corpus_citation_path"] = (
            "ca/policy/other"
        )
    elif mutation == "superseded":
        rule["versions"].append(
            {
                "effective_from": "2025-07-01",
                "effective_to": "2025-12-31",
                "formula": "7",
            }
        )
    elif mutation == "opaque":
        rule["versions"][0]["formula"] = "if eligible: allocation - deduction else: 0"
        for case in cases:
            case["input"]["eligible"] = case["input"][NAME] > 1
    elif mutation == "dead":
        rule["versions"][0]["formula"] = "if False: allocation - deduction else: 0"
    elif mutation == "wrong_threshold":
        rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
            "> 1", "> 2"
        )
    elif mutation == "wrong_assertion":
        cases[1]["output"]["net_income"] = 99999
    elif mutation == "extra_input":
        cases[1]["input"]["deduction"] = 9999
    elif mutation == "alias_collision":
        for case in cases:
            case["input"]["ca:policies/example#input." + NAME] = case["input"][NAME]
    elif mutation == "negative_count":
        cases[0]["input"][NAME] = -1
    missing, spans = run(rule, cases, inputs)
    assert missing
    assert not spans


@pytest.mark.parametrize(
    "tail",
    [
        " Only residents may claim.",
        " unless disabled.",
        " if resident.",
        " and the claimant is eligible.",
        " However, use a different amount.",
    ],
)
def test_complete_restrictive_or_unknown_tail_is_not_consumed(tail):
    rule, cases, inputs = fixture()
    clause = CLAUSE + tail
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    assert run(rule, cases, inputs, SOURCE.replace(CLAUSE, clause), clause)[0]


@pytest.mark.parametrize("wrapper", [('"', '"'), ('"', ""), ("(", ""), ("“", "”")])
def test_quoted_or_malformed_context_rejects(wrapper):
    rule, cases, inputs = fixture()
    source = SOURCE.replace(CLAUSE, wrapper[0] + CLAUSE + wrapper[1])
    assert run(rule, cases, inputs, source)[0]


def test_wrong_path_cannot_borrow_count_evidence():
    rule, cases, inputs = fixture()
    source = SOURCE.replace(CLAUSE, CLAUSE + " Another rule applies.")
    clause = CLAUSE + " Another rule applies."
    assert run(rule, cases, inputs, source, clause)[0]


def test_year_money_occurrence_outside_owned_span_is_preserved():
    rule, cases, inputs = fixture()
    source = SOURCE + "\nThe monetary limit is 2025 dollars."
    missing, spans = run(rule, cases, inputs, source)
    assert not missing
    masked = c._mask_numeric_spans(source, spans)
    assert "limit is 2025 dollars" in masked
    assert "more than one jurisdiction" in masked


def test_ontario_larger_instruction_has_no_assumed_path_ownership():
    rule, cases, inputs = fixture()
    clause = "If you were a resident of Ontario, calculate this amount. " + CLAUSE
    source = SOURCE.replace(CLAUSE, clause)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    assert run(rule, cases, inputs, source, clause)[0]


@pytest.mark.parametrize("period", ["Month", None])
def test_annual_owner_must_declare_year(period):
    rule, cases, inputs = fixture()
    rule["period"] = period
    assert run(rule, cases, inputs)[0]


@pytest.mark.parametrize(
    "extra",
    [
        "Form T9999 instructions\n",
        "Another form controls these instructions.\n",
    ],
)
def test_foreign_preamble_cannot_lend_annual_context(extra):
    rule, cases, inputs = fixture()
    source = SOURCE.replace("Use this form", extra + "Use this form", 1)
    assert run(rule, cases, inputs, source)[0]


def test_foreign_form_header_before_count_rejects():
    rule, cases, inputs = fixture()
    source = SOURCE.replace(CLAUSE, "Form T9999 instructions\n" + CLAUSE)
    assert run(rule, cases, inputs, source)[0]


def test_dated_later_foreign_header_cannot_replace_original_purpose():
    rule, cases, inputs = fixture()
    source = SOURCE.replace("Use this form", "Use that form", 1)
    source += "\n" + HEADER
    assert run(rule, cases, inputs, source)[0]


def test_payable_event_has_distinct_typed_cardinality_identity():
    rule, cases, inputs = fixture()
    name = "jurisdictions_tax_payable_count"
    clause = CLAUSE.replace("paid tax", "have to pay tax")
    source = SOURCE.replace(CLAUSE, clause)
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(NAME, name)
    rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = clause
    inputs[name] = inputs.pop(NAME)
    inputs[name]["name"] = name
    for case in cases:
        case["input"][name] = case["input"].pop(NAME)
    assert not run(rule, cases, inputs, source, clause)[0]


def test_input_map_key_cannot_override_declared_identity():
    rule, cases, inputs = fixture()
    inputs[NAME]["name"] = "different_count"
    assert run(rule, cases, inputs)[0]


def test_wrong_reached_boundary_crossed_by_wider_pair_stays_unproved():
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        "> 1", "> 2"
    )
    cases[1]["input"][NAME] = 3
    assert run(rule, cases, inputs)[0]
    assert not run(rule, cases, inputs)[1]


def test_named_threshold_relational_descriptor_requires_reached_constant_one():
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        "> 1", "> multiple_count_threshold"
    )
    assert not run(rule, cases, inputs, environment={"multiple_count_threshold": 1})[0]
    cases[1]["input"][NAME] = 3
    assert run(rule, cases, inputs, environment={"multiple_count_threshold": 2})[0]


def test_mutable_threshold_input_is_not_independent_constant():
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = rule["versions"][0]["formula"].replace(
        "> 1", "> threshold"
    )
    for case in cases:
        case["input"]["threshold"] = 1
    assert run(rule, cases, inputs)[0]


def test_extra_relational_condition_does_not_prove_closed_count_antecedent():
    rule, cases, inputs = fixture()
    rule["versions"][0]["formula"] = (
        "if jurisdictions_tax_paid_count > threshold and allocation > deduction: allocation - deduction else: 0"
    )
    assert run(rule, cases, inputs, environment={"threshold": 1})[0]

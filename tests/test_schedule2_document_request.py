"""Source-owned ellipsis: existing document witnesses, no monetary veto."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_completeness as s

F = Path(__file__).parent / "fixtures"
BODY = (F / "detached_schedule2_source.txt").read_text()
PAYLOAD = yaml.safe_load((F / "same_page_floor_candidate.yaml").read_text())
CASES = yaml.safe_load((F / "same_page_floor_candidate.test.yaml").read_text())
CITATION = "ca/policy/cra/t1-2025/federal-spouse-transfer-amounts"
OWNER = "schedule_2_send_new_T2201"
SELECTOR = "CRA_requested_new_T2201"
TEXT = "if the CRA asks you to."
BRANCH = s.SourceStructureBranch(
    ("1",),
    "exception-clause",
    "1",
    TEXT,
    BODY.index(TEXT),
    BODY.index(TEXT) + len(TEXT),
)


def context(payload=PAYLOAD, cases=CASES):
    rules = {r["name"]: r for r in payload["rules"] if r["kind"] == "derived"}
    asserted = {
        n: [c for c in cases if n in s._test_case_output_names(c)] for n in rules
    }
    env = s._constant_rule_environment(payload)
    witnesses = s._toggled_formula_boolean_selectors(
        rules, asserted_by_rule=asserted, formula_environment=env
    )
    return rules, asserted, env, witnesses


def witness():
    return next(
        w
        for w in context()[3]
        if w.rule_name == OWNER and w.selector_name == SELECTOR and w.active_value
    )


def relevant(payload=PAYLOAD, branch=BRANCH, w=None, body=BODY, citation=CITATION):
    return s._document_request_selector_is_relevant(
        branch,
        w or witness(),
        principal_rules=context(payload)[0],
        input_declarations={v["name"]: v for v in payload["inputs"]},
        formula_environment=context(payload)[2],
        parameter_rules={
            r["name"]: r for r in payload["rules"] if r["kind"] == "parameter"
        },
        source_text=body,
        corpus_citation_path=citation,
    )


def test_existing_document_pair_needs_source_owned_relevance_only():
    rules, asserted, env, witnesses = context()
    w = witness()
    assert not s._source_exception_selector_is_relevant(TEXT, SELECTOR)
    assert relevant()
    result = s._exception_witnesses_for_branch(
        BRANCH,
        source_text=BODY,
        corpus_citation_path=CITATION,
        input_declarations={v["name"]: v for v in PAYLOAD["inputs"]},
        principal_rules=rules,
        principal_rule_paths={n: {("1",)} for n in rules},
        asserted_by_rule=asserted,
        formula_environment=env,
        parameter_rules={
            r["name"]: r for r in PAYLOAD["rules"] if r["kind"] == "parameter"
        },
        toggled_exception_selectors=witnesses,
        extract_numeric_occurrences=lambda _: (),
    )
    assert w in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("active_value", False),
        ("blocks", True),
        ("boolean_effect", False),
        ("rule_name", "schedule_2_continuing_claim_no_new_T2201"),
        ("rule_name", "federal_amounts_transferred_from_spouse_line_32600"),
        ("rule_name", "schedule_2_attach_completed_certified_T2201"),
    ],
)
def test_wrong_effect_or_output_cannot_substitute(field, value):
    assert not relevant(w=replace(witness(), **{field: value}))


@pytest.mark.parametrize(
    "mutation",
    [
        "and",
        "negated",
        "extra",
        "version",
        "dates",
        "proof",
        "unit",
        "missing_input",
        "input_default",
    ],
)
def test_candidate_binding_mutations_refuse(mutation):
    p = deepcopy(PAYLOAD)
    r = next(r for r in p["rules"] if r["name"] == OWNER)
    if mutation == "and":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            " or ", " and "
        )
    elif mutation == "negated":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            SELECTOR, "not " + SELECTOR
        )
    elif mutation == "extra":
        r["versions"][0]["formula"] += " and filing_paper_return"
    elif mutation == "version":
        r["versions"].append(deepcopy(r["versions"][0]))
    elif mutation == "dates":
        r["versions"][0]["effective_to"] = "2026-12-31"
    elif mutation == "proof":
        r["metadata"]["proof"]["atoms"] = []
    elif mutation == "unit":
        r["dtype"] = "Money"
    elif mutation == "missing_input":
        p["inputs"] = [v for v in p["inputs"] if v["name"] != SELECTOR]
    else:
        next(v for v in p["inputs"] if v["name"] == SELECTOR)["default"] = False
    assert not relevant(p, w=witness())


def test_source_scope_and_coordinates_refuse():
    assert not relevant(body=BODY.replace("asks you to.", "asks you to and approves."))
    assert not relevant(citation=CITATION + "-other")
    assert not relevant(branch=replace(BRANCH, start=BRANCH.start + 1))


@pytest.mark.parametrize(
    "mutation",
    [
        "metadata_null",
        "proof_null",
        "atoms_null",
        "source_null",
        "missing_dates",
        "judgment_unit",
        "raw_whitespace",
        "wrong_amount",
        "wrong_alternative",
        "duplicate_projection",
        "prior_wrong_date",
        "prior_wrong_operator",
    ],
)
def test_independent_review_findings_refuse(mutation):
    p = deepcopy(PAYLOAD)
    r = next(r for r in p["rules"] if r["name"] == OWNER)
    if mutation == "metadata_null":
        r["metadata"] = None
    elif mutation == "proof_null":
        r["metadata"]["proof"] = None
    elif mutation == "atoms_null":
        r["metadata"]["proof"]["atoms"] = None
    elif mutation == "source_null":
        r["metadata"]["proof"]["atoms"][0]["source"] = None
    elif mutation == "missing_dates":
        r["metadata"]["proof"]["atoms"] = r["metadata"]["proof"]["atoms"][:1]
    elif mutation == "judgment_unit":
        r["unit"] = "CAD"
    elif mutation == "raw_whitespace":
        atom = r["metadata"]["proof"]["atoms"][0]
        atom["source"]["excerpt"] = atom["source"]["excerpt"].replace("\n", " ")
    elif mutation == "wrong_amount":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "spouse_disability_amount_line_31600", "spouse_taxable_income_line_26000"
        )
    elif mutation == "wrong_alternative":
        r["versions"][0]["formula"] = r["versions"][0]["formula"].replace(
            "previous_DTC_approval_ended_before_2025",
            "source_separation_blocks_transfer",
        )
    elif mutation == "duplicate_projection":
        duplicate = deepcopy(
            next(
                r
                for r in p["rules"]
                if r["name"] == "schedule_2_disability_amount_line_4"
            )
        )
        duplicate["name"] = "duplicate"
        p["rules"].append(duplicate)
    else:
        prior = next(
            r
            for r in p["rules"]
            if r["name"] == "previous_DTC_approval_ended_before_2025"
        )
        if mutation == "prior_wrong_date":
            prior["versions"][0]["effective_to"] = "2026-12-31"
        else:
            prior["versions"][0]["formula"] = prior["versions"][0]["formula"].replace(
                " < ", " > "
            )
    assert not relevant(p, w=witness())


@pytest.mark.parametrize(
    "mutation", ["money", "missing_dates", "wrong_citation", "default"]
)
def test_cutoff_declaration_custody_refuses(mutation):
    p = deepcopy(PAYLOAD)
    r = next(
        r for r in p["rules"] if r["name"] == "schedule_2_DTC_approval_end_cutoff_year"
    )
    if mutation == "money":
        r["dtype"], r["unit"] = "Money", "CAD"
    elif mutation == "missing_dates":
        r["metadata"]["proof"]["atoms"] = r["metadata"]["proof"]["atoms"][:1]
    elif mutation == "wrong_citation":
        r["metadata"]["proof"]["atoms"][1]["source"]["corpus_citation_path"] = (
            CITATION + "-other"
        )
    else:
        r["default"] = 2025
    assert not relevant(p, w=witness())

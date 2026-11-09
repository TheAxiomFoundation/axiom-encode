"""Reviewed direct factual spelling, never aliases inferred from arithmetic."""

import copy
from dataclasses import replace

import pytest
import yaml

from axiom_encode.harness import source_path_evidence as e
from tests.test_source_path_evidence import CITATION, SOURCE, certify, inputs, rule

STANDARD = "section_428mj_tax_before_foreign_tax_credit_for_residence_jurisdiction"
ALTERNATE = (
    "form_428mj_tax_before_provincial_or_territorial_foreign_tax_credit_for_residence"
)


def alternate():
    p, c = inputs()
    return yaml.safe_load(
        yaml.safe_dump(p).replace(STANDARD, ALTERNATE)
    ), yaml.safe_load(yaml.safe_dump(c).replace(STANDARD, ALTERNATE))


def test_reviewed_residence_spelling_keeps_source_proof_and_execution():
    p, c = alternate()
    before = copy.deepcopy((p, c))
    result = certify(p, c)
    assert len(result.condition_spans) == 4
    assert all(o.certified for o in result.obligations if o.owner == e._TAX)
    assert (p, c) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "both",
        "rule",
        "alternate_rule",
        "missing_unit",
        "computed_input",
        "duplicate",
        "import",
        "module_import",
        "unit",
        "entity",
        "period",
        "dtype",
        "unrelated",
        "proof",
        "wrong_formula_coordinate",
        "wrong_path",
        "wrong_output",
    ],
)
def test_residence_alias_never_grants_unproved_or_ambiguous_evidence(mutation):
    p, c = alternate()
    before = copy.deepcopy((p, c))
    item = next(x for x in p["inputs"] if x["name"] == ALTERNATE)
    if mutation == "both":
        p["inputs"].append({**item, "name": STANDARD})
    elif mutation == "rule":
        p["rules"].append({"name": STANDARD, "kind": "derived"})
    elif mutation == "alternate_rule":
        p["rules"].append({"name": ALTERNATE, "kind": "derived"})
    elif mutation == "missing_unit":
        item.pop("unit")
    elif mutation == "computed_input":
        c[0]["input"][f"{CITATION.replace('ca/', 'ca:', 1)}#{e._TAX}"] = 42
    elif mutation == "duplicate":
        p["inputs"].append(copy.deepcopy(item))
    elif mutation == "import":
        p["imports"] = ["ca:policies/unrelated"]
    elif mutation == "module_import":
        p.setdefault("module", {})["imports"] = ["ca:policies/unrelated"]
    elif mutation in {"unit", "entity", "period", "dtype"}:
        item[mutation] = {
            "unit": "USD",
            "entity": "Household",
            "period": "Month",
            "dtype": "Text",
        }[mutation]
    elif mutation == "unrelated":
        p = yaml.safe_load(yaml.safe_dump(p).replace(ALTERNATE, "unrelated_amount"))
        c = yaml.safe_load(yaml.safe_dump(c).replace(ALTERNATE, "unrelated_amount"))
    elif mutation == "proof":
        rule(p, e._TAX)["metadata"]["proof"]["atoms"] = []
    elif mutation == "wrong_formula_coordinate":
        owner = rule(p, e._TAX)
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            ALTERNATE, "on428mj_line_58_after_entering_zero_on_lines_43_and_45"
        )
    elif mutation == "wrong_path":
        owner = rule(p, e._TAX)
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            "jurisdictions_tax_payable_count >", "jurisdictions_tax_payable_count <"
        )
    else:
        for case in c:
            for key in case["output"]:
                if key.endswith("#" + e._TAX):
                    case["output"][key] += 1
    assert (p, c) != before
    result = certify(p, c)
    assert not any(name == e._TAX for name, *_ in result.condition_spans)


def test_binding_changes_only_closed_other_multi_operand():
    p, _ = alternate()
    contract = e.paths.worksheet_shadow_contract(
        SOURCE, citation=CITATION, line2_owner=e._LINE2, tax_otherwise_owner=e._TAX
    )
    bound = e._bind_residence_428mj_input_name(p, contract)
    assert not bound.unresolved
    for before, after in zip(contract.obligations, bound.obligations):
        if before.identity != "other_multi":
            assert before == after
        else:
            assert after.expected_expression == ALTERNATE
            assert (
                replace(
                    after,
                    expected_expression=before.expected_expression,
                    input_contracts=before.input_contracts,
                )
                == before
            )
            assert [x.name for x in after.input_contracts].count(ALTERNATE) == 1
    malformed = replace(
        contract,
        obligations=tuple(
            replace(o, expected_expression="unrelated")
            if o.identity == "other_multi"
            else o
            for o in contract.obligations
        ),
    )
    assert e._bind_residence_428mj_input_name(p, malformed).unresolved


def test_authoritative_wrong_coordinate_cannot_borrow_alias_contract():
    p, c = alternate()
    changed = SOURCE.replace("T2203", "T9999")
    assert changed != SOURCE
    assert not certify(p, c, source=changed).condition_spans

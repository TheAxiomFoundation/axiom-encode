"""Full graph accounting; historical actual captures wrapped in local test evidence."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.harness.bound_form_operands import (
    bind_ordered_form,
    observed_form_frame,
)
from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    SameCaseEvidence,
    case_key,
)

FIXTURE = Path(__file__).parent / "fixtures/bound_form_obligations/observed-graph.json"
SOURCE = (FIXTURE.parent.parent / "informational_referral/source.txt").read_text()
ROOT = "ca/policies/cra/t1-2025/nl479-credits/primary-document.yaml"
CITATION = "ca/policy/cra/t1-2025/nl479-credits/primary-document"


def data():
    return json.loads(FIXTURE.read_text())


def bind(content):
    return bind_ordered_form(
        recognize_bound_form_structure(SOURCE, CITATION),
        content,
        {CITATION: SOURCE},
        root_relative=ROOT,
    )


def evidence(tmp_path, d):
    content = d["content"]
    original = tmp_path / "original" / ROOT
    original.parent.mkdir(parents=True)
    original.write_text(content)
    snapshot = tmp_path / "snapshot"
    target = snapshot / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(content)
    return SameCaseEvidence(
        (ClosureFile(ROOT, original, content.encode()),),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(d["artifact"]).encode(),
        tuple(
            CaseExecution(
                case_key(d["cases"][r["case_index"] - 1]),
                json.dumps(r["request"]),
                json.dumps(r["response"]),
                r["case_index"],
            )
            for r in d["records"]
        ),
    )


def test_complete_disjoint_graph_observes_both_entity_cohorts(tmp_path):
    d = data()
    b = bind(d["content"])
    assert b is not None and len(b.document_requirements) == 1
    assert len(b.entered) == 4 and len(b.row_owners) == 9
    assert observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


@pytest.mark.parametrize(
    "change",
    [
        "extra_rule",
        "unused_input",
        "row_depends_document",
        "document_depends_row",
        "duplicate_owner",
        "wrong_proof",
    ],
)
def test_no_unaccounted_or_cross_dependent_declarations(change):
    d = data()
    p = yaml.safe_load(d["content"])
    if change == "extra_rule":
        p["rules"].append(dict(p["rules"][-1], name="unknown_requirement"))
    elif change == "unused_input":
        p["inputs"].append(dict(p["inputs"][-1], name="unused_needs_space"))
    elif change == "row_depends_document":
        next(r for r in p["rules"] if r["name"] == "nl479_entered_assistance_line_2")[
            "versions"
        ][0]["formula"] = "needs_additional_space"
    elif change == "document_depends_row":
        p["rules"][-1]["versions"][0]["formula"] = p["inputs"][0]["name"]
    elif change == "duplicate_owner":
        p["rules"].append(copy.deepcopy(p["rules"][-1]))
    else:
        p["rules"][-1]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "If applicable, complete the table."
        )
    assert bind(yaml.safe_dump(p, sort_keys=False)) is None


@pytest.mark.parametrize(
    "change",
    [
        "fact_entity",
        "fact_kind",
        "fact_value",
        "foreign_id",
        "missing_fact",
        "wrong_year",
        "wrong_outcome",
        "wrong_ast",
        "missing_pair",
        "missing_rows",
        "compiled_entity",
        "compiled_ast",
        "compiled_extra",
        "case_wrong_entity_id",
    ],
)
def test_actual_document_and_full_graph_mutations_refuse(tmp_path, change):
    d = data()
    r = d["records"][-1]
    fact = r["request"]["dataset"]["inputs"][0]
    ref = r["request"]["queries"][0]["outputs"][0]
    result = r["response"]["results"][0]
    compiled = next(
        x
        for x in d["artifact"]["program"]["derived"]
        if x["name"] == "additional_sheet_attachment_required"
    )
    if change == "fact_entity":
        fact["entity"] = "Family"
    elif change == "fact_kind":
        fact["value"] = {"kind": "integer", "value": 1}
    elif change == "fact_value":
        fact["value"]["value"] = False
    elif change == "foreign_id":
        fact["entity_id"] = "foreign"
    elif change == "missing_fact":
        r["request"]["dataset"]["inputs"] = []
    elif change == "wrong_year":
        r["request"]["queries"][0]["period"]["start"] = "2024-01-01"
    elif change == "wrong_outcome":
        result["outputs"][ref]["outcome"] = "not_holds"
    elif change == "wrong_ast":
        result["trace"][ref]["executed_expression"] = "true"
    elif change == "missing_pair":
        d["records"].pop()
    elif change == "missing_rows":
        d["records"].pop(0)
    elif change == "compiled_entity":
        compiled["entity"] = "Family"
    elif change == "compiled_ast":
        compiled["expr"]["right"]["value"]["value"] = False
    elif change == "compiled_extra":
        d["artifact"]["program"]["derived"].append(dict(compiled, name="other"))
    else:
        d["cases"][-1]["entity_id"] = "other-filer"
    b = bind(d["content"])
    assert b is not None
    assert not observed_form_frame(b, evidence(tmp_path, d), d["content"], d["cases"])


@pytest.mark.parametrize("index,integer", [(1, 0), (2, 1)])
def test_actual_bool_payload_rejects_integer_even_when_python_equal(
    tmp_path, index, integer
):
    d = data()
    d["records"][index]["request"]["dataset"]["inputs"][0]["value"]["value"] = integer
    assert not observed_form_frame(
        bind(d["content"]), evidence(tmp_path, d), d["content"], d["cases"]
    )


@pytest.mark.parametrize("versioned", [False, True])
def test_compiled_boolean_literal_is_type_exact(tmp_path, versioned):
    d = data()
    declaration = next(
        x
        for x in d["artifact"]["program"]["derived"]
        if x["name"] == "additional_sheet_attachment_required"
    )
    expression = (
        declaration["versions"][0]["expr"] if versioned else declaration["expr"]
    )
    expression["right"]["value"]["value"] = 1
    assert not observed_form_frame(
        bind(d["content"]), evidence(tmp_path, d), d["content"], d["cases"]
    )


def _bool_lookup_indexes(node):
    count = 0
    if isinstance(node, dict):
        if node.get("kind") == "parameter_lookup":
            value = node["index"]["value"]
            assert value["value"] == 0 and type(value["value"]) is int
            value["value"] = False
            count += 1
        for child in node.values():
            count += _bool_lookup_indexes(child)
    elif isinstance(node, list):
        for child in node:
            count += _bool_lookup_indexes(child)
    return count


@pytest.mark.parametrize("site", ["expr", "versions"])
def test_arithmetic_compiled_expression_is_bool_int_type_exact(tmp_path, site):
    d = data()
    changed = sum(
        _bool_lookup_indexes(declaration[site])
        for declaration in d["artifact"]["program"]["derived"]
    )
    assert changed
    assert not observed_form_frame(
        bind(d["content"]), evidence(tmp_path, d), d["content"], d["cases"]
    )

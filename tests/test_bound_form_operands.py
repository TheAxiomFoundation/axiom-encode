"""Ordered source ownership controls; fixture facts are reviewed, not inferred."""

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode.bound_form_structure import recognize_bound_form_structure
from axiom_encode.harness.bound_form_operands import bind_ordered_form
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

FIXTURE = Path(__file__).parent / "fixtures" / "bound_form_operands"
SOURCE = (FIXTURE.parent / "informational_referral" / "source.txt").read_text()
CONTENT = (FIXTURE / "diagnostic.yaml").read_text()
PAYLOAD = yaml.safe_load(CONTENT)
CASES = yaml.safe_load((FIXTURE / "diagnostic.test.yaml").read_text())
CITATION = PAYLOAD["module"]["source_verification"]["corpus_citation_path"]


ROOT = "ca/policies/cra/t1-2025/nl479-credits/primary-document.yaml"


def binding(payload=None, source=SOURCE, root=ROOT):
    text = CONTENT if payload is None else yaml.safe_dump(payload, sort_keys=False)
    form = recognize_bound_form_structure(source, CITATION)
    return (
        bind_ordered_form(form, text, {CITATION: source}, root_relative=root)
        if form
        else None
    )


def rule(payload, row, *, parameter=False):
    prefix = f"Line {row}:" if row != 5 else "Line 5 and line 62000:"
    return next(
        r
        for r in payload["rules"]
        if (r["kind"] == "parameter") == parameter
        and r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"].startswith(prefix)
    )


def matches(b, row, case=None, leaf=None):
    current = rule(PAYLOAD, row)
    span = next(r.span for r in b.form.rows if r.number == row)
    return b.matches(
        content=CONTENT,
        branch=SimpleNamespace(start=span[0], end=span[1], text=SOURCE[slice(*span)]),
        rule=current,
        principal_rules={
            r["name"]: r for r in PAYLOAD["rules"] if r["kind"] == "derived"
        },
        case=CASES[0] if case is None else case,
        leaf=leaf or current["versions"][0]["formula"],
        trace=(),
    )


def issues(payload=None, cases=CASES, source=SOURCE, proof=True):
    cases = copy.deepcopy(cases)
    for case in cases:
        case["input"] = {
            key.rsplit("#", 1)[-1]: value for key, value in case["input"].items()
        }
    pipeline = ValidatorPipeline(
        policy_repo_path=Path("/tmp/rulespec-ca"),
        axiom_rules_path=Path("/tmp/axiom-rules-engine"),
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return pipeline._complete_source_unit_issues(
        CONTENT if payload is None else yaml.safe_dump(payload, sort_keys=False),
        validation_source_texts={CITATION: source},
        proof_source_texts={CITATION: source} if proof else {},
        test_cases=cases,
    )


@pytest.mark.parametrize("row", [3, 5, 7, 9])
def test_exact_ordered_owner_and_case(row):
    assert binding() is not None
    assert matches(binding(), row)


def test_missing_unreached_fact_is_not_required_but_reached_fact_is():
    case = copy.deepcopy(CASES[0])
    del case["input"][next(k for k in case["input"] if k.endswith("t1129_line_6"))]
    assert matches(binding(), 3, case)
    assert not matches(binding(), 9, case)
    del case["input"][next(k for k in case["input"] if k.endswith("line_2"))]
    assert not matches(binding(), 3, case)


@pytest.mark.parametrize(
    "change",
    [
        "reverse",
        "wrong_line",
        "alias",
        "arithmetic_identity",
        "duplicate_owner",
        "wrong_date",
        "second_version",
        "wrong_unit",
        "wrong_entity",
        "missing_unit",
        "bad_hash",
        "bad_excerpt",
        "span",
        "proof_hash",
        "null_module",
        "extra_import",
    ],
)
def test_declared_graph_refusals(change):
    p = copy.deepcopy(PAYLOAD)
    r = rule(p, 5)
    if change == "reverse":
        a, b = r["versions"][0]["formula"].split(" - ")
        r["versions"][0]["formula"] = b + " - " + a
    elif change == "wrong_line":
        r["versions"][0]["formula"] = rule(p, 3)["name"] + " - " + rule(p, 2)["name"]
    elif change == "alias":
        rule(p, 4)["versions"][0]["formula"] = rule(p, 2)["versions"][0]["formula"]
    elif change == "arithmetic_identity":
        rule(p, 4)["versions"][0]["formula"] += " + 0"
    elif change == "duplicate_owner":
        other = copy.deepcopy(r)
        other["name"] = "duplicate"
        p["rules"].append(other)
    elif change == "wrong_date":
        r["versions"][0]["effective_to"] = "2026-12-31"
    elif change == "second_version":
        r["versions"].append(copy.deepcopy(r["versions"][0]))
    elif change == "wrong_unit":
        rule(p, 4)["unit"] = "USD"
    elif change == "wrong_entity":
        rule(p, 4)["entity"] = "Person"
    elif change == "missing_unit":
        p.pop("units")
    elif change == "bad_hash":
        p["module"]["source_verification"]["source_sha256"] = "0" * 64
    elif change == "bad_excerpt":
        r["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "nonexistent"
    elif change == "span":
        r["metadata"]["proof"]["atoms"][0]["source"]["span"] = [0, 1]
    elif change == "proof_hash":
        r["metadata"]["proof"]["atoms"][0]["source"]["sha256"] = "0" * 64
    elif change == "null_module":
        p["module"] = None
    elif change == "extra_import":
        p["imports"] = ["ca:policies/other#value"]
    assert binding(p) is None


def test_alias_collision_and_wrong_execution_are_refused():
    case = copy.deepcopy(CASES[0])
    key = next(iter(case["input"]))
    case["input"][key.rsplit("#", 1)[1]] = case["input"][key]
    assert not matches(binding(), 3, case)
    r = rule(PAYLOAD, 5)
    a, b = r["versions"][0]["formula"].split(" - ")
    assert not matches(binding(), 5, leaf=b + " - " + a)


def test_analyzer_accepts_ordered_rows_but_preserves_conditions_and_header():
    result = issues()
    assert len(result) == 2, result
    assert "2025" in result[0]
    assert result[-1].count("[change]:") == 3
    assert not any("formula branch" in item for item in result)
    assert any("formula branch" in item for item in issues(proof=False))


def test_coherent_reversal_cannot_buy_coverage_with_changed_assertions():
    p = copy.deepcopy(PAYLOAD)
    cases = copy.deepcopy(CASES)
    r = rule(p, 5)
    a, b = r["versions"][0]["formula"].split(" - ")
    r["versions"][0]["formula"] = b + " - " + a
    for case in cases:
        out = case["output"]
        k5 = next(k for k in out if k.endswith("#" + r["name"]))
        k7 = next(k for k in out if k.endswith("#" + rule(p, 7)["name"]))
        k9 = next(k for k in out if k.endswith("#" + rule(p, 9)["name"]))
        credit = next(v for k, v in case["input"].items() if k.endswith("t1129_line_6"))
        out[k5] = -out[k5]
        out[k7] = -out[k7]
        out[k9] = out[k7] + credit
    result = issues(p, cases)
    assert any(
        "formula branch" in item and "Line 3 minus line 4" in item for item in result
    )
    assert result[-1] == issues()[-1]


def test_new_year_and_non_tax_names_do_not_depend_on_allowlist():
    source = SOURCE.replace("2025", "2027")
    p = copy.deepcopy(PAYLOAD)
    for r in p["rules"]:
        for v in r["versions"]:
            v["effective_from"] = v["effective_from"].replace("2025", "2027")
            v["effective_to"] = v["effective_to"].replace("2025", "2027")
        for atom in r["metadata"]["proof"]["atoms"]:
            if atom["kind"] == "effective_period":
                atom["source"]["excerpt"] = "2027"
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    text = yaml.safe_dump(p, sort_keys=False)
    for i, r in enumerate(p["rules"]):
        text = text.replace(r["name"], f"row_owner_{i}")
    for i, item in enumerate(p["inputs"]):
        text = text.replace(item["name"], f"fact_{i}")
    assert binding(yaml.safe_load(text), source) is not None


def test_qualified_names_require_bound_namespace_and_cannot_cross_modules():
    assert matches(binding(), 3)
    assert not matches(binding(root=None), 3)
    foreign = copy.deepcopy(CASES[0])
    foreign["input"] = {
        "xx:policies/foreign#" + k.split("#", 1)[1]: v
        for k, v in foreign["input"].items()
    }
    assert not matches(binding(), 3, foreign)
    short = copy.deepcopy(CASES[0])
    short["input"] = {k.split("#", 1)[1]: v for k, v in short["input"].items()}
    assert matches(binding(root=None), 3, short)


@pytest.mark.parametrize(
    "path",
    [
        "../ca/policies/x.yaml",
        "/ca/policies/x.yaml",
        "ca//policies/x.yaml",
        "ca/policies/../x.yaml",
    ],
)
def test_noncanonical_root_identity_refuses(path):
    assert binding(root=path) is None


def frame_evidence(tmp_path, record=None):
    import json

    from axiom_encode.harness.same_case_execution import (
        CaseExecution,
        ClosureFile,
        SameCaseEvidence,
        case_key,
    )

    data = (
        json.loads((FIXTURE / "observed-case.json").read_text())
        if record is None
        else record
    )
    original = tmp_path / "original" / ROOT
    original.parent.mkdir(parents=True)
    original.write_text(CONTENT)
    snapshot = tmp_path / "snapshot"
    target = snapshot / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(CONTENT)
    return SameCaseEvidence(
        (ClosureFile(ROOT, original, CONTENT.encode()),),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(data["artifact"]).encode(),
        (
            CaseExecution(
                case_key(CASES[0]),
                json.dumps(data["request"]),
                json.dumps(data["response"]),
                1,
            ),
        ),
    )


def test_actual_capture_fixture_binds_family_frame(tmp_path):
    from axiom_encode.harness.bound_form_operands import observed_form_frame

    assert observed_form_frame(binding(), frame_evidence(tmp_path), CONTENT, CASES)


@pytest.mark.parametrize(
    "change",
    [
        "response_value",
        "query_year",
        "query_entity",
        "fact_entity",
        "fact_value",
        "artifact_entity",
        "artifact_ast",
        "trace_ast",
        "trace_dependencies",
        "unit",
        "metadata",
        "request_relation",
        "duplicate_record",
        "wrong_case",
        "stale_source",
        "foreign_input",
    ],
)
def test_frame_observation_refusals(tmp_path, change):
    import json
    from dataclasses import replace

    from axiom_encode.harness.bound_form_operands import observed_form_frame

    data = json.loads((FIXTURE / "observed-case.json").read_text())
    case = copy.deepcopy(CASES[0])
    response = data["response"]
    query = data["request"]["queries"][0]
    trace = response["results"][0]["trace"]
    first = next(iter(trace))
    if change == "response_value":
        trace[first]["value"]["value"] = "999"
    elif change == "query_year":
        query["period"]["end"] = "2026-12-31"
    elif change == "query_entity":
        query["entity_id"] = "other"
    elif change == "fact_entity":
        data["request"]["dataset"]["inputs"][0]["entity"] = "Person"
    elif change == "fact_value":
        data["request"]["dataset"]["inputs"][0]["value"]["value"] = 999
    elif change == "artifact_entity":
        data["artifact"]["program"]["derived"][0]["entity"] = "Person"
    elif change == "artifact_ast":
        data["artifact"]["program"]["derived"][0]["expr"] = {
            "kind": "literal",
            "value": {"kind": "integer", "value": 2000},
        }
    elif change == "trace_ast":
        trace[first]["executed_expression"] = "0"
    elif change == "trace_dependencies":
        trace[first]["dependencies"] = []
    elif change == "unit":
        trace[first]["unit"] = "USD"
    elif change == "metadata":
        response["metadata"]["fallback_reason"] = "fallback"
    elif change == "request_relation":
        data["request"]["relation_binding"] = "lenient"
    elif change == "foreign_input":
        case["input"] = {
            "xx:policies/foreign#" + k.split("#", 1)[1]: v
            for k, v in case["input"].items()
        }
    evidence = frame_evidence(tmp_path, data)
    if change == "duplicate_record":
        evidence = replace(evidence, records=evidence.records * 2)
    elif change == "wrong_case":
        case["name"] = "different"
    elif change == "stale_source":
        evidence.files[0].original.write_text(CONTENT + "\n")
    assert not observed_form_frame(binding(), evidence, CONTENT, [case])


@pytest.mark.parametrize(
    "tail",
    [
        "less assistance",
        "plus a recovery amount",
        "minus other expenses",
        "after deducting assistance",
        "if eligible",
    ],
)
def test_operative_display_suffix_keeps_analyzer_obligation(tail):
    p = copy.deepcopy(PAYLOAD)
    old = "Physical activity expenses = ^"
    new = "Physical activity expenses " + tail + " = ^"
    source = SOURCE.replace(old, new)
    for r in p["rules"]:
        for atom in r["metadata"]["proof"]["atoms"]:
            atom["source"]["excerpt"] = atom["source"]["excerpt"].replace(old, new)
    p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
        source.encode()
    ).hexdigest()
    assert binding(p, source) is None
    result = issues(p, source=source)
    assert any("formula branch" in item and tail in item for item in result)


@pytest.mark.parametrize(
    "atom", [None, {}, {"path": [], "kind": "formula", "source": {}}]
)
def test_extra_malformed_atom_is_not_ignored(atom):
    p = copy.deepcopy(PAYLOAD)
    rule(p, 5)["metadata"]["proof"]["atoms"].append(atom)
    assert binding(p) is None

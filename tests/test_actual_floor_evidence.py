"""Actual case pairs cannot stand in for synthetic floor interventions."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
import yaml

from axiom_encode.harness import actual_floor_evidence as floor
from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.imported_export_evidence import ResolvedImportedExport
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_occurrences_from_text,
    numeric_value_is_grounded,
)
from tests.test_same_case_execution import ROOT, URI, world  # noqa: F401


@pytest.fixture
def pair(world):  # noqa: F811
    text = 'Line 9: Line 5 minus line 8 (if negative, enter "0") Result = ^'
    citation = "ca/policy/test/consumer"
    provider_citation = "ca/policy/test/provider"
    provider_excerpt = (
        'Line 4 and line 10004: Line 1 minus line 2 (if negative, enter "0")'
    )
    provider_body = (
        "Step 1 - Income calculation\n"
        "Line 1: Reported income ^\nLine 2: Adjustment ^\n"
        "Line 3 and line 10003: Add lines 1 to 2. = ^\n"
        + provider_excerpt
        + " Net income = ^"
    )
    deduction_excerpt = "Line 8 and line 10008: Deduction ^"
    body = (
        "Step 2 - Final calculation\n"
        "Line 5: Enter the amount from line 4 of the previous page. ^\n"
        "Line 6: Other amount ^\nLine 7 and line 10007: Add lines 5 to 6. = ^\n"
        + deduction_excerpt
        + "\n"
        + text
    )
    context = {
        citation: body,
        provider_citation: provider_body,
        "ca/policy/test/header": "FORM 2031",
    }

    def proof(rule, source, excerpt):
        atoms = [
            {
                "path": "versions[0].formula",
                "kind": "formula",
                "source": {"corpus_citation_path": source, "excerpt": excerpt},
            }
        ]
        atoms += [
            {
                "path": "versions[0]." + field,
                "kind": "effective_period",
                "source": {
                    "corpus_citation_path": "ca/policy/test/header",
                    "excerpt": "FORM 2031",
                },
            }
            for field in ("effective_from", "effective_to")
        ]
        rule["metadata"] = {"proof": {"atoms": atoms}}

    world.rule["versions"][0]["formula"] = "max(0, reported - deduction)"
    world.rule["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = text
    deduction = {
        "name": "deduction",
        "kind": "derived",
        "entity": "Person",
        "period": "Year",
        "dtype": "Money",
        "unit": "CAD",
        "versions": [
            {
                "effective_from": "2031-01-01",
                "effective_to": "2031-12-31",
                "formula": "local_amount",
            }
        ],
    }
    proof(world.rule, citation, text)
    proof(deduction, citation, deduction_excerpt)
    rules = {"out": world.rule, "deduction": deduction}
    payload = yaml.safe_load(world.content)
    payload["rules"] = list(rules.values())
    payload["module"] = {"source_verification": {"corpus_citation_path": citation}}
    content = yaml.safe_dump(payload)
    root = world.evidence.files[0]
    root.original.write_text(content)
    (world.evidence.snapshot_root / root.relative).write_text(content)
    provider_relative = "ca/policies/test/provider.yaml"
    provider_payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": provider_citation}},
        "inputs": [
            {
                "name": "income",
                "dtype": "Money",
                "unit": "CAD",
                "entity": "Person",
                "period": "Year",
            }
        ],
        "rules": [
            {
                "name": "reported",
                "kind": "derived",
                "dtype": "Money",
                "unit": "CAD",
                "entity": "Person",
                "period": "Year",
                "versions": [
                    {
                        "formula": "income",
                        "effective_from": "2031-01-01",
                        "effective_to": "2031-12-31",
                    }
                ],
            }
        ],
    }
    proof(provider_payload["rules"][0], provider_citation, provider_excerpt)
    provider_raw = yaml.safe_dump(provider_payload).encode()
    provider = root.original.parent / "provider.yaml"
    provider.write_bytes(provider_raw)
    (world.evidence.snapshot_root / provider_relative).write_bytes(provider_raw)
    artifact = copy.deepcopy(world.artifact)
    declaration = copy.deepcopy(artifact["program"]["derived"][-1])
    declaration.update(id=ROOT + "#deduction", name="deduction")
    artifact["program"]["derived"].append(declaration)
    cases = []
    records = []
    for index, income in enumerate([4, 6], 1):
        case = copy.deepcopy(world.case)
        case.update(name=f"floor-{index}", period=world.request["queries"][0]["period"])
        case["input"][ROOT + "#input.local_amount"] = 5
        case["input"][URI.split("#")[0] + "#input.income"] = income
        case["output"] = {ROOT + "#out": max(0, income - 5), ROOT + "#deduction": 5}
        cases.append(case)
        request = copy.deepcopy(world.request)
        request["queries"][0]["entity_id"] = f"case-{index}"
        request["queries"][0]["outputs"] = list(case["output"])
        response = copy.deepcopy(world.response)
        row = response["results"][0]
        row["entity_id"] = f"case-{index}"
        node = copy.deepcopy(row["trace"][ROOT + "#out"])
        node.update(id=ROOT + "#deduction")
        row["trace"][ROOT + "#deduction"] = node
        for ref, value in [
            (URI, income),
            (ROOT + "#out", max(0, income - 5)),
            (ROOT + "#deduction", 5),
        ]:
            row["trace"][ref]["entity_id"] = f"case-{index}"
            row["trace"][ref]["value"]["value"] = str(value)
        row["outputs"] = {
            ref: copy.deepcopy(row["trace"][ref]) for ref in case["output"]
        }
        records.append(
            e.CaseExecution(
                e.case_key(case), json.dumps(request), json.dumps(response), index
            )
        )
    evidence = replace(
        world.evidence,
        files=(
            replace(root, raw=content.encode()),
            e.ClosureFile(provider_relative, provider, provider_raw),
        ),
        artifact_bytes=json.dumps(artifact).encode(),
        records=tuple(records),
    )
    return dict(
        evidence=evidence,
        branch=s.SourceStructureBranch(
            (), "formula-clause", "floor", text, body.index(text), len(body)
        ),
        content=content,
        source_text=body,
        corpus_citation_path=citation,
        source_context=context,
        imported_exports=(
            ResolvedImportedExport(
                URI,
                URI,
                "reported",
                str(provider),
                str(provider.parents[2]),
                hashlib.sha256(provider_raw).hexdigest(),
                provider_raw.decode(),
                (URI,),
            ),
        ),
        principal_rules=rules,
        asserted_by_rule={"out": cases, "deduction": cases},
        formula_environment={},
        extract_numeric_occurrences=extract_typed_numeric_occurrences_from_text,
        numeric_value_is_grounded=numeric_value_is_grounded,
    )


def test_actual_negative_positive_pair(pair):
    assert len(floor.witnesses(**pair)) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "no_index",
        "wrong_index",
        "wrong_entity",
        "wrong_year",
        "wrong_unit",
        "stale_case",
        "extra_input_change",
        "positive_only",
        "equality_only",
        "wrong_operator",
        "conflicting_version",
        "wrong_source",
        "changed_key_set",
    ],
)
def test_floor_refusals(pair, mutation):
    evidence = pair["evidence"]
    records = list(evidence.records)
    cases = pair["asserted_by_rule"]["out"]
    rule = pair["principal_rules"]["out"]
    if mutation == "no_index":
        records[0] = replace(records[0], case_index=None)
    elif mutation == "wrong_index":
        records[0] = replace(records[0], case_index=99)
    elif mutation == "wrong_entity":
        request = json.loads(records[0].request_json)
        request["queries"][0]["entity_id"] = "foreign"
        records[0] = replace(records[0], request_json=json.dumps(request))
    elif mutation == "wrong_year":
        request = json.loads(records[0].request_json)
        response = json.loads(records[0].response_json)
        request["queries"][0]["period"]["start"] = "2030-01-01"
        response["results"][0]["period"] = request["queries"][0]["period"]
        records[0] = replace(
            records[0],
            request_json=json.dumps(request),
            response_json=json.dumps(response),
        )
    elif mutation == "wrong_unit":
        artifact = json.loads(evidence.artifact_bytes)
        artifact["program"]["derived"][0]["unit"] = "USD"
        pair["evidence"] = replace(
            evidence, artifact_bytes=json.dumps(artifact).encode()
        )
    elif mutation == "stale_case":
        cases[0]["name"] = "stale"
    elif mutation == "extra_input_change":
        cases[0]["input"][ROOT + "#input.local_amount"] = 4
        records[0] = replace(records[0], case_sha256=e.case_key(cases[0]))
    elif mutation in ("positive_only", "equality_only"):
        cases[0]["input"][URI.split("#")[0] + "#input.income"] = (
            7 if mutation == "positive_only" else 5
        )
        cases[0]["output"][ROOT + "#out"] = 2 if mutation == "positive_only" else 0
        records[0] = replace(records[0], case_sha256=e.case_key(cases[0]))
        response = json.loads(records[0].response_json)
        for ref, value in [
            (URI, cases[0]["input"][URI.split("#")[0] + "#input.income"]),
            (ROOT + "#out", cases[0]["output"][ROOT + "#out"]),
        ]:
            response["results"][0]["trace"][ref]["value"]["value"] = str(value)
        records[0] = replace(records[0], response_json=json.dumps(response))
    elif mutation == "wrong_operator":
        rule["versions"][0]["formula"] = "max(0, reported + deduction)"
    elif mutation == "conflicting_version":
        rule["versions"].append(
            {"effective_from": "2032-01-01", "formula": "max(0, deduction - reported)"}
        )
    elif mutation == "wrong_source":
        pair["corpus_citation_path"] = "ca/policy/other"
    elif mutation == "changed_key_set":
        cases[0]["input"]["extra"] = 1
        records[0] = replace(records[0], case_sha256=e.case_key(cases[0]))
    if mutation != "wrong_unit":
        pair["evidence"] = replace(evidence, records=tuple(records))
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize("missing", [URI, ROOT + "#deduction"])
def test_negative_case_requires_replayed_import_and_observed_dependency(pair, missing):
    evidence = pair["evidence"]
    records = list(evidence.records)
    response = json.loads(records[0].response_json)
    del response["results"][0]["trace"][missing]
    records[0] = replace(records[0], response_json=json.dumps(response))
    pair["evidence"] = replace(evidence, records=tuple(records))
    # Plausible target assertion and trace remain, but the missing reached
    # dependency cannot be reconstructed from the companion's expected value.
    assert response["results"][0]["trace"][ROOT + "#out"]["value"]["value"] == "0"
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize("wrong_row", [False, True])
def test_floor_reader_refuses_listed_or_wrong_row_assertion(pair, wrong_row):
    evidence = pair["evidence"]
    cases = pair["asserted_by_rule"]["out"]
    cases[0]["output"][ROOT + "#out"] = [0]
    if wrong_row:
        cases[0]["tables"] = {"Person": [{"local_amount": 9}]}
    records = list(evidence.records)
    records[0] = replace(records[0], case_sha256=e.case_key(cases[0]))
    evidence = replace(evidence, records=tuple(records))
    assert evidence.values(cases[0], pair["content"]) is not None
    pair["evidence"] = evidence
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize(
    "shape",
    ["query-null", "row-null", "program-null", "derived-null", "derived-item-null"],
)
def test_malformed_evidence_declines(pair, shape):
    evidence = pair["evidence"]
    records = list(evidence.records)
    if shape == "query-null":
        request = json.loads(records[0].request_json)
        request["queries"] = [None]
        records[0] = replace(records[0], request_json=json.dumps(request))
        evidence = replace(evidence, records=tuple(records))
    elif shape == "row-null":
        response = json.loads(records[0].response_json)
        response["results"] = [None]
        records[0] = replace(records[0], response_json=json.dumps(response))
        evidence = replace(evidence, records=tuple(records))
    else:
        artifact = json.loads(evidence.artifact_bytes)
        if shape == "program-null":
            artifact["program"] = None
        else:
            artifact["program"]["derived"] = None if shape == "derived-null" else [None]
        evidence = replace(evidence, artifact_bytes=json.dumps(artifact).encode())
    pair["evidence"] = evidence
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize(
    "change",
    [
        "duplicate-import-alias",
        "local-import-shadow",
        "null-metadata",
        "null-proof-source",
        "null-version",
        "wrong-operation",
        "reverse-operation",
    ],
)
def test_current_source_shapes_and_conflicts_refuse(pair, change):
    payload = yaml.safe_load(pair["content"])
    rule = payload["rules"][0]
    if change == "duplicate-import-alias":
        payload["imports"].append("ca:policies/foreign#reported")
    elif change == "local-import-shadow":
        payload["imports"].append("ca:policies/foreign#deduction")
    elif change == "null-metadata":
        rule["metadata"] = None
    elif change == "null-proof-source":
        rule["metadata"]["proof"]["atoms"][0]["source"] = None
    elif change == "null-version":
        rule["versions"].append(None)
    else:
        rule["versions"][0]["formula"] = (
            "max(0, reported + deduction)"
            if change == "wrong-operation"
            else "max(0, deduction - reported)"
        )
    content = yaml.safe_dump(payload)
    evidence = pair["evidence"]
    files = list(evidence.files)
    root = files[0]
    root.original.write_text(content)
    (evidence.snapshot_root / root.relative).write_text(content)
    files[0] = replace(root, raw=content.encode())
    pair.update(
        content=content,
        evidence=replace(evidence, files=tuple(files)),
        principal_rules={r["name"]: r for r in payload["rules"]},
    )
    assert pair["evidence"].bound(content, ())
    assert not floor.witnesses(**pair)


def _rebind(pair, payload):
    content = yaml.safe_dump(payload)
    evidence = pair["evidence"]
    root = evidence.files[0]
    root.original.write_text(content)
    (evidence.snapshot_root / root.relative).write_text(content)
    pair.update(
        content=content,
        principal_rules={r["name"]: r for r in payload["rules"]},
        evidence=replace(
            evidence, files=(replace(root, raw=content.encode()), *evidence.files[1:])
        ),
    )
    assert pair["evidence"].bound(content, pair["imported_exports"])


def test_coherent_reversed_operand_results_still_refuse(pair):
    assert floor.witnesses(**pair)
    payload = yaml.safe_load(pair["content"])
    payload["rules"][0]["versions"][0]["formula"] = "max(0, deduction - reported)"
    _rebind(pair, payload)
    cases = pair["asserted_by_rule"]["out"]
    records = []
    for case, record, result in zip(cases, pair["evidence"].records, (1, 0)):
        case["output"][ROOT + "#out"] = result
        response = json.loads(record.response_json)
        row = response["results"][0]
        for container in (row["outputs"], row["trace"]):
            container[ROOT + "#out"]["value"]["value"] = str(result)
        records.append(
            replace(
                record, case_sha256=e.case_key(case), response_json=json.dumps(response)
            )
        )
    pair["evidence"] = replace(pair["evidence"], records=tuple(records))
    assert all(
        pair["evidence"].values(case, pair["content"]) is not None for case in cases
    )
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize(
    "change",
    [
        "missing-context",
        "missing-record",
        "duplicate-grid",
        "malformed-duplicate-grid",
        "wrong-preceding-step",
        "gapped-grid",
        "wrong-frame",
        "short-local-proof",
        "short-provider-proof",
        "foreign-provider",
        "conflicting-other-version",
    ],
)
def test_ordered_source_context_refusals(pair, change):
    assert floor.witnesses(**pair)
    context = pair["source_context"]
    provider_citation = "ca/policy/test/provider"
    if change == "missing-context":
        pair["source_context"] = None
    elif change == "missing-record":
        pair["imported_exports"] = ()
    elif change in ("duplicate-grid", "malformed-duplicate-grid"):
        body = context[provider_citation]
        if change == "malformed-duplicate-grid":
            body += "\nLine 4: Conflicting coordinate ^"
        context["ca/policy/test/conflicting-provider"] = body
    elif change == "wrong-preceding-step":
        context[provider_citation] = context[provider_citation].replace(
            "Step 1", "Step 3"
        )
    elif change == "gapped-grid":
        context[provider_citation] = context[provider_citation].replace(
            "Line 2: Adjustment ^\n", ""
        )
    elif change in ("wrong-frame", "short-local-proof", "conflicting-other-version"):
        payload = yaml.safe_load(pair["content"])
        if change == "wrong-frame":
            payload["rules"][1]["versions"][0]["effective_to"] = "2032-12-31"
        elif change == "short-local-proof":
            payload["rules"][1]["metadata"]["proof"]["atoms"][0]["source"][
                "excerpt"
            ] = "Line 8 and line 10008: D"
        else:
            version = copy.deepcopy(payload["rules"][0]["versions"][0])
            version["formula"] = "max(0, deduction - reported)"
            payload["rules"][0]["versions"].append(version)
        _rebind(pair, payload)
    else:
        record = pair["imported_exports"][0]
        if change == "foreign-provider":
            pair["imported_exports"] = (
                replace(record, canonical_target="ca:policies/foreign#reported"),
            )
        else:
            payload = yaml.safe_load(record.content)
            payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
                "excerpt"
            ] = "Line 4 and line 10004: L"
            content = yaml.safe_dump(payload)
            old = pair["evidence"].files[1]
            old.original.write_text(content)
            (pair["evidence"].snapshot_root / old.relative).write_text(content)
            pair["evidence"] = replace(
                pair["evidence"],
                files=(pair["evidence"].files[0], replace(old, raw=content.encode())),
            )
            pair["imported_exports"] = (
                replace(
                    record,
                    content=content,
                    content_sha256=hashlib.sha256(content.encode()).hexdigest(),
                ),
            )
            assert pair["evidence"].bound(pair["content"], pair["imported_exports"])
    assert not floor.witnesses(**pair)


@pytest.mark.parametrize("padding", [" \n", "\n\n", "\t "])
def test_authenticated_outer_whitespace_preserves_raw_offsets(pair, padding):
    context = pair["source_context"]
    for key in list(context):
        if key.endswith("/header"):
            continue
        context[key] = padding + context[key] + "\n\n"
    assert floor.witnesses(**pair)


def test_internal_source_change_is_not_outer_whitespace(pair):
    key = pair["corpus_citation_path"]
    pair["source_context"][key] = pair["source_context"][key].replace(
        "Final calculation", "Other calculation"
    )
    assert not floor.witnesses(**pair)

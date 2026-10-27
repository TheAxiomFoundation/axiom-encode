"""Signed numeric display evidence cannot authorize clamps or other conditions."""

import copy
import hashlib
import json
from dataclasses import replace

import pytest
import yaml

from axiom_encode.harness import printed_signed_format_evidence as fmt
from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import source_completeness as s
from tests.test_actual_floor_evidence import (  # noqa: F401
    ROOT,
    URI,
    _rebind,
    pair,
    world,
)


def provider_rebind(data, payload):
    record = data["imported_exports"][0]
    content = yaml.safe_dump(payload)
    old = data["evidence"].files[1]
    old.original.write_text(content)
    (data["evidence"].snapshot_root / old.relative).write_text(content)
    data["evidence"] = replace(
        data["evidence"],
        files=(data["evidence"].files[0], replace(old, raw=content.encode())),
    )
    data["imported_exports"] = (
        replace(
            record,
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        ),
    )


@pytest.fixture
def signed(pair):  # noqa: F811
    old = pair["branch"].text
    text = "Line 9 and line 10009: Line 5 minus line 8 (if negative, show in brackets) Result = ^"
    pair["source_text"] = pair["source_text"].replace(old, text)
    pair["source_context"][pair["corpus_citation_path"]] = pair["source_text"]
    pair["branch"] = replace(pair["branch"], text=text, end=len(pair["source_text"]))
    payload = yaml.safe_load(pair["content"])
    payload["units"] = [{"name": "CAD", "kind": "currency", "minor_units": 2}]
    owner = payload["rules"][0]
    owner["versions"][0]["formula"] = "reported - deduction"
    owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = text
    _rebind(pair, payload)
    record = pair["imported_exports"][0]
    provider = yaml.safe_load(record.content)
    proof = provider["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]
    old = proof["excerpt"] + " Net income = ^"
    new = "Line 4 and line 10004: Line 1 plus line 2 Total income = ^"
    pair["source_context"][proof["corpus_citation_path"]] = pair["source_context"][
        proof["corpus_citation_path"]
    ].replace(old, new)
    proof["excerpt"] = new
    provider_rebind(pair, provider)
    records = []
    for case, record, result in zip(
        pair["asserted_by_rule"]["out"], pair["evidence"].records, (-1, 1)
    ):
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
    return pair


def test_signed_negative_and_nonnegative_are_format_observations(signed):
    assert len(fmt.witnesses(**signed)) == 1
    # No counterfactual pair requirement: each signed execution stands alone.
    signed["asserted_by_rule"]["out"][1]["input"][ROOT + "#input.unrelated"] = 9
    case = signed["asserted_by_rule"]["out"][1]
    records = list(signed["evidence"].records)
    records[1] = replace(records[1], case_sha256=e.case_key(case))
    signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    assert fmt.witnesses(**signed)


@pytest.mark.parametrize(
    "change",
    [
        "clamp",
        "reverse",
        "abs",
        "wrong-proof",
        "short-proof",
        "wrong-unit",
        "wrong-frame",
        "other-version",
        "no-units",
        "foreign-family",
        "wrong-import",
        "collision",
        "missing-context",
        "missing-provider",
        "duplicate-grid",
        "malformed-grid",
        "positive-only",
        "negative-only",
        "stale",
        "wrong-year",
        "wrong-person",
        "trace-unit",
        "no-index",
        "structural-boundary",
        "advisory",
        "zero-instruction",
    ],
)
def test_signed_format_refusals(signed, change):
    assert fmt.witnesses(**signed)
    payload = yaml.safe_load(signed["content"])
    owner = payload["rules"][0]
    if change in ("clamp", "reverse", "abs"):
        owner["versions"][0]["formula"] = {
            "clamp": "max(0,reported-deduction)",
            "reverse": "deduction-reported",
            "abs": "abs(reported-deduction)",
        }[change]
    elif change == "wrong-proof":
        owner["metadata"]["proof"]["atoms"][0]["source"]["corpus_citation_path"] = (
            "ca/policy/foreign/consumer"
        )
    elif change == "short-proof":
        owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = "Line 9"
    elif change == "wrong-unit":
        payload["rules"][1]["unit"] = "USD"
    elif change == "wrong-frame":
        payload["rules"][1]["versions"][0]["effective_to"] = "2032-12-31"
    elif change == "other-version":
        owner["versions"].append(
            {**owner["versions"][0], "formula": "deduction-reported"}
        )
    elif change == "no-units":
        payload.pop("units")
    elif change == "foreign-family":
        signed["source_context"]["ca/policy/foreign/header"] = "FORM 2031"
        for r in payload["rules"]:
            for atom in r["metadata"]["proof"]["atoms"][1:]:
                atom["source"]["corpus_citation_path"] = "ca/policy/foreign/header"
    elif change == "wrong-import":
        payload["imports"] = ["ca:policies/test/../provider#reported"]
    elif change == "collision":
        payload["inputs"].append({"name": "reported", "dtype": "Money"})
    elif change == "missing-context":
        signed["source_context"] = None
    elif change == "missing-provider":
        signed["imported_exports"] = ()
    elif change in ("duplicate-grid", "malformed-grid"):
        signed["source_context"]["ca/policy/test/sibling"] = signed["source_context"][
            "ca/policy/test/provider"
        ] + ("\nLine 4: duplicate ^" if change == "malformed-grid" else "")
    elif change in ("positive-only", "negative-only"):
        signed["asserted_by_rule"]["out"] = (
            signed["asserted_by_rule"]["out"][1:]
            if change == "positive-only"
            else signed["asserted_by_rule"]["out"][:1]
        )
    elif change == "stale":
        signed["asserted_by_rule"]["out"][0]["input"][ROOT + "#input.local_amount"] = 6
    elif change in ("wrong-year", "wrong-person", "trace-unit", "no-index"):
        records = list(signed["evidence"].records)
        record = records[0]
        response = json.loads(record.response_json)
        if change == "no-index":
            record = replace(record, case_index=None)
        elif change == "wrong-year":
            signed["asserted_by_rule"]["out"][0]["period"] = "2032"
        elif change == "wrong-person":
            response["results"][0]["entity_id"] = "other"
        else:
            response["results"][0]["trace"][ROOT + "#out"]["unit"] = "USD"
        records[0] = replace(record, response_json=json.dumps(response))
        signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    elif change == "structural-boundary":
        signed["source_branches"] = (
            s.SourceStructureBranch(
                (),
                "x",
                "other",
                "",
                signed["branch"].start + 10,
                signed["branch"].start + 20,
            ),
        )
    elif change in ("advisory", "zero-instruction"):
        old = signed["branch"].text
        new = (
            old + " If other, enter zero."
            if change == "advisory"
            else old.replace("show in brackets", 'enter "0"')
        )
        signed["source_text"] = signed["source_text"].replace(old, new)
        signed["source_context"][signed["corpus_citation_path"]] = signed["source_text"]
        signed["branch"] = replace(
            signed["branch"], text=new, end=len(signed["source_text"])
        )
        owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = new
    _rebind(signed, payload)
    assert not fmt.witnesses(**signed)


@pytest.mark.parametrize("listed", [False, True])
def test_current_case_singleton_wrong_row_is_not_a_signed_observation(signed, listed):
    case = signed["asserted_by_rule"]["out"][0]
    case["output"][ROOT + "#out"] = [-1] if listed else 99
    records = list(signed["evidence"].records)
    records[0] = replace(records[0], case_sha256=e.case_key(case))
    signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    assert signed["evidence"].values(case, signed["content"]) is not None
    assert not fmt.witnesses(**signed)


@pytest.mark.parametrize("missing", [URI, ROOT + "#deduction"])
def test_missing_reached_dependency_cannot_use_expected_values(signed, missing):
    records = list(signed["evidence"].records)
    response = json.loads(records[0].response_json)
    del response["results"][0]["trace"][missing]
    records[0] = replace(records[0], response_json=json.dumps(response))
    signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    assert response["results"][0]["trace"][ROOT + "#out"]["value"]["value"] == "-1"
    assert not fmt.witnesses(**signed)


def test_reversed_arithmetic_with_coherent_actual_outputs_refuses(signed):
    payload = yaml.safe_load(signed["content"])
    payload["rules"][0]["versions"][0]["formula"] = "deduction - reported"
    _rebind(signed, payload)
    records = []
    for case, record, value in zip(
        signed["asserted_by_rule"]["out"], signed["evidence"].records, (1, -1)
    ):
        case["output"][ROOT + "#out"] = value
        response = json.loads(record.response_json)
        for container in (
            response["results"][0]["outputs"],
            response["results"][0]["trace"],
        ):
            container[ROOT + "#out"]["value"]["value"] = str(value)
        records.append(
            replace(
                record, case_sha256=e.case_key(case), response_json=json.dumps(response)
            )
        )
    signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    assert all(
        signed["evidence"].values(c, signed["content"]) is not None
        for c in signed["asserted_by_rule"]["out"]
    )
    assert not fmt.witnesses(**signed)


@pytest.mark.parametrize(
    "shape",
    [
        "null-query",
        "null-row",
        "null-metadata",
        "null-source",
        "null-version",
        "null-imports",
        "null-derived",
        "duplicate-input",
        "duplicate-unit",
    ],
)
def test_malformed_optional_contract_shapes_refuse(signed, shape):
    if shape in ("null-query", "null-row"):
        records = list(signed["evidence"].records)
        record = records[0]
        if shape == "null-query":
            request = json.loads(record.request_json)
            request["queries"] = [None]
            record = replace(record, request_json=json.dumps(request))
        else:
            response = json.loads(record.response_json)
            response["results"] = [None]
            record = replace(record, response_json=json.dumps(response))
        records[0] = record
        signed["evidence"] = replace(signed["evidence"], records=tuple(records))
    elif shape == "null-derived":
        artifact = json.loads(signed["evidence"].artifact_bytes)
        artifact["program"]["derived"] = None
        signed["evidence"] = replace(
            signed["evidence"], artifact_bytes=json.dumps(artifact).encode()
        )
    else:
        payload = yaml.safe_load(signed["content"])
        rule = payload["rules"][0]
        if shape == "null-metadata":
            rule["metadata"] = None
        elif shape == "null-source":
            rule["metadata"]["proof"]["atoms"][0]["source"] = None
        elif shape == "null-version":
            rule["versions"] = [None]
        elif shape == "null-imports":
            payload["imports"] = None
        elif shape == "duplicate-input":
            payload["inputs"].append(copy.deepcopy(payload["inputs"][0]))
        else:
            payload["units"].append({"name": "CAD", "kind": "other"})
        _rebind(signed, payload)
    assert not fmt.witnesses(**signed)


@pytest.mark.parametrize(
    "caption",
    [
        "Result unless spouse retired",
        "Result if eligible",
        "Result for eligible persons",
        "Result when required",
        "Result subject to approval",
    ],
)
def test_caption_cannot_hide_additional_conditions(signed, caption):
    old = signed["branch"].text
    new = old.replace("Result =", caption + " =")
    signed["source_text"] = signed["source_text"].replace(old, new)
    signed["source_context"][signed["corpus_citation_path"]] = signed["source_text"]
    signed["branch"] = replace(
        signed["branch"], text=new, end=len(signed["source_text"])
    )
    payload = yaml.safe_load(signed["content"])
    payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = new
    _rebind(signed, payload)
    assert fmt._FORMAT.fullmatch(new)
    assert not fmt.witnesses(**signed)


@pytest.mark.parametrize(
    "owner,field", [(0, "span"), (0, "sha256"), (1, "span"), (1, "sha256")]
)
def test_unverified_optional_source_metadata_declines(signed, owner, field):
    payload = yaml.safe_load(signed["content"])
    payload["rules"][owner]["metadata"]["proof"]["atoms"][0]["source"][field] = "forged"
    _rebind(signed, payload)
    assert not fmt.witnesses(**signed)


def test_duplicate_owner_formula_proof_declines(signed):
    payload = yaml.safe_load(signed["content"])
    atoms = payload["rules"][0]["metadata"]["proof"]["atoms"]
    atoms.append(copy.deepcopy(atoms[0]))
    _rebind(signed, payload)
    assert not fmt.witnesses(**signed)


def test_valid_all_version_source_frames_do_not_blanket_refuse(signed):
    signed["source_context"]["ca/policy/test/header"] = "FORM 2031\nFORM 2032"

    def extend(payload):
        for rule in payload["rules"]:
            v = copy.deepcopy(rule["versions"][0])
            v.update(effective_from="2032-01-01", effective_to="2032-12-31")
            rule["versions"].append(v)
            atoms = copy.deepcopy(rule["metadata"]["proof"]["atoms"])
            for atom in atoms:
                atom["path"] = atom["path"].replace("[0]", "[1]")
                if atom["kind"] == "effective_period":
                    atom["source"]["excerpt"] = "FORM 2032"
            rule["metadata"]["proof"]["atoms"] += atoms

    payload = yaml.safe_load(signed["content"])
    extend(payload)
    _rebind(signed, payload)
    provider = yaml.safe_load(signed["imported_exports"][0].content)
    extend(provider)
    provider_rebind(signed, provider)
    assert fmt.witnesses(**signed)
    provider["rules"][0]["versions"][1]["effective_to"] = "2033-12-31"
    provider_rebind(signed, provider)
    assert not fmt.witnesses(**signed)


def test_exact_format_obligation_does_not_discharge_sibling_advisory(signed):
    branch = signed["branch"]
    other = replace(
        branch,
        text=branch.text + " If negative, use another worksheet.",
        end=branch.end + 44,
    )
    witnesses = fmt.witnesses(**signed)
    assert witnesses
    assert s._unmatched_evidence_obligations({branch: witnesses, other: set()}) == (
        other,
    )

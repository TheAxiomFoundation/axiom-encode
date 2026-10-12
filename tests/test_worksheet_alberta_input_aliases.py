"""Real source/candidate/case controls for the finite Alberta field contract."""

from __future__ import annotations

import copy
import functools
import hashlib
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import source_path_evidence as evidence
from axiom_encode.harness import validator_pipeline as v

FIXTURE = Path(__file__).parent / "fixtures/source_path_shadow"
SOURCE = (FIXTURE / "source.txt").read_text()
CITATION = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"


def fixture(alternate=True):
    candidate = (FIXTURE / "candidate.yaml").read_text()
    cases = (FIXTURE / "cases.yaml").read_text()
    if alternate:
        for standard, other in evidence._ALBERTA_INPUT_ALIASES.items():
            candidate = candidate.replace(standard, other)
            cases = cases.replace(standard, other)
    return yaml.safe_load(candidate), yaml.safe_load(cases)


def certify(payload, cases, source=SOURCE):
    return evidence.certify_worksheet_paths(
        payload,
        source_text=source,
        corpus_citation_path=CITATION,
        test_cases=cases,
        numeric_value_is_grounded=v.numeric_value_is_grounded,
        extract_numeric_occurrences=functools.partial(
            v.extract_typed_numeric_inventory_occurrences_from_text, profile="legacy"
        ),
    )


def test_both_field_contracts_certify_same_tax_frames_without_mutating_facts():
    p, c = fixture()
    before = copy.deepcopy((p, c))
    result = certify(p, c)
    original = certify(*fixture(False))
    assert result.condition_spans == original.condition_spans
    assert len(result.condition_spans) == 4
    assert all(o.certified for o in result.obligations if o.owner == evidence._TAX)
    assert (p, c) == before
    assert not result.owns_exception(1493, 1807)


@pytest.mark.parametrize(
    "mutation",
    [
        "both",
        "duplicate",
        "rule_collision",
        "import",
        "unknown",
        "unit",
        "dtype",
        "entity",
        "period",
        "wrong_line",
        "missing_proof",
        "wrong_proof",
        "wrong_case",
        "wrong_source_coordinate",
    ],
)
def test_aliases_cannot_replace_source_identity_type_proof_or_execution(mutation):
    p, c = fixture()
    source = SOURCE
    item = next(x for x in p["inputs"] if x["name"] == "alberta_tax_line_63")
    tax = next(x for x in p["rules"] if x["name"] == evidence._TAX)
    if mutation == "both":
        p["inputs"].append({**item, "name": "ab428_line_63"})
    elif mutation == "duplicate":
        p["inputs"].append(copy.deepcopy(item))
    elif mutation == "rule_collision":
        p["rules"].append({**tax, "name": "ab428_line_63"})
    elif mutation == "import":
        p["module"]["imports"] = [{"from": "other", "names": ["ab428_line_63"]}]
    elif mutation == "unknown":
        item["name"] = "unregistered_alberta_63"
    elif mutation in ("unit", "dtype", "entity", "period"):
        item[mutation] = {
            "unit": "USD",
            "dtype": "Text",
            "entity": "Household",
            "period": "Month",
        }[mutation]
    elif mutation == "wrong_line":
        tax["versions"][0]["formula"] = tax["versions"][0]["formula"].replace(
            "alberta_tax_line_63", "alberta_tax_line_68"
        )
    elif mutation in ("missing_proof", "wrong_proof"):
        atoms = tax["metadata"]["proof"]["atoms"]
        if mutation == "missing_proof":
            tax["metadata"]["proof"]["atoms"] = [
                a for a in atoms if a["path"] != "versions[0].formula"
            ]
        else:
            for a in atoms:
                if a["path"] == "versions[0].formula":
                    a["source"]["excerpt"] = "Wrong unpublished source operation."
    elif mutation == "wrong_source_coordinate":
        source = SOURCE.replace("line 63", "line 64")
        assert source != SOURCE
        p["module"]["source_verification"]["source_sha256"] = hashlib.sha256(
            source.encode()
        ).hexdigest()
        for r in p["rules"]:
            for atom in r.get("metadata", {}).get("proof", {}).get("atoms", []):
                atom["source"]["excerpt"] = atom["source"]["excerpt"].replace(
                    "line 63", "line 64"
                )
    else:
        for case in c:
            for k in case["output"]:
                if k.endswith("#" + evidence._TAX):
                    case["output"][k] += 1
    result = certify(p, c, source)
    assert not any(owner == evidence._TAX for owner, *_ in result.condition_spans)

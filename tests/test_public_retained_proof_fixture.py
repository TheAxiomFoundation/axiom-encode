"""Proof-stage coverage of exact public retained inputs, not hosted execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness.proof_validator import validate_rulespec_proofs

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
INPUTS = FIXTURE_ROOT / "proof-parenthetical-219g"
REPLAY = FIXTURE_ROOT / "us_replay_benchmark/local-219g-proof-parenthetical-20261113"
CITATION = "us/statute/26/219/g"
TARGET_RULE = "phaseout_denominator_joint"
UNRELATED_ISSUE = (
    "Proof source evidence not found: rule `applicable_dollar_amount_other_base` "
    "proof atom 0 `source.excerpt` does not appear in `us/statute/26/219/g`."
)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_public_input_bytes_and_json_body_export_are_bound():
    provenance = json.loads((INPUTS / "provenance.json").read_text())
    assert provenance["schema"] == "axiom-encode/public-proof-inputs/v1"
    for name, digest in provenance["files"].items():
        assert hashlib.sha256((INPUTS / name).read_bytes()).hexdigest() == digest

    for label, name, expected_blob in [
        ("candidate", "candidate.yaml", "600cc21117d9af802b73a3406313230c9c661fd9"),
        (
            "companion",
            "candidate.test.yaml",
            "4da3e3e5470c95852e1df04882c8efdbad737305",
        ),
    ]:
        origin = provenance[label]
        assert origin["repository"] == "TheAxiomFoundation/rulespec-us"
        assert origin["commit"] == "a7c6d8058f67756150d92484dcb2afba258501f8"
        assert origin["git_blob_sha1"] == expected_blob
        assert _git_blob_sha1((INPUTS / name).read_bytes()) == expected_blob

    source = provenance["source"]
    assert source["repository"] == "TheAxiomFoundation/axiom-corpus"
    assert source["commit"] == "bcf433c5af81810507402743d8b578ed4cee1792"
    assert source["git_blob_sha1"] == "626aa60d3c32898160ee54bf9b36ee18dc3589e5"
    raw_row = (INPUTS / "source-row.jsonl").read_bytes()
    assert hashlib.sha256(raw_row).hexdigest() == (
        "5149410f9e4774925c0d840c465bce9e97b569108502a8fd0231e30290dfff61"
    )
    assert source["row_bytes_sha256"] == hashlib.sha256(raw_row).hexdigest()
    row = json.loads(raw_row)
    assert row["citation_path"] == CITATION
    body = row["body"].encode("utf-8")
    assert body == (INPUTS / "source.txt").read_bytes()
    assert source["body_utf8_sha256"] == hashlib.sha256(body).hexdigest()
    for name in ["candidate.yaml", "candidate.test.yaml", "source.txt"]:
        assert (INPUTS / name).read_bytes() == (REPLAY / name).read_bytes()

    payload = yaml.safe_load((INPUTS / "candidate.yaml").read_text())
    assert len(payload["rules"]) == 20 and "imports" not in payload
    assert len(yaml.safe_load((INPUTS / "candidate.test.yaml").read_text())) == 6
    joint = next(rule for rule in payload["rules"] if rule["name"] == TARGET_RULE)
    assert joint["versions"][0]["formula"] == "20000"
    assert type(joint["versions"][0]["formula"]) is str
    assert joint["metadata"]["proof"]["atoms"][0]["source"] == {
        "corpus_citation_path": CITATION,
        "excerpt": "$20,000 in the case of a joint return",
    }


def test_full_retained_candidate_fixes_joint_return_but_keeps_ellipsis_refusal():
    result = validate_rulespec_proofs(
        (REPLAY / "candidate.yaml").read_text(),
        source_texts={CITATION: (REPLAY / "source.txt").read_text()},
    )
    assert result.proof_required and result.atoms_checked == 20
    assert not result.passed
    assert result.issues == [UNRELATED_ISSUE]


@pytest.mark.parametrize(
    "replacement",
    [
        "($25,000 in the case of a joint return)",
        "-($20,000 in the case of a joint return)",
    ],
)
def test_full_retained_candidate_refuses_changed_operative_source(replacement):
    source = (REPLAY / "source.txt").read_text()
    changed = source.replace("($20,000 in the case of a joint return)", replacement)
    assert changed != source
    result = validate_rulespec_proofs(
        (REPLAY / "candidate.yaml").read_text(), source_texts={CITATION: changed}
    )
    assert not result.passed
    assert UNRELATED_ISSUE in result.issues
    assert any(f"rule `{TARGET_RULE}`" in issue for issue in result.issues)

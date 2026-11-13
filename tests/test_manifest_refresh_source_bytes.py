"""Real public TD1 bytes survive refresh without bypassing source validation."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli

FIXTURE = Path(__file__).parent / "fixtures/td1_manifest_refresh"


def candidate(tmp_path):
    raw = (FIXTURE / "candidate.yaml").read_bytes()
    manifest = json.loads((FIXTURE / "manifest.json").read_bytes())
    assert hashlib.sha256(raw).hexdigest() == manifest["applied_files"][0]["sha256"]
    target = tmp_path / "td1.yaml"
    target.write_bytes(raw)
    source = (FIXTURE / "source.txt").read_bytes()
    assert (
        hashlib.sha256(source).hexdigest()
        == manifest["source_attestation"]["source_sha256"]
    )
    (tmp_path / "source.txt").write_bytes(source)
    context = tmp_path / "context-manifest.json"
    context.write_text(json.dumps({"source_text_file": "source.txt"}))
    result = SimpleNamespace(
        context_manifest_file=str(context),
        backend=manifest["backend"],
        citation=manifest["citation"],
        source_attestation=manifest["source_attestation"],
    )
    setattr(result, cli._MANIFEST_ONLY_REFRESH_ATTR, True)
    setattr(
        result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR, hashlib.sha256(raw).hexdigest()
    )
    return result, target, raw


def test_real_td1_refresh_preserves_all_original_bytes_and_contract(tmp_path):
    result, target, raw = candidate(tmp_path)
    tests = (FIXTURE / "candidate.test.yaml").read_bytes()
    manifest = json.loads((FIXTURE / "manifest.json").read_bytes())
    assert hashlib.sha256(tests).hexdigest() == manifest["applied_files"][1]["sha256"]
    cases = yaml.safe_load(tests)
    assert len(cases) == 1 and len(cases[0]["output"]) == 72
    assert cases[0]["input"] == {}
    assert cli._immutable_rulespec_digest_issue(result, target) is None
    cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() == raw
    assert cli._immutable_rulespec_digest_issue(result, target) is None


@pytest.mark.parametrize("digest", [None, "", "0" * 63, "A" * 64, "g" * 64, 1])
def test_refresh_requires_separate_valid_original_digest(tmp_path, digest):
    result, target, raw = candidate(tmp_path)
    if digest is None:
        delattr(result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR)
    else:
        setattr(result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR, digest)
    with pytest.raises(RuntimeError, match="requires an immutable RuleSpec digest"):
        cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() == raw


def test_refresh_refuses_changed_policy_even_when_source_binding_matches(tmp_path):
    result, target, raw = candidate(tmp_path)
    target.write_bytes(raw + b"\n")
    with pytest.raises(RuntimeError, match="hash-bound RuleSpec bytes"):
        cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() == raw + b"\n"


@pytest.mark.parametrize(
    "defect",
    ["hash", "citation", "missing_verification", "attestation", "interpretation"],
)
def test_refresh_refuses_invalid_bindings_without_rewriting(tmp_path, defect):
    result, target, raw = candidate(tmp_path)
    if defect == "hash":
        raw = raw.replace(
            result.source_attestation["source_sha256"].encode(), b"0" * 64
        )
    elif defect == "citation":
        raw = raw.replace(result.citation.encode(), b"ca/policy/unbound")
    elif defect == "missing_verification":
        obj = yaml.safe_load(raw)
        del obj["module"]["source_verification"]
        raw = yaml.safe_dump(obj).encode()
    elif defect == "attestation":
        del result.source_attestation["corpus_release"]
    else:
        raw = raw.replace(
            b"module:\n", b"module:\n  interpretations: []\n  interpretations: []\n", 1
        )
    target.write_bytes(raw)
    setattr(
        result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR, hashlib.sha256(raw).hexdigest()
    )
    expected = {
        "hash": "Cannot refresh manifest with unbound source verification",
        "citation": "corpus path\\(s\\) not bound by resolver",
        "missing_verification": "Cannot refresh manifest with unbound source verification",
        "attestation": "incomplete source_attestation",
        "interpretation": "invalid interpretation YAML",
    }[defect]
    assert cli._immutable_rulespec_digest_issue(result, target) is None
    with pytest.raises(RuntimeError, match=expected):
        cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() == raw


@pytest.mark.parametrize("flag", [False, 1, "true", None])
def test_only_exact_internal_true_selects_refresh_preservation(tmp_path, flag):
    result, target, raw = candidate(tmp_path)
    setattr(result, cli._MANIFEST_ONLY_REFRESH_ATTR, flag)
    cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() != raw  # Ordinary stamping still serializes.
    assert yaml.safe_load(target.read_bytes()) == yaml.safe_load(raw)


def test_reviewed_promotion_existing_no_write_path_stays_intact(tmp_path):
    result, target, raw = candidate(tmp_path)
    delattr(result, cli._MANIFEST_ONLY_REFRESH_ATTR)
    delattr(result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR)
    setattr(result, cli._REVIEWED_CANDIDATE_PROMOTION_ATTR, True)
    cli._stamp_generated_source_attestation_for_apply(result, target)
    assert target.read_bytes() == raw

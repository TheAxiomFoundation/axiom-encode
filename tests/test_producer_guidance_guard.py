"""Custody/command tests use verifier seams, not synthetic-signature authority."""

import json
from types import SimpleNamespace

import pytest

from axiom_encode import cli
from axiom_encode.producer_review_guidance import ProducerGuidanceVerificationRequest

IDENTITY = {"repository": "TheAxiomFoundation/axiom-encode", "commit": "a" * 40, "version": "test"}
CITATION = "ca/policy/example"


@pytest.mark.parametrize("count,late_issue,expected_pass", [(1, False, True), (0, False, False), (2, False, False), (1, True, False)])
def test_guard_only_emits_single_record_after_whole_guard_passes(tmp_path, monkeypatch, capsys, count, late_issue, expected_pass):
    args = SimpleNamespace(repo=tmp_path, corpus_path=tmp_path, base_ref="b" * 40, head_ref="HEAD",
        all=False, retrospective=False, json=True, candidate_status=False,
        expected_encoder_checkout=tmp_path, producer_review_guidance_ref="{}",
        producer_review_guidance_citation=CITATION, producer_review_guidance_context_root=tmp_path)
    monkeypatch.setattr(cli, "_resolve_canonical_rulespec_checkout", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(cli, "_current_guard_encoder_execution_identity", lambda: IDENTITY)
    monkeypatch.setattr(cli, "_read_only_guard_encoder_execution_identity", lambda p: IDENTITY)
    record = {"manifest_sha256": "e" * 64, "context_manifest_sha256": "f" * 64}
    def guard(*a, **kw):
        request = kw["producer_guidance_request"]
        assert request.citation == CITATION
        assert request.producer_commit == IDENTITY["commit"]
        kw["producer_guidance_records"].extend([record] * count)
        return ["unrelated protected file mismatch"] if late_issue else []
    monkeypatch.setattr(cli, "guard_generated_change_issues", guard)
    with pytest.raises(SystemExit) as exc:
        cli.cmd_guard_generated(args)
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is expected_pass
    assert exc.value.code == (0 if expected_pass else 1)
    assert ("producer_review_guidance" in result) is expected_pass
    if expected_pass:
        assert result["producer_review_guidance"] == record


@pytest.mark.parametrize("loader_failure", [False, True, "payload_with_issue"])
def test_sink_receives_only_successful_loader_payload_and_digest(tmp_path, monkeypatch, loader_failure):
    request = ProducerGuidanceVerificationRequest("{}", tmp_path, IDENTITY["commit"], CITATION, tmp_path)
    payload = {"citation": CITATION, "applied_files": []}
    digest = "e" * 64
    records = []
    seen = []
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: object())
    def load(*args, **kwargs):
        assert kwargs["local_corpus_release"] is release
        if loader_failure:
            return payload if loader_failure == "payload_with_issue" else None, "ca", digest, ["bad signature"]
        return payload, "ca", digest, []
    def verify(actual_request, **kwargs):
        assert actual_request is request
        assert kwargs["verified_payload"] is payload
        assert kwargs["manifest_sha256"] == digest
        seen.append(True)
        return {"manifest_sha256": digest}
    monkeypatch.setattr(cli, "_load_verified_applied_encoding_manifest_payload", load)
    monkeypatch.setattr(cli, "verify_guidance_for_verified_manifest", verify)
    release = object()
    _, issues = cli._load_applied_encoding_manifest_entries(
        tmp_path, ["ca/.axiom/encoding-manifests/example.json"],
        expected_waiver_set_sha256="c" * 64, expected_encoder_identity=IDENTITY,
        local_corpus_release=release, producer_guidance_request=request,
        producer_guidance_records=records,
    )
    if loader_failure:
        assert issues[0] == "bad signature"
        assert records == [] and seen == []
    else:
        assert issues == [] and seen == [True]
        assert records == [{"manifest_sha256": digest}]


@pytest.mark.parametrize("override", [
    {"producer_review_guidance_ref": None}, {"producer_review_guidance_citation": None},
    {"producer_review_guidance_context_root": None}, {"expected_encoder_checkout": None},
    {"json": False}, {"all": True}, {"retrospective": True}, {"head_ref": "other"},
])
def test_incomplete_or_historical_guard_request_rejected(tmp_path, monkeypatch, override):
    values = dict(repo=tmp_path, corpus_path=tmp_path, base_ref="b" * 40, head_ref="HEAD",
        all=False, retrospective=False, json=True, candidate_status=False,
        expected_encoder_checkout=tmp_path, producer_review_guidance_ref="{}",
        producer_review_guidance_citation=CITATION, producer_review_guidance_context_root=tmp_path)
    values.update(override)
    monkeypatch.setattr(cli, "_resolve_canonical_rulespec_checkout", lambda *a, **kw: tmp_path)
    def unexpected(*a, **kw):
        pytest.fail("guard must not run for an invalid guidance request")
    monkeypatch.setattr(cli, "guard_generated_change_issues", unexpected)
    with pytest.raises(ValueError, match="complete ordinary current guard"):
        cli.cmd_guard_generated(SimpleNamespace(**values))


def test_guard_refuses_checkout_different_from_running_producer(tmp_path, monkeypatch):
    args = SimpleNamespace(repo=tmp_path, corpus_path=tmp_path, base_ref="b" * 40, head_ref="HEAD",
        all=False, retrospective=False, json=True, candidate_status=False,
        expected_encoder_checkout=tmp_path, producer_review_guidance_ref="{}",
        producer_review_guidance_citation=CITATION, producer_review_guidance_context_root=tmp_path)
    monkeypatch.setattr(cli, "_resolve_canonical_rulespec_checkout", lambda *a, **kw: tmp_path)
    monkeypatch.setattr(cli, "_current_guard_encoder_execution_identity", lambda: IDENTITY)
    monkeypatch.setattr(cli, "_read_only_guard_encoder_execution_identity", lambda p: {**IDENTITY, "commit": "f" * 40})
    with pytest.raises(ValueError, match="running producer"):
        cli.cmd_guard_generated(args)

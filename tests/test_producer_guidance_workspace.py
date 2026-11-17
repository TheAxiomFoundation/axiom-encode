"""Workspace transport tests; these do not authenticate a protected caller."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from axiom_encode.harness.evals import parse_runner_spec, prepare_eval_workspace
from axiom_encode.producer_review_guidance import ProducerReviewGuidance


def guidance():
    content = b"Reviewed full guidance\n"
    return ProducerReviewGuidance(
        "a" * 40,
        "review-guidance/reference.json",
        "b" * 64,
        100,
        "ca/policy/example",
        hashlib.sha256(b"source\nbody\n").hexdigest(),
        content,
        hashlib.sha256(content).hexdigest(),
    )


def workspace(tmp_path, record=None, **kwargs):
    root = tmp_path / "rulespec-ca" / "ca"
    root.mkdir(parents=True, exist_ok=True)
    return prepare_eval_workspace(
        citation="ca/policy/example",
        runner=parse_runner_spec("openai:gpt-5.4"),
        output_root=tmp_path / "out",
        source_text="source\r\nbody\r",
        axiom_rules_path=root,
        mode="cold",
        producer_review_guidance=record,
        **kwargs,
    )


def test_exact_workspace_source_and_guidance(tmp_path):
    record = guidance()
    ws = workspace(tmp_path, record)
    manifest = json.loads(ws.manifest_file.read_text())
    assert ws.source_text_file.read_bytes() == b"source\nbody\n"
    assert manifest["producer_review_guidance"]["source_sha256"] == record.source_sha256
    findings = manifest["review_findings_files"]
    assert len(findings) == 1
    assert findings[0]["content"].encode() == record.content
    assert (ws.root / findings[0]["workspace_path"]).read_bytes() == record.content
    assert findings[0]["sha256"] == record.content_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        {"citation": "wrong"},
        {"source_sha256": "0" * 64},
        {"content": b"tampered\n"},
    ],
)
def test_mismatch_refused_before_workspace_creation(tmp_path, mutation):
    with pytest.raises(ValueError):
        workspace(tmp_path, replace(guidance(), **mutation))
    assert not (tmp_path / "out").exists()


def test_mixed_inline_refused(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        workspace(tmp_path, guidance(), review_findings_paths=[tmp_path / "inline"])
    assert not (tmp_path / "out").exists()


def test_absent_guidance_keeps_manifest_shape(tmp_path):
    ws = workspace(tmp_path)
    manifest = json.loads(ws.manifest_file.read_text())
    assert "producer_review_guidance" not in manifest
    assert manifest["review_findings_files"] == []


@pytest.mark.parametrize(
    "content", [b"\xff", b"bad\r\n", b"bad\n\n", b"x" * (512 * 1024 + 1)]
)
def test_invalid_content_preserves_existing_workspace(tmp_path, content):
    ws = workspace(tmp_path)
    sentinel = ws.root / "sentinel"
    sentinel.write_text("preserve")
    record = replace(
        guidance(), content=content, content_sha256=hashlib.sha256(content).hexdigest()
    )
    with pytest.raises(ValueError):
        workspace(tmp_path, record)
    assert sentinel.read_text() == "preserve"


def test_large_guidance_full_prompt_rendering(tmp_path):
    from axiom_encode.harness.evals import _format_mandatory_review_findings

    content = ("BEGIN" + "x" * 250000 + "END\n").encode()
    record = replace(
        guidance(), content=content, content_sha256=hashlib.sha256(content).hexdigest()
    )
    ws = workspace(tmp_path, record)
    rendered = _format_mandatory_review_findings(ws)
    assert content.decode().strip() in rendered
    manifest = json.loads(ws.manifest_file.read_text())
    assert manifest["review_findings_files"][0]["sha256"] == record.content_sha256


@pytest.mark.parametrize(
    "override",
    [
        {"citations": ["one", "two"]},
        {"runner_specs": ["openai:a", "openai:b"]},
        {"review_findings_paths": ["inline"]},
        {"require_complete_source_unit": False},
        {"legacy_replacement": object()},
        {"repair_candidate_tests_only": True},
        {"accept_valid_retry_candidate": True},
    ],
)
def test_harness_rejects_unsupported_guidance_modes_before_resolution(
    tmp_path, monkeypatch, override
):
    from axiom_encode.harness import evals

    def unexpected(*args, **kwargs):
        pytest.fail("source resolution must not run for unsupported guidance mode")

    monkeypatch.setattr(evals, "resolve_corpus_source_unit", unexpected)
    args = dict(
        citations=["ca/policy/example"],
        runner_specs=["openai:gpt-5.4"],
        output_root=tmp_path,
        policy_path=tmp_path,
        runtime_axiom_rules_path=tmp_path,
        corpus_release=object(),
        require_complete_source_unit=True,
        producer_review_guidance=guidance(),
    )
    args.update(override)
    with pytest.raises(ValueError, match="single ordinary full-source"):
        evals.run_model_eval(**args)


@pytest.mark.parametrize("canonical_replacement", [False, True])
def test_harness_forwards_guidance_with_resolved_source(
    tmp_path, monkeypatch, canonical_replacement
):
    from axiom_encode.harness import evals

    record = guidance()
    source = object()
    monkeypatch.setattr(evals, "resolve_corpus_source_unit", lambda *a: source)
    seen = []

    def capture(**kwargs):
        seen.append(kwargs)
        return "sentinel"

    monkeypatch.setattr(evals, "_run_single_eval", capture)
    result = evals.run_model_eval(
        citations=[record.citation],
        runner_specs=["openai:gpt-5.4"],
        output_root=tmp_path,
        policy_path=tmp_path,
        runtime_axiom_rules_path=tmp_path,
        corpus_release=object(),
        require_complete_source_unit=True,
        producer_review_guidance=record,
        replacement_overlay_scope=canonical_replacement,
        target_relative_output=Path("policies/example.yaml")
        if canonical_replacement
        else None,
        validation_retry_candidate=evals.ValidationRetryCandidate(
            "rules: []", context_role="replacement_baseline"
        )
        if canonical_replacement
        else None,
    )
    assert result == ["sentinel"]
    assert len(seen) == 1
    if canonical_replacement:
        assert (
            seen[0]["validation_retry_candidate"].context_role == "replacement_baseline"
        )
        assert seen[0]["target_relative_output"] == Path("policies/example.yaml")
    assert seen[0]["replacement_overlay_scope"] is canonical_replacement
    assert seen[0]["source_unit"] is source
    assert seen[0]["producer_review_guidance"] is record

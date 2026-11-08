"""Immutable file parsing does not stand in for protected source authentication."""

import hashlib
import json
import subprocess

import pytest

from axiom_encode.producer_review_guidance import (
    MAX_GUIDANCE_BYTES,
    read_producer_review_guidance,
)

CITATION = "ca/policy/example"
SOURCE = "a" * 64


@pytest.fixture
def make_repo(tmp_path):
    def build(*, envelope=None, raw=None, mode="regular"):
        repo = tmp_path / "producer"
        repo.mkdir()

        def git(*args):
            return subprocess.check_output(["git", "-C", str(repo), *args])

        git("init", "-q")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "Test")
        path = repo / "review-guidance/reference.json"
        path.parent.mkdir()
        if raw is None:
            raw = json.dumps(
                envelope
                if envelope is not None
                else {
                    "schema": "axiom-encode/producer-review-guidance/v1",
                    "citation": CITATION,
                    "source_sha256": SOURCE,
                    "finding": "Full reviewed guidance\nSecond line",
                }
            ).encode()
        if mode == "symlink":
            (repo / "target").write_bytes(raw)
            path.symlink_to("../target")
        else:
            path.write_bytes(raw)
            if mode == "executable":
                path.chmod(0o755)
        git("add", ".")
        git("commit", "-qm", "Immutable fixture")
        head = git("rev-parse", "HEAD").decode().strip()
        descriptor = {
            "schema": "axiom-encode/producer-review-finding-ref/v1",
            "producer_commit": head,
            "path": "review-guidance/reference.json",
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        return repo, head, descriptor, path

    return build


def read(repo, head, descriptor, **kwargs):
    return read_producer_review_guidance(
        json.dumps(descriptor),
        encoder_repo=repo,
        expected_producer_sha=head,
        citation=kwargs.get("citation", CITATION),
        authenticated_source_sha256=kwargs.get("source", SOURCE),
    )


def test_immutable_blob_and_exact_newline(make_repo):
    repo, head, descriptor, path = make_repo()
    path.write_text("dirty working-tree substitution")
    result = read(repo, head, descriptor)
    assert result.content == b"Full reviewed guidance\nSecond line\n"
    assert result.content_sha256 == hashlib.sha256(result.content).hexdigest()
    assert result.blob_sha256 == descriptor["sha256"]
    assert path.read_text() == "dirty working-tree substitution"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("producer_commit", "0" * 40),
        ("path", "../outside.json"),
        ("path", "/review-guidance/x.json"),
        ("path", "review-guidance/../x.json"),
        ("path", "review-guidance//x.json"),
        ("path", "review-guidance/a\\x.json"),
        ("path", "review-guidance/.hidden.json"),
        ("path", "review-guidance/missing.json"),
        ("size_bytes", True),
        ("size_bytes", 0),
        ("size_bytes", MAX_GUIDANCE_BYTES + 1),
        ("size_bytes", 1),
        ("sha256", "0" * 64),
        ("sha256", None),
        ("url", "https://invalid"),
    ],
)
def test_descriptor_refusals(make_repo, field, value):
    repo, head, descriptor, _ = make_repo()
    descriptor[field] = value
    with pytest.raises(ValueError):
        read(repo, head, descriptor)


@pytest.mark.parametrize("mode", ["symlink", "executable"])
def test_nonregular_blob_refused(make_repo, mode):
    repo, head, descriptor, _ = make_repo(mode=mode)
    with pytest.raises(ValueError):
        read(repo, head, descriptor)


@pytest.mark.parametrize(
    "raw",
    [
        b"{}",
        b"[]",
        b"\xef\xbb\xbf{}",
        b"\xff",
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b"{\r\n}",
        b" " * 20,
    ],
)
def test_invalid_blob_json(make_repo, raw):
    repo, head, descriptor, _ = make_repo(raw=raw)
    with pytest.raises(ValueError):
        read(repo, head, descriptor)


@pytest.mark.parametrize(
    "finding",
    ["", " ", "bad\rtext", "bad\x00text", "bad\x7ftext", "extra newline\n", True, None],
)
def test_finding_normalization_is_rejection(make_repo, finding):
    envelope = {
        "schema": "axiom-encode/producer-review-guidance/v1",
        "citation": CITATION,
        "source_sha256": SOURCE,
        "finding": finding,
    }
    repo, head, descriptor, _ = make_repo(envelope=envelope)
    with pytest.raises(ValueError):
        read(repo, head, descriptor)


@pytest.mark.parametrize(
    "kwargs", [{"citation": "ca/policy/other"}, {"source": "b" * 64}]
)
def test_authenticated_identity_mismatch(make_repo, kwargs):
    repo, head, descriptor, _ = make_repo()
    with pytest.raises(ValueError):
        read(repo, head, descriptor, **kwargs)


def test_large_guidance_is_not_truncated(make_repo):
    finding = "x" * 300000
    envelope = {
        "schema": "axiom-encode/producer-review-guidance/v1",
        "citation": CITATION,
        "source_sha256": SOURCE,
        "finding": finding,
    }
    repo, head, descriptor, _ = make_repo(envelope=envelope)
    assert read(repo, head, descriptor).content == (finding + "\n").encode()


@pytest.mark.parametrize("token", ["1e999", "-1e999", "NaN", "Infinity"])
def test_json_nonfinite_numbers_rejected(token):
    from axiom_encode.producer_review_guidance import _strict_json

    with pytest.raises(ValueError, match="nonfinite"):
        _strict_json(('{"value":' + token + '}').encode(), 4096)


def test_inherited_git_redirection_ignored(make_repo, monkeypatch):
    repo, head, descriptor, _ = make_repo()
    monkeypatch.setenv("GIT_DIR", "/nonexistent/injected-repository")
    monkeypatch.setenv("GIT_WORK_TREE", "/nonexistent/worktree")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.repositoryformatversion")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "999")
    assert read(repo, head, descriptor).content.startswith(b"Full reviewed guidance")


def test_missing_promisor_blob_refused_without_fetch(make_repo, tmp_path):
    repo, head, descriptor, _ = make_repo()
    oid = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", head + ":review-guidance/reference.json"]
    ).decode().strip()
    marker = tmp_path / "fetch-was-attempted"
    helper = tmp_path / "remote-helper"
    helper.write_text("#!/bin/sh\ntouch " + str(marker) + "\nexit 1\n")
    helper.chmod(0o755)
    for key, value in [
        ("extensions.partialclone", "origin"),
        ("remote.origin.promisor", "true"),
        ("remote.origin.url", "ext::" + str(helper)),
        ("protocol.ext.allow", "always"),
    ]:
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True)
    (repo / ".git/objects" / oid[:2] / oid[2:]).unlink()
    with pytest.raises(ValueError, match="immutable producer tree|Git blob size mismatch"):
        read(repo, head, descriptor)
    assert not marker.exists()


@pytest.mark.parametrize("mismatch", [None, "repository", "commit", "version"])
def test_protected_producer_binding_before_blob_read(make_repo, monkeypatch, mismatch):
    from types import SimpleNamespace

    from axiom_encode import cli
    from axiom_encode.harness import evals

    source_text = "source\r\nbody\r"
    source_digest = hashlib.sha256(b"source\nbody\n").hexdigest()
    repo, head, descriptor, _ = make_repo(envelope={
        "schema": "axiom-encode/producer-review-guidance/v1",
        "citation": CITATION, "source_sha256": source_digest, "finding": "reviewed",
    })
    running = {"repository": "TheAxiomFoundation/axiom-encode", "commit": head, "version": "test"}
    checkout = dict(running)
    if mismatch:
        checkout[mismatch] = "wrong"
    monkeypatch.setattr(cli, "_current_guard_encoder_execution_identity", lambda: running)
    monkeypatch.setattr(cli, "_read_only_guard_encoder_execution_identity", lambda p: checkout)
    source = SimpleNamespace(body=source_text, amendment_request=None, requested=CITATION, citation_path=CITATION)
    release = object()
    validated = []
    def validate(unit, corpus, *, target_identifier):
        assert unit is source and corpus is release and target_identifier == CITATION
        validated.append(True)
    monkeypatch.setattr(evals, "_validate_corpus_source_unit", validate)
    kwargs = dict(encoder_checkout=repo, citation=CITATION, source_unit=source, corpus_release=release)
    if mismatch:
        with pytest.raises(ValueError, match="running protected producer"):
            cli._resolve_encode_producer_review_guidance(json.dumps(descriptor), **kwargs)
        assert validated == []
    else:
        result = cli._resolve_encode_producer_review_guidance(json.dumps(descriptor), **kwargs)
        assert validated == [True]
        assert result.content == b"reviewed\n"
        assert result.source_sha256 == source_digest


def test_source_validation_failure_stops_guidance_read(monkeypatch, tmp_path):
    from axiom_encode import cli, producer_review_guidance
    from axiom_encode.harness import evals
    identity = {"repository": "TheAxiomFoundation/axiom-encode", "commit": "a" * 40, "version": "test"}
    monkeypatch.setattr(cli, "_current_guard_encoder_execution_identity", lambda: identity)
    monkeypatch.setattr(cli, "_read_only_guard_encoder_execution_identity", lambda p: identity)
    def reject(*args, **kwargs):
        raise ValueError("source authentication refused")
    def unexpected(*args, **kwargs):
        pytest.fail("guidance reader must not run after source refusal")
    monkeypatch.setattr(evals, "_validate_corpus_source_unit", reject)
    monkeypatch.setattr(producer_review_guidance, "read_producer_review_guidance", unexpected)
    with pytest.raises(ValueError, match="source authentication refused"):
        cli._resolve_encode_producer_review_guidance("{}", encoder_checkout=tmp_path,
            citation=CITATION, source_unit=object(), corpus_release=object())


@pytest.mark.parametrize("citation,requested,resolved,amendment", [
    (CITATION, CITATION, "ca/policy", None),
    (CITATION, "ca/policy/alias", CITATION, None),
    (" ca/policy/example", " ca/policy/example", " ca/policy/example", None),
    (CITATION, CITATION, CITATION, "amendment"),
])
def test_guidance_requires_exact_source_scope(monkeypatch, tmp_path, citation, requested, resolved, amendment):
    from types import SimpleNamespace

    from axiom_encode import cli, producer_review_guidance
    from axiom_encode.harness import evals
    identity = {"repository": "TheAxiomFoundation/axiom-encode", "commit": "a" * 40, "version": "test"}
    monkeypatch.setattr(cli, "_current_guard_encoder_execution_identity", lambda: identity)
    monkeypatch.setattr(cli, "_read_only_guard_encoder_execution_identity", lambda p: identity)
    monkeypatch.setattr(evals, "_validate_corpus_source_unit", lambda *a, **kw: None)
    def unexpected(*args, **kwargs):
        pytest.fail("blob reader must not run for inexact source scope")
    monkeypatch.setattr(producer_review_guidance, "read_producer_review_guidance", unexpected)
    source = SimpleNamespace(body="source", requested=requested, citation_path=resolved, amendment_request=amendment)
    with pytest.raises(ValueError):
        cli._resolve_encode_producer_review_guidance("{}", encoder_checkout=tmp_path,
            citation=citation, source_unit=source, corpus_release=object())


@pytest.mark.parametrize("override", [
    {"producer_review_guidance_ref": None}, {"producer_review_guidance_checkout": None},
    {"apply": False}, {"require_complete_source_unit": False}, {"backend": "codex"},
    {"review_findings": ["inline"]}, {"repair_candidate_root": "prior"},
    {"repair_candidate_path": "prior.yaml"}, {"repair_candidate_rulespec_sha256": "a"*64},
    {"repair_candidate_tests_sha256": "a"*64}, {"repair_candidate_tests_only": True},
    {"replace_legacy_rulespec_path": "old.yaml"}, {"legacy_dependent_rulespec_path": ["dep"]},
    {"legacy_exact_dependent_rulespec_path": ["dep"]}, {"legacy_retained_successor_rulespec_path": ["dep"]},
    {"scheduled_dependent_rulespec_path": ["dep"]}, {"apply_target_only": True},
    {"skip_reviewers": True},
])
def test_cli_guidance_modes_refused_before_signer(monkeypatch, override):
    from types import SimpleNamespace

    from axiom_encode import cli
    values = dict(producer_review_guidance_ref="{}", producer_review_guidance_checkout="encoder",
                  apply=True, require_complete_source_unit=True, backend="openai")
    values.update(override)
    def unexpected(*args, **kwargs):
        pytest.fail("signer must not open for unsupported guidance mode")
    monkeypatch.setattr(cli, "_isolated_apply_manifest_signer", unexpected)
    with pytest.raises(ValueError):
        cli._cmd_encode_under_current_trust(SimpleNamespace(**values))


def test_cli_guidance_allows_ordinary_canonical_replacement():
    from types import SimpleNamespace

    from axiom_encode import cli
    args = SimpleNamespace(producer_review_guidance_ref="{}", producer_review_guidance_checkout="encoder",
                           apply=True, require_complete_source_unit=True, backend="openai",
                           replace_rulespec_path="ca/policies/example.yaml")
    cli._validate_encode_producer_guidance_request(args)
    cli._validate_encode_producer_guidance_request(SimpleNamespace())


@pytest.mark.parametrize("mutation", [None, "source", "producer", "tool", "citation", "context_bytes", "selfhashed_finding", "selfhashed_binding", "extra_finding", "symlink", "outside"])
def test_verified_manifest_guidance_binding(make_repo, tmp_path, mutation):
    from dataclasses import asdict

    from axiom_encode.producer_review_guidance import (
        ProducerGuidanceVerificationRequest,
        verify_guidance_for_verified_manifest,
    )
    repo, head, descriptor, _ = make_repo()
    record = read(repo, head, descriptor)
    context_root = tmp_path / "generated"
    context_root.mkdir()
    context_file = context_root / "context.json"
    binding = asdict(record)
    binding.pop("content")
    context = {"citation": CITATION, "producer_review_guidance": binding,
               "review_findings_files": [{"content": record.content.decode(), "sha256": record.content_sha256}]}
    context_file.write_text(json.dumps(context))
    payload = {"tool": "axiom-encode encode --apply", "backend": "openai", "citation": CITATION,
               "axiom_encode_git": {"commit": head},
               "source_attestation": {"requested_corpus_citation_path": CITATION,
                    "resolved_corpus_citation_path": CITATION, "generation_input_sha256": SOURCE},
               "context_manifest_file": str(context_file),
               "context_manifest_sha256": hashlib.sha256(context_file.read_bytes()).hexdigest()}
    if mutation == "source":
        payload["source_attestation"]["generation_input_sha256"] = "0" * 64
    elif mutation == "producer":
        payload["axiom_encode_git"]["commit"] = "0" * 40
    elif mutation in {"tool", "citation"}:
        payload[mutation] = "wrong"
    elif mutation == "context_bytes":
        context_file.write_text("{}")
    elif mutation in {"selfhashed_finding", "selfhashed_binding", "extra_finding"}:
        if mutation == "selfhashed_finding":
            context["review_findings_files"][0] = {"content": "wrong\n", "sha256": hashlib.sha256(b"wrong\n").hexdigest()}
        elif mutation == "selfhashed_binding":
            context["producer_review_guidance"]["source_sha256"] = "0" * 64
        else:
            context["review_findings_files"].append(context["review_findings_files"][0])
        context_file.write_text(json.dumps(context))
        payload["context_manifest_sha256"] = hashlib.sha256(context_file.read_bytes()).hexdigest()
    elif mutation in {"symlink", "outside"}:
        outside = tmp_path / "outside.json"
        outside.write_bytes(context_file.read_bytes())
        if mutation == "symlink":
            context_file.unlink()
            context_file.symlink_to(outside)
        else:
            payload["context_manifest_file"] = str(outside)
    request = ProducerGuidanceVerificationRequest(json.dumps(descriptor), repo, head, CITATION, context_root)
    kwargs = dict(manifest_path="ca/.axiom/encoding-manifests/test.json",
                  manifest_sha256="e" * 64, verified_payload=payload)
    if mutation:
        with pytest.raises((ValueError, RuntimeError)):
            verify_guidance_for_verified_manifest(request, **kwargs)
    else:
        result = verify_guidance_for_verified_manifest(request, **kwargs)
        assert result["manifest_sha256"] == "e" * 64
        assert result["context_manifest_sha256"] == payload["context_manifest_sha256"]
        assert result["source_sha256"] == SOURCE
        assert result["content_sha256"] == record.content_sha256


@pytest.mark.parametrize("value", ["", "ordinary finding\n", '{"schema":"other"}', " axiom-producer-review-guidance-ref:v1:{}"])
def test_untagged_inline_not_reinterpreted(value):
    from axiom_encode.producer_review_guidance import (
        producer_guidance_descriptor_from_review_finding,
    )
    assert producer_guidance_descriptor_from_review_finding(value) is None


@pytest.mark.parametrize("value", ["axiom-producer-review-guidance-ref:", "axiom-producer-review-guidance-ref:v2:{}", "axiom-producer-review-guidance-ref:v1:{}", "axiom-producer-review-guidance-ref:v1:bad"])
def test_reserved_malformed_tag_refused(value):
    from axiom_encode.producer_review_guidance import (
        producer_guidance_descriptor_from_review_finding,
    )
    with pytest.raises(ValueError):
        producer_guidance_descriptor_from_review_finding(value)


def test_tagged_descriptor_exact_text_no_git_read(make_repo, monkeypatch):
    from axiom_encode import producer_review_guidance as module
    _, _, descriptor, _ = make_repo()
    text = json.dumps(descriptor, indent=2)
    def unexpected(*a, **kw):
        pytest.fail("input parsing must not read Git")
    monkeypatch.setattr(module, "_git", unexpected)
    assert module.producer_guidance_descriptor_from_review_finding(module.GUIDANCE_INPUT_PREFIX + text) == text


@pytest.mark.parametrize("field", ["queue_id", "queue_item_id", "queue_manifest_sha256", "queue_item_generation_sha256", "queue_dispatcher_run_id", "repair_run_id", "replace_legacy_rulespec_path", "legacy_exact_dependent_rulespec_path", "second_legacy_exact_dependent_rulespec_path", "dependent_citation", "second_dependent_citation", "dependent_review_finding", "second_dependent_review_finding"])
def test_workflow_pointer_refuses_other_lanes(make_repo, field):
    from axiom_encode.producer_review_guidance import (
        GUIDANCE_INPUT_PREFIX,
        validate_workflow_guidance_input,
    )
    _, head, descriptor, _ = make_repo()
    inputs = {"review_finding": GUIDANCE_INPUT_PREFIX + json.dumps(descriptor), "citation": CITATION, field: "nonempty"}
    with pytest.raises(ValueError, match="cannot mix"):
        validate_workflow_guidance_input(inputs, producer_commit=head)


def test_workflow_pointer_empty_transaction_and_inline_compatibility(make_repo):
    from axiom_encode.producer_review_guidance import (
        GUIDANCE_INPUT_PREFIX,
        validate_workflow_guidance_input,
    )
    _, head, descriptor, _ = make_repo()
    raw = json.dumps(descriptor)
    inputs = {"review_finding": GUIDANCE_INPUT_PREFIX + raw, "citation": CITATION, "source_bundle_json": "[]"}
    assert validate_workflow_guidance_input(inputs, producer_commit=head) == raw
    with pytest.raises(ValueError, match="workflow producer"):
        validate_workflow_guidance_input(inputs, producer_commit="0"*40)
    inputs["source_bundle_json"] = '["ca/policy/extra"]'
    with pytest.raises(ValueError, match="single full-source"):
        validate_workflow_guidance_input(inputs, producer_commit=head)
    assert validate_workflow_guidance_input({"review_finding": "ordinary", "queue_id": "queue"}, producer_commit=head) is None


@pytest.mark.parametrize("field", ["dependent_review_finding", "second_dependent_review_finding"])
def test_dependent_pointer_is_never_treated_as_inline(field):
    from axiom_encode.producer_review_guidance import validate_workflow_guidance_input
    with pytest.raises(ValueError, match="primary-only"):
        validate_workflow_guidance_input({"review_finding": "ordinary", field: "axiom-producer-review-guidance-ref:v1:{}"}, producer_commit="a" * 40)


@pytest.mark.parametrize("mutation", [None, "manifest", "context", "guard_failure", "guard_issues", "missing_record", "record_context", "record_extra", "event_inline", "event_blob", "event_producer"])
def test_packaging_rebinds_original_event_and_guard_bytes(make_repo, tmp_path, mutation):
    from dataclasses import asdict

    from axiom_encode.producer_review_guidance import (
        GUIDANCE_INPUT_PREFIX,
        ProducerGuidanceVerificationRequest,
        verify_guidance_for_verified_manifest,
        verify_packaged_producer_guidance,
    )
    repo, head, descriptor, _ = make_repo()
    guidance = read(repo, head, descriptor)
    binding = asdict(guidance)
    binding.pop("content")
    root = tmp_path / "generated"
    root.mkdir()
    context_file = root / "context.json"
    context_file.write_text(json.dumps({"citation": CITATION, "producer_review_guidance": binding,
        "review_findings_files": [{"content": guidance.content.decode(), "sha256": guidance.content_sha256}]}))
    manifest = {"tool": "axiom-encode encode --apply", "backend": "openai", "citation": CITATION,
        "axiom_encode_git": {"commit": head}, "source_attestation": {
        "requested_corpus_citation_path": CITATION, "resolved_corpus_citation_path": CITATION,
        "generation_input_sha256": SOURCE}, "context_manifest_file": str(context_file),
        "context_manifest_sha256": hashlib.sha256(context_file.read_bytes()).hexdigest()}
    raw = json.dumps(manifest).encode()
    path = "ca/.axiom/encoding-manifests/example.json"
    request = ProducerGuidanceVerificationRequest(json.dumps(descriptor), repo, head, CITATION, root)
    record = verify_guidance_for_verified_manifest(request, manifest_path=path,
        manifest_sha256=hashlib.sha256(raw).hexdigest(), verified_payload=manifest)
    guard = {"passed": True, "issues": [], "producer_review_guidance": record}
    inputs = {"citation": CITATION, "review_finding": GUIDANCE_INPUT_PREFIX + json.dumps(descriptor)}
    if mutation == "manifest":
        raw += b" "
    elif mutation == "context":
        context_file.write_text("{}")
    elif mutation == "guard_failure":
        guard["passed"] = False
    elif mutation == "guard_issues":
        guard["issues"] = ["later mismatch"]
    elif mutation == "missing_record":
        guard.pop("producer_review_guidance")
    elif mutation == "record_context":
        record["context_manifest_sha256"] = "0" * 64
    elif mutation == "record_extra":
        record["unexpected"] = True
    elif mutation == "event_inline":
        inputs["review_finding"] = "ordinary"
    elif mutation in {"event_blob", "event_producer"}:
        descriptor["sha256" if mutation == "event_blob" else "producer_commit"] = "0" * (64 if mutation == "event_blob" else 40)
        inputs["review_finding"] = GUIDANCE_INPUT_PREFIX + json.dumps(descriptor)
    kwargs = dict(original_inputs=inputs, producer_commit=head, encoder_checkout=repo,
                  context_root=root, manifest_path=path, manifest_bytes=raw, guard_payload=guard)
    if mutation:
        with pytest.raises(ValueError):
            verify_packaged_producer_guidance(**kwargs)
    else:
        assert verify_packaged_producer_guidance(**kwargs).encode() == guidance.content


@pytest.mark.parametrize("mode", ["regular", "symlink", "executable", "wrong-size"])
def test_reader_through_actual_protected_git_wrapper(make_repo, tmp_path, monkeypatch, mode):
    import os
    import shutil
    import sys
    from pathlib import Path

    from scripts import provision_verification_supervisor as provisioner

    repo, head, descriptor, path = make_repo(mode=mode if mode != "wrong-size" else "regular")
    if mode == "wrong-size":
        descriptor["size_bytes"] += 1
    trusted_git = Path(shutil.which("git")).resolve()
    tools = tmp_path / "protected-tools"
    tools.mkdir()
    provisioner._install_trusted_git_wrapper(tools, Path(sys.executable), trusted_git)
    monkeypatch.setenv("PATH", str(tools) + os.pathsep + os.environ["PATH"])
    if mode == "regular":
        path.write_text("working-tree substitution")
        assert read(repo, head, descriptor).content == b"Full reviewed guidance\nSecond line\n"
    else:
        with pytest.raises(ValueError):
            read(repo, head, descriptor)

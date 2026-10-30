"""Git ancestry belongs to the staging origin; execution bytes belong to overlay."""

import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals
from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import validator_pipeline as v
from tests.test_evals import _write_test_corpus_release

PIN = "a" * 40
CITATION = "ca/policy/test"
BODY = "Add: reported + local_amount."
REL = Path("policies/test/consumer.yaml")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root):
    git(root, "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "commit",
        "--allow-empty",
        "-qm",
        "fixture",
    )


@pytest.fixture
def staged(tmp_path, monkeypatch):
    source = tmp_path / "source" / "rulespec-ca"
    content = source / "ca"
    target = content / REL
    target.parent.mkdir(parents=True)
    provider = target.with_name("provider.yaml")
    provider.write_text(
        yaml.safe_dump(
            {
                "format": "rulespec/v1",
                "module": {},
                "rules": [
                    {
                        "name": "reported",
                        "kind": "derived",
                        "versions": [{"formula": "5"}],
                    }
                ],
            }
        )
    )
    payload = {
        "format": "rulespec/v1",
        "module": {},
        "imports": ["ca:policies/test/provider#reported"],
        "rules": [
            {
                "name": "out",
                "kind": "derived",
                "versions": [{"formula": "reported + local_amount"}],
                "metadata": {
                    "proof": {
                        "atoms": [
                            {
                                "kind": "formula",
                                "source": {
                                    "corpus_citation_path": CITATION,
                                    "excerpt": BODY,
                                },
                            }
                        ]
                    }
                },
            }
        ],
    }
    target.write_text(yaml.safe_dump(payload))
    target.with_name("consumer.test.yaml").write_text("[]\n")
    (content / "sibling.txt").write_text("must remain visible to ordinary guards")
    git(source, "init", "-q")
    commit(source)
    overlay = tmp_path / "overlay" / "rulespec-ca"
    overlay.parent.mkdir()
    cli._stage_apply_overlay_dependency_root(source=source, target=overlay)
    binary = tmp_path / "fixture-binary"
    binary.write_bytes(b"fixture-only-not-a-real-engine")
    # The fixture controls source resolution/binary transport only. Actual
    # import collection, source-consumer recognition, copy/clone and identity
    # checks run without substitutes; no compiled or signed result is claimed.
    monkeypatch.setattr(
        v.ValidatorPipeline,
        "_source_texts_for_rulespec_content",
        lambda *a: {CITATION: BODY},
    )
    monkeypatch.setattr(
        v.ValidatorPipeline,
        "_proof_source_texts_for_rulespec_content",
        lambda *a, **k: {CITATION: BODY},
    )
    monkeypatch.setattr(v.ValidatorPipeline, "_axiom_rules_binary", lambda self: binary)
    return SimpleNamespace(
        source=source, overlay=overlay, target=target, binary=binary, tmp=tmp_path
    )


def pipeline(staged, origin=None):
    return v.ValidatorPipeline(
        staged.overlay / "ca",
        staged.tmp / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=PIN,
        execution_overlay_origin=origin,
    )


def origin(staged):
    result = e.bind_execution_overlay_origin(
        e.capture_execution_git_origin(staged.source / "ca"), staged.overlay / "ca"
    )
    assert result is not None
    return result


def test_actual_staging_origin_materializes_overlay_bytes_without_git_laundering(
    staged,
):
    file = staged.overlay / "ca" / REL
    file.write_text(file.read_text() + "\n# new candidate, not original checkout\n")
    assert not (staged.overlay / ".git").exists()
    assert e._snapshot_plan(pipeline(staged), file) is None
    with e.compilation_snapshot(pipeline(staged, origin(staged)), file) as capture:
        assert capture is not None
        assert capture.original_root == staged.overlay
        assert capture.git_head == git(staged.source, "rev-parse", "HEAD")
        assert capture.fresh()
        assert (capture.snapshot_root / "ca" / REL).read_bytes() == file.read_bytes()
        assert file.read_bytes() != staged.target.read_bytes()
        assert not (capture.snapshot_root / "ca/sibling.txt").exists()
        assert (
            staged.overlay / "ca/sibling.txt"
        ).read_text() == "must remain visible to ordinary guards"
        assert not (capture.snapshot_root / ".git/objects/info/alternates").exists()
        assert not (capture.snapshot_root / ".git/commondir").exists()
        assert file.stat().st_ino != (capture.snapshot_root / "ca" / REL).stat().st_ino


@pytest.mark.parametrize(
    "change",
    [
        "head",
        "source_root",
        "overlay_root",
        "jurisdiction",
        "source_symlink",
        "overlay_symlink",
        "malformed",
    ],
)
def test_conflicting_or_stale_origin_does_not_grant_snapshot(staged, change):
    bound = origin(staged)
    if change == "head":
        commit(staged.source)
    elif change == "source_root":
        other = staged.tmp / "other" / "rulespec-ca"
        other.mkdir(parents=True)
        git(other, "init", "-q")
        commit(other)
        bound = replace(bound, source_content_root=other / "ca")
    elif change == "overlay_root":
        bound = replace(bound, overlay_content_root=staged.source / "ca")
    elif change == "jurisdiction":
        (staged.overlay / "us").mkdir()
        bound = replace(bound, overlay_content_root=staged.overlay / "us")
    elif change in ("source_symlink", "overlay_symlink"):
        root = staged.source if change == "source_symlink" else staged.overlay
        old = root.with_name("moved")
        root.rename(old)
        root.symlink_to(old, target_is_directory=True)
    else:
        bound = {"source": str(staged.source)}
    assert (
        e._snapshot_plan(pipeline(staged, bound), staged.overlay / "ca" / REL) is None
    )


def test_origin_factory_refuses_non_git_and_alias_roots(staged):
    assert (
        e.bind_execution_overlay_origin(
            e.capture_execution_git_origin(staged.overlay / "ca"), staged.source / "ca"
        )
        is None
    )
    alias = staged.tmp / "alias"
    alias.symlink_to(staged.source, target_is_directory=True)
    assert (
        e.bind_execution_overlay_origin(
            e.capture_execution_git_origin(alias / "ca"), staged.overlay / "ca"
        )
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "head",
        "candidate",
        "provider",
        "companion",
        "source_symlink",
        "overlay_symlink",
        "closure_symlink",
        "snapshot_symlink",
    ],
)
def test_freshness_and_consumption_reject_mutation_after_capture(staged, change):
    file = staged.overlay / "ca" / REL
    with e.compilation_snapshot(pipeline(staged, origin(staged)), file) as capture:
        assert capture is not None
        # An empty fixture artifact/record suffices to test the independent
        # origin/file freshness gate, not runtime value acceptance.
        artifact = staged.tmp / "artifact.json"
        artifact.write_text("{}")
        capture.compiled(artifact)
        capture.records.append(e.CaseExecution("fixture", "{}", "{}"))
        evidence = capture.evidence()
        assert evidence is not None and evidence.bound(file.read_text(), ())
        if change == "head":
            commit(staged.source)
        elif change in ("source_symlink", "overlay_symlink"):
            root = staged.source if change == "source_symlink" else staged.overlay
            moved = root.with_name("moved")
            root.rename(moved)
            root.symlink_to(moved, target_is_directory=True)
        elif change in ("closure_symlink", "snapshot_symlink"):
            path = (
                file
                if change == "closure_symlink"
                else capture.snapshot_root / "ca" / REL
            )
            moved = staged.tmp / "same-bytes.yaml"
            moved.write_bytes(path.read_bytes())
            path.unlink()
            path.symlink_to(moved)
        else:
            path = {
                "candidate": file,
                "provider": file.with_name("provider.yaml"),
                "companion": file.with_name("consumer.test.yaml"),
            }[change]
            path.write_text(path.read_text() + "\n# changed\n")
        assert not capture.fresh()
        assert capture.evidence() is None
        assert not evidence.bound(file.read_text(), ())


def test_escaped_closure_does_not_acquire_evidence(staged):
    provider = staged.overlay / "ca" / REL.with_name("provider.yaml")
    raw = provider.read_bytes()
    provider.unlink()
    outside = staged.tmp / "outside.yaml"
    outside.write_bytes(raw)
    provider.symlink_to(outside)
    assert (
        e._snapshot_plan(pipeline(staged, origin(staged)), staged.overlay / "ca" / REL)
        is None
    )


def test_missing_origin_preserves_no_clone_path(staged, monkeypatch):
    materialize = Mock(side_effect=AssertionError("must not clone"))
    monkeypatch.setattr(e, "_materialize_snapshot", materialize)
    with e.compilation_snapshot(
        pipeline(staged), staged.overlay / "ca" / REL
    ) as capture:
        assert capture is None
    materialize.assert_not_called()


def test_head_change_during_actual_copy_refuses_binding(staged):
    seed = e.capture_execution_git_origin(staged.source / "ca")
    assert seed is not None
    copied = staged.tmp / "during-copy" / "rulespec-ca"
    copied.parent.mkdir()
    cli._stage_apply_overlay_dependency_root(source=staged.source, target=copied)
    commit(staged.source)
    assert e.bind_execution_overlay_origin(seed, copied / "ca") is None


@pytest.mark.parametrize("stale_during_copy", [False, True])
def test_evaluate_artifact_forwards_pre_copy_origin_to_real_pipeline(
    staged, monkeypatch, stale_during_copy
):
    generated = staged.tmp / "generated" / REL
    generated.parent.mkdir(parents=True)
    generated.write_text(staged.target.read_text() + "\n# evaluated candidate\n")
    generated.with_name("consumer.test.yaml").write_text("[]\n")
    original_copy = evals._copy_validation_overlay_tree

    def copy_then_change(*args, **kwargs):
        original_copy(*args, **kwargs)
        if stale_during_copy:
            commit(staged.source)

    monkeypatch.setattr(evals, "_copy_validation_overlay_tree", copy_then_change)
    observed = []

    def check(pipeline, rules_file):
        # Actual pipeline, actual ordinary staging and actual snapshot clone.
        # Only compiler execution is replaced; no runtime pass is asserted.
        assert pipeline.policy_repo_path != staged.source / "ca"
        assert (pipeline.policy_repo_path / "sibling.txt").is_file()
        with e.compilation_snapshot(pipeline, rules_file) as capture:
            observed.append(capture is not None)
            assert (capture is not None) is (not stale_during_copy)
            if capture is not None:
                assert (
                    capture.snapshot_root / capture.root_relative
                ).read_bytes() == generated.read_bytes()
        return v.ValidationResult("compile", passed=True)

    monkeypatch.setattr(v.ValidatorPipeline, "_run_compile_check", check)
    monkeypatch.setattr(
        v.ValidatorPipeline, "_run_ci", lambda *a: v.ValidationResult("ci", passed=True)
    )
    evals.evaluate_artifact(
        rulespec_file=generated,
        policy_repo_root=staged.source / "ca",
        axiom_rules_path=staged.tmp / "engine",
        local_corpus_release=_write_test_corpus_release(
            staged.tmp, [{"citation_path": CITATION, "body": BODY}]
        ),
        source_text=BODY,
        source_citation_path=CITATION,
        skip_reviewers=True,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=PIN,
    )
    assert observed == [not stale_during_copy]


@pytest.mark.parametrize("stale_during_copy", [False, True])
def test_apply_overlay_forwards_pre_copy_origin_to_real_pipeline(
    staged, monkeypatch, stale_during_copy
):
    from tests.test_cli import (
        _complete_source_attestation,
        _write_test_context_manifest,
    )

    generated = staged.tmp / "output" / "reviewed-candidate" / REL
    generated.parent.mkdir(parents=True)
    payload = yaml.safe_load(staged.target.read_text())
    sha = e.digest(BODY.encode())
    payload["module"]["source_verification"] = {
        "corpus_citation_path": CITATION,
        "source_sha256": sha,
    }
    generated.write_text(yaml.safe_dump(payload))
    generated.with_name("consumer.test.yaml").write_text("[]\n")
    context = _write_test_context_manifest(
        staged.tmp / "context", citation_path=CITATION, source_sha256=sha
    )
    result = SimpleNamespace(
        output_file=str(generated),
        runner="reviewed-candidate",
        backend="openai",
        model="reviewed-candidate-promotion-v1",
        tool=cli.APPLIED_ENCODING_REVIEWED_CANDIDATE_TOOL,
        citation=CITATION,
        generation_prompt_sha256=None,
        trace_file=None,
        context_manifest_file=str(context),
        source_attestation=_complete_source_attestation(CITATION, source_sha256=sha),
    )
    setattr(result, cli._REVIEWED_CANDIDATE_PROMOTION_ATTR, True)
    setattr(
        result, cli._IMMUTABLE_RULESPEC_SHA256_ATTR, e.digest(generated.read_bytes())
    )
    original_copy = cli._stage_apply_overlay_dependency_root

    def copy_then_change(*args, **kwargs):
        original_copy(*args, **kwargs)
        if stale_during_copy:
            commit(staged.source)

    monkeypatch.setattr(cli, "_stage_apply_overlay_dependency_root", copy_then_change)
    observed = []

    def validate(pipeline, *, overlay_target, **kwargs):
        assert isinstance(pipeline, v.ValidatorPipeline)
        assert (pipeline.policy_repo_path / "sibling.txt").is_file()
        with e.compilation_snapshot(pipeline, overlay_target) as capture:
            observed.append(capture is not None)
            assert (capture is not None) is (not stale_during_copy)
            if capture is not None:
                assert (
                    capture.snapshot_root / capture.root_relative
                ).read_bytes() == generated.read_bytes()
        return [(overlay_target, SimpleNamespace(all_passed=True, results={}))]

    monkeypatch.setattr(cli, "_validate_overlay_files", validate)
    # Isolate this staging test from signing/provenance transport. Existing
    # engine-binding suites exercise those separately; this test signs nothing.
    monkeypatch.setattr(cli, "_pinned_apply_engine_binding", lambda *a: None)
    monkeypatch.setattr(
        cli, "_record_successful_apply_validation", lambda *a, **k: None
    )
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        result,
        output_root=staged.tmp / "output",
        policy_repo_path=staged.source / "ca",
        axiom_rules_path=staged.tmp / "engine",
        axiom_rules_engine_ref=PIN,
        local_corpus_release=_write_test_corpus_release(
            staged.tmp, [{"citation_path": CITATION, "body": BODY}]
        ),
        validate_dependents=False,
        require_complete_source_unit=True,
    )
    assert ok, issues
    assert observed == [not stale_during_copy]


def test_ordinary_canonical_checkout_and_git_worktree_remain_eligible(staged):
    ordinary = v.ValidatorPipeline(
        staged.source / "ca",
        staged.tmp / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=PIN,
    )
    assert e._snapshot_plan(ordinary, staged.target) is not None
    linked = staged.tmp / "linked" / "rulespec-ca"
    linked.parent.mkdir()
    git(staged.source, "worktree", "add", "--detach", str(linked), "HEAD")
    linked_pipe = v.ValidatorPipeline(
        linked / "ca",
        staged.tmp / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
        axiom_rules_engine_ref=PIN,
    )
    with e.compilation_snapshot(linked_pipe, linked / "ca" / REL) as capture:
        assert capture is not None and capture.fresh()

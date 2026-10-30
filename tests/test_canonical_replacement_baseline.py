"""Preservation-context binding; these fixtures do not claim signature admission."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness.evals import _format_validation_retry_candidate


def bound_pair(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    target = tmp_path / "us/policies/example.yaml"
    target.parent.mkdir(parents=True)
    companion = target.with_suffix(".test.yaml")
    source = SimpleNamespace(
        body="Actual source",
        requested="us/statute/example",
        citation_path="us/statute/example",
    )
    digest = hashlib.sha256(source.body.encode()).hexdigest()
    target.write_text(
        yaml.safe_dump(
            {
                "module": {
                    "source_verification": {
                        "source_sha256": digest,
                        "corpus_citation_path": source.requested,
                    }
                },
                "rules": [],
            }
        )
    )
    companion.write_text("[]\n")
    manifest = (
        tmp_path
        / cli.APPLIED_ENCODING_MANIFEST_DIR
        / target.relative_to(tmp_path).with_suffix(".json")
    )
    manifest.parent.mkdir(parents=True)
    data = {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "applied_files": [
            {
                "path": p.relative_to(tmp_path).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in (target, companion)
        ],
        "source_attestation": {
            "requested_corpus_citation_path": source.requested,
            "resolved_corpus_citation_path": source.citation_path,
            **{
                k: digest
                for k in (
                    "source_sha256",
                    "resolved_text_sha256",
                    "generation_input_sha256",
                )
            },
        },
    }
    manifest.write_text(json.dumps(data))
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return target, companion, manifest, source


def test_exact_source_pair_and_changed_source_lane(tmp_path):
    target, companion, _, source = bound_pair(tmp_path)
    baseline = cli._canonical_replacement_baseline(
        tmp_path, target, companion, source_unit=source
    )
    assert baseline.context_role == "replacement_baseline"
    assert baseline.rulespec == target.read_text()
    source.body = "Changed authority"
    assert (
        cli._canonical_replacement_baseline(
            tmp_path, target, companion, source_unit=source
        )
        is None
    )


@pytest.mark.parametrize("which", ["policy", "tests", "manifest"])
def test_dirty_pair_refuses(tmp_path, which):
    target, companion, manifest, source = bound_pair(tmp_path)
    path = {"policy": target, "tests": companion, "manifest": manifest}[which]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        cli._canonical_replacement_baseline(
            tmp_path, target, companion, source_unit=source
        )


def test_full_prompt_is_truthful_and_not_truncated(tmp_path):
    target, companion, _, source = bound_pair(tmp_path)
    baseline = cli._canonical_replacement_baseline(
        tmp_path, target, companion, source_unit=source
    )
    from dataclasses import replace

    baseline = replace(
        baseline, rulespec=baseline.rulespec + "# " + "x" * 10000 + "\n# final marker"
    )
    prompt = _format_validation_retry_candidate(
        baseline, target_file_name="example.yaml"
    )
    assert baseline.rulespec in prompt
    assert "# final marker" in prompt
    assert "UNTRUSTED REPLACEMENT BASELINE" in prompt
    assert "...[truncated]" not in prompt
    with pytest.raises(ValueError):
        _format_validation_retry_candidate(
            baseline, target_file_name="example.yaml", tests_only=True
        )


@pytest.mark.parametrize("which", ["policy", "tests", "manifest"])
def test_symlink_pair_refuses(tmp_path, which):
    target, companion, manifest, source = bound_pair(tmp_path)
    path = {"policy": target, "tests": companion, "manifest": manifest}[which]
    copy = tmp_path / "outside-copy"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(copy)
    with pytest.raises(ValueError):
        cli._canonical_replacement_baseline(
            tmp_path, target, companion, source_unit=source
        )


def test_baseline_cannot_take_valid_candidate_shortcut(tmp_path):
    from unittest.mock import patch

    from axiom_encode.harness.evals import (
        ValidationRetryCandidate,
        _run_single_eval,
        parse_runner_spec,
        resolve_corpus_source_unit,
    )
    from tests.test_repair_overlay_failure_is_retryable import (
        _bind_corpus,
        _make_workspace,
    )

    release = _bind_corpus(tmp_path)
    source = resolve_corpus_source_unit("us-ca/regulation/mpp/63-503", release)
    workspace = _make_workspace(tmp_path / "workspace")
    candidate = ValidationRetryCandidate(
        "format: rulespec/v1\nrules: []\n", "[]\n", context_role="replacement_baseline"
    )
    with (
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace", return_value=workspace
        ),
        patch("axiom_encode.harness.evals._run_prompt_eval") as model,
    ):
        with pytest.raises(
            ValueError, match="cannot use retained-candidate acceptance"
        ):
            _run_single_eval(
                citation=source.requested,
                runner=parse_runner_spec("codex:gpt-5.5"),
                output_root=tmp_path / "out",
                policy_path=tmp_path,
                runtime_axiom_rules_path=tmp_path,
                corpus_release=release,
                mode="cold",
                extra_context_paths=[],
                source_unit=source,
                validation_retry_candidate=candidate,
                accept_valid_retry_candidate=True,
            )
    model.assert_not_called()


def test_named_edit_and_second_partial_keep_original_objects(tmp_path):
    from axiom_encode.harness.evals import (
        ValidationRetryCandidate,
        _overlay_validation_retry_candidate,
    )

    original_rule = {
        "name": "untouched",
        "kind": "derived",
        "metadata": {"proof": {"atoms": ["original"]}},
        "versions": [{"effective_from": "2026-01-01", "formula": 7}],
    }
    original_case = {
        "name": "old",
        "period": "2026",
        "input": {"fact": False},
        "output": {"value": False},
    }
    baseline = ValidationRetryCandidate(
        yaml.safe_dump({"rules": [original_rule, {"name": "changed", "formula": 1}]}),
        yaml.safe_dump([original_case]),
        context_role="replacement_baseline",
    )
    target = tmp_path / "target.yaml"
    test = target.with_suffix(".test.yaml")
    target.write_text(yaml.safe_dump({"rules": [{"name": "changed", "formula": 2}]}))
    test.write_text(yaml.safe_dump([{"name": "new", "input": {}, "output": {}}]))
    _overlay_validation_retry_candidate(
        target, artifact_root=tmp_path, candidate=baseline
    )
    first = yaml.safe_load(target.read_text())
    assert next(r for r in first["rules"] if r["name"] == "untouched") == original_rule
    assert next(r for r in first["rules"] if r["name"] == "changed")["formula"] == 2
    assert original_case in yaml.safe_load(test.read_text())
    # Capture occurs after overlay even if ordinary compile/tests reject it.
    captured = ValidationRetryCandidate(target.read_text(), test.read_text())
    target.write_text("rules: []\n")
    test.write_text("[]\n")
    _overlay_validation_retry_candidate(
        target, artifact_root=tmp_path, candidate=captured
    )
    assert yaml.safe_load(target.read_text()) == first
    assert original_case in yaml.safe_load(test.read_text())


def test_real_eval_partial_rejection_capture_then_retry_preserves_baseline(tmp_path):
    from unittest.mock import patch

    from axiom_encode.harness.evals import (
        EvalPromptResponse,
        ValidationRetryCandidate,
        _run_single_eval,
        parse_runner_spec,
        resolve_corpus_source_unit,
    )
    from tests.test_repair_overlay_failure_is_retryable import (
        _bind_corpus,
        _make_workspace,
        _metrics,
    )

    release = _bind_corpus(tmp_path)
    source = resolve_corpus_source_unit("us-ca/regulation/mpp/63-503", release)
    output = tmp_path / "out"
    workspace = _make_workspace(output / "_eval_workspaces" / "workspace")
    baseline = ValidationRetryCandidate(
        'format: rulespec/v1\nrules:\n- name: preserved\n  kind: parameter\n  dtype: Count\n  versions: [{effective_from: "2026-01-01", formula: 1}]\n',
        "- name: historical\n  input: {}\n  output: {}\n",
        context_role="replacement_baseline",
    )
    observed = []

    def reject(**kwargs):
        path = kwargs["rulespec_file"]
        observed.append(yaml.safe_load(path.read_text()))
        return _metrics(passed=False)

    with (
        patch(
            "axiom_encode.harness.evals.prepare_eval_workspace", return_value=workspace
        ),
        patch(
            "axiom_encode.harness.evals._run_prompt_eval",
            return_value=EvalPromptResponse(
                text="format: rulespec/v1\nrules: []\n", duration_ms=1
            ),
        ) as model,
        patch("axiom_encode.harness.evals._hydrate_eval_root"),
        patch(
            "axiom_encode.harness.evals._evaluate_generated_artifact_with_repairs",
            side_effect=reject,
        ),
    ):
        for _ in range(2):
            result = _run_single_eval(
                citation=source.requested,
                runner=parse_runner_spec("codex:gpt-5.5"),
                output_root=output,
                policy_path=tmp_path,
                runtime_axiom_rules_path=tmp_path,
                corpus_release=release,
                mode="cold",
                extra_context_paths=[],
                source_unit=source,
                validation_retry_candidate=baseline,
            )
            assert not result.success
            baseline = cli._capture_validation_retry_candidate(
                result, output_root=output
            )
            assert yaml.safe_load(baseline.rulespec)["rules"][0]["name"] == "preserved"
            assert yaml.safe_load(baseline.tests)[0]["name"] == "historical"
    assert model.call_count == 2
    assert len(observed) == 2
    assert all(item["rules"][0]["name"] == "preserved" for item in observed)


def test_no_usable_candidate_retry_remains_fail_closed():
    from unittest.mock import patch

    with patch("axiom_encode.cli._best_validation_retry_attempt", return_value=None):
        with pytest.raises(RuntimeError, match="candidate is unavailable"):
            cli._encode_validation_retry_context(
                [object()], initial_retry_candidate=None
            )


def test_existing_candidate_size_limit_is_not_silently_truncated(tmp_path):
    from axiom_encode.harness.evals import ValidationRetryCandidate
    from axiom_encode.repair_candidate_contract import (
        VALIDATION_RETRY_CANDIDATE_MAX_FILE_BYTES,
    )

    with pytest.raises(ValueError):
        ValidationRetryCandidate(
            "x" * (VALIDATION_RETRY_CANDIDATE_MAX_FILE_BYTES + 1),
            context_role="replacement_baseline",
        )


def test_duplicate_inline_target_removed_but_other_context_retained(tmp_path):
    from axiom_encode.harness.evals import (
        EvalContextFile,
        ValidationRetryCandidate,
        _build_eval_prompt,
    )
    from tests.test_repair_overlay_failure_is_retryable import _make_workspace

    workspace = _make_workspace(tmp_path / "workspace")
    baseline = ValidationRetryCandidate(
        "rules: []\n# UNIQUE_FULL_BASELINE\n",
        "[]\n",
        context_role="replacement_baseline",
    )
    files = []
    for name, content in [
        ("target.yaml", baseline.rulespec),
        ("target.test.yaml", baseline.tests),
        ("other.yaml", "rules: []\n# OTHER_PRECEDENT\n"),
    ]:
        path = workspace.root / name
        path.write_text(content)
        files.append(
            EvalContextFile(
                str(path), name, "us:statutes/" + name.removesuffix(".yaml"), "rulespec"
            )
        )
    prompt = _build_eval_prompt(
        "us/statute/example",
        "repo-augmented",
        workspace,
        files,
        target_file_name="target.yaml",
        include_tests=True,
        runner_backend="openai",
        validation_retry_candidate=baseline,
    )
    assert prompt.count("UNIQUE_FULL_BASELINE") == 1
    assert prompt.count("=== FILE: target.yaml ===") == 1  # output template only
    assert prompt.count("=== FILE: target.test.yaml ===") == 1
    assert "OTHER_PRECEDENT" in prompt

"""Offline admission command parsing and exit status contracts."""

import argparse
import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from axiom_encode.harness.admission_cli import (
    register_admission_score_parser,
    run_admission_score,
)


def _args(tmp_path):
    (tmp_path / "candidate.yaml").write_text("fixture candidate")
    return argparse.Namespace(
        candidate=tmp_path / "candidate.yaml",
        citation="us/statute/1/1",
        tests=tmp_path / "candidate.test.yaml",
        policy_repo_path=tmp_path / "rulespec-us",
        corpus_path=tmp_path / "axiom-corpus",
        corpus_release_public_key="verification-only-public-key",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        axiom_compose_path=None,
        rulespec_dependency_roots=[],
        expected_source_body_sha256=None,
        expected_policy_repo_commit=None,
        expected_dependency_content_sha256=None,
        expected_corpus_release_content_sha256=None,
    )


@pytest.mark.parametrize(
    "admitted,prerequisite,status",
    [(True, False, 0), (False, False, 1), (False, True, 2)],
)
@pytest.mark.parametrize("candidate_kind", ["file", "broken-symlink", "directory"])
@pytest.mark.parametrize("frozen_digests", [False, True])
def test_admission_cli_emits_json_and_exit_status(
    tmp_path,
    monkeypatch,
    capsys,
    admitted,
    prerequisite,
    status,
    candidate_kind,
    frozen_digests,
):
    import axiom_encode.corpus_resolver as corpus_resolver
    import axiom_encode.harness.admission as admission
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    expected_context = None
    if frozen_digests:
        args.expected_source_body_sha256 = "a" * 64
        args.expected_policy_repo_commit = "b" * 40
        args.expected_dependency_content_sha256 = ["c" * 64, "d" * 64]
        args.expected_corpus_release_content_sha256 = "e" * 64
        expected_context = {
            "source_body_sha256": args.expected_source_body_sha256,
            "policy_repo_commit": args.expected_policy_repo_commit,
            "dependency_content_sha256": args.expected_dependency_content_sha256,
            "corpus_release_content_sha256": (
                args.expected_corpus_release_content_sha256
            ),
        }
    if candidate_kind != "file":
        args.candidate.unlink()
        if candidate_kind == "broken-symlink":
            args.candidate.symlink_to(tmp_path / "missing.yaml")
        else:
            args.candidate.mkdir()
    release = object()
    payload = {"admitted": admitted, "prerequisite_failure": prerequisite}
    captured = {}

    def score(candidate, **kwargs):
        captured.update(kwargs)
        assert candidate.output_file == str(args.candidate)
        assert candidate.backend == ""
        context_path = Path(candidate.context_manifest_file)
        context = json.loads(context_path.read_text())
        generation_input = (
            context_path.parent / context["source_text_file"]
        ).read_bytes()
        assert generation_input == b"Authoritative body\nwith normalized\nnewlines\n"
        generation_input_sha256 = hashlib.sha256(generation_input).hexdigest()
        assert context["source_text_sha256"] == generation_input_sha256
        assert (
            candidate.context_manifest_sha256
            == hashlib.sha256(context_path.read_bytes()).hexdigest()
        )
        assert context["source_metadata"]["source_attestation"] == {
            "requested_corpus_citation_path": args.citation,
            "generation_input_sha256": generation_input_sha256,
        }
        assert (
            candidate.source_attestation
            == context["source_metadata"]["source_attestation"]
        )
        return SimpleNamespace(admitted=admitted, to_dict=lambda: payload)

    monkeypatch.setattr(
        toolchain, "local_corpus_release_verification", lambda _: nullcontext()
    )
    monkeypatch.setattr(
        toolchain, "load_rulespec_local_corpus_release", lambda *_: release
    )
    monkeypatch.setattr(admission, "score_admission", score)
    monkeypatch.setattr(
        corpus_resolver,
        "resolve_local_corpus_source",
        lambda *_: SimpleNamespace(
            body="Authoritative body\r\nwith normalized\rnewlines\n",
            to_attestation=lambda: {
                "requested_corpus_citation_path": args.citation,
            },
        ),
    )

    assert run_admission_score(args) == status
    assert json.loads(capsys.readouterr().out) == payload
    assert captured["local_corpus_release"] is release
    assert captured["companion_test_path"] == args.tests
    assert captured["policy_repo_path"] == args.policy_repo_path
    assert captured["expected_context"] == expected_context


def test_admission_cli_nonexistent_candidate_is_usage_error(
    tmp_path, monkeypatch, capsys
):
    import axiom_encode.harness.admission as admission
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    args.candidate.unlink()

    def unexpected(*_args, **_kwargs):
        pytest.fail("A missing candidate must fail before loading context or scoring")

    monkeypatch.setattr(toolchain, "load_rulespec_local_corpus_release", unexpected)
    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "candidate path does not exist" in captured.err


def test_admission_cli_unverifiable_release_is_prerequisite(
    tmp_path, monkeypatch, capsys
):
    import axiom_encode.toolchain as toolchain

    def unverifiable(_):
        raise ValueError("Corpus release signature is invalid")

    monkeypatch.setattr(toolchain, "local_corpus_release_verification", unverifiable)
    assert run_admission_score(_args(tmp_path)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] is False
    assert payload["prerequisite_failure"] is True
    assert payload["issues"] == ["Corpus release signature is invalid"]


def test_admission_cli_parser_requires_packaged_context():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_admission_score_parser(subparsers)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["admission-score", "candidate.yaml"])
    assert exc.value.code == 2
    args = parser.parse_args(
        [
            "admission-score",
            "candidate.yaml",
            "--citation",
            "us/statute/1/1",
            "--policy-repo",
            "rulespec-us",
            "--corpus-path",
            "axiom-corpus",
            "--axiom-rules-engine-path",
            "axiom-rules-engine",
            "--corpus-release-public-key",
            "public-key",
        ]
    )
    assert args.candidate == Path("candidate.yaml")
    assert args.policy_repo_path == Path("rulespec-us")
    assert args.expected_source_body_sha256 is None
    assert args.expected_policy_repo_commit is None
    assert args.expected_dependency_content_sha256 is None
    assert args.expected_corpus_release_content_sha256 is None


def test_admission_cli_parser_accepts_frozen_context_digests():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_admission_score_parser(subparsers)
    args = parser.parse_args(
        [
            "admission-score",
            "candidate.yaml",
            "--citation",
            "us/statute/1/1",
            "--policy-repo",
            "rulespec-us",
            "--corpus-path",
            "axiom-corpus",
            "--axiom-rules-engine-path",
            "axiom-rules-engine",
            "--corpus-release-public-key",
            "public-key",
            "--expected-source-body-sha256",
            "a" * 64,
            "--expected-policy-repo-commit",
            "b" * 40,
            "--expected-dependency-content-sha256",
            "c" * 64,
            "--expected-dependency-content-sha256",
            "d" * 64,
            "--expected-corpus-release-content-sha256",
            "e" * 64,
        ]
    )
    assert args.expected_source_body_sha256 == "a" * 64
    assert args.expected_policy_repo_commit == "b" * 40
    assert args.expected_dependency_content_sha256 == ["c" * 64, "d" * 64]
    assert args.expected_corpus_release_content_sha256 == "e" * 64


def test_eval_suite_admission_flag_is_registered(monkeypatch):
    from axiom_encode import cli

    captured = []
    monkeypatch.setattr(cli, "cmd_eval_suite", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "axiom-encode",
            "eval-suite",
            "suite.yaml",
            "--admission-score",
            "--policy-repo-path",
            "rulespec-us",
            "--corpus-path",
            "axiom-corpus",
            "--axiom-rules-engine-path",
            "axiom-rules-engine",
        ],
    )
    cli.main()
    assert captured[0].admission_score is True

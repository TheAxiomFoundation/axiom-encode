"""Offline admission command parsing and exit status contracts."""

import argparse
import hashlib
import json
import os
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
        backend="codex",
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
    import axiom_encode.harness.admission as admission
    import axiom_encode.harness.evals as evals
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
        assert candidate.backend == args.backend
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
        evals,
        "resolve_corpus_source_unit",
        lambda *_: SimpleNamespace(
            amendment_documents=(),
            resolved_source=SimpleNamespace(
                body="Authoritative body\r\nwith normalized\rnewlines\n",
                to_attestation=lambda: {
                    "requested_corpus_citation_path": args.citation,
                },
            ),
        ),
    )

    if candidate_kind != "file":
        # The shared guard refuses a non-regular candidate before scoring.
        assert run_admission_score(args) == 1
        refused = json.loads(capsys.readouterr().out)
        assert refused["admitted"] is False
        assert refused["failure_kind"] == "candidate"
        assert refused["prerequisite_failure"] is False
        assert "generated artifact must be a regular file" in refused["issues"][0]
        assert captured == {}
        return
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


def test_admission_cli_loads_the_release_without_remote_sources(
    tmp_path, monkeypatch, capsys
):
    """A lock-file checkout must not reach R2 from the scoring command."""

    import axiom_encode.toolchain as toolchain
    from axiom_encode import corpus_materialize
    from axiom_encode.corpus_resolver import InvalidCorpusReleaseError

    monkeypatch.setattr(
        corpus_materialize,
        "r2_credentials_from_environment",
        lambda _environ: object(),
    )
    monkeypatch.setattr(
        corpus_materialize, "R2ObjectSource", lambda *_args: "remote-r2-source"
    )
    observed = []

    def load_release(_policy_repo, corpus_path):
        corpus_path.mkdir(parents=True, exist_ok=True)
        with corpus_materialize.release_sources(
            corpus_path, git_commit="e" * 40, r2_bucket="axiom-corpus"
        ) as sources:
            observed.append(sources)
        raise InvalidCorpusReleaseError(
            "Corpus release provisions could not be placed from local sources"
        )

    monkeypatch.setattr(toolchain, "load_rulespec_local_corpus_release", load_release)
    monkeypatch.setattr(
        toolchain,
        "local_corpus_release_verification",
        lambda _key: nullcontext(),
    )
    assert run_admission_score(_args(tmp_path)) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["prerequisite_failure"] is True
    assert observed and "remote-r2-source" not in observed[0]
    # Outside the scoring command the same configuration does add R2.
    with corpus_materialize.release_sources(
        tmp_path / "axiom-corpus", git_commit="e" * 40, r2_bucket="axiom-corpus"
    ) as sources:
        assert "remote-r2-source" in sources


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
            "--backend",
            "codex",
        ]
    )
    assert args.candidate == Path("candidate.yaml")
    assert args.backend == "codex"
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
            "--backend",
            "claude",
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


def test_admission_cli_backends_are_the_ones_apply_stamps_for():
    from axiom_encode import cli
    from axiom_encode.harness.admission_cli import MODEL_BACKENDS

    assert frozenset(MODEL_BACKENDS) == cli.APPLIED_ENCODING_ENCODER_BACKENDS


def test_admission_cli_requires_the_generating_backend():
    """Apply stamps model output; a backend-less score would skip that."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_admission_score_parser(subparsers)
    base = [
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
    for extra in ([], ["--backend", ""], ["--backend", "fixture"]):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([*base, *extra])
        assert exc.value.code == 2


def test_admission_cli_companion_elsewhere_is_usage_error(
    tmp_path, monkeypatch, capsys
):
    import axiom_encode.harness.admission as admission
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    (tmp_path / "elsewhere").mkdir()
    args.tests = tmp_path / "elsewhere" / "candidate.test.yaml"
    args.tests.write_text("tests")

    def unexpected(*_args, **_kwargs):
        pytest.fail("A misplaced companion must fail before loading or scoring")

    monkeypatch.setattr(toolchain, "load_rulespec_local_corpus_release", unexpected)
    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "companion test file" in captured.err


def test_admission_cli_amended_source_is_a_prerequisite(tmp_path, monkeypatch, capsys):
    """Without generation context the command cannot check commencement."""

    import axiom_encode.harness.admission as admission
    import axiom_encode.harness.evals as evals
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    monkeypatch.setattr(
        toolchain, "local_corpus_release_verification", lambda _: nullcontext()
    )
    monkeypatch.setattr(
        toolchain, "load_rulespec_local_corpus_release", lambda *_: object()
    )
    monkeypatch.setattr(
        evals,
        "resolve_corpus_source_unit",
        lambda *_: SimpleNamespace(
            amendment_documents=(SimpleNamespace(citation_path="us/statute/9"),),
            resolved_source=None,
        ),
    )

    def unexpected(*_args, **_kwargs):
        pytest.fail("An amended source must not be scored without its context")

    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] is False
    assert payload["prerequisite_failure"] is True
    assert payload["prerequisite_categories"] == ["missing-amendment-context"]


@pytest.mark.parametrize("failure", ["amended-source", "release-load"])
def test_admission_cli_refuses_a_non_regular_candidate_before_any_prerequisite(
    tmp_path, monkeypatch, capsys, failure
):
    """A symlinked candidate is the candidate's refusal, whatever else fails.

    Both the amended-source prerequisite and a release that fails to load
    return before ``score_admission``; the shared guard runs first.
    """

    import axiom_encode.harness.admission as admission
    import axiom_encode.harness.evals as evals
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    target = tmp_path / "real-candidate.yaml"
    args.candidate.rename(target)
    args.candidate.symlink_to(target)
    monkeypatch.setattr(
        toolchain, "local_corpus_release_verification", lambda _: nullcontext()
    )
    if failure == "release-load":

        def unverifiable(*_args):
            raise ValueError("Corpus release signature is invalid")

        monkeypatch.setattr(
            toolchain, "load_rulespec_local_corpus_release", unverifiable
        )
    else:
        monkeypatch.setattr(
            toolchain, "load_rulespec_local_corpus_release", lambda *_: object()
        )
        monkeypatch.setattr(
            evals,
            "resolve_corpus_source_unit",
            lambda *_: SimpleNamespace(
                amendment_documents=(SimpleNamespace(citation_path="us/statute/9"),),
                resolved_source=None,
            ),
        )

    def unexpected(*_args, **_kwargs):
        pytest.fail("A refused artifact must not be scored")

    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] is False
    assert payload["failure_kind"] == "candidate"
    assert payload["prerequisite_failure"] is False
    assert payload["prerequisite_categories"] == []
    assert "generated artifact must be a regular file" in payload["issues"][0]


def test_admission_cli_unreadable_policy_identity_is_a_prerequisite(
    tmp_path, monkeypatch, capsys
):
    """A policy checkout the identity cannot digest is a prerequisite, exit 2.

    A FIFO in the policy checkout makes the policy identity refuse to read
    it. The command must report that as the ``identity-unavailable``
    prerequisite, as ``score_admission`` does, not let the error escape and
    exit 1, the code for a refusal.
    """

    import axiom_encode.harness.admission as admission
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    args.policy_repo_path.mkdir(parents=True)
    os.mkfifo(args.policy_repo_path / "blocked.fifo")

    def unexpected(*_args, **_kwargs):
        pytest.fail("An unidentifiable policy checkout must not be scored")

    monkeypatch.setattr(toolchain, "load_rulespec_local_corpus_release", unexpected)
    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["admitted"] is False
    assert payload["prerequisite_failure"] is True
    assert payload["prerequisite_categories"] == ["identity-unavailable"]
    assert "not a regular file" in payload["issues"][0]


def test_admission_cli_non_regular_candidate_outranks_an_unreadable_policy(
    tmp_path, capsys
):
    """The artifact refusal still wins when the identity cannot be built."""

    args = _args(tmp_path)
    args.policy_repo_path.mkdir(parents=True)
    os.mkfifo(args.policy_repo_path / "blocked.fifo")
    target = tmp_path / "real-candidate.yaml"
    args.candidate.rename(target)
    args.candidate.symlink_to(target)

    assert run_admission_score(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["failure_kind"] == "candidate"
    assert payload["prerequisite_failure"] is False
    assert "generated artifact must be a regular file" in payload["issues"][0]


@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
@pytest.mark.parametrize("later", ["amended-source", "release-load"])
def test_admission_cli_later_identity_failures_fall_back_to_an_empty_identity(
    tmp_path, monkeypatch, capsys, later, error
):
    """An identity that fails after the first build does not crash the command.

    The first identity build succeeds; every later one raises, as when the
    checkout changes mid-run. The amended-source prerequisite and the
    release-load prerequisite then carry an empty identity, and the command
    still exits 2, its code for a prerequisite.
    """

    import axiom_encode.harness.admission as admission
    import axiom_encode.harness.evals as evals
    import axiom_encode.toolchain as toolchain

    args = _args(tmp_path)
    calls = []

    def identity(**_kwargs):
        calls.append(_kwargs)
        if len(calls) > 1:
            raise error("Policy context artifact is not a regular file")
        return {"policy_repo": {"commit": "f" * 40}}

    monkeypatch.setattr(admission, "admission_identity", identity)
    monkeypatch.setattr(
        toolchain, "local_corpus_release_verification", lambda _: nullcontext()
    )
    if later == "release-load":

        def unverifiable(*_args):
            raise ValueError("Corpus release signature is invalid")

        monkeypatch.setattr(
            toolchain, "load_rulespec_local_corpus_release", unverifiable
        )
        expected_category = "infrastructure"
    else:
        monkeypatch.setattr(
            toolchain, "load_rulespec_local_corpus_release", lambda *_: object()
        )
        monkeypatch.setattr(
            evals,
            "resolve_corpus_source_unit",
            lambda *_: SimpleNamespace(
                amendment_documents=(SimpleNamespace(citation_path="us/statute/9"),),
                resolved_source=None,
            ),
        )
        expected_category = "missing-amendment-context"

    def unexpected(*_args, **_kwargs):
        pytest.fail("A prerequisite failure must not be scored")

    monkeypatch.setattr(admission, "score_admission", unexpected)
    assert run_admission_score(args) == 2
    payload = json.loads(capsys.readouterr().out)
    assert len(calls) == 2
    assert payload["prerequisite_failure"] is True
    assert payload["prerequisite_categories"] == [expected_category]
    assert payload["identity"] == {}

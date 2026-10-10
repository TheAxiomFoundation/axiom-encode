"""Differential checks for the signed apply admission boundary."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.corpus_release import (
    VerifiedCorpusReleaseObject,
    VerifiedReleaseArtifact,
    VerifiedReleaseScope,
)
from axiom_encode.corpus_resolver import (
    LocalCorpusRelease,
    resolve_local_corpus_source,
)
from axiom_encode.harness.admission import (
    COMPLETE_SOURCE_UNIT_CATEGORIES,
    admission_identity,
    issue_category,
    score_admission,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline

_RECORD_SUCCESSFUL_APPLY_VALIDATION = cli._record_successful_apply_validation


def _hashes(root: Path) -> dict[str, str]:
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
    )
    return {
        name.decode(): hashlib.sha256((root / name.decode()).read_bytes()).hexdigest()
        for name in listing.stdout.split(b"\0")
        if name
    }


@pytest.fixture
def context(tmp_path, monkeypatch):
    """Use a real release resolver with a test-only verification boundary."""

    def make(
        source="The allowance is $40.",
        *,
        missing_tests=False,
        compile_failure=False,
        backend="",
    ):
        policy = tmp_path / "rulespec-us" / "us"
        policy.mkdir(parents=True, exist_ok=True)
        corpus = tmp_path / "axiom-corpus"
        provisions = corpus / "data/corpus/provisions/us/statute/fixture.jsonl"
        provisions.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            json.dumps(
                {
                    "id": "fixture",
                    "citation_path": "us/statute/26/1",
                    "body": source,
                    "jurisdiction": "us",
                    "document_class": "statute",
                    "version": "fixture",
                    "source_path": "sources/fixture",
                    "source_as_of": "2026-01-01",
                    "expression_date": "2026-01-01",
                }
            )
            + "\n"
        )
        provisions.write_text(raw)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        verified = VerifiedCorpusReleaseObject(
            "admission-fixture",
            "a" * 64,
            "b" * 64,
            (VerifiedReleaseScope("us", "statute", "fixture", 1, "c" * 64, "d" * 64),),
            (
                VerifiedReleaseArtifact(
                    "provisions",
                    provisions.relative_to(corpus).as_posix(),
                    digest,
                    len(raw.encode()),
                    1,
                ),
            ),
        )
        release_object = corpus / "releases/admission-fixture" / ("a" * 64 + ".json")
        release_object.parent.mkdir(parents=True, exist_ok=True)
        release_object.write_text("{}")
        monkeypatch.setattr(
            "axiom_encode.corpus_resolver.verify_release_object",
            lambda *_args, **_kwargs: verified,
        )
        release = LocalCorpusRelease(
            corpus, verified.name, verified.content_sha256, "fixture-public-key"
        )
        source_digest = hashlib.sha256(source.encode()).hexdigest()
        attestation = {
            **(
                resolve_local_corpus_source("us/statute/26/1", release).to_attestation()
                if source.strip()
                else {}
            ),
            "generation_input_sha256": source_digest,
            "source_as_of": "2026-01-01",
            "expression_date": "2026-01-01",
        }
        context_root = tmp_path / "context"
        context_root.mkdir(exist_ok=True)
        source_file = context_root / "source.txt"
        source_file.write_text(source)
        manifest = context_root / "context-manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source_text_file": "source.txt",
                    "source_text_sha256": source_digest,
                    "source_metadata": {"source_attestation": attestation},
                }
            )
        )
        engine = tmp_path / "axiom-rules-engine"
        engine.mkdir(exist_ok=True)
        binary = engine / "axiom-rules-engine"
        binary.write_text("fixture engine")
        binary.chmod(0o755)
        output_root = tmp_path / "out"
        candidate = output_root / "fixture-runner/statutes/26/1.yaml"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rulespec/v1",
            "module": {
                "source_verification": {"corpus_citation_path": "us/statute/26/1"}
            },
            "rules": [
                {
                    "name": "allowance",
                    "kind": "parameter",
                    "dtype": "Money",
                    "unit": "USD",
                    "period": "Month",
                    "source": "26 USC 1(a)",
                    "metadata": {
                        "proof": {
                            "atoms": [
                                {
                                    "path": "versions[0].formula",
                                    "kind": "amount",
                                    "source": {
                                        "corpus_citation_path": "us/statute/26/1",
                                        "excerpt": "The allowance is $40.",
                                    },
                                }
                            ]
                        }
                    },
                    "versions": [{"effective_from": "2026-01-01", "formula": "40"}],
                }
            ],
        }
        if compile_failure:
            payload["rules"][0]["kind"] = "invalid_fixture_kind"
        candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
        test = cli._rulespec_test_path(candidate)
        if missing_tests:
            test.unlink(missing_ok=True)
        else:
            test.write_text(
                yaml.safe_dump(
                    [
                        {
                            "name": "allowance",
                            "period": "2026-01",
                            "input": {},
                            "output": {"us:statutes/26/1#allowance": 40},
                        }
                    ]
                )
            )
        marker = policy / "README.md"
        marker.write_text("Admission fixture\n")
        for args in (
            ["init", "-q"],
            ["add", "."],
            [
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "Fixture",
            ],
        ):
            subprocess.run(
                ["git", "-C", str(policy.parent), *args],
                check=True,
                capture_output=True,
            )
        result = SimpleNamespace(
            output_file=str(candidate),
            runner="fixture-runner",
            backend=backend,
            citation="us/statute/26/1",
            source_attestation=attestation,
            context_manifest_file=str(manifest),
            context_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            generation_input_file=str(source_file),
            generation_input_sha256=source_digest,
        )
        return result, {
            "output_root": output_root,
            "policy_repo_path": policy,
            "axiom_rules_path": engine,
            "local_corpus_release": release,
            "require_complete_source_unit": True,
        }

    def compile_fixture(_self, rules_file, output_path):
        payload = yaml.safe_load(rules_file.read_text())
        if payload["rules"][0]["kind"] == "invalid_fixture_kind":
            return subprocess.CompletedProcess(
                [], 1, "", "Invalid rule kind: invalid_fixture_kind"
            ), None
        compiled = {"rules": [], "parameters": []}
        output_path.write_text(json.dumps(compiled))
        return subprocess.CompletedProcess([], 0, "", ""), compiled

    monkeypatch.setattr(
        ValidatorPipeline, "_compile_rulespec_to_artifact", compile_fixture
    )
    monkeypatch.setattr(
        ValidatorPipeline, "_run_rulespec_test_cases", lambda *_args, **_kwargs: []
    )
    monkeypatch.setattr(
        cli, "_apply_validation_execution_identity", lambda **_kwargs: {}
    )
    monkeypatch.setattr(
        cli,
        "_portable_apply_validation_execution_identity",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        cli, "_record_successful_apply_validation", lambda *_args, **_kwargs: None
    )
    return make


def _score(result, options):
    return score_admission(
        result,
        **{
            name: value
            for name, value in options.items()
            if name != "require_complete_source_unit"
        },
    )


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "source,missing_tests,compile_failure,expected_category",
    [
        ("The allowance is $40.", False, False, None),
        ("The allowance is $40.", False, True, "compile_atomic_kind"),
        ("The allowance is $40.", True, False, "ci_tests_missing"),
        (
            "The allowance is $40. The separate limit is $75.",
            False,
            False,
            "complete-source-unit:numeric-recall",
        ),
        (
            "(a) The allowance is $40.\n(b) The separate limit is $75.",
            False,
            False,
            "complete-source-unit:structure",
        ),
        (
            "The allowance is calculated by multiplying income by $40.",
            False,
            False,
            "complete-source-unit:formula-output",
        ),
    ],
)
def test_admission_differential_production_fixtures(
    context, source, missing_tests, compile_failure, expected_category, backend
):
    result, options = context(
        source,
        missing_tests=missing_tests,
        compile_failure=compile_failure,
        backend=backend,
    )
    before_candidate = Path(result.output_file).read_bytes()
    before_policy = _hashes(options["policy_repo_path"].parent)
    score = _score(result, options)
    assert Path(result.output_file).read_bytes() == before_candidate
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert (score.admitted, score.issues) == (ok, issues)
    assert score.prerequisite_failure is False
    assert _hashes(options["policy_repo_path"].parent) == before_policy
    if expected_category is None:
        assert score.admitted, score.issues
    else:
        assert not score.admitted
        assert expected_category in score.refusal_categories, score.issues
    if score.total_authoritative_occurrences is not None:
        assert (
            score.covered_authoritative_occurrences
            + score.missing_authoritative_occurrences
            == score.total_authoritative_occurrences
        )
    if score.admitted:
        assert score.missing_authoritative_occurrences == 0
    if "complete-source-unit:numeric-recall" in score.refusal_categories:
        assert score.missing_authoritative_occurrences > 0


def test_admission_constructs_the_same_pipeline(context, monkeypatch):
    result, options = context()
    observed = []

    class CapturingPipeline(ValidatorPipeline):
        def __init__(self, **kwargs):
            normalized = dict(kwargs)
            normalized["policy_repo_path"] = Path(kwargs["policy_repo_path"]).name
            normalized["rulespec_dependency_roots"] = tuple(
                Path(root).name for root in kwargs["rulespec_dependency_roots"]
            )
            observed.append(normalized)
            super().__init__(**kwargs)

    monkeypatch.setattr(cli, "ValidatorPipeline", CapturingPipeline)
    _score(result, options)
    cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert len(observed) == 2
    assert observed[0] == observed[1]
    assert observed[0]["enable_oracles"] is False
    assert observed[0]["require_policy_proofs"] is True
    assert observed[0]["require_complete_source_unit"] is True


def test_admission_deterministic_json_and_zero_denominator(context):
    result, options = context()
    assert _score(result, options).to_json() == _score(result, options).to_json()


def test_admission_zero_authoritative_occurrences(context):
    result, options = context("The allowance follows an administrative determination.")
    score = _score(result, options)
    assert score.total_authoritative_occurrences == 0
    assert score.covered_authoritative_occurrences == 0
    assert score.missing_authoritative_occurrences == 0
    assert score.authoritative_recall_percentage is None


def test_admission_pathful_compile_failure_is_deterministic(context, monkeypatch):
    result, options = context()

    def compile_failure(_self, rules_file, _output_path):
        return subprocess.CompletedProcess(
            [], 1, "", f"{rules_file}: relation entity typing failed"
        ), None

    monkeypatch.setattr(
        ValidatorPipeline, "_compile_rulespec_to_artifact", compile_failure
    )
    first = _score(result, options)
    second = _score(result, options)
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert first.to_json() == second.to_json()
    assert first.compile_pass is False
    placeholder = "<rulespec-validation-root>"
    assert placeholder in first.issues[0]
    # The signed apply path keeps its diagnostics byte for byte: it names the
    # real temporary overlay, never the placeholder. Each scorer issue is the
    # matching apply issue with only the overlay root replaced.
    assert first.admitted == ok
    assert len(first.issues) == len(issues) >= 1
    assert all(placeholder not in issue for issue in issues)
    tail = re.search(re.escape(placeholder) + r"(/[^\s:]+)", first.issues[0]).group(1)
    overlay_root = issues[0][: issues[0].index(tail)].rsplit(" ", 1)[-1]
    assert Path(overlay_root).is_absolute()
    assert [issue.replace(overlay_root, placeholder) for issue in issues] == list(
        first.issues
    )


def test_admission_shared_helper_preserves_apply_snapshot(context, monkeypatch):
    result, options = context()
    monkeypatch.setattr(
        cli, "_record_successful_apply_validation", _RECORD_SUCCESSFUL_APPLY_VALIDATION
    )
    monkeypatch.setattr(cli, "_apply_result_metadata", lambda _result: {})
    signed_result = copy.deepcopy(result)
    ok, issues, supplemental = (
        cli._validate_generated_encoding_in_policy_overlay_with_release(
            signed_result, **options
        )
    )
    assert ok and not issues
    shared_result = copy.deepcopy(result)
    shared_ok, shared_issues, shared_supplemental = (
        cli._validate_generated_encoding_candidate_in_policy_overlay_with_release(
            shared_result, **options
        )
    )
    assert (shared_ok, shared_issues, shared_supplemental) == (ok, issues, supplemental)
    _RECORD_SUCCESSFUL_APPLY_VALIDATION(
        shared_result,
        **{
            key: value
            for key, value in options.items()
            if key != "require_complete_source_unit"
        },
        relative_output=Path("statutes/26/1.yaml"),
        supplemental_files=shared_supplemental,
    )
    assert json.dumps(
        vars(signed_result)[cli._APPLY_VALIDATION_SNAPSHOT_ATTR], sort_keys=True
    ) == json.dumps(
        vars(shared_result)[cli._APPLY_VALIDATION_SNAPSHOT_ATTR], sort_keys=True
    )


def test_admission_unknown_prefix_is_reported(context, monkeypatch):
    from axiom_encode.harness.validator_pipeline import ValidationResult

    result, options = context()
    monkeypatch.setattr(
        ValidatorPipeline,
        "_run_ci",
        lambda *_args: ValidationResult(
            "ci", False, issues=["[future:gate] Unknown gate"]
        ),
    )
    score = _score(result, options)
    assert score.issue_categories == ["unknown-prefix"]
    assert score.refusal_categories == ["unknown-prefix"]
    assert score.unknown_issues == score.issues


def test_admission_pathful_dependent_baseline_debt_is_tolerated(context, monkeypatch):
    from axiom_encode.harness.validator_pipeline import ValidationResult

    result, options = context()
    policy = options["policy_repo_path"]
    dependent = policy / "statutes/26/2.yaml"
    dependent.parent.mkdir(parents=True, exist_ok=True)
    dependent.write_text(
        "format: rulespec/v1\nimports: [us:statutes/26/1]\nrules: []\n"
    )
    original_ci = ValidatorPipeline._run_ci

    def run_ci(pipeline, path):
        if path.name == "2.yaml":
            return ValidationResult(
                "ci", False, issues=[f"Existing debt in {path}: known failure"]
            )
        return original_ci(pipeline, path)

    monkeypatch.setattr(ValidatorPipeline, "_run_ci", run_ci)
    first = _score(result, options)
    assert first.admitted, first.issues
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert (first.admitted, first.issues) == (ok, issues)


@pytest.mark.parametrize("category", COMPLETE_SOURCE_UNIT_CATEGORIES)
def test_admission_complete_source_categories(category):
    issue = f"statutes/26/1.yaml: ci: [complete-source-unit:{category}] A diagnostic"
    assert issue_category(issue) == f"complete-source-unit:{category}"


def test_admission_seeded_issue_categories():
    generator = random.Random(2158)
    for _ in range(100):
        category = generator.choice(COMPLETE_SOURCE_UNIT_CATEGORIES)
        issue = f"a.yaml: ci: [complete-source-unit:{category}] {generator.randrange(10000)}"
        assert isinstance(issue_category(issue), str)
        assert issue_category(issue) == f"complete-source-unit:{category}"
    assert issue_category("a.yaml: ci: [new-family] Detail") == "unknown-prefix"
    assert (
        issue_category("a.yaml: compile: rules[0] has invalid kind") != "unknown-prefix"
    )


def test_admission_rejects_runner_escape_without_writes(context):
    result, options = context()
    result.runner = "../outside"
    score = _score(result, options)
    assert score.prerequisite_failure
    assert "escapes output root" in score.issues[0]


@pytest.mark.parametrize(
    "issue,category",
    [
        (
            "a.yaml: ci: Ungrounded generated numeric literal: value 10",
            "ci_ungrounded_numeric",
        ),
        (
            "a.yaml: compile: atomic RuleSpec module must not declare composition",
            "compile_atomic_kind",
        ),
        ("a.yaml: compile: relation entity typing failed", "compile_relation_typing"),
        (
            "a.yaml: compile: YAML type error: rules[0] expected string",
            "compile_yaml_type_error",
        ),
        ("a.test.yaml YAML parse failed: expected list", "ci_tests_yaml"),
        (
            "a.yaml: ci: dataset input x must use an absolute legal RuleSpec reference",
            "ci_input_reference_invalid",
        ),
        (
            "a.yaml: ci: `deferred_outputs` is misplaced at document root",
            "ci_deferral_location",
        ),
        (
            "a.yaml: ci: module.deferred_outputs[1].reason is required",
            "ci_deferral_reason",
        ),
        (
            "a.yaml: ci: derived x has no formula version at 2025",
            "ci_temporal_formula_missing",
        ),
        (
            "a.yaml: ci: Proof claim references are not supported in proof atom",
            "ci_proof_claim_unsupported",
        ),
        (
            "a.yaml: ci: Source sub-paragraph coverage missing",
            "ci_source_subparagraph_coverage",
        ),
        (
            "a.yaml: ci: Judgment rule missing positive companion output coverage",
            "ci_judgment_output_coverage",
        ),
    ],
)
def test_admission_historical_issue_families(issue, category):
    assert issue_category(issue) == category


def test_admission_identity_binds_untracked_policy_context(context):
    _result, options = context()
    identity_options = {
        name: value
        for name, value in options.items()
        if name not in {"output_root", "require_complete_source_unit"}
    }
    before = admission_identity(**identity_options)
    untracked = options["policy_repo_path"] / "statutes/26/context.yaml"
    untracked.parent.mkdir(parents=True, exist_ok=True)
    untracked.write_text("format: rulespec/v1\nrules: []\n")
    after = admission_identity(**identity_options)
    assert before["policy_repo"]["commit"] == after["policy_repo"]["commit"]
    assert (
        before["policy_repo"]["tracked_sha256"]
        == after["policy_repo"]["tracked_sha256"]
    )
    assert (
        before["policy_repo"]["content_sha256"]
        != after["policy_repo"]["content_sha256"]
    )


@contextlib.contextmanager
def _short_timeout():
    """Interrupt any regression that tries to open a special artifact."""

    def expired(_signum, _frame):
        raise TimeoutError("Artifact scoring blocked for three seconds")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 3)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize("artifact_name", ["candidate", "companion"])
@pytest.mark.parametrize(
    "kind",
    [
        "broken-symlink",
        "working-symlink",
        "policy-test-symlink",
        "symlink-loop",
        "hardlink",
        "directory",
        "fifo",
        "socket",
        "outside-root",
    ],
)
def test_admission_original_artifact_guard(
    context, monkeypatch, artifact_name, kind, backend
):
    """Reject original artifacts before staging or model source stamping."""

    from unittest.mock import Mock

    from axiom_encode.harness import admission

    result, options = context(backend=backend)
    candidate = Path(result.output_file)
    companion = cli._rulespec_test_path(candidate)
    existing_companion = options["policy_repo_path"] / "statutes/26/1.test.yaml"
    existing_companion.parent.mkdir(parents=True, exist_ok=True)
    existing_companion.write_bytes(companion.read_bytes())
    artifact = candidate if artifact_name == "candidate" else companion
    original = artifact.with_name("original-" + artifact.name)
    original_bytes = artifact.read_bytes()
    artifact.rename(original)
    score_options = dict(options)
    if kind == "broken-symlink":
        artifact.symlink_to(artifact.with_name("absent.yaml"))
    elif kind == "working-symlink":
        artifact.symlink_to(original)
    elif kind == "policy-test-symlink":
        artifact.symlink_to(existing_companion)
    elif kind == "symlink-loop":
        artifact.symlink_to(artifact)
    elif kind == "hardlink":
        os.link(original, artifact)
    elif kind == "directory":
        artifact.mkdir()
    elif kind == "fifo":
        os.mkfifo(artifact)
    elif kind == "socket":
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket, sys; "
                "sock = socket.socket(socket.AF_UNIX); sock.bind(sys.argv[1])",
                artifact.name,
            ],
            cwd=artifact.parent,
            check=True,
            capture_output=True,
        )
    elif artifact_name == "candidate":
        artifact.write_bytes(original_bytes)
        external = options["output_root"].parent / "external"
        artifact.parent.rename(external)
        artifact.parent.symlink_to(external, target_is_directory=True)
    else:
        external = options["output_root"].parent / "external.test.yaml"
        external.write_bytes(original_bytes)
        score_options["companion_test_path"] = external

    before_policy = _hashes(options["policy_repo_path"].parent)
    before_link = artifact.readlink() if artifact.is_symlink() else None
    before_links = original.stat().st_nlink
    production = (
        cli._validate_generated_encoding_candidate_in_policy_overlay_with_release
    )
    production_spy = Mock(wraps=production)
    staging_spy = Mock(wraps=admission.shutil.copyfile)
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production_spy,
    )
    monkeypatch.setattr(admission.shutil, "copyfile", staging_spy)
    with _short_timeout():
        score = _score(result, score_options)
    production_spy.assert_not_called()
    staging_spy.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert original.read_bytes() == original_bytes
    assert original.stat().st_nlink == before_links
    assert _hashes(options["policy_repo_path"].parent) == before_policy
    if before_link is not None:
        assert artifact.readlink() == before_link
    elif kind == "directory":
        assert artifact.is_dir()
    if kind == "outside-root":
        assert "resolves outside the generation output root" in score.issues[0]

    # Production reads model candidates before its guard. These candidate kinds
    # either fail earlier or can block there; the scorer's earlier refusal is
    # intentional. Companion artifacts never enter source stamping.
    production_reaches_guard = artifact_name == "companion" and kind != "outside-root"
    if artifact_name == "candidate":
        production_reaches_guard = kind in {"working-symlink", "hardlink"} or (
            backend == "" and kind in {"directory", "fifo", "socket"}
        )
    if production_reaches_guard:
        monkeypatch.setattr(
            cli,
            "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
            production,
        )
        with _short_timeout():
            ok, issues, _ = (
                cli._validate_generated_encoding_in_policy_overlay_with_release(
                    copy.deepcopy(result), **options
                )
            )
        assert (score.admitted, score.issues) == (ok, issues)


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_review_linked_target_collision(context, monkeypatch, backend):
    """The reviewer's linked-target hardlink can never reach recursive staging."""

    from unittest.mock import Mock

    result, options = context(backend=backend)
    candidate = Path(result.output_file)
    original = candidate.with_name("linked-target")
    candidate.rename(original)
    os.link(original, original.with_name("other-name"))
    candidate.symlink_to(original)
    production = Mock()
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    with _short_timeout():
        score = _score(result, options)
    production.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert score.issues == [
        "statutes/26/linked-target: generated artifact must be a regular "
        "file, not a link: 1.yaml"
    ]


@pytest.mark.parametrize("kind", ["fifo", "symlink"])
def test_admission_artifact_guard_precedes_missing_frozen_context(
    context, monkeypatch, kind
):
    from unittest.mock import Mock

    from axiom_encode.harness import admission

    result, options = context(backend="codex")
    candidate = Path(result.output_file)
    original = candidate.with_name("original.yaml")
    candidate.rename(original)
    if kind == "fifo":
        os.mkfifo(candidate)
    else:
        candidate.symlink_to(original)
    Path(result.context_manifest_file).unlink()
    production = Mock()
    staging = Mock()
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    monkeypatch.setattr(admission.shutil, "copyfile", staging)
    with _short_timeout():
        score = _score(result, options)
    production.assert_not_called()
    staging.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert "generated artifact must be a regular file, not a link" in score.issues[0]


@pytest.mark.parametrize("artifact_name", ["candidate", "companion"])
@pytest.mark.parametrize("kind", ["fifo", "socket", "symlink"])
def test_admission_original_artifacts_in_policy_context_are_never_read(
    context, monkeypatch, artifact_name, kind
):
    from unittest.mock import Mock

    from axiom_encode.harness import admission

    result, options = context(backend="codex")
    artifact = options["policy_repo_path"] / (
        "generated.yaml" if artifact_name == "candidate" else "generated.test.yaml"
    )
    if kind == "fifo":
        os.mkfifo(artifact)
    elif kind == "socket":
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import socket, sys; "
                "sock = socket.socket(socket.AF_UNIX); sock.bind(sys.argv[1])",
                artifact.name,
            ],
            cwd=artifact.parent,
            check=True,
            capture_output=True,
        )
    else:
        artifact.symlink_to(Path(result.output_file))
    if artifact_name == "candidate":
        result.output_file = str(artifact)
    else:
        options["companion_test_path"] = artifact
    original_read_bytes = Path.read_bytes
    production = Mock()
    staging = Mock()

    def read_bytes(path):
        assert path != artifact, "Scorer tried to read the guarded original artifact"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    monkeypatch.setattr(admission.shutil, "copyfile", staging)
    with _short_timeout():
        score = _score(result, options)
    production.assert_not_called()
    staging.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert "generated artifact must be a regular file, not a link" in score.issues[0]


def test_admission_original_candidate_symlink_coincident_with_tool_is_never_read(
    context, monkeypatch
):
    from unittest.mock import Mock

    from axiom_encode.harness import admission

    result, options = context(backend="codex")
    candidate = Path(result.output_file)
    original = candidate.with_name("original.yaml")
    candidate.rename(original)
    candidate.symlink_to(original)
    options["axiom_rules_path"] = candidate
    original_read_bytes = Path.read_bytes
    production = Mock()
    staging = Mock()

    def read_bytes(path):
        assert path != candidate, "Scorer tried to read the guarded original symlink"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    monkeypatch.setattr(admission.shutil, "copyfile", staging)
    with _short_timeout():
        score = _score(result, options)
    production.assert_not_called()
    staging.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert "generated artifact must be a regular file, not a link" in score.issues[0]


@pytest.mark.parametrize("artifact_role", ["candidate", "companion"])
def test_admission_guarded_fifo_coincident_with_pinned_runtime_is_never_read(
    context, monkeypatch, artifact_role
):
    from unittest.mock import Mock

    from axiom_encode.engine_binding import engine_binding_receipt_path
    from axiom_encode.harness import admission

    result, options = context(backend="codex")
    artifact = options["axiom_rules_path"] / "axiom-rules-engine"
    artifact.unlink()
    os.mkfifo(artifact)
    if artifact_role == "candidate":
        result.output_file = str(artifact)
    else:
        options["companion_test_path"] = artifact
    pin = "e" * 40
    configuration = options["policy_repo_path"].parent / ".axiom/toolchain.toml"
    configuration.parent.mkdir()
    configuration.write_text(f'[toolchain]\naxiom_rules_engine_ref = "{pin}"\n')
    engine_binding_receipt_path(artifact).write_text(
        json.dumps({"engine_commit": pin, "binary_sha256": "f" * 64})
    )
    original_read_bytes = Path.read_bytes
    production = Mock()
    staging = Mock()

    def read_bytes(path):
        assert path != artifact, "Scorer tried to read the guarded original FIFO"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    monkeypatch.setattr(admission.shutil, "copyfile", staging)
    with _short_timeout():
        score = _score(result, options)
    production.assert_not_called()
    staging.assert_not_called()
    assert score.admitted is False
    assert score.failure_kind == "candidate"
    assert score.refusal_categories
    assert "generated artifact must be a regular file, not a link" in score.issues[0]


def test_admission_guarded_artifact_keeps_normal_pinned_engine_identity(
    context, monkeypatch
):
    from unittest.mock import Mock

    from axiom_encode.engine_binding import engine_binding_receipt_path

    result, options = context(backend="codex")
    binary = options["axiom_rules_path"] / "axiom-rules-engine"
    pin = "e" * 40
    configuration = options["policy_repo_path"].parent / ".axiom/toolchain.toml"
    configuration.parent.mkdir()
    configuration.write_text(f'[toolchain]\naxiom_rules_engine_ref = "{pin}"\n')
    engine_binding_receipt_path(binary).write_text(
        json.dumps(
            {
                "engine_commit": pin,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            }
        )
    )
    normal = _score(result, options)
    assert normal.admitted is True, normal.issues
    candidate = Path(result.output_file)
    candidate.unlink()
    os.mkfifo(candidate)
    production = Mock()
    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        production,
    )
    with _short_timeout():
        guarded = _score(result, options)
    production.assert_not_called()
    assert guarded.failure_kind == "candidate"
    assert guarded.identity["engine"] == normal.identity["engine"]
    assert guarded.identity["engine"]["declared_ref"] == pin


def test_admission_artifact_guard_preserves_inaccessible_companion_behavior(
    context, monkeypatch
):
    """An inaccessible entry stays absent to the original pathlib checks."""

    result, options = context()
    candidate = Path(result.output_file)
    companion = cli._rulespec_test_path(candidate)
    existing_companion = options["policy_repo_path"] / "statutes/26/1.test.yaml"
    existing_companion.parent.mkdir(parents=True, exist_ok=True)
    existing_companion.write_bytes(companion.read_bytes())
    original_lstat = Path.lstat
    original_stat = os.stat

    def lstat(path, *args, **kwargs):
        if path == companion:
            raise PermissionError("Controlled companion inspection denial")
        return original_lstat(path, *args, **kwargs)

    def stat(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and Path(path) == companion:
            raise PermissionError("Controlled companion inspection denial")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(os, "stat", stat)
    # These are the unchanged old guard's entry-presence semantics.
    assert not companion.is_symlink() and not companion.exists()
    assert (
        cli._generated_artifact_guard_issue(
            candidate,
            companion,
            Path("statutes/26/1.yaml"),
            options["output_root"] / result.runner,
        )
        is None
    )
    score = _score(result, options)
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert (score.admitted, score.issues) == (ok, issues)
    assert score.admitted is True, score.issues


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_review_source_attestation_repro_is_a_numeric_refusal(
    context, backend
):
    result, options = context(
        "The allowance is $40. The separate limit is $75.", backend=backend
    )
    candidate = Path(result.output_file)
    payload = yaml.safe_load(candidate.read_text())
    payload["module"]["source_verification"]["values"] = {"source_attestation": 40}
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
    before = candidate.read_bytes()
    score = _score(result, options)
    assert candidate.read_bytes() == before
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert (score.admitted, score.issues) == (ok, issues)
    assert not score.admitted
    assert score.failure_kind == "candidate"
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    assert "ci_source_scope" in score.refusal_categories
    assert "complete-source-unit:numeric-recall" in score.refusal_categories
    assert (
        score.total_authoritative_occurrences,
        score.covered_authoritative_occurrences,
        score.missing_authoritative_occurrences,
    ) == (2, 1, 1)


@pytest.mark.parametrize("backend", ["", "codex"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("corpus_citation_path", "us/statute/26/999"),
        ("source_attestation", {"unexpected": "candidate controlled"}),
    ],
)
def test_admission_candidate_source_binding_failure_is_scored(
    context, field, value, backend
):
    result, options = context(
        "The allowance is $40. The separate limit is $75.", backend=backend
    )
    candidate = Path(result.output_file)
    payload = yaml.safe_load(candidate.read_text())
    payload["module"]["source_verification"][field] = value
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
    score = _score(result, options)
    ok, issues, _ = cli._validate_generated_encoding_in_policy_overlay_with_release(
        copy.deepcopy(result), **options
    )
    assert (score.admitted, score.issues) == (ok, issues)
    assert not score.admitted
    assert score.failure_kind == "candidate"
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    assert score.refusal_categories


# Freeze every marker from the removed diagnostic-based prerequisite classifier,
# including the qualifiers it combined with source markers. They are candidate
# data in this regression, never evidence that frozen context is unavailable.
_OLD_PREREQUISITE_MARKERS = (
    "[complete-source-unit:authoritative-source]",
    "ownership manifest",
    "apply-manifest coverage",
    "unverifiable release",
    "localcorpusrelease",
    "corpus release object",
    "corpus release signature",
    "bound named corpus release",
    "corpus source",
    "source verification source missing",
    "proof source unresolved",
    "corpus body",
    "corpus release artifact",
    "provision artifact",
    "provisions artifact",
    "source text unavailable",
    "authoritative source unavailable",
    "not found",
    "missing",
    "unavailable",
    "cannot",
    "required",
    "mismatch",
    "no active",
    "not materialized",
    "unresolvable ref",
    "cannot resolve ref",
    "unknown revision",
    "bad revision",
    "engine binary not found",
    "engine pin",
    "engine checkout",
    "requires an explicit axiom-compose executable",
    "axiom-compose executable is not executable",
    "axiom-compose timed out",
    "permission denied",
    "no such file or directory",
    "operation timed out",
    "source relation verification target unavailable",
    "generated output file not found",
    "no candidate artifact",
    "explicit source metadata",
    "context manifest",
    "source_attestation",
    "generation input digest",
    "source attestation",
)
_BRACKET_LABELS = (
    *(f"[complete-source-unit:{name}]" for name in COMPLETE_SOURCE_UNIT_CATEGORIES),
    "[temporal-dependency-coverage]",
    "[existing-target-oracle-contract]",
    "[existing-target-naming-contract]",
    "[future:gate]",
    "[new-family]",
)


@pytest.mark.parametrize("backend", ["", "codex"])
def test_admission_seeded_candidate_text_never_becomes_prerequisite(
    context, monkeypatch, backend
):
    from axiom_encode.harness import admission

    result, options = context(
        "The allowance is $40. The separate limit is $75.", backend=backend
    )
    original_identity = admission.admission_identity
    frozen_identity = None

    def identity_snapshot(**kwargs):
        nonlocal frozen_identity
        if frozen_identity is None:
            frozen_identity = original_identity(**kwargs)
        return copy.deepcopy(frozen_identity)

    # Only candidate text changes in this property. Pin its real context
    # identity once while keeping source resolution and production validation.
    monkeypatch.setattr(admission, "admission_identity", identity_snapshot)
    candidate = Path(result.output_file)
    original = yaml.safe_load(candidate.read_text())
    markers = list(dict.fromkeys((*_OLD_PREREQUISITE_MARKERS, *_BRACKET_LABELS)))
    random.Random(2159).shuffle(markers)
    fields = (
        "rule-name",
        "formula",
        "module-summary",
        "source-verification-values",
        "source-verification-citation",
        "source-verification-attestation",
    )
    # Rotate the marker assignment so every field sees every string once.
    # Each candidate runs all six injections through the real validator.
    for index in range(len(markers)):
        payload = copy.deepcopy(original)
        injected = {
            field: markers[(index + offset) % len(markers)]
            + ": required, missing, unavailable"
            for offset, field in enumerate(fields)
        }
        payload["rules"][0]["name"] = injected["rule-name"]
        payload["rules"][0]["versions"][0]["formula"] = (
            "40 + (" + injected["formula"] + ")"
        )
        payload["module"]["summary"] = injected["module-summary"]
        verification = payload["module"]["source_verification"]
        verification["values"] = {injected["source-verification-values"]: 40}
        verification["corpus_citation_path"] = injected["source-verification-citation"]
        verification["source_attestation"] = injected["source-verification-attestation"]
        candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
        before = candidate.read_bytes()
        score = _score(result, options)
        assert score.failure_kind != "prerequisite", (injected, score.issues)
        assert score.failure_kind == "candidate", (injected, score.issues)
        assert not score.prerequisite_failure, (injected, score.issues)
        assert not score.admitted, (injected, score.issues)
        assert score.refusal_categories, (injected, score.issues)
        assert candidate.read_bytes() == before


@pytest.mark.parametrize(
    "missing,category",
    [
        ("missing-source", "missing-corpus-text"),
        ("empty-source", "missing-corpus-text"),
        ("unverified-release", "unverifiable-corpus-release"),
        ("missing-engine", "unresolvable-ref"),
        ("unresolvable-encoder", "unresolvable-ref"),
        ("unresolvable-policy", "unresolvable-ref"),
        ("unresolvable-dependency", "unresolvable-ref"),
        ("missing-policy", "missing-policy-context"),
        ("missing-compose", "runtime-unavailable"),
    ],
)
def test_admission_preflight_excludes_missing_context_without_validation(
    context, monkeypatch, tmp_path, missing, category
):
    from axiom_encode.harness import admission

    result, options = context(
        "" if missing == "empty-source" else "The allowance is $40."
    )
    if missing == "missing-source":
        release = options["local_corpus_release"]
        (release.root / release.artifacts[0].path).unlink()
    elif missing == "unverified-release":
        options["local_corpus_release"] = None
    elif missing == "missing-engine":
        (options["axiom_rules_path"] / "axiom-rules-engine").unlink()
    elif missing == "missing-policy":
        options["policy_repo_path"] = tmp_path / "missing-policy"
    elif missing == "missing-compose":
        candidate = Path(result.output_file)
        payload = yaml.safe_load(candidate.read_text())
        payload["module"]["kind"] = "composition"
        baseline = options["policy_repo_path"] / "statutes/26/1.yaml"
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text(yaml.safe_dump(payload, sort_keys=False))
    elif missing.startswith("unresolvable-"):
        original_identity = admission.admission_identity

        def unresolved_identity(**kwargs):
            identity = original_identity(**kwargs)
            if missing == "unresolvable-dependency":
                identity["dependencies"] = [{"commit": None}]
            else:
                target = (
                    "encoder" if missing == "unresolvable-encoder" else "policy_repo"
                )
                identity[target]["commit"] = None
            return identity

        monkeypatch.setattr(admission, "admission_identity", unresolved_identity)
    called = []

    def validation_spy(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("Preflight must stop before production validation")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        validation_spy,
    )
    score = _score(result, options)
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_failure
    assert category in score.prerequisite_categories, score.issues
    assert score.refusal_categories == []
    assert not score.admitted
    assert called == []


def test_admission_candidate_composition_cannot_create_a_prerequisite(
    context, monkeypatch
):
    result, options = context()
    candidate = Path(result.output_file)
    payload = yaml.safe_load(candidate.read_text())
    payload["module"]["kind"] = "composition"
    candidate.write_text(yaml.safe_dump(payload, sort_keys=False))
    original_compile = ValidatorPipeline._compile_rulespec_to_artifact
    called = []

    def compile_fixture(pipeline, rules_file, output_path):
        called.append(rules_file)
        composition = yaml.safe_load(rules_file.read_text())
        if composition.get("module", {}).get("kind") == "composition":
            return pipeline._compose_rulespec_module(rules_file, output_path.parent)
        return original_compile(pipeline, rules_file, output_path)

    monkeypatch.setattr(
        ValidatorPipeline, "_compile_rulespec_to_artifact", compile_fixture
    )
    score = _score(result, options)
    assert called
    assert not score.admitted
    assert score.failure_kind == "candidate"
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    assert score.refusal_categories
    assert any(
        "requires an explicit axiom-compose executable" in issue
        for issue in score.issues
    )


@pytest.mark.parametrize("output", ["missing-path", "no-output"])
def test_admission_no_artifact_is_a_candidate_failure(context, monkeypatch, output):
    result, options = context()
    Path(result.output_file).unlink()
    if output == "no-output":
        result.output_file = None
    called = []

    def validation_spy(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("No artifact cannot reach production validation")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        validation_spy,
    )
    score = _score(result, options)
    assert not score.admitted
    assert score.failure_kind == "candidate"
    assert score.refusal_categories == ["no-artifact"]
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    assert called == []


def _runner_failure_row(result, failure_kind="error"):
    """Shape the row the eval harness writes when its runner raises."""

    Path(result.output_file).unlink()
    result.output_file = ""
    result.trace_file = ""
    result.context_manifest_file = ""
    result.context_manifest_sha256 = None
    result.success = False
    result.failure_kind = failure_kind
    result.timed_out = failure_kind == "timeout"
    return result


def _production_spy(monkeypatch):
    called = []

    def validation_spy(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("An absent artifact cannot reach production validation")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        validation_spy,
    )
    return called


@pytest.mark.parametrize(
    "absent", ["missing-path", "no-output", "runner-error", "runner-timeout"]
)
@pytest.mark.parametrize(
    "missing,category",
    [
        ("empty-source", "missing-corpus-text"),
        ("missing-source", "missing-corpus-text"),
        ("unverified-release", "unverifiable-corpus-release"),
        ("missing-engine", "unresolvable-ref"),
        ("missing-policy", "missing-policy-context"),
        ("unresolvable-policy", "unresolvable-ref"),
        ("expected-context", "expected-context-mismatch"),
    ],
)
def test_admission_absent_artifact_never_hides_a_prerequisite(
    context, monkeypatch, tmp_path, absent, missing, category
):
    """The reviewer's probes: a diagnosed frozen-input failure is not the model's."""

    from axiom_encode.harness import admission

    result, options = context(
        "" if missing == "empty-source" else "The allowance is $40."
    )
    if absent == "missing-path":
        Path(result.output_file).unlink()
    elif absent == "no-output":
        Path(result.output_file).unlink()
        result.output_file = None
    else:
        _runner_failure_row(
            result, "timeout" if absent == "runner-timeout" else "error"
        )
    if missing == "missing-source":
        release = options["local_corpus_release"]
        (release.root / release.artifacts[0].path).unlink()
    elif missing == "unverified-release":
        options["local_corpus_release"] = None
    elif missing == "missing-engine":
        (options["axiom_rules_path"] / "axiom-rules-engine").unlink()
    elif missing == "missing-policy":
        options["policy_repo_path"] = tmp_path / "missing-policy"
    elif missing == "unresolvable-policy":
        original_identity = admission.admission_identity

        def unresolved_identity(**kwargs):
            identity = original_identity(**kwargs)
            identity["policy_repo"]["commit"] = None
            return identity

        monkeypatch.setattr(admission, "admission_identity", unresolved_identity)
    elif missing == "expected-context":
        options["expected_context"] = {"source_body_sha256": "0" * 64}
    called = _production_spy(monkeypatch)
    score = _score(result, options)
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_failure
    assert score.prerequisite_categories == [category], score.issues
    assert score.refusal_categories == []
    assert not score.admitted
    assert called == []


@pytest.mark.parametrize("output", ["missing-path", "no-output"])
@pytest.mark.parametrize(
    "broken,category",
    [
        ("missing-manifest", "missing-context-manifest"),
        ("manifest-hash", "context-manifest-hash-mismatch"),
        ("invalid-manifest", "invalid-context-manifest"),
        ("missing-generation-input", "missing-generation-input"),
        ("generation-input-hash", "generation-input-hash-mismatch"),
    ],
)
def test_admission_absent_artifact_never_hides_broken_frozen_context(
    context, monkeypatch, output, broken, category
):
    result, options = context()
    Path(result.output_file).unlink()
    if output == "no-output":
        result.output_file = None
    manifest = Path(result.context_manifest_file)
    if broken == "missing-manifest":
        manifest.unlink()
    elif broken == "manifest-hash":
        manifest.write_text(manifest.read_text() + "\n")
    elif broken == "invalid-manifest":
        manifest.write_text("[]")
        result.context_manifest_sha256 = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
    elif broken == "missing-generation-input":
        (manifest.parent / "source.txt").unlink()
    else:
        (manifest.parent / "source.txt").write_text("The allowance is $41.")
    called = _production_spy(monkeypatch)
    score = _score(result, options)
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_categories == [category], score.issues
    assert score.refusal_categories == []
    assert called == []


@pytest.mark.parametrize(
    "failure_kind,category",
    [("error", "generation-error"), ("timeout", "generation-timeout")],
)
def test_admission_runner_failure_row_is_named_from_its_own_record(
    context, monkeypatch, failure_kind, category
):
    result, options = context()
    _runner_failure_row(result, failure_kind)
    called = _production_spy(monkeypatch)
    score = _score(result, options)
    assert not score.admitted
    assert score.failure_kind == "candidate"
    assert score.refusal_categories == [category]
    assert score.issue_categories == [category]
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    # No workspace was recorded, so the row binds its source and nothing more.
    assert "context" not in score.identity
    assert score.identity["source"]["requested_corpus_citation_path"] == (
        "us/statute/26/1"
    )
    assert called == []
    assert _score(result, options).to_json() == score.to_json()


def test_admission_timeout_with_a_workspace_is_a_generation_timeout(
    context, monkeypatch
):
    result, options = context()
    Path(result.output_file).unlink()
    result.output_file = ""
    result.success = False
    result.failure_kind = "timeout"
    result.timed_out = True
    called = _production_spy(monkeypatch)
    score = _score(result, options)
    assert score.failure_kind == "candidate"
    assert score.refusal_categories == ["generation-timeout"]
    assert score.identity["context"]["context_manifest_sha256"] == (
        result.context_manifest_sha256
    )
    assert called == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", True),
        ("success", None),
        ("failure_kind", None),
        ("failure_kind", "validation"),
        ("trace_file", "trace.json"),
        ("output_file", "statutes/26/never-written.yaml"),
    ],
)
def test_admission_only_a_recorded_runner_failure_may_omit_its_manifest(
    context, monkeypatch, field, value
):
    result, options = context()
    _runner_failure_row(result)
    setattr(result, field, value)
    called = _production_spy(monkeypatch)
    score = _score(result, options)
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_categories == ["missing-context-manifest"]
    assert score.refusal_categories == []
    assert called == []


@pytest.mark.parametrize("failure", ["authentication", "quota-exhaustion"])
@pytest.mark.parametrize("absent", ["missing-path", "runner-error"])
def test_admission_reported_infrastructure_failure_is_a_prerequisite(
    context, monkeypatch, failure, absent
):
    """The fourth review's probe: the caller proved infrastructure ended it."""

    result, options = context()
    if absent == "missing-path":
        Path(result.output_file).unlink()
    else:
        _runner_failure_row(result)
    called = _production_spy(monkeypatch)
    score = _score(result, {**options, "generation_infrastructure_failure": failure})
    assert not score.admitted
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_categories == [f"generation-{failure}"]
    assert score.refusal_categories == []
    assert called == []
    # Without the caller's evidence the same row is the chain's own outcome.
    unreported = _score(result, options)
    assert unreported.failure_kind == "candidate"
    assert unreported.refusal_categories == [
        "no-artifact" if absent == "missing-path" else "generation-error"
    ]


@pytest.mark.parametrize("failure", ["authentication", "quota-exhaustion"])
@pytest.mark.parametrize("compile_failure", [False, True])
def test_admission_infrastructure_report_does_not_excuse_an_artifact(
    context, failure, compile_failure
):
    """A candidate that exists is judged, whatever the caller reports."""

    result, options = context(compile_failure=compile_failure)
    reported = _score(result, {**options, "generation_infrastructure_failure": failure})
    assert reported.to_json() == _score(result, options).to_json()
    assert reported.prerequisite_categories == []
    if compile_failure:
        assert reported.failure_kind == "candidate"
        assert reported.refusal_categories
    else:
        assert reported.admitted


@pytest.mark.parametrize("failure", ["authentication", "quota-exhaustion"])
def test_admission_infrastructure_report_does_not_excuse_a_nonregular_artifact(
    context, monkeypatch, failure
):
    result, options = context()
    candidate = Path(result.output_file)
    original = candidate.with_name("original.yaml")
    candidate.rename(original)
    candidate.symlink_to(original)
    called = _production_spy(monkeypatch)
    with _short_timeout():
        score = _score(
            result, {**options, "generation_infrastructure_failure": failure}
        )
    assert score.failure_kind == "candidate"
    assert "generated artifact must be a regular file, not a link" in score.issues[0]
    assert called == []


def test_admission_infrastructure_report_yields_to_a_diagnosed_prerequisite(
    context, monkeypatch
):
    result, options = context("")
    Path(result.output_file).unlink()
    called = _production_spy(monkeypatch)
    score = _score(
        result, {**options, "generation_infrastructure_failure": "authentication"}
    )
    assert score.failure_kind == "prerequisite"
    assert score.prerequisite_categories == ["missing-corpus-text"]
    assert called == []


@pytest.mark.parametrize("failure", ["", "harness-crash", "Authentication", 1])
def test_admission_rejects_an_unknown_infrastructure_class(context, failure):
    result, options = context()
    with pytest.raises(ValueError, match="Unknown generation infrastructure failure"):
        _score(result, {**options, "generation_infrastructure_failure": failure})


def _context_with_packaged_file(result, origin):
    """Repackage the fixture context with one copied file from ``origin``."""

    manifest = Path(result.context_manifest_file)
    payload = json.loads(manifest.read_text())
    packaged = manifest.parent / "context/precedent.yaml"
    packaged.parent.mkdir(exist_ok=True)
    packaged.write_text("format: rulespec/v1\n")
    payload["context_files"] = [
        {
            "source_path": origin,
            "workspace_path": "context/precedent.yaml",
            "import_path": "us:statutes/26/2",
            "kind": "implementation_precedent",
        }
    ]
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    result.context_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_admission_context_identity_ignores_the_checkout_it_was_packaged_from(
    context,
):
    """The reviewer's fold repro: only an origin path differs between runs."""

    result, options = context()
    _context_with_packaged_file(result, "/checkouts/left/statutes/26/2.yaml")
    left = _score(result, options)
    _context_with_packaged_file(result, "/checkouts/right/statutes/26/2.yaml")
    right = _score(result, options)
    assert left.admitted and right.admitted
    left_context = dict(left.identity["context"])
    right_context = dict(right.identity["context"])
    assert left_context.pop("context_manifest_sha256") != right_context.pop(
        "context_manifest_sha256"
    )
    assert left_context == right_context
    assert set(left_context) == {
        "context_manifest_canonical_sha256",
        "generation_input_sha256",
        "artifacts",
    }
    assert {key: value for key, value in left.identity.items() if key != "context"} == {
        key: value for key, value in right.identity.items() if key != "context"
    }


@pytest.mark.parametrize(
    "change",
    ["bytes", "workspace_path", "kind", "import_path", "citation-origin", "mode"],
)
def test_admission_context_identity_changes_with_packaged_content(context, change):
    result, options = context()
    _context_with_packaged_file(result, "/checkouts/left/statutes/26/2.yaml")
    reference = _score(result, options).identity["context"]
    manifest = Path(result.context_manifest_file)
    payload = json.loads(manifest.read_text())
    item = payload["context_files"][0]
    if change == "bytes":
        (manifest.parent / item["workspace_path"]).write_text("format: other\n")
    elif change == "workspace_path":
        moved = manifest.parent / "context/other.yaml"
        (manifest.parent / item["workspace_path"]).rename(moved)
        item["workspace_path"] = "context/other.yaml"
    elif change == "kind":
        item["kind"] = "canonical_concept"
    elif change == "import_path":
        item["import_path"] = "us:statutes/26/3"
    elif change == "citation-origin":
        # A citation is not a checkout location, so it stays in the identity.
        item["source_path"] = "us/statute/26/2"
    else:
        payload["mode"] = "repair"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True))
    result.context_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    changed = _score(result, options).identity["context"]
    comparable = {
        key: value for key, value in changed.items() if key != "context_manifest_sha256"
    }
    assert comparable != {
        key: value
        for key, value in reference.items()
        if key != "context_manifest_sha256"
    }
    if change == "bytes":
        assert (
            changed["context_manifest_canonical_sha256"]
            == reference["context_manifest_canonical_sha256"]
        )
        assert changed["artifacts"] != reference["artifacts"]
    else:
        assert (
            changed["context_manifest_canonical_sha256"]
            != reference["context_manifest_canonical_sha256"]
        )


def test_admission_canonical_context_digest_properties():
    """Seeded: only an absolute origin path may change without changing the digest."""

    import copy
    import random

    from axiom_encode.harness.admission import _canonical_context_manifest_sha256

    rng = random.Random(20261010)

    def absolute_origin():
        name = f"{rng.randrange(10**6)}/statutes/{rng.randrange(99)}.yaml"
        return rng.choice([f"/checkouts/{name}", f"C:\\checkouts\\{name}"])

    def item():
        payload = {
            "source_path": rng.choice(
                [absolute_origin(), f"us/statute/26/{rng.randrange(99)}"]
            ),
            "workspace_path": f"context/{rng.randrange(10**6)}.yaml",
            "kind": rng.choice(["implementation_precedent", "definition_stub"]),
        }
        if rng.random() < 0.5:
            payload["import_path"] = f"us:statutes/26/{rng.randrange(99)}"
        return payload

    def is_absolute(origin):
        return origin.startswith("/") or origin[1:3] == ":\\"

    for _ in range(300):
        manifest = {
            "citation": f"us/statute/26/{rng.randrange(99)}",
            "mode": rng.choice(["cold", "repair"]),
            "source_text_file": "source.txt",
            "source_metadata": {"source_attestation": {"source_sha256": "0" * 64}},
            "context_files": [item() for _ in range(rng.randrange(4))],
            "review_findings_files": [item() for _ in range(rng.randrange(3))],
        }
        before = copy.deepcopy(manifest)
        digest = _canonical_context_manifest_sha256(manifest)
        assert manifest == before, "canonicalization mutated its input"
        assert digest == _canonical_context_manifest_sha256(copy.deepcopy(manifest))
        assert digest == _canonical_context_manifest_sha256(
            dict(reversed(list(manifest.items())))
        )

        relocated = copy.deepcopy(manifest)
        for key in ("context_files", "review_findings_files"):
            for entry in relocated[key]:
                if is_absolute(entry["source_path"]):
                    entry["source_path"] = absolute_origin()
        assert _canonical_context_manifest_sha256(relocated) == digest

        items = [
            (key, index)
            for key in ("context_files", "review_findings_files")
            for index in range(len(manifest[key]))
        ]
        changed = copy.deepcopy(manifest)
        if items and rng.random() < 0.7:
            key, index = rng.choice(items)
            entry = changed[key][index]
            field = rng.choice(["workspace_path", "kind", "citation-origin"])
            if field == "citation-origin":
                entry["source_path"] = f"us/statute/27/{rng.randrange(99)}"
            else:
                entry[field] = entry[field] + "-changed"
        else:
            changed["mode"] = changed["mode"] + "-changed"
        assert _canonical_context_manifest_sha256(changed) != digest


@pytest.mark.parametrize("exception_type", [RuntimeError, ValueError, KeyError])
def test_admission_escaped_production_exception_is_a_scorer_error(
    context, monkeypatch, exception_type
):
    result, options = context()

    def explode(*_args, **_kwargs):
        raise exception_type("fixture scorer failure")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        explode,
    )
    score = _score(result, options)
    assert not score.admitted
    assert score.failure_kind == "scorer-error"
    assert not score.prerequisite_failure
    assert score.prerequisite_categories == []
    assert score.refusal_categories == []
    assert score.scorer_error == {
        "type": exception_type.__name__,
        "message": str(exception_type("fixture scorer failure")),
    }
    assert any(exception_type.__name__ in issue for issue in score.issues)
    assert any("fixture scorer failure" in issue for issue in score.issues)


def test_admission_pathful_scorer_exception_is_deterministic(context, monkeypatch):
    result, options = context()

    def explode(staged_result, **_kwargs):
        raise RuntimeError(f"fixture failure: {staged_result.output_file}")

    monkeypatch.setattr(
        cli,
        "_validate_generated_encoding_candidate_in_policy_overlay_with_release",
        explode,
    )
    first = _score(result, options)
    second = _score(result, options)
    assert first.to_json() == second.to_json()
    assert first.failure_kind == "scorer-error"
    assert first.scorer_error == {
        "type": "RuntimeError",
        "message": f"fixture failure: {result.output_file}",
    }
    assert first.refusal_categories == []
    assert first.prerequisite_categories == []


@pytest.mark.parametrize(
    "issue,category",
    [
        (
            "statutes/26/1.yaml: ci: Source claim key: "
            "[complete-source-unit:authoritative-source] required",
            "ci_source_scope",
        ),
        (
            "statutes/26/1.yaml: compile: Invalid rule kind: "
            "[existing-target-oracle-contract]",
            "compile_atomic_kind",
        ),
        (
            "statutes/26/1.yaml: ci: expected text: [future:gate]",
            "ci_test_assertion",
        ),
        (
            "statutes/26/1.yaml: ci: Candidate quoted "
            "[complete-source-unit:numeric-recall]",
            "ci-other",
        ),
    ],
)
def test_admission_bracket_labels_in_candidate_text_do_not_set_categories(
    issue, category
):
    assert issue_category(issue) == category

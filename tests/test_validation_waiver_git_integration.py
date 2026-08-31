"""Real-Git integration coverage for immutable waiver-transition evidence."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import __version__, cli
from axiom_encode.harness.evals import resolve_corpus_source_unit
from tests.eval_evidence_fixtures import (
    TEST_APPLY_PRIVATE_KEY_B64,
    TEST_APPLY_PUBLIC_KEY_B64,
)
from tests.release_object_fixtures import bind_test_corpus_release
from tests.signing_broker_fixtures import SigningBrokerFixture

_CORPUS_RELEASE = "waiver-git-integration-release"
_MODULE_PATH = "us/statutes/module.yaml"
_WAIVER_PATH = "known-validation-gaps.yaml"
_TOOLCHAIN_PATH = ".axiom/toolchain.toml"
_MANIFEST_PATH = ".axiom/encoding-manifests/us/statutes/module.json"
_CORPUS_CITATION = "us/statute/waiver-git-integration"
_ENCODER_IDENTITY = {
    "repository": cli.APPLIED_ENCODING_OFFICIAL_REPOSITORY,
    "commit": "a" * 40,
    "version": __version__,
}
_SIGNING_BROKER = SigningBrokerFixture(
    apply_private_key=TEST_APPLY_PRIVATE_KEY_B64,
    apply_public_key=TEST_APPLY_PUBLIC_KEY_B64,
)
_BASE_EVIDENCE_LABELS = (
    "protected-base validation waiver set",
    "head validation waiver set",
    "changed-paths input",
    "protected-base RuleSpec toolchain",
    "head RuleSpec toolchain",
)
_MANIFEST_EVIDENCE_LABEL = f"pending-consumption apply manifest {_MANIFEST_PATH}"
_RACE_CASES = tuple(
    (phase, label)
    for phase in ("creation", "consumption", "pending-only-consumption")
    for label in _BASE_EVIDENCE_LABELS
) + tuple(
    (phase, _MANIFEST_EVIDENCE_LABEL)
    for phase in ("consumption", "pending-only-consumption")
)


@dataclass(frozen=True)
class _GitTransition:
    repository: Path
    head_worktree: Path
    base_commit: str
    head_commit: str
    corpus: Path
    base_waiver: Path
    base_toolchain: Path
    changed_paths: Path
    expected_execution: tuple[dict[str, object], ...]

    @property
    def evidence_paths(self) -> dict[str, Path]:
        evidence = {
            "protected-base validation waiver set": self.base_waiver,
            "head validation waiver set": self.head_worktree / _WAIVER_PATH,
            "changed-paths input": self.changed_paths,
            "protected-base RuleSpec toolchain": self.base_toolchain,
            "head RuleSpec toolchain": self.head_worktree / _TOOLCHAIN_PATH,
        }
        manifest = self.head_worktree / _MANIFEST_PATH
        if manifest.is_file():
            evidence[_MANIFEST_EVIDENCE_LABEL] = manifest
        return evidence


def _git(repository: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _approval(marker: str) -> str:
    expiry = (date.today() + timedelta(days=30)).isoformat()
    return (
        f'      fingerprint: "sha256:{marker * 64}"\n'
        '      owner: "@MaxGhenis"\n'
        "      issue: "
        '"https://github.com/TheAxiomFoundation/axiom-encode/issues/1558"\n'
        f'      expires: "{expiry}"\n'
    )


def _waiver_yaml(
    *,
    active: str | None = None,
    pending: str | None = None,
) -> bytes:
    if active is None and pending is None:
        return b"validate_failures: {}\n"
    chunks = ["validate_failures:\n", f"  {_MODULE_PATH}:\n"]
    if active is not None:
        chunks.extend(("    active:\n", _approval(active)))
    if pending is not None:
        chunks.extend(("    pending:\n", _approval(pending)))
    return "".join(chunks).encode()


def _build_corpus(tmp_path: Path):
    corpus = tmp_path / "axiom-corpus"
    provision = (
        corpus / "data/corpus/provisions/us/statute/waiver-git-integration.jsonl"
    )
    _write(
        provision,
        (
            json.dumps(
                {
                    "id": "test:us/statute/waiver-git-integration",
                    "citation_path": "us/statute/waiver-git-integration",
                    "body": "authoritative source",
                    "jurisdiction": "us",
                    "document_class": "statute",
                    "version": "waiver-git-integration",
                    "source_path": "sources/us/statute/waiver-git-integration",
                    "source_as_of": "2026-01-01",
                    "expression_date": "2026-01-01",
                },
                sort_keys=True,
            )
            + "\n"
        ).encode(),
    )
    release = bind_test_corpus_release(
        corpus,
        _CORPUS_RELEASE,
        [("us", "statute", "waiver-git-integration")],
    )
    return corpus, release


def _source_backed_module(source_sha256: str, *, head: bool = False) -> bytes:
    suffix = "# module changed for pending activation\n" if head else ""
    return (
        "format: rulespec/v1\n"
        "module:\n"
        "  source_verification:\n"
        f"    corpus_citation_path: {_CORPUS_CITATION}\n"
        f"    source_sha256: {source_sha256}\n"
        "rules: []\n" + suffix
    ).encode()


def _signed_model_manifest(
    *,
    base_waiver: bytes,
    module_bytes: bytes,
    source_attestation: dict[str, object],
) -> bytes:
    waiver_sha256 = hashlib.sha256(base_waiver).hexdigest()
    payload = {
        "schema_version": cli.APPLIED_ENCODING_MANIFEST_SCHEMA,
        "generated_at": "2026-08-31T12:00:00+00:00",
        "tool": cli.APPLIED_ENCODING_MODEL_TOOL,
        "axiom_encode_version": __version__,
        "axiom_encode_git": {
            "root": "/repo/axiom-encode",
            "commit": _ENCODER_IDENTITY["commit"],
            "dirty_tracked": False,
            "version": __version__,
            "version_commit": "b" * 40,
            "identity_source": "git",
        },
        "generation_prompt_sha256": None,
        "run_id": None,
        "citation": _CORPUS_CITATION,
        "runner": "codex",
        "backend": "codex",
        "model": "waiver-integration-fixture",
        "validation_waiver_set_sha256": waiver_sha256,
        "generated_output_root": "/tmp/axiom-generated",
        "generated_output_file": None,
        "generated_output_sha256": None,
        "trace_file": None,
        "trace_sha256": None,
        "context_manifest_file": None,
        "context_manifest_sha256": None,
        "applied_files": [
            {
                "path": _MODULE_PATH,
                "sha256": hashlib.sha256(module_bytes).hexdigest(),
            }
        ],
        "source_attestation": source_attestation,
        "validation_execution": {
            "schema": "axiom-encode/apply-validation-execution/v1",
            "axiom_encode": {
                **_ENCODER_IDENTITY,
                "identity_source": "git",
            },
            "axiom_rules_engine": {
                "repository": "github.com/TheAxiomFoundation/axiom-rules-engine",
                "commit": "e" * 40,
            },
            "policy_pre_apply": {
                "rulespec_root": "rulespec-us/us",
                "pre_apply_content_sha256": "e" * 64,
                "pre_apply_file_count": 1,
                "toolchain_contract_sha256": "d" * 64,
                "validation_waiver_set_sha256": waiver_sha256,
            },
            "rulespec_dependencies": [],
        },
    }
    cli._sign_applied_encoding_manifest(payload, _SIGNING_BROKER)
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _toolchain(waiver: bytes, *, corpus_digest: str) -> bytes:
    waiver_digest = hashlib.sha256(waiver).hexdigest()
    return (
        "[toolchain]\n"
        f'axiom_corpus_release = "{_CORPUS_RELEASE}"\n'
        f'axiom_corpus_release_content_sha256 = "{corpus_digest}"\n'
        f'validation_waiver_set_sha256 = "{waiver_digest}"\n'
    ).encode()


def _materialize_blob(
    repository: Path,
    commit: str,
    relative_path: str,
    destination: Path,
) -> bytes:
    raw = _git(repository, "show", f"{commit}:{relative_path}")
    _write(destination, raw)
    return raw


def _build_git_transition(tmp_path: Path, phase: str) -> _GitTransition:
    repository = tmp_path / "primary" / "rulespec-us"
    repository.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "-q", str(repository)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _git(repository, "config", "user.name", "Waiver Integration Test")
    _git(repository, "config", "user.email", "waiver-test@example.invalid")
    corpus, corpus_release = _build_corpus(tmp_path)
    corpus_digest = corpus_release.content_sha256
    source_unit = resolve_corpus_source_unit(_CORPUS_CITATION, corpus_release)
    source_attestation = dict(source_unit.source_attestation)
    source_attestation["generation_input_sha256"] = source_attestation[
        "resolved_text_sha256"
    ]
    source_attestation["rulespec_root"] = "rulespec-us/us"
    base_module = _source_backed_module(str(source_attestation["source_sha256"]))

    if phase == "creation":
        base_waiver = _waiver_yaml()
        head_waiver = _waiver_yaml(pending="b")
        expected = (
            {
                "path": _MODULE_PATH,
                "passed": True,
                "fingerprint": "sha256:passing",
                "outcome": {},
            },
        )
    elif phase in {"consumption", "pending-only-consumption"}:
        base_waiver = _waiver_yaml(
            active="a" if phase == "consumption" else None,
            pending="b",
        )
        head_waiver = _waiver_yaml(active="b")
        expected = (
            {
                "path": _MODULE_PATH,
                "passed": False,
                "fingerprint": f"sha256:{'b' * 64}",
                "outcome": {},
            },
        )
    elif phase == "semantic-noop":
        base_waiver = _waiver_yaml(active="a")
        head_waiver = base_waiver + b"# byte-only rewrite\n"
        expected = ()
    else:  # pragma: no cover - test helper contract
        raise AssertionError(f"unknown transition phase: {phase}")

    _write(repository / _WAIVER_PATH, base_waiver)
    _write(
        repository / _TOOLCHAIN_PATH,
        _toolchain(base_waiver, corpus_digest=corpus_digest),
    )
    _write(repository / _MODULE_PATH, base_module)
    _git(repository, "add", "--", _WAIVER_PATH, _TOOLCHAIN_PATH, _MODULE_PATH)
    _git(repository, "commit", "-q", "-m", "base waiver state")
    base_commit = _git(repository, "rev-parse", "HEAD").decode().strip()

    _write(repository / _WAIVER_PATH, head_waiver)
    _write(
        repository / _TOOLCHAIN_PATH,
        _toolchain(head_waiver, corpus_digest=corpus_digest),
    )
    if phase in {"consumption", "pending-only-consumption"}:
        head_module = _source_backed_module(
            str(source_attestation["source_sha256"]),
            head=True,
        )
        _write(repository / _MODULE_PATH, head_module)
        _write(
            repository / _MANIFEST_PATH,
            _signed_model_manifest(
                base_waiver=base_waiver,
                module_bytes=head_module,
                source_attestation=source_attestation,
            ),
        )
    head_paths = [_WAIVER_PATH, _TOOLCHAIN_PATH, _MODULE_PATH]
    if phase in {"consumption", "pending-only-consumption"}:
        head_paths.append(_MANIFEST_PATH)
    _git(repository, "add", "--", *head_paths)
    _git(repository, "commit", "-q", "-m", "head waiver state")
    head_commit = _git(repository, "rev-parse", "HEAD").decode().strip()

    _git(repository, "checkout", "-q", "--detach", base_commit)
    head_worktree = tmp_path / "head" / "rulespec-us"
    head_worktree.parent.mkdir()
    _git(
        repository, "worktree", "add", "-q", "--detach", str(head_worktree), head_commit
    )

    evidence = tmp_path / "git-evidence"
    base_waiver_path = evidence / "protected-base-waivers.yaml"
    base_toolchain_path = evidence / "protected-base-toolchain.toml"
    _materialize_blob(repository, base_commit, _WAIVER_PATH, base_waiver_path)
    _materialize_blob(repository, base_commit, _TOOLCHAIN_PATH, base_toolchain_path)
    for relative_path in (_WAIVER_PATH, _TOOLCHAIN_PATH):
        blob = _git(repository, "show", f"{head_commit}:{relative_path}")
        assert (head_worktree / relative_path).read_bytes() == blob

    changed_paths = evidence / "changed-paths.zlist"
    _write(
        changed_paths,
        _git(
            repository,
            "diff",
            "--name-only",
            "-z",
            base_commit,
            head_commit,
            "--",
        ),
    )
    assert _git(repository, "rev-parse", "--git-common-dir").strip()
    assert (head_worktree / ".git").is_file()
    return _GitTransition(
        repository=repository,
        head_worktree=head_worktree,
        base_commit=base_commit,
        head_commit=head_commit,
        corpus=corpus,
        base_waiver=base_waiver_path,
        base_toolchain=base_toolchain_path,
        changed_paths=changed_paths,
        expected_execution=expected,
    )


def _audit_args(transition: _GitTransition) -> SimpleNamespace:
    return SimpleNamespace(
        root=transition.head_worktree,
        corpus_path=transition.corpus,
        waivers=None,
        protected_base=transition.base_waiver,
        protected_base_toolchain=transition.base_toolchain,
        changed_paths=transition.changed_paths,
        changed_paths_format="nul-v1",
        axiom_rules_path=None,
        rulespec_dependency_root=[],
        partition_key=None,
        partition_keys_json=None,
        json=True,
    )


def _audit(
    transition: _GitTransition,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    executor=None,
) -> tuple[int, dict[str, object]]:
    monkeypatch.setenv(cli._WAIVER_AUDIT_WORKERS_ENV, "1")
    execution = executor or (
        lambda *_args, **_kwargs: list(transition.expected_execution)
    )

    with (
        patch.object(
            cli,
            "_fingerprint_validation_waiver_modules",
            side_effect=execution,
        ),
        patch.object(
            cli,
            "_read_only_guard_encoder_execution_identity",
            return_value=_ENCODER_IDENTITY,
        ),
        patch.object(
            cli,
            "_applied_encoding_manifest_verifier",
            return_value=_SIGNING_BROKER,
        ),
    ):
        exit_code = cli._cmd_validation_waivers_audit(_audit_args(transition))
    return exit_code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    "phase", ["creation", "consumption", "pending-only-consumption"]
)
def test_real_git_cross_worktree_accepts_exact_transition_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
):
    transition = _build_git_transition(tmp_path, phase)

    exit_code, report = _audit(transition, monkeypatch, capsys)

    assert exit_code == 0
    assert report["success"] is True
    assert report["checked"] == 1
    assert (
        _git(
            transition.repository,
            "diff",
            "--name-only",
            "-z",
            transition.base_commit,
            transition.head_commit,
        )
        == transition.changed_paths.read_bytes()
    )
    if phase in {"consumption", "pending-only-consumption"}:
        assert _MANIFEST_PATH.encode() + b"\0" in transition.changed_paths.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("signature", "signature"),
        ("stale-module-hash", "stale sha256"),
        ("unlisted-consumed-module", "consumed module must be changed and listed"),
    ],
)
@pytest.mark.parametrize("phase", ["consumption", "pending-only-consumption"])
def test_real_git_consumption_rejects_adversarial_signed_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    expected_error: str,
    phase: str,
):
    transition = _build_git_transition(tmp_path, phase)
    manifest = transition.head_worktree / _MANIFEST_PATH
    payload = json.loads(manifest.read_text())
    if mutation == "signature":
        payload["model"] = "tampered-after-signing"
    elif mutation == "stale-module-hash":
        payload["applied_files"][0]["sha256"] = "0" * 64
        cli._sign_applied_encoding_manifest(payload, _SIGNING_BROKER)
    else:
        other_path = "us/statutes/other.yaml"
        other = transition.head_worktree / other_path
        source_sha256 = payload["source_attestation"]["source_sha256"]
        _write(other, _source_backed_module(str(source_sha256)))
        payload["applied_files"] = [
            {
                "path": other_path,
                "sha256": hashlib.sha256(other.read_bytes()).hexdigest(),
            }
        ]
        cli._sign_applied_encoding_manifest(payload, _SIGNING_BROKER)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    exit_code, report = _audit(transition, monkeypatch, capsys)

    assert exit_code == 1
    assert report["success"] is False
    assert report["checked"] == 0
    assert any(expected_error in error for error in report["errors"])


@pytest.mark.parametrize(
    "mutation",
    ["base-waiver", "base-toolchain", "head-pair", "changed-paths"],
)
@pytest.mark.parametrize(
    "phase", ["creation", "consumption", "pending-only-consumption"]
)
def test_real_git_transition_rejects_mutated_materialized_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
    phase: str,
):
    transition = _build_git_transition(tmp_path, phase)
    if mutation == "base-waiver":
        transition.base_waiver.write_bytes(
            transition.base_waiver.read_bytes() + b"# uncommitted evidence drift\n"
        )
    elif mutation == "base-toolchain":
        transition.base_toolchain.write_bytes(
            transition.base_toolchain.read_bytes() + b"# uncommitted evidence drift\n"
        )
    elif mutation == "head-pair":
        _materialize_blob(
            transition.repository,
            transition.base_commit,
            _WAIVER_PATH,
            transition.head_worktree / _WAIVER_PATH,
        )
        _materialize_blob(
            transition.repository,
            transition.base_commit,
            _TOOLCHAIN_PATH,
            transition.head_worktree / _TOOLCHAIN_PATH,
        )
    elif mutation == "changed-paths":
        transition.changed_paths.write_bytes(
            transition.changed_paths.read_bytes() + b"README.md\0"
        )

    exit_code, report = _audit(transition, monkeypatch, capsys)

    assert exit_code == 1
    assert report["success"] is False
    assert report["checked"] == 0
    assert report["errors"]


def test_real_git_rejects_same_semantics_with_different_waiver_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    transition = _build_git_transition(tmp_path, "semantic-noop")

    exit_code, report = _audit(transition, monkeypatch, capsys)

    assert exit_code == 1
    assert report["success"] is False
    assert report["checked"] == 0
    assert any("semantic no-op" in error for error in report["errors"])


@pytest.mark.parametrize(("phase", "evidence_label"), _RACE_CASES)
def test_real_git_rejects_deterministic_same_byte_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    phase: str,
    evidence_label: str,
):
    transition = _build_git_transition(tmp_path, phase)
    evidence_path = transition.evidence_paths[evidence_label]
    replace_now = threading.Event()
    replacement_done = threading.Event()
    replacement_errors: list[BaseException] = []

    def replace_with_same_bytes() -> None:
        try:
            if not replace_now.wait(timeout=10):
                raise AssertionError("audit never reached the post-snapshot executor")
            replacement = evidence_path.with_name(f"{evidence_path.name}.replacement")
            replacement.write_bytes(evidence_path.read_bytes())
            os.replace(replacement, evidence_path)
        except BaseException as exc:  # surfaced deterministically in the test thread
            replacement_errors.append(exc)
        finally:
            replacement_done.set()

    race = threading.Thread(target=replace_with_same_bytes, daemon=True)
    race.start()

    def execute_after_all_snapshots(*_args, **_kwargs):
        replace_now.set()
        assert replacement_done.wait(timeout=10)
        return list(transition.expected_execution)

    try:
        exit_code, report = _audit(
            transition,
            monkeypatch,
            capsys,
            executor=execute_after_all_snapshots,
        )
    finally:
        replace_now.set()
        race.join(timeout=10)

    assert not race.is_alive()
    assert not replacement_errors
    assert exit_code == 1
    assert report["success"] is False
    assert any(
        f"{evidence_label} changed after its audit snapshot" in error
        for error in report["errors"]
    )

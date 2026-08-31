"""CLI integration tests for atomic legacy-cleanup guard admission."""

from __future__ import annotations

import base64
import json
import os
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import axiom_encode.cli as cli
from axiom_encode.legacy_cleanup import canonical_receipt_bytes
from tests.test_legacy_cleanup_guard import (
    PUBLIC_KEY,
    _atomic_cleanup,
    _base_repo,
    _commit,
    _delete_group,
    _orphan_receipt_base,
    _plan,
    _sign,
    _unsigned_payload,
    _write_receipt,
)


def _install_guard_inputs(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    *,
    verifier: object = PUBLIC_KEY,
) -> None:
    """Keep these tests on the cleanup branch while exercising its real guard."""

    monkeypatch.setattr(cli, "load_rulespec_toolchain", lambda _repo: object())
    monkeypatch.setattr(
        cli,
        "verify_rulespec_validation_waiver_set",
        lambda _repo: "e" * 64,
    )
    monkeypatch.setattr(
        cli,
        "load_rulespec_local_corpus_release",
        lambda _repo, _corpus: object(),
    )
    monkeypatch.setattr(cli, "_applied_encoding_manifest_verifier", lambda: verifier)
    monkeypatch.setattr(
        cli,
        "_legacy_cleanup_toolchain_binding",
        lambda **_kwargs: (deepcopy(payload["toolchain"]), object(), ()),
    )


def _guard_atomic_commit(
    repo: Path,
    *,
    base: str,
    head: str,
    tmp_path: Path,
) -> list[str]:
    return cli.guard_generated_change_issues(
        repo,
        corpus_path=tmp_path / "axiom-corpus",
        base_ref=base,
        head_ref=head,
        expected_encoder_checkout=tmp_path / "axiom-encode",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
    )


def test_guard_generated_accepts_exact_atomic_cleanup_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, _receipt_path = _atomic_cleanup(repo, base)
    _install_guard_inputs(monkeypatch, payload)

    assert _guard_atomic_commit(
        repo,
        base=base,
        head=head,
        tmp_path=tmp_path,
    ) == []


@pytest.mark.parametrize(
    ("missing", "expected"),
    [
        ("base", "requires an exact protected --base-ref"),
        ("encoder", "requires --expected-encoder-checkout"),
        ("engine", "requires --axiom-rules-engine-path"),
        ("trust", "requires the protected apply signature verification key"),
    ],
)
def test_guard_generated_cleanup_requires_every_admission_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
    expected: str,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, receipt_path = _atomic_cleanup(repo, base)
    _install_guard_inputs(
        monkeypatch,
        payload,
        verifier=None if missing == "trust" else PUBLIC_KEY,
    )
    arguments = {
        "repo_path": repo,
        "corpus_path": tmp_path / "axiom-corpus",
        "base_ref": None if missing == "base" else base,
        "head_ref": head,
        "changed_files": [receipt_path.as_posix()],
        "expected_encoder_checkout": (
            None if missing == "encoder" else tmp_path / "axiom-encode"
        ),
        "axiom_rules_path": (
            None if missing == "engine" else tmp_path / "axiom-rules-engine"
        ),
    }

    issues = cli.guard_generated_change_issues(**arguments)

    assert len(issues) == 1
    assert expected in issues[0]


def test_guard_generated_rejects_altered_cleanup_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _base_repo(tmp_path)
    payload = _sign(_unsigned_payload(_plan(repo, base)))
    signature = payload["signature"]
    assert isinstance(signature, dict)
    signature["value"] = base64.b64encode(b"\0" * 64).decode("ascii")
    _write_receipt(repo, payload)
    _delete_group(repo)
    head = _commit(repo, "altered cleanup receipt")
    _install_guard_inputs(monkeypatch, payload)

    issues = _guard_atomic_commit(repo, base=base, head=head, tmp_path=tmp_path)

    assert any("receipt signature is invalid" in issue for issue in issues)


def test_guard_generated_rejects_replayed_historical_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base, payload, historical_path = _orphan_receipt_base(tmp_path)
    replay_path = historical_path.with_name(f"{'f' * 64}.json")
    assert replay_path != historical_path
    target = repo / replay_path
    target.write_bytes(canonical_receipt_bytes(payload))
    os.chmod(target, 0o644)
    _delete_group(repo)
    head = _commit(repo, "replay historical cleanup receipt")
    _install_guard_inputs(monkeypatch, payload)

    issues = _guard_atomic_commit(repo, base=base, head=head, tmp_path=tmp_path)

    assert any("receipt path does not match its identity" in issue for issue in issues)


@pytest.mark.parametrize(
    ("delete", "extra", "expected"),
    [
        ("none", False, "missing exact changes"),
        ("both", True, "extra or mixed changes"),
    ],
    ids=["orphan", "mixed"],
)
def test_guard_generated_rejects_nonatomic_cleanup_change_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    delete: str,
    extra: bool,
    expected: str,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, _receipt_path = _atomic_cleanup(
        repo,
        base,
        delete=delete,
        extra=extra,
    )
    _install_guard_inputs(monkeypatch, payload)

    issues = _guard_atomic_commit(repo, base=base, head=head, tmp_path=tmp_path)

    assert any(expected in issue for issue in issues)


def test_guard_generated_rejects_historical_receipt_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base, payload, historical_path = _orphan_receipt_base(tmp_path)
    target = repo / historical_path
    target.write_bytes(target.read_bytes().replace(b"20:00:00", b"20:00:01"))
    head = _commit(repo, "mutate historical cleanup receipt")
    _install_guard_inputs(monkeypatch, payload)

    issues = _guard_atomic_commit(repo, base=base, head=head, tmp_path=tmp_path)

    assert any("historical cleanup receipt was modified" in issue for issue in issues)


def test_guard_generated_rejects_uncommitted_path_over_atomic_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base = _base_repo(tmp_path)
    head, payload, _receipt_path = _atomic_cleanup(repo, base)
    _install_guard_inputs(monkeypatch, payload)
    (repo / "unrelated.txt").write_text("mixed worktree state\n", encoding="utf-8")

    issues = _guard_atomic_commit(repo, base=base, head=head, tmp_path=tmp_path)

    assert any("uncommitted, omitted, or mixed" in issue for issue in issues)


def test_guard_generated_parser_threads_cleanup_admission_paths(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    corpus = tmp_path / "axiom-corpus"
    encoder = tmp_path / "axiom-encode"
    engine = tmp_path / "axiom-rules-engine"
    dependency = tmp_path / "rulespec-eu"
    argv = [
        "axiom-encode",
        "guard-generated",
        "--repo",
        str(repo),
        "--corpus-path",
        str(corpus),
        "--base-ref",
        "a" * 40,
        "--head-ref",
        "b" * 40,
        "--expected-encoder-checkout",
        str(encoder),
        "--axiom-rules-engine-path",
        str(engine),
        "--rulespec-dependency-root",
        str(dependency),
    ]

    with (
        patch("sys.argv", argv),
        patch("axiom_encode.cli.cmd_guard_generated") as command,
    ):
        cli.main()

    args = command.call_args.args[0]
    assert args.repo == repo
    assert args.corpus_path == corpus
    assert args.base_ref == "a" * 40
    assert args.head_ref == "b" * 40
    assert args.expected_encoder_checkout == encoder
    assert args.axiom_rules_path == engine
    assert args.rulespec_dependency_root == [dependency]


def test_cmd_guard_generated_forwards_cleanup_admission_and_reports_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "rulespec-be"
    corpus = tmp_path / "axiom-corpus"
    encoder = tmp_path / "axiom-encode"
    engine = tmp_path / "axiom-rules-engine"
    dependency = tmp_path / "rulespec-eu"
    args = SimpleNamespace(
        repo=repo,
        corpus_path=corpus,
        base_ref="a" * 40,
        head_ref="b" * 40,
        all=False,
        expected_encoder_checkout=encoder,
        axiom_rules_path=engine,
        rulespec_dependency_root=[dependency],
        json=True,
    )

    with (
        patch(
            "axiom_encode.cli._resolve_canonical_rulespec_checkout",
            return_value=repo,
        ),
        patch(
            "axiom_encode.cli.guard_generated_change_issues",
            return_value=[],
        ) as guard,
        pytest.raises(SystemExit) as exc_info,
    ):
        cli.cmd_guard_generated(args)

    assert exc_info.value.code == 0
    guard.assert_called_once_with(
        repo,
        corpus_path=corpus,
        base_ref="a" * 40,
        head_ref="b" * 40,
        roots=tuple(sorted(cli.RULESPEC_ATOMIC_MODULE_ROOTS)),
        all_files=False,
        expected_encoder_checkout=encoder,
        axiom_rules_path=engine,
        rulespec_dependency_roots=(dependency,),
    )
    assert json.loads(capsys.readouterr().out) == {
        "repo": str(repo),
        "passed": True,
        "issues": [],
    }


def test_stage_signed_backfill_parser_threads_cleanup_inputs(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-be"
    corpus = tmp_path / "axiom-corpus"
    encoder = tmp_path / "axiom-encode"
    engine = tmp_path / "axiom-rules-engine"
    dependency = tmp_path / "rulespec-eu"
    argv = [
        "axiom-encode",
        "stage-signed-backfill",
        "--repo",
        str(repo),
        "--corpus-path",
        str(corpus),
        "--legacy-cleanup-base-ref",
        "a" * 40,
        "--expected-encoder-checkout",
        str(encoder),
        "--axiom-rules-engine-path",
        str(engine),
        "--rulespec-dependency-root",
        str(dependency),
    ]

    with (
        patch("sys.argv", argv),
        patch("axiom_encode.cli.cmd_stage_signed_backfill") as command,
    ):
        cli.main()

    args = command.call_args.args[0]
    assert args.legacy_cleanup_base_ref == "a" * 40
    assert args.expected_encoder_checkout == encoder
    assert args.axiom_rules_path == engine
    assert args.rulespec_dependency_root == [dependency]


def test_cmd_stage_signed_backfill_reverifies_cleanup_before_staging(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "rulespec-be"
    corpus = tmp_path / "axiom-corpus"
    encoder = tmp_path / "axiom-encode"
    engine = tmp_path / "axiom-rules-engine"
    dependency = tmp_path / "rulespec-eu"
    base = "a" * 40
    verifier = object()
    receipt = {"groups": []}
    expected_toolchain = {"bound": True}
    args = SimpleNamespace(
        repo=repo,
        corpus_path=corpus,
        legacy_cleanup_base_ref=base,
        expected_encoder_checkout=encoder,
        axiom_rules_path=engine,
        rulespec_dependency_root=[dependency],
    )

    with (
        patch(
            "axiom_encode.cli._resolve_canonical_rulespec_checkout",
            return_value=repo,
        ),
        patch(
            "axiom_encode.cli._applied_encoding_manifest_verifier",
            return_value=verifier,
        ),
        patch(
            "axiom_encode.cli.verify_worktree_legacy_cleanup_transition",
            return_value=SimpleNamespace(issues=(), receipt=receipt),
        ) as preliminary,
        patch(
            "axiom_encode.cli._legacy_cleanup_expected_toolchain_from_receipt",
            return_value=expected_toolchain,
        ) as bind,
        patch(
            "axiom_encode.prepare_signed_backfill.stage_authorized_changes"
        ) as stage,
    ):
        cli.cmd_stage_signed_backfill(args)

    preliminary.assert_called_once_with(
        repo,
        base_ref=base,
        verifier=verifier,
    )
    bind.assert_called_once_with(
        repo_path=repo,
        base_ref=base,
        receipt=receipt,
        corpus_path=corpus,
        encoder_checkout=encoder,
        rules_engine_checkout=engine,
        dependency_roots=(dependency,),
        provenance_verifier=verifier,
    )
    stage.assert_called_once_with(
        repo,
        corpus_root=corpus,
        legacy_cleanup_base_ref=base,
        legacy_cleanup_verifier=verifier,
        legacy_cleanup_expected_toolchain=expected_toolchain,
    )

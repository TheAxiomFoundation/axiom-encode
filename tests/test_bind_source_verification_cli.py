"""`axiom-encode bind-source-verification` and the encode outcome's binding."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli
from tests.test_evals import _write_test_source_unit

CITATION = "us-zz/manual/unit/chapter/block-1"
MODULE_PATH = Path("us-zz/policies/unit/chapter/block-1.yaml")
UNBOUND = (
    "format: rulespec/v1\n"
    "module:\n"
    "  source_verification:\n"
    f"    corpus_citation_path: {CITATION}\n"
    "  summary: |-\n"
    "    Synthetic text.\n"
    "rules: []\n"
)


@pytest.fixture
def world(tmp_path):
    release, source_unit = _write_test_source_unit(
        tmp_path, "Authoritative synthetic source.", citation_path=CITATION
    )
    repo = tmp_path / "rulespec-us"
    module = repo / MODULE_PATH
    module.parent.mkdir(parents=True)
    module.write_bytes(UNBOUND.encode())
    corpus = tmp_path / "corpus-checkout"
    corpus.mkdir()
    pin = source_unit.source_attestation["source_sha256"]
    bound = UNBOUND.replace(
        f"    corpus_citation_path: {CITATION}\n",
        f"    corpus_citation_path: {CITATION}\n    source_sha256: {pin}\n",
    )
    with patch.object(
        cli, "load_rulespec_local_corpus_release", return_value=release
    ) as load:
        yield SimpleNamespace(
            repo=repo,
            corpus=corpus,
            module=module,
            pin=pin,
            bound=bound.encode(),
            release=release,
            load=load,
        )


def run(world, *extra: str) -> int:
    argv = [
        "axiom-encode",
        "bind-source-verification",
        "--repo",
        str(world.repo),
        "--citation",
        CITATION,
        "--corpus-path",
        str(world.corpus),
        *extra,
    ]
    with patch("sys.argv", argv), pytest.raises(SystemExit) as caught:
        cli.main()
    return caught.value.code


def test_binds_the_module_and_prints_both_digests_and_the_one_line_diff(world, capsys):
    world.module.chmod(0o640)
    assert run(world) == 0
    assert world.module.read_bytes() == world.bound
    assert stat.S_IMODE(world.module.stat().st_mode) == 0o640
    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"citation={CITATION}",
        f"module={MODULE_PATH.as_posix()}",
        f"corpus_release={world.release.name} ({world.release.content_sha256})",
        f"attested_source_sha256={world.pin}",
        f"old_sha256={hashlib.sha256(UNBOUND.encode()).hexdigest()}",
        f"new_sha256={hashlib.sha256(world.bound).hexdigest()}",
        f"diff=+    source_sha256: {world.pin}",
        "status=bound:insert_source_sha256",
    ]
    # The release is the one the checkout pins, read as a promotion reads it.
    world.load.assert_called_once_with(world.repo.resolve(), world.corpus.resolve())
    # Only the module file changed in the checkout.
    assert [p for p in world.repo.rglob("*") if p.is_file()] == [world.module]


def test_check_reports_an_unbound_module_and_writes_nothing(world, capsys):
    assert run(world, "--check") == 1
    assert world.module.read_bytes() == UNBOUND.encode()
    out = capsys.readouterr().out
    assert f"diff=+    source_sha256: {world.pin}" in out
    assert out.splitlines()[-1] == "status=unbound"


def test_a_bound_module_is_left_alone_with_and_without_check(world, capsys):
    world.module.write_bytes(world.bound)
    before = world.module.stat().st_mtime_ns
    for extra in ((), ("--check",)):
        assert run(world, *extra) == 0
        out = capsys.readouterr().out
        digest = hashlib.sha256(world.bound).hexdigest()
        assert f"old_sha256={digest}" in out and f"new_sha256={digest}" in out
        assert "diff=" not in out
        assert out.splitlines()[-1] == "status=already_bound"
    assert world.module.read_bytes() == world.bound
    assert world.module.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "block, status, detail",
    [
        (
            f"    corpus_citation_path: {CITATION}\n    source_sha256: {'0e' * 32}\n",
            "status=refused:conflict",
            "claims another source",
        ),
        (
            "    corpus_citation_path: us-zz/manual/unit/chapter/block-2\n",
            "status=refused:conflict",
            "claims another source",
        ),
        (
            f"    <<: {{corpus_citation_path: {CITATION}}}\n",
            "status=refused:unsupported_shape",
            "merge key",
        ),
    ],
)
@pytest.mark.parametrize("check", [(), ("--check",)])
def test_a_refusal_writes_nothing_and_exits_1(
    world, capsys, block, status, detail, check
):
    text = UNBOUND.replace(f"    corpus_citation_path: {CITATION}\n", block)
    world.module.write_bytes(text.encode())
    assert run(world, *check) == 1
    assert world.module.read_bytes() == text.encode()
    out = capsys.readouterr().out
    assert status in out.splitlines()
    assert detail in out
    assert "new_sha256=" not in out


def test_a_missing_module_or_a_symlink_is_refused(world):
    target = world.module.with_name("real.yaml")
    world.module.rename(target)
    with pytest.raises(ValueError, match="not a regular file in the checkout"):
        run(world)
    world.module.symlink_to(target)
    with pytest.raises(ValueError, match="not a regular file in the checkout"):
        run(world)
    assert target.read_bytes() == UNBOUND.encode()


def test_a_citation_that_is_not_canonical_is_refused_before_any_read(world):
    argv = [
        "axiom-encode",
        "bind-source-verification",
        "--repo",
        str(world.repo),
        "--citation",
        " " + CITATION,
        "--corpus-path",
        str(world.corpus),
    ]
    with patch("sys.argv", argv), pytest.raises(ValueError):
        cli.main()
    world.load.assert_not_called()


def test_the_command_has_no_model_signing_or_network_option():
    with patch("sys.argv", ["axiom-encode", "bind-source-verification", "--help"]):
        with pytest.raises(SystemExit):
            cli.main()
    import inspect

    source = inspect.getsource(cli._bind_source_verification_in_checkout)
    for forbidden in ("run_model_eval", "signing_broker", "urllib", "subprocess"):
        assert forbidden not in source


def test_verification_only_roots_wrap_the_resolution(world):
    seen = []

    class Context:
        def __enter__(self):
            seen.append("enter")

        def __exit__(self, *_exc):
            seen.append("exit")
            return False

    with (
        patch.object(
            cli,
            "local_corpus_release_verification",
            side_effect=lambda *a, **k: seen.append((a, k)) or Context(),
        ),
        patch.object(cli, "protected_process_signal_active", return_value=False),
    ):
        assert run(world, "--corpus-release-public-key", "KEY", "--check") == 1
    assert seen[0] == (
        ("KEY",),
        {"retired_public_keys": (), "apply_public_key": None, "eval_public_key": None},
    )
    assert seen[1:] == ["enter", "exit"]


# --- the outcome of a non-apply encode ---


def _result(tmp_path, text: str | None, *, success: bool = True, pin: str = "ab" * 32):
    output = ""
    if text is not None:
        file = tmp_path / "candidate.yaml"
        file.write_bytes(text.encode())
        output = str(file)
    return SimpleNamespace(
        output_file=output,
        success=success,
        error=None if success else "Generated RuleSpec failed CI validation",
        source_attestation={
            "requested_corpus_citation_path": CITATION,
            "resolved_corpus_citation_path": CITATION,
            "source_sha256": pin,
        },
    )


def _outcome(result) -> dict:
    outcome = cli._initial_encode_outcome(result, apply_requested=False)
    cli._record_encode_source_binding(result, outcome)
    return outcome


def test_a_bound_candidate_is_recorded_as_bound(tmp_path, capsys):
    text = UNBOUND.replace(
        f"    corpus_citation_path: {CITATION}\n",
        f"    corpus_citation_path: {CITATION}\n    source_sha256: {'ab' * 32}\n",
    )
    result = _result(tmp_path, text)
    outcome = _outcome(result)
    assert outcome["source_binding"] == "bound"
    assert outcome["status"] == "standalone_validated"
    assert outcome["final_success"] is True
    assert result.success is True
    assert "source_binding_error" not in outcome
    assert capsys.readouterr().out == "  source_binding=bound\n"


def test_an_unbound_clean_candidate_is_never_a_success(tmp_path, capsys):
    result = _result(tmp_path, UNBOUND)
    outcome = _outcome(result)
    assert outcome["source_binding"] == "unbound"
    assert outcome["status"] == "source_unbound"
    assert outcome["final_success"] is False
    assert outcome["source_binding_error"].startswith("unbound: ")
    assert result.success is False
    assert result.error == outcome["primary_error"]
    assert "not bound to its resolver-attested source" in result.error
    assert capsys.readouterr().out == (
        "  outcome=source_unbound final_success=False source_binding=unbound\n"
    )


@pytest.mark.parametrize(
    "block, kind",
    [
        (
            f"    corpus_citation_path: {CITATION}\n    source_sha256: {'0e' * 32}\n",
            "conflict",
        ),
        (f"    <<: {{corpus_citation_path: {CITATION}}}\n", "unsupported_shape"),
    ],
)
def test_a_failed_try_that_could_not_bind_gets_the_binding_status(
    tmp_path, block, kind
):
    text = UNBOUND.replace(f"    corpus_citation_path: {CITATION}\n", block)
    result = _result(tmp_path, text, success=False)
    outcome = _outcome(result)
    assert outcome["status"] == "source_unbound"
    assert outcome["source_binding"] == "unbound"
    assert outcome["source_binding_error"].startswith(kind + ": ")
    assert outcome["final_success"] is False
    # The validator verdict of the try is kept for the retry loop.
    assert result.error == "Generated RuleSpec failed CI validation"


def test_unparseable_output_keeps_the_validation_status(tmp_path):
    result = _result(tmp_path, "module: [unclosed\n", success=False)
    outcome = _outcome(result)
    assert outcome["status"] == "standalone_failed"
    assert outcome["source_binding"] == "unbound"
    assert outcome["final_success"] is False


def test_no_candidate_file_is_recorded_as_such(tmp_path):
    result = _result(tmp_path, None, success=False)
    outcome = _outcome(result)
    assert outcome["source_binding"] == "no_candidate"
    assert outcome["status"] == "standalone_failed"


def test_a_result_without_an_attestation_is_unbound(tmp_path):
    result = _result(tmp_path, UNBOUND)
    result.source_attestation = None
    outcome = _outcome(result)
    assert outcome["source_binding"] == "unbound"
    assert outcome["final_success"] is False
    assert result.success is False

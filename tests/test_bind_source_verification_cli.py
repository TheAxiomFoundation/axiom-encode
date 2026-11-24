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
        "warning=NO --expect-source-sha256: this pin attests the source as "
        f"corpus release {world.release.name} stores it now. Compare "
        "attested_source_sha256 with the source attestation of the encode "
        "run before you push; if they differ, encode the candidate again",
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


def test_a_missing_module_or_a_symlink_is_a_clean_error(world, capsys):
    target = world.module.with_name("real.yaml")
    world.module.rename(target)
    assert run(world) == 2
    world.module.symlink_to(target)
    assert run(world) == 2
    out = capsys.readouterr().out
    assert out.count("status=error") == 2
    assert out.count("error=candidate RuleSpec is not a regular file") == 2
    assert target.read_bytes() == UNBOUND.encode()


def test_a_citation_that_is_not_canonical_is_a_clean_error_before_any_read(
    world, capsys
):
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
    with patch("sys.argv", argv), pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert "status=error" in capsys.readouterr().out
    world.load.assert_not_called()


def test_a_wrong_checkout_or_a_symlinked_path_is_a_clean_error(world, capsys, tmp_path):
    """No traceback for a checkout the toolchain refuses or a symlinked path."""
    link = tmp_path / "link-to-repo"
    link.symlink_to(world.repo)
    world.repo = link
    assert run(world) == 2
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "status=error"
    assert out[1].startswith("error=RuleSpec checkout path contains a symlink")
    world.repo = link.resolve()
    from axiom_encode.toolchain import RuleSpecToolchainError

    world.load.side_effect = RuleSpecToolchainError("checkout is not canonical")
    assert run(world) == 2
    assert capsys.readouterr().out.splitlines() == [
        "status=error",
        "error=checkout is not canonical",
    ]
    assert world.module.read_bytes() == UNBOUND.encode()


def test_expect_source_sha256_that_matches_binds_without_the_warning(world, capsys):
    assert run(world, "--expect-source-sha256", world.pin) == 0
    assert world.module.read_bytes() == world.bound
    out = capsys.readouterr().out
    assert "warning=" not in out
    assert f"expected_source_sha256={world.pin}" in out.splitlines()
    assert out.splitlines()[-1] == "status=bound:insert_source_sha256"


@pytest.mark.parametrize("check", [(), ("--check",)])
@pytest.mark.parametrize("start", ["unbound", "bound"])
def test_expect_source_sha256_refuses_a_source_that_changed(
    world, capsys, check, start
):
    """The checkout's corpus attests another body than the encode run recorded:
    never re-pin, and never report a stale pin of the new body as bound."""
    before = UNBOUND.encode() if start == "unbound" else world.bound
    world.module.write_bytes(before)
    assert run(world, "--expect-source-sha256", "0e" * 32, *check) == 1
    assert world.module.read_bytes() == before
    out = capsys.readouterr().out.splitlines()
    assert "status=refused:source_changed" in out
    assert f"expected_source_sha256={'0e' * 32}" in out
    assert out[-1].startswith("error=the corpus release pinned by this checkout")
    assert not any(line.startswith(("new_sha256=", "diff=")) for line in out)


@pytest.mark.parametrize("value", ["0E" * 32, "0e" * 31, "sha256:" + "0e" * 32, ""])
def test_expect_source_sha256_must_be_a_digest(world, capsys, value):
    assert run(world, "--expect-source-sha256", value) == 2
    assert "--expect-source-sha256 must be 64 lowercase" in capsys.readouterr().out
    assert world.module.read_bytes() == UNBOUND.encode()
    world.load.assert_not_called()


@pytest.mark.parametrize("check", [(), ("--check",)])
def test_right_values_outside_the_plain_block_form_are_never_reported_bound(
    world, capsys, check
):
    """What the command calls bound, the admission reader accepts."""
    from axiom_encode import reviewed_pr_admission

    for text in (b"---\n" + world.bound, b"---\n" + UNBOUND.encode()):
        world.module.write_bytes(text)
        assert run(world, *check) == 1
        assert world.module.read_bytes() == text
        out = capsys.readouterr().out
        assert "status=refused:not_plain_form" in out.splitlines()
        assert "not in the plain block form" in out
        assert "already_bound" not in out and "status=bound" not in out
    world.module.write_bytes(UNBOUND.encode())
    assert run(world) == 0
    reviewed_pr_admission.require_candidate_source_pin(
        world.module.read_bytes(), CITATION
    )


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


def test_a_clean_result_whose_recorded_file_cannot_be_read_is_unbound(tmp_path, capsys):
    result = _result(tmp_path, UNBOUND)
    result.output_file = str(tmp_path / "absent.yaml")
    result.generated_output_sha256 = "ab" * 32
    outcome = _outcome(result)
    assert outcome["source_binding"] == "unbound"
    assert outcome["source_binding_error"].startswith("unreadable: ")
    assert outcome["status"] == "source_unbound"
    assert outcome["final_success"] is False
    assert result.success is False
    assert "source_binding=unbound" in capsys.readouterr().out


@pytest.mark.parametrize(
    "success, recorded",
    [(False, "ab" * 32), (True, None), (True, "not-a-digest"), (True, object())],
)
def test_only_a_failed_or_digestless_result_is_unverified(
    tmp_path, capsys, success, recorded
):
    result = _result(tmp_path, UNBOUND, success=success)
    result.output_file = str(tmp_path / "absent.yaml")
    result.generated_output_sha256 = recorded
    outcome = _outcome(result)
    # `unverified` is its own state: it is never recorded as `bound`.
    assert outcome["source_binding"] == "unverified"
    assert "source_binding_error" not in outcome
    assert outcome["final_success"] is success
    assert result.success is success
    assert capsys.readouterr().out == "  source_binding=unverified\n"

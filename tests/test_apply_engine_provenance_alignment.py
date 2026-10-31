"""An explicit apply binding must agree with its portable checkout attribution."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from axiom_encode import cli
from axiom_encode.engine_binding import (
    EnginePin,
    resolve_pinned_engine_binary,
    write_engine_binding_receipt,
)
from tests.test_engine_binding import _drift_commit, _engine_checkout, _git


@pytest.fixture
def engines(tmp_path):
    checkout, old = _engine_checkout(tmp_path)
    _git(
        checkout,
        "remote",
        "add",
        "origin",
        "https://github.com/TheAxiomFoundation/axiom-rules-engine.git",
    )
    debug = checkout / "target/debug/axiom-rules-engine"
    debug.parent.mkdir(parents=True)
    debug.write_bytes(b"stub binary A")
    debug.chmod(0o755)
    write_engine_binding_receipt(debug, old, "debug")
    current = _drift_commit(checkout)
    release = checkout / "target/release/axiom-rules-engine"
    release.parent.mkdir(parents=True)
    release.write_bytes(b"stub binary B")
    release.chmod(0o755)
    write_engine_binding_receipt(release, current, "release")
    assert _git(checkout, "status", "--porcelain") == ""
    return checkout, old, current, debug, release


def _execution(tmp_path, engines):
    checkout, _, current, _, release = engines
    policy = tmp_path / "rulespec-ca/ca"
    policy.mkdir(parents=True, exist_ok=True)
    return {
        "axiom_encode_identity": {
            "kind": "git",
            "dirty": False,
            "path": str(tmp_path / "encoder"),
            "commit": "c" * 40,
            "version": "0.0.0",
            "origin_repository": "github.com/TheAxiomFoundation/axiom-encode",
        },
        "axiom_rules_identity": cli._git_checkout_execution_identity(checkout),
        "axiom_rules_engine_binding": {
            "ref": current,
            "binary": str(release),
            "sha256": cli._sha256_file(release),
        },
        "policy_root_identity": {
            "path": str(policy),
            "content_sha256": "d" * 64,
            "file_count": 0,
            "toolchain_contract_sha256": "e" * 64,
            "validation_waiver_set_sha256": "f" * 64,
        },
        "rulespec_dependency_identities": [],
    }


def test_apply_rejects_receipted_old_binary_before_resolver_or_build(engines):
    checkout, old, current, debug, release = engines
    # The general receipt-first resolver remains valid for non-apply callers.
    assert (
        resolve_pinned_engine_binary(
            checkout, EnginePin(old, Path("fixture")), allow_build=False
        )
        == debug
    )
    with patch.object(
        cli, "resolve_pinned_engine_binary", wraps=resolve_pinned_engine_binary
    ) as resolve:
        with pytest.raises(
            cli.EngineBindingError, match="binding revision must match.*HEAD"
        ):
            cli._pinned_apply_engine_binding(checkout, old)
    resolve.assert_not_called()
    binding = cli._pinned_apply_engine_binding(checkout, current)
    assert binding == {
        "ref": current,
        "binary": str(release),
        "sha256": cli._sha256_file(release),
    }


def test_portable_rejects_independently_constructed_revision_disagreement(
    tmp_path, engines
):
    execution = _execution(tmp_path, engines)
    execution["axiom_rules_engine_binding"]["ref"] = engines[1]
    with pytest.raises(RuntimeError, match="binding revision does not match portable"):
        cli._portable_apply_validation_execution_identity(
            execution, replacement_overlay_scope=False
        )


def test_matching_and_absent_binding_preserve_portable_schema(tmp_path, engines):
    execution = _execution(tmp_path, engines)
    bound = cli._portable_apply_validation_execution_identity(
        execution, replacement_overlay_scope=False
    )
    del execution["axiom_rules_engine_binding"]
    legacy = cli._portable_apply_validation_execution_identity(
        execution, replacement_overlay_scope=False
    )
    assert bound == legacy
    assert bound["axiom_rules_engine"]["commit"] == engines[2]
    assert set(bound) == {
        "schema",
        "rulespec_scope",
        "axiom_encode",
        "axiom_rules_engine",
        "policy_pre_apply",
        "rulespec_dependencies",
    }


@pytest.mark.parametrize(
    "malformation",
    [None, [], {}, "bad", "bad_ref", "relative_binary", "bad_hash", "extra"],
)
def test_portable_present_malformed_binding_never_downgrades(
    tmp_path, engines, malformation
):
    execution = _execution(tmp_path, engines)
    binding = execution["axiom_rules_engine_binding"]
    if malformation == "bad_ref":
        binding["ref"] = "main"
    elif malformation == "relative_binary":
        binding["binary"] = "relative"
    elif malformation == "bad_hash":
        binding["sha256"] = "bad"
    elif malformation == "extra":
        binding["extra"] = True
    else:
        execution["axiom_rules_engine_binding"] = malformation
    with pytest.raises(RuntimeError, match="malformed engine binding"):
        cli._portable_apply_validation_execution_identity(
            execution, replacement_overlay_scope=False
        )


def test_legacy_helper_does_not_read_checkout_or_resolve():
    with (
        patch.object(cli, "_git_checkout_execution_identity") as identity,
        patch.object(cli, "resolve_pinned_engine_binary") as resolve,
    ):
        assert cli._pinned_apply_engine_binding(Path("unused"), None) is None
    identity.assert_not_called()
    resolve.assert_not_called()


def test_refresh_overlay_mismatch_precedes_pipeline(engines, tmp_path):
    checkout, old, _, _, _ = engines
    result = SimpleNamespace()
    with patch.object(cli, "ValidatorPipeline") as pipeline:
        with pytest.raises(
            cli.EngineBindingError, match="binding revision must match.*HEAD"
        ):
            cli._validate_generated_encoding_in_policy_overlay_with_release(
                result,
                output_root=tmp_path,
                policy_repo_path=tmp_path,
                axiom_rules_path=checkout,
                axiom_rules_engine_ref=old,
                local_corpus_release=object(),
            )
    pipeline.assert_not_called()
    assert getattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR) is None


def test_native_attempt_mismatch_precedes_generation_and_signing(engines, tmp_path):
    checkout, old, _, _, _ = engines
    args = SimpleNamespace(
        apply=True,
        axiom_rules_engine_ref=old,
        backend="openai",
        rulespec_dependency_roots=(),
        axiom_compose_path=None,
        corpus_path=tmp_path,
        axiom_rules_path=checkout,
    )
    signer = MagicMock(spec=cli.SigningBroker)
    with patch.object(cli, "run_model_eval") as generate:
        with pytest.raises(
            cli.EngineBindingError, match="binding revision must match.*HEAD"
        ):
            cli._run_encode_attempt(args, model="unused", apply_signing_broker=signer)
    generate.assert_not_called()
    assert signer.mock_calls == []


def test_matching_apply_snapshot_and_portable_identity_agree(tmp_path, engines):
    expected = _execution(tmp_path, engines)
    policy = Path(expected["policy_root_identity"]["path"])
    with (
        patch.object(
            cli,
            "_apply_encoder_execution_identity",
            return_value=expected["axiom_encode_identity"],
        ),
        patch.object(
            cli,
            "_rulespec_root_execution_identity",
            return_value=expected["policy_root_identity"],
        ),
    ):
        execution = cli._apply_validation_execution_identity(
            axiom_rules_path=engines[0],
            policy_repo_path=policy,
            relative_output=Path("rule.yaml"),
            rulespec_dependency_roots=(),
            axiom_rules_engine_ref=engines[2],
        )
    assert execution["axiom_rules_engine_binding"]["ref"] == engines[2]
    assert execution["axiom_rules_identity"]["commit"] == engines[2]
    portable = cli._portable_apply_validation_execution_identity(
        execution, replacement_overlay_scope=False
    )
    assert portable["axiom_rules_engine"]["commit"] == engines[2]


def test_explicit_apply_requires_checkout_even_with_bare_receipt(tmp_path):
    binary = tmp_path / "axiom-rules-engine"
    binary.write_bytes(b"prebuilt receipt fixture")
    binary.chmod(0o755)
    ref = "a" * 40
    write_engine_binding_receipt(binary, ref, "prebuilt")
    assert (
        resolve_pinned_engine_binary(
            tmp_path, EnginePin(ref, Path("fixture")), allow_build=False
        )
        == binary
    )
    with patch.object(cli, "resolve_pinned_engine_binary") as resolve:
        with pytest.raises(
            cli.EngineBindingError, match="binding revision must match.*HEAD"
        ):
            cli._pinned_apply_engine_binding(tmp_path, ref)
    resolve.assert_not_called()

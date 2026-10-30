"""Explicit refresh pin must survive validation snapshot and apply rechecks."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from axiom_encode import cli


@pytest.mark.parametrize("ref", [None, "a" * 40])
def test_successful_overlay_snapshot_accepts_explicit_or_legacy_binding(tmp_path, ref):
    binding = {"ref": ref, "binary": str(tmp_path / "engine"), "sha256": "b" * 64}
    execution = {"axiom_rules_engine_binding": binding} if ref else {}
    result = SimpleNamespace()
    with (
        patch.object(
            cli, "_apply_validation_execution_identity", return_value=execution
        ) as identity,
        patch.object(
            cli,
            "_build_apply_validation_snapshot",
            side_effect=lambda *a, **kw: kw["validation_execution_identity"],
        ),
    ):
        cli._record_successful_apply_validation(
            result,
            output_root=tmp_path,
            policy_repo_path=tmp_path,
            relative_output=Path("rule.yaml"),
            supplemental_files={},
            local_corpus_release=object(),
            axiom_rules_path=tmp_path,
            axiom_rules_engine_ref=ref,
            expected_engine_binding=binding if ref else None,
        )
    assert identity.call_args.kwargs["axiom_rules_engine_ref"] == ref
    assert getattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR) == execution


def test_snapshot_rejects_engine_bytes_changed_since_overlay_start(tmp_path):
    before = {"ref": "a" * 40, "binary": str(tmp_path / "engine"), "sha256": "b" * 64}
    after = {**before, "sha256": "c" * 64}
    result = SimpleNamespace()
    with (
        patch.object(
            cli,
            "_apply_validation_execution_identity",
            return_value={"axiom_rules_engine_binding": after},
        ),
        patch.object(cli, "_build_apply_validation_snapshot") as build,
        pytest.raises(RuntimeError, match="changed during overlay"),
    ):
        cli._record_successful_apply_validation(
            result,
            output_root=tmp_path,
            policy_repo_path=tmp_path,
            relative_output=Path("rule.yaml"),
            supplemental_files={},
            local_corpus_release=object(),
            axiom_rules_path=tmp_path,
            axiom_rules_engine_ref="a" * 40,
            expected_engine_binding=before,
        )
    build.assert_not_called()
    assert not hasattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR)


def test_pinned_binding_uses_existing_verified_resolver_and_exact_bytes(tmp_path):
    binary = tmp_path / "engine"
    binary.write_bytes(b"actual engine bytes")
    with patch.object(
        cli, "resolve_pinned_engine_binary", return_value=binary
    ) as resolve:
        actual = cli._pinned_apply_engine_binding(tmp_path, "a" * 40)
    assert resolve.call_args.args[1].sha == "a" * 40
    assert actual == {
        "ref": "a" * 40,
        "binary": str(binary),
        "sha256": cli._sha256_file(binary),
    }


def test_pin_resolution_failure_prevents_overlay_snapshot(tmp_path):
    result = SimpleNamespace()
    with (
        patch.object(
            cli,
            "resolve_pinned_engine_binary",
            side_effect=cli.EngineBindingError("wrong pin"),
        ),
        patch.object(cli, "ValidatorPipeline") as pipeline,
        pytest.raises(cli.EngineBindingError, match="wrong pin"),
    ):
        cli._validate_generated_encoding_in_policy_overlay_with_release(
            result,
            output_root=tmp_path,
            policy_repo_path=tmp_path,
            axiom_rules_path=tmp_path,
            axiom_rules_engine_ref="a" * 40,
            local_corpus_release=object(),
        )
    pipeline.assert_not_called()
    assert getattr(result, cli._APPLY_VALIDATION_SNAPSHOT_ATTR) is None


@pytest.mark.parametrize(
    "binding",
    [
        None,
        {},
        {"ref": "main"},
        {"ref": "a" * 40, "binary": "relative", "sha256": "b" * 64},
    ],
)
def test_snapshot_explicit_malformed_binding_cannot_downgrade(binding):
    with pytest.raises(RuntimeError, match="malformed engine binding"):
        cli._apply_snapshot_engine_ref({"axiom_rules_engine_binding": binding})


def test_legacy_snapshot_has_no_new_engine_requirement():
    assert cli._apply_snapshot_engine_ref({}) is None
    with patch.object(cli, "resolve_pinned_engine_binary") as resolve:
        assert cli._pinned_apply_engine_binding(Path("unused"), None) is None
    resolve.assert_not_called()

"""Required review refuses finalized exact-dependent migrations before signing.

Reuse the existing migration graph fixture through overlay finalization. The
probe stops before unrelated manifest census/provenance work and uses fixture
trust roots. It supplies no real engine or production signing-key evidence.
"""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

import axiom_encode.cli as cli
from axiom_encode.generalist_review_record import (
    GeneralistReviewState,
    ReviewRequirementRefused,
    build_generalist_review_record,
    review_evidence_digests,
)
from tests.test_cli import TestCmdEncode as _MigrationCases
from tests.test_cli import (
    _pin_test_encoder_execution_identity as _pin_test_encoder_execution_identity,
)
from tests.test_cli import _test_eval_evidence_keys as _test_eval_evidence_keys


class _ReachedSignerBoundary(Exception):
    pass


class _MigrationProbeComplete(Exception):
    pass


@pytest.mark.parametrize("dependent_owner", ["manual", "generated"])
@pytest.mark.parametrize("required", [True, False], ids=["required", "record-only"])
def test_finalized_exact_dependents_refuse_required_review_before_signing(
    tmp_path, monkeypatch, capsys, dependent_owner, required
):
    fixture = _MigrationCases()
    reached = []

    def probe_apply(result, *, output_root, policy_repo_path, corpus_path, **_kwargs):
        contract = cli._result_legacy_replacement_contract(result)
        assert isinstance(contract, cli._LegacyReplacementContract)
        assert len(contract.exact_dependents) == 2
        dependent_files = [
            item
            for dependent in contract.exact_dependents
            for item in dependent.live_files
        ]
        rewritten_modules = [
            item.raw
            for item in dependent_files
            if not item.path.name.endswith(".test.yaml")
        ]
        assert len(rewritten_modules) == 2
        assert all(b"us-la:statutes/47/32#amount" in raw for raw in rewritten_modules)
        assert all(
            b"us-la:statutes/47:32#amount" not in raw for raw in rewritten_modules
        )

        output = Path(result.output_file)
        companion = output.with_name("32.test.yaml")
        state = GeneralistReviewState(
            metrics_pass=True,
            score=9,
            issue_count=0,
            prompt_sha256="4" * 64,
            evidence={
                **review_evidence_digests(output.read_bytes(), companion.read_bytes()),
                "rulespec_complete": True,
                "test_complete": True,
                "reviewer_backend": "codex",
                "reviewer_model": "fixture",
            },
            required=required,
        )
        pairs = cli._review_applied_pairs([output])
        record = build_generalist_review_record(
            state,
            pairs,
            [
                digest
                for pair in pairs
                for digest in (pair.rulespec_sha256, pair.test_sha256)
            ],
        )
        assert record["status"] == "passed"
        assert record["required"] is required
        setattr(result, cli._GENERALIST_REVIEW_STATE_ATTR, state)

        # Exercise public admission too, with this real finalized contract but
        # mocked generation and unrelated overlay validation from its CLI helper.
        cli_root = tmp_path / "cli"
        cli_root.mkdir()
        original_state_builder = cli._encode_generalist_review_state

        def attach_contract(cli_result, **kwargs):
            setattr(cli_result, cli._LEGACY_REPLACEMENT_ATTR, contract)
            return original_state_builder(cli_result, **kwargs)

        with patch.object(
            cli, "_encode_generalist_review_state", side_effect=attach_contract
        ):
            run = fixture._run_review_gate(
                cli_root,
                capsys,
                review=fixture._REVIEW_PASSED,
                require_flag=required,
            )
        assert run.outcome["generalist_review"]["status"] == "passed"
        if required:
            assert run.code == 1
            assert run.outcome["status"] == "apply_blocked_review"
            assert "rewritten exact dependents" in run.outcome["apply_error"]
            run.apply.assert_not_called()
        else:
            assert run.code == 0
            assert run.outcome["status"] == "apply_applied"
            run.apply.assert_called_once()

        checkout = policy_repo_path.parent
        before = {
            item.path: (checkout / item.path).read_bytes() for item in dependent_files
        }
        with (
            patch.object(cli, "_recover_apply_transaction") as recover,
            patch.object(
                cli,
                "_require_applied_encoding_manifest_signer",
                side_effect=_ReachedSignerBoundary,
            ) as acquire_signer,
            patch.object(cli, "_stage_signed_apply_manifest") as stage,
            patch.object(cli, "_install_apply_transaction") as install,
        ):
            expected = ReviewRequirementRefused if required else _ReachedSignerBoundary
            with pytest.raises(expected) as error:
                cli._apply_generated_encoding_result(
                    result,
                    output_root=output_root,
                    policy_repo_path=policy_repo_path,
                    corpus_path=corpus_path,
                )
            if required:
                assert "rewritten exact dependents" in str(error.value)
                recover.assert_not_called()
                acquire_signer.assert_not_called()
            else:
                recover.assert_called_once()
                acquire_signer.assert_called_once()
            stage.assert_not_called()
            install.assert_not_called()
        assert {
            item.path: (checkout / item.path).read_bytes() for item in dependent_files
        } == before
        reached.append(contract)
        # Stop the reused migration test before its own full apply assertions.
        raise _MigrationProbeComplete

    monkeypatch.setattr(fixture, "_record_apply_validation", probe_apply)
    preflights = contextmanager(
        _MigrationCases._neutralize_encode_preflights.__wrapped__
    )
    with preflights(fixture, tmp_path), pytest.raises(_MigrationProbeComplete):
        fixture.test_apply_atomically_migrates_exact_legacy_dependent(
            tmp_path, dependent_owner
        )
    assert len(reached) == 1

import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from decimal import Decimal
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from axiom_oracles.bridges.registry import (
    PolicyEngineMapping,
    PolicyEngineOracleRegistry,
    load_policyengine_registry,
)
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode.corpus_resolver import (
    PROOF_EVIDENCE_SEGMENT_SEPARATOR,
    AmbiguousCorpusSourceError,
    LocalCorpusRelease,
)
from axiom_encode.engine_binding import (
    EngineBindingError,
    write_engine_binding_receipt,
)
from axiom_encode.harness import validator_pipeline
from axiom_encode.harness.evals import _rulespec_validation_target
from axiom_encode.harness.policyengine_runtime import (
    PolicyEngineRuntime,
    PolicyEngineRuntimeError,
)
from axiom_encode.harness.proof_validator import (
    _rule_source_numeric_parent_alpha_scope,
    _source_top_level_marker_for_numeric_parent,
    find_rulespec_proof_issues,
    validate_rulespec_proofs,
)
from axiom_encode.harness.validator_pipeline import (
    _HEBREW_PREFIX_STACK_FRAGMENT,
    HEBREW_HEADED_LIST_IN_CONDITION,
    NumericOccurrence,
    OracleSubprocessResult,
    _clean_source_text_for_numeric_extraction_tracked,
    _corpus_citation_to_normalized_target,
    _extract_json_object,
    _formula_is_syntactically_unsatisfiable_false,
    _infer_us_state_code_from_rulespec_path,
    _literal_comparison_truth_value,
    _normalize_us_tax_filing_status,
    _normalize_validation_staging_text,
    _NumericTextView,
    _policyengine_expected_float,
    _policyengine_period_string,
    _policyengine_us_snap_input_aliases,
    _tax_unit_member_aged_flags,
    build_existing_target_oracle_contract,
    extract_embedded_source_text,
    extract_grounding_values,
    extract_named_scalar_occurrences,
    extract_numbers_from_text,
    extract_numeric_occurrences_from_text,
    extract_typed_numeric_inventory_occurrences_from_text,
    extract_typed_numeric_occurrences_from_text,
    find_aggregate_exception_predicate_issues,
    find_anaphoric_scope_omission_issues,
    find_broad_application_passthrough_issues,
    find_child_fragment_reencoding_issues,
    find_copied_cross_reference_source_issues,
    find_current_purpose_placeholder_issues,
    find_current_year_final_amount_table_issues,
    find_deferred_output_issues,
    find_deferred_purpose_specific_limitation_issues,
    find_delegated_policy_setting_issues,
    find_deprecated_source_url_issues,
    find_employer_scoped_entity_issues,
    find_empty_rules_module_issues,
    find_entity_limited_aggregation_order_issues,
    find_exception_test_coverage_issues,
    find_existing_target_oracle_contract_issues,
    find_filtered_entity_dependency_issues,
    find_formula_absolute_reference_issues,
    find_formula_date_literal_issues,
    find_helper_only_definition_issues,
    find_import_shape_issues,
    find_imported_deferred_branch_composition_issues,
    find_imported_person_scoped_definition_unit_issues,
    find_interval_table_reencoding_candidates,
    find_interval_table_reencoding_issues,
    find_judgment_conditional_formula_issues,
    find_judgment_positive_companion_output_issues,
    find_local_dependency_temporal_coverage_issues,
    find_missing_child_exception_import_issues,
    find_missing_derived_companion_output_issues,
    find_missing_same_section_subsection_import_issues,
    find_mixed_case_rule_name_token_issues,
    find_nonnegative_amount_reduction_issues,
    find_out_of_scope_rule_source_issues,
    find_partial_extent_zeroing_issues,
    find_person_scoped_definition_unit_issues,
    find_person_scoped_rate_base_unit_issues,
    find_proof_import_hash_consistency_issues,
    find_proof_import_reference_issues,
    find_relation_aggregate_syntax_issues,
    find_role_limited_relation_scope_issues,
    find_rule_name_path_suffix_issues,
    find_rule_source_metadata_issues,
    find_scoped_exception_category_gate_issues,
    find_shared_statutory_rate_entity_suffix_name_issues,
    find_sibling_rule_name_collision_issues,
    find_source_claim_reference_issues,
    find_source_condition_coverage_issues,
    find_source_limitation_application_issues,
    find_source_relation_issues,
    find_source_scope_consistency_issues,
    find_source_subparagraph_coverage_issues,
    find_source_table_row_scalar_parameter_issues,
    find_source_verification_issues,
    find_synthetic_source_authorized_input_issues,
    find_tax_filing_status_enum_representation_issues,
    find_tax_filing_status_local_input_issues,
    find_tax_filing_status_surviving_spouse_issues,
    find_tax_filing_status_test_input_issues,
    find_tax_status_component_local_input_issues,
    find_temporal_value_fact_name_issues,
    find_test_input_assignment_issues,
    find_unconsumed_local_exception_output_issues,
    find_ungrounded_numeric_issues,
    find_ungrounded_numeric_issues_scoped,
    find_unused_import_issues,
    find_unused_modifier_parameter_issues,
    find_upstream_placement_issues,
    find_versioned_derived_formula_issues,
    find_zero_branch_test_coverage_issues,
    hebrew_ambiguous_reading_groups,
    numeric_value_is_grounded,
    repair_copied_cross_reference_summary,
    repair_nonnegative_amount_reductions,
    repair_source_table_band_scalar_parameters,
    repair_source_table_interval_row_alignment,
    repair_source_table_interval_tests,
)
from axiom_encode.harness.validator_pipeline import (
    ValidatorPipeline as _ValidatorPipeline,
)
from axiom_encode.repo_routing import find_policy_repo_root
from tests.process_limits import HANG_TIMEOUT_SECONDS
from tests.real_engine import ABSENT_ROOT
from tests.release_object_fixtures import bind_test_corpus_release
from tests.scaling import assert_linear_growth, lowest_work

# Every test here that executes the engine takes the real_engine fixture;
# the rest get a checkout that never exists, here and in CI alike.
AXIOM_RULES_PATH = ABSENT_ROOT
TEST_CORPUS_RELEASE_NAME = "test-release"
TEST_CORPUS_VERSION = "test-version"


def _clear_numeric_extraction_caches():
    # A fresh tokenization can still reuse its paragraph/currency index.
    # Clear both text-result caches before every measured extraction.
    validator_pipeline._tokenize_numeric_occurrences_from_text.cache_clear()
    validator_pipeline._hebrew_paragraph_and_heading_index.cache_clear()


@pytest.mark.parametrize(
    "section",
    [
        "39-28.5-107",
        "39-30.5-104",
        "39-22-123.5",
        "39-1-104.5",
        "39-26-702",
    ],
)
def test_colorado_corpus_citation_lookup_keeps_crs_segments_dot_atomic(section):
    assert _corpus_citation_to_normalized_target(f"us-co/statute/39/{section}") == (
        "statutes",
        "39",
        section,
    )


def test_claude_reviewer_disables_tools_and_scrubs_signing_capabilities(
    tmp_path,
    monkeypatch,
):
    signing_names = (
        "AXIOM_ENCODE_EVAL_SIGNING_PRIVATE_KEY",
        "AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY",
        "AXIOM_ENCODE_SIGNING_BROKER_FD",
        "AXIOM_ENCODE_SIGNING_BROKER_PID",
        "AXIOM_ENCODE_SIGNING_BROKER_ACTIVE",
    )
    for index, name in enumerate(signing_names):
        monkeypatch.setenv(name, str(index + 1))

    with patch.object(
        validator_pipeline,
        "_run_subprocess_with_idle_timeout",
        return_value=validator_pipeline._SubprocessRunResult(
            output='{"passed": true}',
            returncode=0,
        ),
    ) as mock_run:
        output, returncode = validator_pipeline.run_claude_code(
            "review this",
            model="claude-test",
            timeout=30,
            cwd=tmp_path,
        )

    command = mock_run.call_args.args[0]
    assert command[command.index("--permission-mode") + 1] == "dontAsk"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--allowed-tools") + 1] == ""
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers": {}}'
    for flag in (
        "--safe-mode",
        "--no-session-persistence",
        "--disable-slash-commands",
        "--no-chrome",
        "--strict-mcp-config",
    ):
        assert flag in command
    child_environment = mock_run.call_args.kwargs["env"]
    assert all(name not in child_environment for name in signing_names)
    assert mock_run.call_args.kwargs["cwd"] == tmp_path
    assert output == '{"passed": true}'
    assert returncode == 0


def test_validator_subprocess_capture_read_errors_are_reproducible():
    captured_paths: list[Path] = []

    def fail_capture_read(path: Path, *_args, **_kwargs):
        captured_paths.append(path)
        raise OSError(5, "synthetic capture read failure", str(path))

    errors: list[str] = []
    with patch.object(Path, "read_text", autospec=True, side_effect=fail_capture_read):
        for _ in range(2):
            with pytest.raises(RuntimeError) as exc_info:
                validator_pipeline._run_subprocess_with_idle_timeout(
                    [sys.executable, "-c", "pass"],
                    timeout=HANG_TIMEOUT_SECONDS,
                    idle_timeout=HANG_TIMEOUT_SECONDS,
                    poll_interval=0.001,
                )
            errors.append(str(exc_info.value))

    assert captured_paths[0].parent != captured_paths[1].parent
    assert errors[0] == errors[1]
    assert "<validator-subprocess-capture>/stdout.log" in errors[0]
    assert all(str(path.parent) not in errors[0] for path in captured_paths)


def test_validator_subprocess_capture_output_paths_are_reproducible():
    captured_paths: list[Path] = []

    def expose_capture_path(path: Path, *_args, **_kwargs):
        captured_paths.append(path)
        return f"capture={path}\n"

    outputs: list[str] = []
    with patch.object(
        Path,
        "read_text",
        autospec=True,
        side_effect=expose_capture_path,
    ):
        for _ in range(2):
            result = validator_pipeline._run_subprocess_with_idle_timeout(
                [sys.executable, "-c", "pass"],
                timeout=HANG_TIMEOUT_SECONDS,
                idle_timeout=HANG_TIMEOUT_SECONDS,
                poll_interval=0.001,
            )
            outputs.append(result.output)

    assert captured_paths[0].parent == captured_paths[1].parent
    assert captured_paths[2].parent == captured_paths[3].parent
    assert captured_paths[0].parent != captured_paths[2].parent
    assert (
        outputs
        == [
            (
                "capture=<validator-subprocess-capture>/stdout.log\n"
                "capture=<validator-subprocess-capture>/stderr.log\n"
            )
        ]
        * 2
    )


def test_validator_subprocess_cleanup_error_cannot_mask_timeout():
    class NeverCompletes:
        returncode = None

        def poll(self):
            return None

        def kill(self):
            self.returncode = -9

        def wait(self):
            return self.returncode

    real_rmtree = shutil.rmtree
    cleanup_failures: list[str] = []

    def rmtree_with_cleanup_failure(path, *args, onexc=None, **kwargs):
        assert onexc is not None
        failing_path = str(Path(path) / "stdout.log")
        cleanup_failures.append(failing_path)
        onexc(
            os.unlink,
            failing_path,
            OSError(5, "synthetic cleanup failure", failing_path),
        )
        return real_rmtree(path)

    with (
        patch.object(
            validator_pipeline.subprocess,
            "Popen",
            return_value=NeverCompletes(),
        ),
        patch.object(validator_pipeline.time, "time", side_effect=[0.0, 2.0]),
        patch.object(shutil, "rmtree", side_effect=rmtree_with_cleanup_failure),
        pytest.raises(subprocess.TimeoutExpired),
    ):
        validator_pipeline._run_subprocess_with_idle_timeout(
            ["unused"],
            timeout=1,
            idle_timeout=10,
            poll_interval=0,
        )

    assert len(cleanup_failures) == 1
    assert cleanup_failures[0].endswith("/stdout.log")


def test_rulespec_run_compiled_timeout_is_reproducible_across_temp_roots(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    def timeout_with_command_path(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    issue_sets: list[list[str]] = []
    with patch.object(
        validator_pipeline.subprocess,
        "run",
        side_effect=timeout_with_command_path,
    ):
        for staging_name in ("tmpABC123", "tmpXYZ789"):
            outputs, issues = pipeline._run_rulespec_derived_test_case(
                binary=tmp_path / "axiom-rules-engine",
                compiled_path=tmp_path / staging_name / "compiled.json",
                case={"input": {}},
                case_name="timeout_case",
                case_index=1,
                period={
                    "period_kind": "month",
                    "start": "2026-01-01",
                    "end": "2026-01-31",
                },
                output_names=["benefit"],
                derived_by_key={"benefit": {"entity": "Household"}},
                require_legal_input_keys=False,
                legal_ids_by_friendly_name={},
                declared_relation_names=set(),
                module_target=None,
            )
            assert outputs is None
            issue_sets.append(issues)

    assert issue_sets == [
        ["Test case `timeout_case` execution timed out after 60 seconds."],
        ["Test case `timeout_case` execution timed out after 60 seconds."],
    ]
    assert all(
        "tmpABC123" not in issue and "tmpXYZ789" not in issue for issue in issues
    )


def _canonical_rulespec_content_root(base: Path, jurisdiction: str) -> Path:
    """Create and return ``rulespec-<country>/<jurisdiction>``."""

    country = jurisdiction.split("-", 1)[0]
    content_root = base / f"rulespec-{country}" / jurisdiction
    content_root.mkdir(parents=True, exist_ok=True)
    return content_root


def _canonical_rulespec_test_file(
    base: Path,
    *,
    jurisdiction: str = "us",
    relative: str = "policies/example/rules.yaml",
) -> tuple[Path, Path]:
    """Create a canonical content root and return one RuleSpec fixture path."""

    policy_repo = _canonical_rulespec_content_root(base, jurisdiction)
    rules_file = policy_repo / relative
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    return policy_repo, rules_file


def _composition_validation_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    checkout = tmp_path / "rulespec-us"
    policy_repo = checkout / "us-az"
    rules_file = policy_repo / "policies/example/composition.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  kind: composition
rules: []
"""
    )
    program_spec = checkout / "programs/us-az/snap/fy-2026.yaml"
    program_spec.parent.mkdir(parents=True)
    program_spec.write_text(
        """program: us-az/snap
period: 2026-01
outputs: [result]
scope:
  federal: []
  state:
    - policies/example/composition
"""
    )
    return policy_repo, rules_file, program_spec


def test_composition_module_requires_explicit_compose_executable(tmp_path):
    policy_repo, rules_file, _program_spec = _composition_validation_fixture(tmp_path)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "engine",
        enable_oracles=False,
    )
    pipeline._axiom_rules_binary = lambda: tmp_path / "engine"

    result = pipeline._run_rulespec_compile_check(rules_file)

    assert not result.passed
    assert "requires an explicit axiom-compose executable" in result.issues[0]


def test_composition_module_is_composed_before_engine_compile(tmp_path):
    policy_repo, rules_file, program_spec = _composition_validation_fixture(tmp_path)
    compose = tmp_path / "axiom-compose"
    compose.write_text(
        f"""#!{sys.executable}
import pathlib
import sys
args = sys.argv[1:]
output = pathlib.Path(args[args.index('-o') + 1])
output.write_text('format: rulespec/v1\\nrules: []\\n')
"""
    )
    compose.chmod(0o755)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "engine",
        axiom_compose_path=compose,
        enable_oracles=False,
    )
    pipeline._axiom_rules_binary = lambda: tmp_path / "engine"
    observed: dict[str, object] = {}

    def fake_compile(*, program, output, rulespec_roots, **_kwargs):
        observed["program"] = program
        observed["roots"] = rulespec_roots
        observed["composed"] = _kwargs["composed"]
        output.write_text(
            json.dumps({"program": {"parameters": [], "derived": [], "relations": []}})
        )
        return subprocess.CompletedProcess([], 0, "", "")

    with patch.object(validator_pipeline, "run_rulespec_compile", fake_compile):
        result = pipeline._run_rulespec_compile_check(rules_file)

    assert result.passed
    assert Path(observed["program"]).name == "composed-program.yaml"
    assert observed["roots"] == (policy_repo.parent,)
    assert observed["composed"] is True
    assert pipeline._owning_program_specs(rules_file) == (program_spec,)


def test_composition_owner_normalizes_qualified_fragment_scope_target(tmp_path):
    policy_repo, rules_file, program_spec = _composition_validation_fixture(tmp_path)
    program_spec.write_text(
        program_spec.read_text().replace(
            "- policies/example/composition",
            "- us-az:policies/example/composition#result",
        )
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "engine",
        enable_oracles=False,
    )

    assert pipeline._owning_program_specs(rules_file) == (program_spec,)


def test_composition_owner_follows_transitive_composition_imports(tmp_path):
    policy_repo, nested_rules, program_spec = _composition_validation_fixture(tmp_path)
    outer_rules = policy_repo / "policies/example/outer-composition.yaml"
    outer_rules.write_text(
        """format: rulespec/v1
module:
  kind: composition
imports:
  - us-az:policies/example/composition
"""
    )
    program_spec.write_text(
        program_spec.read_text().replace(
            "- policies/example/composition",
            "- policies/example/outer-composition",
        )
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "engine",
        enable_oracles=False,
    )

    assert pipeline._owning_program_specs(nested_rules) == (program_spec,)


def test_composition_uses_explicit_roots_and_credential_free_environment(
    tmp_path,
    monkeypatch,
):
    policy_repo, rules_file, program_spec = _composition_validation_fixture(tmp_path)
    compose = tmp_path / "axiom-compose"
    compose.write_text("#!/bin/sh\nexit 0\n")
    compose.chmod(0o755)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "engine",
        axiom_compose_path=compose,
        enable_oracles=False,
    )
    observed: dict[str, object] = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        output = Path(command[command.index("-o") + 1])
        output.write_text("format: rulespec/v1\nrules: []\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-composer")
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", "/untrusted/root")
    monkeypatch.setattr(subprocess, "run", fake_run)

    result, composed = pipeline._compose_rulespec_module(rules_file, tmp_path)

    assert result.returncode == 0
    assert composed == tmp_path / "composed-program.yaml"
    assert observed["command"] == [
        str(compose),
        str(program_spec),
        "-o",
        str(composed),
        "--rulespec-root",
        str(policy_repo.parent),
    ]
    assert "OPENAI_API_KEY" not in observed["env"]
    assert "AXIOM_RULESPEC_REPO_ROOTS" not in observed["env"]


def _admitted_runtime_stub(
    root: Path,
    *,
    country: str,
    rulespec_checkout_root: Path | None = None,
) -> PolicyEngineRuntime:
    """Construct an already-admitted identity for boundary-only unit tests."""

    identity = {
        "schema": "axiom-policyengine-runtime/v2",
        "country": country,
        "repository_root": str(root),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    runtime = object.__new__(PolicyEngineRuntime)
    object.__setattr__(runtime, "root", root)
    object.__setattr__(runtime, "country", country)
    object.__setattr__(runtime, "python_path", root / ".venv" / "bin" / "python")
    object.__setattr__(
        runtime,
        "site_packages_path",
        root / ".venv" / "lib" / "python3.13" / "site-packages",
    )
    object.__setattr__(
        runtime,
        "rulespec_checkout_root",
        rulespec_checkout_root or root.parent / f"rulespec-{country}",
    )
    object.__setattr__(runtime, "identity", identity)
    object.__setattr__(
        runtime,
        "identity_sha256",
        hashlib.sha256(canonical.encode()).hexdigest(),
    )
    return runtime


class ValidatorPipeline(_ValidatorPipeline):
    """Test wrapper that binds a signed release before exercising CI."""

    def __init__(self, *args, local_corpus_release=None, **kwargs):
        # Legacy oracle behavior tests below isolate mapping/scenario logic with
        # a stubbed subprocess.  The explicit runtime admission boundary itself
        # is exercised against _ValidatorPipeline in dedicated hard-cut tests.
        stub_policyengine = (
            bool(kwargs.get("enable_oracles"))
            and kwargs.get("policyengine_runtime") is None
        )
        if stub_policyengine:
            kwargs["enable_oracles"] = False
        super().__init__(
            *args,
            local_corpus_release=local_corpus_release,
            **kwargs,
        )
        if stub_policyengine:
            self.enable_oracles = True

    def _run_policyengine(self, rulespec_file):
        if self.policyengine_runtime is None:
            return self._run_policyengine_bound(rulespec_file, "us")
        return super()._run_policyengine(rulespec_file)

    def _run_ci(self, rulespec_file):
        if self.local_corpus_release is None:
            self.local_corpus_release = _write_local_corpus_provision(
                self.policy_repo_path,
                "us/statute/26/1",
            )
        return super()._run_ci(rulespec_file)


def test_policyengine_oracle_requires_explicit_runtime_before_any_discovery(
    tmp_path,
    monkeypatch,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    monkeypatch.setenv(
        "AXIOM_ENCODE_POLICYENGINE_US_PYTHON",
        "/attacker/ambient/python",
    )
    monkeypatch.setenv("HOME", "/attacker/home")
    with (
        patch("axiom_encode.harness.validator_pipeline.subprocess.run") as run,
        pytest.raises(PolicyEngineRuntimeError, match="explicit admitted runtime"),
    ):
        _ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=AXIOM_RULES_PATH,
            local_corpus_release=None,
            enable_oracles=True,
            oracle_validators=("policyengine",),
        )
    run.assert_not_called()


@pytest.mark.parametrize("oracle", ["taxsim", "all"])
def test_validator_rejects_removed_oracle_modes(tmp_path, oracle):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    with pytest.raises(ValueError, match="Unsupported oracle validator"):
        _ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=AXIOM_RULES_PATH,
            local_corpus_release=None,
            enable_oracles=True,
            oracle_validators=(oracle,),
        )


def test_policyengine_runtime_country_must_match_canonical_rulespec_root(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "uk")
    runtime = _admitted_runtime_stub(
        tmp_path / "policyengine-us",
        country="us",
        rulespec_checkout_root=policy_repo.parent,
    )
    with pytest.raises(PolicyEngineRuntimeError, match="does not match"):
        _ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=AXIOM_RULES_PATH,
            local_corpus_release=None,
            enable_oracles=True,
            oracle_validators=("policyengine",),
            policyengine_runtime=runtime,
        )


def test_policyengine_oracle_reprobes_and_rejects_post_execution_mutation(
    tmp_path,
):
    policy_repo, rules_file = _canonical_rulespec_test_file(tmp_path)
    runtime = _admitted_runtime_stub(
        tmp_path / "policyengine-us",
        country="us",
        rulespec_checkout_root=policy_repo.parent,
    )
    pipeline = _ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=AXIOM_RULES_PATH,
        local_corpus_release=None,
        enable_oracles=True,
        oracle_validators=("policyengine",),
        policyengine_runtime=runtime,
    )
    pipeline._run_policyengine_bound = lambda *_args: (
        validator_pipeline.ValidationResult(
            "policyengine",
            True,
            score=1.0,
            details={"coverage": {"comparable": 1}},
        )
    )

    with (
        patch.object(
            PolicyEngineRuntime,
            "assert_unchanged",
            side_effect=[
                None,
                PolicyEngineRuntimeError(
                    "PolicyEngine runtime identity changed after it was admitted"
                ),
            ],
        ),
        pytest.raises(PolicyEngineRuntimeError, match="identity changed"),
    ):
        pipeline._run_policyengine(rules_file)


def test_policyengine_oracle_binds_runtime_and_uses_root_country_only(tmp_path):
    policy_repo, rules_file = _canonical_rulespec_test_file(tmp_path)
    rules_file.write_text(
        "# legislation.gov.uk must not select UK\nformat: rulespec/v1\n"
    )
    runtime = _admitted_runtime_stub(
        tmp_path / "policyengine-us",
        country="us",
        rulespec_checkout_root=policy_repo.parent,
    )
    pipeline = _ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=AXIOM_RULES_PATH,
        local_corpus_release=None,
        enable_oracles=True,
        oracle_validators=("policyengine",),
        policyengine_runtime=runtime,
    )
    observed: list[str] = []

    def run_bound(_rules_file, country):
        observed.append(country)
        return validator_pipeline.ValidationResult(
            "policyengine",
            True,
            score=1.0,
            details={"coverage": {"comparable": 1}},
        )

    pipeline._run_policyengine_bound = run_bound
    with patch.object(PolicyEngineRuntime, "assert_unchanged", return_value=None):
        result = pipeline._run_policyengine(rules_file)

    assert observed == ["us"]
    assert result.details["runtime_identity"] == runtime.identity
    assert result.details["runtime_identity_sha256"] == runtime.identity_sha256


def test_validator_rejects_duplicate_rulespec_mapping_keys(tmp_path):
    repo = tmp_path / "rulespec-us"
    rules_file = (
        repo / "us-ia" / "regulations" / "iac" / "441" / "41" / "41" / "28.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
rules:
  - name: shelter_basic_needs_component
    kind: parameter
    entity: Household
    dtype: Money
    period: Month
    values:
      10: 20.58
      10: 20.58
"""
    )

    release = _write_local_corpus_provision(tmp_path, "us/statute/26/1")
    pipeline = ValidatorPipeline(
        policy_repo_path=repo / "us-ia",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=release,
    )

    result = pipeline.validate(rules_file, skip_reviewers=True)

    assert result.all_passed is False
    assert any(
        "duplicate key" in issue and "10" in issue
        for issue in result.results["compile"].issues
    )
    assert any(
        "duplicate key" in issue and "10" in issue
        for issue in result.results["ci"].issues
    )


def test_validator_validate_requires_bound_local_release(tmp_path):
    rules_file = tmp_path / "rulespec-us" / "statutes" / "26" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    with pytest.raises(RuntimeError, match="requires a bound LocalCorpusRelease"):
        pipeline.validate(rules_file, skip_reviewers=True)


def test_rulespec_numeric_output_comparison_tolerates_decimal_residue(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._rulespec_scalar_values_equal(
        {"kind": "decimal", "value": Decimal("19.999999999999999999999999992")},
        {"kind": "integer", "value": 20},
    )
    assert not pipeline._rulespec_scalar_values_equal(
        {"kind": "decimal", "value": Decimal("19.99")},
        {"kind": "integer", "value": 20},
    )


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("197.51000000000001", "197.51"),
        ("100.48999999999999", "100.49"),
        ("197.50000000000002", "197.5"),
        ("100.49999999999998", "100.5"),
    ],
)
def test_rulespec_numeric_output_comparison_tolerates_one_binary64_ulp(
    tmp_path, actual, expected
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._rulespec_scalar_values_equal(
        {"kind": "decimal", "value": Decimal(actual)},
        {"kind": "decimal", "value": Decimal(expected)},
    )


def test_rulespec_numeric_output_comparison_rejects_more_than_one_binary64_ulp(
    tmp_path,
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    expected = 100.5
    more_than_one_ulp = math.nextafter(math.nextafter(expected, math.inf), math.inf)
    assert not pipeline._rulespec_scalar_values_equal(
        {"kind": "decimal", "value": more_than_one_ulp},
        {"kind": "decimal", "value": expected},
    )


@pytest.mark.parametrize(
    ("actual", "actual_kind", "expected", "expected_kind"),
    [
        (2**53, "integer", 2**53 + 1, "integer"),
        (Decimal(2**53), "decimal", 2**53 + 1, "integer"),
        (Decimal("0.9999999999999998"), "decimal", 1, "integer"),
        (Decimal("-1"), "decimal", Decimal("-0.9999999999999998"), "decimal"),
    ],
)
def test_rulespec_numeric_output_comparison_rejects_unsafe_float_collapses(
    tmp_path, actual, actual_kind, expected, expected_kind
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert not pipeline._rulespec_scalar_values_equal(
        {"kind": actual_kind, "value": actual},
        {"kind": expected_kind, "value": expected},
    )


def test_rulespec_numeric_output_comparison_accepts_quoted_decimal(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-us",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert (
        pipeline._compare_rulespec_output(
            case_name="one_third",
            output_name="rate",
            expected_value="0.3333333333333333333333333333",
            actual_output={
                "kind": "scalar",
                "value": {
                    "kind": "decimal",
                    "value": Decimal("0.3333333333333333333333333333"),
                },
            },
        )
        is None
    )


def test_complete_mode_test_mismatch_steers_estg_dual_effective_versions(tmp_path):
    amendment_citation = (
        "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1"
    )
    amendment_source = (
        "4. In § 66 Absatz 1 wird die Angabe „250 Euro“ durch die Angabe "
        "„255 Euro“ ersetzt. "
        "4. § 66 wird wie folgt geändert: a) In Absatz 1 wird die Angabe "
        "„255 Euro“ durch die Angabe „259 Euro“ ersetzt. "
        "Artikel 10 Inkrafttreten (1) Dieses Gesetz tritt vorbehaltlich des "
        "Absatzes 2 am 1. Januar 2025 in Kraft. (2) Die Artikel 2, 4 und 6 "
        "treten am 1. Januar 2026 in Kraft."
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-de/de",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        require_complete_source_unit=True,
        source_text=(
            "(1) Das Kindergeld beträgt monatlich für jedes Kind 259 Euro.\n"
            "(2) Das Kindergeld wird monatlich vom Beginn des Monats an gezahlt."
        ),
        source_citation_path="de/statute/estg/66",
        amendment_source_texts={amendment_citation: amendment_source},
    )

    output_name = "de:statutes/estg/66#child_benefit_amount"
    issues = pipeline._run_rulespec_test_cases(
        rules_file=tmp_path / "rulespec-de/de/statutes/estg/66.yaml",
        compiled_path=tmp_path / "compiled.json",
        compiled_payload={
            "program": {
                "derived": [],
                "parameters": [
                    {
                        "name": "child_benefit_amount",
                        "id": output_name,
                        "versions": [
                            {
                                "effective_from": "2025-01-01",
                                "values": {"0": {"kind": "integer", "value": "259"}},
                            }
                        ],
                    }
                ],
            }
        },
        cases=[
            {
                "name": "kindergeld_2025",
                "period": {
                    "period_kind": "tax_year",
                    "start": "2025-01-01",
                    "end": "2025-12-31",
                },
                "input": {},
                "output": {output_name: 255},
            }
        ],
    )

    assert len(issues) == 1
    issue = issues[0]
    assert "expected integer 255, got integer 259" in issue
    assert "Complete-source effective-version hint" in issue
    assert amendment_citation in issue
    assert "dual effective versions" in issue
    assert "`255` for 2025 from the amendment" in issue
    assert "`259` for the 2026 consolidated version" in issue


@pytest.mark.parametrize(
    ("require_complete_source_unit", "primary_source", "amendment_sources"),
    (
        (
            False,
            "Das Kindergeld beträgt monatlich für jedes Kind 259 Euro.",
            {
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1": (
                    "Die Angabe „250 Euro“ wird durch „255 Euro“ ersetzt. "
                    "Das Gesetz gilt 2025 und 2026."
                )
            },
        ),
        (
            True,
            "2025 betrug das Kindergeld 255 Euro; nun beträgt es 259 Euro.",
            {
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1": (
                    "Die Angabe „250 Euro“ wird durch „255 Euro“ ersetzt. "
                    "Das Gesetz gilt 2025 und 2026."
                )
            },
        ),
        (
            True,
            "Das Kindergeld beträgt monatlich für jedes Kind 259 Euro.",
            {
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1": (
                    "Die Angabe „250 Euro“ wird durch „254 Euro“ ersetzt. "
                    "Das Gesetz gilt 2025 und 2026."
                )
            },
        ),
        (
            True,
            "Das Kindergeld beträgt monatlich für jedes Kind 260 Euro.",
            {
                "de/statute/bgbl-2024-i-449/steuerfortentwicklungsgesetz/document-1": (
                    "Die Angabe „250 Euro“ wird durch „255 Euro“ ersetzt. "
                    "Das Gesetz gilt 2025 und 2026."
                )
            },
        ),
    ),
)
def test_effective_version_hint_requires_complete_amendment_primary_split(
    tmp_path,
    require_complete_source_unit,
    primary_source,
    amendment_sources,
):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path / "rulespec-de/de",
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        require_complete_source_unit=require_complete_source_unit,
        source_text=primary_source,
        source_citation_path="de/statute/estg/66",
        amendment_source_texts=amendment_sources,
    )

    issue = pipeline._compare_rulespec_output(
        case_name="kindergeld_2025",
        output_name="de:statutes/estg/66#child_benefit_amount",
        expected_value=255,
        actual_output={
            "kind": "scalar",
            "value": {"kind": "integer", "value": 259},
        },
        period={
            "period_kind": "tax_year",
            "start": "2025-01-01",
            "end": "2025-12-31",
        },
    )

    assert issue is not None
    assert "expected integer 255, got integer 259" in issue
    assert "Complete-source effective-version hint" not in issue


def test_rulespec_compile_roots_use_only_explicit_checkouts(monkeypatch, tmp_path):
    repo_parent = tmp_path / "repos"
    policy_repo = _canonical_rulespec_content_root(repo_parent, "us-ny")
    dependency_root = _canonical_rulespec_content_root(
        tmp_path / "dependencies", "uk"
    ).parent
    ambient_root = _canonical_rulespec_content_root(tmp_path / "ambient", "us").parent
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(ambient_root))

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        rulespec_dependency_roots=(dependency_root,),
    )

    roots = pipeline._rulespec_compile_roots()
    assert roots == (policy_repo.parent.resolve(), dependency_root.resolve())
    assert ambient_root not in roots
    assert pipeline._rulespec_root_args() == [
        "--rulespec-root",
        str(policy_repo.parent.resolve()),
        "--rulespec-root",
        str(dependency_root.resolve()),
    ]


def test_rulespec_compile_roots_accept_composition_checkouts(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "active", "us")
    dependency_root = _canonical_rulespec_content_root(
        tmp_path / "dependencies", "uk"
    ).parent
    (policy_repo.parent / "programs/us/snap").mkdir(parents=True)
    (dependency_root / "programs/uk/universal-credit").mkdir(parents=True)

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        rulespec_dependency_roots=(dependency_root,),
    )

    assert pipeline._rulespec_compile_roots() == (
        policy_repo.parent.resolve(),
        dependency_root.resolve(),
    )


def test_rulespec_engine_env_excludes_model_and_repository_credentials(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-secret")
    monkeypatch.setenv("GH_TOKEN", "github-test-secret")
    monkeypatch.setenv("AXIOM_REPO_TOKEN", "repository-test-secret")
    monkeypatch.setenv("NON_SENSITIVE_SENTINEL", "preserved")
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    env = pipeline._rulespec_engine_env()

    assert env["NON_SENSITIVE_SENTINEL"] == "preserved"
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "GH_TOKEN" not in env
    assert "AXIOM_REPO_TOKEN" not in env


def test_rulespec_compile_roots_ignore_ambient_env_and_sibling_roots(
    monkeypatch,
    tmp_path,
):
    repo_parent = tmp_path / "worktrees"
    policy_repo = _canonical_rulespec_content_root(repo_parent, "us-co")
    stale_sibling = _canonical_rulespec_content_root(tmp_path / "ambient", "us").parent
    dependency_root = _canonical_rulespec_content_root(
        tmp_path / "configured-roots", "uk"
    ).parent
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_sibling))

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        rulespec_dependency_roots=(dependency_root,),
    )

    roots = pipeline._rulespec_compile_roots()
    assert roots == (policy_repo.parent.resolve(), dependency_root.resolve())
    assert stale_sibling not in roots
    assert repo_parent not in roots


def test_rulespec_compile_roots_reject_nested_checkout_symlink(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: true\n")
    nested = policy_repo / "statutes" / "outside.yaml"
    nested.parent.mkdir(parents=True)
    nested.symlink_to(outside)
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="contains a symlink",
    ):
        pipeline._rulespec_compile_roots()


def test_rulespec_dependency_root_rejects_nested_symlink(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "active", "us")
    dependency_root = _canonical_rulespec_content_root(
        tmp_path / "dependencies", "uk"
    ).parent
    outside = tmp_path / "outside.yaml"
    outside.write_text("secret: true\n")
    nested = dependency_root / "uk" / "statutes" / "outside.yaml"
    nested.parent.mkdir(parents=True)
    nested.symlink_to(outside)

    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="contains a symlink",
    ):
        ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            enable_oracles=False,
            rulespec_dependency_roots=(dependency_root,),
        )


def test_rulespec_dependency_roots_reject_duplicate_namespace(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "active", "us")
    dependency_one = _canonical_rulespec_content_root(
        tmp_path / "dependencies-one", "uk"
    ).parent
    dependency_two = _canonical_rulespec_content_root(
        tmp_path / "dependencies-two", "uk"
    ).parent

    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="same namespace 'rulespec-uk'",
    ):
        ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            enable_oracles=False,
            rulespec_dependency_roots=(dependency_one, dependency_two),
        )


def test_rulespec_dependency_root_rejects_active_namespace_collision(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "active", "us")
    conflicting_dependency = _canonical_rulespec_content_root(
        tmp_path / "dependency", "us-ca"
    ).parent

    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="active checkout namespace 'rulespec-us'",
    ):
        ValidatorPipeline(
            policy_repo_path=policy_repo,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            enable_oracles=False,
            rulespec_dependency_roots=(conflicting_dependency,),
        )


def test_rulespec_target_resolution_rejects_active_namespace_fallback(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "active", "us")
    conflicting_content = _canonical_rulespec_content_root(
        tmp_path / "dependency", "us-ca"
    )
    target = conflicting_content / "regulations" / "example.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    target_ref = validator_pipeline._parse_rulespec_target("us-ca:regulations/example")
    assert target_ref is not None

    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="active checkout namespace 'rulespec-us'",
    ):
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
            rulespec_dependency_roots=(conflicting_content.parent,),
        )


def test_rulespec_ci_rejects_noncanonical_runner_before_ambient_lookup(tmp_path):
    runner_root = tmp_path / "runner"
    rules_file = runner_root / "source" / "example.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    ambient = runner_root / "statutes" / "1" / "ambient.yaml"
    ambient.parent.mkdir(parents=True)
    ambient.write_text("format: rulespec/v1\nrules: []\n")
    pipeline = ValidatorPipeline(
        policy_repo_path=runner_root,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=_write_local_corpus_provision(
            tmp_path / "release",
            "us/statute/26/1",
        ),
    )

    with patch.object(pipeline, "_run_rulespec_ci") as run_rulespec_ci:
        result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert "not canonical" in (result.error or "")
    run_rulespec_ci.assert_not_called()


def test_rulespec_compile_passes_only_explicit_root_arguments(monkeypatch, tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "1" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    output_path = tmp_path / "compiled.json"
    captured: dict[str, object] = {}
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    monkeypatch.setattr(
        pipeline,
        "_axiom_rules_binary",
        lambda: tmp_path / "axiom-rules-engine-bin",
    )
    original_run = validator_pipeline.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        captured["command"] = command
        captured["env"] = kwargs["env"]
        output_path.write_text('{"program": {"derived": []}}')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    result, payload = pipeline._compile_rulespec_to_artifact(
        rules_file,
        output_path,
    )

    assert result.returncode == 0
    assert payload == {"program": {"derived": []}}
    assert captured["command"] == [
        str(tmp_path / "axiom-rules-engine-bin"),
        "compile",
        "--program",
        str(rules_file),
        "--rulespec-root",
        str(policy_repo.parent.resolve()),
        "--output",
        str(output_path),
    ]
    assert "AXIOM_RULESPEC_REPO_ROOTS" not in captured["env"]
    assert "AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE" not in captured["env"]


def test_rulespec_engine_binary_ignores_ambient_path(monkeypatch, tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    engine_root = tmp_path / "declared-axiom-rules-engine"
    engine_root.mkdir()
    ambient_bin = tmp_path / "ambient-bin"
    ambient_bin.mkdir()
    ambient_engine = ambient_bin / "axiom-rules-engine"
    ambient_engine.write_text("#!/bin/sh\nexit 0\n")
    ambient_engine.chmod(0o755)
    monkeypatch.setenv("PATH", f"{ambient_bin}{os.pathsep}{os.environ['PATH']}")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=engine_root,
        enable_oracles=False,
    )

    with pytest.raises(FileNotFoundError, match="explicitly declared checkout"):
        pipeline._axiom_rules_binary()


def test_rulespec_validation_run_compiled_scrubs_ambient_root_env(
    monkeypatch, tmp_path
):
    repo_parent = tmp_path / "repos"
    policy_repo = _canonical_rulespec_content_root(repo_parent, "us")
    stale_root = tmp_path / "stale-rulespec-us"
    stale_root.mkdir()
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_root))
    dependency_root = _canonical_rulespec_content_root(
        tmp_path / "dependencies", "uk"
    ).parent

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        rulespec_dependency_roots=(dependency_root,),
    )
    captured_env: dict[str, str] | None = None
    original_run = validator_pipeline.subprocess.run

    def fake_run(cmd, **kwargs):
        nonlocal captured_env
        if cmd[0] == "git":
            return original_run(cmd, **kwargs)
        captured_env = kwargs.get("env")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            **json.loads(kwargs["input"])["queries"][0],
                            "outputs": {
                                "us:statutes/1/1#benefit": {
                                    "kind": "scalar",
                                    "id": "us:statutes/1/1#benefit",
                                    "value": {"kind": "integer", "value": 6},
                                }
                            },
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={"input": {}},
        case_name="computes_benefit",
        case_index=0,
        period={
            "period_kind": "month",
            "start": "2026-01-01",
            "end": "2026-01-31",
        },
        output_names=["us:statutes/1/1#benefit"],
        derived_by_key={"us:statutes/1/1#benefit": {"entity": "Household"}},
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        declared_relation_names=set(),
        module_target=None,
    )

    assert issues == []
    assert outputs is not None
    assert outputs["us:statutes/1/1#benefit"]["value"]["value"] == 6
    assert captured_env is not None
    assert "AXIOM_RULESPEC_REPO_ROOTS" not in captured_env
    assert "AXIOM_RULESPEC_REPO_ROOTS_EXCLUSIVE" not in captured_env
    assert str(stale_root) not in captured_env.values()


def _write_engine_pinned_toolchain(checkout_root: Path, engine_ref: str) -> Path:
    """Write a legacy multi-key toolchain.toml declaring an engine pin."""

    config_dir = checkout_root / ".axiom"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "toolchain.toml"
    config_path.write_text(
        "[toolchain]\n"
        'axiom_corpus_release = "test-release"\n'
        f'axiom_corpus_release_content_sha256 = "{"0" * 64}"\n'
        f'validation_waiver_set_sha256 = "{"0" * 64}"\n'
        f'axiom_rules_engine_ref = "{engine_ref}"\n'
    )
    return config_path


def test_rulespec_engine_binary_prefers_release_and_warns_once_when_unpinned(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    engine_root = tmp_path / "axiom-rules-engine"
    release = engine_root / "target" / "release" / "axiom-rules-engine"
    debug = engine_root / "target" / "debug" / "axiom-rules-engine"
    for candidate in (release, debug):
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(b"bin")
    events: list[dict] = []

    class _RecordingDB:
        def log_event(self, **kwargs):
            events.append(kwargs)

    pipeline = _ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
        encoding_db=_RecordingDB(),
        session_id="engine-binding-test",
    )

    assert pipeline._axiom_rules_binary() == release
    assert pipeline._axiom_rules_binary() == release

    unverified = [
        event for event in events if event["event_type"] == "engine_binding_unverified"
    ]
    assert len(unverified) == 1
    metadata = unverified[0]["metadata"]
    assert metadata["binary"] == str(release)
    assert str(debug) in metadata["other_existing_candidates"]
    assert metadata["declared_engine_pin"] is None


def test_rulespec_engine_binary_nonexistent_policy_repo_raises(tmp_path):
    # A typo'd policy repo path must fail loudly, not downgrade to unpinned
    # binding of whatever stale binary exists.
    engine_root = tmp_path / "axiom-rules-engine"
    debug = engine_root / "target" / "debug" / "axiom-rules-engine"
    debug.parent.mkdir(parents=True)
    debug.write_bytes(b"stale debug")

    pipeline = _ValidatorPipeline(
        policy_repo_path=tmp_path / "missing" / "us",
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
    )

    with pytest.raises(EngineBindingError, match="does not exist"):
        pipeline._axiom_rules_binary()


def test_rulespec_engine_binary_resolution_is_memoized_per_pipeline(
    monkeypatch, tmp_path
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    engine_root = tmp_path / "axiom-rules-engine"
    engine_root.mkdir()
    resolved = engine_root / "target" / "release" / "axiom-rules-engine"
    # Memoization is stat-guarded: it only holds for a real, unchanged binary.
    resolved.parent.mkdir(parents=True)
    resolved.write_bytes(b"pinned release")
    calls = {"load": 0, "resolve": 0}

    def fake_load(path):
        calls["load"] += 1
        return validator_pipeline.EnginePin(
            sha="a" * 40,
            source=tmp_path / "toolchain.toml",
        )

    def fake_resolve(checkout, pin, *, allow_build=True):
        calls["resolve"] += 1
        return resolved

    monkeypatch.setattr(validator_pipeline, "load_declared_engine_pin", fake_load)
    monkeypatch.setattr(
        validator_pipeline, "resolve_pinned_engine_binary", fake_resolve
    )
    pipeline = _ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
    )

    assert pipeline._axiom_rules_binary() == resolved
    assert pipeline._axiom_rules_binary() == resolved
    assert calls == {"load": 1, "resolve": 1}


def test_rulespec_engine_ref_override_wins_over_toolchain_pin(monkeypatch, tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    _write_engine_pinned_toolchain(policy_repo.parent, "c" * 40)
    captured: dict[str, object] = {}

    def fake_resolve(checkout, pin, *, allow_build=True):
        captured["pin"] = pin
        return checkout / "target" / "release" / "axiom-rules-engine"

    monkeypatch.setattr(
        validator_pipeline, "resolve_pinned_engine_binary", fake_resolve
    )
    pipeline = _ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        local_corpus_release=None,
        enable_oracles=False,
        axiom_rules_engine_ref="d" * 40,
    )

    pipeline._axiom_rules_binary()

    pin = captured["pin"]
    assert pin.sha == "d" * 40
    assert str(pin.source) == "<constructor>"


def test_rulespec_compile_uses_pinned_receipted_binary(monkeypatch, tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "1" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    pin_sha = "f" * 40
    _write_engine_pinned_toolchain(policy_repo.parent, pin_sha)
    engine_root = tmp_path / "axiom-rules-engine"
    release = engine_root / "target" / "release" / "axiom-rules-engine"
    debug = engine_root / "target" / "debug" / "axiom-rules-engine"
    for candidate, payload in ((release, b"pinned release"), (debug, b"stale debug")):
        candidate.parent.mkdir(parents=True)
        candidate.write_bytes(payload)
    write_engine_binding_receipt(release, pin_sha, "release")

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=engine_root,
        enable_oracles=False,
    )
    output_path = tmp_path / "compiled.json"
    captured: dict[str, object] = {}
    original_run = validator_pipeline.subprocess.run

    def fake_run(command, **kwargs):
        if command[0] == "git":
            return original_run(command, **kwargs)
        captured["command"] = command
        output_path.write_text('{"program": {"derived": []}}')
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    result, payload = pipeline._compile_rulespec_to_artifact(rules_file, output_path)

    assert result.returncode == 0
    assert payload == {"program": {"derived": []}}
    assert captured["command"][0] == str(release)


def test_rulespec_engine_binary_swap_after_binding_is_rebound_not_reused(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    pin_sha = "f" * 40
    _write_engine_pinned_toolchain(policy_repo.parent, pin_sha)
    engine_root = tmp_path / "axiom-rules-engine"
    release = engine_root / "target" / "release" / "axiom-rules-engine"
    release.parent.mkdir(parents=True)
    release.write_bytes(b"pinned release")
    write_engine_binding_receipt(release, pin_sha, "release")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=engine_root,
        local_corpus_release=None,
        enable_oracles=False,
    )

    assert pipeline._axiom_rules_binary() == release

    release.write_bytes(b"swapped bytes that no receipt attests")
    with pytest.raises(EngineBindingError):
        pipeline._axiom_rules_binary()

    write_engine_binding_receipt(release, pin_sha, "release")
    assert pipeline._axiom_rules_binary() == release


def test_rulespec_target_resolution_uses_explicit_dependency_root(
    monkeypatch, tmp_path
):
    workspace = tmp_path / "worktrees"
    policy_repo = _canonical_rulespec_content_root(workspace, "us-co")
    stale_sibling = _canonical_rulespec_content_root(tmp_path / "ambient", "uk")
    stale_file = stale_sibling / "statutes" / "1" / "example.yaml"
    stale_file.parent.mkdir(parents=True)
    stale_file.write_text(
        """format: rulespec/v1
rules:
  - name: stale_snap_income_rule
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: '0'
"""
    )
    configured_parent = tmp_path / "configured"
    configured_file = (
        configured_parent / "rulespec-uk" / "uk" / "statutes" / "1" / "example.yaml"
    )
    configured_file.parent.mkdir(parents=True)
    configured_file.write_text(
        """format: rulespec/v1
rules:
  - name: snap_standard_utility_allowance_state_option
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: '0'
"""
    )
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(stale_sibling))
    target_ref = validator_pipeline._parse_rulespec_target(
        "uk:statutes/1/example#snap_standard_utility_allowance_state_option"
    )

    assert target_ref is not None
    assert (
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
            rulespec_dependency_roots=(configured_parent / "rulespec-uk",),
        )
        == configured_file
    )


def test_rulespec_target_resolution_ignores_ambient_env(monkeypatch, tmp_path):
    ambient_repo = _canonical_rulespec_content_root(tmp_path / "ambient", "us")
    target = ambient_repo / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    policy_repo = _canonical_rulespec_content_root(tmp_path, "uk")
    monkeypatch.setenv("AXIOM_RULESPEC_REPO_ROOTS", str(ambient_repo.parent))
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    assert (
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
        )
        is None
    )


def test_rulespec_target_resolution_ignores_cwd_and_sibling_checkouts(
    monkeypatch,
    tmp_path,
):
    ambient_us = _canonical_rulespec_content_root(tmp_path, "us")
    target = ambient_us / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    policy_repo = _canonical_rulespec_content_root(tmp_path / "workspace", "uk")
    monkeypatch.chdir(tmp_path)
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    assert (
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
        )
        is None
    )


def test_rulespec_resolution_cache_audits_checkout_once_per_scope(
    monkeypatch,
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    original_walk = os.walk
    walked_roots = []

    def counting_walk(root, *args, **kwargs):
        walked_roots.append(Path(root))
        return original_walk(root, *args, **kwargs)

    monkeypatch.setattr(validator_pipeline.os, "walk", counting_walk)

    with validator_pipeline._rulespec_resolution_cache_scope():
        assert (
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
            == policy_repo.parent
        )
        assert (
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
            == policy_repo.parent
        )

    assert walked_roots == [policy_repo.parent]

    validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
    assert walked_roots == [policy_repo.parent, policy_repo.parent]


def test_rulespec_resolution_cache_retries_rejected_symlink_audit(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    outside = tmp_path / "outside"
    outside.mkdir()
    unsafe = policy_repo / "unsafe"
    unsafe.symlink_to(outside, target_is_directory=True)

    with validator_pipeline._rulespec_resolution_cache_scope():
        with pytest.raises(
            validator_pipeline.UnsafeRulespecContextPath,
            match="contains a symlink",
        ):
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
        unsafe.unlink()
        assert (
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
            == policy_repo.parent
        )


@pytest.mark.parametrize("replace_existing", [False, True])
def test_rulespec_resolution_cache_invalidates_added_or_replaced_symlink(
    tmp_path,
    replace_existing,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    nested = policy_repo / "statutes/1"
    nested.mkdir(parents=True)
    candidate = nested / "unsafe.yaml"
    if replace_existing:
        candidate.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    outside = tmp_path / "outside.yaml"
    outside.write_text("outside\n", encoding="utf-8")

    with validator_pipeline._rulespec_resolution_cache_scope():
        assert (
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)
            == policy_repo.parent
        )
        if replace_existing:
            candidate.unlink()
        candidate.symlink_to(outside)
        with pytest.raises(
            validator_pipeline.UnsafeRulespecContextPath,
            match="contains a symlink",
        ):
            validator_pipeline._rulespec_checkout_root_for_active_path(policy_repo)


def test_rulespec_resolution_cache_reuses_target_lookup_inside_scope(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    target = policy_repo / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    with patch.object(
        validator_pipeline,
        "_candidate_rulespec_repo_roots",
        wraps=validator_pipeline._candidate_rulespec_repo_roots,
    ) as candidates:
        with validator_pipeline._rulespec_resolution_cache_scope():
            assert (
                validator_pipeline._resolve_rulespec_target_file(
                    target_ref,
                    policy_repo,
                )
                == target
            )
            assert (
                validator_pipeline._resolve_rulespec_target_file(
                    target_ref,
                    policy_repo,
                )
                == target
            )

    candidates.assert_called_once()


def test_rulespec_resolution_cache_does_not_cache_missing_target(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    target = policy_repo / "statutes/1/target.yaml"
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    with validator_pipeline._rulespec_resolution_cache_scope():
        assert (
            validator_pipeline._resolve_rulespec_target_file(target_ref, policy_repo)
            is None
        )
        target.parent.mkdir(parents=True)
        target.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
        assert (
            validator_pipeline._resolve_rulespec_target_file(target_ref, policy_repo)
            == target
        )


def test_rulespec_resolution_cache_invalidates_changed_target_file(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    target = policy_repo / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n", encoding="utf-8")
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    with patch.object(
        validator_pipeline,
        "_candidate_rulespec_repo_roots",
        wraps=validator_pipeline._candidate_rulespec_repo_roots,
    ) as candidates:
        with validator_pipeline._rulespec_resolution_cache_scope():
            assert (
                validator_pipeline._resolve_rulespec_target_file(
                    target_ref,
                    policy_repo,
                )
                == target
            )
            target.write_text(
                "format: rulespec/v1\nmodule:\n  name: changed\nrules: []\n",
                encoding="utf-8",
            )
            assert (
                validator_pipeline._resolve_rulespec_target_file(
                    target_ref,
                    policy_repo,
                )
                == target
            )

    assert candidates.call_count == 2


def test_rulespec_target_resolution_rejects_workspace_dependency_root(tmp_path):
    workspace = tmp_path / "dependencies"
    target = workspace / "rulespec-us" / "us" / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    policy_repo = _canonical_rulespec_content_root(tmp_path, "uk")
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="exact canonical checkout roots",
    ):
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
            rulespec_dependency_roots=(workspace,),
        )


def test_rulespec_target_resolution_accepts_macos_system_path_alias(tmp_path):
    resolved_tmp = tmp_path.resolve()
    for alias_root, physical_root in (
        (Path("/var"), Path("/private/var")),
        (Path("/tmp"), Path("/private/tmp")),
    ):
        if (
            resolved_tmp.is_relative_to(physical_root)
            and alias_root.is_symlink()
            and alias_root.resolve() == physical_root
        ):
            aliased_tmp = alias_root / resolved_tmp.relative_to(physical_root)
            break
    else:
        pytest.skip("temporary directory has no macOS system path alias")
    assert aliased_tmp != resolved_tmp
    assert aliased_tmp.resolve() == resolved_tmp
    checkout = aliased_tmp / "rulespec-us"
    target = checkout / "us" / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    policy_repo = _canonical_rulespec_content_root(tmp_path, "uk")
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    assert (
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
            rulespec_dependency_roots=(checkout,),
        )
        == target.resolve()
    )


def test_rulespec_target_resolution_rejects_explicit_checkout_symlink(tmp_path):
    checkout = tmp_path / "private" / "rulespec-us"
    target = checkout / "us" / "statutes/1/target.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    aliases = tmp_path / "aliases"
    aliases.mkdir()
    (aliases / "rulespec-us").symlink_to(checkout, target_is_directory=True)
    policy_repo = _canonical_rulespec_content_root(tmp_path, "uk")
    target_ref = validator_pipeline._parse_rulespec_target("us:statutes/1/target")

    assert target_ref is not None
    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="contains a symlink",
    ):
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
            rulespec_dependency_roots=(aliases / "rulespec-us",),
        )


def test_rulespec_target_resolution_rejects_repo_controlled_axiom_alias(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path / "workspace", "us")
    external_repo = _canonical_rulespec_content_root(tmp_path / "private", "uk")
    target = external_repo / "statutes/1/secret.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    axiom_dir = policy_repo / "_axiom"
    axiom_dir.mkdir()
    (axiom_dir / "rulespec-uk").symlink_to(
        external_repo,
        target_is_directory=True,
    )
    target_ref = validator_pipeline._parse_rulespec_target("uk:statutes/1/secret")

    assert target_ref is not None
    with pytest.raises(
        validator_pipeline.UnsafeRulespecContextPath,
        match="active RuleSpec checkout contains a symlink",
    ):
        validator_pipeline._resolve_rulespec_target_file(
            target_ref,
            policy_repo,
        )


@pytest.mark.parametrize(
    "import_target",
    ["us:statutes/1/target", "us:statutes/1/target#target_amount"],
)
def test_imported_rulespec_exports_uses_canonical_target_base(
    tmp_path,
    import_target,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "1" / "current.yaml"
    target_file = policy_repo / "statutes" / "1" / "target.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text("format: rulespec/v1\nrules: []\n")
    target_file.write_text(
        """format: rulespec/v1
rules:
  - name: target_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: '1'
"""
    )

    exports = validator_pipeline._imported_rulespec_exports(
        {"imports": [import_target]},
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert exports["target_amount"].target == ("us:statutes/1/target#target_amount")


def test_unrelated_same_section_term_import_rejects_other_section_standin(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_checks_policy_repo_for_temp_output(
    tmp_path,
):
    repo = _canonical_rulespec_content_root(tmp_path, "us")
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    generated_section_root = tmp_path / "run/codex/statutes/26/3134"
    generated_section_root.mkdir(parents=True)
    rules_file = generated_section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    with _rulespec_validation_target(rules_file, repo) as validation_file:
        validation_root = find_policy_repo_root(validation_file)
        assert validation_root is not None
        pipeline = ValidatorPipeline(
            policy_repo_path=validation_root,
            axiom_rules_path=tmp_path / "axiom-rules-engine",
            enable_oracles=False,
        )
        issues = pipeline._check_unrelated_same_section_term_imports(validation_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_rejects_file_level_standin(tmp_path):
    repo = _canonical_rulespec_content_root(tmp_path, "us")
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    other_root = repo / "statutes/26/45A"
    other_root.mkdir(parents=True)
    (other_root / "b.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Section 45A wage amount.
rules:
  - name: qualified_wages_and_health_costs_taken_into_account
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: wages
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b` overlaps same-section term `qualified_wages` "
        "defined or deferred in `statutes/26/3134/c.yaml`. Import the "
        "same-section output or defer the dependent output instead of using an "
        "unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_rejects_exclusion_citation(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages exclude wages taken into account under section 45A.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_unrelated_same_section_term_imports(rules_file)

    assert issues == [
        "Unrelated same-section term import: "
        "`us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account` "
        "overlaps same-section term `qualified_wages` defined or deferred in "
        "`statutes/26/3134/c.yaml`. Import the same-section output or defer the "
        "dependent output instead of using an unrelated section's same-named concept."
    ]


def test_unrelated_same_section_term_import_allows_same_section_definition(tmp_path):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
rules:
  - name: qualified_wages
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: wages
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3134/c#qualified_wages
module:
  summary: Qualified wages with respect to any employee shall not exceed $10,000.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_unrelated_same_section_term_import_allows_defined_cross_reference(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/45A/b#qualified_wages_and_health_costs_taken_into_account
module:
  summary: Qualified wages have the meaning given by section 45A.
rules:
  - name: qualified_wages_taken_into_account_for_employee
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: min(10000, qualified_wages_and_health_costs_taken_into_account)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_unrelated_same_section_term_import_allows_narrower_cross_reference_credit(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/3134"
    section_root.mkdir(parents=True)
    (section_root / "c.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definitions for section 3134.
  deferred_outputs:
    - output: us:statutes/26/3134/c#qualified_wages
      reason: Needs calendar-quarter mechanics.
rules: []
"""
    )
    rules_file = section_root / "b.yaml"
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/3131/e#qualified_sick_leave_wages_credit_with_collectively_bargained_contributions_increase
module:
  summary: Applicable employment taxes are reduced by credits allowed under sections 3131 and 3132.
rules:
  - name: employment_tax_limit_for_credit
    kind: derived
    entity: Employer
    dtype: Money
    period: Year
    versions:
      - effective_from: '2021-01-01'
        formula: max(0, applicable_employment_taxes - qualified_sick_leave_wages_credit_with_collectively_bargained_contributions_increase)
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_unrelated_same_section_term_import_allows_cross_section_tax_rate(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    section_root = repo / "statutes/26/1402"
    section_root.mkdir(parents=True)
    (section_root / "b.yaml").write_text(
        """format: rulespec/v1
module:
  summary: Definition of self-employment income.
  deferred_outputs:
    - output: us:statutes/26/1402/b#self_employment_income
      reason: Needs trade-or-business income mechanics.
rules: []
"""
    )
    rules_file = section_root / "a/12.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate
module:
  summary: Paragraph (12) uses one-half of the section 1401(b)(1) rate.
rules:
  - name: paragraph_12_deduction_rate
    kind: derived
    entity: Person
    dtype: Rate
    period: Year
    versions:
      - effective_from: '1990-01-01'
        formula: self_employment_income_tax_rate / 2
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    assert pipeline._check_unrelated_same_section_term_imports(rules_file) == []


def test_rulespec_companion_runner_uses_rows_for_absolute_list_outputs(
    monkeypatch, tmp_path
):
    repo_parent = tmp_path / "repos"
    policy_repo = _canonical_rulespec_content_root(repo_parent, "us")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )
    captured_request: dict[str, object] | None = None

    def fake_run(cmd, **kwargs):
        nonlocal captured_request
        if "input" not in kwargs:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        captured_request = json.loads(kwargs["input"])
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "results": [
                        {
                            **json.loads(kwargs["input"])["queries"][0],
                            "outputs": {
                                "excluded_from_wages": {
                                    "kind": "scalar",
                                    "id": "us:statutes/26/3121/a/6#excluded_from_wages",
                                    "value": {"kind": "money", "value": 300},
                                }
                            },
                        }
                    ]
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(validator_pipeline.subprocess, "run", fake_run)

    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={
            "input": {},
            "tables": {
                "Payment": [
                    {
                        "payment_amount": 300,
                    }
                ]
            },
            "output": {
                "us:statutes/26/3121/a/6#excluded_from_wages": [300],
            },
        },
        case_name="excluded_payment",
        case_index=1,
        period={
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        output_names=["excluded_from_wages"],
        output_runtime_keys={
            "us:statutes/26/3121/a/6#excluded_from_wages": "excluded_from_wages",
        },
        derived_by_key={"excluded_from_wages": {"entity": "Payment"}},
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        declared_relation_names=set(),
        module_target=None,
    )

    assert issues == []
    assert outputs is not None
    assert captured_request is not None
    assert captured_request["queries"][0]["entity_id"] == "payment-1"


def test_rulespec_row_output_length_issue_order_is_deterministic(tmp_path):
    class StableHashName(str):
        def __new__(cls, value: str, hash_value: int):
            instance = super().__new__(cls, value)
            instance.hash_value = hash_value
            return instance

        def __hash__(self):
            return self.hash_value

    policy_repo = _canonical_rulespec_content_root(tmp_path / "repos", "us")
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )
    z_output = StableHashName("z_output", 0)
    a_output = StableHashName("a_output", 1)

    outputs, issues = pipeline._run_rulespec_derived_test_case(
        binary=tmp_path / "engine",
        compiled_path=tmp_path / "compiled.json",
        case={
            "input": {},
            "tables": {"Payment": [{"payment_amount": 300}]},
            "output": {
                z_output: [300, 400],
                a_output: [300, 400],
            },
        },
        case_name="ordered_failure",
        case_index=1,
        period={
            "period_kind": "tax_year",
            "start": "2026-01-01",
            "end": "2026-12-31",
        },
        output_names=[z_output, a_output],
        derived_by_key={
            z_output: {"entity": "Payment"},
            a_output: {"entity": "Payment"},
        },
        require_legal_input_keys=False,
        legal_ids_by_friendly_name={},
        declared_relation_names=set(),
        module_target=None,
    )

    assert outputs is None
    assert issues == [
        "Test case `ordered_failure` output `a_output` expected 2 row value(s), "
        "but `tables.Payment` has 1 row(s)."
    ]


def test_cross_statute_definition_import_check_uses_cited_title_and_existing_targets(
    tmp_path,
):
    repo = tmp_path / "rulespec-us"
    target = repo / "statutes" / "7" / "2014" / "e.yaml"
    target.parent.mkdir(parents=True)
    target.write_text("format: rulespec/v1\nrules: []\n")
    rules_file = repo / "statutes" / "7" / "2015" / "e.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    Eligibility is described in section 2014(e). A controlled substance is
    defined in section 802 of title 21.
rules: []
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=repo,
        axiom_rules_path=tmp_path / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_cross_statute_definition_imports(rules_file)

    assert issues == [
        "Cross-statute definition import missing: source text references "
        "section 2014(e) but file does not import from 7/2014/e"
    ]

    rules_file.write_text(
        """format: rulespec/v1
imports:
  - us:statutes/7/2014/e#snap_earned_income_deduction
module:
  summary: |-
    Eligibility is described in section 2014(e).
rules: []
"""
    )

    assert pipeline._check_cross_statute_definition_imports(rules_file) == []


def test_formula_absolute_reference_rejects_import_targets_in_formula():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
imports:
  - us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies
rules:
  - name: person_exempt_from_paragraph_1_work_requirements
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies
"""

    issues = find_formula_absolute_reference_issues(content)

    assert issues == [
        "Formula absolute import reference: "
        "`person_exempt_from_paragraph_1_work_requirements` contains "
        "`us:statutes/7/2015/d/2/A#title_iv_work_registration_exemption_applies` "
        "inside a formula. Add that target to `imports:` and reference the "
        "imported rule by bare local name in formula text."
    ]


def test_versioned_derived_formula_allows_multiple_formula_versions():
    content = """format: rulespec/v1
rules:
  - name: savers_credit_gross_contributions
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: qualified_retirement_contributions
      - effective_from: '2027-01-01'
        formula: able_account_contributions
  - name: inflation_adjusted_threshold
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: '50000'
      - effective_from: '2027-01-01'
        formula: '52000'
"""

    issues = find_versioned_derived_formula_issues(content)

    assert issues == []


def test_unbounded_derived_cannot_outlive_direct_finite_local_dependency():
    content = """\
format: rulespec/v1
rules:
  - name: al_pit_2026_first_rate
    kind: parameter
    dtype: Rate
    period: Year
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: 0.02
  - name: al_pit_2026_schedule_before_credits
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: al_pit_2026_first_rate * taxable_income
inputs:
  - name: taxable_income
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
"""

    issues = find_local_dependency_temporal_coverage_issues(content)

    assert len(issues) == 1
    assert "al_pit_2026_schedule_before_credits" in issues[0]
    assert "al_pit_2026_first_rate" in issues[0]
    assert "2026-12-31" in issues[0]


def test_bounded_derived_matches_direct_local_dependency_coverage():
    content = """\
format: rulespec/v1
rules:
  - name: rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: 0.02
  - name: tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: rate * income
inputs:
  - name: income
    entity: TaxUnit
    dtype: Money
"""

    assert find_local_dependency_temporal_coverage_issues(content) == []


def test_later_bounded_derived_cannot_outlive_direct_local_dependency():
    content = """\
format: rulespec/v1
rules:
  - name: rate
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: 0.02
  - name: tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    versions:
      - effective_from: '2027-01-01'
        effective_to: '2027-12-31'
        formula: rate * income
inputs:
  - name: income
    entity: TaxUnit
    dtype: Money
"""

    issues = find_local_dependency_temporal_coverage_issues(content)

    assert len(issues) == 1
    assert "tax" in issues[0]
    assert "rate" in issues[0]
    assert "2027-01-01 through 2027-12-31" in issues[0]


def test_implicitly_superseded_derived_version_is_not_treated_as_unbounded():
    content = """\
format: rulespec/v1
rules:
  - name: rate_2026
    kind: parameter
    dtype: Rate
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: 0.02
  - name: tax
    kind: derived
    entity: TaxUnit
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: rate_2026 * income
      - effective_from: '2027-01-01'
        formula: income
inputs:
  - name: income
    entity: TaxUnit
    dtype: Money
"""

    assert find_local_dependency_temporal_coverage_issues(content) == []


def test_exact_oracle_replacement_contract_preserves_surface_and_valid_input():
    target = "us-al:policies/income_tax/2026_section_40_18_5"
    existing = """\
format: rulespec/v1
rules:
  - name: al_pit_2026_schedule_before_credits
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: max(0, al_taxable_income) if filing_status == 'single' else 0
inputs:
  - name: al_taxable_income
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
  - name: filing_status
    entity: TaxUnit
    dtype: String
    period: Year
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={f"{target}#al_pit_2026_schedule_before_credits": object()}
    )
    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
        invalid_input_names={"filing_status"},
    )
    assert contract is not None
    assert [item.name for item in contract.inputs] == ["al_taxable_income"]

    renamed = existing.replace(
        "al_pit_2026_schedule_before_credits", "al_income_tax"
    ).replace("al_taxable_income", "taxable_income")
    issues = find_existing_target_oracle_contract_issues(renamed, contract)

    assert len(issues) == 2
    assert "#al_pit_2026_schedule_before_credits" in issues[0]
    assert "#input.al_taxable_income" in issues[1]
    assert find_existing_target_oracle_contract_issues(existing, contract) == []


def test_not_comparable_registry_entry_does_not_freeze_replacement_shape():
    target = "us-az:policies/des/faa5/na-categorical-eligibility/expanded-categorical-eligibility"
    existing = """\
format: rulespec/v1
rules:
  - name: expanded_categorical_eligibility_deems_resources_met
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: true
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#expanded_categorical_eligibility_deems_resources_met": SimpleNamespace(
                mapping_type="not_comparable"
            )
        }
    )

    assert (
        build_existing_target_oracle_contract(
            existing,
            target=target,
            policyengine_registry=registry,
        )
        is None
    )


def test_mixed_oracle_registry_preserves_only_comparable_surface_shape():
    target = "us-az:policies/des/faa5/na-categorical-eligibility/expanded-categorical-eligibility"
    existing = """\
format: rulespec/v1
rules:
  - name: unmapped_arizona_ece_deeming
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: true
  - name: mapped_categorical_eligibility
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: true
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#unmapped_arizona_ece_deeming": SimpleNamespace(
                mapping_type="not_comparable"
            ),
            f"{target}#mapped_categorical_eligibility": SimpleNamespace(
                mapping_type="direct_variable"
            ),
        }
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert [surface.name for surface in contract.surfaces] == [
        "mapped_categorical_eligibility"
    ]


def test_exact_oracle_replacement_contract_preserves_visibility_and_rejects_path_identity_helpers():
    target = "us-al:policies/income_tax/2026_section_40_18_5_schedule_before_credits"
    mapped_name = "al_pit_2026_section_40_18_5_schedule_before_credits"
    existing = f"""\
format: rulespec/v1
rules:
  - name: {mapped_name}
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        effective_to: '2026-12-31'
        formula: al_pit_completed_taxable_income
inputs:
  - name: al_pit_completed_taxable_income
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={f"{target}#{mapped_name}": object()}
    )
    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert contract.replacement_name_identity == "2026_section_40_18_5"
    assert contract.surfaces[0].private is False

    replacement = existing.replace(
        "    unit: USD\n    versions:",
        "    unit: USD\n    metadata:\n      private: true\n    versions:",
        1,
    ).replace(
        "inputs:\n"
        "  - name: al_pit_completed_taxable_income\n"
        "    entity: TaxUnit\n"
        "    dtype: Money\n",
        "inputs:\n"
        "  - name: al_pit_completed_taxable_income\n"
        "    entity: TaxUnit\n"
        "    dtype: Decimal\n",
    )
    replacement = replacement.replace(
        "inputs:\n",
        "  - name: al_pit_2026_section_40_18_5_taxable_income_floor\n"
        "    kind: parameter\n"
        "    dtype: Money\n"
        "    unit: USD\n"
        "    versions:\n"
        "      - effective_from: '2026-01-01'\n"
        "        formula: 0\n"
        "inputs:\n",
    )

    issues = find_existing_target_oracle_contract_issues(replacement, contract)

    assert len(issues) == 3
    assert "metadata.private/public contract" in issues[0]
    assert "#input.al_pit_completed_taxable_income" in issues[1]
    assert "[existing-target-naming-contract]" in issues[2]
    assert "`2026_section_40_18_5`" in issues[2]
    assert mapped_name not in issues[2]


def test_exact_oracle_replacement_naming_contract_allows_semantic_helpers():
    target = "us-al:policies/income_tax/2026_section_40_18_5_schedule_before_credits"
    mapped_name = "al_pit_2026_section_40_18_5_schedule_before_credits"
    existing = f"""\
format: rulespec/v1
rules:
  - name: {mapped_name}
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income_floor
  - name: taxable_income_floor
    kind: parameter
    dtype: Money
    unit: USD
    metadata:
      private: true
    versions:
      - effective_from: '2026-01-01'
        formula: 0
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={f"{target}#{mapped_name}": object()}
    )
    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert find_existing_target_oracle_contract_issues(existing, contract) == []


def test_exact_oracle_contract_allows_only_proven_relation_entity_repair():
    target = "us-az:policies/snap/basic_categorical_eligibility"
    existing = """\
format: rulespec/v1
rules:
  - name: member_of_budgetary_unit
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: participant_qualifies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: participant_has_status
  - name: all_participants_qualify
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: count_where(member_of_budgetary_unit, participant_qualifies) == len(member_of_budgetary_unit)
  - name: basic_eligibility
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: all_participants_qualify
inputs:
  - name: participant_has_status
    entity: Person
    dtype: Judgment
    period: Month
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#participant_qualifies": object(),
            f"{target}#all_participants_qualify": object(),
            f"{target}#basic_eligibility": object(),
        }
    )
    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    surfaces = {surface.name: surface for surface in contract.surfaces}
    assert surfaces["all_participants_qualify"].replacement_entity == "Household"
    assert surfaces["basic_eligibility"].replacement_entity == "Household"
    assert surfaces["participant_qualifies"].replacement_entity == ""

    corrected = existing.replace(
        "  - name: all_participants_qualify\n    kind: derived\n    entity: Person\n",
        "  - name: all_participants_qualify\n"
        "    kind: derived\n"
        "    entity: Household\n",
    )
    corrected = corrected.replace(
        "  - name: basic_eligibility\n    kind: derived\n    entity: Person\n",
        "  - name: basic_eligibility\n    kind: derived\n    entity: Household\n",
    )
    assert find_existing_target_oracle_contract_issues(corrected, contract) == []

    arbitrary_entity = corrected.replace(
        "  - name: all_participants_qualify\n"
        "    kind: derived\n"
        "    entity: Household\n",
        "  - name: all_participants_qualify\n    kind: derived\n    entity: TaxUnit\n",
    )
    arbitrary_issues = find_existing_target_oracle_contract_issues(
        arbitrary_entity, contract
    )
    assert len(arbitrary_issues) == 1
    assert "#all_participants_qualify" in arbitrary_issues[0]
    assert "entity `Household`" in arbitrary_issues[0]

    invalid_helper = corrected.replace(
        "  - name: participant_qualifies\n    kind: derived\n    entity: Person\n",
        "  - name: participant_qualifies\n    kind: derived\n    entity: Household\n",
    )
    helper_issues = find_existing_target_oracle_contract_issues(
        invalid_helper, contract
    )
    assert len(helper_issues) == 1
    assert "#participant_qualifies" in helper_issues[0]
    assert "kind/entity/dtype" in helper_issues[0]

    invalid_dtype = corrected.replace(
        "  - name: all_participants_qualify\n"
        "    kind: derived\n"
        "    entity: Household\n"
        "    dtype: Judgment\n",
        "  - name: all_participants_qualify\n"
        "    kind: derived\n"
        "    entity: Household\n"
        "    dtype: Decimal\n",
    )
    dtype_issues = find_existing_target_oracle_contract_issues(invalid_dtype, contract)
    assert len(dtype_issues) == 1
    assert "#all_participants_qualify" in dtype_issues[0]
    assert "kind/entity/dtype/period" in dtype_issues[0]


def test_relation_entity_repair_does_not_escape_aggregate_scope():
    target = "us-az:policies/snap/aggregate_scope"
    existing = """\
format: rulespec/v1
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: dependent_of_person
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Dependent, Person]
  - name: person_has_dependents
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: len(dependent_of_person) > 0
  - name: household_has_person_with_dependents
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: count_where(member_of_household, person_has_dependents) > 0
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#household_has_person_with_dependents": object(),
        }
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert contract.surfaces[0].replacement_entity == ""
    assert find_existing_target_oracle_contract_issues(existing, contract) == []


def test_relation_entity_repair_requires_unambiguous_aggregate_scope():
    target = "us-az:policies/snap/conflicting_aggregate_scope"
    existing = """\
format: rulespec/v1
rules:
  - name: people
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: dependents
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Dependent, Person]
  - name: ambiguous_count
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: len(people) + len(dependents)
  - name: outer_count
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: ambiguous_count + len(people)
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#ambiguous_count": object(),
            f"{target}#outer_count": object(),
        }
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert all(surface.replacement_entity == "" for surface in contract.surfaces)
    assert find_existing_target_oracle_contract_issues(existing, contract) == []


def test_relation_entity_repair_propagates_correct_helper_scope_not_string_text():
    target = "us-az:policies/snap/helper_scope"
    existing = """\
format: rulespec/v1
rules:
  - name: people
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: household_member_count
    kind: derived
    entity: Household
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: len(people)
  - name: mapped_count
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: household_member_count
  - name: mapped_label
    kind: derived
    entity: Person
    dtype: String
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: '"len(people)"'
  - name: mapped_amount
    kind: derived
    entity: Person
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2026-01-01'
        formula: income # do not use sum(people.income) here
inputs:
  - name: income
    entity: Person
    dtype: Money
    period: Month
    unit: USD
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#mapped_count": object(),
            f"{target}#mapped_label": object(),
            f"{target}#mapped_amount": object(),
        }
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    surfaces = {surface.name: surface for surface in contract.surfaces}
    assert surfaces["mapped_count"].replacement_entity == "Household"
    assert surfaces["mapped_label"].replacement_entity == ""
    assert surfaces["mapped_amount"].replacement_entity == ""


def test_relation_entity_repair_uses_only_outer_aggregate_scope():
    target = "us-az:policies/snap/nested_aggregate_scope"
    existing = """\
format: rulespec/v1
rules:
  - name: people
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: dependents
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Dependent, Person]
  - name: mapped_count
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: count_where(people, len(dependents) > 0)
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={f"{target}#mapped_count": object()}
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert contract.surfaces[0].replacement_entity == "Household"


def test_relation_entity_repair_does_not_cache_cycle_truncated_constraints():
    target = "us-az:policies/snap/cyclic_scope"
    existing = """\
format: rulespec/v1
rules:
  - name: people
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: dependents
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Dependent, Person]
  - name: a
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: len(people) + b
  - name: b
    kind: derived
    entity: Person
    dtype: Decimal
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: len(dependents) + a
"""
    registry = SimpleNamespace(
        mappings_by_legal_id={
            f"{target}#a": object(),
            f"{target}#b": object(),
        }
    )

    contract = build_existing_target_oracle_contract(
        existing,
        target=target,
        policyengine_registry=registry,
    )

    assert contract is not None
    assert all(surface.replacement_entity == "" for surface in contract.surfaces)


def test_rule_source_metadata_rejects_executable_rules_without_rule_source():
    content = """format: rulespec/v1
module:
  summary: The standard is 20 hours.
  source_verification:
    corpus_citation_path: us/statute/7/2015
rules:
  - name: minimum_hours
    kind: parameter
    dtype: Count
    versions:
      - effective_from: '2026-01-01'
        formula: '20'
  - name: runtime_membership
    kind: data_relation
    data_relation:
      arity: 2
  - name: restates_upstream
    kind: source_relation
    source_relation:
      type: restates
      target: us:statutes/7/2015#minimum_hours
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source metadata required: `minimum_hours` is an executable rule "
        "and must include `source:` with the legal citation/span supporting it."
    ]


def test_rule_source_metadata_rejects_derived_relation_without_rule_source():
    content = """format: rulespec/v1
module:
  summary: Household members who meet the eligibility tests form the SNAP unit.
  source_verification:
    corpus_citation_path: us/regulation/7/273/1
rules:
  - name: snap_unit
    kind: derived_relation
    derived_relation:
      arity: 2
      source_relation: member_of_household
      entity: SnapUnit
      member_relation: members
      slot_entities: [Person, Household]
    versions:
      - effective_from: '2026-01-01'
        formula: snap_member_eligible
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source metadata required: `snap_unit` is an executable rule "
        "and must include `source:` with the legal citation/span supporting it."
    ]


def test_rule_source_metadata_rejects_missing_module_source_locator():
    content = """format: rulespec/v1
module:
  summary: The standard is 20 hours.
rules:
  - name: minimum_hours
    kind: parameter
    dtype: Count
    source: 7 USC 2015(e)(4)
    versions:
      - effective_from: '2026-01-01'
        formula: '20'
"""

    issues = find_rule_source_metadata_issues(content)

    assert issues == [
        "Rule source locator required: module.source_verification must include "
        "exactly one `corpus_citation_path` when executable rules are present."
    ]


def test_missing_derived_companion_output_rejects_uncovered_derived_rule(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "7" / "2015" / "e.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: student_age_exception_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2015(e)(1)
    versions:
      - effective_from: '2026-01-01'
        formula: person_age_years < 18
  - name: student_single_parent_exception_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Month
    source: 7 USC 2015(e)(8)
    versions:
      - effective_from: '2026-01-01'
        formula: person_is_single_parent
"""
    cases = [
        {
            "name": "age_exception",
            "period": "2026-01",
            "input": {"us:statutes/7/2015/e#input.person_age_years": 17},
            "output": {"us:statutes/7/2015/e#student_age_exception_applies": "holds"},
        }
    ]

    issues = find_missing_derived_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Derived rule missing companion output coverage: "
        "`us:statutes/7/2015/e#student_single_parent_exception_applies` "
        "is not asserted by the companion `.test.yaml` file."
    ]


def test_missing_derived_companion_output_uses_canonical_prefix_with_matching_origin(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "26" / "63" / "f.yaml"
    rules_file.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "init"], cwd=policy_repo.parent, check=True, capture_output=True
    )
    subprocess.run(
        [
            "git",
            "remote",
            "add",
            "origin",
            "https://github.com/TheAxiomFoundation/rulespec-us.git",
        ],
        cwd=policy_repo.parent,
        check=True,
        capture_output=True,
    )
    content = """format: rulespec/v1
rules:
  - name: blind_under_subsection_f
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    source: 26 USC 63(f)
    versions:
      - effective_from: '2026-01-01'
        formula: taxpayer_is_blind
"""
    cases = [
        {
            "name": "empty",
            "period": "2026",
            "input": {},
            "output": {},
        }
    ]

    issues = find_missing_derived_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Derived rule missing companion output coverage: "
        "`us:statutes/26/63/f#blind_under_subsection_f` "
        "is not asserted by the companion `.test.yaml` file."
    ]


def test_missing_derived_companion_output_strips_country_subdir(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "nz")
    rules_file = policy_repo / "statutes" / "income_tax" / "rates.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: income_tax_before_credits
    kind: derived
    entity: Person
    dtype: Money
    period: Year
    source: Income Tax Act 2007 Schedule 1
    versions:
      - effective_from: '2026-01-01'
        formula: taxable_income * 0.105
"""
    cases = [
        {
            "name": "basic",
            "period": "2026",
            "input": {},
            "output": {"nz:statutes/income_tax/rates#income_tax_before_credits": 105},
        }
    ]

    issues = find_missing_derived_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == []


def test_judgment_positive_companion_output_rejects_never_holds(tmp_path):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us")
    rules_file = policy_repo / "statutes" / "26" / "3102" / "f" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3102(f)(1)
    versions:
      - effective_from: '2013-01-01'
        formula: |-
          tax_is_imposed_by_section_3101_b_2
          and wages_from_employer_in_excess_of_additional_medicare_collection_threshold > 0
"""
    cases = [
        {
            "name": "below_threshold",
            "period": "2026",
            "input": {},
            "output": {
                "us:statutes/26/3102/f/1#subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold": "not_holds"
            },
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Judgment rule missing positive companion output coverage: "
        "`us:statutes/26/3102/f/1#subsection_a_applies_to_additional_medicare_tax_wages_above_employer_threshold` "
        "is not asserted as `holds` by the companion `.test.yaml` file."
    ]


def test_judgment_positive_companion_output_allows_unsatisfiable_false_rule(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-co")
    rules_file = policy_repo / "regulations" / "10-ccr-2506-1" / "4.801.43.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: voluntary_payment_reactivates_terminated_claim
    kind: derived
    entity: SnapClaim
    dtype: Judgment
    period: Month
    source: 10 CCR 2506-1 4.801.43(A)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          claim_is_terminated
          and voluntary_payment_made_on_terminated_claim
          and false
"""
    cases = [
        {
            "name": "terminated_claim_nonreactivation",
            "period": "2026-01",
            "input": {
                "us-co:regulations/10-ccr-2506-1/4.801.43#input.claim_is_terminated": True,
                "us-co:regulations/10-ccr-2506-1/4.801.43#input.voluntary_payment_made_on_terminated_claim": True,
            },
            "output": {
                "us-co:regulations/10-ccr-2506-1/4.801.43#voluntary_payment_reactivates_terminated_claim": "not_holds"
            },
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == []


@pytest.mark.parametrize(
    "formula",
    [
        "1 == 0",
        "true == false",
        '"same" != "same"',
        "9007199254740992.0 == 9007199254740993.0",
        'source_condition and "same" != "same"',
        "(true or false) and false",
    ],
)
def test_judgment_positive_companion_output_allows_false_literal_comparison(
    tmp_path, formula
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-nj")
    rules_file = policy_repo / "statutes" / "54a" / "4-7.yaml"
    rules_file.parent.mkdir(parents=True)
    content = f"""format: rulespec/v1
rules:
  - name: subsection_f_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    source: N.J.S.54A:4-7
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          {formula}
"""
    cases = [
        {
            "name": "subsection_f_does_not_apply",
            "period": "2026",
            "input": {},
            "output": {"us-nj:statutes/54a/4-7#subsection_f_applies": "not_holds"},
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == []


@pytest.mark.parametrize(
    "formula",
    [
        "1 == 1",
        '"A" != "a"',
        "9007199254740992.0 != 9007199254740993.0",
        'source_condition and "A" != "a"',
        "true or false and false",
    ],
)
def test_judgment_positive_companion_output_requires_positive_for_true_literal_comparison(
    tmp_path, formula
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-nj")
    rules_file = policy_repo / "statutes" / "54a" / "4-7.yaml"
    rules_file.parent.mkdir(parents=True)
    content = f"""format: rulespec/v1
rules:
  - name: subsection_f_applies
    kind: derived
    entity: Person
    dtype: Judgment
    period: Year
    source: N.J.S.54A:4-7
    versions:
      - effective_from: '2000-01-01'
        formula: |-
          {formula}
"""
    cases = [
        {
            "name": "incorrect_negative_only_case",
            "period": "2026",
            "input": {},
            "output": {"us-nj:statutes/54a/4-7#subsection_f_applies": "not_holds"},
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Judgment rule missing positive companion output coverage: "
        "`us-nj:statutes/54a/4-7#subsection_f_applies` is not asserted as "
        "`holds` by the companion `.test.yaml` file."
    ]


@pytest.mark.parametrize(
    ("formula", "expected"),
    [
        ("1 == 0", False),
        ("1 != 0", True),
        ("1 < 2", True),
        ("2 <= 1", False),
        ("2 > 1", True),
        ("2 >= 3", False),
        ('"A" != "a"', True),
        ("9007199254740992.0 != 9007199254740993.0", True),
        ("true == false", False),
        ("False != false", False),
        (r'"line\n" == "line\n"', True),
        ('(")") == (")")', True),
        ('("(") == (")")', False),
    ],
)
def test_literal_comparison_truth_value_matches_rulespec_semantics(formula, expected):
    assert _literal_comparison_truth_value(formula) is expected


def test_constant_false_detection_respects_rulespec_boolean_precedence():
    assert not _formula_is_syntactically_unsatisfiable_false("true or false and false")
    assert _formula_is_syntactically_unsatisfiable_false("(true or false) and false")


def test_constant_false_detection_fails_closed_above_nesting_limit():
    deeply_wrapped_false = f"{'(' * 300}1 == 0{')' * 300}"
    deeply_nested_conjunction = "false"
    for _ in range(300):
        deeply_nested_conjunction = f"true and ({deeply_nested_conjunction})"

    assert not _formula_is_syntactically_unsatisfiable_false(deeply_wrapped_false)
    assert not _formula_is_syntactically_unsatisfiable_false(deeply_nested_conjunction)


def test_literal_comparison_handles_deep_parentheses_without_recursive_parsing():
    deeply_wrapped_false = f"{'(' * 10_000}1{')' * 10_000} == 0"

    assert _literal_comparison_truth_value(deeply_wrapped_false) is False


@pytest.mark.parametrize(
    "formula",
    [
        '"a" < "b"',
        "true > false",
        "1 == true",
        "1 < 2 < 3",
        "None == None",
        "claim_count == 0",
        "1e3 == 1000",
        '"unterminated == 1',
        "9223372036854775808 == 0",
        "-9223372036854775808 == 0",
        f"{'9' * 5000} == 0",
        "0.00000000000000000000000000001 == 0.0",
        f"{'9' * 5000}.0 == 0.0",
    ],
)
def test_literal_comparison_truth_value_rejects_unsupported_rulespec_forms(
    formula,
):
    assert _literal_comparison_truth_value(formula) is None


def test_synthetic_source_authorized_input_rejected_for_prohibition():
    content = """format: rulespec/v1
rules:
  - name: county_postponement_entitled
    kind: derived
    entity: SnapFairHearing
    dtype: Judgment
    period: Day
    source: 10 CCR 2506-1 section 4.802.61(A)(3)
    versions:
      - effective_from: '2025-10-01'
        formula: |-
          county_requested_postponement
          and county_has_source_authorized_postponement_entitlement
"""

    issues = find_synthetic_source_authorized_input_issues(content)

    assert issues == [
        "Synthetic source-authorized local input: "
        "`county_postponement_entitled` depends on local input "
        "`county_has_source_authorized_postponement_entitlement`. "
        "Encode the source-stated authorization, prohibition, or entitlement "
        "directly, or import a source-backed upstream authorization rule; do "
        "not let tests supply a `source_authorized` escape hatch."
    ]


def test_judgment_positive_companion_output_requires_positive_for_or_false_rule(
    tmp_path,
):
    policy_repo = _canonical_rulespec_content_root(tmp_path, "us-co")
    rules_file = policy_repo / "regulations" / "10-ccr-2506-1" / "4.801.43.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: terminated_claim_reactivation_prohibited
    kind: derived
    entity: SnapClaim
    dtype: Judgment
    period: Month
    source: 10 CCR 2506-1 4.801.43(A)
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          claim_is_terminated
          or false
"""
    cases = [
        {
            "name": "not_terminated",
            "period": "2026-01",
            "input": {
                "us-co:regulations/10-ccr-2506-1/4.801.43#input.claim_is_terminated": False,
            },
            "output": {
                "us-co:regulations/10-ccr-2506-1/4.801.43#terminated_claim_reactivation_prohibited": "not_holds"
            },
        }
    ]

    issues = find_judgment_positive_companion_output_issues(
        content,
        cases,
        rules_file=rules_file,
        policy_repo_path=policy_repo,
    )

    assert issues == [
        "Judgment rule missing positive companion output coverage: "
        "`us-co:regulations/10-ccr-2506-1/4.801.43#terminated_claim_reactivation_prohibited` "
        "is not asserted as `holds` by the companion `.test.yaml` file."
    ]


def test_judgment_positive_companion_output_allows_holds_case(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    rules_file = policy_repo / "statutes" / "26" / "3102" / "f" / "1.yaml"
    rules_file.parent.mkdir(parents=True)
    content = """format: rulespec/v1
rules:
  - name: additional_medicare_collection_applies
    kind: derived
    entity: Payment
    dtype: Judgment
    period: Year
    source: 26 USC 3102(f)(1)
    versions:
      - effective_from: '2013-01-01'
        formula: wages_above_threshold > 0
"""
    cases = [
        {
            "name": "above_threshold",
            "period": "2026",
            "input": {},
            "output": {
                "us:statutes/26/3102/f/1#additional_medicare_collection_applies": [
                    "holds"
                ]
            },
        }
    ]

    assert (
        find_judgment_positive_companion_output_issues(
            content,
            cases,
            rules_file=rules_file,
            policy_repo_path=policy_repo,
        )
        == []
    )


def test_test_input_assignment_scopes_inputs_to_asserted_outputs():
    content = """format: rulespec/v1
module:
  proof_validation:
    required: true
rules:
  - name: table_value
    kind: parameter
    dtype: Money
    indexed_by: household_size
    versions:
      - effective_from: '2026-01-01'
        values:
          1: 209
          2: 209
  - name: deduction
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: table_value[household_size]
  - name: asset_limit_amount
    kind: parameter
    dtype: Money
    versions:
      - effective_from: '2026-01-01'
        formula: '3000'
  - name: asset_limit
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    versions:
      - effective_from: '2026-01-01'
        formula: asset_limit_amount
"""
    cases = [
        {
            "name": "asset_limit_does_not_need_household_size",
            "period": "2026-01",
            "input": {},
            "output": {
                "us:policies/usda/snap/fy-2026-cola/deductions#asset_limit": 3000
            },
        }
    ]

    assert find_test_input_assignment_issues(content, cases) == []


def test_same_section_subsection_import_accepts_transitive_child_import(tmp_path):
    policy_repo = tmp_path / "rulespec-us"
    child = policy_repo / "statutes" / "7" / "2015" / "d" / "2" / "C.yaml"
    child.parent.mkdir(parents=True)
    child.write_text(
        "format: rulespec/v1\nimports:\n  - us:statutes/7/2015/e\nrules: []\n"
    )
    parent = policy_repo / "statutes" / "7" / "2015" / "d" / "2.yaml"
    parent.parent.mkdir(parents=True, exist_ok=True)
    parent.write_text("placeholder\n")
    content = """format: rulespec/v1
module:
  summary: |-
    A person shall be exempt if the person is a student, except that a person
    enrolled in an institution of higher education is ineligible unless the
    person meets the requirements of subsection (e) of this section.
imports:
  - us:statutes/7/2015/d/2/C
rules: []
"""

    issues = find_missing_same_section_subsection_import_issues(
        content,
        rules_file=parent,
        policy_repo_path=policy_repo,
    )

    assert issues == []


def test_unused_imports_are_rejected():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/63/c#standard_deduction
  - us:statutes/26/163/a#interest_deduction
rules:
  - name: taxable_income
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: adjusted_gross_income - standard_deduction
"""

    issues = find_unused_import_issues(content)

    assert issues == [
        "Unused import `us:statutes/26/163/a#interest_deduction`: imported "
        "symbol `interest_deduction` is not referenced by any formula or proof "
        "import."
    ]


def test_proof_imports_must_be_referenced_by_rule_formula():
    content = """format: rulespec/v1
imports:
  - us:statutes/26/151#section_151_exemption_deduction
rules:
  - name: section_931_disallowed_deductions_excluding_section_151
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:statutes/26/151#section_151_exemption_deduction
              output: section_151_exemption_deduction
              hash: sha256:abc
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          min(deductions_before_denial, allocable_deductions)
"""

    issues = find_proof_import_reference_issues(content)

    assert issues == [
        "Proof import not referenced: "
        "`section_931_disallowed_deductions_excluding_section_151` proof imports "
        "`section_151_exemption_deduction`, but the rule formula does not "
        "reference that imported symbol."
    ]


def test_import_shape_rejects_map_entries():
    content = """format: rulespec/v1
imports:
  - target: us:statutes/26/45A/a
    symbols:
      - base_year_1993_indian_employment_costs
rules: []
"""

    issues = find_import_shape_issues(content)

    assert len(issues) == 1
    assert "Import shape invalid" in issues[0]
    assert "imports[0]" in issues[0]
    assert "scalar string" in issues[0]


def test_import_shape_rejects_unprefixed_targets():
    content = """format: rulespec/v1
imports:
  - statutes/26/24/h#ctc_refundable_maximum_under_subsection_h
rules: []
"""

    issues = find_import_shape_issues(content)

    assert len(issues) == 1
    assert "Import target invalid" in issues[0]
    assert "imports[0]" in issues[0]
    assert "absolute RuleSpec targets" in issues[0]


@pytest.mark.parametrize(
    "target",
    [
        "programs/snap/fy-2026#snap_eligible",
        "us:programs/snap/fy-2026#snap_eligible",
    ],
)
def test_import_shape_rejects_composition_spec_targets(target):
    content = f"""format: rulespec/v1
imports:
  - {target}
rules: []
"""

    issues = find_import_shape_issues(content)

    assert len(issues) == 1
    assert "Import target invalid" in issues[0]
    if target.startswith("us:"):
        assert "axiom-compose ProgramSpecs" in issues[0]


def _mock_corpus_source_text(monkeypatch, text: str) -> None:
    monkeypatch.setattr(
        "axiom_encode.harness.validator_pipeline._fetch_corpus_source_text",
        lambda _citation_path: text,
    )


def _write_local_corpus_provision(
    repo_parent: Path,
    citation_path: str,
    body: str = "Authoritative source text.",
) -> LocalCorpusRelease:
    version = TEST_CORPUS_VERSION
    parts = citation_path.split("/")
    corpus_root = repo_parent / "axiom-corpus"
    provisions_dir = corpus_root / "data" / "corpus" / "provisions"
    provisions_dir = provisions_dir / parts[0] / parts[1]
    provisions_dir.mkdir(parents=True, exist_ok=True)
    (provisions_dir / f"{version}.jsonl").write_text(
        json.dumps(
            _active_corpus_record(
                citation_path,
                body,
                version=version,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return bind_test_corpus_release(
        corpus_root,
        TEST_CORPUS_RELEASE_NAME,
        [(parts[0], parts[1], version)],
    )


def _active_corpus_record(
    citation_path: str,
    body: str | None,
    *,
    version: str = TEST_CORPUS_VERSION,
    **extra,
) -> dict:
    jurisdiction, document_class, *_rest = citation_path.split("/")
    return {
        "id": f"test:{citation_path}",
        "citation_path": citation_path,
        "body": body,
        "jurisdiction": jurisdiction,
        "document_class": document_class,
        "version": version,
        "source_path": f"sources/{jurisdiction}/{document_class}/test",
        "source_as_of": "2026-01-01",
        "expression_date": "2026-01-01",
        **extra,
    }


def _test_corpus_release(
    corpus_root: Path,
    *scopes: tuple[str, str, str],
) -> LocalCorpusRelease:
    return bind_test_corpus_release(
        corpus_root,
        TEST_CORPUS_RELEASE_NAME,
        list(scopes),
    )


def test_promoted_stub_check_uses_corpus_provisions(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    release = _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")

    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=release,
    )

    issues = pipeline._check_promoted_stub_file(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


def test_promoted_stub_ignores_unsigned_duplicate_artifact(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    citation = "us/statute/7/2014/e/4"
    release = _write_local_corpus_provision(repo_parent, citation)
    provisions = repo_parent / "axiom-corpus/data/corpus/provisions/us/statute"
    duplicate = _active_corpus_record(citation, "Duplicate active source.")
    duplicate["id"] = "duplicate-active-row"
    provisions.joinpath("duplicate.jsonl").write_text(
        json.dumps(duplicate) + "\n", encoding="utf-8"
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=release,
    )

    issues = pipeline._check_promoted_stub_file(rules_file)

    assert len(issues) == 1
    assert "corpus.provisions has source text" in issues[0]
    assert "Dependency corpus check failed" not in issues[0]


def test_local_only_promoted_stub_check_ignores_ambient_corpus(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")
    explicit_corpus = tmp_path / "explicit-corpus"
    explicit_provisions = explicit_corpus / "data/corpus/provisions/us/statute"
    explicit_provisions.mkdir(parents=True)
    canonical_provisions = explicit_provisions / f"{TEST_CORPUS_VERSION}.jsonl"
    canonical_provisions.write_text(
        json.dumps(_active_corpus_record("us/statute/2", "unrelated")) + "\n",
        encoding="utf-8",
    )
    explicit_release = _test_corpus_release(
        explicit_corpus,
        ("us", "statute", TEST_CORPUS_VERSION),
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=explicit_release,
    )

    assert pipeline._check_promoted_stub_file(rules_file) == []

    canonical_provisions.write_text(
        json.dumps(
            _active_corpus_record(
                "us/statute/7/2014/e/4",
                "authoritative",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    pipeline.local_corpus_release = _test_corpus_release(
        explicit_corpus,
        ("us", "statute", TEST_CORPUS_VERSION),
    )
    assert pipeline._check_promoted_stub_file(rules_file)


def test_imported_stub_dependency_check_uses_corpus_provisions(tmp_path):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_file = rules_repo / "statutes" / "7" / "2014" / "root.yaml"
    target_file = rules_repo / "statutes" / "7" / "2014" / "e" / "4.yaml"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        "format: rulespec/v1\nmodule:\n  status: stub\nrules: []\n",
        encoding="utf-8",
    )
    rules_file.parent.mkdir(parents=True, exist_ok=True)
    rules_file.write_text(
        "format: rulespec/v1\n"
        "imports:\n"
        "  - statutes/7/2014/e/4#snap_state_uses_child_support_deduction\n"
        "rules: []\n",
        encoding="utf-8",
    )
    release = _write_local_corpus_provision(repo_parent, "us/statute/7/2014/e/4")

    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
        local_corpus_release=release,
    )

    issues = pipeline._check_imported_stub_dependencies(rules_file)

    assert issues
    assert "corpus.provisions has source text" in issues[0]


@pytest.mark.parametrize("unsafe_kind", ["traversal", "symlink"])
def test_imported_stub_dependency_rejects_unsafe_target(tmp_path, unsafe_kind):
    repo_parent = tmp_path / "repos"
    rules_repo = repo_parent / "rulespec-us"
    rules_repo.mkdir(parents=True)
    outside_file = repo_parent / "outside.yaml"
    outside_file.write_text(
        "format: rulespec/v1\n"
        "module:\n"
        "  status: stub\n"
        "  summary: OPENAI_API_KEY=sentinel-secret-value\n"
        "rules: []\n",
        encoding="utf-8",
    )
    if unsafe_kind == "traversal":
        import_path = "../outside"
    else:
        target_file = rules_repo / "statutes" / "7" / "unsafe.yaml"
        target_file.parent.mkdir(parents=True)
        target_file.symlink_to(outside_file)
        import_path = "statutes/7/unsafe"

    rules_file = rules_repo / "root.yaml"
    rules_file.write_text(
        f"format: rulespec/v1\nimports:\n  - {import_path}#unsafe_symbol\nrules: []\n",
        encoding="utf-8",
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=rules_repo,
        axiom_rules_path=repo_parent / "axiom-rules-engine",
        enable_oracles=False,
    )

    issues = pipeline._check_imported_stub_dependencies(rules_file)

    assert len(issues) == 1
    assert "Unsafe imported RuleSpec dependency" in issues[0]


def test_rulespec_compile_ci_and_grounding(real_engine, tmp_path, monkeypatch):
    _mock_corpus_source_text(
        monkeypatch,
        "The official source states the standard utility allowance is $451.",
    )

    policy_repo, rules_file = _canonical_rulespec_test_file(tmp_path)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: |-
    The standard utility allowance is $451.
  source_verification:
    corpus_citation_path: us/guidance/example/sua
rules:
  - name: standard_utility_allowance_value
    kind: parameter
    dtype: Money
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          451
  - name: standard_utility_allowance
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2024-01-01'
        formula: |-
          standard_utility_allowance_value
"""
    )
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: base
  period: 2024-01
  input: {}
  output:
    us:policies/example/rules#standard_utility_allowance: 451
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    assert pipeline._run_compile_check(rules_file).passed is True
    assert pipeline._run_ci(rules_file).passed is True
    assert extract_embedded_source_text(rules_file.read_text()).startswith(
        "The standard utility allowance"
    )
    assert extract_grounding_values(rules_file.read_text()) == [(1, "451", 451.0)]
    assert [
        (item.name, item.value)
        for item in extract_named_scalar_occurrences(rules_file.read_text())
    ] == [("standard_utility_allowance_value", 451.0)]


def test_rulespec_ci_rejects_repo_backed_friendly_output_keys(real_engine, tmp_path):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    household_allotment_input: 298
  output:
    snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "must use legal RuleSpec id" in issue
        and "us:statutes/7/2017/a#snap_regular_month_allotment" in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_unresolved_output_reference_path(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.household_allotment_input: 298
  output:
    us:statutes/7/9999/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/9999/a#snap_regular_month_allotment` points to a RuleSpec file that could not be resolved"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_input_reference_in_output_position(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: household_allotment_input
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.household_allotment_input: 298
  output:
    us:statutes/7/2017/a#input.household_allotment_input: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "output `us:statutes/7/2017/a#input.household_allotment_input` resolves to an input slot, which is not allowed here"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_friendly_input_keys(real_engine, tmp_path):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `snap_maximum_allotment` must use an absolute legal RuleSpec id" in issue
        for issue in result.issues
    )


def test_rulespec_ci_executes_repo_backed_absolute_input_keys(real_engine, tmp_path):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#input.snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_input_reference_path(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/9999/a#input.snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `us:statutes/7/9999/a#input.snap_maximum_allotment` points to a RuleSpec file that could not be resolved"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_unresolved_input_reference_fragment(
    real_engine,
    tmp_path,
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2017"
        / "a.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2017(a) sets the regular SNAP allotment.
rules:
  - name: snap_regular_month_allotment
    kind: derived
    entity: Household
    dtype: Money
    period: Month
    unit: USD
    versions:
      - effective_from: '2025-10-01'
        formula: snap_maximum_allotment
"""
    )
    rules_file.with_name("a.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2017/a#snap_maximum_allotment: 298
  output:
    us:statutes/7/2017/a#snap_regular_month_allotment: 298
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `us:statutes/7/2017/a#snap_maximum_allotment` does not resolve to an input slot, derived rule, or parameter"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_rejects_repo_backed_friendly_relation_child_input_keys(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2012"
        / "j.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "input `snap_member_is_elderly_or_disabled` must use an absolute legal RuleSpec id"
        in issue
        for issue in result.issues
    )


def test_rulespec_ci_executes_repo_backed_absolute_relation_child_input_keys(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2012"
        / "j.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
        enforce_repository_layout=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is True
    assert result.issues == []


def test_rulespec_ci_rejects_repo_backed_unresolved_relation_reference(
    real_engine, tmp_path
):
    rules_file = (
        _canonical_rulespec_content_root(tmp_path, "us")
        / "statutes"
        / "7"
        / "2012"
        / "j.yaml"
    )
    rules_file.parent.mkdir(parents=True)
    rules_file.write_text(
        """format: rulespec/v1
module:
  summary: 7 USC 2012(j) defines SNAP elderly or disabled household members.
rules:
  - name: member_of_household
    kind: data_relation
    data_relation:
      arity: 2
      arguments: [Person, Household]
  - name: snap_household_has_elderly_or_disabled_member
    kind: derived
    entity: Household
    dtype: Judgment
    period: Month
    versions:
      - effective_from: '2025-10-01'
        formula: count_where(member_of_household, snap_member_is_elderly_or_disabled) > 0
"""
    )
    rules_file.with_name("j.test.yaml").write_text(
        """- name: base
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.not_member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    pipeline = ValidatorPipeline(
        policy_repo_path=_canonical_rulespec_content_root(tmp_path, "us"),
        axiom_rules_path=real_engine.root,
        enable_oracles=False,
    )

    result = pipeline._run_ci(rules_file)

    assert result.passed is False
    assert any(
        "relation input `us:statutes/7/2012/j#relation.not_member_of_household` does not resolve to a declared relation"
        in issue
        for issue in result.issues
    )


def test_oracle_test_extraction_normalizes_legal_output_ids(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: base
  period: 2026-01
  input:
    household_size: 1
  output:
    us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction: 209
"""
    )

    assert tests == [
        {
            "variable": "snap_standard_deduction",
            "raw_variable": "us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction",
            "name": "base",
            "period": "2026-01",
            "inputs": {"household_size": 1},
            "expect": 209,
        }
    ]


def test_oracle_test_extraction_aliases_legal_input_ids(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: wage_tax
  period: 2026
  input:
    us:statutes/26/3101/a#input.wages: 100000
  output:
    us:statutes/26/3101/a#oasdi_wage_tax: 6200
"""
    )

    assert tests[0]["inputs"]["us:statutes/26/3101/a#input.wages"] == 100000
    assert tests[0]["inputs"]["wages"] == 100000


def test_oracle_test_extraction_preserves_policyengine_only_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: utility_region
  period: 2026-01
  input:
    us-ny:regulations/18-nycrr/387/12/f/3/v/a#input.household_resides_in_new_york_city: false
  oracle_inputs:
    policyengine:
      snap_utility_region_str: NY_NAS
  output:
    us-ny:regulations/18-nycrr/387/12/f/3/v/a#snap_standard_utility_allowance: 988
"""
    )

    assert tests[0]["inputs"] == {
        "us-ny:regulations/18-nycrr/387/12/f/3/v/a#input.household_resides_in_new_york_city": False,
        "household_resides_in_new_york_city": False,
    }
    assert tests[0]["oracle_inputs"] == {
        "policyengine": {"snap_utility_region_str": "NY_NAS"}
    }


def test_policyengine_expected_float_normalizes_judgment_expectations():
    assert _policyengine_expected_float("holds") == 1.0
    assert _policyengine_expected_float("not_holds") == 0.0
    assert _policyengine_expected_float(209) == 209.0


def test_oracle_test_extraction_preserves_relation_list_inputs(tmp_path):
    pipeline = ValidatorPipeline(
        policy_repo_path=tmp_path,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=False,
    )

    tests = pipeline._extract_rulespec_tests(
        """- name: relation_case
  period: 2026-01
  input:
    us:statutes/7/2012/j#relation.member_of_household:
      - us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled: true
  output:
    us:statutes/7/2012/j#snap_household_has_elderly_or_disabled_member: holds
"""
    )

    assert tests[0]["inputs"]["us:statutes/7/2012/j#relation.member_of_household"] == [
        {"us:statutes/7/2012/j#input.snap_member_is_elderly_or_disabled": True}
    ]


def test_policyengine_oracle_does_not_score_unmapped_outputs(tmp_path):
    policy_repo, rules_file = _canonical_rulespec_test_file(tmp_path)
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: mapped
  period: 2026-01
  input: {}
  output:
    us:policies/usda/snap/fy-2026-cola/deductions#snap_standard_deduction: 209
- name: unmapped
  period: 2026-01
  input: {}
  output:
    us:test/fake#unmapped_var: 99
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")
    pipeline._should_compare_pe_test_output = lambda *_args, **_kwargs: True
    pipeline._is_pe_test_mappable = lambda *_args, **_kwargs: (True, None)
    pipeline._build_pe_scenario_script = lambda *_args, **_kwargs: ""
    pipeline._run_pe_subprocess_detailed = lambda *_args, **_kwargs: (
        OracleSubprocessResult(returncode=0, stdout="RESULT:209\n")
    )

    result = pipeline._run_policyengine(rules_file)

    assert result.score == 1.0
    assert result.passed is True
    assert result.issues == []
    assert result.details["coverage"]["comparable"] == 1
    assert result.details["coverage"]["passed"] == 1
    assert result.details["coverage"]["unmapped"] == 1


def test_policyengine_oracle_skips_unprojectable_legal_inputs(tmp_path):
    policy_repo, rules_file = _canonical_rulespec_test_file(tmp_path)
    rules_file.write_text("format: rulespec/v1\n")
    rules_file.with_name("rules.test.yaml").write_text(
        """- name: statutory_taxable_income_proof
  period: 2026
  input:
    us:statutes/26/63#input.gross_income: 100000
    us:statutes/26/63#input.deductions_allowed_by_this_chapter_other_than_standard_deduction: 15000
  output:
    us:statutes/26/63#taxable_income: 85000
"""
    )
    pipeline = ValidatorPipeline(
        policy_repo_path=policy_repo,
        axiom_rules_path=AXIOM_RULES_PATH,
        enable_oracles=True,
        oracle_validators=("policyengine",),
    )
    pipeline._detect_policyengine_country = lambda *_args, **_kwargs: "us"
    pipeline._find_pe_python = lambda _country: Path("python")

    ran_policyengine = False

    def run_policyengine(*_args, **_kwargs):
        nonlocal ran_policyengine
        ran_policyengine = True
        return OracleSubprocessResult(returncode=0, stdout="RESULT:0\n")

    pipeline._run_pe_subprocess_detailed = run_policyengine

    result = pipeline._run_policyengine(rules_file)

    assert ran_policyengine is False
    assert result.passed is False
    assert result.score is None
    assert result.error == "PolicyEngine produced zero comparable oracle evidence"
    assert result.details["coverage"]["unsupported"] == 1
    assert any("unprojectable RuleSpec legal input" in issue for issue in result.issues)


def test_policyengine_registry_is_legal_id_keyed():
    registry = load_policyengine_registry()

    mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/e/2#snap_earned_income_deduction",
        country="us",
    )

    assert mapping is not None
    assert mapping.policyengine_variable == "snap_earned_income_deduction"
    assert registry.mapping_for_legal_id("snap_earned_income_deduction") is None
    earned_income_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/e/2#snap_earned_income_deduction_rate",
        country="us",
    )
    assert earned_income_rate_mapping.mapping_type == "parameter_value"
    assert (
        earned_income_rate_mapping.policyengine_parameter
        == "gov.usda.snap.income.deductions.earned_income"
    )
    ssi_general_exclusion_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382a/b/2#annual_general_income_exclusion_limit",
        country="us",
    )
    assert ssi_general_exclusion_mapping.mapping_type == "parameter_value"
    assert (
        ssi_general_exclusion_mapping.policyengine_parameter
        == "gov.ssa.ssi.income.exclusions.general"
    )
    assert ssi_general_exclusion_mapping.result_multiplier == 12
    ssi_earned_exclusion_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382a/b/4#annual_earned_income_initial_exclusion_limit",
        country="us",
    )
    assert ssi_earned_exclusion_mapping.mapping_type == "parameter_value"
    assert (
        ssi_earned_exclusion_mapping.policyengine_parameter
        == "gov.ssa.ssi.income.exclusions.earned"
    )
    assert ssi_earned_exclusion_mapping.result_multiplier == 12
    ssi_earned_share_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382a/b/4#earned_income_remainder_exclusion_rate",
        country="us",
    )
    assert ssi_earned_share_mapping.mapping_type == "parameter_value"
    assert (
        ssi_earned_share_mapping.policyengine_parameter
        == "gov.ssa.ssi.income.exclusions.earned_share"
    )
    ssi_blind_expense_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382a/b/4#blind_branch_earning_expenses_excluded",
        country="us",
    )
    assert ssi_blind_expense_mapping.mapping_type == "not_comparable"
    assert ssi_blind_expense_mapping.policyengine_variable == "ssi_countable_income"
    ssi_individual_fbr_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382f/a#amount_determined_under_section_1382f_for_section_1382_b_1",
        country="us",
    )
    assert ssi_individual_fbr_mapping.mapping_type == "parameter_value"
    assert (
        ssi_individual_fbr_mapping.policyengine_parameter
        == "gov.ssa.ssi.amount.individual"
    )
    assert ssi_individual_fbr_mapping.result_multiplier == 12
    ssi_couple_fbr_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382f/a#amount_determined_under_section_1382f_for_section_1382_b_2",
        country="us",
    )
    assert ssi_couple_fbr_mapping.mapping_type == "parameter_value"
    assert ssi_couple_fbr_mapping.policyengine_parameter == "gov.ssa.ssi.amount.couple"
    assert ssi_couple_fbr_mapping.result_multiplier == 12
    ssi_aged_threshold_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382c/a/1#aged_age_threshold_years",
        country="us",
    )
    assert ssi_aged_threshold_mapping.mapping_type == "parameter_value"
    assert (
        ssi_aged_threshold_mapping.policyengine_parameter
        == "gov.ssa.ssi.eligibility.aged_threshold"
    )
    ssi_full_abd_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382c/a/1#aged_blind_or_disabled_individual",
        country="us",
    )
    assert ssi_full_abd_mapping.mapping_type == "not_comparable"
    assert ssi_full_abd_mapping.policyengine_variable == "is_ssi_aged_blind_disabled"
    ssi_blind_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382c/a/2#blind_for_subchapter",
        country="us",
    )
    assert ssi_blind_mapping.mapping_type == "not_comparable"
    assert ssi_blind_mapping.policyengine_variable == "is_blind"
    ssi_individual_eligibility_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382/a/1#eligible_individual",
        country="us",
    )
    assert ssi_individual_eligibility_mapping.mapping_type == "not_comparable"
    assert ssi_individual_eligibility_mapping.policyengine_variable == "is_ssi_eligible"
    ssi_couple_eligibility_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382/a/2#eligible_individual_with_eligible_spouse",
        country="us",
    )
    assert ssi_couple_eligibility_mapping.mapping_type == "not_comparable"
    assert (
        ssi_couple_eligibility_mapping.policyengine_variable == "is_ssi_eligible_spouse"
    )
    ssi_individual_benefit_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382/b#annual_benefit_without_eligible_spouse",
        country="us",
    )
    assert ssi_individual_benefit_mapping.mapping_type == "not_comparable"
    assert ssi_individual_benefit_mapping.policyengine_variable == "ssi"
    ssi_couple_benefit_mapping = registry.mapping_for_legal_id(
        "us:statutes/42/1382/b#annual_benefit_with_eligible_spouse",
        country="us",
    )
    assert ssi_couple_benefit_mapping.mapping_type == "not_comparable"
    assert ssi_couple_benefit_mapping.policyengine_variable == "tax_unit_ssi"
    maximum_allotment_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/maximum-allotments#snap_maximum_allotment_table",
        country="us",
    )
    assert maximum_allotment_mapping.mapping_type == "parameter_value"
    assert (
        maximum_allotment_mapping.policyengine_parameter
        == "gov.usda.snap.max_allotment.main.CONTIGUOUS_US"
    )
    assert maximum_allotment_mapping.parameter_key_input == "household_size"
    gross_income_limit_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/income-eligibility-standards#snap_gross_income_limit_130_percent_fpl_48_states_dc",
        country="us",
    )
    assert gross_income_limit_mapping.mapping_type == "not_comparable"
    assert gross_income_limit_mapping.candidate_priority == "P4"
    assert gross_income_limit_mapping.policyengine_variable is None
    colorado_ccap_smi_limit_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/8-ccr-1403-1/3.111/h-low-income-eligibility-guidelines#monthly_state_median_income_85_limit",
        country="us",
    )
    assert colorado_ccap_smi_limit_mapping.mapping_type == "direct_variable"
    assert colorado_ccap_smi_limit_mapping.policyengine_variable == "co_ccap_smi"
    assert colorado_ccap_smi_limit_mapping.result_multiplier == 0.85
    colorado_oap_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/9-ccr-2503-5/3.532#oap_authorized_grant_payment_for_month",
        country="us",
    )
    assert colorado_oap_mapping.mapping_type == "direct_variable"
    assert colorado_oap_mapping.policyengine_variable == "co_oap"
    assert colorado_oap_mapping.result_multiplier == pytest.approx(1 / 12)
    colorado_oap_grant_standard_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/9-ccr-2503-5/3.530#oap_total_monthly_grant_standard",
        country="us",
    )
    assert colorado_oap_grant_standard_mapping.mapping_type == "parameter_value"
    assert (
        colorado_oap_grant_standard_mapping.policyengine_parameter
        == "gov.states.co.ssa.oap.grant_standard"
    )
    colorado_ssp_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/9-ccr-2503-5/3.548#and_cs_authorized_grant_payment",
        country="us",
    )
    assert colorado_ssp_mapping.mapping_type == "direct_variable"
    assert colorado_ssp_mapping.policyengine_variable == "co_state_supplement"
    assert colorado_ssp_mapping.result_multiplier is None
    colorado_ssp_legacy_name_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/9-ccr-2503-5/3.548#and_cs_authorized_grant_payment_for_month",
        country="us",
    )
    assert colorado_ssp_legacy_name_mapping.mapping_type == "direct_variable"
    assert (
        colorado_ssp_legacy_name_mapping.policyengine_variable == "co_state_supplement"
    )
    assert colorado_ssp_legacy_name_mapping.result_multiplier is None
    ca_capi_mapping = registry.mapping_for_legal_id(
        "us-ca:regulations/cdss/eas/49/49-055#ca_capi",
        country="us",
    )
    assert ca_capi_mapping.mapping_type == "direct_variable"
    assert ca_capi_mapping.policyengine_variable == "ca_capi"
    assert ca_capi_mapping.result_multiplier is None
    il_aabd_personal_allowance_mapping = registry.mapping_for_legal_id(
        "us-il:policies/dhs/csmm/11-01-01/personal-allowance#il_aabd_personal_allowance",
        country="us",
    )
    assert il_aabd_personal_allowance_mapping.mapping_type == "direct_variable"
    assert (
        il_aabd_personal_allowance_mapping.policyengine_variable
        == "il_aabd_personal_allowance"
    )
    assert il_aabd_personal_allowance_mapping.period == "month"
    assert il_aabd_personal_allowance_mapping.comparison == "money"
    colorado_ssp_grant_standard_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/9-ccr-2503-5/3.546#and_cs_total_grant_standard",
        country="us",
    )
    assert colorado_ssp_grant_standard_mapping.mapping_type == "parameter_value"
    assert (
        colorado_ssp_grant_standard_mapping.policyengine_parameter
        == "gov.states.co.ssa.state_supplement.grant_standard"
    )
    michigan_ssp_independent_living_mapping = registry.mapping_for_legal_id(
        "us-mi:policies/mdhhs/rft/248#independent_living_individual_state_ssi_payment",
        country="us",
    )
    assert michigan_ssp_independent_living_mapping.mapping_type == "parameter_value"
    assert (
        michigan_ssp_independent_living_mapping.policyengine_parameter
        == "gov.states.mi.mdhhs.ssp.payment.individual"
    )
    assert michigan_ssp_independent_living_mapping.parameter_key_path == (
        "INDEPENDENT_LIVING",
    )
    michigan_ssp_couple_independent_living_mapping = registry.mapping_for_legal_id(
        "us-mi:policies/mdhhs/rft/248#independent_living_couple_state_ssi_payment",
        country="us",
    )
    assert michigan_ssp_couple_independent_living_mapping.mapping_type == (
        "parameter_value"
    )
    assert (
        michigan_ssp_couple_independent_living_mapping.policyengine_parameter
        == "gov.states.mi.mdhhs.ssp.payment.couple"
    )
    assert michigan_ssp_couple_independent_living_mapping.parameter_key_path == (
        "INDEPENDENT_LIVING",
    )
    assert michigan_ssp_couple_independent_living_mapping.result_multiplier == 0.5
    michigan_ssp_couple_domiciliary_care_mapping = registry.mapping_for_legal_id(
        "us-mi:policies/mdhhs/rft/248#domiciliary_care_couple_dhs_supplement",
        country="us",
    )
    assert michigan_ssp_couple_domiciliary_care_mapping.mapping_type == (
        "parameter_value"
    )
    assert (
        michigan_ssp_couple_domiciliary_care_mapping.policyengine_parameter
        == "gov.states.mi.mdhhs.ssp.payment.couple"
    )
    assert michigan_ssp_couple_domiciliary_care_mapping.parameter_key_path == (
        "DOMICILIARY_CARE",
    )
    assert michigan_ssp_couple_domiciliary_care_mapping.result_multiplier is None
    michigan_ssp_child_mapping = registry.mapping_for_legal_id(
        "us-mi:policies/mdhhs/rft/248#eligible_child_state_ssi_payment",
        country="us",
    )
    assert michigan_ssp_child_mapping.mapping_type == "not_comparable"
    michigan_ssp_person_mapping = registry.mapping_for_legal_id(
        "us-mi:policies/mdhhs/rft/248#mi_ssp_person",
        country="us",
    )
    assert michigan_ssp_person_mapping.mapping_type == "not_comparable"
    assert michigan_ssp_person_mapping.policyengine_variable == "mi_ssp_person"
    wa_ssp_eligibility_mapping = registry.mapping_for_legal_id(
        "us-wa:regulations/388/388-474/388-474-0012#individual_is_eligible_for_ssp_as_described",
        country="us",
    )
    assert wa_ssp_eligibility_mapping.mapping_type == "not_comparable"
    assert wa_ssp_eligibility_mapping.policyengine_variable == "wa_ssp"
    wa_ssp_ineligible_spouse_mapping = registry.mapping_for_legal_id(
        "us-wa:regulations/388/388-474/388-474-0012#ssp_eligibility_based_on_ineligible_spouse_status",
        country="us",
    )
    assert wa_ssp_ineligible_spouse_mapping.mapping_type == "not_comparable"
    assert wa_ssp_ineligible_spouse_mapping.policyengine_variable == "wa_ssp"
    section_2014c_net_failure_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_net_income_exceeds_poverty_line",
        country="us",
    )
    assert section_2014c_net_failure_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_net_failure_mapping.policyengine_variable
        == "meets_snap_net_income_test"
    )
    wic_certification_mapping = registry.mapping_for_legal_id(
        "us:regulations/7-cfr/246/7/c#applicant_meets_basic_wic_eligibility_criteria",
        country="us",
    )
    assert wic_certification_mapping.mapping_type == "not_comparable"
    assert wic_certification_mapping.program == "wic"
    assert wic_certification_mapping.candidate_priority == "P4"
    assert "certification-procedure helper outputs" in (
        wic_certification_mapping.rationale or ""
    )
    wic_income_mapping = registry.mapping_for_legal_id(
        "us:regulations/7-cfr/246/7/d#applicant_adjunctively_income_eligible",
        country="us",
    )
    assert wic_income_mapping.mapping_type == "not_comparable"
    assert wic_income_mapping.program == "wic"
    early_head_start_age_mapping = registry.mapping_for_legal_id(
        "us:regulations/45-cfr/1302/12/b#early_head_start_age_threshold_years",
        country="us",
    )
    assert early_head_start_age_mapping.mapping_type == "parameter_value"
    assert (
        early_head_start_age_mapping.policyengine_parameter
        == "gov.hhs.head_start.early_head_start.age_limit"
    )
    head_start_age_mapping = registry.mapping_for_legal_id(
        "us:regulations/45-cfr/1302/12/b#head_start_preschool_age_threshold_years",
        country="us",
    )
    assert head_start_age_mapping.mapping_type == "parameter_value"
    assert (
        head_start_age_mapping.policyengine_parameter == "gov.hhs.head_start.age_range"
    )
    assert head_start_age_mapping.parameter_key_path == ("thresholds", 1)
    head_start_overincome_exception_mapping = registry.mapping_for_legal_id(
        "us:regulations/45-cfr/1302/12#paragraph_c_overincome_exception_participant_may_be_enrolled",
        country="us",
    )
    assert head_start_overincome_exception_mapping.mapping_type == "not_comparable"
    assert head_start_overincome_exception_mapping.program == "head_start"
    chip_definition_mappings = [
        "us:statutes/42/1397jj/b/1#medicaid_applicable_income_level_additional_percentage_points",
        "us:statutes/42/1397jj/b/1#targeted_low_income_child",
        "us:statutes/42/1397jj/c/1#child",
        "us:statutes/42/1397jj/c/1#child_age_limit",
        "us:statutes/42/1397jj/c/4#low_income_child",
        "us:statutes/42/1397jj/c/4#low_income_child_family_income_limit",
        "us:statutes/42/1397jj/c/8#uncovered_child",
    ]
    for legal_id in chip_definition_mappings:
        mapping = registry.mapping_for_legal_id(legal_id, country="us")
        assert mapping.mapping_type == "not_comparable"
        assert mapping.program == "chip"
        assert mapping.candidate_priority == "P4"
        assert mapping.policyengine_variable == "is_chip_eligible_child"
    pregnant_chip_definition_mappings = [
        "us:statutes/42/1397ll/d/2#standard_postpartum_period_days",
        "us:statutes/42/1397ll/d/2#extended_postpartum_period_months",
        "us:statutes/42/1397ll/d/2#targeted_low_income_pregnant_woman_extended_postpartum_period_months",
        "us:statutes/42/1397ll/d/2#pregnant_woman_income_floor_rate",
        "us:statutes/42/1397ll/d/2#applicable_pregnant_woman_income_floor_rate",
        "us:statutes/42/1397ll/d/2#within_pregnancy_or_applicable_postpartum_period",
        "us:statutes/42/1397ll/d/2#targeted_low_income_pregnant_woman",
    ]
    for legal_id in pregnant_chip_definition_mappings:
        mapping = registry.mapping_for_legal_id(legal_id, country="us")
        assert mapping.mapping_type == "not_comparable"
        assert mapping.program == "chip"
        assert mapping.candidate_priority == "P4"
        assert (
            mapping.policyengine_variable == "is_chip_eligible_standard_pregnant_person"
        )
    child_chip_composition_mapping = registry.mapping_for_legal_id(
        "us-co:policies/cms/colorado-chip-eligibility#is_chip_eligible_child",
        country="us",
    )
    assert child_chip_composition_mapping.mapping_type == "direct_variable"
    assert child_chip_composition_mapping.program == "chip"
    assert (
        child_chip_composition_mapping.policyengine_variable == "is_chip_eligible_child"
    )
    pregnant_chip_composition_mapping = registry.mapping_for_legal_id(
        "us-co:policies/cms/colorado-chip-eligibility#is_chip_eligible_standard_pregnant_person",
        country="us",
    )
    assert pregnant_chip_composition_mapping.mapping_type == "direct_variable"
    assert (
        pregnant_chip_composition_mapping.policyengine_variable
        == "is_chip_eligible_standard_pregnant_person"
    )
    child_chip_availability_mapping = registry.mapping_for_legal_id(
        "us-ca:policies/cms/california-chip-eligibility#california_separate_chip_child_eligibility_available",
        country="us",
    )
    assert child_chip_availability_mapping.mapping_type == "not_comparable"
    assert (
        child_chip_availability_mapping.policyengine_variable
        == "is_chip_eligible_child"
    )
    pregnant_chip_availability_mapping = registry.mapping_for_legal_id(
        "us-co:policies/cms/colorado-chip-eligibility#colorado_standard_pregnant_chip_eligibility_available",
        country="us",
    )
    assert pregnant_chip_availability_mapping.mapping_type == "not_comparable"
    assert (
        pregnant_chip_availability_mapping.policyengine_variable
        == "is_chip_eligible_standard_pregnant_person"
    )
    fcep_person_mapping = registry.mapping_for_legal_id(
        "us-me:policies/cms/maine-chip-eligibility#is_chip_fcep_eligible_person",
        country="us",
    )
    assert fcep_person_mapping.mapping_type == "not_comparable"
    assert fcep_person_mapping.program == "chip"
    assert fcep_person_mapping.candidate_priority == "P4"
    assert fcep_person_mapping.policyengine_variable == "is_chip_fcep_eligible_person"
    fcep_availability_mapping = registry.mapping_for_legal_id(
        "us-me:policies/cms/maine-chip-eligibility#maine_fcep_eligibility_available",
        country="us",
    )
    assert fcep_availability_mapping.mapping_type == "not_comparable"
    assert (
        fcep_availability_mapping.policyengine_variable
        == "is_chip_fcep_eligible_person"
    )
    fcep_fpl_limit_mapping = registry.mapping_for_legal_id(
        "us-me:policies/cms/maine-chip-eligibility#maine_fcep_fpl_limit",
        country="us",
    )
    assert fcep_fpl_limit_mapping.mapping_type == "not_comparable"
    assert fcep_fpl_limit_mapping.program == "chip"
    assert (
        fcep_fpl_limit_mapping.policyengine_variable == "is_chip_fcep_eligible_person"
    )
    fcep_effective_fpl_limit_mapping = registry.mapping_for_legal_id(
        "us-me:policies/cms/maine-chip-eligibility#maine_fcep_effective_fpl_limit",
        country="us",
    )
    assert fcep_effective_fpl_limit_mapping.mapping_type == "not_comparable"
    assert (
        fcep_effective_fpl_limit_mapping.policyengine_variable
        == "is_chip_fcep_eligible_person"
    )
    residential_clean_energy_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25D#residential_clean_energy_credit",
        country="us",
    )
    assert residential_clean_energy_mapping.mapping_type == "not_comparable"
    assert (
        residential_clean_energy_mapping.policyengine_variable
        == "residential_clean_energy_credit"
    )
    assert "PolicyEngine/policyengine-us#8694" in (
        residential_clean_energy_mapping.rationale or ""
    )
    residential_clean_energy_percentage_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25D#residential_clean_energy_credit_applicable_percentage",
        country="us",
    )
    assert residential_clean_energy_percentage_mapping.mapping_type == "not_comparable"
    assert (
        residential_clean_energy_percentage_mapping.policyengine_parameter
        == "gov.irs.credits.residential_clean_energy.applicable_percentage"
    )
    section_2014c_gross_failure_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#household_fails_gross_income_standard",
        country="us",
    )
    assert section_2014c_gross_failure_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_gross_failure_mapping.policyengine_variable
        == "meets_snap_gross_income_test"
    )
    section_2014c_income_ineligible_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#household_ineligible_to_participate_due_to_income_standards",
        country="us",
    )
    assert section_2014c_income_ineligible_mapping.mapping_type == "not_comparable"
    assert (
        section_2014c_income_ineligible_mapping.policyengine_variable
        == "is_snap_eligible"
    )
    section_2014c_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_gross_income_excess_rate_over_poverty_line",
        country="us",
    )
    assert section_2014c_rate_mapping.mapping_type == "not_comparable"
    section_2014c_poverty_line_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2014/c#snap_income_standard_poverty_line_with_territory_cap",
        country="us",
    )
    assert section_2014c_poverty_line_mapping.mapping_type == "not_comparable"
    section_1402a12_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1402/a/12#self_employment_tax_equivalent_deduction",
        country="us",
    )
    assert section_1402a12_mapping.mapping_type == "not_comparable"
    assert section_1402a12_mapping.program == "tax"
    assert section_2014c_poverty_line_mapping.policyengine_variable == "snap_fpg"
    ca_snap_benefit_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/state-plan-composition#snap_benefit",
        country="us",
    )
    assert ca_snap_benefit_mapping.mapping_type == "not_comparable"
    assert ca_snap_benefit_mapping.policyengine_variable == "snap"
    ca_snap_monthly_income_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/state-plan-composition#snap_monthly_household_income",
        country="us",
    )
    assert ca_snap_monthly_income_mapping.mapping_type == "not_comparable"
    assert ca_snap_monthly_income_mapping.policyengine_variable == "snap_gross_income"
    ca_calworks_region1_map_mapping = registry.mapping_for_legal_id(
        "us-ca:policies/cdss/calworks/maximum-aid-payment-region-1#calworks_region_1_maximum_aid_payment",
        country="us",
    )
    assert ca_calworks_region1_map_mapping.mapping_type == "parameter_value"
    assert (
        ca_calworks_region1_map_mapping.policyengine_parameter
        == "gov.states.ca.cdss.tanf.cash.monthly_payment.region1"
    )
    assert ca_calworks_region1_map_mapping.parameter_key_path == (
        {
            "input": "assistance_unit_is_exempt",
            "key_map": {"True": "exempt", "False": "non_exempt"},
        },
        {"input": "persons_on_aid", "max_value": 10},
    )
    ca_calworks_resource_mapping = registry.mapping_for_legal_id(
        "us-ca:policies/cdss/calworks/maximum-resource-limit#calworks_maximum_resource_limit",
        country="us",
    )
    assert ca_calworks_resource_mapping.mapping_type == "parameter_value"
    assert (
        ca_calworks_resource_mapping.policyengine_parameter
        == "gov.states.ca.cdss.tanf.cash.resources.limit"
    )
    shelter_cap_mapping = registry.mapping_for_legal_id(
        "us:policies/usda/snap/fy-2026-cola/deductions#snap_maximum_excess_shelter_deduction_alaska",
        country="us",
    )
    assert shelter_cap_mapping.parameter_keys == (
        "AK_URBAN",
        "AK_RURAL_1",
        "AK_RURAL_2",
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.207.2#snap_initial_month_prorated_allotment",
            country="us",
        ).mapping_type
        == "not_comparable"
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.403.1#manual_specific_output",
            country="us",
        ).match_type
        == "prefix"
    )
    assert (
        registry.mapping_for_legal_id(
            "us-co:regulations/10-ccr-2506-1/4.407.31#snap_standard_utility_allowance",
            country="us",
        ).mapping_type
        == "direct_variable"
    )
    phone_allowance_mapping = registry.mapping_for_legal_id(
        "us-co:regulations/10-ccr-2506-1/4.407.31#snap_individual_utility_allowance",
        country="us",
    )
    assert phone_allowance_mapping.mapping_type == "not_comparable"
    assert phone_allowance_mapping.policyengine_variable == (
        "snap_individual_utility_allowance"
    )
    assert phone_allowance_mapping.candidate_priority == "P4"
    ny_standard_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/a#snap_standard_utility_allowance",
        country="us",
    )
    assert ny_standard_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_standard_allowance_mapping.policyengine_variable
        == "snap_standard_utility_allowance"
    )
    ny_limited_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/b#snap_limited_utility_allowance",
        country="us",
    )
    assert ny_limited_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_limited_allowance_mapping.policyengine_variable
        == "snap_limited_utility_allowance"
    )
    ny_phone_allowance_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_individual_utility_allowance",
        country="us",
    )
    assert ny_phone_allowance_mapping.mapping_type == "direct_variable"
    assert (
        ny_phone_allowance_mapping.policyengine_variable
        == "snap_individual_utility_allowance"
    )
    ny_composition_mapping = registry.mapping_for_legal_id(
        "us-ny:policies/otda/snap/fy-2026-benefit-calculation#snap_allotment",
        country="us",
    )
    assert ny_composition_mapping.mapping_type == "not_comparable"
    ny_initial_proration_mapping = registry.mapping_for_legal_id(
        "us-ny:regulations/18-nycrr/387/14/a/1#snap_initial_month_prorated_allotment",
        country="us",
    )
    assert ny_initial_proration_mapping.mapping_type == "not_comparable"
    assert (
        registry.mapping_for_legal_id(
            "us-ny:regulations/18-nycrr/387/12/f/3/v/c#snap_telephone_allowance_eligible",
            country="us",
        ).match_type
        == "prefix"
    )
    regular_allotment_mapping = registry.mapping_for_legal_id(
        "us:statutes/7/2017/a#snap_regular_month_allotment",
        country="us",
    )
    assert regular_allotment_mapping.mapping_type == "not_comparable"
    assert regular_allotment_mapping.policyengine_variable == "snap_normal_allotment"
    assert regular_allotment_mapping.candidate_priority == "P4"
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/3101/a#oasdi_wage_tax",
            country="us",
        ).policyengine_variable
        == "employee_social_security_tax"
    )
    section_1222_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1222#net_capital_gain",
        country="us",
    )
    assert section_1222_mapping.mapping_type == "not_comparable"
    assert section_1222_mapping.match_type == "prefix"
    assert section_1222_mapping.candidate_priority == "P4"
    section_1211_limit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_loss_limit",
        country="us",
    )
    assert section_1211_limit_mapping.mapping_type == "parameter_value"
    assert (
        section_1211_limit_mapping.policyengine_parameter
        == "gov.irs.ald.loss.capital.max"
    )
    assert section_1211_limit_mapping.parameter_key == "SINGLE"
    section_1211_selected_limit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_loss_limit_by_filing_status",
        country="us",
    )
    assert section_1211_selected_limit_mapping.parameter_key_input == "filing_status"
    section_1211_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1211#other_taxpayer_capital_losses_allowed",
        country="us",
    )
    assert section_1211_formula_mapping.mapping_type == "not_comparable"
    assert section_1211_formula_mapping.candidate_priority == "P4"
    section_1212_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1212/a/1#corporation_capital_loss_carryback_to_taxable_year",
        country="us",
    )
    assert section_1212_mapping.mapping_type == "not_comparable"
    assert section_1212_mapping.match_type == "prefix"
    assert section_1212_mapping.candidate_priority == "P4"
    oasdi_wage_base_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3121/a/1#oasdi_wage_base_excess_excluded_remuneration",
        country="us",
    )
    assert oasdi_wage_base_mapping.mapping_type == "not_comparable"
    assert (
        oasdi_wage_base_mapping.policyengine_variable
        == "taxable_earnings_for_social_security"
    )
    oasdi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3101/a#oasdi_wage_tax_rate",
        country="us",
    )
    assert oasdi_rate_mapping.mapping_type == "parameter_value"
    assert (
        oasdi_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.social_security.rate.employee"
    )
    assert oasdi_rate_mapping.comparable is True
    self_employment_oasdi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/a/rate#old_age_survivors_and_disability_insurance_tax_rate",
        country="us",
    )
    assert self_employment_oasdi_rate_mapping.mapping_type == "parameter_value"
    assert (
        self_employment_oasdi_rate_mapping.policyengine_parameter
        == "gov.irs.self_employment.rate.social_security"
    )
    self_employment_oasdi_tax_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/a#old_age_survivors_and_disability_insurance_tax",
        country="us",
    )
    assert self_employment_oasdi_tax_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_oasdi_tax_mapping.policyengine_variable
        == "self_employment_social_security_tax"
    )
    employee_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3101/b/1#hospital_insurance_wage_tax_rate",
        country="us",
    )
    assert employee_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        employee_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.medicare.rate.employee"
    )
    self_employment_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/b/1/rate#self_employment_income_tax_rate",
        country="us",
    )
    assert self_employment_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        self_employment_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.self_employment.rate.medicare"
    )
    self_employment_medicare_tax_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/1401/b/1#self_employment_income_tax",
        country="us",
    )
    assert self_employment_medicare_tax_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_medicare_tax_mapping.policyengine_variable
        == "self_employment_medicare_tax"
    )
    rrta_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3201#tier_2_employee_tax",
        country="us",
    )
    assert rrta_tier_2_mapping.mapping_type == "not_comparable"
    rrta_employee_representative_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3211#tier_2_employee_representative_tax",
        country="us",
    )
    assert rrta_employee_representative_tier_2_mapping.mapping_type == "not_comparable"
    rrta_employer_tier_2_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3221#tier_2_employer_tax",
        country="us",
    )
    assert rrta_employer_tier_2_mapping.mapping_type == "not_comparable"
    employer_medicare_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/b#hospital_insurance_employer_tax_rate",
        country="us",
    )
    assert employer_medicare_rate_mapping.mapping_type == "parameter_value"
    assert (
        employer_medicare_rate_mapping.policyengine_parameter
        == "gov.irs.payroll.medicare.rate.employer"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/3111/b#hospital_insurance_employer_tax",
            country="us",
        ).policyengine_variable
        == "employer_medicare_tax"
    )
    aca_36b_mappings = {
        "us:statutes/26/36B#applicable_taxpayer_minimum_household_income_poverty_line_ratio": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B#applicable_taxpayer_maximum_household_income_poverty_line_ratio": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B#applicable_taxpayer_income_percentage_test_met": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B/c/1/A#applicable_taxpayer_lower_poverty_line_percentage": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B/c/1/A#applicable_taxpayer_upper_poverty_line_percentage": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B/c/1/A#applicable_taxpayer": (
            "policyengine_parameter",
            "gov.aca.ptc_income_eligibility",
        ),
        "us:statutes/26/36B#employer_sponsored_coverage_affordability_ratio": (
            "policyengine_variable",
            "offered_aca_disqualifying_esi",
        ),
        "us:statutes/26/36B#employer_sponsored_plan_minimum_value_share": (
            "policyengine_variable",
            "offered_aca_disqualifying_esi",
        ),
        "us:statutes/26/36B#qualified_small_employer_hra_affordability_ratio": (
            None,
            None,
        ),
        "us:statutes/26/36B#qualified_small_employer_hra_permitted_benefit_month_count": (
            None,
            None,
        ),
        "us:statutes/26/36B#qualified_small_employer_hra_monthly_permitted_benefit_fraction": (
            None,
            None,
        ),
        "us:statutes/26/36B/b#required_contribution_annual_to_monthly_divisor": (
            "policyengine_variable",
            "aca_ptc",
        ),
        "us:statutes/26/36B/b#required_monthly_contribution": (
            "policyengine_variable",
            "aca_ptc",
        ),
        "us:statutes/26/36B/b#premium_assistance_amount": (
            "policyengine_variable",
            "premium_tax_credit",
        ),
        "us:statutes/26/36B/b#premium_assistance_credit_amount": (
            "policyengine_variable",
            "aca_ptc",
        ),
    }
    for legal_id, (target_attr, target_value) in aca_36b_mappings.items():
        mapping = registry.mapping_for_legal_id(legal_id, country="us")
        assert mapping.mapping_type == "not_comparable"
        assert mapping.program == "aca_ptc"
        assert mapping.candidate_priority == "P4"
        if target_attr is not None:
            assert getattr(mapping, target_attr) == target_value
    qualified_veteran_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/e#veteran_employment_credit_against_subsection_a_tax",
        country="us",
    )
    assert qualified_veteran_credit_mapping.mapping_type == "not_comparable"
    assert qualified_veteran_credit_mapping.match_type == "prefix"
    research_payroll_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3111/f#research_credit_allowed_against_subsection_a_tax",
        country="us",
    )
    assert research_payroll_credit_mapping.mapping_type == "not_comparable"
    assert research_payroll_credit_mapping.match_type == "prefix"
    section_3306_k_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/3306/k#local_agricultural_labor",
        country="us",
    )
    assert section_3306_k_mapping.mapping_type == "not_comparable"
    assert section_3306_k_mapping.match_type == "prefix"
    assert (
        section_3306_k_mapping.policyengine_variable
        == "taxable_earnings_for_federal_unemployment_tax"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:policies/irs/rev-proc-2025-32/standard-deduction#basic_standard_deduction_amount",
            country="us",
        ).policyengine_variable
        == "basic_standard_deduction"
    )
    standard_deduction_single_mapping = registry.mapping_for_legal_id(
        "us:policies/irs/rev-proc-2025-32/standard-deduction#standard_deduction_single",
        country="us",
    )
    assert standard_deduction_single_mapping.mapping_type == "parameter_value"
    assert (
        standard_deduction_single_mapping.policyengine_parameter
        == "gov.irs.deductions.standard.amount"
    )
    assert standard_deduction_single_mapping.parameter_key == "SINGLE"
    married_additional_mapping = registry.mapping_for_legal_id(
        "us:policies/irs/rev-proc-2025-32/standard-deduction#additional_standard_deduction_married_or_surviving_spouse",
        country="us",
    )
    assert married_additional_mapping.mapping_type == "parameter_value"
    assert married_additional_mapping.parameter_keys == (
        "JOINT",
        "SEPARATE",
        "SURVIVING_SPOUSE",
    )
    assert (
        registry.mapping_for_legal_id(
            "us:policies/irs/rev-proc-2025-32/standard-deduction#standard_deduction",
            country="us",
        ).policyengine_variable
        == "standard_deduction"
    )
    basic_standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#basic_standard_deduction",
        country="us",
    )
    assert basic_standard_deduction_mapping.mapping_type == "not_comparable"
    assert (
        basic_standard_deduction_mapping.policyengine_variable
        == "basic_standard_deduction"
    )
    assert basic_standard_deduction_mapping.candidate_priority == "P4"
    additional_standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#additional_standard_deduction",
        country="us",
    )
    assert additional_standard_deduction_mapping.mapping_type == "not_comparable"
    assert (
        additional_standard_deduction_mapping.policyengine_variable
        == "additional_standard_deduction"
    )
    assert additional_standard_deduction_mapping.candidate_priority == "P4"
    standard_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/63/c#standard_deduction",
        country="us",
    )
    assert standard_deduction_mapping.mapping_type == "not_comparable"
    assert standard_deduction_mapping.policyengine_variable == "standard_deduction"
    assert standard_deduction_mapping.candidate_priority == "P4"
    ctc_phase_out_threshold_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/24/h#ctc_phase_out_threshold",
        country="us",
    )
    assert ctc_phase_out_threshold_mapping.mapping_type == "direct_variable"
    assert (
        ctc_phase_out_threshold_mapping.policyengine_variable
        == "ctc_phase_out_threshold"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1401#self_employment_tax",
            country="us",
        ).mapping_type
        == "not_comparable"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1402/a#self_employment_tax_equivalent_deduction_fraction",
            country="us",
        ).policyengine_parameter
        == "gov.irs.ald.misc.employer_share"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/1402/a#net_earnings_from_self_employment",
            country="us",
        ).match_type
        == "prefix"
    )
    qualified_tips_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/224#qualified_tips_deduction",
        country="us",
    )
    assert qualified_tips_mapping.mapping_type == "not_comparable"
    assert qualified_tips_mapping.match_type == "prefix"
    qualified_overtime_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/225#qualified_overtime_deduction",
        country="us",
    )
    assert qualified_overtime_mapping.mapping_type == "not_comparable"
    assert qualified_overtime_mapping.match_type == "prefix"
    nonitemizer_charitable_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/170/p#nonitemizer_charitable_deduction",
        country="us",
    )
    assert nonitemizer_charitable_mapping.mapping_type == "not_comparable"
    assert nonitemizer_charitable_mapping.match_type == "prefix"
    senior_deduction_amount_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/151#senior_deduction_base_amount",
        country="us",
    )
    assert senior_deduction_amount_mapping.mapping_type == "parameter_value"
    assert (
        senior_deduction_amount_mapping.policyengine_parameter
        == "gov.irs.deductions.senior_deduction.amount"
    )
    assert senior_deduction_amount_mapping.match_type == "exact"
    section_151_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/151#senior_deduction",
        country="us",
    )
    assert section_151_formula_mapping.mapping_type == "not_comparable"
    assert section_151_formula_mapping.match_type == "prefix"
    qbi_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qbi_deduction_rate",
        country="us",
    )
    assert qbi_rate_mapping.mapping_type == "parameter_value"
    assert qbi_rate_mapping.policyengine_parameter == "gov.irs.deductions.qbi.max.rate"
    assert qbi_rate_mapping.match_type == "exact"
    qbi_phasein_other_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qbi_phasein_range_other",
        country="us",
    )
    assert qbi_phasein_other_mapping.mapping_type == "parameter_value"
    assert qbi_phasein_other_mapping.parameter_keys == (
        "SINGLE",
        "SEPARATE",
        "HEAD_OF_HOUSEHOLD",
    )
    qbi_floor_amount_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#minimum_active_qbi_deduction_base",
        country="us",
    )
    assert qbi_floor_amount_mapping.mapping_type == "parameter_value"
    assert qbi_floor_amount_mapping.parameter_key_path == (
        "brackets",
        1,
        "amount",
    )
    qbi_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/199A#qualified_business_income_deduction",
        country="us",
    )
    assert qbi_formula_mapping.mapping_type == "not_comparable"
    assert qbi_formula_mapping.match_type == "prefix"
    medical_floor_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/213#medical_expense_agi_floor_rate",
        country="us",
    )
    assert medical_floor_mapping.mapping_type == "parameter_value"
    assert (
        medical_floor_mapping.policyengine_parameter
        == "gov.irs.deductions.itemized.medical.floor"
    )
    assert medical_floor_mapping.match_type == "exact"
    medical_formula_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/213#medical_care_deduction",
        country="us",
    )
    assert medical_formula_mapping.mapping_type == "not_comparable"
    assert medical_formula_mapping.match_type == "prefix"
    filing_requirement_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/6012#individual_income_tax_return_required_under_2018_2025_rule",
        country="us",
    )
    assert filing_requirement_mapping.mapping_type == "not_comparable"
    assert filing_requirement_mapping.match_type == "prefix"
    self_employment_tax_deduction_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/164/f#self_employment_tax_deduction",
        country="us",
    )
    assert self_employment_tax_deduction_mapping.mapping_type == "direct_variable"
    assert (
        self_employment_tax_deduction_mapping.policyengine_variable
        == "self_employment_tax_ald"
    )
    social_security_taxable_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/86#social_security_benefits_included_in_gross_income",
        country="us",
    )
    assert social_security_taxable_mapping.mapping_type == "direct_variable"
    assert (
        social_security_taxable_mapping.policyengine_variable
        == "tax_unit_taxable_social_security"
    )
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/86#social_security_base_amount_joint",
            country="us",
        ).policyengine_parameter
        == "gov.irs.social_security.taxability.threshold.base.main"
    )
    savers_credit_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit",
        country="us",
    )
    assert savers_credit_mapping.mapping_type == "not_comparable"
    assert savers_credit_mapping.policyengine_variable == "savers_credit_potential"
    savers_credit_cap_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_contribution_cap",
        country="us",
    )
    assert savers_credit_cap_mapping.mapping_type == "parameter_value"
    assert (
        savers_credit_cap_mapping.policyengine_parameter
        == "gov.irs.credits.retirement_saving.contributions_cap"
    )
    savers_credit_rate_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_middle_rate",
        country="us",
    )
    assert savers_credit_rate_mapping.parameter_key_path == ("amounts", 1)
    threshold_multiplier_mapping = registry.mapping_for_legal_id(
        "us:statutes/26/25B#savers_credit_threshold_multiplier",
        country="us",
    )
    assert threshold_multiplier_mapping.mapping_type == "parameter_value"
    assert (
        threshold_multiplier_mapping.policyengine_parameter
        == "gov.irs.credits.retirement_saving.rate.threshold_adjustment"
    )
    assert threshold_multiplier_mapping.parameter_key_input == "filing_status"
    assert (
        registry.mapping_for_legal_id(
            "us:statutes/26/25B#savers_credit_applicable_percentage",
            country="us",
        ).match_type
        == "prefix"
    )


def test_packaged_kentucky_2026_oracle_registry_and_pin_are_synchronized():
    root = Path(__file__).parents[1]
    document = yaml.safe_load(
        (root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml").read_text()
    )
    records = {
        item["legal_id"]: item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith("us-ky:policies/income_tax/2026_")
    }
    source_hold_prefix = (
        "us-ky:policies/income_tax/2026_full_year_resident_source_hold#"
    )
    source_holds = {
        legal_id: item
        for legal_id, item in records.items()
        if legal_id.startswith(source_hold_prefix)
    }
    schedule_prefix = (
        "us-ky:policies/income_tax/2026_krs_141_020_schedule_before_credits#"
    )
    schedule = {
        legal_id.removeprefix(schedule_prefix): item
        for legal_id, item in records.items()
        if legal_id.startswith(schedule_prefix)
    }

    assert len(source_holds) == 15
    assert {item["mapping_type"] for item in source_holds.values()} == {
        "not_comparable"
    }
    assert {item["candidate_priority"] for item in source_holds.values()} == {"P4"}
    assert set(schedule) == {
        "ky_pit_2026_krs_141_020_rate",
        "ky_pit_2026_krs_141_020_net_income_boundary",
        "ky_pit_2026_krs_141_020_schedule_before_credits",
    }
    assert (
        schedule["ky_pit_2026_krs_141_020_rate"]["policyengine_parameter"]
        == "gov.states.ky.tax.income.rate"
    )
    assert (
        schedule["ky_pit_2026_krs_141_020_schedule_before_credits"][
            "policyengine_variable"
        ]
        == "ky_income_tax_before_non_refundable_credits_unit"
    )

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def test_packaged_alabama_2026_schedule_registry_and_fallback_are_synchronized():
    root = Path(__file__).parents[1]
    document = yaml.safe_load(
        (root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml").read_text()
    )
    schedule_prefix = (
        "us-al:policies/income_tax/2026_section_40_18_5_schedule_before_credits#"
    )
    schedule = {
        item["legal_id"].removeprefix(schedule_prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(schedule_prefix)
    }

    assert set(schedule) == {
        "al_pit_2026_section_40_18_5_first_rate",
        "al_pit_2026_section_40_18_5_second_rate",
        "al_pit_2026_section_40_18_5_third_rate",
        "al_pit_2026_section_40_18_5_nonjoint_first_bracket_ceiling",
        "al_pit_2026_section_40_18_5_nonjoint_second_bracket_ceiling",
        "al_pit_2026_section_40_18_5_joint_first_bracket_ceiling",
        "al_pit_2026_section_40_18_5_joint_second_bracket_ceiling",
        "al_pit_2026_section_40_18_5_taxable_income_boundary",
        "al_pit_2026_section_40_18_5_schedule_before_credits",
    }
    assert {name: item["mapping_type"] for name, item in schedule.items()} == {
        "al_pit_2026_section_40_18_5_first_rate": "parameter_value",
        "al_pit_2026_section_40_18_5_second_rate": "parameter_value",
        "al_pit_2026_section_40_18_5_third_rate": "parameter_value",
        "al_pit_2026_section_40_18_5_nonjoint_first_bracket_ceiling": (
            "parameter_value"
        ),
        "al_pit_2026_section_40_18_5_nonjoint_second_bracket_ceiling": (
            "parameter_value"
        ),
        "al_pit_2026_section_40_18_5_joint_first_bracket_ceiling": ("parameter_value"),
        "al_pit_2026_section_40_18_5_joint_second_bracket_ceiling": ("parameter_value"),
        "al_pit_2026_section_40_18_5_taxable_income_boundary": "direct_variable",
        "al_pit_2026_section_40_18_5_schedule_before_credits": "direct_variable",
    }

    registry = load_policyengine_registry()
    exact = registry.mapping_for_legal_id(
        f"{schedule_prefix}al_pit_2026_section_40_18_5_first_rate",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-al:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert exact is not None
    assert exact.match_type == "exact"
    assert exact.mapping_type == "parameter_value"
    assert fallback is not None
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == "9434ce833bacbc60bc1572ea0245913ff4e48501"
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def test_packaged_connecticut_2026_ordinary_tax_registry_is_exactly_synchronized():
    import axiom_oracles.bridges.registry as runtime_registry_module

    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_document = yaml.safe_load(bundled_path.read_text())
    runtime_document = yaml.safe_load(runtime_path.read_text())
    schedule_prefix = (
        "us-ct:policies/income_tax/2026_resident_ordinary_tax_before_personal_credit#"
    )

    def schedule_records(document):
        return {
            item["legal_id"].removeprefix(schedule_prefix): item
            for item in document["mappings"]
            if item.get("legal_id", "").startswith(schedule_prefix)
        }

    bundled = schedule_records(bundled_document)
    runtime = schedule_records(runtime_document)
    not_comparable = {
        "ct_pit_2026_ordinary_tax_filing_status_is_valid",
        "ct_pit_2026_ordinary_tax_input_domain_is_valid",
        "ct_pit_2026_ordinary_tax_schedule_scale",
    }
    parameter_values = {
        "ct_pit_2026_ordinary_tax_head_first_ceiling",
        "ct_pit_2026_ordinary_tax_joint_first_ceiling",
        *(f"ct_pit_2026_ordinary_tax_rate_{index}" for index in range(1, 8)),
        *(f"ct_pit_2026_ordinary_tax_single_floor_{index}" for index in range(2, 8)),
        "ct_pit_2026_ordinary_tax_single_phaseout_start",
        "ct_pit_2026_ordinary_tax_head_phaseout_start",
        "ct_pit_2026_ordinary_tax_joint_phaseout_start",
        "ct_pit_2026_ordinary_tax_separate_phaseout_start",
        "ct_pit_2026_ordinary_tax_single_phaseout_increment",
        "ct_pit_2026_ordinary_tax_head_phaseout_increment",
        "ct_pit_2026_ordinary_tax_separate_phaseout_increment",
        "ct_pit_2026_ordinary_tax_single_phaseout_amount",
        "ct_pit_2026_ordinary_tax_single_phaseout_maximum",
        "ct_pit_2026_ordinary_tax_single_low_recapture_start",
        "ct_pit_2026_ordinary_tax_single_middle_recapture_start",
        "ct_pit_2026_ordinary_tax_single_high_recapture_start",
        "ct_pit_2026_ordinary_tax_single_recapture_increment",
        "ct_pit_2026_ordinary_tax_single_low_recapture_amount",
        "ct_pit_2026_ordinary_tax_single_middle_recapture_amount",
        "ct_pit_2026_ordinary_tax_single_high_recapture_amount",
        "ct_pit_2026_ordinary_tax_single_low_recapture_maximum",
        "ct_pit_2026_ordinary_tax_single_middle_recapture_maximum",
        "ct_pit_2026_ordinary_tax_single_high_recapture_maximum",
        "ct_pit_2026_ordinary_tax_head_middle_recapture_amount",
        "ct_pit_2026_ordinary_tax_head_middle_recapture_maximum",
    }
    direct_variables = {
        "ct_pit_2026_ordinary_low_rate_phaseout_addback",
        "ct_pit_2026_ordinary_low_recapture",
        "ct_pit_2026_ordinary_middle_recapture",
        "ct_pit_2026_ordinary_high_recapture",
    }
    derived_expressions = {
        "ct_pit_2026_ordinary_schedule_tax",
        "ct_pit_2026_resident_ordinary_tax_before_personal_credit",
    }
    expected_types = {
        **dict.fromkeys(not_comparable, "not_comparable"),
        **dict.fromkeys(parameter_values, "parameter_value"),
        **dict.fromkeys(direct_variables, "direct_variable"),
        **dict.fromkeys(derived_expressions, "derived_expression"),
    }

    assert len(expected_types) == 45
    assert set(bundled) == set(expected_types)
    assert {name: item["mapping_type"] for name, item in bundled.items()} == (
        expected_types
    )
    assert bundled == runtime

    registry = load_policyengine_registry()
    exact = registry.mapping_for_legal_id(
        f"{schedule_prefix}ct_pit_2026_ordinary_tax_rate_1",
        country="us",
    )
    final_component = registry.mapping_for_legal_id(
        f"{schedule_prefix}ct_pit_2026_resident_ordinary_tax_before_personal_credit",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-ct:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert exact is not None
    assert exact.match_type == "exact"
    assert exact.mapping_type == "parameter_value"
    assert final_component is not None
    assert final_component.match_type == "exact"
    assert final_component.mapping_type == "derived_expression"
    assert fallback is not None
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == "9434ce833bacbc60bc1572ea0245913ff4e48501"
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def test_packaged_georgia_2026_annual_tax_registry_is_exactly_synchronized():
    import axiom_oracles.bridges.registry as runtime_registry_module

    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_document = yaml.safe_load(bundled_path.read_text())
    runtime_document = yaml.safe_load(runtime_path.read_text())
    schedule_prefix = (
        "us-ga:policies/income_tax/2026_annual_tax_before_nonrefundable_credits#"
    )

    def schedule_records(document):
        return {
            item["legal_id"].removeprefix(schedule_prefix): item
            for item in document["mappings"]
            if item.get("legal_id", "").startswith(schedule_prefix)
        }

    bundled = schedule_records(bundled_document)
    runtime = schedule_records(runtime_document)

    assert set(bundled) == {
        "ga_pit_2026_annual_input_is_nonnegative",
        "ga_pit_2026_annual_tax_before_nonrefundable_credits",
    }
    assert bundled == runtime

    private = bundled["ga_pit_2026_annual_input_is_nonnegative"]
    assert private["mapping_type"] == "not_comparable"
    assert private["candidate_priority"] == "P4"
    assert "policyengine_variable" not in private

    public = bundled["ga_pit_2026_annual_tax_before_nonrefundable_credits"]
    assert public["mapping_type"] == "direct_variable"
    assert (
        public["policyengine_variable"] == "ga_income_tax_before_non_refundable_credits"
    )
    assert public["entity"] == "tax_unit"
    assert public["period"] == "year"
    assert public["unit"] == "USD"
    assert public["comparison"] == "money"
    assert "candidate_priority" not in public
    assert "final Form 500 liability" in public["rationale"]

    registry = load_policyengine_registry()
    exact_private = registry.mapping_for_legal_id(
        f"{schedule_prefix}ga_pit_2026_annual_input_is_nonnegative",
        country="us",
    )
    exact_public = registry.mapping_for_legal_id(
        f"{schedule_prefix}ga_pit_2026_annual_tax_before_nonrefundable_credits",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-ga:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert exact_private is not None
    assert exact_private.match_type == "exact"
    assert exact_private.mapping_type == "not_comparable"
    assert exact_private.candidate_priority == "P4"
    assert exact_public is not None
    assert exact_public.match_type == "exact"
    assert exact_public.mapping_type == "direct_variable"
    assert (
        exact_public.policyengine_variable
        == "ga_income_tax_before_non_refundable_credits"
    )
    assert fallback is not None
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == "9434ce833bacbc60bc1572ea0245913ff4e48501"
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def test_packaged_mississippi_2026_schedule_registry_is_exactly_synchronized():
    import axiom_oracles.bridges.registry as runtime_registry_module

    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_document = yaml.safe_load(bundled_path.read_text())
    runtime_document = yaml.safe_load(runtime_path.read_text())
    schedule_prefix = "us-ms:policies/income_tax/2026_section_27_7_5_schedule#"

    def schedule_records(document):
        return {
            item["legal_id"].removeprefix(schedule_prefix): item
            for item in document["mappings"]
            if item.get("legal_id", "").startswith(schedule_prefix)
        }

    bundled = schedule_records(bundled_document)
    runtime = schedule_records(runtime_document)
    assert set(bundled) == {
        "ms_pit_2026_zero_tax_ceiling",
        "ms_pit_2026_rate",
        "ms_pit_2026_schedule_taxable_income",
        "ms_pit_2026_section_27_7_5_schedule_tax",
    }
    assert bundled == runtime
    bundled_fallbacks = [
        item
        for item in bundled_document["prefixes"]
        if item.get("legal_id_prefix") == "us-ms:"
    ]
    runtime_fallbacks = [
        item
        for item in runtime_document["prefixes"]
        if item.get("legal_id_prefix") == "us-ms:"
    ]
    assert len(bundled_fallbacks) == 1
    assert bundled_fallbacks == runtime_fallbacks

    ceiling = bundled["ms_pit_2026_zero_tax_ceiling"]
    assert ceiling["mapping_type"] == "parameter_value"
    assert ceiling["policyengine_parameter"] == "gov.states.ms.tax.income.rate"
    assert ceiling["parameter_key_path"] == ["thresholds", 1]
    assert ceiling["comparison"] == "money"

    rate = bundled["ms_pit_2026_rate"]
    assert rate["mapping_type"] == "parameter_value"
    assert rate["policyengine_parameter"] == "gov.states.ms.tax.income.rate"
    assert rate["parameter_key_path"] == ["rates", 1]
    assert rate["comparison"] == "rate"

    taxable_income = bundled["ms_pit_2026_schedule_taxable_income"]
    assert taxable_income["mapping_type"] == "not_comparable"
    assert taxable_income["candidate_priority"] == "P4"
    assert "policyengine_variable" not in taxable_income

    schedule_tax = bundled["ms_pit_2026_section_27_7_5_schedule_tax"]
    assert schedule_tax["mapping_type"] == "direct_variable"
    assert schedule_tax["policyengine_variable"] == (
        "ms_income_tax_before_credits_joint"
    )
    assert schedule_tax["entity"] == "person"
    assert schedule_tax["period"] == "year"
    assert schedule_tax["unit"] == "USD"
    assert schedule_tax["comparison"] == "money"
    assert "final liability" in schedule_tax["rationale"]

    registry = load_policyengine_registry()
    exact = registry.mapping_for_legal_id(
        f"{schedule_prefix}ms_pit_2026_section_27_7_5_schedule_tax",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-ms:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert exact is not None
    assert exact.match_type == "exact"
    assert exact.mapping_type == "direct_variable"
    assert exact.policyengine_variable == "ms_income_tax_before_credits_joint"
    assert fallback is not None
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == "9434ce833bacbc60bc1572ea0245913ff4e48501"
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def _kansas_2026_k40es_records(document):
    schedule_prefix = "us-ks:policies/income_tax/2026_k40es_schedule_before_credits#"
    return {
        item["legal_id"].removeprefix(schedule_prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(schedule_prefix)
    }


def _kansas_2026_shared_mapping_material(text):
    def exact_section(start, end):
        start_index = text.index(start)
        end_index = text.index(end, start_index) + len(end)
        return text[start_index:end_index]

    schedule = exact_section(
        "  # The official Kansas K-40ES",
        "    rationale: Both outputs select the joint or all-other Kansas "
        "schedule and apply it to completed Kansas taxable income before "
        "credits. The Populace runner fails closed if PolicyEngine's separate "
        "adjusted-gross-income gate would suppress an otherwise positive "
        "K-40ES schedule domain.",
    )
    fallback = exact_section(
        '  - legal_id_prefix: "us-ks:"',
        "    rationale: PolicyEngine-US does not model every Kansas statute, "
        "agency manual, regulation, or source-level output one to one; "
        "independently reviewed comparable outputs carry exact mappings which "
        "take precedence over this jurisdiction-wide fallback.",
    )
    return f"{schedule}\n\n{fallback}".replace("\r\n", "\n").replace("\r", "\n")


def test_packaged_kansas_2026_k40es_registry_has_exact_bounded_slice():
    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    bundled_document = yaml.safe_load(bundled_path.read_text())
    bundled = _kansas_2026_k40es_records(bundled_document)
    expected_types = {
        "ks_pit_2026_k40es_lower_rate": "parameter_value",
        "ks_pit_2026_k40es_upper_rate": "parameter_value",
        "ks_pit_2026_k40es_joint_lower_bracket_ceiling": "not_comparable",
        "ks_pit_2026_k40es_other_lower_bracket_ceiling": "not_comparable",
        "ks_pit_2026_k40es_taxable_income_boundary": "direct_variable",
        "ks_pit_2026_k40es_schedule_before_credits": "direct_variable",
    }

    assert set(bundled) == set(expected_types)
    assert {name: item["mapping_type"] for name, item in bundled.items()} == (
        expected_types
    )

    for output_name, index in (
        ("ks_pit_2026_k40es_lower_rate", 0),
        ("ks_pit_2026_k40es_upper_rate", 1),
    ):
        rate = bundled[output_name]
        assert rate["policyengine_parameter"] == (
            "gov.states.ks.tax.income.rates.joint"
        )
        assert rate["parameter_key_path"] == ["rates", index]
        assert rate["period"] == "year"
        assert rate["comparison"] == "rate"

    for output_name, scale in (
        ("ks_pit_2026_k40es_joint_lower_bracket_ceiling", "joint"),
        ("ks_pit_2026_k40es_other_lower_bracket_ceiling", "other"),
    ):
        ceiling = bundled[output_name]
        assert ceiling["candidate_priority"] == "P4"
        assert ceiling["policyengine_parameter"] == (
            f"gov.states.ks.tax.income.rates.{scale}"
        )
        assert "first dollar of the upper bracket" in ceiling["rationale"]

    for output_name, variable in (
        ("ks_pit_2026_k40es_taxable_income_boundary", "ks_taxable_income"),
        (
            "ks_pit_2026_k40es_schedule_before_credits",
            "ks_income_tax_before_credits",
        ),
    ):
        output = bundled[output_name]
        assert output["policyengine_variable"] == variable
        assert (
            output["entity"],
            output["period"],
            output["unit"],
            output["comparison"],
        ) == ("tax_unit", "year", "USD", "money")

    bundled_fallbacks = [
        item
        for item in bundled_document["prefixes"]
        if item.get("legal_id_prefix") == "us-ks:"
    ]
    assert len(bundled_fallbacks) == 1
    assert bundled_fallbacks[0]["mapping_type"] == "not_comparable"
    assert bundled_fallbacks[0]["candidate_priority"] == "P4"


def test_packaged_kansas_2026_k40es_registry_is_exactly_synchronized():
    import axiom_oracles.bridges.registry as runtime_registry_module

    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_text = bundled_path.read_text()
    runtime_text = runtime_path.read_text()
    bundled_document = yaml.safe_load(bundled_text)
    runtime_document = yaml.safe_load(runtime_text)
    bundled = _kansas_2026_k40es_records(bundled_document)
    runtime = _kansas_2026_k40es_records(runtime_document)

    assert bundled == runtime
    assert _kansas_2026_shared_mapping_material(
        bundled_text
    ) == _kansas_2026_shared_mapping_material(runtime_text)

    bundled_fallbacks = [
        item
        for item in bundled_document["prefixes"]
        if item.get("legal_id_prefix") == "us-ks:"
    ]
    runtime_fallbacks = [
        item
        for item in runtime_document["prefixes"]
        if item.get("legal_id_prefix") == "us-ks:"
    ]
    assert len(bundled_fallbacks) == 1
    assert bundled_fallbacks == runtime_fallbacks

    registry = load_policyengine_registry()
    exact = registry.mapping_for_legal_id(
        "us-ks:policies/income_tax/2026_k40es_schedule_before_credits"
        "#ks_pit_2026_k40es_schedule_before_credits",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-ks:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert exact is not None
    assert exact.match_type == "exact"
    assert exact.mapping_type == "direct_variable"
    assert exact.policyengine_variable == "ks_income_tax_before_credits"
    assert fallback is not None
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == "9434ce833bacbc60bc1572ea0245913ff4e48501"
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()


def _dc_2026_section_47_1806_03_records(document):
    schedule_prefix = (
        "us-dc:policies/income_tax/2026_section_47_1806_03_schedule_before_credits#"
    )
    return {
        item["legal_id"].removeprefix(schedule_prefix): item
        for item in document["mappings"]
        if item.get("legal_id", "").startswith(schedule_prefix)
    }


def _dc_2026_shared_mapping_material(text):
    def exact_section(start, end):
        start_index = text.index(start)
        end_index = text.index(end, start_index) + len(end)
        return text[start_index:end_index]

    schedule = exact_section(
        "  # The canonical section 47-1806.03(a)(11) module is a narrow joint-method",
        "    rationale: Both outputs apply the post-2021 seven-band District "
        "schedule to completed joint-method District taxable income before "
        "credits. The exact joint-method target intentionally excludes "
        "PolicyEngine's broader filing-method selector "
        "dc_income_tax_before_credits.",
    )
    fallback = exact_section(
        '  - legal_id_prefix: "us-dc:"',
        "    rationale: PolicyEngine-US does not model every District statute, "
        "agency manual, regulation, or source-level output one to one; "
        "independently reviewed comparable outputs carry exact mappings which "
        "take precedence over this jurisdiction-wide fallback.",
    )
    return f"{schedule}\n\n{fallback}".replace("\r\n", "\n").replace("\r", "\n")


def test_packaged_dc_2026_section_47_1806_03_has_exact_bounded_slice():
    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    bundled_document = yaml.safe_load(bundled_path.read_text())
    bundled = _dc_2026_section_47_1806_03_records(bundled_document)
    not_comparable = {
        "dc_pit_2026_section_47_1806_03_bracket_upper",
        "dc_pit_2026_section_47_1806_03_bracket_floor",
        "dc_pit_2026_section_47_1806_03_bracket_base",
        "dc_pit_2026_section_47_1806_03_bracket_rate",
        "dc_pit_2026_section_47_1806_03_bracket_selector",
    }
    direct_variables = {
        "dc_pit_2026_section_47_1806_03_taxable_income_boundary",
        "dc_pit_2026_section_47_1806_03_schedule_before_credits",
    }
    expected_types = {
        **dict.fromkeys(not_comparable, "not_comparable"),
        **dict.fromkeys(direct_variables, "direct_variable"),
    }

    assert len(expected_types) == 7
    assert set(bundled) == set(expected_types)
    assert {name: item["mapping_type"] for name, item in bundled.items()} == (
        expected_types
    )
    assert {item["program"] for item in bundled.values()} == {"tax"}
    for output_name in not_comparable:
        output = bundled[output_name]
        assert output["candidate_priority"] == "P4"
        assert "policyengine_variable" not in output

    boundary = bundled["dc_pit_2026_section_47_1806_03_taxable_income_boundary"]
    assert boundary["policyengine_variable"] == "dc_taxable_income_joint"
    assert "potentially negative" in boundary["rationale"]
    assert "independently floor it at zero" in boundary["rationale"]

    schedule = bundled["dc_pit_2026_section_47_1806_03_schedule_before_credits"]
    assert schedule["policyengine_variable"] == ("dc_income_tax_before_credits_joint")
    assert "joint-method" in schedule["rationale"]
    assert "before credits" in schedule["rationale"]
    assert "broader filing-method selector" in schedule["rationale"]
    assert schedule["policyengine_variable"] != "dc_income_tax_before_credits"

    for output in (boundary, schedule):
        assert (
            output["entity"],
            output["period"],
            output["unit"],
            output["comparison"],
        ) == ("tax_unit", "year", "USD", "money")

    bundled_fallbacks = [
        item
        for item in bundled_document["prefixes"]
        if item.get("legal_id_prefix") == "us-dc:"
    ]
    assert len(bundled_fallbacks) == 1
    assert bundled_fallbacks[0]["mapping_type"] == "not_comparable"
    assert bundled_fallbacks[0]["candidate_priority"] == "P4"
    assert "take precedence" in bundled_fallbacks[0]["rationale"]


def test_packaged_dc_2026_registry_text_hash_runtime_and_precedence_are_exact():
    import axiom_oracles.bridges.registry as runtime_registry_module

    durable_oracle_merge = "9434ce833bacbc60bc1572ea0245913ff4e48501"
    root = Path(__file__).parents[1]
    bundled_path = root / "src/axiom_encode/oracles/policyengine/mappings/us.yaml"
    runtime_path = (
        Path(runtime_registry_module.__file__).with_name("mappings") / "us.yaml"
    )
    bundled_text = bundled_path.read_text()
    runtime_text = runtime_path.read_text()
    bundled_document = yaml.safe_load(bundled_text)
    runtime_document = yaml.safe_load(runtime_text)
    bundled = _dc_2026_section_47_1806_03_records(bundled_document)
    runtime = _dc_2026_section_47_1806_03_records(runtime_document)
    bundled_material = _dc_2026_shared_mapping_material(bundled_text)
    runtime_material = _dc_2026_shared_mapping_material(runtime_text)
    encoded_material = bundled_material.encode()

    assert bundled == runtime
    assert bundled_material == runtime_material
    assert len(bundled_material.splitlines()) == 66
    assert len(encoded_material) == 4_340
    assert hashlib.sha256(encoded_material).hexdigest() == (
        "30c426331da1ded1c90ad5db9a0104d91a425c3c0e5e08bfd3375bbf239e4ebf"
    )

    bundled_fallbacks = [
        item
        for item in bundled_document["prefixes"]
        if item.get("legal_id_prefix") == "us-dc:"
    ]
    runtime_fallbacks = [
        item
        for item in runtime_document["prefixes"]
        if item.get("legal_id_prefix") == "us-dc:"
    ]
    assert len(bundled_fallbacks) == 1
    assert bundled_fallbacks == runtime_fallbacks

    prefix = (
        "us-dc:policies/income_tax/2026_section_47_1806_03_schedule_before_credits#"
    )
    registry = load_policyengine_registry()
    boundary = registry.mapping_for_legal_id(
        f"{prefix}dc_pit_2026_section_47_1806_03_taxable_income_boundary",
        country="us",
    )
    schedule = registry.mapping_for_legal_id(
        f"{prefix}dc_pit_2026_section_47_1806_03_schedule_before_credits",
        country="us",
    )
    fallback = registry.mapping_for_legal_id(
        "us-dc:policies/income_tax/unrelated#future_output",
        country="us",
    )
    assert boundary is not None
    assert boundary.match_type == "exact"
    assert boundary.mapping_type == "direct_variable"
    assert boundary.policyengine_variable == "dc_taxable_income_joint"
    assert schedule is not None
    assert schedule.match_type == "exact"
    assert schedule.mapping_type == "direct_variable"
    assert schedule.policyengine_variable == ("dc_income_tax_before_credits_joint")
    assert fallback is not None
    assert fallback.legal_id == "us-dc:"
    assert fallback.match_type == "prefix"
    assert fallback.mapping_type == "not_comparable"
    assert fallback.candidate_priority == "P4"

    dependency_pin = re.search(
        r"axiom-oracles@[0-9a-f]{40}", (root / "pyproject.toml").read_text()
    )
    assert dependency_pin is not None
    pin = dependency_pin.group(0).removeprefix("axiom-oracles@")
    assert pin == durable_oracle_merge
    assert f"?rev={pin}#{pin}" in (root / "uv.lock").read_text()
    assert (
        (root / "src/axiom_encode/__init__.py")
        .read_text()
        .startswith('__version__ = "0.2.2297"')

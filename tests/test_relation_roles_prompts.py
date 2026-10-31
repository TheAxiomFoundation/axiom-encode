"""Keep relation-role instructions in both encoder prompts and retry paths."""

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode.harness.evals import (
    EvalWorkspace,
    _build_empty_artifact_retry_prompt,
    _build_rulespec_eval_prompt,
)
from axiom_encode.prompts.encoder import (
    RELATION_ROLES_FORMULA_PROTOCOL,
    RELATION_ROLES_SELF_CHECK,
    RELATION_ROLES_SHAPE,
    RELATION_ROLES_STRUCTURE_PROTOCOL,
    RELATION_ROLES_TESTS_PROTOCOL,
    get_encoder_prompt,
)

ROLE_BLOCKS = (
    RELATION_ROLES_STRUCTURE_PROTOCOL,
    RELATION_ROLES_FORMULA_PROTOCOL,
    RELATION_ROLES_TESTS_PROTOCOL,
    RELATION_ROLES_SELF_CHECK,
    RELATION_ROLES_SHAPE,
)


@settings(
    max_examples=36,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    include_tests=st.booleans(),
    complete_source_unit=st.booleans(),
    backend=st.sampled_from(["codex", "openai", "claude"]),
    retry=st.sampled_from(["none", "cached", "uncached"]),
    source=st.sampled_from(
        [
            "The household contains people who live together.",
            "The applicant and the applicant's spouse form a household.",
            "The source requires the stated group and excludes a participant.",
        ]
    ),
)
def test_relation_role_rules_match_across_prompt_paths(
    tmp_path: Path,
    include_tests: bool,
    complete_source_unit: bool,
    backend: str,
    retry: str,
    source: str,
):
    source_file = tmp_path / "source.txt"
    source_file.write_text(source)
    manifest_file = tmp_path / "context-manifest.json"
    manifest_file.write_text("{}")
    workspace = EvalWorkspace(
        root=tmp_path,
        source_text_file=source_file,
        manifest_file=manifest_file,
    )
    encoder_prompt = get_encoder_prompt(
        citation="us/regulation/7/273/1",
        output_path="regulations/7/273/1.yaml",
        corpus_citation_path="us/regulation/7/273/1",
        require_complete_source_unit=complete_source_unit,
    )
    eval_prompt = _build_rulespec_eval_prompt(
        citation="us/regulation/7/273/1",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="1.yaml",
        target_ref_prefix="us:regulations/7/273/1",
        include_tests=include_tests,
        runner_backend=backend,
        policyengine_rule_hint=None,
        require_complete_source_unit=complete_source_unit,
    )
    if retry != "none":
        # A plain string exercises the older source-scope trimming path; the
        # prompt subclass exercises the path that preserves its cache boundary.
        eval_prompt = _build_empty_artifact_retry_prompt(
            str(eval_prompt) if retry == "uncached" else eval_prompt,
            target_file_name="1.yaml",
            include_tests=include_tests,
        )

    for block in ROLE_BLOCKS:
        assert encoder_prompt.count(block.strip()) == 1
        assert eval_prompt.count(block.strip()) == 1


@pytest.mark.parametrize(
    "instruction",
    [
        "A contested legal reading gets no default",
        "a provision or cited State policy requires that closure",
        "composition or participation",
        "including false assertions",
        "Never approximate a closure or silently truncate a chain",
        "Never give an election a default",
        "does not yet declare completeness",
    ],
)
def test_relation_role_protocol_keeps_legal_and_runtime_limits(instruction: str):
    normalized = " ".join("\n".join(ROLE_BLOCKS).split())
    assert instruction in normalized


def test_relation_role_shape_preserves_filtered_entity_route():
    prompt = get_encoder_prompt("example", "example.yaml")
    assert "entity: SnapUnit" in prompt
    assert "member_relation: members" in prompt
    assert "source_relation: member_of_household" in prompt
    assert "source_path:" in prompt
    assert "candidate.snap_household_unit == anchor.snap_household_unit" in prompt
    assert "It does not encode any provision in full" in " ".join(prompt.split())

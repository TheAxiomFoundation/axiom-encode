"""Tests for canonical-concept registry injection into the encoder prompt."""

from __future__ import annotations

from pathlib import Path

from axiom_encode.concepts.registry import load_concept_registry
from axiom_encode.harness.evals import (
    EvalContextFile,
    EvalWorkspace,
    _build_rulespec_eval_prompt,
    _format_canonical_concept_registry_guidance,
)


def _minimal_workspace(tmp_path: Path, source_text: str) -> EvalWorkspace:
    source_text_file = tmp_path / "source.txt"
    source_text_file.write_text(source_text)
    manifest_file = tmp_path / "context-manifest.json"
    manifest_file.write_text("{}")
    return EvalWorkspace(
        root=tmp_path,
        source_text_file=source_text_file,
        manifest_file=manifest_file,
    )


def test_canonical_concept_section_present_when_source_mentions_blocked_synonym(
    tmp_path: Path,
):
    source_text = (
        "Household snap_gross_monthly_income is the sum of earned and unearned "
        "income before exclusions and deductions."
    )
    workspace = _minimal_workspace(tmp_path, source_text)

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=[],
    )

    assert "Canonical concept names:" in section
    assert "`snap_total_gross_income`" in section
    assert "`snap_gross_monthly_income`" in section
    assert "do not use:" in section


def test_canonical_concept_section_present_when_source_mentions_canonical(
    tmp_path: Path,
):
    source_text = "Eligibility uses snap_total_gross_income as the comparison value."
    workspace = _minimal_workspace(tmp_path, source_text)

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=[],
    )

    assert "`snap_total_gross_income`" in section


def test_canonical_concept_section_absent_when_no_registered_tokens(tmp_path: Path):
    source_text = (
        "Any taxpayer who makes an election under this section shall include "
        "the amount in gross income for the taxable year."
    )
    workspace = _minimal_workspace(tmp_path, source_text)

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=[],
    )

    assert section == ""


def test_canonical_concept_section_scans_copied_context_files(tmp_path: Path):
    source_text = "Operative source paragraph with no canonical tokens."
    workspace = _minimal_workspace(tmp_path, source_text)
    context_path = tmp_path / "context" / "policies" / "usda" / "snap.yaml"
    context_path.parent.mkdir(parents=True)
    context_path.write_text(
        "format: rulespec/v1\nrules:\n  - name: snap_total_gross_income\n"
    )
    context_files = [
        EvalContextFile(
            source_path=str(context_path),
            workspace_path=str(context_path.relative_to(tmp_path)),
            import_path="us:policies/usda/snap",
            kind="canonical_concept",
        )
    ]

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=context_files,
    )

    assert "`snap_total_gross_income`" in section


def test_canonical_concept_section_only_lists_matched_concepts(tmp_path: Path):
    source_text = "Only snap_gross_monthly_income appears in this provision."
    workspace = _minimal_workspace(tmp_path, source_text)

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=[],
    )

    registry = load_concept_registry()
    unmatched = [
        c.canonical_name
        for c in registry.concepts_by_id.values()
        if c.canonical_name != "snap_total_gross_income"
    ]
    for name in unmatched:
        assert f"`{name}`" not in section


def test_canonical_concept_section_skipped_for_partial_substrings(tmp_path: Path):
    # "snap_total_gross_income_after_x" should not match the bare canonical
    # name even though it contains it as a substring.
    source_text = "see snap_total_gross_income_after_x for the override formula."
    workspace = _minimal_workspace(tmp_path, source_text)

    section = _format_canonical_concept_registry_guidance(
        source_text,
        workspace,
        context_files=[],
    )

    assert section == ""


def test_build_rulespec_eval_prompt_splices_canonical_concept_section(tmp_path: Path):
    source_text = "Calculate snap_gross_monthly_income before applying exclusions."
    workspace = _minimal_workspace(tmp_path, source_text)

    prompt = _build_rulespec_eval_prompt(
        citation="us/regulation/7-cfr/273/10",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="regulations/7-cfr/273/10.yaml",
        target_ref_prefix="us:regulations/7-cfr/273/10",
        include_tests=False,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )

    assert "Canonical concept names:" in prompt
    assert "`snap_total_gross_income`" in prompt
    assert "`snap_gross_monthly_income`" in prompt


def test_build_rulespec_eval_prompt_omits_section_when_no_match(tmp_path: Path):
    source_text = (
        "The taxpayer's gross income for the taxable year shall include "
        "amounts described in this paragraph."
    )
    workspace = _minimal_workspace(tmp_path, source_text)

    prompt = _build_rulespec_eval_prompt(
        citation="us/statute/26/61",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="statutes/26/61.yaml",
        target_ref_prefix="us:statutes/26/61",
        include_tests=False,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )

    assert "Canonical concept names:" not in prompt


def test_build_rulespec_eval_prompt_includes_scoped_exception_test_guidance(
    tmp_path: Path,
):
    source_text = (
        "Any qualifying expense is excluded except that this paragraph does "
        "not apply to expenses for a nonqualified service to the extent paid."
    )
    workspace = _minimal_workspace(tmp_path, source_text)

    prompt = _build_rulespec_eval_prompt(
        citation="7 CFR 273.9(d)(1)",
        mode="cold",
        workspace=workspace,
        context_files=[],
        target_file_name="regulations/7-cfr/273/9/d/1.yaml",
        target_ref_prefix="us:regulations/7-cfr/273/9/d/1",
        include_tests=True,
        runner_backend="codex",
        policyengine_rule_hint=None,
    )

    assert "predicate for the excepted category" in prompt
    test_file_rules = prompt.split("Test file rules:", maxsplit=1)[1].split(
        "Do not respond with summaries",
        maxsplit=1,
    )[0]
    assert "`subject to`" in test_file_rules
    assert "positive/nonzero" in test_file_rules
    assert "toggle each gate at least once" in test_file_rules
